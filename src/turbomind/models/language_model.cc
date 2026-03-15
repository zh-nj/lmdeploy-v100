
#include "src/turbomind/models/language_model.h"

#include <memory>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/interval.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/generation/generation.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mtp_predictor.h"
#include "src/turbomind/models/llama/rejection_sampling.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/output_processor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

struct LanguageModel::Impl {
    const DataType       dtype_;
    const ModelParam     param_;
    const AttentionParam attn_param_;
    const Communicators& comm_;
    const LlamaWeight&   weights_;
    LlamaLinear&         linear_;

    const int  tp_size_;
    const int  tp_rank_;
    const bool use_ag2d_;
    const bool speculative_decoding_;

    const bool debug_;

    Buffer_<bool> false_;

    // mutable state
    State finished_;
    State sequence_length_;  // length of known tokens
    // immutable state
    Buffer_<int> autoreg_ids_;
    // Buffer_<int> autoreg_ids_offsets_;

    // Symmetric buffer for holding global hidden states or logits
    Buffer_<uint8_t> symm_buf_;

    // Max chunk size for compute / output full logits
    int max_logits_len_ = 0;

    Buffer_<int>  sequence_length_buf_;
    Buffer_<bool> finished_buf_;

    struct Data {
        Buffer_<int>  sequence_length;
        Buffer_<bool> finished;

        Buffer_<bool> autoregres;
        Buffer_<bool> generating;

        int n_generating;
    };

    vector<Data> data_;

    std::optional<InputProcessor>   input_processor_;
    std::unique_ptr<UnifiedDecoder> unified_decoder_;
    std::optional<OutputProcessor>  output_processor_;
    std::unique_ptr<Generation>     generation_;  // token generator
    std::unique_ptr<MTPPredictor>   mtp_predictor_;  // MTP speculative decoding predictor

    Tensor       last_hidden_states_;  // [batch, hidden_size] preserved from Forward for DraftTokens

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        switch (op) {
            case BatchOp::kSetup:
                return Setup(phase, env);
            case BatchOp::kPrepare:
                return Prepare(phase, env);
            case BatchOp::kForward:
                return Forward(phase, env);
            case BatchOp::kUnprep:
                return Unprep(phase, env);
            case BatchOp::kFetch:
                return Fetch(phase, env);
            case BatchOp::kVerify:
                TM_CHECK(0) << "kVerify is deprecated in Verify-in-Next-Forward mode";
                return;
            case BatchOp::kSnapshotGDN:
                unified_decoder_->SnapshotGDNState(core::Context::stream().handle());
                return;
            case BatchOp::kRestoreGDN:
                unified_decoder_->RestoreGDNState(core::Context::stream().handle());
                return;
            case BatchOp::kSwapGDN:
                TM_CHECK(0) << "kSwapGDN is deprecated";
                return;
            case BatchOp::kDraft:
                return DraftTokens(phase, env);
            case BatchOp::kReject:
                return RejectDrafts(phase, env);
            case BatchOp::kRollback:
                return Rollback(phase, env);
            default:
                input_processor_->Run(op, phase, env);
                unified_decoder_->Run(op, phase, env);
                generation_->Run(op, phase, env);
                output_processor_->Run(op, phase, env);
        }
    }

    Impl(DataType              dtype,
         const ModelParam&     model,
         const EngineParam&    engine,
         const AttentionParam& attn,
         const MoeParam&       moe,
         const Context&        ctx,
         const LlamaWeight&    weights,
         int                   phases);

    Tensor LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf);
    Tensor PostEmbedding(const Tensor& features, Buffer symm_buf);

    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);
    void Forward(int phase, TensorMap& env);
    void Rollback(int phase, TensorMap& env);
    void DraftTokens(int phase, TensorMap& env);
    void RejectDrafts(int phase, TensorMap& env);
    void Unprep(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);
};

LanguageModel::Impl::Impl(DataType              dtype,
                          const ModelParam&     model,
                          const EngineParam&    engine,
                          const AttentionParam& attn,
                          const MoeParam&       moe,
                          const Context&        ctx,
                          const LlamaWeight&    weights,
                          int                   phases):
    dtype_{dtype},
    param_{model},
    attn_param_{attn},
    comm_{ctx.comm},
    weights_{weights},
    linear_{*ctx.linear},
    tp_size_{comm_.h_tp_group->n_ranks()},
    tp_rank_{comm_.h_tp_group->rank()},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)},
    speculative_decoding_{engine.speculative_decoding},
    debug_{isDebug()}
{

    false_ = {engine.max_batch_size, kDEVICE};
    Clear(false_);

    finished_buf_ = {engine.max_batch_size, kCPUpinned};
    finished_     = {{engine.max_batch_size}, kBool, kDEVICE};

    autoreg_ids_ = {engine.max_batch_size, kDEVICE};
    // autoreg_ids_offsets_ = {engine.max_batch_size + 1, kCPU};
    // std::fill_n(autoreg_ids_offsets_.data(), autoreg_ids_offsets_.size(), 0);

    sequence_length_buf_ = {engine.max_batch_size, kCPUpinned};
    sequence_length_     = {{engine.max_batch_size}, kInt, kDEVICE};
    for (int i = 0; i < phases; ++i) {
        auto& d           = data_.emplace_back();
        d.sequence_length = empty_like(sequence_length_buf_, kDEVICE);
        d.finished        = empty_like(finished_buf_, kDEVICE);
        d.autoregres      = {engine.max_batch_size, kCPU};
        d.generating      = {engine.max_batch_size, kCPU};
    }

    input_processor_.emplace(engine, param_, phases);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, ctx, phases);

    generation_ = std::make_unique<Generation>(kFloat32,
                                               engine.max_batch_size,
                                               engine.session_len,
                                               model.vocab_size,
                                               weights.post_decoder_embedding.output_dim * tp_size_,
                                               comm_.h_tp_group,
                                               phases);

    const int     vocab_size     = weights_.post_decoder_embedding.output_dim * tp_size_;
    const ssize_t max_fwd_tokens = engine.max_forward_token_num;

    if (ctx.comm.d_comm) {
        auto symm_alloc = GetSymmAllocator(ctx.comm.d_comm);
        // Native comm fuses allreduce & rmsnorm in token granularity
        TM_CHECK(engine.max_forward_token_num % tp_size_ == 0);

        ssize_t bytes{};
        bytes = std::max(bytes, byte_size(dtype_, max_fwd_tokens * engine.attn_dp_size * model.hidden_units));
        bytes = std::max(bytes, byte_size(dtype_, engine.max_batch_size * vocab_size));

        symm_buf_ = {bytes, symm_alloc};
        // Compute max logits length based on symm buffer size
        max_logits_len_ = symm_buf_.view(dtype_).size() / vocab_size;
    }
    else {
        max_logits_len_ = std::max<int>(max_fwd_tokens * model.hidden_units / vocab_size, engine.max_batch_size);
    }

    output_processor_.emplace(param_, max_logits_len_, tp_rank_, phases, [this](const Tensor& hstate) {
        return PostEmbedding(hstate, symm_buf_);
    });

    // Instantiate MTP predictor for speculative decoding (only when enabled and weights are available)
    if (speculative_decoding_ && param_.num_mtp_layers > 0 && !weights.mtp_layer_weights.empty()) {
        mtp_predictor_ = std::make_unique<MTPPredictor>(model, engine, attn, moe, ctx, weights);
        TM_LOG_INFO("[LanguageModel] MTPPredictor initialized with %d MTP layers, K=%d",
                    param_.num_mtp_layers,
                    param_.num_draft_tokens);
    }
}

Tensor LanguageModel::Impl::LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf)
{
    const auto st = core::Context::stream().handle();

    const int hidden_units = param_.hidden_units;

    const auto& embedding_table = weights_.pre_decoder_embedding.weight;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units);

    const int token_num = input_ids.size();

    Tensor input_embeds{{token_num, hidden_units}, dtype_, kDEVICE};

    if (token_num == 0) {
        return input_embeds;
    }

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, st);
        sync_check_cuda_error();
    }
    else if (use_ag2d_) {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(dtype_), {token_num, tp_size_, local_hidden_units}};
        Tensor local{temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        sync_check_cuda_error();

        comm_.d_comm->AllGather2D(local.raw_data(),
                                  temp.raw_data(),
                                  hidden_units,
                                  local_hidden_units,
                                  local_hidden_units,
                                  token_num,
                                  local.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        sync_check_cuda_error();

        Copy(temp.buffer(), input_embeds.buffer());
    }
    else {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(dtype_), {tp_size_, token_num, local_hidden_units}};
        Tensor local{temp.slice(tp_rank_).squeeze(0)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        sync_check_cuda_error();

        comm_.d_comm->AllGather(local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_.d_tp_group, st);
        sync_check_cuda_error();

        invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                  (uint16_t*)temp.raw_data(),
                                  tp_size_,
                                  token_num,
                                  local_hidden_units,
                                  false,
                                  st);
        sync_check_cuda_error();
    }

    return input_embeds;
}

Tensor LanguageModel::Impl::PostEmbedding(const Tensor& features, Buffer symm_buf)
{
    NvtxScope scope("postDecodeEmbedding");

    const auto st = core::Context::stream().handle();

    const int bsz              = features.shape(0);
    const int local_vocab_size = weights_.post_decoder_embedding.output_dim;
    const int vocab_size       = local_vocab_size * tp_size_;

    if (bsz == 0) {
        return Tensor{{0, vocab_size}, dtype_, kDEVICE};
    }

    if (tp_size_ == 1) {
        Tensor logits{{bsz, vocab_size}, dtype_, kDEVICE};
        linear_.Forward(features, weights_.post_decoder_embedding, logits);
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(logits, "logits", 1);
        return logits;
    }
    else if (use_ag2d_) {
        Tensor logits{symm_buf.view(dtype_), {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        linear_.Forward(features, weights_.post_decoder_embedding, local.squeeze(1));
        sync_check_cuda_error();
        comm_.d_comm->AllGather2D(local.raw_data(),
                                  logits.raw_data(),
                                  vocab_size,
                                  local_vocab_size,
                                  local_vocab_size,
                                  bsz,
                                  logits.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        sync_check_cuda_error();
        return logits.view({bsz, -1});
    }
    else {
        Tensor logits{symm_buf.view(dtype_), {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, weights_.post_decoder_embedding, local.squeeze(0));
        sync_check_cuda_error();
        comm_.d_comm->AllGather(local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, st);
        sync_check_cuda_error();
        Tensor out{{bsz, vocab_size}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, st);
        sync_check_cuda_error();
        return out;
    }
}

void LanguageModel::Impl::Setup(int phase, TensorMap& env)
{
    input_processor_->Run(BatchOp::kSetup, phase, env);

    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    const auto& rc = env.at("batch").data<BatchData*>()[0]->rc;

    d.n_generating = 0;

    for (int i = 0; i < rc.size(); ++i) {
        auto& c         = *rc[i];
        d.autoregres[i] = c.autoregres;
        d.generating[i] = c.generating;
        d.n_generating += c.generating;
        if (TM_UNLIKELY(!c.autoregres)) {
            sequence_length_buf_[i] = c.history_len + c.alpha + c.input_len;
        }
    }

    copy(sequence_length_buf_, rc.size(), d.sequence_length);

    unified_decoder_->Run(BatchOp::kSetup, phase, env);
    generation_->Run(BatchOp::kSetup, phase, env);
    output_processor_->Run(BatchOp::kSetup, phase, env);
}

void LanguageModel::Impl::Prepare(int phase, TensorMap& env)
{
    env.emplace("autoreg_ids", autoreg_ids_);

    input_processor_->Run(BatchOp::kPrepare, phase, env);

    auto& d = data_.at(phase);

    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    // core::CopyT copy{};

    if (auto group = copy.group()) {
        for (int i = 0; i < b.bsz; ++i) {
            if (const int j = b.perm[i]; j < b.bs0) {
                copy(finished_.front().data<bool>() + j, 1, finished_.back().data<bool>() + i);
            }
            else {
                copy(false_.data() + i, 1, finished_.back().data<bool>() + i);
            }
        }
        finished_.Swap();
    }

    if (auto group = copy.group()) {
        // sequence_length = history_len + input_len
        for (int i = 0; i < b.bsz; ++i) {
            if (const int j = b.perm[i]; j < b.bs0 && d.autoregres[i]) {
                copy(sequence_length_.front().data<int>() + j, 1, sequence_length_.back().data<int>() + i);
            }
            else {
                copy(d.sequence_length.data() + i, 1, sequence_length_.back().data<int>() + i);
            }
        }
        sequence_length_.Swap();
    }

    Buffer_<int> k_offsets{b.bsz + 1, kDEVICE};
    // PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), core::Context::stream().handle());

    // Buffer_<int> k_offsets_tmp{k_offsets.size(), kCPU};
    // Buffer_<int> sequence_length_tmp{sequence_length_.front().size(), kCPU};

    // Copy(k_offsets, k_offsets_tmp);
    // Copy(sequence_length_.front().buffer(), sequence_length_tmp);

    // core::Context::stream().Sync();

    // dbg(core::to_vector<int>(sequence_length_tmp.slice(0, bsz)));
    // dbg(core::to_vector<int>(k_offsets_tmp.slice(0, bsz + 1)));

    env.produce("finished", finished_.front());
    env.produce("sequence_length", sequence_length_.front());
    env.produce("k_offsets", k_offsets);

    copy.Run(); // Synchronize copies before metadata is consumed to prevent race conditions

    unified_decoder_->Run(BatchOp::kPrepare, phase, env);
    generation_->Run(BatchOp::kPrepare, phase, env);
    output_processor_->Run(BatchOp::kPrepare, phase, env);
}

void LanguageModel::Impl::Forward(int phase, TensorMap& env)
{

    auto& d = data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    // Detect steady-state speculative decoding: any request has draft tokens from previous round
    bool has_drafts = false;
    if (speculative_decoding_) {
        for (int i = 0; i < b.bsz; ++i) {
            if (b.rc[i]->num_drafts > 0) {
                has_drafts = true;
                break;
            }
        }
    }

    {
        Buffer_<int> k_offsets = env.at("k_offsets").buffer();
        PrefixSum(sequence_length_.front().data<int>(), b.bsz, k_offsets.data(), core::Context::stream().handle());
    }

    {  // compute input embeddings
        auto input_ids = env.at("input_ids").buffer();

        Tensor input_embeds = LookupEmbedding(input_ids, symm_buf_);
        TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];
        input_processor_->PatchEmbedding(phase, input_embeds, copy);
        copy.Run();

        env.produce("input_embeds", std::move(input_embeds));
        // dbg(env);
    }

    if (symm_buf_) {
        env.produce("symm_buf", symm_buf_);
    }

    env.produce("output_norm_weight", weights_.output_norm_weight);

    // When speculative decoding is enabled in steady-state (has_drafts), override
    // selected_token_pos to select K+1 positions per request (bonus + K drafts) instead
    // of just the last token. This avoids the expensive output_hidden_states path that
    // forces the decoder to output ALL tokens' hidden states (30% perf regression on >32K).
    // For first decode (!has_drafts), keep the default selected_token_pos (1 per request).
    if (has_drafts) {
        const int K   = param_.num_draft_tokens;
        const int bsz = b.bsz;
        auto q_offsets_dev = env.at("q_offsets").buffer();

        // Build selected_token_pos entirely on GPU: output[i*(K+1)+j] = q_offsets[i] + j
        const int select_count = bsz * (K + 1);
        Buffer_<int> spec_selected_pos_dev{select_count, kDEVICE};
        invokeBuildSpecSelectedPos(
            spec_selected_pos_dev.data(), q_offsets_dev.data<int>(), bsz, K, core::Context::stream().handle());
        env.at("selected_token_pos") = spec_selected_pos_dev;
    }

    // Step3p5 MTP consumes the un-normed residual, matching vLLM's
    // forward()/compute_logits split.
    if (speculative_decoding_ && d.n_generating) {
        env.produce("output_last_residual", Tensor{});
    }

    unified_decoder_->Forward(phase, env, weights_.decoder_layer_weights);

    // env.at("batch").data<BatchData*>()[0]->Notify();

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 2);

    auto& hidden_states = env.at("hidden_states");

    // Preserve hidden_states for DraftTokens (MTP needs the last accepted position's hidden state)
    if (speculative_decoding_ && d.n_generating) {
        // Save the un-normed residual for MTP. The normalized hidden_states are
        // still used for logits, but Step3p5's MTP stack expects the residual.
        auto* residual_ptr = env.try_("last_residual");
        if (residual_ptr && residual_ptr->raw_data()) {
            last_hidden_states_ = *residual_ptr;
        }
        else {
            last_hidden_states_ = hidden_states;
        }
    }

    // hidden_states is already the right shape:
    //   has_drafts=true:  [bsz*(K+1), hidden_size] (selected by overridden selected_token_pos)
    //   has_drafts=false: [bsz, hidden_size] (default selected_token_pos, 1 per request)
    env.produce("logits", PostEmbedding(hidden_states, symm_buf_));

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 1);

    if (has_drafts) {
        // Steady-state spec decode: skip Generation::Forward (no sampling).
        // Rollback will handle seq_len updates and token_ids writes.
    }
    else if (d.n_generating) {
        // First decode or non-spec-decode: normal sampling
        generation_->Run(BatchOp::kForward, phase, env);
        Copy(env.at("output_ids").buffer(), autoreg_ids_);
    }
}

void LanguageModel::Impl::DraftTokens(int phase, TensorMap& env)
{
    if (!mtp_predictor_) {
        TM_LOG_ERROR("[LanguageModel] DraftTokens called but mtp_predictor_ is null");
        return;
    }

    const auto st = core::Context::stream().handle();
    auto& b       = *env.at("batch").data<BatchData*>()[0];
    const int K   = param_.num_draft_tokens;
    const int bsz = b.bsz;

    // Set up MTP attention layer dispatch data.
    // Use operator[] instead of produce() because keys may already exist from Forward/Reject.
    {
        BatchCopy copy{};
        env["copy"] = copy.buf();

        // MTP processes 1 token per request (decode-like), but the main model's rc may have
        // input_len=2 (bonus + draft) from the steady-state setup. Temporarily override to
        // decode parameters for MTP attention setup.
        std::vector<std::pair<int, bool>> saved(bsz);
        for (int i = 0; i < bsz; ++i) {
            auto& c = *b.rc[i];
            saved[i] = {c.input_len, c.autoregres};
            c.input_len = 1;
            c.autoregres = true;
        }

        // For decode-only batch: q_offsets = [0, 1, 2, ..., bsz], k_offsets from sequence_length
        Buffer_<int> q_offsets{bsz + 1, kDEVICE};
        {
            Buffer_<int> q_off_host{bsz + 1, kCPU};
            for (int i = 0; i <= bsz; ++i) {
                q_off_host[i] = i;
            }
            Copy(q_off_host, q_offsets);
        }
        env["q_offsets"] = q_offsets;

        // k_offsets: prefix sum of sequence_length
        Buffer_<int> k_offsets{bsz + 1, kDEVICE};
        PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), st);
        env["k_offsets"] = k_offsets;

        // finished: all false (decode requests are not finished)
        env["finished"] = finished_.front();

        mtp_predictor_->SetupAttention(0, env);
        copy.Run();

        // Restore original values
        for (int i = 0; i < bsz; ++i) {
            auto& c = *b.rc[i];
            c.input_len = saved[i].first;
            c.autoregres = saved[i].second;
        }
    }

    // Select the correct hidden state for each request from last_hidden_states_.
    // In steady-state (has_drafts), last_hidden_states_ = [bsz*(K+1), hidden_size]
    // with request i's K+1 positions at indices i*(K+1)..i*(K+1)+K.
    // We need the hidden state at the last accepted position (bonus + N accepted drafts).
    // In first decode, last_hidden_states_ = [bsz, hidden_size] (already correct).
    Tensor hidden_states;
    {
        auto* num_accepted_ptr = env.try_("num_accepted");

        if (num_accepted_ptr && last_hidden_states_.shape(0) > bsz) {
            // Steady-state: select hidden state at the last accepted position using GPU gather.
            // last_hidden_states_ layout: [bsz*(K+1), hidden_size]
            // Request i's K+1 positions are at indices i*(K+1)..i*(K+1)+K
            // Position 0 = bonus, position N = last accepted draft
            auto num_accepted_buf = num_accepted_ptr->buffer();

            hidden_states = Tensor{{bsz, (int)param_.hidden_units}, dtype_, kDEVICE};
            invokeGatherHiddenByAccepted(hidden_states.raw_data(),
                                         last_hidden_states_.raw_data(),
                                         num_accepted_buf.data<int>(),
                                         bsz,
                                         K,
                                         param_.hidden_units,
                                         dtype_,
                                         st);
        }
        else {
            // First decode: last_hidden_states_ is already [bsz, hidden_size]
            hidden_states = last_hidden_states_;
        }
    }

    // Call MTPPredictor::Draft to generate K draft tokens
    Buffer_<int> last_tokens{bsz, kDEVICE};
    cudaMemcpyAsync(last_tokens.data(), autoreg_ids_.data(), bsz * sizeof(int), cudaMemcpyDeviceToDevice, st);

    auto result = mtp_predictor_->Draft(bsz, hidden_states, last_tokens, K, env);

    // Save draft tokens to RequestCache for next iteration's Schedule to inject
    // result.draft_tokens layout is [K, bsz]: draft_tokens[k * bsz + i] = k-th draft for request i
    {
        Buffer_<int> draft_host{K * bsz, kCPU};
        Copy(Buffer{result.draft_tokens.data(), K * bsz, kDEVICE}, draft_host);
        core::Context::stream().Sync();

        for (int i = 0; i < bsz; ++i) {
            auto& c = *b.rc[i];
            c.num_drafts = K;
            for (int k = 0; k < K; ++k) {
                c.draft_tokens[k] = draft_host[k * bsz + i];
            }
        }
    }
}

void LanguageModel::Impl::RejectDrafts(int phase, TensorMap& env)
{
    if (!mtp_predictor_) {
        TM_LOG_ERROR("[LanguageModel] RejectDrafts called but mtp_predictor_ is null");
        return;
    }

    const auto st = core::Context::stream().handle();
    auto& b       = *env.at("batch").data<BatchData*>()[0];
    const int K   = param_.num_draft_tokens;
    const int bsz = b.bsz;

    // Read logits from Forward output.
    // In Verify-in-Next-Forward mode, logits are already in [bsz*(K+1), vocab_size] format
    // (Forward used overridden selected_token_pos to select K+1 positions per request).
    auto& logits = env.at("logits");

    // vocab_size from the post_decoder_embedding weight
    const int vocab_size  = weights_.post_decoder_embedding.output_dim * tp_size_;
    const auto logit_bytes = byte_size(dtype_, vocab_size);

    // Logits are already [bsz*(K+1), vocab_size] — use directly as verify_logits.
    auto& verify_logits = logits;

    // Get draft tokens from RequestCache (saved by previous round's DraftTokens).
    // Layout: [bsz, K] for GreedyReject.
    Buffer_<int> draft_tokens_bk{bsz * K, kDEVICE};
    {
        Buffer_<int> draft_host{bsz * K, kCPU};
        for (int i = 0; i < bsz; ++i) {
            for (int k = 0; k < K; ++k) {
                draft_host[i * K + k] = b.rc[i]->draft_tokens[k];
            }
        }
        Copy(draft_host, draft_tokens_bk);
    }

    // verify_logits layout: [bsz * (K+1), vocab_size]
    // GreedyReject expects [batch, K+1, vocab_size] which is the same flat layout
    auto result = GreedyReject(verify_logits.raw_data(),
                               draft_tokens_bk.data(),
                               bsz,
                               K,
                               vocab_size,
                               dtype_,
                               st);

    // Produce num_accepted and bonus_tokens into env for kRollback
    env.produce("num_accepted", result.num_accepted);
    env.produce("bonus_tokens", result.bonus_tokens);

    // Update autoreg_ids_ with bonus tokens (the next token to generate)
    cudaMemcpyAsync(autoreg_ids_.data(),
                    result.bonus_tokens.data(),
                    bsz * sizeof(int),
                    cudaMemcpyDeviceToDevice,
                    st);
}

void LanguageModel::Impl::Rollback(int phase, TensorMap& env)
{
    // Verify-in-Next-Forward Rollback:
    //
    // In the new flow, Forward processed K+1 tokens (bonus + K drafts) per request
    // but SKIPPED Generation::Forward (no sampling, no seq_len += 1).
    //
    // sequence_length_.front() = history_len + input_len = history_len + 1 + K
    // (the full KV cache length after Forward wrote all K+1 tokens' KV entries)
    //
    // After rejection with N accepted drafts (0 ≤ N ≤ K):
    //   - Tokens to keep: bonus (argmax of logits[0]) + N accepted drafts
    //   - New sequence_length = history_len + 1 (bonus) + N (accepted drafts)
    //   - KV cache: positions history_len..history_len+N are valid, rest are stale
    //     (stale entries will be overwritten by next Forward)
    //
    // Token writing:
    //   - bonus token goes at c.token_ids[c.seq_len] (where c.seq_len is the pre-Forward value)
    //   - accepted draft tokens go at c.token_ids[c.seq_len+1], ..., c.token_ids[c.seq_len+N]
    //   - Update will set c.seq_len = sequence_length[i] and compute new_tokens
    //
    // GDN state:
    //   - Snapshot was taken before Forward (S_prev).
    //   - Live state = S_prev + bonus + all_drafts (after Forward).
    //   - N=K (all accepted): keep live state, discard snapshot.
    //   - N<K (partial reject): restore S_prev from snapshot.
    //     Next Forward will re-process bonus + accepted drafts + new drafts from S_prev.

    auto& b       = *env.at("batch").data<BatchData*>()[0];
    auto& d       = data_.at(phase);
    const int K   = param_.num_draft_tokens;
    const auto st = core::Context::stream().handle();

    // Read num_accepted and bonus_tokens from env (produced by kReject)
    auto num_accepted_buf = env.at("num_accepted").buffer();
    auto bonus_tokens_buf = env.at("bonus_tokens").buffer();

    Buffer_<int> num_accepted_host{b.bsz, kCPU};
    Buffer_<int> bonus_tokens_host{b.bsz, kCPU};
    Copy(num_accepted_buf.slice(0, b.bsz), num_accepted_host);
    Copy(bonus_tokens_buf.slice(0, b.bsz), bonus_tokens_host);

    // Write tokens and compute new sequence_length
    Buffer_<int> new_seq_len{b.bsz, kCPU};
    // Read current sequence_length from device (set by Forward/Prepare) for non-generating requests
    Buffer_<int> cur_seq_len{b.bsz, kCPU};
    Copy(sequence_length_.front().buffer().slice(0, b.bsz), cur_seq_len);

    // Single sync for all three D2H copies above
    core::Context::stream().Sync();

    for (int i = 0; i < b.bsz; ++i) {
        auto& c = *b.rc[i];

        if (!d.generating[i]) {
            // Non-generating request (chunked prefill): keep sequence_length from Forward unchanged
            new_seq_len[i] = cur_seq_len[i];
            continue;
        }

        const int N = num_accepted_host[i];
        const int old_seq_len = c.seq_len;  // pre-Forward value

        // In Verify-in-Next-Forward flow:
        // Schedule already injected draft tokens at token_ids[old_seq_len..old_seq_len+K-1].
        // The first N drafts were accepted (verified by RejectDrafts).
        // These are already in the correct positions — no need to re-write them.
        // The bonus token (next token after last accepted) goes at old_seq_len + N.
        c.token_ids[old_seq_len + N] = bonus_tokens_host[i];

        // New sequence_length = old_seq_len + N (accepted drafts) + 1 (bonus)
        new_seq_len[i] = old_seq_len + N + 1;

        // Save rejection results to RequestCache for next iteration
        c.all_accepted = (N == K);
    }

    // Write new sequence_length to device
    Copy(Buffer{new_seq_len.data(), b.bsz, kCPU}, sequence_length_.front().buffer().slice(0, b.bsz));

    // Sync data_[phase].sequence_length so kFetch/Update see the correct value.
    // kUnprep already ran (before spec decode) and set d.sequence_length to the old value.
    // We must overwrite it with the post-rollback value.
    Copy(sequence_length_.front().buffer().slice(0, b.bsz), d.sequence_length.slice(0, b.bsz));

    // Stop criteria check: Generation::Forward is skipped in steady-state spec decode,
    // so stop_criteria_->Forward() never runs. We must check length limit and EOS here.
    {
        // Length criterion: check on host (we already have new_seq_len and max_seq_len)
        Buffer_<bool> finished_host{b.bsz, kCPU};
        Copy(finished_.front().buffer().slice(0, b.bsz), finished_host);
        core::Context::stream().Sync();

        bool any_finished = false;
        for (int i = 0; i < b.bsz; ++i) {
            if (!d.generating[i]) continue;  // Skip non-generating requests
            auto& c = *b.rc[i];
            // Length criterion
            if (new_seq_len[i] >= c.max_seq_len) {
                finished_host[i] = true;
                any_finished = true;
            }
            // EOS criterion: check bonus token and accepted draft tokens
            if (!finished_host[i]) {
                const auto& eos_ids = c.gen_cfg.eos_ids;
                if (!eos_ids.empty()) {
                    const int N = num_accepted_host[i];
                    // Check bonus token
                    int bonus = bonus_tokens_host[i];
                    for (int eid : eos_ids) {
                        if (bonus == eid) {
                            finished_host[i] = true;
                            any_finished = true;
                            break;
                        }
                    }
                    // Check accepted draft tokens
                    if (!finished_host[i]) {
                        for (int k = 0; k < N && !finished_host[i]; ++k) {
                            int tok = c.draft_tokens[k];
                            for (int eid : eos_ids) {
                                if (tok == eid) {
                                    finished_host[i] = true;
                                    any_finished = true;
                                    // Truncate seq_len to exclude tokens after EOS
                                    // Drafts at old_seq_len..old_seq_len+N-1, bonus at old_seq_len+N
                                    // EOS at draft[k] means seq_len = old_seq_len + k + 1
                                    int truncated = c.seq_len + k + 1;
                                    if (truncated < new_seq_len[i]) {
                                        new_seq_len[i] = truncated;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (any_finished) {
            // Write updated finished flags and possibly truncated seq_len back to device
            Copy(Buffer{finished_host.data(), b.bsz, kCPU}, finished_.front().buffer().slice(0, b.bsz));
            Copy(Buffer{new_seq_len.data(), b.bsz, kCPU}, sequence_length_.front().buffer().slice(0, b.bsz));
            Copy(sequence_length_.front().buffer().slice(0, b.bsz), d.sequence_length.slice(0, b.bsz));
        }

        // Also sync finished to data_[phase].finished (kUnprep already ran)
        Copy(finished_.front().buffer().slice(0, b.bsz), d.finished.slice(0, b.bsz));
    }

    // GDN state: no snapshot/restore needed.
    // The live GDN state (S_prev + bonus + drafts) is kept as-is.
    // Rejected drafts' contributions remain as small noise in the recurrent state.
    // This avoids the critical bug where restoring S_prev loses the bonus token's
    // GDN contribution, causing output degradation after consecutive rejections.
}

void LanguageModel::Impl::Unprep(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(sequence_length_.front().buffer(), d.sequence_length.size(), d.sequence_length);

    copy(finished_.front().buffer(), d.finished.size(), d.finished);

    generation_->Run(BatchOp::kUnprep, phase, env);
}

void LanguageModel::Impl::Fetch(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(d.sequence_length, d.sequence_length.size(), sequence_length_buf_);
    env.produce("sequence_length", sequence_length_buf_);

    copy(d.finished, d.finished.size(), finished_buf_);
    env.produce("finished", finished_buf_);

    env.produce("generating", d.generating);

    generation_->Run(BatchOp::kFetch, phase, env);
}

LanguageModel::~LanguageModel() = default;

LanguageModel::LanguageModel(LanguageModel&&) noexcept = default;

LanguageModel::LanguageModel(DataType              dtype,
                             const ModelParam&     model,
                             const EngineParam&    engine,
                             const AttentionParam& attn,
                             const MoeParam&       moe,
                             const Context&        ctx,
                             const LlamaWeight&    weights,
                             int                   phases)
{
    impl_ = std::make_unique<Impl>(dtype, model, engine, attn, moe, ctx, weights, phases);
}

void LanguageModel::Run(BatchOp op, int phase, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Run(op, phase, env);
}

const ModelParam& LanguageModel::model_param() const noexcept
{
    return TM_CHECK_NOTNULL(impl_)->param_;
}

const AttentionParam& LanguageModel::attn_param() const noexcept
{
    return TM_CHECK_NOTNULL(impl_)->attn_param_;
}

}  // namespace turbomind

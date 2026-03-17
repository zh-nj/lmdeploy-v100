/*
 * Copyright (c) OpenMMLab. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/turbomind/models/llama/mtp_predictor.h"

#include <algorithm>
#include <numeric>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

// MTP_PROFILE: compile-time switch for per-step timing in ForwardStep
// NOTE: Per-step timing uses cudaStreamSynchronize which is INCOMPATIBLE with TP>1
// Only use for TP=1 debugging. For TP>1, use pipeline-level profiling only.
#ifndef MTP_PROFILE
#define MTP_PROFILE 0
#endif

// MTP_PROFILE_PIPELINE: pipeline-level profiling (safe for TP>1)
// Enabled by MTP_PROFILE but uses deferred sync at iteration boundary
#ifndef MTP_PROFILE_PIPELINE
#define MTP_PROFILE_PIPELINE MTP_PROFILE
#endif

#if MTP_PROFILE
#include <cstdio>
#include <chrono>
static FILE* g_mtp_profile_fp   = nullptr;
static int   g_mtp_profile_iter = 0;
static int   g_mtp_layer_idx;
static std::chrono::high_resolution_clock::time_point g_mtp_tp;

static void mtp_profile_init()
{
    if (!g_mtp_profile_fp) {
        g_mtp_profile_fp = fopen("/tmp/mtp_profile.csv", "w");
        if (g_mtp_profile_fp) {
            fprintf(g_mtp_profile_fp, "iter,layer,step,ms\n");
        }
    }
}

// These macros use cudaStreamSynchronize — ONLY safe for TP=1
#define MTP_PROFILER_INIT() g_mtp_layer_idx = mtp_layer_idx
#define MTP_TIMER_BEGIN(name) \
    do { cudaStreamSynchronize(st); g_mtp_tp = std::chrono::high_resolution_clock::now(); } while(0)
#define MTP_TIMER_END(name) \
    do { \
        cudaStreamSynchronize(st); \
        auto _now = std::chrono::high_resolution_clock::now(); \
        double _ms = std::chrono::duration<double, std::milli>(_now - g_mtp_tp).count(); \
        mtp_profile_init(); \
        if (g_mtp_profile_fp) { \
            fprintf(g_mtp_profile_fp, "%d,%d,%s,%.4f\n", g_mtp_profile_iter, g_mtp_layer_idx, #name, _ms); \
        } \
    } while(0)
#define MTP_PROFILER_FLUSH() \
    do { if (g_mtp_profile_fp) fflush(g_mtp_profile_fp); } while(0)
#define MTP_ITER_INC() (++g_mtp_profile_iter)

#else
#define MTP_PROFILER_INIT()   ((void)0)
#define MTP_TIMER_BEGIN(name) ((void)0)
#define MTP_TIMER_END(name)   ((void)0)
#define MTP_PROFILER_FLUSH()  ((void)0)
#define MTP_ITER_INC()        ((void)0)
#endif

namespace turbomind {

MTPPredictor::MTPPredictor(const ModelParam&     model,
                           const EngineParam&    engine,
                           const AttentionParam& attn,
                           const MoeParam&       moe,
                           const Context&        ctx,
                           const LlamaWeight&    weights):
    param_{model},
    norm_eps_{model.norm_eps},
    dtype_{model.data_type},
    hidden_units_{static_cast<int>(model.hidden_units)},
    tp_size_{ctx.comm.h_tp_group->n_ranks()},
    tp_rank_{ctx.comm.h_tp_group->rank()},
    mtp_attn_layer_offset_{[&] {
        // MTP attention layer indices start after main model's attention layers
        // Only count main model layers (layer_num), not MTP layers appended to the array
        if (model.layer_types.empty()) {
            return static_cast<int>(model.layer_num);
        }
        auto end = model.layer_types.begin()
                   + std::min(model.layer_types.size(), static_cast<size_t>(model.layer_num));
        return static_cast<int>(std::count(model.layer_types.begin(), end, 0));
    }()},
    linear_{*ctx.linear},
    weights_{weights},
    comm_{ctx.comm},
    d_comm_{ctx.comm.d_comm}
{
    TM_LOG_INFO("[MTPPredictor] Initializing with hidden_units=%d, tp_size=%d, mtp_attn_layer_offset=%d",
                hidden_units_,
                tp_size_,
                mtp_attn_layer_offset_);

    // Create MTP's own attention layer instance.
    // The MTP decoder layer is a full-attention layer, so we create a new UnifiedAttentionLayer.
    // `init=true` triggers internal buffer allocation; `phases=1` since MTP only runs in decode phase.
    const bool has_moe = std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL) > 0;
    attn_layer_ = std::make_unique<UnifiedAttentionLayer>(model, attn, engine, engine.attn_tp_size, ctx, 1, has_moe);

    // Create MTP's own MoE FFN layer instance
    if (has_moe) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
    }

    // Create shared expert FFN layer if the model has inter_size (shared expert)
    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL) > 0) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }
}

MTPPredictor::~MTPPredictor() = default;

void MTPPredictor::SetupAttention(int phase, TensorMap& env)
{
    attn_layer_->Run(BatchOp::kSetup, 0, env);
    attn_layer_->Run(BatchOp::kPrepare, 0, env);
}

Tensor MTPPredictor::LookupEmbedding(const Buffer_<int>& token_ids, int batch_size)
{
    const auto st = core::Context::stream().handle();

    const auto& embedding_table = weights_.pre_decoder_embedding.weight;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units_);

    Tensor input_embeds{{batch_size, hidden_units_}, dtype_, kDEVICE};

    if (batch_size == 0) {
        return input_embeds;
    }

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(input_embeds, token_ids, embedding_table, st);
        sync_check_cuda_error();
    }
    else {
        // TP > 1: each rank has a shard of the embedding table.
        // AllGather the shards to reconstruct the full embedding.
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{{tp_size_, batch_size, local_hidden_units}, dtype_, kDEVICE};
        Tensor local = temp.slice(tp_rank_).squeeze(0);

        invokeEmbeddingLookup(local, token_ids, embedding_table, st);
        sync_check_cuda_error();

        d_comm_->AllGather(
            local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_.d_tp_group, st);
        sync_check_cuda_error();

        invokeTransposeAxis01(
            (uint16_t*)input_embeds.raw_data(), (uint16_t*)temp.raw_data(), tp_size_, batch_size, local_hidden_units, st);
        sync_check_cuda_error();
    }

    return input_embeds;
}

Tensor MTPPredictor::PostEmbedding(const Tensor& features, int batch_size)
{
    const auto st = core::Context::stream().handle();

    const int local_vocab_size = weights_.post_decoder_embedding.output_dim;
    const int vocab_size       = local_vocab_size * tp_size_;

    if (batch_size == 0) {
        return Tensor{{0, vocab_size}, dtype_, kDEVICE};
    }

    if (tp_size_ == 1) {
        Tensor logits{{batch_size, vocab_size}, dtype_, kDEVICE};
        linear_.Forward(features, weights_.post_decoder_embedding, logits);
        sync_check_cuda_error();
        return logits;
    }
    else {
        Tensor logits{{tp_size_, batch_size, local_vocab_size}, dtype_, kDEVICE};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, weights_.post_decoder_embedding, local.squeeze(0));
        sync_check_cuda_error();

        d_comm_->AllGather(
            local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, st);
        sync_check_cuda_error();

        Tensor out{{batch_size, vocab_size}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, batch_size, local_vocab_size, st);
        sync_check_cuda_error();
        return out;
    }
}

Tensor MTPPredictor::PostEmbeddingLocal(const Tensor& features, int batch_size, const LlamaDenseWeight& lm_head)
{
    const auto st = core::Context::stream().handle();

    const int local_vocab_size = lm_head.output_dim;
    const int vocab_size       = local_vocab_size * tp_size_;

    if (batch_size == 0) {
        return Tensor{{0, vocab_size}, dtype_, kDEVICE};
    }

    if (tp_size_ == 1) {
        Tensor logits{{batch_size, vocab_size}, dtype_, kDEVICE};
        linear_.Forward(features, lm_head, logits);
        sync_check_cuda_error();
        return logits;
    }
    else {
        Tensor logits{{tp_size_, batch_size, local_vocab_size}, dtype_, kDEVICE};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, lm_head, local.squeeze(0));
        sync_check_cuda_error();

        d_comm_->AllGather(
            local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, st);
        sync_check_cuda_error();

        Tensor out{{batch_size, vocab_size}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, batch_size, local_vocab_size, st);
        sync_check_cuda_error();
        return out;
    }
}

Buffer_<int> MTPPredictor::Argmax(const Tensor& logits, int batch_size)
{
    const auto st = core::Context::stream().handle();

    const int vocab_size = logits.shape(1);

    Buffer_<int> output_ids{batch_size, kDEVICE};

    if (batch_size > 0) {
        invokeArgmax(output_ids.data(), logits.raw_data(), batch_size, vocab_size, dtype_, st);
        sync_check_cuda_error();
    }

    return output_ids;
}

ForwardStepResult MTPPredictor::ForwardStep(int           mtp_layer_idx,
                                       int           batch_size,
                                       const Tensor& prev_embedding,
                                       const Tensor& hidden_states,
                                       int           step_idx,
                                       TensorMap&    env)
{
    const auto st = core::Context::stream().handle();

    TM_CHECK(mtp_layer_idx < (int)weights_.mtp_layer_weights.size());
    const auto& mtp_w = *weights_.mtp_layer_weights[mtp_layer_idx];

    MTP_PROFILER_INIT();

    // Step 1-2: Dual RMSNorm
    MTP_TIMER_BEGIN(dual_rmsnorm);
    Tensor normed_emb{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normed_emb, prev_embedding, mtp_w.pre_fc_norm_embedding, norm_eps_, st);
    sync_check_cuda_error();

    Tensor normed_hidden{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normed_hidden, hidden_states, mtp_w.pre_fc_norm_hidden, norm_eps_, st);
    sync_check_cuda_error();
    MTP_TIMER_END(dual_rmsnorm);

    // Step 3: Concatenate [normed_emb, normed_hidden] → [batch, hidden_size*2]
    // We allocate a fused buffer and copy both halves into it.
    Tensor fused{{batch_size, hidden_units_ * 2}, dtype_, kDEVICE};
    {
        // Copy normed_emb to first half, normed_hidden to second half
        const size_t half_bytes = byte_size(dtype_, (size_t)batch_size * hidden_units_);
        cudaMemcpyAsync(fused.raw_data(), normed_emb.raw_data(), half_bytes, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync((char*)fused.raw_data() + half_bytes,
                        normed_hidden.raw_data(),
                        half_bytes,
                        cudaMemcpyDeviceToDevice,
                        st);
    }

    // Step 4: fc projection (ColumnParallelLinear with gather_output=True)
    // fc weight: [hidden_size/tp, hidden_size*2] → local output [batch, hidden_size/tp]
    // Then all-gather to get [batch, hidden_size]
    MTP_TIMER_BEGIN(fc_proj);
    Tensor projected;
    if (tp_size_ == 1) {
        projected = Tensor{{batch_size, hidden_units_}, dtype_, kDEVICE};
        linear_.Forward(fused, mtp_w.fc, projected);
        sync_check_cuda_error();
    }
    else {
        const int local_hidden = mtp_w.fc.output_dim;  // hidden_size / tp
        Tensor gathered{{tp_size_, batch_size, local_hidden}, dtype_, kDEVICE};
        Tensor local = gathered.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(fused, mtp_w.fc, local.squeeze(0));
        sync_check_cuda_error();

        d_comm_->AllGather(
            local.raw_data(), gathered.raw_data(), local.size(), dtype_, comm_.d_tp_group, st);
        sync_check_cuda_error();

        projected = Tensor{{batch_size, hidden_units_}, dtype_, kDEVICE};
        invokeTransposeAxis01(
            (uint16_t*)projected.raw_data(), (uint16_t*)gathered.raw_data(), tp_size_, batch_size, local_hidden, st);
        sync_check_cuda_error();
    }

    // Step 5: Attention layer (MTP's own KV cache layer)
    MTP_TIMER_END(fc_proj);
    const auto& decoder_w = *mtp_w.decoder_layer;

    // Pre-attention RMSNorm
    MTP_TIMER_BEGIN(attn_norm);
    Tensor residual = projected;
    Tensor attn_input{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(attn_input, projected, decoder_w.self_attn_norm, norm_eps_, st);
    sync_check_cuda_error();
    MTP_TIMER_END(attn_norm);

    // Attention forward
    MTP_TIMER_BEGIN(attention);
    const int mtp_kv_layer_idx = mtp_attn_layer_offset_ + mtp_layer_idx;
    attn_layer_->Forward({0,  // phase=0 (decode)
                          attn_input,
                          attn_input,
                          decoder_w.self_attn_weights.get(),
                          mtp_kv_layer_idx});

    // Post-attention: allreduce + residual + RMSNorm (pre-FFN norm)
    // attn_input now contains the attention output
    MTP_TIMER_END(attention);
    MTP_TIMER_BEGIN(post_attn_allreduce);
    const Tensor& attn_output_bias = decoder_w.self_attn_weights->output.bias;
    if (d_comm_) {
        d_comm_->AllreduceResidualBiasRMSnorm(attn_input.raw_data(),
                                              residual.data_or((void*)nullptr),
                                              attn_output_bias.data_or((void*)nullptr),
                                              decoder_w.ffn_norm.raw_data(),
                                              norm_eps_,
                                              hidden_units_,
                                              batch_size,
                                              dtype_,
                                              0,
                                              st);
        sync_check_cuda_error();
    }
    else {
        invokeResidualBiasRMSNorm(attn_input.raw_data(),
                                  residual.data_or((void*)nullptr),
                                  decoder_w.ffn_norm.raw_data(),
                                  attn_output_bias.data_or((void*)nullptr),
                                  dtype_,
                                  hidden_units_,
                                  batch_size,
                                  norm_eps_,
                                  st);
        sync_check_cuda_error();
    }

    // Step 6: MoE FFN
    // After the fused allreduce+residual+rmsnorm:
    //   attn_input = RMSNorm(residual + allreduce(attn_out) + bias, ffn_norm)
    //   residual = residual + allreduce(attn_out) + bias
    // Now attn_input is the FFN input (normed), residual holds the running residual.
    MTP_TIMER_END(post_attn_allreduce);

    MTP_TIMER_BEGIN(moe_ffn);
    std::optional<MoeFfnLayer::ForwardParam> moe_fwd_param;

    if (decoder_w.moe_weights) {
        moe_fwd_param = MoeFfnLayer::ForwardParam{
            attn_input, attn_input, decoder_w.moe_weights.get(), ffn_layer_ ? 1.f : 0.f, 0};
        moe_ffn_layer_->Forward(*moe_fwd_param);
    }

    if (decoder_w.ffn_weights) {
        ffn_layer_->forward({attn_input, attn_input, decoder_w.ffn_weights.get(), 0});
    }

    if (moe_fwd_param) {
        moe_ffn_layer_->Combine(*moe_fwd_param);
    }

    // Post-FFN: allreduce + residual + final RMSNorm
    MTP_TIMER_END(moe_ffn);
    MTP_TIMER_BEGIN(post_ffn_allreduce);
    if (d_comm_) {
        d_comm_->AllreduceResidualBiasRMSnorm(attn_input.raw_data(),
                                              residual.data_or((void*)nullptr),
                                              nullptr,  // no bias
                                              mtp_w.final_norm.raw_data(),
                                              norm_eps_,
                                              hidden_units_,
                                              batch_size,
                                              dtype_,
                                              0,
                                              st);
        sync_check_cuda_error();
    }
    else {
        invokeResidualBiasRMSNorm(attn_input.raw_data(),
                                  residual.data_or((void*)nullptr),
                                  mtp_w.final_norm.raw_data(),
                                  nullptr,  // no bias
                                  dtype_,
                                  hidden_units_,
                                  batch_size,
                                  norm_eps_,
                                  st);
        sync_check_cuda_error();
    }

    // Step 7: lm_head → logits
    MTP_TIMER_END(post_ffn_allreduce);
    MTP_TIMER_BEGIN(lm_head);
    Tensor logits;
    if (mtp_w.has_shared_head && mtp_w.shared_head_output) {
        // Step3p5: per-layer shared_head (norm + output)
        // The Post-FFN allreduce+residual+RMSNorm above used mtp_w.final_norm.
        // For Step3p5, final_norm IS shared_head.norm (same weight).
        // Now apply per-layer lm_head using shared_head_output.
        logits = PostEmbeddingLocal(attn_input, batch_size, *mtp_w.shared_head_output);
    }
    else {
        // Qwen3.5: shared main model lm_head
        logits = PostEmbedding(attn_input, batch_size);
    }

    // Step 8: argmax → draft token IDs
    MTP_TIMER_END(lm_head);
    MTP_TIMER_BEGIN(argmax);
    auto draft_ids = Argmax(logits, batch_size);
    MTP_TIMER_END(argmax);

    MTP_PROFILER_FLUSH();

    return ForwardStepResult{std::move(draft_ids), residual};
}

MTPPredictor::DraftResult MTPPredictor::Draft(int                 batch_size,
                                               const Tensor&       hidden_states,
                                               const Buffer_<int>& last_tokens,
                                               int                 num_draft_tokens,
                                               TensorMap&          env)
{
    TM_CHECK_GT(num_draft_tokens, 0);
    TM_CHECK_GT(batch_size, 0);
    TM_CHECK(!weights_.mtp_layer_weights.empty());

    const int num_mtp_layers = static_cast<int>(weights_.mtp_layer_weights.size());

    DraftResult result;
    result.draft_tokens = Buffer_<int>{num_draft_tokens * batch_size, kDEVICE};
    result.num_drafts   = num_draft_tokens;

    MTP_ITER_INC();

    // Initial embedding: lookup the last accepted tokens
    Tensor prev_embedding = LookupEmbedding(last_tokens, batch_size);

    Tensor current_hidden = hidden_states;  // Start with main model output (un-normed residual)

    for (int k = 0; k < num_draft_tokens; ++k) {
        // Use MTP layer index = k % num_mtp_layers (allows reusing layers for K > num_mtp_layers)
        const int mtp_layer_idx = k % num_mtp_layers;

        // ForwardStep produces draft token IDs and output hidden_states for this step
        auto [step_tokens, output_hidden] = ForwardStep(mtp_layer_idx, batch_size, prev_embedding, current_hidden, k, env);

        // Copy step tokens into the result buffer at position k
        const auto st = core::Context::stream().handle();
        cudaMemcpyAsync(result.draft_tokens.data() + k * batch_size,
                        step_tokens.data(),
                        batch_size * sizeof(int),
                        cudaMemcpyDeviceToDevice,
                        st);

        // Chain hidden_states for multi-layer MTP (Step3p5: 3 independent layers)
        // For single-layer MTP (Qwen3.5), current_hidden stays as main model output
        if (num_mtp_layers > 1) {
            current_hidden = output_hidden;
        }

        // Prepare embedding for next step: lookup the draft tokens we just produced
        if (k < num_draft_tokens - 1) {
            prev_embedding = LookupEmbedding(step_tokens, batch_size);
        }
    }

    return result;
}

}  // namespace turbomind

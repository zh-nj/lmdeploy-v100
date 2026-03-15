
#include "src/turbomind/engine/model_executor.h"

#include <memory>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/anomaly_handler.h"

// #include "dbg.h"

namespace turbomind {

using std::shared_ptr;
using std::unique_ptr;

struct ModelExecutor::Impl {

    LanguageModel& model_;
    LlamaLinear&   linear_;

    const int device_id_;

    Queue<unique_ptr<BatchData>>& inbound_;
    Queue<unique_ptr<BatchData>>& outbound_;

    std::thread internal_thread_;

    static bool HasDrafts(const BatchData& d)
    {
        for (const auto& c : d.rc) {
            if (c && c->num_drafts > 0) {
                return true;
            }
        }
        return false;
    }

    static bool HasDecodeRequests(const BatchData& d)
    {
        for (const auto& c : d.rc) {
            if (c && c->autoregres) {
                return true;
            }
        }
        return false;
    }

    void InternalThreadEntry()
    {
        check_cuda_error(cudaSetDevice(device_id_));

        Stream    stream  = Stream::create();
        Allocator h_alloc = Allocator(kCPU);
        Allocator d_alloc = Allocator(stream, false);

        AnomalyHandler::instance().Init(0, 1000, 0, 1000, stream.handle());

        core::ContextGuard ctx{stream, h_alloc, d_alloc};

        unique_ptr<BatchData> d;

        while (inbound_.pop(d)) {
            TM_CHECK_NOTNULL(d);
            core::Context::stream().Wait(d->ready);

            // Check if this is a steady-state speculative decode iteration
            // (batch has requests with draft tokens injected by Schedule)
            bool has_drafts = d->spec_decode_enabled && HasDrafts(*d);

            if (has_drafts) {
                RunWithDrafts(*d);
            }
            else {
                Run(*d);

                // After first decode, run MTP Draft if spec decode is enabled.
                // The first decode advances GDN state by the bonus token (S0 → S1).
                // This is correct: the next steady-state Forward processes tokens AFTER
                // the bonus, so GDN should start from S1.
                if (d->spec_decode_enabled && HasDecodeRequests(*d)) {
                    RunDraftOnly(*d);
                }
            }

            d->done.Record(core::Context::stream());
            outbound_.push(std::move(d));
        }
    }

    // Steady-state speculative decode pipeline (Verify-in-Next-Forward):
    //   Forward(K+1 tokens) → Reject → Rollback → Draft
    //
    // GDN state handling: We do NOT snapshot/restore GDN state.
    // The Forward processes [bonus, D0..D_{K-1}] through GDN, advancing the state.
    // On rejection (N<K), the rejected drafts' GDN contributions remain in the state.
    // This is a small error for K=1 (one wrong token per rejection) that doesn't
    // accumulate significantly. The alternative (restoring S_prev) loses the bonus
    // token's GDN contribution, which causes severe output degradation.
    // TODO: Implement two-phase Forward for exact GDN state management with K>1.
    void RunWithDrafts(BatchData& d)
    {
        // Mark this batch as having gone through the spec decode pipeline.
        // Update uses this to know that output_ids is invalid (Generation::Forward was skipped).
        d.spec_decode = true;

        TensorMap env{{"batch", d.buf()}};

        // 1. Normal Prepare → Forward → Unprep (K+1 tokens/req, prefill-like)
        BatchCopy copy;
        env["copy"] = copy.buf();
        model_.Run(BatchOp::kPrepare, d.phase, env);
        copy.Run();
        model_.Run(BatchOp::kForward, d.phase, env);
        model_.Run(BatchOp::kUnprep, d.phase, env);
        copy.Run();

        // 2. Reject (from Forward logits vs draft tokens)
        model_.Run(BatchOp::kReject, d.phase, env);

        // 3. Rollback (seq_len + token_ids, NO GDN restore)
        model_.Run(BatchOp::kRollback, d.phase, env);

        // 4. Draft (MTP generates new draft tokens, saves to RequestCache)
        // Pass block_ptrs for MTP SetupAttention (KV cache access)
        env["block_ptrs"]         = d.spec_block_ptrs;
        env["block_ptrs_offsets"] = d.spec_block_ptrs_offsets;
        model_.Run(BatchOp::kDraft, d.phase, env);

        AnomalyHandler::instance().Summarize([](...) {});
        AnomalyHandler::instance().Reset();
    }

    // First decode iteration: run MTP Draft after normal Forward to generate initial drafts
    void RunDraftOnly(BatchData& d)
    {
        TensorMap env{{"batch", d.buf()},
                      {"block_ptrs", d.spec_block_ptrs},
                      {"block_ptrs_offsets", d.spec_block_ptrs_offsets}};

        model_.Run(BatchOp::kDraft, d.phase, env);
    }

    void Run(BatchData& d)
    {
        auto batch = &d;

        // Not a steady-state spec decode iteration
        d.spec_decode = false;

        BatchCopy copy;
        TensorMap env{{"batch", d.buf()}, {"copy", copy.buf()}};

        model_.Run(BatchOp::kPrepare, d.phase, env);
        // dbg(copy);
        copy.Run();

        model_.Run(BatchOp::kForward, d.phase, env);

        model_.Run(BatchOp::kUnprep, d.phase, env);
        // dbg(copy);
        copy.Run();

        // TM_CHECK(0);
        AnomalyHandler::instance().Summarize([](...) {});
        AnomalyHandler::instance().Reset();
    }

    Impl(LanguageModel&                model,
         Context&                      context,
         int                           device_id,
         Queue<unique_ptr<BatchData>>& inbound,
         Queue<unique_ptr<BatchData>>& outbound):
        model_{model}, linear_{*context.linear}, device_id_{device_id}, inbound_{inbound}, outbound_{outbound}
    {
    }

    ~Impl()
    {
        if (internal_thread_.joinable()) {
            internal_thread_.join();
        }
    }

    void Start()
    {
        internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
    }
};

ModelExecutor::~ModelExecutor() = default;

ModelExecutor::ModelExecutor()                         = default;
ModelExecutor::ModelExecutor(ModelExecutor&&) noexcept = default;
ModelExecutor& ModelExecutor::operator=(ModelExecutor&&) noexcept = default;

ModelExecutor::ModelExecutor(LanguageModel&                model,
                             Context&                      context,
                             int                           device_id,
                             Queue<unique_ptr<BatchData>>& inbound,
                             Queue<unique_ptr<BatchData>>& outbound):
    impl_{std::make_unique<Impl>(model, context, device_id, inbound, outbound)}
{
}

void ModelExecutor::Start()
{
    return impl_->Start();
}

}  // namespace turbomind

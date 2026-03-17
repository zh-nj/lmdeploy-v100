/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

#include <algorithm>
#include <functional>
#include <math.h>
#include <numeric>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/sigmoid_gate.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

// #include "dbg.h"

namespace turbomind {

// #define DISABLE_MULTI_TOKEN_DECODE  // Uncomment to disable MTD for baseline testing
#define MTD_DIAGNOSTIC 0

struct AttentionData {
    struct Stat {
        int n;
        int q_sum;
        int q_max;
        int k_sum;
        int k_max;
    } decode, prefill;

    Buffer_<void*> block_ptrs;
    Buffer_<int>   block_ptrs_offsets;

    Buffer_<float> rope_base;

    Tensor_<int> mrope_position_ids;
    Buffer_<int> mrope_position_delta;
    Buffer_<int> mrope_length;

    // Per-phase owned copy of finished flags (avoids async race with State double-buffering)
    Buffer_<bool> finished_owned;
    // borrowed from env
    Buffer_<bool> finished;
    Buffer_<int>  q_offsets;
    Buffer_<int>  k_offsets;

    // Multi-token decode: treat small uniform prefill as repeated decode calls
    // 0 = normal, >0 = number of tokens per request to process via repeated decode calls
    int multi_token_decode = 0;

    // Temporary buffers for multi-token decode per-call cu_q_len and cu_k_len
    Buffer_<int> mtd_cu_q_len;
    Buffer_<int> mtd_cu_k_len;
    Tensor       mtd_temp_out;  // temp output buffer [bsz, out_dim]
};

UnifiedAttentionLayer::~UnifiedAttentionLayer()
{

    check_cuda_error(cudaEventDestroy(aux_event_));
    check_cuda_error(cudaEventDestroy(qkv_event_));
    check_cuda_error(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

UnifiedAttentionLayer::UnifiedAttentionLayer(const ModelParam&     model,
                                             const AttentionParam& attn,
                                             const EngineParam&    engine,
                                             int                   tp_size,
                                             const Context&        ctx,
                                             int                   phases,
                                             bool                  init):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    local_head_num_(head_num_ / tp_size),
    local_kv_head_num_(model.kv_head_num / tp_size),
    param_(attn),
    model_param_(model),
    engine_param_(engine),
    cp_fn_ctx_(ctx.comm.d_comm, ctx.comm.d_cp_group),
    is_warm_up_{*ctx.is_warm_up},
    context_(ctx),
    linear_(*ctx.linear),
    arch_(getSMVersion())
{
    TM_CHECK_EQ(head_num_ % tp_size, 0) << head_num_ << " " << tp_size;
    TM_CHECK_EQ(head_num_ % kv_head_num_, 0) << head_num_ << " " << kv_head_num_;

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    init_rope_kernel_param(param_.rope, rope_param_);

    // Pre-compute per-layer RopeKernelParam if available
    if (!param_.rope_per_layer.empty()) {
        rope_params_per_layer_.resize(param_.rope_per_layer.size());
        for (size_t i = 0; i < param_.rope_per_layer.size(); ++i) {
            init_rope_kernel_param(param_.rope_per_layer[i], rope_params_per_layer_[i]);
        }
    }

    Allocator alloc            = core::Context::device_alloc();
    ssize_t   workspace_tokens = kMaxWorkspaceTokens;
    if (engine_param_.attn_cp_size > 1) {
        alloc = GetSymmAllocator(ctx.comm.d_comm);
        workspace_tokens += engine_param_.max_forward_token_num;
    }
    // partial_O layout:
    //   w/  cp, decode(q, h, k, 2) + prefill(q, h, 1, 2)
    //   w/o cp, decode(q, h, k, 2)
    partial_O_  = Tensor_<float>({workspace_tokens, local_head_num_, size_per_head_}, kDEVICE);
    partial_ML_ = Tensor_<float>({engine_param_.attn_cp_size, workspace_tokens, local_head_num_, 2}, alloc);
    split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);
    if (init) {
        const int dim = (int)local_head_num_ * (int)size_per_head_;
        tmp_attn_     = Tensor{{engine_param_.max_forward_token_num, dim}, model.data_type, kDEVICE};
    }

    Clear(split_cnt_.buffer());

    // Pre-allocate variance buffer for per-token QK norm with TP>1
    if (model.qk_norm && model.qk_norm_type == "per_token" && tp_size > 1) {
        qk_norm_var_ = {{2 * engine.max_forward_token_num}, kDEVICE};
    }

    const int bsz = engine.max_batch_size;

    if (rope_param_.type == RopeType::kDynamic) {
        rope_base_buf_ = {bsz + 1, kCPUpinned};
    }
    else if (rope_param_.type == RopeType::kMrope) {
        // `mrope_position_ids` is not buffered
        mrope_position_delta_buf_ = {bsz, kCPUpinned};
        mrope_length_buf_         = {bsz, kCPUpinned};
    }
    const int max_blocks = bsz * cdiv(engine.session_len, param_.cache_block_seq_len);
    for (int i = 0; i < phases; ++i) {
        auto& d               = data_.emplace_back(std::make_shared<AttentionData>());
        d->block_ptrs         = {max_blocks + 16, kDEVICE};
        d->block_ptrs_offsets = {bsz + 1, kDEVICE};
        d->finished_owned     = {bsz, kDEVICE};
        if (rope_param_.type == RopeType::kDynamic) {
            d->rope_base = empty_like(rope_base_buf_, kDEVICE);
        }
        else if (rope_param_.type == RopeType::kMrope) {
            /// TODO: total space for `mrope_position_ids` can be reduced to (max_fwd_tokens, 3)
            d->mrope_position_ids    = {{bsz, engine.session_len, 3}, kDEVICE};
            d->mrope_position_delta  = empty_like(mrope_position_delta_buf_, kDEVICE);
            d->mrope_length          = empty_like(mrope_length_buf_, kDEVICE);
            rope_param_.mrope.stride = d->mrope_position_ids.stride(0);
        }
    }
}

static void init_dynamic_ntk(RequestCache& cache, const RopeParam& rope)
{
    cache.rope_base = rope.base;
    if (auto scaling_factor = rope.factor; scaling_factor > 1.f) {
        const auto max_seq_len = cache.prompt_len;
        const auto max_pos_emb = rope.max_position_embeddings;
        if (max_seq_len > max_pos_emb) {
            scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
            cache.rope_base *= powf(scaling_factor, rope.dim / (rope.dim - 2.f));
            // clang-format off
            TM_LOG_INFO("[ProcessInferRequests] %ld rope_scaling_factor: %f, rope_theta = %f",
                        (long)cache.req->id, scaling_factor, cache.rope_base);
            // clang-format on
        }
    }
}

void UnifiedAttentionLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        Buffer_<RequestCache*> rc = env.at("requests").buffer();
        if (rope_param_.type == RopeType::kDynamic) {
            for (int i = 0; i < rc.size(); ++i) {
                init_dynamic_ntk(*rc[i], param_.rope);
            }
        }
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        data_.at(phase)->finished  = env.at("finished").buffer().borrow();
        data_.at(phase)->q_offsets = env.at("q_offsets").buffer().borrow();
        data_.at(phase)->k_offsets = env.at("k_offsets").buffer().borrow();

        // This is needed in async mode to clear the `attn` buffer for the finished sequences. Ohterwise random NaNs
        // will crash the MoE router later
        /// TODO: use better solution, this increase memory usage and heterogenous attention layers may still break it
        if (tmp_attn_) {
            auto& d = data_.at(phase);
            const int total_q = d->multi_token_decode > 0 ? d->decode.q_sum : d->decode.n + d->prefill.q_sum;
            core::Clear(tmp_attn_.slice(0, total_q));
            core::Clear(split_cnt_);
        }
    }
}

void UnifiedAttentionLayer::Setup(int phase, TensorMap& env)
{
    const auto& rc  = env.at("batch").data<BatchData*>()[0]->rc;
    const int   bsz = rc.size();

    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    {  /// Upload KV cache ptrs
        const Buffer_<int> offsets = env.at("block_ptrs_offsets").buffer();
        copy(env.at("block_ptrs").buffer(), offsets[bsz], d.block_ptrs);
        copy(offsets, bsz + 1, d.block_ptrs_offsets);
    }

    /// prepare Q/K stats for decode/prefill
    d.decode = d.prefill = {};
    d.multi_token_decode = 0;

    d.decode.n  = std::find_if(rc.begin(), rc.end(), [](auto r) { return r->input_len > 1; }) - rc.begin();
    d.prefill.n = bsz - d.decode.n;

    // Detect uniform small prefill: all requests have same input_len > 1 and input_len <= threshold.
    // Reclassify as multi-token decode to use decode kernel (split-k, no ProcessKV/FlattenKV).
    //
    // This includes speculative verification batches (bonus + K drafts, input_len=K+1).
    // MTD uses decode kernel with split-k which may produce slightly different floating-point
    // results vs prefill kernel, but rejection sampling uses argmax (greedy) so the verification
    // outcome is robust to small numerical differences.
#ifndef DISABLE_MULTI_TOKEN_DECODE
    if (d.decode.n == 0 && d.prefill.n > 0) {
        const int first_input_len = rc[0]->input_len;
        bool uniform = first_input_len > 1 && first_input_len <= 8;
        for (int i = 1; uniform && i < bsz; ++i) {
            uniform = (rc[i]->input_len == first_input_len);
        }
        if (uniform) {
            d.multi_token_decode = first_input_len;
            d.decode.n  = bsz;
            d.prefill.n = 0;
        }
    }
#endif

    for (int i = 0; i < bsz; ++i) {
        const auto& c = *rc[i];

        auto& s = i < d.decode.n ? d.decode : d.prefill;
        s.q_sum += c.input_len;
        s.k_sum += c.history_len + c.alpha + c.input_len;
        s.q_max = std::max(s.q_max, c.input_len);
        s.k_max = std::max(s.k_max, c.history_len + c.alpha + c.input_len);
    }

    // auto &D = d.decode, &P = d.prefill;
    // dbg(D.n, D.k_sum, D.k_max, P.n, P.q_sum, P.q_max, P.k_sum, P.k_max);

    /// handling different RoPE types
    if (rope_param_.type == RopeType::kDynamic) {
        for (int i = 0; i < bsz; ++i) {
            rope_base_buf_[i] = rc[i]->rope_base;
        }
        copy(rope_base_buf_, bsz, d.rope_base);
    }
    else if (rope_param_.type == RopeType::kMrope) {
        const auto stride = d.mrope_position_ids.stride(0);
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            auto& r = *c.req;
            if (auto pos_ids = r.inputs.try_("mrope_position_ids")) {
                int length                   = pos_ids->shape(0);
                mrope_length_buf_[i]         = length;
                mrope_position_delta_buf_[i] = *r.inputs.at("mrope_position_delta").data<int>();
                if (auto o = Interval{0, length} & Interval{c.history_len + c.alpha, Interval::Size{c.input_len}}) {
                    copy(pos_ids->data<int>() + o.begin() * 3,
                         (int)o.size() * 3,
                         d.mrope_position_ids.data() + i * stride + o.begin() * 3);
                }
            }
            else {
                mrope_length_buf_[i] = mrope_position_delta_buf_[i] = 0;
            }
        }
        copy(mrope_length_buf_, rc.size(), d.mrope_length);
        copy(mrope_position_delta_buf_, rc.size(), d.mrope_position_delta);
    }
}

void UnifiedAttentionLayer::Forward(ForwardParam p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = p.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = p.layer_id;

    const auto& weights = *p.weights;

    Tensor qkv;

    auto& d = *data_.at(p.phase);

    if (weights.qkv.output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        qkv = linear_.Forward(p.input, weights.qkv);
        sync_check_cuda_error();

        if (model_param_.qk_norm) {
            qk_norm(qkv, weights);
        }
    }
    else {
        qkv = forward_mla(p.input, weights);
    }

    TM_DEBUG_TENSOR(qkv, Concat("qkv", layer_id), 3);

    auto invoke = [&](auto t) -> Tensor {
        using T = decltype(t);
        return core_attention<T>(qkv, p, weights);
    };

    Tensor attn = [&]() -> Tensor { TM_DISPATCH_PRIMARY_DTYPES_RET(qkv.dtype(), invoke); }();

    TM_DEBUG_TENSOR(attn, Concat("attn", layer_id), 3);

    // Gated attention: output *= sigmoid(gate)
    if (weights.gate && model_param_.attn_output_gate) {
        Tensor gate_out = linear_.Forward(p.input, weights.gate);
        sync_check_cuda_error();

        const int token_num = attn.shape(0);
        if (model_param_.attn_output_gate_per_head) {
            // Per-head scalar gate (Step3p5): gate shape (tokens, num_heads/tp)
            // Broadcast across head_dim: output[t,h*D+d] *= sigmoid(gate[t,h])
            const int layer_local_head_num = weights.gate.output_dim;
            invokeSigmoidGateBroadcast(attn.raw_data(), gate_out.raw_data(),
                                       token_num, layer_local_head_num, size_per_head_,
                                       byte_size(attn.dtype(), 1), core::Context::stream().handle());
        }
        else {
            // Per-element gate (Qwen3-Next): gate shape = attn shape
            // weights.gate.output_dim is the total output dimension (num_heads * head_dim / tp),
            // NOT the number of heads. So count = token_num * output_dim directly.
            const int count = token_num * weights.gate.output_dim;
            invokeSigmoidGate(attn.raw_data(), gate_out.raw_data(), count,
                              byte_size(attn.dtype(), 1), core::Context::stream().handle());
        }
        sync_check_cuda_error();
    }

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>

    (void)linear_.Forward(attn, weights.output, p.output);
    sync_check_cuda_error();
}

template<class T>
Tensor UnifiedAttentionLayer::core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    auto& d = *data_.at(p.phase);

    const int batch_size = d.decode.n + d.prefill.n;
    const int q_count    = qkv.shape(0);

    // Derive per-layer local_head_num from QKV weight output_dim
    // output_dim = (head_num + 2 * kv_head_num) * head_dim / tp
    // local_head_num = output_dim / head_dim - 2 * local_kv_head_num
    const int layer_local_head_num = weights.qkv.output_dim
        ? (weights.qkv.output_dim / size_per_head_ - 2 * local_kv_head_num_)
        : local_head_num_;

    TM_CHECK_EQ(d.prefill.q_sum + (d.multi_token_decode > 0 ? d.decode.q_sum : d.decode.n), q_count);

    const int local_q_kv_head_num = layer_local_head_num + 2 * local_kv_head_num_;

    const int layer_attn_dim = layer_local_head_num * size_per_head_;
    Tensor attn;
    if (tmp_attn_) {
        // tmp_attn_ is allocated with global max head_num; reshape to per-layer dim
        // The underlying buffer is large enough (layer_attn_dim <= local_head_num_ * size_per_head_)
        attn = Tensor{tmp_attn_.raw_data(), {q_count, layer_attn_dim}, dtype, device};
    }
    else {
        attn = {{q_count, layer_attn_dim}, dtype, device};
    }
    Tensor tmp_kv{{(int)local_kv_head_num_, 2, d.prefill.k_sum + MAX_CTA_S, (int)size_per_head_}, dtype, device};

    auto CreateParams = [&](int offset, AttentionData::Stat stat, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q      = (T*)qkv.raw_data();
        params.k      = params.q + layer_local_head_num * size_per_head_;
        params.v      = params.k + local_kv_head_num_ * size_per_head_;
        params.stride = (layer_local_head_num + 2 * local_kv_head_num_) * size_per_head_;

        if (weights.qkv.bias) {
            params.q_bias = (T*)weights.qkv.bias.data_or<T>(nullptr);
            params.k_bias = params.q_bias + layer_local_head_num * size_per_head_;
            params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;
        }

        params.batch_size = stat.n;

        params.token_num = stat.q_sum;
        params.max_q_len = stat.q_max;
        params.max_k_len = stat.k_max;

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       p.layer_id,
                                                       (int)param_.cache_block_seq_len};

        // prefill only
        params.linear_iter_params = LinearIteratorParams{tmp_kv.raw_data(),  //
                                                         int(2 * stat.k_sum * size_per_head_),
                                                         int(stat.k_sum * size_per_head_)};

        params.finished = d.finished.data() + offset;
        params.cu_q_len = d.q_offsets.data() + offset;
        params.cu_k_len = d.k_offsets.data() + offset;

        params.num_heads     = layer_local_head_num;
        params.num_kv_heads  = local_kv_head_num_;
        params.size_per_head = size_per_head_;
        params.layer_id      = p.layer_id;

        double scaling = 1.;
        if (param_.softmax_scale) {  // model predefined softmax scale
            scaling *= param_.softmax_scale;
        }
        else {  // default value
            scaling /= std::sqrt((float)params.size_per_head);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.sinks       = weights.sinks.data_or((T*)nullptr);
        params.scale_sinks = scaling;

        params.window_size = weights.window_size;
        if (!params.window_size) {
            params.window_size = 256 << 20;  // 256 M
        }

        params.rope_param = rope_param_;
        // Per-layer RoPE override
        if (!rope_params_per_layer_.empty()
            && p.layer_id < (int)rope_params_per_layer_.size()) {
            params.rope_param = rope_params_per_layer_[p.layer_id];
        }
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base = d.rope_base.data() + offset;
        }
        else if (rope_param_.type == RopeType::kMrope) {
            params.rope_param.mrope.position_ids   = d.mrope_position_ids.data() + offset * rope_param_.mrope.stride;
            params.rope_param.mrope.position_delta = d.mrope_position_delta.data() + offset;
            params.rope_param.mrope.length         = d.mrope_length.data() + offset;
        }

        // logn attn
        params.use_logn_attn           = param_.use_logn_attn;
        params.max_position_embeddings = param_.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_ML  = partial_ML_.data();
        params.partial_O   = partial_O_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        // context parallel
        params.cp_rank = engine_param_.attn_cp_rank;
        params.cp_size = engine_param_.attn_cp_size;
        if (params.cp_size > 1) {
            params.cp_size = cutlass::FastDivmod(params.cp_size);

            // update ML,O offset if both prefill and decode present
            const int offset_ML_stage =
                engine_param_.attn_cp_size * (offset ? kMaxWorkspaceTokens * layer_local_head_num * 2 : 0);
            const int offset_ML_rank = params.cp_rank * params.token_num * layer_local_head_num * params.max_split_k * 2;
            const int offset_O       = offset ? kMaxWorkspaceTokens * layer_local_head_num * size_per_head_ : 0;

            params.partial_ML = partial_ML_.data() + offset_ML_stage + offset_ML_rank;
            params.partial_O  = partial_O_.data() + offset_O;
            params.offset_q   = offset;

            // postprocess func
            params.cp_fn          = CpPost;
            params.cp_fn_ctx      = (void*)&cp_fn_ctx_;
            cp_fn_ctx_.cp_rank    = params.cp_rank;
            cp_fn_ctx_.count      = params.token_num * layer_local_head_num * params.max_split_k * 2;
            cp_fn_ctx_.partial_ML = partial_ML_.data() + offset_ML_stage;
            cp_fn_ctx_.stream     = stream;
        }

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = model_param_.quant_policy;
        return params;
    };

    const cudaStream_t stream = core::Context::stream().handle();

    cudaStream_t pf_stream = stream;
    cudaStream_t dc_stream = pf_stream;

    if (d.decode.n && d.prefill.n) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (d.prefill.n && !is_warm_up_) {
        const int offset = d.decode.n;
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, d.prefill, 1, pf_stream);

        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            sync_check_cuda_error();

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, d.prefill.k_sum);
            sync_check_cuda_error();

            dispatchAttention(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && !is_warm_up_) {
        if (d.multi_token_decode > 0) {            // Multi-token decode: dispatch N decode calls, one per token position.
            // Each call processes 1 token per request using the decode kernel (with split-k).
            const int N = d.multi_token_decode;
            const int out_dim = layer_local_head_num * size_per_head_;
            const int64_t original_stride = (layer_local_head_num + 2 * local_kv_head_num_) * size_per_head_;

            TM_LOG_INFO("[MTD] layer=%d N=%d bsz=%d q_count=%d k_max=%d stride=%ld",
                        p.layer_id, N, batch_size, q_count, d.decode.k_max, original_stride);

            // Allocate temp output buffer [bsz, out_dim] (reuse across calls)
            if (!d.mtd_temp_out || d.mtd_temp_out.shape(0) < batch_size) {
                d.mtd_temp_out = Tensor{{batch_size, out_dim}, dtype, kDEVICE};
            }

            // Build standard decode cu_q_len = [0, 1, 2, ..., bsz] (once, reuse)
            if (!d.mtd_cu_q_len || d.mtd_cu_q_len.size() < batch_size + 1) {
                d.mtd_cu_q_len = Buffer_<int>{batch_size + 1, kDEVICE};
                invokeIota(d.mtd_cu_q_len.data(), batch_size + 1, dc_stream);
            }

            // Allocate cu_k_len buffer for per-call adjustment
            if (!d.mtd_cu_k_len || d.mtd_cu_k_len.size() < batch_size + 1) {
                d.mtd_cu_k_len = Buffer_<int>{batch_size + 1, kDEVICE};
            }

            for (int t = 0; t < N; ++t) {
                // Build per-call decode stat (1 token per request)
                AttentionData::Stat per_call_stat{};
                per_call_stat.n     = batch_size;
                per_call_stat.q_sum = batch_size;
                per_call_stat.q_max = 1;
                // k_sum and k_max need per-call context_len = original_context_len - (N-1-t)
                // We compute approximate values from d.decode stats
                const int delta = N - 1 - t;
                per_call_stat.k_sum = d.decode.k_sum - batch_size * delta;
                per_call_stat.k_max = d.decode.k_max - delta;

                auto params = CreateParams(0, per_call_stat, kMaxKVSplits, dc_stream);

                // Override Q/K/V pointers: offset to token t, stride = N * original_stride
                params.q = (T*)qkv.raw_data() + t * original_stride;
                params.k = params.q + layer_local_head_num * size_per_head_;
                params.v = params.k + local_kv_head_num_ * size_per_head_;
                params.stride = N * original_stride;

                // Override output to temp buffer
                params.out = (T*)d.mtd_temp_out.raw_data();

                // Override cu_q_len to standard decode [0, 1, ..., bsz]
                params.cu_q_len = d.mtd_cu_q_len.data();

                // Build adjusted cu_k_len: new[i] = orig[i] - i * delta
                invokeBuildMtdCuKLen(d.mtd_cu_k_len.data(), d.k_offsets.data(), batch_size, delta, dc_stream);
                params.cu_k_len = d.mtd_cu_k_len.data();

                // Override token_num for reduce kernel
                params.token_num = batch_size;
                params.max_q_len = 1;


                if constexpr (sizeof(T) == 2) {
                    dispatchDecoding<T>(params);
                    sync_check_cuda_error();
                }

                // Scatter output: temp_out[i,:] → attn[(N*i + t) * out_dim .. +out_dim]
                invokeScatterRows(attn.raw_data(), d.mtd_temp_out.raw_data(),
                                  batch_size, N, t, out_dim * sizeof(T), dc_stream);
                sync_check_cuda_error();
            }

#if MTD_DIAGNOSTIC
            // === Compare MTD output with prefill reference ===
            if (p.layer_id == 0 && ref_attn.raw_data()) {
                cudaStreamSynchronize(dc_stream);
                const int total = q_count * out_dim;
                std::vector<T> h_ref(total), h_mtd(total);
                cudaMemcpy(h_ref.data(), ref_attn.raw_data(), total * sizeof(T), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_mtd.data(), attn.raw_data(), total * sizeof(T), cudaMemcpyDeviceToHost);

                // Per-head comparison for each row
                for (int row = 0; row < q_count && row < 4; ++row) {
                    for (int h = 0; h < (int)layer_local_head_num; ++h) {
                        float head_max_diff = 0;
                        int head_mismatch = 0;
                        for (int d = 0; d < (int)size_per_head_; ++d) {
                            int idx = row * out_dim + h * size_per_head_ + d;
                            float diff = std::abs((float)h_ref[idx] - (float)h_mtd[idx]);
                            if (diff > head_max_diff) head_max_diff = diff;
                            if (diff > 0.01f) head_mismatch++;
                        }
                        if (head_mismatch > 0) {
                            int base = row * out_dim + h * size_per_head_;
                            TM_LOG_WARNING("[MTD_DIAG] rank=%d row=%d head=%d max_diff=%f mismatches=%d/%d "
                                           "ref=[%f,%f,%f,%f] mtd=[%f,%f,%f,%f]",
                                           (int)engine_param_.attn_tp_rank, row, h, head_max_diff, head_mismatch, (int)size_per_head_,
                                           (float)h_ref[base], (float)h_ref[base+1], (float)h_ref[base+2], (float)h_ref[base+3],
                                           (float)h_mtd[base], (float)h_mtd[base+1], (float)h_mtd[base+2], (float)h_mtd[base+3]);
                        }
                    }
                }

                // Also check if mtd row0 == mtd row1 (symptom from previous session)
                if (q_count >= 2) {
                    bool rows_identical = true;
                    for (int j = 0; j < out_dim && rows_identical; ++j) {
                        if ((float)h_mtd[j] != (float)h_mtd[out_dim + j]) rows_identical = false;
                    }
                    if (rows_identical) {
                        TM_LOG_WARNING("[MTD_DIAG] rank=%d WARNING: mtd row0 == row1 (identical output for both tokens!)",
                                       (int)engine_param_.attn_tp_rank);
                    }
                }
            }
#endif
        }
        else {
            auto params = CreateParams(0, d.decode, kMaxKVSplits, dc_stream);
            if constexpr (sizeof(T) == 2) {
                dispatchDecoding<T>(params);
                sync_check_cuda_error();
            }
        }
    }

    if (d.decode.n && d.prefill.n) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
    }

    if (is_warm_up_) {
        rng_.set_stream(stream);
        rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}

Tensor UnifiedAttentionLayer::forward_mla(const Tensor& hidden_state, const WeightType& w)
{
    const int q_lora_rank  = w.q_a_proj.output_dim;
    const int kv_lora_rank = w.kv_b_proj.input_dim;
    const int qk_rope_dim  = w.kv_a_proj.output_dim - kv_lora_rank;
    const int qk_nope_dim  = std::max(w.q_b_proj.output_dim, w.q_proj.output_dim) / local_head_num_ - qk_rope_dim;
    const int v_head_dim   = w.kv_b_proj.output_dim / local_head_num_ - qk_nope_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    Tensor q;

    const auto stream = core::Context::stream().handle();

    if (w.q_proj.weight) {
        q = linear_.Forward(hidden_state, w.q_proj);
        sync_check_cuda_error();
    }
    else {
        Tensor q_a = linear_.Forward(hidden_state, w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm, model_param_.norm_eps, stream);
        sync_check_cuda_error();

        q = linear_.Forward(q_a, w.q_b_proj);
        sync_check_cuda_error();
    }

    Tensor kv_a_k_pe = linear_.Forward(hidden_state, w.kv_a_proj);
    sync_check_cuda_error();

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm, model_param_.norm_eps, stream);
    sync_check_cuda_error();

    Tensor kv_b = linear_.Forward(kv_a, w.kv_b_proj);
    sync_check_cuda_error();

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    Tensor qkv{{token_num, local_q_kv_head_num, (int)size_per_head_}, dtype, hidden_state.device()};
    MLACopyQKV(dtype,
               qkv.raw_data(),
               q.raw_data(),
               kv_a.raw_data(),
               kv_b.raw_data(),
               token_num,
               local_head_num_,
               qk_nope_dim,
               qk_rope_dim,
               kv_lora_rank,
               v_head_dim,
               stream);
    sync_check_cuda_error();

    return qkv;
}

void UnifiedAttentionLayer::qk_norm(Tensor& qkv, const WeightType& weights)
{
    const auto stream = core::Context::stream().handle();

    TM_CHECK(model_param_.attn_bias == false) << "not implemented";

    const auto token_num = qkv.shape(0);

    // Derive per-layer local_head_num from QKV weight output_dim
    const int layer_local_head_num = weights.qkv.output_dim
        ? (weights.qkv.output_dim / size_per_head_ - 2 * local_kv_head_num_)
        : local_head_num_;
    // Derive global head_num for this layer
    const int tp_size = head_num_ / local_head_num_;
    const int layer_head_num = layer_local_head_num * tp_size;

    if (model_param_.qk_norm_type == "per_token") {
        // MiniMax-M2 style: RMS norm across all heads combined (per-token, not per-head).
        // Q shape: (tokens, local_heads * head_dim), K shape: (tokens, local_kv_heads * head_dim)
        const int q_dim = layer_local_head_num * size_per_head_;
        const int k_dim = local_kv_head_num_ * size_per_head_;

        // Global dimensions across all TP ranks
        const int global_q_dim = layer_head_num * size_per_head_;
        const int global_k_dim = kv_head_num_ * size_per_head_;

        if (tp_size > 1 && context_.comm.d_comm) {
            // Fused two-phase RMS norm with TP all-reduce for correct global variance.
            // All work on main stream — no aux_stream, no event sync needed.
            auto q_var = qk_norm_var_.slice(0, token_num).view({token_num, 1});
            auto k_var = qk_norm_var_.slice(token_num, token_num).view({token_num, 1});

            // Phase 1: compute local sum of squares for Q and K in a single fused kernel
            auto qkv_2d = qkv.view({token_num, -1});
            invokeRMSNormVarianceQK(q_var, k_var, qkv_2d, q_dim, k_dim, stream);
            sync_check_cuda_error();

            // All-reduce sum of squares across TP ranks (single call for both Q and K)
            const int tp_group = context_.comm.d_tp_group;
            context_.comm.d_comm->AllReduceSum(
                qk_norm_var_.data(), qk_norm_var_.data(), 2 * token_num, kFloat, tp_group, stream);
            sync_check_cuda_error();

            // Phase 2: apply normalization to Q and K in a single fused kernel
            invokeRMSNormApplyQK(
                qkv_2d, q_dim, k_dim, weights.q_a_layernorm, weights.kv_a_layernorm, q_var, k_var, model_param_.norm_eps, global_q_dim, global_k_dim, stream);
            sync_check_cuda_error();
        }
        else {
            // TP=1: use simple single-phase RMS norm (no all-reduce needed)
            // Fork K to aux_stream for parallelism
            check_cuda_error(cudaEventRecord(qkv_event_, stream));
            check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

            auto q = qkv.view({token_num, -1}).slice({0, 0}, {-1, (int)q_dim});
            auto k = qkv.view({token_num, -1}).slice({0, (int)q_dim}, {-1, (int)k_dim});

            invokeRMSNorm(q, q, weights.q_a_layernorm, model_param_.norm_eps, stream);
            sync_check_cuda_error();

            invokeRMSNorm(k, k, weights.kv_a_layernorm, model_param_.norm_eps, aux_stream_);
            sync_check_cuda_error();

            check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
            check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
        }
    }
    else {
        // Standard per-head QK norm (e.g. Qwen3)
        // Fork K to aux_stream for parallelism
        check_cuda_error(cudaEventRecord(qkv_event_, stream));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

        auto qkv3 = qkv.view({token_num, -1, (int)size_per_head_});

        auto q = qkv3.slice({0, 0, 0}, {-1, (int)layer_local_head_num, -1});
        invokeRMSNormQK(q, weights.q_a_layernorm, model_param_.norm_eps, stream);
        sync_check_cuda_error();

        auto k = qkv3.slice({0, (int)layer_local_head_num, 0}, {-1, (int)local_kv_head_num_, -1});
        invokeRMSNormQK(k, weights.kv_a_layernorm, model_param_.norm_eps, aux_stream_);
        sync_check_cuda_error();

        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
    }
}

}  // namespace turbomind

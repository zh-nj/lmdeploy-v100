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

#include "src/turbomind/models/llama/gated_delta_net_layer.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cstring>

#include "src/turbomind/kernels/gated_delta_net/causal_conv1d.h"
#include "src/turbomind/kernels/gated_delta_net/delta_rule.h"
#include "src/turbomind/kernels/gated_delta_net/gated_rmsnorm.h"
#include "src/turbomind/kernels/gated_delta_net/gdn_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

GatedDeltaNetLayer::GatedDeltaNetLayer(const ModelParam&          model,
                                       const GatedDeltaNetParam&  gdn,
                                       const EngineParam&         engine,
                                       const Context&             ctx):
    linear_(*ctx.linear),
    hidden_size_(model.hidden_units),
    num_k_heads_(gdn.num_key_heads / engine.attn_tp_size),
    num_v_heads_(gdn.num_value_heads / engine.attn_tp_size),
    head_k_dim_(gdn.key_head_dim),
    head_v_dim_(gdn.value_head_dim),
    conv_kernel_size_(gdn.conv_kernel_dim),
    kv_ratio_(gdn.num_value_heads / gdn.num_key_heads),
    key_dim_(num_k_heads_ * head_k_dim_),
    value_dim_(num_v_heads_ * head_v_dim_),
    conv_dim_(key_dim_ * 2 + value_dim_),
    qkvz_dim_(key_dim_ * 2 + value_dim_ * 2),
    ba_dim_(num_v_heads_ * 2),
    all_proj_dim_(qkvz_dim_ + ba_dim_),
    norm_eps_(model.norm_eps),
    max_batch_size_(engine.max_batch_size),
    max_forward_token_num_(engine.max_forward_token_num),
    num_linear_layers_(0),
    tp_size_(engine.attn_tp_size)
{
    const auto stream = ctx.stream;

    // Count linear attention layers from layer_types
    int linear_idx = 0;
    layer_id_to_linear_idx_.resize(model.layer_num, -1);
    for (int i = 0; i < (int)model.layer_num; ++i) {
        if (i < (int)model.layer_types.size() && model.layer_types[i] == 1) {
            layer_id_to_linear_idx_[i] = linear_idx++;
        }
    }
    const_cast<int&>(num_linear_layers_) = linear_idx;

    // Pre-allocate state buffers using raw cudaMalloc
    conv_state_raw_ = nullptr;
    recurrent_state_raw_ = nullptr;
    if (num_linear_layers_ > 0) {
        conv_state_size_ = (size_t)max_batch_size_ * num_linear_layers_ * conv_dim_ * conv_kernel_size_;
        check_cuda_error(cudaMalloc(&conv_state_raw_, conv_state_size_ * sizeof(half)));
        check_cuda_error(cudaMemsetAsync(conv_state_raw_, 0, conv_state_size_ * sizeof(half), stream));

        recurrent_state_size_ = (size_t)max_batch_size_ * num_linear_layers_ * num_v_heads_ * head_k_dim_ * head_v_dim_;
        check_cuda_error(cudaMalloc(&recurrent_state_raw_, recurrent_state_size_ * sizeof(float)));
        check_cuda_error(cudaMemsetAsync(recurrent_state_raw_, 0, recurrent_state_size_ * sizeof(float), stream));
        check_cuda_error(cudaStreamSynchronize(stream));

        // Allocate snapshot buffers only when speculative decoding is enabled
        if (engine.speculative_decoding) {
            check_cuda_error(cudaMalloc(&conv_state_snapshot_, conv_state_size_ * sizeof(half)));
            check_cuda_error(cudaMalloc(&recurrent_state_snapshot_, recurrent_state_size_ * sizeof(float)));
            // snapshot2_ buffers are only needed for SwapStateAndSnapshot (deprecated in Verify-in-Next-Forward flow)
            // Removed to save ~4.5 GB GPU memory
        }
    }

    // Pre-allocate temporary buffers to avoid per-forward allocation
    const int max_tokens = max_forward_token_num_;
    check_cuda_error(cudaMalloc(&conv_out_buf_,  (size_t)max_tokens * conv_dim_ * sizeof(half)));
    check_cuda_error(cudaMalloc(&delta_out_buf_, (size_t)max_tokens * num_v_heads_ * head_v_dim_ * sizeof(half)));
    check_cuda_error(cudaMalloc(&norm_out_buf_,  (size_t)max_tokens * value_dim_ * sizeof(half)));

    // Host staging buffers
    cu_seqlens_host_ = {max_batch_size_ + 1, kCPUpinned};
    seq_idx_host_    = {max_forward_token_num_, kCPUpinned};
}

GatedDeltaNetLayer::~GatedDeltaNetLayer()
{
    cudaFree(conv_state_raw_);
    cudaFree(recurrent_state_raw_);
    cudaFree(conv_state_snapshot_);
    cudaFree(recurrent_state_snapshot_);
    cudaFree(conv_state_snapshot2_);
    cudaFree(recurrent_state_snapshot2_);
    cudaFree(conv_out_buf_);
    cudaFree(delta_out_buf_);
    cudaFree(norm_out_buf_);
}

void GatedDeltaNetLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kSetup) {
        const auto stream = core::Context::stream().handle();
        const auto& rc  = env.at("batch").data<BatchData*>()[0]->rc;
        const int   bsz = rc.size();
        auto& copy      = *env.at("copy").data<BatchCopy*>()[0];

        if ((int)phase_data_.size() <= phase) {
            phase_data_.resize(phase + 1);
            for (auto& pd : phase_data_) {
                if (!pd.cu_seqlens) {
                    pd.cu_seqlens = {max_batch_size_ + 1, kDEVICE};
                }
                if (!pd.seq_idx) {
                    pd.seq_idx = {max_forward_token_num_, kDEVICE};
                }
            }
        }

        auto& pd = phase_data_[phase];
        pd.batch_size = bsz;

        cu_seqlens_host_[0] = 0;
        int total_tokens = 0;
        for (int i = 0; i < bsz; ++i) {
            const auto& c = *rc[i];
            int seq_len = c.input_len;
            for (int t = 0; t < seq_len; ++t) {
                seq_idx_host_[total_tokens + t] = i;
            }
            total_tokens += seq_len;
            cu_seqlens_host_[i + 1] = total_tokens;
        }
        pd.token_num = total_tokens;

        copy(cu_seqlens_host_, bsz + 1, pd.cu_seqlens);
        copy(seq_idx_host_, total_tokens, pd.seq_idx);

        if (num_linear_layers_ > 0) {
            const size_t conv_layer_stride = (size_t)conv_dim_ * conv_kernel_size_;
            const size_t conv_batch_stride = (size_t)num_linear_layers_ * conv_layer_stride;
            const size_t rs_layer_size     = (size_t)num_v_heads_ * head_k_dim_ * head_v_dim_;
            const size_t rs_batch_stride   = (size_t)num_linear_layers_ * rs_layer_size;
            for (int b = 0; b < bsz; ++b) {
                const bool do_reset = rc[b]->history_len == 0;
                if (do_reset) {
                    check_cuda_error(cudaMemsetAsync(
                        conv_state_raw_ + b * conv_batch_stride,
                        0, conv_batch_stride * sizeof(half), stream));
                    check_cuda_error(cudaMemsetAsync(
                        recurrent_state_raw_ + b * rs_batch_stride,
                        0, rs_batch_stride * sizeof(float), stream));
                }
            }
        }
    }
}

void GatedDeltaNetLayer::Forward(ForwardParam p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto stream = core::Context::stream().handle();

    const int token_num = p.input.shape(0);
    if (token_num == 0) {
        return;
    }

    const int    layer_id   = p.layer_id;
    const auto&  weights    = *p.weights;
    const int    linear_idx = layer_id_to_linear_idx_[layer_id];

    TM_CHECK(linear_idx >= 0) << "layer " << layer_id << " is not a linear attention layer";

    auto& pd = phase_data_[p.phase];
    const int batch_size = pd.batch_size;
    const bool is_decode = (token_num == batch_size);

    // 1. Single merged GEMM: in_proj_all = [QKVZ, BA]
    Tensor proj_all = linear_.Forward(p.input, weights.in_proj_all);
    sync_check_cuda_error();

    // proj_all layout: [token_num, qkvz_dim_ + ba_dim_]
    // where qkvz_dim_ = key_dim_*2 + value_dim_*2
    //       ba_dim_    = num_v_heads_*2
    // Offsets within proj_all row:
    //   Q:     [0, key_dim_)
    //   K:     [key_dim_, key_dim_*2)
    //   V:     [key_dim_*2, key_dim_*2 + value_dim_)
    //   Z:     [key_dim_*2 + value_dim_, qkvz_dim_)  = [conv_dim_, qkvz_dim_)
    //   b(beta_raw): [qkvz_dim_, qkvz_dim_ + num_v_heads_)
    //   a(alpha):    [qkvz_dim_ + num_v_heads_, all_proj_dim_)
    // Runtime uses BA suffix, so alpha_col_offset points to a(alpha) and
    // beta_col_offset points to b(beta_raw).
    const half* proj_ptr = (const half*)proj_all.raw_data();

    // 2. Causal Conv1d (reads QKV from proj_all, stride = all_proj_dim_)
    const size_t conv_state_layer_stride = (size_t)conv_dim_ * conv_kernel_size_;
    const size_t conv_state_batch_stride = (size_t)num_linear_layers_ * conv_state_layer_stride;
    half* conv_state = conv_state_raw_ + linear_idx * conv_state_layer_stride;

    half* conv_out = conv_out_buf_;

    if (is_decode) {
        invokeCausalConv1dDecode(conv_out,
                                 conv_state,
                                 proj_ptr,
                                 (const half*)weights.conv1d_weight.raw_data(),
                                 batch_size,
                                 conv_dim_,
                                 conv_kernel_size_,
                                 stream,
                                 all_proj_dim_,
                                 (int)conv_state_batch_stride);
    }
    else {
        invokeCausalConv1dPrefill(conv_out,
                                  conv_state,
                                  proj_ptr,
                                  (const half*)weights.conv1d_weight.raw_data(),
                                  pd.seq_idx.data(),
                                  batch_size,
                                  token_num,
                                  conv_dim_,
                                  conv_kernel_size_,
                                  stream,
                                  all_proj_dim_,
                                  (int)conv_state_batch_stride,
                                  pd.cu_seqlens.data());
    }
    sync_check_cuda_error();

    // 3+4. Delta Rule with inline gating (fused: eliminates separate gating kernel)
    const size_t rs_head_size    = (size_t)head_k_dim_ * head_v_dim_;
    const size_t rs_layer_size   = (size_t)num_v_heads_ * rs_head_size;
    const size_t rs_batch_stride = (size_t)num_linear_layers_ * rs_layer_size;
    float* rs_state = recurrent_state_raw_ + linear_idx * rs_layer_size;

    half* delta_out = delta_out_buf_;

    if (is_decode) {
        invokeRecurrentDeltaRuleDecode(delta_out,
                                       rs_state,
                                       conv_out,
                                       conv_out + key_dim_,
                                       conv_out + key_dim_ * 2,
                                       (const half*)weights.A_log.raw_data(),
                                       (const half*)weights.dt_bias.raw_data(),
                                       proj_ptr,   // alpha source (a in BA suffix)
                                       proj_ptr,   // beta source (b in BA suffix)
                                       batch_size,
                                       num_v_heads_,
                                       num_k_heads_,
                                       kv_ratio_,
                                       conv_dim_,
                                       conv_dim_,
                                       conv_dim_,
                                       head_k_dim_,
                                       head_v_dim_,
                                       stream,
                                       (int)rs_batch_stride,
                                       all_proj_dim_,              // alpha_stride
                                       all_proj_dim_,              // beta_stride
                                       qkvz_dim_ + num_v_heads_,  // alpha_col_offset -> a(alpha)
                                       qkvz_dim_);                 // beta_col_offset  -> b(beta_raw)
    }
    else {
        invokeChunkDeltaRulePrefill(delta_out,
                                    rs_state,
                                    conv_out,
                                    conv_out + key_dim_,
                                    conv_out + key_dim_ * 2,
                                    (const half*)weights.A_log.raw_data(),
                                    (const half*)weights.dt_bias.raw_data(),
                                    proj_ptr,   // alpha source (a in BA suffix)
                                    proj_ptr,   // beta source (b in BA suffix)
                                    pd.cu_seqlens.data(),
                                    batch_size,
                                    token_num,
                                    num_v_heads_,
                                    num_k_heads_,
                                    kv_ratio_,
                                    conv_dim_,
                                    conv_dim_,
                                    conv_dim_,
                                    head_k_dim_,
                                    head_v_dim_,
                                    stream,
                                    (int)rs_batch_stride,
                                    all_proj_dim_,              // alpha_stride
                                    all_proj_dim_,              // beta_stride
                                    qkvz_dim_ + num_v_heads_,  // alpha_col_offset -> a(alpha)
                                    qkvz_dim_);                 // beta_col_offset  -> b(beta_raw)
    }
    sync_check_cuda_error();

    // 5. Gated RMSNorm (reads Z directly from proj_all at offset conv_dim_)
    half* norm_out = norm_out_buf_;

    invokeGatedRMSNorm(norm_out,
                       delta_out,
                       proj_ptr + conv_dim_,  // Z starts at conv_dim_ within proj_all
                       (const half*)weights.norm_weight.raw_data(),
                       token_num, num_v_heads_, head_v_dim_, norm_eps_, stream,
                       0,              // x_stride (compact delta_out)
                       all_proj_dim_); // z_stride (Z is in proj_all with stride all_proj_dim_)
    sync_check_cuda_error();

    // 6. out_proj GEMM
    Tensor norm_tensor{(void*)norm_out, {token_num, value_dim_}, DataType::kFloat16, kDEVICE};
    (void)linear_.Forward(norm_tensor, weights.out_proj, p.output);
    sync_check_cuda_error();
}

void GatedDeltaNetLayer::SnapshotState(cudaStream_t stream)
{
    TM_CHECK(conv_state_snapshot_ != nullptr) << "Snapshot buffers not allocated (speculative_decoding disabled?)";
    TM_CHECK(conv_state_raw_ != nullptr) << "State buffers not allocated";

    check_cuda_error(cudaMemcpyAsync(
        conv_state_snapshot_, conv_state_raw_, conv_state_size_ * sizeof(half),
        cudaMemcpyDeviceToDevice, stream));
    check_cuda_error(cudaMemcpyAsync(
        recurrent_state_snapshot_, recurrent_state_raw_, recurrent_state_size_ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    snapshot_valid_ = true;
}

void GatedDeltaNetLayer::RestoreState(cudaStream_t stream)
{
    TM_CHECK(snapshot_valid_) << "No valid snapshot to restore";
    TM_CHECK(conv_state_snapshot_ != nullptr && conv_state_raw_ != nullptr);

    check_cuda_error(cudaMemcpyAsync(
        conv_state_raw_, conv_state_snapshot_, conv_state_size_ * sizeof(half),
        cudaMemcpyDeviceToDevice, stream));
    check_cuda_error(cudaMemcpyAsync(
        recurrent_state_raw_, recurrent_state_snapshot_, recurrent_state_size_ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    snapshot_valid_ = false;
}

void GatedDeltaNetLayer::DiscardSnapshot()
{
    snapshot_valid_ = false;
}

void GatedDeltaNetLayer::SwapStateAndSnapshot(cudaStream_t stream)
{
    TM_CHECK(snapshot_valid_) << "No valid snapshot to swap with";
    TM_CHECK(conv_state_snapshot2_ != nullptr) << "Swap temp buffers not allocated";

    // 3-way swap using slot 1 as temp: state <-> snapshot
    // Step 1: state -> temp (slot 1)
    check_cuda_error(cudaMemcpyAsync(
        conv_state_snapshot2_, conv_state_raw_, conv_state_size_ * sizeof(half),
        cudaMemcpyDeviceToDevice, stream));
    check_cuda_error(cudaMemcpyAsync(
        recurrent_state_snapshot2_, recurrent_state_raw_, recurrent_state_size_ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    // Step 2: snapshot -> state
    check_cuda_error(cudaMemcpyAsync(
        conv_state_raw_, conv_state_snapshot_, conv_state_size_ * sizeof(half),
        cudaMemcpyDeviceToDevice, stream));
    check_cuda_error(cudaMemcpyAsync(
        recurrent_state_raw_, recurrent_state_snapshot_, recurrent_state_size_ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    // Step 3: temp -> snapshot
    check_cuda_error(cudaMemcpyAsync(
        conv_state_snapshot_, conv_state_snapshot2_, conv_state_size_ * sizeof(half),
        cudaMemcpyDeviceToDevice, stream));
    check_cuda_error(cudaMemcpyAsync(
        recurrent_state_snapshot_, recurrent_state_snapshot2_, recurrent_state_size_ * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    // snapshot_valid_ remains true (snapshot now holds old live state)
}

}  // namespace turbomind

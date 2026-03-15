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

#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/gated_delta_net_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class GatedDeltaNetLayer {
public:
    using WeightType = GatedDeltaNetWeight;

    struct ForwardParam {
        int               phase;
        Tensor            input;
        Tensor            output;
        const WeightType* weights;
        int               layer_id;
    };

    GatedDeltaNetLayer(const ModelParam&          model,
                       const GatedDeltaNetParam&  gdn,
                       const EngineParam&         engine,
                       const Context&             ctx);

    ~GatedDeltaNetLayer();

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

    /// Snapshot current conv_state and recurrent_state to slot 0 (primary snapshot).
    /// Only valid when speculative_decoding is enabled (snapshot buffers allocated).
    void SnapshotState(cudaStream_t stream);

    /// Restore conv_state and recurrent_state from slot 0 (primary snapshot).
    /// Asserts that a valid snapshot exists.
    void RestoreState(cudaStream_t stream);

    /// Mark the primary snapshot as invalid (no-op on memory).
    void DiscardSnapshot();

    /// Swap live state with primary snapshot (slot 0).
    /// After swap: live state = old snapshot, snapshot = old live state.
    /// Uses slot 1 as temporary storage for the 3-way swap.
    /// Preserves snapshot_valid_ = true.
    void SwapStateAndSnapshot(cudaStream_t stream);

private:
    LlamaLinear& linear_;

    // Model dimensions (after TP sharding)
    const int hidden_size_;
    const int num_k_heads_;   // local
    const int num_v_heads_;   // local
    const int head_k_dim_;
    const int head_v_dim_;
    const int conv_kernel_size_;
    const int kv_ratio_;      // num_v_heads / num_k_heads

    const int key_dim_;       // num_k_heads * head_k_dim (local)
    const int value_dim_;     // num_v_heads * head_v_dim (local)
    const int conv_dim_;      // key_dim * 2 + value_dim  (local)
    const int qkvz_dim_;      // key_dim * 2 + value_dim * 2 (local)
    const int ba_dim_;        // num_v_heads * 2 (local)
    const int all_proj_dim_;  // qkvz_dim + ba_dim (merged projection output dim)

    const float norm_eps_;

    const int max_batch_size_;
    const int max_forward_token_num_;
    const int num_linear_layers_;  // number of linear attention layers

    const int tp_size_;

    // Per-phase batch metadata
    struct PhaseData {
        int        batch_size = 0;
        int        token_num  = 0;
        Buffer_<int> cu_seqlens;   // [batch+1] on device
        Buffer_<int> seq_idx;      // [token_num] on device
    };
    std::vector<PhaseData> phase_data_;

    // Host staging buffers for Setup
    Buffer_<int> cu_seqlens_host_;
    Buffer_<int> seq_idx_host_;

    // Pre-allocated state buffers (raw cudaMalloc, not pool allocator)
    half*  conv_state_raw_ = nullptr;
    float* recurrent_state_raw_ = nullptr;
    size_t conv_state_size_ = 0;
    size_t recurrent_state_size_ = 0;

    // Pre-allocated snapshot buffers for speculative decoding rollback.
    // Slot 0 (primary): used for S0 snapshot (pre-main-Forward state).
    // Slot 1 (swap temp): used as temporary during SwapStateAndSnapshot.
    // Same size as state buffers; only allocated when speculative_decoding is enabled.
    half*  conv_state_snapshot_       = nullptr;
    float* recurrent_state_snapshot_  = nullptr;
    half*  conv_state_snapshot2_      = nullptr;  // slot 1 (swap temp)
    float* recurrent_state_snapshot2_ = nullptr;  // slot 1 (swap temp)
    bool   snapshot_valid_            = false;

    // Pre-allocated temporary buffers (avoid per-forward allocation).
    // These are per-layer scratch buffers, so Forward() on the same layer
    // instance must not run concurrently.
    half*  conv_out_buf_  = nullptr;  // [max_tokens, conv_dim]
    half*  delta_out_buf_ = nullptr;  // [max_tokens, num_v_heads * head_v_dim]
    half*  norm_out_buf_  = nullptr;  // [max_tokens, value_dim]

    // Mapping from global layer_id to linear-attention layer index
    std::vector<int> layer_id_to_linear_idx_;
};

}  // namespace turbomind

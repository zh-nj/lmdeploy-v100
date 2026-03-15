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

#include <memory>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/MTPWeight.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

struct ForwardStepResult {
    Buffer_<int> draft_tokens;      // [batch] draft token IDs
    Tensor       output_hidden;     // [batch, hidden_size] post-FFN residual
};

class MTPPredictor {
public:
    MTPPredictor(const ModelParam&     model,
                 const EngineParam&    engine,
                 const AttentionParam& attn,
                 const MoeParam&       moe,
                 const Context&        ctx,
                 const LlamaWeight&    weights);

    ~MTPPredictor();

    struct DraftResult {
        Buffer_<int> draft_tokens;  // [K] draft token IDs
        int          num_drafts;    // actual number of drafts generated (≤ K)
    };

    /// Generate K draft tokens from main model's hidden_states and last accepted token.
    DraftResult Draft(int                 batch_size,
                      const Tensor&       hidden_states,  // [batch, hidden_size] last-layer hidden states
                      const Buffer_<int>& last_tokens,    // [batch] last accepted token IDs
                      int                 num_draft_tokens,  // K
                      TensorMap&          env);

    /// Set up the MTP attention layer's dispatch data for a decode-only batch.
    /// Must be called before Draft() with the same env that contains batch, block_ptrs, etc.
    void SetupAttention(int phase, TensorMap& env);

    /// Forward a single MTP step: embed → dual RMSNorm → concat → fc → attn → MoE → norm → lm_head → argmax
    /// Returns the draft token IDs and output hidden_states for this step.
    ForwardStepResult ForwardStep(int              mtp_layer_idx,
                                  int              batch_size,
                                  const Tensor&    prev_embedding,  // [batch, hidden_size]
                                  const Tensor&    hidden_states,   // [batch, hidden_size]
                                  int              step_idx,
                                  TensorMap&       env);

private:
    /// Lookup embedding for token IDs using shared embed_tokens
    Tensor LookupEmbedding(const Buffer_<int>& token_ids, int batch_size);

    /// Compute logits using shared lm_head (PostEmbedding)
    Tensor PostEmbedding(const Tensor& features, int batch_size);

    /// Compute logits using per-layer lm_head (PostEmbeddingLocal)
    Tensor PostEmbeddingLocal(const Tensor& features, int batch_size, const LlamaDenseWeight& lm_head);

    /// Simple argmax over logits → token IDs
    Buffer_<int> Argmax(const Tensor& logits, int batch_size);

    const ModelParam     param_;
    const float          norm_eps_;
    const DataType       dtype_;
    const int            hidden_units_;
    const int            tp_size_;
    const int            tp_rank_;
    const int            mtp_attn_layer_offset_;  // KV cache layer index for MTP attention

    LlamaLinear&                    linear_;
    const LlamaWeight&              weights_;
    const Communicators&            comm_;
    comm::DeviceCommImpl* const     d_comm_;

    // MTP's own attention and MoE layer instances
    std::unique_ptr<UnifiedAttentionLayer> attn_layer_;
    std::unique_ptr<MoeFfnLayer>           moe_ffn_layer_;
    std::unique_ptr<LlamaFfnLayer>         ffn_layer_;  // for shared expert
};

}  // namespace turbomind

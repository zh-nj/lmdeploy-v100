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
#include <optional>

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"

namespace turbomind {

struct MTPLayerWeight: core::Module {

    MTPLayerWeight() = default;

    // Dual RMSNorm weights for embedding and hidden_states branches
    Tensor pre_fc_norm_embedding;  // [hidden_size]
    Tensor pre_fc_norm_hidden;     // [hidden_size]

    // Linear projection: hidden_size*2 → hidden_size
    LlamaDenseWeight fc;

    // Full-attention + MoE decoder layer (reuses existing structure)
    std::unique_ptr<LlamaDecoderLayerWeight> decoder_layer;

    // Final RMSNorm before lm_head
    Tensor final_norm;  // [hidden_size]

    // Shared pointers to main model weights (not owned)
    const LlamaDenseWeight* embed_tokens = nullptr;  // → LlamaWeight::pre_decoder_embedding
    const LlamaDenseWeight* lm_head      = nullptr;  // → LlamaWeight::post_decoder_embedding

    // Per-layer shared_head (Step3p5 MTP only, optional)
    Tensor shared_head_norm;                              // [hidden_size]
    std::optional<LlamaDenseWeight> shared_head_output;   // hidden→vocab
    bool has_shared_head = false;
};

}  // namespace turbomind

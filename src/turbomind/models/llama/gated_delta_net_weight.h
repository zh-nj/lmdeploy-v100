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

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct GatedDeltaNetWeight: public core::Module {

    GatedDeltaNetWeight() = default;

    GatedDeltaNetWeight(int                       hidden_size,
                        const GatedDeltaNetParam& gdn,
                        int                       tp_size,
                        int                       tp_rank,
                        DataType                  data_type);

    // Linear projection weights (non-quantized, fp16)
    LlamaDenseWeight in_proj_all;   // [hidden_size, (key_dim*2 + value_dim*2 + num_v_heads*2) / tp]
    LlamaDenseWeight out_proj;      // [value_dim / tp, hidden_size]

    // Non-linear weights
    Tensor conv1d_weight;  // [conv_dim / tp, 1, kernel_size], depthwise conv
    Tensor A_log;          // [num_v_heads / tp], log decay factor
    Tensor dt_bias;        // [num_v_heads / tp], time step bias
    Tensor norm_weight;    // [head_v_dim], gated RMS norm weight (not TP-sharded)
};

}  // namespace turbomind

/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.cc

#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/gated_delta_net_weight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

static bool is_fuse_silu_act()
{
    static const bool value = [] {
        const auto str = std::getenv("TM_FUSE_SILU_ACT");
        if (str) {
            try {
                auto v = std::stoi(str) != 0;
                TM_LOG_INFO("TM_FUSE_SILU_ACT=%d", (int)v);
                return v;
            }
            catch (...) {
            }
        }
        // TM_LOG_INFO("TM_FUSE_SILU_ACT=1");
        return true;
    }();
    return value;
}

LlamaDecoderLayerWeight::LlamaDecoderLayerWeight(
    DataType data_type, int layer_id, const ModelParam& model, const EngineParam& engine, const MoeParam& moe_param):
    head_num_(layer_id < (int)model.head_num_per_layer.size() ? model.head_num_per_layer[layer_id] : model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    inter_size_(model.inter_size.at(layer_id)),
    data_type_{data_type},
    weight_type_(model.weight_type),
    expert_weight_type_(model.expert_weight_type),
    attn_bias_(model.attn_bias),
    attn_tp_size_(engine.attn_tp_size),
    attn_tp_rank_(engine.attn_tp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    mlp_tp_rank_(engine.mlp_tp_rank)
{
    // Determine layer type: 0=full_attention (default), 1=linear_attention
    const int layer_type = (layer_id < (int)model.layer_types.size()) ? model.layer_types[layer_id] : 0;

    if (layer_type == 1) {
        // linear_attention layer: use GatedDeltaNetWeight
        linear_attn_weights.reset(
            new GatedDeltaNetWeight{hidden_units_, model.gdn, attn_tp_size_, attn_tp_rank_, data_type_});
        register_module("linear_attn", *linear_attn_weights);
    }
    else {
        // full_attention layer: use LlamaAttentionWeight
        self_attn_weights.reset(new LlamaAttentionWeight{hidden_units_,
                                                         size_per_head_,
                                                         head_num_,
                                                         kv_head_num_,
                                                         model.mla,
                                                         attn_bias_,
                                                         model.qk_norm,
                                                         model.attn_output_gate,
                                                         model.attn_output_gate_per_head,
                                                         attn_tp_size_,
                                                         attn_tp_rank_,
                                                         data_type_,
                                                         weight_type_,
                                                         model.group_size,
                                                         model.window_size.empty() ? 0 : model.window_size.at(layer_id),
                                                         model.attn_sink});
        register_module("attention", *self_attn_weights);
    }

    if (inter_size_) {
        const bool is_cublas_gemm = byte_size(weight_type_, 8) == 16;
        // Use per-layer FFN weight type if available (e.g. Step3p5 where
        // shared expert is fp16 but dense MLP is int4).
        DataType ffn_wt = weight_type_;
        if (layer_id < (int)model.ffn_weight_types.size()) {
            ffn_wt = model.ffn_weight_types[layer_id];
        }
        const int ffn_group_size = (ffn_wt == kUint4 || ffn_wt == kUint8) ? model.group_size : 1;
        const bool is_ffn_cublas = byte_size(ffn_wt, 8) == 16;
        
        // ffn_weights handles both dense MLP (non-MoE layers) and shared expert (MoE layers).
        // Both use swiglu_limits_shared (not swiglu_limits, which is for MoE experts).
        // See official modeling_step3p5.py: Step3p5MLP uses swiglu_limit_shared for both cases.
        const float ffn_swiglu_limit = model.swiglu_limits_shared.empty()
            ? 0.f : model.swiglu_limits_shared.at(layer_id);

        bool allow_fuse = is_fuse_silu_act() && !is_ffn_cublas;
        // Disable fused SiLU if this layer has clamped SwiGLU activation
        if (ffn_swiglu_limit > 0.f) {
            allow_fuse = false;
        }

        ffn_weights.reset(new LlamaFfnWeight{
            hidden_units_,
            inter_size_,
            model.mlp_bias,
            mlp_tp_size_,
            mlp_tp_rank_,
            data_type_,
            ffn_wt,
            ffn_group_size,
            model.act_type,
            allow_fuse,
            ffn_swiglu_limit
        });
        register_module("feed_forward", *ffn_weights);
    }

    if (layer_id < moe_param.expert_num.size() && moe_param.expert_num[layer_id]) {
        // Use per-layer expert GGML type if available (UD models).
        gguf::GGMLType layer_ggml_type = static_cast<gguf::GGMLType>(model.expert_ggml_type);
        if (layer_id < (int)model.expert_ggml_types.size()) {
            layer_ggml_type = static_cast<gguf::GGMLType>(model.expert_ggml_types[layer_id]);
        }
        // Per-layer w2 (down_proj) GGML type — UD models may differ from w1/w3.
        gguf::GGMLType layer_ggml_type_w2 = gguf::GGML_TYPE_COUNT;
        if (layer_id < (int)model.expert_ggml_types_w2.size()) {
            layer_ggml_type_w2 = static_cast<gguf::GGMLType>(model.expert_ggml_types_w2[layer_id]);
        }
        // Use per-layer expert weight type if available (mixed-quantization models).
        DataType layer_expert_wt = expert_weight_type_;
        if (layer_id < (int)model.expert_weight_types.size()) {
            layer_expert_wt = model.expert_weight_types[layer_id];
            if (layer_expert_wt != expert_weight_type_) {
                TM_LOG_INFO("[Layer %d] using per-layer expert_weight_type=%d (default=%d)",
                            layer_id, (int)layer_expert_wt, (int)expert_weight_type_);
            }
        }

        bool allow_fuse = is_fuse_silu_act();
        // Step3.5: Disable fused SiLU if this layer has clamped SwiGLU activation
        if (!model.swiglu_limits.empty() && model.swiglu_limits.at(layer_id) > 0.f) {
            allow_fuse = false;
        }
        moe_weights.reset(new MoeFfnWeight{layer_id,
                                           moe_param,
                                           hidden_units_,
                                           model.mlp_bias,
                                           data_type_,
                                           layer_expert_wt,
                                           model.group_size,
                                           mlp_tp_size_,
                                           mlp_tp_rank_,
                                           model.act_type,
                                           allow_fuse,
                                           layer_ggml_type,
                                           layer_ggml_type_w2,
                                           model.swiglu_limits.empty() ? 0.f : model.swiglu_limits.at(layer_id),
                                           model.swiglu_limits_shared.empty() ? 0.f : model.swiglu_limits_shared.at(layer_id)});
        register_module("moe_ffn", *moe_weights);
    }

    self_attn_norm = Tensor{{hidden_units_}, data_type_, kDEVICE};
    ffn_norm       = Tensor{{hidden_units_}, data_type_, kDEVICE};
    register_parameter("attention_norm.weight", self_attn_norm);
    register_parameter("ffn_norm.weight", ffn_norm);
}

LlamaDecoderLayerWeight::~LlamaDecoderLayerWeight() = default;

void LlamaDecoderLayerWeight::prepare(const cudaDeviceProp& prop, cudaStream_t st)
{
    if (self_attn_weights) {
        self_attn_weights->prepare();
    }

    if (ffn_weights) {
        ffn_weights->prepare(false);
    }

    if (moe_weights) {
        moe_weights->prepare();
    }
}

}  // namespace turbomind

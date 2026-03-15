// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>
#include <map>
#include <regex>
#include <string>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind {

struct MLAParam {
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_dim;
    int v_head_dim;
};

struct GatedDeltaNetParam {
    int key_head_dim;    // 128
    int value_head_dim;  // 128
    int num_key_heads;   // 16
    int num_value_heads; // 32
    int conv_kernel_dim; // 4
};

struct ModelParam {
    size_t   head_num;
    size_t   head_dim;
    size_t   kv_head_num;
    size_t   hidden_units;
    size_t   layer_num;
    size_t   vocab_size;
    size_t   embedding_size;
    float    norm_eps;
    int      quant_policy;
    bool     attn_bias;
    bool     attn_sink;
    bool     attn_output_gate;  // gated attention output (Qwen3-Next: output *= sigmoid(gate))
    bool     attn_output_gate_per_head;  // true: per-head scalar gate (Step3p5), false: per-element gate (Qwen3-Next)
    bool     mlp_bias;
    DataType data_type;
    DataType weight_type;
    DataType expert_weight_type;
    // Per-layer expert weight types for mixed-quantization models
    // (e.g. AWQ models where some layers are not quantized).
    // Empty means use expert_weight_type for all layers.
    std::vector<DataType> expert_weight_types;
    // Per-layer FFN weight types for models where shared expert / dense MLP
    // uses a different precision than the global weight_type.
    // (e.g. Step3p5 where dense MLP is int4 but shared expert is fp16)
    // Empty means use weight_type for all layers.
    std::vector<DataType> ffn_weight_types;
    int      group_size;
    int      expert_ggml_type;  // gguf::GGMLType for GGUF expert weights (GGML_TYPE_COUNT = not GGUF)
    // Per-layer expert GGML types for UD (Unsloth Dynamic) models.
    // Empty means use expert_ggml_type for all layers.
    std::vector<int> expert_ggml_types;
    // Per-layer expert GGML types for w2 (down_proj) only.
    // UD models may use different types for w2 vs w1/w3.
    // Empty means w2 uses the same type as w1/w3 (expert_ggml_types).
    std::vector<int> expert_ggml_types_w2;
    MLAParam           mla;
    GatedDeltaNetParam gdn;
    bool               qk_norm;
    std::string        qk_norm_type;  // "per_head" (default) or "per_token" (MiniMax-M2 style)
    int                tune_layer_num;

    std::vector<int> layer_types;  // 0=full_attention, 1=linear_attention

    ActivationType act_type;

    std::vector<float> swiglu_limits;
    std::vector<float> swiglu_limits_shared;

    std::vector<int> window_size;
    std::vector<int> inter_size;

    // Per-layer attention head counts (empty = use head_num for all layers)
    std::vector<int> head_num_per_layer;

    // MTP speculative decoding
    int         num_mtp_layers   = 0;  // Number of MTP predictor layers (0 = disabled)
    int         num_draft_tokens = 0;  // Max draft tokens K per speculative decoding cycle
    std::string mtp_expert_weight_type;  // "fp16" = force fp16, empty = inherit from ref layer
    bool        mtp_has_shared_head = false;  // Step3p5 MTP: per-layer shared_head (norm + lm_head)
};

/// TODO: rename all `gate` in the context of MoE router to `router`
struct MoeParam {
    enum Method
    {
        kNaive,
        kFused
    } method;

    int   experts_per_token;
    int   inter_size;
    bool  norm_topk_prob;
    bool  shared_gate;
    float routed_scale;

    bool router_bias;

    int         topk_group;
    std::string topk_method;
    std::string scoring_func;  // "softmax" or "sigmoid"
    int         n_group;
    bool        fp32_gate;  // use fp32 precision for gate matmul (Step3p5: need_fp32_gate=true)

    std::vector<int> expert_num;
};

struct AttentionParam {
    float softmax_scale;
    int   cache_block_seq_len;
    // logn attention
    bool use_logn_attn;
    int  max_position_embeddings;
    // rotary embedding
    RopeParam rope;
    // Per-layer RoPE overrides (empty = use global `rope` for all layers)
    std::vector<RopeParam> rope_per_layer;
};

struct EngineParam {
    // batch params
    int max_batch_size;
    int session_len;
    int step_length;

    // cache params
    float cache_max_block_count;
    int   cache_chunk_size;
    bool  enable_prefix_caching;
    bool  enable_metrics;

    // chunking params
    int max_forward_token_num;
    int max_context_token_num;
    int num_tokens_per_iter;
    int max_prefill_iters;

    // parallel params
    int outer_dp_size;
    int outer_dp_rank;
    int attn_dp_size;
    int attn_dp_rank;
    int attn_tp_size;
    int attn_tp_rank;
    int attn_cp_size;
    int attn_cp_rank;
    int mlp_tp_size;
    int mlp_tp_rank;

    // multi-node
    int nnodes;
    int node_rank;

    std::vector<int> devices;

    // MTP speculative decoding
    bool speculative_decoding = false;  // Top-level enable flag
};

}  // namespace turbomind

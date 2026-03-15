// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cublas_v2.h>
#include <unordered_map>

#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class MoeFfnLayer {
public:
    MoeFfnLayer(const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx);

    ~MoeFfnLayer();

    struct ForwardParam {
        Tensor              input;
        Tensor              output;
        const MoeFfnWeight* weights;
        float               scale;
        int                 layer_id;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    Tensor_<float> Gate(const Tensor& input, const LlamaDenseWeight& gate, bool apply_bias = true);

    void dump_logits(int token_num, int layer_id, int expert_num);

    const int inter_size_;
    const int hidden_dim_;
    const int tp_size_;

    const MoeParam param_;

    int& is_warm_up_;

    LlamaLinear& linear_;

    std::unique_ptr<LlamaFfnLayer> expert_ffn_;

    ///////////////////////////////////////////////////////
    /// runtime states
    Buffer_<int> h_offsets_;

    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;

    Tensor         temp_;
    Tensor         inter_buf_;
    Tensor_<float> shared_scales_;

    std::unordered_map<int, Buffer_<float>> router_bias_f32_;  // per-layer cached float32 router bias

    cublasHandle_t cublas_handle_{};  // for fp32 gate matmul
    ///////////////////////////////////////////////////////
};

}  // namespace turbomind

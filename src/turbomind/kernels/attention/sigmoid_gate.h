// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

namespace turbomind {

// Fused sigmoid gate: output[i] = output[i] * sigmoid(gate[i])
// Both output and gate have shape (token_num, dim)
void invokeSigmoidGate(void* output, const void* gate, int count, int dtype_size, cudaStream_t stream);

// Broadcast sigmoid gate: output[t, h*head_dim + d] *= sigmoid(gate[t, h])
// output shape: (token_num, num_heads * head_dim), gate shape: (token_num, num_heads)
void invokeSigmoidGateBroadcast(
    void* output, const void* gate, int token_num, int num_heads, int head_dim, int dtype_size, cudaStream_t stream);

}  // namespace turbomind

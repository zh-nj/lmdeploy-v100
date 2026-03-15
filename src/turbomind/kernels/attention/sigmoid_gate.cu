// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/sigmoid_gate.h"
#include <cuda_fp16.h>

namespace turbomind {

template<typename T>
__global__ void sigmoid_gate_kernel(T* output, const T* gate, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float g = (float)gate[idx];
        float s = 1.0f / (1.0f + expf(-g));
        float o = (float)output[idx];
        output[idx] = (T)(o * s);
    }
}

void invokeSigmoidGate(void* output, const void* gate, int count, int dtype_size, cudaStream_t stream)
{
    const int block = 256;
    const int grid  = (count + block - 1) / block;
    if (dtype_size == 2) {
        sigmoid_gate_kernel<<<grid, block, 0, stream>>>((half*)output, (const half*)gate, count);
    }
}

// Broadcast variant: gate has shape (token_num, num_heads), output has shape (token_num, num_heads * head_dim)
// Each gate scalar is broadcast across head_dim elements.
template<typename T>
__global__ void sigmoid_gate_broadcast_kernel(T* output, const T* gate, int total, int num_heads, int head_dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        // idx = t * (num_heads * head_dim) + h * head_dim + d
        const int out_dim = num_heads * head_dim;
        const int t       = idx / out_dim;
        const int rem     = idx % out_dim;
        const int h       = rem / head_dim;
        float     g       = (float)gate[t * num_heads + h];
        float     s       = 1.0f / (1.0f + expf(-g));
        float     o       = (float)output[idx];
        output[idx]       = (T)(o * s);
    }
}

void invokeSigmoidGateBroadcast(
    void* output, const void* gate, int token_num, int num_heads, int head_dim, int dtype_size, cudaStream_t stream)
{
    const int total = token_num * num_heads * head_dim;
    const int block = 256;
    const int grid  = (total + block - 1) / block;
    if (dtype_size == 2) {
        sigmoid_gate_broadcast_kernel<<<grid, block, 0, stream>>>(
            (half*)output, (const half*)gate, total, num_heads, head_dim);
    }
}

}  // namespace turbomind

// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/kernels/gated_delta_net/gdn_utils.h"

namespace turbomind {

__global__ void extractColumnsKernel(half*       dst,
                                     const half* src,
                                     int         rows,
                                     int         width,
                                     int         col_offset,
                                     int         src_stride)
{
    const int r = blockIdx.x;
    if (r >= rows) return;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        dst[r * width + c] = src[r * src_stride + col_offset + c];
    }
}

void invokeExtractColumns(half*        dst,
                           const half*  src,
                           int          rows,
                           int          width,
                           int          col_offset,
                           int          src_stride,
                           cudaStream_t stream)
{
    if (rows == 0 || width == 0) return;
    int block = min(width, 1024);
    extractColumnsKernel<<<rows, block, 0, stream>>>(dst, src, rows, width, col_offset, src_stride);
}

__global__ void repeatInterleaveKernel(half*       dst,
                                       const half* src,
                                       int         token_num,
                                       int         num_k_heads,
                                       int         head_dim,
                                       int         kv_ratio)
{
    const int t = blockIdx.x;
    if (t >= token_num) return;
    const int num_v_heads = num_k_heads * kv_ratio;
    const int src_row     = t * num_k_heads * head_dim;
    const int dst_row     = t * num_v_heads * head_dim;
    for (int idx = threadIdx.x; idx < num_v_heads * head_dim; idx += blockDim.x) {
        int vh = idx / head_dim;
        int d  = idx % head_dim;
        int kh = vh / kv_ratio;
        dst[dst_row + vh * head_dim + d] = src[src_row + kh * head_dim + d];
    }
}

void invokeRepeatInterleave(half*        dst,
                             const half*  src,
                             int          token_num,
                             int          num_k_heads,
                             int          head_dim,
                             int          kv_ratio,
                             cudaStream_t stream)
{
    if (token_num == 0) return;
    int num_v_heads = num_k_heads * kv_ratio;
    int block = min(num_v_heads * head_dim, 1024);
    repeatInterleaveKernel<<<token_num, block, 0, stream>>>(
        dst, src, token_num, num_k_heads, head_dim, kv_ratio);
}

}  // namespace turbomind

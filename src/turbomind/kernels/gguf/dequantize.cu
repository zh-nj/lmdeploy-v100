// Copyright (c) OpenMMLab. All rights reserved.
// GGUF dequantization CUDA kernels.
// Reference: llama.cpp ggml-cuda/convert.cu

#include "dequantize.h"

#include <cuda_fp16.h>
#include <cstdio>

namespace turbomind {
namespace gguf {

// ---- Helper: decode 6-bit scale/min from K-quant scales[12] ----
// Matches llama.cpp get_scale_min_k4
static __device__ __forceinline__ void
get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m)
{
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    }
    else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ==================================================================
// Q8_0 kernel: block_size=32, 32 threads per block
// ==================================================================
static __global__ void dequantize_block_q8_0(const void* __restrict__ vx,
                                              half* __restrict__ y,
                                              int64_t k)
{
    const BlockQ8_0* x = (const BlockQ8_0*)vx;
    const int64_t i  = (int64_t)blockIdx.x;  // block index
    const int     tid = threadIdx.x;          // 0..31

    if (i * 32 + tid >= k) return;

    const float d = __half2float(*reinterpret_cast<const half*>(&x[i].d));
    y[i * 32 + tid] = __float2half(x[i].qs[tid] * d);
}

// ==================================================================
// Q6_K kernel: super-block=256, 64 threads per block
// ==================================================================
static __global__ void dequantize_block_q6_k(const void* __restrict__ vx,
                                              half* __restrict__ yy)
{
    const BlockQ6K* x = (const BlockQ6K*)vx;
    const int64_t i   = blockIdx.x;
    const int     tid = threadIdx.x;  // 0..63
    const int     ip  = tid / 32;
    const int     il  = tid - 32 * ip;
    const int     is  = 8 * ip + il / 16;

    half* y = yy + i * QK_K + 128 * ip + il;

    const float d = __half2float(*reinterpret_cast<const half*>(&x[i].d));

    const uint8_t* ql = x[i].ql + 64 * ip + il;
    const uint8_t  qh = x[i].qh[32 * ip + il];
    const int8_t*  sc = x[i].scales + is;

    y[0]  = __float2half(d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = __float2half(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = __float2half(d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = __float2half(d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}


// ==================================================================
// Q5_K kernel: super-block=256, 64 threads per block
// ==================================================================
static __global__ void dequantize_block_q5_k(const void* __restrict__ vx,
                                              half* __restrict__ yy)
{
    const BlockQ5K* x = (const BlockQ5K*)vx;
    const int64_t i   = blockIdx.x;
    const int     tid = threadIdx.x;  // 0..63
    const int     il  = tid / 16;
    const int     ir  = tid % 16;
    const int     is  = 2 * il;

    half* y = yy + i * QK_K + 64 * il + 2 * ir;

    const float dall = __half2float(*reinterpret_cast<const half*>(&x[i].d));
    const float dmin = __half2float(*reinterpret_cast<const half*>(&x[i].dmin));

    const uint8_t* ql = x[i].qs + 32 * il + 2 * ir;
    const uint8_t* qh = x[i].qh + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    uint8_t hm = 1 << (2 * il);
    y[0]  = __float2half(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
    y[1]  = __float2half(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = __float2half(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
    y[33] = __float2half(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
}

// ==================================================================
// Q4_K kernel: super-block=256, 32 threads per block
// ==================================================================
static __global__ void dequantize_block_q4_k(const void* __restrict__ vx,
                                              half* __restrict__ yy)
{
    const BlockQ4K* x = (const BlockQ4K*)vx;
    const int64_t i   = blockIdx.x;
    const int     tid = threadIdx.x;  // 0..31
    const int     il  = tid / 8;
    const int     ir  = tid % 8;
    const int     is  = 2 * il;
    const int     n   = 4;

    half* y = yy + i * QK_K + 64 * il + n * ir;

    const float dall = __half2float(*reinterpret_cast<const half*>(&x[i].d));
    const float dmin = __half2float(*reinterpret_cast<const half*>(&x[i].dmin));

    const uint8_t* q = x[i].qs + 32 * il + n * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    for (int l = 0; l < n; ++l) {
        y[l + 0]  = __float2half(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = __float2half(d2 * (q[l] >> 4) - m2);
    }
}

// ==================================================================
// MXFP4 kernel: block_size=32, 1 thread per block
// ==================================================================
__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static __global__ void dequantize_block_mxfp4(const void* __restrict__ vx,
                                               half* __restrict__ y,
                                               int64_t n_blocks)
{
    const BlockMXFP4* x = (const BlockMXFP4*)vx;
    const int64_t bid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_blocks) return;

    const float d = ldexpf(1.0f, (int)x[bid].e - 128);
    half* out = y + bid * 32;
    for (int j = 0; j < 16; ++j) {
        uint8_t q = x[bid].qs[j];
        out[j]      = __float2half(kvalues_mxfp4[q & 0xF] * d);
        out[j + 16] = __float2half(kvalues_mxfp4[q >> 4] * d);
    }
}

// ==================================================================
// Unified dispatch
// ==================================================================
void dequantize_gguf(GGMLType     type,
                     const void*  src,
                     half*        dst,
                     int64_t      n,
                     cudaStream_t stream)
{
    switch (type) {
        case GGML_TYPE_Q8_0: {
            const int nb = (int)(n / 32);
            dequantize_block_q8_0<<<nb, 32, 0, stream>>>(src, dst, n);
            break;
        }
        case GGML_TYPE_Q6_K: {
            const int nb = (int)(n / QK_K);
            dequantize_block_q6_k<<<nb, 64, 0, stream>>>(src, dst);
            break;
        }
        case GGML_TYPE_Q5_K: {
            const int nb = (int)(n / QK_K);
            dequantize_block_q5_k<<<nb, 64, 0, stream>>>(src, dst);
            break;
        }
        case GGML_TYPE_Q4_K: {
            const int nb = (int)(n / QK_K);
            dequantize_block_q4_k<<<nb, 32, 0, stream>>>(src, dst);
            break;
        }
        case GGML_TYPE_F16: {
            // Direct copy (src is already fp16)
            cudaMemcpyAsync(dst, src, n * sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
            break;
        }
        case GGML_TYPE_MXFP4: {
            const int nb = (int)(n / QK_MXFP4);
            const int threads = 256;
            const int blocks = (nb + threads - 1) / threads;
            dequantize_block_mxfp4<<<blocks, threads, 0, stream>>>(
                src, dst, nb);
            break;
        }
        default:
            fprintf(stderr,
                    "[GGUF] Unsupported dequantize type: %d\n", (int)type);
            break;
    }
}

}  // namespace gguf
}  // namespace turbomind

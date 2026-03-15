// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "src/turbomind/kernels/gated_delta_net/delta_rule.h"

namespace {

void checkCuda(cudaError_t code, const char* what)
{
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(code));
        std::exit(1);
    }
}

int chooseSplitV(int batch_heads, int head_v_dim)
{
    if (batch_heads <= 0 || head_v_dim <= 0) return 1;
    int split_v = 1;
    if (batch_heads < 128 && head_v_dim >= 64) {
        split_v = 128 / batch_heads;
        if (split_v < 1) split_v = 1;
        if (split_v > 8) split_v = 8;
        if (split_v > head_v_dim) split_v = head_v_dim;
        while (split_v > 1) {
            const int vc = (head_v_dim + split_v - 1) / split_v;
            if (vc >= 16) break;
            split_v >>= 1;
        }
    }
    return split_v;
}

int parseOrDefault(char** argv, int idx, int default_value)
{
    return argv[idx] ? std::atoi(argv[idx]) : default_value;
}

}  // namespace

int main(int argc, char** argv)
{
    // Defaults are chosen to hit v_chunk > 16 path on local V heads.
    const int batch      = argc > 1 ? parseOrDefault(argv, 1, 32) : 32;
    const int token_num  = argc > 2 ? parseOrDefault(argv, 2, 2048) : 2048;
    const int num_heads  = argc > 3 ? parseOrDefault(argv, 3, 8) : 8;
    const int q_num_heads = argc > 4 ? parseOrDefault(argv, 4, 4) : 4;
    const int head_k_dim = argc > 5 ? parseOrDefault(argv, 5, 128) : 128;
    const int head_v_dim = argc > 6 ? parseOrDefault(argv, 6, 128) : 128;
    const int warmup     = argc > 7 ? parseOrDefault(argv, 7, 20) : 20;
    const int iters      = argc > 8 ? parseOrDefault(argv, 8, 100) : 100;

    const int kv_ratio = num_heads / std::max(1, q_num_heads);
    const int q_stride = q_num_heads * head_k_dim;
    const int k_stride = q_num_heads * head_k_dim;
    const int v_stride = num_heads * head_v_dim;

    const int batch_heads = batch * num_heads;
    int split_v = chooseSplitV(batch_heads, head_v_dim);
    if (token_num >= 1024 && batch_heads < 64 && head_v_dim >= 64) {
        while (split_v < 16) {
            const int next_split = split_v << 1;
            const int next_chunk = (head_v_dim + next_split - 1) / next_split;
            if (next_chunk < 8) break;
            split_v = next_split;
        }
    }
    const int v_chunk = (head_v_dim + split_v - 1) / split_v;

    const bool use_reg       = (v_chunk <= 16);
    const bool use_tiled_reg = (!use_reg && token_num >= 512 && v_chunk >= 32);
    const char* path = use_reg ? "reg" : (use_tiled_reg ? "tiled_reg" : "generic");

    std::printf("Config: batch=%d token_num=%d num_heads=%d q_num_heads=%d "
                "head_k_dim=%d head_v_dim=%d split_v=%d v_chunk=%d path=%s\n",
                batch, token_num, num_heads, q_num_heads, head_k_dim, head_v_dim, split_v, v_chunk, path);

    const size_t q_elems = (size_t)token_num * q_stride;
    const size_t k_elems = (size_t)token_num * k_stride;
    const size_t v_elems = (size_t)token_num * v_stride;
    const size_t out_elems = (size_t)token_num * num_heads * head_v_dim;
    const size_t state_elems = (size_t)batch * num_heads * head_k_dim * head_v_dim;
    const int alpha_stride = num_heads * 2;  // [b, a] per token row
    const int beta_stride  = num_heads * 2;
    const int alpha_col_offset = num_heads;
    const int beta_col_offset  = 0;
    const size_t ba_elems = (size_t)token_num * alpha_stride;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<half> h_q(q_elems), h_k(k_elems), h_v(v_elems), h_a_log(num_heads), h_dt_bias(num_heads), h_ba(ba_elems);
    for (auto& x : h_q) x = __float2half(dist(rng));
    for (auto& x : h_k) x = __float2half(dist(rng));
    for (auto& x : h_v) x = __float2half(dist(rng));
    for (auto& x : h_a_log) x = __float2half(-2.5f + dist(rng));
    for (auto& x : h_dt_bias) x = __float2half(dist(rng));
    for (auto& x : h_ba) x = __float2half(dist(rng));

    std::vector<int> h_cu_seqlens(batch + 1, 0);
    const int base = token_num / batch;
    const int rem  = token_num % batch;
    int acc = 0;
    h_cu_seqlens[0] = 0;
    for (int b = 0; b < batch; ++b) {
        acc += base + (b < rem ? 1 : 0);
        h_cu_seqlens[b + 1] = acc;
    }

    half* d_q = nullptr;
    half* d_k = nullptr;
    half* d_v = nullptr;
    half* d_out = nullptr;
    half* d_a_log = nullptr;
    half* d_dt_bias = nullptr;
    half* d_ba = nullptr;
    float* d_state = nullptr;
    int* d_cu_seqlens = nullptr;

    checkCuda(cudaMalloc(&d_q, q_elems * sizeof(half)), "cudaMalloc d_q");
    checkCuda(cudaMalloc(&d_k, k_elems * sizeof(half)), "cudaMalloc d_k");
    checkCuda(cudaMalloc(&d_v, v_elems * sizeof(half)), "cudaMalloc d_v");
    checkCuda(cudaMalloc(&d_out, out_elems * sizeof(half)), "cudaMalloc d_out");
    checkCuda(cudaMalloc(&d_a_log, num_heads * sizeof(half)), "cudaMalloc d_a_log");
    checkCuda(cudaMalloc(&d_dt_bias, num_heads * sizeof(half)), "cudaMalloc d_dt_bias");
    checkCuda(cudaMalloc(&d_ba, ba_elems * sizeof(half)), "cudaMalloc d_ba");
    checkCuda(cudaMalloc(&d_state, state_elems * sizeof(float)), "cudaMalloc d_state");
    checkCuda(cudaMalloc(&d_cu_seqlens, (batch + 1) * sizeof(int)), "cudaMalloc d_cu_seqlens");

    checkCuda(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice), "copy q");
    checkCuda(cudaMemcpy(d_k, h_k.data(), k_elems * sizeof(half), cudaMemcpyHostToDevice), "copy k");
    checkCuda(cudaMemcpy(d_v, h_v.data(), v_elems * sizeof(half), cudaMemcpyHostToDevice), "copy v");
    checkCuda(cudaMemcpy(d_a_log, h_a_log.data(), num_heads * sizeof(half), cudaMemcpyHostToDevice), "copy a_log");
    checkCuda(cudaMemcpy(d_dt_bias, h_dt_bias.data(), num_heads * sizeof(half), cudaMemcpyHostToDevice), "copy dt_bias");
    checkCuda(cudaMemcpy(d_ba, h_ba.data(), ba_elems * sizeof(half), cudaMemcpyHostToDevice), "copy ba");
    checkCuda(cudaMemcpy(d_cu_seqlens, h_cu_seqlens.data(), (batch + 1) * sizeof(int), cudaMemcpyHostToDevice), "copy cu");
    checkCuda(cudaMemset(d_out, 0, out_elems * sizeof(half)), "memset out");
    checkCuda(cudaMemset(d_state, 0, state_elems * sizeof(float)), "memset state");

    for (int i = 0; i < warmup; ++i) {
        turbomind::invokeChunkDeltaRulePrefill(d_out,
                                               d_state,
                                               d_q,
                                               d_k,
                                               d_v,
                                               d_a_log,
                                               d_dt_bias,
                                               d_ba,
                                               d_ba,
                                               d_cu_seqlens,
                                               batch,
                                               token_num,
                                               num_heads,
                                               q_num_heads,
                                               kv_ratio,
                                               q_stride,
                                               k_stride,
                                               v_stride,
                                               head_k_dim,
                                               head_v_dim,
                                               0,
                                               0,
                                               alpha_stride,
                                               beta_stride,
                                               alpha_col_offset,
                                               beta_col_offset);
    }
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t e0, e1;
    checkCuda(cudaEventCreate(&e0), "create e0");
    checkCuda(cudaEventCreate(&e1), "create e1");

    checkCuda(cudaEventRecord(e0), "record e0");
    for (int i = 0; i < iters; ++i) {
        turbomind::invokeChunkDeltaRulePrefill(d_out,
                                               d_state,
                                               d_q,
                                               d_k,
                                               d_v,
                                               d_a_log,
                                               d_dt_bias,
                                               d_ba,
                                               d_ba,
                                               d_cu_seqlens,
                                               batch,
                                               token_num,
                                               num_heads,
                                               q_num_heads,
                                               kv_ratio,
                                               q_stride,
                                               k_stride,
                                               v_stride,
                                               head_k_dim,
                                               head_v_dim,
                                               0,
                                               0,
                                               alpha_stride,
                                               beta_stride,
                                               alpha_col_offset,
                                               beta_col_offset);
    }
    checkCuda(cudaEventRecord(e1), "record e1");
    checkCuda(cudaEventSynchronize(e1), "sync e1");

    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    const float avg_ms = ms / iters;
    const float tok_s = avg_ms > 0.f ? (token_num * 1000.f / avg_ms) : 0.f;
    std::printf("Result: avg_ms=%.4f tok_per_s=%.2f\n", avg_ms, tok_s);

    checkCuda(cudaEventDestroy(e0), "destroy e0");
    checkCuda(cudaEventDestroy(e1), "destroy e1");

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_a_log);
    cudaFree(d_dt_bias);
    cudaFree(d_ba);
    cudaFree(d_state);
    cudaFree(d_cu_seqlens);
    return 0;
}


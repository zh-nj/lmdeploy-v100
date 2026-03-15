// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeRMSNorm(Tensor& out, const Tensor& x, const Tensor& w, float eps, cudaStream_t st, int global_dim = 0);

// Two-phase RMS norm for TP-aware variance computation:
// Phase 1: compute per-token sum of squares (local partial sum)
void invokeRMSNormVariance(Tensor& variance, const Tensor& x, cudaStream_t st);

// Phase 2: apply normalization using pre-computed variance (after all-reduce)
// variance: [num_tokens, 1] float, containing the global sum of squares
// global_dim: the total dimension across all TP ranks
void invokeRMSNormApply(Tensor& out, const Tensor& x, const Tensor& w, const Tensor& variance, float eps, int global_dim, cudaStream_t st);

void invokeRMSNormQK(Tensor& x, const Tensor& w, float eps, cudaStream_t st);

// Fused two-phase QK norm for per-token RMS norm with TP all-reduce:
// Phase 1: compute sum-of-squares for Q and K regions of QKV tensor in a single kernel
void invokeRMSNormVarianceQK(Tensor& q_var, Tensor& k_var, const Tensor& qkv, int q_dim, int k_dim, cudaStream_t st);

// Phase 2: apply normalization to Q and K regions using pre-computed variance in a single kernel
void invokeRMSNormApplyQK(Tensor&       qkv,
                           int           q_dim,
                           int           k_dim,
                           const Tensor& q_weight,
                           const Tensor& k_weight,
                           const Tensor& q_var,
                           const Tensor& k_var,
                           float         eps,
                           int           global_q_dim,
                           int           global_k_dim,
                           cudaStream_t  st);

template<class T>
void invokeBiasResidualRMSNorm(
    T* residual, T* hidden_states, const T* weights, const T* bias, int dims, int num, float eps, cudaStream_t st);

void invokeResidualBiasRMSNorm(void*        hidden_states,
                               void*        residual,
                               const void*  weights,
                               const void*  bias,
                               DataType     dtype,
                               int          dims,
                               int          num,
                               float        eps,
                               cudaStream_t st);

void ApplyBias(Tensor& x, const Tensor& bias, const Buffer_<int>& offsets, float scale, cudaStream_t st);

void ApplyBias(Tensor& x, const Tensor& bias, cudaStream_t st);

}  // namespace turbomind

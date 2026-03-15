// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/core/thread_map.h"

#include <cmath>
#include <type_traits>

namespace turbomind::attention {

// ============================================================================
// Decode-specific MMA_884 impl for sm70.
//
// Dimension mapping (same as MMA_81616 decode): S→M, H→N, D→K
//
// QK: S[S,H] = K[S,D] · Q[H,D]^T  →  mma_m8n8k4_row_col(S, K, Q, S)
//   K = A operand (row-major), Q = B operand (col-major)
// PV: O[D,H] = V^T[D,S] · P[S,H]  →  mma_m8n8k4_row_row(O, V, P, O)
//   V = A operand (row-major), P = B operand (row-major)
//
// mma_m8n8k4 output: Array<float, 8> per [m][n] tile
//   8 values indexed as [s1*4 + q*2 + s0]
//   M-dim (=S): determined by q ∈ {0,1} and lane bits {0,3,4}
//   N-dim (=H): determined by s1,s0 ∈ {0,1} and lane bits {1,2}
//
// Thread-to-position mapping:
//   S position: m*16 + (lane&8) + (lane&1) + lane/16*4 + q*2
//   H position: n*16 + (lane&4)*2 + (lane&2) + s1*4 + s0
//
// S reduction shuffles: xor 1, 8, 16 (lane bits 0, 3, 4)
// H positions per thread: 4 (s1*4+s0 = {0,1,4,5})
// ============================================================================
template<class T_,
         int CTA_H_,
         int CTA_Q_,
         int CTA_S_,
         int WARP_H_,
         int WARP_Q,
         int WARP_S,
         int HeadDim,
         int Stages>
struct Impl<MMA_884_DEC, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q, WARP_S, HeadDim, Stages> {
    using T   = T_;
    using Tkv = T_;

    static_assert(CTA_Q_ == 1, "MMA_884_DEC is decode-only (CTA_Q=1)");

    static constexpr int CTA_H = CTA_H_;
    static constexpr int CTA_Q = CTA_Q_;
    static constexpr int CTA_S = CTA_S_;
    static constexpr int WARP_H = WARP_H_;
    static constexpr int kHeadDim = HeadDim;

    static constexpr int kWarpCntH = CTA_H / WARP_H;
    static constexpr int kWarpCntQ = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    // mma_m8n8k4 tile sizes
    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 4;

    // QK dimensions: S[S,H] = K[S,D] · Q[H,D]^T
    static constexpr int K_M = WARP_S / OP_M;               // 1
    static constexpr int K_N = (CTA_H + OP_N - 1) / OP_N;   // 1 (CTA_H<=16)
    static constexpr int K_K = HeadDim / OP_K;               // HeadDim/4

    // PV dimensions: O[D,H] = V^T[D,S] · P[S,H]
    static constexpr int V_M = HeadDim / OP_M;               // HeadDim/16
    static constexpr int V_N = K_N;                           // same H dimension
    static constexpr int V_K = WARP_S / OP_K;                // 4

    static constexpr int CTA_H1 = (CTA_H + OP_N - 1) / OP_N * OP_N;

    // ---- Fragment types ----
    // QK: K=A(row), Q=B(col)
    using FragK = Array<half, 4>[K_K][K_M];
    using FragQ = Array<half, 4>[K_N][K_K];
    using FragS = Array<float, 8>[K_M][K_N];

    // PV: V=A(row), P=B(row)
    using FragV = Array<half, 4>[V_M][V_K];
    using FragP = Array<half, 4>[V_K][V_N];
    using FragO = Array<float, 8>[V_M][V_N];

    // Per-H max/sum: 4 H sub-positions per thread (s1*2+s0 → H offsets {0,1,4,5})
    using FragM = Array<float, 4>[K_N];
    using FragL = FragM;

    // Cross-warp reduction storage
    // SmemM: 4 floats per thread-group × kWarpCntS warps × 4 thread-groups per warp
    using SmemM = Array<float, 4>[K_N][kWarpCntS][4];
    using SmemO = Array<float, 8>[V_M][V_N][kWarpCntS][WARP_SIZE];

    struct SwizzleV {
        __device__ static int apply(int offset)
        {
            offset = ((offset & 8) << 2) ^ offset;
            offset = ((offset & ~20) | (((offset & 16) >> 2) | ((offset & 4) << 2)));
            offset = ((offset & (0x3 << 6)) >> 3) ^ offset;
            return offset;
        }
        __device__ int operator()(int offset) { return apply(offset); }
    };

    using SmemLayoutQ = SmemLayoutV2<CTA_H1, HeadDim + 4, 1, 1, Identity>;
    using SmemLayoutK = SmemLayoutV2<CTA_S, HeadDim + 4, 1, 1, Identity>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, HeadDim, CTA_S, 64, SwizzleV>;
    using SmemLayoutP = SmemLayoutV2<CTA_H1, CTA_S + 4, 1, 1, Identity>;

    using SmemLayoutKVp = void;

    union SharedStorage {
        __align__(16) T Q[SmemLayoutQ::kSize];
        struct {
            __align__(16) T K[SmemLayoutK::kSize];
            __align__(16) T V[SmemLayoutV::kSize];
            __align__(16) T P[SmemLayoutP::kSize];
        };
        struct {
            __align__(16) SmemM M;
            __align__(16) SmemM L;
            __align__(16) SmemO O;
        };
        __align__(16) float O1[CTA_H1][kHeadDim];
    };

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_H1, 4, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 4, kWarpCount>;
    using ThreadMapKVp = void;

    static constexpr bool kDeferReduceL = true;

    __device__ static void Sync()
    {
        __syncthreads();
    }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        gmem_K.SetSmem(storage.K);
        gmem_V.SetSmem(storage.V);
    }

    // ========================================================================
    // ForeachS: iterate over S fragment positions
    // ========================================================================
    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            const int si = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2
                                           + warp_id * WARP_S;
                            const int hi = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            ((Func &&) func)(hi, /*qi*/ 0, si, /*ri*/ 0, S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // ForeachML: iterate over per-H M/L values
    // FragM = Array<float, 4>[K_N], indexed [n][s1*2+s0]
    // 4 H sub-positions per thread: s1*4+s0 = {0,1,4,5}
    // ri = S reduction index within warp
    // ========================================================================
    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int s1 = 0; s1 < 2; ++s1) {
                PRAGMA_UNROLL
                for (int s0 = 0; s0 < 2; ++s0) {
                    const int hi = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                    // S positions per thread: (lane&8)/8 + (lane&1) + lane/16*4 → 0..7
                    const int ri = (lane_id >> 3 & 1) + (lane_id & 1) + (lane_id >> 4) * 4;
                    ((Func &&) func)(hi, /*qi*/ 0, ri, frag_M[n][s1 * 2 + s0], frag_L[n][s1 * 2 + s0]);
                }
            }
        }
    }

    // ========================================================================
    // TransformQ: load Q[CTA_H1, HeadDim] from smem into FragQ
    // B operand of mma_m8n8k4_row_col
    // B thread mapping: col(N=H) = (lane&8) + lane%4 + lane/16*4
    //                   row(K=D) = k*4 consecutive
    // ========================================================================
    __device__ static void TransformQ(T* smem_Q, FragQ& frag_Q)
    {
        if constexpr (!kUseSmemQ) {
            const int lane_id = threadIdx.x % WARP_SIZE;
            SmemAccessor<T, SmemLayoutQ> sQ{smem_Q};

            // Zero-pad unused rows of Q smem.
            // Q is stored at rows 0..CTA_Q*CTA_H-1 (= 0..CTA_H-1 for decode).
            // MMA_884 reads from rows 0..CTA_H1-1 (padded to OP_N=16 boundary).
            // Rows CTA_H..CTA_H1-1 must be zero to avoid NaN in QK computation.
            if constexpr (CTA_H < CTA_H1) {
                const int tid = threadIdx.x;
                constexpr int stride = CTA_H1 - CTA_H;  // rows to zero
                constexpr int cols = kHeadDim;
                // Each thread zeros a portion of the unused rows
                for (int idx = tid; idx < stride * cols; idx += kWarpCount * WARP_SIZE) {
                    const int row = CTA_H + idx / cols;
                    const int col = idx % cols;
                    sQ(row, col) = T(0);
                }
                __syncthreads();
            }

            // B operand col-major: col = lane%4 [+4 if lane>=16], row = i
            // The (lane&4)*2 term differentiates Comp 1/3 (H offset 0) vs Comp 2/4 (H offset 8)
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; ++k) {
                    const int hi = n * OP_N + lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
                    const int di = k * 4;
                    Lds(frag_Q[n][k], &sQ(hi, di));
                }
            }
        }
    }

    // ========================================================================
    // StateQK: K=A operand (row), Q=B operand (col)
    // A thread mapping: row(M=S) = (lane&8) + lane%4 + lane/16*4
    //                   col(K=D) = k*4 consecutive
    // ========================================================================
    struct StateQK {
        SmemAccessor<T, SmemLayoutK> smem_K;
        FragQ frag_Q;
        FragK frag_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_): smem_K{storage.K}
        {
            static_assert(!kUseSmemQ);
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; ++k) {
                    frag_Q[n][k] = frag_Q_[n][k];
                }
            }
        }

        __device__ void Load(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                const int s = m * 16 + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * WARP_S;
                const int c = k * 4;
                Lds(frag_K[k][m], &smem_K(s, c));
            }
        }

        __device__ void Transform(int k) {}
    };

    // ========================================================================
    // ComputeQK: S = K · Q^T using mma_m8n8k4_row_col
    // ========================================================================
    template<class Prefetch, class Preload>
    __device__ static void
    ComputeQK(StateQK& state_QK, FragS& frag_S, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.Load(k + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    mma_m8n8k4_row_col(frag_S[m][n], state_QK.frag_K[k][m], state_QK.frag_Q[n][k], frag_S[m][n]);
                }
            }
        }
    }

    // ========================================================================
    // StatePV: V=A operand (row), P=B operand (row)
    // V in smem is [CTA_S, HeadDim], need V^T[D,S] as A[M=D, K=S]
    //
    // A operand for mma_m8n8k4 row-major (8×4):
    //   Each thread holds 4 elements = one row of A = V^T[d_thread, s..s+3]
    //   d_thread within 16-row tile: (lane&8) + (lane>>4)*4 + (lane&3)
    //   s values: s_base + 0..3
    //
    // V^T[d, s] = V[s, d], so we need V[s_base+i, d_thread] for i=0..3
    // These are at 4 different rows in smem V[S, D], NOT consecutive.
    // ========================================================================
    struct StatePV {
        T* smem_V;

        static_assert(V_M % 2 == 0 || V_M == 1);

        FragP frag_P;
        FragV frag_V;

        __device__ StatePV(SharedStorage& storage, bool offset): smem_V{storage.V}
        {
            assert(offset);
        }

        __device__ void Load(int m, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            const int d = m * 16 + (lane_id & 8) + (lane_id >> 4) * 4 + (lane_id & 3);
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                const int s_base = k * 4 + warp_id * WARP_S;
                PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    frag_V[m][k][i] = smem_V[SmemLayoutV::apply(s_base + i, d)];
                }
            }
        }

        __device__ void Transform(int m) {}
    };

    // ========================================================================
    // ComputePV: O = V^T · P using mma_m8n8k4_row_row
    // ========================================================================
    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV& state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            if (m < V_M - 1) {
                state_PV.Load(m + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m8n8k4_row_row(frag_O[m][n], state_PV.frag_V[m][k], state_PV.frag_P[k][n], frag_O[m][n]);
                }
            }
        }
    }

    // ========================================================================
    // Softmax: online softmax over S dimension
    //
    // FragS[m][n] has 8 values: [s1*4 + q*2 + s0]
    //   q ∈ {0,1} → S positions (reduce over these)
    //   s1,s0 → H positions (keep these)
    //
    // FragM[n] has 4 values: [s1*2+s0] → per-H max
    //
    // S reduction within thread: reduce over q (2 values)
    // S reduction across threads: shuffle xor 1, 8, 16
    // ========================================================================
    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragO& frag_O, float qk_scale)
    {
#if 0  // DEBUG: dump S values before softmax
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            printf("[MMA884DEC] HD=%d Softmax: S[0][0] = {%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f}\n",
                   kHeadDim,
                   frag_S[0][0][0], frag_S[0][0][1], frag_S[0][0][2], frag_S[0][0][3],
                   frag_S[0][0][4], frag_S[0][0][5], frag_S[0][0][6], frag_S[0][0][7]);
        }
#endif
        FragM prev_M;
        copy(frag_M, prev_M);

        // Step 1: Find max over S for each H position
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int s0 = 0; s0 < 2; ++s0) {
                        // Reduce over q (S positions) for this H position
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            frag_M[n][s1 * 2 + s0] =
                                fmaxf(frag_M[n][s1 * 2 + s0], frag_S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }

        // Cross-thread S reduction: lanes with same H but different S positions
        // S is in M-dim, distributed via lane bits 0, 3, 4
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                frag_M[n][i] = fmaxf(frag_M[n][i], __shfl_xor_sync(uint32_t(-1), frag_M[n][i], 1));
                frag_M[n][i] = fmaxf(frag_M[n][i], __shfl_xor_sync(uint32_t(-1), frag_M[n][i], 8));
                frag_M[n][i] = fmaxf(frag_M[n][i], __shfl_xor_sync(uint32_t(-1), frag_M[n][i], 16));
            }
        }

        // Step 2: Compute expdiff and rescale previous L and O
        FragM expdiff_M;
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                expdiff_M[n][i] = exp2f((prev_M[n][i] - frag_M[n][i]) * qk_scale);
                if (is_residue && frag_M[n][i] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M[n][i] = 0.f;
                }
                frag_L[n][i] *= expdiff_M[n][i];
            }
        }

        // Rescale frag_O: each O value's H position is determined by s1,s0
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int s0 = 0; s0 < 2; ++s0) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M[n][s1 * 2 + s0];
                        }
                    }
                }
            }
        }

        // Step 3: Compute exp(S - M) and accumulate L
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int s1 = 0; s1 < 2; ++s1) {
                PRAGMA_UNROLL
                for (int s0 = 0; s0 < 2; ++s0) {
                    float tmp_L{};
                    PRAGMA_UNROLL
                    for (int m = 0; m < K_M; ++m) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            float& val = frag_S[m][n][s1 * 4 + q * 2 + s0];
                            float  p   = exp2f(val * qk_scale - frag_M[n][s1 * 2 + s0] * qk_scale);
                            if (is_residue && frag_M[n][s1 * 2 + s0] == -std::numeric_limits<float>::infinity()) {
                                p = 0.f;
                            }
                            tmp_L += p;
                            val = p;
                        }
                    }
                    if constexpr (!kDeferReduceL) {
                        tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 1);
                        tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 8);
                        tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 16);
                    }
                    frag_L[n][s1 * 2 + s0] += tmp_L;
                }
            }
        }
    }

    // ========================================================================
    // ConvertStoP: convert softmax output (FragS) to PV input (FragP)
    //
    // Goes through shared memory because QK output and PV input have
    // different thread-to-element mappings.
    //
    // 1. Write FragS to smem P[H, S] using ForeachS positions
    // 2. Sync
    // 3. Read FragP from smem using PV B operand positions
    //
    // For PV: O[D,H] = V^T[D,S] · P[S,H], mma_m8n8k4_row_row
    // B = P[S, H] is row-major. Each thread needs 4 consecutive H values
    // at the same S position.
    //
    // B operand lane mapping (from prefill V loading):
    //   K-dim (=S): lane_id % 4
    //   N-dim (=H) base: lane_id / 16 * 4 + (lane_id & 4) * 2
    //   4 consecutive N (=H) values: hi_base + 0..3
    //
    // P is stored as P[H, S] in smem, so consecutive H values have stride
    // CTA_S+4 — must use scalar loads.
    // ========================================================================
    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, SharedStorage& storage)
    {
        // Write S values to smem P[H, S]
        ForeachS(frag_S,
                 [&](int hi, int qi, int si, int ri, float p) { storage.P[SmemLayoutP::apply(hi, si)] = half(p); });

        if constexpr (!kUseSmemP) {
            __syncthreads();

            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    // B operand row-major: 4 consecutive H at same S
                    const int si      = k * 4 + lane_id % 4 + warp_id * WARP_S;
                    const int hi_base = n * OP_N + lane_id / 16 * 4 + (lane_id & 4) * 2;
                    PRAGMA_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        frag_P[k][n][i] = storage.P[SmemLayoutP::apply(hi_base + i, si)];
                    }
                }
            }
        }
    }

    // ========================================================================
    // Merge: cross-warp reduction of O, M, L
    // Following MMA_81616's pattern but with FragM = Array<float, 4>[K_N]
    // ========================================================================
    __device__ static void Merge(FragO& frag_O, FragM& frag_M, FragL& frag_L, float qk_scale, SharedStorage& storage)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_s = warp_id % kWarpCntS;

        FragM prev_M;
        copy(frag_M, prev_M);

        __syncthreads();

        // Store per-warp M to smem
        // SmemM = Array<float, 4>[K_N][kWarpCntS][4]
        // 4 thread-groups per warp: lane_id / 8 gives 4 groups (0..3)
        // But we need threads that share the same H positions to store together.
        // H positions are determined by (lane&4)*2 + (lane&2) + s1*4 + s0
        // Threads with same (lane&4, lane&2) share the same H positions.
        // (lane&4)/4 + (lane&2)/2 gives 4 groups (0..3)
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
            if ((lane_id & ~6) == 0) {  // lane_id bits 0,3,4 all zero → lane 0 of each H group
                Store((float*)&storage.M[n][warp_id_s][h_group], frag_M[n]);
            }
        }

        __syncthreads();

        // Compute global maximum across warps
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                PRAGMA_UNROLL
                for (int w = 0; w < kWarpCntS - 1; ++w) {
                    const int src_warp = (warp_id_s + w + 1) % kWarpCntS;
                    frag_M[n][i] = fmaxf(frag_M[n][i], storage.M[n][src_warp][h_group][i]);
                }
            }
        }

        // Rescale and store O, L to smem
        FragM expdiff_M;
        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                expdiff_M[n][i] = exp2f((prev_M[n][i] - frag_M[n][i]) * qk_scale);
                if (frag_M[n][i] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M[n][i] = 0.f;
                }
            }
            // Rescale O and store to smem
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int s0 = 0; s0 < 2; ++s0) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M[n][s1 * 2 + s0];
                        }
                    }
                }
                Store((float*)&storage.O[m][n][warp_id_s][lane_id], frag_O[m][n]);
            }
            // Rescale L and reduce deferred L
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                frag_L[n][i] *= expdiff_M[n][i];
                if constexpr (kDeferReduceL) {
                    frag_L[n][i] += __shfl_xor_sync(uint32_t(-1), frag_L[n][i], 1);
                    frag_L[n][i] += __shfl_xor_sync(uint32_t(-1), frag_L[n][i], 8);
                    frag_L[n][i] += __shfl_xor_sync(uint32_t(-1), frag_L[n][i], 16);
                }
            }
            const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
            if ((lane_id & ~6) == 0) {
                Store((float*)&storage.L[n][warp_id_s][h_group], frag_L[n]);
            }
        }

        __syncthreads();

        // Accumulate O and L across warps
        clear(frag_O);
        clear(frag_L);

        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int w = 0; w < kWarpCntS; ++w) {
                using namespace ops;
                PRAGMA_UNROLL
                for (int m = 0; m < V_M; ++m) {
                    Array<float, 8> tmp_O;
                    Load(tmp_O, storage.O[m][n][w][lane_id].data());
                    frag_O[m][n] = frag_O[m][n] + tmp_O;
                }
                const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
                frag_L[n] = frag_L[n] + storage.L[n][w][h_group];
            }
        }
    }

    // ========================================================================
    // StoreO: write output from registers to smem O1[CTA_H1, HeadDim],
    // then read back with a simple thread map and call func.
    //
    // FragO = Array<float, 8>[V_M][V_N], indexed [s1*4+q*2+s0]
    // For PV output (same layout as QK output but with D→M, H→N):
    //   D position: m*16 + (lane&8) + (lane&1) + lane/16*4 + q*2
    //   H position: n*16 + (lane&4)*2 + (lane&2) + s1*4 + s0
    // ========================================================================
    template<bool is_norm, class Func>
    __device__ static void StoreO(FragO& frag_O, const FragL& frag_L, SharedStorage& storage, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        FragL inv_L;
        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                inv_L[n][i] = fdividef(1.f, frag_L[n][i]);
            }
        }

        __syncthreads();

        // Write frag_O to smem O1[H, D]
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            const int di = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2;
                            const int hi = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            float val = frag_O[m][n][s1 * 4 + q * 2 + s0];
                            if constexpr (is_norm) {
                                val *= inv_L[n][s1 * 2 + s0];
                            }
                            if (warp_id == 0) {
                                storage.O1[hi][di] = val;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();

        // Read back from O1 with a simple thread map
        using Map = RakedThreadMap<kHeadDim, CTA_H1, 4, kWarpCount>;

        Array<float, 4> tmp_O[Map::kIterS][Map::kIterC];
        const int2      offset = Map::get_offset(warp_id, lane_id);
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                const int hi = offset.y + s * Map::kDeltaS;
                const int di = offset.x + c * Map::kDeltaC;
                if (hi < CTA_H) {
                    Load(tmp_O[s][c], &storage.O1[hi][di]);
                    ((Func &&) func)(hi, 0, di, tmp_O[s][c]);
                }
            }
        }
    }
};

// ============================================================================
// HeadDim=256 partial specialization for decode MMA_884_DEC.
// Two-segment 128 approach: splits 256-dim into two halves of 128.
// QK: two dot products into same FragS. PV: two outputs to FragO halves.
// ============================================================================
template<class T_,
         int CTA_H_,
         int CTA_Q_,
         int CTA_S_,
         int WARP_H_,
         int WARP_Q,
         int WARP_S,
         int Stages>
struct Impl<MMA_884_DEC, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q, WARP_S, 256, Stages> {
    using T   = T_;
    using Tkv = T_;

    static_assert(CTA_Q_ == 1, "MMA_884_DEC is decode-only (CTA_Q=1)");

    static constexpr int CTA_H    = CTA_H_;
    static constexpr int CTA_Q    = CTA_Q_;
    static constexpr int CTA_S    = CTA_S_;
    static constexpr int WARP_H   = WARP_H_;
    static constexpr int kHeadDim = 256;

    static constexpr int kWarpCntH = CTA_H / WARP_H;
    static constexpr int kWarpCntQ = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 4;

    static constexpr int kHalfDim = 128;

    // QK: per segment (128-dim)
    static constexpr int K_M = WARP_S / OP_M;               // 1
    static constexpr int K_N = (CTA_H + OP_N - 1) / OP_N;   // 1
    static constexpr int K_K = kHalfDim / OP_K;              // 32

    // PV
    static constexpr int V_M_half = kHalfDim / OP_M;         // 8
    static constexpr int V_M      = kHeadDim / OP_M;         // 16
    static constexpr int V_N      = K_N;                      // 1
    static constexpr int V_K      = WARP_S / OP_K;            // 4

    static constexpr int K_K_full = kHeadDim / OP_K;          // 64
    static constexpr int CTA_H1 = (CTA_H + OP_N - 1) / OP_N * OP_N;

    using FragK = Array<half, 4>[K_K][K_M];
    using FragQ = Array<half, 4>[K_N][K_K_full];  // full 256-dim
    using FragS = Array<float, 8>[K_M][K_N];

    using FragV = Array<half, 4>[V_M_half][V_K];  // 128-dim per segment
    using FragP = Array<half, 4>[V_K][V_N];
    using FragO = Array<float, 8>[V_M][V_N];      // full 256-dim

    using FragM = Array<float, 4>[K_N];
    using FragL = FragM;

    using SmemM = Array<float, 4>[K_N][kWarpCntS][4];
    using SmemO = Array<float, 8>[V_M][V_N][kWarpCntS][WARP_SIZE];

    struct SwizzleV {
        __device__ static int apply(int offset)
        {
            offset = ((offset & 8) << 2) ^ offset;
            offset = ((offset & ~20) | (((offset & 16) >> 2) | ((offset & 4) << 2)));
            offset = ((offset & (0x3 << 6)) >> 3) ^ offset;
            return offset;
        }
        __device__ int operator()(int offset) { return apply(offset); }
    };

    using SmemLayoutQ = SmemLayoutV2<CTA_H1, kHeadDim + 4, 1, 1, Identity>;
    using SmemLayoutK = SmemLayoutV2<CTA_S, kHeadDim + 4, 1, 1, Identity>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, kHeadDim, CTA_S, 64, SwizzleV>;
    using SmemLayoutP = SmemLayoutV2<CTA_H1, CTA_S + 4, 1, 1, Identity>;
    using SmemLayoutKVp = void;

    union SharedStorage {
        __align__(16) T Q[SmemLayoutQ::kSize];
        struct {
            __align__(16) T K[SmemLayoutK::kSize];
            __align__(16) T V[SmemLayoutV::kSize];
            __align__(16) T P[SmemLayoutP::kSize];
        };
        struct {
            __align__(16) SmemM M;
            __align__(16) SmemM L;
            __align__(16) SmemO O;
        };
        __align__(16) float O1[CTA_H1][kHeadDim];
    };

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    using ThreadMapQ   = RakedThreadMap<kHeadDim, CTA_H1, 4, kWarpCount, 32>;
    using ThreadMapKV  = RakedThreadMap<kHeadDim, CTA_S, 4, kWarpCount, 32>;
    using ThreadMapKVp = void;

    static constexpr bool kDeferReduceL = true;

    __device__ static void Sync() { __syncthreads(); }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        gmem_K.SetSmem(storage.K);
        gmem_V.SetSmem(storage.V);
    }

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            const int si = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2
                                           + warp_id * WARP_S;
                            const int hi = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            ((Func &&) func)(hi, 0, si, 0, S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }
    }

    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int s1 = 0; s1 < 2; ++s1) {
                PRAGMA_UNROLL
                for (int s0 = 0; s0 < 2; ++s0) {
                    const int hi = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                    const int ri = (lane_id >> 3 & 1) + (lane_id & 1) + (lane_id >> 4) * 4;
                    ((Func &&) func)(hi, 0, ri, frag_M[n][s1 * 2 + s0], frag_L[n][s1 * 2 + s0]);
                }
            }
        }
    }

    // TransformQ: load full 256-dim Q, split into lo/hi halves
    __device__ static void TransformQ(T* smem_Q, FragQ& frag_Q)
    {
        if constexpr (!kUseSmemQ) {
            const int lane_id = threadIdx.x % WARP_SIZE;
            SmemAccessor<T, SmemLayoutQ> sQ{smem_Q};

            // Zero-pad unused rows of Q smem (same fix as generic template).
            // Q has CTA_H valid rows, MMA reads CTA_H1 rows. Rows CTA_H..CTA_H1-1 must be zero.
            if constexpr (CTA_H < CTA_H1) {
                const int tid = threadIdx.x;
                constexpr int stride = CTA_H1 - CTA_H;
                constexpr int cols = kHeadDim;
                for (int idx = tid; idx < stride * cols; idx += kWarpCount * WARP_SIZE) {
                    const int row = CTA_H + idx / cols;
                    const int col = idx % cols;
                    sQ(row, col) = T(0);
                }
                __syncthreads();
            }

            // B operand col-major: col = lane%4 [+4 if lane>=16], row = i
            // The (lane&4)*2 term differentiates Comp 1/3 (H offset 0) vs Comp 2/4 (H offset 8)
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K_full; ++k) {
                    const int hi = n * OP_N + lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
                    const int di = k * 4;
                    Lds(frag_Q[n][k], &sQ(hi, di));
                }
            }
        }
    }

    struct StateQK {
        SmemAccessor<T, SmemLayoutK> smem_K;

        Array<half, 4> frag_Q_lo[K_N][K_K];
        Array<half, 4> frag_Q_hi[K_N][K_K];
        FragK frag_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_): smem_K{storage.K}
        {
            static_assert(!kUseSmemQ);
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; ++k) {
                    frag_Q_lo[n][k] = frag_Q_[n][k];
                    frag_Q_hi[n][k] = frag_Q_[n][k + K_K];
                }
            }
        }

        __device__ void LoadLo(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                const int s = m * 16 + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * WARP_S;
                const int c = k * 4;
                Lds(frag_K[k][m], &smem_K(s, c));
            }
        }

        __device__ void LoadHi(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                const int s = m * 16 + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * WARP_S;
                const int c = k * 4 + kHalfDim;
                Lds(frag_K[k][m], &smem_K(s, c));
            }
        }

        __device__ void Load(int k, int pipe_iter) { LoadLo(k, pipe_iter); }
        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputeQK(StateQK& state_QK, FragS& frag_S, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        // Segment 1: lo half (D=0..127)
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.LoadLo(k + 1, offset);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    mma_m8n8k4_row_col(frag_S[m][n], state_QK.frag_K[k][m], state_QK.frag_Q_lo[n][k], frag_S[m][n]);
                }
            }
        }

        state_QK.LoadHi(0, offset);

        // Segment 2: hi half (D=128..255)
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.LoadHi(k + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    mma_m8n8k4_row_col(frag_S[m][n], state_QK.frag_K[k][m], state_QK.frag_Q_hi[n][k], frag_S[m][n]);
                }
            }
        }
    }

    struct StatePV {
        T* smem_V;

        static_assert(V_M_half % 2 == 0);

        FragP frag_P;
        FragV frag_V;

        __device__ StatePV(SharedStorage& storage, bool offset): smem_V{storage.V}
        {
            assert(offset);
        }

        __device__ void LoadLo(int m, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            const int d = m * 16 + (lane_id & 8) + (lane_id >> 4) * 4 + (lane_id & 3);
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                const int s_base = k * 4 + warp_id * WARP_S;
                PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    frag_V[m][k][i] = smem_V[SmemLayoutV::apply(s_base + i, d)];
                }
            }
        }

        __device__ void LoadHi(int m, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            const int d = m * 16 + (lane_id & 8) + (lane_id >> 4) * 4 + (lane_id & 3) + kHalfDim;
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                const int s_base = k * 4 + warp_id * WARP_S;
                PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    frag_V[m][k][i] = smem_V[SmemLayoutV::apply(s_base + i, d)];
                }
            }
        }

        __device__ void Load(int m, int pipe_iter) { LoadLo(m, pipe_iter); }
        __device__ void Transform(int m) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV& state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        // Segment 1: V lo half → frag_O[0..V_M_half-1]
        PRAGMA_UNROLL
        for (int m = 0; m < V_M_half; ++m) {
            if (m < V_M_half - 1) {
                state_PV.LoadLo(m + 1, offset);
            }
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m8n8k4_row_row(frag_O[m][n], state_PV.frag_V[m][k], state_PV.frag_P[k][n], frag_O[m][n]);
                }
            }
        }

        state_PV.LoadHi(0, offset);

        // Segment 2: V hi half → frag_O[V_M_half..V_M-1]
        PRAGMA_UNROLL
        for (int m = 0; m < V_M_half; ++m) {
            if (m < V_M_half - 1) {
                state_PV.LoadHi(m + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m8n8k4_row_row(
                        frag_O[m + V_M_half][n], state_PV.frag_V[m][k], state_PV.frag_P[k][n], frag_O[m + V_M_half][n]);
                }
            }
        }
    }

    // Softmax: identical to generic decode MMA_884_DEC (same FragS/FragM layout)
    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        copy(frag_M, prev_M);

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int s0 = 0; s0 < 2; ++s0) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            frag_M[n][s1 * 2 + s0] =
                                fmaxf(frag_M[n][s1 * 2 + s0], frag_S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                frag_M[n][i] = fmaxf(frag_M[n][i], __shfl_xor_sync(uint32_t(-1), frag_M[n][i], 1));
                frag_M[n][i] = fmaxf(frag_M[n][i], __shfl_xor_sync(uint32_t(-1), frag_M[n][i], 8));
                frag_M[n][i] = fmaxf(frag_M[n][i], __shfl_xor_sync(uint32_t(-1), frag_M[n][i], 16));
            }
        }

        FragM expdiff_M;
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                expdiff_M[n][i] = exp2f((prev_M[n][i] - frag_M[n][i]) * qk_scale);
                if (is_residue && frag_M[n][i] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M[n][i] = 0.f;
                }
                frag_L[n][i] *= expdiff_M[n][i];
            }
        }

        // Rescale frag_O (full V_M=16 tiles)
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int s0 = 0; s0 < 2; ++s0) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M[n][s1 * 2 + s0];
                        }
                    }
                }
            }
        }

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int s1 = 0; s1 < 2; ++s1) {
                PRAGMA_UNROLL
                for (int s0 = 0; s0 < 2; ++s0) {
                    float tmp_L{};
                    PRAGMA_UNROLL
                    for (int m = 0; m < K_M; ++m) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            float& val = frag_S[m][n][s1 * 4 + q * 2 + s0];
                            float  p   = exp2f(val * qk_scale - frag_M[n][s1 * 2 + s0] * qk_scale);
                            if (is_residue && frag_M[n][s1 * 2 + s0] == -std::numeric_limits<float>::infinity()) {
                                p = 0.f;
                            }
                            tmp_L += p;
                            val = p;
                        }
                    }
                    if constexpr (!kDeferReduceL) {
                        tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 1);
                        tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 8);
                        tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 16);
                    }
                    frag_L[n][s1 * 2 + s0] += tmp_L;
                }
            }
        }
    }

    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, SharedStorage& storage)
    {
        ForeachS(frag_S,
                 [&](int hi, int qi, int si, int ri, float p) { storage.P[SmemLayoutP::apply(hi, si)] = half(p); });

        if constexpr (!kUseSmemP) {
            __syncthreads();
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    // B operand row-major: 4 consecutive H at same S
                    const int si      = k * 4 + lane_id % 4 + warp_id * WARP_S;
                    const int hi_base = n * OP_N + lane_id / 16 * 4 + (lane_id & 4) * 2;
                    PRAGMA_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        frag_P[k][n][i] = storage.P[SmemLayoutP::apply(hi_base + i, si)];
                    }
                }
            }
        }
    }

    __device__ static void Merge(FragO& frag_O, FragM& frag_M, FragL& frag_L, float qk_scale, SharedStorage& storage)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id_s = warp_id % kWarpCntS;

        FragM prev_M;
        copy(frag_M, prev_M);

        __syncthreads();

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
            if ((lane_id & ~6) == 0) {
                Store((float*)&storage.M[n][warp_id_s][h_group], frag_M[n]);
            }
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                PRAGMA_UNROLL
                for (int w = 0; w < kWarpCntS - 1; ++w) {
                    const int src_warp = (warp_id_s + w + 1) % kWarpCntS;
                    frag_M[n][i] = fmaxf(frag_M[n][i], storage.M[n][src_warp][h_group][i]);
                }
            }
        }

        FragM expdiff_M;
        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                expdiff_M[n][i] = exp2f((prev_M[n][i] - frag_M[n][i]) * qk_scale);
                if (frag_M[n][i] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M[n][i] = 0.f;
                }
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int s0 = 0; s0 < 2; ++s0) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M[n][s1 * 2 + s0];
                        }
                    }
                }
                Store((float*)&storage.O[m][n][warp_id_s][lane_id], frag_O[m][n]);
            }
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                frag_L[n][i] *= expdiff_M[n][i];
                if constexpr (kDeferReduceL) {
                    frag_L[n][i] += __shfl_xor_sync(uint32_t(-1), frag_L[n][i], 1);
                    frag_L[n][i] += __shfl_xor_sync(uint32_t(-1), frag_L[n][i], 8);
                    frag_L[n][i] += __shfl_xor_sync(uint32_t(-1), frag_L[n][i], 16);
                }
            }
            const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
            if ((lane_id & ~6) == 0) {
                Store((float*)&storage.L[n][warp_id_s][h_group], frag_L[n]);
            }
        }

        __syncthreads();

        clear(frag_O);
        clear(frag_L);

        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int w = 0; w < kWarpCntS; ++w) {
                using namespace ops;
                PRAGMA_UNROLL
                for (int m = 0; m < V_M; ++m) {
                    Array<float, 8> tmp_O;
                    Load(tmp_O, storage.O[m][n][w][lane_id].data());
                    frag_O[m][n] = frag_O[m][n] + tmp_O;
                }
                const int h_group = ((lane_id & 4) >> 2) * 2 + ((lane_id & 2) >> 1);
                frag_L[n] = frag_L[n] + storage.L[n][w][h_group];
            }
        }
    }

    template<bool is_norm, class Func>
    __device__ static void StoreO(FragO& frag_O, const FragL& frag_L, SharedStorage& storage, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        FragL inv_L;
        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                inv_L[n][i] = fdividef(1.f, frag_L[n][i]);
            }
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            const int di = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2;
                            const int hi = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            float val = frag_O[m][n][s1 * 4 + q * 2 + s0];
                            if constexpr (is_norm) {
                                val *= inv_L[n][s1 * 2 + s0];
                            }
                            if (warp_id == 0) {
                                storage.O1[hi][di] = val;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();

        using Map = RakedThreadMap<kHeadDim, CTA_H1, 4, kWarpCount, 32>;
        Array<float, 4> tmp_O[Map::kIterS][Map::kIterC];
        const int2      offset = Map::get_offset(warp_id, lane_id);
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                const int hi = offset.y + s * Map::kDeltaS;
                const int di = offset.x + c * Map::kDeltaC;
                if (hi < CTA_H) {
                    Load(tmp_O[s][c], &storage.O1[hi][di]);
                    ((Func &&) func)(hi, 0, di, tmp_O[s][c]);
                }
            }
        }
    }
};

}  // namespace turbomind::attention


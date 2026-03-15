// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/core/thread_map.h"

#include <cmath>

namespace turbomind::attention {

template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H_, int WARP_Q, int WARP_S, int HeadDim>
struct Impl<MMA_884, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q, WARP_S, HeadDim> {
    using T   = T_;
    using Tkv = T_;

    static constexpr int CTA_H    = CTA_H_;
    static constexpr int CTA_Q    = CTA_Q_;
    static constexpr int CTA_S    = CTA_S_;
    static constexpr int kHeadDim = HeadDim;

    static constexpr int kWarpCntQ  = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 4;

    static constexpr int K_M = WARP_Q / OP_M;   // 1
    static constexpr int K_N = WARP_S / OP_N;   // 4
    static constexpr int K_K = HeadDim / OP_K;  // 32

    static constexpr int V_M = WARP_Q / OP_M;   // 1
    static constexpr int V_N = HeadDim / OP_N;  // 8
    static constexpr int V_K = WARP_S / OP_K;   // 16

    //  +---+---+
    //  | 0 | 1 |
    //  +---+---+
    //  | 2 | 3 |
    //  +---+---+
    using FragQ = Array<half, 4>[K_K][K_M];   //    (q2,q2,x2,q4) (Dk,Qm) (d4)
                                              //      4  8  0  1    4 16    1
    using FragK = Array<half, 4>[K_K][K_N];   //    (s2,x2,s2,s4) (Dk,Sn) (d4)
                                              //      4  0  8  1    4 16    1
    using FragS = Array<float, 8>[K_M][K_N];  // (q2,q2,s2,s2,q2) (Qm,Sn) (s2,q2,s2)
                                              //   4  8  8  2  1   16 16    4  2  1
    using FragP = Array<half, 4>[V_K][V_M];   //    (q2,q2,x2,q4) (Sk,Qm) (s4)
                                              //      4  8  0  1    4 16    1
    using FragV = Array<half, 4>[V_K][V_N];   //    (d2,x2,d2,s4) (Sk,Dn) (d4)       [row major]
                                              //      4  0  8  1    4 16    1
    using FragO = Array<float, 8>[V_M][V_N];  // (q2,q2,d2,d2,q2) (Qm,Dn) (d2,q2,d2)
                                              //   4  8  8  2  1   16 16    4  2  1
    using FragM = Array<float, 2>[V_M];       // (q2,q2,_2,_2,q2) (Qm)    (q2))
    using FragL = FragM;

    // using Swizzle = Identity;

    struct SwizzleV {

        __device__ static int apply(int offset)
        {
            // Rearrange for LDS.128 (also avoid bank-conflict along C)
            // 6543210
            // dDDDDdd
            offset = ((offset & 8) << 2) ^ offset;                                     // x[5] ^= x[3]
            offset = ((offset & ~20) | (((offset & 16) >> 2) | ((offset & 4) << 2)));  // swap(x[4], x[2])

            // Shuffle C according S to avoid bank-conflict
            // ssssSSdDDddd
            offset = ((offset & (0x3 << 6)) >> 3) ^ offset;
            return offset;
        }

        __device__ int operator()(int offset)
        {
            return apply(offset);
        }
    };

    using SmemLayoutQ = SmemLayoutV2<CTA_Q, HeadDim + 4, 1, 1, Identity>;
    using SmemLayoutP = SmemLayoutV2<CTA_Q, CTA_S + 4, 1, 1, Identity>;
    using SmemLayoutK = SmemLayoutV2<CTA_S, HeadDim + 4, 1, 1, Identity>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, HeadDim, CTA_S, 64, SwizzleV>;

    using SmemLayoutKVp = void;

    struct SharedStorage {
        union {
            __align__(16) T Q[SmemLayoutQ::kSize];
            struct {
                __align__(16) T K[SmemLayoutK::kSize];
                __align__(16) T V[SmemLayoutV::kSize];
                __align__(16) T P[SmemLayoutP::kSize];
            };
        };
    };

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_Q, 4, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 4, kWarpCount>;

    using ThreadMapKVp = void;

    static constexpr bool kDeferReduceL = true;

    __device__ static void Sync()
    {
        __syncthreads();
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
                            const int qi = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2;
                            const int si = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            ((Func &&) func)(0, warp_id * WARP_Q + qi, si, /*ri*/ 0, S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }
    }

    __device__ static void TransformQ(const T* smem_Q, FragQ& frag_Q)
    {
        if constexpr (!kUseSmemQ) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    const int qi = m * OP_M + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * WARP_Q;
                    const int di = k * 4;
                    Lds(frag_Q[k][m], &smem_Q[SmemLayoutQ::apply(qi, di)]);
                }
            }
        }
    }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        gmem_K.SetSmem(storage.K);
        gmem_V.SetSmem(storage.V);
    }

    struct StateQK {
        SmemAccessor<T, SmemLayoutK> smem_K;

        FragQ frag_Q;
        FragK frag_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_): smem_K{storage.K}
        {
            static_assert(!kUseSmemQ, "not implemented");
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    frag_Q[k][m] = frag_Q_[k][m];
                }
            }
        }

        __device__ void Load(int k, int pipe_iter)
        {
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                const int s = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
                const int c = k * 4;
                Lds(frag_K[k][n], &smem_K(s, c));
            }
        }

        __device__ void Transform(int k) {}
    };

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
                    const int nn = n ^ 1;
                    mma_m8n8k4_row_col(frag_S[m][nn], state_QK.frag_Q[k][m], state_QK.frag_K[k][nn], frag_S[m][nn]);
                }
            }
        }
    }

    struct StatePV {
        T* smem_V;

        static_assert(V_N % 2 == 0);
        Array<int, V_N / 2> idxs_;

        FragP frag_P;
        FragV frag_V;

        __device__ StatePV(SharedStorage& storage, bool offset): smem_V{storage.V}
        {
            assert(offset);
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < 8; n += 2) {
                const int s  = 0 * 4 + lane_id % 4;
                const int c  = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2;
                idxs_[n / 2] = SmemLayoutV::apply(s, c);
            }
        }

        __device__ void Load(int k, int pipe_iter)
        {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; n += 2) {
                const int idx = idxs_[n / 2] + k * 4 * SmemLayoutV::C0;
                Lds((Array<half, 8>&)frag_V[k][n], &smem_V[idx]);
            }
        }

        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV& state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                state_PV.Load(k + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m8n8k4_row_row(frag_O[m][n], state_PV.frag_P[k][m], state_PV.frag_V[k][n], frag_O[m][n]);
                }
            }
        }
    }

    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            prev_M[m] = frag_M[m];
        }

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
                            frag_M[m][q] =
                                fmaxf(frag_M[m][q], frag_S[m][n][s1 * 4 + q * 2 + s0]);  // reduce over local quad
                        }
                    }
                }
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {  // reduce over thread group within warp (within warp tiles)
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 2));
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 4));
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                // exp(M - M'), isinf(frag_M) => isnan(expdiff_M)
                float expdiff_M = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M;  // Rescale previous output
                        }
                    }
                }
                frag_L[m][q] *= expdiff_M;
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            // unnormalized prob, optimized to FFMA
                            float p = exp2f(frag_S[m][n][s1 * 4 + q * 2 + s0] * qk_scale - frag_M[m][q] * qk_scale);
                            if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                                p = 0.f;
                            }
                            tmp_L += p;
                            frag_S[m][n][s1 * 4 + q * 2 + s0] = p;
                        }
                    }
                }
                if constexpr (!kDeferReduceL) {
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 2);
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 4);
                }
                frag_L[m][q] = frag_L[m][q] + tmp_L;  // update L
            }
        }
    }

    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, SharedStorage& storage)
    {
        ForeachS(frag_S,
                 [&](int, int qi, int si, int ri, float p) { storage.P[SmemLayoutP::apply(qi, si)] = half(p); });

        if constexpr (!kUseSmemP) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < V_M; ++m) {
                    const int qi = m * OP_M + lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4 + warp_id * WARP_Q;
                    const int si = k * OP_K;
                    Lds(frag_P[k][m], &storage.P[SmemLayoutP::apply(qi, si)]);
                }
            }
        }
    }

    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func)
    {
        /// FIXME: implement this
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {  // Q,16
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {  // Q,2
                const int qi = (lane_id & 1) * 1 + (lane_id & 16) / 4 + (lane_id & 8) + m * OP_M + q * 2;
                const int ri = (lane_id & 2) / 2 + (lane_id & 4) / 2;
                ((Func &&) func)(0, warp_id * WARP_Q + qi, ri, frag_M[m][q], frag_L[m][q]);
            }
        }
    };

    template<class Storage>
    __device__ static void Merge(FragO& frag_O, FragM& frag_M, FragL& frag_L, float qk_scale, Storage& storage)
    {
        static_assert(kWarpCntS == 1);

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                if constexpr (kDeferReduceL) {
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 2);
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 4);
                }
            }
        }
    }

    template<bool is_norm, class Func>
    __device__ static void StoreO(FragO& frag_O, FragL& frag_L, SharedStorage& storage, Func&& func)
    {
        FragL inv_L;
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                inv_L[m][q] = fdividef(1.f, frag_L[m][q] + 1e-8f);
            }
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int mm = lane_id / 16 * 4 + (lane_id & 8) + (lane_id & 1);
        const int nn = (lane_id & 4) * 2 + (lane_id & 2);

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int d1 = 0; d1 < 2; ++d1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        const int qi = m * OP_M + mm + q * 2 + warp_id * WARP_Q;
                        const int di = n * OP_N + nn + d1 * 4;
                        if constexpr (is_norm) {
                            PRAGMA_UNROLL
                            for (int d0 = 0; d0 < 2; ++d0) {
                                frag_O[m][n][d1 * 4 + q * 2 + d0] *= inv_L[m][q];
                            }
                        }
                        ((Func &&) func)(0, qi, di, (Array<float, 2>&)frag_O[m][n][d1 * 4 + q * 2]);
                    }
                }
            }
        }
    }
};

// ============================================================================
// HeadDim=256 partial specialization: two-segment 128 MMA_884
// Splits HeadDim=256 into two halves of 128, executes MMA_884 on each half,
// accumulates in a single kernel. QK: two dot products into same FragS.
// PV: two independent outputs to corresponding FragO positions.
// ============================================================================
template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H_, int WARP_Q, int WARP_S>
struct Impl<MMA_884, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q, WARP_S, 256> {
    using T   = T_;
    using Tkv = T_;

    static constexpr int CTA_H    = CTA_H_;
    static constexpr int CTA_Q    = CTA_Q_;
    static constexpr int CTA_S    = CTA_S_;
    static constexpr int kHeadDim = 256;

    static constexpr int kWarpCntQ  = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 4;

    // Internal half dimension for two-segment MMA
    static constexpr int kHalfDim = 128;

    static constexpr int K_M = WARP_Q / OP_M;        // 1
    static constexpr int K_N = WARP_S / OP_N;         // 4
    static constexpr int K_K = kHalfDim / OP_K;       // 32 (per segment)

    static constexpr int V_M      = WARP_Q / OP_M;    // 1
    static constexpr int V_N_half = kHalfDim / OP_N;   // 8 (per segment)
    static constexpr int V_N      = kHeadDim / OP_N;   // 16 (total)
    static constexpr int V_K      = WARP_S / OP_K;     // 16

    // FragQ holds full 256 dims: first K_K entries = lo half, next K_K = hi half
    // K_K_full = kHeadDim / OP_K = 64
    static constexpr int K_K_full = kHeadDim / OP_K;   // 64

    using FragQ = Array<half, 4>[K_K_full][K_M];
    using FragK = Array<half, 4>[K_K][K_N];            // 128-dim per segment
    using FragS = Array<float, 8>[K_M][K_N];
    using FragP = Array<half, 4>[V_K][V_M];
    using FragV = Array<half, 4>[V_K][V_N_half];       // 128-dim per segment
    using FragO = Array<float, 8>[V_M][V_N];           // full 256-dim output (V_N=16)
    using FragM = Array<float, 2>[V_M];
    using FragL = FragM;

    struct SwizzleV {
        __device__ static int apply(int offset)
        {
            offset = ((offset & 8) << 2) ^ offset;
            offset = ((offset & ~20) | (((offset & 16) >> 2) | ((offset & 4) << 2)));
            offset = ((offset & (0x3 << 6)) >> 3) ^ offset;
            return offset;
        }
        __device__ int operator()(int offset)
        {
            return apply(offset);
        }
    };

    // Q: full 256+4 width (loaded once from global)
    using SmemLayoutQ = SmemLayoutV2<CTA_Q, kHeadDim + 4, 1, 1, Identity>;
    using SmemLayoutP = SmemLayoutV2<CTA_Q, CTA_S + 4, 1, 1, Identity>;
    // K: full 256+4 width (mainloop loads full width, we read halves from smem)
    using SmemLayoutK = SmemLayoutV2<CTA_S, kHeadDim + 4, 1, 1, Identity>;
    // V: full 256 width with SwizzleV
    using SmemLayoutV = SmemLayoutV2<CTA_S, kHeadDim, CTA_S, 64, SwizzleV>;

    using SmemLayoutKVp = void;

    struct SharedStorage {
        union {
            __align__(16) T Q[SmemLayoutQ::kSize];
            struct {
                __align__(16) T K[SmemLayoutK::kSize];
                __align__(16) T V[SmemLayoutV::kSize];
                __align__(16) T P[SmemLayoutP::kSize];
            };
        };
    };

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    // Manually specify WarpThreadC=32 to avoid lowbit(256)/4=64 > WARP_SIZE
    using ThreadMapQ  = RakedThreadMap<kHeadDim, CTA_Q, 4, kWarpCount, 32>;
    using ThreadMapKV = RakedThreadMap<kHeadDim, CTA_S, 4, kWarpCount, 32>;

    using ThreadMapKVp = void;

    static constexpr bool kDeferReduceL = true;

    __device__ static void Sync()
    {
        __syncthreads();
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
                            const int qi = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2;
                            const int si = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            ((Func &&) func)(0, warp_id * WARP_Q + qi, si, /*ri*/ 0, S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }
    }

    // Load full 256-dim Q from smem into FragQ (first K_K = lo, next K_K = hi)
    __device__ static void TransformQ(const T* smem_Q, FragQ& frag_Q)
    {
        if constexpr (!kUseSmemQ) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < K_K_full; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    const int qi = m * OP_M + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * WARP_Q;
                    const int di = k * 4;
                    Lds(frag_Q[k][m], &smem_Q[SmemLayoutQ::apply(qi, di)]);
                }
            }
        }
    }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        gmem_K.SetSmem(storage.K);
        gmem_V.SetSmem(storage.V);
    }

    struct StateQK {
        SmemAccessor<T, SmemLayoutK> smem_K;

        // Two halves of Q fragments
        Array<half, 4> frag_Q_lo[K_K][K_M];
        Array<half, 4> frag_Q_hi[K_K][K_M];
        FragK frag_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_): smem_K{storage.K}
        {
            static_assert(!kUseSmemQ, "not implemented");
            // Split full FragQ into lo (k=0..K_K-1) and hi (k=K_K..K_K_full-1)
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    frag_Q_lo[k][m] = frag_Q_[k][m];
                    frag_Q_hi[k][m] = frag_Q_[k + K_K][m];
                }
            }
        }

        // Load K from the lo half (columns 0..127)
        __device__ void LoadLo(int k, int pipe_iter)
        {
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                const int s = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
                const int c = k * 4;  // columns 0..127
                Lds(frag_K[k][n], &smem_K(s, c));
            }
        }

        // Load K from the hi half (columns 128..255)
        __device__ void LoadHi(int k, int pipe_iter)
        {
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                const int s = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
                const int c = k * 4 + kHalfDim;  // columns 128..255
                Lds(frag_K[k][n], &smem_K(s, c));
            }
        }

        // Standard Load interface for mainloop compatibility (loads lo half at k=0)
        __device__ void Load(int k, int pipe_iter)
        {
            LoadLo(k, pipe_iter);
        }

        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputeQK(StateQK& state_QK, FragS& frag_S, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        // Segment 1: Q[:,0:128] · K[:,0:128] → accumulate into frag_S
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.LoadLo(k + 1, offset);
            }
            // At end of segment 1, load first K of segment 2
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int nn = n ^ 1;
                    mma_m8n8k4_row_col(frag_S[m][nn], state_QK.frag_Q_lo[k][m], state_QK.frag_K[k][nn], frag_S[m][nn]);
                }
            }
        }

        // Load first K slice of hi half
        state_QK.LoadHi(0, offset);

        // Segment 2: Q[:,128:256] · K[:,128:256] → accumulate into same frag_S
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
                    const int nn = n ^ 1;
                    mma_m8n8k4_row_col(frag_S[m][nn], state_QK.frag_Q_hi[k][m], state_QK.frag_K[k][nn], frag_S[m][nn]);
                }
            }
        }
    }

    struct StatePV {
        T* smem_V;

        static_assert(V_N_half % 2 == 0);
        // Pre-computed smem indices for lo and hi halves of V
        Array<int, V_N_half / 2> idxs_lo_;
        Array<int, V_N_half / 2> idxs_hi_;

        FragP frag_P;
        FragV frag_V;

        __device__ StatePV(SharedStorage& storage, bool offset): smem_V{storage.V}
        {
            assert(offset);
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < V_N_half; n += 2) {
                const int s     = 0 * 4 + lane_id % 4;
                const int c_lo  = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2;
                const int c_hi  = c_lo + kHalfDim;
                idxs_lo_[n / 2] = SmemLayoutV::apply(s, c_lo);
                idxs_hi_[n / 2] = SmemLayoutV::apply(s, c_hi);
            }
        }

        // Load V from the lo half (columns 0..127)
        __device__ void LoadLo(int k, int pipe_iter)
        {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N_half; n += 2) {
                const int idx = idxs_lo_[n / 2] + k * 4 * SmemLayoutV::C0;
                Lds((Array<half, 8>&)frag_V[k][n], &smem_V[idx]);
            }
        }

        // Load V from the hi half (columns 128..255)
        __device__ void LoadHi(int k, int pipe_iter)
        {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N_half; n += 2) {
                const int idx = idxs_hi_[n / 2] + k * 4 * SmemLayoutV::C0;
                Lds((Array<half, 8>&)frag_V[k][n], &smem_V[idx]);
            }
        }

        // Standard Load interface (loads lo half)
        __device__ void Load(int k, int pipe_iter)
        {
            LoadLo(k, pipe_iter);
        }

        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV& state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        // Segment 1: P · V[:,0:128] → frag_O[:,0:V_N_half]
        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                state_PV.LoadLo(k + 1, offset);
            }
            // At last k of segment 1, we'll load first hi-half V after the loop
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N_half; ++n) {
                    mma_m8n8k4_row_row(frag_O[m][n], state_PV.frag_P[k][m], state_PV.frag_V[k][n], frag_O[m][n]);
                }
            }
        }

        // Load first V slice of hi half
        state_PV.LoadHi(0, offset);

        // Segment 2: P · V[:,128:256] → frag_O[:,V_N_half:V_N]
        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                state_PV.LoadHi(k + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N_half; ++n) {
                    mma_m8n8k4_row_row(frag_O[m][n + V_N_half], state_PV.frag_P[k][m], state_PV.frag_V[k][n], frag_O[m][n + V_N_half]);
                }
            }
        }
    }

    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            prev_M[m] = frag_M[m];
        }

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
                            frag_M[m][q] = fmaxf(frag_M[m][q], frag_S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 2));
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 4));
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float expdiff_M = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                // Rescale previous output — full V_N=16 tiles
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M;
                        }
                    }
                }
                frag_L[m][q] *= expdiff_M;
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            float p = exp2f(frag_S[m][n][s1 * 4 + q * 2 + s0] * qk_scale - frag_M[m][q] * qk_scale);
                            if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                                p = 0.f;
                            }
                            tmp_L += p;
                            frag_S[m][n][s1 * 4 + q * 2 + s0] = p;
                        }
                    }
                }
                if constexpr (!kDeferReduceL) {
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 2);
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 4);
                }
                frag_L[m][q] = frag_L[m][q] + tmp_L;
            }
        }
    }

    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, SharedStorage& storage)
    {
        ForeachS(frag_S,
                 [&](int, int qi, int si, int ri, float p) { storage.P[SmemLayoutP::apply(qi, si)] = half(p); });

        if constexpr (!kUseSmemP) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < V_M; ++m) {
                    const int qi = m * OP_M + lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4 + warp_id * WARP_Q;
                    const int si = k * OP_K;
                    Lds(frag_P[k][m], &storage.P[SmemLayoutP::apply(qi, si)]);
                }
            }
        }
    }

    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                const int qi = (lane_id & 1) * 1 + (lane_id & 16) / 4 + (lane_id & 8) + m * OP_M + q * 2;
                const int ri = (lane_id & 2) / 2 + (lane_id & 4) / 2;
                ((Func &&) func)(0, warp_id * WARP_Q + qi, ri, frag_M[m][q], frag_L[m][q]);
            }
        }
    };

    template<class Storage>
    __device__ static void Merge(FragO& frag_O, FragM& frag_M, FragL& frag_L, float qk_scale, Storage& storage)
    {
        static_assert(kWarpCntS == 1);

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                if constexpr (kDeferReduceL) {
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 2);
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 4);
                }
            }
        }
    }

    template<bool is_norm, class Func>
    __device__ static void StoreO(FragO& frag_O, FragL& frag_L, SharedStorage& storage, Func&& func)
    {
        FragL inv_L;
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                inv_L[m][q] = fdividef(1.f, frag_L[m][q] + 1e-8f);
            }
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int mm = lane_id / 16 * 4 + (lane_id & 8) + (lane_id & 1);
        const int nn = (lane_id & 4) * 2 + (lane_id & 2);

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int d1 = 0; d1 < 2; ++d1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        const int qi = m * OP_M + mm + q * 2 + warp_id * WARP_Q;
                        const int di = n * OP_N + nn + d1 * 4;
                        if constexpr (is_norm) {
                            PRAGMA_UNROLL
                            for (int d0 = 0; d0 < 2; ++d0) {
                                frag_O[m][n][d1 * 4 + q * 2 + d0] *= inv_L[m][q];
                            }
                        }
                        ((Func &&) func)(0, qi, di, (Array<float, 2>&)frag_O[m][n][d1 * 4 + q * 2]);
                    }
                }
            }
        }
    }
};

}  // namespace turbomind::attention

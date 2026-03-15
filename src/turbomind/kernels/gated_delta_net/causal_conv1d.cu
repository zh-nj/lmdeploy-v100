// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/kernels/gated_delta_net/causal_conv1d.h"

namespace turbomind {

// ---------------------------------------------------------------------------
// Causal Conv1d Prefill Kernel  (depthwise, SiLU activation)
//
// Grid:  (cdiv(conv_dim, block_dim),)
// Block: (block_dim,)
//
// Each thread handles one channel and walks the full token sequence
// sequentially, maintaining a sliding window of kernel_size values.
// Adjacent threads process adjacent channels → coalesced global memory
// accesses on the input/output rows.
//
// Sequence boundaries are detected via seq_idx; the sliding window is
// reset to zero at each boundary so no information leaks across sequences.
//
// After the convolution pass, a second backward scan saves the last
// kernel_size input values of each sequence into conv_state for decode.
// ---------------------------------------------------------------------------

static constexpr int kMaxKernelSize = 8;

__device__ __forceinline__ float silu_f(float x)
{
    return x / (1.0f + expf(-x));
}

__global__ void causalConv1dPrefillSeqKernel(half*       output,       // [token_num, conv_dim]
                                             half*       conv_state,   // [batch, conv_dim, kernel_size]
                                             const half* input,        // [token_num, conv_dim]
                                             const half* weight,       // [conv_dim, kernel_size]
                                             const int*  seq_idx,      // [token_num]
                                             int         batch,
                                             int         token_num,
                                             int         conv_dim,
                                             int         kernel_size,
                                             int         input_stride,
                                             int         conv_state_batch_stride)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    // Load convolution weights for this channel into registers
    float w[kMaxKernelSize];
    for (int k = 0; k < kernel_size; ++k) {
        w[k] = __half2float(weight[ch * kernel_size + k]);
    }

    // Sliding window: window[0] = oldest, window[kernel_size-1] = newest
    float window[kMaxKernelSize];
    for (int k = 0; k < kernel_size; ++k) {
        window[k] = 0.0f;
    }

    int prev_seq = -1;

    // ---- Forward pass: compute convolution + SiLU ----
    for (int t = 0; t < token_num; ++t) {
        int cur_seq = seq_idx[t];

        // At sequence boundary, load conv_state instead of resetting to zero
        if (cur_seq != prev_seq) {
            const half* state_ptr = conv_state + cur_seq * conv_state_batch_stride + ch * kernel_size;
            for (int k = 0; k < kernel_size; ++k) {
                window[k] = __half2float(state_ptr[k]);
            }
            prev_seq = cur_seq;
        }

        // Shift window left, insert new value
        float x_val = __half2float(input[t * input_stride + ch]);
        for (int k = 0; k < kernel_size - 1; ++k) {
            window[k] = window[k + 1];
        }
        window[kernel_size - 1] = x_val;

        // Dot product
        float acc = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            acc += window[k] * w[k];
        }

        // SiLU activation and store
        output[t * conv_dim + ch] = __float2half(silu_f(acc));
    }

    // ---- Save conv_state for each sequence ----
    // Walk backwards to find the last token of each sequence, then save
    // the last kernel_size input values (merged with old conv_state if needed).
    int saved_count = 0;

    for (int t = token_num - 1; t >= 0 && saved_count < batch; --t) {
        int cur_seq = seq_idx[t];
        bool is_last = (t == token_num - 1) || (seq_idx[t + 1] != cur_seq);

        if (is_last) {
            // conv_state layout: [batch, conv_dim, kernel_size]
            half* state_ptr = conv_state + cur_seq * conv_state_batch_stride + ch * kernel_size;

            // Count how many tokens this sequence has up to position t
            int seq_start = t;
            while (seq_start > 0 && seq_idx[seq_start - 1] == cur_seq) {
                --seq_start;
            }
            const int seq_len = t - seq_start + 1;

            if (seq_len >= kernel_size) {
                // All state values come from input
                for (int k = 0; k < kernel_size; ++k) {
                    int src_t = t - kernel_size + 1 + k;
                    state_ptr[k] = __float2half(__half2float(input[src_t * input_stride + ch]));
                }
            }
            else {
                // Merge old conv_state with new input values
                float old_state[kMaxKernelSize];
                for (int k = 0; k < kernel_size; ++k) {
                    old_state[k] = __half2float(state_ptr[k]);
                }
                const int shift = kernel_size - seq_len;
                for (int k = 0; k < shift; ++k) {
                    state_ptr[k] = __float2half(old_state[k + seq_len]);
                }
                for (int k = 0; k < seq_len; ++k) {
                    state_ptr[shift + k] = input[(seq_start + k) * input_stride + ch];
                }
            }
            saved_count++;
        }
    }
}

__global__ void causalConv1dPrefillForwardKernel(half*       output,       // [token_num, conv_dim]
                                                 const half* input,        // [token_num, conv_dim]
                                                 const half* weight,       // [conv_dim, kernel_size]
                                                 const half* conv_state,   // [batch, conv_dim, kernel_size]
                                                 const int*  cu_seqlens,   // [batch + 1]
                                                 int         batch,
                                                 int         token_num,
                                                 int         conv_dim,
                                                 int         kernel_size,
                                                 int         input_stride,
                                                 int         conv_state_batch_stride)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int t  = blockIdx.y;
    if (ch >= conv_dim || t >= token_num) {
        return;
    }

    // Find which sequence this token belongs to via cu_seqlens
    // (binary search for the batch index)
    int batch_idx = 0;
    {
        int lo = 0, hi = batch;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cu_seqlens[mid + 1] <= t) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        batch_idx = lo;
    }
    const int seq_start = cu_seqlens[batch_idx];

    // conv_state layout: [batch, conv_dim, kernel_size]
    const half* state_ptr = conv_state + batch_idx * conv_state_batch_stride + ch * kernel_size;

    float w[kMaxKernelSize];
    for (int k = 0; k < kernel_size; ++k) {
        w[k] = __half2float(weight[ch * kernel_size + k]);
    }

    float acc = 0.f;
    for (int k = 0; k < kernel_size; ++k) {
        const int src_t = t - kernel_size + 1 + k;
        if (src_t >= seq_start) {
            // Token is within the current sequence in the input array
            acc += __half2float(input[src_t * input_stride + ch]) * w[k];
        }
        else {
            // Token is before the current sequence start — read from conv_state
            // conv_state stores the last kernel_size inputs: state[0]=oldest, state[ks-1]=newest
            // We need the value at position (kernel_size + src_t - seq_start) from the end of state
            const int state_idx = kernel_size + (src_t - seq_start);
            if (state_idx >= 0) {
                acc += __half2float(state_ptr[state_idx]) * w[k];
            }
        }
    }
    output[t * conv_dim + ch] = __float2half(silu_f(acc));
}

__global__ void causalConv1dSaveStateKernel(half*       conv_state,                // [batch, conv_dim, kernel_size]
                                            const half* input,                     // [token_num, conv_dim]
                                            const int*  cu_seqlens,                // [batch + 1]
                                            int         batch,
                                            int         conv_dim,
                                            int         kernel_size,
                                            int         input_stride,
                                            int         conv_state_batch_stride)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int b  = blockIdx.y;
    if (ch >= conv_dim || b >= batch) {
        return;
    }

    const int seq_start = cu_seqlens[b];
    const int seq_end   = cu_seqlens[b + 1];
    const int seq_len   = seq_end - seq_start;
    const int last_t    = seq_end - 1;
    half*     state_ptr = conv_state + b * conv_state_batch_stride + ch * kernel_size;

    if (seq_len >= kernel_size) {
        // All state values come from input
        for (int k = 0; k < kernel_size; ++k) {
            const int src_t = last_t - kernel_size + 1 + k;
            state_ptr[k] = input[src_t * input_stride + ch];
        }
    }
    else {
        // Need to merge old conv_state with new input values
        // Old state: [s0, s1, ..., s_{ks-1}]
        // After seq_len new values, state should be:
        //   [s_{seq_len}, s_{seq_len+1}, ..., s_{ks-1}, input[seq_start], ..., input[last_t]]
        // Read old state values first (they will be overwritten)
        float old_state[kMaxKernelSize];
        for (int k = 0; k < kernel_size; ++k) {
            old_state[k] = __half2float(state_ptr[k]);
        }
        // Write merged state
        const int shift = kernel_size - seq_len;
        for (int k = 0; k < shift; ++k) {
            state_ptr[k] = __float2half(old_state[k + seq_len]);
        }
        for (int k = 0; k < seq_len; ++k) {
            state_ptr[shift + k] = input[(seq_start + k) * input_stride + ch];
        }
    }
}

void invokeCausalConv1dPrefill(half*        output,
                               half*        conv_state,
                               const half*  input,
                               const half*  weight,
                               const int*   seq_idx,
                               int          batch,
                               int          token_num,
                               int          conv_dim,
                               int          kernel_size,
                               cudaStream_t stream,
                               int          input_stride,
                               int          conv_state_batch_stride,
                               const int*   cu_seqlens)
{
    constexpr int kBlockDim = 256;
    const int     grid_x    = (conv_dim + kBlockDim - 1) / kBlockDim;
    if (input_stride == 0) {
        input_stride = conv_dim;
    }
    if (conv_state_batch_stride == 0) {
        conv_state_batch_stride = conv_dim * kernel_size;
    }
    if (cu_seqlens != nullptr) {
        dim3 grid_forward(grid_x, token_num);
        causalConv1dPrefillForwardKernel<<<grid_forward, kBlockDim, 0, stream>>>(
            output, input, weight, conv_state, cu_seqlens, batch,
            token_num, conv_dim, kernel_size, input_stride, conv_state_batch_stride);

        dim3 grid_state(grid_x, batch);
        causalConv1dSaveStateKernel<<<grid_state, kBlockDim, 0, stream>>>(
            conv_state, input, cu_seqlens, batch, conv_dim, kernel_size, input_stride, conv_state_batch_stride);
    }
    else {
        causalConv1dPrefillSeqKernel<<<grid_x, kBlockDim, 0, stream>>>(
            output,
            conv_state,
            input,
            weight,
            seq_idx,
            batch,
            token_num,
            conv_dim,
            kernel_size,
            input_stride,
            conv_state_batch_stride);
    }
}

// ---------------------------------------------------------------------------
// Causal Conv1d Decode Kernel  (single-step update, SiLU activation)
//
// Grid:  (cdiv(conv_dim, block_dim), batch)
// Block: (block_dim,)
//
// Each thread handles one (batch_idx, channel) pair:
//   1. Shift conv_state left by one position
//   2. Insert the new input value at the rightmost position
//   3. Compute dot product of updated conv_state with weight
//   4. Apply SiLU activation
//   5. Write output and updated conv_state
// ---------------------------------------------------------------------------

__global__ void causalConv1dDecodeKernel(half*       output,       // [batch, conv_dim]
                                         half*       conv_state,   // [batch, conv_dim, kernel_size]
                                         const half* input,        // [batch, conv_dim]
                                         const half* weight,       // [conv_dim, kernel_size]
                                         int         batch,
                                         int         conv_dim,
                                         int         kernel_size,
                                         int         input_stride,
                                         int         conv_state_batch_stride)
{
    const int ch        = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    if (ch >= conv_dim || batch_idx >= batch)
        return;

    // Load convolution weights for this channel
    float w[kMaxKernelSize];
    for (int k = 0; k < kernel_size; ++k) {
        w[k] = __half2float(weight[ch * kernel_size + k]);
    }

    // Pointer to this (batch, channel)'s conv_state slice
    half* state_ptr = conv_state + batch_idx * conv_state_batch_stride + ch * kernel_size;

    // Load new input value
    float x_new = __half2float(input[batch_idx * input_stride + ch]);

    // Shift state left by 1 and insert new value at the end
    float s[kMaxKernelSize];
    for (int k = 0; k < kernel_size - 1; ++k) {
        s[k] = __half2float(state_ptr[k + 1]);
    }
    s[kernel_size - 1] = x_new;

    // Write updated state back
    for (int k = 0; k < kernel_size; ++k) {
        state_ptr[k] = __float2half(s[k]);
    }

    // Dot product
    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        acc += s[k] * w[k];
    }

    // SiLU activation and store output
    output[batch_idx * conv_dim + ch] = __float2half(silu_f(acc));
}

void invokeCausalConv1dDecode(half*        output,
                              half*        conv_state,
                              const half*  input,
                              const half*  weight,
                              int          batch,
                              int          conv_dim,
                              int          kernel_size,
                              cudaStream_t stream,
                              int          input_stride,
                              int          conv_state_batch_stride)
{
    constexpr int kBlockDim = 128;
    const int     grid_x    = (conv_dim + kBlockDim - 1) / kBlockDim;
    dim3          grid(grid_x, batch);
    if (input_stride == 0) {
        input_stride = conv_dim;
    }
    if (conv_state_batch_stride == 0) {
        conv_state_batch_stride = conv_dim * kernel_size;
    }

    causalConv1dDecodeKernel<<<grid, kBlockDim, 0, stream>>>(
        output, conv_state, input, weight, batch, conv_dim, kernel_size, input_stride, conv_state_batch_stride);
}

}  // namespace turbomind

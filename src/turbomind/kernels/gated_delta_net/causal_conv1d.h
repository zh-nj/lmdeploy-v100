// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

// Causal Conv1d Prefill
//
// Performs depthwise causal convolution with SiLU activation on a packed token sequence.
// Each channel is convolved independently (groups = conv_dim).
// Sequence boundaries are tracked via seq_idx to prevent cross-sequence leakage.
//
// After convolution, the last kernel_size inputs of each sequence are saved
// into conv_state for subsequent decode steps (the full sliding window).
//
// Parameters:
//   output       [token_num, conv_dim]  fp16, convolution output with SiLU activation
//   conv_state   [batch, conv_dim, kernel_size]  fp16, saved sliding-window state per sequence
//   input        [token_num, conv_dim]  fp16, input activations
//   weight       [conv_dim, kernel_size]  fp16, depthwise conv weights
//   seq_idx      [token_num]  int32, sequence index for each token (0-based)
//   batch        number of sequences in the batch
//   token_num    total number of tokens across all sequences
//   conv_dim     number of channels (= groups for depthwise conv)
//   kernel_size  convolution kernel size (typically 4)
//   input_stride stride between adjacent tokens in input (in half elements).
//                0 means compact layout: conv_dim.
//   conv_state_batch_stride  stride between adjacent batches in conv_state (in half elements).
//                            0 means compact layout: conv_dim * kernel_size.
//   stream       CUDA stream
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
                               int          input_stride = 0,
                               int          conv_state_batch_stride = 0,
                               const int*   cu_seqlens = nullptr);

// Causal Conv1d Decode (single-step update)
//
// Performs a single decode step of depthwise causal convolution with SiLU activation.
// For each (batch, channel) pair:
//   1. Shifts conv_state left by one position
//   2. Inserts the new input value at the rightmost position
//   3. Computes dot product of updated conv_state with weight
//   4. Applies SiLU activation
//
// Parameters:
//   output       [batch, conv_dim]  fp16, convolution output with SiLU activation
//   conv_state   [batch, conv_dim, kernel_size]  fp16, updated in-place
//   input        [batch, conv_dim]  fp16, new input for this decode step
//   weight       [conv_dim, kernel_size]  fp16, depthwise conv weights
//   batch        number of sequences in the batch
//   conv_dim     number of channels (= groups for depthwise conv)
//   kernel_size  convolution kernel size (typically 4)
//   input_stride stride between adjacent batches in input (in half elements).
//                0 means compact layout: conv_dim.
//   conv_state_batch_stride  stride between adjacent batches in conv_state (in half elements).
//                            0 means compact layout: conv_dim * kernel_size.
//   stream       CUDA stream
void invokeCausalConv1dDecode(half*        output,
                              half*        conv_state,
                              const half*  input,
                              const half*  weight,
                              int          batch,
                              int          conv_dim,
                              int          kernel_size,
                              cudaStream_t stream,
                              int          input_stride = 0,
                              int          conv_state_batch_stride = 0);

}  // namespace turbomind

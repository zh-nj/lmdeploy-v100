// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"

#include <cuda_runtime.h>

namespace turbomind {

struct RejectionResult {
    Buffer_<int> num_accepted;  // [batch] number of accepted drafts per request (0 ≤ N ≤ K)
    Buffer_<int> bonus_tokens;  // [batch] bonus token at first mismatch position
};

/// Greedy rejection sampling: compare draft tokens against target logits.
///
/// For each request in the batch:
///   target[i] = argmax(verification_logits[i])
///   Accept draft[i] if draft[i] == target[i], for consecutive i starting from 0
///   First mismatch at position p: num_accepted = p, bonus_token = target[p]
///   All K match: num_accepted = K, bonus_token = argmax(logits[K])
///
/// @param verification_logits  [batch, K+1, vocab_size] logits from verification forward
/// @param draft_tokens         [batch, K] draft token IDs
/// @param batch_size           number of requests
/// @param K                    number of draft tokens per request
/// @param vocab_size           vocabulary size
/// @param dtype                data type of logits (kFloat16, kBfloat16, or kFloat32)
/// @param stream               CUDA stream
RejectionResult GreedyReject(const void*  verification_logits,
                             const int*   draft_tokens,
                             int          batch_size,
                             int          K,
                             int          vocab_size,
                             DataType     dtype,
                             cudaStream_t stream);

}  // namespace turbomind

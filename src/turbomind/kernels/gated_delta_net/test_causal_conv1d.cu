// Quick correctness test for causal_conv1d prefill kernel.
// Compile: nvcc -o test_causal_conv1d test_causal_conv1d.cu causal_conv1d.cu
//          -I/path/to/lmdeploy --std=c++17 -arch=sm_70
// Run:     ./test_causal_conv1d

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

#include "src/turbomind/kernels/gated_delta_net/causal_conv1d.h"

static float silu_ref(float x)
{
    return x / (1.0f + expf(-x));
}

// Reference implementation: causal conv1d + SiLU, with seq_idx boundary handling
static void causal_conv1d_ref(std::vector<float>&       output,
                              std::vector<float>&       conv_state,
                              const std::vector<float>& input,
                              const std::vector<float>& weight,
                              const std::vector<int>&   seq_idx,
                              int                       batch,
                              int                       token_num,
                              int                       conv_dim,
                              int                       kernel_size)
{
    output.resize(token_num * conv_dim);
    conv_state.resize(batch * conv_dim * kernel_size, 0.0f);

    for (int ch = 0; ch < conv_dim; ++ch) {
        std::vector<float> window(kernel_size, 0.0f);
        int                prev_seq = -1;

        for (int t = 0; t < token_num; ++t) {
            int cur_seq = seq_idx[t];
            if (cur_seq != prev_seq) {
                for (int k = 0; k < kernel_size; ++k)
                    window[k] = 0.0f;
                prev_seq = cur_seq;
            }

            float x_val = input[t * conv_dim + ch];
            for (int k = 0; k < kernel_size - 1; ++k)
                window[k] = window[k + 1];
            window[kernel_size - 1] = x_val;

            float acc = 0.0f;
            for (int k = 0; k < kernel_size; ++k)
                acc += window[k] * weight[ch * kernel_size + k];

            output[t * conv_dim + ch] = silu_ref(acc);
        }
    }

    // Save conv_state: last kernel_size inputs per sequence per channel
    for (int ch = 0; ch < conv_dim; ++ch) {
        int saved = 0;
        for (int t = token_num - 1; t >= 0 && saved < batch; --t) {
            int  cur_seq = seq_idx[t];
            bool is_last = (t == token_num - 1) || (seq_idx[t + 1] != cur_seq);
            if (is_last) {
                for (int k = 0; k < kernel_size; ++k) {
                    int   src_t = t - kernel_size + 1 + k;
                    float val   = 0.0f;
                    if (src_t >= 0 && seq_idx[src_t] == cur_seq) {
                        val = input[src_t * conv_dim + ch];
                    }
                    conv_state[cur_seq * conv_dim * kernel_size + ch * kernel_size + k] = val;
                }
                saved++;
            }
        }
    }
}

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

int main()
{
    // Test parameters
    const int batch       = 3;
    const int token_num   = 20;
    const int conv_dim    = 8;
    const int kernel_size = 4;

    // Sequence layout: seq0 has 8 tokens, seq1 has 5 tokens, seq2 has 7 tokens
    std::vector<int> seq_idx(token_num);
    for (int i = 0; i < 8; ++i)
        seq_idx[i] = 0;
    for (int i = 8; i < 13; ++i)
        seq_idx[i] = 1;
    for (int i = 13; i < 20; ++i)
        seq_idx[i] = 2;

    // Random input and weights
    srand(42);
    std::vector<float> h_input(token_num * conv_dim);
    std::vector<float> h_weight(conv_dim * kernel_size);
    for (auto& v : h_input)
        v = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;
    for (auto& v : h_weight)
        v = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;

    // Reference
    std::vector<float> ref_output, ref_state;
    causal_conv1d_ref(ref_output, ref_state, h_input, h_weight, seq_idx, batch, token_num, conv_dim, kernel_size);

    // Convert to half
    std::vector<half> h_input_h(token_num * conv_dim);
    std::vector<half> h_weight_h(conv_dim * kernel_size);
    for (int i = 0; i < (int)h_input.size(); ++i)
        h_input_h[i] = __float2half(h_input[i]);
    for (int i = 0; i < (int)h_weight.size(); ++i)
        h_weight_h[i] = __float2half(h_weight[i]);

    // Allocate device memory
    half *d_input, *d_weight, *d_output, *d_conv_state;
    int*  d_seq_idx;
    CUDA_CHECK(cudaMalloc(&d_input, token_num * conv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, conv_dim * kernel_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, token_num * conv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_conv_state, batch * conv_dim * kernel_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_seq_idx, token_num * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input_h.data(), token_num * conv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_h.data(), conv_dim * kernel_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_idx, seq_idx.data(), token_num * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_conv_state, 0, batch * conv_dim * kernel_size * sizeof(half)));

    // Run kernel
    turbomind::invokeCausalConv1dPrefill(
        d_output, d_conv_state, d_input, d_weight, d_seq_idx, batch, token_num, conv_dim, kernel_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    std::vector<half> h_output_h(token_num * conv_dim);
    std::vector<half> h_state_h(batch * conv_dim * kernel_size);
    CUDA_CHECK(cudaMemcpy(h_output_h.data(), d_output, token_num * conv_dim * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(h_state_h.data(), d_conv_state, batch * conv_dim * kernel_size * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare output
    float max_err_out = 0.0f;
    for (int i = 0; i < token_num * conv_dim; ++i) {
        float cuda_val = __half2float(h_output_h[i]);
        float ref_val  = ref_output[i];
        float err      = fabsf(cuda_val - ref_val);
        if (err > max_err_out) {
            max_err_out = err;
        }
    }

    // Compare conv_state
    float max_err_state = 0.0f;
    for (int i = 0; i < batch * conv_dim * kernel_size; ++i) {
        float cuda_val = __half2float(h_state_h[i]);
        float ref_val  = ref_state[i];
        float err      = fabsf(cuda_val - ref_val);
        if (err > max_err_state) {
            max_err_state = err;
        }
    }

    printf("Output  max abs error: %.6f (threshold: 1e-3)\n", max_err_out);
    printf("State   max abs error: %.6f (threshold: 1e-3)\n", max_err_state);

    bool pass = (max_err_out < 1e-3f) && (max_err_state < 1e-3f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");

    // Cleanup prefill test
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_conv_state);
    cudaFree(d_seq_idx);

    if (!pass) {
        return 1;
    }

    // ================================================================
    // Test 2: Decode kernel
    //
    // Strategy: prefill a sequence of length L, then decode 1 step.
    // Compare decode output against full prefill of length L+1 at the
    // last position.
    // ================================================================
    printf("\n--- Decode kernel test ---\n");

    const int decode_batch       = 2;
    const int prefill_len        = 10;
    const int decode_conv_dim    = 8;
    const int decode_kernel_size = 4;

    // Build seq_idx for prefill: 2 sequences, each of length prefill_len/2
    const int seq0_len = 6;
    const int seq1_len = prefill_len - seq0_len;  // 4
    std::vector<int> decode_seq_idx(prefill_len);
    for (int i = 0; i < seq0_len; ++i)
        decode_seq_idx[i] = 0;
    for (int i = seq0_len; i < prefill_len; ++i)
        decode_seq_idx[i] = 1;

    // Random input for prefill + 1 extra token per sequence for decode
    std::vector<float> h_prefill_input(prefill_len * decode_conv_dim);
    std::vector<float> h_decode_input(decode_batch * decode_conv_dim);
    std::vector<float> h_decode_weight(decode_conv_dim * decode_kernel_size);
    for (auto& v : h_prefill_input)
        v = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;
    for (auto& v : h_decode_input)
        v = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;
    for (auto& v : h_decode_weight)
        v = (float(rand()) / RAND_MAX - 0.5f) * 2.0f;

    // --- Reference: full prefill of extended sequences ---
    // Build extended input: seq0 tokens + decode_input[0], seq1 tokens + decode_input[1]
    int ext_token_num = prefill_len + decode_batch;
    std::vector<float> h_ext_input(ext_token_num * decode_conv_dim);
    std::vector<int>   ext_seq_idx(ext_token_num);

    // Copy seq0 tokens
    for (int t = 0; t < seq0_len; ++t) {
        for (int c = 0; c < decode_conv_dim; ++c)
            h_ext_input[t * decode_conv_dim + c] = h_prefill_input[t * decode_conv_dim + c];
        ext_seq_idx[t] = 0;
    }
    // Append decode token for seq0
    for (int c = 0; c < decode_conv_dim; ++c)
        h_ext_input[seq0_len * decode_conv_dim + c] = h_decode_input[0 * decode_conv_dim + c];
    ext_seq_idx[seq0_len] = 0;

    // Copy seq1 tokens
    for (int t = 0; t < seq1_len; ++t) {
        int src_t = seq0_len + t;
        int dst_t = seq0_len + 1 + t;
        for (int c = 0; c < decode_conv_dim; ++c)
            h_ext_input[dst_t * decode_conv_dim + c] = h_prefill_input[src_t * decode_conv_dim + c];
        ext_seq_idx[dst_t] = 1;
    }
    // Append decode token for seq1
    int last_ext = seq0_len + 1 + seq1_len;
    for (int c = 0; c < decode_conv_dim; ++c)
        h_ext_input[last_ext * decode_conv_dim + c] = h_decode_input[1 * decode_conv_dim + c];
    ext_seq_idx[last_ext] = 1;

    std::vector<float> ref_ext_output, ref_ext_state;
    causal_conv1d_ref(ref_ext_output, ref_ext_state, h_ext_input, h_decode_weight,
                      ext_seq_idx, decode_batch, ext_token_num, decode_conv_dim, decode_kernel_size);

    // The reference decode outputs are at positions seq0_len (for seq0) and last_ext (for seq1)
    std::vector<float> ref_decode_output(decode_batch * decode_conv_dim);
    for (int c = 0; c < decode_conv_dim; ++c) {
        ref_decode_output[0 * decode_conv_dim + c] = ref_ext_output[seq0_len * decode_conv_dim + c];
        ref_decode_output[1 * decode_conv_dim + c] = ref_ext_output[last_ext * decode_conv_dim + c];
    }

    // --- CUDA: prefill then decode ---
    // Convert to half
    std::vector<half> h_prefill_h(prefill_len * decode_conv_dim);
    std::vector<half> h_dinput_h(decode_batch * decode_conv_dim);
    std::vector<half> h_dweight_h(decode_conv_dim * decode_kernel_size);
    for (int i = 0; i < (int)h_prefill_input.size(); ++i)
        h_prefill_h[i] = __float2half(h_prefill_input[i]);
    for (int i = 0; i < (int)h_decode_input.size(); ++i)
        h_dinput_h[i] = __float2half(h_decode_input[i]);
    for (int i = 0; i < (int)h_decode_weight.size(); ++i)
        h_dweight_h[i] = __float2half(h_decode_weight[i]);

    half *d_prefill_in, *d_dweight, *d_prefill_out, *d_dstate, *d_dinput, *d_doutput;
    int*  d_dseq_idx;
    CUDA_CHECK(cudaMalloc(&d_prefill_in, prefill_len * decode_conv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dweight, decode_conv_dim * decode_kernel_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_prefill_out, prefill_len * decode_conv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dstate, decode_batch * decode_conv_dim * decode_kernel_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dinput, decode_batch * decode_conv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_doutput, decode_batch * decode_conv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dseq_idx, prefill_len * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_prefill_in, h_prefill_h.data(), prefill_len * decode_conv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dweight, h_dweight_h.data(), decode_conv_dim * decode_kernel_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dseq_idx, decode_seq_idx.data(), prefill_len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dstate, 0, decode_batch * decode_conv_dim * decode_kernel_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_dinput, h_dinput_h.data(), decode_batch * decode_conv_dim * sizeof(half), cudaMemcpyHostToDevice));

    // Step 1: Prefill
    turbomind::invokeCausalConv1dPrefill(
        d_prefill_out, d_dstate, d_prefill_in, d_dweight, d_dseq_idx,
        decode_batch, prefill_len, decode_conv_dim, decode_kernel_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Decode
    turbomind::invokeCausalConv1dDecode(
        d_doutput, d_dstate, d_dinput, d_dweight,
        decode_batch, decode_conv_dim, decode_kernel_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back decode output
    std::vector<half> h_doutput_h(decode_batch * decode_conv_dim);
    CUDA_CHECK(cudaMemcpy(h_doutput_h.data(), d_doutput, decode_batch * decode_conv_dim * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare
    float max_err_decode = 0.0f;
    for (int i = 0; i < decode_batch * decode_conv_dim; ++i) {
        float cuda_val = __half2float(h_doutput_h[i]);
        float ref_val  = ref_decode_output[i];
        float err      = fabsf(cuda_val - ref_val);
        if (err > max_err_decode) {
            max_err_decode = err;
            if (err > 1e-3f) {
                int b = i / decode_conv_dim;
                int c = i % decode_conv_dim;
                printf("  Mismatch at batch=%d ch=%d: cuda=%.6f ref=%.6f err=%.6f\n",
                       b, c, cuda_val, ref_val, err);
            }
        }
    }

    printf("Decode  max abs error: %.6f (threshold: 1e-3)\n", max_err_decode);
    bool decode_pass = (max_err_decode < 1e-3f);
    printf("Decode result: %s\n", decode_pass ? "PASS" : "FAIL");

    // Cleanup decode test
    cudaFree(d_prefill_in);
    cudaFree(d_dweight);
    cudaFree(d_prefill_out);
    cudaFree(d_dstate);
    cudaFree(d_dinput);
    cudaFree(d_doutput);
    cudaFree(d_dseq_idx);

    return (pass && decode_pass) ? 0 : 1;
}

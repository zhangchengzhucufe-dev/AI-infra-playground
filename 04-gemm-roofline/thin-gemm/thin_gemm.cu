// Thin GEMM: measuring where the tensor core stops helping, and the
// arithmetic-intensity reasoning behind vLLM's skinny-GEMM dispatch.
//
// Shapes come from vLLM mainline's Kimi K3 decode GEMM dispatch table
// (vllm/models/kimi_k3/nvidia/low_latency_gemm.py, targeting SM103 BF16):
// each row is a real projection layer's (N, K); M is the token count this
// forward step processes -- decode: the batch (order 1-16), chunked
// prefill: the chunk size (thousands to tens of thousands); the weight
// shape N/K does not depend on context length. Upstream's move, which this
// program measures against: for M<=16 vLLM abandons the cuBLAS/tensor core
// path for a skinny kernel doing plain CUDA core FMA (skipping the TMA and
// tensor core setup), 8%-100% faster in their microbenchmarks.
//
// The physics: arithmetic intensity AI = 2MNK / (2MK + 2NK + 2MN)
// [flop/byte] falls off as M shrinks. Against the machine balance point
// (peak TFLOPS / peak bandwidth), rows below it are memory-bound with a
// theoretical ceiling of AI x peak bandwidth -- far under peak TFLOPS no
// matter how good the kernel is. The sweep makes the %-of-tensor-core-peak
// collapse at small M visible and separates the two denominators: % of TC
// peak says how much of the tensor core is wasted; % of the bandwidth
// roofline says how close the call already is to the memory bound.
//
// What it runs: cuBLAS bf16 (f32 accumulate, bf16 out) over the (N, K)
// shape table x M from 1 to 65536, printing us / TFLOPS / GB/s / AI per
// row.
//
// Usage: ./bin/thin-gemm/thin_gemm [peak TFLOPS peak GB/s]
//   With both peaks given, two extra %-of-peak columns are printed
//   (%TCpeak and %BW); without them only the base columns appear.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../common.h"

struct Shape {
    int n, k;
    const char* name;
};

// Representative shapes picked from KIMI_K3_PROJECTIONS in
// low_latency_gemm.py (local shapes after TP sharding); each row is
// derivable from the K3 config (hidden=7168, 96 heads, q_lora=1536,
// kv_lora=512, rope=64, dense intermediate=33792):
//   f_b_proj   second half of the gated rank-128 bottleneck, 12288/TP8;
//              K=128 is an extreme small K
//   q_b_proj   96x(128+64)=18432/TP8
//   o_proj     96x128=12288/TP8 into hidden
//   fused_qkv_a_proj  1536+512+64=2112, not sharded
//   in_proj_qkvgfab   KDA input projection (q+k+v+g bulk)/TP8
//   dense_down_proj   33792/TP4 (only layer 0 is dense)
//   dense_gate_up_proj 2x33792/TP4
static const Shape SHAPES[] = {
    {1536, 128, "f_b_proj"},
    {2304, 1536, "q_b_proj"},
    {7168, 1536, "o_proj"},
    {2112, 7168, "fused_qkv_a_proj"},
    {6288, 7168, "in_proj_qkvgfab"},
    {7168, 8448, "dense_down_proj"},
    {16896, 7168, "dense_gate_up_proj"},
};
// The M axis covers the full decode-to-prefill range: 1-16 is the decode
// batch (where the skinny kernel takes over), 64-256 is the transition,
// 1024-65536 matches chunked prefill's per-step token count (K3 is a 1M
// context model, but any prompt enters the GEMM in chunks; per-step
// M = chunk size; the large-M end shows the %-of-peak saturation plateau).
static const int MS[] = {1, 8, 16, 64, 256, 1024, 4096, 16384, 65536};

int main(int argc, char** argv) {
    double peak_tflops = argc > 2 ? atof(argv[1]) : 0;
    double peak_gbps = argc > 2 ? atof(argv[2]) : 0;

    int maxM = 65536, maxN = 0, maxK = 0;
    for (auto& s : SHAPES) {
        if (s.n > maxN) maxN = s.n;
        if (s.k > maxK) maxK = s.k;
    }
    size_t nA = (size_t)maxM * maxK, nW = (size_t)maxN * maxK,
           nD = (size_t)maxM * maxN;
    // Data content does not affect timing; fill with a cheap xorshift of
    // nonzero values
    std::vector<__nv_bfloat16> hA(nA);
    uint32_t x = 0x12345678;
    for (auto& v : hA) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v = __float2bfloat16((float)(int)(x % 7) - 3.f);
    }
    __nv_bfloat16 *dA, *dW, *dD;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dW, nW * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 2));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    // Weight buffer is large; tile-fill it with A's contents
    for (size_t off = 0; off < nW; off += nA)
        CUDA_CHECK(cudaMemcpy(dW + off, dA,
                              (nW - off < nA ? nW - off : nA) * 2,
                              cudaMemcpyDeviceToDevice));

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;

    printf("%-20s %5s %6s %6s %9s %9s %9s %7s", "layer", "M", "N", "K",
           "us", "TFLOPS", "GB/s", "AI");
    if (peak_tflops > 0) printf(" %8s %8s", "%TCpeak", "%BW");
    printf("\n");
    for (auto& s : SHAPES) {
        for (int M : MS) {
            // D[M,N] row-major: C_col[N,M] = W_col[K,N]^T x A_col[K,M]
            auto launch = [&] {
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, M, s.k, &alpha,
                             dW, CUDA_R_16BF, s.k, dA, CUDA_R_16BF, s.k,
                             &beta, dD, CUDA_R_16BF, s.n, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT);
            };
            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            int iters = M <= 256 ? 200 : (M <= 4096 ? 50 : 20);
            float ms = time_avg_ms(launch, iters);
            double flop = 2.0 * M * s.n * s.k;
            double bytes = 2.0 * ((double)M * s.k + (double)s.n * s.k +
                                  (double)M * s.n);
            double tflops = flop / (ms * 1e9);
            double gbps = bytes / (ms * 1e6);
            double ai = flop / bytes;
            printf("%-20s %5d %6d %6d %9.1f %9.1f %9.1f %7.1f", s.name, M,
                   s.n, s.k, ms * 1e3, tflops, gbps, ai);
            if (peak_tflops > 0)
                printf(" %7.1f%% %7.1f%%", 100.0 * tflops / peak_tflops,
                       100.0 * gbps / peak_gbps);
            printf("\n");
        }
        printf("\n");
    }
    cublasDestroy(h);
    return 0;
}

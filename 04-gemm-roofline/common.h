// Shared helpers for every program in this directory: CUDA error checks,
// event timing, and the averaged-launch benchmark helper all the
// TFLOPS/bandwidth numbers come from.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_KERNEL()                        \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Average of iters launches after warmup, in ms. Use for bandwidth and TFLOPS alike.
template <typename F>
static inline float time_avg_ms(F&& launch, int iters, int warmup = 20) {
    for (int i = 0; i < warmup; i++) launch();
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; i++) launch();
    float ms = t.stop_ms();
    CUDA_CHECK(cudaGetLastError());
    return ms / iters;
}

// Effective bandwidth: bytes is traffic that must cross HBM (reads + writes).
static inline double effective_gbps(double bytes, float ms) {
    return bytes / (ms * 1e6);
}

// Tolerance for half-precision checks: tensor core accumulation order
// differs from the CPU loop, so GEMM with fp16/bf16 inputs and fp32
// accumulate usually is not bit-exact against a CPU reference; loosen rtol
// with the magnitude of K (rule of thumb: ~1e-2 at K=4096, plus another
// 2^-8 of output rounding for bf16 outputs). Cases built from pure or small
// integers can match exactly; each program's header states which check it
// uses.
static inline int check_close(const float *got, const float *want, long n,
                              float rtol) {
    long bad = 0;
    for (long i = 0; i < n; i++) {
        float w = want[i];
        if (fabsf(got[i] - w) > rtol * (1.0f + fabsf(w))) {
            if (bad < 5)
                fprintf(stderr, "MISMATCH at %ld: got %f, want %f\n", i,
                        got[i], w);
            bad++;
        }
    }
    if (bad) fprintf(stderr, "total mismatches: %ld / %ld\n", bad, n);
    return bad == 0;
}

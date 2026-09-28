// Shared utilities for the programs in this directory: error checks, a
// CUDA-event timer, and effective-bandwidth math. Same lineage as the
// common.h in the other topic directories, plus the bf16 header.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Wrap every CUDA API call; on error, report file, line, and cause immediately.
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

// Kernel launches return no error code; catch launch errors with these.
#define CUDA_CHECK_KERNEL()                        \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)

// cudaEvent-based timer; measures elapsed time on the GPU timeline (ms).
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

// Average of iters launches after warmup, in ms. Used for bandwidth and
// TFLOPS numbers alike.
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

// Effective bandwidth: bytes is the traffic that must cross HBM
// (reads + writes); work it out per kernel before passing it in.
static inline double effective_gbps(double bytes, float ms) {
    return bytes / (ms * 1e6);
}

// Element-wise compare against a reference; tolerances come from the
// caller, which knows the accumulation-order and output-rounding story.
static inline int check_close(const float* got, const float* want, long n,
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

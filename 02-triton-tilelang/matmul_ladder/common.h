// Small utilities shared by the CUDA programs here: checked error macros, a
// cudaEvent timer, seeded data fill, tolerance-based comparison, PASS/FAIL
// reporting, and a speedup printer. (first-kernels/saxpy.cu is deliberately
// self-contained and re-declares the error macros and event timing itself.)
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

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

// Kernel launch returns no error code; these two lines are how you catch launch errors.
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

// Fixed-seed pseudo-random fill so every run sees identical data.
// Values land in [0, 10).
static inline void fill_random(float *p, long n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    for (long i = 0; i < n; i++) p[i] = dist(rng);
}

// Element-wise compare; fails when relative error exceeds eps, returns 1 on pass.
static inline int check_close(const float *got, const float *want, long n,
                              float eps = 1e-4f) {
    for (long i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > eps * (1.0f + fabsf(want[i]))) {
            fprintf(stderr, "MISMATCH at %ld: got %f, want %f\n", i,
                    (double)got[i], (double)want[i]);
            return 0;
        }
    }
    return 1;
}

#define REPORT(ok)                       \
    do {                                 \
        if (ok) {                        \
            printf("PASS\n");            \
        } else {                         \
            printf("FAIL\n");            \
            exit(1);                     \
        }                                \
    } while (0)

// ---------------- Performance report ----------------
// Print the speedup ratio of two versions. If ratio < warn_below, print an
// extra hint without touching the exit code -- the program only lays out the
// numbers; interpreting fast vs slow is up to the reader.
// warn_below <= 0 means no speedup is expected; no hint is printed.
static inline float report_speedup(const char *label, float base_ms,
                                   float opt_ms, float warn_below,
                                   const char *hint) {
    float ratio = opt_ms > 0.f ? base_ms / opt_ms : 0.f;
    printf("%s = %.2fx\n", label, ratio);
    if (warn_below > 0.f && ratio < warn_below) {
        printf("WARN: %s (does not affect PASS)\n", hint);
    }
    return ratio;
}

// ---------------- Machine-readable result line ----------------
// One ##RESULT JSON line for external harnesses to parse. Opt-in via
// WMHPC_RESULT=1; off by default so everyday output stays clean.
// Convention: the exit code expresses correctness only; timing never affects it.
static inline void emit_result(const char *prob, const char *status,
                               const char *metrics_json) {
    const char *on = getenv("WMHPC_RESULT");
    if (!on || on[0] == '0' || on[0] == '\0') return;
    if (!metrics_json) metrics_json = "{}";

    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        printf("##RESULT {\"prob\":\"%s\",\"status\":\"%s\",\"metrics\":%s,"
               "\"device\":\"%s\",\"sm\":%d,\"cc\":\"%d.%d\"}\n",
               prob, status, metrics_json, prop.name,
               prop.multiProcessorCount, prop.major, prop.minor);
    } else {
        printf("##RESULT {\"prob\":\"%s\",\"status\":\"%s\",\"metrics\":%s}\n",
               prob, status, metrics_json);
    }
    fflush(stdout);
}

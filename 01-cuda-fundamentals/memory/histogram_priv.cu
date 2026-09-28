// Histogram privatization. The naive version funnels every increment through
// the same 256 global counters; the privatized version gives each block its
// own shared-memory histogram: per-block collisions stay on-chip, and only
// after the block is done are its 256 bins merged into the global histogram
// with one atomicAdd per bin. That collapse of global atomic traffic is where
// the speedup comes from.
// main() judges and times both; expected output PASS per kernel plus the
// naive/priv ratio.
#include "common.h"

#define BINS 256

__global__ void histogram_naive(const unsigned char *data, unsigned int *hist,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        atomicAdd(&hist[data[i]], 1u);
    }
}

__global__ void histogram_priv(const unsigned char *data, unsigned int *hist,
                               int n) {
    // One private shared histogram per block: block-internal collisions are
    // absorbed on-chip, and the global counters are touched exactly once per
    // block per bin at the end.
    __shared__ unsigned int local[BINS];

    int t = threadIdx.x;
    for (int b = t; b < BINS; b += blockDim.x) local[b] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        atomicAdd(&local[data[i]], 1u);
    }
    __syncthreads();

    for (int b = t; b < BINS; b += blockDim.x) {
        atomicAdd(&hist[b], local[b]);
    }
}

// ---------------- Judge and timing harness ----------------

typedef void (*hist_fn)(const unsigned char *, unsigned int *, int);

static float run_one(hist_fn fn, const char *name, const unsigned char *d_data,
                     unsigned int *d_hist, const unsigned int *h_ref, int n,
                     int blocks, int threads) {
    unsigned int h_hist[BINS];
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
    fn<<<blocks, threads>>>(d_data, d_hist, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    for (int b = 0; b < BINS; b++) {
        if (h_hist[b] != h_ref[b]) {
            fprintf(stderr, "bin %d: got %u, want %u\n", b, h_hist[b], h_ref[b]);
            printf("%s: FAIL\n", name);
            emit_result("histogram-priv", "fail", "{}");
            exit(1);
        }
    }

    const int reps = 50;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < reps; r++) fn<<<blocks, threads>>>(d_data, d_hist, n);
    float ms = timer.stop_ms() / reps;
    CUDA_CHECK_KERNEL();
    printf("%s: PASS  avg %.4f ms  (%.2f GB/s)\n", name, ms, n / ms / 1e6);
    return ms;
}

int main() {
    const int n = 1 << 24;

    unsigned char *h_data = (unsigned char *)malloc(n);
    unsigned int h_ref[BINS] = {0};
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> byte(0, BINS - 1);
    for (int i = 0; i < n; i++) h_data[i] = (unsigned char)byte(rng);
    for (int i = 0; i < n; i++) h_ref[h_data[i]]++;

    unsigned char *d_data;
    unsigned int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));

    int threads = 256, blocks = 1024;
    float ms_naive = run_one(histogram_naive, "naive", d_data, d_hist, h_ref, n,
                             blocks, threads);
    float ms_priv = run_one(histogram_priv, "priv ", d_data, d_hist, h_ref, n,
                            blocks, threads);
    // Threshold 10x: measured 149x on A100, 86x on V100; a ~1x ratio is the
    // signal that privatization did not actually engage.
    float ratio = report_speedup("naive / priv", ms_naive, ms_priv, 10.0f,
                                 "speedup below 10x; check whether privatization is actually taking effect");

    // Report how much shared memory the privatized kernel actually uses --
    // informational only, not a pass/fail condition.
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, histogram_priv));
    if (attr.sharedSizeBytes == 0) {
        printf("WARN: priv version did not use shared memory (does not affect PASS)\n");
    }

    char metrics[256];
    snprintf(metrics, sizeof(metrics),
             "{\"naive_ms\":%.4f,\"priv_ms\":%.4f,\"speedup\":%.3f,"
             "\"shared_bytes\":%zu}",
             ms_naive, ms_priv, ratio, attr.sharedSizeBytes);
    emit_result("histogram-priv", "pass", metrics);
    return 0;
}

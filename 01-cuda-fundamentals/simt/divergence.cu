// Warp divergence, measured. Both kernels do identical work per thread; the
// only difference is how the branch splits threads. Branching on tid % 2 splits
// every warp half-and-half, so both paths execute serially; branching on
// (tid / 32) % 2 keeps each warp entirely on one path. main() times both and
// prints the slowdown ratio.
#include "common.h"

// Branch on odd/even: every warp is split half and half.
__global__ void diverge_in_warp(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;
    if (tid % 2 == 0) {
        for (int i = 0; i < iters; i++) x = x * 1.000001f + 0.5f;
    } else {
        for (int i = 0; i < iters; i++) x = x * 0.999999f - 0.5f;
    }
    out[tid] = x;
}

// Branch by warp: every thread in a warp takes the same path.
__global__ void diverge_by_warp(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;
    if ((tid / 32) % 2 == 0) {
        for (int i = 0; i < iters; i++) x = x * 1.000001f + 0.5f;
    } else {
        for (int i = 0; i < iters; i++) x = x * 0.999999f - 0.5f;
    }
    out[tid] = x;
}

int main() {
    const int blocks = 1024, threads = 256, iters = 20000;
    const int n = blocks * threads;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));

    // Warm up both once.
    diverge_in_warp<<<blocks, threads>>>(d_out, iters);
    diverge_by_warp<<<blocks, threads>>>(d_out, iters);
    CUDA_CHECK_KERNEL();

    GpuTimer timer;

    timer.start();
    diverge_in_warp<<<blocks, threads>>>(d_out, iters);
    float ms_in = timer.stop_ms();

    timer.start();
    diverge_by_warp<<<blocks, threads>>>(d_out, iters);
    float ms_by = timer.stop_ms();

    CUDA_CHECK_KERNEL();
    printf("diverge in warp (tid %% 2)   : %8.3f ms\n", ms_in);
    printf("diverge by warp (tid/32 %% 2): %8.3f ms\n", ms_by);
    printf("ratio: %.2f\n", ms_in / ms_by);
    return 0;
}

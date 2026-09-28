// Timing traps: the same kernel, timed three ways, gives three different
// numbers. Host timing without a sync measures launch overhead only -- the
// host returns long before the GPU starts. Host timing after
// cudaDeviceSynchronize is real wall-clock time but pays the launch cost on
// the host's critical path. cudaEvents measure elapsed time on the GPU
// timeline itself -- the right choice for a performance index. Prints all
// three numbers back to back.
#include <chrono>
#include "common.h"

__global__ void busy(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 0.5f;
    for (int i = 0; i < iters; i++) x = x * 1.000001f + 0.5f;
    out[tid] = x;
}

int main() {
    const int blocks = 2048, threads = 256, iters = 5000;
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)blocks * threads * sizeof(float)));

    busy<<<blocks, threads>>>(d_out, iters);  // warm-up
    CUDA_CHECK_KERNEL();

    // Way 1: host clock, stopped immediately after launch.
    auto t0 = std::chrono::steady_clock::now();
    busy<<<blocks, threads>>>(d_out, iters);
    auto t1 = std::chrono::steady_clock::now();
    double ms_nosync = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Way 2: host clock, stopped once the GPU is actually done.
    t0 = std::chrono::steady_clock::now();
    busy<<<blocks, threads>>>(d_out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::steady_clock::now();
    double ms_sync = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Way 3: cudaEvent timing.
    GpuTimer timer;
    timer.start();
    busy<<<blocks, threads>>>(d_out, iters);
    float ms_event = timer.stop_ms();

    printf("host clock, no sync : %10.4f ms\n", ms_nosync);
    printf("host clock, synced  : %10.4f ms\n", ms_sync);
    printf("cudaEvent           : %10.4f ms\n", ms_event);
    return 0;
}

// Vector addition on unified memory: the same kernel as vector_add.cu, but the
// three buffers come from cudaMallocManaged instead of malloc + cudaMalloc, and
// every cudaMemcpy disappears -- CPU and GPU dereference the same pointers while
// the driver migrates pages between host memory and VRAM on fault.
// The timing window is kernel + explicit sync + CPU checksum readback;
// allocation, data fill, and the precomputed checksum all sit outside it.
// This window is not comparable with saxpy.cu's kernel-only event
// timing: it includes host wake-up on purpose, to show the full
// unified-memory round trip. Expected output: one timing line, then PASS.
#include <chrono>
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 24;  // 16M elements
    size_t bytes = (size_t)n * sizeof(float);

    // Create the CUDA context first. The first CUDA API call costs hundreds
    // of ms of init; inside the timing window it would drown out the effect
    // being measured.
    CUDA_CHECK(cudaFree(0));

    // Unified memory: one set of pointers both CPU and GPU can dereference;
    // on a page fault the driver migrates pages between host memory and VRAM.
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    fill_random(a, n, 1);
    fill_random(b, n, 2);

    // Expected checksum, precomputed on the host; also outside the timing.
    double want = 0;
    for (int i = 0; i < n; i++) want += (double)(a[i] + b[i]);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // ================= timing window opens =================
    auto t0 = std::chrono::steady_clock::now();

    vectorAdd<<<blocks, threads>>>(a, b, c, n);
    // Kernel launch is async: the host runs on without waiting for the GPU,
    // but the next line reads c on the CPU, so sync first or you race the GPU.
    // (In the explicit version this sync hides inside cudaMemcpy -- it is a
    // synchronous call.)
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU reads all results. Under unified memory this is the step that pulls
    // the result pages back to the host.
    double got = 0;
    for (int i = 0; i < n; i++) got += (double)c[i];

    auto t1 = std::chrono::steady_clock::now();
    // ================= timing window closes =================

    printf("copy + kernel + readback: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    REPORT(fabs(got - want) <= 1e-3 * (1.0 + fabs(want)));
    return 0;
}

// Vector addition, the canonical first CUDA kernel: one thread per element,
// 256-thread blocks, grid sized by rounding up, bounds check against n.
// main() judges the result against a CPU reference; expected output PASS.
#include "common.h"

// __global__ marks the function as a kernel: host code calls it with <<<...>>>.
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Global element index this thread owns.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check: the round-up grid launches more threads than elements.
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1000003;  // deliberately not a multiple of 256
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Host -> device copies; the direction argument is the last one.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    // Round up so the grid covers all n elements.
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch: the execution configuration goes inside <<<blocks, threads>>>.
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}

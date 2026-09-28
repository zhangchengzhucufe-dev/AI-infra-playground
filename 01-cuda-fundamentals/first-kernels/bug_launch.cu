// A launch-configuration bug and its fix, kept for reference: with
// threads = 2048 the launch exceeded maxThreadsPerBlock (1024 on this GPU),
// the launch was invalid, the kernel never ran, and the result check failed --
// with no error message, because nothing ever queried the launch error.
// threads = 1024 plus the CUDA_CHECK_KERNEL() macro after the launch is the
// corrected version.
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1000003;
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
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    int threads = 1024;  // fix: 2048 exceeds the max threads per block (1024 on
                         // this GPU), so the launch was invalid and the kernel
                         // never ran; add the error checks and you'll see
                         // cudaErrorInvalidConfiguration.
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();  // kernel launch has no return value; this catches launch errors

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}

// Why the failure is silent without CUDA_CHECK_KERNEL(): a kernel launch
// returns no error code, and launch errors stay queued until the next CUDA
// call queries them -- here nothing between the bad launch and the result
// check would ever surface cudaErrorInvalidConfiguration. The related limit
// is maxThreadsPerBlock, one of the fields device_query prints.

// Why __syncthreads() matters. Each block reverses its own 256 elements:
// stage into shared memory, sync, read back out crossed. Comment out the
// marked __syncthreads() line and the program races: threads can read buf
// before other threads' writes land, so results come out wrong -- or
// nondeterministically right. As-is, expected output PASS.
#include "common.h"

#define BLOCK 256

__global__ void reverse_blocks(const float *in, float *out, int n) {
    __shared__ float buf[BLOCK];
    int base = blockIdx.x * BLOCK;
    int t = threadIdx.x;

    buf[t] = in[base + t];
    __syncthreads();  // <-- comment this out to see the race
    out[base + t] = buf[BLOCK - 1 - t];
}

int main() {
    const int nblocks = 4096;
    const int n = nblocks * BLOCK;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_in, n, 7);
    for (int b = 0; b < nblocks; b++)
        for (int t = 0; t < BLOCK; t++)
            h_ref[b * BLOCK + t] = h_in[b * BLOCK + (BLOCK - 1 - t)];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    reverse_blocks<<<nblocks, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_out, h_ref, n));
    return 0;
}

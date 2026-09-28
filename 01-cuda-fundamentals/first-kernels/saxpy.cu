// SAXPY: y = 2.0 * x + y in single precision. Usage: ./saxpy <n>
// Inputs come from a fixed formula (no file I/O, easy to verify independently):
//   x[i] = ((i % 2048) - 1024) * 0.5f
//   y[i] = (i % 1024) - 512
// After computing, y is copied back to the host, accumulated in double, and
// one line SUM=<total> is printed; exit code 0.
// For n = 0 print SUM=0 (a 0-block launch is illegal, so special-case it).
// Deliberately self-contained: the error-check macros and cudaEvent timing
// are written out here instead of included from common.h. Verified by
// judge_saxpy.sh, which rebuilds this file and checks SUM for 7 values of n.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA API call; on error, report file, line, and cause immediately.
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err_ = (call);                                \
        if (err_ != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",       \
                    cudaGetErrorName(err_), __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                    \
            exit(1);                                              \
        }                                                         \
    } while (0)

// Kernel launch returns no error code; these two lines are how you catch launch errors.
#define CUDA_CHECK_KERNEL()                   \
    do {                                      \
        CUDA_CHECK(cudaGetLastError());       \
        CUDA_CHECK(cudaDeviceSynchronize());  \
    } while (0)

__global__ void saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <n>\n", argv[0]);
        return 1;
    }
    long n = atol(argv[1]);
    if (n < 0) n = 0;

    // n = 0: nothing to compute and a 0-block launch is illegal; special-case it.
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    if (!h_x || !h_y) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    for (long i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (float)((i % 1024) - 512);
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    // Round up: cover every element even when n is not a multiple of 256.
    int blocks = (int)((n + threads - 1) / threads);

    // cudaEvent timing: elapsed time on the GPU timeline over this interval (ms).
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    saxpy<<<blocks, threads>>>(/*n=*/(int)n, /*a=*/2.0f, d_x, d_y);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double s = 0;
    for (long i = 0; i < n; i++) s += (double)h_y[i];
    printf("SUM=%.0f (n=%ld, %.3f ms)\n", s, n, (double)ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);
    return 0;
}

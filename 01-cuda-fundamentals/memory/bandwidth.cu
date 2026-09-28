// Memory-access pattern vs effective bandwidth: the same kernel, run with read
// strides of 1, 2, 4, ..., 32 floats. Stride 1 is fully coalesced; as the
// stride grows, adjacent lanes in a warp land stride floats apart, each 32-lane
// access spreads over more 128-byte transactions, and throughput falls.
// n is a power of two, so & (n-1) is a cheap modulo. Prints a stride/ms/GB/s
// table.
#include "common.h"

// stride = 1 is the coalesced case; at stride s, neighboring lanes in a warp
// read addresses s floats apart. n is a power of two, so & (n-1) == % n.
__global__ void strided_copy(const float *in, float *out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (long)i * stride & (n - 1);
        out[i] = in[j];
    }
}

int main() {
    const int n = 1 << 24;  // 16M elements, power of two
    size_t bytes = (size_t)n * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    strided_copy<<<blocks, threads>>>(d_in, d_out, n, 1);  // warm-up
    CUDA_CHECK_KERNEL();

    const int reps = 20;
    int strides[] = {1, 2, 4, 8, 16, 32};
    printf("%8s %12s %12s\n", "stride", "ms", "GB/s");
    for (int s : strides) {
        GpuTimer timer;
        timer.start();
        for (int r = 0; r < reps; r++)
            strided_copy<<<blocks, threads>>>(d_in, d_out, n, s);
        float ms = timer.stop_ms() / reps;
        CUDA_CHECK_KERNEL();
        // 4 bytes read + 4 bytes written per element.
        double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;
        printf("%8d %12.4f %12.1f\n", s, ms, gbps);
    }
    return 0;
}

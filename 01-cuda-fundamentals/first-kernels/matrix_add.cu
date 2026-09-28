// 2D matrix addition: 16x16 blocks of dim3 and a 2D grid over an M x N matrix,
// rounded up independently in both dimensions; the row-major flat index is
// row * N + col. main() judges the result against a CPU reference; expected
// output PASS.
#include "common.h"

__global__ void matrixAdd(const float *a, const float *b, float *c, int M, int N) {
    // Row this thread handles: y-direction built-ins.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Column this thread handles: x-direction built-ins.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 2D bounds check: both row and column can overrun the matrix.
    if (row < M && col < N) {
        int idx = row * N + col;  // row-major flattened to a 1D index
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int M = 1000, N = 700;  // neither is a multiple of 16
    const long total = (long)M * N;
    size_t bytes = total * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, total, 1);
    fill_random(h_b, total, 2);
    for (long i = 0; i < total; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);  // 16 columns along x, 16 rows along y
    // 2D grid: rounded up independently in both directions.
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    matrixAdd<<<blocks, threads>>>(d_a, d_b, d_c, M, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, total));
    return 0;
}

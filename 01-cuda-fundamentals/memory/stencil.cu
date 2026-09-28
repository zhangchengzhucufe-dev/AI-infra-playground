// Three-point averaging stencil: out[i] = (in[i-1] + in[i] + in[i+1]) / 3,
// positions past either edge read as 0. Two kernels, identical math: one with
// a statically declared shared tile, one with a dynamically sized extern
// __shared__ tile whose byte size is passed at launch. The shared tile exists
// to cut global-memory traffic: each element is read from VRAM once, and the
// block's threefold reuse then hits on-chip shared memory, whose latency is
// far below VRAM's.
// main() checks both versions against a CPU reference; expected output
// "static PASS", "dynamic PASS", PASS.
#include "common.h"

#define BLOCK 256
#define RADIUS 1

__global__ void stencil_static(const float *in, float *out, int n) {
    // Static shared tile: BLOCK elements plus a halo of RADIUS on each side.
    __shared__ float tile[BLOCK + 2 * RADIUS];

    int g = blockIdx.x * blockDim.x + threadIdx.x;  // global index
    int l = threadIdx.x + RADIUS;                   // position inside the tile

    tile[l] = (g < n) ? in[g] : 0.f;
    // Threads at the block edges also fetch one halo element per side.
    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }

    // Barrier: the tile must be fully populated before anyone reads it.
    __syncthreads();

    if (g < n) {
        // Three-point average, read from the tile (never from global in[]).
        out[g] = (tile[l - 1] + tile[l] + tile[l + 1]) / 3.f;
    }
}

__global__ void stencil_dynamic(const float *in, float *out, int n) {
    // Dynamic shared memory: declared extern, sized at launch time.
    extern __shared__ float tile[];

    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int l = threadIdx.x + RADIUS;

    tile[l] = (g < n) ? in[g] : 0.f;
    if (threadIdx.x < RADIUS) {
        int left = g - RADIUS;
        int right = g + BLOCK;
        tile[l - RADIUS] = (left >= 0) ? in[left] : 0.f;
        tile[l + BLOCK] = (right < n) ? in[right] : 0.f;
    }
    __syncthreads();
    if (g < n) {
        out[g] = (tile[l - 1] + tile[l] + tile[l + 1]) / 3.f;
    }
}

int main() {
    const int n = 1000003;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_in, n, 3);
    for (int i = 0; i < n; i++) {
        float l = (i > 0) ? h_in[i - 1] : 0.f;
        float r = (i < n - 1) ? h_in[i + 1] : 0.f;
        h_ref[i] = (l + h_in[i] + r) / 3.f;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK - 1) / BLOCK;

    stencil_static<<<blocks, BLOCK>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_ref, n)) REPORT(0);
    printf("static  PASS\n");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    // Dynamic launch: the third config argument is the tile size in bytes.
    stencil_dynamic<<<blocks, BLOCK, (BLOCK + 2 * RADIUS) * sizeof(float)>>>(
        d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_out, h_ref, n)) REPORT(0);
    printf("dynamic PASS\n");

    REPORT(1);
    return 0;
}

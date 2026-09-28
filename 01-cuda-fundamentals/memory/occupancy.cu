// Occupancy vs bandwidth, measured. Shared memory is allocated per block, so
// the more a block takes, the fewer blocks fit on one SM and the fewer warps
// stay resident. stream_add declares dynamic shared memory it never uses for
// data: compute and memory traffic are identical in every row of the table;
// only the SM's parallelism changes. For each allocation level the program
//   1. queries cudaOccupancyMaxActiveBlocksPerMultiprocessor for the
//      theoretical resident blocks per SM and converts that to occupancy, and
//   2. times the effective bandwidth of the same elementwise add.
// Prints the table, then the block size suggested by
// cudaOccupancyMaxPotentialBlockSize.
#include "common.h"

#define BLOCK 256

__global__ void stream_add(const float *a, const float *b, float *c, int n) {
    extern __shared__ float ballast[];  // occupies shared memory only, never read or written
    (void)ballast;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int smem_sm = (int)prop.sharedMemPerMultiprocessor;
    int smem_blk_max = (int)prop.sharedMemPerBlockOptin;
    int max_threads = prop.maxThreadsPerMultiProcessor;
    printf("%s: %d KB shared memory / SM, max %d resident threads / SM\n\n",
           prop.name, smem_sm / 1024, max_threads);

    // Opt in above the default 48 KB dynamic shared memory per block.
    CUDA_CHECK(cudaFuncSetAttribute((const void *)stream_add,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_blk_max));

    const int n = 1 << 26;
    size_t bytes = (size_t)n * sizeof(float);
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));
    int nblocks = (n + BLOCK - 1) / BLOCK;

    // Shared-memory levels, as fractions of the per-SM total: a block that
    // takes 1/x of it leaves room for roughly x blocks per SM. On this card
    // the six fractions land on four distinct resident-block counts (6, 5,
    // 3, 1 -- the middle two fractions both give 6); the exact levels are
    // architecture-dependent -- the API's numbers are authoritative.
    const double fracs[] = {0.0, 0.132, 0.15, 0.18, 0.29, 0.55};
    printf("%-14s %-16s %-11s %s\n",
           "shared/block", "blocks/SM (theo)", "occupancy", "measured BW");
    for (int k = 0; k < 6; k++) {
        int smem = (int)(smem_sm * fracs[k]);
        if (smem > smem_blk_max) smem = smem_blk_max;

        int active = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, stream_add, BLOCK, smem));
        double occ = 100.0 * active * BLOCK / max_threads;

        stream_add<<<nblocks, BLOCK, smem>>>(d_a, d_b, d_c, n);  // warm-up
        CUDA_CHECK_KERNEL();
        const int reps = 20;
        GpuTimer timer;
        timer.start();
        for (int r = 0; r < reps; r++)
            stream_add<<<nblocks, BLOCK, smem>>>(d_a, d_b, d_c, n);
        float ms = timer.stop_ms() / reps;
        CUDA_CHECK_KERNEL();
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;

        printf("%8.1f KB %10d %14.1f%% %10.1f GB/s\n",
               smem / 1024.0, active, occ, gbps);
    }

    // Second occupancy API: let the runtime suggest a block size that
    // maximizes occupancy.
    int min_grid = 0, best_block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &best_block, stream_add, 0, 0));
    printf("\ncudaOccupancyMaxPotentialBlockSize suggestion (smem = 0): blockSize = %d\n",
           best_block);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}

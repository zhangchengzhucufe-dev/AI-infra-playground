// Prints the key properties of the installed GPU: the fields of cudaDeviceProp
// that matter most for writing kernels -- SM count, warp size, shared memory
// per block, max resident threads per SM, global memory, max threads per block.
#include "common.h"

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU name            : %s\n", prop.name);
    printf("compute capability  : %d.%d\n", prop.major, prop.minor);

    // Number of streaming multiprocessors on the device.
    printf("SM count            : %d\n", prop.multiProcessorCount);

    // Threads per warp (32 on every current NVIDIA GPU).
    printf("warp size           : %d\n", prop.warpSize);

    // Largest shared memory allocation a single block can request (bytes).
    printf("shared mem / block  : %zu\n", (size_t)prop.sharedMemPerBlock);

    // Hard ceiling on threads resident per SM; occupancy is measured against it.
    printf("max threads / SM    : %d\n", prop.maxThreadsPerMultiProcessor);

    // Total device global memory (bytes).
    printf("global mem          : %zu\n", (size_t)prop.totalGlobalMem);

    printf("max threads / block : %d\n", prop.maxThreadsPerBlock);
    return 0;
}

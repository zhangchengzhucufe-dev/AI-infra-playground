// Minimal first CUDA program: hello<<<4, 8>>> prints its block and thread
// coordinates from inside the kernel. Also the guinea pig for the PTX/SASS
// compile experiments (see the Makefile header).
// Build & run: make bin/basics/hello
#include "common.h"

__global__ void hello() {
    printf("hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<4, 8>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}

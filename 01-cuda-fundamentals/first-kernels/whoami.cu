// Block scheduling order, observed: 16 blocks each check in with printf from
// thread 0. Run it a few times in a row -- the order the blocks appear in
// changes between runs, because the order in which blocks execute is simply
// not specified; the scheduler fills SMs however resources allow.
#include "common.h"

__global__ void whoami() {
    // Thread 0 of each block checks in.
    if (threadIdx.x == 0) {
        printf("block %d checks in\n", blockIdx.x);
    }
}

int main() {
    whoami<<<16, 32>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}

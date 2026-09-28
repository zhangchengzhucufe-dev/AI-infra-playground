// How row stride affects ldmatrix throughput.
//
// The same 16x16 fp16 tile sits in smem with four row strides:
// 32 B (packed), 64 B, 128 B (the tile embedded in a 64-column fp16
// matrix), and 128+16 B (padding). The kernel issues ldmatrix .x4 in a
// loop and prints the average cycle count per issue.
//
// What the strides do: packed rows make the same 16B fragment rows land
// on overlapping banks, so wavefronts and conflict replays grow with
// the stride -- in the ncu capture (ldsm_stride_ncu.csv), 8/16/32
// wavefronts and 4/12/28 conflicts per ldmatrix issue at 32/64/128 B,
// while the padded stride is conflict-free at the 4-wavefront minimum.
// The cycle ratios come out smaller than the wavefront ratios because
// with 8 warps resident the LSU is not the only bottleneck -- other
// stages overlap the serialization.
// ncu command (two counters):
//   ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
//       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
//       ./bin/ldmatrix/ldsm_stride
//
// Run: make run/ldmatrix/ldsm_stride
#include <cuda_fp16.h>
#include <cstdint>
#include "../common.h"

constexpr int ITERS = 4096;

// 8 warps issue together to saturate the LSU, so throughput is set by
// bank conflicts alone; with a single warp, pipelining hides most of
// the serialization. Each warp uses its own smem region with the same
// access pattern.
template <int STRIDE>
__global__ void ldsm_kernel(unsigned* out, long long* cycles) {
    constexpr int WARPS = 8;
    __shared__ __align__(16) uint8_t smem[WARPS * 16 * 160];  // ldmatrix row addresses need 16B alignment
    for (int i = threadIdx.x; i < WARPS * 16 * 160; i += WARPS * 32)
        smem[i] = (uint8_t)i;
    __syncthreads();
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int r = lane & 15, h = lane >> 4;
    unsigned addr = (unsigned)__cvta_generic_to_shared(
        &smem[w * 16 * 160 + r * STRIDE + h * 16]);
    unsigned acc = 0, r0, r1, r2, r3;
    __syncthreads();
    long long t0 = clock64();
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(addr));
        acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
    long long t1 = clock64();
    __syncthreads();
    if (threadIdx.x == 0) {
        *cycles = t1 - t0;
        *out = acc;
    }
}

template <int STRIDE>
static void run_one(const char* name) {
    unsigned* dout;
    long long* dcyc;
    CUDA_CHECK(cudaMalloc(&dout, 4));
    CUDA_CHECK(cudaMalloc(&dcyc, 8));
    ldsm_kernel<STRIDE><<<1, 256>>>(dout, dcyc);  // warmup
    ldsm_kernel<STRIDE><<<1, 256>>>(dout, dcyc);
    CUDA_CHECK_KERNEL();
    long long cyc;
    CUDA_CHECK(cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost));
    printf("stride %-8s %6.2f cycles / ldmatrix (amortized over 8 warps)\n",
           name, (double)cyc / ITERS);
    cudaFree(dout);
    cudaFree(dcyc);
}

int main() {
    run_one<32>("32B");
    run_one<64>("64B");
    run_one<128>("128B");
    run_one<144>("128B+pad");
    return 0;
}

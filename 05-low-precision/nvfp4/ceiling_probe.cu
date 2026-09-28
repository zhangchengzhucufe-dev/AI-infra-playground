// Bandwidth ceiling probe for the NVFP4 quantization access pattern.
//
// The kernel performs exactly the same memory traffic as the quant kernel
// (reads the same bf16, writes the 8 bytes of data and 1 byte of scale
// factor to the same places) with no math -- xor the bits read in and
// write them straight out. Its runtime is the floor for this access
// pattern on this card.
//
// Read three numbers together: probe GB/s, the quant kernel's GB/s, and
// their ratio -- how far the quant kernel is from its own ceiling, and
// whether the remaining gap is memory or compute (ncu's SM% / DRAM%
// settles that).
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16* __restrict__ in,
                             uint8_t* __restrict__ dataOut,
                             uint8_t* __restrict__ sfOut, int M, int K) {
    // Exactly the quant kernel's access shape: one thread per group, reads
    // 16 bf16, writes 8 bytes of data + 1 byte of SF; the only difference
    // is no quantization -- xor the bits read in and pass them through
    // (keeps the compiler from optimizing the traffic away).
    long g = (long)blockIdx.x * BLOCK + threadIdx.x;
    int groupsPerRow = K / NVFP4_GROUP;
    long total = (long)M * groupsPerRow;
    if (g >= total) return;
    int r = (int)(g / groupsPerRow);
    int kg = (int)(g % groupsPerRow);

    const uint16_t* row = reinterpret_cast<const uint16_t*>(
        in + (size_t)r * K + kg * NVFP4_GROUP);
    uint8_t out[NVFP4_GROUP / 2];
    uint8_t acc = 0;
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP; i += 2) {
        uint16_t a = row[i], b = row[i + 1];
        uint8_t lo = (uint8_t)(a ^ b);
        uint8_t hi = (uint8_t)((a >> 8) ^ (b >> 8));
        out[i / 2] = (uint8_t)((hi << 4) | lo);
        acc ^= lo ^ hi;
    }
    uint8_t* dst =
        dataOut + ((size_t)r * K / 2 + kg * (NVFP4_GROUP / 2));
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP / 2; i++) dst[i] = out[i];
    sfOut[sf_swizzled_offset(r, kg, nvfp4_num_ktiles(K))] = acc;
}

static void launch_probe(const __nv_bfloat16* in, uint8_t* dataOut,
                         uint8_t* sfOut, int M, int K, int sms) {
    // Launch config identical to launch_nvfp4_quant so the comparison is fair.
    (void)sms;  // kept in the signature to mirror the quant kernel's launcher
    constexpr int BLOCK = 256;
    long total = (long)M * (K / NVFP4_GROUP);
    int grid = (int)((total + BLOCK - 1) / BLOCK);
    probe_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    for (const auto& shape :
         {std::pair{4096, 7168}, {16384, 4096}, {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, nvfp4_sf_bytes(M, K)));
        CUDA_CHECK(cudaMemset(dx, 0x3c, n * 2));
        float ms = time_avg_ms(
            [&] { launch_probe(dx, dd, dsf, M, K, sms); }, 50);
        CUDA_CHECK_KERNEL();
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-6d K=%-5d  probe %8.2f us  %6.0f GB/s\n", M, K, ms * 1e3,
               effective_gbps(bytes, ms));
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return 0;
}

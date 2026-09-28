// NVFP4 quantization kernel.
//
// Input bf16 matrix [M, K] (K a multiple of 16), outputs:
//   dataOut: packed e2m1, K/2 bytes per row, the low nibble holds the
//            even-index element
//   sfOut:   e4m3 SF in the swizzled layout (offsets via nvfp4_common.h's
//            sf_swizzled_offset; the whole SF tensor is zeroed before the
//            call)
//
// Per-group computation order (the host reference runs the same float
// math in the same order, so the comparison is byte-exact):
//   amax = max |v| over the group's 16 values
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// e2m1 conversion goes through e2m1_encode.h's software encoder (host/
// device, plain C++), so this pipeline runs on any sm_80+ GPU. The
// production path on sm_100 is the hardware cvt instruction; the encoder
// was verified against an independent float64 reference over 402,864
// values including all midpoints, so the packed output is identical.
//
// Organization: 16 elements = 32 bytes, so one thread owns exactly one
// group and intra-group cooperation is not needed; quant has no cross-row
// dependencies, the grid layout is entirely free.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "e2m1_encode.h"
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                   uint8_t* __restrict__ dataOut,
                                   uint8_t* __restrict__ sfOut, int M, int K) {
    // one thread per group: global group id -> (row, group within row)
    long g = (long)blockIdx.x * BLOCK + threadIdx.x;
    int groupsPerRow = K / NVFP4_GROUP;
    long total = (long)M * groupsPerRow;
    if (g >= total) return;
    int r = (int)(g / groupsPerRow);
    int kg = (int)(g % groupsPerRow);

    const __nv_bfloat16* row = in + (size_t)r * K + kg * NVFP4_GROUP;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP; i++)
        amax = fmaxf(amax, fabsf(__bfloat162float(row[i])));

    __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
    float s = float(sf8);
    float inv = s != 0.f ? 1.0f / s : 0.0f;
    sfOut[sf_swizzled_offset(r, kg, nvfp4_num_ktiles(K))] =
        *(uint8_t*)&sf8;

    // convert and pack pairwise: the low nibble holds the even-index element
    uint8_t* dst = dataOut + ((size_t)r * K / 2 + kg * (NVFP4_GROUP / 2));
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP; i += 2) {
        float v0 = __bfloat162float(row[i]) * inv;
        float v1 = __bfloat162float(row[i + 1]) * inv;
        dst[i / 2] = (uint8_t)(e2m1_encode(v0) | (e2m1_encode(v1) << 4));
    }
}

// The quantize driver and the fused rms+quant program both call through
// this signature; the grid size is defined here.
inline void launch_nvfp4_quant(const __nv_bfloat16* in, uint8_t* dataOut,
                               uint8_t* sfOut, int M, int K, int sms) {
    constexpr int BLOCK = 256;
    long total = (long)M * (K / NVFP4_GROUP);
    int grid = (int)((total + BLOCK - 1) / BLOCK);
    nvfp4_quant_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}

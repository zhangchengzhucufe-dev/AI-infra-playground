// NVFP4 format conventions: data layout, scale-factor layout, and the
// e2m1 lattice decode.
//
// Data format:
//   - a quantization group is 16 consecutive elements along K, one scale
//     factor (SF) per group
//   - SF = the group's amax / 6.0 (6.0 is e2m1's max), stored as e4m3;
//     the scale used for dequantization is float(e4m3(SF)), so SF itself
//     goes through quantization
//   - data is e2m1, two per byte, the low nibble holds the even-index
//     element
//
// SF is not stored row-major: it uses the swizzled layout the tensor core
// consumes, [numMTiles, numKTiles, 32, 4, 4]: tiles of 128 rows along M,
// numKTiles = ceil(K/64) (one K tile covers 4 groups).
// Byte offset formula in sf_swizzled_offset -- this is the layout
// cuBLASLt / tcgen05 kind::mxf4nvf4 read on the sm_100 family, so a
// quantized tensor stays consumable there even though this pipeline
// itself runs the software encoder.
#pragma once
#include <cstdint>
#include <cuda_fp8.h>

constexpr int NVFP4_GROUP = 16;

__host__ __device__ inline int nvfp4_num_ktiles(int K) {
    return (K / NVFP4_GROUP + 3) / 4;
}

// Total SF tensor bytes (padding from the M-side 128 alignment is allocated
// and zeroed too).
__host__ __device__ inline int64_t nvfp4_sf_bytes(int M, int K) {
    return (int64_t)((M + 127) / 128) * nvfp4_num_ktiles(K) * 512;
}

// SF byte offset for row, group kGroup (= k/16).
__host__ __device__ inline int64_t sf_swizzled_offset(int row, int kGroup,
                                                      int numKTiles) {
    int mTileIdx = row >> 7;         // row / 128
    int outerM   = row & 31;         // row % 32
    int innerM   = (row >> 5) & 3;   // (row / 32) % 4
    int kTileIdx = kGroup >> 2;      // kGroup / 4
    int innerK   = kGroup & 3;       // kGroup % 4
    return ((int64_t)(mTileIdx * numKTiles + kTileIdx) << 9) |
           (outerM << 4) | (innerM << 2) | innerK;
}

// e2m1 decode: the lattice is fixed, so decode is a table lookup (the
// matching encode lives in e2m1_encode.h).
__host__ __device__ inline float e2m1_decode(uint8_t nib) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = mag[nib & 7];
    return (nib & 8) ? -m : m;
}

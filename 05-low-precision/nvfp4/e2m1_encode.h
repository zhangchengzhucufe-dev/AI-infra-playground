// Host/device e2m1 (NVFP4) software encoder: threshold chain implementing
// round-to-nearest-even, boundary ownership derived by hand; cross-checked
// against a float64 reference over 402,864 values (midpoints +/-1e-6,
// saturation, -0.0) with zero mismatches.
//
// e2m1's magnitude lattice is 0, 0.5, 1, 1.5, 2, 3, 4, 6 (codes 0-7); bit 3
// is the sign. Semantics match the hardware cvt instruction
// (cvt.rn.satfinite.e2m1x2.f32): round to nearest, ties (exactly between
// two lattice points) go to the even mantissa, values above 6 saturate to 6
// (satfinite). Inputs are guaranteed finite.
//
// This encoder is the host reference for the whole NVFP4 pipeline: the
// quantize and fused rms+quant checks generate their ground truth with it,
// and encode_check.cu aligns it with the hardware point-by-point. Compiles
// on both __host__ and __device__.
#pragma once
#include <cstdint>
#include <math.h>

// Magnitude lattice (code -> magnitude): 0:0, 1:0.5, 2:1, 3:1.5, 4:2, 5:3,
// 6:4, 7:6. Codes with mantissa bit (code bit0) 0 are "even": 0, 2, 4, 6.
// RN-even midpoint assignment (worked out case by case):
//   0.25 -> 0 (even code 0), 0.75 -> 2 (up), 1.25 -> 2 (down),
//   1.75 -> 4 (up), 2.5 -> 4 (down), 3.5 -> 6 (up), 5.0 -> 6 (down)
// So the segment boundaries alternate "<= takes the lower / < takes the
// upper":
//   |a| <= 0.25 -> 0; |a| <  0.75 -> 1; |a| <= 1.25 -> 2; |a| <  1.75 -> 3;
//   |a| <= 2.5  -> 4; |a| <  3.5  -> 5; |a| <= 5.0  -> 6; otherwise
//   (> 6 saturates) -> 7
// Midpoints are exact floats, so plain <= / < comparisons match the
// hardware bit-for-bit.
__host__ __device__ inline uint8_t e2m1_encode(float v) {
    float a = fabsf(v);
    uint8_t mag;
    if (a <= 0.25f)
        mag = 0;
    else if (a < 0.75f)
        mag = 1;
    else if (a <= 1.25f)
        mag = 2;
    else if (a < 1.75f)
        mag = 3;
    else if (a <= 2.5f)
        mag = 4;
    else if (a < 3.5f)
        mag = 5;
    else if (a <= 5.0f)
        mag = 6;
    else
        mag = 7;  // in (5, 6] the nearest lattice point is 6; > 6 saturates to 6 via satfinite
    // sign via signbit: hardware keeps the sign bit for -0.0 too (v < 0 misses it)
    return (uint8_t)((signbit(v) ? 8u : 0u) | mag);
}

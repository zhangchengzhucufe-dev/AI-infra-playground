# 05 - Low precision: NVFP4 and block scaling

Why 4-bit formats exist, what outliers do to naive quantization, and an NVFP4 quantization pipeline built from the encoder up - running end to end on this machine. One scope note: the e2m1 conversion goes through the software encoder (verified against an independent float64 reference), not the sm_100-only hardware `cvt` instruction, so everything here runs on any sm_80+ GPU including my RTX 3060. The packed bytes are identical; what you give up without the hardware instruction is throughput, which the ceiling probe measures.

## Layout

```
nvfp4/            e2m1_encode.h         software e2m1 encoder, round-to-nearest-even
                  nvfp4_quant_kernel.h  quantize kernel: amax -> scale -> swizzled SF -> pack
                  nvfp4_common.h        group size, swizzled SF offset math
                  quantize.cu           end-to-end quantization + byte-exact check + bandwidth
                  ceiling_probe.cu      bandwidth ceiling, same shape as the quantizer
                  fused_rms_quantize.cu RMSNorm + quantize in one kernel, vs the two-step baseline
experiments/      quant_outlier.py      outlier-poisoning experiment (pure python/torch)
                  block_scale_sim.py    block-scale GEMM algebra
tests/            3 tests for the algebra
```

## The outlier experiment (experiments/quant_outlier.py)

Quantize a tensor that contains one value of magnitude 3000 with a single global scale. e4m3's max is 448, so the scale becomes 3000/448 = 6.7 and every normal-sized value loses precision: error at x~0.5 goes from 4.6e-2 to 3.1e-4 (149x better) once the outlier is removed. That one number costs the whole tensor.

Values below scale x 2^-10 (about 0.0065 here) round to exactly zero, which is half of e4m3's smallest subnormal code point.

Per-block scales (1x128 groups) shrink the blast radius: outlier-free blocks improve ~34x, and only the 127 neighbors inside the outlier's own block still suffer. That's the whole reason MX/NVFP4-style formats use grouped scales.

## Why scale groups run along K (experiments/block_scale_sim.py)

If the scale is constant over the whole dot product (per row/col), it factors out of the sum. If scales change along K, they don't: each segment has to be rescaled before accumulating. My test file includes a deliberately wrong implementation that restores only once at the end, and it really does produce a large error (that's the third test).

This is why CUTLASS lays scales out as M x ceil(K/SV), one scale per 16 or 32 contiguous K: the mma consumes contiguous K chunks anyway, so scale factors can stream in with the K tile. DeepSeek-V3 uses 128-wide K groups, NVFP4 uses 16. Finer groups = 8x more scale metadata, but an outlier poisons 16 values instead of 128.

## The NVFP4 pipeline (nvfp4/)

- **e2m1_encode.h**: software encoder with the RN-even boundary table derived by hand (0.25 rounds to 0, 0.75 to 2, 1.25 to 2, 1.75 to 4 ... the "even" side alternates, so the comparison chain uses <= and < alternating). Cross-checked against a float64 reference over 402,864 values (all midpoints +/-1e-6, the saturation range, and -0.0): zero mismatches. One real bug found that way: testing `v < 0` for the sign misses -0.0, which encodes with the sign bit set (0x8); the check has to be `signbit(v)`.
- **quantize.cu + nvfp4_quant_kernel.h**: one thread per 16-element group: amax -> e4m3 scale -> swizzled scale layout write -> pack two nibbles per byte. Device and host run the identical float chain, so the check is byte-exact. Measured on the 3060: PASS at all shapes, 242 GB/s at M=4096 K=7168 against the probe's 273-290 GB/s ceiling - within ~85% of pure copy bandwidth for a kernel doing real math.
- **ceiling_probe.cu**: the same tiling and the same bytes moved as the quantizer, with xor pass-through instead of math. It exists so the quant kernel's bandwidth has an honest ceiling to be compared against, instead of a datasheet number.
- **fused_rms_quantize.cu**: RMSNorm + quantize in one kernel (float4 loads, shuffle reduction for sum-of-squares, per-group recompute -> scale -> pack, no bf16 intermediate) against a tuned two-step baseline. Byte math says 2.56x is the ceiling (two-step writes and re-reads a bf16 intermediate); measured on the 3060:

| M | K | two-step (us) | fused (us) | speedup |
| --- | --- | --- | --- | --- |
| 1 | 4096 | 24.7 | 13.8 | 1.78x |
| 256 | 4096 | 37.9 | 25.6 | 1.48x |
| 1024 | 4096 | 106.5 | 80.2 | 1.33x |
| 4096 | 4096 | 396.6 | 290.4 | 1.37x |
| 16384 | 4096 | 1549.6 | 1049.2 | 1.48x |
| 4096 | 7168 | 687.7 | 440.9 | 1.56x |
| 16384 | 7168 | 2617.3 | 1566.1 | 1.67x |
| 4096 | 8192 | 787.1 | 411.2 | 1.91x |
| 16384 | 8192 | 3224.8 | 1685.0 | 1.91x |

The fused version wins everywhere, but the margin moves between runs (an earlier run showed 1.1x at small M and 2.2x at 16384x7168; laptop thermals are part of the measurement). The stable part of the pattern: launch overhead eats into the gain at small M, the gain grows with K as the bf16 round trip becomes the dominant cost, and everything stays below the 2.56x byte-account ceiling. Both the fused output and the two-step baseline's rms_norm intermediate are validated against host references, so the speedups are not coming from a quietly broken baseline. Byte differences vs the host reference stay within the 1e-4 tolerance (different sumsq reduction order flips values sitting on rounding boundaries).

## W4A16 vs NVFP4, and when you'd want which

W4A16 (Marlin-style) quantizes weights for storage: int4 saves memory and decode-time weight bandwidth, but compute dequantizes back to fp16. NVFP4 quantizes the compute itself, and fp4 tensor cores are 4x bf16 peak. At decode shapes (M<=16 - topic 04's study shows them at ~100% bandwidth, ~1% TC) the bandwidth saving is what pays, which is why W4A16 was the popular choice before fp4 hardware existed.

## Run it

```bash
make run/nvfp4/quantize
make run/nvfp4/fused_rms_quantize      # the fusion table, ~1 minute
cd .. && pytest 05-low-precision/tests
python 05-low-precision/experiments/quant_outlier.py    # runs anywhere
```

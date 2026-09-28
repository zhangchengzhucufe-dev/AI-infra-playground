# 03 - Tensor cores: mma, ldmatrix, smem descriptors

sm_80 tensor core programming, learned from the bottom up: which lane holds which fragment element, how to load them without hundreds of extra instructions, and how shared memory has to be laid out so the hardware can actually consume it. Everything here runs on any sm_80+ card; I developed it on the RTX 3060 (sm_86).

## First, the numbers that explain why this topic exists

I derived the tensor core peaks for the 5090 and B300 from core counts and clocks (dense, boost clocks, one FMA = 2 FLOPs). B300 numbers are folded from public GB300 NVL72 specs since I don't have the card or a real datasheet, so treat that column as approximate.

| quantity | 5090 (sm_120) | B300 (sm_103) |
| --- | --- | --- |
| bf16 FLOP/cycle/SM | 1024 (4 TC x 128 FMA x 2) | ~2048 (back-derived) |
| bf16 peak | ~419 TFLOPS (170 SM x 2.41 GHz) | ~2.25 PFLOPS |
| fp8 / fp4 peak | ~838 / ~1676 TFLOPS | ~4.5 / ~15 PFLOPS |
| bandwidth | ~1792 GB/s (GDDR7) | ~8 TB/s (HBM3e) |
| balance point | ~234 FLOP/byte | ~281 FLOP/byte |

Sanity check that convinced me the 5090 numbers are right: NVIDIA's "3352 AI TOPS" is exactly the fp4 sparse number (1676 x 2), so dividing down 1676 / 2 / 2 = 419 lands on my derived bf16 peak.

Now compare with one m16n8k16 mma instruction: 4096 FLOP over 1280 B of operands + result = **3.2 FLOP/byte**. Against a balance point of 234-281, that's a 70-90x gap. If every mma read its operands from DRAM, the tensor core would be waiting ~99% of the time. This one calculation is why topics 03-05 are mostly about data supply rather than the mma itself.

## Files

```
fragments/   first_mma.cu      minimal m16n8k16 fp16 example + arch mismatch experiments
             fragment_map.cu   per-lane fragment truth table for mma.m16n8k32
             bug_fragment.cu   a layout bug, its exact symptom, the fix
ldmatrix/    ldmatrix.cu       manual element loads vs ldmatrix, both in one kernel
             ldsm_stride.cu    bank conflicts vs row stride, measured with ncu
smem/        descriptor.cu     wgmma/TMA smem descriptors, three scenarios
             swizzle.cu        the three XOR swizzle patterns
```

## Fragment layouts

`fragment_map.cu` writes out the full mapping for `mma.m16n8k32` and checks every (lane, register, byte) against a host truth table. For A: row = group + (r&1)*8, col = tig*4 + (r>>1)*16 + j. For B (col-major): row = tig*4 + r*16 + j, col = group. It passed on the first run, which was a good feeling, because it means the PTX ISA doc and the hardware actually agree with each other.

`bug_fragment.cu` plants a bug I made once for real: register pairs a2/a3 and a6/a7 of the A fragment were copied with the wrong row index (missing +8). The symptom is very regular: rows 0-7 of D perfect, rows 8-15 wrong in 59/128 elements. Four registers, one +8, and the mismatch pattern matches the theory exactly.

## ldmatrix

`ldmatrix.cu` loads the same m16n8k16 A/B fragments two ways and keeps both in one kernel (template switch): per-element 16-bit smem reads + half2 packing, vs ldmatrix. Both paths pass strict equality against a CPU reference on several seeds, and I counted SASS instructions with cuobjdump on the sm_86 build:

| | manual | ldmatrix |
| --- | --- | --- |
| total instructions | 280 | 216 |
| smem->fragment reads | 8 16-bit LDS + half2 packing | 2 LDSM (.x4 + .x2) |
| address arithmetic | 76 | 83 |

The mma section (1 HMMA.16816.F32) is identical in both, so the whole difference is the load segment: the manual path pays for the per-element reads and the packing around them, ldmatrix pays slightly more address math and buys back all the rest. Why a manual version can't win in general: the fragment layout scatters data across 32 lanes while smem holds it row-major (and B's k-adjacent pairs are not even contiguous in the k-major layout), so somebody has to do that reshuffle. ldmatrix is the instruction the hardware provides for exactly this.

`ldsm_stride.cu` measures bank conflicts as a function of row stride, with ncu counters (raw CSV kept next to the source). One ldmatrix.x4 moves 512B and smem serves 128B per wavefront, so 4 wavefronts is the floor:

| row stride | wavefronts/issue | conflicts/issue | cycles |
| --- | --- | --- | --- |
| 32B | 8 | 4 | 64 |
| 64B | 16 | 12 | 128 |
| 128B | 32 | 28 | 256 |
| 128B+16 pad | 4 | 0 | 32 |

wavefronts = 4 + conflicts, every single time, and cycles scale with wavefronts almost 1:1 here because 8 warps hammer the LSU back to back with nothing else to hide the stalls behind. In a real GEMM there are mma and address instructions between loads, so the effect would be much smaller.

## Descriptors and swizzle

`descriptor.cu` builds wgmma smem descriptors by hand: address>>4 in bits [0,14), LBO in [16,30), SBO in [32,46), layout in [61,64). For a 64x64 bf16 tile, K-major without swizzle gives LBO=128, SBO=1024; with 128B swizzle the LBO field is ignored so it's written 0. All three scenarios pass against descriptors checked on a B300.

Something that confused me at first: scenarios 2 (K-major) and 3 (MN-major) produce the same descriptor here. The direction difference isn't in the descriptor fields, it's in how the data was staged into smem in the first place. A square tile makes both directions' SBO come out 1024B, and swizzle ignores LBO, so they coincide. A non-square tile would separate them.

`swizzle.cu` implements all three patterns with one rule: XOR the 16B chunk index within the row with the row's low bits (3 bits for 128B rows, 2 for 64B, 1 for 32B):

```
128B: row*128 + (colByte ^ ((row & 7) << 4))
 64B: row*64  + (colByte ^ ((row & 3) << 4))
 32B: row*32  + (colByte ^ ((row & 1) << 4))
```

Verified as bijections with conflict-free column access. The 128B one is in the PTX ISA docs; the 64B/32B ones I worked out as the same pattern truncated, and the judge confirms them.

## Build and run

```bash
make run/fragments/first_mma
make run/ldmatrix/ldmatrix
make ptx/fragments/first_mma             # read the generated PTX
```

Everything here runs on any sm_80+ GPU; ARCH defaults to native (developed on sm_86).

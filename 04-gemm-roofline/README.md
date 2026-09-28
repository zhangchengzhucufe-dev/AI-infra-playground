# 04 - GEMM roofline: the thin-GEMM study: where cuBLAS stops using tensor cores

A measured roofline study of the shapes that actually matter for LLM inference: 7 decoder layers x 9 M values (M = batch x seq) = 63 GEMM shapes run through cuBLAS bf16, reporting tensor-core and bandwidth utilization against reference peaks.

Everything here runs on any GPU with cuBLAS; the numbers below are from my RTX 3060 laptop. On a datacenter card the absolute numbers move (bigger peaks) but the shape of the curve does not.

## Layout

```
thin-gemm/    thin_gemm.cu    the study: cuBLAS bf16, per-shape time / GB/s / %TC / %BW
common.h
Makefile
```

## The shapes

The 7 layers are real sizes from a vLLM decoder (q_b_proj 2304x1536, o_proj 7168x1536, fused_qkv_a 2112x7168, in_proj 6288x7168, dense_down 7168x8448, dense_gate_up 16896x7168, and f_b_proj 1536x128, the short-K outlier). M sweeps 1, 8, 16, 64, 256, 1024, 4096, 16384, 65536 - decode (M<=16) through prefill. The sweep needs ~3.6 GB VRAM and a few minutes.

%TC and %BW are against reference peaks passed on the command line; on the 3060 I used 25 TFLOPS / 300 GB/s (rows exceed 100% %TC, so the card's real bf16 peak is ~26.5):

| layer \ M | 1 | 8 | 16 | 64 | 256 | 1024 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f_b_proj (1536,128) | 0.0 | 0.3 | 1.6 | 6.8 | 22.1 | 60.4 | 83.2 |
| q_b_proj (2304,1536) | 0.9 | 7.6 | 15.5 | 53.3 | 79.0 | 91.9 | 101 |
| o_proj (7168,1536) | 1.0 | 7.8 | 15.8 | 51.6 | 99.3 | 106 | 106 |
| fused_qkv_a (2112,7168) | 1.2 | 7.5 | 15.0 | 59.7 | 89.8 | 96.4 | 100 |
| in_proj_qkvgfab (6288,7168) | 1.2 | 8.6 | 16.3 | 48.3 | 93.3 | 96.0 | 107 |
| dense_down (7168,8448) | 1.2 | 9.0 | 18.6 | 51.0 | 96.8 | 97.4 | 102 |
| dense_gate_up (16896,7168) | 1.1 | 9.1 | 16.9 | 67.9 | 85.5 | 93.1 | 98.9 |

(The full 63-row table prints M=16384 and M=65536 too; everything plateaus by M=1024.)

## What the table says

- **M=1 is weight traffic, nothing else**: every layer except f_b_proj runs at 78-104% of bandwidth with ~1% tensor-core utilization. Activations are a few KB, weights are several MB, and each weight byte feeds only M FMAs, so arithmetic intensity is ~M - nowhere near the machine balance point (234-281 FLOP/byte, derived in topic 03).
- **The crossover is M=16 to M=256**: TC utilization climbs through it and everything plateaus at 93-107% (cuBLAS's own ceiling) by M=1024.
- **f_b_proj fails both roofs**: at M=1 it moves 4.8 GB/s - the weights are 384 KB and the whole GEMM is so small that fixed costs (launch, tile scheduling) dominate; at large M it tops out ~85% because K=128 is two BK=64 k-steps, and prologue/epilogue never amortize. Its limiter is the shape itself plus fixed overhead, not either roof.
- **Why vLLM has a separate decode path**: this table is the quantitative case for it. At M<=16 the TC path pays TMA/tile setup for a kernel that is bandwidth-bound anyway; a skinny CUDA-core kernel skips that machinery and wins on latency.

## Run it

```bash
make bin/thin-gemm/thin_gemm
./bin/thin-gemm/thin_gemm 25 300     # args: peak TFLOPS, GB/s of your card
```

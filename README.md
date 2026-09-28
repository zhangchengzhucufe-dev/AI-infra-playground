# AI-infra-playground

My GPU programming practice code. I started from "how do I even write a kernel" and worked up to tensor core fragments, GEMM rooflines, and NVFP4 quantization. Everything here runs on what I have: an RTX 3060 Laptop with 6 GB VRAM under WSL2, CUDA 13.3. I kept the scope to what my hardware can actually execute, so every program in this repo produces real measured output on my machine.

## What's in here

| dir | what it is |
| --- | --- |
| [01-cuda-fundamentals](01-cuda-fundamentals/) | the basics: why GPUs are fast, warps and divergence, memory, timing. ~20 small programs |
| [02-triton-tilelang](02-triton-tilelang/) | the same kernels rewritten in Triton/TileLang, matmuls tuned to cuBLAS parity, and a lowering study (sm_90a vs sm_100a) |
| [03-mma-tensor-cores](03-mma-tensor-cores/) | sm_80 tensor cores: mma fragment layouts, ldmatrix, shared memory descriptors and swizzle |
| [04-gemm-roofline](04-gemm-roofline/) | a thin-GEMM study: 63 real LLM shapes through cuBLAS, finding exactly where tensor cores stop mattering |
| [05-low-precision](05-low-precision/) | why one outlier ruins int8 quantization, block scaling, and a full NVFP4 pipeline |

Topics 01 and 02 are the foundations. Topics 03 to 05 all come from one observation I kept running into: a single mma instruction only has about 3 FLOP/byte of arithmetic intensity, but the GPU can compute at 200+ FLOP/byte, so if you feed operands straight from DRAM the tensor core sits idle ~99% of the time. Almost everything after topic 02 is different ways of attacking that gap.

## Running things

Python tests (one venv at the repo root):

```bash
pip install -e .            # or uv sync
pytest                      # 27 tests
```

TileLang kernels need `pip install -e '.[tilelang]'`.

Each CUDA topic has its own Makefile:

```bash
cd 01-cuda-fundamentals && make run/simt/reduce
cd 03-mma-tensor-cores  && make run/ldmatrix/ldmatrix
cd 04-gemm-roofline     && make run/thin-gemm
```

CUDA programs print PASS/FAIL and exit nonzero on failure. Performance numbers never affect exit codes, and all timing numbers in the READMEs are from my 3060 laptop, so don't compare them against datasheets or other GPUs.

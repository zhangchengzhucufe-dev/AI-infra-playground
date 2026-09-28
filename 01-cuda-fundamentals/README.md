# 01 - CUDA fundamentals

The first programs I wrote when learning CUDA, kept as a series of small experiments. Machine: RTX 3060 Laptop (sm_86). Every program checks its result against a CPU reference and prints PASS/FAIL, so they double as regression tests.

## What's here

```
basics/          hello.cu, device_query.cu, scaling.cu
first-kernels/   vector_add, matrix_add, unified memory, grid stride, SAXPY + judge
simt/            divergence, sync_matters, reduce (3 versions), simt_sim.py
memory/          stencil, constant memory, 2 histograms, bandwidth, occupancy
timing/          timing_trap.cu
common.h         error checking, cudaEvent timer, result diff
```

## Things I measured and what they taught me

**One GPU thread is slow.** In `basics/scaling.cu`, one GPU thread takes ~20x longer per element than one CPU thread. The CPU has caches, branch prediction, out-of-order execution; the GPU thread is just a dumb lane. Then I ran the same job with 16384 blocks x 256 threads and it was ~900x faster than a single block. So the speed was never about fast cores, it was about having enough threads in flight to hide memory latency. A GPU with one block is basically a slow little CPU.

**Divergence means both sides run.** `simt/divergence.cu`: putting the branch *inside* warps (`tid % 2`) takes about 2x the time of the same branch split *between* warps, because in the first case every warp executes both paths. I also wrote `simt/simt_sim.py`, a small CPU simulator of this execution model (5 tests), which made the "run both sides under a mask" idea concrete.

**Same FLOPs, different speed.** The two block reductions in `simt/reduce.cu` do the exact same number of adds. The only difference is where the active threads sit in each warp: spread out (interleaved) vs packed at the low end (contiguous). Measured 1.5x between them on my card, and a warp-shuffle version that skips shared memory was faster still. This one surprised me: algorithm-class Big-O says they're identical.

**Atomic contention is the whole cost.** `memory/histogram.cu` vs `memory/histogram_priv.cu`: 16M atomicAdd calls straight to global memory took ~5.9 ms. Giving each block a shared-memory histogram and doing only 256 global atomics per block took ~0.14 ms. That's ~40x from restructuring where the contention happens, not from doing less work.

**Bandwidth needs coalescing and enough warps.** `memory/bandwidth.cu`: stride-1 reads hit ~250 GB/s; the number halves for each stride doubling and flattens around 56 GB/s once every thread's address lands in a different 128B transaction. `memory/occupancy.cu`: dropping occupancy from 100% to 83% or 50% costs almost nothing, but 16.7% (8 warps/SM) cuts bandwidth hard, because there aren't enough outstanding loads to cover DRAM latency (Little's law, but measured).

**Timing is easy to get wrong.** `timing/timing_trap.cu` times one kernel three ways: host clock without sync (~0.04 ms, that's just launch overhead), host clock with sync (~1 ms, includes host wake-up), cudaEvent (~0.5 ms, the number that actually belongs in a report).

## SAXPY from scratch

`first-kernels/saxpy.cu` is written without common.h: my own error-check macros and cudaEvent timing, plus an n=0 special case (a 0-block launch is an error in CUDA). `judge_saxpy.sh` builds it and checks 7 sizes (0, 1, 31, 1024, 1025, 1048576, 1048579) against sums computed independently on the CPU: 7/7 pass.

One thing I noticed while writing the judge: every value the formula generates is an integer or half-integer, which float represents exactly, so the sums have no rounding at all and can be compared as integers.

## Run it

```bash
make run/simt/reduce                # any single program
bash first-kernels/judge_saxpy.sh first-kernels/saxpy.cu
cd .. && pytest 01-cuda-fundamentals/tests    # simt_sim tests
```

`ARCH=sm_80 make ...` if building on a node without a GPU.

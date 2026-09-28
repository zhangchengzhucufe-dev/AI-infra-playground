"""TileLang tiled matmul: C = A @ B, fp16 in, fp32 accumulate.

The five load-bearing pieces: the two shared tiles, the fragment
accumulator, the pipelined loop over K, the T.copy tile loads, and T.gemm.
Explicit where Triton is implicit -- you place the shared/fragment levels
and pick the stage count yourself.

Needs a GPU and tilelang (pip install -e '.[tilelang]'):

    pytest tests/test_tilelang.py -k matmul

Tuning sweep over configs (results in this topic's README; on the RTX 3060
with TileLang 0.1.14 the best is 128,128,32,128,3 at ~26 TFLOPS, cuBLAS
parity; the 128x256 config compiles but fails at first launch -- it wants
144 KB dynamic smem, this card caps at 100 KB):

    python -c "from matmul_ladder.tilelang import bench; bench()"
"""

import tilelang
import tilelang.language as T


def make_matmul(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                threads=128, num_stages=3,
                dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, BLOCK_N),
            T.ceildiv(M, BLOCK_M),
            threads=threads,
        ) as (bx, by):
            # shared tiles: A (M-tile x K-step), B (K-step x N-tile)
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)

            # accumulator lives in registers (fragment), fp32 for precision
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

            T.clear(C_local)

            # software pipeline along K
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                # stage the current A/B tiles into shared
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
                # tile-level multiply-accumulate
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return main


def bench(M=2048, N=2048, K=2048):
    import torch
    import triton

    assert torch.cuda.is_available(), "benchmark requires a GPU"
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # (block_M, block_N, block_K, threads, num_stages)
    configs = [
        (64, 64, 32, 128, 1),
        (64, 64, 32, 128, 3),
        (128, 128, 32, 128, 3),
        (128, 128, 64, 256, 3),
        (128, 256, 64, 256, 3),
    ]

    checked = False
    for bm, bn, bk, threads, stages in configs:
        # the whole per-config work sits inside the try: failures can come
        # from compile (config doesn't lower) or from the first launch
        # (dynamic smem over the card's limit -- tilelang only sets that at
        # run time). Either way, skip the config instead of aborting the sweep.
        try:
            kernel = tilelang.compile(
                make_matmul(M, N, K, bm, bn, bk, threads=threads, num_stages=stages),
                out_idx=[2],
            )
            if not checked:  # first successful config is checked once against torch
                ref = a.float() @ b.float()
                torch.testing.assert_close(kernel(a, b), ref, rtol=1e-2, atol=1e-1)
                checked = True
            ms = triton.testing.do_bench(lambda: kernel(a, b))
        except Exception as e:
            print(f"block_M={bm:4d} block_N={bn:4d} block_K={bk:3d} "
                  f"threads={threads:3d} stages={stages}  skipped ({str(e)[:80]})")
            continue
        tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
        print(f"block_M={bm:4d} block_N={bn:4d} block_K={bk:3d} "
              f"threads={threads:3d} stages={stages}  {ms:8.3f} ms  {tflops:6.1f} TFLOPS")

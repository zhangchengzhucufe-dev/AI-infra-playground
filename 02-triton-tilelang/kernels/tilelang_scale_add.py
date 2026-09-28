"""TileLang scale-add: Y = 2 * X + 1 on an (M, N) tile.

The two lines that matter map to TileLang's two basic abstractions: the 2D
CTA grid (T.ceildiv per axis) and per-block parallel tile traversal
(T.Parallel). Needs a GPU and tilelang (pip install -e '.[tilelang]'):

    pytest tests/test_tilelang.py -k scale_add
"""

import tilelang
import tilelang.language as T


def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale_add(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # 2D CTA grid: block count per axis from tile size
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            # visit every element of the tile in parallel within the block
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if gi < M and gj < N:
                    Y[gi, gj] = X[gi, gj] * 2.0 + 1.0

    return scale_add

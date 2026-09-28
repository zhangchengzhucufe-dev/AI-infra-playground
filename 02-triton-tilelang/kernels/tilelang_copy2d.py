"""TileLang 2D scaled copy through shared memory: Y = 2 * X on an (M, N) tile.

Same grid as tilelang_scale_add, but the tile now stages through shared
memory -- and every data movement is one T.copy call, which absorbs the
bounds handling for ragged tails (M/N need not divide the tile size).
Compare with a hand-written CUDA 2D kernel: the row/col tile origin and the
grid math are still yours; the per-element index math and bounds guard are
the compiler's.

Needs a GPU and tilelang (pip install -e '.[tilelang]'):

    pytest tests/test_tilelang.py -k copy2d
"""

import tilelang
import tilelang.language as T


def make_scale2d(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale2d(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # 2D CTA grid: x covers the N columns, y the M rows
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)

            # stage the current tile into shared; T.copy handles the tail
            T.copy(X[by * block_M, bx * block_N], X_shared)

            for i, j in T.Parallel(block_M, block_N):
                X_shared[i, j] = X_shared[i, j] * 2.0

            # store the computed tile back at the same position in Y
            T.copy(X_shared, Y[by * block_M, bx * block_N])

    return scale2d

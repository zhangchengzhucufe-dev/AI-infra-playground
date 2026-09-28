"""Row softmax in TileLang, written from scratch.

Design:
- softmax(x) takes a float32 CUDA tensor of shape (M, N) and returns the
  same-shape result, softmaxed independently per row; one block per row.
- Numerically safe: subtract the row max before exp (the tests include rows
  with huge values; an unstable implementation produces inf/nan).
- Row width N is arbitrary (assumed <= 4096). TileLang compiles per shape,
  so make_softmax(M, N) generates the kernel and the wrapper caches it by
  shape -- the standard TileLang pattern.
- Fragment width is the next power of 2 >= N, tail padded with -inf
  (T.if_then_else + T.infinity); otherwise layout inference can fail with
  "no available layout".
- Reductions are explicit here: T.reduce_max / T.reduce_sum.

    pytest tests/test_tilelang_softmax.py

Benchmarked against torch.softmax in this topic's README: bandwidth-bound,
at parity (183-291 GB/s depending on N).
"""

import torch
import tilelang
import tilelang.language as T

_kernel_cache = {}


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def make_softmax(M, N, dtype="float32", threads=128):
    # fragment width = next pow2 >= N; pad the tail with -inf.
    BLOCK_N = _next_pow2(max(N, 1))

    @T.prim_func
    def softmax_kernel(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(M, threads=threads) as (m):
            row = T.alloc_fragment((BLOCK_N,), dtype)
            row_max = T.alloc_fragment((1,), dtype)
            row_sum = T.alloc_fragment((1,), dtype)

            # load: pad j >= N with -inf (clamp the global index at N-1 to
            # stay in bounds; the value is overridden by -inf anyway).
            for j in T.Parallel(BLOCK_N):
                row[j] = T.if_then_else(
                    j < N, X[m, T.min(j, N - 1)], -T.infinity(dtype))

            # numerical stability: subtract the row max first, so exp's
            # argument is at most 0 and cannot overflow.
            T.reduce_max(row, row_max, dim=0, clear=True)
            for j in T.Parallel(BLOCK_N):
                row[j] = T.exp(row[j] - row_max[0])
            T.reduce_sum(row, row_sum, dim=0, clear=True)

            # store: only the first N positions.
            for j in T.Parallel(N):
                Y[m, j] = row[j] / row_sum[0]

    return softmax_kernel


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    key = (M, N, x.dtype)
    if key not in _kernel_cache:
        _kernel_cache[key] = tilelang.compile(make_softmax(M, N, dtype=str(x.dtype).removeprefix('torch.')))
    y = torch.empty_like(x)
    _kernel_cache[key](x, y)
    return y

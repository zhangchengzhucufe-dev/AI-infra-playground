"""Row softmax in Triton: one program per row, arbitrary row width via mask.

Runs fine without a GPU (conftest.py switches to interpreter mode
automatically).

Design:
- softmax(x) takes a 2D tensor of shape (M, N) and returns the same shape,
  softmaxed independently per row;
- numerically safe: subtract the row max before exp and sum (a test input
  has huge values; an unstable implementation gets inf/nan);
- BLOCK_SIZE = triton.next_power_of_2(N); out-of-range lanes load -inf,
  which cannot skew the max and contributes exp(-inf) = 0.

    pytest tests/test_softmax.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_row, n, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    # Fill out-of-bounds positions with -inf: they can't skew the max below,
    # and exp(-inf) happens to be 0.
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask,
                other=-float("inf"))

    # The key to numerical stability: subtract the row max first, so the
    # largest exp argument is 0 and nothing overflows.
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(y_ptr + row * stride_row + offs, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    # the kernel indexes rows by stride and columns by 1, so it needs
    # contiguous rows; a transposed/strided view would silently read
    # the wrong elements
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y

"""Triton vector addition -- the four load-bearing lines of any Triton kernel:
program id, the global index block it covers, the bounds mask, the masked store.

    pytest tests/test_vector_add.py
Runs without a GPU too -- conftest.py switches to interpreter mode automatically.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    # index of this program in the 1-D grid
    pid = tl.program_id(0)
    # the block of global indices this program covers (BLOCK_SIZE long)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask that screens out out-of-bounds positions
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # write x + y back to z, respecting the mask
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # the kernel addresses memory flat, so it needs one contiguous layout;
    # a transposed/strided view would silently compute on wrong addresses
    x = x.contiguous()
    y = y.contiguous()
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z

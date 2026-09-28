"""Fused elementwise kernel: z = relu(a * x + b).

scale_kernel is the reference: identical skeleton (program id, offsets, mask,
masked load/store), only the compute line and the parameter list differ.
That's the payoff of the tile view -- swapping the elementwise formula means
swapping one expression, not rewriting the data movement.

    pytest tests/test_fused_op.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = x * 2.0
    tl.store(z_ptr + offsets, z, mask=mask)


def scale(x: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scale_kernel[grid](x, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z


# same skeleton as scale_kernel; only the compute line and parameters change

@triton.jit
def fused_kernel(x_ptr, z_ptr, n, a, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = tl.maximum(a * x + b, 0.0)  # relu(a * x + b)
    tl.store(z_ptr + offsets, z, mask=mask)


def fused(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    fused_kernel[grid](x, z, n, a, b, BLOCK_SIZE=BLOCK_SIZE)
    return z

"""Where the block scale multiplies back: two valid placements and one
deliberately wrong control.

fp64 only -- this simulates where the scale sits algebraically, not
narrow-precision rounding.

gemm_scale_per_row_col:
    One scale per row of A, one per row of B (i.e. per output column of
    the GEMM). The scale product is constant across the whole K
    reduction, so it can be multiplied back once, after the full dot
    product.

gemm_scale_along_k:
    The scale changes every SEG elements along K. Each K block first
    computes its partial sum in quantized view, multiplies that
    segment's sA*sB back in, then accumulates into the output.

gemm_scale_along_k_one_restore:
    The wrong control. It adds the normalized partial sums of the
    different K blocks first and multiplies back only the first
    segment's scale product at the end. Because the scale product
    changes from K block to K block, that factor cannot be pulled out of
    the full K sum, so the result must differ from the reference.

Tests: pytest tests/test_block_scale.py
"""

import torch

SEG = 128


def gemm_fp64(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A.double() @ B.double().T


def gemm_scale_per_row_col(A: torch.Tensor, B: torch.Tensor,
                           sA: torch.Tensor, sB: torch.Tensor) -> torch.Tensor:
    """sA: [M], sB: [N], both positive.

    The row/column scales are constant over the whole dot product and
    factor out of the sum:
    sum_k (a_k/sA)(b_k/sB) * sA*sB = sum_k a_k b_k,
    so take the full dot product of the normalized inputs, then multiply
    sA x sB back into [M, N] once.
    """
    qA = A.double() / sA[:, None]
    qB = B.double() / sB[:, None]
    return (qA @ qB.T) * sA[:, None] * sB[None, :]


def gemm_scale_along_k(A: torch.Tensor, B: torch.Tensor,
                       sA: torch.Tensor, sB: torch.Tensor) -> torch.Tensor:
    """sA: [M, K//SEG], sB: [N, K//SEG], both positive.

    The scale product changes per K segment and does not factor out of
    the whole sum --
    sum_k x_k * c_k != (sum_k x_k) * c_anything.
    So each K block's normalized partial sum must be multiplied by that
    segment's sA*sB before it is accumulated.
    """
    M, K = A.shape
    assert K % SEG == 0, "scale groups must tile K exactly; a ragged tail would be silently dropped"
    N = B.shape[0]
    out = torch.zeros((M, N), dtype=torch.float64)
    for block in range(K // SEG):
        sl = slice(block * SEG, (block + 1) * SEG)
        qA = A[:, sl].double() / sA[:, block, None]
        qB = B[:, sl].double() / sB[:, block, None]
        out += (qA @ qB.T) * sA[:, block, None] * sB[None, :, block]
    return out


def gemm_scale_along_k_one_restore(A: torch.Tensor, B: torch.Tensor,
                                   sA: torch.Tensor,
                                   sB: torch.Tensor) -> torch.Tensor:
    """Deliberately wrong control: multiplies back only the first segment's scale, after the whole K reduction."""
    M, K = A.shape
    assert K % SEG == 0, "scale groups must tile K exactly; a ragged tail would be silently dropped"
    N = B.shape[0]
    normalized_sum = torch.zeros((M, N), dtype=torch.float64)
    for block in range(K // SEG):
        sl = slice(block * SEG, (block + 1) * SEG)
        qA = A[:, sl].double() / sA[:, block, None]
        qB = B[:, sl].double() / sB[:, block, None]
        normalized_sum += qA @ qB.T
    return normalized_sum * sA[:, 0, None] * sB[None, :, 0]

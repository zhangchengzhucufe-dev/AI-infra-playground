"""Unit tests for the block-scale placement simulation in
experiments/block_scale_sim.py (fp64)."""

import torch

from experiments.block_scale_sim import (
    SEG,
    gemm_fp64,
    gemm_scale_along_k,
    gemm_scale_along_k_one_restore,
    gemm_scale_per_row_col,
)


def _data(m=7, n=5, k=512):
    g = torch.Generator().manual_seed(3)
    A = torch.randn(m, k, generator=g, dtype=torch.float64)
    B = torch.randn(n, k, generator=g, dtype=torch.float64)
    sA_k = torch.rand(m, k // SEG, generator=g, dtype=torch.float64) + 0.5
    sB_k = torch.rand(n, k // SEG, generator=g, dtype=torch.float64) + 0.5
    sA_row = torch.rand(m, generator=g, dtype=torch.float64) + 0.5
    sB_col = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    return A, B, sA_k, sB_k, sA_row, sB_col


def test_per_row_col_restores_once():
    A, B, _, _, sA, sB = _data()
    ref = gemm_fp64(A, B)
    got = gemm_scale_per_row_col(A, B, sA, sB)
    torch.testing.assert_close(got, ref, rtol=2e-13, atol=2e-13)


def test_along_k_restores_each_partial_sum():
    A, B, sA, sB, _, _ = _data()
    ref = gemm_fp64(A, B)
    got = gemm_scale_along_k(A, B, sA, sB)
    # segmenting changes fp64 summation grouping, so check algebraic
    # equivalence, not bit-exactness
    torch.testing.assert_close(got, ref, rtol=2e-13, atol=2e-13)


def test_along_k_cannot_restore_only_once():
    A, B, sA, sB, _, _ = _data()
    ref = gemm_fp64(A, B)
    got = gemm_scale_along_k_one_restore(A, B, sA, sB)
    assert (got - ref).abs().max().item() > 1e-3

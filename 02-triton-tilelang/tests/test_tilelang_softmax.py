import pytest
import torch

pytest.importorskip("tilelang", reason="tilelang not installed (uv sync --extra tilelang)")

if not torch.cuda.is_available():
    pytest.skip("TileLang problems need a GPU; run on the cluster", allow_module_level=True)

from kernels.tilelang_softmax import softmax


def test_basic():
    torch.manual_seed(0)
    x = torch.randn(33, 127, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_wide_rows():
    torch.manual_seed(1)
    x = torch.randn(8, 1000, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_numerical_stability():
    # rows with huge values; without subtracting the max first,
    # exp overflows to inf/nan.
    torch.manual_seed(2)
    x = torch.randn(4, 256, device="cuda") * 1000.0
    got = softmax(x)
    assert torch.isfinite(got).all(), "inf/nan produced, subtract the row max before exp"
    torch.testing.assert_close(got, torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_single_element_rows():
    x = torch.randn(5, 1, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1),
                               atol=1e-5, rtol=1e-5)

import torch

from matmul_ladder.triton import matmul


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# fp16 inputs: on GPU, tl.dot with fp32 inputs defaults to TF32
# (10-bit mantissa), which won't match the CPU reference;
# fp16 inputs + fp32 accumulation behave the same on both backends.
def _check(M, K, N, seed):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device=_device(), dtype=torch.float16)
    b = torch.randn(K, N, device=_device(), dtype=torch.float16)
    got = matmul(a, b, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16)
    ref = a.float() @ b.float()
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)


def test_matmul_small():
    _check(64, 48, 32, seed=0)


def test_matmul_ragged():
    _check(100, 70, 55, seed=1)

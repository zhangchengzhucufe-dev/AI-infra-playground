"""Outlier poisoning study: one 3000-magnitude outlier in a tensor
quantized with a single global scale.

Ten thousand elements uniform on [-1, 1] plus one outlier of 3000,
quantized per-tensor to E4M3 (scale = amax / 448, cast via
torch.float8_e4m3fn), dequantized, then pointwise relative error. Three
comparisons: with vs without the outlier, the threshold below which
values quantize to zero, and how much per-block (1x128) scaling buys
back.

Run: python kernels/quant_outlier.py
"""

import torch

E4M3_MAX = 448.0


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor E4M3 quantize, then dequantize."""
    amax = x.abs().max()
    scale = amax / E4M3_MAX
    q = (x / scale).to(torch.float8_e4m3fn)
    return q.float() * scale


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """Find the element of x closest to value, return its relative error."""
    i = (x - value).abs().argmin()
    return ((y[i] - x[i]) / x[i]).abs().item()


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    print("with outlier:")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        print(f"  x~{v:<8} rel_err={rel_err_at(x, y, v):.3e}")

    # Re-quantize without the outlier, compare the error at 0.5
    x_no = build_tensor()[:-1]
    y_no = quant_dequant_per_tensor(x_no)
    e_with = rel_err_at(x, y, 0.5)
    e_without = rel_err_at(x_no, y_no, 0.5)
    print(f"\n(a) rel err at 0.5: with outlier {e_with:.3e}, "
          f"without {e_without:.3e}, {e_with / e_without:.1f}x change")

    # Threshold for quantizing to zero: E4M3's smallest nonzero magnitude
    #     is the subnormal 2^-9, so values / scale below half of that step
    #     (2^-10) round to 0.
    scale = x.abs().max() / E4M3_MAX
    small = torch.tensor([1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2])
    for v in small:
        q = (v / scale).to(torch.float8_e4m3fn)
        print(f"  (b) x={v:.4f} -> quantized {q.float().item() * scale.item():.3e}"
              f" ({'zero' if q.float().item() == 0 else 'nonzero'})")

    # 1x128 per-block scale: the 3000 pushes up the scale of the block
    #     holding it; every other block is unaffected.
    xu = x[:-1]                    # the ten thousand uniform elements
    base = xu[: (len(xu) // 128) * 128]  # trim to a multiple of 128, 78 blocks
    blocks = base.view(-1, 128)
    s = blocks.abs().amax(dim=1, keepdim=True) / E4M3_MAX
    qb = (blocks / s).to(torch.float8_e4m3fn).float() * s
    for v in (0.5, 0.01):
        i = int((base - v).abs().argmin())
        b = i // 128
        err_blk = ((qb[b, i % 128] - base[i]) / base[i]).abs().item()
        print(f"  (c) block without outlier, x~{v}: "
              f"per-tensor {rel_err_at(x, y, v):.3e}, "
              f"per-block {err_blk:.3e}")
    # block with the outlier: stuff 3000 into one 128-element block and
    # quantize; its block-mates fall back to per-tensor accuracy.
    blk = torch.cat([torch.tensor([3000.0]), xu[:127]])
    s2 = blk.abs().max() / E4M3_MAX
    q2 = (blk / s2).to(torch.float8_e4m3fn).float() * s2
    for off in (1, 60):
        err2 = ((q2[off] - blk[off]) / blk[off]).abs().item()
        print(f"  (c) block with outlier, x~{blk[off].item():.3f}: "
              f"err {err2:.3e}")


if __name__ == "__main__":
    main()

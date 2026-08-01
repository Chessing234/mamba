"""CPU tests for TileLang contiguity helper used by Mamba3 MIMO (#985)."""

import torch

from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import ensure_tilelang_contiguous


def test_ensure_tilelang_contiguous_none():
    assert ensure_tilelang_contiguous(None) is None


def test_ensure_tilelang_contiguous_already_ok():
    t = torch.randn(2, 3, 4)
    out = ensure_tilelang_contiguous(t)
    assert out is not None
    assert out.stride(-1) == 1
    assert out.shape == t.shape


def test_ensure_tilelang_contiguous_fixes_size1_trailing_stride():
    # Transpose so the last dim has size 1 but stride(-1) != 1 after .contiguous()'s
    # false-positive "already contiguous" for size-1 dims — construct explicitly.
    base = torch.randn(2, 4, 1)
    weird = base.transpose(1, 2)  # (2, 1, 4) — last stride is fine
    # Build (B, D, 1) view whose last stride is not 1:
    src = torch.randn(2, 8)
    view = src[:, :1]  # shape (2, 1), often stride (8, 1) which is ok
    # Force non-unit last stride: as_strided
    t = torch.as_strided(src, size=(2, 3, 1), stride=(8, 1, 2))
    assert t.shape[-1] == 1
    # .contiguous() may still leave stride(-1) != 1 for size-1 trailing dims
    c = t.contiguous()
    # Regardless of contiguous() quirks, helper must guarantee stride(-1)==1
    out = ensure_tilelang_contiguous(t)
    assert out is not None
    assert out.shape == t.shape
    assert out.stride(-1) == 1
    assert torch.allclose(out, t)

"""
Mamba-3 decode-path shape contract tests.

forward() hands the mixer's (B, 1, D) decode slice straight to step(), while step() was
written against (B, D). These tests pin the 3-D-tolerant entry / rank-preserving exit
behaviour and check that decoding one token still matches the last position of a full
sequence forward.

Run: pytest tests/modules/test_mamba3_step_shapes.py
"""
import pytest
import torch

from mamba_ssm.modules import mamba3 as mamba3_module
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.utils.generation import InferenceParams

try:
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
except ImportError:
    mamba3_step_fn = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_decode():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if mamba3_step_fn is None:
        pytest.skip("CuteDSL Mamba-3 step kernel not available (needs nvidia-cutlass-dsl + quack-kernels)")


def _require_mimo():
    if mamba3_module.mamba3_mimo_combined is None:
        pytest.skip("TileLang MIMO kernels not available")


def _make_model(is_mimo: bool, dtype=torch.float32, **kwargs) -> Mamba3:
    defaults = dict(
        d_model=128,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        is_mimo=is_mimo,
        mimo_rank=2,
        chunk_size=16,
        layer_idx=0,
        device="cuda",
        dtype=dtype,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return Mamba3(**defaults).eval()


def _inference_params(batch: int, max_seqlen: int) -> InferenceParams:
    return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch)


def _prefill(model: Mamba3, u: torch.Tensor, inference_params: InferenceParams) -> torch.Tensor:
    """Run the prefill and advance the offset so the next call takes the decode branch."""
    out = model(u, inference_params=inference_params)
    inference_params.seqlen_offset += u.shape[1]
    return out


def _input(batch: int, seqlen: int, model: Mamba3, dtype=torch.float32) -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(batch, seqlen, model.d_model, device="cuda", dtype=dtype)


# ---------------------------------------------------------------------------
# (a) forward()'s decode branch survives a (B, 1, D) slice and returns (B, 1, D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_mimo", [False, True])
def test_forward_decode_branch_accepts_3d(is_mimo):
    _require_decode()
    if is_mimo:
        _require_mimo()
    batch, seqlen = 2, 8
    model = _make_model(is_mimo)
    u = _input(batch, seqlen, model)
    inference_params = _inference_params(batch, seqlen)

    with torch.no_grad():
        _prefill(model, u[:, :-1], inference_params)
        assert inference_params.seqlen_offset > 0, "test would not exercise the decode branch"
        out = model(u[:, -1:], inference_params=inference_params)

    # Block.forward adds this straight onto a (B, 1, D) residual, so the rank must survive.
    assert out.shape == (batch, 1, model.d_model)
    assert out.dtype == u.dtype
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# (b) calling step() directly with 2-D input keeps its old behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_mimo", [False, True])
def test_step_2d_call_unchanged(is_mimo):
    _require_decode()
    if is_mimo:
        _require_mimo()
    batch, seqlen = 2, 8
    model = _make_model(is_mimo)
    u = _input(batch, seqlen, model)

    # Two independent caches primed with the identical prefill, so the only difference
    # between the two step() calls below is the rank of the token passed in.
    states = []
    for _ in range(2):
        inference_params = _inference_params(batch, seqlen)
        with torch.no_grad():
            _prefill(model, u[:, :-1], inference_params)
        states.append(model._get_states_from_cache(inference_params, batch))

    with torch.no_grad():
        out_2d, *rest_2d = model.step(u[:, -1], *states[0])
        out_3d, *rest_3d = model.step(u[:, -1:], *states[1])

    assert out_2d.shape == (batch, model.d_model), "2-D callers must still get (B, D) back"
    assert out_3d.shape == (batch, 1, model.d_model)
    assert torch.allclose(out_2d, out_3d.squeeze(1), atol=0.0, rtol=0.0), (
        "squeezing the seqlen dim must not change the computation"
    )
    for s_2d, s_3d in zip(rest_2d, rest_3d):
        assert torch.equal(s_2d, s_3d), "state updates must be identical for both entry ranks"


# ---------------------------------------------------------------------------
# (c) decode output == last token of the full-sequence forward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_mimo", [False, True])
def test_decode_matches_full_forward(is_mimo):
    _require_decode()
    if is_mimo:
        _require_mimo()
    batch, seqlen = 2, 8
    # fp32 model, but the prefill kernels (Triton for SISO, TileLang for MIMO)
    # accumulate fp32 GEMMs via TF32 tensor cores, while the CuTe step kernel runs
    # true fp32, so the two legitimately diverge at ~1e-2 absolute on outputs of
    # magnitude ~2-5. 1e-5 is unattainable here.
    atol, rtol = 2e-2, 2e-2
    model = _make_model(is_mimo, dtype=torch.float32)
    u = _input(batch, seqlen, model)

    with torch.no_grad():
        out_full = model(u)  # no inference_params -> pure sequence path
        inference_params = _inference_params(batch, seqlen)
        _prefill(model, u[:, :-1], inference_params)
        out_decode = model(u[:, -1:], inference_params=inference_params)

    assert out_decode.shape == out_full[:, -1:].shape
    torch.testing.assert_close(out_decode, out_full[:, -1:], atol=atol, rtol=rtol)

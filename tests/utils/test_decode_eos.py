"""EOS stop conditions for decode() without requiring a real Mamba model."""
from types import SimpleNamespace

import torch

from mamba_ssm.utils.generation import decode


class _TinyLM(torch.nn.Module):
    """Deterministic stand-in: always emits the next id from a fixed sequence."""

    def __init__(self, next_ids):
        super().__init__()
        self.next_ids = list(next_ids)
        self._i = 0

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=1):
        vocab = 32
        logits = torch.full((input_ids.shape[0], vocab), -1e9)
        tok = self.next_ids[min(self._i, len(self.next_ids) - 1)]
        logits[:, tok] = 0.0
        self._i += 1
        return SimpleNamespace(logits=logits.unsqueeze(1))


def test_decode_stops_on_list_eos():
    model = _TinyLM([5, 7, 9, 11])
    out = decode(
        torch.tensor([[1, 2]]),
        model,
        max_length=10,
        top_k=1,
        eos_token_id=[7, 99],
    )
    # prompt [1,2] + sampled 5 + sampled 7 (EOS) -> stop before 9
    assert out.sequences.tolist() == [[1, 2, 5, 7]]


def test_decode_stops_on_int_eos():
    model = _TinyLM([3, 4, 5])
    out = decode(
        torch.tensor([[0]]),
        model,
        max_length=10,
        top_k=1,
        eos_token_id=4,
    )
    assert out.sequences.tolist() == [[0, 3, 4]]

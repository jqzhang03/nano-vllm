"""Correctness coverage for segmented, packed-INT4 MoE expert GEMMs."""
from __future__ import annotations

import pytest
import torch

from nanovllm.layers.moe import MoE


def _quantized_moe(*, top_k: int = 2) -> MoE:
    torch.manual_seed(23)
    moe = MoE(128, 256, num_experts=8, top_k=top_k).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for param in moe.parameters():
            param.uniform_(-0.05, 0.05)
    for expert in moe.experts:
        for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
            linear.quantize_int4(dense_path=False, group_size=64)
    return moe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("top_k", [2, 6])
@pytest.mark.parametrize("fuse_gate_up", [False, True])
def test_segmented_int4_matches_per_expert_loop(top_k: int, fuse_gate_up: bool):
    moe = _quantized_moe(top_k=top_k)
    x = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = moe._forward_loop(x)
        moe.int4_backend = "segment"
        moe.fuse_int4_gate_up = fuse_gate_up
        actual = moe(x)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_segmented_int4_handles_empty_expert_segments():
    moe = _quantized_moe(top_k=1)
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0].fill_(1.0 / moe.hidden_size)
    x = torch.ones(7, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = moe._forward_loop(x)
        moe.int4_backend = "segment"
        actual = moe(x)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert torch.equal(moe._route(x)[1], torch.zeros(7, 1, device="cuda", dtype=torch.long))


def test_moe_rejects_unknown_int4_backend():
    with pytest.raises(ValueError, match="unknown MoE INT4 backend"):
        MoE(16, 32, num_experts=2, top_k=1, int4_backend="unknown")

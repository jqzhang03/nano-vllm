"""loader._resolve_tensor 的点分段匹配单测（阶段 1.5：MoE 3D 权重免疫 dense key 子串）。"""
import pytest
import torch
from torch import nn

from nanovllm.utils.loader import _resolve_tensor

MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


def test_dense_mapping_segment_style():
    """dense 权重名（末段 = key）照旧解析。"""
    assert _resolve_tensor(None, MAPPING, "model.layers.3.mlp.gate_proj.weight") == \
        ("model.layers.3.mlp.gate_up_proj.weight", 0)
    assert _resolve_tensor(None, MAPPING, "model.layers.3.self_attn.q_proj.weight") == \
        ("model.layers.3.self_attn.qkv_proj.weight", "q")


def test_moe_3d_names_pass_through():
    """MoE 3D 权重名（末段 gate_up_proj/down_proj）不被 dense 的 up_proj/gate_proj 误匹配。

    旧实现是任意子串 replace："up_proj" in "experts.gate_up_proj" 为 True →
    会错误替换成 gate_gate_up_proj。段匹配后 gate_up_proj 与 key up_proj 不相等 → 直通。
    """
    for name in ["model.layers.1.mlp.experts.gate_up_proj.weight",
                 "model.layers.1.mlp.experts.down_proj.weight",
                 "model.layers.1.mlp.gate.weight"]:
        assert _resolve_tensor(None, MAPPING, name) == (name, None), name


def test_non_weight_suffix_untouched():
    """非 weight/bias 结尾的名字（如 buffer）不改动。"""
    name = "model.layers.0.self_attn.rotary_emb.inv_freq"
    assert _resolve_tensor(None, MAPPING, name) == (name, None)

"""MoE 层 CPU 单测：结构 + 位级对照 + 量化豁免标记（无需 GPU）。

对照用例与 benchmarks/_moe_check.py 同源（forward 的 gather/index_add 组织 vs
reference 的全行掩码组织——数学同构、e 升序同序累加 → fp32 CPU 上位级一致）。
"""
import torch

from nanovllm.layers.moe import MoE


def _make(H=64, I=128, E=8, k=2, seed=0):
    torch.manual_seed(seed)
    moe = MoE(H, I, num_experts=E, top_k=k)
    with torch.no_grad():
        for p in moe.parameters():
            p.uniform_(-0.05, 0.05)
    return moe


def _bitclose(a, b, atol=1e-6, rtol=1e-4):
    """位级或仅 GEMM 行数相关的 K-归约尾差（~1e-9 fp32；记账错为 O(1) 级）。

    GPU fp16 下多数形状是 0 diff（benchmarks/_moe_check.py）；CPU/异行数时
    matmul kernel 的 K 归约顺序可差 1-2 ulp。
    """
    return torch.allclose(a, b, atol=atol, rtol=rtol)


def test_forward_bit_exact_reference():
    moe = _make()
    x = torch.randn(37, 64)  # 非 k 倍数 → 存在空专家
    with torch.no_grad():
        y_fwd = moe(x)
        y_ref = moe.reference(x)
    assert _bitclose(y_fwd, y_ref), "forward vs reference must agree to rounding"


def test_route_shapes_and_bounds():
    moe = _make()
    x = torch.randn(10, 64)
    vals, idx = moe._route(x)
    assert vals.shape == (10, 2) and idx.shape == (10, 2)
    assert idx.max().item() < 8 and idx.min().item() >= 0
    assert torch.all(vals >= 0) and torch.all(vals <= 1)
    # topk 概率（无 norm 时）= softmax 子集
    logits = moe.gate(x).float()
    probs = logits.softmax(-1)
    p_ref = probs.gather(1, idx)
    assert torch.allclose(vals.float(), p_ref, atol=1e-6)


def test_norm_topk_prob():
    moe = _make()
    moe.norm_topk_prob = True
    x = torch.randn(10, 64)
    vals, idx = moe._route(x)
    # 归一化后每行和为 1
    assert torch.allclose(vals.float().sum(-1), torch.ones(10), atol=1e-5)


def test_gate_quantize_exclude_flag():
    """router 永不量化：quantize_exclude=True；专家线性默认可量化。"""
    moe = _make()
    assert moe.gate.quantize_exclude is True
    assert len(moe.experts) == 8
    e0 = moe.experts[0]
    assert e0.gate_proj.quantize_exclude is False
    assert e0.up_proj.quantize_exclude is False
    assert e0.down_proj.quantize_exclude is False
    # 参数名与 HF checkpoint 直配（experts.{i}.gate_proj 等）
    names = {n for n, _ in moe.named_parameters()}
    assert "gate.weight" in names
    assert "experts.0.gate_proj.weight" in names
    assert "experts.7.down_proj.weight" in names


def test_k1_and_kE():
    for k in (1, 8):  # 8 = 全专家（等价稠密化路径的极端）
        moe = _make(k=k)
        x = torch.randn(5, 64)
        with torch.no_grad():
            assert _bitclose(moe(x), moe.reference(x))

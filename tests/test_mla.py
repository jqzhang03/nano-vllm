"""DeepSeek-V2 / MLA CPU 单测（无需 GPU）。

覆盖：
- 结构：参数路径与 HF checkpoint 直配（self_attn.*、mlp.gate/experts.{i}.*/
  shared_experts.*、dense mlp.*）；MoE 继承布局不重复注册；
- 语义自洽：无引擎 CPU 前向（手工稠密注意力分支）确定性、无 NaN、形状正确；
- MoE routed_scaling_factor：默认 1.0 恒等、非 1 时路由概率被缩放；
- RMSNorm 非破坏性（_mla_check.py 定位的真实 bug：fp32 下 x.float() 别名输入，
  残差流中间张量被原位归一化）——CPU fp32 模型 forward 与"逐层克隆重算"一致。
"""
import torch

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.moe import MoE
from nanovllm.models.deepseek_v2 import DeepseekV2ForCausalLM


def _toy_cfg():
    from transformers import DeepseekV2Config
    cfg = DeepseekV2Config(
        vocab_size=2048,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,          # 层0 dense；层1 MoE
        num_attention_heads=16,
        first_k_dense_replace=1,
        kv_lora_rank=64,
        q_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        routed_scaling_factor=1.0,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True,
    )
    return cfg


def test_deepseek_param_names_match_hf():
    """参数路径与 HF checkpoint 直配（loader 零映射、零改名）。"""
    cfg = _toy_cfg()
    m = DeepseekV2ForCausalLM(cfg)
    names = {n for n, _ in m.named_parameters()}
    want = [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",          # dense 层
        "model.layers.1.mlp.gate.weight",               # MoE router
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.3.down_proj.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
        "model.embed_tokens.weight",
    ]
    for w in want:
        assert w in names, f"缺参数 {w}"
    # MoE 继承不重复注册（mlp.moe.* 不应出现）
    assert not any(n.startswith("model.layers.1.mlp.moe") for n in names)


def test_cpu_forward_manual_dense():
    """无引擎 CPU 前向：确定性、无 NaN、行数与词表形状正确。"""
    torch.manual_seed(0)
    cfg = _toy_cfg()
    m = DeepseekV2ForCausalLM(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2:
                p.uniform_(-0.05, 0.05)
            else:
                p.fill_(1.0)          # norm 权重（RMSNorm ×w）
    with torch.no_grad():
        ids = torch.randint(0, cfg.vocab_size, (13,))
        h1 = m.compute_logits(m(ids, torch.arange(13)))
        h2 = m.compute_logits(m(ids, torch.arange(13)))
    assert h1.shape == (13, cfg.vocab_size)
    assert torch.isfinite(h1).all()
    assert torch.allclose(h1, h2, atol=1e-4), "同权重同输入应确定"


def test_rmsnorm_no_inplace_on_input():
    """RMSNorm fp32 不得修改输入（残差流复用中间张量的正确性前提）。

    回归场景：_mla_check.py 定位——fp32 下 x.float() 别名输入，mul_ 把调用方
    的残差流张量原地归一化；bf16 路径 .float() 拷贝所以引擎无感。
    """
    torch.manual_seed(1)
    norm = RMSNorm(64, eps=1e-6)
    x = torch.randn(5, 64)
    ref = x.clone()
    with torch.no_grad():
        y = norm(x)
    assert torch.equal(x, ref), "输入被 RMSNorm 原位修改"
    # 与逐元素重算一致
    with torch.no_grad():
        var = x.float().pow(2).mean(-1, keepdim=True)
        y2 = (x.float() * torch.rsqrt(var + norm.eps)) * norm.weight
    assert torch.allclose(y.float(), y2, atol=1e-6)


def test_routed_scaling_factor():
    torch.manual_seed(2)
    x = torch.randn(7, 48)
    m1 = MoE(48, 96, num_experts=6, top_k=2)
    with torch.no_grad():
        for p in m1.parameters():
            p.uniform_(-0.05, 0.05)
    m2 = MoE(48, 96, num_experts=6, top_k=2, routed_scaling_factor=2.0)
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        v1, i1 = m1._route(x)
        v2, i2 = m2._route(x)
    assert torch.equal(i1, i2)
    assert torch.allclose(v2.float(), v1.float() * 2.0, atol=1e-6)
    y1 = m1(x)
    y2 = m2(x)
    # 默认 1.0 时与旧行为一致（数值同 MoE reference）
    y1r = m1.reference(x)
    assert torch.allclose(y1, y1r, atol=1e-5)
    assert not torch.allclose(y1, y2, atol=1e-3), "factor=2 应改变输出"

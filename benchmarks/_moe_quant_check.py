"""MoE 量化机制核验：量化确实逐专家生效、router 排除、数值误差在 RTN 预期内。

背景：_moe_quant.py 的 toy 模型 int4/fp8 top-1 = 0%——先判定是机制错还是
"随机权重顶层贴边"。本脚本给出三个证据：
1) 模型级：int4 引擎里 experts 的每个线性 int4=True，而 MoE.gate 未被量化；
2) 层级：单层 MoE 专家量化后 forward vs fp16 forward 的 rel 误差（RTN 预期 ~1e-2 级；
   若机制错会 >0.1）；
3) logits 尺度：随机 toy 的 fp16 logits std/顶层间距（顶层贴边 → 微小扰动翻 top）。

用法: python benchmarks/_moe_quant_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

TOY = os.path.expanduser("~/moe_toy")


def evidence_logits():
    """3) logits 尺度：随机 toy 顶层是否贴边。"""
    lg = torch.load("/tmp/_moe_e2e_logits.pt", map_location="cpu")
    print(f"fp16 logits: std={lg.std().item():.4f} "
          f"absmax={lg.abs().max().item():.4f}")
    for i in range(lg.size(0)):
        top2 = lg[i].topk(2).values
        print(f"  prompt{i}: top1={top2[0].item():.4f} top2={top2[1].item():.4f} "
              f"gap={ (top2[0]-top2[1]).item():.4f} "
              f"-> 顶层间距/标度 = {(top2[0]-top2[1]).item()/lg.std().item():.2f} sigma")


def evidence_layer_quant_error():
    """2) 层级误差：专家 int4/fp8 后 forward vs fp16 forward（bf16——int4 Triton
    内核的激活要求 bf16，与引擎默认一致）。用 2-范数比（‖Δy‖/‖y‖）而非 max/极值。"""
    from nanovllm.layers.moe import MoE
    torch.manual_seed(0)
    moe = MoE(512, 768, num_experts=8, top_k=2).cuda().half().to(torch.bfloat16)
    with torch.no_grad():
        for p in moe.parameters():
            p.uniform_(-0.05, 0.05)
    x = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y16 = moe(x)
    for name, quant in [("int4(dense_path)", lambda m: m.quantize_int4(dense_path=True)),
                        ("int4(pure)", lambda m: m.quantize_int4(dense_path=False)),
                        ("fp8", lambda m: m.quantize_fp8())]:
        import copy
        m2 = copy.deepcopy(moe)  # 同权重副本（quantize 会 del self.weight）
        with torch.no_grad():
            # 只量化专家（gate 排除——验证 quantize_exclude 生效）
            for e in m2.experts:
                for lin in (e.gate_proj, e.up_proj, e.down_proj):
                    quant(lin)
            assert not getattr(m2.gate, "int4", False) and not getattr(m2.gate, "fp8", False), \
                "gate 不应被量化"
            yq = m2(x)
        rel = (yq.float() - y16.float()).norm() / y16.float().norm()
        print(f"layer {name:14s}: gate_int4={m2.gate.int4} rel_norm_err={rel:.3e}")


def evidence_engine_quantized():
    """1) 引擎级：int4 引擎内 experts 全量化、gate 未量化。"""
    from nanovllm import LLM, SamplingParams
    llm = LLM(TOY, max_model_len=128, quantization="int4", kv_swap=False,
              enforce_eager=True)
    model = llm.model_runner.model
    n_exp_int4 = n_gate_q = 0
    for name, m in model.named_modules():
        if hasattr(m, "int4"):
            if "mlp.experts" in name and name.endswith(("gate_proj", "up_proj", "down_proj")):
                n_exp_int4 += int(m.int4)
            if name.endswith("mlp.gate"):
                n_gate_q += int(m.int4 or m.fp8)
    print(f"engine int4: quantized expert linears={n_exp_int4} (期望=2 MoE层×8×3=48), "
          f"quantized gates={n_gate_q} (期望 0)")
    llm.exit()


if __name__ == "__main__":
    print("=== MoE 量化机制核验 ===")
    evidence_logits()
    print()
    evidence_layer_quant_error()
    print()
    evidence_engine_quantized()
    print("\nDONE")

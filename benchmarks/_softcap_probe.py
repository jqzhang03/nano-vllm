"""Gemma-2 attn logit soft-cap 的近似误差测量（真实模型上）。

问题背景：attn_logit_softcapping = cap·tanh(scores/cap)，在 Gemma-2 的尺度
（scale=1/16 或 1/√head_dim，logits 通常在 ±2 内）下 tanh 处于近线性区——
"如果跳过 softcap，logits 差多少？" 这个数字决定能否在 fp8 KV 路径（自研内核
无 softcap）上近似省略。

方法：gemma2 引擎（fp16 KV）跑真实 prompt prefill，用 hook 抓某层注意力前的
原始 logits（q·kᵀ·scale）与 softcap 后的值，统计相对误差；同时对比
final_logit_softcapping 对最终 logits 的裁剪幅度（这个必须实现，量级大）。

用法: python benchmarks/_softcap_probe.py <gemma2_model_dir>
"""
import os
import sys

import torch

from nanovllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "To bake a chocolate cake, you need",
    "Machine learning is",
]


def main():
    model = os.path.expanduser(sys.argv[1])
    llm = LLM(model, max_model_len=512, quantization="none", kv_swap=False)
    hf = llm.config.hf_config
    cap = getattr(hf, "attn_logit_softcapping", None)
    final_cap = getattr(hf, "final_logit_softcapping", None)
    print(f"attn_logit_softcapping={cap} final_logit_softcapping={final_cap}")

    # hook：抓某个 Attention 里 softmax 之前的原始 logits（重算 q·kᵀ·scale）
    # 用 forward hook 在 Attention.forward 之后拿不到内部量 → 直接 monkey-patch
    # 一层（层0），记录 q、k 的原始值。**单 seq**（多 seq 打包后 view 的 stride
    # 会触发 store_kvcache 的 stride 断言；单 seq 与 _gemma2_debug 同款已验证）。
    from nanovllm.engine.sequence import Sequence
    attn = llm.model_runner.model.model.layers[0].self_attn
    rec = {}

    def rec_fwd(positions, hidden_states):
        qkv = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = q.view(-1, attn.num_heads, attn.head_dim)
        k = k.view(-1, attn.num_kv_heads, attn.head_dim)
        v = v.view(-1, attn.num_kv_heads, attn.head_dim)  # v 必须 reshape（漏了会踩 store 断言）
        q, k = attn.rotary_emb(positions, q, k)
        rec["q"] = q.clone().float()
        rec["k"] = k.clone().float()
        o = attn.attn(q, k, v)
        return attn.o_proj(o.flatten(1, -1))

    orig = attn.forward
    attn.forward = rec_fwd
    try:
        bm = llm.scheduler.block_manager
        seq = Sequence(llm.tokenizer.encode(PROMPTS[0]))
        seq.num_scheduled_tokens = len(seq)
        bm.allocate(seq, bm.can_allocate(seq))
        llm.model_runner.call("run", [seq], "prefill", True, False)
        bm.deallocate(seq)
    finally:
        attn.forward = orig
    q, k = rec["q"], rec["k"]  # [T, H, D] / [T, K_H, D]
    g = q.shape[1] // k.shape[1]
    ke = k.repeat_interleave(g, dim=1)
    scale = attn.scaling
    scores = torch.einsum("qhd,khd->qhk", q, ke) * scale  # [T, H, T]
    T = q.shape[0]
    rows = torch.arange(T, device=q.device)
    causal = rows[:, None] >= rows[None, :]
    scores = scores.masked_fill(~causal[:, None, :], float("-inf"))
    finite = torch.isfinite(scores)
    if cap:
        capped = cap * torch.tanh(scores / cap)
        rel = ((capped - scores).abs() / (scores.abs() + 1e-8))[finite]
        print(f"\nattn softcap 影响（层0，{T} token）:")
        print(f"  原始 logits: min={scores[finite].min().item():.3f} "
              f"max={scores[finite].max().item():.3f} "
              f"mean|.|={scores[finite].abs().mean().item():.3f}")
        print(f"  softcap 后相对变化: mean={rel.mean().item()*100:.3f}% "
              f"max={rel.max().item()*100:.2f}%")
        print(f"  结论: {'近恒等（省略 softcap 误差 <1%）' if rel.max().item() < 0.01 else '必须实现'}")
    if final_cap:
        print(f"\nfinal logit softcap={final_cap}: 作用于 lm_head 输出 "
              f"(量级可达 ±30+，tanh 明显非线性——必须实现，本引擎已实现)")

    # 生成冒烟（验证 softcap 全链路）
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=16), use_tqdm=False)
    for o in outs:
        print(f"  -> {o['text'][:60]!r}")
    llm.exit()
    print("\nSOFTCAP PROBE OK")


if __name__ == "__main__":
    main()

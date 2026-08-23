"""Medusa头单元级验证：头是否学到了模型的next-token分布？

在推理数据（与训练同源的prompt）上：
- head_0..head_4 的 top-1 与真实 next token 的重合率（≈模型top-1准确率上限）
- head_0 的 top-1 与模型 LM head argmax 的重合率（头学习质量，应~90%+）
若 head≈model 而引擎α≈0 → 集成bug（行索引/时机）；若 head≈model 差 → 训练/加载问题。

用法（WSL，GPU）：
    python benchmarks/_medusa_debug.py [--medusa-path results/medusa_heads.pt]
"""
from __future__ import annotations

import argparse
import os

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
REAL_PROMPTS = [
    "The capital of France is", "To bake a chocolate cake, you need",
    "The three laws of robotics are", "A summary of the water cycle:",
    "Machine learning is", "The best way to learn programming is",
    "Photosynthesis happens when", "In 1969, humans",
    "The history of the steam engine begins with", "A good morning routine starts with",
    "The solar system consists of", "To improve your sleep, you should",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录（默认 Qwen3-0.6B）")
    p.add_argument("--medusa-path", default="results/medusa_heads.pt")
    args = p.parse_args()
    args.model = os.path.expanduser(args.model)  # bash argv 不展开 ~ → 手动展开

    llm = LLM(args.model, speculative="medusa", medusa_path=args.medusa_path, gpu_memory_utilization=0.9)
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    runner = llm.model_runner
    heads = runner.medusa_heads
    gamma = llm.config.max_draft_len
    tokenizer = llm.tokenizer

    # 与训练相同的提取流程：prefill → 最后一层hidden
    from nanovllm.utils.context import reset_context
    seqs = [tokenizer.encode(pp) for pp in REAL_PROMPTS]
    num_blocks = runner.config.num_kvcache_blocks
    seq_objs = [Sequence(t) for t in seqs]
    for s in seq_objs:
        s.num_scheduled_tokens = s.num_tokens
        s.block_table = [k % num_blocks for k in range(s.num_blocks)]
    input_ids, positions = runner.prepare_prefill(seq_objs)
    with torch.inference_mode():
        hidden = runner.model(input_ids, positions).float()  # [T, H] fp32
    reset_context()

    all_tok = torch.tensor([t for s in seqs for t in s], dtype=torch.long).cuda()
    T = len(all_tok)
    n = T - gamma - 2
    h = hidden[:n]
    y_true = [all_tok[k + 1:k + 1 + n] for k in range(gamma + 1)]

    # 模型 LM head 的 argmax（位置t的logits预测t+1；模型/头权重都是bf16）
    h16 = h.to(torch.bfloat16)
    with torch.inference_mode():
        logits = llm.model_runner.model.compute_logits(h16)
        y_model = logits.argmax(dim=-1)  # [n]
        y_head = [hd(h16).argmax(dim=-1) for hd in heads.heads]  # 每头 [n]

    def acc(a, b):
        return (a == b).float().mean().item()

    print(f"模型 top-1 准确率（LM head vs 真实next）: {acc(y_model, y_true[0]):.3f}")
    for k in range(gamma + 1):
        # head_k(h_t) 预测 t+k+1；模型在位置 t+k 的logits也预测 t+k+1 → 对齐 y_model[k:]
        m = y_model[k:]
        L = len(m)
        print(f"head_{k} top-1 vs 真实next(t+{k + 1}): {acc(y_head[k][:L], y_true[k][:L]):.3f} | "
              f"vs 模型argmax(t+{k + 1}): {acc(y_head[k][:L], m):.3f}")
    # head_0 应该预测 t+1：与 y_true[0] / y_model 对比
    print(f"head_0 vs head_1 top-1 重合: {acc(y_head[0], y_head[1]):.3f}（应低——不同偏移）")
    llm.exit()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

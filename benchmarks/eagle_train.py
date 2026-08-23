"""EAGLE-1 草稿层训练（自蒸馏）：用目标模型自己的数据训无 RoPE 的草稿 transformer 层。

语义：F(h_t, e(w_{t+1})) → h̃_{t+1}（预测下一位置的 hidden），草稿分布 =
LM_head(h̃_{t+1}) 预测 w_{t+2}。损失 = CE(LM_head(F(h_t, e(w_{t+1}))), w_{t+2})
+ λ·MSE(F(...), h_{t+1})（EAGLE-1 双损失：LM head 对齐 + 特征回归）。

流程：
1. 数据生成：真实 prompt + 随机 token prompt，engine 自回归生成（同 medusa_train）；
2. 特征提取：整批 prefill 拿最后一层 hidden h_t；标签 = w_{t+2}、特征目标 = h_{t+1}；
3. 训练前复制 embed/lm_head 权重（引擎 exit 后仍需要做草稿分布与输入 embedding）；
4. 训练：AdamW，λ=0.5 特征 MSE；
5. 保存 state_dict。

用法（WSL，GPU）：
    python benchmarks/eagle_train.py [--out results/eagle_layer.pt] [--steps 3000]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm import LLM, SamplingParams
from nanovllm.layers.eagle import EagleLayer

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
REAL_PROMPTS = [
    "The capital of France is",
    "To bake a chocolate cake, you need",
    "The three laws of robotics are",
    "A summary of the water cycle:",
    "Machine learning is",
    "The best way to learn programming is",
    "Photosynthesis happens when",
    "In 1969, humans",
    "The history of the steam engine begins with",
    "A good morning routine starts with",
    "The solar system consists of",
    "To improve your sleep, you should",
    "Quantum computing works by",
    "The main difference between TCP and UDP is",
    "A healthy diet should include",
    "The rules of chess are",
    "Deep learning models are trained by",
    "The Amazon rainforest is",
    "Cooking pasta correctly requires",
    "The invention of the telephone is credited to",
]


def generate_data(llm: LLM, num_seqs: int, seq_len: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    tokenizer = llm.tokenizer
    prompts = [tokenizer.encode(p) for p in REAL_PROMPTS]
    for i in range(len(prompts), num_seqs):
        prompts.append(rng.sample(range(0, 50000), 24))
    sps = [SamplingParams(temperature=0.8, ignore_eos=True, max_tokens=seq_len)] * len(prompts)
    out = llm.generate(prompts, sps, use_tqdm=False)
    return [o["token_ids"] for o in out]


def extract_features(llm: LLM, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
    """整批 prefill 一次，返回 EAGLE 训练四元组（跳过 LM head）。

    对位置 t（0..n-1）：h_t（输入特征）、w_{t+1}（下一 token id，embed 用）、
    label = w_{t+2}、h_target = h_{t+1}（MSE 目标）。
    """
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import reset_context
    runner = llm.model_runner
    num_blocks = runner.config.num_kvcache_blocks
    seq_objs = [Sequence(t) for t in seqs]
    for s in seq_objs:
        s.num_scheduled_tokens = s.num_tokens
        s.block_table = [k % num_blocks for k in range(s.num_blocks)]
    input_ids, positions = runner.prepare_prefill(seq_objs)
    with torch.inference_mode():
        hidden = runner.model(input_ids, positions)  # [T, H]
    reset_context()
    hidden = hidden.float()
    all_tok = torch.tensor([t for s in seqs for t in s], dtype=torch.long)
    n = len(all_tok) - 3  # 需要 h_t, w_{t+1}, w_{t+2}, h_{t+1}
    assert n > 0 and hidden.shape[0] >= n + 1
    return (hidden[:n],                       # h_t
            all_tok[1:n + 1],                 # w_{t+1}
            all_tok[2:n + 2],                 # w_{t+2} (label)
            hidden[1:n + 1])                  # h_{t+1} (MSE 目标)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help="模型目录（默认 Qwen3-0.6B）")
    p.add_argument("--out", default="results/eagle_layer.pt")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-seqs", type=int, default=576, help="自蒸馏数据序列数（24M 参数 vs 49k 样本会过拟合——数据越多越好）")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-feat", type=float, default=0.5, help="特征 MSE 权重（EAGLE-1）")
    args = p.parse_args()
    args.model = os.path.expanduser(args.model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"== 1/4 生成自蒸馏数据（{args.num_seqs} seqs × {args.seq_len} tok） ==")
    llm = LLM(args.model, gpu_memory_utilization=0.9)
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.8, max_tokens=8), use_tqdm=False)
    seqs = generate_data(llm, args.num_seqs, args.seq_len, args.seed)
    print(f"   生成 {len(seqs)} 条序列")

    print("== 2/4 提取 hidden / 标签 / 特征目标 ==")
    t0 = time.perf_counter()
    h_t, w_next, label, h_target = extract_features(llm, seqs)
    n = h_t.shape[0]
    print(f"   h_t {tuple(h_t.shape)}（{time.perf_counter() - t0:.1f}s）")

    # 训练需要目标模型的 embed（输入 e(w_{t+1})）与 LM head（草稿分布）——
    # 在 exit 前复制（引擎占 14GB，训练前必须释放，否则 allocator 慢 60×）
    embed_w = llm.model_runner.model.model.embed_tokens.weight.detach().clone()
    lm_head_w = llm.model_runner.model.lm_head.weight.detach().clone()
    hf = llm.config.hf_config
    llm.exit()
    torch.cuda.empty_cache()
    h_t = h_t.cuda()
    w_next = w_next.cuda()
    label = label.cuda()
    h_target = h_target.cuda()
    embed_w = embed_w.cuda()
    lm_head_w = lm_head_w.cuda()

    print(f"== 3/4 训练（{args.steps} 步, batch {args.batch}, lr {args.lr}, λ={args.lambda_feat}） ==")
    torch.backends.cuda.matmul.allow_tf32 = True
    layer = EagleLayer(hf.hidden_size, hf.num_attention_heads,
                       getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads,
                       hf.intermediate_size, hf.rms_norm_eps).cuda().float()
    # LinearBase 的权重是 torch.empty（引擎靠加载填充）——从头训练必须先初始化
    from nanovllm.layers.linear import LinearBase
    for m in layer.modules():
        if isinstance(m, LinearBase):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    opt = torch.optim.AdamW(layer.parameters(), lr=args.lr, weight_decay=0.05)
    n = h_t.shape[0]
    t0 = time.perf_counter()
    for step in range(args.steps):
        batch = torch.randint(0, n, (args.batch,), generator=torch.Generator().manual_seed(step))
        h = h_t[batch]                                    # [B, H]
        emb = F.embedding(w_next[batch], embed_w)         # e(w_{t+1})
        h_pred = layer(h, emb)                            # [B, H]
        logits = F.linear(h_pred.to(lm_head_w.dtype), lm_head_w)  # [B, V] 草稿分布
        loss_ce = F.cross_entropy(logits, label[batch])
        loss_feat = F.mse_loss(h_pred, h_target[batch])
        loss = loss_ce + args.lambda_feat * loss_feat
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            el = time.perf_counter() - t0
            print(f"   step {step:5d}: loss={loss.item():.4f} (ce={loss_ce.item():.4f} "
                  f"feat={loss_feat.item():.4f}) | {el / (step + 1) * 1000:.0f}ms/步")

    torch.save(layer.state_dict(), args.out)
    print(f"   eagle layer -> {args.out}（{sum(p.numel() for p in layer.parameters()) / 1e6:.1f}M 参数）")


if __name__ == "__main__":
    main()

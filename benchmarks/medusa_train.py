"""Medusa头训练（自蒸馏）：用目标模型自己生成的数据训 γ+1 个预测头。

流程：
1. 数据生成：混合真实prompt + 随机token prompt，engine 自回归生成 L token
   （ignore_eos），得到 N 条完成序列（标签 = 模型自己的输出，自蒸馏）。
2. 特征提取：每条序列整体做一次 prefill 前向（跳过LM head，避免9GB logits），
   取最后一层hidden [T,H]；head_k 的标签 = 序列内 token_{t+k}。
3. 训练：CE 损失 Σ_k CE(head_k(h_t), token_{t+k})，AdamW。
4. 保存 state_dict；评估：生成时统计接受率 α 与吞吐。

语义：head_k(h_t) 预测位置 t+k 的 token；推理时 draft 输入 = 验收后新
t_last 的 hidden（见 nanovllm/layers/medusa.py 的约定）。

用法（WSL，GPU）：
    python benchmarks/medusa_train.py [--out results/medusa_heads.pt] [--steps 5000]
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.medusa import MedusaHeads

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
    """engine自回归生成N条序列（真实prompt + 随机token prompt混合）。"""
    rng = random.Random(seed)
    tokenizer = llm.tokenizer
    prompts = [tokenizer.encode(p) for p in REAL_PROMPTS]
    for i in range(len(prompts), num_seqs):
        prompts.append(rng.sample(range(0, 50000), 24))
    sps = [SamplingParams(temperature=0.8, ignore_eos=True, max_tokens=seq_len)] * len(prompts)
    out = llm.generate(prompts, sps, use_tqdm=False)
    return [o["token_ids"] for o in out]


def extract_features(llm: LLM, seqs: list[list[int]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """整批prefill一次，返回 (hidden [n,H] fp32, 每头标签 [n])；跳过LM head。"""
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import reset_context
    runner = llm.model_runner
    num_blocks = runner.config.num_kvcache_blocks
    seq_objs = [Sequence(t) for t in seqs]
    for s in seq_objs:
        s.num_scheduled_tokens = s.num_tokens
        s.block_table = [k % num_blocks for k in range(s.num_blocks)]  # slot walk用（内容无意义）
    input_ids, positions = runner.prepare_prefill(seq_objs)
    with torch.inference_mode():
        hidden = runner.model(input_ids, positions)  # [T, H] fp16（不经过LM head）
    reset_context()
    hidden = hidden.float()
    gamma = llm.config.max_draft_len
    all_tok = torch.tensor([t for s in seqs for t in s], dtype=torch.long)
    # head_k(h_t) 预测位置 t+k+1（k=0..γ；推理draft_k=head_{shift+k-1}，见medusa.py约定）
    n = len(all_tok) - (gamma + 2)
    assert n > 0 and hidden.shape[0] >= n
    labels = [all_tok[k + 1:k + 1 + n] for k in range(gamma + 1)]
    return hidden[:n], labels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/medusa_heads.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-seqs", type=int, default=192)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"== 1/4 生成自蒸馏数据（{args.num_seqs} seqs × {args.seq_len} tok） ==")
    llm = LLM(MODEL, gpu_memory_utilization=0.9)
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.8, max_tokens=8), use_tqdm=False)
    seqs = generate_data(llm, args.num_seqs, args.seq_len, args.seed)
    print(f"   生成 {len(seqs)} 条序列")

    print("== 2/4 提取最后一层hidden与标签 ==")
    t0 = time.perf_counter()
    hidden, labels = extract_features(llm, seqs)
    print(f"   hidden {tuple(hidden.shape)} fp32（{time.perf_counter() - t0:.1f}s）")
    # 关键：训练前释放引擎（模型+KV缓存+图池 ~14GB）——实测显存压力下每步
    # 分配临时张量慢 60×（7.3s vs 124ms/步）。hidden/labels已被引用，exit不影响。
    llm.exit()
    torch.cuda.empty_cache()
    hidden = hidden.cuda()
    labels = [lab.cuda() for lab in labels]

    print(f"== 3/4 训练（{args.steps} 步, batch {args.batch}, lr {args.lr}） ==")
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32头GEMM走TF32张量核（~5×）
    hf = llm.config.hf_config
    heads = MedusaHeads(llm.config.max_draft_len + 1, hf.hidden_size,
                        llm.config.medusa_hidden, hf.vocab_size).cuda().float()
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    n = labels[0].shape[0]
    t0 = time.perf_counter()
    t_step0 = t0
    for step in range(args.steps):
        batch = torch.randint(0, n, (args.batch,), generator=torch.Generator().manual_seed(step))
        h = hidden[batch]
        loss = torch.tensor(0.0, device="cuda")
        for k, hd in enumerate(heads.heads):
            loss = loss + torch.nn.functional.cross_entropy(hd(h), labels[k][batch])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            now = time.perf_counter()
            print(f"   step {step:5d} loss {loss.item() / len(heads.heads):.3f} "
                  f"({(now - t_step0) / 100 * 1000:.0f}ms/步)", flush=True)
            t_step0 = now
    heads.cpu()
    torch.cuda.empty_cache()
    torch.save(heads.state_dict(), args.out)
    print(f"   heads -> {args.out}")

    print("== 4/4 评估：baseline vs ngram vs medusa（α 与吞吐） ==")
    tokenizer = llm.tokenizer
    eval_prompts = [tokenizer.encode(pp) for pp in REAL_PROMPTS[:12]]
    sps = [SamplingParams(temperature=0.6, max_tokens=96, ignore_eos=True)] * len(eval_prompts)
    for mode, mpath in (("none", ""), ("ngram", ""), ("medusa", args.out)):
        evallm = LLM(MODEL, speculative=mode, medusa_path=mpath, gpu_memory_utilization=0.9)
        evallm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
        t0 = time.perf_counter()
        evallm.generate(eval_prompts, sps, use_tqdm=False)
        wall = time.perf_counter() - t0
        stats = evallm.collect_metrics()["step_stats"]
        tok = stats["decode_tokens"]
        line = f"   {mode:<6}: {tok} token / {wall:.2f}s = {tok / wall:.0f} tok/s"
        if mode != "none":
            d = stats["spec_draft_tokens"]
            acc = stats["spec_accepted_drafts"]
            line += f" | α = {acc / d if d else float('nan'):.3f}"
        print(line, flush=True)
        evallm.exit()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

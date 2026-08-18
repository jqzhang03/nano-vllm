"""verify前向微基准 v2：定位 verify 步每token成本≈decode（≈7.5× 长查询prefill）的根源。

v1 发现：分页K/V只贵15%（不是瓶颈）；但探针的"长查询prefill形状"（8×256）
165ms/2048tok 与引擎真实prefill阶段（~35ms/步）差5倍——v2 排除首轮编译、
加入真实prefill形状（多序列变长、混合chunk）与真实verify形状（draining批次）：
  A. 8 seqs × 256（均匀长查询）       ——v1的慢形状
  A2. ~30 seqs 变长 ~2048 tokens       ——引擎真实prefill步形状
  B. 256 seqs × 5 连续K/V             ——短查询、无分页
  C. 256 seqs × 5 分页K/V             ——真实verify形状
  D. 引擎真实draining序列（行数逐步变）  ——含形状变化的影响

用法（WSL，GPU）：
    python benchmarks/_verify_probe.py [--fp8] [--repeats 5]
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import reset_context

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def make_runner(fp8: bool) -> ModelRunner:
    config = Config(model=MODEL, kv_cache_dtype="fp8_e4m3" if fp8 else "auto",
                    gpu_memory_utilization=0.9)
    return ModelRunner(config, 0, [])


@torch.inference_mode()
def time_forward(runner, input_ids, positions, repeats, skip_first=True) -> list[float]:
    times = []
    torch.cuda.synchronize()
    for i in range(repeats):
        t0 = time.perf_counter()
        runner.model.compute_logits(runner.model(input_ids, positions))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    if skip_first:
        times = times[1:]
    return times


def _add_blocks(runner, seq):
    num_blocks = runner.config.num_kvcache_blocks
    seq.block_table = [k % num_blocks for k in range(seq.num_blocks)]


def case_long_uniform(runner, repeats):
    """A: 2048 tokens / 8 seqs × 256（均匀长查询，连续K/V）。"""
    seqs = [Sequence(torch.randint(0, 50000, (256,)).tolist()) for _ in range(8)]
    for seq in seqs:
        seq.num_scheduled_tokens = 256
        _add_blocks(runner, seq)
    input_ids, positions = runner.prepare_prefill(seqs)
    ms = time_forward(runner, input_ids, positions, repeats)
    reset_context()
    return ms, 2048


def case_real_prefill(runner, repeats):
    """A2: ~2048 tokens / 30 seqs 变长（引擎真实prefill步形状：chunked多序列）。"""
    rng = torch.Generator(device="cpu").manual_seed(0)
    lengths = torch.randint(40, 110, (30,), generator=rng).tolist()
    total = sum(lengths)
    seqs = []
    for ln in lengths:
        seq = Sequence(torch.randint(0, 50000, (ln,)).tolist())
        seq.num_scheduled_tokens = ln
        _add_blocks(runner, seq)
        seqs.append(seq)
    input_ids, positions = runner.prepare_prefill(seqs)
    ms = time_forward(runner, input_ids, positions, repeats)
    reset_context()
    return ms, total


def case_short_fresh(runner, repeats):
    """B: 1280 tokens / 256 seqs × 5（短查询，连续K/V）。"""
    num_blocks = runner.config.num_kvcache_blocks
    seqs = [Sequence(torch.randint(0, 50000, (5,)).tolist()) for _ in range(256)]
    for seq in seqs:
        seq.num_scheduled_tokens = 5
        _add_blocks(runner, seq)
    input_ids, positions = runner.prepare_prefill(seqs)
    ms = time_forward(runner, input_ids, positions, repeats)
    reset_context()
    return ms, 1280


def case_verify(runner, repeats, fp8: bool):
    """C: 1280 tokens / 256 seqs × 5，分页K/V（真实verify形状）。"""
    num_blocks = runner.config.num_kvcache_blocks
    seqs = []
    for _ in range(256):
        seq = Sequence([7] * 24)
        seq.draft_tokens = [1, 2, 3, 4]
        seq.num_scheduled_tokens = 5
        seq.num_tokens = 24
        seq.num_cached_tokens = 23
        seq.block_table = [k % num_blocks for k in range(1)]
        seqs.append(seq)
    input_ids, positions = runner.prepare_spec(seqs)
    ms = time_forward(runner, input_ids, positions, repeats)
    reset_context()
    return ms, 1280


def case_drain(runner, repeats, fp8: bool):
    """D: 引擎真实draining序列（256行→0，行数逐步变，含形状变化）。"""
    num_blocks = runner.config.num_kvcache_blocks
    rng = torch.Generator(device="cpu").manual_seed(1)
    all_ms = []
    total_tok = 0
    rows = 256
    while rows > 0:
        seqs = []
        for _ in range(rows):
            seq = Sequence([7] * 24)
            seq.draft_tokens = [1, 2, 3, 4]
            seq.num_scheduled_tokens = 5
            seq.num_tokens = 24
            seq.num_cached_tokens = 23
            seq.block_table = [k % num_blocks for k in range(1)]
            seqs.append(seq)
        input_ids, positions = runner.prepare_spec(seqs)
        times = time_forward(runner, input_ids, positions, 2, skip_first=True)
        all_ms.extend(times)
        total_tok += rows * 5 * len(times)
        reset_context()
        rows = rows // 2
    return all_ms, total_tok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--repeats", type=int, default=6)
    args = p.parse_args()

    print(f"model: {os.path.basename(MODEL)} | kv_cache: {'fp8' if args.fp8 else 'fp16'} | "
          f"repeats={args.repeats}（排除首轮编译）")
    runner = make_runner(args.fp8)
    try:
        ms_a, tok_a = case_long_uniform(runner, args.repeats)
        ms_a2, tok_a2 = case_real_prefill(runner, args.repeats)
        ms_b, tok_b = case_short_fresh(runner, args.repeats)
        ms_c, tok_c = case_verify(runner, args.repeats, args.fp8)
        ms_d, tok_d = case_drain(runner, args.repeats, args.fp8)
    finally:
        runner.call("exit")
        torch.cuda.empty_cache()

    print(f"\nA  8×256=2048tok 连续K/V : {stat(ms_a):7.1f}ms/步 ({stat(ms_a) / tok_a * 1000:6.2f} µs/tok)")
    print(f"A2 ~30seq变长~2048tok    : {stat(ms_a2):7.1f}ms/步 ({stat(ms_a2) / tok_a2 * 1000:6.2f} µs/tok)")
    print(f"B  256×5=1280tok 连续K/V : {stat(ms_b):7.1f}ms/步 ({stat(ms_b) / tok_b * 1000:6.2f} µs/tok)")
    print(f"C  256×5=1280tok 分页K/V : {stat(ms_c):7.1f}ms/步 ({stat(ms_c) / tok_c * 1000:6.2f} µs/tok)")
    print(f"D  draining 256→0（形状变化）: {stat(ms_d):7.1f}ms/步 平均 ({tok_d / sum(ms_d) * 1000:6.2f} µs/tok)")
    print(f"\n短查询vs长查询(均匀): {stat(ms_b) / stat(ms_a):.2f}× | 分页vs连续: {stat(ms_c) / stat(ms_b):.2f}× "
          f"| 真实prefill vs 均匀: {stat(ms_a2) / stat(ms_a):.2f}× | draining vs 固定: {stat(ms_d) / stat(ms_c):.2f}×")


def stat(v):
    return sum(v) / len(v)


if __name__ == "__main__":
    main()

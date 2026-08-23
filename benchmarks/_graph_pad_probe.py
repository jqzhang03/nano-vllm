"""CUDA-graph 化 verify 前向的两个风险点验证：

1. flash varlen 的 max_seqlen_k 烘焙成 4096（真实值~28）是否会变慢/出错
   （CUDA graph 捕获时标量参数固定，每步真实值必须 ≤ 捕获值）。
2. 用零长度行填充到固定容量（cu_seqlens 重复值）是否正确、高效——
   graph 化需要固定形状，真实行数变化时用空行补齐。
"""
from __future__ import annotations

import os
import sys
import time

import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import set_context, reset_context

MODEL = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B/")
from flash_attn import flash_attn_varlen_func


def make_runner() -> ModelRunner:
    config = Config(model=MODEL, kv_cache_dtype="auto", gpu_memory_utilization=0.9)
    return ModelRunner(config, 0, [])


@torch.inference_mode()
def main():
    runner = make_runner()
    num_blocks = runner.config.num_kvcache_blocks
    try:
        # ---- 构造 256 行 verify 形状（真实行） ----
        n_real = 256
        seqs = []
        for _ in range(n_real):
            seq = Sequence([7] * 24)
            seq.draft_tokens = [1, 2, 3, 4]
            seq.num_scheduled_tokens = 5
            seq.num_tokens = 24
            seq.num_cached_tokens = 23
            seq.block_table = [k % num_blocks for k in range(1)]
            seqs.append(seq)
        input_ids, positions = runner.prepare_spec(seqs)
        ctx = runner.model_runner if False else None
        from nanovllm.utils.context import get_context
        c = get_context()

        # ---- 参考：真实 max_seqlen_k（~28） ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            runner.model.compute_logits(runner.model(input_ids, positions))
        torch.cuda.synchronize()
        ms_true = (time.perf_counter() - t0) / 5 * 1000
        reset_context()

        # ---- 1) max_seqlen_k 烘焙成 4096 ----
        set_context(True, c.cu_seqlens_q, c.cu_seqlens_k, c.max_seqlen_q, 4096,
                    c.slot_mapping, None, c.block_tables, is_spec=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            runner.model.compute_logits(runner.model(input_ids, positions))
        torch.cuda.synchronize()
        ms_4096 = (time.perf_counter() - t0) / 5 * 1000
        reset_context()

        # ---- 2) 填充到 512 行（1280→2560 real+pad），零长度行 ----
        n_pad = 512 - n_real
        cu_q = torch.cat([c.cu_seqlens_q, c.cu_seqlens_q[-1].expand(n_pad)], dim=0).int()
        cu_k = torch.cat([c.cu_seqlens_k, c.cu_seqlens_k[-1].expand(n_pad)], dim=0).int()
        bt = torch.cat([c.block_tables, torch.zeros(n_pad, c.block_tables.shape[1], dtype=torch.int32, device="cuda")], dim=0)
        sm = torch.cat([c.slot_mapping, torch.full((n_pad * 5,), -1, dtype=torch.int32, device="cuda")], dim=0)
        ids = torch.cat([input_ids, torch.zeros(n_pad * 5, dtype=torch.int64, device="cuda")])
        pos = torch.cat([positions, torch.zeros(n_pad * 5, dtype=torch.int64, device="cuda")])
        set_context(True, cu_q.cuda(), cu_k.cuda(), 5, 4096, sm.cuda(), None, bt.cuda(), is_spec=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            runner.model.compute_logits(runner.model(ids.cuda(), pos.cuda()))
        torch.cuda.synchronize()
        ms_pad = (time.perf_counter() - t0) / 5 * 1000
        reset_context()

        # ---- 正确性：填充版的真实行 logits == 未填充版 ----
        set_context(True, c.cu_seqlens_q, c.cu_seqlens_k, c.max_seqlen_q, c.max_seqlen_k,
                    c.slot_mapping, None, c.block_tables, is_spec=True)
        l_true = runner.model.compute_logits(runner.model(input_ids, positions))
        set_context(True, cu_q.cuda(), cu_k.cuda(), 5, 4096, sm.cuda(), None, bt.cuda(), is_spec=True)
        l_pad = runner.model.compute_logits(runner.model(ids.cuda(), pos.cuda()))
        diff = (l_pad[:n_real * 5].float() - l_true.float()).abs().max().item()

        print(f"真实max_seqlen_k(~28): {ms_true:6.1f}ms | 烘焙4096: {ms_4096:6.1f}ms "
              f"({ms_4096 / ms_true:.2f}×)")
        print(f"填充512行(2560tok, 含空行): {ms_pad:6.1f}ms ({ms_pad / 1280 * 1000:.1f}µs/tok, "
              f"vs 未填充 {ms_true / 1280 * 1000:.1f}µs/tok)")
        print(f"填充版真实行 logits max|Δ| vs 未填充: {diff:.6f} "
              f"({'PASS: 零长度行不影响真实行结果' if diff < 1e-3 else 'FAIL'})")
    finally:
        runner.call("exit")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

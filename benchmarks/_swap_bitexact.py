"""KV swap bit-exact 验证：swap_out 拷出 → 释放 → allocate_private 新块 → swap_in 拷回，
对比新块与旧块的 kv_cache 内容逐位一致（swap 的核心正确性：无损拷贝，免重新 prefill）。"""
import os

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(PATH, kv_swap=True, max_model_len=2048)
llm.generate(["warm up"] * 2, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
runner = llm.model_runner
scheduler = llm.scheduler
bm: BlockManager = scheduler.block_manager

# 一个 decode 序列：prefill 2048 token（8 块）后进入 decode
seq = Sequence([7] * 2048)
seq.num_scheduled_tokens = 2048
scheduler.add(seq)
llm.step()  # prefill 步（写 KV）
print(f"after prefill: len={len(seq)} cached={seq.num_cached_tokens} bt={len(seq.block_table)}")
seq.is_prefill = False
seq.num_cached_tokens = seq.num_tokens
seq.status = type(seq.status).RUNNING
old_block_ids = list(seq.block_table)
old_kv = runner.kv_cache[:, :, old_block_ids].clone()  # 快照旧 KV

# swap_out（记账 + 等 engine 拷贝）
scheduler.running.append(seq)
scheduler.preempt(seq)
assert seq in scheduler.swapped
assert len(scheduler.swap_pairs) == 1
_, out_ids, buf, direction = scheduler.swap_pairs[0]
assert direction == "out" and out_ids == old_block_ids
# 手动执行 engine 的 swap 拷贝逻辑（模拟 engine.step）
runner.swap_out(out_ids, buf)
scheduler.finish_swap_out(seq, out_ids)
assert not seq.block_table
print(f"after swap_out: len={len(seq)} cached={seq.num_cached_tokens} swapped={seq.swapped} "
      f"buf==old_kv: {(buf.float() - old_kv.float().cpu()).abs().max().item():.6f}")

# 释放后 KV 内容应保持（块还没被重分配）——立刻换入验证拷贝
scheduler._try_swap_in()
assert seq in scheduler.running and not seq.swapped
assert len(scheduler.swap_pairs) == 2
_, in_ids, buf2, direction = scheduler.swap_pairs[1]
assert direction == "in" and buf2 is buf
print(f"before swap_in: len={len(seq)} cached={seq.num_cached_tokens} bt={len(seq.block_table)} "
      f"in_ids={in_ids}")
runner.swap_in(in_ids, buf)

# 对比：换入的新块 vs 换出前的旧块（bit-exact）
new_kv = runner.kv_cache[:, :, in_ids[:len(old_block_ids)]]
d = (new_kv.float() - old_kv.float()).abs().max().item()
d_buf = (new_kv.float() - buf.float().cuda()).abs().max().item()
print(f"old_blocks={len(old_block_ids)} new_blocks={len(in_ids)} "
      f"kv max|Δ|={d:.6f} new_kv vs buf: {d_buf:.6f} dtype={runner.kv_cache.dtype}")
print(f"old_block_ids={old_block_ids} in_ids={in_ids}")
assert d == 0.0, "KV swap not bit-exact!"
print("KV SWAP BIT-EXACT OK")
llm.exit()

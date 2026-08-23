"""流式加载 vs eager 加载的参数逐项对比（定位 streaming loader 的权重映射 bug）。

用法:
  python benchmarks/_stream_weights_check.py [model]            # standalone 流式 vs eager
  python benchmarks/_stream_weights_check.py [model] --runner   # ModelRunner 完整路径 vs eager
输出：每个参数 max|eager - streaming|；buffer 对比；META 残留。
"""
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.registry import get_model_class
from nanovllm.utils.loader import load_model

path = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen2.5-0.5B")
use_runner = "--runner" in sys.argv

dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
torch.cuda.set_device(0)
cfg = AutoConfig.from_pretrained(path)
default_dtype = torch.get_default_dtype()
torch.set_default_dtype(cfg.dtype)
torch.set_default_device("cuda")
cls = get_model_class(cfg.model_type)

# eager（正确基线）
m1 = cls(cfg)
load_model(m1, path)
# get_rope 是 lru_cache(1)：不清缓存会让 m2 复用 m1 的 RoPE 实例，buffer 对比变平凡
# （正是 cos_sin_cache 被 to_empty 清零却没被抓到的原因）
from nanovllm.layers.rotary_embedding import get_rope
get_rope.cache_clear()

if use_runner:
    # 走 ModelRunner 的完整 streaming 路径（meta + hook + finalize + warmup）；
    # ModelRunner 会自己 init dist → 先销毁我们的
    dist.destroy_process_group()
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    runner = ModelRunner(Config(path, streaming_load=True, enforce_eager=True,
                                quantization="none", max_model_len=2048), 0, [])
    m2 = runner.model
else:
    with torch.device("meta"):
        m2 = cls(cfg)
    load_model(m2, path, streaming=True)
    # 复刻 _finalize_streaming 的重绑
    if getattr(cfg, "tie_word_embeddings", False):
        head = m2.lm_head
        if head.weight.is_meta:
            head.to_empty(device="cuda")
        head.weight.data = m2.model.embed_tokens.weight.data

n_bad = 0
for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
    assert n1 == n2, f"param order mismatch: {n1} vs {n2}"
    if p2.is_meta:
        print(f"META  {n1}")
        n_bad += 1
        continue
    if p1.shape != p2.shape or p1.dtype != p2.dtype:
        print(f"SHAPE {n1}: eager{p1.shape} {p1.dtype} vs stream{p2.shape} {p2.dtype}")
        n_bad += 1
        continue
    d = (p1.float() - p2.float()).abs().max().item()
    if d > 1e-3:
        n_bad += 1
        print(f"DIFF  {n1}: maxdiff={d:.4f}  <-- BAD")
b1d = dict(m1.named_buffers())
b2d = dict(m2.named_buffers())
print("buffers:", sorted(b1d))
for k in b1d:
    if k not in b2d:
        print(f"BUFFER-MISSING {k}")
        n_bad += 1
        continue
    b1, b2 = b1d[k], b2d[k]
    if b1.is_meta or b2.is_meta:
        print(f"BUFFER-META {k}")
        n_bad += 1
        continue
    d = (b1.float() - b2.float()).abs().max().item()
    if d > 1e-3:
        n_bad += 1
        print(f"BUFFER-DIFF {k}: maxdiff={d:.4f}  <-- BAD")
print(f"\nbad params/buffers: {n_bad}")
torch.set_default_device("cpu")
torch.set_default_dtype(default_dtype)
if not use_runner:
    dist.destroy_process_group()

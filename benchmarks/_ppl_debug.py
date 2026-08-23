"""裸模型评估路径对 qwen2 输出 uniform logits 的定位（_quant_ppl 用）。

对比：同一输入在 (a) 裸模型 eval_ce 式前向 与 (b) HF 参考 上的 hidden/logits。
"""
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.registry import get_model_class
from nanovllm.utils.loader import load_model

PATH = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen2.5-0.5B")
DEV = "cuda"

dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
torch.cuda.set_device(0)

hf = AutoConfig.from_pretrained(PATH)
model = get_model_class(hf.model_type)(hf).to(DEV, dtype=hf.dtype)
load_model(model, PATH)
if getattr(hf, "tie_word_embeddings", False):
    model.lm_head.weight.data = model.model.embed_tokens.weight.data
model.eval()

# 一段固定输入
ids = torch.tensor([1140, 2822, 1159, 5193, 5741, 1751, 1446, 2116, 21738, 220, 13, 5, 2870, 1159, 3060, 220],
                   device=DEV)
positions = torch.arange(ids.numel(), device=DEV)
T = ids.numel()
cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)

from nanovllm.utils.context import set_context, reset_context
set_context(True, cu, cu, T, T, torch.tensor([], dtype=torch.int32, device=DEV), None, None)
with torch.inference_mode():
    hidden = model(ids, positions)
reset_context()
print(f"hidden: shape {tuple(hidden.shape)} dtype {hidden.dtype} "
      f"mean {hidden.mean().item():.5f} std {hidden.std().item():.5f} max {hidden.abs().max().item():.4f}")
print(f"hidden rows all-equal: {(hidden[0] == hidden[1]).all().item()}")

with torch.inference_mode():
    logits = model.compute_logits(hidden)
print(f"logits: top-1 {logits.argmax(-1).tolist()}")
print(f"logits uniform check: row0 max-min {(logits[0].max() - logits[0].min()).item():.4f}")

# 权重诊断
w_head = model.lm_head.weight
w_emb = model.model.embed_tokens.weight
print(f"lm_head.weight: {tuple(w_head.shape)} {w_head.dtype} abs_sum {w_head.float().abs().sum().item():.3f}")
print(f"embed.weight  : {tuple(w_emb.shape)} {w_emb.dtype} abs_sum {w_emb.float().abs().sum().item():.3f}")
print(f"share storage : {w_head.data_ptr() == w_emb.data_ptr()}")
print(f"embed row1140 : {w_emb[1140][:8].tolist()}")
with torch.inference_mode():
    direct = torch.nn.functional.linear(hidden, w_head)
print(f"F.linear direct: abs_max {direct.abs().max().item():.4f} row0 max-min {(direct[0].max() - direct[0].min()).item():.4f}")

# HF 参考（同输入）
from transformers import AutoModelForCausalLM
ref = AutoModelForCausalLM.from_pretrained(PATH, dtype=hf.dtype).to(DEV).eval()
with torch.inference_mode():
    out = ref(input_ids=ids.unsqueeze(0), use_cache=False)
h_ref = out.hidden_states[-1][0] if out.hidden_states is not None else None
if h_ref is not None:
    d = (hidden.float() - h_ref.float()).abs()
    print(f"HF hidden: mean {h_ref.mean().item():.5f} std {h_ref.std().item():.5f}")
    print(f"hidden vs HF: max diff {d.max().item():.4f} mean {d.mean().item():.6f}")
l_ref = out.logits[0, -1].float()
d_l = (logits[-1].float() - l_ref).abs()
print(f"logits vs HF: max diff {d_l.max().item():.4f} mean {d_l.mean().item():.6f} "
      f"top1 agree {(logits[-1].argmax() == l_ref.argmax()).item()}")
dist.destroy_process_group()

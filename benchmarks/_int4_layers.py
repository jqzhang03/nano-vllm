"""INT4 逐层 hidden-state 漂移诊断：三层分离。

1. fp16 基线 → hidden16/logits16
2. quantize_int4 内核路径 → hidden8/logits8
3. 同权重反量化回 bf16（torch 参考，无内核）→ logits_deq
判定：logits8 vs logits_deq 小 = 内核正确（误差纯来自量化）；logits_deq vs logits16
大 = RTN int4 量化本身的代价（AWQ 的用武之地）。
"""
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.registry import get_model_class
from nanovllm.utils.loader import load_model

PATH = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B/")
DEV = "cuda"

dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
torch.cuda.set_device(0)

torch.manual_seed(0)
hf = AutoConfig.from_pretrained(PATH)
model = get_model_class(hf.model_type)(hf).to(DEV, dtype=hf.dtype)
load_model(model, PATH)
# CPU 构造→.to(cuda) 打破 tie 共享；tie 模型 checkpoint 常不含 lm_head.weight → 重绑
if getattr(hf, "tie_word_embeddings", False):
    model.lm_head.weight.data = model.model.embed_tokens.weight.data
model.eval()

ids = torch.randint(0, 50000, (8, 256), device=DEV).flatten()
positions = torch.arange(256, device=DEV).repeat(8)


def run(model, capture, hooks_layers=True):
    from nanovllm.utils.context import set_context, reset_context
    T = ids.numel()
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    set_context(True, cu, cu, T, T, torch.tensor([], dtype=torch.int32, device=DEV), None, None)
    outs = {}
    hooks = []
    if hooks_layers:
        for name, m in model.named_modules():
            if m.__class__.__name__ == "Qwen3DecoderLayer":
                def make_hook(nm):
                    def hook(mod, args, out):
                        outs[nm] = (out[0] if isinstance(out, tuple) else out).float().clone()
                    return hook
                hooks.append(m.register_forward_hook(make_hook(name)))
    with torch.inference_mode():
        hidden = model(ids, positions)
        logits = model.compute_logits(hidden)
    for h in hooks:
        h.remove()
    reset_context()
    return logits, outs


print("== fp16 基线 ==")
logits16, hid16 = run(model, True)

from nanovllm.layers.linear import LinearBase
linears = [m for m in model.modules() if isinstance(m, LinearBase)]
for m in linears:
    m.quantize_int4()

print("== int4 内核路径 ==")
logits8, hid8 = run(model, False, hooks_layers=True)

# 参考：反量化回 bf16（不跑内核）
for m in linears:
    q = m.w_int4  # (N, K//2) int8
    scale = m.w_int4_scale  # (N, K//g) bf16
    K = q.shape[1] * 2
    N = scale.shape[0]
    lo = (q & 0x0F).to(torch.int8) - 8
    hi = ((q >> 4).to(torch.int8) & 0x0F) - 8
    w_hat = torch.zeros(N, K, device=q.device, dtype=torch.bfloat16)
    w_hat[:, 0::2] = (lo.float() * scale[:, torch.arange(K // 2) // 64].to(q.device)).to(torch.bfloat16)
    w_hat[:, 1::2] = (hi.float() * scale[:, torch.arange(K // 2) // 64].to(q.device)).to(torch.bfloat16)
    m.weight = torch.nn.Parameter(w_hat.contiguous())  # 装回反量化权重走稠密路径
    del m.w_int4, m.w_int4_scale
    m.int4 = False

print("== 反量化参考（无内核） ==")
logits_deq, _ = run(model, False, hooks_layers=False)

for name in hid16:
    d_k = (hid16[name] - hid8[name]).abs()
    print(f"{name:<30}  int4 max_abs={d_k.max().item():.4f}  mean_abs={d_k.mean().item():.5f}")

dl = (logits16.float() - logits8.float()).abs()
print(f"\nlogits fp16 vs int4内核:  max={dl.max().item():.4f}  mean={dl.mean().item():.5f}")
dk = (logits8.float() - logits_deq.float()).abs()
print(f"logits 内核 vs 反量化参考: max={dk.max().item():.5f}  mean={dk.mean().item():.6f}  "
      f"(应≈0，隔离内核正确性)")
dq = (logits16.float() - logits_deq.float()).abs()
print(f"logits fp16 vs 量化本身:  max={dq.max().item():.4f}  mean={dq.mean().item():.5f}")

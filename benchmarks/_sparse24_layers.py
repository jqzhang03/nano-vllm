"""2:4 稀疏逐层诊断：剪枝代价（fp16 vs 剪枝稠密参考 vs 打包内核）。

三层分离：fp16 基线 → 剪枝后稠密权重（参考）→ 打包 v+idx 内核。
判定：内核 vs 剪枝参考 应≈0（存储重排）；fp16 vs 剪枝 = 一次性幅值剪枝的真实代价。
另报每层权重 Frobenius 相对误差 ||w_pruned-w||/||w||（50%权重被丢弃的质量占比）。
"""
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

PATH = "/home/zjq/huggingface/Qwen3-0.6B/"
DEV = "cuda"

dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
torch.cuda.set_device(0)

torch.manual_seed(0)
hf = AutoConfig.from_pretrained(PATH)
model = Qwen3ForCausalLM(hf).to(DEV, dtype=hf.dtype)
load_model(model, PATH)
model.eval()

ids = torch.randint(0, 50000, (8, 256), device=DEV).flatten()
positions = torch.arange(256, device=DEV).repeat(8)


def run(model):
    from nanovllm.utils.context import set_context, reset_context
    T = ids.numel()
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    set_context(True, cu, cu, T, T, torch.tensor([], dtype=torch.int32, device=DEV), None, None)
    outs = {}
    hooks = []
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
logits16, hid16 = run(model)

from nanovllm.layers.linear import LinearBase
linears = [m for m in model.modules() if isinstance(m, LinearBase)]

# 权重剪枝质量统计（用同一剪枝函数）
def prune_2_4(w):
    w4 = w.view(*w.shape[:-1], w.shape[-1] // 4, 4)
    keep = w4.abs().argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(w4, dtype=torch.bool).scatter_(-1, keep, True)
    return (w4 * mask).view(w.shape)

rel_errs = []
for i, m in enumerate(linears):
    wp = prune_2_4(m.weight.detach())
    rel = (wp.float() - m.weight.float()).norm() / m.weight.float().norm()
    rel_errs.append(rel.item())
    m.w_pruned = wp.contiguous()
print(f"权重 Frobenius 相对误差（2:4剪枝丢弃质量）: min={min(rel_errs):.4f} "
      f"mean={sum(rel_errs) / len(rel_errs):.4f} max={max(rel_errs):.4f}")

# 剪枝稠密参考（把剪枝后的稠密权重装回去走稠密路径）
for m in linears:
    m.weight = torch.nn.Parameter(m.w_pruned)
    del m.w_pruned
print("== 剪枝稠密参考（无内核） ==")
logits_p, hid_p = run(model)

# 打包内核路径（对已剪枝权重再quantize_sparse24是幂等的）
for m in linears:
    m.quantize_sparse24()
print("== 打包内核路径 ==")
logits_k, hid_k = run(model)

for name in hid16:
    dk = (hid16[name] - hid_k[name]).abs()
    print(f"{name:<30}  sparse24 max_abs={dk.max().item():.4f}  mean_abs={dk.mean().item():.5f}")

dl = (logits16.float() - logits_k.float()).abs()
print(f"\nlogits fp16 vs 2:4:    max={dl.max().item():.4f}  mean={dl.mean().item():.5f}")
dk = (logits_k.float() - logits_p.float()).abs()
print(f"logits 内核 vs 剪枝参考: max={dk.max().item():.6f}  mean={dk.mean().item():.7f}  (应≈0)")

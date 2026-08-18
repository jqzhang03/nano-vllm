"""Per-layer hidden-state drift: fp16 vs W8A8 quantization."""
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

def run(model, capture):
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

logits16, hid16 = run(model, True)

# 校准：采集每个线性层输入的逐通道amax（SmoothQuant的s向量）
from nanovllm.layers.linear import LinearBase
linears = [m for m in model.modules() if isinstance(m, LinearBase)]
hooks = []
for m in linears:
    m.x_max = None
    def make_hook(mod):
        def hook(_mod, args):
            amax = args[0].float().abs().amax(dim=0)
            if mod.x_max is None:
                mod.x_max = amax.cpu()
            else:
                mod.x_max = torch.maximum(mod.x_max, amax.cpu())
        return hook
    hooks.append(m.register_forward_pre_hook(make_hook(m)))
with torch.inference_mode():
    from nanovllm.utils.context import set_context, reset_context
    T = ids.numel()
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    set_context(True, cu, cu, T, T, torch.tensor([], dtype=torch.int32, device=DEV), None, None)
    model(ids, positions)
    reset_context()
for h in hooks:
    h.remove()

# quantize with smoothing
for m in linears:
    m.quantize_w8a8(m.x_max)
    del m.x_max

logits8, hid8 = run(model, False)

for name in hid16:
    d = (hid16[name] - hid8[name]).abs()
    rel = d / (hid16[name].abs() + 1e-3)
    print(f"{name:<30} layer{hid16[name].shape[1]:>4d}  max_abs={d.max().item():.4f}  mean_abs={d.mean().item():.5f}  mean_rel={rel.mean().item():.5f}")

dl = (logits16.float() - logits8.float()).abs()
print(f"logits max_abs={dl.max().item():.4f} mean_abs={dl.mean().item():.5f}")

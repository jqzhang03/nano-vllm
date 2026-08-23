"""诊断：Medusa训练循环为什么每步7.3s（预期~30ms）。逐段计时。

用法（WSL，GPU）：
    python benchmarks/_medusa_train_probe.py
"""
import os
import sys
import time

import torch

MODEL = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B/")
from nanovllm.layers.medusa import MedusaHeads

torch.backends.cuda.matmul.allow_tf32 = True

hf_vocab = 151936
hidden_dim = 1024
gamma = 4
n = 18427
B = 512

heads = MedusaHeads(gamma + 1, hidden_dim, 256, hf_vocab).cuda().float()
opt = torch.optim.AdamW(heads.parameters(), lr=3e-3, weight_decay=0.01)
hidden = torch.randn(n, hidden_dim, device="cuda")
labels = [torch.randint(0, 1000, (n,), device="cuda") for _ in range(gamma + 1)]


def bench(name, fn, iters=3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    print(f"{name:<28} {(time.perf_counter() - t0) / iters * 1000:8.1f} ms/次")


def batch_cpu():
    return torch.randint(0, n, (B,), generator=torch.Generator().manual_seed(1))


def batch_gpu():
    return torch.randint(0, n, (B,), device="cuda")


def gather(idx):
    h = hidden[idx]
    ys = [lab[idx] for lab in labels]
    return h, ys


def forward(h):
    loss = torch.tensor(0.0, device="cuda")
    for k, hd in enumerate(heads.heads):
        loss = loss + torch.nn.functional.cross_entropy(hd(h), labels[k][:B])
    return loss


def full_step(idx):
    h = hidden[idx]
    loss = torch.tensor(0.0, device="cuda")
    for k, hd in enumerate(heads.heads):
        loss = loss + torch.nn.functional.cross_entropy(hd(h), labels[k][idx])
    opt.zero_grad()
    loss.backward()
    opt.step()


# 预热（含首次kernel编译）
idx = batch_gpu()
full_step(idx)
torch.cuda.synchronize()

print("== 逐段（预热后） ==")
bench("randint CPU generator", batch_cpu)
bench("randint CUDA", batch_gpu)
idx = batch_gpu()
h = hidden[idx]
bench("gather hidden+labels", lambda: gather(idx))
bench("forward 4 heads + CE", lambda: forward(h))
bench("backward", lambda: forward(h).backward())
bench("AdamW step", lambda: (opt.zero_grad(), forward(h).backward(), opt.step()))
bench("完整一步 (GPU batch)", lambda: full_step(batch_gpu()))
idx_cpu = batch_cpu()
bench("完整一步 (CPU batch索引)", lambda: full_step(idx_cpu))

# 首次（冷）调用成本
print("\n== 冷启动（新形状/新优化器状态） ==")
heads2 = MedusaHeads(gamma + 1, hidden_dim, 256, hf_vocab).cuda().float()
opt2 = torch.optim.AdamW(heads2.parameters(), lr=3e-3)
t0 = time.perf_counter()
full_step(idx)
torch.cuda.synchronize()
print(f"冷完整一步: {(time.perf_counter() - t0) * 1000:.0f} ms")

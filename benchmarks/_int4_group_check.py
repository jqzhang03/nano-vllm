import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F

from nanovllm.layers.linear import ColumnParallelLinear


def check(group):
    g = torch.Generator(device="cuda").manual_seed(1)
    N, K = 512, 1088 if group == 64 else 1024  # 1088 = 64×17（非128倍数）
    lin = ColumnParallelLinear(K, N).cuda()
    lin.weight.data.normal_(0, 0.05, generator=g)
    w = lin.weight.detach().float()
    lin.quantize_int4(group_size=group, dense_path=True)  # w_deq = 反量化
    x = torch.randn(37, K, device="cuda") * 0.3
    y = lin(x)  # int4 内核（M=37 小）
    ref = F.linear(x, lin.w_deq)
    d = (y.float() - ref).abs()
    rel = d.max().item() / max(ref.abs().max().item(), 1e-6)
    print(f"group {group}: int4 内核 vs w_deq 稠密 max abs {d.max().item():.2e} "
          f"rel {rel:.2e}")
    assert rel < 1e-3, "int4 内核 vs 反量化不一致"
    # 大 M 也应一致（prefill 走 w_deq 本身）
    xb = torch.randn(513, K, device="cuda") * 0.3
    yb = lin(xb)
    rb = F.linear(xb, lin.w_deq)
    db = (yb.float() - rb).abs()
    assert db.max().item() < 1e-3 * max(rb.abs().max().item(), 1e-6) + 1e-2
    print("group", group, "OK")


def check64_big():
    g = torch.Generator(device="cuda").manual_seed(2)
    N, K = 2048, 10944  # V2-Lite dense down_proj K（64×171，非128倍数）
    lin = ColumnParallelLinear(K, N).cuda()
    lin.weight.data.normal_(0, 0.05, generator=g)
    lin.quantize_int4(group_size=64, dense_path=True)
    x = torch.randn(2, K, device="cuda", dtype=torch.bfloat16) * 0.3
    y = lin(x)
    ref = F.linear(x.float(), lin.w_deq.float())
    d = (y.float() - ref).abs()
    print(f"K=10944 group64: max {d.max().item():.2e} "
          f"(含 NaN {int(torch.isnan(d).sum().item())})")
    assert not torch.isnan(d).any() and d.max().item() < 0.1
    print("10944-tail OK")


check(128)
check(64)
check64_big()
print("INT4 GROUP-64 KERNEL OK")

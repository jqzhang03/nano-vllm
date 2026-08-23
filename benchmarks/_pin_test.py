"""WSL2 pinned 内存压力测试：swap 缓冲同款形状批量分配是否导致 VM 崩溃。"""
import torch

bufs = []
total = 0
try:
    for i in range(50):
        b = torch.empty(2, 28, 3, 256, 8, 128, dtype=torch.bfloat16, pin_memory=True)
        bufs.append(b)
        total += b.numel() * b.element_size() / 1e6
        if i % 10 == 0:
            print(f"allocated {i + 1} bufs = {total:.0f} MB pinned", flush=True)
    print(f"OK: {len(bufs)} bufs = {total:.0f} MB pinned")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

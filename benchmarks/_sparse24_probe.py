"""2:4 稀疏探针 v2：CUTLASS vs cuSPARSELt 后端在 sm_120 的正确性/性能/图捕获。

背景：v1 显示 cuSPARSELt 后端在 MLP 层（4096x1024）只有稠密 0.12-0.18×（每调用开销），
仅 lm_head 小 M 有 1.7×；且 CUDA graph 重放 bit-exact=False。torch 2.8 新增 CUTLASS
后端（Blackwell 专用），本探针对比两后端并列出可用算子。
"""
import time

import torch
import torch.nn.functional as F
import torch.sparse.semi_structured as ss
from torch.sparse import to_sparse_semi_structured

torch.manual_seed(0)
print("torch", torch.__version__, "| capability", torch.cuda.get_device_capability())
print("semi_structured submodule:", [n for n in dir(ss) if not n.startswith("_")])
print("CUTLASS class:", torch.sparse.SparseSemiStructuredTensorCUTLASS)
print("CUSPARSELT class:", torch.sparse.SparseSemiStructuredTensorCUSPARSELT)


def prune_2_4(w):
    """2:4 幅值剪枝（组内保留最大 2 个）。"""
    w4 = w.view(*w.shape[:-1], w.shape[-1] // 4, 4)
    keep = w4.abs().argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(w4, dtype=torch.bool).scatter_(-1, keep, True)
    return (w4 * mask).view(w.shape)


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def make_sparse(w, backend):
    wp = prune_2_4(w)
    if backend == "cutlass":
        return torch.sparse.SparseSemiStructuredTensorCUTLASS.from_dense(wp)
    if backend == "cusparselt":
        return torch.sparse.SparseSemiStructuredTensorCUSPARSELT.from_dense(wp)
    return to_sparse_semi_structured(wp)


print("\n=== 后端正确性 + 性能 ===")
for dtype in (torch.bfloat16, torch.float16):
    for name, N, K in [("gate_up 4096x1024", 4096, 1024), ("lm_head 151936x1024", 151936, 1024)]:
        w = (torch.randn(N, K, device="cuda") * 0.02).to(dtype)
        for backend in ("cutlass", "cusparselt"):
            try:
                ws = make_sparse(w, backend)
                for M in (8, 256):
                    x = (torch.randn(M, K, device="cuda") * 0.5).to(dtype)
                    y = F.linear(x, ws)
                    ref = F.linear(x, prune_2_4(w))
                    err = (y.float() - ref.float()).abs().max().item()
                    t_dense = bench(lambda: F.linear(x, w))
                    t_sp = bench(lambda: F.linear(x, ws))
                    print(f"[{dtype} {name} M={M} {backend}] max|Δ|={err:.4f}  "
                          f"dense {t_dense:.3f}ms  sparse {t_sp:.3f}ms  speedup {t_dense / t_sp:.2f}x")
            except Exception as e:
                print(f"[{dtype} {name} {backend}] FAIL {type(e).__name__}: {str(e)[:160]}")

print("\n=== CUDA graph 捕获（CUTLASS） ===")
try:
    N, K = 4096, 1024
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    ws = make_sparse(w, "cutlass")
    x = (torch.randn(8, K, device="cuda") * 0.5).to(torch.bfloat16)
    for _ in range(3):
        F.linear(x, ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = F.linear(x, ws)
    y_ref = F.linear(x, ws)
    torch.cuda.synchronize()
    # 多次重放看确定性
    det = all(torch.equal(y, F.linear(x, ws)) for _ in range(5))
    print(f"CUDA graph capture: OK  replay bit-exact vs eager={torch.equal(y, y_ref)}  "
          f"replay deterministic={det}")
except Exception as e:
    print(f"CUDA graph capture: FAIL {type(e).__name__}: {str(e)[:200]}")

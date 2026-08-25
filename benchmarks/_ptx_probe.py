"""探测 Triton 3.x CompiledKernel 的属性 + 从 PTX 统计寄存器/溢出（n_regs 替代）。

结论用途：内核调优的"寄存器压力"证据（哪些配置 spill、占用率如何）。
"""
import re

import torch
import triton
import triton.language as tl


@triton.jit
def k_simple(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x * 2.0)


@triton.jit
def k_gemm(a_ptr, b_ptr, c_ptr, M, N, K,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def ptx_stats(ptx: str):
    """从 PTX 统计：寄存器号上限（近似寄存器数）、.local 访问（溢出证据）。"""
    regs = 0
    for m in re.finditer(r"\.reg \.\w+ \t?%r<(\d+)>", ptx):
        regs = max(regs, int(m.group(1)))
    # 也统计其它命名空间（%f<..> 浮点、%rs 等）
    for m in re.finditer(r"\.reg \.\w+ \t?%r<(\d+)>", ptx):
        regs = max(regs, int(m.group(1)) + 1)
    spills = len(re.findall(r"ld\.local|st\.local", ptx))
    return regs, spills


def main():
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    o = torch.empty_like(x)
    kern = k_simple.warmup(x, o, BLOCK=1024, grid=(1,))
    print("CompiledKernel attrs:", [a for a in dir(kern) if not a.startswith("__")])
    print("has n_regs:", hasattr(kern, "n_regs"), "| has asm:", hasattr(kern, "asm"))
    try:
        asm_keys = list(kern.asm.keys()) if hasattr(kern, "asm") else []
        print("asm keys:", asm_keys)
    except Exception as e:
        print("asm access err:", e)
    try:
        ptx = kern.asm["ptx"]
        regs, spills = ptx_stats(ptx)
        print(f"simple kernel: ptx regs~{regs} spills~{spills}")
    except Exception as e:
        print("ptx err:", e)

    # GEMM kernel：对比 w4 vs w8（寄存器压力差异）
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    for bm, bn, bk, w in ((64, 128, 128, 4), (64, 128, 128, 8), (128, 128, 64, 4)):
        try:
            kern = k_gemm.warmup(a, b, c, M, N, K, BM=bm, BN=bn, BK=bk,
                                 num_warps=w, num_stages=2, grid=(1,))
            ptx = kern.asm["ptx"]
            regs, spills = ptx_stats(ptx)
            print(f"gemm BM{bm} BN{bn} BK{bk} w{w}: ptx regs~{regs} spills={spills}")
        except Exception as e:
            print(f"gemm BM{bm} BN{bn} BK{bk} w{w}: err {e}")


if __name__ == "__main__":
    main()

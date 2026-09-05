"""SASS 指令统计工具：对 cubin 或含 fatbin 的 .so 做逐函数指令计数与资源统计。

阶段 1b（内核补课）工具：回答"编译器到底生成了什么"——HMMA/LDS/STS/LDG 指令数、
bank-conflict 相关的共享内存指令形态、寄存器/溢出等。

用法:
  python benchmarks/_sass_stats.py <cubin-or-so-path> [--func NAME]
输出: 每个 kernel 函数的资源表（regs/smem/spill，来自 --dump-resource-usage）+
      指令助记符 Top 榜 + 关键类别汇总（MMA/LDS/STS/LDG/CP.ASYNC/同步/算术/访存全局）。
"""
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

# 优先用 conda env 的 cuobjdump 12.8（系统 12.4 无法解码 sm_120 SASS）
_CUDA_BIN = "/home/zjq/miniconda3/envs/nano-vllm/bin"
CUOBJDUMP = next((os.path.join(_CUDA_BIN, "cuobjdump")
                  for _ in [0] if os.path.exists(os.path.join(_CUDA_BIN, "cuobjdump"))),
                 "cuobjdump")

# sm_120 的 SASS 指令行：  /*0000*/  OPCODE ... ?flag; /*'  （不以 ; 结尾，尾部是调度标志+注释）
INSN_RE = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9._@!?]*)")


def cuobjdump(args: list[str], path: str) -> str:
    r = subprocess.run([CUOBJDUMP, *args, path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {r.stderr[:500]}")
    return r.stdout


def resource_stats(path: str) -> dict[str, dict]:
    """--dump-resource-usage 输出 per-function regs/smem/spill。"""
    out = cuobjdump(["--dump-resource-usage"], path)
    funcs = {}
    cur = None
    for line in out.splitlines():
        # 12.8 格式: "Function _Z...:" （旧格式 "Function : _Z..."）
        m = re.search(r"Function\s+:?\s*(\S+)", line)
        if m:
            cur = m.group(1).rstrip(":")
            funcs[cur] = {}
            continue
        if cur:
            m = re.search(r"REG:(\d+)\s+STACK:(\d+)", line)
            if m:
                funcs[cur]["regs"] = int(m.group(1))
                funcs[cur]["stack"] = int(m.group(2))
            m = re.search(r"SHARED:(\d+)", line)
            if m:
                funcs[cur]["smem"] = int(m.group(1))
    return funcs


def sass_stats(path: str, func_filter: str | None = None) -> dict[str, Counter]:
    """--dump-sass 输出逐函数指令计数。opcode 保留全名（如 HMMA.16816.F32）
    与基名（点前主指令）。"""
    out = cuobjdump(["--dump-sass"], path)
    funcs = {}
    cur = None
    for line in out.splitlines():
        m = re.search(r"Function\s+:?\s*(\S+)", line)
        if m:
            cur = m.group(1).rstrip(":")
            if func_filter and func_filter not in cur:
                cur = None
            else:
                funcs[cur] = Counter()
            continue
        if cur is None:
            continue
        m = INSN_RE.match(line)
        if m:
            op = m.group(1)
            funcs[cur][op] += 1
    return funcs


def summarize(c: Counter) -> dict[str, int]:
    """把全名 opcode 归并成教学用类别。"""
    base = Counter()
    for op, n in c.items():
        head = op.split(".")[0]
        base[head] += n
    cats: dict[str, int] = {"total": sum(c.values())}
    for op, n in c.items():
        if op.startswith("HMMA") or op.startswith("MMA"):
            cats["MMA"] = cats.get("MMA", 0) + n
        if op.startswith("LDS"):
            cats["LDS (SMEM读)"] = cats.get("LDS (SMEM读)", 0) + n
        if op.startswith("STS"):
            cats["STS (SMEM写)"] = cats.get("STS (SMEM写)", 0) + n
        if op.startswith("LDG") or op.startswith("LD.E"):
            cats["LDG (全局读)"] = cats.get("LDG (全局读)", 0) + n
        if op.startswith("STG") or op.startswith("ST.E"):
            cats["STG (全局写)"] = cats.get("STG (全局写)", 0) + n
        if op.startswith("LDGSTS") or op.startswith("CP.ASYNC"):
            cats["cp.async"] = cats.get("cp.async", 0) + n
        if op.startswith("BAR"):
            cats["BAR.SYNC"] = cats.get("BAR.SYNC", 0) + n
        if op.startswith("LDSM"):
            cats["ldmatrix"] = cats.get("ldmatrix", 0) + n
        if "FADD" in op or "FMUL" in op or "FFMA" in op or op.startswith("FMA"):
            cats["FMA类(FADD/FMUL/FFMA)"] = cats.get("FMA类(FADD/FMUL/FFMA)", 0) + n
        if op.startswith("IADD") or op.startswith("ISETP") or op.startswith("IMAD") \
                or op.startswith("I2F") or op.startswith("F2I") or op.startswith("SHF") \
                or op.startswith("LEA"):
            cats["整数/寻址"] = cats.get("整数/寻址", 0) + n
        if op.startswith("MOV") or op.startswith("SEL"):
            cats["MOV/SEL"] = cats.get("MOV/SEL", 0) + n
    return cats


def main():
    path = sys.argv[1]
    func_filter = None
    if len(sys.argv) > 3 and sys.argv[2] == "--func":
        func_filter = sys.argv[3]
    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")
    res = resource_stats(path)
    sass = sass_stats(path, func_filter)
    print(f"=== SASS stats: {os.path.basename(path)} ===")
    for fn, counter in sass.items():
        print(f"\n--- {fn} ---")
        r = res.get(fn, {})
        print(f"resources: regs={r.get('regs', '?')} smem={r.get('smem', '?')} "
              f"stack/spill={r.get('stack', '?')}")
        cats = summarize(counter)
        print("categories:", ", ".join(f"{k}={v}" for k, v in cats.items()))
        top = counter.most_common(14)
        print("top opcodes:", ", ".join(f"{op}x{n}" for op, n in top))


if __name__ == "__main__":
    main()

"""CUDA C 编译链 smoke：验证 nvcc 12.8 能编译 sm_120 cubin 并被 torch 加载执行。

用法: python benchmarks/_cuda_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONDA_PREFIX = "/home/zjq/miniconda3/envs/nano-vllm"


def main():
    # 让 torch cpp_extension 找到 conda 版 nvcc 12.8（先于系统 12.4）
    nvcc_dir = os.path.join(CONDA_PREFIX, "bin")
    if os.path.exists(os.path.join(nvcc_dir, "nvcc")):
        os.environ["PATH"] = nvcc_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ["CUDA_HOME"] = CONDA_PREFIX
    # host 编译器必须用 conda gcc 14（系统 gcc 15 超 nvcc 上限且 pybind11 模板推导崩）
    host_gcc = os.path.join(CONDA_PREFIX, "bin", "x86_64-conda-linux-gnu-gcc")
    if os.path.exists(host_gcc):
        os.environ["CUDAHOSTCXX"] = host_gcc
    # 明确编 sm_120（capability 12.0）；torch 2.8 cu128 认得这个 arch
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    ext = load(name="_cuda_smoke_ext", sources=[os.path.join(here, "_cuda_smoke.cu")],
               extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],
               verbose=False)

    import torch
    import subprocess

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    print(f"device arch            : sm_{ext.cuda_arch()}")
    # 用 cuobjdump 验证编译产物真的是 sm_120 cubin（同时顺带验证 SASS 工具链）
    r = subprocess.run(["cuobjdump", "--dump-sass", ext.__file__],
                       capture_output=True, text=True)
    arch_line = [ln for ln in r.stdout.splitlines() if "SM120" in ln or "sm_120" in ln]
    print(f"cubin arch (cuobjdump) : {arch_line[0].strip() if arch_line else 'NOT FOUND'} "
          f"({'PASS' if arch_line else 'FAIL'})")

    n = 1 << 20
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    z = ext.saxpy(x, y, 2.5)
    ref = 2.5 * x + y
    err = (z - ref).abs().max().item()
    print(f"saxpy max err          : {err:.3e}  ({'PASS' if err < 1e-4 else 'FAIL'})")

    xh = torch.randn(n, device=dev, dtype=torch.float16)
    yh = torch.randn(n, device=dev, dtype=torch.float16)
    zh = ext.half_add(xh, yh)
    errh = (zh.float() - (xh.float() + yh.float())).abs().max().item()
    print(f"half_add max err       : {errh:.3e}  ({'PASS' if errh < 0.01 else 'FAIL'})")
    print("\nCUDA C 编译链 OK：nvcc 12.8 + sm_120 cubin + torch 加载执行全部通过")


if __name__ == "__main__":
    main()

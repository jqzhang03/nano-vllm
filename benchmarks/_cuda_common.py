"""CUDA C 开发公共环境：conda nvcc 12.8 + gcc14 + sm_120 编译配置。

本机工具链要点（踩坑记录，见 note.md 候选故事）：
- pip 的 nvidia-cuda-nvcc-cu12 wheel 只有 ptxas 没有 nvcc 主程序（拆包）→
  用 conda: `conda install -n nano-vllm -c nvidia cuda-nvcc=12.8.93`
- 系统 gcc 15.2 超 nvcc 12.8 上限（host_config.h #error）且 pybind11 模板推导崩 →
  conda gcc 14.3（x86_64-conda-linux-gnu-gcc）；CUDAHOSTCXX 不生效 → 手动建
  env/bin/gcc、g++ symlink 并前置 PATH
- torch cpp_extension 需要 ninja
- sm_120 需要 TORCH_CUDA_ARCH_LIST=12.0（torch 2.8 cu128 认得）
"""
import os

CONDA_PREFIX = "/home/zjq/miniconda3/envs/nano-vllm"


def ensure_cuda_env():
    """设置 PATH/CUDA_HOME，让 nvcc 12.8 与 gcc 14 被 torch cpp_extension 找到。"""
    nvcc_dir = os.path.join(CONDA_PREFIX, "bin")
    if os.path.exists(os.path.join(nvcc_dir, "nvcc")):
        os.environ["PATH"] = nvcc_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ["CUDA_HOME"] = CONDA_PREFIX
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"


EXTRA_FLAGS = ["-O3", "-allow-unsupported-compiler"]


def load_ext(name, cu_path, extra_flags=None):
    """编译 CUDA 扩展（缓存于 ~/.cache/torch_extensions）。"""
    ensure_cuda_env()
    from torch.utils.cpp_extension import load

    return load(name=name, sources=[cu_path],
                extra_cuda_cflags=EXTRA_FLAGS + (extra_flags or []),
                verbose=False)

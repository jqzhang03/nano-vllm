"""下载 HF 模型到本地（新版 huggingface_hub 弃用了 huggingface-cli 子命令，用 API 更稳）。

用法: python benchmarks/_hf_download.py <repo_id> <local_dir>

直连 huggingface.co 在本机（WSL）不通（ConnectTimeout），走 hf-mirror.com 镜像
（HF_ENDPOINT 必须在 import huggingface_hub 之前设置——常量在模块导入时求值）。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 镜像不代理 xet（xethub.hf.co）CAS 服务器 → 禁用 Xet 走经典 HTTP 分块下载
# （否则大文件报 401 Unauthorized: cas-server.xethub.hf.co）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import sys  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402

repo = sys.argv[1]
dest = os.path.expanduser(sys.argv[2])
os.makedirs(os.path.dirname(dest), exist_ok=True)
print(f"downloading {repo} -> {dest} (endpoint={os.environ['HF_ENDPOINT']})", flush=True)
snapshot_download(repo_id=repo, local_dir=dest)
print("done", flush=True)

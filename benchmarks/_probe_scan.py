import sys, os
sys.path.insert(0, '/mnt/d/Project/nano-vllm/benchmarks')
from _cuda_common import load_ext
import torch
import numpy as np

ext = load_ext('_cuda_gemm_dev_ext', os.path.abspath('/mnt/d/Project/nano-vllm/benchmarks/_cuda_gemm_dev.cu'))
M, N = 64, 128
c = torch.full((M, N), -1.0, device='cuda', dtype=torch.float16)
ext.store_probe(c)
c = c.cpu().numpy()
written = (c != -1)
for r0 in range(0, 64, 8):
    line = ''
    for c0 in range(0, 128, 8):
        blk = written[r0:r0 + 8, c0:c0 + 8]
        vals = np.unique(c[r0:r0 + 8, c0:c0 + 8])
        tag = 'W' if blk.any() else '.'
        line += f'[{r0 // 8},{c0 // 8}]:{tag}{vals[:3].tolist() if blk.any() else ""} '
    print(line)

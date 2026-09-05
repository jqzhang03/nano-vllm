import sys
print("PY", sys.version[:6])
import torch
print("CUDA", torch.cuda.is_available())
print("ENVCHECK_OK")

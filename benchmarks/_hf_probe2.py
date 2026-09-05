import inspect
import sys
sys.path.insert(0, '/mnt/d/Project/nano-vllm/benchmarks')
from transformers.models.qwen3_moe import modeling_qwen3_moe as MM

for cls_name in ['Qwen3MoeDecoderLayer', 'Qwen3MoeAttention', 'Qwen3MoeMLP']:
    cls = getattr(MM, cls_name)
    print(f'\n===== {cls_name} =====')
    src = inspect.getsource(cls)
    print(src[:5200])

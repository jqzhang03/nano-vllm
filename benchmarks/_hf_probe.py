import sys
import inspect
sys.path.insert(0, '/mnt/d/Project/nano-vllm/benchmarks')
import transformers

names = ['Qwen3MoeConfig', 'Qwen3MoeForCausalLM', 'Qwen3MoeDecoderLayer', 'Qwen3MoeConfig']
for n in names:
    try:
        getattr(transformers, n)
        print('have', n)
    except Exception:
        print('MISSING', n)

from transformers.models.qwen3_moe import modeling_qwen3_moe as M
import transformers.models.qwen3_moe.modeling_qwen3_moe as MM

c = M.Qwen3MoeConfig()
print('\nmodel_type:', c.model_type)
for k in ['num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
          'shared_expert_intermediate_size', 'norm_topk_prob', 'hidden_act',
          'intermediate_size', 'attention_bias', 'head_dim', 'rope_theta',
          'tie_word_embeddings', 'max_position_embeddings', 'num_hidden_layers',
          'num_attention_heads', 'num_key_value_heads', 'hidden_size', 'vocab_size',
          'rms_norm_eps']:
    print(f'  {k} = {getattr(c, k, "N/A")}')

print('\n--- module classes ---')
for n in dir(MM):
    if 'Moe' in n or 'Router' in n or 'Expert' in n:
        print(' ', n)

# 关键语义：experts 路由/共享专家处理
for cls_name in ['Qwen3MoeExperts', 'Qwen3MoeTopKRouter']:
    cls = getattr(MM, cls_name, None)
    if cls is None:
        continue
    print(f'\n===== {cls_name} =====')
    print(inspect.getsource(cls)[:6000])

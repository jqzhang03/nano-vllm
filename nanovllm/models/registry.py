"""模型注册表：按 HuggingFace config.model_type 分发到对应模型类。

新模型适配的入口（见 LEARNING.md 阶段7）：
  1. 在 models/ 下新建 <family>.py（以 qwen3.py/qwen2.py 为模板）；
  2. 在 _MODEL_REGISTRY 里登记 model_type → 模型类；
  3. 引擎侧（model_runner.py）只通过 get_model_class(model_type) 构造，
     不再硬编码具体模型。
"""
from typing import Type

from torch import nn

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.llama3 import Llama3ForCausalLM
from nanovllm.models.mistral import MistralForCausalLM
from nanovllm.models.gemma2 import Gemma2ForCausalLM

# model_type（HF config.model_type）→ 模型类。
# 已支持：qwen3 / qwen2（Qwen2.5 系列同属 qwen2）/ llama（Llama-3.x）/
# mistral（Mistral-7B-v0.x，滑动窗口注意力 SWA）/ gemma2（Gemma-2，交替
# local/global 注意力 + attn logit soft-cap + final logit soft-cap）。
_MODEL_REGISTRY: dict[str, Type[nn.Module]] = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "llama": Llama3ForCausalLM,
    "mistral": MistralForCausalLM,
    "gemma2": Gemma2ForCausalLM,
}

# 规划中的模型：model_type → 未实现的具体卡点（见 LEARNING.md 阶段7 卡点清单）。
# 构造时给出可操作的报错（指明缺什么），而不是笼统的 "unsupported"。
_PLANNED_BLOCKERS: dict[str, str] = {
    "mixtral": "Mixtral 端口未实现：MoE 层（router + top-k + expert FFN + load-balancing "
               "aux loss + 专家并行分片）未做——见 LEARNING.md 阶段7 MoE 卡点。",
}

# 常见框架卡点（跨模型）：
# - rotary_embedding.get_rope 不支持 rope_scaling（YaRN/linear/dynamic）——长上下文扩展
#   模型（如 Llama-3.2、Mistral-Nemo）需要先实现。
# - 流式加载限制：int4 强制纯 int4、w8a8 无 SmoothQuant、awq 仅预生成 scales。


def get_model_class(model_type: str) -> Type[nn.Module]:
    """按 HF model_type 返回模型类；未实现/未知类型给出可操作的报错。"""
    if model_type in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_type]
    if model_type in _PLANNED_BLOCKERS:
        raise NotImplementedError(
            f"model_type {model_type!r} 已列入规划但未实现：{_PLANNED_BLOCKERS[model_type]}"
        )
    raise ValueError(
        f"unsupported model_type {model_type!r}; supported: {sorted(_MODEL_REGISTRY)} | "
        f"planned: {sorted(_PLANNED_BLOCKERS)}"
    )

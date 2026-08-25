import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Gemma2Config

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, ColumnParallelLinear, RowParallelLinear, tp_size
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


def gemma2_layer_types(config: Gemma2Config, num_layers: int) -> list[str]:
    """每层的注意力类型："global"（全注意力）| "sliding"（SWA，窗口=config.sliding_window）。

    与安装版 transformers 的行为逐一对齐（parity 基准，benchmarks/_gemma2_lt.py 实证）：
      1. config 显式带 layer_types（新 transformers 字段）→ 直接使用（归一化
         "sliding_attention"→"sliding"、"full_attention"→"global"、"local"→"sliding"）；
      2. 只有旧式 attention_types（google 原始 9B 的 [[["global","local"],5]] 格式，
         两种写法都兼容）→ 展开类型列表 × 重复次数；层数超出取最后一项
         （9B 的 42 层 = [global,local]×5 后全部 local）；
      3. 两者都缺（**unsloth 的 2B / 9B-it 转换版就是这种**）→ 安装版 transformers 的
         Gemma2Config.__post_init__ 默认：layer i 为 "sliding_attention" if (i+1)%2
         else "full_attention"，即 **sliding 开头交替**（13/13、21/21 实证）——必须复刻
         这个默认，否则与 HF 参考不一致（parity top-1 塌掉）。
    """
    def norm(t: str) -> str:
        return {"sliding_attention": "sliding", "local": "sliding",
                "full_attention": "global", "global": "global"}.get(t, "global")

    lt = getattr(config, "layer_types", None)
    if isinstance(lt, list) and lt:
        types = [norm(t) for t in lt]
        if len(types) < num_layers:
            types += [types[-1]] * (num_layers - len(types))
        return types[:num_layers]
    at = getattr(config, "attention_types", None)
    if at:
        pairs = []
        try:
            pairs = list(zip(at[0], at[1]))
        except (IndexError, TypeError):
            pairs = [(t, r) for t, r in at]
        types: list[str] = []
        for type_list, repeats in pairs:
            for _ in range(repeats):
                types.extend(type_list)
        types = [norm(t) for t in types]
        if len(types) < num_layers:
            types += [types[-1]] * (num_layers - len(types))
        return types[:num_layers]
    return ["sliding" if (i + 1) % 2 else "global" for i in range(num_layers)]


# Gemma-2 注意力：head_dim=256（config 显式）、GQA、**logit soft-cap**（attn_logit_softcapping，
# flash-attn softcap 参数：内核内 cap·tanh(logits/cap) 后 softmax）、交替 local/global
# 层（local 层传 sliding_window → SWA 窗口掩码）。无 QK-Norm、无 bias。
# 缩放：config.query_pre_attn_scalar^-0.5（2B 为 256 → 1/16，等于 head_dim^-0.5；
# 9B 无此字段 → 回落 head_dim^-0.5）。
class Gemma2Attention(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
        layer_type: str,
    ) -> None:
        super().__init__()
        ts = tp_size()  # 进程组未初始化（裸模型/CPU 测试）时按 1
        hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % ts == 0
        self.num_heads = self.total_num_heads // ts
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % ts == 0
        self.num_kv_heads = self.total_num_kv_heads // ts
        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        scalar = getattr(config, "query_pre_attn_scalar", None) or self.head_dim
        self.scaling = scalar ** -0.5

        # QKV 合并投影（checkpoint 是独立 q/k/v 张量，weight_loader 按 packed 映射分片）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
        )
        window = config.sliding_window if layer_type == "sliding" else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            window_size=window,
            logit_softcapping=getattr(config, "attn_logit_softcapping", None),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


# Gemma-2 MLP：gate/up/down **独立**投影（区别于 qwen/llama 的 gate_up 合并），
# 激活 = gelu_pytorch_tanh（F.gelu approximate="tanh"）——非 silu。
class Gemma2MLP(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
    ) -> None:
        super().__init__()
        # transformers 4.x 用 hidden_act、5.x 改名为 hidden_activation——都兼容
        act = getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", None)
        assert act in ("gelu_pytorch_tanh", "gelu"), f"unsupported gemma2 activation {act!r}"
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma2DecoderLayer(nn.Module):
    # Gemma-2 特有：**四个 RMSNorm + 层内两个残差加**（区别于 llama/qwen 的
    # "两个 norm + 单残差"）：
    #   x1 = ln2(attn(ln1(x0))) + x0      ← 块A（attention 后残差加）
    #   out = x1 + ln4(mlp(ln3(x1)))      ← 块B（feedforward 后残差加）
    # 融合残差模式（add_rms_forward）表达不了"第二个残差基 = x1 而非 x0"——
    # 这里每层自包含：x0 = 上一层完整输出，全部用纯 norm + 显式加法，
    # 返回 (out, None)（下一层重新以自身输入为 x0；最终 norm 也是纯 norm）。
    # 漏掉 pre/post_feedforward_layernorm 会在加载时报错 + 输出错乱。
    def __init__(
        self,
        config: Gemma2Config,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma2Attention(config, layer_type)
        self.mlp = Gemma2MLP(config)
        # Gemma-2 全部 RMSNorm 用 (1+weight) 残差式缩放（weight_offset=True）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, weight_offset=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, weight_offset=True)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, weight_offset=True)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, weight_offset=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x0 = hidden_states
        h = self.input_layernorm(x0)                 # ln1(x0)
        h = self.self_attn(positions, h)             # attn(ln1(x0))
        h = self.post_attention_layernorm(h)         # ln2
        x1 = h + x0                                  # 块A残差加
        h = self.pre_feedforward_layernorm(x1)       # ln3(x1)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)       # ln4
        return x1 + h, None                          # 块B残差加；残差流置空（每层自包含）


class Gemma2Model(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        layer_types = gemma2_layer_types(config, config.num_hidden_layers)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config, layer_types[i]) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, weight_offset=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Gemma-2 的 **scaled word embedding**（HF Gemma2TextScaledWordEmbedding 同款）：
        # embed 输出 × √hidden_size（2B: √2304=48）——隐藏在源码里的细节，不加它
        # parity top-1 直接 0%（逐层 debug 定位：embed 层 diff 恒为 48×，见 note.md）。
        # 注意 lm_head 无对应缩放（仅 embed 侧）。
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states, None)
        # Gemma-2 最终 norm 是纯 norm（无残差加；层内已自包含残差）
        return self.norm(hidden_states)


class Gemma2ForCausalLM(nn.Module):
    # q/k/v 合并进 qkv_proj；gate/up 是独立模块（与 HF 同名），不进 packed 映射
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        config: Gemma2Config
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        # **final logit soft-cap**（config.final_logit_softcapping，2B/9B=30）：
        # logits = cap·tanh(logits/cap)——把 logits 压进 [-cap, cap]，HF 同款
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits

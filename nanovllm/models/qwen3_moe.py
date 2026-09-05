"""Qwen3-MoE（model_type "qwen3_moe"）—— 阶段 1.5 模型适配。

结构对齐 transformers 5.x 的 Qwen3Moe：
- 注意力与 dense Qwen3 相同（复用 qwen3.py 的 Qwen3Attention：qkv 融合 + q/k RMSNorm
  + RoPE；权重名 q_proj/k_proj/v_proj/o_proj 走同一 packed 映射）；
- MLP 按层混合：`(layer_idx+1) % decoder_sparse_step == 0` 且 num_experts>0 →
  MoE 层（layers/moe.py：experts.{i}.gate_proj/up_proj/down_proj 独立 2D，参数名与
  HF checkpoint 完全一致 → loader 零改动）；其余层为 dense（gate_proj/up_proj/
  down_proj 独立，同 HF Qwen3MoeMLP——不用 qwen3 的 merged gate_up，避免 packed
  映射把 experts 的 up_proj/gate_proj 也改写到不存在的模块上）；
- packed_modules_mapping 只含 qkv（与 gemma2 相同的"独立 gate/up"模式）；
- mlp_only_layers 例外集合照抄（这些层强制 dense）。
"""
import torch
import torch.nn.functional as F
from torch import nn

from transformers import Qwen3MoeConfig

from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.moe import MoE
from nanovllm.models.qwen3 import Qwen3Attention


def _is_moe_layer(config: Qwen3MoeConfig, layer_idx: int) -> bool:
    mlp_only = getattr(config, "mlp_only_layers", ())
    num_experts = getattr(config, "num_experts", 0)
    step = getattr(config, "decoder_sparse_step", 1)
    return (layer_idx not in mlp_only) and num_experts > 0 and \
        (layer_idx + 1) % step == 0


class Qwen3MoeMLP(nn.Module):
    """dense 层 FFN（独立 gate/up，与 HF Qwen3MoeMLP 权重名一一对应）。"""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size,
                                              bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size,
                                            bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size,
                                           bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=getattr(config, "max_position_embeddings", 32768),
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        if _is_moe_layer(config, layer_idx):
            self.mlp = MoE(
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                norm_topk_prob=getattr(config, "norm_topk_prob", False),
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # 注意：gate_proj/up_proj 不入映射（dense 层与 experts 都是独立 2D 参数，
        # 名字与 HF checkpoint 直配；merged 会与 experts 的 up_proj 冲突）
    }

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

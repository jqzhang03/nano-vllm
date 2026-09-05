"""DeepSeek-V2（model_type "deepseek_v2"）—— 阶段 2a 模型适配。

结构对齐 transformers 5.15 modeling_deepseek_v2.py（权重名直配，loader 零映射）：
- 注意力 = **MLA**（layers/attention_mla.py：潜在压缩 c_KV + decoupled rope +
  q/kv 潜在层 RMSNorm；fused cache [c_kv | k̃_pe]，576 元素/token/层）；
  self_attn 直接挂 MLAAttention（参数路径 self_attn.q_a_proj.* 等 = HF）；
- MLP 按层混合：layer_idx >= first_k_dense_replace → MoE（mlp.moe.gate router +
  mlp.moe.experts.{i}.gate_proj/up_proj/down_proj（2D 存盘布局）+
  **mlp.shared_experts**（加性共享专家 FFN，intermediate = moe_inter × n_shared，
  输入 = MoE 前残差）；其余层 dense MLP（mlp.gate_proj/up_proj/down_proj）；
- loader 段改名：HF "mlp.gate.weight" → 本实现 "mlp.moe.gate.weight"（router
  在 MoE 核心内；experts/shared_experts 段名天然直配）；
- routed scaling：top-k 概率 × routed_scaling_factor（MoE 新参数，V2 默认 1.0）；
- rope：64 维 interleaved-pairs（DeepSeek 约定）；
- 不支持：topk_method="group_limited_greedy"（V2-Lite 为 greedy）、rope_scaling。
"""
import torch
import torch.nn.functional as F
from torch import nn

from transformers import DeepseekV2Config

from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.moe import MoE
from nanovllm.layers.attention_mla import MLAAttention


def _is_moe_layer(config: DeepseekV2Config, layer_idx: int) -> bool:
    n_routed = getattr(config, "n_routed_experts", 0) or 0
    return n_routed > 0 and layer_idx >= config.first_k_dense_replace


def _rope_theta(config: DeepseekV2Config) -> float:
    rp = getattr(config, "rope_parameters", None) or {}
    return float(rp.get("rope_theta", getattr(config, "rope_theta", 10000.0)))


class DeepseekV2MLP(nn.Module):
    """dense/shared FFN（独立 gate/up/down，名与 HF mlp(.shared_experts) 直配）。"""

    def __init__(self, config: DeepseekV2Config,
                 intermediate_size: int | None = None) -> None:
        super().__init__()
        inter = (intermediate_size if intermediate_size is not None
                 else config.intermediate_size)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, inter,
                                              bias=config.mlp_bias)
        self.up_proj = ColumnParallelLinear(config.hidden_size, inter,
                                            bias=config.mlp_bias)
        self.down_proj = RowParallelLinear(inter, config.hidden_size,
                                           bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV2Moe(MoE):
    """MoE + 加性共享专家：y = MoE(x) + shared_experts(x)（HF 同构）。

    直接继承 layers/moe.py 的 MoE → gate/experts 就以 mlp.gate/mlp.experts.*
    命名（与 HF checkpoint 直配）；共享专家挂兄弟模块 mlp.shared_experts。
    """

    def __init__(self, config: DeepseekV2Config) -> None:
        n_routed = getattr(config, "n_routed_experts")
        top_k = config.num_experts_per_tok
        assert config.topk_method in (None, "greedy"), \
            f"topk_method={config.topk_method!r} 未实现（group_limited_greedy）"
        super().__init__(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=n_routed,
            top_k=top_k,
            norm_topk_prob=False,               # DeepSeek 无 topk 概率归一化
            routed_scaling_factor=config.routed_scaling_factor,
            segment_backend=bool(getattr(config, "moe_segment_backend", False)),
        )
        # 共享专家 = 更宽的 dense FFN（intermediate = moe_intermediate × n_shared）
        self.shared_experts = DeepseekV2MLP(
            config,
            intermediate_size=config.moe_intermediate_size
            * config.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.shared_experts(x)


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(self, config: DeepseekV2Config, layer_idx: int) -> None:
        super().__init__()
        rp = getattr(config, "rope_parameters", None) or {}
        rope_theta = _rope_theta(config)
        self.self_attn = MLAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            max_position=config.max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
        )
        if _is_moe_layer(config, layer_idx):
            self.mlp = DeepseekV2Moe(config)
        else:
            self.mlp = DeepseekV2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), \
                hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states,
                                                           residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states,
                                                                residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV2Model(nn.Module):

    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV2DecoderLayer(config, i)
             for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):
    # 权重名与 HF checkpoint 逐段直配（self_attn.*、mlp.gate/experts.{i}.*/
    # shared_experts.*、dense mlp.* 同名），loader 零映射、零改名
    packed_modules_mapping = {}

    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

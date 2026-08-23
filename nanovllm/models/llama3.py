import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

# Llama-3.x 注意力模块：与 Qwen2 同构（无 QK-Norm），关键差异是
# attention_bias 默认必须为 False（Llama 的 qkv 无 bias；qwen 系默认 True，
# 若沿用会错误创建 bias 参数 → checkpoint 无对应张量 → 残留未初始化 → logits 垃圾）。
class Llama3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        rope_theta: float = 500000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size() # 张量并行中，GPU数量
        self.total_num_heads = num_heads # 全局注意力头总数
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size # 每张GPU分配的注意力头数
        self.total_num_kv_heads = num_kv_heads # 全局KV头总数
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size # 每张GPU分配的KV头数
        self.head_dim = head_dim or hidden_size // self.total_num_heads # 每个注意力头的维度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # QKV并行线性投影层，同时计算QKV（Llama 无 bias）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 如果提供了RoPE缩放配置：无操作缩放（default/None）只取 rope_theta；
        # "llama3" 缩放（Llama-3.2 官方 / Llama-3.1 的 unsloth 转换版都带）传给
        # RotaryEmbedding 实现（波长分段缩放，见 rotary_embedding.py）；其余报错。
        scaling_key = None
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type")
            if rope_type in (None, "default"):
                rope_theta = rope_scaling.get("rope_theta", rope_theta)
            elif rope_type == "llama3":
                scaling_key = tuple(sorted(rope_scaling.items()))
            else:
                raise NotImplementedError(
                    f"rope_scaling type {rope_type!r} unsupported: only 'default'/'llama3' handled "
                    f"(yarn/linear/dynamic 未实现，见 LEARNING.md 阶段7 卡点清单)")

        # 初始化旋转位置编码模块
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            scaling_key=scaling_key,
        )
        # 初始化实际的注意力计算模块
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    # 接收位置编码张量和隐藏状态，返回注意力输出
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 一次性投影得到QKV拼接的结果
        qkv = self.qkv_proj(hidden_states)
        # 沿最后一维拆分QKV
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 重塑QKV
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 将旋转位置编码应用在QK上（Llama 无 QK-Norm）
        q, k = self.rotary_emb(positions, q, k)
        # 执行注意力计算
        o = self.attn(q, k, v)
        # 将注意力输出的最后两维展平，并通过行并行输出投影，得到最终输出
        output = self.o_proj(o.flatten(1, -1))
        return output

# 前馈网络模块
class Llama3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # 门控和上投影合并为一个列并行线性层
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 下投影为行并行线性层
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # SiLu + 逐元素乘法实现门控
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

# 单Tranformer解码层
class Llama3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        # 注意力子层（attention_bias 默认 False——Llama 特有，区别于 qwen 系）
        self.self_attn = Llama3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 500000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # MLP子层
        self.mlp = Llama3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 注意力前的归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP前的归一化层
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

# Llama-3 主体
class Llama3Model(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        # 词嵌入层
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 创建config.num_hidden_layers个解码层
        self.layers = nn.ModuleList([Llama3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 在输出之前的归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

# 用于因果语言模型的完整Llama-3，包含主体和LM Head
class Llama3ForCausalLM(nn.Module):
    # 将Hugging face的权重名称映射到自定义实现的参数名和索引，用于权重加载时的转换
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"), # q_proj对应qkv_proj的q部分
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig
    ) -> None:
        super().__init__()
        self.model = Llama3Model(config)
        # 将模型的最后一个隐藏状态从hidden_size映射到vocab_size
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 如果配置要求权重绑定，将LM Head的权重和词嵌入权重共享
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # 前向传播返回隐藏状态，不计算logits
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    # 通过LM Head计算logits
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

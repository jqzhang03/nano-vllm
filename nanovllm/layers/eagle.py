"""EAGLE-1 草稿层（Draft Model）：一个无 RoPE 的 transformer 层。

F(h_t, e(w_{t+1})) → h̃_{t+1}：输入 = 目标模型最后一层 hidden（"features"）与
下一 token 的 embedding 拼接（[n, 2H]），输出 = 预测的下一位置 hidden（[n, H]），
再交给目标模型的共享 LM head 得到草稿分布。

与 Medusa 的差异（EAGLE 论文的核心卖点）：
- Medusa：γ+1 个并行 MLP 头，每个头独立从 h_t 预测 t+k+1（无 token 间依赖）；
- EAGLE：**自回归**——h̃_{t+1} 依赖 h_t 与 w_{t+1}，而 w_{t+1} 依赖上一步的采样，
  草稿分布逐 token 条件化于已草拟内容 → 更接近目标分布、接受率更高。
- 代价：草稿是 γ 步串行小前向（每步一个 transformer 层 + LM head）。

结构要点（EAGLE-1 同款）：
- **无 RoPE**（论文：hidden 已编码位置信息，位置编码反而有害）；
- attention 支路输入 2H（norm→qkv→因果SDPA→o_proj），残差加回 h（H 维）；
- FFN 支路在 H 维（norm→gate_up→SiLU→down），残差；
- 自注意力用 torch SDPA（草稿序列每步每 seq 只有 1 行，无需 flash/分页）。
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, tp_size


class _EagleRMSNorm(nn.Module):
    """EagleLayer 专用 RMSNorm（非 in-place，autograd 安全）。

    引擎的 RMSNorm 为推理优化做了 in-place mul_（@torch.compile），训练反向会报
    "modified by an inplace operation"；草稿层训练需要可微版本。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps).to(x.dtype)) * self.weight


class EagleLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        diagonal_attn: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        # 对角注意力：每行只 attend 自己。推理是 1 token/步（批量行 = 不同序列），
        # 训练若用全局 causal 会让独立样本互相 attend（噪声）——对角使训练/推理一致
        self.diagonal_attn = diagonal_attn
        ts = tp_size()
        self.num_heads = num_heads // ts
        # attention 支路：输入 2H（concat），输出 H 更新量
        self.input_layernorm = _EagleRMSNorm(2 * hidden_size, eps=rms_norm_eps)
        self.qkv_proj = QKVParallelLinear(2 * hidden_size, head_dim, num_heads, num_heads,
                                          bias=False)
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False)
        # FFN 支路：H 维
        self.post_attention_layernorm = _EagleRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2,
                                                       bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """h: [n, H] 目标模型 hidden（features）；emb: [n, H] 下一 token 的 embedding。

        返回 h̃: [n, H]（下一位置的预测 hidden）。"""
        x = torch.cat([h, emb], dim=-1)                     # [n, 2H]
        x = self.input_layernorm(x)
        qkv = self.qkv_proj(x)                              # [n, 3·(heads·hd)]
        qd = self.num_heads * self.head_dim
        q, k, v = qkv.split([qd] * 3, dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        # 自注意力。q/k/v 是 [n, heads, hd]；SDPA 的 3-D 形式会把 hd 当序列维（语义错），
        # 正确形式是 [1, heads, n, hd]。
        n = q.shape[0]
        v3 = v  # [n, heads, hd]（对角分支用）
        q = q.unsqueeze(0)   # [1, heads, n, hd]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        if self.diagonal_attn:
            # 每行只 attend 自己（训练随机批量行 = 独立样本，推理每 seq 1 行 → 语义一致）。
            # softmax(单元素)=1 → 输出恒等于 v（q/k 投影冗余但无害，结构保留便于切 causal）
            o = v3
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # 序列内因果
            o = o.squeeze(0).transpose(0, 1).contiguous()   # [heads, n, hd] → [n, heads, hd]
        o = self.o_proj(o.flatten(1, -1))                   # [n, H]
        h = h + o                                           # 残差在 H 维
        h = self.post_attention_layernorm(h)
        h = h + self.down_proj(self.act_fn(self.gate_up_proj(h)))
        return h

"""MoE（Mixture of Experts）FFN 层——阶段 1.5。

结构对齐 HF 惯例（experts.E.gate_proj / up_proj / down_proj 独立张量）：
- `ExpertFFN`：单专家 = gate/up（Column 并行）+ down（Row 并行），激活 SiLU；
  每个专家是独立 `LinearBase` 子类实例 → **继承全部量化路径**
  （int4/fp8/w8a8/sparse24，逐专家独立打包/scale）与 weight_loader 分片；
- `MoE`：router（ReplicatedLinear hidden→E）+ `nn.ModuleList` of ExpertFFN +
  可选 shared_expert（gate/up/down + shared_expert_gate 标量加权）。

前向（正确性优先的"循环专家"路径，vLLM 早期同款）：
    logits = router(x)（fp32 softmax）→ top_k 每 token 选 k 个专家
    → (token, slot) 展平 → 按专家循环：
      选中该专家的 token 行 gather → ExpertFFN → 乘路由概率 → index_add 散回
  与 naive 参考（全专家算、按 e 升序累加、fp32 累加）数学同构且求和顺序相同
  → 对照应位级一致（见 benchmarks/_moe_check.py）。

已知边界（诚实标注）：
- 数据相关动态形状（gather 行数随路由变）→ 不能进 CUDA graph；MoE 模型
  decode 图优化需要"路由 padding"（未做），基准时用 enforce_eager。
- 循环专家 = 每层 E×(gather+3 GEMM+scatter) 启动；E 大/负载不均时每步时间 =
  max 负载专家（后续优化：分组/融合 GEMM）。
"""
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear


class ExpertFFN(nn.Module):
    """单专家 FFN：y = down(silu(gate(x)) * up(x))。权重 [out,in] 同 HF。"""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 bias: bool = False, linear_cls: Type = ColumnParallelLinear,
                 row_cls: Type = RowParallelLinear):
        super().__init__()
        self.gate_proj = linear_cls(hidden_size, intermediate_size, bias=bias)
        self.up_proj = linear_cls(hidden_size, intermediate_size, bias=bias)
        self.down_proj = row_cls(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """MoE FFN：router top-k + 循环专家 +（可选）shared expert。

    Args:
        hidden_size: 输入/输出维
        intermediate_size: 每专家 FFN 中间维
        num_experts: 专家数 E
        top_k: 每 token 激活专家数 k
        num_shared_experts: shared expert 数（0/1；Qwen3-MoE 风格）
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, top_k: int,
                 num_shared_experts: int = 0,
                 bias: bool = False,
                 expert_cls: Type = ExpertFFN):
        super().__init__()
        assert num_experts > 0 and 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [expert_cls(hidden_size, intermediate_size, bias=bias)
             for _ in range(num_experts)])
        # shared expert（Qwen3-MoE 风格）：所有 token 都过一遍的稠密 FFN，
        # 由 shared_expert_gate 的逐 token 标量（sigmoid 后）加权。
        self.num_shared_experts = num_shared_experts
        if num_shared_experts > 0:
            self.shared_expert = expert_cls(hidden_size, intermediate_size, bias=bias)
            self.shared_expert_gate = ReplicatedLinear(hidden_size, num_shared_experts,
                                                       bias=False)

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """路由：返回 (topk 概率 [T,k], topk 专家号 [T,k])，fp32 计算保证稳定。"""
        logits = self.router(x).float()
        probs = logits.softmax(dim=-1)
        return probs.topk(self.top_k, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        top_vals, top_idx = self._route(x)          # fp32 [T,k]
        idx_flat = top_idx.reshape(-1)              # [T*k]
        w_flat = top_vals.reshape(-1).to(x.dtype)   # 路由概率
        tgt = (torch.arange(T, device=x.device)
               .unsqueeze(1).expand(T, self.top_k).reshape(-1))  # 每 (t,slot) 的源行
        out = torch.zeros(T, x.size(1), device=x.device,
                          dtype=torch.float32)      # fp32 累加（顺序=e 升序，见 reference）
        for e in range(self.num_experts):
            m = idx_flat == e                       # [T*k] bool：谁路由给了 e
            if bool(m.any()):
                xs = x.index_select(0, tgt[m])      # [n_e, H] gather
                ys = self.experts[e](xs)            # 复用量化路径
                out.index_add_(0, tgt[m], ys.float() * w_flat[m][:, None].float())
        if self.num_shared_experts > 0:
            sg = torch.sigmoid(self.shared_expert_gate(x).float())  # [T, n_shared]
            out += self.shared_expert(x).float() * sg
        return out.to(x.dtype)

    @torch.no_grad()
    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """naive 参考：全专家都算，路由概率为 0 的项贡献 0。

        与 forward 的差异只在**组织方式**（这里不用 gather/index_add，用全行
        掩码向量逐专家累加）；数学同构且按 e 升序累加 → 应位级一致。
        """
        T = x.size(0)
        top_vals, top_idx = self._route(x)          # 与 forward 同一路由
        # 每 token 对每个 e 的概率：选中 → 对应 slot 的 prob；否则 0
        p_full = torch.zeros(T, self.num_experts, device=x.device, dtype=x.dtype)
        p_full.scatter_(1, top_idx, top_vals.to(x.dtype))
        out = torch.zeros(T, x.size(1), device=x.device, dtype=torch.float32)
        for e in range(self.num_experts):
            y_e = self.experts[e](x).float()        # [T, H] 全行（含 p=0 行，贡献精确为 0）
            out += p_full[:, e:e + 1].float() * y_e
        if self.num_shared_experts > 0:
            sg = torch.sigmoid(self.shared_expert_gate(x).float())
            out += self.shared_expert(x).float() * sg
        return out.to(x.dtype)

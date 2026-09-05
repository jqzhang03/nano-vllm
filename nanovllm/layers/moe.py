"""MoE（Mixture of Experts）FFN 层——阶段 1.5。

权重布局对齐 transformers checkpoint 格式（`experts.{i}.gate_proj/up_proj/down_proj`
2D per-expert——transformers 5.x 的 use_experts_implementation 在内存用 3D、存盘转
2D，parity 以存盘格式为准）：
- `mlp.gate.weight`            [E, H]         router（每 token 在 E 专家上的 logits）
- `mlp.experts.{i}.gate_proj`  [I, H] / up_proj [I, H] / down_proj [H, I]
参数全名直接匹配 HF state_dict → **loader 零改动**（全量复制，权重装载/流式
chunk（层级）与量化路径（逐专家独立）都自动成立）。

前向语义（照 transformers Qwen3MoeTopKRouter/Qwen3MoeExperts）：
    logits = gate(x)（fp32 softmax）→ top_k →（可选 norm_topk_prob 归一化）→
    概率转 x dtype → 逐专家循环（e 升序，与 HF nonzero 排序一致）：
      gather 选中行 → silu(gate)·up → down → ×路由概率 → index_add（x dtype 累加）

`reference`：独立组织（全行掩码概率向量，不用 gather/index_add）——数学同构、
累加顺序同为 e 升序 → 对照应位级一致（除 GEMM 行数相关的 K-归约尾差）。

已知边界（诚实标注）：
- 动态 gather 形状 → 不能进 CUDA graph（capture 烘焙形状重放时对不上，静默错）；
  MoE 模型基准用 enforce_eager；graph 化需"路由 padding"（未做）。
- TP>1 语义：gate/up/down 各自 Column/Row 分片可用，但"同一专家分散在多个文件"
  的 TP 存储格式未验证（本项目 TP 未实测）。
- 循环专家 = 每层 E 次 (gather + 3 GEMM + scatter) 启动；负载不均时步时 = max 负载专家。
- router 不应量化（gate 精度决定路由）。
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear


class ExpertFFN(nn.Module):
    """单专家 FFN：y = down(silu(gate(x))·up(x))。属性名与 HF experts.{i} 对齐。"""

    def __init__(self, hidden_size: int, moe_intermediate_size: int,
                 bias: bool = False):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, moe_intermediate_size,
                                              bias=bias)
        self.up_proj = ColumnParallelLinear(hidden_size, moe_intermediate_size,
                                            bias=bias)
        self.down_proj = RowParallelLinear(moe_intermediate_size, hidden_size,
                                           bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """MoE FFN：router top-k + 循环专家（ModuleList 2D 权重）。

    Args:
        hidden_size: 输入/输出维
        moe_intermediate_size: 每专家 FFN 中间维
        num_experts: 专家数 E
        top_k: 每 token 激活专家数（num_experts_per_tok）
        norm_topk_prob: 是否把 top-k 概率归一化（Qwen3-MoE 配置项）
    """

    def __init__(self, hidden_size: int, moe_intermediate_size: int,
                 num_experts: int, top_k: int, norm_topk_prob: bool = False):
        super().__init__()
        assert num_experts > 0 and 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)  # [E, H]
        self.gate.quantize_exclude = True  # router 精度决定路由 → 永不量化
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_size, moe_intermediate_size)
             for _ in range(num_experts)])

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 路由：返回 (topk 概率 [T,k]（x dtype，可能已归一）, 专家号 [T,k])。"""
        logits = self.gate(x).float()
        probs = logits.softmax(dim=-1)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        return top_vals.to(x.dtype), top_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        top_vals, top_idx = self._route(x)                   # [T,k]
        idx_flat = top_idx.reshape(-1)                       # [T*k]
        w_flat = top_vals.reshape(-1)                        # 路由概率（x dtype）
        tgt = (torch.arange(T, device=x.device)
               .unsqueeze(1).expand(T, self.top_k).reshape(-1))
        out = torch.zeros_like(x)                            # x dtype 累加（同 HF）
        for e in range(self.num_experts):
            m = idx_flat == e                                # 谁路由给了 e
            # 无条件执行（不做 bool(m.any()) host sync）：空专家 → 空 gather/FFN/add
            # = GPU 侧无操作，但省掉每专家一次 device→host 同步（~ms 级固定税，
            # 见 benchmarks/_moe_imbalance.py 的 T 扫描：时间与 T 无关）
            xs = x.index_select(0, tgt[m])                   # [n_e, H] gather
            ys = self.experts[e](xs)                         # 复用量化路径
            out.index_add_(0, tgt[m], ys * w_flat[m][:, None])
        return out

    @torch.no_grad()
    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """naive 参考：全行掩码路径（无 gather/index_add），e 升序同 forward。"""
        T = x.size(0)
        top_vals, top_idx = self._route(x)
        p_full = torch.zeros(T, self.num_experts, device=x.device, dtype=x.dtype)
        p_full.scatter_(1, top_idx, top_vals)                # 非选中专家 = 0
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            y_e = self.experts[e](x)                         # 全行（含 p=0 行，贡献 0）
            out += p_full[:, e:e + 1] * y_e
        return out

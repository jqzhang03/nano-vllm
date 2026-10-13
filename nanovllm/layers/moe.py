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
                 num_experts: int, top_k: int, norm_topk_prob: bool = False,
                 segment_backend: bool = False,
                 int4_backend: str = "loop", fuse_int4_gate_up: bool = True,
                 routed_scaling_factor: float = 1.0):
        super().__init__()
        assert num_experts > 0 and 1 <= top_k <= num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        # DeepSeek 路由：top-k 概率 × routed_scaling_factor（V2/V3 默认 1.0；
        # Qwen3-MoE 无此因子 → 默认 1.0 恒等，行为不变）
        self.routed_scaling_factor = routed_scaling_factor
        # grouped 的批量实现：False = padded bmm（cuBLAS batched，默认——实测
        # 全尺寸最优或接近）；True = Triton 真段式内核（无 padding；实测仅在
        # 小 E + 极长段赢 43%，见 nanovllm/layers/moe_segment.py 头部边界表）
        self.segment_backend = segment_backend
        if int4_backend not in ("loop", "segment"):
            raise ValueError(f"unknown MoE INT4 backend: {int4_backend!r}")
        self.int4_backend = int4_backend
        self.fuse_int4_gate_up = fuse_int4_gate_up
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)  # [E, H]
        self.gate.quantize_exclude = True  # router 精度决定路由 → 永不量化
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_size, moe_intermediate_size)
             for _ in range(num_experts)])
        # grouped 后端状态（首次 forward 惰性判定；见 _ensure_grouped_weights）
        self._gup_t = None    # [E, H, 2I] 3D 堆叠转置（gate/up 融合）
        self._dn_t = None     # [E, I, H] 3D 堆叠转置
        self._grouped_ok = None  # None=未判定；True/False=可用/回退 loop
        self._int4_segment_ok = None
        self._int4_segment_reason = None
        self._int4_segment_tables = None
        self._int4_segment_signature = None

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 路由：返回 (topk 概率 [T,k]（x dtype，含 routed_scaling_factor）, 专家号 [T,k])。"""
        logits = self.gate(x).float()
        probs = logits.softmax(dim=-1)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            top_vals = top_vals * self.routed_scaling_factor
        return top_vals.to(x.dtype), top_idx

    # ------------------------------------------------------------------
    # 权重可组性：所有专家的三类线性都有 float 权重（未量化，或有 w_deq）
    # → 可堆叠 3D 走 grouped 批量 GEMM。纯 int4 / nvfp4 / fp8 / w8a8 / sparse24
    # （权重是打包/定点格式，无 float 视图）
    # → 回退逐专家循环（每专家走各自的量化内核）。
    # ------------------------------------------------------------------
    def _float_weight(self, lin: nn.Module) -> torch.Tensor | None:
        w = getattr(lin, "w_deq", None)
        if w is not None:
            return w
        if not (lin.int4 or lin.nvfp4 or lin.fp8 or lin.w8a8 or lin.sparse24):
            return lin.weight
        return None

    def _ensure_grouped_weights(self) -> bool:
        if self._grouped_ok is not None:
            return self._grouped_ok
        w_g = [self._float_weight(e.gate_proj) for e in self.experts]
        w_u = [self._float_weight(e.up_proj) for e in self.experts]
        w_d = [self._float_weight(e.down_proj) for e in self.experts]
        ok = all(w is not None for w in w_g + w_u + w_d)
        if ok:
            dtype = w_g[0].dtype
            dev = w_g[0].device
            # [E, 2I, H] → 转置 [E, H, 2I]（bmm 用：dst @ Wt）
            gup = torch.stack([torch.cat([w_g[e], w_u[e]], dim=0)
                               for e in range(self.num_experts)])
            dn = torch.stack([w_d[e] for e in range(self.num_experts)])
            self._gup_t = gup.transpose(1, 2).contiguous().to(dtype).to(dev)
            self._dn_t = dn.transpose(1, 2).contiguous().to(dtype).to(dev)
        self._grouped_ok = ok
        return ok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """自动后端：float 权重可组 → grouped 批量 GEMM；否则逐专家循环。"""
        if self._ensure_grouped_weights():
            return self._forward_grouped(x)
        if self.int4_backend == "segment" and self._can_segment_int4(x):
            return self._forward_segment_int4(x)
        return self._forward_loop(x)

    def _expert_linears(self):
        return [
            (expert.gate_proj, expert.up_proj, expert.down_proj)
            for expert in self.experts
        ]

    def _can_segment_int4(self, x: torch.Tensor) -> bool:
        """Check the supported packed-INT4 layout once after model quantization."""
        if self._int4_segment_ok is True:
            return True
        linears = [lin for triple in self._expert_linears() for lin in triple]
        reason = None
        if not linears or any(not getattr(lin, "int4", False) for lin in linears):
            reason = "all routed expert projections must use INT4"
        elif any(getattr(lin, "w_deq", None) is not None for lin in linears):
            reason = "dual-path INT4 uses the existing float grouped backend"
        elif any(hasattr(lin, "awq_scale") for lin in linears):
            reason = "AWQ activation scaling is not supported by segmented INT4"
        elif any(getattr(lin, "bias", None) is not None for lin in linears):
            reason = "expert projection bias is not supported by segmented INT4"
        elif any(getattr(lin, "tp_size", 1) != 1 for lin in linears):
            reason = "segmented INT4 currently requires tensor_parallel_size=1"
        elif x.dtype not in (torch.bfloat16, torch.float16):
            reason = f"activation dtype {x.dtype} is not supported"
        else:
            group_sizes = set()
            for lin in linears:
                packed = getattr(lin, "w_int4", None)
                scales = getattr(lin, "w_int4_scale", None)
                if packed is None or scales is None or packed.ndim != 2 or scales.ndim != 2:
                    reason = "packed weights/scales have an unsupported shape"
                    break
                k = packed.shape[1] * 2
                if scales.shape[0] != packed.shape[0] or k % scales.shape[1]:
                    reason = "INT4 scale layout does not match the packed weights"
                    break
                group_sizes.add(k // scales.shape[1])
                if scales.dtype != x.dtype or packed.device != x.device or scales.device != x.device:
                    reason = "INT4 weights, scales, and activations must share device/dtype"
                    break
            if reason is None and len(group_sizes) != 1:
                reason = "all expert projections must use the same INT4 group size"
            if reason is None:
                group_size = next(iter(group_sizes))
                if group_size % 2 or 128 % group_size:
                    reason = f"INT4 group size {group_size} must be even and divide 128"
        self._int4_segment_reason = reason
        self._int4_segment_ok = reason is None
        return self._int4_segment_ok

    def _ensure_int4_segment_tables(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        linears = self._expert_linears()
        attrs = {
            "gate_w": [item[0].w_int4 for item in linears],
            "gate_s": [item[0].w_int4_scale for item in linears],
            "up_w": [item[1].w_int4 for item in linears],
            "up_s": [item[1].w_int4_scale for item in linears],
            "down_w": [item[2].w_int4 for item in linears],
            "down_s": [item[2].w_int4_scale for item in linears],
        }
        signature = (x.device, tuple(t.data_ptr() for tensors in attrs.values()
                                     for t in tensors))
        if signature != self._int4_segment_signature:
            self._int4_segment_tables = {
                name: torch.tensor([t.data_ptr() for t in tensors],
                                   dtype=torch.int64, device=x.device)
                for name, tensors in attrs.items()
            }
            self._int4_segment_signature = signature
        return self._int4_segment_tables

    def _forward_segment_int4(self, x: torch.Tensor) -> torch.Tensor:
        """Sort routed rows once and run grouped, no-padding packed-INT4 GEMMs."""
        from nanovllm.layers.moe_int4_segment import (
            moe_segment_int4_gup,
            moe_segment_int4_mm,
        )

        T = x.size(0)
        top_vals, top_idx = self._route(x)
        idx_flat = top_idx.reshape(-1)
        tgt = (torch.arange(T, device=x.device)
               .unsqueeze(1).expand(T, self.top_k).reshape(-1))
        if idx_flat.numel() == 0:
            return torch.zeros_like(x)

        order = torch.argsort(idx_flat, stable=True)
        sorted_idx = idx_flat[order]
        xs = x.index_select(0, tgt[order])
        counts = torch.bincount(sorted_idx, minlength=self.num_experts)
        offsets = torch.cumsum(counts, 0) - counts
        max_count = int(counts.max())
        tables = self._ensure_int4_segment_tables(x)
        group_size = self.experts[0].gate_proj.w_int4.shape[1] * 2 \
            // self.experts[0].gate_proj.w_int4_scale.shape[1]

        if self.fuse_int4_gate_up:
            gup = moe_segment_int4_gup(
                xs, tables["gate_w"], tables["gate_s"],
                tables["up_w"], tables["up_s"], offsets, counts,
                self.moe_intermediate_size, group_size,
                max_tokens_per_expert=max_count)
            gate, up = gup.chunk(2, dim=-1)
        else:
            gate = moe_segment_int4_mm(
                xs, tables["gate_w"], tables["gate_s"], offsets, counts,
                self.moe_intermediate_size, group_size,
                max_tokens_per_expert=max_count)
            up = moe_segment_int4_mm(
                xs, tables["up_w"], tables["up_s"], offsets, counts,
                self.moe_intermediate_size, group_size,
                max_tokens_per_expert=max_count)
        hidden = F.silu(gate) * up
        ys = moe_segment_int4_mm(
            hidden, tables["down_w"], tables["down_s"], offsets, counts,
            self.hidden_size, group_size, max_tokens_per_expert=max_count)

        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=x.device)
        # Match _forward_loop's expert-id accumulation order. A single index_add
        # over all slots permits BF16 atomics to sum them in a different order,
        # which becomes visible at top_k=6 on a real DeepSeek router.
        y_by_slot = ys[inverse].view(T, self.top_k, self.hidden_size)
        slot_order = top_idx.argsort(dim=-1, stable=True)
        y_expert_order = y_by_slot.gather(
            1, slot_order[:, :, None].expand(T, self.top_k, self.hidden_size))
        weights_expert_order = top_vals.gather(1, slot_order)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            out += y_expert_order[:, slot, :] * weights_expert_order[:, slot, None]
        return out

    def _forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        """逐专家循环后端（量化专家回退路径；与 reference 同序对照源）。"""
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
            # = GPU 侧无操作，但省掉每专家一次 device→host 同步
            xs = x.index_select(0, tgt[m])                   # [n_e, H] gather
            ys = self.experts[e](xs)                         # 复用量化路径
            out.index_add_(0, tgt[m], ys * w_flat[m][:, None])
        return out

    # ------------------------------------------------------------------
    # grouped 批量后端（消除逐专家 Python 循环/启动）：
    #   top-k 路由 → 按专家稳定排序（段连续）→ 排序行 gather →
    #   padded [E, max_n, H]（排序拼接前缀即各段；空段占 0 行）→
    #   单次批量 bmm × 3D gate_up（gate/up 融合）→ silu(g)·u →
    #   单次批量 bmm × 3D down → 反排 → 加权 index_add（token 的 k 个 slot 合并）。
    # 代价：padded 批量把计算放大到 E·max_n 行（均衡路由 max≈Tk/E → 近 1×；
    # 不均衡时 = E·max/(T·k) 浪费——见 benchmarks/_moe_grouped_bench.py）。
    # 与 loop 的差异：同一 token 的 k 个 slot 累加顺序 = slot 序（概率降序），
    # 非 e 升序 → fp 尾差级（对照用阈值）。
    # ------------------------------------------------------------------
    def _forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        k = self.top_k
        top_vals, top_idx = self._route(x)                   # [T,k]
        idx_flat = top_idx.reshape(-1)                       # [R], R = T*k
        w_flat = top_vals.reshape(-1)
        tgt = (torch.arange(T, device=x.device)
               .unsqueeze(1).expand(T, k).reshape(-1))
        R = idx_flat.numel()
        if R == 0:
            return torch.zeros_like(x)
        # 1) 按专家稳定排序 → 段连续；gather 排序后的行
        order = torch.argsort(idx_flat, stable=True)         # [R]
        xs = x.index_select(0, tgt[order])                   # [R, H] 段连续（e 升序）
        counts = torch.bincount(idx_flat[order], minlength=self.num_experts)
        if self.segment_backend:
            # Triton 真段式：offsets 驱动，无 padding（moe_segment.py）
            from nanovllm.layers.moe_segment import moe_segment_mm
            offs = torch.cumsum(counts, 0) - counts
            gup = moe_segment_mm(xs, self._gup_t, offs, counts)      # [R, 2I]
            g, u = gup.chunk(2, dim=-1)
            h = torch.nn.functional.silu(g) * u                     # [R, I]
            ys = moe_segment_mm(h, self._dn_t, offs, counts)        # [R, H]
        else:
            # padded bmm：每段放回自己的行带（e*max_n + 段内偏移），其余 0
            max_n = int(counts.max())
            E = self.num_experts
            e_row = idx_flat[order].to(torch.long)           # [R] 每行专家（非降）
            seg_start = torch.cumsum(counts, 0) - counts     # [E] 每段在 xs 的起始
            pos = e_row * max_n + (torch.arange(R, device=x.device)
                                   - seg_start[e_row])       # [R] dst 平面行号
            dst = torch.zeros(E, max_n, x.size(1), device=x.device, dtype=x.dtype)
            dst.reshape(-1, x.size(1)).index_copy_(0, pos, xs)
            # 3) fused gate_up 批量 GEMM + silu·up（padding 行输入 0 → 输出精确 0）
            gup = torch.bmm(dst, self._gup_t)                # [E, max_n, 2I]
            g, u = gup.chunk(2, dim=-1)
            h = torch.nn.functional.silu(g) * u              # [E, max_n, I]
            out3 = torch.bmm(h, self._dn_t)                  # [E, max_n, H]
            ys = out3.reshape(-1, x.size(1))[pos]            # 按平面行号取回 → xs 序
        # 4) 反排回 (t, slot) 序 → 加权 index_add 合并同一 token 的 slot
        inv = torch.empty_like(order)
        inv[order] = torch.arange(R, device=x.device)
        y_slot = ys[inv] * w_flat[:, None]
        out = torch.zeros_like(x)
        out.index_add_(0, tgt, y_slot)
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

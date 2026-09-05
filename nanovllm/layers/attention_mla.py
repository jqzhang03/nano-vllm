"""MLA（Multi-head Latent Attention）——DeepSeek-V2 潜在注意力（阶段 2a）。

数学（对齐 transformers 5.15 modeling_deepseek_v2.py DeepseekV2Attention）：
    c_q   = q_a_layernorm(q_a_proj(h))                          [T, q_lora_rank]
    q     = q_b_proj(c_q) → view [T, h, qk_nope + qk_rope]
    c_kv  = kv_a_layernorm(kv_a_proj_with_mqa(h)[..., :kv_lora]) [T, kv_lora]
    k_pe  = kv_a_proj_with_mqa(h)[..., kv_lora:]（不 norm，共享单头 rope 64）
    q_pe/k_pe 各自旋转（interleaved-pairs RoPE，见 rotary_embedding.py）
    kv_b 逐头输出 [k_nope(128) | v(128)]：kv_b_proj(c_kv) → view [T,h,nope+v]
    k_head = [k_nope | k_pe(expand)]（qk_head_dim = 192），v_head_dim = 128
    scaling = qk_head_dim ** -0.5（192 维 key）

KV cache（引擎 allocate_kv_cache 按 cache_kind="mla" 绑定 fused 张量）：
    每层 [num_blocks, block_size, kv_lora + qk_rope]，token 行 = [c_kv | k̃_pe]
    ——只存压缩潜在 + 旋转后的 rope key；逐头 k_nope/v 不落盘（账本：等效 GQA
    4096 元素/token/层 vs 576 ≈ 7.1×，见 INTERVIEW §2.8）。

注意力路径（跑通≠写对：每条都对照稠密展开参考）：
1) 无引擎（CPU 单测）：全稠密 torch 手工因果注意力；
2) prefill / cache-shaped 行（spec、分块续写、前缀复用）：本步 fresh 行直接用
   逐头稠密 K/V；前缀行从 fused cache gather → kv_b 稠密化 → 行主序组装成
   连续 [Σ(c+f)] 喂 flash_attn_varlen（c = 行 query 起点 = 缓存 token 数）；
3) decode 行：
   a) **吸收式 Triton 内核**（kv_b 有 float 视图时默认）：W_UK 吸收进 q
      （q_abs = q_nope @ W_UKᵀ，score = q_abs·c 的 512 点积）、W_UV 吸收进
      输出侧（kernel 内只累加 [h, kv_lora] 潜在加权和，出内核后一次性 v 投影）
      → 每 token 每层只读 576 元素（GQA 读逐头 K/V 4096 元素）；
   b) `dense 兜底`（纯 int4/fp8 无 float 视图）：整段缓存稠密化再 flash——
      正确性等价、带宽优势消失（诚实边界见文件尾注）。
decode 兜底与 varlen 前缀行的行信息（start/槽位索引）由 ModelRunner prepare_*
构建进 Context（mla_pre_* 对齐 varlen 行、mla_dec_* 对齐 decode 行）。
"""
import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_context


# ---------------------------------------------------------------------------
# fused cache 写内核：逐 token 写 [c_kv | k̃_pe] 到层缓存 [nb, B, kv_lora+rope]
# ---------------------------------------------------------------------------
@triton.jit
def mla_store_kernel(
    c_ptr, k_pe_ptr, cache_ptr, slot_ptr,
    KV_LORA: tl.constexpr, ROPE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_ptr + idx)
    if slot == -1:
        return
    D = KV_LORA + ROPE
    row = (slot // BLOCK_SIZE) * (BLOCK_SIZE * D) + (slot % BLOCK_SIZE) * D
    offs_c = tl.arange(0, KV_LORA)
    c = tl.load(c_ptr + idx * KV_LORA + offs_c)
    tl.store(cache_ptr + row + offs_c, c)
    offs_r = tl.arange(0, ROPE)
    k = tl.load(k_pe_ptr + idx * ROPE + offs_r)
    tl.store(cache_ptr + row + KV_LORA + offs_r, k)


def mla_store(c_kv: torch.Tensor, k_pe: torch.Tensor, mla_cache: torch.Tensor,
              slot_mapping: torch.Tensor):
    """写 [c_kv | k̃_pe] 到 fused cache（T 与 slot_mapping 对齐；-1 跳过）。"""
    T, kv_lora = c_kv.shape
    rope = k_pe.shape[-1]
    assert k_pe.shape[0] == T and mla_cache.shape[-1] == kv_lora + rope
    assert slot_mapping.numel() == T
    assert kv_lora % 2 == 0 and rope % 2 == 0
    mla_store_kernel[(T,)](c_kv, k_pe, mla_cache, slot_mapping,
                           KV_LORA=kv_lora, ROPE=rope,
                           BLOCK_SIZE=mla_cache.shape[1])


# ---------------------------------------------------------------------------
# MLA decode 吸收式内核：grid = (行数,)；每行一个 q token、全 h 头。
#
#   score_i(t) = q_abs_i·c_t + q̃_pe_i·k̃_t         （W_UK 吸收：q_abs = q_nope @ W_UKᵀ
#   在 kernel 外每步每头做一次 128×512 小 GEMM；k̃ 旋转后存盘、q̃_pe 用旋转后 q）
#   o_c[h, kv_lora] = Σ_t p_it·c_t                 （W_UV 吸收进输出侧，出内核后
#   逐头 512×128 投影——见 MLAAttention._decode_kernel 的 einsum）
# 每 token 每层 DRAM = kv_lora+rope 元素读一次、跨 h 头共享 —— MLA 带宽账本来源。
#
# 布局约定：矩阵行 = token（T）、列 = 头（H）→ dot(c_t [T,K], q_absᵀ [K,H])；
# softmax 沿 token 轴；o_c [H,K] 按 α[h] 逐头缩放（m/l 为 [H,1]）。
# ---------------------------------------------------------------------------
@triton.jit
def mla_decode_kernel(
    q_abs_t_ptr, q_pe_t_ptr, cache_ptr, block_table_ptr, cache_seqlens_ptr,
    o_ptr, softmax_scale, max_blocks,
    KV_LORA: tl.constexpr, ROPE: tl.constexpr, H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    seqlen = tl.load(cache_seqlens_ptr + pid)
    offs_h = tl.arange(0, H)
    offs_k = tl.arange(0, KV_LORA)
    # 吸收后 query：q_absᵀ [KV, H]、q̃_peᵀ [R, H]（每行自己的）
    q_abs = tl.load(q_abs_t_ptr + pid * KV_LORA * H
                    + offs_k[:, None] * H + offs_h[None, :]).to(tl.float16)
    offs_r = tl.arange(0, ROPE)
    q_pe = tl.load(q_pe_t_ptr + pid * ROPE * H
                   + offs_r[:, None] * H + offs_h[None, :]).to(tl.float16)

    # flash 状态沿 token 轴归约 → [1, H]（s [T,H] 可广播）；o_c [H,K] 的
    # 逐头缩放用 alpha 的转置 [H,1]
    m = tl.full([1, H], float("-inf"), dtype=tl.float32)
    l = tl.zeros([1, H], dtype=tl.float32)
    o_c = tl.zeros([H, KV_LORA], dtype=tl.float32)

    num_blocks = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    D = KV_LORA + ROPE
    block_stride = BLOCK_SIZE * D
    for b in range(num_blocks):
        block_id = tl.load(block_table_ptr + pid * max_blocks + b)
        base = block_id * block_stride
        for t in range(0, BLOCK_SIZE, BLOCK_T):
            offs_t = t + tl.arange(0, BLOCK_T)
            tok_mask = (b * BLOCK_SIZE + offs_t) < seqlen
            c_ptrs = cache_ptr + base + offs_t[:, None] * D + offs_k[None, :]
            c_t = tl.load(c_ptrs, mask=tok_mask[:, None],
                          other=0.0).to(tl.float16)                 # [T, KV]
            s = tl.dot(c_t, q_abs, out_dtype=tl.float32)            # [T, H] nope
            k_ptrs = cache_ptr + base + offs_t[:, None] * D \
                + KV_LORA + offs_r[None, :]
            k_t = tl.load(k_ptrs, mask=tok_mask[:, None],
                          other=0.0).to(tl.float16)
            s = (s + tl.dot(k_t, q_pe, out_dtype=tl.float32)) \
                * softmax_scale
            s = tl.where(tok_mask[:, None], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=0)[None, :])
            alpha = tl.exp(m - m_new)                               # [1,H]
            p = tl.exp(s - m_new)
            l = l * alpha + tl.sum(p, axis=0)[None, :]
            o_c = o_c * tl.trans(alpha) + tl.dot(tl.trans(p.to(tl.float16)),
                                                 c_t, out_dtype=tl.float32)
            m = m_new
    o_c = o_c * tl.trans(1.0 / l)
    tl.store(o_ptr + pid * H * KV_LORA
             + offs_h[:, None] * KV_LORA + offs_k[None, :],
             o_c.to(q_abs_t_ptr.dtype.element_ty))


def mla_decode_attention(q_abs: torch.Tensor, q_pe: torch.Tensor,
                         mla_cache: torch.Tensor, block_table: torch.Tensor,
                         cache_seqlens: torch.Tensor,
                         softmax_scale: float) -> torch.Tensor:
    """吸收式 MLA decode 内核入口。

    q_abs: [bs, H, kv_lora]（= q_nope @ W_UKᵀ）；q_pe: [bs, H, rope]（已旋转）
    返回 o_c [bs, H, kv_lora]（潜在加权和；v 投影在 kernel 外）。
    """
    bs, H, kv_lora = q_abs.shape
    rope = q_pe.shape[-1]
    kv_lora_r = triton.next_power_of_2(kv_lora)
    rope_r = triton.next_power_of_2(rope)
    assert mla_cache.shape[-1] == kv_lora + rope
    o = torch.empty(bs, H, kv_lora_r, device=mla_cache.device,
                    dtype=mla_cache.dtype)
    grid = (bs,)
    mla_decode_kernel[grid](
        q_abs.transpose(1, 2).contiguous(),      # [bs, KV, H]
        q_pe.transpose(1, 2).contiguous(),       # [bs, R, H]
        mla_cache, block_table, cache_seqlens, o,
        softmax_scale, block_table.shape[1],
        KV_LORA=kv_lora_r, ROPE=rope_r, H=H,
        BLOCK_SIZE=mla_cache.shape[1], BLOCK_T=32,
        num_warps=4,
    )
    return o[..., :kv_lora]


class MLAAttention(nn.Module):
    """DeepSeek MLA 注意力层（参数名与 HF checkpoint 直配，loader 零映射）。"""

    cache_kind = "mla"      # 引擎 allocate_kv_cache 识别 fused cache（见 runner）

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position: int,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # DeepSeek-V2 官方：scaling = qk_head_dim ** -0.5（192 维 key，含 rope）
        self.scaling = self.qk_head_dim ** -0.5

        if q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(hidden_size,
                                               num_heads * self.qk_head_dim,
                                               bias=attention_bias)
        else:
            self.q_a_proj = ColumnParallelLinear(hidden_size, q_lora_rank,
                                                 bias=attention_bias)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 num_heads * self.qk_head_dim,
                                                 bias=False)
        self.kv_a_proj_with_mqa = ColumnParallelLinear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=attention_bias)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False)
        self.o_proj = RowParallelLinear(num_heads * v_head_dim, hidden_size,
                                        bias=attention_bias)
        # rope 头：64 维 interleaved-pairs（DeepSeek 约定，见 rotary_embedding）
        assert rope_scaling is None, "MLA rope_scaling 未实现"
        self.rotary_emb = get_rope(
            qk_rope_head_dim, qk_rope_head_dim, max_position, rope_theta,
            interleaved=1)
        # 引擎绑定（allocate_kv_cache 按 cache_kind）：[nb, B, kv_lora+rope]
        self.mla_cache = torch.tensor([])
        self.mla_fp8 = False          # MLA cache 仅 bf16（fp8 KV 在 runner 断言）
        self._wuk_t = None            # [h, qk_nope, kv_lora] 逐头 W_UK（惰性）
        self._wuv_t = None            # [h, v_head, kv_lora] 逐头 W_UV

    # ------------------------------------------------------------------
    # float 视图（与 MoE._float_weight 同规则）：未量化 weight / int4 dual-path
    # 的 w_deq；纯 int4/fp8/... 无视图 → decode 走稠密兜底（_decode_rows）
    # ------------------------------------------------------------------
    @staticmethod
    def _float_weight(lin: nn.Module) -> torch.Tensor | None:
        w = getattr(lin, "w_deq", None)
        if w is not None:
            return w
        if not (lin.int4 or lin.fp8 or lin.w8a8 or lin.sparse24):
            return lin.weight
        return None

    def has_float_views(self) -> bool:
        wb = self._float_weight(self.kv_b_proj)
        if wb is None:
            return False
        if self._wuk_t is None:
            h, nope, vd = (self.num_heads, self.qk_nope_head_dim,
                           self.v_head_dim)
            w3 = wb.detach().reshape(h, nope + vd, self.kv_lora_rank)
            self._wuk_t = w3[:, :nope, :].contiguous()   # [h, nope, kv_lora]
            self._wuv_t = w3[:, nope:, :].contiguous()   # [h, vd, kv_lora]
        return True

    # ------------------------------------------------------------------
    # 纯数学部件（与引擎 Context 解耦，无缓存单测/CPU 参考共用）
    # ------------------------------------------------------------------
    def _project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """h → (q_nope, q_pe_raw, c_kv, k_pe_raw)（潜在层 norm 与 HF 同构）。"""
        T = hidden.shape[0]
        if self.q_lora_rank is None:
            q = self.q_proj(hidden)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        q = q.view(T, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        kv = self.kv_a_proj_with_mqa(hidden)
        c_raw, k_pe_raw = kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return q_nope, q_pe, self.kv_a_layernorm(c_raw), k_pe_raw

    def _expand(self, c_kv: torch.Tensor, k_pe_r: torch.Tensor,
                T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """c_kv → 逐头稠密 (k_head [T,h,nope+rope], v [T,h,v])。k_pe_r 已旋转、
        跨头共享（cat 落连续）。"""
        kkv = self.kv_b_proj(c_kv).view(
            T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kkv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe_r[:, None, :].expand(T, self.num_heads,
                                         self.qk_rope_head_dim)
        return torch.cat([k_nope, k_pe], dim=-1), v

    def _manual_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     positions: torch.Tensor) -> torch.Tensor:
        """无引擎参考：fp32 手工因果注意力（跨行无泄漏，行内 causal）。

        q/k head dim 192（nope+rope）≠ v head dim 128 → 拆字母维的 einsum。
        """
        s = torch.einsum("phj,khj->phk", q.float(), k.float()) * self.scaling
        causal = positions[:, None] >= positions[None, :]     # [p, k]
        s = s.masked_fill(~causal.unsqueeze(1), float("-inf"))
        p = s.softmax(dim=-1)
        return torch.einsum("phk,khv->phv", p, v.float()).to(q.dtype)

    # ------------------------------------------------------------------
    # cache-shaped 行的稠密化组装
    #   rows r = 0..R-1：缓存段 [0, start_r)（槽位 gather_idx 行主序）；
    #   fresh 段 = 本步 dense k/v 的 [q_off_r, q_off_r + qlen_r)（= 本行 query token
    #   自身——verify/续写行的 query 就是刚写缓存的 token）；
    #   目标段 = [seg_r, seg_r + start_r + qlen_r)，行主序连续 → flash 的
    #   cu_seqlens_k 差分恰为 start_r + qlen_r（引擎原 cu_k 语义，无需新数组）。
    # 无前缀行（start=0）→ 整段 fresh 拷贝 = 布局不变（纯开销，spec 混合行用）。
    # ------------------------------------------------------------------
    def _assemble(self, k_dense: torch.Tensor, v_dense: torch.Tensor,
                  starts: torch.Tensor, qlens: torch.Tensor,
                  q_off: torch.Tensor, gather_idx: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = k_dense.device
        R = starts.numel()
        s = starts.long()
        f = qlens.long()
        seg_start = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=dev),
             torch.cumsum(s + f, 0)[:-1]])
        total = int((s + f).sum().item())
        flat = self.mla_cache.reshape(-1, self.mla_cache.shape[-1])
        gathered = flat.index_select(0, gather_idx.long())     # [Σs, kv+rope]
        gc, gk = gathered.split([self.kv_lora_rank,
                                 self.qk_rope_head_dim], dim=-1)
        gk_head, gv = self._expand(gc, gk, gc.shape[0])        # 缓存前缀稠密化
        h, dk = gk_head.shape[1], gk_head.shape[2]
        dv = gv.shape[2]
        k_out = torch.empty(total, h, dk, device=dev, dtype=k_dense.dtype)
        v_out = torch.empty(total, h, dv, device=dev, dtype=v_dense.dtype)
        s_cum = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                           torch.cumsum(s, 0)])
        q_cum = q_off.long()
        for r in range(R):
            dst = int(seg_start[r])
            cs, ce = int(s_cum[r]), int(s_cum[r + 1])
            qs, qe = int(q_cum[r]), int(q_cum[r + 1])
            if ce - cs:
                k_out[dst:dst + ce - cs] = gk_head[cs:ce]
                v_out[dst:dst + ce - cs] = gv[cs:ce]
            if qe - qs:
                k_out[dst + ce - cs:dst + ce - cs + qe - qs] = k_dense[qs:qe]
                v_out[dst + ce - cs:dst + ce - cs + qe - qs] = v_dense[qs:qe]
        return k_out, v_out

    def _flash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               cu_q: torch.Tensor, cu_k: torch.Tensor,
               max_q: int, max_k: int) -> torch.Tensor:
        # flash-attn varlen 要求 v 的 head dim == k 的 head dim；MLA 的 v
        # (128) < qk (192) → 把 v 零填充到 qk 维（额外列贡献 0，数学恒等），
        # 输出按原 v 维截取。
        if v.shape[-1] < k.shape[-1]:
            pad = torch.zeros(*v.shape[:-1], k.shape[-1], device=v.device,
                              dtype=v.dtype)
            pad[..., :v.shape[-1]] = v
            v = pad
        o = flash_attn_varlen_func(
            q, k, v, max_seqlen_q=max_q, cu_seqlens_q=cu_q,
            max_seqlen_k=max_k, cu_seqlens_k=cu_k,
            softmax_scale=self.scaling, causal=True)
        if v.shape[-1] != self.v_head_dim:
            o = o[..., :self.v_head_dim].contiguous()
        return o

    # ==================================================================
    # 引擎入口（ModelRunner.run → 模型 DecoderLayer 调用）
    # ==================================================================
    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]
        q_nope, q_pe_raw, c_kv, k_pe_raw = self._project(hidden_states)
        q_pe_r, k_pe_r = self.rotary_emb(positions, q_pe_raw, k_pe_raw)
        k_dense, v_dense = self._expand(c_kv, k_pe_r, T)

        ctx = get_context()
        cache_ready = self.mla_cache.numel() > 0
        # ---- 无引擎（CPU 单测 / 独立参考）----
        if not cache_ready and ctx.slot_mapping is None and not ctx.is_prefill:
            q = torch.cat([q_nope, q_pe_r], dim=-1)
            return self.o_proj(
                self._manual_attn(q, k_dense, v_dense, positions)
                .reshape(T, -1).to(hidden_states.dtype))

        # ---- 写路径：全批次 token 的 [c_kv | k̃]（写后才读，见各路径注释）----
        if cache_ready and ctx.slot_mapping is not None:
            mla_store(c_kv, k_pe_r, self.mla_cache, ctx.slot_mapping)

        q = torch.cat([q_nope, q_pe_r], dim=-1)      # [T, h, qk_head_dim]

        # ================= 混合批次：prefill 行 + decode 行 =================
        if ctx.is_mixed:
            if ctx.is_spec:
                # 投机混合：全批次 varlen（verify 行恒有前缀复用；prefill 行
                # start=num_cached 可能为 0 → fresh-only 直通）
                kk, vv = self._assemble(k_dense, v_dense,
                                        ctx.mla_pre_starts,
                                        ctx.cu_seqlens_q[1:] -
                                        ctx.cu_seqlens_q[:-1],
                                        ctx.cu_seqlens_q[:-1],
                                        ctx.mla_pre_idx)
                o = self._flash(q, kk, vv, ctx.cu_seqlens_q,
                                ctx.cu_seqlens_k, ctx.max_seqlen_q,
                                ctx.max_seqlen_k)
                return self.o_proj(o.reshape(T, -1))
            n_pre = ctx.n_prefill_tokens
            n_pre_rows = ctx.cu_seqlens_q.size(0) - 1
            if ctx.prefill_block_tables is not None:
                k_pre, v_pre = self._assemble(
                    k_dense[:n_pre], v_dense[:n_pre],
                    ctx.mla_pre_starts,
                    ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1],
                    ctx.cu_seqlens_q[:-1], ctx.mla_pre_idx)
            else:
                k_pre, v_pre = k_dense[:n_pre], v_dense[:n_pre]
            o_pre = self._flash(q[:n_pre], k_pre, v_pre,
                                ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                                ctx.max_seqlen_q, ctx.max_seqlen_k)
            o_dec = self._decode_rows(q[n_pre:], q_nope[n_pre:],
                                      q_pe_r[n_pre:])
            return self.o_proj(
                torch.cat([o_pre.reshape(n_pre, -1),
                           o_dec.reshape(T - n_pre, -1)], dim=0))
        # ================= 纯 prefill / 纯 spec ============================
        if ctx.is_prefill:
            if ctx.block_tables is not None:
                # cache-shaped：缓存稠密化 + fresh，行主序组装（cu_k 差分不变）
                kk, vv = self._assemble(k_dense, v_dense,
                                        ctx.mla_pre_starts,
                                        ctx.cu_seqlens_q[1:] -
                                        ctx.cu_seqlens_q[:-1],
                                        ctx.cu_seqlens_q[:-1],
                                        ctx.mla_pre_idx)
            else:
                kk, vv = k_dense, v_dense
            o = self._flash(q, kk, vv, ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                            ctx.max_seqlen_q, ctx.max_seqlen_k)
            return self.o_proj(o.reshape(T, -1))
        # ================= 纯 decode =======================================
        return self.o_proj(
            self._decode_rows(q, q_nope, q_pe_r).reshape(T, -1))

    # ------------------------------------------------------------------
    # decode 行：吸收式内核（float 视图）或稠密兜底（量化无视图）
    # ------------------------------------------------------------------
    def _decode_rows(self, q: torch.Tensor, q_nope: torch.Tensor,
                     q_pe_r: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        bs = q.shape[0]
        if bs == 0:
            return q.new_zeros(0, self.num_heads, self.qk_head_dim)
        if (ctx.context_lens is not None and ctx.block_tables is not None
                and self.has_float_views()):
            return self._decode_kernel(q_nope, q_pe_r)
        # ---- dense 兜底：整段缓存稠密化 + 逐行 1-token varlen ----
        assert ctx.mla_dec_starts is not None, "MLA decode 兜底缺行信息"
        # decode 行：keys = 整段 [0, lens)（本行刚写的自身 token 也在缓存里，
        # fresh = 0）；每行 1 个 query
        lens = ctx.mla_dec_starts
        cu_q = torch.arange(bs + 1, device=q.device, dtype=torch.int32)
        cu_k = torch.cat([torch.zeros(1, dtype=torch.int32, device=q.device),
                          torch.cumsum(lens, 0)])
        kk, vv = self._assemble(q, q.new_empty(0, 1, 1), lens,
                                torch.zeros(bs, dtype=torch.long,
                                            device=q.device),
                                torch.zeros(bs, dtype=torch.long,
                                            device=q.device),
                                ctx.mla_dec_idx)
        max_k = int(cu_k[-1].item())
        return self._flash(q, kk, vv, cu_q, cu_k, 1, max_k)

    def _decode_kernel(self, q_nope: torch.Tensor,
                       q_pe_r: torch.Tensor) -> torch.Tensor:
        """吸收式内核路径：每 token 每层只读 576 元素缓存。"""
        ctx = get_context()
        bs = q_nope.shape[0]
        wuk = self._wuk_t
        # q_abs = q_nope @ W_UKᵀ → [bs, h, kv_lora]（每步每头 128×512，K=nope）
        q_abs = torch.einsum("bhj,hjk->bhk",
                             q_nope.to(wuk.dtype), wuk).to(torch.float16)
        o_c = mla_decode_attention(q_abs, q_pe_r.to(torch.float16),
                                   self.mla_cache, ctx.block_tables,
                                   ctx.context_lens, self.scaling)
        # v 投影（吸收进输出侧）：o_i = W_UV_i · o_c_i → [bs, h, v_head]
        wuv = self._wuv_t
        out = torch.einsum("bhk,hvk->bhv", o_c.to(wuv.dtype), wuv)
        return out.reshape(bs, self.num_heads, self.v_head_dim)

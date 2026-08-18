import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def paged_decode_attention_fp8_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, cache_seqlens_ptr, o_ptr,
    k_scale, v_scale, softmax_scale,
    max_blocks, num_heads, kv_heads,
    head_dim: tl.constexpr, num_groups: tl.constexpr, QPAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Paged decode attention over an FP8 (E4M3) KV cache（v6，MMA版）。

    - GQA融合：一program处理 (seq, kv_head) 及该组全部q头，KV只读一次；
    - 直接 `tl.load` fp8 + `.to(tl.float32)`：Triton编译成硬件 cvt（无LUT gather）；
    - 计算走 MMA（tl.dot）：GQA组 G=2 不满足 dot 的 N>=16，用 QPAD=16 填充，
      q 按 mask 加载（填充列=0），softmax 前把填充列 mask 成 -inf，输出按 mask 存回；
      8x 的MMA计算浪费换内存效率（decode是memory-bound，算力有余）；
    - load 不做 mask（越界槽位是合法内存脏值），正确性由 scores 的 tok_mask 保证；
    - p 转 fp16 参与 acc 的 dot（flash-attn 同款做法）；num_warps=1 实测最准最快
      （w>=2 时跨warp归约顺序变化使误差放大到 1e-2）；
    - 实测（RTX 5060 Ti）：vs v4(LUT,BT32,w1) 全面 0.71-0.74x；vs v5(直接load)
      再快 ~25%。BLOCK_T∈{64,128} 反而更慢（寄存器压力）。
    """
    pid = tl.program_id(0)
    seq_id = pid // kv_heads
    kv_head = pid % kv_heads
    seqlen = tl.load(cache_seqlens_ptr + seq_id)
    offs_d = tl.arange(0, head_dim)
    offs_g = tl.arange(0, QPAD)
    g_valid = offs_g[None, :] < num_groups
    q_base = q_ptr + seq_id * num_heads * head_dim + kv_head * num_groups * head_dim
    q = tl.load(q_base + offs_d[:, None] + offs_g[None, :] * head_dim,
                mask=g_valid, other=0.0).to(tl.float32)                   # [D, QPAD]
    q16 = q.to(tl.float16)

    acc = tl.zeros([head_dim, QPAD], dtype=tl.float32)                    # [D, QPAD]
    m = tl.full([1, QPAD], float("-inf"), dtype=tl.float32)
    l = tl.zeros([1, QPAD], dtype=tl.float32)

    num_blocks = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_stride = BLOCK_SIZE * kv_heads * head_dim
    for b in range(num_blocks):
        block_id = tl.load(block_table_ptr + seq_id * max_blocks + b)
        base = block_id * block_stride + kv_head * head_dim
        for t in range(0, BLOCK_SIZE, BLOCK_T):
            offs_t = t + tl.arange(0, BLOCK_T)
            tok_mask = (b * BLOCK_SIZE + offs_t) < seqlen
            k_ptrs = k_cache_ptr + base + offs_t[:, None] * (kv_heads * head_dim) + offs_d[None, :]
            k16 = (tl.load(k_ptrs).to(tl.float32) * k_scale).to(tl.float16)  # [T, D]
            s = tl.dot(k16, q16, out_dtype=tl.float32) * softmax_scale       # [T, QPAD]
            s = tl.where(tok_mask[:, None] & g_valid, s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=0)[None, :])
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new)
            l = l * alpha + tl.sum(p, axis=0)[None, :]
            v_ptrs = v_cache_ptr + base + offs_t[:, None] * (kv_heads * head_dim) + offs_d[None, :]
            v_t = (tl.load(v_ptrs).to(tl.float32) * v_scale).to(tl.float16)  # [T, D]
            acc = acc * alpha + tl.dot(tl.trans(v_t), p.to(tl.float16), out_dtype=tl.float32)
            m = m_new
    o = acc / l                                                            # [D, QPAD]
    tl.store(o_ptr + seq_id * num_heads * head_dim + kv_head * num_groups * head_dim
             + offs_d[:, None] + offs_g[None, :] * head_dim,
             o.to(q_ptr.dtype.element_ty), mask=g_valid)


def paged_decode_attention_fp8(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                               block_table: torch.Tensor, cache_seqlens: torch.Tensor,
                               k_scale: float, v_scale: float, softmax_scale: float) -> torch.Tensor:
    bs, num_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    max_blocks = block_table.shape[1]
    num_groups = num_heads // kv_heads
    qpad = max(16, num_groups)
    o = torch.empty_like(q)
    grid = (bs * kv_heads,)
    paged_decode_attention_fp8_kernel[grid](
        q, k_cache, v_cache, block_table, cache_seqlens, o,
        k_scale, v_scale, softmax_scale,
        max_blocks, num_heads, kv_heads,
        head_dim=head_dim, num_groups=num_groups, QPAD=qpad,
        BLOCK_SIZE=k_cache.shape[1], BLOCK_T=32,
        num_warps=1,
    )
    return o


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # ---- FP8 KV cache 状态（由ModelRunner在allocate_kv_cache/校准时设置） ----
        self.use_fp8 = False                 # 是否启用fp8(E4M3) KV存储
        self.k_scale = 1.0                   # 本层K的固定反量化scale（warmup校准）
        self.v_scale = 1.0                   # 本层V的固定反量化scale
        self.inv_k_scale = 1.0
        self.inv_v_scale = 1.0
        self.calibrating = False             # 校准阶段：记录本层K/V的max|·|
        self.cal_max_k = 0.0
        self.cal_max_v = 0.0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 校准时记录K/V动态范围（在store之前）
        if self.calibrating:
            self.cal_max_k = max(self.cal_max_k, k.abs().max().item())
            self.cal_max_v = max(self.cal_max_v, v.abs().max().item())
        if k_cache.numel() and v_cache.numel():
            if self.use_fp8:
                # 写路径：先用本层固定scale量化为fp8（E4M3）。
                # 注意：torch的 fp32->fp8 cast 溢出不饱和而是产生NaN位模式(0x7F/0xFF)
                # （实测 500 -> 0x7F），必须先 clamp 到 E4M3 最大值 448。
                # 校准数据外出现更大激活（真实prompt > 校准token）时，溢出必须饱和为
                # 448 而非 NaN——v4的LUT把NaN位模式读成0.0掩盖了此bug（见BENCHMARKS.md）。
                kq = (k.float() * self.inv_k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                vq = (v.float() * self.inv_v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            else:
                kq, vq = k, v
            store_kvcache(kq, vq, k_cache, v_cache, context.slot_mapping)
        if context.is_mixed:
            # 混合批次（vLLM V1同款调度）：prefill行在前、decode行在后。
            # 写路径已在上方覆盖全批次（slot_mapping含两组槽位）。
            if context.is_spec:
                # 投机混合步：verify行恒有前缀复用（num_cached=len-1）→ 全批次varlen，
                # K/V必须为缓存形状[blocks, block_size, ...]（flash按k.shape[1]推断
                # block size；见BENCHMARKS.md §5.3 的varlen+分块序列坑）。
                if self.use_fp8:
                    k_pre = k_cache.to(k.dtype) * self.k_scale
                    v_pre = v_cache.to(v.dtype) * self.v_scale
                else:
                    k_pre, v_pre = k_cache, v_cache
                o = flash_attn_varlen_func(q, k_pre, v_pre,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True,
                                           block_table=context.prefill_block_tables)
                return o
            # prefill组：flash varlen；若本组存在分块序列（key_len>query_len，含自己
            # 上一chunk写入的缓存）则k/v必须用缓存形状[blocks, block_size, ...]——
            # flash varlen的block_table按k.shape[1]推断block size；
            # decode组：fp16走flash_attn_with_kvcache / fp8走自研内核（读缓存）。
            n_pre = context.n_prefill_tokens
            if context.prefill_block_tables is not None:
                if self.use_fp8:
                    k_pre = k_cache.to(k.dtype) * self.k_scale
                    v_pre = v_cache.to(v.dtype) * self.v_scale
                else:
                    k_pre, v_pre = k_cache, v_cache
            else:
                k_pre, v_pre = k[:n_pre], v[:n_pre]
            o_pre = flash_attn_varlen_func(q[:n_pre], k_pre, v_pre,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True,
                                           block_table=context.prefill_block_tables)
            q_dec = q[n_pre:]
            if self.use_fp8:
                o_dec = paged_decode_attention_fp8(q_dec, k_cache, v_cache,
                                                   context.block_tables, context.context_lens,
                                                   self.k_scale, self.v_scale, self.scale)
            else:
                o_dec = flash_attn_with_kvcache(q_dec.unsqueeze(1), k_cache, v_cache,
                                                cache_seqlens=context.context_lens,
                                                block_table=context.block_tables,
                                                softmax_scale=self.scale, causal=True).squeeze(1)
            return torch.cat([o_pre, o_dec], dim=0)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache：KV来自缓存
                if self.use_fp8:
                    # 反量化成模型dtype再交给flash-attn（正确性优先；性能限制见BENCHMARKS.md）
                    k = k_cache.to(k.dtype) * self.k_scale
                    v = v_cache.to(v.dtype) * self.v_scale
                else:
                    k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if self.use_fp8:
                # 读路径：自研Triton内核直接读fp8缓存，寄存器内反量化
                o = paged_decode_attention_fp8(q, k_cache, v_cache,
                                               context.block_tables, context.context_lens,
                                               self.k_scale, self.v_scale, self.scale)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        return o

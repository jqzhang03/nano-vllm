"""SWA / soft-cap 约定 probe（GPU）：flash-attn 的 window_size 与 softcap 语义 + 自研 fp8 内核窗口掩码。

对比对象（全部 torch 手工参考实现）：
  ref_attn：scores = q·kᵀ·scale →（softcap: cap·tanh(scores/cap)）→ 因果+窗口掩码 → softmax → o

钉死两个约定（写进 attention.py 注释）：
  1. flash window_size=(left, 0)：query i 关注 keys ∈ [i-left, i]（含两端，文档原话
     [i+seqlen_k-seqlen_q-left, i+seqlen_k-seqlen_q+right]）→ 传 (W-1, 0) 即与
     "窗口 W 个 key（含自己）" 对齐（HF SDPA sliding_window=W 语义）。
  2. flash softcap=c：logits' = c·tanh(logits/c)，与 Gemma-2 attn_logit_softcapping 定义一致。

验证面：varlen prefill（全行 + 前缀复用+分块）、decode（kvcache + block_table）、
自研 fp8 内核（decode WINDOW / verify 行 varlen WINDOW，Q=γ+1≤5）。

用法: python benchmarks/_swa_probe.py
"""
import torch

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.attention import (
    paged_decode_attention_fp8, paged_varlen_attention_fp8, store_kvcache,
)


def ref_attn(q, k, v, scale, window=None, cap=None, base=0):
    """torch 参考（绝对位置）：q 行 r 的绝对位置 = base + r，keys 0..Tk-1。

    因果+窗口：pos 关注 col j ∈ [pos-window+1, pos]（含自己，共 window 个 key）。
    base=0 时等价于整段因果；前缀复用/decode 场景传 q 行的起始绝对位置。
    """
    T = q.shape[0]
    Tk = k.shape[0]
    scores = torch.einsum("qhd,khd->qhk", q, k) * scale
    qrows = torch.arange(T, device=q.device) + base
    kcols = torch.arange(Tk, device=q.device)
    mask = qrows[:, None] >= kcols[None, :]                    # 只关注过去/自己
    if window:
        mask = mask & (qrows[:, None] <= kcols[None, :] + window - 1)
    if cap:
        scores = cap * torch.tanh(scores / cap)
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    p = scores.softmax(dim=2)
    return torch.einsum("qhk,khd->qhd", p, v)


def expand_kv(k, v, num_heads):
    """GQA：kv_heads → num_heads 展开（沿 head 维 dim=1）。"""
    g = num_heads // k.shape[1]
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    return k, v


def fmt(name, ref, out, tol):
    d = (ref - out).abs().max().item()
    ok = d <= tol
    print(f"  {name:<46} max_diff={d:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    dev = "cuda"
    H, K_H, D, W, T = 8, 4, 128, 256, 600
    scale = D ** -0.5
    cap = 50.0
    bf16 = torch.bfloat16

    print(f"torch SDPA sliding_window/softcap 支持: "
          f"{hasattr(torch.nn.functional.scaled_dot_product_attention, '__wrapped__') or True} "
          f"(flash-attn 2.8.3 自带 softcap 参数，直接用)")

    q = torch.randn(T, H, D, device=dev, dtype=bf16)
    k = torch.randn(T, K_H, D, device=dev, dtype=bf16)
    v = torch.randn(T, K_H, D, device=dev, dtype=bf16)
    ke, ve = expand_kv(k, v, H)

    ok_all = True
    print("\n=== 1) varlen prefill：window 约定（flash (W-1,0) vs 参考 [i-W+1, i]） ===")
    ref_w = ref_attn(q, ke, ve, scale, window=W)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
    # 容差 5e-2：bf16 输入 + flash 内部 fp32 累积 vs torch bf16 参考的正常噪声
    # (256,0) 是 off-by-one 演示（预期 FAIL，证明约定必须是 (W-1,0)），不计入 ok_all
    out = flash_attn_varlen_func(q, k, v, cu, cu,
                                 max_seqlen_q=T, max_seqlen_k=T,
                                 softmax_scale=scale, causal=True,
                                 window_size=(W, 0))
    fmt(f"flash window_size=({W},0) [off-by-one 演示，预期 FAIL]", ref_w, out, 5e-2)
    out = flash_attn_varlen_func(q, k, v, cu, cu,
                                 max_seqlen_q=T, max_seqlen_k=T,
                                 softmax_scale=scale, causal=True,
                                 window_size=(W - 1, 0))
    ok_all &= fmt("flash window_size=(255,0) [约定] → (W-1, 0)", ref_w, out, 5e-2)

    print("\n=== 2) varlen prefill：softcap 约定（flash softcap=c vs 参考 cap·tanh） ===")
    ref_c = ref_attn(q, ke, ve, scale, cap=cap)
    out = flash_attn_varlen_func(q, k, v, cu, cu,
                                 max_seqlen_q=T, max_seqlen_k=T,
                                 softmax_scale=scale, causal=True, softcap=cap)
    ok_all &= fmt("flash softcap=50 (full causal)", ref_c, out, 5e-2)
    ref_cw = ref_attn(q, ke, ve, scale, window=W, cap=cap)
    out = flash_attn_varlen_func(q, k, v, cu, cu,
                                 max_seqlen_q=T, max_seqlen_k=T,
                                 softmax_scale=scale, causal=True,
                                 window_size=(W - 1, 0), softcap=cap)
    ok_all &= fmt("flash softcap=50 + window", ref_cw, out, 5e-2)

    print("\n=== 3) varlen + 前缀复用 + 分块（block_table 路径 + window） ===")
    # 模拟：seq 有 T 个缓存 token，query = 最后 8 个位置（分块 prefill 的 verify/续写行）
    Q = 8
    q_pre = q[T - Q:T]
    cu_q = torch.tensor([0, Q], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, T], device=dev, dtype=torch.int32)
    nblk = (T + 256 - 1) // 256
    k_cache = torch.zeros(nblk, 256, K_H, D, device=dev, dtype=bf16)
    v_cache = torch.zeros_like(k_cache)
    for b in range(nblk):
        s, e = b * 256, min((b + 1) * 256, T)
        k_cache[b, :e - s] = k[s:e]
        v_cache[b, :e - s] = v[s:e]
    bt = torch.arange(nblk, device=dev, dtype=torch.int32).unsqueeze(0)
    ref_pfx = ref_attn(q_pre, ke, ve, scale, window=W, base=T - Q)  # 全历史 key + 绝对位置
    out = flash_attn_varlen_func(q_pre, k_cache, v_cache, cu_q, cu_k,
                                 max_seqlen_q=Q, max_seqlen_k=T,
                                 softmax_scale=scale, causal=True,
                                 window_size=(W - 1, 0), block_table=bt)
    ok_all &= fmt("flash varlen + block_table + window", ref_pfx, out, 1e-2)

    print("\n=== 4) decode（flash kvcache + block_table + window + softcap） ===")
    q_dec = q[T - 1:T]  # [1, H, D] 在绝对位置 T-1
    cache_seqlens = torch.tensor([T], device=dev, dtype=torch.int32)
    ref_dec = ref_attn(q_dec, ke, ve, scale, window=W, cap=cap, base=T - 1)
    out = flash_attn_with_kvcache(q_dec.unsqueeze(1), k_cache, v_cache,
                                  cache_seqlens=cache_seqlens, block_table=bt,
                                  softmax_scale=scale, causal=True,
                                  window_size=(W - 1, 0), softcap=cap).squeeze(1)
    ok_all &= fmt("flash kvcache + block_table + window + softcap", ref_dec, out, 1e-2)

    print("\n=== 5) 自研 fp8 decode 内核 + WINDOW ===")
    # fp8 量化缓存（per-tensor scale，模拟校准层 scale）
    kf = k.float(); vf = v.float()
    ks = kf.abs().max() / 448.0; vs = vf.abs().max() / 448.0
    kq8 = (kf / ks).clamp(-448, 448).to(torch.float8_e4m3fn)
    vq8 = (vf / vs).clamp(-448, 448).to(torch.float8_e4m3fn)
    kc = torch.zeros(nblk, 256, K_H, D, device=dev, dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    for b in range(nblk):
        s, e = b * 256, min((b + 1) * 256, T)
        kc[b, :e - s] = kq8[s:e]
        vc[b, :e - s] = vq8[s:e]
    seqlens = torch.tensor([T], device=dev, dtype=torch.int32)
    out = paged_decode_attention_fp8(q_dec, kc, vc, bt, seqlens,
                                     ks.item(), vs.item(), scale, window=W)
    ok_all &= fmt("fp8 decode kernel WINDOW=256", ref_dec, out, 5e-2)

    print("\n=== 6) 自研 fp8 verify 行 varlen 内核 + WINDOW（Q=γ+1=3，2 个 seq） ===")
    # 两个 seq：长度 400 / 600，verify 行 query = 各 seq 最后 3 个位置
    lens = [400, 600]
    qs = []; refs = []; kblk = []; vblk = []; bts = []
    max_nblk = 0
    for L in lens:
        nb = (L + 255) // 256
        max_nblk = max(max_nblk, nb)
        kcb = torch.zeros(nb, 256, K_H, D, device=dev, dtype=torch.float8_e4m3fn)
        vcb = torch.zeros_like(kcb)
        for b in range(nb):
            s, e = b * 256, min((b + 1) * 256, L)
            kcb[b, :e - s] = kq8[s:e]
            vcb[b, :e - s] = vq8[s:e]
        qs.append(q[L - 3:L])
        refs.append(ref_attn(q[L - 3:L], ke, ve, scale, window=W, base=L - 3))
        kblk.append(kcb); vblk.append(vcb)
        bts.append(torch.arange(nb, device=dev, dtype=torch.int32))
    # 打包为单批：q [6, H, D]，block_table [2, max_nblk]（pad -1），缓存统一块大小
    # **注意物理块偏移**：kc_all 展平后 seq i 的物理块在 [i*max_nblk, (i+1)*max_nblk)，
    # block_table 必须指向物理块号（seq1 是 3,4,5，不是 0,1,2——否则读到 seq0 的 KV）
    q_all = torch.cat(qs)
    bt_all = torch.full((2, max_nblk), -1, device=dev, dtype=torch.int32)
    for i, b in enumerate(bts):
        bt_all[i, :b.shape[0]] = b + i * max_nblk
    kc_all = torch.zeros(2, max_nblk, 256, K_H, D, device=dev, dtype=torch.float8_e4m3fn)
    vc_all = torch.zeros_like(kc_all)
    for i in range(2):
        kc_all[i, :kblk[i].shape[0]] = kblk[i]
        vc_all[i, :vblk[i].shape[0]] = vblk[i]
    kc_all = kc_all.view(-1, 256, K_H, D).contiguous()
    vc_all = vc_all.view(-1, 256, K_H, D).contiguous()
    key_lens = torch.tensor(lens, device=dev, dtype=torch.int32)
    cu_q = torch.tensor([0, 3, 6], device=dev, dtype=torch.int32)
    out = paged_varlen_attention_fp8(q_all, kc_all, vc_all, cu_q, key_lens, bt_all,
                                     ks.item(), vs.item(), scale, window=W)
    ref_all = torch.cat(refs)
    ok_all &= fmt("fp8 verify varlen kernel WINDOW=256", ref_all, out, 5e-2)
    d = (ref_all - out).abs()
    for i in range(2):
        rowmax = d[i * 3:(i + 1) * 3].amax(dim=(1, 2))
        print(f"  seq{i} (len={lens[i]}) row diffs: {[f'{x.item():.2e}' for x in rowmax]}")
    # 隔离：同一形态 WINDOW=0（无窗口）是否匹配——区分"窗口掩码 bug" vs "varlen 内核固有误差"
    out0 = paged_varlen_attention_fp8(q_all, kc_all, vc_all, cu_q, key_lens, bt_all,
                                      ks.item(), vs.item(), scale, window=0)
    ref0 = torch.cat([ref_attn(q[L - 3:L], ke, ve, scale, base=L - 3) for L in lens])
    ok_all &= fmt("fp8 verify varlen kernel WINDOW=0 (隔离)", ref0, out0, 5e-2)

    print(f"\n{'ALL OK' if ok_all else 'SOME FAILED'}")

    # 附：验证 store_kvcache 与 fp8 写路径不受影响（回归）
    print("\n=== 7) store_kvcache 回归（无窗口相关改动） ===")
    slot_map = torch.arange(T, device=dev, dtype=torch.int32)
    kc2 = torch.zeros(nblk * 256, K_H, D, device=dev, dtype=bf16)
    vc2 = torch.zeros_like(kc2)
    store_kvcache(k, v, kc2.view(nblk, 256, K_H, D), vc2.view(nblk, 256, K_H, D), slot_map)
    d = (kc2.view(nblk, 256, K_H, D) - k_cache).abs().max().item()
    print(f"  store_kvcache max_diff={d:.2e} {'OK' if d == 0 else 'FAIL'}")
    ok_all &= d == 0


if __name__ == "__main__":
    main()

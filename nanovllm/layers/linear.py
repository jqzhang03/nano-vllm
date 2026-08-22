import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


@triton.jit
def gemm_int8_kernel(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K, num_groups,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """W8A8 int8 GEMM（per-group 权重 scale）: C = (A_int8 @ B_int8.T) * a_scale * w_scale。

    A: [M, K] int8（per-token scale a_scale[M]）；B: [N, K] int8。
    w_scale: [N, num_groups]，按 K 维分组（GROUP=128，AWQ 标准）；BLOCK_K == GROUP_SIZE，
    每个 K 块恰好一组 → 块内 int32 累加后乘本组 scale、以 fp32 跨组累加
    （组间 scale 不同，不能像 per-channel 那样最后统一乘）。
    per-channel 是 num_groups=1 的特例。
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_mask = offs_m[:, None] < M
    b_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        group = (k // BLOCK_K) % num_groups  # per-channel (num_groups=1) 时恒为0，防越界
        b_s = tl.load(b_scale_ptr + offs_n[None, :] * num_groups + group,
                      mask=b_mask, other=1.0)
        acc += tl.dot(a, b, out_dtype=tl.int32).to(tl.float32) * b_s
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_s = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0)
    out = acc * a_s[:, None]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=a_mask & b_mask)


def w8a8_gemm(a_int8: torch.Tensor, b_int8: torch.Tensor,
              a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """W8A8线性：x [M, K] int8 × W [N, K] int8 → [M, N]（fp16/bf16）。

    b_scale: [N, num_groups] per-group 权重scale（num_groups=1 即 per-channel）。
    """
    M, K = a_int8.shape
    N = b_int8.shape[0]
    assert K % 128 == 0, "group size 128 must divide K"
    out = torch.empty(M, N, device=a_int8.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    gemm_int8_kernel[grid](
        a_int8, b_int8, out, a_scale, b_scale,
        M, N, K, b_scale.shape[1],
        a_int8.stride(0), a_int8.stride(1),
        b_int8.stride(1), b_int8.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_M=8,
        num_warps=4,
    )
    return out


@triton.jit
def gemm_int4_kernel(
    a_ptr, b_ptr, c_ptr, b_scale_ptr,
    M, N, K, num_groups,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """INT4 反量化 GEMM: C = A @ W_dequant（bf16 激活 × int4 权重，寄存器内反量化）。

    A: [M, K] bf16；B: [N, K//2] int8 打包权重——字节 (n, j) 的低/高半字节分别是
    输出通道 n 在输入 k=2j / 2j+1 上的 int4 码（offset 编码：码-8=真值，对称量化）；
    b_scale: [N, num_groups]（K 维按 group_size=BLOCK_K 分组，组amax/7）。
    反量化在寄存器内完成：拆半字节 → (码-8)*组scale → bf16；**2-dot 拆分**——
    a_e/a_o 按 K 奇偶步长2加载，与 lo/hi 分别 dot（避免 interleave 的布局转换，
    调优见 benchmarks/_int4_tune.py：小 M 权重带宽减半 → 最多 4.4×）。
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # 输出通道
    offs_k2 = tl.arange(0, BLOCK_K // 2)               # 打包列（k 对）

    a_e_ptrs = a_ptr + offs_m[:, None] * stride_am + (2 * offs_k2)[None, :] * stride_ak
    a_o_ptrs = a_e_ptrs + stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k2[:, None] * stride_bk
    a_mask = offs_m[:, None] < M
    b_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_e = tl.load(a_e_ptrs, mask=a_mask, other=0.0)                  # (BM, BK/2) 偶数k
        a_o = tl.load(a_o_ptrs, mask=a_mask, other=0.0)                  # (BM, BK/2) 奇数k
        b8 = tl.load(b_ptrs, mask=b_mask, other=0)                       # (BK/2, BN) int8
        group = (k // BLOCK_K) % num_groups                              # 本K块恰为一组
        s = tl.load(b_scale_ptr + offs_n * num_groups + group, mask=offs_n < N, other=1.0)
        lo = ((b8 & 0x0F) - 8).to(tl.float32) * s[None, :]               # 偶数k真值
        hi = (((b8 >> 4) & 0x0F) - 8).to(tl.float32) * s[None, :]        # 奇数k（>>算术移位，&0x0F遮符号扩展）
        acc += tl.dot(a_e, lo.to(tl.bfloat16), out_dtype=tl.float32)
        acc += tl.dot(a_o, hi.to(tl.bfloat16), out_dtype=tl.float32)
        a_e_ptrs += BLOCK_K * stride_ak
        a_o_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=a_mask & b_mask)


def int4_gemm(a: torch.Tensor, b_int4: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """INT4 线性：a [M, K] bf16 × 打包 int4 [N, K//2] → [M, N]（bf16）。

    b_scale: [N, num_groups]（K 维按 128 分组）；b_int4 低/高半字节 = k=2j/2j+1。
    tile 按 M 自适应：小 M（decode，权重带宽主导）用 16×128；大 M 用 64×256。
    """
    M, K = a.shape
    N = b_scale.shape[0]
    assert b_int4.shape == (N, K // 2), f"expected packed [N, K//2], got {b_int4.shape}"
    assert K % 128 == 0, "group size 128 must divide K"
    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    if M <= 128:
        bm, bn, warps = 16, 128, 4
    else:
        bm, bn, warps = 64, 256, 8
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_int4_kernel[grid](
        a, b_int4, out, b_scale,
        M, N, K, b_scale.shape[1],
        a.stride(0), a.stride(1),
        b_int4.stride(0), b_int4.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
        num_warps=warps, num_stages=2,
    )
    return out


@triton.jit
def gemm_sparse24_kernel(
    a_ptr, v_ptr, idx_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_vn, stride_vk, stride_in, stride_ik, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """2:4 稀疏 GEMM: C = A @ W_dequant（每 4 个连续输入通道保留 2 个非零）。

    A: [M, K] bf16；2:4 剪枝后的打包权重：
      v:   [N, K//2] bf16  行 i=(g, s) = 组 g 的槽 s 的非零值（每槽恰好一个）
      idx: [N, K//4] uint8 字节 (n, j) 低2位 = 组 j 槽0 的 k 偏移(0..3)，bit2-3 = 槽1
    （两槽偏移必不同——组内两个非零位置互斥）。
    内核 4 路拆分：对 p∈{0..3}，a_p = A[:, 4g+p]（步长4的stride加载，L2复用同一缓存行），
    v_p = 槽值按 idx==p 掩码（两槽至多一个命中，无求和误差）→ acc += dot(a_p, v_p)。
    权重字节流量 = 0.625×稠密（v 半量 bf16 + idx 1/4 量）；MMA 数与稠密相同——
    软件 2:4 是带宽优化，计算加速需要硬件稀疏 MMA（cuSPARSELt/CUTLASS）。
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, BLOCK_K // 4)          # 组索引（本K块内）
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # A 的 4 路 strided 加载（每路 (BM, BK/4)，K 步长 4）
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + 4 * offs_g)[None, :] * stride_ak
        a0 = tl.load(a_ptrs, mask=m_mask, other=0.0)
        a1 = tl.load(a_ptrs + stride_ak, mask=m_mask, other=0.0)
        a2 = tl.load(a_ptrs + 2 * stride_ak, mask=m_mask, other=0.0)
        a3 = tl.load(a_ptrs + 3 * stride_ak, mask=m_mask, other=0.0)
        # v 的两路加载：槽0（行 2g）与槽1（行 2g+1），均为 (BK/4, BN)
        v_ptrs = v_ptr + offs_n[None, :] * stride_vn + (k // 2 + 2 * offs_g)[:, None] * stride_vk
        v_lo = tl.load(v_ptrs, mask=n_mask, other=0.0)
        v_hi = tl.load(v_ptrs + stride_vk, mask=n_mask, other=0.0)
        # idx 打包字节（列=组，行=n）
        i_ptrs = idx_ptr + offs_n[None, :] * stride_in + (k // 4 + offs_g)[:, None] * stride_ik
        b8 = tl.load(i_ptrs, mask=n_mask, other=0)                 # (BK/4, BN) uint8
        i_lo = b8 & 0x3
        i_hi = (b8 >> 2) & 0x3
        # 4 路掩码重建 [K, N] 权重块并 dot（每路 (BM, BK/4)×(BK/4, BN)）
        vp0 = tl.where(i_lo == 0, v_lo, 0.0) + tl.where(i_hi == 0, v_hi, 0.0)
        acc += tl.dot(a0, vp0.to(tl.bfloat16), out_dtype=tl.float32)
        vp1 = tl.where(i_lo == 1, v_lo, 0.0) + tl.where(i_hi == 1, v_hi, 0.0)
        acc += tl.dot(a1, vp1.to(tl.bfloat16), out_dtype=tl.float32)
        vp2 = tl.where(i_lo == 2, v_lo, 0.0) + tl.where(i_hi == 2, v_hi, 0.0)
        acc += tl.dot(a2, vp2.to(tl.bfloat16), out_dtype=tl.float32)
        vp3 = tl.where(i_lo == 3, v_lo, 0.0) + tl.where(i_hi == 3, v_hi, 0.0)
        acc += tl.dot(a3, vp3.to(tl.bfloat16), out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask & (offs_n[None, :] < N))


def sparse24_gemm(a: torch.Tensor, v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """2:4 稀疏线性：a [M, K] bf16 × 剪枝打包 [v (N, K//2) bf16, idx (N, K//4) uint8] → [M, N]。

    tile 按 M 自适应（调优见 benchmarks/_sparse24_tune.py）：小 M（权重带宽主导）
    用 16×128 + stages=3；大 M 用 64×256。
    """
    M, K = a.shape
    N = v.shape[0]
    assert v.shape == (N, K // 2), f"expected v [N, K//2], got {v.shape}"
    assert idx.shape == (N, K // 4), f"expected idx [N, K//4], got {idx.shape}"
    assert K % 4 == 0
    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    if M <= 128:
        bm, bn, warps, stages = 16, 128, 4, 3
    else:
        bm, bn, warps, stages = 64, 256, 8, 2
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_sparse24_kernel[grid](
        a, v, idx, out, M, N, K,
        a.stride(0), a.stride(1), v.stride(0), v.stride(1), idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    return out


class WeightQuantMixin:
    """int4 / sparse24 量化共享实现（LinearBase 与 ParallelLMHead 复用）。"""

    # 稠密反量化路径的路由阈值（微基准 _int4_tune.py 结论：int4 只在"小M大N"的
    # 权重带宽主导形态赢）——M≤128 且 N≥2048 走 int4 内核，其余（大M prefill/
    # decode、小N 的 o_proj/down_proj）走 w_deq 的 cuBLAS 稠密路径，收掉大M亏损
    # 与 TTFT 回归（见 BENCHMARKS.md §10）。
    int4_max_m = 128
    int4_min_n = 2048

    def quantize_int4(self, awq_scale: torch.Tensor | None = None, group_size: int = 128,
                      dense_path: bool = True):
        """per-group（K维按group_size分组，AWQ标准）int4权重量化（对称，组scale=amax/7，码偏移+8）。

        awq_scale: [K] 校准得到的激活感知缩放（AWQ式 s = mean|X|^α 归一化），折叠进权重
        W'=W·s、推理时输入侧除 s（恒等变换 W·X = (W·s)·(X/s)，见 AWQ 论文：
        大激活通道的权重被放大→组内量化相对误差小；激活被缩小→误差贡献小）。
        注意方向：权重**乘** s、激活**除** s（反了会塌缩——大激活通道权重被除到~0、
        组 amax 被其他通道主导，实测 KL 1.08 → 12.4，且 α 搜索会收敛到 0）。
        打包沿输入维：
        字节 (n, j) 低/高半字节 = 输出通道 n 在 k=2j/2j+1 的 int4 码 → 存储 [N, K//2] int8
        （内核 2-dot 拆分：a_e/a_o 按 K 奇偶步长2加载，与 lo/hi 分别 dot，
        避免 interleave 的寄存器布局转换——调优见 benchmarks/_int4_tune.py）。
        """
        w = self.weight.detach().float()
        if awq_scale is not None:
            awq_scale = awq_scale.to(w.device)
            w = w * awq_scale[None, :]
            self.register_buffer("awq_scale", awq_scale.to(self.weight.dtype).contiguous())
        N, K = w.shape
        assert K % group_size == 0 and K % 2 == 0
        w_g = w.view(N, K // group_size, group_size)
        w_scale = w_g.abs().amax(dim=2).clamp(min=1e-8) / 7.0              # [N, K//g]
        q = torch.clamp(torch.round(w / w_scale[:, torch.arange(K) // group_size]), -7, 7)
        q4 = (q.to(torch.int8) + 8).to(torch.uint8)                        # offset码 1..15
        packed = (q4[:, 0::2] | (q4[:, 1::2] << 4)).to(torch.int8)         # [N, K//2]
        self.register_buffer("w_int4", packed.contiguous())                # 字节低半=偶数k、高半=奇数k
        self.register_buffer("w_int4_scale", w_scale.to(self.weight.dtype).contiguous())
        # 稠密反量化副本（dense_path=True 时）：dequant(q, scale)（AWQ 时再 /s 还原到
        # 原始尺度）——大 M / 小 N 层走 F.linear 稠密路径，与 int4 路径数学恒等（同一份
        # q/scale）。显存代价：量化部分从 0.27× 变 1.27×（int4+bf16 双份，比 fp16 原权重
        # 还大）——这是"吞吐无损模式"的定价，关掉（dense_path=False）即纯 int4 显存模式。
        if dense_path:
            packed_i = packed  # [N, K//2] int8
            lo = (packed_i & 0x0F) - 8
            hi = ((packed_i >> 4).to(torch.int8) & 0x0F) - 8
            w_deq = torch.zeros(N, K, device=w.device, dtype=torch.float32)
            w_deq[:, 0::2] = lo.float() * w_scale[:, torch.arange(K // 2) // (group_size // 2)]
            w_deq[:, 1::2] = hi.float() * w_scale[:, torch.arange(K // 2) // (group_size // 2)]
            if awq_scale is not None:
                w_deq = w_deq / awq_scale.clamp(min=1e-8)[None, :]
            self.register_buffer("w_deq", w_deq.to(self.weight.dtype).contiguous())
        self.int4 = True
        del self.weight  # 释放fp16权重（weight_loader只在加载期使用）

    def _int4_forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = getattr(self, "bias", None)
        # 路由（dense_path 模式）：小M大N → int4 内核（带宽赢）；否则稠密反量化路径
        # （cuBLAS，w_deq 已含 AWQ 还原，无需激活侧缩放）
        w_deq = getattr(self, "w_deq", None)
        if w_deq is not None and (x.shape[0] > self.int4_max_m
                                  or self.w_int4.shape[0] < self.int4_min_n):
            y = F.linear(x, w_deq, bias)
            return y
        awq = getattr(self, "awq_scale", None)
        if awq is not None:
            x = x / awq[None, :]  # AWQ恒等变换的激活侧（权重侧已折叠 W'=W·s）
        y = int4_gemm(x, self.w_int4, self.w_int4_scale)
        if bias is not None:
            y = y + bias
        return y

    def quantize_sparse24(self):
        """2:4 结构化剪枝 + 自研 Triton 稀疏 GEMM（打包 v + idx，见 gemm_sparse24_kernel）。

        每 4 个连续输入通道按幅值保留 2 个（magnitude pruning，NVIDIA 2:4 结构）。
        打包：v [N, K//2] bf16 非零值 + idx [N, K//4] uint8 槽位偏移（每槽 2bit，两槽打包）。
        权重字节 = 0.625×稠密。为什么不走 torch semi-structured（cuSPARSELt/CUTLASS）：
        CUTLASS 后端仅支持 sm_8x；cuSPARSELt 在 sm_120 可用但每调用开销 0.3-0.5ms
        （MLP 层实测只有稠密 0.02-0.17×，仅 lm_head 大权重小M有 1.7×）——见
        benchmarks/_sparse24_probe.py。
        """
        w = self.weight.detach()
        N, K = w.shape
        assert K % 4 == 0
        w4 = w.view(N, K // 4, 4)
        keep = w4.abs().argsort(dim=-1, descending=True)[..., :2]     # (N, K//4, 2) 槽位偏移
        mask = torch.zeros_like(w4, dtype=torch.bool).scatter_(-1, keep, True)
        v = torch.gather(w4 * mask, -1, keep).view(N, K // 2)          # 非零值（行 i=2g+s）
        idx = keep.to(torch.uint8).view(N, K // 2)                     # 每槽 2bit 偏移
        idx_packed = (idx[:, 0::2] | (idx[:, 1::2] << 2)).contiguous()  # (N, K//4)
        self.register_buffer("w_s24_v", v.contiguous())
        self.register_buffer("w_s24_idx", idx_packed)
        self.sparse24 = True
        del self.weight  # 释放fp16权重（weight_loader只在加载期使用）

    def _sparse24_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sparse24_gemm(x, self.w_s24_v, self.w_s24_idx)
        bias = getattr(self, "bias", None)
        if bias is not None:
            y = y + bias
        return y


class LinearBase(WeightQuantMixin, nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        # ---- 量化状态（load_model之后由ModelRunner调用对应quantize方法） ----
        self.w8a8 = False  # w_int8/w_scale/smooth缓冲由quantize_w8a8创建（不能先占普通属性名）
        self.int4 = False  # w_int4/w_int4_scale/awq_scale缓冲由quantize_int4创建
        self.sparse24 = False  # w_s24_v/w_s24_idx缓冲由quantize_sparse24创建

    def quantize_w8a8(self, x_max: torch.Tensor | None = None):
        """per-group（K维按128分组，AWQ标准）int8权重量化 + scale；TP分片按各自输出维量化。

        x_max: 校准得到的本层输入逐通道amax（[K]），提供时启用SmoothQuant式平滑：
        激活离群值会让per-token量化的max/mean比值变大（naive W8A8单层误差~2%），
        按 s[k] = x_max[k]^0.5 / w_col_max[k]^0.5 把激活尺度折进权重（X'=X·s, W'=W/s，
        数学恒等），使X'更平滑、量化误差显著下降。
        权重 scale 按 K 维 128 一组（组内amax/127）——比 per-channel 细 8 倍
        （K=1024），是精度缺口的主要修复手段（见BENCHMARKS.md §8）。
        """
        w = self.weight.detach().float()
        if x_max is not None:
            x_max = x_max.to(w.device)
            w_col = w.abs().amax(dim=0).clamp(min=1e-8)                      # [K] 每通道权重max
            s = (x_max.clamp(min=1e-8) ** 0.5) / (w_col ** 0.5)              # 平滑向量
            s = s / s.mean()                                                 # 归一化保持量级
            w = w * s[None, :]                                               # W' = W·s（激活侧除s）
            self.register_buffer("smooth", s.to(w.dtype).contiguous())
        N, K = w.shape
        group = 128
        assert K % group == 0, f"group size {group} must divide K={K}"
        w_g = w.view(N, K // group, group)
        w_scale = w_g.abs().amax(dim=2).clamp(min=1e-8) / 127.0             # [N, K//group]
        w_int8 = torch.clamp(torch.round(w / w_scale[:, torch.arange(K) // group]),
                             -127, 127).to(torch.int8)
        self.register_buffer("w_int8", w_int8.contiguous())
        self.register_buffer("w_scale", w_scale.contiguous())
        self.w8a8 = True
        del self.weight  # 释放fp16权重（显存减半；weight_loader只在加载期使用）

    def _w8a8_forward(self, x: torch.Tensor) -> torch.Tensor:
        # SmoothQuant平滑（若启用）：X' = X / s，权重侧已折叠 W' = W·s，恒等变换
        smooth = getattr(self, "smooth", None)
        if smooth is not None:
            x = x / smooth[None, :]
        # per-token激活量化（scale按K维amax）
        a_scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0  # [M, 1]
        a_int8 = torch.clamp(torch.round(x.float() / a_scale), -127, 127).to(torch.int8)
        y = w8a8_gemm(a_int8, self.w_int8, a_scale.squeeze(-1), self.w_scale.squeeze(-1))
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse24:
            return self._sparse24_forward(x)
        if self.int4:
            return self._int4_forward(x)
        if self.w8a8:
            return self._w8a8_forward(x)
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse24:
            return self._sparse24_forward(x)
        if self.int4:
            return self._int4_forward(x)
        if self.w8a8:
            return self._w8a8_forward(x)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse24:
            y = self._sparse24_forward(x)
        elif self.int4:
            y = self._int4_forward(x)
        elif self.w8a8:
            y = self._w8a8_forward(x)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

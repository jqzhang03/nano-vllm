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


class LinearBase(nn.Module):

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
        # ---- W8A8 量化状态（load_model之后由ModelRunner调用quantize_w8a8） ----
        self.w8a8 = False  # w_int8/w_scale/smooth缓冲由quantize_w8a8创建（不能先占普通属性名）

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
        if self.w8a8:
            y = self._w8a8_forward(x)
            if self.tp_size > 1:
                dist.all_reduce(y)
            return y
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

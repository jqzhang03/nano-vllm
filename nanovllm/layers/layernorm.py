import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        weight_offset: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        # weight_offset=True = **Gemma-2 的残差式缩放**（HF Gemma2RMSNorm 同款）：
        # 输出 = norm(x) × (1 + weight)，权重初始化为 0（checkpoint 存的是偏移量）。
        # 标准 llama/qwen 是 × weight。Gemma-2 用 (1+w) 而初始化为 0 → 权重只存"增量"，
        # 数值上 = 把缩放基线挪到 1（初始化即恒等）。用错（×w 而非 ×(1+w)）时输出
        # 差 (1+w)/w 倍——小权重列放大到 ~10×，parity top-1 直接 0%（见 note.md）。
        self.weight_offset = weight_offset
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        w = self.weight + (1.0 if self.weight_offset else 0.0)
        x = x.to(orig_dtype).mul_(w)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        w = self.weight + (1.0 if self.weight_offset else 0.0)
        x = x.to(orig_dtype).mul_(w)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

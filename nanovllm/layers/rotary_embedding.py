from functools import lru_cache
import math

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # "两半配对"式 RoPE（llama/qwen 约定）：x1 = 前一半、x2 = 后一半，
    # 配对 (i, i+d/2) 按 θ_i 旋转。cos/sin [.., d/2]（θ 序列 arange(0,d,2)/d）。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def apply_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """interleaved-pairs RoPE（DeepSeek MLA 的 q_pe/k_pe，与 HF 5.15 DeepseekV2
    的 view_as_complex 复数实现同构）：相邻维配对 (2k, 2k+1) 按 θ_k 旋转。

    与 apply_rotary_emb 的频率序列相同（cos/sin 同为 [.., d/2]），但**配对方式
    不同**（(2k,2k+1) vs (k, k+d/2)）——同一权重下两种约定输出不等价，必须按
    模型 checkpoint 的训练约定选。DeepSeek-V2 官方/HF 用 interleaved。
    """
    xf = x.float()
    xe, xo = xf[..., 0::2], xf[..., 1::2]
    ye = xe * cos - xo * sin
    yo = xe * sin + xo * cos
    return torch.stack((ye, yo), dim=-1).flatten(-2).to(x.dtype)


def _scaled_inv_freq_llama3(
    inv_freq: torch.Tensor,
    factor: float,
    high_freq_factor: float,
    low_freq_factor: float,
    original_max_position_embeddings: float,
) -> torch.Tensor:
    """Llama-3.2 的 "llama3" RoPE 缩放（transformers 同款，Llama-3.1 的 unsloth 转换
    checkpoint 也带此配置）。

    按波长分段：短波长（高频）不缩放；长波长（低频）除 factor；中间平滑插值——
    unsloth 用它把 8192 训练上下文外推到 131072。逐频率向量化：
    wavelen = 2π / freq；< high_freq_wavelen 不变；> low_freq_wavelen 除 factor；
    中间按 smooth 因子在"除 factor"与"不变"之间线性混合。
    """
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2.0 * math.pi / inv_freq
    is_low = wavelen > low_freq_wavelen
    is_high = wavelen < high_freq_wavelen
    is_mid = ~(is_low | is_high)
    smooth = ((original_max_position_embeddings / wavelen) - low_freq_factor) / (
        high_freq_factor - low_freq_factor)
    scaled = inv_freq / factor
    return torch.where(is_low, scaled,
                       torch.where(is_mid, (1 - smooth) * scaled + smooth * inv_freq,
                                   inv_freq))


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # True：相邻维配对 (2k, 2k+1)（DeepSeek MLA rope 头）；False：两半配对
        # (i, i+d/2)（llama/qwen）。两种约定的 cos/sin 表相同，仅 apply 不同。
        self.interleaved = interleaved
        # rope_scaling 只支持无操作（default/None）与 llama3 变体；yarn/linear/dynamic
        # 未实现 → 构造时报错（见 INTERVIEW.md §1.7 卡点清单）
        if rope_scaling:
            rtype = rope_scaling.get("rope_type")
            if rtype not in ("default", "llama3"):
                raise NotImplementedError(
                    f"rope_scaling type {rtype!r} unsupported: only 'default'/'llama3' handled")
        self.rope_scaling = rope_scaling
        self.build_cache()

    def build_cache(self) -> None:
        """（重新）计算 cos/sin 缓存。

        meta 设备构造时算出的值不落地（meta 张量无数据）；按层流式加载的
        to_empty 物化只给未初始化内存 → 必须在这里重算（见 INTERVIEW.md §1.7
        与 §6 故事 9 的坑：cos_sin_cache 全零 → q/k 被零旋转 → 逐层发散）。
        """
        inv_freq = 1.0 / (self.base**(torch.arange(0, self.head_size, 2, dtype=torch.float) / self.head_size))
        if self.rope_scaling and self.rope_scaling.get("rope_type") == "llama3":
            inv_freq = _scaled_inv_freq_llama3(
                inv_freq,
                self.rope_scaling["factor"],
                self.rope_scaling["high_freq_factor"],
                self.rope_scaling["low_freq_factor"],
                self.rope_scaling["original_max_position_embeddings"],
            )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]     # [T, 1, 2J]
        cos, sin = cos_sin.chunk(2, dim=-1)         # [T, 1, J]
        # MLA 共享 rope key 是 2D [T, J]（无 head 维）——去掉 cos/sin 的
        # 单例 head 维，否则 (T,J)×(T,1,J) 右对齐广播成 (T,T,J)
        if query.dim() == 2:
            cos, sin = cos[:, 0], sin[:, 0]
        elif key.dim() == 2 and query.dim() != 2:
            k_cos, k_sin = cos[:, 0], sin[:, 0]
            apply = apply_rotary_emb_interleaved if self.interleaved \
                else apply_rotary_emb
            query = apply(query, cos, sin)
            return query, apply(key, k_cos, k_sin)
        apply = apply_rotary_emb_interleaved if self.interleaved \
            else apply_rotary_emb
        query = apply(query, cos, sin)
        key = apply(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    scaling_key: tuple | None = None,
    interleaved: int = 0,
):
    """scaling_key = rope_scaling dict 的哈希化（tuple(sorted(items))）；
    lru_cache 要求参数可哈希，dict 不行。interleaved：DeepSeek MLA 相邻维配对。"""
    rope_scaling = dict(scaling_key) if scaling_key else None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                                 rope_scaling, interleaved=bool(interleaved))
    return rotary_emb

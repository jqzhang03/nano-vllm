from functools import lru_cache
import math

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


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
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # rope_scaling 只支持无操作（default/None）与 llama3 变体；yarn/linear/dynamic
        # 未实现 → 构造时报错（见 LEARNING.md 阶段7 卡点清单）
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
        to_empty 物化只给未初始化内存 → 必须在这里重算（见 LEARNING.md 阶段7
        的坑：cos_sin_cache 全零 → q/k 被零旋转 → 逐层发散）。
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
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    scaling_key: tuple | None = None,
):
    """scaling_key = rope_scaling dict 的哈希化（tuple(sorted(items))）；
    lru_cache 要求参数可哈希，dict 不行。"""
    rope_scaling = dict(scaling_key) if scaling_key else None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)
    return rotary_emb

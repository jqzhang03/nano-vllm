"""Host-staged KV cache transfer between disaggregated prefill/decode workers."""

from math import ceil

import torch


def export_kv_cache(kv_cache: torch.Tensor, block_ids: list[int],
                    num_tokens: int, block_size: int, *, mla: bool) -> torch.Tensor:
    """Gather the first ``num_tokens`` cached rows into a contiguous CPU tensor.

    MHA input/output layout is ``[2, layers, tokens, kv_heads, head_dim]``;
    MLA uses ``[layers, tokens, latent_dim]``. Block ids are in logical order.
    """
    if num_tokens <= 0:
        raise ValueError("KV transfer requires at least one cached token")
    n_blocks = ceil(num_tokens / block_size)
    if len(block_ids) < n_blocks:
        raise ValueError(f"KV transfer needs {n_blocks} blocks, got {len(block_ids)}")

    ids = torch.tensor(block_ids[:n_blocks], dtype=torch.long, device=kv_cache.device)
    if mla:
        # [layers, blocks, block_size, latent_dim]
        rows = kv_cache.index_select(1, ids).flatten(1, 2)[:, :num_tokens]
    else:
        # [K/V, layers, blocks, block_size, kv_heads, head_dim]
        rows = kv_cache.index_select(2, ids).flatten(2, 3)[:, :, :num_tokens]
    return rows.contiguous().to(device="cpu")


def import_kv_cache(kv_cache: torch.Tensor, block_ids: list[int],
                    num_tokens: int, block_size: int, rows: torch.Tensor,
                    *, mla: bool) -> None:
    """Scatter contiguous CPU KV rows into newly allocated private GPU blocks."""
    n_blocks = ceil(num_tokens / block_size)
    if len(block_ids) != n_blocks:
        raise ValueError(f"KV transfer needs {n_blocks} destination blocks, got {len(block_ids)}")

    ids = torch.tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    src = rows.to(device=kv_cache.device, dtype=kv_cache.dtype)
    if mla:
        expected = (kv_cache.shape[0], num_tokens, kv_cache.shape[-1])
        if tuple(src.shape) != expected:
            raise ValueError(f"MLA KV rows have shape {tuple(src.shape)}, expected {expected}")
        blocks = torch.zeros((kv_cache.shape[0], n_blocks, block_size,
                              kv_cache.shape[-1]), device=kv_cache.device,
                             dtype=kv_cache.dtype)
        blocks.flatten(1, 2)[:, :num_tokens].copy_(src)
        kv_cache.index_copy_(1, ids, blocks)
    else:
        expected = (kv_cache.shape[0], kv_cache.shape[1], num_tokens,
                    kv_cache.shape[-2], kv_cache.shape[-1])
        if tuple(src.shape) != expected:
            raise ValueError(f"MHA KV rows have shape {tuple(src.shape)}, expected {expected}")
        blocks = torch.zeros((kv_cache.shape[0], kv_cache.shape[1], n_blocks,
                              block_size, kv_cache.shape[-2], kv_cache.shape[-1]),
                             device=kv_cache.device, dtype=kv_cache.dtype)
        blocks.flatten(2, 3)[:, :, :num_tokens].copy_(src)
        kv_cache.index_copy_(2, ids, blocks)

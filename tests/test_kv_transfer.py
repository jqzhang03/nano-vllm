import pytest
import torch

from nanovllm.engine.kv_transfer import export_kv_cache, import_kv_cache


@pytest.mark.parametrize("mla", [False, True])
def test_kv_transfer_round_trip_partial_block(mla):
    block_size = 4
    block_ids = [1, 3]
    num_tokens = 6
    if mla:
        source = torch.arange(2 * 5 * block_size * 3, dtype=torch.float32)
        source = source.reshape(2, 5, block_size, 3)
    else:
        source = torch.arange(2 * 2 * 5 * block_size * 2 * 3,
                              dtype=torch.float32)
        source = source.reshape(2, 2, 5, block_size, 2, 3)

    rows = export_kv_cache(source, block_ids, num_tokens, block_size, mla=mla)
    destination = torch.zeros_like(source)
    import_kv_cache(destination, block_ids, num_tokens, block_size, rows,
                    mla=mla)

    indices = torch.tensor(block_ids)
    if mla:
        source_rows = source.index_select(1, indices).flatten(1, 2)[:, :num_tokens]
        destination_rows = destination.index_select(1, indices).flatten(1, 2)[:, :num_tokens]
    else:
        source_rows = source.index_select(2, indices).flatten(2, 3)[:, :, :num_tokens]
        destination_rows = destination.index_select(2, indices).flatten(2, 3)[:, :, :num_tokens]
    assert torch.equal(rows, source_rows)
    assert torch.equal(rows, destination_rows)


def test_kv_transfer_rejects_missing_blocks():
    cache = torch.zeros(2, 1, 4, 4, 2, 2)
    with pytest.raises(ValueError, match="needs 2 blocks"):
        export_kv_cache(cache, [0], 5, 4, mla=False)


def test_kv_transfer_rejects_wrong_destination_block_count():
    cache = torch.zeros(2, 1, 4, 4, 2, 2)
    rows = torch.zeros(2, 1, 5, 2, 2)
    with pytest.raises(ValueError, match="destination blocks"):
        import_kv_cache(cache, [0], 5, 4, rows, mla=False)

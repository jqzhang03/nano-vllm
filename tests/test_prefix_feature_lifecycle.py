from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from conftest import _simulate_prefill


def _assert_manager_invariants(manager: BlockManager) -> None:
    free_ids = list(manager.free_block_ids)
    assert len(free_ids) == len(set(free_ids))
    assert set(free_ids).isdisjoint(manager.used_block_ids)
    assert set(free_ids) == {
        block.block_id for block in manager.blocks
        if block.ref_count == 0 and not block.pending_free
    }
    assert manager.used_block_ids == {
        block.block_id for block in manager.blocks if block.ref_count > 0
    }
    for block_id in manager.deferred_free_block_ids:
        assert manager.blocks[block_id].pending_free
        assert manager.blocks[block_id].ref_count > 0
    for key, block_id in manager.hash_to_block_id.items():
        block = manager.blocks[block_id]
        assert block.hash == key
        if block.pending_free:
            assert block_id in manager.deferred_free_block_ids
    assert set(manager.hash_to_block_id.values()).issubset(manager.prefix_lru)


def test_prefix_feature_context_reuses_until_kv_generation_changes():
    manager = BlockManager(4, Sequence.block_size)
    published = Sequence([3] * Sequence.block_size)
    _simulate_prefill(manager, published, len(published))
    candidate = Sequence(list(published.token_ids))
    original_lru = list(manager.prefix_lru)

    features, parsed = manager.resolve_prefix_features(candidate)
    assert parsed
    assert features.cached_tokens == Sequence.block_size
    assert features.cached_blocks == 1
    assert features.first_miss_block == 1
    assert list(manager.prefix_lru) == original_lru  # probes do not alter recency

    reused, parsed = manager.resolve_prefix_features(candidate)
    assert not parsed
    assert reused is features

    generation = manager.kv_generation
    unrelated = Sequence([8] * Sequence.block_size)
    manager.allocate(unrelated, 0)
    assert manager.kv_generation > generation

    refreshed, parsed = manager.resolve_prefix_features(candidate)
    assert parsed
    assert refreshed.kv_generation == manager.kv_generation
    assert refreshed.cached_tokens == Sequence.block_size
    assert manager.prefix_feature_invalidations == 1
    assert manager.kv_generation_mutations > 0
    _assert_manager_invariants(manager)


def test_prefix_feature_cache_ablation_reparses_without_changing_features():
    manager = BlockManager(2, Sequence.block_size, prefix_feature_cache=False)
    published = Sequence([4] * Sequence.block_size)
    _simulate_prefill(manager, published, len(published))
    candidate = Sequence(list(published.token_ids))

    first, parsed_first = manager.resolve_prefix_features(candidate)
    second, parsed_second = manager.resolve_prefix_features(candidate)

    assert parsed_first and parsed_second
    assert first is not second
    assert first.cached_tokens == second.cached_tokens == Sequence.block_size
    assert first.cached_blocks == second.cached_blocks == 1
    assert candidate.prefix_feature_context is None
    assert manager.prefix_feature_invalidations == 0
    _assert_manager_invariants(manager)


def test_allocator_uses_uncached_free_blocks_before_evicting_prefixes():
    manager = BlockManager(3, Sequence.block_size)
    cached = Sequence([5] * Sequence.block_size)
    _simulate_prefill(manager, cached, len(cached))
    cached_id = cached.block_table[0]
    manager.deallocate(cached)

    new_id = manager._allocate_block()
    assert new_id != cached_id
    assert manager.blocks[cached_id].hash != -1
    assert manager.prefix_cache_evictions == 0
    manager.blocks[new_id].ref_count -= 1
    manager._deallocate_block(new_id)

    # When all free blocks are cached, allocation sacrifices the LRU entry.
    second = Sequence([6] * Sequence.block_size)
    _simulate_prefill(manager, second, len(second))
    second_id = second.block_table[0]
    manager.deallocate(second)
    third = Sequence([9] * Sequence.block_size)
    _simulate_prefill(manager, third, len(third))
    manager.deallocate(third)
    expected_victim = next(iter(manager.prefix_lru))
    pressure = Sequence([7] * Sequence.block_size)
    assert manager.can_allocate(pressure) == 0
    manager.allocate(pressure, 0)
    assert pressure.block_table == [expected_victim]
    assert manager.blocks[expected_victim].hash == -1
    assert manager.prefix_cache_evictions == 1
    assert manager.blocks[second_id].hash != -1
    _assert_manager_invariants(manager)


def test_free_prefix_limit_evicts_until_configured_bound():
    manager = BlockManager(4, Sequence.block_size, max_free_prefix_blocks=1)
    sequences = [Sequence([token] * Sequence.block_size) for token in (11, 12)]
    for seq in sequences:
        _simulate_prefill(manager, seq, len(seq))
        manager.deallocate(seq)

    reusable_free = [
        block_id for block_id in manager.free_block_ids
        if manager._is_reusable_prefix_block(manager.blocks[block_id])
    ]
    assert len(reusable_free) == 1
    assert manager.prefix_cache_evictions == 1
    _assert_manager_invariants(manager)


def test_deferred_free_hides_pending_shared_blocks_then_commits_once():
    manager = BlockManager(4, Sequence.block_size)
    owner = Sequence([13] * Sequence.block_size)
    _simulate_prefill(manager, owner, len(owner))
    shared = Sequence(list(owner.token_ids))
    _simulate_prefill(manager, shared, 0)
    block_id = owner.block_table[0]
    assert shared.block_table == [block_id]
    assert manager.blocks[block_id].ref_count == 2

    candidate = Sequence(list(owner.token_ids))
    initial, parsed = manager.resolve_prefix_features(candidate)
    assert parsed and initial.cached_tokens == Sequence.block_size

    manager.deallocate(owner, deferred=True)
    manager.deallocate(shared, deferred=True)
    block = manager.blocks[block_id]
    assert block.pending_free
    assert block.ref_count == 2
    assert block_id not in manager.free_block_ids
    assert manager.deferred_free_block_ids == [block_id, block_id]

    pending_features, parsed = manager.resolve_prefix_features(candidate)
    assert parsed
    assert pending_features.cached_tokens == 0
    assert manager.can_allocate(candidate) == 0
    _assert_manager_invariants(manager)

    manager.flush_deferred_free()
    assert block.ref_count == 0
    assert not block.pending_free
    assert block_id in manager.free_block_ids
    assert manager.deferred_free_block_ids == []
    assert manager.deferred_free_refs_queued == 2
    assert manager.deferred_free_refs_committed == 2
    assert manager.deferred_free_flushes == 1
    assert manager.deferred_free_peak_blocks == 1
    assert manager.deferred_free_peak_refs == 2
    assert manager.can_allocate(candidate) == 1

    manager.allocate(candidate, 1)
    assert candidate.block_table == [block_id]
    assert manager.blocks[block_id].ref_count == 1
    _assert_manager_invariants(manager)


def test_swap_release_invalidates_prefix_entry_before_reuse():
    manager = BlockManager(2, Sequence.block_size)
    seq = Sequence([17] * Sequence.block_size)
    _simulate_prefill(manager, seq, len(seq))
    block_id = seq.block_table[0]
    token_hash = manager.blocks[block_id].hash

    manager.release_blocks(list(seq.block_table))
    seq.block_table.clear()

    assert token_hash not in manager.hash_to_block_id
    assert block_id not in manager.prefix_lru
    assert manager.blocks[block_id].hash == -1
    assert manager.prefix_cache_evictions == 1
    candidate = Sequence([17] * Sequence.block_size)
    assert manager.can_allocate(candidate) == 0
    _assert_manager_invariants(manager)


def test_private_cache_modes_resolve_empty_features_without_sharing():
    for manager in (
        BlockManager(2, Sequence.block_size, no_share=True),
        BlockManager(2, Sequence.block_size, rolling_window=Sequence.block_size * 2),
    ):
        seq = Sequence([19] * Sequence.block_size)
        features, parsed = manager.resolve_prefix_features(seq)
        assert parsed
        assert (features.cached_tokens, features.cached_blocks,
                features.first_miss_block) == (0, 0, 0)
        _, parsed = manager.resolve_prefix_features(seq)
        assert not parsed


def test_resetting_lifecycle_metrics_keeps_generation_and_cache_state():
    manager = BlockManager(2, Sequence.block_size)
    seq = Sequence([23] * Sequence.block_size)
    _simulate_prefill(manager, seq, len(seq))
    manager.deallocate(seq, deferred=True)
    generation = manager.kv_generation
    indexed = dict(manager.hash_to_block_id)

    manager.reset_metrics()

    assert manager.kv_generation == generation
    assert manager.hash_to_block_id == indexed
    assert manager.deferred_free_block_ids
    assert manager.kv_generation_mutations == 0
    assert manager.prefix_feature_invalidations == 0
    assert manager.prefix_cache_evictions == 0
    assert manager.deferred_free_refs_queued == 0
    assert manager.deferred_free_refs_committed == 0
    assert manager.deferred_free_flushes == 0
    assert manager.deferred_free_peak_blocks == len(
        set(manager.deferred_free_block_ids))
    assert manager.deferred_free_peak_refs == len(manager.deferred_free_block_ids)
    _assert_manager_invariants(manager)

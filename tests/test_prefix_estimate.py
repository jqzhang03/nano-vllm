"""Cross-thread prefix-cache estimate used by the serving admission controller.

The admission estimate runs in the server's event loop while the engine mutates
the live `hash_to_block_id` table in its worker thread, so it can only read the
immutable snapshot from `BlockManager.prefix_snapshot()`. These tests pin that
the snapshot-based estimate agrees with the authoritative committed reuse
(`num_prefix_cached_tokens`), that the snapshot version moves whenever the table
changes, and that a stale snapshot never reports more than the live table holds.

Capacity note: `Sequence.num_tokens` is `len(prompt) + max_tokens`, so the tests
cap `max_tokens` at 1 and size pools from `Sequence.num_blocks`.
"""
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from conftest import _simulate_prefill

BLOCK = Sequence.block_size


def _seq(token_ids) -> Sequence:
    return Sequence(token_ids, SamplingParams(temperature=0.6, max_tokens=1))


def _estimate(manager: BlockManager, seq: Sequence, snapshot=None) -> int:
    if snapshot is None:
        snapshot = manager.prefix_snapshot()
    return manager.estimate_cached_tokens(seq, snapshot)


def _allocate_fresh(manager: BlockManager, seq: Sequence) -> int:
    """Allocate like the prefill path does, honouring the cache-aware block count."""
    num_cached = manager.can_allocate(seq)
    assert num_cached >= 0, "pool too small for this test sequence"
    manager.allocate(seq, num_cached)
    return num_cached


def test_estimate_matches_committed_reuse_for_exact_and_partial_prefixes():
    manager = BlockManager(8, BLOCK)
    published = _seq([7] * (2 * BLOCK + 10))
    _simulate_prefill(manager, published, len(published.token_ids))

    # Exact repeat: both full blocks plus the published partial block (10 tokens)
    # are reusable, and the estimate matches the authoritative probe.
    exact = _seq(list(published.token_ids))
    snapshot = manager.prefix_snapshot()
    assert _estimate(manager, exact, snapshot) == 2 * BLOCK + 10
    assert _estimate(manager, exact, snapshot) == manager.get_prefix_cached_tokens(exact)
    _allocate_fresh(manager, exact)
    assert exact.num_prefix_cached_tokens == 2 * BLOCK + 10

    # Longer prompt whose tail block differs: reuse stops at the last full block
    # because the published tail block only held 10 tokens.
    longer = _seq(list(published.token_ids) + [11] * 5)
    assert _estimate(manager, longer, manager.prefix_snapshot()) == 2 * BLOCK
    assert manager.get_prefix_cached_tokens(longer) == 2 * BLOCK

    # Exact repeat of the published tail length keeps the partial block reusable.
    same_tail = _seq(list(published.token_ids))
    assert _estimate(manager, same_tail, manager.prefix_snapshot()) == 2 * BLOCK + 10

    # Unrelated prompt: nothing is reusable.
    other = _seq([9] * BLOCK)
    assert _estimate(manager, other, manager.prefix_snapshot()) == 0


def test_estimate_never_exceeds_what_allocate_commits():
    manager = BlockManager(4, BLOCK)
    published = _seq([5] * (2 * BLOCK))
    _simulate_prefill(manager, published, len(published.token_ids))
    snapshot = manager.prefix_snapshot()

    candidate = _seq(list(published.token_ids))
    estimate = _estimate(manager, candidate, snapshot)
    _allocate_fresh(manager, candidate)
    assert estimate == candidate.num_prefix_cached_tokens == 2 * BLOCK


def test_pending_free_blocks_are_not_estimated_as_reusable():
    manager = BlockManager(6, BLOCK)
    published = _seq([6] * BLOCK)
    _simulate_prefill(manager, published, len(published.token_ids))
    version = manager.prefix_map_version()
    live = manager.prefix_snapshot()
    assert _estimate(manager, _seq([6] * BLOCK), live) == BLOCK

    # Deferred release marks the block pending_free: it must stop counting as
    # reusable, and the snapshot version must move so callers rebuild it.
    manager.deallocate(published, deferred=True)
    assert manager.prefix_map_version() > version
    assert all(block.pending_free for block in manager.blocks if block.hash != -1)
    rebuilt = manager.prefix_snapshot()
    assert _estimate(manager, _seq([6] * BLOCK), rebuilt) == 0  # rebuilt
    assert _estimate(manager, _seq([6] * BLOCK), live) == BLOCK  # stale snapshot is optimistic

    # Committing the release keeps the KV unreusable while the hash entry stays.
    manager.flush_deferred_free()
    assert manager.prefix_map_version() > version
    # The committed block keeps its valid KV and hash entry, so it is reusable
    # again (the pending_free guard was the only thing blocking it).
    assert _estimate(manager, _seq([6] * BLOCK), manager.prefix_snapshot()) == BLOCK
    fresh = _seq([6] * BLOCK)
    assert manager.can_allocate(fresh) == 1
    _allocate_fresh(manager, fresh)
    assert fresh.num_prefix_cached_tokens == BLOCK


def test_publishing_and_evicting_bumps_the_snapshot_version():
    # Pool of 3: the released published sequence leaves one reusable free entry,
    # and the next sequence needs all three blocks, so allocation has to
    # sacrifice that entry instead of skipping past it.
    manager = BlockManager(3, BLOCK)
    published = _seq([4] * BLOCK)
    _simulate_prefill(manager, published, len(published.token_ids))
    manager.deallocate(published)
    assert manager.prefix_snapshot() != {}
    version = manager.prefix_map_version()

    _allocate_fresh(manager, _seq([12] * (3 * BLOCK)))
    assert manager.prefix_map_version() > version
    assert manager.prefix_snapshot() == {}
    assert manager.prefix_cache_evictions >= 1


def test_stale_snapshot_lags_the_live_table_after_eviction():
    published = _seq([8] * BLOCK)
    small = BlockManager(3, BLOCK)
    _simulate_prefill(small, published, len(published.token_ids))
    small.deallocate(published)
    stale_snapshot = small.prefix_snapshot()
    assert stale_snapshot != {}

    # Same cache contents in another pool: the snapshot still describes them,
    # which shows the estimate reads the snapshot and not a live table.
    roomy = BlockManager(4, BLOCK)
    _simulate_prefill(roomy, _seq(list(published.token_ids)), len(published.token_ids))
    assert _estimate(roomy, _seq([8] * BLOCK), stale_snapshot) == BLOCK

    # Evicting from the live pool makes the stale snapshot optimistic: callers
    # must compare prefix_map_version() before trusting it.
    _allocate_fresh(small, _seq([30] * (3 * BLOCK)))
    assert small.prefix_snapshot() == {}
    assert _estimate(small, _seq([8] * BLOCK), stale_snapshot) == BLOCK
    assert _estimate(small, _seq([8] * BLOCK), small.prefix_snapshot()) == 0

def test_snapshot_is_an_immutable_copy():
    manager = BlockManager(6, BLOCK)
    published = _seq([5] * BLOCK)
    _simulate_prefill(manager, published, len(published.token_ids))
    snapshot = manager.prefix_snapshot()
    assert len(snapshot) == 1
    entry = next(iter(snapshot.values()))
    assert isinstance(entry[2], tuple)
    assert entry[1] is False

    # Mutating the live pool (allocate + deferred free) must leave the snapshot
    # object itself untouched and still self-consistent.
    _allocate_fresh(manager, _seq([17] * BLOCK))
    manager.deallocate(_seq([19] * BLOCK), deferred=True)
    assert len(snapshot) == 1
    assert isinstance(next(iter(snapshot.values()))[2], tuple)
    assert next(iter(snapshot.values()))[1] is False

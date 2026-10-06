"""Regressions for two measurement/correctness bugs found on 2026-10-06.

1. Per-request metrics were recorded twice for sequences that finished inside a
   multi-step decode burst: `_run_decode_burst` returns those sequences *and* they
   are still in the outer batch, so ``_req_metrics`` grew duplicate ``seq_id``
   entries. Benchmarks count completions from that list, so a 128-request run
   recorded 133 rows over only 95 distinct sequences and the driver stopped early
   (it had counted the duplicates) - the reported throughput was computed on a
   subset of the workload.
2. The TPOT-driven prefill throttle had no deadband and no queue-depth gate, so
   any overshoot pinned the per-step prefill budget at `prefill_reserve_tokens`,
   which starved prefill and kept measured TPOT high (self-locking).
"""
import os

import pytest

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def _config(**overrides) -> Config:
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.isdir(path):
        pytest.skip("Qwen3-0.6B checkpoint not available")
    kwargs = dict(model=path, num_kvcache_blocks=64, kvcache_block_size=256,
                  max_num_batched_tokens=2048)
    kwargs.update(overrides)
    return Config(**kwargs)


def _bare_engine(**overrides) -> LLMEngine:
    """Engine shell with just the bookkeeping the metrics path touches."""
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = _config(**overrides)
    engine._req_metrics = []
    engine._recorded_seq_ids = set()
    return engine


def test_request_metrics_are_recorded_once_per_sequence():
    engine = _bare_engine()
    seq = Sequence([1] * 300)
    seq.status = SequenceStatus.FINISHED
    seq.t_submitted = 1.0
    seq.t_first_token = 1.1
    seq.t_completed = 1.2
    # First sighting (e.g. the outer batch) records it; the burst reporting the
    # same sequence afterwards must be ignored.
    engine._record_once(seq)
    engine._record_once(seq)
    engine._record_once(seq)
    assert len(engine._req_metrics) == 1
    assert engine._req_metrics[0]["seq_id"] == seq.seq_id


def test_reset_benchmark_metrics_clears_the_dedup_set():
    engine = _bare_engine()
    seq = Sequence([1] * 300)
    seq.status = SequenceStatus.FINISHED
    engine._record_once(seq)
    assert engine._req_metrics
    engine.scheduler = Scheduler(engine.config)
    engine._pd = False
    engine._step_stats = LLMEngine._empty_step_stats()
    engine.reset_benchmark_metrics()
    assert engine._req_metrics == []
    assert engine._recorded_seq_ids == set()
    # After a reset the same sequence may be recorded again (a fresh run).
    engine._record_once(seq)
    assert len(engine._req_metrics) == 1


def _stub_scheduler_with_waiting(depth: int):
    config = _config()
    scheduler = Scheduler(config)
    scheduler.waiting.extend(Sequence([2] * 64) for _ in range(depth))
    running = Sequence([3] * 64)
    # target 20 ms, observed 25 ms: inside the 0.5 deadband (target + 10 ms)
    running.tpot_ewma_ms = 25.0
    running.tpot_slo_ms = 20.0
    scheduler.running.append(running)
    scheduler.external_tpot_pressure = 0.0
    return scheduler


def test_tpot_throttle_is_gated_by_waiting_depth():
    scheduler = _stub_scheduler_with_waiting(depth=1)
    assert scheduler._tpot_decode_pressure() == 0.0, \
        "25 ms against a 20 ms target with a 0.5 deadband is within tolerance"
    scheduler.running[0].tpot_ewma_ms = 1000.0
    assert scheduler._tpot_decode_pressure() > 0.0, \
        "a massive overshoot must throttle prefill while the queue is shallow"

    deep = _stub_scheduler_with_waiting(depth=scheduler.tpot_throttle_max_waiting)
    deep.running[0].tpot_ewma_ms = 1000.0
    assert deep._tpot_decode_pressure() == 0.0, \
        "at depth the bottleneck is prefill throughput: throttling it self-locks"


def test_deadband_is_reachable_through_config():
    config = _config(tpot_prefill_throttle_margin=0.0, tpot_throttle_max_waiting=0)
    scheduler = Scheduler(config)
    assert scheduler.tpot_prefill_throttle_margin == 0.0
    assert scheduler.tpot_throttle_max_waiting == 0
    # Legacy behaviour: the same 1.5x overshoot now saturates the throttle.
    s = _stub_scheduler_with_waiting(depth=1)
    s.tpot_prefill_throttle_margin = 0.0
    s.tpot_throttle_max_waiting = 0
    assert s._tpot_decode_pressure() == 1.0

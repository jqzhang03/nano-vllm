"""Decode-burst arrival pressure and burst return contract.

Two regressions motivated this file:

1. ``_run_decode_burst`` started returning ``(tokens, finished_sequences)`` while
   the function still returned a bare int, so every engine step that entered a
   pure-decode burst raised ``TypeError: cannot unpack non-sequence int``.
2. The arrival-pressure hook (``decode_burst_yield_on_arrival``) was referenced by
   the engine and the server but the config field, the server-side
   ``_sync_decode_burst_pressure`` and the step-stats key were missing.

Both are checked here without a GPU by exercising the engine object's own
predicate and by calling the burst helper with stub scheduler/runner objects.
"""
import threading

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine


class _StubScheduler:
    """Minimal scheduler surface used by ``_run_decode_burst``'s early exits."""

    def __init__(self, waiting=(), running=(), swapped=()):
        self.waiting = list(waiting)
        self.running = list(running)
        self.swapped = list(swapped)
        self.calls = []

    def _order_running_by_tpot(self):
        self.calls.append("order")

    def _schedule_decode(self):
        self.calls.append("schedule")
        return [], "decode"


class _StubRunner:
    def call(self, *args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("runner must not be called for a yielding burst")


def _engine_with_config(**overrides) -> LLMEngine:
    engine = LLMEngine.__new__(LLMEngine)
    defaults = dict(
        model=".",                      # Config requires an existing dir; see below
        multi_step_decode=True,
        max_decode_steps=4,
        decode_burst_yield=True,
        decode_burst_yield_on_arrival=True,
        speculative="none",
    )
    defaults.update(overrides)
    engine.config = _FakeConfig(**defaults)
    engine._step_stats = LLMEngine._empty_step_stats()
    engine._last_step_multistep_tokens = 0
    engine._last_step_decode_iterations = 0
    engine._last_step_decode_burst_yielded = False
    engine._decode_burst_yield_signal = threading.Event()
    engine._decode_burst_yield_callback = None
    return engine


class _FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_arrival_pressure_predicate_covers_signal_and_callback():
    engine = _engine_with_config()
    assert not engine._decode_burst_pressure_active()

    engine.request_decode_burst_yield()
    assert engine._decode_burst_pressure_active()
    engine.clear_decode_burst_yield()
    assert not engine._decode_burst_pressure_active()

    state = {"due": False}
    engine.set_decode_burst_yield_callback(lambda: state["due"])
    assert not engine._decode_burst_pressure_active()
    state["due"] = True
    assert engine._decode_burst_pressure_active()
    engine.set_decode_burst_yield_callback(None)
    assert not engine._decode_burst_pressure_active()


def test_burst_returns_tokens_and_finished_sequences():
    engine = _engine_with_config()
    scheduler = _StubScheduler(waiting=[object()])  # waiting => burst is skipped
    result = engine._run_decode_burst(
        scheduler, _StubRunner(), 3, collect_logits=False)
    assert isinstance(result, tuple) and len(result) == 2, \
        "callers unpack (tokens, finished_sequences) from every burst path"
    tokens, finished = result
    assert tokens == 3
    assert finished == []


def test_arrival_pressure_ends_burst_before_next_round():
    engine = _engine_with_config()
    engine.request_decode_burst_yield()
    scheduler = _StubScheduler(running=[object()])
    tokens, finished = engine._run_decode_burst(
        scheduler, _StubRunner(), 7, collect_logits=False)
    assert (tokens, finished) == (7, [])
    assert engine._last_step_decode_burst_yielded
    assert scheduler.calls == []          # yielding happens before scheduling
    assert engine._step_stats["decode_burst_rounds"] == 1


def test_arrival_yield_ablation_runs_the_full_budget():
    engine = _engine_with_config(decode_burst_yield_on_arrival=False)
    engine.request_decode_burst_yield()
    scheduler = _StubScheduler(running=[object()])
    tokens, finished = engine._run_decode_burst(
        scheduler, _StubRunner(), 7, collect_logits=False)
    # Ablation: the pressure signal is ignored, so the burst proceeds to schedule
    # its follow-up round (the stub returns no rows, ending the loop).
    assert (tokens, finished) == (7, [])
    assert not engine._last_step_decode_burst_yielded
    assert "schedule" in scheduler.calls


def test_config_exposes_the_arrival_switch():
    fields = {field.name for field in __import__("dataclasses").fields(Config)}
    assert "decode_burst_yield_on_arrival" in fields
    assert "decode_burst_yield" in fields

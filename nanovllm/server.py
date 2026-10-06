"""OpenAI-compatible HTTP serving with incremental token streaming."""

import argparse
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import logging
from math import isfinite
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=1e-10)
    max_tokens: int = Field(default=256, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_token_budget: int | None = Field(
        default=None, ge=1,
        description="Maximum prompt-context tokens retained for this managed conversation")
    ttft_slo_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False,
                                      description="Request TTFT target in milliseconds")
    tpot_slo_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False,
                                      description="Request average TPOT target in milliseconds")


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    temperature: float = Field(default=0.7, ge=1e-10)
    max_tokens: int = Field(default=256, ge=1)
    stream: bool = False
    ttft_slo_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False,
                                      description="Request TTFT target in milliseconds")
    tpot_slo_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False,
                                      description="Request average TPOT target in milliseconds")


@dataclass(slots=True)
class GenerationHandle:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    submitted_at: float
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    stream: bool = False
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    result: asyncio.Future | None = None
    seq_id: int | None = None
    cancelled: bool = False
    generated_tokens: int = 0
    prefill_tokens_done: int = 0
    first_token_at: float | None = None
    completed_at: float | None = None
    admission_decision: str = "accepted"
    admission_wait_ms: float = 0.0
    predicted_ttft_ms: float = 0.0
    queue_pressure: float = 0.0
    estimated_output_tokens: float = 0.0
    estimated_cache_hit_tokens: float = 0.0
    hit_estimate_counted: bool = False
    prefix_cached_tokens: int = 0


class GenerationManager:
    """Serializes engine access while continuously batching submitted requests."""

    def __init__(self, engine: LLM, executor: ThreadPoolExecutor,
                 max_queued_requests: int = 256, *,
                 dynamic_admission: bool = True,
                 prefix_cache_aware_admission: bool = True,
                 max_deferred_requests: int = 64,
                 max_admission_wait_ms: float = 2000.0,
                 admission_work_budget_ms: float = 10000.0,
                 admission_target_ttft_ms: float = 2000.0,
                 admission_soft_pressure: float = 0.85,
                 prefill_tps_fallback: float = 10000.0,
                 decode_tps_fallback: float = 1000.0,
                 cache_affinity_admission: bool = True):
        if max_deferred_requests < 0:
            raise ValueError("max_deferred_requests must be non-negative")
        if max_queued_requests < 1:
            raise ValueError("max_queued_requests must be at least 1")
        if not isfinite(max_admission_wait_ms) or max_admission_wait_ms < 0:
            raise ValueError("max_admission_wait_ms must be non-negative")
        if (not isfinite(admission_work_budget_ms) or admission_work_budget_ms <= 0
                or not isfinite(admission_target_ttft_ms)
                or admission_target_ttft_ms <= 0):
            raise ValueError("admission time budgets must be positive")
        if not isfinite(admission_soft_pressure) or not 0 < admission_soft_pressure <= 1:
            raise ValueError("admission_soft_pressure must be in (0, 1]")
        if (not isfinite(prefill_tps_fallback) or prefill_tps_fallback <= 0
                or not isfinite(decode_tps_fallback) or decode_tps_fallback <= 0):
            raise ValueError("admission throughput fallbacks must be positive")
        self.engine = engine
        self.executor = executor
        self.max_queued_requests = max_queued_requests
        self.dynamic_admission = dynamic_admission
        self.prefix_cache_aware_admission = prefix_cache_aware_admission
        self.max_deferred_requests = max_deferred_requests
        self.max_admission_wait_ms = max_admission_wait_ms
        self.admission_work_budget_ms = admission_work_budget_ms
        self.admission_target_ttft_ms = admission_target_ttft_ms
        self.admission_soft_pressure = admission_soft_pressure
        self.prefill_tps = prefill_tps_fallback
        self.decode_tps = decode_tps_fallback
        # 准入是否按前缀缓存命中折算 prompt 工作量（消融开关）。
        self.cache_affinity_admission = cache_affinity_admission
        # 跨线程只读的前缀缓存快照（hash -> (block_id, pending_free, token_ids)）与
        # 其版本号：只在 engine 空闲时重建（`_refresh_prefix_snapshot`），准入估算
        # 只读这份不可变副本，绝不读 worker 线程正在改的活表。
        self._prefix_snapshot: dict[int, tuple[int, bool, tuple]] = {}
        self._prefix_snapshot_version = -1
        self._prefix_snapshot_age_steps = 0
        self._admission_hit_estimate_tokens = 0
        self._admission_hit_actual_tokens = 0
        # Begin with the requested output cap as a conservative prior, then
        # learn the generated/capped ratio from completed requests.
        self.output_length_ratio = 1.0
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.active: dict[int, GenerationHandle] = {}
        self.pending: OrderedDict[str, GenerationHandle] = OrderedDict()
        self.deferred: OrderedDict[str, GenerationHandle] = OrderedDict()
        self._admission_condition = asyncio.Condition()
        self._admission_counts = {"accepted": 0, "deferred_total": 0, "rejected": 0,
                                  "deferred_timeouts": 0,
                                  "prefix_cache_estimated_requests": 0,
                                  "prefix_cache_estimated_tokens": 0,
                                  "prefix_cache_actual_requests": 0,
                                  "prefix_cache_actual_tokens": 0}
        self._last_admission_estimate: dict[str, float] = {}
        self._kv_pool_stats: list[tuple[int, int]] = []
        self._kv_pool_peak_used: list[int] = []
        self._prefix_cache_snapshots_by_manager: dict[
            BlockManager,
            tuple[int, dict[int, tuple[tuple[int, ...], int, bool]]],
        ] = {}
        self._prefix_cache_snapshots: tuple[
            tuple[BlockManager, dict[int, tuple[tuple[int, ...], int, bool]]], ...
        ] = ()
        self._engine_busy = False
        self.step_stats = {"steps": 0, "prefill_tokens": 0, "decode_tokens": 0,
                           "prefill_seconds": 0.0, "decode_seconds": 0.0,
                           "decode_iterations": 0, "multi_step_decode_steps": 0,
                           "multi_step_decode_tokens": 0,
                           # 引擎侧权威前缀复用计数（scheduler 累加），用于和准入估算对账：
                           # 每完成一次 prefill 分配，按 seq.num_prefix_cached_tokens 累加。
                           "prefix_cache_hit_tokens": 0,
                           "prefix_cache_hit_requests": 0,
                           "prefix_cache_evictions": 0,
                           "engine_seconds": 0.0}
        self._refresh_kv_pool_stats()
        self._refresh_prefix_snapshot()
        self.task: asyncio.Task | None = None
        self.stopping = False
        self.accepting = True

    def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="nanovllm-generation-loop")

    async def submit(self, prompt_token_ids: list[int], sampling_params: SamplingParams,
                     *, stream: bool = False, submitted_at: float | None = None,
                     ttft_slo_ms: float | None = None,
                     tpot_slo_ms: float | None = None
                     ) -> GenerationHandle:
        loop = asyncio.get_running_loop()
        handle = GenerationHandle(
            request_id=f"chatcmpl-{uuid.uuid4().hex}",
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            submitted_at=(time.perf_counter() if submitted_at is None else submitted_at),
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
            stream=stream,
            result=loop.create_future(),
        )
        self._validate_prompt_capacity(handle)
        deferred_at: float | None = None
        try:
            async with self._admission_condition:
                while True:
                    if not self.accepting:
                        raise HTTPException(status_code=503,
                                            detail="generation worker is stopping")
                    is_deferred = handle.request_id in self.deferred
                    outstanding = len(self.active) + len(self.pending) + len(self.deferred)
                    estimate = self._estimate_admission(handle)
                    self._last_admission_estimate = estimate
                    is_head = (not self.deferred
                               or next(iter(self.deferred)) == handle.request_id)
                    capacity_available = (outstanding < self.max_queued_requests
                                          or is_deferred)
                    engine_outstanding = len(self.active) + len(self.pending)
                    can_admit = (is_head and capacity_available
                                 and self._should_admit(estimate, engine_outstanding))
                    if can_admit:
                        if handle.request_id in self.deferred:
                            self.deferred.pop(handle.request_id)
                            handle.admission_decision = "accepted_after_defer"
                            handle.admission_wait_ms = (
                                time.perf_counter() - deferred_at) * 1000.0
                        handle.predicted_ttft_ms = estimate["predicted_ttft_ms"]
                        handle.queue_pressure = estimate["queue_pressure"]
                        handle.estimated_output_tokens = estimate["candidate_output_tokens"]
                        handle.estimated_cache_hit_tokens = estimate[
                            "estimated_cache_hit_tokens"]
                        if not handle.hit_estimate_counted:
                            handle.hit_estimate_counted = True
                            self._admission_hit_estimate_tokens += int(
                                estimate["estimated_cache_hit_tokens"])
                        self.pending[handle.request_id] = handle
                        self.incoming.put_nowait(handle)
                        self._sync_decode_burst_pressure()
                        self._admission_counts["accepted"] += 1
                        self._admission_condition.notify_all()
                        return handle

                    if outstanding >= self.max_queued_requests and not is_deferred:
                        self._admission_counts["rejected"] += 1
                        raise self._overloaded(estimate, "hard request limit reached")

                    if deferred_at is None:
                        if len(self.deferred) >= self.max_deferred_requests:
                            self._admission_counts["rejected"] += 1
                            raise self._overloaded(estimate, "deferred request limit reached")
                        deferred_at = time.perf_counter()
                        handle.admission_decision = "deferred"
                        self.deferred[handle.request_id] = handle
                        self._sync_decode_burst_pressure()
                        self._admission_counts["deferred_total"] += 1

                    elapsed_ms = (time.perf_counter() - deferred_at) * 1000.0
                    remaining_ms = self.max_admission_wait_ms - elapsed_ms
                    if remaining_ms <= 0:
                        self.deferred.pop(handle.request_id, None)
                        self._admission_counts["rejected"] += 1
                        self._admission_counts["deferred_timeouts"] += 1
                        self._admission_condition.notify_all()
                        raise self._overloaded(estimate, "admission wait budget expired")
                    try:
                        await asyncio.wait_for(
                            self._admission_condition.wait(),
                            timeout=min(remaining_ms / 1000.0, 0.1),
                        )
                    except asyncio.TimeoutError:
                        pass
        finally:
            # A disconnected/cancelled HTTP task must release its deferred slot.
            async with self._admission_condition:
                if self.deferred.pop(handle.request_id, None) is not None:
                    self._admission_condition.notify_all()
                self._sync_decode_burst_pressure()

    def _validate_prompt_capacity(self, handle: GenerationHandle) -> None:
        block_size = self.engine.config.kvcache_block_size
        required_blocks = (len(handle.prompt_token_ids) + block_size - 1) // block_size
        schedulers = ((self.engine.prefill_scheduler, self.engine.decode_scheduler)
                      if self.engine._pd else (self.engine.scheduler,))
        capacities = []
        for scheduler in schedulers:
            capacities.append(len(scheduler.block_manager.blocks))
            if scheduler.full_block_manager is not None:
                capacities.append(len(scheduler.full_block_manager.blocks))
        if capacities and required_blocks > min(capacities):
            raise HTTPException(
                status_code=400,
                detail=(f"prompt needs {required_blocks} KV blocks, but the active pool "
                        f"can hold at most {min(capacities)}"),
            )

    def _block_managers(self):
        schedulers = ((self.engine.prefill_scheduler, self.engine.decode_scheduler)
                      if self.engine._pd else (self.engine.scheduler,))
        managers = []
        for scheduler in schedulers:
            managers.append(scheduler.block_manager)
            if scheduler.full_block_manager is not None:
                managers.append(scheduler.full_block_manager)
        return managers

    def _prefix_cache_managers(self) -> list[BlockManager]:
        if not self.prefix_cache_aware_admission:
            return []
        # PD currently constructs no_share managers for both pools, so it has no
        # reusable prefixes to estimate. If PD-side sharing is added later, only
        # the prefill pool can predict prompt work; decode imports private KV.
        schedulers = ((self.engine.prefill_scheduler,)
                      if self.engine._pd else (self.engine.scheduler,))
        managers = []
        for scheduler in schedulers:
            for manager in (scheduler.block_manager, scheduler.full_block_manager):
                if (manager is not None and not manager.rolling and not manager.no_share):
                    managers.append(manager)
        return managers

    def _refresh_kv_pool_stats(self) -> None:
        # Only refresh while the inference worker is idle, avoiding concurrent reads
        # of BlockManager state while engine.step() mutates it in the worker thread.
        managers = self._block_managers()
        self._kv_pool_stats = [
            (len(manager.blocks), len(manager.free_block_ids))
            for manager in managers
        ]
        snapshots_by_manager = {}
        snapshots = []
        for manager in self._prefix_cache_managers():
            cached = self._prefix_cache_snapshots_by_manager.get(manager)
            if cached is not None and cached[0] == manager.kv_generation:
                snapshot = cached[1]
            else:
                snapshot: dict[int, tuple[tuple[int, ...], int, bool]] = {}
                free_ids = set(manager.free_block_ids)
                for prefix_hash, block_id in manager.hash_to_block_id.items():
                    block = manager.blocks[block_id]
                    if block.hash == prefix_hash and not block.pending_free:
                        snapshot[prefix_hash] = (
                            tuple(block.token_ids), block_id, block_id in free_ids)
                cached = (manager.kv_generation, snapshot)
            snapshots_by_manager[manager] = cached
            snapshots.append((manager, cached[1]))
        self._prefix_cache_snapshots_by_manager = snapshots_by_manager
        self._prefix_cache_snapshots = tuple(snapshots)
        if len(self._kv_pool_peak_used) != len(self._kv_pool_stats):
            self._kv_pool_peak_used = [0] * len(self._kv_pool_stats)
        for index, (total, free) in enumerate(self._kv_pool_stats):
            self._kv_pool_peak_used[index] = max(
                self._kv_pool_peak_used[index], total - free)

    def _refresh_prefix_snapshot(self) -> None:
        """Rebuild the cross-thread prefix-cache snapshot (engine idle only).

        Same idle-only rule as ``_refresh_kv_pool_stats``: the version check keeps
        the cost near zero on decode-only steps, and a snapshot that is one engine
        step stale only shifts the admission estimate by the blocks published in
        that step.
        """
        managers = self._block_managers()
        if not managers:
            return
        primary = managers[0]
        version = primary.prefix_map_version()
        if version == self._prefix_snapshot_version:
            return
        self._prefix_snapshot = primary.prefix_snapshot()
        self._prefix_snapshot_version = version

    def _estimate_candidate_hit_tokens(self, candidate: GenerationHandle) -> int:
        """Estimate how many prompt tokens the prefix cache can serve for a request.

        Uses the same chained block hashes as ``BlockManager.can_allocate`` against
        the idle-time snapshot, so the number is the block-aligned reuse the engine
        would commit if the cache does not change before this request is scheduled.
        """
        if not self.cache_affinity_admission or not self._prefix_snapshot:
            return 0
        manager = self._block_managers()[0]
        probe = Sequence(candidate.prompt_token_ids, candidate.sampling_params)
        return manager.estimate_cached_tokens(probe, self._prefix_snapshot)

    def _estimate_admission(self, candidate: GenerationHandle) -> dict[str, float]:
        prefill_tokens = 0
        decode_tokens = 0
        block_size = self.engine.config.kvcache_block_size
        pending_blocks = 0
        prefix_matches: dict[
            tuple[int, ...], tuple[int, tuple[tuple[BlockManager, int, bool], ...]]
        ] = {}
        reserved_free_prefix_blocks: set[tuple[BlockManager, int]] = set()

        def prefix_match(
            handle: GenerationHandle,
        ) -> tuple[int, tuple[tuple[BlockManager, int, bool], ...]]:
            key = tuple(handle.prompt_token_ids)
            match = prefix_matches.get(key)
            if match is None:
                match = self._estimate_prefix_cache_match(handle.prompt_token_ids)
                prefix_matches[key] = match
            return match

        projected_prefix_cached_tokens = 0
        for handle in self.active.values():
            remaining_prefill = max(
                0, len(handle.prompt_token_ids) - handle.prefill_tokens_done)
            if handle.prefill_tokens_done == 0:
                match = prefix_match(handle)
                cached_tokens = min(remaining_prefill, match[0])
                projected_prefix_cached_tokens += cached_tokens
                remaining_prefill -= cached_tokens
            prefill_tokens += remaining_prefill
            decode_tokens += max(
                0, handle.sampling_params.max_tokens - handle.generated_tokens)
            if handle.prefill_tokens_done == 0 and handle.generated_tokens == 0:
                # The engine accepted the sequence, but no prefill progress has
                # been reported yet; its prompt blocks are not in current KV use.
                pending_blocks += self._projected_kv_blocks(
                    handle, prefix_match(handle), reserved_free_prefix_blocks)
            else:
                pending_blocks += (max(0, handle.sampling_params.max_tokens
                                       - handle.generated_tokens) + block_size - 1) // block_size
        for handle in self.pending.values():
            match = prefix_match(handle)
            projected_prefix_cached_tokens += match[0]
            prefill_tokens += max(0, len(handle.prompt_token_ids) - match[0])
            decode_tokens += self._expected_output_tokens(handle.sampling_params.max_tokens)
            pending_blocks += self._projected_kv_blocks(
                handle, match, reserved_free_prefix_blocks)

        candidate_output_tokens = self._expected_output_tokens(
            candidate.sampling_params.max_tokens)
        # 前缀缓存感知：估算这个候选能直接复用多少 prompt token（块对齐）。
        # cache_affinity_admission=False 时恒为 0，即"准入不感知缓存"的消融口径。
        candidate_hit_tokens = self._estimate_candidate_hit_tokens(candidate)
        candidate_prefill_tokens = len(candidate.prompt_token_ids) - candidate_hit_tokens
        prefill_tokens += candidate_prefill_tokens
        decode_tokens += candidate_output_tokens
        pending_blocks += (candidate_prefill_tokens
                           + candidate.sampling_params.max_tokens + block_size - 1) // block_size
        self._admission_hit_estimate_tokens += candidate_hit_tokens

        prefill_rate = max(1.0, self.prefill_tps)
        decode_rate = max(1.0, self.decode_tps)
        prefill_seconds = prefill_tokens / prefill_rate
        decode_seconds = decode_tokens / decode_rate
        prior_decode_tokens = max(0, decode_tokens - candidate_output_tokens)
        prior_decode_seconds = prior_decode_tokens / decode_rate
        target_ms = (candidate.ttft_slo_ms if candidate.ttft_slo_ms is not None
                     else self.admission_target_ttft_ms)
        request_age_ms = max(0.0, time.perf_counter() - candidate.submitted_at) * 1000.0
        first_decode_seconds = 1.0 / decode_rate if candidate_output_tokens > 0 else 0.0
        predicted_ttft_ms = 1000.0 * (
            prefill_seconds
            + first_decode_seconds
            + min(prior_decode_seconds * 0.25, target_ms / 2000.0)
        ) + request_age_ms
        projected_work_ms = 1000.0 * (prefill_seconds + decode_seconds)

        kv_pressure = 0.0
        for total, free in self._kv_pool_stats:
            if total:
                used = total - free
                kv_pressure = max(kv_pressure, min(1.0, (used + pending_blocks) / total))
        projected_count = len(self.active) + len(self.pending) + 1
        count_pressure = projected_count / self.max_queued_requests
        work_pressure = projected_work_ms / self.admission_work_budget_ms
        return {
            "predicted_ttft_ms": predicted_ttft_ms,
            "target_ttft_ms": target_ms,
            "projected_work_ms": projected_work_ms,
            "projected_prefill_tokens": float(prefill_tokens),
            "candidate_prefill_tokens": float(candidate_prefill_tokens),
            "candidate_prefix_cached_tokens": float(candidate_prefix_cached_tokens),
            "projected_prefix_cached_tokens": float(projected_prefix_cached_tokens),
            "projected_output_tokens": float(decode_tokens),
            "candidate_output_tokens": candidate_output_tokens,
            "candidate_prompt_tokens": float(len(candidate.prompt_token_ids)),
            "estimated_cache_hit_tokens": float(candidate_hit_tokens),
            "candidate_prefill_tokens": float(candidate_prefill_tokens),
            "output_length_ratio": self.output_length_ratio,
            "queue_pressure": max(count_pressure, work_pressure, kv_pressure),
            "kv_pressure": kv_pressure,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "projected_requests": float(projected_count),
        }

    def _expected_output_tokens(self, max_tokens: int) -> float:
        return min(float(max_tokens), max(1.0, max_tokens * self.output_length_ratio))

    def _should_admit(self, estimate: dict[str, float], outstanding: int) -> bool:
        if not self.dynamic_admission or outstanding == 0:
            return True
        return (estimate["predicted_ttft_ms"] <= estimate["target_ttft_ms"]
                and estimate["queue_pressure"] < self.admission_soft_pressure)

    def _overloaded(self, estimate: dict[str, float], reason: str) -> HTTPException:
        retry_after = max(1, int((estimate.get("projected_work_ms", 1000.0) + 999) // 1000))
        return HTTPException(
            status_code=429,
            detail={"message": "server admission limit reached",
                    "admission": "rejected", "reason": reason,
                    "predicted_ttft_ms": round(estimate.get("predicted_ttft_ms", 0.0), 1),
                    "estimated_output_tokens": round(
                        estimate.get("candidate_output_tokens", 0.0), 1),
                    "estimated_prefill_tokens": int(round(
                        estimate.get("candidate_prefill_tokens", 0.0))),
                    "estimated_prefix_cached_tokens": int(round(
                        estimate.get("candidate_prefix_cached_tokens", 0.0))),
                    "queue_pressure": round(estimate.get("queue_pressure", 1.0), 3)},
            headers={"Retry-After": str(retry_after)},
        )

    def admission_snapshot(self) -> dict[str, Any]:
        return {
            **self._admission_counts,
            "dynamic_enabled": self.dynamic_admission,
            "prefix_cache_aware": self.prefix_cache_aware_admission,
            "active": len(self.active),
            "queued": len(self.pending),
            "deferred": len(self.deferred),
            "max_queued_requests": self.max_queued_requests,
            "max_deferred_requests": self.max_deferred_requests,
            "prefill_tps_estimate": round(self.prefill_tps, 1),
            "decode_tps_estimate": round(self.decode_tps, 1),
            "output_length_ratio": round(self.output_length_ratio, 3),
            "cache_affinity_admission": self.cache_affinity_admission,
            # 准入估算命中 vs 引擎实际复用：两个累加器口径一致（都按请求前缀 token 数），
            # 前者在准入时按快照算，后者在 prefill 分配后由引擎回报。
            "estimated_cache_hit_tokens": self._admission_hit_estimate_tokens,
            "actual_cache_hit_tokens": self._admission_hit_actual_tokens,
            "hit_estimate_error_tokens": (self._admission_hit_estimate_tokens
                                          - self._admission_hit_actual_tokens),
            "prefix_snapshot_version": self._prefix_snapshot_version,
            "prefix_snapshot_entries": len(self._prefix_snapshot),
            "last_estimate": dict(self._last_admission_estimate),
            "step_stats": dict(self.step_stats),
            "kv_pools": [
                {"total_blocks": total, "free_blocks": free,
                 "current_used_blocks": total - free,
                 "peak_used_blocks": self._kv_pool_peak_used[index],
                 "capacity_tokens": total * self.engine.config.kvcache_block_size}
                for index, (total, free) in enumerate(self._kv_pool_stats)
            ],
        }

    @staticmethod
    def admission_metadata(handle: GenerationHandle) -> dict[str, Any]:
        return {
            "decision": handle.admission_decision,
            "wait_ms": round(handle.admission_wait_ms, 1),
            "predicted_ttft_ms": round(handle.predicted_ttft_ms, 1),
            "queue_pressure": round(handle.queue_pressure, 3),
            "estimated_output_tokens": round(handle.estimated_output_tokens, 1),
            # 估算（准入时按前缀缓存快照算出）；实测命中见 context.prefix_cache_hit_tokens
            "estimated_cache_hit_tokens": round(handle.estimated_cache_hit_tokens, 1),
        }

    async def close(self) -> None:
        if self.task is None or self.task.done():
            return
        async with self._admission_condition:
            self.accepting = False
            self._admission_condition.notify_all()
        self.engine.request_decode_burst_yield()
        await self.incoming.put(None)
        await self.task

    async def _notify_admission_waiters(self) -> None:
        async with self._admission_condition:
            self._admission_condition.notify_all()

    def _observe_rates(self, prefill_tokens: int, decode_tokens: int,
                       elapsed_seconds: float) -> None:
        if elapsed_seconds <= 0:
            return
        if prefill_tokens > 0:
            observed = prefill_tokens / elapsed_seconds
            self.prefill_tps = 0.8 * self.prefill_tps + 0.2 * observed
        if decode_tokens > 0:
            observed = decode_tokens / elapsed_seconds
            self.decode_tps = 0.8 * self.decode_tps + 0.2 * observed

    def _observe_output_length(self, handle: GenerationHandle) -> None:
        max_tokens = handle.sampling_params.max_tokens
        if max_tokens > 0:
            observed_ratio = min(1.0, handle.generated_tokens / max_tokens)
            self.output_length_ratio = 0.8 * self.output_length_ratio + 0.2 * observed_ratio

    def _sync_prefill_progress(self) -> None:
        schedulers = ((self.engine.prefill_scheduler, self.engine.decode_scheduler)
                      if self.engine._pd else (self.engine.scheduler,))
        for scheduler in schedulers:
            for queue in (scheduler.waiting, scheduler.running, scheduler.swapped):
                for seq in queue:
                    handle = self.active.get(seq.seq_id)
                    if handle is not None:
                        handle.prefill_tokens_done = max(
                            handle.prefill_tokens_done,
                            min(len(handle.prompt_token_ids), seq.num_cached_tokens),
                        )

    async def _admit_pending(self) -> None:
        while True:
            try:
                handle = self.incoming.get_nowait()
            except asyncio.QueueEmpty:
                self._sync_decode_burst_pressure()
                return
            if handle is None:
                self.stopping = True
                self._sync_decode_burst_pressure()
                return
            await self._admit_one(handle)

    async def _admit_one(self, handle: GenerationHandle) -> None:
        if handle.cancelled:
            self.pending.pop(handle.request_id, None)
            self._finish(handle, {"error": "request cancelled"})
            return
        try:
            handle.seq_id = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.engine.add_request,
                handle.prompt_token_ids,
                handle.sampling_params,
                handle.submitted_at,
                handle.ttft_slo_ms,
                handle.tpot_slo_ms,
            )
            self.pending.pop(handle.request_id, None)
            self.active[handle.seq_id] = handle
        except Exception as exc:
            self.pending.pop(handle.request_id, None)
            self._finish(handle, {"error": str(exc)})

    def _finish(self, handle: GenerationHandle, result: dict[str, Any]) -> None:
        handle.events.put_nowait({"type": "done", "result": result})
        if handle.result is not None and not handle.result.done():
            handle.result.set_result(result)

    async def _cancel_requests(self) -> None:
        cancelled = [(seq_id, handle) for seq_id, handle in self.active.items()
                     if handle.cancelled]
        if cancelled:
            self._engine_busy = True
            try:
                for seq_id, handle in cancelled:
                    await asyncio.get_running_loop().run_in_executor(
                        self.executor, self.engine.cancel_request, seq_id)
                    self.active.pop(seq_id, None)
                    self._finish(handle, {"error": "request cancelled"})
            finally:
                self._engine_busy = False
            self._refresh_kv_pool_stats()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self.active:
                handle = await self.incoming.get()
                if handle is None:
                    self.stopping = True
                else:
                    await self._admit_one(handle)
            await self._admit_pending()
            await self._cancel_requests()
            if self.stopping:
                for handle in list(self.active.values()):
                    handle.cancelled = True
                await self._cancel_requests()
                return
            if not self.active:
                continue
            try:
                step_started = time.perf_counter()
                self._engine_busy = True
                try:
                    finished, kind, n_prefill, n_decode = await loop.run_in_executor(
                        self.executor, self.engine.step)
                finally:
                    self._engine_busy = False
                step_elapsed = time.perf_counter() - step_started
                self._observe_rates(n_prefill, n_decode, step_elapsed)
                self.step_stats["steps"] += 1
                self.step_stats["decode_iterations"] += (
                    self.engine._last_step_decode_iterations)
                self.step_stats["multi_step_decode_steps"] += int(
                    self.engine._last_step_decode_iterations > 1)
                self.step_stats["multi_step_decode_tokens"] += (
                    self.engine._last_step_multistep_tokens)
                self.step_stats["decode_burst_pressure_yields"] += int(
                    self.engine._last_step_decode_burst_yielded)
                self.step_stats["prefill_tokens"] += n_prefill
                self.step_stats["decode_tokens"] += n_decode
                self.step_stats["engine_seconds"] += step_elapsed
                if kind == "prefill":
                    self.step_stats["prefill_seconds"] += step_elapsed
                elif kind in ("decode", "spec"):
                    self.step_stats["decode_seconds"] += step_elapsed
                else:
                    token_total = n_prefill + n_decode
                    if token_total:
                        self.step_stats["prefill_seconds"] += (
                            step_elapsed * n_prefill / token_total)
                        self.step_stats["decode_seconds"] += (
                            step_elapsed * n_decode / token_total)
                self._sync_prefill_progress()
                self._refresh_kv_pool_stats()
                self._refresh_prefix_snapshot()
                # 引擎侧权威前缀复用计数（scheduler 累加，读标量不算开销）
                schedulers = ((self.engine.prefill_scheduler, self.engine.decode_scheduler)
                              if self.engine._pd else (self.engine.scheduler,))
                self.step_stats["prefix_cache_hit_tokens"] = sum(
                    scheduler.prefix_cache_hit_tokens for scheduler in schedulers)
                self.step_stats["prefix_cache_hit_requests"] = sum(
                    scheduler.prefix_cache_hit_requests for scheduler in schedulers)
                self.step_stats["prefix_cache_evictions"] = sum(
                    scheduler.block_manager.prefix_cache_evictions
                    for scheduler in schedulers)
                for seq_id, cached_tokens in self.engine._last_step_prefix_hits.items():
                    handle = self.active.get(seq_id)
                    if handle is not None:
                        # 实测命中：按增量累计（max 更新可能分多个 step 到齐）
                        growth = max(0, cached_tokens - handle.prefix_cached_tokens)
                        self._admission_hit_actual_tokens += growth
                        handle.prefix_cached_tokens = max(
                            handle.prefix_cached_tokens, cached_tokens)
                for seq_id, token_ids in self.engine._last_step_tokens.items():
                    handle = self.active.get(seq_id)
                    if handle is not None and token_ids:
                        if handle.first_token_at is None:
                            handle.first_token_at = time.perf_counter()
                        handle.generated_tokens += len(token_ids)
                        handle.prefill_tokens_done = len(handle.prompt_token_ids)
                        if handle.stream:
                            handle.events.put_nowait({"type": "tokens", "token_ids": token_ids})
                for seq_id, token_ids in finished:
                    handle = self.active.pop(seq_id, None)
                    if handle is None:
                        continue
                    is_stop = (bool(token_ids)
                               and token_ids[-1] == self.engine.config.eos
                               and not handle.sampling_params.ignore_eos)
                    result = {
                        "token_ids": token_ids,
                        "finish_reason": "stop" if is_stop else "length",
                    }
                    self._observe_output_length(handle)
                    handle.completed_at = time.perf_counter()
                    self._finish(handle, result)
                await self._notify_admission_waiters()
            except Exception as exc:
                logger.exception("inference step failed")
                async with self._admission_condition:
                    self.accepting = False
                    self._admission_condition.notify_all()
                for seq_id, handle in list(self.active.items()):
                    self.active.pop(seq_id, None)
                    try:
                        await loop.run_in_executor(
                            self.executor, self.engine.cancel_request, seq_id)
                    except Exception:
                        logger.exception("failed to cancel sequence after inference error")
                    self._finish(handle, {"error": str(exc)})
                self.stopping = True
                while not self.incoming.empty():
                    queued = self.incoming.get_nowait()
                    if queued is not None:
                        self.pending.pop(queued.request_id, None)
                        self._finish(queued, {"error": "inference worker stopped"})
                return


@dataclass(slots=True)
class Conversation:
    messages: list[dict[str, str]]
    updated_at: float
    token_budget: int
    turns: int = 0
    prompt_tokens_total: int = 0
    prefix_cache_hit_requests: int = 0
    prefix_cache_hit_tokens: int = 0
    last_prompt_tokens: int = 0
    last_prefix_cache_hit_tokens: int = 0


class ConversationStore:
    """Bounded, process-local text history; KV cache remains owned by the engine."""

    def __init__(self, max_sessions: int = 256, ttl_seconds: int = 86400):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.sessions: OrderedDict[str, Conversation] = OrderedDict()

    def _purge_expired(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        expired = [key for key, session in self.sessions.items()
                   if session.updated_at < cutoff]
        for key in expired:
            self.sessions.pop(key, None)

    def get_session(self, conversation_id: str) -> Conversation | None:
        self._purge_expired()
        session = self.sessions.get(conversation_id)
        if session is None:
            return None
        session.updated_at = time.monotonic()
        self.sessions.move_to_end(conversation_id)
        return session

    def get(self, conversation_id: str) -> list[dict[str, str]] | None:
        session = self.get_session(conversation_id)
        return None if session is None else [dict(message) for message in session.messages]

    def put(self, conversation_id: str, messages: list[dict[str, str]], *,
            token_budget: int, prompt_tokens: int,
            prefix_cached_tokens: int) -> Conversation:
        self._purge_expired()
        previous = self.sessions.get(conversation_id)
        session = Conversation(
            messages=[dict(message) for message in messages],
            updated_at=time.monotonic(),
            token_budget=token_budget,
            turns=(previous.turns if previous else 0) + 1,
            prompt_tokens_total=(previous.prompt_tokens_total if previous else 0)
            + prompt_tokens,
            prefix_cache_hit_requests=(previous.prefix_cache_hit_requests if previous else 0)
            + int(prefix_cached_tokens > 0),
            prefix_cache_hit_tokens=(previous.prefix_cache_hit_tokens if previous else 0)
            + prefix_cached_tokens,
            last_prompt_tokens=prompt_tokens,
            last_prefix_cache_hit_tokens=prefix_cached_tokens,
        )
        self.sessions[conversation_id] = session
        self.sessions.move_to_end(conversation_id)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
        return session

    @staticmethod
    def metadata(session: Conversation) -> dict[str, int | float]:
        return {
            "session_token_budget": session.token_budget,
            "turns": session.turns,
            "prompt_tokens_total": session.prompt_tokens_total,
            "last_prompt_tokens": session.last_prompt_tokens,
            "prefix_cache_hit_requests": session.prefix_cache_hit_requests,
            "prefix_cache_hit_tokens": session.prefix_cache_hit_tokens,
            "last_prefix_cache_hit_tokens": session.last_prefix_cache_hit_tokens,
            "prefix_cache_hit_request_rate": (
                session.prefix_cache_hit_requests / session.turns if session.turns else 0.0),
            "prefix_cache_hit_token_rate": (
                session.prefix_cache_hit_tokens / session.prompt_tokens_total
                if session.prompt_tokens_total else 0.0),
        }

    def delete(self, conversation_id: str) -> bool:
        self._purge_expired()
        return self.sessions.pop(conversation_id, None) is not None


class KeyedLocks:
    """Serialize turns within one conversation without blocking other sessions."""

    def __init__(self):
        self._guard = asyncio.Lock()
        self._entries: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, key: str):
        async with self._guard:
            lock, users = self._entries.get(key, (asyncio.Lock(), 0))
            self._entries[key] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                lock, users = self._entries[key]
                if users == 1:
                    del self._entries[key]
                else:
                    self._entries[key] = (lock, users - 1)


class IncrementalDecoder:
    """Decode accumulated token ids while holding a short mutable UTF-8 tail."""

    def __init__(self, tokenizer, tail_chars: int = 8):
        self.tokenizer = tokenizer
        self.tail_chars = tail_chars
        self.token_ids: list[int] = []
        self.emitted_chars = 0

    def push(self, token_ids: list[int], *, final: bool = False) -> str:
        self.token_ids.extend(token_ids)
        text = self.tokenizer.decode(
            self.token_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        stable_end = len(text) if final else max(0, len(text) - self.tail_chars)
        stable_end = max(self.emitted_chars, stable_end)
        delta = text[self.emitted_chars:stable_end]
        self.emitted_chars = stable_end
        return delta


def _encode_chat(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        return list(token_ids)
    prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    return tokenizer.encode(f"{prompt}\nassistant:", add_special_tokens=False)


def _truncate_history(tokenizer, messages: list[dict[str, str]], token_budget: int
                      ) -> tuple[list[dict[str, str]], int]:
    """Drop oldest complete user turns until the prompt fits the model window."""
    compacted = [dict(message) for message in messages]
    removed_turns = 0
    while len(_encode_chat(tokenizer, compacted)) > token_budget:
        user_indices = [index for index, message in enumerate(compacted)
                        if message["role"] == "user"]
        if len(user_indices) < 2:
            raise HTTPException(
                status_code=400,
                detail="prompt is longer than the available context window after reserving max_tokens",
            )
        first_turn = user_indices[0]
        next_turn = user_indices[1]
        old_turn = compacted[first_turn:next_turn]
        preserved_system = [message for message in old_turn if message["role"] == "system"]
        del compacted[first_turn:next_turn]
        if preserved_system:
            compacted[0:0] = preserved_system
        removed_turns += 1
    return compacted, removed_turns


async def _summarize_history(tokenizer, manager: GenerationManager,
                             messages: list[dict[str, str]], token_budget: int,
                             max_model_len: int) -> tuple[list[dict[str, str]], int]:
    """Summarize old complete turns with the served model, preserving recent turns."""
    if len(_encode_chat(tokenizer, messages)) <= token_budget:
        return [dict(message) for message in messages], 0
    summary_prefix = "Earlier conversation summary:\n"
    system_messages = []
    previous_summary = []
    dialogue = []
    for message in messages:
        if message["role"] != "system":
            dialogue.append(dict(message))
        elif message["content"].startswith(summary_prefix):
            previous_summary.append(message["content"][len(summary_prefix):])
        else:
            system_messages.append(dict(message))

    turns: list[list[dict[str, str]]] = []
    for message in dialogue:
        if message["role"] == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    if not turns or turns[-1][-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="the current turn must end with a user message")

    recent_turns = turns[:-1]
    current_turn = turns[-1]
    dropped_turns: list[list[dict[str, str]]] = []
    summary_budget = min(256, max(32, token_budget // 8))
    placeholder = {"role": "system", "content": summary_prefix + "x " * summary_budget}

    def candidate(summary: str, kept: list[list[dict[str, str]]]) -> list[dict[str, str]]:
        result = list(system_messages)
        if summary:
            result.append({"role": "system", "content": summary_prefix + summary})
        result.extend(message for turn in kept for message in turn)
        result.extend(current_turn)
        return result

    def placeholder_candidate(kept: list[list[dict[str, str]]]) -> list[dict[str, str]]:
        result = list(system_messages)
        result.append(placeholder)
        result.extend(message for turn in kept for message in turn)
        result.extend(current_turn)
        return result

    while len(_encode_chat(tokenizer, placeholder_candidate(recent_turns))) > token_budget:
        if not recent_turns:
            without_summary = candidate("", [])
            if len(_encode_chat(tokenizer, without_summary)) <= token_budget:
                return without_summary, len(dropped_turns)
            raise HTTPException(
                status_code=400,
                detail="current system prompt and user message exceed the available context window",
            )
        dropped_turns.append(recent_turns.pop(0))

    if not dropped_turns and not previous_summary:
        return _truncate_history(tokenizer, messages, token_budget)

    dropped_text = "\n".join(previous_summary + [
        "\n".join(f"{message['role']}: {message['content']}" for message in turn)
        for turn in dropped_turns
    ])
    instruction = (
        "Compress the earlier conversation into concise factual notes. Preserve names, "
        "preferences, decisions, constraints, and unresolved tasks. Do not answer the user. "
        "Do not invent facts."
    )
    summary_max_tokens = max(16, summary_budget)
    raw_ids = tokenizer.encode(dropped_text, add_special_tokens=False)
    while True:
        summary_input = tokenizer.decode(raw_ids, skip_special_tokens=True)
        summary_prompt_ids = _encode_chat(tokenizer, [
            {"role": "system", "content": instruction},
            {"role": "user", "content": summary_input},
        ])
        if len(summary_prompt_ids) + summary_max_tokens <= max_model_len:
            break
        if not raw_ids:
            raise HTTPException(status_code=400, detail="unable to fit the context summary prompt")
        raw_ids = raw_ids[max(1, len(raw_ids) // 8):]

    summary_handle = await manager.submit(
        summary_prompt_ids, SamplingParams(temperature=0.2, max_tokens=summary_max_tokens))
    summary_result = await _wait_for_result(summary_handle)
    if "error" in summary_result:
        raise HTTPException(status_code=500, detail=f"context summarization failed: {summary_result['error']}")
    summary_ids = tokenizer.encode(
        tokenizer.decode(summary_result["token_ids"], skip_special_tokens=True),
        add_special_tokens=False)[:summary_budget]
    summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
    compacted = candidate(summary, recent_turns)
    while len(_encode_chat(tokenizer, compacted)) > token_budget and summary_ids:
        summary_ids = summary_ids[:max(0, len(summary_ids) - 16)]
        summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
        compacted = candidate(summary, recent_turns)
    if len(_encode_chat(tokenizer, compacted)) > token_budget:
        raise HTTPException(status_code=400, detail="unable to compact conversation to the context window")
    return compacted, len(dropped_turns)


def _normalize_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    normalized = [{"role": message.role, "content": message.content} for message in messages]
    if not normalized or normalized[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="the last chat message must have role 'user'")
    return normalized


async def _wait_for_result(handle: GenerationHandle) -> dict[str, Any]:
    assert handle.result is not None
    try:
        return await handle.result
    except asyncio.CancelledError:
        handle.cancelled = True
        raise


def _sse(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(model: str, *, engine_kwargs: dict[str, Any] | None = None,
               max_queued_requests: int = 256, max_sessions: int = 256,
               session_ttl_seconds: int = 86400,
               session_token_budget: int | None = None,
               context_compaction: Literal["truncate", "summarize"] = "truncate",
               dynamic_admission: bool = True,
               prefix_cache_aware_admission: bool = True,
               max_deferred_requests: int = 64,
               max_admission_wait_ms: float = 2000.0,
               admission_work_budget_ms: float = 10000.0,
               admission_target_ttft_ms: float = 2000.0,
               admission_soft_pressure: float = 0.85,
               admission_prefill_tps_fallback: float = 10000.0,
               admission_decode_tps_fallback: float = 1000.0,
               cache_affinity_admission: bool = True) -> FastAPI:
    """Create an OpenAI-compatible app; model loading happens in the lifespan."""
    engine_kwargs = dict(engine_kwargs or {})
    if max_queued_requests < 1:
        raise ValueError("max_queued_requests must be at least 1")
    if max_deferred_requests < 0:
        raise ValueError("max_deferred_requests must be non-negative")
    if not isfinite(max_admission_wait_ms) or max_admission_wait_ms < 0:
        raise ValueError("max_admission_wait_ms must be non-negative")
    if (not isfinite(admission_work_budget_ms) or admission_work_budget_ms <= 0
            or not isfinite(admission_target_ttft_ms) or admission_target_ttft_ms <= 0):
        raise ValueError("admission time budgets must be positive")
    if not isfinite(admission_soft_pressure) or not 0 < admission_soft_pressure <= 1:
        raise ValueError("admission_soft_pressure must be in (0, 1]")
    if (not isfinite(admission_prefill_tps_fallback)
            or admission_prefill_tps_fallback <= 0
            or not isfinite(admission_decode_tps_fallback)
            or admission_decode_tps_fallback <= 0):
        raise ValueError("admission throughput fallbacks must be positive")
    if max_sessions < 1:
        raise ValueError("max_sessions must be at least 1")
    if session_ttl_seconds < 1:
        raise ValueError("session_ttl_seconds must be at least 1")
    if session_token_budget is not None and session_token_budget < 1:
        raise ValueError("session_token_budget must be at least 1")
    if context_compaction not in ("truncate", "summarize"):
        raise ValueError("context_compaction must be 'truncate' or 'summarize'")
    app = FastAPI(title="nano-vllm", version="0.2.0")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nanovllm-engine")
        loop = asyncio.get_running_loop()
        engine = None
        manager = None
        try:
            engine = await loop.run_in_executor(executor, lambda: LLM(model, **engine_kwargs))
            if (session_token_budget is not None
                    and session_token_budget > engine.config.max_model_len):
                raise ValueError("session_token_budget cannot exceed max_model_len")
            manager = GenerationManager(
                engine, executor, max_queued_requests,
                dynamic_admission=dynamic_admission,
                prefix_cache_aware_admission=prefix_cache_aware_admission,
                max_deferred_requests=max_deferred_requests,
                max_admission_wait_ms=max_admission_wait_ms,
                admission_work_budget_ms=admission_work_budget_ms,
                admission_target_ttft_ms=admission_target_ttft_ms,
                admission_soft_pressure=admission_soft_pressure,
                prefill_tps_fallback=admission_prefill_tps_fallback,
                decode_tps_fallback=admission_decode_tps_fallback,
                cache_affinity_admission=cache_affinity_admission,
            )
            manager.start()
            app.state.engine = engine
            app.state.manager = manager
            app.state.conversations = ConversationStore(max_sessions, session_ttl_seconds)
            app.state.conversation_locks = KeyedLocks()
            app.state.session_token_budget = session_token_budget
            yield
        finally:
            try:
                if manager is not None:
                    await manager.close()
            finally:
                try:
                    if engine is not None:
                        await loop.run_in_executor(executor, engine.exit)
                finally:
                    executor.shutdown(wait=True)

    app.router.lifespan_context = lifespan

    @app.get("/health")
    async def health():
        engine = getattr(app.state, "engine", None)
        if engine is None:
            return {"status": "starting"}
        return {
            "status": "ok",
            "execution_mode": engine.execution_mode,
            "active_requests": len(app.state.manager.active),
            "admission": app.state.manager.admission_snapshot(),
            "conversations": len(app.state.conversations.sessions),
            "max_model_len": engine.config.max_model_len,
            "decode": {
                "multi_step_enabled": engine.config.multi_step_decode,
                "max_steps": engine.config.max_decode_steps,
                "burst_yield": engine.config.decode_burst_yield,
                "tpot_aware": engine.config.tpot_aware_scheduling,
                "default_tpot_slo_ms": engine.config.default_tpot_slo_ms,
            },
        }

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{
            "id": model, "object": "model", "owned_by": "nano-vllm",
        }]}

    async def submit_prompt(prompt_ids: list[int], temperature: float, max_tokens: int,
                            *, stream: bool, submitted_at: float | None = None,
                            ttft_slo_ms: float | None = None,
                            tpot_slo_ms: float | None = None):
        engine = app.state.engine
        if max_tokens >= engine.config.max_model_len:
            raise HTTPException(status_code=400, detail="max_tokens must be smaller than max_model_len")
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        return await app.state.manager.submit(
            prompt_ids, params, stream=stream, submitted_at=submitted_at,
            ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)

    def stream_response(handle: GenerationHandle, *, chat: bool,
                        model_name: str) -> StreamingResponse:
        async def events():
            decoder = IncrementalDecoder(app.state.engine.tokenizer)
            created = int(time.time())
            completed = False
            if chat:
                first = {"id": handle.request_id, "object": "chat.completion.chunk",
                         "created": created, "model": model_name,
                         "choices": [{"index": 0, "delta": {"role": "assistant"},
                                      "finish_reason": None}]}
            else:
                first = {"id": handle.request_id, "object": "text_completion",
                         "created": created, "model": model_name,
                         "choices": [{"index": 0, "text": "", "finish_reason": None}]}
            try:
                yield _sse(first)
                while True:
                    event = await handle.events.get()
                    if event["type"] == "tokens":
                        text = decoder.push(event["token_ids"])
                        if text:
                            if chat:
                                payload = {"id": handle.request_id,
                                           "object": "chat.completion.chunk",
                                           "created": created, "model": model_name,
                                           "choices": [{"index": 0,
                                                        "delta": {"content": text},
                                                        "finish_reason": None}]}
                            else:
                                payload = {"id": handle.request_id,
                                           "object": "text_completion",
                                           "created": created, "model": model_name,
                                           "choices": [{"index": 0, "text": text,
                                                        "finish_reason": None}]}
                            yield _sse(payload)
                        continue
                    result = event["result"]
                    if "error" in result:
                        payload = {"error": {"message": result["error"],
                                              "type": "server_error"},
                                   "admission": GenerationManager.admission_metadata(handle)}
                        yield _sse(payload)
                        completed = True
                        break
                    tail = decoder.push([], final=True)
                    finish_reason = result["finish_reason"]
                    if chat:
                        payload = {"id": handle.request_id,
                                   "object": "chat.completion.chunk",
                                   "created": created, "model": model_name,
                                   "choices": [{"index": 0,
                                                "delta": {"content": tail} if tail else {},
                                                "finish_reason": finish_reason}]}
                    else:
                        payload = {"id": handle.request_id,
                                   "object": "text_completion",
                                   "created": created, "model": model_name,
                                   "choices": [{"index": 0, "text": tail,
                                                "finish_reason": finish_reason}]}
                    payload["admission"] = GenerationManager.admission_metadata(handle)
                    yield _sse(payload)
                    completed = True
                    break
                yield _sse("[DONE]")
            except asyncio.CancelledError:
                handle.cancelled = True
                raise
            finally:
                if not completed:
                    handle.cancelled = True

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Admission-Decision": handle.admission_decision,
            "X-Admission-Wait-Ms": f"{handle.admission_wait_ms:.1f}",
            "X-Predicted-TTFT-Ms": f"{handle.predicted_ttft_ms:.1f}",
            "X-Queue-Pressure": f"{handle.queue_pressure:.3f}",
            "X-Estimated-Output-Tokens": f"{handle.estimated_output_tokens:.1f}",
            "X-Prefix-Cache-Hit-Tokens": str(handle.prefix_cached_tokens),
        })

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        request_received_at = time.perf_counter()
        engine = app.state.engine
        prompt_ids = engine.tokenizer.encode(request.prompt, add_special_tokens=False)
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        max_tokens = request.max_tokens
        if len(prompt_ids) + max_tokens > engine.config.max_model_len:
            raise HTTPException(status_code=400, detail="prompt plus max_tokens exceeds max_model_len")
        handle = await submit_prompt(prompt_ids, request.temperature, max_tokens,
                                     stream=request.stream,
                                     submitted_at=request_received_at,
                                     ttft_slo_ms=request.ttft_slo_ms,
                                     tpot_slo_ms=request.tpot_slo_ms)
        served_model = request.model or model
        if request.stream:
            return stream_response(handle, chat=False, model_name=served_model)
        result = await _wait_for_result(handle)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        text = engine.tokenizer.decode(
            result["token_ids"], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return {
            "id": handle.request_id, "object": "text_completion", "created": int(time.time()),
            "model": served_model,
            "choices": [{"index": 0, "text": text, "finish_reason": result["finish_reason"]}],
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": len(result["token_ids"]),
                      "total_tokens": len(prompt_ids) + len(result["token_ids"])},
            # 与 /v1/chat/completions 对齐：暴露实际前缀复用（无会话时 session 为 None）
            "context": {"prefix_cache_hit_tokens": handle.prefix_cached_tokens,
                        "estimated_cache_hit_tokens": int(
                            handle.estimated_cache_hit_tokens)},
            "admission": app.state.manager.admission_metadata(handle),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        request_received_at = time.perf_counter()
        engine = app.state.engine
        served_model = request.model or model
        max_tokens = request.max_completion_tokens or request.max_tokens
        if max_tokens >= engine.config.max_model_len:
            raise HTTPException(status_code=400, detail="max_tokens must be smaller than max_model_len")
        incoming = _normalize_messages(request.messages)
        conversation_id = request.conversation_id
        lock_context = (app.state.conversation_locks.hold(conversation_id)
                        if conversation_id else _empty_context())
        await lock_context.__aenter__()
        try:
            store: ConversationStore = app.state.conversations
            existing_session = (store.get_session(conversation_id)
                                if conversation_id else None)
            existing = (existing_session.messages if existing_session is not None else None)
            if existing is not None:
                if any(message["role"] != "user" for message in incoming):
                    raise HTTPException(
                        status_code=400,
                        detail="managed conversations accept only new user messages after creation",
                    )
                prepared = existing + incoming
            else:
                prepared = incoming
            session_budget = (request.session_token_budget
                              or (existing_session.token_budget
                                  if existing_session is not None else None)
                              or app.state.session_token_budget
                              or engine.config.max_model_len)
            if session_budget > engine.config.max_model_len:
                raise HTTPException(
                    status_code=400,
                    detail="session_token_budget cannot exceed max_model_len",
                )
            prompt_budget = min(engine.config.max_model_len - max_tokens, session_budget)
            if context_compaction == "summarize":
                prompt_messages, removed_turns = await _summarize_history(
                    engine.tokenizer, app.state.manager, prepared, prompt_budget,
                    engine.config.max_model_len)
            else:
                prompt_messages, removed_turns = _truncate_history(
                    engine.tokenizer, prepared, prompt_budget)
            prompt_ids = _encode_chat(engine.tokenizer, prompt_messages)
            handle = await app.state.manager.submit(
                prompt_ids, SamplingParams(temperature=request.temperature, max_tokens=max_tokens),
                stream=request.stream, submitted_at=request_received_at,
                ttft_slo_ms=request.ttft_slo_ms,
                tpot_slo_ms=request.tpot_slo_ms)
        except BaseException:
            await lock_context.__aexit__(None, None, None)
            raise

        if request.stream:
            async def session_stream():
                completed = False
                try:
                    async for chunk in stream_response(
                            handle, chat=True, model_name=served_model).body_iterator:
                        if isinstance(chunk, bytes):
                            chunk = chunk.decode("utf-8")
                        if chunk.strip() == "data: [DONE]":
                            result = await _wait_for_result(handle)
                            if "error" not in result:
                                final_messages = prompt_messages + [{
                                    "role": "assistant",
                                    "content": engine.tokenizer.decode(
                                        result["token_ids"], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False),
                                }]
                                if conversation_id:
                                    store.put(
                                        conversation_id, final_messages,
                                        token_budget=session_budget,
                                        prompt_tokens=len(prompt_ids),
                                        prefix_cached_tokens=handle.prefix_cached_tokens,
                                    )
                            completed = True
                        yield chunk
                finally:
                    if not completed:
                        handle.cancelled = True
                    await lock_context.__aexit__(None, None, None)

            return StreamingResponse(session_stream(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache", "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Admission-Decision": handle.admission_decision,
                "X-Admission-Wait-Ms": f"{handle.admission_wait_ms:.1f}",
                "X-Predicted-TTFT-Ms": f"{handle.predicted_ttft_ms:.1f}",
                "X-Queue-Pressure": f"{handle.queue_pressure:.3f}",
                "X-Estimated-Output-Tokens": f"{handle.estimated_output_tokens:.1f}",
                "X-Estimated-Prefill-Tokens": str(handle.estimated_prefill_tokens),
                "X-Estimated-Prefix-Cached-Tokens": str(
                    handle.estimated_prefix_cached_tokens),
            })
        try:
            result = await _wait_for_result(handle)
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            answer = engine.tokenizer.decode(
                result["token_ids"], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            if conversation_id:
                session = store.put(
                    conversation_id,
                    prompt_messages + [{"role": "assistant", "content": answer}],
                    token_budget=session_budget,
                    prompt_tokens=len(prompt_ids),
                    prefix_cached_tokens=handle.prefix_cached_tokens,
                )
                session_context = store.metadata(session)
            else:
                session_context = None
            completion_tokens = len(result["token_ids"])
            return {
                "id": handle.request_id, "object": "chat.completion",
                "created": int(time.time()), "model": served_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": result["finish_reason"]}],
                "usage": {"prompt_tokens": len(prompt_ids),
                          "completion_tokens": completion_tokens,
                          "total_tokens": len(prompt_ids) + completion_tokens},
                "context": {"conversation_id": conversation_id,
                            "removed_turns": removed_turns,
                            "session_token_budget": session_budget,
                            "prefix_cache_hit_tokens": handle.prefix_cached_tokens,
                            "estimated_cache_hit_tokens": int(
                                handle.estimated_cache_hit_tokens),
                            "session_statistics": session_context},
                "admission": app.state.manager.admission_metadata(handle),
            }
        finally:
            await lock_context.__aexit__(None, None, None)

    @app.get("/v1/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        session = app.state.conversations.get_session(conversation_id)
        if session is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        messages = [dict(message) for message in session.messages]
        token_count = len(_encode_chat(app.state.engine.tokenizer, messages))
        return {"conversation_id": conversation_id, "messages": messages,
                "prompt_tokens": token_count,
                "session": app.state.conversations.metadata(session)}

    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        async with app.state.conversation_locks.hold(conversation_id):
            deleted = app.state.conversations.delete(conversation_id)
        return {"conversation_id": conversation_id, "deleted": deleted}

    return app


@asynccontextmanager
async def _empty_context():
    yield


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nano-vLLM OpenAI-compatible server")
    parser.add_argument("model", help="local Hugging Face model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execution-mode", choices=("auto", "mixed", "pd"), default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prefix-cache-max-free-blocks", type=int, default=0,
                        help="cap idle reusable prefix blocks; 0 leaves the cap unset")
    parser.add_argument("--no-mixed-cudagraph", action="store_true",
                        help="run mixed prefill/decode batches eagerly")
    parser.add_argument("--mixed-cudagraph-max-graphs", type=int, default=4)
    parser.add_argument("--mixed-cudagraph-max-tokens", type=int, default=4096,
                        help="use eager execution for larger mixed batches to bound capture memory")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8_e4m3"), default="auto")
    parser.add_argument("--kv-calibration-path", default="",
                        help="FP8 KV calibration token IDs JSON")
    parser.add_argument("--kv-fp8-scale-margin", type=float, default=1.1,
                        help="FP8 KV calibration scale margin; values below 1 clip outliers")
    parser.add_argument("--no-kv-swap", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=256)
    parser.add_argument("--session-ttl-seconds", type=int, default=86400)
    parser.add_argument("--session-token-budget", type=int, default=None,
                        help="default retained prompt-token budget per managed conversation")
    parser.add_argument("--max-queued-requests", type=int, default=256)
    parser.add_argument("--no-dynamic-admission", action="store_true",
                        help="disable predictive accept/defer admission; keep the hard queue cap")
    parser.add_argument("--no-admission-prefix-cache-awareness", action="store_true",
                        help="ignore reusable prompt prefixes in admission estimates (ablation)")
    parser.add_argument("--max-deferred-requests", type=int, default=64)
    parser.add_argument("--max-admission-wait-ms", type=float, default=2000.0)
    parser.add_argument("--admission-work-budget-ms", type=float, default=10000.0)
    parser.add_argument("--admission-target-ttft-ms", type=float, default=2000.0)
    parser.add_argument("--admission-soft-pressure", type=float, default=0.85)
    parser.add_argument("--admission-prefill-tps-fallback", type=float, default=10000.0)
    parser.add_argument("--admission-decode-tps-fallback", type=float, default=1000.0)
    parser.add_argument("--context-compaction", choices=("truncate", "summarize"),
                        default="truncate")
    parser.add_argument("--no-latency-aware-scheduling", action="store_true")
    parser.add_argument("--no-cache-affinity-admission", action="store_true",
                        help="disable Top-W cache-affinity ordering AND cache-aware "
                             "admission estimates (prefill work is charged in full)")
    parser.add_argument("--no-aging-fairness", action="store_true")
    parser.add_argument("--no-recompute-aware-preemption", action="store_true")
    parser.add_argument("--admission-window", type=int, default=16)
    parser.add_argument("--aging-timeout-ms", type=float, default=2000.0)
    parser.add_argument("--prefill-reserve-tokens", type=int, default=256)
    parser.add_argument("--no-slo-aware-scheduling", action="store_true")
    parser.add_argument("--default-ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--default-tpot-slo-ms", type=float, default=None,
                        help="TPOT scheduling target for requests without an explicit target (ms)")
    parser.add_argument("--no-tpot-aware-scheduling", action="store_true",
                        help="disable active-request TPOT prioritization and prefill throttling")
    parser.add_argument("--tpot-decode-ms-fallback", type=float, default=20.0,
                        help="estimated per-token latency before request decode samples exist (ms)")
    parser.add_argument("--no-multi-step-decode", action="store_true",
                        help="disable bounded multi-step decode (ablation path)")
    parser.add_argument("--max-decode-steps", type=int, default=4,
                        help="maximum pure-decode model forwards per engine step")
    parser.add_argument("--no-decode-burst-yield", action="store_true",
                        help="disable early burst yield: run the full multi-step round "
                             "budget even when prefill work is queued (ablation path)")
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=4096)
    parser.add_argument("--queue-depth-for-full-prefill", type=int, default=16)
    parser.add_argument("--preempt-prefill-tps", type=float, default=10000.0)
    parser.add_argument("--preempt-kv-transfer-gbps", type=float, default=12.0)
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        args.model,
        engine_kwargs={
            "execution_mode": args.execution_mode,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "prefix_cache_max_free_blocks": args.prefix_cache_max_free_blocks,
            "mixed_cudagraph": not args.no_mixed_cudagraph,
            "mixed_cudagraph_max_graphs": args.mixed_cudagraph_max_graphs,
            "mixed_cudagraph_max_tokens": args.mixed_cudagraph_max_tokens,
            "kv_cache_dtype": args.kv_cache_dtype,
            "kv_calibration_path": args.kv_calibration_path,
            "kv_fp8_scale_margin": args.kv_fp8_scale_margin,
            "kv_swap": not args.no_kv_swap,
            "latency_aware_scheduling": not args.no_latency_aware_scheduling,
            "cache_affinity_admission": not args.no_cache_affinity_admission,
            "aging_fairness": not args.no_aging_fairness,
            "recompute_aware_preemption": not args.no_recompute_aware_preemption,
            "admission_window": args.admission_window,
            "aging_timeout_ms": args.aging_timeout_ms,
            "prefill_reserve_tokens": args.prefill_reserve_tokens,
            "slo_aware_scheduling": not args.no_slo_aware_scheduling,
            "default_ttft_slo_ms": args.default_ttft_slo_ms,
            "tpot_aware_scheduling": not args.no_tpot_aware_scheduling,
            "default_tpot_slo_ms": args.default_tpot_slo_ms,
            "tpot_decode_ms_fallback": args.tpot_decode_ms_fallback,
            "multi_step_decode": not args.no_multi_step_decode,
            "max_decode_steps": args.max_decode_steps,
            "decode_burst_yield": not args.no_decode_burst_yield,
            "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
            "queue_depth_for_full_prefill": args.queue_depth_for_full_prefill,
            "preempt_prefill_tokens_per_second": args.preempt_prefill_tps,
            "preempt_kv_transfer_gbps": args.preempt_kv_transfer_gbps,
        },
        max_queued_requests=args.max_queued_requests,
        max_sessions=args.max_sessions,
        session_ttl_seconds=args.session_ttl_seconds,
        session_token_budget=args.session_token_budget,
        context_compaction=args.context_compaction,
        dynamic_admission=not args.no_dynamic_admission,
        prefix_cache_aware_admission=(
            not args.no_admission_prefix_cache_awareness),
        max_deferred_requests=args.max_deferred_requests,
        max_admission_wait_ms=args.max_admission_wait_ms,
        admission_work_budget_ms=args.admission_work_budget_ms,
        admission_target_ttft_ms=args.admission_target_ttft_ms,
        admission_soft_pressure=args.admission_soft_pressure,
        admission_prefill_tps_fallback=args.admission_prefill_tps_fallback,
        admission_decode_tps_fallback=args.admission_decode_tps_fallback,
        cache_affinity_admission=not args.no_cache_affinity_admission,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

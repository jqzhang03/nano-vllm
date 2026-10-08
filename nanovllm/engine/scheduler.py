from collections import deque
from math import ceil
from time import perf_counter

import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.ngram import find_ngram_draft


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.latency_aware_scheduling = config.latency_aware_scheduling
        self.cache_affinity_admission = config.cache_affinity_admission
        self.admission_window = config.admission_window
        self.aging_fairness = config.aging_fairness
        self.aging_timeout_s = config.aging_timeout_ms / 1000.0
        self.prefill_reserve_tokens = config.prefill_reserve_tokens
        self.slo_aware_scheduling = config.slo_aware_scheduling
        self.default_ttft_slo_s = (None if config.default_ttft_slo_ms is None else
                                    config.default_ttft_slo_ms / 1000.0)
        self.tpot_aware_scheduling = config.tpot_aware_scheduling
        self.default_tpot_slo_ms = config.default_tpot_slo_ms
        self.tpot_decode_ms_fallback = config.tpot_decode_ms_fallback
        # TPOT 压力对 prefill 的收缩死区（见 _tpot_decode_pressure 的注释）
        self.tpot_prefill_throttle_margin = config.tpot_prefill_throttle_margin
        # 等待队列达到该深度时停用 TPOT 对 prefill 的压缩（防自锁，见同处注释）
        self.tpot_throttle_max_waiting = config.tpot_throttle_max_waiting
        self.external_tpot_pressure = 0.0
        self.external_tpot_target_active = False
        self.max_prefill_chunk_tokens = config.max_prefill_chunk_tokens
        self.queue_depth_for_full_prefill = config.queue_depth_for_full_prefill
        self.recompute_aware_preemption = config.recompute_aware_preemption
        self._prefill_seconds_per_token = 1.0 / config.preempt_prefill_tokens_per_second
        self._swap_seconds_per_byte = 1.0 / (config.preempt_kv_transfer_gbps * 1e9)
        self._single_slot_prefill_turn = True
        # ---- SWA 滚动缓冲（阶段 2b/2b扩展）----
        # 前置：统一窗口（mistral）或交替 local/global（gemma2，**split 模式**：
        # local 层走环池、global 层走 full 池普通分页、永不驱逐）的 bf16/fp8 KV
        # MHA 模型；投机仅限 ngram；窗口 ≥ 块大小。
        self.rolling = config.rolling_cache
        ring_window = None
        self.rolling_split = False
        if self.rolling:
            hf = config.hf_config
            assert config.speculative in ("none", "ngram"), \
                "rolling_cache + medusa/eagle 投机未实现（verify 行窗口回读的环余量语义）"
            assert config.kv_cache_dtype in ("auto", "fp8_e4m3"), \
                "rolling_cache 需要 bf16 或 fp8 KV"
            assert hf.model_type in ("mistral", "gemma2"), \
                f"rolling_cache 支持 mistral（统一窗口）与 gemma2（交替窗口）；model_type={hf.model_type!r}"
            ring_window = getattr(hf, "sliding_window", None)
            assert ring_window, "模型缺 sliding_window"
            assert ring_window >= config.kvcache_block_size, \
                "滚动缓冲要求 sliding_window ≥ 块大小（spec 写 span/驱逐分离）"
            # gemma2 交替 local/global：窗口层与非窗口层并存 → split 模式
            # （runner 在 allocate 后写回 config.ring_blocks/full_blocks）
            if hf.model_type == "gemma2":
                from nanovllm.models.gemma2 import gemma2_layer_types
                types = gemma2_layer_types(hf, hf.num_hidden_layers)
                windows = {ring_window if t == "sliding" else None for t in types}
                assert windows == {None, ring_window} or windows == {ring_window}, \
                    f"gemma2 窗口模式未覆盖：{windows}"
                self.rolling_split = windows == {None, ring_window}
            if self.rolling_split:
                assert config.speculative == "none", \
                    "split（交替窗口）模式暂不支持投机"
        # split 模式（交替窗口，阶段 2b 扩展）：两个独立池
        #   block_manager（环池，rolling）：local 层，窗口内容
        #   full_block_manager（full 池，no_share）：global 层，全历史
        if self.rolling_split:
            nb_r = config.num_ring_kvcache_blocks
            nb_f = config.num_full_kvcache_blocks
            assert nb_r > 0 and nb_f > 0, \
                "split（交替窗口）模式缺环池/full 池块数（runner 未设置？）"
            self.block_manager = BlockManager(
                nb_r, config.kvcache_block_size,
                rolling_window=ring_window,
                ring_slack=config.max_draft_len + 2,
                max_free_prefix_blocks=config.prefix_cache_max_free_blocks,
                prefix_feature_cache=config.prefix_feature_cache)
            self.full_block_manager = BlockManager(
                nb_f, config.kvcache_block_size,
                no_share=True, table_attr="kv_table")
        else:
            self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                              rolling_window=ring_window,
                                              ring_slack=config.max_draft_len + 2,
                                              no_share=config.pd_separation,
                                              max_free_prefix_blocks=config.prefix_cache_max_free_blocks,
                                              prefix_feature_cache=config.prefix_feature_cache)
            self.full_block_manager = None
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()  # KV swap 抢占：KV 已换出到 CPU 的序列
        self.num_preemptions = 0  # 抢占次数统计（基准测试使用）
        self.num_swaps = 0  # KV swap 换出次数统计（基准测试使用）
        self.num_recompute_preemptions = 0
        self.recompute_tokens = 0
        self.estimated_recompute_seconds = 0.0
        self.estimated_swap_seconds = 0.0
        self.num_affinity_probes = 0
        self.num_prefix_feature_parses = 0
        self.num_prefix_feature_reuses = 0
        self.prefix_cache_hit_tokens = 0
        self.prefix_cache_hit_requests = 0
        self.num_aging_promotions = 0
        self.num_slo_prefill_steps = 0
        self.slo_prefill_budget_sum = 0
        self.slo_prefill_budget_max = 0
        self.slo_prefill_rows_sum = 0
        self.num_tpot_prefill_steps = 0
        self.tpot_prefill_budget_sum = 0
        self.num_tpot_priority_steps = 0
        # TPOT 压力对 prefill 配额的收缩观测：scale = prefill_pressure·(1−tpot_pressure)，
        # scale→0 表示 prefill 被压到 min_budget（饥饿）。
        self.slo_prefill_budget_scale_sum = 0.0
        self.slo_prefill_budget_scale_min = 1.0
        self._last_prefill_budget_scale = 1.0
        self.num_tpot_starved_steps = 0
        self._last_tpot_pressure = 0.0
        self.cow_pairs: list[tuple[int, int]] = []  # 本轮调度产生的COW复制对 (old_block_id, new_block_id)
        self.swap_pairs: list[tuple[Sequence, list[int] | dict[str, list[int]],
                                    object, str]] = []  # GPU拷贝由engine在run前执行
        self._swap_buffers: dict[int, object] = {}  # seq_id → CPU 缓冲（换出时分配，换入后释放）
        # FP8 KV 在 CPU 侧按 uint8 原始字节保存并往返，避免 float8 的 CPU 算子限制。
        # TP>1 的 spawn 进程尚未实现每个 rank 的独立 offload buffer，因此仍回退 recompute。
        self.kv_swap = config.kv_swap and config.tensor_parallel_size == 1
        self._swap_max_bytes = int(config.kv_swap_space_gb * 1e9)
        self._swap_bytes = 0  # 当前换出缓冲累计字节（超预算回落 recompute）
        if self.kv_swap:
            hf = config.hf_config
            self._swap_mla = hasattr(hf, "kv_lora_rank")   # MLA fused cache
            self._swap_layers = hf.num_hidden_layers
            self._swap_kv_heads = hf.num_key_value_heads // config.tensor_parallel_size
            self._swap_head_dim = (getattr(hf, "head_dim", None)
                                   or hf.hidden_size // hf.num_attention_heads)
            self._swap_mla_d = (hf.kv_lora_rank + hf.qk_rope_head_dim
                                if self._swap_mla else 0)
            self._swap_fp8 = config.kv_cache_dtype == "fp8_e4m3"
            self._swap_dtype = torch.uint8 if self._swap_fp8 else hf.dtype
            itemsize = 1 if self._swap_fp8 else hf.dtype.itemsize
            if self.rolling_split:
                from nanovllm.models.gemma2 import gemma2_layer_types
                types = gemma2_layer_types(hf, hf.num_hidden_layers)
                layer_counts = {
                    "ring": sum(t == "sliding" for t in types),
                    "full": sum(t != "sliding" for t in types),
                }
            else:
                layer_counts = {"main": self._swap_layers}
            self._swap_pool_layers = layer_counts
            self._swap_pool_bytes_per_block = {}
            for pool, layers in layer_counts.items():
                if self._swap_mla:
                    block_bytes = (layers * self.block_size
                                   * self._swap_mla_d * itemsize)
                else:
                    block_bytes = (2 * layers * self.block_size
                                   * self._swap_kv_heads * self._swap_head_dim
                                   * itemsize)
                self._swap_pool_bytes_per_block[pool] = block_bytes
            # Retain the aggregate for callers that only need an upper-bound estimate.
            self._swap_bytes_per_block = sum(self._swap_pool_bytes_per_block.values())
        # ---- 投机解码（n-gram / Medusa） ----
        self.spec_decode = config.speculative in ("ngram", "medusa", "eagle")
        self.spec_mode = config.speculative   # "ngram" | "medusa" | "eagle"
        self.ngram_window = config.ngram_window
        self.ngram_min_window = config.ngram_min_window
        self.max_draft_len = config.max_draft_len

    def is_finished(self):
        return not self.waiting and not self.running and not self.swapped

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def reset_metrics(self) -> None:
        self.num_preemptions = 0
        self.num_swaps = 0
        self.num_recompute_preemptions = 0
        self.recompute_tokens = 0
        self.estimated_recompute_seconds = 0.0
        self.estimated_swap_seconds = 0.0
        self.num_affinity_probes = 0
        self.num_prefix_feature_parses = 0
        self.num_prefix_feature_reuses = 0
        self.prefix_cache_hit_tokens = 0
        self.prefix_cache_hit_requests = 0
        self.block_manager.reset_metrics()
        if self.full_block_manager is not None:
            self.full_block_manager.reset_metrics()
        self.num_aging_promotions = 0
        self.num_slo_prefill_steps = 0
        self.slo_prefill_budget_sum = 0
        self.slo_prefill_budget_max = 0
        self.slo_prefill_rows_sum = 0
        self.num_tpot_prefill_steps = 0
        self.tpot_prefill_budget_sum = 0
        self.num_tpot_priority_steps = 0
        self.slo_prefill_budget_scale_sum = 0.0
        self.slo_prefill_budget_scale_min = 1.0
        self._last_prefill_budget_scale = 1.0
        self.num_tpot_starved_steps = 0
        self._last_tpot_pressure = 0.0

    def observe_prefill(self, num_tokens: int, elapsed_seconds: float) -> None:
        """Update the recompute cost estimate from observed pure-prefill steps."""
        if num_tokens <= 0 or elapsed_seconds <= 0:
            return
        observed = elapsed_seconds / num_tokens
        self._prefill_seconds_per_token = (
            0.8 * self._prefill_seconds_per_token + 0.2 * observed)

    def observe_kv_transfer(self, num_bytes: int, elapsed_seconds: float) -> None:
        """Update the swap bandwidth estimate from completed D2H/H2D transfers."""
        if num_bytes <= 0 or elapsed_seconds <= 0:
            return
        observed = elapsed_seconds / num_bytes
        self._swap_seconds_per_byte = 0.8 * self._swap_seconds_per_byte + 0.2 * observed

    def _estimated_prefill_tokens(self, seq: Sequence) -> int:
        cached_tokens = seq.num_cached_tokens
        if self.cache_affinity_admission:
            features, parsed = self.block_manager.resolve_prefix_features(seq)
            if parsed:
                self.num_prefix_feature_parses += 1
            else:
                self.num_prefix_feature_reuses += 1
            cached_tokens = max(cached_tokens, features.cached_tokens)
        return max(1, seq.num_tokens - min(cached_tokens, seq.num_tokens))

    def _estimated_ttft_slack(self, seq: Sequence, now: float) -> float | None:
        target = (seq.ttft_slo_ms / 1000.0 if seq.ttft_slo_ms is not None
                  else self.default_ttft_slo_s)
        if target is None:
            return None
        submitted = seq.t_submitted if seq.t_submitted is not None else now
        estimated_prefill_seconds = (
            self._estimated_prefill_tokens(seq) * self._prefill_seconds_per_token)
        return target - max(0.0, now - submitted) - estimated_prefill_seconds

    def _tpot_target_ms(self, seq: Sequence) -> float | None:
        return (seq.tpot_slo_ms if seq.tpot_slo_ms is not None
                else self.default_tpot_slo_ms)

    def _has_active_tpot_target(self) -> bool:
        return (self.tpot_aware_scheduling
                and (self.external_tpot_target_active
                     or any(self._tpot_target_ms(seq) is not None for seq in self.running)))

    def _estimated_tpot_slack_ms(self, seq: Sequence) -> float | None:
        target = self._tpot_target_ms(seq)
        if target is None:
            return None
        observed = (seq.tpot_ewma_ms if seq.tpot_ewma_ms is not None
                    else self.tpot_decode_ms_fallback)
        return target - observed

    def _tpot_decode_pressure(self) -> float:
        """How hard decode work should throttle prefill, in [0, 1].

        ``slack = target − observed``, so the raw ratio ``1 − slack/target`` is
        ``observed/target``: it reaches 1.0 as soon as the observed interval merely
        exceeds the target, and every larger overshoot saturates there too. That
        cliff is what starves prefill — the quota collapses to
        ``prefill_reserve_tokens`` regardless of *how far* off the target is, and the
        resulting 256-token prefill fragments then keep measured TPOT high, which
        keeps the pressure pinned. Measured on the RTX 5060 Ti with 128 requests at
        16 req/s: budget scale 0.001 (min 0.000), prefill quota exactly 256 tokens
        over 355 steps, TTFT p50 43.7 s, and the same result for TPOT targets of
        10/20/40/80/160 ms once observed TPOT (~85 ms) exceeded them.

        ``tpot_prefill_throttle_margin`` adds a deadband: within
        ``margin × target`` of the target the throttle stays off, and beyond it the
        throttle ramps linearly with the overshoot instead of clipping at 1.0. A
        target of 0 keeps the legacy cliff behaviour for comparison.

        ``tpot_throttle_max_waiting`` additionally disables the throttle once the
        waiting queue is at least that deep. The throttle's own justification is
        "decode is the bottleneck", but at depth the bottleneck is prefill
        throughput — starving prefill keeps every late request without a first
        token while the already-running requests keep missing their target, so the
        signal reinforces itself. Measured at 128 requests / 16 req/s with a 20 ms
        target: throttle pinned the budget at 256 tokens for 355 steps and left
        TTFT p50 at 43.7 s, and a target of 10/20/40/80/160 ms all behaved the same.
        """
        if not self.tpot_aware_scheduling:
            return 0.0
        if 0 < self.tpot_throttle_max_waiting <= len(self.waiting):
            return 0.0
        pressure = self.external_tpot_pressure
        margin = max(0.0, self.tpot_prefill_throttle_margin)
        for seq in self.running:
            target = self._tpot_target_ms(seq)
            if target is None or target <= 0:
                continue
            observed = (seq.tpot_ewma_ms if seq.tpot_ewma_ms is not None
                        else self.tpot_decode_ms_fallback)
            if margin > 0:
                deadband = margin * target
                overshoot = observed - target - deadband
                ratio = (0.0 if overshoot <= 0
                         else overshoot / (target + deadband))
            else:
                ratio = observed / target
            pressure = max(pressure, min(1.0, max(0.0, ratio)))
        return pressure

    def _order_running_by_tpot(self) -> None:
        """Prioritize active requests with the least remaining TPOT slack."""
        if not self.tpot_aware_scheduling or len(self.running) < 2:
            return
        items = list(self.running)
        if not any(self._tpot_target_ms(seq) is not None for seq in items):
            return
        ranked = sorted(
            enumerate(items),
            key=lambda pair: (
                0 if self._tpot_target_ms(pair[1]) is not None else 1,
                float("inf") if self._estimated_tpot_slack_ms(pair[1]) is None
                else self._estimated_tpot_slack_ms(pair[1]),
                pair[0],
            ),
        )
        self.running = deque(seq for _, seq in ranked)

    def _slo_prefill_controls(self, now: float | None = None) -> tuple[int, int, float]:
        """Choose a per-step prefill token quota and row quota from SLO pressure."""
        self._last_tpot_pressure = 0.0
        if not self.waiting:
            return self.max_num_batched_tokens, 0, 0.0
        if not (self.slo_aware_scheduling or self._has_active_tpot_target()):
            return self.max_num_batched_tokens, self.max_num_seqs, 0.0

        now = perf_counter() if now is None else now
        if self.slo_aware_scheduling or self._has_active_tpot_target():
            depth_target = self.queue_depth_for_full_prefill
            if depth_target <= 1:
                queue_pressure = 1.0
            else:
                queue_pressure = min(1.0, max(0.0, (len(self.waiting) - 1)
                                               / (depth_target - 1)))
        else:
            queue_pressure = 0.0

        items = list(self.waiting)
        urgency = 0.0
        if self.slo_aware_scheduling:
            for seq in items[:self.admission_window]:
                slack = self._estimated_ttft_slack(seq, now)
                target = (seq.ttft_slo_ms / 1000.0 if seq.ttft_slo_ms is not None
                          else self.default_ttft_slo_s)
                if slack is not None and target is not None:
                    urgency = max(urgency, min(1.0, max(0.0, 1.0 - slack / target)))

        prefill_pressure = max(queue_pressure, urgency)
        tpot_pressure = self._tpot_decode_pressure()
        self._last_tpot_pressure = tpot_pressure
        max_budget = min(self.max_num_batched_tokens, self.max_prefill_chunk_tokens)
        min_budget = min(self.prefill_reserve_tokens, max_budget)
        # TPOT 压力对 prefill 配额的收缩是"乘法"的：pressure→1 时配额直接落到
        # min_budget（prefill_reserve_tokens）。目标不可达时（如 20ms 目标 vs 实测
        # 170ms）这会把 prefill 切成碎片步——饥饿就是这么来的。下面记录收缩因子，
        # 供消融报告直接观测，不必再从吞吐反推。
        budget_scale = prefill_pressure * (1.0 - tpot_pressure)
        self._last_prefill_budget_scale = budget_scale
        self.slo_prefill_budget_scale_sum += budget_scale
        self.slo_prefill_budget_scale_min = min(self.slo_prefill_budget_scale_min,
                                                budget_scale)
        if tpot_pressure >= 0.5:
            self.num_tpot_starved_steps += 1
        token_budget = round(min_budget + budget_scale * (max_budget - min_budget))
        if self.max_num_seqs <= 1:
            prefill_rows = 1
        else:
            row_capacity = self.max_num_seqs - 1
            prefill_rows = min(len(self.waiting), max(1, ceil(row_capacity * budget_scale)))
        return max(1, token_budget), prefill_rows, prefill_pressure

    def _record_slo_controls(self, token_budget: int, prefill_rows: int) -> None:
        if not (self.slo_aware_scheduling or self._has_active_tpot_target()) or not self.waiting:
            return
        budget_scale = getattr(self, "_last_prefill_budget_scale", 1.0)
        self.slo_prefill_budget_scale_sum += budget_scale
        self.slo_prefill_budget_scale_min = min(self.slo_prefill_budget_scale_min,
                                                budget_scale)
        if self._last_tpot_pressure >= 0.5:
            self.num_tpot_starved_steps += 1
        if self.slo_aware_scheduling:
            self.num_slo_prefill_steps += 1
            self.slo_prefill_budget_sum += token_budget
            self.slo_prefill_budget_max = max(self.slo_prefill_budget_max, token_budget)
            self.slo_prefill_rows_sum += prefill_rows
        if self._has_active_tpot_target():
            self.num_tpot_prefill_steps += 1
            self.tpot_prefill_budget_sum += token_budget
            self.num_tpot_priority_steps += int(self._last_tpot_pressure >= 0.5)

    def _waiting_score(self, seq: Sequence, now: float, original_index: int) -> tuple:
        submitted = seq.t_submitted if seq.t_submitted is not None else now
        age = max(0.0, now - submitted)
        if self.aging_fairness and age >= self.aging_timeout_s:
            # Once aged, old requests outrank cache/short-prompt preference.
            return (0, submitted, original_index)

        cached_tokens = seq.num_cached_tokens
        if self.cache_affinity_admission:
            features, parsed = self.block_manager.resolve_prefix_features(seq)
            if parsed:
                self.num_prefix_feature_parses += 1
            else:
                self.num_prefix_feature_reuses += 1
            cached_tokens = max(cached_tokens, features.cached_tokens)
        cached_tokens = min(cached_tokens, seq.num_tokens)
        remaining_tokens = max(0, seq.num_tokens - cached_tokens)
        slack = self._estimated_ttft_slack(seq, now)
        if self.slo_aware_scheduling or self._has_active_tpot_target():
            # Least laxity first within Top-W; cache reuse and prompt cost break ties.
            return (1, float("inf") if slack is None else slack,
                    -cached_tokens, remaining_tokens,
                    submitted, original_index)
        if self.cache_affinity_admission and self.latency_aware_scheduling:
            return (1, -cached_tokens, remaining_tokens, submitted, original_index)
        if self.cache_affinity_admission:
            return (1, -cached_tokens, submitted, original_index)
        if self.latency_aware_scheduling:
            return (1, remaining_tokens, submitted, original_index)
        return (1, submitted, original_index)

    def _order_waiting(self) -> None:
        """Re-rank only the Top-W admission window; preserve FIFO beyond it."""
        if len(self.waiting) < 2:
            return
        if not (self.latency_aware_scheduling or self.cache_affinity_admission
                or self.aging_fairness or self.slo_aware_scheduling):
            return
        items = list(self.waiting)
        now = perf_counter()
        if self.aging_fairness:
            aged = [(index, seq) for index, seq in enumerate(items)
                    if seq.t_submitted is not None
                    and now - seq.t_submitted >= self.aging_timeout_s]
            if aged:
                original_index, oldest = min(
                    aged, key=lambda pair: (pair[1].t_submitted, pair[0]))
                rest = items[:original_index] + items[original_index + 1:]
                width = min(max(0, self.admission_window - 1), len(rest))
                window = rest[:width]
                ranked = sorted(
                    enumerate(window),
                    key=lambda pair: self._waiting_score(pair[1], now, pair[0]),
                )
                ordered = [oldest] + [seq for _, seq in ranked] + rest[width:]
                if original_index > 0 and not oldest.age_promoted:
                    oldest.age_promoted = True
                    self.num_aging_promotions += 1
                self.waiting = deque(ordered)
                return

        width = min(self.admission_window, len(items))
        window = items[:width]
        ranked = sorted(
            enumerate(window),
            key=lambda pair: self._waiting_score(pair[1], now, pair[0]),
        )
        ordered = [seq for _, seq in ranked]
        if ordered and ordered[0] is not window[0]:
            first = ordered[0]
            submitted = first.t_submitted if first.t_submitted is not None else now
            if (self.aging_fairness and not first.age_promoted
                    and now - submitted >= self.aging_timeout_s):
                first.age_promoted = True
                self.num_aging_promotions += 1
        self.waiting = deque(ordered + items[width:])

    def _prefill_reserve(self) -> int:
        if not self.waiting or not self.latency_aware_scheduling:
            return 0
        if self.max_num_batched_tokens <= 1:
            return 0
        seq = self.waiting[0]
        cached_tokens = seq.num_cached_tokens
        if self.cache_affinity_admission:
            features, parsed = self.block_manager.resolve_prefix_features(seq)
            if parsed:
                self.num_prefix_feature_parses += 1
            else:
                self.num_prefix_feature_reuses += 1
            cached_tokens = max(cached_tokens, features.cached_tokens)
        remaining = max(1, seq.num_tokens - min(cached_tokens, seq.num_tokens))
        return min(self.prefill_reserve_tokens, remaining,
                   self.max_num_batched_tokens - 1)

    def _decode_row_limit(self, slo_controls: tuple[int, int, float] | None = None) -> int:
        if not self.waiting:
            return self.max_num_seqs
        if self.slo_aware_scheduling or self._has_active_tpot_target():
            token_budget, prefill_rows, _ = (slo_controls or
                                               self._slo_prefill_controls())
            if self.max_num_batched_tokens <= 1 or self.max_num_seqs == 1:
                if self._single_slot_prefill_turn:
                    self._single_slot_prefill_turn = False
                    return 0
                self._single_slot_prefill_turn = True
                return min(1, self.max_num_seqs)
            return min(self.max_num_seqs - prefill_rows,
                       max(0, self.max_num_batched_tokens - token_budget))
        if not self.latency_aware_scheduling:
            return self.max_num_seqs
        if self.max_num_batched_tokens <= 1 or self.max_num_seqs == 1:
            if self._single_slot_prefill_turn:
                self._single_slot_prefill_turn = False
                return 0
            self._single_slot_prefill_turn = True
            return min(1, self.max_num_seqs)
        reserve = self._prefill_reserve()
        token_limit = max(0, self.max_num_batched_tokens - reserve)
        return min(self.max_num_seqs - 1, token_limit)

    def _mixed_prefill_row_limit(
        self, slo_controls: tuple[int, int, float], decode_limit: int,
    ) -> int:
        if not (self.slo_aware_scheduling or self._has_active_tpot_target()):
            return self.max_num_seqs
        if self.max_num_seqs == 1:
            return max(0, self.max_num_seqs - decode_limit)
        return slo_controls[1]

    def mark_prefill_started(self, seq: Sequence) -> None:
        """Record admission start once, preserving request-arrival timestamps."""
        now = perf_counter()
        if seq.t_prefill_started is None:
            seq.t_prefill_started = now

    def schedule(self) -> tuple[list[Sequence], str]:
        """返回 (被调度序列, kind)；kind ∈ {"prefill", "decode", "mixed", "spec"}。

        waiting与running都非空时返回混合批次（decode行在后，prefill行在前）：
        早完成prefill的请求立即开始decode，不用等全部prefill跑完
        （vLLM V1同款策略；此前"先prefill后decode"会让早完成者空等数秒，
        见INTERVIEW.md §10.3.2）。decode与prefill共享max_num_batched_tokens预算。

        投机解码（spec_decode）时：先给所有running序列算n-gram草稿；只要有任一
        草稿非空，running行全部变为verify行（γ=0的行退化为1-token varlen行），
        kind = "spec"（无waiting）或 mixed（有waiting，prefill行在前）。全部无
        草稿时回落纯decode（CUDA graph路径，不损失）。
        """
        self.cow_pairs = []
        self.swap_pairs = []
        self._order_waiting()
        # KV swap 换入优先：把 KV 已换出到 CPU 的序列换回 GPU（free块足够时），
        # 换入后直接参与本步 decode（KV 完整，无需重新 prefill）
        self._try_swap_in()
        self._order_running_by_tpot()
        if self.spec_decode:
            for seq in self.running:
                self._compute_draft(seq)
            if self.waiting and self.running:
                return self._schedule_mixed()
            if self.waiting:
                return self._schedule_prefill()
            if any(seq.draft_tokens for seq in self.running):
                return self._schedule_spec()
            for seq in self.running:
                seq.draft_tokens = None
            return self._schedule_decode()
        if self.waiting and self.running:
            return self._schedule_mixed()
        if self.waiting:
            return self._schedule_prefill()
        return self._schedule_decode()

    # ------------------------------------------------------------------
    # split（交替窗口）模式双表辅助：非 split 时 full_block_manager=None，
    # 辅助退化为原单表行为。
    # ------------------------------------------------------------------
    def _dec_can(self, seq: Sequence) -> bool:
        """decode 步可追加：环表（可驱逐腾位）且 full 表（需新块时）都有位。"""
        return self.block_manager.can_append(seq) and (
            self.full_block_manager is None
            or self.full_block_manager.can_append(seq))

    def _alloc_prefill_blocks(self, seq: Sequence) -> bool:
        """prefill 首次分配（双表同量块）：块不足返回 False（不做任何分配）。"""
        if self.full_block_manager is not None and not seq.kv_table \
                and self.full_block_manager.can_allocate(seq) == -1:
            return False
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:
            return False
        self.block_manager.allocate(seq, num_cached_blocks)
        if self.full_block_manager is not None and not seq.kv_table:
            self.full_block_manager.allocate(seq, 0)
        self.num_affinity_probes += 1
        if seq.num_prefix_cached_tokens > 0:
            self.prefix_cache_hit_requests += 1
            self.prefix_cache_hit_tokens += seq.num_prefix_cached_tokens
        return True

    def _compute_draft(self, seq: Sequence):
        """给一个running序列准备本步草稿。

        - ngram模式：每步CPU重新搜索（历史窗口）。
        - medusa模式：草稿由engine在上一轮verify后用GPU头前向算出并写回
          seq.draft_tokens——已设置的保留；未设置的（刚完成prefill、或上一轮
          是回落步）用n-gram兜底，保证第一步也能投机。

        上限 = min(最大草稿数, 剩余输出预算-1)：每步至少产出1个token（bonus/
        拒绝样本），所以草稿数最多 = remaining-1，保证追加后不超max_tokens。
        """
        if self.spec_mode in ("medusa", "eagle") and seq.draft_tokens is not None:
            return  # engine已设置（GPU头前向/草稿层自回归），保持
        remaining = seq.max_tokens - seq.num_completion_tokens - 1
        max_len = min(self.max_draft_len, remaining)
        if max_len <= 0:
            seq.draft_tokens = []
            return
        seq.draft_tokens = find_ngram_draft(seq.token_ids, self.ngram_window,
                                            self.ngram_min_window, max_len, self.eos)

    def _schedule_mixed(self) -> tuple[list[Sequence], str]:
        """混合批次：先安排decode（复用can_append/抢占逻辑），再用剩余预算做prefill。"""
        if self.spec_decode:
            return self._schedule_mixed_spec()
        # 1) decode部分（batch行序在后）
        decode_seqs = []
        slo_controls = self._slo_prefill_controls()
        decode_limit = self._decode_row_limit(slo_controls)
        prefill_row_limit = self._mixed_prefill_row_limit(
            slo_controls, decode_limit)
        self._record_slo_controls(slo_controls[0], prefill_row_limit)
        while self.running and len(decode_seqs) < decode_limit:
            seq = self.running.popleft()
            while not self._dec_can(seq):
                if self.running:
                    self._preempt_another_running()
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                if self.full_block_manager is not None:
                    self.full_block_manager.may_append(seq)
                pair = self.block_manager.cow_block(seq, seq.num_tokens - 1)
                if pair is not None:
                    self.cow_pairs.append(pair)
                decode_seqs.append(seq)
        self.running.extendleft(reversed(decode_seqs))

        # 2) prefill部分（batch行序在前），共享token预算
        prefill_seqs = []
        num_batched_tokens = len(decode_seqs)  # decode每序列1 token计入预算
        prefill_tokens_used = 0
        prefill_token_limit = (slo_controls[0]
                               if self.slo_aware_scheduling or self._has_active_tpot_target()
                               else self.max_num_batched_tokens)
        while (self.waiting and len(prefill_seqs) < prefill_row_limit
               and len(prefill_seqs) + len(decode_seqs) < self.max_num_seqs):
            seq = self.waiting[0]
            remaining = min(self.max_num_batched_tokens - num_batched_tokens,
                            prefill_token_limit - prefill_tokens_used)
            if remaining == 0:
                break
            if not seq.block_table:
                if not self._alloc_prefill_blocks(seq):
                    break
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and prefill_seqs:  # only allow chunked prefill for the first seq
                break
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            prefill_tokens_used += seq.num_scheduled_tokens
            if seq.num_scheduled_tokens > 0:
                pair = self.block_manager.cow_block(seq, seq.num_cached_tokens)
                if pair is not None:
                    self.cow_pairs.append(pair)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.mark_prefill_started(seq)
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            if num_tokens != 0:
                self.mark_prefill_started(seq)
                prefill_seqs.append(seq)

        if prefill_seqs and decode_seqs:
            return prefill_seqs + decode_seqs, "mixed"
        if prefill_seqs:
            return prefill_seqs, "prefill"
        if decode_seqs:
            return decode_seqs, "decode"
        # A full-prefix hit can enter running without contributing a prefill row.
        # Let it make decode progress rather than asserting on an empty mixed batch.
        if self.running:
            return self._schedule_decode()
        return self._schedule_prefill()

    def _spec_rows(self, max_rows: int, budget: int) -> list[Sequence]:
        """把running序列编排为verify行（草稿已由_compute_draft算好）。

        - 每行query长度 n = γ+1（含末token + 草稿），共享max_num_batched_tokens预算：
          预算不足时截断后续行的草稿（截断总是安全的，验收只验证剩余部分）；
        - can_append_spec 检查写span（可能跨块）所需的新块+COW副本，不足则抢占；
        - 写span内每个被共享的块都COW（含跨块时第二个块）。
        """
        rows = []
        used = 0
        while self.running and len(rows) < max_rows:
            seq = self.running.popleft()
            n = len(seq.draft_tokens) + 1
            avail = max(1, budget - used)
            if n > avail:
                seq.draft_tokens = seq.draft_tokens[:avail - 1]
                n = avail
            while not self.block_manager.can_append_spec(seq, n):
                if self.running:
                    self._preempt_another_running()
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = n
                seq.is_prefill = False
                self.block_manager.may_append_spec(seq, n)
                # COW 只对有共享的写块有意义；滚动块恒私有（ref=1）→ 跳过
                # （表项下标 = 逻辑块 − kv_j0，cow_block 的 write_start//B 语义
                # 只对非滚动表成立）
                if not self.block_manager.rolling:
                    start = len(seq) - 1
                    first_blk = start // self.block_size
                    last_blk = (start + n - 1) // self.block_size
                    for b in range(first_blk, last_blk + 1):
                        pair = self.block_manager.cow_block(seq, b * self.block_size)
                        if pair is not None:
                            self.cow_pairs.append(pair)
                rows.append(seq)
                used += n
        self.running.extendleft(reversed(rows))
        return rows

    def _schedule_spec(self) -> tuple[list[Sequence], str]:
        """纯verify步（无waiting）：所有running行做一次并行验证。"""
        rows = self._spec_rows(self.max_num_seqs, self.max_num_batched_tokens)
        assert rows
        return rows, "spec"

    def _schedule_mixed_spec(self) -> tuple[list[Sequence], str]:
        """投机混合步：verify行（后）+ prefill行（前），共享token预算。"""
        slo_controls = self._slo_prefill_controls()
        row_limit = self._decode_row_limit(slo_controls)
        prefill_row_limit = self._mixed_prefill_row_limit(slo_controls, row_limit)
        self._record_slo_controls(slo_controls[0], prefill_row_limit)
        if self.slo_aware_scheduling or self._has_active_tpot_target():
            prefill_token_limit = slo_controls[0]
            spec_budget = max(1, self.max_num_batched_tokens - prefill_token_limit)
        else:
            reserve = self._prefill_reserve()
            prefill_token_limit = self.max_num_batched_tokens
            spec_budget = max(1, self.max_num_batched_tokens - reserve)
        spec_rows = self._spec_rows(row_limit, spec_budget)
        prefill_seqs = []
        num_batched_tokens = sum(seq.num_scheduled_tokens for seq in spec_rows)
        prefill_tokens_used = 0
        while (self.waiting and len(prefill_seqs) < prefill_row_limit
               and len(prefill_seqs) + len(spec_rows) < self.max_num_seqs):
            seq = self.waiting[0]
            remaining = min(self.max_num_batched_tokens - num_batched_tokens,
                            prefill_token_limit - prefill_tokens_used)
            if remaining == 0:
                break
            if not seq.block_table:
                if not self._alloc_prefill_blocks(seq):
                    break
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and prefill_seqs:  # only allow chunked prefill for the first seq
                break
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            prefill_tokens_used += seq.num_scheduled_tokens
            if seq.num_scheduled_tokens > 0:
                pair = self.block_manager.cow_block(seq, seq.num_cached_tokens)
                if pair is not None:
                    self.cow_pairs.append(pair)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.mark_prefill_started(seq)
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            if num_tokens != 0:
                self.mark_prefill_started(seq)
                prefill_seqs.append(seq)

        if prefill_seqs and spec_rows:
            return prefill_seqs + spec_rows, "mixed"
        if prefill_seqs:
            return prefill_seqs, "prefill"
        if spec_rows:
            return spec_rows, "spec"
        if self.running:
            for seq in self.running:
                if seq.draft_tokens is None:
                    self._compute_draft(seq)
            return self._schedule_spec()
        return self._schedule_prefill()

    def _schedule_prefill(self) -> tuple[list[Sequence], str]:
        # 需要被调度的序列列表
        scheduled_seqs = []
        # 在prefill阶段需要处理的token数量
        num_batched_tokens = 0
        slo_controls = self._slo_prefill_controls()
        prefill_token_limit = (slo_controls[0]
                               if self.slo_aware_scheduling or self._has_active_tpot_target()
                               else self.max_num_batched_tokens)
        prefill_row_limit = (slo_controls[1]
                             if self.slo_aware_scheduling or self._has_active_tpot_target()
                             else self.max_num_seqs)
        self._record_slo_controls(slo_controls[0], prefill_row_limit)

        # prefill
        while self.waiting and len(scheduled_seqs) < prefill_row_limit:
            seq = self.waiting[0]
            remaining = min(self.max_num_batched_tokens - num_batched_tokens,
                            prefill_token_limit - num_batched_tokens)
            if remaining == 0:
                break
            # 判断当前序列是否占用KV Cache block块
            if not seq.block_table:
                # 双表分配（split 模式）；can_allocate 返回-1则无法分配，不做任何分配
                if not self._alloc_prefill_blocks(seq):
                    break
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 当可处理的token数小于需要处理的token数时，只有当前序列是第一个被调度的序列时才允许将其进行分块处理
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 部分块共享后，若本次prefill的写起点落在被共享的缓存块内（如共享了44-token的尾块
            # 且要继续写入），先把该块复制一份，避免污染其他共享者
            if seq.num_scheduled_tokens > 0:
                pair = self.block_manager.cow_block(seq, seq.num_cached_tokens)
                if pair is not None:
                    self.cow_pairs.append(pair)
            # 如果缓存的前缀token数量+当前调度的token数量等于总token数量，说明当前序列已经完成了prefill阶段，
            # 将其状态修改为RUNNING，加入decode序列当中
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.mark_prefill_started(seq)
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 当前序列prefill如果未完成，再加入调度序列中
            if num_tokens != 0:
                self.mark_prefill_started(seq)
                scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, "prefill"
        # 本步无prefill可调度（如全部full-hit）→ 回落decode（与旧行为一致）
        return self._schedule_decode()

    def _schedule_decode(self) -> tuple[list[Sequence], str]:
        # decode
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查当前分配的KV Cache块能否追加上一轮生成的、但还没写入KV Cache块的token
            # KV Cache写入逻辑是：如果当前序列的token数量%block_size==1，说明需要申请一个新的KV Cache块来存储上一轮生成的token
            # 否则直接在最后一个KV Cache块中追加即可
            while not self._dec_can(seq):
                # 如果运行队列中还有其他序列，则中断当前序列，将其放回至等待队列中，释放其占用的资源
                if self.running:
                    self._preempt_another_running()
                else: # 否则，运行队列中没有其他队列，只能将自己释放，自己回退到等待队列
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                if self.full_block_manager is not None:
                    self.full_block_manager.may_append(seq)
                # decode写入末块前，若末块是共享的部分块，先复制一块（COW）
                pair = self.block_manager.cow_block(seq, seq.num_tokens - 1)
                if pair is not None:
                    self.cow_pairs.append(pair)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, "decode"

    def _estimate_preemption_cost(self, seq: Sequence, num_swap_bytes: int
                                  ) -> tuple[int, float, float]:
        reusable_tokens = min(
            seq.num_tokens,
            self.block_manager.get_prefix_cached_tokens(seq),
        )
        recompute_tokens = max(1, seq.num_tokens - reusable_tokens)
        recompute_seconds = recompute_tokens * self._prefill_seconds_per_token
        swap_seconds = 2 * num_swap_bytes * self._swap_seconds_per_byte
        return recompute_tokens, recompute_seconds, swap_seconds

    def _preemption_plan(self, seq: Sequence) -> tuple[bool, int, float, float, int]:
        required_bytes = self._swap_required_bytes(seq) if self.kv_swap else 0
        has_complete_tables = bool(seq.block_table) and (
            self.full_block_manager is None or bool(seq.kv_table))
        can_swap = (self.kv_swap and not seq.is_prefill and has_complete_tables
                    and self._swap_bytes + required_bytes <= self._swap_max_bytes)
        recompute_tokens, recompute_cost, swap_cost = self._estimate_preemption_cost(
            seq, required_bytes)
        use_swap = bool(can_swap and (
            not self.recompute_aware_preemption or swap_cost < recompute_cost))
        managers = [(self.block_manager, seq.block_table)]
        if self.full_block_manager is not None:
            managers.append((self.full_block_manager, seq.kv_table))
        reclaimable = [
            sum(manager.blocks[block_id].ref_count == 1 for block_id in table)
            for manager, table in managers
        ]
        reclaimable_blocks = min(reclaimable, default=0)
        selected_cost = swap_cost if use_swap else recompute_cost
        return use_swap, recompute_tokens, recompute_cost, selected_cost, reclaimable_blocks

    def _swap_required_bytes(self, seq: Sequence) -> int:
        if not self.kv_swap:
            return 0
        if self.rolling_split:
            return (len(seq.block_table) * self._swap_pool_bytes_per_block["ring"]
                    + len(seq.kv_table) * self._swap_pool_bytes_per_block["full"])
        return len(seq.block_table) * self._swap_pool_bytes_per_block["main"]

    @staticmethod
    def swap_buffer_nbytes(buffer: torch.Tensor | dict[str, torch.Tensor]) -> int:
        """Return host storage bytes for a regular or split-pool swap payload."""
        buffers = buffer.values() if isinstance(buffer, dict) else (buffer,)
        return sum(tensor.numel() * tensor.element_size() for tensor in buffers)

    def _new_swap_buffer(self, num_blocks: int, pool: str) -> torch.Tensor:
        layers = self._swap_pool_layers[pool]
        if self._swap_mla:
            return torch.empty(self._swap_layers, num_blocks, self.block_size,
                               self._swap_mla_d, dtype=self._swap_dtype)
        # Split pools differ only in layer count; keeping the same six-dimensional
        # layout lets ModelRunner copy either pool with the same indexing code.
        return torch.empty(2, layers, num_blocks, self.block_size,
                           self._swap_kv_heads, self._swap_head_dim,
                           dtype=self._swap_dtype)

    def _swap_in_required_blocks(self, seq: Sequence, manager: BlockManager) -> int:
        cached_blocks = (seq.num_cached_tokens + self.block_size - 1) // self.block_size
        if manager.rolling:
            cached_blocks = max(1, cached_blocks - seq.kv_j0)
        needs_append = len(seq) % self.block_size == 1
        if needs_append and not (manager.rolling and
                                 manager._front_dead(seq, len(seq) + 1)):
            cached_blocks += 1
        return cached_blocks

    def _choose_preemption_victim(self, candidates) -> Sequence:
        if not self.recompute_aware_preemption:
            return candidates[-1]

        plans = {seq: self._preemption_plan(seq) for seq in candidates}
        useful = [seq for seq in candidates if plans[seq][4] > 0]
        if useful:
            candidates = useful

        def score(seq: Sequence) -> tuple[float, float, float]:
            _, _, _, cost, reclaimable = plans[seq]
            retry_penalty = 1 + seq.preemption_count
            adjusted_cost = cost * retry_penalty
            unit_cost = adjusted_cost / max(1, reclaimable)
            submitted = seq.t_submitted if seq.t_submitted is not None else perf_counter()
            return unit_cost, adjusted_cost, -submitted

        return min(candidates, key=score)

    def _preempt_another_running(self) -> None:
        victim = self._choose_preemption_victim(list(self.running))
        self.running.remove(victim)
        self.preempt(victim)

    def preempt(self, seq: Sequence):
        """抢占：KV 块不足时中断序列。

        - kv_swap 开启且序列 KV 完整（decode/spec 序列）→ **swap_out**：KV 拷到
          CPU、释放 GPU 块；恢复时直接换回（bit-exact，免重新 prefill）。
        - 否则（prefill 中途 / swap 关闭）→ **recompute**：释放块、回 waiting，
        恢复时按前缀缓存重新 prefill（块哈希命中部分免算）。
        """
        self.num_preemptions += 1
        seq.preemption_count += 1
        use_swap, recompute_tokens, recompute_cost, swap_cost, _ = \
            self._preemption_plan(seq)
        if use_swap:
            self.estimated_swap_seconds += swap_cost
            self.swap_out(seq)
        else:
            self.num_recompute_preemptions += 1
            self.recompute_tokens += recompute_tokens
            self.estimated_recompute_seconds += recompute_cost
            seq.status = SequenceStatus.WAITING
            seq.is_prefill = True
            seq.draft_tokens = None  # 回waiting的序列下次以prefill行重新调度，草稿作废
            seq.swapped = False
            self.block_manager.deallocate(seq)
            if self.full_block_manager is not None and seq.kv_table:
                self.full_block_manager.deallocate(seq)
            self.waiting.appendleft(seq)

    def swap_out(self, seq: Sequence):
        """KV swap 换出（记账）：记录待拷贝的块与 CPU 缓冲，**不立即释放块**。

        块保持占用直到 engine 完成 GPU→CPU 拷贝（swap_pairs 机制，同 COW）——
        否则本步后续调度可能从 free 池重分配该块、覆盖内容，拷贝读到脏数据。
        engine 拷贝后调用 finish_swap_out 释放块。
        """
        self.num_swaps += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = False       # 恢复时直接 decode，不是 prefill
        seq.draft_tokens = None
        seq.swapped = True
        # 注意：decode 序列最后 token 的块可能未分配（can_append 失败正是缺这块）
        # → 只换出已有块；恢复后本步 decode 正常追加。普通模式只有一个 KV 池；
        # Gemma-2 split rolling 模式需要同时快照 local 环池和 global full 池。
        if self.rolling_split:
            block_ids = {
                "ring": list(seq.block_table),
                "full": list(seq.kv_table),
            }
            buffers = {
                "ring": self._new_swap_buffer(len(block_ids["ring"]), "ring"),
                "full": self._new_swap_buffer(len(block_ids["full"]), "full"),
            }
            assert block_ids["ring"] and block_ids["full"]
        else:
            block_ids = list(seq.block_table)
            assert block_ids
            buffers = self._new_swap_buffer(len(block_ids), "main")
        actual_bytes = self.swap_buffer_nbytes(buffers)
        assert actual_bytes == self._swap_required_bytes(seq)
        self._swap_bytes += actual_bytes
        self._swap_buffers[seq.seq_id] = buffers
        self.swap_pairs.append((seq, block_ids, buffers, "out"))
        self.swapped.appendleft(seq)

    def finish_swap_out(self, seq: Sequence,
                        block_ids: list[int] | dict[str, list[int]]) -> None:
        """engine 完成 GPU→CPU 拷贝后：释放块、清块表（num_cached_tokens 保留=num_tokens）。"""
        if self.rolling_split:
            self.block_manager.release_blocks(block_ids["ring"])
            self.full_block_manager.release_blocks(block_ids["full"])
            seq.kv_table.clear()
        else:
            self.block_manager.release_blocks(block_ids)
        seq.block_table.clear()

    def swap_in(self, seq: Sequence):
        """KV swap 换入：重新分配私有 GPU 块，KV 从 CPU 拷回（bit-exact），直接 decode。"""
        seq.status = SequenceStatus.RUNNING
        seq.swapped = False
        # 拷贝完成前仍计入 host buffer 预算，避免同一步骤里先换入再换出
        # 导致 CPU 缓冲峰值超过 kv_swap_space_gb。
        buf = self._swap_buffers[seq.seq_id]
        self.block_manager.allocate_private(seq)  # 全新私有块（num_cached_tokens 保留）
        if self.rolling_split:
            self.full_block_manager.allocate_private(seq)
            block_ids = {"ring": list(seq.block_table), "full": list(seq.kv_table)}
        else:
            block_ids = list(seq.block_table)
        self.swap_pairs.append((seq, block_ids, buf, "in"))
        self.running.appendleft(seq)

    def finish_swap_in(self, seq: Sequence):
        """engine 完成 CPU→GPU 拷贝后释放 host buffer 并更新预算。"""
        buf = self._swap_buffers.pop(seq.seq_id)
        self._swap_bytes -= self.swap_buffer_nbytes(buf)

    def _try_swap_in(self):
        """把 swapped 队列里 KV 足够的序列换回 GPU（free 块够一个换一个）。

        预留 1 块给换入后的首个 decode 追加（can_append 在块边界需新块），
        否则 swap_in→can_append 失败→又 swap_out 的死循环。
        """
        if not self.swapped:
            return
        remaining = deque()
        while self.swapped:
            seq = self.swapped.popleft()
            ring_need = self._swap_in_required_blocks(seq, self.block_manager)
            can_swap_in = len(self.block_manager.free_block_ids) >= ring_need
            if self.full_block_manager is not None:
                full_need = self._swap_in_required_blocks(seq, self.full_block_manager)
                can_swap_in = (can_swap_in and
                               len(self.full_block_manager.free_block_ids) >= full_need)
            if can_swap_in:
                self.swap_in(seq)
            else:
                remaining.append(seq)
        self.swapped = remaining

    def _maybe_finish(self, seq: Sequence, token_id: int):
        # 判断当前序列是否满足结束条件
        # 注意 >= 而非 ==：投机步一次可接受多个token，completion数可能跳过max_tokens
        # （如 62→65 跳过 64）——精确相等会让序列永不结束、一路长到max_model_len
        # （实测 EAGLE 草稿无上限时序列长到 4093 → 块表17列溢出 spec graph 的16列）
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            seq.t_completed = perf_counter()
            self.block_manager.deallocate(seq, deferred=True)
            if self.full_block_manager is not None and seq.kv_table:
                self.full_block_manager.deallocate(seq, deferred=True)
            self.running.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        # 混合批次里prefill与decode序列并存：按各序列的is_prefill（调度器设置）分支
        try:
            for seq, token_id in zip(seqs, token_ids):
                self.block_manager.hash_blocks(seq, seq.is_prefill)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                    continue
                seq.record_output_timing()
                seq.append_token(token_id)
                self._maybe_finish(seq, token_id)
        finally:
            self._flush_deferred_frees()

    def _flush_deferred_frees(self) -> None:
        self.block_manager.flush_deferred_free()
        if self.full_block_manager is not None:
            self.full_block_manager.flush_deferred_free()

    def postprocess_spec(self, seqs: list[Sequence], token_lists: list[list[int]]):
        """投机步后处理：verify行按已接受token数更新缓存与哈希；prefill行同原逻辑。

        KV提交语义：verify写span [len-1, len-1+num_scheduled) 含被拒草稿的槽位，
        不回滚（下一步覆盖即可），只截断逻辑长度；前缀缓存哈希只发布到接受长度
        （[num_tokens-n_acc-1, num_tokens-1)，追加后调用）——被拒token永不进哈希。
        """
        try:
            for seq, tokens in zip(seqs, token_lists):
                if seq.draft_tokens is not None:
                    n_acc = len(tokens)
                    seq.record_output_timing(len(tokens))
                    seq.append_tokens(tokens)
                    self.block_manager.hash_blocks(seq, False,
                                                   start=seq.num_tokens - n_acc - 1,
                                                   end=seq.num_tokens - 1)
                    seq.num_cached_tokens = seq.num_tokens
                    seq.num_scheduled_tokens = 0
                    seq.draft_tokens = None
                    self._maybe_finish(seq, tokens[-1])
                else:
                    self.block_manager.hash_blocks(seq, seq.is_prefill)
                    seq.num_cached_tokens += seq.num_scheduled_tokens
                    seq.num_scheduled_tokens = 0
                    # 如果在prefill阶段，缓存的token数量小于总逻辑长度，说明序列的prefill阶段还没有结束
                    # 走到这步说明序列被分块处理了
                    if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                        continue
                    seq.record_output_timing()
                    seq.append_token(tokens[0])
                    self._maybe_finish(seq, tokens[0])
        finally:
            self._flush_deferred_frees()

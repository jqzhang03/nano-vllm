import atexit
import logging
import threading
from math import isfinite
from copy import copy
from dataclasses import fields
from time import perf_counter
from typing import Callable
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        requested_mode = config.execution_mode
        cuda_available = torch.cuda.is_available()
        n_devices = torch.cuda.device_count() if cuda_available else 0
        pd_incompatibility = config.pd_mode_incompatibility()
        if requested_mode == "auto":
            if n_devices <= 1:
                execution_mode = "mixed"
                reason = "one or fewer visible CUDA devices"
            elif pd_incompatibility:
                execution_mode = "mixed"
                reason = pd_incompatibility
            elif min(config.prefill_device, config.decode_device) < 0 \
                    or max(config.prefill_device, config.decode_device) >= n_devices \
                    or config.prefill_device == config.decode_device:
                execution_mode = "mixed"
                reason = "configured PD devices are not a valid distinct visible pair"
            else:
                execution_mode = "pd"
                reason = f"{n_devices} visible CUDA devices and PD-compatible settings"
            logger.info("execution_mode=auto selected %s: %s", execution_mode, reason)
        else:
            execution_mode = requested_mode
        if execution_mode == "pd":
            if not cuda_available:
                raise RuntimeError("PD separation requires CUDA")
            if n_devices < 2:
                raise ValueError(
                    f"PD separation requires two visible CUDA devices, found {n_devices}")
            if pd_incompatibility:
                raise ValueError(pd_incompatibility)
            if min(config.prefill_device, config.decode_device) < 0 \
                    or max(config.prefill_device, config.decode_device) >= n_devices:
                raise ValueError(
                    f"PD separation requested CUDA devices {config.prefill_device} and "
                    f"{config.decode_device}, but only {n_devices} device(s) are visible")
            if config.prefill_device == config.decode_device:
                raise ValueError("PD separation requires distinct prefill/decode devices")
        config.execution_mode = execution_mode
        config.pd_separation = execution_mode == "pd"
        self.config = config
        self.execution_mode = execution_mode
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self._pd = execution_mode == "pd"
        if self._pd:
            # Each role owns an independent model and KV pool. Copy Config because
            # ModelRunner writes the per-device KV block count back into it.
            self.prefill_config = copy(config)
            self.prefill_config.kv_swap = False
            self.prefill_config.enforce_eager = True
            self.decode_config = copy(config)
            try:
                self.prefill_runner = ModelRunner(
                    self.prefill_config, 0, [], device=config.prefill_device)
                self.decode_runner = ModelRunner(
                    self.decode_config, 0, [], device=config.decode_device)
            except Exception:
                if hasattr(self, "prefill_runner"):
                    self.prefill_runner.call("exit")
                raise
            self.model_runner = self.decode_runner
        else:
            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        if self._pd:
            self.prefill_config.eos = config.eos
            self.decode_config.eos = config.eos
            self.prefill_scheduler = Scheduler(self.prefill_config)
            self.decode_scheduler = Scheduler(self.decode_config)
            # Keep the familiar attribute for metrics and downstream inspection.
            self.scheduler = self.decode_scheduler
        else:
            self.scheduler = Scheduler(config)
        # 基准计时数据：已结束请求的per-request时间戳快照 + 逐step聚合统计
        self._req_metrics: list[dict] = []
        # 已落盘的 seq_id（per-request 记录幂等）：burst 收尾与外层批次可能同时
        # 报同一条序列，重复记录会让驱动侧 completed 计数虚高、提前结束跑批。
        self._recorded_seq_ids: set[int] = set()
        self._step_stats: dict[str, float | int] = self._empty_step_stats()
        self._collect_logits = False
        self._collect_decode_logits_only = False
        self.collected_logits = []
        self._last_step_tokens: dict[int, list[int]] = {}
        self._last_step_prefix_hits: dict[int, int] = {}
        self._last_step_decode_iterations = 0
        self._last_step_multistep_tokens = 0
        self._last_step_decode_burst_yielded = False
        self._decode_burst_yield_signal = threading.Event()
        self._decode_burst_yield_callback: Callable[[], bool] | None = None
        atexit.register(self.exit)

    @staticmethod
    def _empty_step_stats() -> dict[str, float | int]:
        return dict(prefill_steps=0, decode_steps=0, prefill_tokens=0, decode_tokens=0,
                    prefill_time=0.0, decode_time=0.0, decode_iterations=0,
                    multi_step_decode_steps=0, multi_step_decode_tokens=0,
                    decode_bursts=0, decode_burst_rounds=0, decode_burst_yields=0,
                    decode_burst_skipped_slots=0, decode_burst_pressure_yields=0,
                    spec_steps=0, spec_rows=0,
                    spec_verify_tokens=0, spec_draft_tokens=0, spec_accepted_drafts=0)

    def _record_once(self, seq: Sequence) -> None:
        """Append this sequence's timing snapshot exactly once.

        A sequence that finishes inside a multi-step decode burst is reported both
        by ``_run_decode_burst`` (it collects each round's completions) and by the
        outer batch list, so an unguarded append produces duplicate ``seq_id``
        records. Benchmarks derive their completion count from
        ``collect_metrics()["per_request"]``, so duplicates made a 128-request run
        stop after 95 distinct sequences while reporting 133 rows.
        """
        if seq.seq_id in self._recorded_seq_ids:
            return
        self._recorded_seq_ids.add(seq.seq_id)
        self._req_metrics.append({
            "seq_id": seq.seq_id,
            "prompt_tokens": seq.num_prompt_tokens,
            "completion_tokens": len(seq.completion_token_ids),
            "t_submitted": seq.t_submitted,
            "t_prefill_started": seq.t_prefill_started,
            "t_first_token": seq.t_first_token,
            "t_completed": seq.t_completed,
            "prefix_cached_tokens": seq.num_prefix_cached_tokens,
            "ttft_slo_ms": seq.ttft_slo_ms,
            "ttft_slo_met": self._ttft_slo_met(seq),
            "tpot_slo_ms": seq.tpot_slo_ms,
            "tpot_slo_met": self._tpot_slo_met(seq),
        })

    def reset_benchmark_metrics(self) -> None:
        """Drop per-request/step timing so a benchmark run starts from a clean slate.

        Benchmarks used to assign ``engine._req_metrics = []`` directly, which left the
        per-seq dedup set populated and silently dropped the next run's records.
        """
        self._req_metrics = []
        self._recorded_seq_ids = set()
        self._step_stats = self._empty_step_stats()
        for scheduler in ((self.prefill_scheduler, self.decode_scheduler) if self._pd
                          else (self.scheduler,)):
            scheduler.reset_metrics()

    def exit(self):
        # 幂等：显式调用与atexit可能都触发，且同一进程可能先后创建多个引擎（如精度对比）
        if getattr(self, "_exited", False):
            return
        self._exited = True
        if self._pd:
            self.decode_runner.call("exit")
            self.prefill_runner.call("exit")
            del self.model_runner, self.decode_runner, self.prefill_runner
        else:
            self.model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,
                    submitted_at: float | None = None,
                    ttft_slo_ms: float | None = None,
                    tpot_slo_ms: float | None = None) -> int:
        # isinstance(a, b)：检查a是不是b类型的对象
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if ttft_slo_ms is not None and (not isfinite(ttft_slo_ms) or ttft_slo_ms <= 0):
            raise ValueError("ttft_slo_ms must be a finite positive number")
        if tpot_slo_ms is not None and (not isfinite(tpot_slo_ms) or tpot_slo_ms <= 0):
            raise ValueError("tpot_slo_ms must be a finite positive number")
        seq = Sequence(prompt, sampling_params)
        seq.ttft_slo_ms = (self.config.default_ttft_slo_ms
                            if ttft_slo_ms is None else ttft_slo_ms)
        seq.tpot_slo_ms = (self.config.default_tpot_slo_ms
                            if tpot_slo_ms is None else tpot_slo_ms)
        schedulers = ((self.prefill_scheduler, self.decode_scheduler) if self._pd
                      else (self.scheduler,))
        prompt_blocks = seq.num_blocks
        capacity = min(
            len(scheduler.block_manager.blocks)
            for scheduler in schedulers
        )
        for scheduler in schedulers:
            if scheduler.full_block_manager is not None:
                capacity = min(capacity, len(scheduler.full_block_manager.blocks))
        if prompt_blocks > capacity:
            raise ValueError(
                f"prompt needs {prompt_blocks} KV blocks, but the active pool can hold "
                f"at most {capacity}; reduce prompt length or increase KV cache capacity"
            )
        seq.t_submitted = perf_counter() if submitted_at is None else submitted_at
        schedulers[0].add(seq)
        return seq.seq_id

    def cancel_request(self, seq_id: int) -> bool:
        """Remove a queued/running request and release its KV blocks."""
        schedulers = ((self.prefill_scheduler, self.decode_scheduler) if self._pd
                      else (self.scheduler,))
        for scheduler in schedulers:
            for queue in (scheduler.waiting, scheduler.running, scheduler.swapped):
                seq = next((item for item in queue if item.seq_id == seq_id), None)
                if seq is None:
                    continue
                queue.remove(seq)
                if seq.swapped:
                    buf = scheduler._swap_buffers.pop(seq.seq_id, None)
                    if buf is not None:
                        scheduler._swap_bytes -= buf.numel() * buf.element_size()
                    seq.swapped = False
                if seq.block_table:
                    scheduler.block_manager.deallocate(seq)
                if scheduler.full_block_manager is not None and seq.kv_table:
                    scheduler.full_block_manager.deallocate(seq)
                seq.status = SequenceStatus.FINISHED
                seq.t_completed = perf_counter()
                return True
        return False

    def _record_step_tokens(self, seqs: list[Sequence], before: dict[int, int]) -> None:
        for seq in seqs:
            start = before.get(seq.seq_id, seq.num_completion_tokens)
            new_tokens = seq.completion_token_ids[start:]
            if new_tokens:
                self._last_step_tokens.setdefault(seq.seq_id, []).extend(new_tokens)

    def request_decode_burst_yield(self) -> None:
        """Ask the current decode burst to stop after its active forward."""
        self._decode_burst_yield_signal.set()

    def clear_decode_burst_yield(self) -> None:
        """Clear service pressure after queued requests reach the engine."""
        self._decode_burst_yield_signal.clear()

    def set_decode_burst_yield_callback(
        self, callback: Callable[[], bool] | None,
    ) -> None:
        """Install an optional arrival-pressure probe, primarily for trace replay."""
        self._decode_burst_yield_callback = callback

    def _decode_burst_pressure_active(self) -> bool:
        if self._decode_burst_yield_signal.is_set():
            return True
        callback = self._decode_burst_yield_callback
        return bool(callback is not None and callback())

    def _verify(self, seqs: list[Sequence], token_ids: list[int]):
        """投机验收：把逐行样本与草稿比对，返回每序列已接受token列表。

        logits行序 = 批次行序：普通行每seq 1行（LM head已归约）；verify行每seq
        γ+1行（第i行预测位置len-1+i → 样本samples[i]验证草稿drafts[i]；
        最后一行是全接受时的bonus）。接受语义见 nanovllm/engine/ngram.verify_drafts。

        返回 (token_lists, n_decode, n_draft, n_draft_acc, n_verify, n_acc_list)：
        n_decode = 本步产出token总数（= Σ接受数）；
        n_draft = 草稿总数；
        n_draft_acc = 被接受草稿数（α = n_draft_acc / n_draft）；
        n_verify = verify forward处理的token数（Σ γ_i+1，含末token重算）；
        n_acc_list = 每seq的接受数（与seqs对齐，Medusa draft选行用）。
        """
        from nanovllm.engine.ngram import verify_drafts
        token_lists = []
        idx = 0
        n_decode = n_draft = n_draft_acc = n_verify = 0
        n_acc_list = []
        for seq in seqs:
            if seq.draft_tokens is None:
                token_lists.append([token_ids[idx]])
                idx += 1
                n_decode += 1
                n_acc_list.append(0)  # 非verify行（prefill行）；_medusa_drafts用它识别
                continue
            drafts = seq.draft_tokens
            n = len(drafts) + 1
            samples = token_ids[idx:idx + n]
            idx += n
            accepted, n_acc = verify_drafts(drafts, samples)
            token_lists.append(accepted)
            n_decode += n_acc
            n_draft += len(drafts)
            n_draft_acc += n_acc - 1
            n_verify += n
            n_acc_list.append(n_acc)
        return token_lists, n_decode, n_draft, n_draft_acc, n_verify, n_acc_list

    def _medusa_drafts(self, seqs: list[Sequence], hidden: torch.Tensor,
                       n_acc_list: list[int], n_rows_list: list[int]):
        """从verify步的hidden批量计算下一轮Medusa draft（语义见 layers/medusa.py）。

        draft输入行 = 验收后新t_last的hidden（当前verify输入的第min(n_acc,γ_i)行：
        非全接受时t_last是第n_acc个输入token；全接受时bonus无hidden，用第γ_i行）+
        head偏移（全接受 shift=1）。EOS/剩余预算截断与n-gram一致。
        n_rows_list 必须在postprocess前捕获（postprocess会清零num_scheduled_tokens）。
        """
        heads = self.model_runner.medusa_heads
        gamma = self.config.max_draft_len
        eos = self.config.eos
        # mixed步里prefill行在hidden前面，spec组从n_prefill_tokens起
        h_offset = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        groups = {0: [], 1: []}  # shift → [(seq, hidden行号)]
        row = h_offset
        for seq, n_acc, n_rows in zip(seqs, n_acc_list, n_rows_list):
            if n_acc == 0:
                continue  # 非verify行（prefill行）
            gamma_i = n_rows - 1  # 本行实际γ（可能<全局γ）
            if n_acc <= gamma_i:
                # 非全接受：t_last（位置len+n_acc-1）是当前verify输入的第n_acc行
                idx = row + n_acc
                shift = 0
            else:
                # 全接受：t_last是bonus采样产物、无hidden → 用第γ_i行 + head偏移1
                idx = row + gamma_i
                shift = 1
            groups[shift].append((seq, idx))
            row += n_rows
        for shift, items in groups.items():
            if not items:
                continue
            h = torch.stack([hidden[i] for _, i in items])  # [n, hidden]（fp16）
            toks = [hd(h).argmax(dim=-1).tolist() for hd in heads.heads[shift:shift + gamma]]
            for i, (seq, _) in enumerate(items):
                cap = min(gamma, seq.max_tokens - seq.num_completion_tokens - 1)
                drafts = []
                for k in range(cap):
                    t = toks[k][i]
                    if t == eos:
                        break
                    drafts.append(t)
                seq.draft_tokens = drafts

    def _eagle_drafts(self, seqs: list[Sequence], hidden: torch.Tensor,
                      n_acc_list: list[int], n_rows_list: list[int]):
        """EAGLE-1 草稿：从验收后新 t_last 的 hidden 自回归生成 γ 个草稿 token。

        F(h_t, e(w_{t+1})) → h̃_{t+1} → LM_head → argmax 采样 w_{t+2}；下一步
        F(h̃_{t+1}, e(w_{t+2}))…（hidden 与 token 双自回归，草稿分布条件化于已草拟
        内容——比 Medusa 的并行头更接近目标分布，α 更高）。

        hidden 行选择与 _medusa_drafts 同语义：非全接受用第 n_acc 行（t_last 的
        feature），全接受（bonus 无 hidden）用第 γ_i 行；起始 token = seq.last_token
        （postprocess_spec 已追加）。EOS 截断与 n-gram 一致。
        按步跨 seq 批量（每步一个 [m, H] 前向 + LM head），γ 步串行。
        """
        layer = self.model_runner.eagle_layer
        gamma = self.config.max_draft_len
        eos = self.config.eos
        # mixed步里prefill行在hidden前面，spec组从n_prefill_tokens起
        h_offset = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        row = h_offset
        items = []  # (seq, hidden行)
        for seq, n_acc, n_rows in zip(seqs, n_acc_list, n_rows_list):
            if n_acc == 0:
                continue  # 非verify行（prefill行）
            gamma_i = n_rows - 1
            idx = row + (n_acc if n_acc <= gamma_i else gamma_i)
            items.append((seq, idx))
            row += n_rows
        if not items:
            return
        embed = self.model_runner.model.model.embed_tokens
        lm_head = self.model_runner.model.lm_head
        h = torch.stack([hidden[i] for _, i in items])            # [m, H]
        w = torch.tensor([s.last_token for s, _ in items], device=h.device)
        active_seqs = [s for s, _ in items]
        drafts = {id(s): [] for s, _ in items}
        for _ in range(gamma):
            emb = embed(w)                                        # [m, H]
            h = layer(h, emb)                                     # [m, H]
            logits = F.linear(h, lm_head.weight)                  # [m, V]
            # 草稿用 argmax：草稿分布已接近目标（teacher-forced top-1 ~62%），
            # 温度采样反而稀释接受率（temp=0.6 时实测 α 从 0.13 → 0.06）
            w_new = logits.argmax(dim=-1)
            keep_idx = []
            for j, seq in enumerate(active_seqs):
                t = w_new[j].item()
                # per-seq 输出预算（同 ngram 的 remaining-1）：保证追加后不超 max_tokens，
                # 否则投机接受可能跳过 max_tokens 让序列永不结束（_maybe_finish 已改 >= 兜底）
                if t == eos or len(drafts[id(seq)]) >= seq.max_tokens - seq.num_completion_tokens - 1:
                    continue
                drafts[id(seq)].append(t)
                keep_idx.append(j)
            if not keep_idx:
                break
            h = h[keep_idx]
            w = w_new[keep_idx]
            active_seqs = [active_seqs[j] for j in keep_idx]
        for s, _ in items:
            s.draft_tokens = drafts[id(s)]

    def _pd_transfer_sequence(self, seq: Sequence) -> None:
        """Move a completed prefill sequence and its prompt KV into the decode pool."""
        cached_tokens = seq.num_cached_tokens
        assert cached_tokens == seq.num_prompt_tokens, (
            f"prefill handoff has {cached_tokens}/{seq.num_prompt_tokens} cached tokens")
        source_blocks = list(seq.block_table)
        host_kv = self.prefill_runner.call("export_kv", source_blocks, cached_tokens)

        self.prefill_scheduler.running.remove(seq)
        self.prefill_scheduler.block_manager.deallocate(seq)
        seq.num_cached_tokens = cached_tokens
        seq.is_prefill = False
        seq.draft_tokens = None
        self.decode_scheduler.block_manager.allocate_private(seq)
        try:
            self.decode_runner.call("import_kv", list(seq.block_table),
                                    cached_tokens, host_kv)
        except Exception:
            self.decode_scheduler.block_manager.deallocate(seq)
            raise
        self.decode_scheduler.running.append(seq)

    def _step_pd_prefill(self) -> tuple[list[Sequence], int, bool]:
        scheduler = self.prefill_scheduler
        if not scheduler.waiting:
            return [], 0, False

        scheduler._order_waiting()
        seqs, kind = scheduler._schedule_prefill()
        self._last_step_prefix_hits.update(
            (seq.seq_id, seq.num_prefix_cached_tokens) for seq in seqs)
        if kind != "prefill":
            raise RuntimeError("PD prefill scheduler produced a non-prefill batch")
        before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        n_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        for old_id, new_id in scheduler.cow_pairs:
            self.prefill_runner.call("cow_block", old_id, new_id)

        run_started = perf_counter()
        collect_logits = (self._collect_logits
                          and not self._collect_decode_logits_only)
        result = self.prefill_runner.call(
            "run", seqs, "prefill", collect_logits, False)
        scheduler.observe_prefill(n_tokens, perf_counter() - run_started)
        if collect_logits:
            token_ids, logits = result
            self.collected_logits.append(("prefill", logits))
        else:
            token_ids = result
        scheduler.postprocess(seqs, token_ids)
        self._record_step_tokens(seqs, before)

        finished = []
        for seq in seqs:
            if seq.is_finished:
                finished.append(seq)
            elif (seq.status == SequenceStatus.RUNNING
                  and seq.num_completion_tokens > 0
                  and seq.num_cached_tokens == seq.num_prompt_tokens):
                self._pd_transfer_sequence(seq)
        return finished, n_tokens, True

    def _run_decode_burst(
        self,
        scheduler: Scheduler,
        runner: ModelRunner,
        initial_decode_tokens: int,
        *,
        collect_logits: bool,
        yield_for_prefill: bool = False,
    ) -> tuple[int, list[Sequence]]:
        """Run bounded follow-up decode forwards before yielding to the outer loop.

        The first decode batch was prepared by ``schedule`` and has already run.
        Follow-up rounds use the same scheduler state and normal KV append/COW/swap
        accounting. Prefill, PD prefill handoff, and speculative paths never enter
        this helper, so each returned engine step remains streamable.

        ``decode_burst_yield``（默认开）让 burst 在后续轮开始前检查 prefill 队列：
        mixed 模式下本步是纯 decode，说明本步调度时无 prefill 可排；但 burst 期间
        仍在到达的新请求会落进 waiting，此时提前收尾让外循环尽快调度 prefill。
        关闭后跑满 ``max_decode_steps`` 再回外循环（消融口径：多步复用 vs 新请求
        TTFT 的取舍）。
        """
        total_decode_tokens = initial_decode_tokens
        finished: list[Sequence] = []
        if (not self.config.multi_step_decode or self.config.max_decode_steps <= 1
                or self.config.speculative != "none" or scheduler.waiting
                or scheduler.swapped or yield_for_prefill):
            return total_decode_tokens, finished
        self._step_stats["decode_bursts"] += 1
        rounds_done = 1  # 首轮已由 schedule() + 本步 run() 完成

        for _ in range(1, self.config.max_decode_steps):
            if not scheduler.running or scheduler.swapped:
                break
            # 让出检查：decode_burst_yield=False 时故意跳过，继续跑满余下轮次（消融）
            if self.config.decode_burst_yield and (scheduler.waiting or yield_for_prefill):
                if scheduler.waiting:
                    # 放弃的 decode 轮次/行槽位（估算值，不代表实际会产出的 token 数）
                    self._step_stats["decode_burst_yields"] += 1
                    self._step_stats["decode_burst_skipped_slots"] += (
                        (self.config.max_decode_steps - rounds_done)
                        * len(scheduler.running))
                break
            if (self.config.decode_burst_yield_on_arrival
                    and self._decode_burst_pressure_active()):
                self._last_step_decode_burst_yielded = True
                break
            scheduler._order_running_by_tpot()
            scheduler.cow_pairs = []
            scheduler.swap_pairs = []
            seqs, kind = scheduler._schedule_decode()
            if kind != "decode" or not seqs:
                break
            for old_id, new_id in scheduler.cow_pairs:
                runner.call("cow_block", old_id, new_id)
            for seq, block_ids, buf, direction in scheduler.swap_pairs:
                transfer_bytes = buf.numel() * buf.element_size()
                transfer_started = perf_counter()
                if direction == "out":
                    runner.call("swap_out", block_ids, buf)
                    scheduler.finish_swap_out(seq, block_ids)
                else:
                    runner.call("swap_in", block_ids, buf)
                    scheduler.finish_swap_in(seq)
                scheduler.observe_kv_transfer(
                    transfer_bytes, perf_counter() - transfer_started)

            before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
            result = runner.call("run", seqs, "decode", collect_logits, False)
            if collect_logits:
                token_ids, logits = result
                self.collected_logits.append(("decode", logits))
            else:
                token_ids = result
            scheduler.postprocess(seqs, token_ids)
            self._record_step_tokens(seqs, before)
            finished.extend(seq for seq in seqs if seq.is_finished)
            total_decode_tokens += len(token_ids)
            self._last_step_multistep_tokens += len(token_ids)
            self._last_step_decode_iterations += 1
            rounds_done += 1
        self._step_stats["decode_burst_rounds"] += rounds_done
        # burst 内结束的序列必须回传给外层：它们的 per-request 计时快照在 step() 里落盘
        return total_decode_tokens, finished

    def _step_pd_decode(self) -> tuple[list[Sequence], int, int, bool]:
        scheduler = self.decode_scheduler
        if not (scheduler.waiting or scheduler.running or scheduler.swapped):
            return [], 0, 0, False

        seqs, kind = scheduler.schedule()
        self._last_step_prefix_hits.update(
            (seq.seq_id, seq.num_prefix_cached_tokens) for seq in seqs)
        before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        n_prefill = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        n_decode = sum(1 for seq in seqs if not seq.is_prefill)
        for old_id, new_id in scheduler.cow_pairs:
            self.decode_runner.call("cow_block", old_id, new_id)
        for seq, block_ids, buf, direction in scheduler.swap_pairs:
            transfer_bytes = buf.numel() * buf.element_size()
            transfer_started = perf_counter()
            if direction == "out":
                self.decode_runner.call("swap_out", block_ids, buf)
                scheduler.finish_swap_out(seq, block_ids)
            else:
                self.decode_runner.call("swap_in", block_ids, buf)
                scheduler.finish_swap_in(seq)
            scheduler.observe_kv_transfer(
                transfer_bytes, perf_counter() - transfer_started)

        prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        run_started = perf_counter()
        collect_logits = (self._collect_logits
                          and (not self._collect_decode_logits_only or kind == "decode"))
        result = self.decode_runner.call(
            "run", seqs, kind, collect_logits, False)
        if kind == "prefill":
            scheduler.observe_prefill(prefill_tokens, perf_counter() - run_started)
        if collect_logits:
            token_ids, logits = result
            self.collected_logits.append((kind, logits))
        else:
            token_ids = result
        scheduler.postprocess(seqs, token_ids)
        self._record_step_tokens(seqs, before)
        if kind in ("decode", "mixed", "spec") and n_decode:
            self._last_step_decode_iterations = 1
        if kind == "decode":
            n_decode, burst_finished = self._run_decode_burst(
                scheduler, self.decode_runner, n_decode,
                collect_logits=self._collect_logits,
                yield_for_prefill=bool(self.prefill_scheduler.waiting))
        else:
            burst_finished = []
        finished = [seq for seq in seqs if seq.is_finished]
        finished.extend(burst_finished)
        return finished, n_prefill, n_decode, True

    def _step_pd(self):
        """Run separate prefill/decode batches and hand prompt KV between GPU pools."""
        self._last_step_prefix_hits = {}
        self.prefill_scheduler.external_tpot_target_active = (
            self.decode_scheduler._has_active_tpot_target())
        self.prefill_scheduler.external_tpot_pressure = (
            self.decode_scheduler._tpot_decode_pressure())
        prefill_done, n_prefill, _ = self._step_pd_prefill()
        decode_done, decode_prefill, n_decode, did_decode = self._step_pd_decode()
        n_prefill += decode_prefill
        finished = prefill_done + decode_done
        for seq in finished:
            self._req_metrics.append({
                "seq_id": seq.seq_id,
                "prompt_tokens": seq.num_prompt_tokens,
                "completion_tokens": len(seq.completion_token_ids),
                "t_submitted": seq.t_submitted,
                "t_prefill_started": seq.t_prefill_started,
                "t_first_token": seq.t_first_token,
                "t_completed": seq.t_completed,
                "prefix_cached_tokens": seq.num_prefix_cached_tokens,
                "ttft_slo_ms": seq.ttft_slo_ms,
                "ttft_slo_met": self._ttft_slo_met(seq),
                "tpot_slo_ms": seq.tpot_slo_ms,
                "tpot_slo_met": self._tpot_slo_met(seq),
            })
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished]
        if n_prefill and n_decode:
            kind = "mixed"  # separate prefill/decode batches ran during this engine step
        elif n_prefill:
            kind = "prefill"
        elif did_decode:
            kind = "decode"
        else:
            raise RuntimeError("PD engine has pending requests but neither stage made progress")
        return outputs, kind, n_prefill, n_decode

    def step(self):
        self._last_step_tokens = {}
        self._last_step_prefix_hits = {}
        self._last_step_decode_iterations = 0
        self._last_step_multistep_tokens = 0
        self._last_step_decode_burst_yielded = False
        if self._pd:
            return self._step_pd()
        seqs, kind = self.scheduler.schedule()  # kind ∈ {"prefill", "decode", "mixed", "spec"}
        self._last_step_prefix_hits.update(
            (seq.seq_id, seq.num_prefix_cached_tokens) for seq in seqs)
        before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        scheduled_prefill_tokens = sum(
            seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        scheduled_decode_rows = sum(1 for seq in seqs if not seq.is_prefill)
        # 本步是否含verify行（投机）：任一行带draft_tokens（[]也算，表示γ=0的verify行）
        has_spec = any(seq.draft_tokens is not None for seq in seqs)
        # Medusa/EAGLE模式：spec步需要最后一层hidden（下轮草稿输入）
        return_hidden = self.config.speculative in ("medusa", "eagle") and kind == "spec"
        # 在运行模型之前执行COW复制：任何序列写共享部分块之前，先把旧块复制给写者。
        # 必须发生在 run() 之前，prepare_decode/prepare_prefill 才能基于换表后的新块计算slot
        for old_id, new_id in self.scheduler.cow_pairs:
            self.model_runner.call("cow_block", old_id, new_id)
        # KV swap 拷贝（同 COW 时机）：换出 GPU→CPU、换入 CPU→GPU（新私有块，run 前填好）。
        # 换出对在拷贝完成后释放块（finish_swap_out）——块在拷贝前保持占用，
        # 避免本步内被重分配覆盖内容。TP=1 时启用；FP8 在 CPU 端按 uint8 原始字节暂存。
        for seq, block_ids, buf, direction in self.scheduler.swap_pairs:
            transfer_bytes = buf.numel() * buf.element_size()
            transfer_started = perf_counter()
            if direction == "out":
                self.model_runner.call("swap_out", block_ids, buf)
                self.scheduler.finish_swap_out(seq, block_ids)
            else:
                self.model_runner.call("swap_in", block_ids, buf)
                self.scheduler.finish_swap_in(seq)
            self.scheduler.observe_kv_transfer(
                transfer_bytes, perf_counter() - transfer_started)
        # 运行模型，返回采样出的token（精度检查模式下同时返回本步logits）
        run_started = perf_counter()
        collect_logits = (self._collect_logits
                          and (not self._collect_decode_logits_only or kind == "decode"))
        result = self.model_runner.call("run", seqs, kind, collect_logits, return_hidden)
        run_elapsed = perf_counter() - run_started
        if kind == "prefill":
            self.scheduler.observe_prefill(
                scheduled_prefill_tokens, run_elapsed)
        elif kind == "mixed" and scheduled_prefill_tokens:
            # Continuous-arrival batches are often mixed for long stretches. Use
            # the same proportional attribution as benchmark metrics so the
            # recompute-cost EWMA does not stay at its fallback forever.
            estimated_prefill_elapsed = run_elapsed * scheduled_prefill_tokens / (
                scheduled_prefill_tokens + scheduled_decode_rows)
            self.scheduler.observe_prefill(
                scheduled_prefill_tokens, estimated_prefill_elapsed)
        hidden = None
        if collect_logits:
            if return_hidden:
                token_ids, logits, hidden = result
            else:
                token_ids, logits = result
            self.collected_logits.append((kind, logits))
        else:
            if return_hidden:
                token_ids, hidden = result
            else:
                token_ids = result
        if has_spec:
            # 统计必须在postprocess_spec之前（之后draft_tokens/num_scheduled_tokens被清空）
            n_spec_rows = sum(1 for seq in seqs if seq.draft_tokens is not None)
            # Medusa：postprocess前捕获每行行数（_medusa_drafts选hidden行用）
            n_rows_list = [seq.num_scheduled_tokens if seq.draft_tokens is not None else 0
                           for seq in seqs]
            (token_lists, n_decode, n_draft, n_draft_acc, n_verify,
             n_acc_list) = self._verify(seqs, token_ids)
            self.scheduler.postprocess_spec(seqs, token_lists)
            # 投机统计（与kind无关，混合步同样累计）
            self._step_stats["spec_steps"] += 1
            self._step_stats["spec_rows"] += n_spec_rows
            self._step_stats["spec_verify_tokens"] += n_verify
            self._step_stats["spec_draft_tokens"] += n_draft
            self._step_stats["spec_accepted_drafts"] += n_draft_acc
        else:
            self.scheduler.postprocess(seqs, token_ids)
        # Medusa/EAGLE：用本步verify的hidden生成下一轮草稿（写回seq.draft_tokens）
        if return_hidden:
            if self.config.speculative == "eagle":
                self._eagle_drafts(seqs, hidden, n_acc_list, n_rows_list)
            else:
                self._medusa_drafts(seqs, hidden, n_acc_list, n_rows_list)
        self._record_step_tokens(seqs, before)
        if kind in ("decode", "mixed", "spec") and scheduled_decode_rows:
            self._last_step_decode_iterations = 1
        if kind == "decode":
            n_decode, burst_finished = self._run_decode_burst(
                self.scheduler, self.model_runner, scheduled_decode_rows,
                collect_logits=self._collect_logits)
        else:
            burst_finished = []
        # 统计用token数：prefill步为正（prefill token数），decode步为负（序列数），
        # mixed步拆分返回prefill/decode各自的数量
        if kind == "prefill":
            n_prefill, n_decode = scheduled_prefill_tokens, 0
        elif kind == "decode":
            n_prefill = 0
        elif kind == "spec":
            n_prefill = 0  # n_decode 已在验收中按实际接受数统计
        else:  # mixed
            n_prefill = scheduled_prefill_tokens
            if not has_spec:
                n_decode = scheduled_decode_rows
        # 调度器完成后处理，如追加token、更新缓存、判断是否结束等
        # 快照已结束请求的计时信息（基准测试使用；driver侧数据完整）
        #
        # 去重是必须的：`_run_decode_burst` 每跑完一轮就把该轮结束的序列收进
        # `burst_finished` 并**当场**落盘（序列在自己的轮次结束时即完成），而外层
        # `seqs` 里同一条序列也会出现在 `finished_seqs` 中——不去重会得到同 seq_id
        # 的多条记录，驱动侧 `completed` 计数随之虚高、提前结束循环（实测 128 请求
        # 只统计到 95 个不同 seq_id、另有 38 条重复）。
        finished_seqs = list(seqs)
        seen_seq_ids = {seq.seq_id for seq in finished_seqs}
        for seq in burst_finished:
            if seq.seq_id not in seen_seq_ids:
                seen_seq_ids.add(seq.seq_id)
                finished_seqs.append(seq)
        for seq in finished_seqs:
            if seq.is_finished:
                self._record_once(seq)
        # 收集已经完成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in finished_seqs if seq.is_finished]
        return outputs, kind, n_prefill, n_decode

    def is_finished(self):
        if self._pd:
            return (self.prefill_scheduler.is_finished()
                    and self.decode_scheduler.is_finished())
        return self.scheduler.is_finished()

    @staticmethod
    def _ttft_slo_met(seq: Sequence) -> bool | None:
        if (seq.ttft_slo_ms is None or seq.t_submitted is None
                or seq.t_first_token is None):
            return None
        return (seq.t_first_token - seq.t_submitted) * 1000 <= seq.ttft_slo_ms

    @staticmethod
    def _tpot_slo_met(seq: Sequence) -> bool | None:
        if (seq.tpot_slo_ms is None or seq.t_first_token is None
                or seq.t_completed is None or seq.num_completion_tokens <= 1):
            return None
        tpot_ms = ((seq.t_completed - seq.t_first_token) * 1000.0
                   / (seq.num_completion_tokens - 1))
        return tpot_ms <= seq.tpot_slo_ms

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True, # use_tqdm:是否显示进度条
        collect_logits: bool = False, # 精度检查：逐step收集logits到self.collected_logits
        collect_decode_logits_only: bool = False,
    ) -> list[str]:
        # 创建一个进度条，总共有len(prompts)个任务，进度条前缀显示"Generating",
        # dynamic_ncols=True：让进度条自适应终端宽度，disabel：是否禁用进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 重置基准计时统计
        self._req_metrics = []
        self._recorded_seq_ids = set()
        self._step_stats = self._empty_step_stats()
        for scheduler in ((self.prefill_scheduler, self.decode_scheduler) if self._pd
                          else (self.scheduler,)):
            scheduler.reset_metrics()
        graph_runners = ([self.prefill_runner, self.decode_runner] if self._pd
                         else [self.model_runner])
        for runner in graph_runners:
            runner.mixed_cudagraph_capture_count = 0
            runner.mixed_cudagraph_replay_count = 0
            runner.mixed_cudagraph_eager_fallbacks = 0
        self._collect_logits = collect_logits
        self._collect_decode_logits_only = collect_decode_logits_only
        self.collected_logits = []
        while not self.is_finished():
            t = perf_counter()
            output, kind, n_prefill, n_decode = self.step()
            dt = perf_counter() - t
            self._step_stats["decode_iterations"] += self._last_step_decode_iterations
            self._step_stats["multi_step_decode_tokens"] += (
                self._last_step_multistep_tokens)
            self._step_stats["decode_burst_pressure_yields"] += int(
                self._last_step_decode_burst_yielded)
            if self._last_step_decode_iterations > 1:
                self._step_stats["multi_step_decode_steps"] += 1
            # 累计逐step统计（基准测试使用）；mixed步按token比例拆分时间归属
            if kind == "prefill":
                self._step_stats["prefill_steps"] += 1
                self._step_stats["prefill_tokens"] += n_prefill
                self._step_stats["prefill_time"] += dt
                prefill_throughput = n_prefill / dt
            elif kind in ("decode", "spec"):
                self._step_stats["decode_steps"] += 1
                self._step_stats["decode_tokens"] += n_decode
                self._step_stats["decode_time"] += dt
                decode_throughput = n_decode / dt
            else:  # mixed
                self._step_stats["prefill_steps"] += 1
                self._step_stats["decode_steps"] += 1
                self._step_stats["prefill_tokens"] += n_prefill
                self._step_stats["decode_tokens"] += n_decode
                total = n_prefill + n_decode
                self._step_stats["prefill_time"] += dt * n_prefill / total
                self._step_stats["decode_time"] += dt * n_decode / total
                if n_prefill:
                    prefill_throughput = n_prefill / dt
                if n_decode:
                    decode_throughput = n_decode / dt
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    def collect_metrics(self) -> dict:
        """返回最近一次generate()的基准计时原始数据（供benchmarks/使用）。

        - per_request: 每个已结束请求的 {seq_id, prompt_tokens, completion_tokens,
          t_submitted, t_first_token, t_completed}，时间为秒（perf_counter基准）；
          t_first_token/t_completed 为None表示请求未生成token/未完成。
        - step_stats: 逐step聚合 {prefill_steps, decode_steps, prefill_tokens,
          decode_tokens, prefill_time, decode_time, decode_iterations,
          multi_step_decode_tokens, decode_burst_pressure_yields}；投机解码时另有
          {spec_steps, spec_verify_tokens, spec_draft_tokens, spec_accepted_drafts}
          （α = spec_accepted_drafts / spec_draft_tokens）。
        - num_preemptions: 本次generate中的KV cache抢占次数。
        - prefix feature/cache lifecycle: parse/reuse由scheduler统计；generation失效、KV
          mutation、LRU eviction与deferred-free refs/flush/peak由BlockManager统计。
        """
        schedulers = ((self.prefill_scheduler, self.decode_scheduler) if self._pd
                      else (self.scheduler,))
        block_managers = [s.block_manager for s in schedulers]
        block_managers.extend(
            s.full_block_manager for s in schedulers
            if s.full_block_manager is not None)
        slo_steps = sum(s.num_slo_prefill_steps for s in schedulers)
        tpot_steps = sum(s.num_tpot_prefill_steps for s in schedulers)
        graph_runner = self.decode_runner if self._pd else self.model_runner
        return {
            "per_request": list(self._req_metrics),
            "step_stats": dict(self._step_stats),
            "num_preemptions": sum(s.num_preemptions for s in schedulers),
            "num_swaps": sum(s.num_swaps for s in schedulers),
            "num_recompute_preemptions": sum(
                s.num_recompute_preemptions for s in schedulers),
            "recompute_tokens": sum(s.recompute_tokens for s in schedulers),
            "estimated_recompute_seconds": sum(
                s.estimated_recompute_seconds for s in schedulers),
            "estimated_swap_seconds": sum(s.estimated_swap_seconds for s in schedulers),
            "num_affinity_probes": sum(s.num_affinity_probes for s in schedulers),
            "num_prefix_feature_parses": sum(
                s.num_prefix_feature_parses for s in schedulers),
            "num_prefix_feature_reuses": sum(
                s.num_prefix_feature_reuses for s in schedulers),
            "prefix_cache_hit_tokens": sum(s.prefix_cache_hit_tokens for s in schedulers),
            "prefix_cache_hit_requests": sum(
                s.prefix_cache_hit_requests for s in schedulers),
            "prefix_cache_evictions": sum(
                manager.prefix_cache_evictions for manager in block_managers),
            "prefix_cache_lru_entries": sum(
                len(manager.prefix_lru) for manager in block_managers),
            "prefix_cache_generation": sum(
                manager.kv_generation for manager in block_managers),
            "kv_generation_mutations": sum(
                manager.kv_generation_mutations for manager in block_managers),
            "prefix_feature_invalidations": sum(
                manager.prefix_feature_invalidations for manager in block_managers),
            "deferred_free_refs_queued": sum(
                manager.deferred_free_refs_queued for manager in block_managers),
            "deferred_free_refs_committed": sum(
                manager.deferred_free_refs_committed for manager in block_managers),
            "deferred_free_flushes": sum(
                manager.deferred_free_flushes for manager in block_managers),
            "deferred_free_peak_blocks": max(
                (manager.deferred_free_peak_blocks for manager in block_managers),
                default=0),
            "deferred_free_peak_refs": max(
                (manager.deferred_free_peak_refs for manager in block_managers),
                default=0),
            "deferred_free_pending_blocks": sum(
                len(set(manager.deferred_free_block_ids))
                for manager in block_managers),
            "deferred_free_pending_refs": sum(
                len(manager.deferred_free_block_ids) for manager in block_managers),
            "mixed_cudagraph": {
                "enabled": self.config.mixed_cudagraph and not self.config.enforce_eager,
                "max_graphs": self.config.mixed_cudagraph_max_graphs,
                "max_tokens": self.config.mixed_cudagraph_max_tokens,
                "cached_shapes": len(getattr(graph_runner, "mixed_graphs", {})),
                "captures": graph_runner.mixed_cudagraph_capture_count,
                "replays": graph_runner.mixed_cudagraph_replay_count,
                "eager_fallbacks": graph_runner.mixed_cudagraph_eager_fallbacks,
            },
            "num_aging_promotions": sum(s.num_aging_promotions for s in schedulers),
            "slo_prefill_steps": slo_steps,
            "adaptive_prefill_tokens_avg": (
                sum(s.slo_prefill_budget_sum for s in schedulers) / slo_steps
                if slo_steps else 0.0),
            "adaptive_prefill_tokens_max": max(
                (s.slo_prefill_budget_max for s in schedulers), default=0),
            "adaptive_prefill_rows_avg": (
                sum(s.slo_prefill_rows_sum for s in schedulers) / slo_steps
                if slo_steps else 0.0),
            "tpot_prefill_steps": tpot_steps,
            "tpot_priority_steps": sum(s.num_tpot_priority_steps for s in schedulers),
            "tpot_adaptive_prefill_tokens_avg": (
                sum(s.tpot_prefill_budget_sum for s in schedulers) / tpot_steps
                if tpot_steps else 0.0),
            # Prefill 饥饿观测：budget_scale = prefill_pressure·(1−tpot_pressure)。
            # 目标不可达时 scale→0，prefill 配额落到 prefill_reserve_tokens 下限。
            "tpot_budget_scale_avg": (
                sum(s.slo_prefill_budget_scale_sum for s in schedulers) / slo_steps
                if slo_steps else 1.0),
            "tpot_budget_scale_min": min(
                (s.slo_prefill_budget_scale_min for s in schedulers), default=1.0),
            "tpot_starved_prefill_steps": sum(
                s.num_tpot_starved_steps for s in schedulers),
            "prefill_reserve_tokens": self.config.prefill_reserve_tokens,
            "multi_step_decode_tokens": self._step_stats["multi_step_decode_tokens"],
            "multi_step_decode_enabled": self.config.multi_step_decode,
            "max_decode_steps": self.config.max_decode_steps,
            "decode_burst_yield_enabled": self.config.decode_burst_yield,
            "decode_bursts": self._step_stats["decode_bursts"],
            "decode_burst_rounds": self._step_stats["decode_burst_rounds"],
            "decode_burst_yields": self._step_stats["decode_burst_yields"],
            "decode_burst_skipped_slots": self._step_stats["decode_burst_skipped_slots"],
        }

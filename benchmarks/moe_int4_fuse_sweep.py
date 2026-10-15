"""Paired full-model sweep for segmented INT4 MoE gate/up fusion.

Example::

    python benchmarks/moe_int4_fuse_sweep.py \
        --model ~/huggingface/DeepSeek-V2-Lite \
        --seeds 7,17,42 --arrival-rates 4,8 --output-lengths 16,64

The checkpoint is loaded once. Each split/fused pair receives the same prompts,
arrival times, and SLO targets; the order alternates by condition. Idle prefix KV
is cleared after warmup so the second run cannot inherit cache hits from the first.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    from .bench import build_workload, build_workload_data, env_info
    from .scheduling_ablation import (
        build_arrival_trace,
        build_ttft_slo_targets,
        create_arrival_engine,
        run_arrival_trace,
        scheduler_list,
        variant_table,
        workload_fingerprint,
    )
except ImportError:  # direct script execution from benchmarks/
    from bench import build_workload, build_workload_data, env_info
    from scheduling_ablation import (
        build_arrival_trace,
        build_ttft_slo_targets,
        create_arrival_engine,
        run_arrival_trace,
        scheduler_list,
        variant_table,
        workload_fingerprint,
    )

from nanovllm import SamplingParams
from nanovllm.layers.moe import MoE


def _parse_ints(value: str, name: str, *, allow_zero: bool = False) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    minimum = 0 if allow_zero else 1
    if not values or any(item < minimum for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain {qualifier} integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _parse_rates(value: str) -> list[float]:
    try:
        rates = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--arrival-rates must be a comma-separated number list") from exc
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("--arrival-rates must contain positive finite values")
    if len(rates) != len(set(rates)):
        raise ValueError("--arrival-rates must not contain duplicates")
    return rates


def _parse_positive_floats(value: str, name: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated number list") from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive finite values")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _set_fusion_mode(moes: list[MoE], fused: bool) -> None:
    for moe in moes:
        moe.int4_backend = "segment"
        moe.fuse_int4_gate_up = fused


def _validate_segment_layers(engine) -> list[MoE]:
    moes = [module for module in engine.model_runner.model.modules()
            if isinstance(module, MoE)]
    if not moes:
        raise RuntimeError("the selected checkpoint has no MoE layers")
    for layer_index, moe in enumerate(moes):
        dtype = moe.experts[0].gate_proj.w_int4_scale.dtype
        probe = torch.empty((1, moe.hidden_size), device="cuda", dtype=dtype)
        if not moe._can_segment_int4(probe):
            raise RuntimeError(
                f"MoE layer {layer_index} cannot use segmented INT4: "
                f"{moe._int4_segment_reason}")
        moe._ensure_int4_segment_tables(probe)
    return moes


def _install_greedy_sampler(engine) -> None:
    """Make token checks independent of random-number consumption order."""
    sampler = engine.model_runner.sampler

    def greedy_forward(logits: torch.Tensor, _temperatures: torch.Tensor):
        return logits.argmax(dim=-1)

    sampler.forward = greedy_forward


def _prime_segment_variants(engine, moes: list[MoE], prompts: list[list[int]]) -> None:
    """Compile eager kernel paths for each single-trace batch size before timing."""
    warmup_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)
    for fused in (False, True):
        _set_fusion_mode(moes, fused)
        for count in range(1, len(prompts) + 1):
            engine.generate(
                prompts[:count],
                [warmup_params] * count,
                use_tqdm=False,
            )
            _clear_idle_prefix_cache(engine)


def _clear_idle_prefix_cache(engine) -> None:
    """Remove cached KV only after all requests have drained from this trace."""
    if not engine.is_finished():
        raise RuntimeError("cannot clear prefix KV while requests are active")
    for scheduler in scheduler_list(engine):
        managers = (scheduler.block_manager, scheduler.full_block_manager)
        for manager in managers:
            if manager is None:
                continue
            if manager.used_block_ids or manager.deferred_free_block_ids:
                raise RuntimeError("prefix KV clear requires all blocks to be idle")
            for block in manager.blocks:
                if block.ref_count != 0 or block.pending_free:
                    raise RuntimeError("prefix KV clear found a referenced block")
                if block.hash != -1:
                    if block.block_id not in manager.free_block_ids:
                        raise RuntimeError("cached KV block is missing from the free list")
                    manager._evict_prefix_block(block)
            if manager.hash_to_block_id or manager.prefix_lru:
                raise RuntimeError("prefix cache did not empty after idle KV clear")


def _ordered_outputs(run: dict[str, Any]) -> list[list[int]]:
    output_by_seq = run.pop("completion_token_ids")
    return [output_by_seq[str(row["seq_id"])]
            for row in sorted(run["per_request"], key=lambda row: row["seq_id"])]


def _output_fingerprint(outputs: list[list[int]]) -> str:
    raw = json.dumps(outputs, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_fingerprint(value: Any) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _percent_delta(after: float | int | None,
                   before: float | int | None) -> float | None:
    if after is None or before in (None, 0):
        return None
    return 100.0 * (after - before) / before


def _pair_metrics(split: dict, fused: dict) -> dict[str, float | int | None]:
    split_summary = split["summary"]
    fused_summary = fused["summary"]
    return {
        "output_throughput_delta_percent": _percent_delta(
            fused_summary["throughput_output_tok_per_s"],
            split_summary["throughput_output_tok_per_s"]),
        "joint_slo_output_goodput_delta_percent": _percent_delta(
            fused_summary["slo_goodput"]["joint_slo_output_tokens_per_second"],
            split_summary["slo_goodput"]["joint_slo_output_tokens_per_second"]),
        "joint_slo_output_goodput_delta_tok_s": (
            fused_summary["slo_goodput"]["joint_slo_output_tokens_per_second"]
            - split_summary["slo_goodput"]["joint_slo_output_tokens_per_second"]),
        "joint_slo_request_delta": (
            fused_summary["slo_goodput"]["joint_slo_requests"]
            - split_summary["slo_goodput"]["joint_slo_requests"]),
        "ttft_p50_delta_percent": _percent_delta(
            fused_summary["ttft"]["p50"], split_summary["ttft"]["p50"]),
        "ttft_p99_delta_percent": _percent_delta(
            fused_summary["ttft"]["p99"], split_summary["ttft"]["p99"]),
        "tpot_p50_delta_percent": _percent_delta(
            fused_summary["tpot"]["p50"], split_summary["tpot"]["p50"]),
        "tpot_p99_delta_percent": _percent_delta(
            fused_summary["tpot"]["p99"], split_summary["tpot"]["p99"]),
    }


def _distribution(values: list[float | int | None]) -> dict[str, float | int | None]:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return {"count": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "count": len(defined),
        "median": statistics.median(defined),
        "mean": statistics.fmean(defined),
        "min": min(defined),
        "max": max(defined),
    }


class DecodeBatchShapeRecorder:
    """Collect model-call decode/prefill shapes without synchronizing CUDA."""

    def __init__(self, model_runner):
        self.active = False
        self.shape_counts: Counter[tuple[str, int, int, int, int]] = Counter()
        self.decode_batch_sizes: list[int] = []
        self.decode_model_calls = 0
        self._original_call = model_runner.call
        model_runner.call = self._call

    def _call(self, method_name, *args):
        if self.active and method_name == "run" and len(args) >= 2:
            seqs, kind = args[:2]
            prefill = [seq for seq in seqs if seq.is_prefill]
            decode = [seq for seq in seqs if not seq.is_prefill]
            prefill_tokens = sum(seq.num_scheduled_tokens for seq in prefill)
            decode_tokens = sum(seq.num_scheduled_tokens for seq in decode)
            shape = (str(kind), len(decode), len(prefill),
                     prefill_tokens, decode_tokens)
            self.shape_counts[shape] += 1
            if decode:
                self.decode_batch_sizes.append(len(decode))
                self.decode_model_calls += 1
        return self._original_call(method_name, *args)

    def start(self) -> None:
        self.shape_counts.clear()
        self.decode_batch_sizes.clear()
        self.decode_model_calls = 0
        self.active = True

    def stop(self) -> dict[str, Any]:
        self.active = False
        histogram = [
            {
                "kind": kind,
                "decode_rows": decode_rows,
                "prefill_sequences": prefill_sequences,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "model_calls": count,
            }
            for (kind, decode_rows, prefill_sequences, prefill_tokens, decode_tokens), count
            in sorted(self.shape_counts.items())
        ]
        return {
            "model_runner_calls": sum(self.shape_counts.values()),
            "decode_model_calls": self.decode_model_calls,
            "decode_batch_size_rows": _distribution(self.decode_batch_sizes),
            "shape_histogram": histogram,
        }


class GpuClockSampler:
    """Sample GPU clocks at low frequency while each measured arrival trace runs."""

    FIELDS = (
        "sm_clock_mhz", "memory_clock_mhz", "graphics_clock_mhz",
        "power_w", "temperature_c",
    )

    def __init__(self, interval_ms: int):
        self.interval_ms = interval_ms
        self.samples: list[dict[str, Any]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _number(value: str) -> float | None:
        value = value.strip()
        if not value or value.upper() in {"N/A", "NA", "NOT SUPPORTED"}:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def start(self) -> None:
        query_fields = (
            "timestamp,clocks.current.sm,clocks.current.memory,"
            "clocks.current.graphics,power.draw,temperature.gpu"
        )
        self.process = subprocess.Popen(
            [
                "nvidia-smi", f"--query-gpu={query_fields}",
                "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read_samples, daemon=True)
        self.thread.start()
        time.sleep(0.1)
        if self.process.poll() is not None:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"nvidia-smi GPU clock sampling failed: {stderr.strip()}")

    def _read_samples(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                fields = next(csv.reader([line]))
            except (csv.Error, StopIteration):
                continue
            if len(fields) < 6:
                continue
            sample = {
                "monotonic_seconds": time.monotonic(),
                "driver_timestamp": fields[0].strip(),
                "sm_clock_mhz": self._number(fields[1]),
                "memory_clock_mhz": self._number(fields[2]),
                "graphics_clock_mhz": self._number(fields[3]),
                "power_w": self._number(fields[4]),
                "temperature_c": self._number(fields[5]),
            }
            with self._lock:
                self.samples.append(sample)

    def summarize(self, started: float, ended: float) -> dict[str, Any]:
        with self._lock:
            samples = [dict(sample) for sample in self.samples
                       if started <= sample["monotonic_seconds"] <= ended]
        summary = {
            key: _distribution([sample[key] for sample in samples])
            for key in self.FIELDS
        }
        raw_samples = [
            {
                "elapsed_seconds": round(sample["monotonic_seconds"] - started, 3),
                "driver_timestamp": sample["driver_timestamp"],
                **{key: sample[key] for key in self.FIELDS},
            }
            for sample in samples
        ]
        return {
            "sample_interval_ms": self.interval_ms,
            "sample_count": len(samples),
            "clock_summary": summary,
            "samples": raw_samples,
        }

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.thread is not None:
            self.thread.join(timeout=2)


def _aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    key_metrics = (
        "output_throughput_delta_percent",
        "joint_slo_output_goodput_delta_percent",
        "joint_slo_output_goodput_delta_tok_s",
        "joint_slo_request_delta",
        "ttft_p50_delta_percent",
        "ttft_p99_delta_percent",
        "tpot_p50_delta_percent",
        "tpot_p99_delta_percent",
    )
    result = {key: _distribution([pair["comparison"][key] for pair in pairs])
              for key in key_metrics}
    throughput_deltas = [pair["comparison"]["output_throughput_delta_percent"]
                         for pair in pairs]
    result["throughput_fused_wins"] = sum(
        value is not None and value > 0 for value in throughput_deltas)
    result["throughput_ties"] = sum(
        value is not None and value == 0 for value in throughput_deltas)
    result["throughput_fused_losses"] = sum(
        value is not None and value < 0 for value in throughput_deltas)
    result["output_token_parity_pairs"] = sum(
        pair["output_parity"]["all_requests_match"] for pair in pairs)
    result["output_token_parity_failures"] = sum(
        not pair["output_parity"]["all_requests_match"] for pair in pairs)
    return result


def _build_engine_args(cli_args) -> SimpleNamespace:
    return SimpleNamespace(
        model=os.path.abspath(os.path.expanduser(cli_args.model)),
        tp=1,
        max_model_len=cli_args.max_model_len,
        gpu_memory_utilization=cli_args.gpu_memory_utilization,
        enforce_eager=True,
        kv_cache_dtype="auto",
        quantization="int4",
        nvfp4_backend="software",
        int4_group_size=cli_args.int4_group_size,
        moe_int4_backend="segment",
        no_moe_int4_fuse_gate_up=False,
        streaming_load=True,
        no_swap_kv=True,
        kv_swap_space_gb=2.0,
        admission_window=16,
        aging_timeout_ms=2000.0,
        prefill_reserve_tokens=256,
        slo_tpot_ms=cli_args.slo_tpot_ms,
        max_decode_steps=4,
        max_prefill_chunk_tokens=4096,
        queue_depth_for_full_prefill=16,
        preempt_prefill_tps=10000.0,
        preempt_kv_transfer_gbps=12.0,
        tpot_prefill_throttle_margin=0.5,
        tpot_throttle_max_waiting=16,
        warmup_seqs=cli_args.warmup_seqs,
        seed=0,
        slo_ttft_ms=cli_args.slo_ttft_ms,
    )


def _sample_request_trace(prompts: list[list[int]], metadata: list[dict],
                          count: int, seed: int) -> tuple[list[list[int]], list[dict]]:
    """Select distinct real requests once per seed, shared by all paired runs."""
    by_hash = {}
    for index, request in enumerate(metadata):
        prompt_hash = request.get("prompt_token_hash")
        if prompt_hash and prompt_hash not in by_hash:
            by_hash[prompt_hash] = index
    candidates = list(by_hash.values())
    if len(candidates) < count:
        raise ValueError(
            f"prompt trace has {len(candidates)} distinct prompts; "
            f"--num-seqs {count} needs at least that many")
    selected = random.Random(seed).sample(candidates, count)
    selected_prompts = [prompts[index] for index in selected]
    selected_metadata = []
    for index in selected:
        request = dict(metadata[index])
        request["source_trace_arrival_time_s"] = request.pop(
            "arrival_time_s", None)
        selected_metadata.append(request)
    return selected_prompts, selected_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.expanduser(
        "~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--seeds", default="7,17,42")
    parser.add_argument("--arrival-rates", default="4,8",
                        help="comma-separated offered rates in requests/second")
    parser.add_argument("--output-lengths", default="16,64")
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--prompts-file", default=None,
                        help="optional real-prompt JSONL trace; arrival schedules and output caps are generated by this sweep")
    parser.add_argument("--arrival-mode", choices=("constant", "poisson"),
                        default="constant")
    parser.add_argument("--input-length", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--int4-group-size", type=int, default=64)
    parser.add_argument("--slo-ttft-ms", type=float, default=10000.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=600.0)
    parser.add_argument("--slo-tpot-targets", default=None,
                        help="comma-separated TPOT SLO targets in ms; overrides --slo-tpot-ms")
    parser.add_argument("--warmup-seqs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2,
                        help="paired repetitions per workload condition")
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=1000,
                        help="nvidia-smi polling interval while each trace runs")
    parser.add_argument("--sampling-mode", choices=("greedy", "random"),
                        default="greedy",
                        help="greedy makes full-model output-token parity deterministic")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="temperature used when --sampling-mode=random")
    parser.add_argument("--require-output-parity", action="store_true",
                        help="return nonzero if any continuous-arrival pair emits different token IDs")
    parser.add_argument("--output", default="results/moe_int4_fuse_sweep.json")
    cli_args = parser.parse_args()

    seeds = _parse_ints(cli_args.seeds, "--seeds", allow_zero=True)
    output_lengths = _parse_ints(cli_args.output_lengths, "--output-lengths")
    arrival_rates = _parse_rates(cli_args.arrival_rates)
    tpot_targets = (_parse_positive_floats(
        cli_args.slo_tpot_targets, "--slo-tpot-targets")
        if cli_args.slo_tpot_targets is not None else [cli_args.slo_tpot_ms])
    if cli_args.num_seqs < 1 or cli_args.input_length < 1:
        raise ValueError("--num-seqs and --input-length must be positive")
    if cli_args.max_model_len < cli_args.input_length + max(output_lengths):
        raise ValueError("--max-model-len must cover input plus output tokens")
    if cli_args.int4_group_size != 64:
        raise ValueError("the DeepSeek-V2-Lite INT4 checkpoint requires group size 64")
    if cli_args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if cli_args.gpu_sample_interval_ms < 250:
        raise ValueError("--gpu-sample-interval-ms must be at least 250")
    if cli_args.temperature <= 1e-10:
        raise ValueError("--temperature must be greater than 1e-10")
    if not torch.cuda.is_available():
        raise RuntimeError("the full-model sweep requires a CUDA GPU")

    source_prompts = None
    source_requests = None
    trace_sha256 = None
    prompts_by_seed = {}
    requests_by_seed = {}
    if cli_args.prompts_file:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            cli_args.model, use_fast=True, local_files_only=True)
        trace_workload = build_workload_data(SimpleNamespace(
            seed=0,
            prompts_file=cli_args.prompts_file,
            num_seqs=0,
            min_input_len=1,
            max_input_len=65536,
            min_output_len=1,
            max_output_len=max(output_lengths),
            shared_prefix_len=0,
        ), tokenizer)
        source_prompts, source_requests = (
            trace_workload.prompts, trace_workload.requests)
        if len(source_prompts) < cli_args.num_seqs:
            raise ValueError(
                f"prompt trace has {len(source_prompts)} requests; "
                f"--num-seqs {cli_args.num_seqs} is too large")
        for seed in seeds:
            prompts_by_seed[seed], requests_by_seed[seed] = _sample_request_trace(
                source_prompts, source_requests, cli_args.num_seqs, seed)
            if max(map(len, prompts_by_seed[seed])) + max(output_lengths) > cli_args.max_model_len:
                raise ValueError(
                    "--max-model-len must cover the selected real prompt and "
                    "the largest output cap")
        trace_sha256 = hashlib.sha256(
            Path(cli_args.prompts_file).expanduser().read_bytes()).hexdigest()

    # The runner's default is set to the first target; each measured request also
    # receives its condition-specific target explicitly below.
    cli_args.slo_tpot_ms = tpot_targets[0]
    args = _build_engine_args(cli_args)
    policy = variant_table()["all_on"]
    condition_count = (len(seeds) * len(tpot_targets)
                       * len(arrival_rates) * len(output_lengths))
    print(f"Loading {args.model} once for {condition_count} "
          f"paired conditions ({cli_args.repeats} measured repetitions each)", flush=True)
    engine = create_arrival_engine(args, policy)
    pairs = []
    gpu_sampler = None
    try:
        moes = _validate_segment_layers(engine)
        if cli_args.sampling_mode == "greedy":
            _install_greedy_sampler(engine)
        print(f"GPU={torch.cuda.get_device_name()} MoE layers={len(moes)} "
              f"INT4 group size={cli_args.int4_group_size}; "
              f"TTFT target={cli_args.slo_ttft_ms:g} ms; "
              f"TPOT targets={','.join(f'{value:g}' for value in tpot_targets)} ms",
              flush=True)
        workload_args = SimpleNamespace(
            seed=0,
            num_seqs=cli_args.num_seqs,
            min_input_len=cli_args.input_length,
            max_input_len=cli_args.input_length,
            min_output_len=1,
            max_output_len=1,
            shared_prefix_len=0,
            prompts_file=None,
        )
        if source_prompts is None:
            prime_args = SimpleNamespace(**vars(workload_args))
            prime_args.seed = seeds[0]
            prime_prompts, _ = build_workload(prime_args)
        else:
            prime_prompts = prompts_by_seed[seeds[0]]
        print("Priming split/fused CUDA kernel shapes (not timed) ...", flush=True)
        _prime_segment_variants(engine, moes, prime_prompts)
        shape_recorder = DecodeBatchShapeRecorder(engine.model_runner)
        gpu_sampler = GpuClockSampler(cli_args.gpu_sample_interval_ms)
        gpu_sampler.start()
        all_on = policy
        condition_index = 0
        for tpot_target in tpot_targets:
            args.slo_tpot_ms = tpot_target
            for seed in seeds:
                if source_prompts is not None:
                    prompts = prompts_by_seed[seed]
                    request_metadata = requests_by_seed[seed]
                else:
                    workload_args.seed = seed
                    prompts, _ = build_workload(workload_args)
                    request_metadata = None
                for arrival_rate in arrival_rates:
                    for output_length in output_lengths:
                        sampling_params = [
                            SamplingParams(
                                temperature=(1.0 if cli_args.sampling_mode == "greedy"
                                             else cli_args.temperature),
                                ignore_eos=True,
                                max_tokens=output_length,
                            )
                            for _ in prompts
                        ]
                        arrivals = build_arrival_trace(
                            len(prompts), cli_args.arrival_mode, arrival_rate, seed + 1)
                        ttft_targets = build_ttft_slo_targets(
                            len(prompts), cli_args.slo_ttft_ms,
                            cli_args.slo_ttft_ms, seed + 2)
                        tpot_request_targets = [tpot_target] * len(prompts)
                        fingerprint = workload_fingerprint(
                            prompts, sampling_params, arrivals, ttft_targets,
                            tpot_request_targets, seed,
                            request_metadata=request_metadata)
                        condition = {
                            "seed": seed,
                            "arrival_mode": cli_args.arrival_mode,
                            "arrival_rate_requests_per_second": arrival_rate,
                            "num_requests": len(prompts),
                            "input_tokens": (cli_args.input_length
                                             if request_metadata is None else None),
                            "prompt_token_lengths": [len(prompt) for prompt in prompts],
                            "prompt_source": (
                                "synthetic_token_ids" if request_metadata is None
                                else "request_trace_prompts_with_synthetic_arrivals"),
                            "prompt_request_ids": ([request["request_id"]
                                                     for request in request_metadata]
                                                    if request_metadata is not None else None),
                            "prompt_token_hashes": ([request["prompt_token_hash"]
                                                      for request in request_metadata]
                                                     if request_metadata is not None else None),
                            "output_tokens": output_length,
                            "ttft_slo_ms": cli_args.slo_ttft_ms,
                            "tpot_slo_ms": tpot_target,
                            "prompt_fingerprint": _json_fingerprint(prompts),
                            "arrival_schedule_fingerprint": _json_fingerprint(arrivals),
                            "workload_fingerprint": fingerprint,
                        }
                        condition_index += 1
                        for repeat_index in range(cli_args.repeats):
                            order = (["segment_split", "segment_fused"]
                                     if (condition_index + repeat_index) % 2 == 0
                                     else ["segment_fused", "segment_split"])
                            runs = {}
                            output_sequences = {}
                            print(
                                f"\nTPOT={tpot_target:g} ms seed={seed} "
                                f"rate={arrival_rate:g} req/s output={output_length} "
                                f"repeat={repeat_index + 1}/{cli_args.repeats} "
                                f"order={' -> '.join(order)} "
                                f"fingerprint={fingerprint[:16]}",
                                flush=True,
                            )
                            for variant in order:
                                fused = variant == "segment_fused"
                                _set_fusion_mode(moes, fused)
                                args.seed = seed
                                args.no_moe_int4_fuse_gate_up = not fused
                                run_window = {}

                                def prepare_trace(trace_engine):
                                    _clear_idle_prefix_cache(trace_engine)
                                    shape_recorder.start()
                                    run_window["started"] = time.monotonic()

                                try:
                                    run = run_arrival_trace(
                                        args,
                                        prompts,
                                        sampling_params,
                                        arrivals,
                                        ttft_targets,
                                        all_on,
                                        engine=engine,
                                        request_metadata=request_metadata,
                                        capture_completion_tokens=True,
                                        before_trace=prepare_trace,
                                    )
                                finally:
                                    ended = time.monotonic()
                                    shape_summary = shape_recorder.stop()
                                if "started" not in run_window:
                                    raise RuntimeError("arrival trace telemetry did not start")
                                run["runtime_telemetry"] = {
                                    "gpu_clocks": gpu_sampler.summarize(
                                        run_window["started"], ended),
                                    "decode_batch_shapes": shape_summary,
                                }
                                if run["scheduler_metrics"]["prefix_cache_hit_tokens"] != 0:
                                    raise AssertionError(
                                        "paired workload unexpectedly hit prefix KV; "
                                        "cache state was not isolated")
                                if run["summary"]["output_tokens"] != len(prompts) * output_length:
                                    raise AssertionError(
                                        "run did not emit the requested fixed output length")
                                outputs = _ordered_outputs(run)
                                output_sequences[variant] = outputs
                                run["output_token_fingerprint"] = _output_fingerprint(outputs)
                                runs[variant] = run
                                summary = run["summary"]
                                goodput = summary["slo_goodput"]
                                gpu_clocks = run["runtime_telemetry"]["gpu_clocks"]
                                sm_clock = gpu_clocks["clock_summary"]["sm_clock_mhz"]["median"]
                                decode_shape = run["runtime_telemetry"]["decode_batch_shapes"]
                                decode_sizes = decode_shape["decode_batch_size_rows"]
                                sm_text = "n/a" if sm_clock is None else f"{sm_clock:.0f} MHz"
                                decode_text = (
                                    "n/a" if decode_sizes["median"] is None else
                                    f"p50/max={decode_sizes['median']:.0f}/"
                                    f"{decode_sizes['max']:.0f} rows"
                                )
                                print(
                                    f"  {variant:14s} output={summary['throughput_output_tok_per_s']:.2f} tok/s "
                                    f"joint-SLO={goodput['joint_slo_requests']}/"
                                    f"{summary['requests_completed']} "
                                    f"goodput={goodput['joint_slo_output_tokens_per_second']:.2f} tok/s "
                                    f"TTFT p50={summary['ttft']['p50'] * 1000:.1f} ms "
                                    f"TPOT p50={summary['tpot']['p50'] * 1000:.1f} ms | "
                                    f"SM clock median={sm_text} "
                                    f"({gpu_clocks['sample_count']} samples), "
                                    f"decode batch {decode_text}",
                                    flush=True,
                                )
                            mismatches = []
                            split_outputs = output_sequences["segment_split"]
                            fused_outputs = output_sequences["segment_fused"]
                            for request_index, (split_tokens, fused_tokens) in enumerate(
                                    zip(split_outputs, fused_outputs)):
                                if split_tokens != fused_tokens:
                                    first_diff = next((index for index, (a, b) in enumerate(
                                        zip(split_tokens, fused_tokens)) if a != b), None)
                                    mismatches.append({
                                        "request_index": request_index,
                                        "first_different_token_index": first_diff,
                                        "split_length": len(split_tokens),
                                        "fused_length": len(fused_tokens),
                                    })
                            parity = {
                                "all_requests_match": not mismatches,
                                "matching_requests": len(prompts) - len(mismatches),
                                "total_requests": len(prompts),
                                "mismatches": mismatches,
                                "split_output_fingerprint": runs["segment_split"][
                                    "output_token_fingerprint"],
                                "fused_output_fingerprint": runs["segment_fused"][
                                    "output_token_fingerprint"],
                                "sampling_mode": cli_args.sampling_mode,
                                "sampling_temperature": (
                                    1.0 if cli_args.sampling_mode == "greedy"
                                    else cli_args.temperature),
                            }
                            comparison = _pair_metrics(
                                runs["segment_split"], runs["segment_fused"])
                            pairs.append({
                                "condition": condition,
                                "repeat_index": repeat_index + 1,
                                "execution_order": order,
                                "output_parity": parity,
                                "comparison": comparison,
                                "runs": runs,
                            })
                            delta = comparison["output_throughput_delta_percent"]
                            print(
                                f"  fused vs split: throughput {delta:+.2f}% | "
                                f"SLO output goodput delta="
                                f"{comparison['joint_slo_output_goodput_delta_tok_s']:+.2f} tok/s | "
                                f"token parity={'PASS' if parity['all_requests_match'] else 'FAIL'}",
                                flush=True,
                            )

        groups = {}
        for pair in pairs:
            key = (pair["condition"]["tpot_slo_ms"],
                   pair["condition"]["arrival_rate_requests_per_second"],
                   pair["condition"]["output_tokens"])
            groups.setdefault(key, []).append(pair)
        by_load = {
            f"tpot_{tpot:g}_rate_{rate:g}_output_{output_length}": {
                "tpot_slo_ms": tpot,
                "arrival_rate_requests_per_second": rate,
                "output_tokens": output_length,
                "seed_count": len({pair["condition"]["seed"] for pair in group}),
                "replicate_count": len(group),
                **_aggregate(group),
            }
            for (tpot, rate, output_length), group in groups.items()
        }
        global_summary = {
            "paired_conditions": condition_count,
            "paired_measurements": len(pairs),
            "seed_count": len(seeds),
            "repetitions_per_condition": cli_args.repeats,
            "tpot_targets_ms": tpot_targets,
            "arrival_rates": arrival_rates,
            "output_lengths": output_lengths,
            **_aggregate(pairs),
        }
        payload = {
            "meta": {
                "model": args.model,
                "date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "gpu_clock_sample_interval_ms": cli_args.gpu_sample_interval_ms,
                "gpu_clock_fields": list(GpuClockSampler.FIELDS),
                **env_info(),
            },
            "benchmark_config": {
                "execution_mode": "mixed",
                "tensor_parallel_size": 1,
                "max_model_len": cli_args.max_model_len,
                "gpu_memory_utilization": cli_args.gpu_memory_utilization,
                "enforce_eager": True,
                "kv_cache_dtype": "auto",
                "quantization": "int4",
                "int4_group_size": cli_args.int4_group_size,
                "moe_int4_backend": "segment",
                "streaming_load": True,
                "kv_swap": False,
                "scheduler_policy": "all_on",
                "slo_ttft_ms": cli_args.slo_ttft_ms,
                "slo_tpot_ms": tpot_targets[0] if len(tpot_targets) == 1 else None,
                "slo_tpot_targets_ms": tpot_targets,
                "warmup_seqs": cli_args.warmup_seqs,
                "sampling_temperature": (
                    1.0 if cli_args.sampling_mode == "greedy"
                    else cli_args.temperature),
                "sampling_mode": cli_args.sampling_mode,
                "variants": ["segment_split", "segment_fused"],
                "prefix_cache_cleared_after_warmup": True,
                "alternating_pair_order": True,
                "repetitions_per_condition": cli_args.repeats,
                "kernel_shape_priming": "split/fused and batch sizes 1..num-seqs",
                "runtime_telemetry": {
                    "gpu_clock_sampling": "nvidia-smi during measured arrival trace",
                    "decode_batch_shapes": (
                        "model kind, decode rows, prefill sequence count, "
                        "prefill tokens, and decode tokens per model call"),
                },
                "require_output_token_parity": cli_args.require_output_parity,
            },
            "sweep": {
                "seeds": seeds,
                "tpot_targets_ms": tpot_targets,
                "arrival_mode": cli_args.arrival_mode,
                "arrival_rates_requests_per_second": arrival_rates,
                "num_requests_per_condition": cli_args.num_seqs,
                "input_tokens": cli_args.input_length,
                "prompts_file": cli_args.prompts_file,
                "prompt_source_sha256": trace_sha256,
                "real_prompt_source_arrivals_ignored": bool(cli_args.prompts_file),
                "output_lengths": output_lengths,
                "moe_layer_count": len(moes),
            },
            "summary": global_summary,
            "by_load": by_load,
            "pairs": pairs,
        }
        output_path = Path(cli_args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(f"\nSaved {output_path}")
        print("\nPaired sweep summary:")
        print("TPOT/load        seeds reps win/tie/loss  throughput delta median/range  "
              "SLO goodput delta median (tok/s)  token parity")
        for name, group in by_load.items():
            delta = group["output_throughput_delta_percent"]
            print(
                f"{name:<17} {group['seed_count']:>5} {group['replicate_count']:>4} "
                f"{group['throughput_fused_wins']}/"
                f"{group['throughput_ties']}/"
                f"{group['throughput_fused_losses']}  "
                f"{delta['median']:+.2f}% "
                f"[{delta['min']:+.2f}, {delta['max']:+.2f}]  "
                f"{group['joint_slo_output_goodput_delta_tok_s']['median']:+.2f}  "
                f"{group['output_token_parity_pairs']}/{group['seed_count']}",
                flush=True,
            )
        print(
            f"overall: throughput wins/ties/losses="
            f"{global_summary['throughput_fused_wins']}/"
            f"{global_summary['throughput_ties']}/"
            f"{global_summary['throughput_fused_losses']}; "
            f"median delta="
            f"{global_summary['output_throughput_delta_percent']['median']:+.2f}%; "
            f"output parity="
            f"{global_summary['output_token_parity_pairs']}/{len(pairs)}",
            flush=True,
        )
        if (cli_args.require_output_parity
                and global_summary["output_token_parity_failures"]):
            raise SystemExit("split/fused generated-token parity failed; see the saved JSON")
        if global_summary["output_token_parity_failures"]:
            print(
                "Note: some sequence tokens differ across continuous-arrival runs. "
                "Use the fixed-input MoE-layer correctness matrix to isolate kernel parity; "
                "this token check is diagnostic unless --require-output-parity is set.",
                flush=True,
            )
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

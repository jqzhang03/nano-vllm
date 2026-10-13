"""Diagnose and benchmark split/fused INT4 MoE with fixed model-call shapes.

Example (WSL nano-vllm environment)::

    python benchmarks/moe_int4_fixed_shape_diag.py \
        --model ~/huggingface/DeepSeek-V2-Lite \
        --seeds 7,17,42 --temperatures 0.6,0.8,1.0 \
        --output-lengths 64,128 --repeats 3

Each pair uses identical prompts and all requests arrive together. By default,
prompts match the seeded synthetic workload used by the earlier arrival sweeps;
``--prompt`` repeats one tokenizer-encoded text prompt instead. Natural random
and greedy sampling compare split/fused output tokens. Performance runs replay
the split token traces on both variants so every model input and output length
is equal. CUDA Graphs and multi-step decode are disabled. A mismatch triggers a
teacher-forced probe of full logits, MoE inputs/outputs, routes, and INT4
gate/up/down projection results at the first differing token position.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

SCRIPT_DIR = str(Path(__file__).resolve().parent)
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

try:
    from .moe_int4_fuse_sweep import (
        GpuClockSampler,
        _clear_idle_prefix_cache,
        _distribution,
        _ordered_outputs,
        _output_fingerprint,
        _pair_metrics,
        _set_fusion_mode,
        _validate_segment_layers,
        build_workload,
    )
    from .scheduling_ablation import (
        create_arrival_engine,
        run_arrival_trace,
        variant_table,
        workload_fingerprint,
    )
except ImportError:  # direct script execution from benchmarks/
    from moe_int4_fuse_sweep import (
        GpuClockSampler,
        _clear_idle_prefix_cache,
        _distribution,
        _ordered_outputs,
        _output_fingerprint,
        _pair_metrics,
        _set_fusion_mode,
        _validate_segment_layers,
        build_workload,
    )
    from scheduling_ablation import (
        create_arrival_engine,
        run_arrival_trace,
        variant_table,
        workload_fingerprint,
    )

from nanovllm import SamplingParams
from nanovllm.layers.moe import MoE
import nanovllm.layers.moe_int4_segment as int4_segment


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique positive integers")
    return result


def _parse_seeds(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated integer list") from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("--seeds must contain unique non-negative integers")
    return result


def _parse_temperatures(value: str) -> list[float]:
    try:
        result = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--temperatures must be a comma-separated number list") from exc
    if (not result or any(not math.isfinite(item) or item <= 1e-10 for item in result)
            or len(set(result)) != len(result)):
        raise ValueError("--temperatures must contain unique finite values greater than 0")
    return result


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _tensor_metrics(left: torch.Tensor | None,
                    right: torch.Tensor | None) -> dict[str, Any] | None:
    if left is None or right is None:
        return None
    if tuple(left.shape) != tuple(right.shape):
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    a = left.detach().float().cpu()
    b = right.detach().float().cpu()
    diff = (a - b).abs()
    scale = torch.maximum(a.abs(), b.abs()).clamp_min(1e-8)
    max_index = int(diff.reshape(-1).argmax()) if diff.numel() else None
    return {
        "shape_equal": True,
        "bitwise_equal": torch.equal(left, right),
        "elements": diff.numel(),
        "different_elements": int(torch.count_nonzero(diff)),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "rms_abs": float(torch.mean(diff.square()).sqrt()) if diff.numel() else 0.0,
        "max_rel": float((diff / scale).max()) if diff.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(diff)
                              / torch.linalg.vector_norm(a).clamp_min(1e-8)),
        "max_difference_index": max_index,
    }


def _top_logits(logits: torch.Tensor, temperature: float, count: int = 8) -> dict[str, Any]:
    values, ids = torch.topk(logits.detach().float().cpu(),
                             k=min(count, logits.numel()))
    return {
        "ids": ids.tolist(),
        "logits": values.tolist(),
        "temperature": temperature,
    }


class FixedShapeTraceController:
    """Trace model-call shapes, natural samples, or replay a fixed token path."""

    def __init__(self, engine, num_requests: int, input_length: int):
        self.engine = engine
        self.num_requests = num_requests
        self.input_length = input_length
        self.runner = engine.model_runner
        self.sampler = self.runner.sampler
        self._original_add_request = engine.add_request
        self._original_call = self.runner.call
        self._original_sampler_forward = self.sampler.forward
        self._active = False
        self._mode = "idle"
        self._greedy = False
        self._force_tokens: torch.Tensor | None = None
        self._force_index_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._probe_request_index: int | None = None
        self._capture_step: int | None = None
        self._request_index_by_seq_id: dict[int, int] = {}
        self._request_order: list[int] = []
        self._current_request_indices: list[int] = []
        self._current_kind: str | None = None
        self._current_prefill_rows = 0
        self._current_output_index = 0
        self._sample_calls = 0
        self._shape_events: list[dict[str, Any]] = []
        self._natural_sample_trace: list[dict[str, Any]] = []
        self._full_logits: torch.Tensor | None = None
        self._debug_sample: dict[str, Any] | None = None
        self._install()

    def _install(self) -> None:
        def add_request(prompt, sampling_params, submitted_at=None,
                        ttft_slo_ms=None, tpot_slo_ms=None):
            seq_id = self._original_add_request(
                prompt, sampling_params, submitted_at=submitted_at,
                ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
            if self._active:
                index = len(self._request_order)
                if index >= self.num_requests:
                    raise AssertionError("more requests were submitted than configured")
                self._request_order.append(seq_id)
                self._request_index_by_seq_id[seq_id] = index
            return seq_id

        def call(method_name, *args):
            if self._active and method_name == "run" and len(args) >= 2:
                seqs, kind = args[:2]
                indices = [self._request_index_by_seq_id[seq.seq_id] for seq in seqs]
                prefill = [seq for seq in seqs if seq.is_prefill]
                decode = [seq for seq in seqs if not seq.is_prefill]
                self._current_request_indices = indices
                self._current_kind = str(kind)
                self._current_prefill_rows = len(prefill)
                self._current_output_index = self._sample_calls
                self._shape_events.append({
                    "kind": str(kind),
                    "request_indices": list(indices),
                    "prefill_sequences": len(prefill),
                    "decode_sequences": len(decode),
                    "prefill_tokens": sum(seq.num_scheduled_tokens for seq in prefill),
                    "decode_tokens": sum(seq.num_scheduled_tokens for seq in decode),
                })
            try:
                return self._original_call(method_name, *args)
            finally:
                if self._active and method_name == "run":
                    self._current_request_indices = []
                    self._current_kind = None
                    self._current_prefill_rows = 0

        def sample_forward(logits: torch.Tensor, temperatures: torch.Tensor):
            sampled = self._original_sampler_forward(logits, temperatures)
            if not self._active:
                return sampled
            if self._greedy:
                sampled = logits.argmax(dim=-1)
            indices = list(self._current_request_indices)
            if logits.ndim != 2 or len(indices) != logits.size(0):
                raise AssertionError(
                    f"sampler rows {tuple(logits.shape)} do not match model-call rows "
                    f"{len(indices)}")
            output_index = self._sample_calls
            if self._mode == "natural":
                top_values, top_ids = torch.topk(logits.detach().float(), k=2, dim=-1)
                self._natural_sample_trace.append({
                    "output_index": output_index,
                    "request_indices": indices,
                    "top2_values": top_values.detach().cpu().tolist(),
                    "top2_ids": top_ids.detach().cpu().tolist(),
                    "sampled_tokens": sampled.detach().cpu().tolist(),
                })
            if self._capture_step == output_index and self._probe_request_index is not None:
                row = indices.index(self._probe_request_index)
                self._full_logits = logits[row].detach().float().cpu().clone()
                self._debug_sample = {
                    "request_index": self._probe_request_index,
                    "output_index": output_index,
                    "sampled_token": int(sampled[row]),
                    "top_logits": _top_logits(logits[row], float(temperatures[row])),
                }
            if self._force_tokens is not None:
                if output_index >= self._force_tokens.size(1):
                    raise AssertionError("model sampled beyond the forced output trace")
                key = tuple(indices)
                row_ids = self._force_index_cache.get(key)
                if row_ids is None:
                    row_ids = torch.tensor(key, dtype=torch.long,
                                           device=self._force_tokens.device)
                    self._force_index_cache[key] = row_ids
                sampled = self._force_tokens[row_ids, output_index]
            self._sample_calls += 1
            return sampled

        self.engine.add_request = add_request
        self.runner.call = call
        self.sampler.forward = sample_forward

    def begin(self, *, mode: str, force_tokens: list[list[int]] | None = None,
              probe_request_index: int | None = None,
              capture_step: int | None = None,
              greedy: bool = False) -> None:
        if self._active:
            raise RuntimeError("a fixed-shape trace is already active")
        if mode not in ("natural", "force"):
            raise ValueError(f"unknown trace mode {mode}")
        self._mode = mode
        self._greedy = greedy
        self._active = True
        self._force_tokens = (torch.tensor(force_tokens, dtype=torch.long, device="cuda")
                              if force_tokens is not None else None)
        if mode == "force" and self._force_tokens is None:
            raise ValueError("force mode requires a token trace")
        if self._force_tokens is not None and self._force_tokens.size(0) != self.num_requests:
            raise ValueError("forced token trace request count does not match the batch")
        self._probe_request_index = probe_request_index
        self._capture_step = capture_step
        self._request_index_by_seq_id.clear()
        self._request_order.clear()
        self._current_request_indices.clear()
        self._shape_events.clear()
        self._natural_sample_trace.clear()
        self._full_logits = None
        self._debug_sample = None
        self._sample_calls = 0
        self._force_index_cache.clear()

    def finish(self) -> dict[str, Any]:
        self._active = False
        histogram: Counter[tuple[str, int, int, int, int]] = Counter()
        for event in self._shape_events:
            histogram[(event["kind"], event["decode_sequences"],
                       event["prefill_sequences"], event["prefill_tokens"],
                       event["decode_tokens"])] += 1
        return {
            "sample_calls": self._sample_calls,
            "request_order": list(self._request_order),
            "shape_events": list(self._shape_events),
            "shape_histogram": [
                {
                    "kind": kind,
                    "decode_rows": decode_rows,
                    "prefill_sequences": prefill_sequences,
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": decode_tokens,
                    "model_calls": count,
                }
                for (kind, decode_rows, prefill_sequences, prefill_tokens, decode_tokens), count
                in sorted(histogram.items())
            ],
            "natural_sample_trace": list(self._natural_sample_trace),
            "full_logits": self._full_logits,
            "debug_sample": self._debug_sample,
        }

    def close(self) -> None:
        self._active = False
        self.engine.add_request = self._original_add_request
        self.runner.call = self._original_call
        self.sampler.forward = self._original_sampler_forward


class MoeTraceProbe:
    """Capture one request's MoE state during one forced decode position."""

    def __init__(self, controller: FixedShapeTraceController, moes: list[MoE],
                 input_length: int, probe_request_index: int, capture_step: int):
        self.controller = controller
        self.input_length = input_length
        self.probe_request_index = probe_request_index
        self.capture_step = capture_step
        self.snapshots: dict[int, dict[str, Any]] = {}
        self._handles = []
        self._route_originals: list[tuple[MoE, Any, bool]] = []
        self._active: dict[str, Any] | None = None
        self._closed = False
        self._old_mm = int4_segment.moe_segment_int4_mm
        self._old_gup = int4_segment.moe_segment_int4_gup
        self._install(moes)

    def _is_capture_step(self) -> bool:
        return (self.controller._active
                and self.controller._sample_calls == self.capture_step)

    def _row_index(self, rows: int) -> int:
        if self.controller._current_prefill_rows:
            if self.controller._current_prefill_rows != self.controller.num_requests:
                raise AssertionError("fixed-shape probe encountered mixed prefill/decode")
            row = self.probe_request_index * self.input_length + self.input_length - 1
        else:
            row = self.controller._current_request_indices.index(
                self.probe_request_index)
        if not 0 <= row < rows:
            raise AssertionError(f"probe row {row} is outside MoE input with {rows} rows")
        return row

    def _record_assignments(self, module: MoE, layer_index: int,
                            x: torch.Tensor, top_values: torch.Tensor,
                            top_ids: torch.Tensor) -> None:
        if self._active is None or self._active["layer_index"] != layer_index:
            return
        row = self._active["row"]
        flat_experts = top_ids.reshape(-1)
        target_rows = torch.arange(x.size(0), device=x.device).repeat_interleave(
            module.top_k)
        order = torch.argsort(flat_experts, stable=True)
        sorted_rows = target_rows[order]
        selected_positions = torch.nonzero(sorted_rows == row, as_tuple=False).flatten()
        sorted_experts = flat_experts[order][selected_positions]
        self._active["positions"] = selected_positions
        self._active["experts"] = sorted_experts.detach().cpu().tolist()
        self.snapshots[layer_index]["route"] = {
            "expert_ids_topk_order": top_ids[row].detach().cpu().tolist(),
            "weights_topk_order": top_values[row].detach().float().cpu().tolist(),
            "expert_ids_sorted_order": self._active["experts"],
        }

    def _store_projection(self, label: str, output: torch.Tensor) -> None:
        if self._active is None or self._active["positions"] is None:
            return
        positions = self._active["positions"]
        selected = output.index_select(0, positions).detach().float().cpu()
        data = {}
        for expert_id, row in zip(self._active["experts"], selected):
            data[str(expert_id)] = row.clone()
        self.snapshots[self._active["layer_index"]].setdefault(
            "projections", {})[label] = data

    def _install(self, moes: list[MoE]) -> None:
        for layer_index, module in enumerate(moes):
            self.snapshots[layer_index] = {}
            had_route_override = "_route" in module.__dict__
            old_route = module.__dict__.get("_route", None)
            original_route = module._route
            self._route_originals.append((module, old_route if had_route_override
                                          else original_route, had_route_override))

            def route(x, *, _module=module, _index=layer_index,
                      _original=original_route):
                values, ids = _original(x)
                if self._active is not None and self._active["layer_index"] == _index:
                    self._record_assignments(_module, _index, x, values, ids)
                return values, ids

            module._route = route

            def pre_hook(current_module, inputs, *, _index=layer_index):
                if not self._is_capture_step():
                    self._active = None
                    return
                x = inputs[0]
                row = self._row_index(x.size(0))
                self._active = {
                    "layer_index": _index,
                    "row": row,
                    "positions": None,
                    "experts": None,
                    "mm_calls": 0,
                    "fused": bool(current_module.fuse_int4_gate_up),
                }
                self.snapshots[_index]["input"] = (
                    x[row].detach().float().cpu().clone())

            def post_hook(current_module, inputs, output, *, _index=layer_index):
                if self._active is not None and self._active["layer_index"] == _index:
                    self.snapshots[_index]["output"] = (
                        output[self._active["row"]].detach().float().cpu().clone())
                    self._active = None

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))

        def mm(*args, **kwargs):
            output = self._old_mm(*args, **kwargs)
            if self._active is not None:
                call_index = self._active["mm_calls"]
                self._active["mm_calls"] += 1
                if self._active["fused"]:
                    label = "down" if call_index == 0 else f"mm_{call_index}"
                else:
                    label = ("gate", "up", "down")[call_index] \
                        if call_index < 3 else f"mm_{call_index}"
                if label in ("gate", "up", "down"):
                    self._store_projection(label, output)
            return output

        def gup(*args, **kwargs):
            output = self._old_gup(*args, **kwargs)
            if self._active is not None and self._active["positions"] is not None:
                intermediate = int(args[7])
                selected = output.index_select(
                    0, self._active["positions"]).detach().float().cpu()
                experts = self._active["experts"]
                gate_data = {}
                up_data = {}
                for expert_id, row in zip(experts, selected):
                    gate_data[str(expert_id)] = row[:intermediate].clone()
                    up_data[str(expert_id)] = row[intermediate:].clone()
                projections = self.snapshots[self._active["layer_index"]].setdefault(
                    "projections", {})
                projections["gate"] = gate_data
                projections["up"] = up_data
            return output

        int4_segment.moe_segment_int4_mm = mm
        int4_segment.moe_segment_int4_gup = gup

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in self._handles:
            handle.remove()
        for module, old_route, had_route_override in self._route_originals:
            if had_route_override:
                module._route = old_route
            else:
                delattr(module, "_route")
        int4_segment.moe_segment_int4_mm = self._old_mm
        int4_segment.moe_segment_int4_gup = self._old_gup
        self._active = None
        self.controller = None


def _projection_metrics(left: dict[str, torch.Tensor] | None,
                        right: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    left = left or {}
    right = right or {}
    left_ids, right_ids = set(left), set(right)
    common_ids = sorted(left_ids & right_ids, key=int)
    joined_left = torch.cat([left[key].reshape(-1) for key in common_ids]) \
        if common_ids else None
    joined_right = torch.cat([right[key].reshape(-1) for key in common_ids]) \
        if common_ids else None
    return {
        "matching_expert_ids": [int(key) for key in common_ids],
        "left_only_expert_ids": sorted(int(key) for key in left_ids - right_ids),
        "right_only_expert_ids": sorted(int(key) for key in right_ids - left_ids),
        "values": _tensor_metrics(joined_left, joined_right),
    }


def _compare_moe_probes(split: MoeTraceProbe, fused: MoeTraceProbe) -> dict[str, Any]:
    layer_rows = []
    first_input_difference = None
    first_route_difference = None
    first_gate_up_difference = None
    first_down_difference = None
    first_output_difference = None
    for layer_index in sorted(set(split.snapshots) | set(fused.snapshots)):
        left = split.snapshots.get(layer_index, {})
        right = fused.snapshots.get(layer_index, {})
        input_metrics = _tensor_metrics(left.get("input"), right.get("input"))
        output_metrics = _tensor_metrics(left.get("output"), right.get("output"))
        left_route = left.get("route")
        right_route = right.get("route")
        route_equal = (left_route is not None and right_route is not None
                       and left_route["expert_ids_topk_order"]
                       == right_route["expert_ids_topk_order"])
        route_weights = _tensor_metrics(
            torch.tensor(left_route["weights_topk_order"]) if left_route else None,
            torch.tensor(right_route["weights_topk_order"]) if right_route else None)
        left_proj = left.get("projections", {})
        right_proj = right.get("projections", {})
        projection_metrics = {
            label: _projection_metrics(left_proj.get(label), right_proj.get(label))
            for label in ("gate", "up", "down")
        }
        if first_input_difference is None and input_metrics is not None \
                and not input_metrics.get("bitwise_equal", False):
            first_input_difference = layer_index
        if first_route_difference is None and left_route is not None \
                and right_route is not None and not route_equal:
            first_route_difference = layer_index
        gate_up_differs = any(
            item["values"] is not None and not item["values"].get("bitwise_equal", False)
            for item in (projection_metrics["gate"], projection_metrics["up"]))
        if first_gate_up_difference is None and gate_up_differs:
            first_gate_up_difference = layer_index
        down_differs = (projection_metrics["down"]["values"] is not None
                        and not projection_metrics["down"]["values"].get(
                            "bitwise_equal", False))
        if first_down_difference is None and down_differs:
            first_down_difference = layer_index
        if first_output_difference is None and output_metrics is not None \
                and not output_metrics.get("bitwise_equal", False):
            first_output_difference = layer_index
        layer_rows.append({
            "moe_layer": layer_index,
            "input": input_metrics,
            "routes_equal": route_equal,
            "route_weights": route_weights,
            "projections": projection_metrics,
            "output": output_metrics,
        })
    return {
        "first_moe_input_difference_layer": first_input_difference,
        "first_router_expert_difference_layer": first_route_difference,
        "first_gate_or_up_projection_difference_layer": first_gate_up_difference,
        "first_down_projection_difference_layer": first_down_difference,
        "first_moe_output_difference_layer": first_output_difference,
        "layers": layer_rows,
    }


def _natural_top2(trace: list[dict[str, Any]], request_index: int,
                  output_index: int) -> dict[str, Any] | None:
    item = next((entry for entry in trace
                 if entry["output_index"] == output_index), None)
    if item is None:
        return None
    row = item["request_indices"].index(request_index)
    return {
        "top2_ids": item["top2_ids"][row],
        "top2_logits": item["top2_values"][row],
        "sampled_token": item["sampled_tokens"][row],
    }


def _find_mismatches(left: list[list[int]], right: list[list[int]]) -> list[dict[str, Any]]:
    if len(left) != len(right):
        raise AssertionError("paired natural runs used different request counts")
    mismatches = []
    for request_index, (a, b) in enumerate(zip(left, right)):
        if len(a) != len(b):
            raise AssertionError("paired natural runs emitted different output lengths")
        positions = [index for index, (left_id, right_id) in enumerate(zip(a, b))
                     if left_id != right_id]
        if positions:
            mismatches.append({
                "request_index": request_index,
                "first_different_token_index": positions[0],
                "different_token_count": len(positions),
                "split_length": len(a),
                "fused_length": len(b),
            })
    return mismatches


def _validate_shape(trace: dict[str, Any], num_requests: int,
                    input_length: int, output_length: int) -> None:
    if len(trace["request_order"]) != num_requests:
        raise AssertionError("trace did not submit the configured request count")
    if trace["sample_calls"] != output_length:
        raise AssertionError(
            f"expected {output_length} sampler calls, saw {trace['sample_calls']}")
    events = trace["shape_events"]
    if len(events) != output_length:
        raise AssertionError(
            f"expected one prefill plus {output_length - 1} decode calls, saw {len(events)}")
    prefill = [event for event in events if event["prefill_sequences"]]
    decode = [event for event in events if event["decode_sequences"]]
    if len(prefill) != 1 or len(decode) != output_length - 1:
        raise AssertionError("fixed batch was split across prefill/decode model calls")
    if prefill[0]["prefill_sequences"] != num_requests \
            or prefill[0]["prefill_tokens"] != num_requests * input_length \
            or prefill[0]["decode_sequences"] != 0:
        raise AssertionError("prefill model-call shape did not match the fixed workload")
    for event in decode:
        if (event["decode_sequences"] != num_requests
                or event["decode_tokens"] != num_requests
                or event["prefill_sequences"] != 0):
            raise AssertionError("decode batch did not retain the full fixed row count")


def _model_input_fingerprint(prompts: list[list[int]],
                             outputs: list[list[int]]) -> str:
    inputs = [prompt + output[:-1] for prompt, output in zip(prompts, outputs)]
    return _digest(inputs)


def _compact_run(run: dict[str, Any], outputs: list[list[int]],
                 trace: dict[str, Any], prompts: list[list[int]],
                 clocks: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "wall_seconds": run["wall_seconds"],
        "summary": run["summary"],
        "scheduler_metrics": run["scheduler_metrics"],
        "per_request": run["per_request"],
        "output_token_fingerprint": _output_fingerprint(outputs),
        "model_input_fingerprint": _model_input_fingerprint(prompts, outputs),
        "shape": {
            "model_runner_calls": len(trace["shape_events"]),
            "decode_model_calls": sum(
                event["decode_sequences"] > 0 for event in trace["shape_events"]),
            "shape_histogram": trace["shape_histogram"],
        },
        "gpu_clocks": clocks,
    }


def _run_trace(engine, controller: FixedShapeTraceController, args,
               prompts: list[list[int]], sampling: list[SamplingParams],
               seed: int, output_length: int, *, mode: str,
               force_tokens: list[list[int]] | None = None,
               probe_request_index: int | None = None,
               capture_step: int | None = None,
               greedy: bool = False) -> tuple[dict[str, Any], list[list[int]], dict[str, Any]]:
    args.seed = seed
    policy = variant_table()["baseline"]
    trace_holder: dict[str, Any] = {}

    def prepare(active_engine):
        _clear_idle_prefix_cache(active_engine)
        controller.begin(
            mode=mode,
            force_tokens=force_tokens,
            probe_request_index=probe_request_index,
            capture_step=capture_step,
            greedy=greedy,
        )

    try:
        raw_run = run_arrival_trace(
            args,
            prompts,
            sampling,
            [0.0] * len(prompts),
            [args.slo_ttft_ms] * len(prompts),
            policy,
            engine=engine,
            capture_completion_tokens=True,
            before_trace=prepare,
        )
    finally:
        trace_holder = controller.finish()
    outputs = _ordered_outputs(raw_run)
    if any(len(output) != output_length for output in outputs):
        raise AssertionError("a request did not emit the requested fixed output length")
    if raw_run["summary"]["output_tokens"] != len(prompts) * output_length:
        raise AssertionError("the run emitted an unexpected number of output tokens")
    _validate_shape(trace_holder, len(prompts), controller.input_length, output_length)
    return raw_run, outputs, trace_holder


def _build_prompt(engine, text: str, input_length: int) -> list[int]:
    tokens = engine.tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError("--prompt tokenizes to an empty sequence")
    while len(tokens) < input_length:
        tokens.extend(engine.tokenizer.encode(" " + text, add_special_tokens=False))
    return list(tokens[:input_length])


def _engine_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=os.path.abspath(os.path.expanduser(cli.model)),
        tp=1,
        max_model_len=cli.max_model_len,
        gpu_memory_utilization=cli.gpu_memory_utilization,
        enforce_eager=True,
        kv_cache_dtype="auto",
        quantization="int4",
        nvfp4_backend="software",
        int4_group_size=64,
        moe_int4_backend="segment",
        no_moe_int4_fuse_gate_up=False,
        streaming_load=True,
        no_swap_kv=True,
        kv_swap_space_gb=0.0,
        admission_window=16,
        aging_timeout_ms=2000.0,
        prefill_reserve_tokens=256,
        slo_tpot_ms=cli.slo_tpot_ms,
        max_decode_steps=1,
        max_prefill_chunk_tokens=max(4096, cli.num_seqs * cli.input_length),
        queue_depth_for_full_prefill=16,
        preempt_prefill_tps=10000.0,
        preempt_kv_transfer_gbps=12.0,
        tpot_prefill_throttle_margin=0.5,
        tpot_throttle_max_waiting=16,
        warmup_seqs=0,
        seed=0,
        slo_ttft_ms=cli.slo_ttft_ms,
    )


def _condition_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [pair["comparison"]["output_throughput_delta_percent"] for pair in pairs]
    defined = [float(value) for value in deltas if value is not None]
    return {
        "pairs": len(pairs),
        "output_throughput_delta_percent": _distribution(defined),
        "fused_wins": sum(value > 0 for value in defined),
        "ties": sum(value == 0 for value in defined),
        "fused_losses": sum(value < 0 for value in defined),
        "tpot_p50_delta_percent": _distribution(
            [pair["comparison"]["tpot_p50_delta_percent"] for pair in pairs]),
        "tpot_p99_delta_percent": _distribution(
            [pair["comparison"]["tpot_p99_delta_percent"] for pair in pairs]),
        "ttft_p50_delta_percent": _distribution(
            [pair["comparison"]["ttft_p50_delta_percent"] for pair in pairs]),
    }


def _sampler_difference(left: torch.Tensor | None, right: torch.Tensor | None,
                        temperature: float) -> dict[str, Any]:
    result = {"logits": _tensor_metrics(left, right)}
    if left is None or right is None or tuple(left.shape) != tuple(right.shape):
        return result
    a = left.float()
    b = right.float()
    p = torch.softmax(a / temperature, dim=-1)
    q = torch.softmax(b / temperature, dim=-1)
    mean = (p + q) * 0.5
    js = 0.5 * (torch.sum(p * (torch.log(p.clamp_min(1e-30))
                               - torch.log(mean.clamp_min(1e-30))))
                + torch.sum(q * (torch.log(q.clamp_min(1e-30))
                                 - torch.log(mean.clamp_min(1e-30)))))
    result.update({
        "jensen_shannon_divergence_nats": float(js),
        "total_variation_distance": float(0.5 * torch.sum((p - q).abs())),
        "max_probability_delta": float((p - q).abs().max()),
        "split_top_logits": _top_logits(left, temperature),
        "fused_top_logits": _top_logits(right, temperature),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.expanduser(
        "~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--prompt", default=None,
                        help="repeat one text prompt for all requests; default uses the "
                             "seeded synthetic prompts from the arrival-sweep harness")
    parser.add_argument("--num-seqs", type=int, default=24)
    parser.add_argument("--input-length", type=int, default=64)
    parser.add_argument("--output-lengths", default="64,128")
    parser.add_argument("--seeds", default="7,17,42")
    parser.add_argument("--temperatures", default="0.6,0.8,1.0",
                        help="SamplingParams.temperature values; each is paired identically")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=2,
                        help="full fixed-batch/output-length warmups per fusion variant")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--slo-ttft-ms", type=float, default=60000.0,
                        help="permissive report-only target for fixed-batch runs")
    parser.add_argument("--slo-tpot-ms", type=float, default=10000.0,
                        help="permissive report-only target for fixed-batch runs")
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=250)
    parser.add_argument("--diagnostic-sentinel", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="probe seed 42 / request 23 / greedy output token 48 when covered")
    parser.add_argument("--output", default="results/moe_int4_fixed_shape_diag_20261012.json")
    cli = parser.parse_args()

    output_lengths = _parse_ints(cli.output_lengths, "--output-lengths")
    seeds = _parse_seeds(cli.seeds)
    temperatures = _parse_temperatures(cli.temperatures)
    if cli.num_seqs < 1 or cli.input_length < 1:
        raise ValueError("--num-seqs and --input-length must be positive")
    if cli.repeats < 1 or cli.warmup_repeats < 1:
        raise ValueError("--repeats and --warmup-repeats must be positive")
    if cli.max_model_len < cli.input_length + max(output_lengths):
        raise ValueError("--max-model-len must cover input plus maximum output length")
    if cli.gpu_sample_interval_ms < 250:
        raise ValueError("--gpu-sample-interval-ms must be at least 250")
    if not torch.cuda.is_available():
        raise RuntimeError("this full-model diagnostic requires a CUDA GPU")

    args = _engine_args(cli)
    total_conditions = len(seeds) * len(temperatures) * len(output_lengths)
    print(f"Loading {args.model}; {total_conditions} fixed-shape conditions, "
          f"{cli.repeats} measured pairs each", flush=True)
    engine = create_arrival_engine(args, variant_table()["baseline"])
    controller = None
    clocks = None
    try:
        moes = _validate_segment_layers(engine)
        prompts_by_seed: dict[int, list[list[int]]] = {}
        for seed in seeds:
            if cli.prompt is not None:
                prompt_ids = _build_prompt(engine, cli.prompt, cli.input_length)
                prompts_by_seed[seed] = [list(prompt_ids) for _ in range(cli.num_seqs)]
            else:
                workload_args = SimpleNamespace(
                    seed=seed,
                    num_seqs=cli.num_seqs,
                    min_input_len=cli.input_length,
                    max_input_len=cli.input_length,
                    min_output_len=1,
                    max_output_len=1,
                    shared_prefix_len=0,
                    prompts_file=None,
                )
                prompts_by_seed[seed], _ = build_workload(workload_args)
        prompt_fingerprints = {
            seed: _digest(prompts_by_seed[seed]) for seed in seeds
        }
        controller = FixedShapeTraceController(engine, cli.num_seqs, cli.input_length)
        print(f"GPU={torch.cuda.get_device_name(0)} MoE layers={len(moes)} "
              f"batch={cli.num_seqs} input={cli.input_length} "
              f"prompt_sha256(seed={seeds[0]})="
              f"{prompt_fingerprints[seeds[0]][:16]}", flush=True)

        natural_pairs: list[dict[str, Any]] = []
        greedy_pairs: list[dict[str, Any]] = []
        reference_outputs: dict[tuple[int, float, int], list[list[int]]] = {}
        greedy_reference_outputs: dict[tuple[int, int], list[list[int]]] = {}
        natural_trace_by_condition: dict[tuple[int, float, int], dict[str, Any]] = {}
        condition_counter = 0

        # First compare native sampling. This is correctness/diagnostic data only.
        for output_length in output_lengths:
            for seed in seeds:
                for temperature in temperatures:
                    key = (seed, temperature, output_length)
                    prompts = prompts_by_seed[seed]
                    sampling = [SamplingParams(
                        temperature=temperature, ignore_eos=True,
                        max_tokens=output_length) for _ in prompts]
                    arrivals = [0.0] * len(prompts)
                    workload_fp = workload_fingerprint(
                        prompts, sampling, arrivals, [args.slo_ttft_ms] * len(prompts),
                        [args.slo_tpot_ms] * len(prompts), seed)
                    condition = {
                        "seed": seed,
                        "temperature": temperature,
                        "sampling_mode": "random",
                        "num_requests": cli.num_seqs,
                        "input_tokens_per_request": cli.input_length,
                        "output_tokens_per_request": output_length,
                        "prompt_fingerprint": prompt_fingerprints[seed],
                        "workload_fingerprint": workload_fp,
                        "arrival_times_seconds": "all zero",
                    }
                    order = (["segment_split", "segment_fused"]
                             if condition_counter % 2 == 0 else
                             ["segment_fused", "segment_split"])
                    condition_counter += 1
                    runs = {}
                    outputs_by_variant = {}
                    sample_traces = {}
                    for variant in order:
                        fused = variant == "segment_fused"
                        _set_fusion_mode(moes, fused)
                        raw, outputs, trace = _run_trace(
                            engine, controller, args, prompts, sampling, seed,
                            output_length, mode="natural")
                        outputs_by_variant[variant] = outputs
                        sample_traces[variant] = trace["natural_sample_trace"]
                        runs[variant] = _compact_run(raw, outputs, trace, prompts, None)
                        print(f"natural {variant} seed={seed} temp={temperature:g} "
                              f"out={output_length} wall={raw['wall_seconds']:.2f}s "
                              f"shape={cli.num_seqs}xdecode", flush=True)

                    split_outputs = outputs_by_variant["segment_split"]
                    fused_outputs = outputs_by_variant["segment_fused"]
                    mismatches = _find_mismatches(split_outputs, fused_outputs)
                    reference_outputs[key] = split_outputs
                    natural_trace_by_condition[key] = {
                        "condition": condition,
                        "order": order,
                        "output_parity": {
                            "all_requests_match": not mismatches,
                            "matching_requests": cli.num_seqs - len(mismatches),
                            "total_requests": cli.num_seqs,
                            "mismatches": mismatches,
                            "split_output_fingerprint": _output_fingerprint(split_outputs),
                            "fused_output_fingerprint": _output_fingerprint(fused_outputs),
                        },
                    }
                    first = (min(mismatches,
                                 key=lambda item: (item["first_different_token_index"],
                                                   item["request_index"]))
                             if mismatches else None)
                    divergence = None
                    if first is not None:
                        request_index = first["request_index"]
                        output_index = first["first_different_token_index"]
                        divergence = {
                            **first,
                            "split_natural_logits": _natural_top2(
                                sample_traces["segment_split"], request_index, output_index),
                            "fused_natural_logits": _natural_top2(
                                sample_traces["segment_fused"], request_index, output_index),
                        }
                    natural_pairs.append({
                        "condition": condition,
                        "execution_order": order,
                        "output_parity": natural_trace_by_condition[key]["output_parity"],
                        "first_divergence": divergence,
                        "runs": runs,
                    })
                    status = "match" if first is None else (
                        f"first diff req={first['request_index']} "
                        f"token={first['first_different_token_index']}")
                    print(f"  parity={cli.num_seqs - len(mismatches)}/{cli.num_seqs} "
                          f"{status}", flush=True)

        # Greedy parity directly probes the previous argmax divergence while
        # retaining random seed/temperature coverage above.
        greedy_counter = 0
        for output_length in output_lengths:
            for seed in seeds:
                prompts = prompts_by_seed[seed]
                sampling = [SamplingParams(
                    temperature=1.0, ignore_eos=True,
                    max_tokens=output_length) for _ in prompts]
                arrivals = [0.0] * len(prompts)
                workload_fp = workload_fingerprint(
                    prompts, sampling, arrivals, [args.slo_ttft_ms] * len(prompts),
                    [args.slo_tpot_ms] * len(prompts), seed)
                condition = {
                    "seed": seed,
                    "temperature": 1.0,
                    "sampling_mode": "greedy",
                    "num_requests": cli.num_seqs,
                    "input_tokens_per_request": cli.input_length,
                    "output_tokens_per_request": output_length,
                    "prompt_fingerprint": prompt_fingerprints[seed],
                    "workload_fingerprint": workload_fp,
                    "arrival_times_seconds": "all zero",
                }
                order = (["segment_split", "segment_fused"]
                         if greedy_counter % 2 == 0 else
                         ["segment_fused", "segment_split"])
                greedy_counter += 1
                outputs_by_variant = {}
                sample_traces = {}
                runs = {}
                for variant in order:
                    fused = variant == "segment_fused"
                    _set_fusion_mode(moes, fused)
                    raw, outputs, trace = _run_trace(
                        engine, controller, args, prompts, sampling, seed,
                        output_length, mode="natural", greedy=True)
                    outputs_by_variant[variant] = outputs
                    sample_traces[variant] = trace["natural_sample_trace"]
                    runs[variant] = _compact_run(raw, outputs, trace, prompts, None)
                    print(f"greedy {variant} seed={seed} out={output_length} "
                          f"wall={raw['wall_seconds']:.2f}s", flush=True)
                split_outputs = outputs_by_variant["segment_split"]
                fused_outputs = outputs_by_variant["segment_fused"]
                mismatches = _find_mismatches(split_outputs, fused_outputs)
                greedy_reference_outputs[(seed, output_length)] = split_outputs
                first = (min(mismatches,
                             key=lambda item: (item["first_different_token_index"],
                                               item["request_index"]))
                         if mismatches else None)
                divergence = None
                if first is not None:
                    request_index = first["request_index"]
                    output_index = first["first_different_token_index"]
                    divergence = {
                        **first,
                        "split_natural_logits": _natural_top2(
                            sample_traces["segment_split"], request_index, output_index),
                        "fused_natural_logits": _natural_top2(
                            sample_traces["segment_fused"], request_index, output_index),
                    }
                parity = {
                    "all_requests_match": not mismatches,
                    "matching_requests": cli.num_seqs - len(mismatches),
                    "total_requests": cli.num_seqs,
                    "mismatches": mismatches,
                    "split_output_fingerprint": _output_fingerprint(split_outputs),
                    "fused_output_fingerprint": _output_fingerprint(fused_outputs),
                }
                greedy_pairs.append({
                    "condition": condition,
                    "execution_order": order,
                    "output_parity": parity,
                    "first_divergence": divergence,
                    "runs": runs,
                })
                status = "match" if first is None else (
                    f"first diff req={first['request_index']} "
                    f"token={first['first_different_token_index']}")
                print(f"  greedy parity={cli.num_seqs - len(mismatches)}/"
                      f"{cli.num_seqs} {status}", flush=True)

        # For each mismatching condition, replay the split token path through the
        # first divergent output position and compare logits and MoE internals.
        detailed_diagnostics = []
        diagnostic_jobs = []
        for pair in natural_pairs:
            if pair["first_divergence"] is not None:
                condition = pair["condition"]
                key = (condition["seed"], condition["temperature"],
                       condition["output_tokens_per_request"])
                diagnostic_jobs.append((pair, reference_outputs[key], False,
                                        "natural_token_mismatch"))
        for pair in greedy_pairs:
            if pair["first_divergence"] is not None:
                condition = pair["condition"]
                key = (condition["seed"], condition["output_tokens_per_request"])
                diagnostic_jobs.append((pair, greedy_reference_outputs[key], True,
                                        "greedy_token_mismatch"))
        sentinel_coverage = {
            "enabled": cli.diagnostic_sentinel,
            "seed": 42,
            "request_index": 23,
            "output_token_index": 48,
            "scheduled": False,
        }
        if cli.diagnostic_sentinel:
            sentinel_key = (42, 64)
            if (sentinel_key in greedy_reference_outputs
                    and cli.num_seqs > sentinel_coverage["request_index"]):
                source_pair = next(
                    pair for pair in greedy_pairs
                    if pair["condition"]["seed"] == sentinel_key[0]
                    and pair["condition"]["output_tokens_per_request"] == sentinel_key[1])
                sentinel_pair = {
                    "condition": source_pair["condition"],
                    "execution_order": source_pair["execution_order"],
                    "first_divergence": {
                        "request_index": sentinel_coverage["request_index"],
                        "first_different_token_index": sentinel_coverage[
                            "output_token_index"],
                    },
                }
                diagnostic_jobs.append((
                    sentinel_pair,
                    greedy_reference_outputs[sentinel_key],
                    True,
                    "historical_mismatch_sentinel",
                ))
                sentinel_coverage["scheduled"] = True
        for pair, reference, is_greedy, diagnostic_reason in diagnostic_jobs:
            divergence = pair["first_divergence"]
            condition = pair["condition"]
            prompts = prompts_by_seed[condition["seed"]]
            request_index = divergence["request_index"]
            output_index = divergence["first_different_token_index"]
            replay_length = output_index + 1
            ref = [row[:replay_length] for row in reference]
            sampling = [SamplingParams(
                temperature=condition["temperature"], ignore_eos=True,
                max_tokens=replay_length) for _ in prompts]
            branch_data = {}
            order = pair["execution_order"]
            for variant in order:
                fused = variant == "segment_fused"
                _set_fusion_mode(moes, fused)
                probe = MoeTraceProbe(
                    controller, moes, cli.input_length, request_index, output_index)
                try:
                    raw, outputs, trace = _run_trace(
                        engine, controller, args, prompts, sampling,
                        condition["seed"], replay_length,
                        mode="force", force_tokens=ref,
                        probe_request_index=request_index, capture_step=output_index,
                        greedy=is_greedy)
                    if outputs != ref:
                        raise AssertionError("teacher-forced diagnostic changed its token path")
                    branch_data[variant] = {
                        "raw": raw,
                        "trace": trace,
                        "moe_probe": probe,
                    }
                finally:
                    probe.close()
            split_data = branch_data["segment_split"]
            fused_data = branch_data["segment_fused"]
            split_trace = split_data["trace"]
            fused_trace = fused_data["trace"]
            logits_metrics = _sampler_difference(
                split_trace["full_logits"], fused_trace["full_logits"],
                condition["temperature"])
            moe_metrics = _compare_moe_probes(
                split_data["moe_probe"], fused_data["moe_probe"])
            detailed_diagnostics.append({
                "condition": condition,
                "diagnostic_reason": diagnostic_reason,
                "sampling_mode": "greedy" if is_greedy else "random",
                "request_index": request_index,
                "first_different_output_token_index": output_index,
                "forced_common_prefix_length": output_index,
                "replay_output_length": replay_length,
                "teacher_forced_path_fingerprint": _output_fingerprint(ref),
                "raw_sampler_candidate_tokens": {
                    "split": split_trace["debug_sample"],
                    "fused": fused_trace["debug_sample"],
                },
                "full_logit_comparison": logits_metrics,
                "moe_layer_comparison": moe_metrics,
                "shape_histograms_equal": (
                    split_trace["shape_histogram"] == fused_trace["shape_histogram"]),
            })
            print(f"diagnosed seed={condition['seed']} temp={condition['temperature']:g} "
                  f"out={condition['output_tokens_per_request']} token={output_index}: "
                  f"first MoE output diff="
                  f"{moe_metrics['first_moe_output_difference_layer']}", flush=True)

        # Full 128-token, 24-row warmups compile and exercise both kernel paths.
        warmup_length = max(output_lengths)
        warmup_prompts = prompts_by_seed[seeds[0]]
        warmup_params = [SamplingParams(
            temperature=0.8, ignore_eos=True, max_tokens=warmup_length)
            for _ in warmup_prompts]
        print(f"Full-batch warmup: {cli.warmup_repeats} x {warmup_length} tokens "
              "per split/fused variant", flush=True)
        for fused in (False, True):
            _set_fusion_mode(moes, fused)
            for warmup_index in range(cli.warmup_repeats):
                engine.generate(warmup_prompts, warmup_params, use_tqdm=False)
                _clear_idle_prefix_cache(engine)
                torch.cuda.synchronize()
                print(f"  warmup {'fused' if fused else 'split'} "
                      f"{warmup_index + 1}/{cli.warmup_repeats} done", flush=True)

        clocks = GpuClockSampler(cli.gpu_sample_interval_ms)
        clocks.start()
        performance_pairs: list[dict[str, Any]] = []
        pair_counter = 0
        try:
            for output_length in output_lengths:
                for seed in seeds:
                    for temperature in temperatures:
                        key = (seed, temperature, output_length)
                        prompts = prompts_by_seed[seed]
                        reference = reference_outputs[key]
                        sampling = [SamplingParams(
                            temperature=temperature, ignore_eos=True,
                            max_tokens=output_length) for _ in prompts]
                        condition = natural_trace_by_condition[key]["condition"]
                        for repeat_index in range(cli.repeats):
                            order = (["segment_split", "segment_fused"]
                                     if pair_counter % 2 == 0 else
                                     ["segment_fused", "segment_split"])
                            pair_counter += 1
                            run_results = {}
                            for variant in order:
                                fused = variant == "segment_fused"
                                _set_fusion_mode(moes, fused)
                                started = time.monotonic()
                                raw, outputs, trace = _run_trace(
                                    engine, controller, args, prompts, sampling, seed,
                                    output_length, mode="force", force_tokens=reference)
                                ended = time.monotonic()
                                if outputs != reference:
                                    raise AssertionError(
                                        f"{variant} did not replay the shared token trace")
                                clock_summary = clocks.summarize(started, ended)
                                run_results[variant] = _compact_run(
                                    raw, outputs, trace, prompts, clock_summary)
                                print(f"perf {variant} seed={seed} temp={temperature:g} "
                                      f"out={output_length} repeat={repeat_index + 1}/"
                                      f"{cli.repeats} wall={raw['wall_seconds']:.2f}s "
                                      f"tok/s={raw['summary']['throughput_output_tok_per_s']:.2f} "
                                      f"shape={cli.num_seqs}xdecode", flush=True)
                            split_run = run_results["segment_split"]
                            fused_run = run_results["segment_fused"]
                            if (split_run["model_input_fingerprint"]
                                    != fused_run["model_input_fingerprint"]):
                                raise AssertionError("paired model inputs were not identical")
                            if split_run["shape"]["shape_histogram"] \
                                    != fused_run["shape"]["shape_histogram"]:
                                raise AssertionError("paired model-call shape histograms differed")
                            comparison = _pair_metrics(split_run, fused_run)
                            performance_pairs.append({
                                "condition": condition,
                                "repeat_index": repeat_index + 1,
                                "execution_order": order,
                                "shared_output_trace_fingerprint": _output_fingerprint(reference),
                                "model_input_fingerprint": split_run["model_input_fingerprint"],
                                "shape_histograms_equal": True,
                                "output_parity": True,
                                "comparison": comparison,
                                "runs": run_results,
                            })
                            print(f"  fused throughput delta="
                                  f"{comparison['output_throughput_delta_percent']:+.2f}% "
                                  f"TPOT p50 delta="
                                  f"{comparison['tpot_p50_delta_percent']:+.2f}%", flush=True)
        finally:
            clocks.stop()

        grouped = defaultdict(list)
        for pair in performance_pairs:
            key = (pair["condition"]["output_tokens_per_request"],
                   pair["condition"]["temperature"])
            grouped[key].append(pair)
        aggregate = {
            "all": _condition_metrics(performance_pairs),
            "by_output_length_and_temperature": {
                f"out={output_length},temp={temperature:g}": _condition_metrics(pairs)
                for (output_length, temperature), pairs in sorted(grouped.items())
            },
        }
        result = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": "moe_int4_fixed_shape_split_fused_diagnostic",
            "environment": {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "triton": __import__("triton").__version__,
                "model": args.model,
                "quantization": "INT4 group_size=64, streaming_load=True",
                "moe_backend": "segment",
                "execution_mode": "mixed, TP=1, eager",
                "cuda_graph": False,
                "multi_step_decode": False,
                "kv_swap": False,
            },
            "workload": {
                "num_requests": cli.num_seqs,
                "arrival_schedule": "all requests submitted at t=0",
                "prompt_source": (
                    "one repeated tokenizer-encoded text prompt" if cli.prompt is not None
                    else "benchmarks/bench.py build_workload(seed), fixed input length"),
                "prompt_text": cli.prompt,
                "prompt_token_ids_by_seed": {
                    str(seed): prompts_by_seed[seed] for seed in seeds
                },
                "input_tokens_per_request": cli.input_length,
                "prompt_fingerprint_by_seed": {
                    str(seed): prompt_fingerprints[seed] for seed in seeds
                },
                "output_lengths": output_lengths,
                "seeds": seeds,
                "temperatures": temperatures,
                "sampling_parameters": {
                    "temperature": "varied as listed",
                    "ignore_eos": True,
                    "max_tokens": "fixed by condition",
                    "other_sampling_controls": "not currently supported by SamplingParams",
                },
                "slo_targets_are_report_only": {
                    "ttft_ms": cli.slo_ttft_ms,
                    "tpot_ms": cli.slo_tpot_ms,
                },
            },
            "warmup": {
                "full_batch_runs_per_variant": cli.warmup_repeats,
                "output_tokens": warmup_length,
                "measured_after_warmup": True,
            },
            "natural_sampling_diagnostics": natural_pairs,
            "greedy_sampling_diagnostics": greedy_pairs,
            "diagnostic_sentinel": sentinel_coverage,
            "detailed_divergence_diagnostics": detailed_diagnostics,
            "fixed_token_performance_pairs": performance_pairs,
            "aggregate": aggregate,
            "clock_sampler": {
                "sample_interval_ms": cli.gpu_sample_interval_ms,
                "total_samples": len(clocks.samples),
            },
        }
        output = Path(cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Wrote {output}; {len(natural_pairs)} natural pairs, "
              f"{len(detailed_diagnostics)} detailed divergence probes, "
              f"{len(performance_pairs)} fixed-token performance pairs", flush=True)
        print(json.dumps(aggregate, indent=2), flush=True)
    finally:
        if clocks is not None:
            clocks.stop()
        if controller is not None:
            controller.close()
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

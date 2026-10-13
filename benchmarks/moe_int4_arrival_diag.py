"""Replay an INT4 MoE arrival trace and isolate MLA decode row ordering.

This diagnostic records the request order, per-call decode rows, sequence
positions, and sampler RNG states. At the selected request/token it saves
top-k logits, decoder/MoE intermediates, and the logical MLA KV rows. It then
replays the same hidden states and KV values under both backends' observed row
orders. It is instrumented for diagnosis; do not use its wall times as
performance results.

Example from Windows:

    wsl -e bash /mnt/d/Project/nano-vllm/benchmarks/run_in_wsl.sh \
        benchmarks/moe_int4_arrival_diag.py \
        --model /home/zjq/huggingface/DeepSeek-V2-Lite
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    from .bench import build_workload, env_info
    from .moe_int4_fuse_sweep import (
        _build_engine_args,
        _clear_idle_prefix_cache,
        _json_fingerprint,
        _ordered_outputs,
        _output_fingerprint,
        _parse_ints,
        _parse_positive_floats,
        _parse_rates,
        _prime_segment_variants,
        _set_fusion_mode,
        _validate_segment_layers,
    )
    from .moe_int4_fixed_shape_diag import _compare_moe_probes, _tensor_metrics, MoeTraceProbe
    from .scheduling_ablation import (
        build_arrival_trace,
        build_ttft_slo_targets,
        create_arrival_engine,
        run_arrival_trace,
        variant_table,
        workload_fingerprint,
    )
except ImportError:  # direct script execution from benchmarks/
    from bench import build_workload, env_info
    from moe_int4_fuse_sweep import (
        _build_engine_args,
        _clear_idle_prefix_cache,
        _json_fingerprint,
        _ordered_outputs,
        _output_fingerprint,
        _parse_ints,
        _parse_positive_floats,
        _parse_rates,
        _prime_segment_variants,
        _set_fusion_mode,
        _validate_segment_layers,
    )
    from moe_int4_fixed_shape_diag import _compare_moe_probes, _tensor_metrics, MoeTraceProbe
    from scheduling_ablation import (
        build_arrival_trace,
        build_ttft_slo_targets,
        create_arrival_engine,
        run_arrival_trace,
        variant_table,
        workload_fingerprint,
    )

from nanovllm import SamplingParams
from nanovllm.utils.context import (
    get_context,
    restore_context,
    set_context,
)


def _state_digest(state: torch.Tensor) -> str:
    return hashlib.sha256(bytes(state.detach().cpu().tolist())).hexdigest()


class ArrivalMoeCaptureContext:
    """Dynamic row mapping consumed by ArrivalMoeTraceProbe."""

    def __init__(self):
        self.capture_active = False
        self.target_moe_row: int | None = None
        self._sample_calls = -1
        self._current_prefill_rows = 0
        self._current_request_indices: list[int] = []
        self._current_decode_request_indices: list[int] = []


class ArrivalMoeTraceProbe(MoeTraceProbe):
    """Reuse the fixed-shape MoE probe with an online mixed-batch row index."""

    def _is_capture_step(self) -> bool:
        return bool(self.controller.capture_active)

    def _row_index(self, rows: int) -> int:
        row = self.controller.target_moe_row
        if row is None or not 0 <= row < rows:
            raise AssertionError(
                f"online MoE probe row {row} is outside the {rows}-row input")
        return row


class DecoderLayerTraceProbe:
    """Capture decoder stages for the selected token on one online batch row."""

    _STAGES = (
        "block_input",
        "block_residual_input",
        "input_layernorm",
        "input_layernorm_residual",
        "attention_output",
        "post_attention_layernorm",
        "post_attention_layernorm_residual",
        "mlp_output",
        "block_output",
        "block_output_residual",
    )

    def __init__(self, model, controller: ArrivalMoeCaptureContext):
        decoder = getattr(model, "model", None)
        self.layers = getattr(decoder, "layers", None)
        if self.layers is None:
            raise RuntimeError(
                "expected DeepSeek model.model.layers for decoder-stage diagnostics")
        self.controller = controller
        self.activations: dict[int, dict[str, torch.Tensor]] = {
            index: {} for index in range(len(self.layers))}
        self._handles = []
        self._install()

    @staticmethod
    def _tensors(value: Any) -> list[torch.Tensor]:
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, (tuple, list)):
            return [item for item in value if torch.is_tensor(item)]
        return []

    def _record(self, layer_index: int, name: str, value: Any) -> None:
        if not self.controller.capture_active:
            return
        row = self.controller.target_moe_row
        if row is None:
            raise AssertionError("decoder probe activated without a mapped batch row")
        tensors = self._tensors(value)
        for output_index, tensor in enumerate(tensors):
            if tensor.ndim < 2 or row >= tensor.size(0):
                raise AssertionError(
                    f"decoder probe row {row} is outside {name} shape "
                    f"{tuple(tensor.shape)}")
            label = name if output_index == 0 else f"{name}_residual"
            self.activations[layer_index][label] = (
                tensor[row].detach().float().cpu().clone())

    def _install(self) -> None:
        for layer_index, layer in enumerate(self.layers):
            def block_pre(_module, inputs, *, _index=layer_index):
                if len(inputs) < 2:
                    raise AssertionError("DeepSeek decoder layer input signature changed")
                self._record(_index, "block_input", inputs[1])
                if len(inputs) > 2 and inputs[2] is not None:
                    self._record(_index, "block_residual_input", inputs[2])

            self._handles.append(layer.register_forward_pre_hook(block_pre))
            for name, module in (
                ("input_layernorm", layer.input_layernorm),
                ("attention_output", layer.self_attn),
                ("post_attention_layernorm", layer.post_attention_layernorm),
                ("mlp_output", layer.mlp),
            ):
                def post(_module, _inputs, output, *,
                         _index=layer_index, _name=name):
                    self._record(_index, _name, output)

                self._handles.append(module.register_forward_hook(post))

            def block_post(_module, _inputs, output, *, _index=layer_index):
                self._record(_index, "block_output", output)

            self._handles.append(layer.register_forward_hook(block_post))

    def reset(self) -> None:
        for values in self.activations.values():
            values.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class MlaRowPermutationCaptureProbe:
    """Capture the live MLA decode inputs and logical KV rows for one call."""

    def __init__(self, model, controller: ArrivalMoeCaptureContext,
                 layer_index: int = 1):
        decoder = getattr(model, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None or not 0 <= layer_index < len(layers):
            raise RuntimeError(f"DeepSeek decoder has no layer {layer_index}")
        self.controller = controller
        self.layer_index = layer_index
        self.module = layers[layer_index].self_attn
        self.capture: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self._handles = [
            self.module.register_forward_pre_hook(self._before_attention),
            self.module.register_forward_hook(self._after_attention),
        ]

    def _before_attention(self, module, inputs) -> None:
        if not self.controller.capture_active:
            return
        if len(inputs) < 2:
            raise AssertionError("MLA forward must receive positions and hidden states")
        ctx = get_context()
        if ctx.is_spec:
            self.capture = {
                "supported": False,
                "reason": "focused call uses speculative MLA attention",
                "layer_index": self.layer_index,
            }
            return
        if ctx.is_prefill and not ctx.is_mixed:
            self.capture = {
                "supported": False,
                "reason": "focused call has no decode rows",
                "layer_index": self.layer_index,
            }
            return
        decode_start = ctx.n_prefill_tokens if ctx.is_mixed else 0
        request_indices = list(self.controller._current_decode_request_indices)
        positions = inputs[0][decode_start:]
        hidden_states = inputs[1][decode_start:]
        if ctx.context_lens is None or ctx.block_tables is None:
            self.capture = {
                "supported": False,
                "reason": "decode Context has no context_lens or block_tables",
                "layer_index": self.layer_index,
            }
            return
        if (positions.size(0) != len(request_indices)
                or hidden_states.size(0) != len(request_indices)
                or ctx.context_lens.size(0) != len(request_indices)
                or ctx.block_tables.size(0) != len(request_indices)):
            raise AssertionError(
                "MLA decode rows do not align with the recorded request rows")
        if not module.mla_cache.numel():
            self.capture = {
                "supported": False,
                "reason": "MLA cache is not allocated",
                "layer_index": self.layer_index,
            }
            return
        self._pending = {
            "module": module,
            "request_indices": request_indices,
            "positions": positions.detach().cpu().clone(),
            "hidden_states": hidden_states.detach().cpu().clone(),
            "context_lens": ctx.context_lens.detach().cpu().clone(),
            "block_tables": ctx.block_tables.detach().cpu().clone(),
            "decode_start": decode_start,
            "block_size": int(module.mla_cache.shape[1]),
            "cache_dim": int(module.mla_cache.shape[-1]),
            "cache_dtype": module.mla_cache.dtype,
        }

    def _after_attention(self, module, _inputs, output) -> None:
        pending = self._pending
        if pending is None or module is not self.module:
            return
        if not torch.is_tensor(output):
            raise AssertionError("MLA attention output must be a tensor")
        cache = module.mla_cache
        block_size = pending["block_size"]
        cache_by_request: dict[int, torch.Tensor] = {}
        positions_by_request: dict[int, int] = {}
        hidden_by_request: dict[int, torch.Tensor] = {}
        output_by_request: dict[int, torch.Tensor] = {}
        lengths = [int(value) for value in pending["context_lens"].tolist()]
        block_tables = pending["block_tables"]
        decode_output = output[pending["decode_start"]:]
        for row, request_index in enumerate(pending["request_indices"]):
            length = lengths[row]
            num_blocks = (length + block_size - 1) // block_size
            block_ids = block_tables[row, :num_blocks].to(
                device=cache.device, dtype=torch.long)
            logical = cache.index_select(0, block_ids).reshape(
                -1, pending["cache_dim"])[:length]
            cache_by_request[request_index] = logical.detach().cpu().clone()
            positions_by_request[request_index] = int(pending["positions"][row])
            hidden_by_request[request_index] = (
                pending["hidden_states"][row].clone())
            output_by_request[request_index] = (
                decode_output[row].detach().cpu().clone())
        if any(length != positions_by_request[request] + 1
               for request, length in zip(pending["request_indices"], lengths)):
            raise AssertionError("decode position and context length are inconsistent")
        self.capture = {
            "supported": True,
            "layer_index": self.layer_index,
            "request_indices": list(pending["request_indices"]),
            "positions_by_request": positions_by_request,
            "hidden_by_request": hidden_by_request,
            "attention_output_by_request": output_by_request,
            "logical_kv_by_request": cache_by_request,
            "context_lens_by_request": {
                request: lengths[row]
                for row, request in enumerate(pending["request_indices"])
            },
            "block_size": block_size,
            "cache_dim": pending["cache_dim"],
            "cache_dtype": pending["cache_dtype"],
            "module": module,
        }
        self._pending = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending = None


class ArrivalTraceRecorder:
    """Capture scheduler/model row order and one token's logits and RNG state."""

    def __init__(self, engine, num_requests: int, focus_request_index: int,
                 focus_token_index: int, probe: ArrivalMoeTraceProbe):
        self.engine = engine
        self.runner = engine.model_runner
        self.num_requests = num_requests
        self.focus_request_index = focus_request_index
        self.focus_token_index = focus_token_index
        self.probe = probe
        self.controller = probe.controller
        self.active = False
        self.request_index_by_seq_id: dict[int, int] = {}
        self.request_order: list[int] = []
        self.model_calls: list[dict[str, Any]] = []
        self.current_event: dict[str, Any] | None = None
        self.focus_logits: torch.Tensor | None = None
        self._original_add_request = engine.add_request
        self._original_call = self.runner.call
        self._original_sampler_forward = self.runner.sampler.forward
        self._install()

    @staticmethod
    def _sequence_row(request_index: int, seq) -> dict[str, Any]:
        if seq.is_prefill:
            input_end = seq.num_cached_tokens + seq.num_scheduled_tokens
            input_position = input_end - 1
            input_token_id = seq.token_ids[input_position]
        else:
            input_position = seq.num_tokens - 1
            input_token_id = seq.last_token
        return {
            "request_index": request_index,
            "seq_id": int(seq.seq_id),
            "completion_index": int(seq.num_completion_tokens),
            "input_position": int(input_position),
            "input_token_id": int(input_token_id),
            "scheduled_tokens": int(seq.num_scheduled_tokens),
            "is_prefill": bool(seq.is_prefill),
        }

    def _install(self) -> None:
        def add_request(prompt, sampling_params, submitted_at=None,
                        ttft_slo_ms=None, tpot_slo_ms=None):
            seq_id = self._original_add_request(
                prompt, sampling_params, submitted_at=submitted_at,
                ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
            if self.active:
                request_index = len(self.request_order)
                if request_index >= self.num_requests:
                    raise AssertionError("more requests arrived than configured")
                self.request_order.append(int(seq_id))
                self.request_index_by_seq_id[int(seq_id)] = request_index
            return seq_id

        def call(method_name, *args):
            if not self.active or method_name != "run" or len(args) < 2:
                return self._original_call(method_name, *args)

            seqs, kind = args[:2]
            rows = []
            for seq in seqs:
                request_index = self.request_index_by_seq_id.get(int(seq.seq_id))
                if request_index is None:
                    raise AssertionError(
                        f"model call contains unregistered sequence {seq.seq_id}")
                rows.append(self._sequence_row(request_index, seq))

            prefill_count = sum(row["is_prefill"] for row in rows)
            if (any(not row["is_prefill"] for row in rows[:prefill_count])
                    or any(row["is_prefill"] for row in rows[prefill_count:])):
                raise AssertionError(
                    "mixed model-call rows must place prefill sequences before decode")
            prefill_tokens = sum(
                row["scheduled_tokens"] for row in rows if row["is_prefill"])
            decode_tokens = sum(
                row["scheduled_tokens"] for row in rows if not row["is_prefill"])
            event = {
                "model_call_index": len(self.model_calls),
                "kind": str(kind),
                "prefill_sequences": prefill_count,
                "decode_sequences": len(rows) - prefill_count,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "request_indices": [row["request_index"] for row in rows],
                "sequence_ids": [row["seq_id"] for row in rows],
                "completion_indices": [row["completion_index"] for row in rows],
                "input_positions": [row["input_position"] for row in rows],
                "input_token_ids": [row["input_token_id"] for row in rows],
                "scheduled_tokens_by_sequence": [
                    row["scheduled_tokens"] for row in rows],
                "prefill_mask": [row["is_prefill"] for row in rows],
            }

            focus_row = next(
                (row_index for row_index, row in enumerate(rows)
                 if row["request_index"] == self.focus_request_index),
                None)
            target_moe_row = None
            if focus_row is not None:
                focus = rows[focus_row]
                if focus["is_prefill"]:
                    target_moe_row = (
                        sum(row["scheduled_tokens"] for row in rows[:focus_row])
                        + focus["scheduled_tokens"] - 1)
                else:
                    decode_row = focus_row - prefill_count
                    target_moe_row = prefill_tokens + decode_row
                event["focus_request"] = {
                    "request_index": focus["request_index"],
                    "seq_id": focus["seq_id"],
                    "completion_index": focus["completion_index"],
                    "input_position": focus["input_position"],
                    "input_token_id": focus["input_token_id"],
                    "is_prefill": focus["is_prefill"],
                    "scheduled_tokens": focus["scheduled_tokens"],
                    "sampler_row_index": focus_row,
                    "moe_input_row_index": target_moe_row,
                }

            capture = (
                focus_row is not None
                and rows[focus_row]["completion_index"] == self.focus_token_index)
            if capture and self.focus_logits is not None:
                raise AssertionError(
                    "the focus request/token was sampled more than once in one trace")
            self.model_calls.append(event)
            self.current_event = event
            self.controller._current_request_indices = [
                row["request_index"] for row in rows]
            self.controller._current_decode_request_indices = [
                row["request_index"] for row in rows[prefill_count:]]
            self.controller._current_prefill_rows = prefill_count
            self.controller._sample_calls = (
                self.focus_token_index if capture else -1)
            self.controller.target_moe_row = target_moe_row if capture else None
            self.controller.capture_active = capture
            try:
                return self._original_call(method_name, *args)
            finally:
                self.controller.capture_active = False
                self.controller.target_moe_row = None
                self.controller._sample_calls = -1
                self.controller._current_decode_request_indices = []
                self.current_event = None

        def sampler_forward(logits: torch.Tensor, _temperatures: torch.Tensor):
            # The historical serving trace used greedy sampling. This matches
            # that path and makes RNG consumption explicitly absent.
            sampled = logits.argmax(dim=-1)
            event = self.current_event
            if not self.active or event is None:
                return sampled
            if logits.ndim != 2 or logits.size(0) != len(event["request_indices"]):
                raise AssertionError(
                    f"sampler rows {tuple(logits.shape)} do not match the recorded "
                    f"model-call rows {len(event['request_indices'])}")
            focus = event.get("focus_request")
            if (focus is None
                    or focus["completion_index"] != self.focus_token_index):
                return sampled

            row = focus["sampler_row_index"]
            cpu_rng_before = _state_digest(torch.random.get_rng_state())
            cuda_rng_before = _state_digest(torch.cuda.get_rng_state(logits.device))
            top_values, top_ids = torch.topk(
                logits[row].detach().float(),
                k=min(8, logits.size(-1)))
            self.focus_logits = logits[row].detach().float().cpu().clone()
            top_values_cpu = top_values.detach().cpu().tolist()
            top_ids_cpu = top_ids.detach().cpu().tolist()
            sampled_token = int(sampled[row].item())
            event["focus_capture"] = {
                "output_token_index": self.focus_token_index,
                "sampled_token_id": sampled_token,
                "top1_top2_margin": (
                    float(top_values_cpu[0] - top_values_cpu[1])
                    if len(top_values_cpu) > 1 else None),
                "topk_token_ids": [int(item) for item in top_ids_cpu],
                "topk_logits": [float(item) for item in top_values_cpu],
                "temperature": float(_temperatures[row].detach().cpu()),
                "sampler": "argmax",
                "cpu_rng_state_before_sha256": cpu_rng_before,
                "cpu_rng_state_after_sha256": _state_digest(
                    torch.random.get_rng_state()),
                "cuda_rng_state_before_sha256": cuda_rng_before,
                "cuda_rng_state_after_sha256": _state_digest(
                    torch.cuda.get_rng_state(logits.device)),
            }
            return sampled

        self.engine.add_request = add_request
        self.runner.call = call
        self.runner.sampler.forward = sampler_forward

    def start(self) -> None:
        if self.active:
            raise RuntimeError("arrival trace recorder already active")
        self.active = True
        self.request_index_by_seq_id.clear()
        self.request_order.clear()
        self.model_calls.clear()
        self.focus_logits = None
        self.controller._current_decode_request_indices.clear()

    def finish(self) -> dict[str, Any]:
        self.active = False
        if len(self.request_order) != self.num_requests:
            raise AssertionError(
                f"recorded {len(self.request_order)} of {self.num_requests} arrivals")
        focus_events = [
            event for event in self.model_calls
            if event.get("focus_capture") is not None]
        if len(focus_events) != 1:
            raise AssertionError(
                f"expected one focused token capture, found {len(focus_events)}")
        return {
            "request_submission_order": [
                {"request_index": index, "seq_id": seq_id}
                for index, seq_id in enumerate(self.request_order)
            ],
            "model_calls": list(self.model_calls),
            "focus_capture": focus_events[0]["focus_capture"],
        }

    def close(self) -> None:
        self.active = False
        self.engine.add_request = self._original_add_request
        self.runner.call = self._original_call
        self.runner.sampler.forward = self._original_sampler_forward


def _call_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["kind"],
        event["prefill_sequences"],
        event["decode_sequences"],
        event["prefill_tokens"],
        event["decode_tokens"],
        tuple(event["request_indices"]),
        tuple(event["completion_indices"]),
        tuple(event["input_positions"]),
        tuple(event["input_token_ids"]),
        tuple(event["scheduled_tokens_by_sequence"]),
        tuple(event["prefill_mask"]),
    )


def _compare_decoder_activations(
    left: dict[int, dict[str, torch.Tensor]],
    right: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    differences = []
    labels = set()
    first_difference = None
    for layer_index in sorted(set(left) | set(right)):
        left_layer = left.get(layer_index, {})
        right_layer = right.get(layer_index, {})
        for label in DecoderLayerTraceProbe._STAGES:
            if label not in left_layer and label not in right_layer:
                continue
            labels.add(label)
            metrics = _tensor_metrics(left_layer.get(label), right_layer.get(label))
            if metrics is None or metrics.get("bitwise_equal", False):
                continue
            entry = {
                "decoder_layer": layer_index,
                "stage": label,
                "metrics": metrics,
            }
            differences.append(entry)
            if first_difference is None:
                first_difference = entry
    return {
        "captured_decoder_layers": len(set(left) | set(right)),
        "captured_stages": sorted(labels),
        "different_activation_count": len(differences),
        "all_captured_activations_equal": not differences,
        "first_difference": first_difference,
        "differences": differences,
    }


def _summarize_row_metrics(left: dict[int, torch.Tensor],
                           right: dict[int, torch.Tensor],
                           request_indices: list[int]) -> dict[str, Any]:
    metrics = [
        _tensor_metrics(left.get(request_index), right.get(request_index))
        for request_index in request_indices
    ]
    metrics = [item for item in metrics if item is not None]
    if not metrics:
        return {
            "rows": 0,
            "bitwise_equal_rows": 0,
            "different_elements": 0,
            "max_abs": None,
            "max_rel": None,
            "all_rows_bitwise_equal": False,
        }
    return {
        "rows": len(metrics),
        "bitwise_equal_rows": sum(bool(item.get("bitwise_equal"))
                                   for item in metrics),
        "different_elements": sum(int(item.get("different_elements", 0))
                                  for item in metrics),
        "max_abs": max(float(item.get("max_abs", 0.0)) for item in metrics),
        "max_rel": max(float(item.get("max_rel", 0.0)) for item in metrics),
        "all_rows_bitwise_equal": all(
            bool(item.get("bitwise_equal")) for item in metrics),
    }


def _compare_mla_captures(left: dict[str, Any] | None,
                          right: dict[str, Any] | None,
                          focus_request_index: int) -> dict[str, Any]:
    if not left or not right or not left.get("supported") or not right.get("supported"):
        return {
            "supported": False,
            "split_reason": (None if left and left.get("supported")
                             else (left or {}).get("reason", "capture missing")),
            "fused_reason": (None if right and right.get("supported")
                             else (right or {}).get("reason", "capture missing")),
        }
    left_pos = left["positions_by_request"]
    right_pos = right["positions_by_request"]
    common = sorted(set(left_pos) & set(right_pos))
    matched = [request for request in common
               if left_pos[request] == right_pos[request]]
    position_mismatches = [request for request in common
                           if left_pos[request] != right_pos[request]]
    left_maps = {
        "hidden_state": left["hidden_by_request"],
        "logical_kv": left["logical_kv_by_request"],
        "attention_output": left["attention_output_by_request"],
    }
    right_maps = {
        "hidden_state": right["hidden_by_request"],
        "logical_kv": right["logical_kv_by_request"],
        "attention_output": right["attention_output_by_request"],
    }
    details = {
        name: _summarize_row_metrics(left_maps[name], right_maps[name], matched)
        for name in left_maps
    }
    focus = None
    if focus_request_index in left_pos and focus_request_index in right_pos:
        focus = {
            "request_index": focus_request_index,
            "split_position": left_pos[focus_request_index],
            "fused_position": right_pos[focus_request_index],
            "same_position": (
                left_pos[focus_request_index] == right_pos[focus_request_index]),
        }
        if focus["same_position"]:
            focus.update({
                name: _tensor_metrics(
                    left_maps[name].get(focus_request_index),
                    right_maps[name].get(focus_request_index))
                for name in left_maps
            })
    return {
        "supported": True,
        "layer_index": left["layer_index"],
        "split_decode_rows": len(left_pos),
        "fused_decode_rows": len(right_pos),
        "row_order_equal": left["request_indices"] == right["request_indices"],
        "same_request_set": set(left_pos) == set(right_pos),
        "matched_request_rows_same_position": matched,
        "request_rows_with_position_mismatch": position_mismatches,
        "matched_row_metrics": details,
        "focus_request": focus,
    }


def _replay_mla_order(snapshot: dict[str, Any],
                      request_order: list[int]) -> dict[int, torch.Tensor]:
    """Rebuild a compact cache and replay one frozen MLA decode row order."""
    module = snapshot["module"]
    if set(request_order) != set(snapshot["request_indices"]):
        raise ValueError("replay order must contain exactly the captured decode rows")
    block_size = snapshot["block_size"]
    cache_by_request = snapshot["logical_kv_by_request"]
    lengths = [len(cache_by_request[request]) for request in request_order]
    blocks_per_request = [
        (length + block_size - 1) // block_size for length in lengths]
    num_blocks = max(blocks_per_request, default=0)
    total_blocks = sum(blocks_per_request)
    device = module.mla_cache.device
    cache_dtype = snapshot["cache_dtype"]
    cache = torch.zeros(
        total_blocks, block_size, snapshot["cache_dim"],
        dtype=cache_dtype, device=device)
    block_tables = torch.zeros(
        len(request_order), num_blocks, dtype=torch.int32, device=device)
    context_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    positions = torch.tensor(
        [snapshot["positions_by_request"][request]
         for request in request_order], dtype=torch.int64, device=device)
    hidden_states = torch.stack(
        [snapshot["hidden_by_request"][request] for request in request_order]
    ).to(device=device)
    cursor = 0
    for row, request in enumerate(request_order):
        logical = cache_by_request[request].to(device=device, dtype=cache_dtype)
        row_blocks = blocks_per_request[row]
        block_tables[row, :row_blocks] = torch.arange(
            cursor, cursor + row_blocks, dtype=torch.int32, device=device)
        for block in range(row_blocks):
            start = block * block_size
            end = min(start + block_size, lengths[row])
            cache[cursor + block, :end - start].copy_(logical[start:end])
        cursor += row_blocks

    previous_cache = module.mla_cache
    previous_context = get_context()
    try:
        module.mla_cache = cache
        set_context(False, slot_mapping=None,
                    context_lens=context_lens, block_tables=block_tables)
        with torch.inference_mode():
            output = module(positions, hidden_states)
        return {
            request: output[row].detach().cpu().clone()
            for row, request in enumerate(request_order)
        }
    finally:
        module.mla_cache = previous_cache
        restore_context(previous_context)


def _mla_row_permutation_report(split: dict[str, Any] | None,
                                fused: dict[str, Any] | None,
                                focus_request_index: int) -> dict[str, Any]:
    if (not split or not fused or not split.get("supported")
            or not fused.get("supported")):
        return {
            "supported": False,
            "reason": "both variants need a captured pure MLA decode call",
        }
    split_set = set(split["request_indices"])
    fused_set = set(fused["request_indices"])
    if split_set != fused_set:
        return {
            "supported": False,
            "reason": "split and fused focused calls contain different decode requests",
            "split_request_indices": sorted(split_set),
            "fused_request_indices": sorted(fused_set),
        }
    split_order = list(split["request_indices"])
    fused_order = list(fused["request_indices"])
    if split_order == fused_order:
        split_test_order = split_order[::-1]
        fused_test_order = fused_order[::-1]
        split_test_source = fused_test_source = "reversed control order"
    else:
        split_test_order = fused_order
        fused_test_order = split_order
        split_test_source = "fused observed order"
        fused_test_source = "split observed order"

    state_tests = []
    for name, snapshot, base_order, test_order, test_source in (
        ("segment_split", split, split_order, split_test_order,
         split_test_source),
        ("segment_fused", fused, fused_order, fused_test_order,
         fused_test_source),
    ):
        live = snapshot["attention_output_by_request"]
        base = _replay_mla_order(snapshot, base_order)
        permuted = _replay_mla_order(snapshot, test_order)
        all_requests = sorted(split_set)
        focus_metrics = None
        if focus_request_index in split_set:
            focus_metrics = {
                "request_index": focus_request_index,
                "live_vs_base_replay": _tensor_metrics(
                    live.get(focus_request_index), base.get(focus_request_index)),
                "base_vs_test_order": _tensor_metrics(
                    base.get(focus_request_index),
                    permuted.get(focus_request_index)),
            }
        state_tests.append({
            "state_source": name,
            "captured_row_order": base_order,
            "test_row_order": test_order,
            "test_row_order_source": test_source,
            "live_vs_base_replay": _summarize_row_metrics(
                live, base, all_requests),
            "base_vs_test_order": _summarize_row_metrics(
                base, permuted, all_requests),
            "focus_request": focus_metrics,
        })
    return {
        "supported": True,
        "layer_index": split["layer_index"],
        "decode_rows": len(split_order),
        "observed_backend_orders_equal": split_order == fused_order,
        "state_tests": state_tests,
        "diagnostic_only": True,
    }


def _output_mismatches(left: list[list[int]],
                       right: list[list[int]]) -> list[dict[str, Any]]:
    mismatches = []
    for request_index, (left_tokens, right_tokens) in enumerate(zip(left, right)):
        first_diff = next(
            (index for index, (a, b) in enumerate(zip(left_tokens, right_tokens))
             if a != b),
            None)
        if (first_diff is not None or len(left_tokens) != len(right_tokens)):
            mismatches.append({
                "request_index": request_index,
                "first_different_token_index": first_diff,
                "split_length": len(left_tokens),
                "fused_length": len(right_tokens),
                "split_token_id": (
                    left_tokens[first_diff] if first_diff is not None else None),
                "fused_token_id": (
                    right_tokens[first_diff] if first_diff is not None else None),
            })
    return mismatches


def _run_variant(engine, moes, args, prompts, sampling_params, arrivals,
                 ttft_targets, policy, condition, variant: str,
                 focus_request_index: int, focus_token_index: int):
    fused = variant == "segment_fused"
    _set_fusion_mode(moes, fused)
    args.seed = condition["seed"]
    args.slo_tpot_ms = condition["tpot_slo_ms"]
    args.no_moe_int4_fuse_gate_up = not fused
    controller = ArrivalMoeCaptureContext()
    probe = ArrivalMoeTraceProbe(
        controller,
        moes,
        input_length=condition["input_tokens"],
        probe_request_index=focus_request_index,
        capture_step=focus_token_index,
    )
    decoder_probe = DecoderLayerTraceProbe(engine.model_runner.model, controller)
    mla_probe = MlaRowPermutationCaptureProbe(
        engine.model_runner.model, controller, layer_index=1)
    recorder = ArrivalTraceRecorder(
        engine,
        len(prompts),
        focus_request_index,
        focus_token_index,
        probe,
    )
    run = None
    trace = None
    try:
        def before_trace(trace_engine):
            _clear_idle_prefix_cache(trace_engine)
            probe.snapshots = {index: {} for index in range(len(moes))}
            decoder_probe.reset()
            recorder.start()

        run = run_arrival_trace(
            args,
            prompts,
            sampling_params,
            arrivals,
            ttft_targets,
            policy,
            engine=engine,
            capture_completion_tokens=True,
            before_trace=before_trace,
        )
        trace = recorder.finish()
        outputs = _ordered_outputs(run)
        if run["summary"]["output_tokens"] != len(prompts) * condition["output_tokens"]:
            raise AssertionError("run did not emit the requested fixed output length")
        return {
            "trace": trace,
            "outputs": outputs,
            "focus_logits": recorder.focus_logits,
            "decoder_activations": decoder_probe.activations,
            "mla_capture": mla_probe.capture,
            "output_token_fingerprint": _output_fingerprint(outputs),
            "diagnostic_only_summary": {
                "wall_seconds": run["wall_seconds"],
                "throughput_output_tok_per_s": (
                    run["summary"]["throughput_output_tok_per_s"]),
                "joint_slo_requests": (
                    run["summary"]["slo_goodput"]["joint_slo_requests"]),
                "joint_slo_output_tokens_per_second": (
                    run["summary"]["slo_goodput"][
                        "joint_slo_output_tokens_per_second"]),
            },
            "moe_probe": probe,
        }
    finally:
        recorder.close()
        mla_probe.close()
        decoder_probe.close()
        probe.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/zjq/huggingface/DeepSeek-V2-Lite")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--arrival-rates", default="32")
    parser.add_argument("--output-lengths", default="64")
    parser.add_argument("--tpot-targets", default="115")
    parser.add_argument("--num-seqs", type=int, default=24)
    parser.add_argument("--input-length", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--int4-group-size", type=int, default=64)
    parser.add_argument("--slo-ttft-ms", type=float, default=10000.0)
    parser.add_argument("--warmup-seqs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--focus-request-index", type=int, default=23)
    parser.add_argument("--focus-token-index", type=int, default=48)
    parser.add_argument(
        "--output",
        default="results/moe_int4_mla_row_permutation_rep5_20261012.json")
    cli_args = parser.parse_args()

    seeds = _parse_ints(cli_args.seeds, "--seeds", allow_zero=True)
    arrival_rates = _parse_rates(cli_args.arrival_rates)
    output_lengths = _parse_ints(cli_args.output_lengths, "--output-lengths")
    tpot_targets = _parse_positive_floats(cli_args.tpot_targets, "--tpot-targets")
    if cli_args.num_seqs < 1 or cli_args.input_length < 1:
        raise ValueError("--num-seqs and --input-length must be positive")
    if not 0 <= cli_args.focus_request_index < cli_args.num_seqs:
        raise ValueError("--focus-request-index must name a configured request")
    if min(output_lengths) <= cli_args.focus_token_index:
        raise ValueError("every output length must include the focused token")
    if cli_args.max_model_len < cli_args.input_length + max(output_lengths):
        raise ValueError("--max-model-len must cover input plus output tokens")
    if cli_args.int4_group_size != 64:
        raise ValueError("the DeepSeek-V2-Lite INT4 checkpoint requires group size 64")
    if cli_args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the online diagnostic requires a CUDA GPU")

    cli_args.slo_tpot_ms = tpot_targets[0]
    args = _build_engine_args(cli_args)
    policy = variant_table()["all_on"]
    engine = create_arrival_engine(args, policy)
    pairs = []
    try:
        moes = _validate_segment_layers(engine)
        print(
            f"GPU={torch.cuda.get_device_name()} model={args.model} "
            f"MoE layers={len(moes)}; capturing request "
            f"{cli_args.focus_request_index}, output token "
            f"{cli_args.focus_token_index}",
            flush=True,
        )
        workload_args = SimpleNamespace(
            seed=seeds[0],
            num_seqs=cli_args.num_seqs,
            min_input_len=cli_args.input_length,
            max_input_len=cli_args.input_length,
            min_output_len=1,
            max_output_len=1,
            shared_prefix_len=0,
            prompts_file=None,
        )
        prime_prompts, _ = _build_workload_for_seed(workload_args, seeds[0])
        print("Priming split/fused kernel shapes (not timed) ...", flush=True)
        _prime_segment_variants(engine, moes, prime_prompts)

        condition_index = 0
        for tpot_target in tpot_targets:
            for seed in seeds:
                for arrival_rate in arrival_rates:
                    for output_length in output_lengths:
                        workload_args.seed = seed
                        prompts, _ = _build_workload_for_seed(workload_args, seed)
                        sampling_params = [
                            SamplingParams(
                                temperature=1.0,
                                ignore_eos=True,
                                max_tokens=output_length,
                            )
                            for _ in prompts
                        ]
                        arrivals = build_arrival_trace(
                            len(prompts), "constant", arrival_rate, seed + 1)
                        ttft_targets = build_ttft_slo_targets(
                            len(prompts), cli_args.slo_ttft_ms,
                            cli_args.slo_ttft_ms, seed + 2)
                        tpot_request_targets = [tpot_target] * len(prompts)
                        fingerprint = workload_fingerprint(
                            prompts, sampling_params, arrivals, ttft_targets,
                            tpot_request_targets, seed)
                        condition = {
                            "seed": seed,
                            "arrival_mode": "constant",
                            "arrival_rate_requests_per_second": arrival_rate,
                            "num_requests": len(prompts),
                            "input_tokens": cli_args.input_length,
                            "output_tokens": output_length,
                            "ttft_slo_ms": cli_args.slo_ttft_ms,
                            "tpot_slo_ms": tpot_target,
                            "prompt_fingerprint": _json_fingerprint(prompts),
                            "arrival_schedule_fingerprint": _json_fingerprint(arrivals),
                            "workload_fingerprint": fingerprint,
                        }
                        condition_index += 1
                        print(
                            f"\nseed={seed} rate={arrival_rate:g} req/s "
                            f"output={output_length} TPOT={tpot_target:g} ms "
                            f"trace={fingerprint[:16]}",
                            flush=True,
                        )
                        for repeat_index in range(cli_args.repeats):
                            order = (
                                ["segment_split", "segment_fused"]
                                if (condition_index + repeat_index) % 2 == 0
                                else ["segment_fused", "segment_split"])
                            measurements = {}
                            for variant in order:
                                print(
                                    f"  repeat={repeat_index + 1}/"
                                    f"{cli_args.repeats} {variant}",
                                    flush=True,
                                )
                                measurements[variant] = _run_variant(
                                    engine, moes, args, prompts, sampling_params,
                                    arrivals, ttft_targets, policy, condition,
                                    variant, cli_args.focus_request_index,
                                    cli_args.focus_token_index)

                            split = measurements["segment_split"]
                            fused = measurements["segment_fused"]
                            mismatches = _output_mismatches(
                                split["outputs"], fused["outputs"])
                            split_events = split["trace"]["model_calls"]
                            fused_events = fused["trace"]["model_calls"]
                            first_call_diff = next(
                                (index for index, (left, right) in enumerate(
                                    zip(split_events, fused_events))
                                 if _call_signature(left) != _call_signature(right)),
                                None)
                            if (first_call_diff is None
                                    and len(split_events) != len(fused_events)):
                                first_call_diff = min(len(split_events), len(fused_events))
                            split_focus_event = next(
                                event for event in split_events
                                if event.get("focus_capture") is not None)
                            fused_focus_event = next(
                                event for event in fused_events
                                if event.get("focus_capture") is not None)
                            split_focus = split_focus_event["focus_request"]
                            fused_focus = fused_focus_event["focus_request"]
                            mla_input_state = _compare_mla_captures(
                                split["mla_capture"], fused["mla_capture"],
                                cli_args.focus_request_index)
                            mla_order_diagnostic = _mla_row_permutation_report(
                                split["mla_capture"], fused["mla_capture"],
                                cli_args.focus_request_index)
                            trace_comparison = {
                                "split_model_calls": len(split_events),
                                "fused_model_calls": len(fused_events),
                                "all_call_signatures_equal": (
                                    first_call_diff is None
                                    and len(split_events) == len(fused_events)),
                                "first_call_signature_difference": first_call_diff,
                                "focus_model_call_index_split": (
                                    split_focus_event["model_call_index"]),
                                "focus_model_call_index_fused": (
                                    fused_focus_event["model_call_index"]),
                                "focus_call_signature_equal": (
                                    _call_signature(split_focus_event)
                                    == _call_signature(fused_focus_event)),
                                "focus_request_row_order_equal": (
                                    split_focus_event["request_indices"]
                                    == fused_focus_event["request_indices"]),
                                "focus_input_equal": {
                                    "output_position_equal": (
                                        split_focus["completion_index"]
                                        == fused_focus["completion_index"]),
                                    "model_input_position_equal": (
                                        split_focus["input_position"]
                                        == fused_focus["input_position"]),
                                    "model_input_token_equal": (
                                        split_focus["input_token_id"]
                                        == fused_focus["input_token_id"]),
                                },
                                "focus_logits": _tensor_metrics(
                                    split["focus_logits"], fused["focus_logits"]),
                                "mla_input_state": mla_input_state,
                                "mla_row_permutation": mla_order_diagnostic,
                                "decoder_layer_activations": (
                                    _compare_decoder_activations(
                                        split["decoder_activations"],
                                        fused["decoder_activations"])),
                                "moe_layers": _compare_moe_probes(
                                    split["moe_probe"], fused["moe_probe"]),
                            }
                            pair = {
                                "condition": condition,
                                "repeat_index": repeat_index + 1,
                                "execution_order": order,
                                "sampling_mode": "greedy",
                                "output_parity": {
                                    "all_requests_match": not mismatches,
                                    "matching_requests": (
                                        len(prompts) - len(mismatches)),
                                    "total_requests": len(prompts),
                                    "mismatches": mismatches,
                                    "split_output_fingerprint": (
                                        split["output_token_fingerprint"]),
                                    "fused_output_fingerprint": (
                                        fused["output_token_fingerprint"]),
                                },
                                "trace_comparison": trace_comparison,
                                "runs": {
                                    "segment_split": {
                                        "request_submission_order": (
                                            split["trace"]["request_submission_order"]),
                                        "focus_capture": (
                                            split["trace"]["focus_capture"]),
                                        "focus_completion_prefix": (
                                            split["outputs"][
                                                cli_args.focus_request_index]
                                            [:cli_args.focus_token_index + 1]),
                                        "model_calls": split_events,
                                        "diagnostic_only_summary": (
                                            split["diagnostic_only_summary"]),
                                    },
                                    "segment_fused": {
                                        "request_submission_order": (
                                            fused["trace"]["request_submission_order"]),
                                        "focus_capture": (
                                            fused["trace"]["focus_capture"]),
                                        "focus_completion_prefix": (
                                            fused["outputs"][
                                                cli_args.focus_request_index]
                                            [:cli_args.focus_token_index + 1]),
                                        "model_calls": fused_events,
                                        "diagnostic_only_summary": (
                                            fused["diagnostic_only_summary"]),
                                    },
                                },
                            }
                            pairs.append(pair)
                            print(
                                f"  token parity={'PASS' if not mismatches else 'FAIL'} "
                                f"focus rows equal="
                                f"{trace_comparison['focus_request_row_order_equal']} "
                                f"logits equal="
                                f"{trace_comparison['focus_logits'].get('bitwise_equal')}",
                                flush=True,
                            )

        payload = {
            "schema_version": 1,
            "benchmark": "moe_int4_continuous_arrival_mla_row_permutation_diag",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                **env_info(),
                "gpu": torch.cuda.get_device_name(),
                "model": args.model,
                "quantization": "INT4 group_size=64",
                "moe_backend": "segment",
                "execution_mode": "mixed, TP=1, eager",
                "cuda_graph": False,
                "multi_step_decode": False,
                "kv_swap": False,
            },
            "benchmark_config": {
                "sampling_mode": "greedy",
                "sampler": "argmax; sampler RNG is not consumed",
                "warmup_seqs": cli_args.warmup_seqs,
                "repetitions_per_condition": cli_args.repeats,
                "focus_request_index": cli_args.focus_request_index,
                "focus_output_token_index": cli_args.focus_token_index,
                "mla_capture_layer_index": 1,
                "mla_row_order_replay": (
                    "rehydrate captured logical KV; replay frozen hidden states "
                    "under both observed decode row orders"),
                "instrumentation_note": (
                    "Detailed trace and MoE/logit capture perturb runtime; "
                    "decoder-stage, MoE, and logit capture perturb runtime; "
                    "diagnostic summaries are not performance measurements."),
            },
            "summary": {
                "paired_conditions": len({
                    pair["condition"]["workload_fingerprint"] for pair in pairs}),
                "paired_measurements": len(pairs),
                "token_parity_pairs": sum(
                    pair["output_parity"]["all_requests_match"] for pair in pairs),
                "token_parity_failures": sum(
                    not pair["output_parity"]["all_requests_match"]
                    for pair in pairs),
                "focus_logits_equal_pairs": sum(
                    bool(pair["trace_comparison"]["focus_logits"]
                         and pair["trace_comparison"]["focus_logits"].get(
                             "bitwise_equal"))
                    for pair in pairs),
                "focus_call_signature_equal_pairs": sum(
                    pair["trace_comparison"]["focus_call_signature_equal"]
                    for pair in pairs),
                "mla_capture_supported_pairs": sum(
                    pair["trace_comparison"]["mla_input_state"]["supported"]
                    for pair in pairs),
                "mla_row_permutation_supported_pairs": sum(
                    pair["trace_comparison"]["mla_row_permutation"]["supported"]
                    for pair in pairs),
                "mla_replay_roundtrip_exact_state_tests": sum(
                    state_test["live_vs_base_replay"][
                        "all_rows_bitwise_equal"]
                    for pair in pairs
                    for state_test in pair["trace_comparison"][
                        "mla_row_permutation"].get("state_tests", [])),
                "mla_row_permutation_exact_state_tests": sum(
                    state_test["base_vs_test_order"][
                        "all_rows_bitwise_equal"]
                    for pair in pairs
                    for state_test in pair["trace_comparison"][
                        "mla_row_permutation"].get("state_tests", [])),
            },
            "pairs": pairs,
        }
        output = Path(cli_args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False),
                          encoding="utf-8")
        print(f"\nWrote {output}", flush=True)
        print(json.dumps(payload["summary"], indent=2), flush=True)
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def _build_workload_for_seed(workload_args, seed: int):
    workload_args.seed = seed
    return build_workload(workload_args)


if __name__ == "__main__":
    main()

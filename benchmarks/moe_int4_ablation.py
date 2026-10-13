"""Profile pure-INT4 MoE experts: loop, segmented, and fused gate/up variants.

Use a local INT4-capable MoE checkpoint, preferably DeepSeek-V2-Lite::

    python benchmarks/moe_int4_ablation.py ~/huggingface/DeepSeek-V2-Lite

The script reports per-layer CUDA-event timings and a stage breakdown. For
continuous-arrival SLO goodput, use ``scheduling_ablation.py`` with the same
trace and each of ``--moe-int4-backend loop``, ``--moe-int4-backend segment
--no-moe-int4-fuse-gate-up``, and ``--moe-int4-backend segment``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from nanovllm import LLM
from nanovllm.layers.moe import MoE
from nanovllm.layers.moe_int4_segment import (
    moe_segment_int4_gup,
    moe_segment_int4_mm,
)


VARIANTS = (
    ("loop", "loop", False),
    ("segment_split", "segment", False),
    ("segment_fused", "segment", True),
)


def _parse_int_list(value: str, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError(f"{name} must contain positive integers")
    return parsed


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    diff = (actual.float() - expected.float()).abs()
    relative_error = (actual.float() - expected.float()).norm() \
        / expected.float().norm().clamp(min=1e-12)
    return {
        "relative_error": float(relative_error),
        "max_abs_error": float(diff.max()),
        "exact_match": bool(torch.equal(actual, expected)),
    }


def verify_real_layer(moe: MoE, token_batches: list[int], seeds: list[int],
                      *, atol: float, rtol: float,
                      label: str = "") -> dict:
    """Compare loop/split/fused MoE outputs over independent real-weight inputs."""
    checks = {}
    all_close = True
    for token_batch in token_batches:
        checks[str(token_batch)] = {}
        dtype = moe.experts[0].gate_proj.w_int4_scale.dtype
        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            x = torch.randn(token_batch, moe.hidden_size, device="cuda",
                            dtype=dtype, generator=generator)
            with torch.inference_mode():
                expected = moe._forward_loop(x)
            seed_checks = {}
            for name, backend, fused in VARIANTS:
                _set_variant([moe], backend, fused)
                forward = (moe._forward_loop if backend == "loop"
                           else moe._forward_segment_int4)
                with torch.inference_mode():
                    actual = forward(x)
                torch.cuda.synchronize()
                metrics = _error_metrics(actual, expected)
                metrics["within_tolerance"] = bool(torch.allclose(
                    actual, expected, atol=atol, rtol=rtol, equal_nan=False))
                all_close = all_close and metrics["within_tolerance"]
                seed_checks[name] = metrics
            checks[str(token_batch)][str(seed)] = seed_checks
            print(
                f"  {label}correctness tokens={token_batch:3d} seed={seed:3d} "
                f"split max_abs={seed_checks['segment_split']['max_abs_error']:.3e} "
                f"fused max_abs={seed_checks['segment_fused']['max_abs_error']:.3e} "
                f"exact={seed_checks['segment_split']['exact_match']}/"
                f"{seed_checks['segment_fused']['exact_match']}",
                flush=True,
            )
    return {
        "seeds": seeds,
        "token_batches": token_batches,
        "atol": atol,
        "rtol": rtol,
        "within_tolerance": all_close,
        "checks": checks,
    }


def cuda_median_ms(fn, *, warmup: int, iterations: int, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def _loop_dispatch(moe: MoE, x: torch.Tensor, top_vals: torch.Tensor,
                   top_idx: torch.Tensor) -> dict:
    T = x.size(0)
    idx_flat = top_idx.reshape(-1)
    w_flat = top_vals.reshape(-1)
    tgt = (torch.arange(T, device=x.device).unsqueeze(1)
           .expand(T, moe.top_k).reshape(-1))
    experts = []
    for expert_id, expert in enumerate(moe.experts):
        mask = idx_flat == expert_id
        target = tgt[mask]
        experts.append({
            "module": expert,
            "target": target,
            "weights": w_flat[mask],
            "x": x.index_select(0, target),
        })
    return {"experts": experts, "target": tgt, "weights": w_flat}


def _segment_dispatch(moe: MoE, x: torch.Tensor, top_vals: torch.Tensor,
                      top_idx: torch.Tensor) -> dict:
    T = x.size(0)
    idx_flat = top_idx.reshape(-1)
    weights = top_vals.reshape(-1)
    target = (torch.arange(T, device=x.device).unsqueeze(1)
              .expand(T, moe.top_k).reshape(-1))
    order = torch.argsort(idx_flat, stable=True)
    sorted_idx = idx_flat[order]
    xs = x.index_select(0, target[order])
    counts = torch.bincount(sorted_idx, minlength=moe.num_experts)
    offsets = torch.cumsum(counts, 0) - counts
    max_count = int(counts.max()) if counts.numel() else 0
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    return {
        "x": xs,
        "counts": counts,
        "offsets": offsets,
        "max_count": max_count,
        "inverse": inverse,
        "target": target,
        "weights": weights,
    }


def _profile_components(moe: MoE, x: torch.Tensor, backend: str, fused: bool,
                        *, warmup: int, iterations: int, repeats: int) -> dict:
    with torch.inference_mode():
        top_vals, top_idx = moe._route(x)
        route_ms = cuda_median_ms(
            lambda: moe._route(x), warmup=warmup,
            iterations=iterations, repeats=repeats)

        state: dict = {}
        # Keep conditional construction explicit; this is the actual timed
        # dispatch closure, including the segment count-to-host grid bound.
        if backend == "loop":
            dispatch_fn = lambda: state.update(_loop_dispatch(moe, x, top_vals, top_idx))
        else:
            dispatch_fn = lambda: state.update(_segment_dispatch(moe, x, top_vals, top_idx))
        dispatch_ms = cuda_median_ms(
            dispatch_fn, warmup=warmup, iterations=iterations, repeats=repeats)
        dispatch_fn()

        if backend == "loop":
            def gate_up_fn():
                state["gates"] = [item["module"].gate_proj(item["x"])
                                  for item in state["experts"]]
                state["ups"] = [item["module"].up_proj(item["x"])
                                for item in state["experts"]]

            def activation_fn():
                state["hidden"] = [F.silu(g) * u
                                    for g, u in zip(state["gates"], state["ups"])]

            def down_fn():
                state["ys"] = [item["module"].down_proj(h)
                               for item, h in zip(state["experts"], state["hidden"])]

            def scatter_fn():
                out = torch.zeros_like(x)
                for item, y in zip(state["experts"], state["ys"]):
                    out.index_add_(0, item["target"], y * item["weights"][:, None])
                state["out"] = out
        else:
            tables = moe._ensure_int4_segment_tables(x)
            group_size = (moe.experts[0].gate_proj.w_int4.shape[1] * 2
                          // moe.experts[0].gate_proj.w_int4_scale.shape[1])

            def gate_up_fn():
                plan = state
                if fused:
                    state["gup"] = moe_segment_int4_gup(
                        plan["x"], tables["gate_w"], tables["gate_s"],
                        tables["up_w"], tables["up_s"], plan["offsets"],
                        plan["counts"], moe.moe_intermediate_size, group_size,
                        max_tokens_per_expert=plan["max_count"])
                else:
                    state["gate"] = moe_segment_int4_mm(
                        plan["x"], tables["gate_w"], tables["gate_s"],
                        plan["offsets"], plan["counts"],
                        moe.moe_intermediate_size, group_size,
                        max_tokens_per_expert=plan["max_count"])
                    state["up"] = moe_segment_int4_mm(
                        plan["x"], tables["up_w"], tables["up_s"],
                        plan["offsets"], plan["counts"],
                        moe.moe_intermediate_size, group_size,
                        max_tokens_per_expert=plan["max_count"])

            def activation_fn():
                if fused:
                    gate, up = state["gup"].chunk(2, dim=-1)
                else:
                    gate, up = state["gate"], state["up"]
                state["hidden"] = F.silu(gate) * up

            def down_fn():
                plan = state
                state["ys"] = moe_segment_int4_mm(
                    plan["hidden"], tables["down_w"], tables["down_s"],
                    plan["offsets"], plan["counts"], moe.hidden_size,
                    group_size, max_tokens_per_expert=plan["max_count"])

            def scatter_fn():
                plan = state
                y_slots = plan["ys"][plan["inverse"]] * plan["weights"][:, None]
                out = torch.zeros_like(x)
                out.index_add_(0, plan["target"], y_slots)
                state["out"] = out

        gate_up_fn()
        gate_up_ms = cuda_median_ms(
            gate_up_fn, warmup=warmup, iterations=iterations, repeats=repeats)
        activation_fn()
        activation_ms = cuda_median_ms(
            activation_fn, warmup=warmup, iterations=iterations, repeats=repeats)
        down_fn()
        down_ms = cuda_median_ms(
            down_fn, warmup=warmup, iterations=iterations, repeats=repeats)
        scatter_ms = cuda_median_ms(
            scatter_fn, warmup=warmup, iterations=iterations, repeats=repeats)
        return {
            "route_topk_ms": route_ms,
            "dispatch_sort_gather_ms": dispatch_ms,
            "gate_up_gemm_ms": gate_up_ms,
            "silu_mul_ms": activation_ms,
            "down_gemm_ms": down_ms,
            "weighted_scatter_ms": scatter_ms,
        }


def _set_variant(moes: list[MoE], backend: str, fused: bool) -> None:
    for moe in moes:
        moe.int4_backend = backend
        moe.fuse_int4_gate_up = fused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?",
                        default=os.path.expanduser("~/huggingface/DeepSeek-V2-Lite"))
    parser.add_argument("--int4-group-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--token-batches", default="1,4,16,64,128")
    parser.add_argument("--profile-token-batches", default="1,16,64")
    parser.add_argument("--correctness-seeds", default="7,17,42",
                        help="independent real-weight layer input seeds")
    parser.add_argument("--correctness-token-batches", default="1,8,32,128",
                        help="MoE row counts to cover in the correctness matrix")
    parser.add_argument("--correctness-all-layers", action="store_true",
                        help="repeat the correctness matrix on every real MoE layer")
    parser.add_argument("--all-layer-correctness-seeds", default="7,42")
    parser.add_argument("--all-layer-correctness-token-batches", default="1,8,32")
    parser.add_argument("--correctness-atol", type=float, default=2e-2)
    parser.add_argument("--correctness-rtol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="results/moe_int4_ablation.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MoE INT4 timing requires a CUDA GPU")

    batches = _parse_int_list(args.token_batches, "--token-batches")
    profile_batches = set(_parse_int_list(
        args.profile_token_batches, "--profile-token-batches"))
    correctness_seeds = _parse_int_list(
        args.correctness_seeds, "--correctness-seeds")
    correctness_batches = _parse_int_list(
        args.correctness_token_batches, "--correctness-token-batches")
    all_layer_seeds = _parse_int_list(
        args.all_layer_correctness_seeds, "--all-layer-correctness-seeds")
    all_layer_batches = _parse_int_list(
        args.all_layer_correctness_token_batches,
        "--all-layer-correctness-token-batches")
    if args.correctness_atol < 0 or args.correctness_rtol < 0:
        raise ValueError("correctness tolerances must be non-negative")
    max_batch = max(batches)
    print(f"Loading pure INT4 model from {args.model}", flush=True)
    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=max(8, max_batch),
        max_num_batched_tokens=max(1024, max_batch),
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization="int4",
        int4_group_size=args.int4_group_size,
        int4_dense_path=False,
        streaming_load=True,
        kv_swap=False,
        enforce_eager=True,
        mixed_cudagraph=False,
        moe_int4_backend="loop",
    )
    try:
        moes = [module for module in llm.model_runner.model.modules()
                if isinstance(module, MoE)]
        if not moes:
            raise RuntimeError("the selected checkpoint has no MoE layers")
        moe = moes[0]
        if not all(linear.int4 for expert in moe.experts for linear in
                   (expert.gate_proj, expert.up_proj, expert.down_proj)):
            raise RuntimeError("the selected MoE expert weights did not load as INT4")
        print(f"GPU={torch.cuda.get_device_name()} MoE layers={len(moes)} "
              f"shape H={moe.hidden_size} I={moe.moe_intermediate_size} "
              f"E={moe.num_experts} top_k={moe.top_k}", flush=True)

        results = {
            "environment": {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": __import__("triton").__version__,
                "model": os.path.abspath(args.model),
                "quantization": "int4",
                "int4_group_size": args.int4_group_size,
                "expert_shape": {
                    "hidden_size": moe.hidden_size,
                    "intermediate_size": moe.moe_intermediate_size,
                    "num_experts": moe.num_experts,
                    "top_k": moe.top_k,
                },
            },
            "benchmark_config": {
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "token_batches": batches,
                "profile_token_batches": sorted(profile_batches),
                "correctness_seeds": correctness_seeds,
                "correctness_token_batches": correctness_batches,
                "correctness_all_layers": args.correctness_all_layers,
                "all_layer_correctness_seeds": all_layer_seeds,
                "all_layer_correctness_token_batches": all_layer_batches,
                "correctness_atol": args.correctness_atol,
                "correctness_rtol": args.correctness_rtol,
                "warmup": args.warmup,
                "iterations_per_sample": args.iterations,
                "sample_repeats": args.repeats,
            },
            "correctness": {},
            "runs": {},
        }
        print("\nReal-weight layer correctness matrix", flush=True)
        results["correctness"] = verify_real_layer(
            moe, correctness_batches, correctness_seeds,
            atol=args.correctness_atol, rtol=args.correctness_rtol)
        if not results["correctness"]["within_tolerance"]:
            raise AssertionError(
                "segmented INT4 MoE output exceeded the configured correctness tolerance")
        if args.correctness_all_layers:
            all_layers = {}
            for layer_index, layer_moe in enumerate(moes):
                print(f"\nReal-weight MoE layer {layer_index + 1}/{len(moes)}",
                      flush=True)
                all_layers[str(layer_index)] = verify_real_layer(
                    layer_moe, all_layer_batches, all_layer_seeds,
                    atol=args.correctness_atol, rtol=args.correctness_rtol,
                    label=f"layer={layer_index:02d} ")
            results["correctness"]["all_moe_layers"] = {
                "layer_count": len(moes),
                "seeds": all_layer_seeds,
                "token_batches": all_layer_batches,
                "within_tolerance": all(
                    result["within_tolerance"] for result in all_layers.values()),
                "layers": all_layers,
            }
            if not results["correctness"]["all_moe_layers"]["within_tolerance"]:
                raise AssertionError(
                    "at least one real-weight MoE layer exceeded correctness tolerance")

        for token_batch in batches:
            dtype = moe.experts[0].gate_proj.w_int4_scale.dtype
            x = torch.randn(token_batch, moe.hidden_size, device="cuda", dtype=dtype)
            if not moe._can_segment_int4(x):
                raise RuntimeError(
                    "segmented INT4 is not active for this MoE layer: "
                    f"{moe._int4_segment_reason}")
            with torch.inference_mode():
                expected = moe._forward_loop(x)
            print(f"\nMoE input tokens={token_batch}", flush=True)
            for name, backend, fused in VARIANTS:
                _set_variant([moe], backend, fused)
                forward = (moe._forward_loop if backend == "loop"
                           else moe._forward_segment_int4)
                with torch.inference_mode():
                    actual = forward(x)
                torch.cuda.synchronize()
                error_metrics = _error_metrics(actual, expected)
                median_ms = cuda_median_ms(
                    lambda: forward(x), warmup=args.warmup,
                    iterations=args.iterations, repeats=args.repeats)
                row = {
                    "median_ms": median_ms,
                    **error_metrics,
                }
                if token_batch in profile_batches:
                    row["stages"] = _profile_components(
                        moe, x, backend, fused, warmup=max(1, args.warmup // 2),
                        iterations=max(1, args.iterations // 2), repeats=args.repeats)
                results["runs"].setdefault(str(token_batch), {})[name] = row
                print(f"  {name:14s} {median_ms:8.3f} ms  "
                      f"rel_err={row['relative_error']:.3e} "
                      f"max_abs={row['max_abs_error']:.3e}", flush=True)
                if "stages" in row:
                    stages = " ".join(f"{key.removesuffix('_ms')}={value:.3f}"
                                      for key, value in row["stages"].items())
                    print(f"    stages (ms): {stages}", flush=True)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
        print(f"\nSaved {out}", flush=True)
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

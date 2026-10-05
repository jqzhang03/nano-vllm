# Repository Guidelines

## Project Structure & Module Organization

All source code lives under `nanovllm/`:

- `engine/` – orchestration and storage: `LLMEngine`, `Scheduler`, `BlockManager`, `Sequence`, `ModelRunner`, host-side KV transfer
- `server.py` – OpenAI-style HTTP/SSE service and in-memory conversation handling
- `layers/` – model operations: attention, linear, layernorm, rotary embedding, sampler
- `models/` – model definitions and `model_type` registry
- `utils/` – weight loading and the global inference context

Root-level `example.py` and `bench.py` are runnable demos; `assets/` holds images; `pyproject.toml` defines packaging and dependencies. `benchmarks/` holds the performance tooling (see `INTERVIEW.md` §10 基准档案): `bench.py` (throughput/latency/SLO/vLLM comparison), `context_concurrency.py` (concurrency × context-length sweep and live KV capacity budget), `kv_fp8_calibrate.py` (real-text FP8 KV calibration and held-out precision comparison), `scheduling_ablation.py` (continuous arrivals, scheduler-policy ablations, and prefix/deferred-free lifecycle counters), `admission_ablation.py` (online accept/defer/reject comparison), `profiler.py` (torch.profiler prefill/decode breakdown), plus dev scripts to drive runs from Windows into WSL. Tests live under `tests/`; the pure-Python portions do not need a GPU. `test_prefix_feature_lifecycle.py` covers generation invalidation, cache allocation/eviction, swap release, deferred-free commit, and metric reset boundaries.

## Build, Test, and Development Commands

```bash
pip install -e .        # install the package in editable mode
python example.py       # end-to-end inference; expects a local Qwen3-0.6B checkpoint
python bench.py         # throughput benchmark (256 sequences)
python benchmarks/bench.py --num-seqs 256            # full metrics: TTFT/TPOT/E2E/p50/p99/SLO
python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512   # prefix-cache workload
python benchmarks/context_concurrency.py --context-lengths 512,1024,2048,4096 --concurrency 1,2,4,8,16 --repeats 3
python benchmarks/kv_fp8_calibrate.py --calibration-file calibration.jsonl --eval-file heldout.jsonl --output results/kv_eval.json
python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64  # prefill/decode breakdown
python benchmarks/scheduling_ablation.py --num-seqs 128 --arrival-rate 16 --arrival-mode poisson
python benchmarks/admission_ablation.py --num-seqs 128 --arrival-rate 16 --arrival-mode poisson
nanovllm-serve /path/to/model --host 127.0.0.1 --port 8000         # OpenAI-style HTTP/SSE service
```

`benchmarks/bench.py` consumes the timing instrumentation on `Sequence` (`t_submitted`/`t_first_token`/`t_completed`, driver-side only) exported through `LLMEngine.collect_metrics()`; keep those fields when touching the engine. Never name a script `profile.py` under `benchmarks/` — it shadows the stdlib `profile` module that torch's `cProfile` import chain needs.

Requires Python 3.10–3.12 and an NVIDIA GPU; `flash-attn`, `triton`, and NCCL are hard dependencies. There is no build step beyond `pip install`.

## Coding Style & Naming Conventions

- Python, 4-space indentation, type hints on all public signatures.
- Prefer dataclasses with `slots=True` for config-like objects (`Config`, `SamplingParams`, `Context`).
- Name modules after their layer (`engine/`, `layers/`) and classes after the concept (`Scheduler`, `BlockManager`, `Sequence`).
- No linter or formatter is configured; keep new code consistent with surrounding style.

## Testing Guidelines

A pure-Python pytest suite exists under `tests/` (scheduler/block-manager/registration, model metadata, and host KV-transfer logic; no GPU required for those cases). When adding tests:

- Place them under `tests/` with `test_*.py` naming.
- Run with `pytest`.
- Scheduler and block-manager logic is pure Python and testable without a GPU; GPU paths (`ModelRunner`, attention kernels) require CUDA hardware.

## Commit & Pull Request Guidelines

Git history uses short imperative summaries, often `fix(scope): message` (e.g., `fix(model_runner): correct seqlen_k to chunk boundary`). Changes land via pull requests on branches named after the change (e.g., `fix/decoding-positions`). PR descriptions should state the problem, the change, and any benchmark impact; link related issues when present.

## Architecture Notes for Contributors

Offline inference runs as: `LLM.generate` → `LLMEngine` submits sequences → scheduler picks prefill/decode/mixed/spec batches → `ModelRunner` packs tensors → registry-selected model forward → sampler returns tokens. The `Context` singleton in `nanovllm/utils/context.py` passes per-step tensors (slot mappings, block tables) from the runner into attention kernels; keep this interface stable when modifying layers.

The serving path accepts requests continuously through `nanovllm/server.py` and feeds them to one engine worker. Predictive admission estimates prompt work, output length from request caps and a global completion-length EWMA, online Prefill/Decode rates, queue pressure, and projected KV use; it accepts, FIFO-defers, or rejects requests with a bounded wait and a hard queue cap. Managed conversations have a prompt-token budget and cumulative prefix-hit statistics; their text history stays in host memory while KV reuse is supplied by the shared prefix cache. Scheduler policies include mixed/chunked prefill, generation-invalidated lazy Top-W cache-affinity features (`prefix_feature_cache=False` reparses each query for ablation), aging, recompute-aware preemption, and TTFT/TPOT-target-aware quotas. Request-level `tpot_slo_ms` and optional `default_tpot_slo_ms` targets prioritize active decode work using a request-local inter-token EWMA; the policy adapts prefill budgets and reports target attainment, but does not provide a hard real-time guarantee. Bounded multi-step decode (default maximum 4 forwards) runs only on pure non-speculative decode windows and stops when local prefill work is waiting; PD mode also yields to its separate prefill queue. SSE delivers tokens accumulated within a burst together. Use `multi_step_decode=False` / `--no-multi-step-decode` and `tpot_aware_scheduling=False` / `--no-tpot-aware-scheduling` for ablations. Idle reusable prefix blocks are reclaimed LRU under allocation pressure or by an optional configured free-block cap; sequence completion defers physical release until the whole postprocess batch is committed. Ordinary MHA mixed batches can use bounded lazy CUDA Graph capture up to a configurable token cap; MLA, rolling/split, oversized, and spec-mixed paths fall back to eager. `execution_mode="auto"` selects mixed scheduling on one visible GPU and experimental local PD separation on a compatible multi-GPU setup. CPU KV swap supports TP=1 MHA/MLA caches in `auto` and FP8 formats; transfers are synchronous and use a bounded host buffer.

## Documentation maintenance

When changing a feature, update the relevant Markdown in the same change: `README.md` for user-facing behavior and limits, `AGENTS.md` / `CLAUDE.md` for development and architecture guidance, `INTERVIEW.md` for the maintained implementation/roadmap/benchmark record, and experiment reports when their evidence or status changes. Keep dated benchmark measurements as historical observations; do not rewrite them to imply the new code was measured. State supported combinations, fallbacks, and unverified paths next to feature claims.

# Repository Guidelines

## Project Structure & Module Organization

All source code lives under `nanovllm/`:

- `engine/` – orchestration and storage: `LLMEngine`, `Scheduler`, `BlockManager`, `Sequence`, `ModelRunner`, host-side KV transfer
- `server.py` – OpenAI-style HTTP/SSE service and in-memory conversation handling
- `layers/` – model operations: attention, linear, layernorm, rotary embedding, sampler
- `models/` – model definitions and `model_type` registry
- `utils/` – weight loading and the global inference context

Root-level `example.py` and `bench.py` are runnable demos; `assets/` holds images; `pyproject.toml` defines packaging and dependencies. `benchmarks/` holds the performance tooling (see `INTERVIEW.md` §10 基准档案): `bench.py` (throughput/latency/SLO/vLLM comparison), `context_concurrency.py` (concurrency × context-length sweep and live KV capacity budget), `kv_fp8_calibrate.py` (real-text FP8 KV calibration and held-out precision comparison), `scheduling_ablation.py` (continuous arrivals, scheduler-policy ablations, and prefix/deferred-free lifecycle counters), `admission_ablation.py` (online accept/defer/reject comparison across admission-off, cache-blind, and cache-aware variants), `prefix_cache_probe.py` / `prefix_cache_verify.py` / `prefix_estimate_check.py` (prefix-hit estimate vs committed reuse, in-process and over HTTP), `profiler.py` (torch.profiler prefill/decode breakdown), plus dev scripts to drive runs from Windows into WSL. Tests live under `tests/`; the pure-Python portions do not need a GPU. `test_prefix_feature_lifecycle.py` covers generation invalidation, cache allocation/eviction, swap release, deferred-free commit, and metric reset boundaries; `test_prefix_estimate.py` covers the cross-thread snapshot estimate, version-based staleness, and `pending_free` exclusion; `test_decode_burst_yield.py` covers the burst return contract and the arrival-pressure predicate; `test_tpot_throttle.py` covers per-request metric idempotency and the TPOT throttle deadband/queue gate.

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
python benchmarks/admission_ablation.py --num-seqs 128 --arrival-rate 16 --arrival-mode poisson --shared-prefix-len 512
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

The serving path accepts requests continuously through `nanovllm/server.py` and feeds them to one engine worker. Predictive admission estimates prompt work, output length from request caps and a global completion-length EWMA, online Prefill/Decode rates, queue pressure, and projected KV use; it accepts, FIFO-defers, or rejects requests with a bounded wait and a hard queue cap. Prompt work is charged net of the block-aligned prefix the cache can serve: while the worker is idle the manager rebuilds one snapshot per prefix-cache pool (`{hash: (token_ids, block_id, in_free)}`, keyed by `kv_generation`) and walks the candidate's chained block hashes through `_estimate_prefix_cache_match`, comparing token ids as well as hashes; the match names the physical blocks so `_projected_kv_blocks` can promise each block to a single request instead of over-counting a shared prefix. This runs on the event loop and must not read the live tables the engine thread is mutating. Responses report `admission.estimated_cache_hit_tokens` next to `context.prefix_cache_hit_tokens`, and `/health` adds the engine-side cumulative reuse counters; `--no-cache-affinity-admission` disables both scheduler cache affinity and the cache-aware estimate. Managed conversations have a prompt-token budget and cumulative prefix-hit statistics; their text history stays in host memory while KV reuse is supplied by the shared prefix cache. Scheduler policies include mixed/chunked prefill, generation-invalidated lazy Top-W cache-affinity features (`prefix_feature_cache=False` reparses each query for ablation), aging, recompute-aware preemption, and TTFT/TPOT-target-aware quotas. Request-level `tpot_slo_ms` and optional `default_tpot_slo_ms` targets prioritize active decode work using a request-local inter-token EWMA; the policy adapts prefill budgets and reports target attainment, but does not provide a hard real-time guarantee. Bounded multi-step decode (default maximum 4 forwards) runs only on pure non-speculative decode windows and stops when local prefill work is waiting; a burst also ends when the arrival signal is set (`decode_burst_yield_on_arrival`: `LLMEngine.request_decode_burst_yield()` driven by server-side `_sync_decode_burst_pressure()`, or the `set_decode_burst_yield_callback()` probe used by trace replay). The queued-prefill check alone cannot fire in mixed mode, because requests are only submitted between engine steps and a burst starts only with an empty waiting queue, so the arrival signal is the path that matters there (measured 53.6 ms vs 68.3 ms p50 for a request arriving mid-burst, signal on/off, `max_decode_steps=8`). TPOT-aware quotas shrink the per-step prefill budget toward `prefill_reserve_tokens` when an active TPOT target is missed; two guards keep that from starving prefill — `tpot_prefill_throttle_margin` (deadband + proportional ramp instead of a cliff that saturates on any overshoot) and `tpot_throttle_max_waiting` (disable the throttle once the waiting queue is deep, where the bottleneck is prefill throughput, not decode). Without them the throttle self-locked at the quota floor for every target from 10 to 160 ms; with them the same 128-request / 16 req/s / 20 ms-target run went 350 → 1140 tok/s, TTFT p50 44.5 s → 1.6 s, and a target the hardware cannot deliver is still missed (`tpot_starvation` in the ablation JSON reports budget-scale avg/min and starved-step counts). Per-request metrics are recorded through `LLMEngine._record_once()`, which is idempotent per `seq_id`: a sequence that finishes inside a burst is reported by both the burst and the outer batch, and the duplicate records made benchmarks stop early (128 requests → 95 distinct records). SSE delivers tokens accumulated within a burst together. Use `multi_step_decode=False` / `--no-multi-step-decode` and `tpot_aware_scheduling=False` / `--no-tpot-aware-scheduling` for ablations. Idle reusable prefix blocks are reclaimed LRU under allocation pressure or by an optional configured free-block cap; sequence completion defers physical release until the whole postprocess batch is committed. Ordinary MHA mixed batches can use bounded lazy CUDA Graph capture up to a configurable token cap; MLA, rolling/split, oversized, and spec-mixed paths fall back to eager. `execution_mode="auto"` selects mixed scheduling on one visible GPU and experimental local PD separation on a compatible multi-GPU setup. CPU KV swap supports TP=1 MHA/MLA caches in `auto` and FP8 formats; transfers are synchronous and use a bounded host buffer.

## Documentation maintenance

When changing a feature, update the relevant Markdown in the same change: `README.md` for user-facing behavior and limits, `AGENTS.md` / `CLAUDE.md` for development and architecture guidance, `INTERVIEW.md` for the maintained implementation/roadmap/benchmark record, and experiment reports when their evidence or status changes. Keep dated benchmark measurements as historical observations; do not rewrite them to imply the new code was measured. State supported combinations, fallbacks, and unverified paths next to feature claims.

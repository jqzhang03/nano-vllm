# Repository Guidelines

## Project Structure & Module Organization

All source code lives under `nanovllm/`:

- `engine/` – orchestration and storage: `LLMEngine`, `Scheduler`, `BlockManager`, `Sequence`, `ModelRunner`, host-side KV transfer
- `server.py` – OpenAI-style HTTP/SSE service and in-memory conversation handling
- `layers/` – model operations: attention, linear, layernorm, rotary embedding, sampler
- `models/` – model definitions and `model_type` registry
- `utils/` – weight loading and the global inference context

Root-level `example.py` and `bench.py` are runnable demos; `assets/` holds images; `pyproject.toml` defines packaging and dependencies. `benchmarks/` holds the performance tooling (see `INTERVIEW.md` §10 基准档案): `bench.py` (throughput/latency/SLO/vLLM comparison), `context_concurrency.py` (concurrency × context-length sweep and live KV capacity budget), `kv_fp8_calibrate.py` (real-text FP8 KV calibration and held-out precision comparison), `scheduling_ablation.py` (continuous arrivals, scheduler-policy ablations, and prefix/deferred-free lifecycle counters), `admission_ablation.py` (online accept/defer/reject comparison across admission-off, cache-blind, and cache-aware variants), `compare_quantization.py` (paired INT4/NVFP4 arrival-trace comparison), `prefix_cache_probe.py` / `prefix_cache_verify.py` / `prefix_estimate_check.py` (prefix-hit estimate vs committed reuse, in-process and over HTTP), `profiler.py` (torch.profiler prefill/decode breakdown), plus dev scripts to drive runs from Windows into WSL. Tests live under `tests/`; the pure-Python portions do not need a GPU. `test_nvfp4.py` covers E2M1 packing, both scale layouts, software/native CUDA GEMMs, and empty MoE expert batches; `test_quantization_comparison.py` checks paired workload validation; `test_prefix_feature_lifecycle.py` covers generation invalidation, cache allocation/eviction, swap release, deferred-free commit, and metric reset boundaries; `test_prefix_estimate.py` covers the cross-thread snapshot estimate, version-based staleness, and `pending_free` exclusion; `test_decode_burst_yield.py` covers the burst return contract and the arrival-pressure predicate; `test_tpot_throttle.py` covers per-request metric idempotency and the TPOT throttle deadband/queue gate.

`benchmarks/compare_ablation.py` compares scheduling/admission reports only when their exact
prompt, sampling, arrival-time, seed, and SLO workload fingerprints match. Admission reports
include joint TTFT/TPOT SLO request and output-token goodput.

`benchmarks/moe_int4_ablation.py` profiles real packed-INT4 MoE layers by input-token count
and reports independent routing, dispatch, gate/up, activation, down, and scatter timings.
`benchmarks/moe_int4_fuse_sweep.py` compares split/fused serving on paired seeded arrival
traces across request rates and output lengths, including token fingerprints and SLO goodput.
`benchmarks/moe_int4_fixed_shape_diag.py` isolates split/fused correctness and performance with
fixed prompts and output traces, and verifies model-input fingerprints and model-call shapes.
`benchmarks/moe_int4_arrival_diag.py` records online request-to-row traces and captures target
logits, decoder/MoE activations, logical MLA KV rows, and frozen-KV row-order replays.
`tests/test_moe_int4_segment.py` checks loop parity for fused/split gate/up and empty expert
segments. Pure INT4 MoE segmentation is opt-in and currently supports TP=1 with BF16/FP16
activations; AWQ, `w_deq`, and other quantization modes use the loop path. CUDA Graph capture
is unverified.

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
python benchmarks/moe_int4_ablation.py ~/huggingface/DeepSeek-V2-Lite --int4-group-size 64 --token-batches 1,8,32
python benchmarks/moe_int4_fixed_shape_diag.py ~/huggingface/DeepSeek-V2-Lite --num-seqs 24 --input-length 64 --seeds 7,17,42 --temperatures 0.6,0.8,1.0 --output-lengths 64,128 --repeats 3 --warmup-repeats 2 --output results/moe_int4_fixed_shape_diag_20261012.json
python benchmarks/compare_quantization.py results/int4.json results/nvfp4.json --variant all_on
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

The serving path accepts requests continuously through `nanovllm/server.py` and feeds them to one engine worker. Predictive admission estimates prompt work, output length from request caps and a global completion-length EWMA, online Prefill/Decode rates, queue pressure, and projected KV use; it accepts, FIFO-defers, or rejects requests with a bounded wait and a hard queue cap. Prompt work is charged net of the block-aligned prefix the cache can serve: while the worker is idle the manager rebuilds one snapshot per prefix-cache pool (`{hash: (token_ids, block_id, in_free)}`, keyed by `kv_generation`) and walks the candidate's chained block hashes through `_estimate_prefix_cache_match`, comparing token ids as well as hashes; the match names the physical blocks so `_projected_kv_blocks` can promise each block to a single request instead of over-counting a shared prefix. This runs on the event loop and must not read the live tables the engine thread is mutating. Responses report `admission.estimated_cache_hit_tokens` next to `context.prefix_cache_hit_tokens`, and `/health` adds the engine-side cumulative reuse counters; `--no-cache-affinity-admission` disables both scheduler cache affinity and the cache-aware estimate. Managed conversations have a prompt-token budget and cumulative prefix-hit statistics; their text history stays in host memory while KV reuse is supplied by the shared prefix cache. Scheduler policies include mixed/chunked prefill, generation-invalidated lazy Top-W cache-affinity features (`prefix_feature_cache=False` reparses each query for ablation), aging, recompute-aware preemption, and TTFT/TPOT-target-aware quotas. Request-level `tpot_slo_ms` and optional `default_tpot_slo_ms` targets prioritize active decode work using a request-local inter-token EWMA; the policy adapts prefill budgets and reports target attainment, but does not provide a hard real-time guarantee. Bounded multi-step decode (default maximum 4 forwards) runs only on pure non-speculative decode windows and stops when local prefill work is waiting; a burst also ends when the arrival signal is set (`decode_burst_yield_on_arrival`: `LLMEngine.request_decode_burst_yield()` driven by server-side `_sync_decode_burst_pressure()`, or the `set_decode_burst_yield_callback()` probe used by trace replay). The queued-prefill check alone cannot fire in mixed mode, because requests are only submitted between engine steps and a burst starts only with an empty waiting queue, so the arrival signal is the path that matters there (measured 53.6 ms vs 68.3 ms p50 for a request arriving mid-burst, signal on/off, `max_decode_steps=8`). TPOT-aware quotas shrink the per-step prefill budget toward `prefill_reserve_tokens` when an active TPOT target is missed; two guards keep that from starving prefill — `tpot_prefill_throttle_margin` (deadband + proportional ramp instead of a cliff that saturates on any overshoot) and `tpot_throttle_max_waiting` (disable the throttle once the waiting queue is deep, where the bottleneck is prefill throughput, not decode). Without them the throttle self-locked at the quota floor for every target from 10 to 160 ms; with them the same 128-request / 16 req/s / 20 ms-target run went 350 → 1140 tok/s, TTFT p50 44.5 s → 1.6 s, and a target the hardware cannot deliver is still missed (`tpot_starvation` in the ablation JSON reports budget-scale avg/min and starved-step counts). Per-request metrics are recorded through `LLMEngine._record_once()`, which is idempotent per `seq_id`: a sequence that finishes inside a burst is reported by both the burst and the outer batch, and the duplicate records made benchmarks stop early (128 requests → 95 distinct records). SSE delivers tokens accumulated within a burst together. Use `multi_step_decode=False` / `--no-multi-step-decode` and `tpot_aware_scheduling=False` / `--no-tpot-aware-scheduling` for ablations. Idle reusable prefix blocks are reclaimed LRU under allocation pressure or by an optional configured free-block cap; sequence completion defers physical release until the whole postprocess batch is committed. MHA mixed batches, including Mistral rolling and Gemma-2 split, can use bounded lazy CUDA Graph capture up to a configurable token cap. MLA mixed, spec-mixed, oversized, and sparse24 graph forwards fall back to eager; decode graphs cover supported rolling/split inputs, while rolling verification remains eager. `execution_mode="auto"` selects mixed scheduling on one visible GPU and experimental local PD separation on a compatible multi-GPU setup. CPU KV swap supports TP=1 MHA/MLA caches in `auto` and FP8 formats, including Mistral ring and Gemma-2 split pools; TP>1 falls back to recompute, transfers are synchronous, and PD mode retains its separate KV-format/rolling constraints. New combinations need GPU validation.

NVFP4 weights use E2M1 nibbles, E4M3 scales per 16 K values, and one FP32 tensor scale. `nvfp4_backend="software"` is the default Triton BF16-dequantization path; `nvfp4_backend="cublaslt"` dynamically quantizes BF16 activations and calls cuBLASLt block-scaled FP4 MMA on Blackwell (SM100+) with cuBLASLt 12.8+. Native weights store one swizzled scale buffer. MoE routers stay unquantized, routed experts still use the per-expert path, empty expert batches skip the GEMM, and MLA `kv_b` retains a small `w_deq` view for absorbed decode. Compare both NVFP4 backends and INT4 with the same `scheduling_ablation.py` trace and `compare_quantization.py`; DeepSeek-V2-Lite INT4 requires group size 64. The tested native combination is TP=1 with eager execution on RTX 5060 Ti / SM120; CUDA Graph capture and TP>1 are unverified. On one 8-request trace, heuristic-ranked cuBLASLt reached the same joint-SLO goodput as INT4 and higher goodput than software NVFP4, while TTFT remained slower than INT4; see `INTERVIEW.md` §10.3.15.

Pure packed-INT4 MoE experts can opt into `moe_int4_backend="segment"`: routed rows are sorted into expert offset/count segments and Triton runs GEMMs without padded batches, using pointer tables to retain the per-expert weight buffers. `moe_int4_fuse_gate_up=True` combines gate/up work in one launch; false enables the split ablation. The backend currently supports TP=1 BF16/FP16 activations, common supported INT4 group size, and no AWQ or `w_deq`; unsupported combinations retain the existing automatic grouped or per-expert path. CUDA Graph capture is unverified. On DeepSeek-V2-Lite, split and fused outputs matched the loop bitwise across all 26 MoE layers, three seeds, and batch sizes 1/8/32 (234 layer cases). Across 24 paired 8-request traces at 4/8 req/s and 16/64 output tokens, fused throughput had a -2.4% median delta (7 wins, 17 losses); all runs met 10 s TTFT / 600 ms TPOT. A separate 16 req/s, 64-output-token stress point measured +5.4% median but ranged from -20.3% to +31.9% across six pairs; all requests met its 150 ms TPOT target. The SLO targets were not tight enough to expose a goodput knee, and these measurements do not establish a stable fusion gain. See `INTERVIEW.md` §10.3.16.

The 2026-10-11 follow-up used three seeds × three repeats × 24 requests per condition at
16/24/32 req/s. With a 150 ms TPOT target, 64- and 128-token outputs stayed at 100% joint-SLO
attainment through 32 req/s. At 125 ms, 64-token split fell to 192/216 requests at 32 req/s
while 128-token outputs stayed at 216/216. At 100 ms, attainment fell for both lengths;
128-token fused runs reached 143/216 vs split 63/216 at 24 req/s and 141/216 vs 72/216 at
32 req/s. This is workload-specific: 100 ms fused throughput was +3.53% median over 54 pairs
(range -20.31% to +28.38%), with token parity in 42/54 pairs. See `INTERVIEW.md` §10.3.16,
`results/moe_int4_fuse_load150_20261011.json`, `results/moe_int4_fuse_load125_20261011.json`,
and `results/moe_int4_fuse_load100_long_20261011.json`.

The 2026-10-11 100–125 ms refinement used 24 requests per condition, seeds 7/17/42, five paired repeats, 24/32 req/s, and 64/128 output tokens (240 pairs). Fused throughput median delta was +2.24% for 128 tokens (70 wins/50 losses; -17.93% to +18.16%) and +0.81% for 64 tokens (64/56; -28.65% to +29.87%); joint-SLO goodput was not repeat-stable. GPU SM-clock medians were 2775 MHz and decode batch p50/max was 24 rows, but exact model-call shape histograms matched in only 53/240 pairs. Token parity was 212/240; every mismatch was seed 42, request 23, token index 48. Subsequent fixed-shape and online row-order diagnostics are recorded in §§10.3.17–10.3.18; fusion remains experimental. See `benchmarks/_moe_report.md`, `INTERVIEW.md` §10.3.16, and `results/moe_int4_fuse_load105_120_rep5_20261011.json`.

The 2026-10-12 fixed-shape diagnostic used DeepSeek-V2-Lite, TP=1 INT4 group64, eager execution, 24 all-at-once requests with 64-token synthetic prompts, seeds 7/17/42, temperatures 0.6/0.8/1.0, and fixed 64/128-token outputs. Each variant received two full-batch 128-token warmups. Random sampling matched for 432/432 request outputs and greedy matched for 144/144. Performance replayed the split token trace on both backends: all 54 pairs had matching model-input fingerprints and shape histograms. Fused throughput was +3.52% median overall (38/54 wins; -10.05% to +21.60%), with +3.49% at 64 tokens and +3.56% at 128 tokens; per-length ranges crossed zero. The seed-42/request-23/token-48 greedy forced-prefix sentinel had identical full logits and bitwise-identical inputs, routes, gate/up/down projections, and outputs across 26 MoE layers. SLO targets were report-only (60 s TTFT / 10 s TPOT), so these data do not establish serving goodput. See `benchmarks/moe_int4_fixed_shape_diag.py`, `results/moe_int4_fixed_shape_diag_20261012.json`, and `INTERVIEW.md` §10.3.17.

The 2026-10-12 online MLA follow-up replayed the seed-42, 32 req/s, 24-request, 64-in/64-out, 115 ms TPOT trace for five paired repetitions. All five split/fused target calls had different 24-row request orders. With each side's layer-1 hidden states and logical KV fixed, rehydrating the same cache and replaying under the other side's order reproduced all 24 attention outputs bitwise in both directions (10/10 state/order checks); the actual-order replays also matched live outputs. One of five pairs again diverged at request 23 / output token 48. Its current layer-1 query hidden was bitwise equal, but 19,575 of 64,512 logical KV elements differed (max abs 0.125); the attention output differed in 958/2,048 elements (max abs 9.77e-4). The evidence rules out the current decode-row permutation for this captured state and points to an earlier difference in KV formation or cache state; it does not yet locate the first differing cached position or identify a block-mapping bug. Keep fusion experimental and do not use diagnostic wall times as performance results. See `benchmarks/moe_int4_arrival_diag.py`, `benchmarks/_moe_report.md`, `INTERVIEW.md` §10.3.18, and `results/moe_int4_mla_row_permutation_rep5_20261012.json`.

For the HTTP path, `create_app()` and `nanovllm-serve` default to eager mixed prefill/decode
batches; `--mixed-cudagraph` opts into lazy capture. The direct `LLM`/`Config` default remains
unchanged. This serving default follows the 2026-10-09 Qwen3-0.6B/RTX 5060 Ti same-trace result;
other models and GPUs need their own GPU validation.

## Documentation maintenance

When changing a feature, update the relevant Markdown in the same change: `README.md` for user-facing behavior and limits, `AGENTS.md` / `CLAUDE.md` for development and architecture guidance, `INTERVIEW.md` for the maintained implementation/roadmap/benchmark record, and experiment reports when their evidence or status changes. Keep dated benchmark measurements as historical observations; do not rewrite them to imply the new code was measured. State supported combinations, fallbacks, and unverified paths next to feature claims.

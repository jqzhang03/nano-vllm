<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM

A from-scratch, vLLM-style offline inference engine built on **PyTorch + Triton + flash-attn** — no inference framework dependency. It is developed and measured on an RTX 5060 Ti 16GB (Blackwell sm_120) under WSL2; every claimed number is backed by the evidence tables in [`INTERVIEW.md`](INTERVIEW.md) (Chinese master doc) and raw JSON under `results/`.

## Key Features

* 🚀 **Scheduler** — vLLM-V1-style mixed batches (prefill rows + decode rows sharing one token budget), chunked prefill, spec batches, KV-swap preemption (bit-exact CPU swap for decode sequences, with an honest cost model).
* 📖 **Paged KV cache** — block size 256; chained-hash **prefix cache** (partial blocks included) with copy-on-write safety; **SWA rolling cache** (`rolling_cache=True`, per-sequence ring eviction to window+B — bounded decode KV, Mistral full-window / Gemma-2 local layers) incl. a split dual-pool mode for Gemma-2's alternating local/global windows.
* ⚡ **Kernels** — custom Triton attention kernels for **FP8 KV cache** (paged decode + varlen/verify; vLLM's FA3 path is Hopper-only and cannot run fp8 KV on sm_120), plus Triton GEMMs for w8a8 / int4 / fp8 / 2:4-sparse weights.
* 🎛️ **Weight quantization** — `--quantization none|w8a8|int4|awq|fp8|sparse24`: per-group int8 + SmoothQuant; int4 with an adaptive **dual-path router** (int4 kernel for bandwidth-bound small-M GEMMs, dequantized-cuBLAS elsewhere); AWQ per-layer α search; fp8 via hardware `torch._scaled_mm` on prefill; honest accuracy/perf ledger (real-text PPL, run-to-run variance documented).
* 🤖 **Speculative decoding** — n-gram / Medusa / EAGLE-1; verify step = prefix-reusing varlen prefill, fully CUDA-graphed (fixed-capacity families + zero-length padding rows); fp8-KV verify and ring-cache verify supported; acceptance keeps the target distribution exactly.
* 🧩 **Model zoo** — registry dispatch by `hf_config.model_type`; streaming layer-wise load + quantize-on-load (7B+ on 16GB).
* 🌐 **Tensor parallelism** — NCCL + shared-memory command channel, `weight_loader`-based sharding.

## Supported Models

| Family | `model_type` | Notes |
|---|---|---|
| Qwen3 / Qwen3-MoE | `qwen3` / `qwen3_moe` | MoE: router + per-expert FFN (+ grouped/segment GEMM backends) |
| Qwen2.5 | `qwen2` | 0.5B / 7B validated |
| Llama-3.x | `llama` | incl. `llama3` RoPE scaling |
| Mistral-7B | `mistral` | sliding-window attention (mask-only by default, optional rolling cache) |
| Gemma-2 | `gemma2` | alternating local/global windows, attn + final logit soft-cap, embed ×√d, RMSNorm (1+w) |
| DeepSeek-V2 | `deepseek_v2` | **MLA** (fused latent cache, absorbed-decode Triton kernel, 7.11× KV compression) + MoE with shared experts |

## Installation

```bash
pip install -e .        # the only build step
```

Python 3.10–3.12 and an NVIDIA GPU are required. On Blackwell (sm_120) use a CUDA 12.8+ torch build (`torch==2.8.0`, cu128 wheels); install flash-attn from the Dao-AILab prebuilt-wheel releases matching your torch build (PyPI ships sdists only). Details and env gotchas: [`CLAUDE.md`](CLAUDE.md).

Or install straight from the repository:

```bash
pip install git+https://github.com/jqzhang03/nano-vllm.git
```

## Model Download

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

The API mirrors vLLM's offline interface:

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

Feature flags (all live on `LLM(...)`): `quantization="int4|awq|fp8|w8a8|sparse24"`,
`kv_cache_dtype="fp8_e4m3"`, `speculative="ngram|medusa|eagle"`, `rolling_cache=True`,
`kv_swap=True` (default). `example.py` is an end-to-end demo; `benchmarks/bench.py` produces
the full TTFT/TPOT/E2E/p50/p99/SLO report and a JSON artifact under `results/`.

## Benchmark

**Test configuration (2026-08/09):** RTX 5060 Ti 16GB (sm_120, 36 SM) · WSL2 · torch 2.8.0+cu128 ·
flash-attn 2.8.3.post1 · bf16 · Qwen3-0.6B · same workload + seed + flash-attn on both sides
(vLLM 0.10.2 in an isolated env; its offline API exposes aggregate histograms only, so p50/p99
are bucket-interpolated approximations — per-request timestamps are nano-side only).

| Workload (128 seqs, in 64–128 / out 64–128) | nano-vllm | vLLM 0.10.2 | ratio |
|---|---|---|---|
| throughput | **6,587 tok/s** | 4,624 | **1.42×** |
| TPOT p50 / p99 | **13.5 / 13.7 ms** | 17.4 / 24.9 | 0.78 / 0.55 |
| E2E avg / p99 | **1.62 / 1.89 s** | 2.27 / 4.92 | 0.71 / 0.38 |

| Workload (256 seqs, in 128–1024 / out 64–512, both sides over-subscribed) | nano-vllm | vLLM 0.10.2 | ratio |
|---|---|---|---|
| throughput | **2,552 tok/s** | 1,888 | **1.35×** |
| TPOT p50 / p99 | **45.1 / 73.6 ms** | 56.0 / 149.2 | 0.81 / 0.49 |

| Long context (128 seqs, 1024 in + 128 out, 147k total tokens) | nano-vllm | vLLM 0.10.2 | ratio |
|---|---|---|---|
| throughput (fp8 KV) | **1,854 tok/s** (0 preemptions) | 1,150 (fp16) | **1.61×** |
| throughput (fp16 KV) | 1,421 tok/s | 1,150 | 1.24× |

**The differentiator:** vLLM cannot run FP8 KV cache on this card at all — its V0/V1 fp8 paths
dispatch to FA3, which is Hopper (sm_90) only. nano-vllm's custom Triton paged kernels are the
only fp8-KV implementation that runs on sm_120: capacity ×1.9, KL 0.0073, top-1 100% on aligned
logits. Single-engine peak: **5,825 tok/s** (fp8 weights, Qwen3-0.6B, bs=256).

Honest boundaries: these are single-card WSL2 numbers for a 0.6B model; decode single-step
latencies are at parity with vLLM fp16, and several features (e.g. pure-int4 large batch,
2:4 sparsity, EAGLE γ=4, KV swap on this machine) are *documented losses* — see
[`INTERVIEW.md`](INTERVIEW.md) §10 (基准档案) for the full tables, conditions, and what to
re-run after changing environment.

## Documentation

* [`CLAUDE.md`](CLAUDE.md) — architecture deep-dive (request lifecycle, Context contract, KV/paging, quantization, speculative decoding) and dev-machine gotchas.
* [`INTERVIEW.md`](INTERVIEW.md) — **master doc** (Chinese): learning route with per-feature code pointers, interview narrative, 18 pitfall stories, deep-dive Q&A, benchmark archive, and a verified line-numbered code map. (Consolidated from the former BENCHMARKS.md / LEARNING.md / INTERVIEW.md.)
* [`AGENTS.md`](AGENTS.md) — contribution guidelines (style, commit conventions, testing).
* `benchmarks/_stage2b_ext_report.md` — evidence tables for the stage-2 MLA / rolling-ring / fp8-KV-combination work.

## Tests

```bash
python -m pytest tests/ -q     # 57 cases, pure Python, no GPU required
```

## Repository Layout

```
nanovllm/
├─ engine/    LLMEngine · Scheduler · BlockManager · Sequence · ModelRunner · ngram
├─ layers/    attention (+ MLA) · linear (quant kernels) · moe · layernorm · rope · embed_head · sampler · medusa · eagle
├─ models/    per-family modules + model_type registry
└─ utils/     weight loader · Context singleton (per-step data contract)
tests/        pytest suite
benchmarks/   bench.py · profiler.py · verification probes & reports (see INTERVIEW.md §1.8)
```

## License

[MIT](LICENSE)

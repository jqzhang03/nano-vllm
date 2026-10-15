<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM

A from-scratch, vLLM-style inference engine for offline generation and local online serving, built on **PyTorch + Triton + flash-attn** — no inference framework dependency. It is developed and measured on an RTX 5060 Ti 16GB (Blackwell sm_120) under WSL2; benchmark claims are tied to dated evidence in [`INTERVIEW.md`](INTERVIEW.md) (Chinese master doc) and raw JSON under `results/`.

## Key Features

* 🚀 **Scheduler** — vLLM-V1-style mixed batches (prefill rows + decode rows sharing one token budget), chunked prefill, bounded multi-step pure decode, TTFT/TPOT-target-aware quotas, spec batches, and KV-swap preemption (bit-exact CPU swap for decode sequences, including FP8 KV byte-preserving offload).
* 📖 **Paged KV cache** — block size 256; chained-hash **prefix cache** (partial blocks included) with copy-on-write safety, generation-invalidated lazy Top-W scheduling features, and free-block LRU reclamation; **SWA rolling cache** (`rolling_cache=True`, per-sequence ring eviction to window+B — bounded decode KV, Mistral full-window / Gemma-2 local layers) incl. a split dual-pool mode for Gemma-2's alternating local/global windows.
* ⚡ **Kernels** — custom Triton attention kernels for **FP8 KV cache** (paged decode + varlen/verify), Triton GEMMs for w8a8 / int4 / NVFP4 software / fp8 / 2:4-sparse weights, and a cuBLASLt NVFP4 native MMA backend.
* 🎛️ **Weight quantization** — `--quantization none|w8a8|int4|nvfp4|awq|fp8|sparse24`: per-group int8 + SmoothQuant; int4 with an adaptive **dual-path router**; NVFP4 E2M1/E4M3 block scaling with a Triton software path or native cuBLASLt FP4 MMA on Blackwell; AWQ per-layer α search; fp8 via hardware `torch._scaled_mm` on prefill.
* 🤖 **Speculative decoding** — n-gram / Medusa / EAGLE-1; verify uses prefix-reusing varlen prefill. Eligible non-rolling pure-spec batches use fixed-capacity CUDA Graphs (including FP8 KV); ordinary MHA mixed Prefill/Decode batches use bounded, lazy shape-keyed CUDA Graphs, including Mistral rolling and Gemma-2 split-cache inputs. MLA and spec-mixed batches fall back to eager. sparse24 keeps its quantized Triton GEMM but uses eager forward because graph replay has produced mismatches in the engine probe. Acceptance preserves the target sampler's distribution.
* 🧩 **Model zoo** — registry dispatch by `hf_config.model_type`; streaming layer-wise load + quantize-on-load (7B+ on 16GB).
* 🌐 **Tensor parallelism** — NCCL + shared-memory command channel, `weight_loader`-based sharding.
* 🔀 **PD stage separation (experimental)** — separate Prefill and Decode model/KV pools on two GPUs; prompt KV moves through host memory.

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

Feature flags (all live on `LLM(...)`): `quantization="int4|nvfp4|awq|fp8|w8a8|sparse24"`,
`nvfp4_backend="software|cublaslt"`,
`kv_cache_dtype="fp8_e4m3"`, `speculative="ngram|medusa|eagle"`, `rolling_cache=True`,
`kv_swap=True` (default), and `execution_mode="auto|mixed|pd"`. `example.py` is an end-to-end demo; `benchmarks/bench.py` produces
the full TTFT/TPOT/E2E/p50/p99/SLO report and a JSON artifact under `results/`.
For pure INT4 MoE experts, `moe_int4_backend="loop|segment"` selects the per-expert or
segmented GEMM path; `moe_int4_fuse_gate_up` controls gate/up launch fusion on the segmented
path.

The 2026-10-12 DeepSeek-V2-Lite fixed-shape diagnostic matched split/fused token outputs
in every random and greedy correctness case and found bitwise-equal sentinel logits and MoE
values across all 26 MoE layers. Across 54 paired performance runs, fused output throughput
had a +3.52% median delta (38 wins, 16 losses; -10.05% to +21.60%). The range crosses zero,
so a stable gain is not established. This all-at-once batch with report-only SLO targets does
not establish online serving goodput; keep fusion experimental. See
[`INTERVIEW.md` §10.3.17](INTERVIEW.md) and
[`benchmarks/_moe_report.md`](benchmarks/_moe_report.md).

A later online diagnostic replayed five paired arrival traces and found one recurring
request-23/token-48 mismatch. The layer-1 query input was identical, but its logical KV
state differed in the failing pair. Replaying either frozen state under both observed
24-row orders produced bitwise-identical outputs, so the current batch row permutation
does not explain the mismatch. The earlier KV difference remains under investigation;
fusion stays experimental. See [`INTERVIEW.md` §10.3.18](INTERVIEW.md),
[`benchmarks/_moe_report.md`](benchmarks/_moe_report.md), and
[`benchmarks/moe_int4_arrival_diag.py`](benchmarks/moe_int4_arrival_diag.py).

### 2026-10-15 supplemental screening

The new ShareGPT screening did not support enabling the all-on scheduling bundle or
the current predictive-admission thresholds by default. Five 8 req/s scheduling pairs
showed lower output throughput and joint-SLO goodput. Admission screening covered 12
target/work points at TPOT≤20 ms and 24 soft-pressure/wait points at TPOT≤80 ms;
candidate checks used three calibration seeds across TPOT 20/40/80/120 ms and two
held-out seeds on 100 prompts at 384 req/s. At 96 req/s / 400 prompts / TPOT≤80 ms,
the cache-aware candidate rejected a median 58.2% of arrivals and produced 4458 token/s
joint goodput, versus 33.8% and 5217 token/s for admission-off. At TPOT≤20 ms it
increased goodput from zero but raised rejection by about 25 percentage points. The
held-out trace was shorter and used a different offered rate, so these measurements do
not establish a default threshold; no product rejection budget is defined.

`benchmarks/sharegpt_trace.py` creates de-identified request JSONL; reference assistant
reply length only supplies a capped `max_tokens` budget, and arrival timestamps are
synthetic. `bench.py`, `scheduling_ablation.py`, and `admission_ablation.py` retain
prompt-only JSONL compatibility and can replay stored arrival times. The results below
use synthetic Poisson arrivals, not production request logs. Shared-prefix variants use
a synthetic 256-token template, not a real system prompt.

On Qwen3-0.6B, the measured BF16 and FP8 KV pools held 425 and 835 blocks respectively.
The default cost policy recomputed its two observed preemptions. A forced-path diagnostic
with recompute cost comparison disabled produced 470 token/s with swap enabled versus
511 token/s without swap across three repeats (-8.1%); this verifies offload execution,
not a serving benefit. FP8 quality checks used 16 held-out prompts at the first decode
position, so they do not establish long-generation quality.

DeepSeek-V2-Lite INT4 segment split/fused kernels matched the loop bitwise across 260
layer/seed/batch cases, while the 36-pair online sweep had a +0.91% median throughput
delta with a -14.94% to +23.07% range and only 19/36 pairs with identical request
outputs. A follow-up 5-seed × 5-repeat condition measured +0.51% median throughput
delta (95% bootstrap CI -0.35% to +5.11%) and identical request outputs in 14/25 pairs.
Keep fusion experimental. MLA cache assembly now computes each query end as
`start + qlen`; a follow-up 16-request replay still recorded one output mismatch, so
the earlier online divergence is not fully resolved. N-gram acceptance tracked output
repetition in the sampled content classes, but no Medusa/EAGLE draft weights were
available. Full conditions and limits are in
[`benchmarks/_supplemental_report.md`](benchmarks/_supplemental_report.md); raw files
are under the ignored `results/supplemental/` directory.

### NVFP4 weight experiments

`quantization="nvfp4"` packs E2M1 values two per byte, stores one E4M3 scale per
16 K values, and keeps one FP32 scale per weight matrix. `nvfp4_backend="software"`
is the default Triton path: it dequantizes weights in-register and computes BF16 dot
products. Set `nvfp4_backend="cublaslt"` (or benchmark option
`--nvfp4-backend cublaslt`) to dynamically quantize BF16 activations and call
cuBLASLt's block-scaled native FP4 MMA. The native path requires Blackwell (SM100+)
and cuBLASLt 12.8 or newer; it was exercised on an RTX 5060 Ti (SM120), PyTorch
2.8/CUDA 12.8. Weight scales are stored once in cuBLASLt's swizzled layout. Empty
MoE expert batches return without a GEMM. The full-model measurement below used
TP=1 and eager execution; CUDA Graph capture and TP>1 remain unverified.

DeepSeek-V2-Lite INT4 runs must set group size 64 because its 10,944-wide projections
are not divisible by 128. For paired continuous-arrival comparisons, keep the trace,
seed, scheduler, and SLO flags identical. Set both aggregate and per-request TTFT
targets so joint SLO goodput uses the intended deadline:

```bash
python benchmarks/scheduling_ablation.py --model ~/huggingface/DeepSeek-V2-Lite \
  --variants all_on --num-seqs 8 --min-input-len 64 --max-input-len 64 \
  --min-output-len 16 --max-output-len 16 --arrival-mode constant --arrival-rate 1 \
  --slo-ttft-ms 5000 --min-request-ttft-slo-ms 5000 --max-request-ttft-slo-ms 5000 \
  --slo-tpot-ms 1500 --max-model-len 512 --gpu-memory-utilization 0.9 \
  --streaming-load --enforce-eager --no-swap-kv --warmup-seqs 2 \
  --quantization int4 --int4-group-size 64 --seed 7 \
  --output results/v2lite_int4_nvfp4_trace_int4.json
python benchmarks/scheduling_ablation.py --model ~/huggingface/DeepSeek-V2-Lite \
  --variants all_on --num-seqs 8 --min-input-len 64 --max-input-len 64 \
  --min-output-len 16 --max-output-len 16 --arrival-mode constant --arrival-rate 1 \
  --slo-ttft-ms 5000 --min-request-ttft-slo-ms 5000 --max-request-ttft-slo-ms 5000 \
  --slo-tpot-ms 1500 --max-model-len 512 --gpu-memory-utilization 0.9 \
  --streaming-load --enforce-eager --no-swap-kv --warmup-seqs 2 \
  --quantization nvfp4 --nvfp4-backend software --int4-group-size 64 --seed 7 \
  --output results/v2lite_int4_nvfp4_trace_nvfp4.json
python benchmarks/compare_quantization.py \
  results/v2lite_int4_nvfp4_trace_int4.json \
  results/v2lite_int4_nvfp4_trace_nvfp4.json --variant all_on \
  --output results/v2lite_int4_nvfp4_trace_comparison.json
python benchmarks/scheduling_ablation.py --model ~/huggingface/DeepSeek-V2-Lite \
  --variants all_on --num-seqs 8 --min-input-len 64 --max-input-len 64 \
  --min-output-len 16 --max-output-len 16 --arrival-mode constant --arrival-rate 1 \
  --slo-ttft-ms 5000 --min-request-ttft-slo-ms 5000 --max-request-ttft-slo-ms 5000 \
  --slo-tpot-ms 1500 --max-model-len 512 --gpu-memory-utilization 0.9 \
  --streaming-load --enforce-eager --no-swap-kv --warmup-seqs 2 \
  --quantization nvfp4 --nvfp4-backend software --int4-group-size 64 --seed 7 \
  --output results/v2lite_nvfp4_software_trace.json
python benchmarks/scheduling_ablation.py --model ~/huggingface/DeepSeek-V2-Lite \
  --variants all_on --num-seqs 8 --min-input-len 64 --max-input-len 64 \
  --min-output-len 16 --max-output-len 16 --arrival-mode constant --arrival-rate 1 \
  --slo-ttft-ms 5000 --min-request-ttft-slo-ms 5000 --max-request-ttft-slo-ms 5000 \
  --slo-tpot-ms 1500 --max-model-len 512 --gpu-memory-utilization 0.9 \
  --streaming-load --enforce-eager --no-swap-kv --warmup-seqs 2 \
  --quantization nvfp4 --nvfp4-backend cublaslt --int4-group-size 64 --seed 7 \
  --output results/v2lite_nvfp4_cublaslt_ranked_trace.json
python benchmarks/compare_quantization.py \
  results/v2lite_nvfp4_software_trace.json results/v2lite_nvfp4_cublaslt_ranked_trace.json \
  --variant all_on --output results/v2lite_nvfp4_software_vs_cublaslt_ranked_comparison.json
```

The paired report rejects different workload fingerprints or scheduler settings
and reports joint TTFT/TPOT SLO goodput, throughput, TTFT, TPOT, and E2E latency.
An earlier 2026-10-09 full streamed DeepSeek-V2-Lite run on an RTX 5060 Ti 16 GB
(PyTorch 2.8/CUDA 12.8, Triton 3.4; eight synthetic requests) used INT4's pure
packed group64 path (streaming disables its dense dual path; both modes retain the
small MLA `kv_b` float view) and the software NVFP4 backend. INT4 measured 4.51
output tok/s with 8/8 requests meeting both SLOs. Software NVFP4 measured 3.86 tok/s
and 3/8 joint-SLO requests; joint-SLO output goodput fell from 4.51 to 1.45 tok/s.
TTFT p50/p99 changed from 2.19/2.74 s to 2.80/4.59 s, and TPOT p50/p99 from
1.29/1.32 s to 1.57/1.61 s. These earlier results are preserved in
`results/v2lite_int4_nvfp4_trace_*`.

A later paired run used the same current code, seed, 8-request constant-arrival
trace, and SLO settings across INT4, software NVFP4, and native cuBLASLt NVFP4.
With the heuristic-ranked cuBLASLt algorithm, throughput / joint-SLO output goodput
were 4.42 / 4.42 tok/s for INT4, 4.46 / 3.90 tok/s for software NVFP4, and
4.41 / 4.41 tok/s for cuBLASLt NVFP4. Native completed 8/8 requests within both
SLOs, versus 7/8 for software NVFP4 and 8/8 for INT4. Native throughput and joint
goodput were within 1% of INT4 and native goodput was 13% higher than software
NVFP4. Native TTFT p50/p99 were 2.98/3.98 s, still slower than INT4's 2.01/2.42 s;
TPOT p50/p99 were 1.33/1.40 s. The initial cuBLASLt run selected the smallest
workspace algorithm and measured 4.21 tok/s / 1.58 tok/s goodput; choosing the
heuristic-ranked candidate on the same trace raised goodput to 4.41 tok/s. These
8-request results are directional, not a stable performance estimate. The full-model
run exercised native calls on real model shapes, while `tests/test_nvfp4.py` checks
GEMM values against a dequantized reference; native-backend full-model logit drift
has not been measured.

A four-case, same-input logits smoke check on the 4-layer real-weight slice using
the software NVFP4 backend found lower NVFP4 mean absolute logit drift than INT4 (0.231 vs 0.349), but higher KL
(0.054 vs 0.029) and lower top-1 agreement (75% vs 100%). This small sample does not
establish model-quality superiority; see `results/v2lite_4l_quant_accuracy.json`.
Native activation-quantization accuracy also remains unmeasured. The INT4 MoE segmented
path and gate/up fusion have now been checked across real MoE layers and paired serving
traces below. Results depend on the workload and do not establish a stable service-level
fusion gain. These measurements apply only to this GPU and model.

### Segmented INT4 MoE expert GEMM

`moe_int4_backend="segment"` sorts routed rows by expert, then uses per-expert offsets
and counts to run packed INT4 GEMMs without padding expert batches. Pointer tables refer
to the existing expert weight buffers, so the path does not duplicate the quantized
weights. `moe_int4_fuse_gate_up=True` computes both projections with one Triton launch;
set it to `False` to compare separate gate and up launches. The default backend remains
`loop`. Segmentation currently requires TP=1, BF16/FP16 activations, pure INT4 expert
weights with a common supported group size, and no AWQ scaling or dequantized `w_deq`
copy. Unsupported combinations retain the existing automatic grouped or per-expert path.
CUDA Graph capture is unverified.

The `benchmarks/moe_int4_ablation.py` profile separates routing, dispatch/sort/gather,
gate/up GEMMs, SiLU multiply, down GEMM, and weighted scatter. On DeepSeek-V2-Lite
(H=2048, I=1408, E=64, top-k=6; RTX 5060 Ti / WSL2), the first real-weight MoE layer
matched the loop output bitwise for five input seeds and 1, 8, 32, and 128 input tokens.
A wider fixed-input check covered all 26 MoE layers, three seeds, and batch sizes 1, 8,
and 32: split and fused outputs matched the loop bitwise in all 234 layer cases.

The earlier two full-model runs on one 8-request trace measured fused gate/up 3.3–19.3%
faster than split; those are preliminary observations. The expanded paired sweep used
three seeds, 4/8 requests per second, 16/64 output tokens, and two repetitions per
condition (24 paired runs, 8 requests each). Every run met the 10 s TTFT / 600 ms TPOT
targets, so joint-SLO output goodput tracked total output throughput. Fused throughput
had a -2.4% median delta versus split, winning 7/24 pairs and losing 17/24. Per-load
median deltas were -5.1% (4 req/s, 16 tokens), +2.2% (4 req/s, 64 tokens), -4.0%
(8 req/s, 16 tokens), and -2.7% (8 req/s, 64 tokens). This workload does not show a
reliable service-level fusion gain; keep the split option available and treat fusion as
an experimental choice. Continuous-arrival generated-token parity was 22/24. For the two
mismatched pairs, a repeated run of the same backend also changed its output fingerprint,
while fixed-input MoE layer parity remained exact; the sequence differences therefore
cannot be attributed to fusion alone.

An additional stress sweep used 16 requests arriving at 16 req/s, 64 output tokens,
three seeds, two repeats, and a 150 ms TPOT target (six paired measurements). Fused
throughput had a +5.4% median delta versus split, with five wins and one loss; the range
was -20.3% to +31.9%. Both backends met joint SLO for all 16 requests in every run, and
generated tokens matched in five of six pairs. Same-backend repeats also changed output
fingerprints in three of six repeat comparisons. Since every run passed SLO, the tests
have not located the joint-goodput knee; this positive median is not enough to claim a
stable fusion benefit. See `INTERVIEW.md` §10.3.16.

A follow-up sweep on 2026-10-11 used 24 requests per condition, three seeds, and three
repeats per seed (nine paired traces per cell), with 16/24/32 req/s arrival rates and
150/125/100 ms TPOT targets. At 150 ms, 64- and 128-token outputs all met joint SLO through
32 req/s. At 125 ms, 128-token outputs still passed every request, while 64-token split
reached 88.9% at 32 req/s. At 100 ms, joint-SLO attainment fell to 40.7–63.0% for 64-token
outputs and 29.2–66.2% for 128-token outputs. For the 128-token/100 ms cases, fused passed
143/216 vs split 63/216 at 24 req/s and 141/216 vs 72/216 at 32 req/s; this is a promising
load-specific signal, but per-run results remain noisy. Across the 54 pairs at 100 ms, fused
throughput had a +3.5% median delta with a -20.3% to +28.4% range, and outputs matched in
42/54 pairs. Keep fusion experimental pending more repeats around this boundary; full results
are in `results/moe_int4_fuse_load150_20261011.json`,
`results/moe_int4_fuse_load125_20261011.json`, and
`results/moe_int4_fuse_load100_long_20261011.json`.

A finer 2026-10-11 sweep measured 105/110/115/120 ms TPOT at 24/32 req/s with 64/128 output tokens, three seeds, and five paired repetitions per cell (240 pairs). For 128-token output, fused throughput had a +2.24% median delta (70 wins, 50 losses; -17.93% to +18.16%); 64-token output had +0.81% (64 wins, 56 losses; -28.65% to +29.87%). Joint-SLO goodput deltas varied widely, so the longer-output benefit is still conditional and not repeat-stable. SM-clock medians stayed at 2775 MHz (2760–2775); decode batch p50/max was 24 rows, although the full per-pair shape histograms matched in only 53/240 pairs. Greedy output parity was 212/240; all 28 mismatches were one seed-42 request at output token index 48. Keep fusion experimental while investigating this batch-sensitive divergence. See `INTERVIEW.md` §10.3.16, `benchmarks/_moe_report.md`, and `results/moe_int4_fuse_load105_120_rep5_20261011.json`.

```bash
python benchmarks/moe_int4_ablation.py ~/huggingface/DeepSeek-V2-Lite \
  --int4-group-size 64 --token-batches 1,8,32 \
  --profile-token-batches 1,8,32 --correctness-seeds 7,17,42,91,123 \
  --correctness-token-batches 1,8,32,128 --correctness-all-layers \
  --all-layer-correctness-seeds 7,17,42 \
  --all-layer-correctness-token-batches 1,8,32 \
  --output results/moe_int4_ablation.json
python benchmarks/moe_int4_fuse_sweep.py \
  --model ~/huggingface/DeepSeek-V2-Lite \
  --seeds 7,17,42 --arrival-rates 4,8 --output-lengths 16,64 \
  --repeats 2 --sampling-mode greedy \
  --output results/moe_int4_fuse_sweep.json
python benchmarks/moe_int4_fuse_sweep.py \
  --model ~/huggingface/DeepSeek-V2-Lite \
  --seeds 7,17,42 --arrival-rates 16 --output-lengths 64 --num-seqs 16 \
  --slo-ttft-ms 10000 --slo-tpot-ms 150 --repeats 2 --sampling-mode greedy \
  --output results/moe_int4_fuse_stress.json
```

`execution_mode="auto"` is the default: one visible GPU selects the regular mixed
prefill/decode scheduler; with two or more visible GPUs it selects PD separation when
`tensor_parallel_size=1`, the configured distinct device IDs are visible, and the PD
feature constraints are met. Tensor parallelism above one, speculative decoding, rolling
cache, or a non-`auto` KV dtype keeps the mixed mode.
Choose `execution_mode="mixed"` or `execution_mode="pd"` to select a mode manually. The
older `pd_separation=True/False` option remains as a compatibility alias.

### PD stage separation (experimental)

```python
llm = LLM("/YOUR/MODEL/PATH", execution_mode="pd",
          prefill_device=0, decode_device=1)
```

This mode requires two visible NVIDIA GPUs and `tensor_parallel_size=1`. Each GPU loads a
separate model replica and owns an independent KV pool. Once a prompt finishes Prefill, the
engine copies its cached KV rows to host memory and imports them into the Decode pool. The
current implementation is local to one process and stages the two batches serially; it does
not provide remote workers, network/RDMA KV transport, or asynchronous P/D overlap. It currently
requires `kv_cache_dtype="auto"`, disables prefix-cache reuse and rolling cache, and does
not support speculative decoding. The Prefill pool uses recompute if it runs short
on blocks. The Decode pool can use `kv_swap=True` to move complete sequences to CPU memory
under KV pressure. Set `kv_swap=False` to disable that behavior in the Decode pool.

### KV cache offload

`kv_swap=True` (default) lets the scheduler move a complete decode sequence's KV blocks to
CPU memory when GPU cache pressure forces preemption, then restore them without recomputing
the prompt. It supports MHA and MLA caches with `kv_cache_dtype="auto"` or `"fp8_e4m3"`;
FP8 values are staged as raw `uint8` bytes so the E4M3 bit patterns are preserved exactly.
Weight quantization is independent of the KV cache dtype. The configured `kv_swap_space_gb`
is a hard buffer budget: if the next sequence will exceed it, the scheduler uses recompute
preemption instead. KV swap currently requires `tensor_parallel_size=1`; TP>1 continues to
use recompute preemption. In PD mode, KV swap applies to the Decode pool. Mistral rolling
cache snapshots its active ring blocks; Gemma-2 split rolling snapshots both the local ring
and global full-history pools. FP8 KV is transferred as raw bytes. Weight quantization is
independent of KV storage dtype.
Transfers use ordinary CPU memory and are synchronous; this is a memory-pressure fallback,
not asynchronous layer-wise offload.

### Online API and conversation history

Install the package in WSL (FastAPI and Uvicorn are installed with it), then start the
OpenAI-compatible server from the WSL shell:

```bash
nanovllm-serve /mnt/d/models/Qwen3-0.6B --host 127.0.0.1 --port 8000
```

The server provides `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`.
Set `stream=true` for Server-Sent Events (SSE) with incremental text chunks. A single engine
worker owns GPU access and batches requests arriving while it is decoding; HTTP clients can
submit concurrently. Disconnecting a stream cancels its queued/running sequence.

The HTTP server runs mixed prefill/decode batches eagerly by default so a live request does not
pay the first-use cost of capturing a new mixed CUDA Graph shape. Pure decode graphs remain
available when supported. Pass `--mixed-cudagraph` to opt in to lazy mixed-graph capture; the
first request with each unseen shape still pays its capture cost. Direct `LLM`/`Config` use keeps
its existing mixed-graph default. This serving choice is based on the Qwen3-0.6B WSL2 results in
`INTERVIEW.md` §10.3.13; other models and GPUs need their own same-trace measurement.

Pass a `conversation_id` to keep chat history between calls. The first call can include a
system message and a user message; later calls send only the new user message. History lives
in host RAM, defaults to at most 256 sessions with a 24-hour idle timeout, and is lost when the
server exits. `GET /v1/conversations/{id}` reads a session and `DELETE` removes it. When a
prompt would exceed the model window, the server drops the oldest complete user/assistant
turns while retaining the system prompt and current user message. This is token-budgeted
history trimming. Stored sessions are text on the host; the GPU KV cache is a temporary
working set for active requests and is not pinned to a conversation between turns. To preserve
a compact model-written summary of removed turns, start with
`--context-compaction summarize`; summarization adds an inference step and may omit details.
Set `--session-token-budget` on the server or pass `session_token_budget` on a chat request
to limit the prompt-context tokens retained for that conversation (the request value updates
the session's budget). Conversation GET responses and non-streaming chat replies include
per-session turn, prompt-token, and prefix-cache-hit counters. The KV pool remains globally
bounded by the engine's startup allocation; cached free prefix blocks are reused by real hits,
then the oldest free cached block is reclaimed when an allocation needs space. Set
`--prefix-cache-max-free-blocks N` to cap idle reusable prefix blocks; zero leaves the cap unset.

Example streaming chat request from PowerShell:

```powershell
curl.exe -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen3-0.6B","conversation_id":"demo","messages":[{"role":"user","content":"你好"}],"ttft_slo_ms":300,"tpot_slo_ms":20,"stream":true}'
```

### Continuous arrivals and latency-aware scheduling

The HTTP worker keeps accepting requests while a GPU step is running, then admits the
queued requests into the next mixed batch. The request-arrival timestamp is carried through
admission so TTFT and queue-wait metrics include time spent waiting in the service queue.
Before placing a request in the engine backlog, dynamic admission estimates prefill work
from prompt tokens, expected decode work from the requested output limit and the observed
completion-length ratio, plus pressure from active work, queued work, and projected KV blocks.
When cache-affinity admission is enabled (default), the prompt work is charged net of the
block-aligned prefix the shared prefix cache can serve: the manager keeps a read-only snapshot
of the block-manager hash table (rebuilt between engine steps while the worker is idle, keyed by
a version counter) and walks the candidate's chained block hashes against it. The estimate and
the tokens actually reused are both reported (`admission.estimated_cache_hit_tokens` and
`context.prefix_cache_hit_tokens`); `/health` additionally reports the engine-side cumulative
reuse counters. `--no-cache-affinity-admission` turns both the scheduler ordering and the
cache-aware estimate off, giving a "no cache awareness" admission variant.
The output estimate starts at the full requested limit, then adapts using completed requests;
KV capacity checks still reserve against the full limit. Prefill/Decode rates start from configurable
fallbacks and adapt from completed engine steps. When the engine is idle, it admits one
request to preserve progress. With work in flight, requests below the soft-pressure and TTFT
limits are accepted immediately; others wait in a FIFO deferral queue for up to 2 seconds by
default. The deferred queue is capped at 64 requests, and the overall hard limit remains 256.
If pressure does not fall in time, the server rejects the request with HTTP 429 and
`Retry-After`; its error body includes that request's final prefill and prefix-hit estimates.
An explicit `ttft_slo_ms` is also used as the admission target; the default admission target is
2000 ms. `/health` reports accepted-request prefix-hit estimates, actual hit counters, and the
last estimate. Non-streaming replies include estimate and actual-hit fields in `admission`; SSE
headers include the estimates and the final chunk includes the completed admission metadata.
The scheduler enables these policies by default:

* **Latency-aware scheduling** reserves a decode row and a configurable minimum prefill
  token budget (256 by default) in mixed batches. Within the admission window it can prefer
  requests with lower estimated remaining prefill cost.
* **Top-W cache affinity** evaluates up to 16 waiting requests using the block manager's
  actual chained-prefix hashes. Reusable prefix length is ranked alongside TTFT slack and
  remaining prefill cost; aging can promote an older request ahead of both.
* **Aging fairness** promotes the oldest request after 2 seconds of waiting, overriding the
  cache and short-prompt preference so long prompts cannot starve behind a steady arrival
  stream.
* **Recompute-aware preemption** estimates replay cost from reusable prefix blocks and
  observed prefill speed, then compares it with the estimated KV swap round trip. Before
  measurements are available it uses configurable fallback estimates (10k prefill tokens/s
  and 12 GB/s KV transfer); disable the policy to restore swap-first behavior.
* **TTFT SLO-aware batching** orders candidates by estimated TTFT deadline slack. It scales
  the prefill token and row quotas with queue depth, request urgency, and observed prefill
  speed. A long prompt advances in bounded chunks while decode work keeps using its share of
  each mixed batch. Chat/completion requests may pass `ttft_slo_ms`; the server default is
  500 ms and can be changed at startup.
* **TPOT-target-aware batching** accepts a request-level `tpot_slo_ms` (or the optional
  `default_tpot_slo_ms`). It tracks each active request's observed token interval, prioritizes
  requests with less TPOT slack, and shrinks prefill token/row quotas when decode latency is
  near or over target. Two guards keep that throttle from starving prefill: within
  `tpot_prefill_throttle_margin` (0.5) of the target it stays off and beyond it ramps with the
  overshoot instead of clipping, and once the waiting queue reaches
  `tpot_throttle_max_waiting` (16) the throttle is disabled entirely — at depth the bottleneck
  is prefill throughput, so throttling it only deepens the backlog. Before those guards a 20 ms
  target pinned the quota at `prefill_reserve_tokens`: measured 350 tok/s, TTFT p50 44.5 s and
  TPOT p50 173 ms at 128 requests / 16 req/s; with them the same run measures 1140 tok/s,
  1.6 s and 82 ms. A target the hardware cannot deliver at the offered load is still missed —
  the guards remove the collapse, not the physics. Set the margin to 0 or the depth to 0 to
  reproduce the legacy cliff.
  Benchmark TPOT SLO attainment uses the same per-request average as the
  reported TPOT: `(completed - first_token) / (completion_tokens - 1)`. Requests that emit
  fewer than two completion tokens have no measurable inter-token interval and are excluded
  from the TPOT SLO denominator.
* **Multi-step decode** executes up to `max_decode_steps` consecutive forwards (default 4,
  configurable from 1 to 16) during an otherwise idle pure-decode scheduling window, reducing
  engine/server-loop handoffs between forwards. Each round performs ordinary KV
  append/COW/preemption accounting. Prefill or
  speculative work ends the burst; mixed, prefill, and speculative batches remain one forward.
  `decode_burst_yield` (default on) additionally stops a running burst when prefill work is
  waiting, but in `mixed` mode that queue check cannot trigger — requests are only submitted
  between engine steps and a burst only starts with an empty waiting queue. The path that does
  work there is the arrival signal (`decode_burst_yield_on_arrival`, default on): the service sets
  it whenever an accepted request is still queued in `pending`/`incoming`, and a trace driver can
  install `LLMEngine.set_decode_burst_yield_callback(...)`. With `max_decode_steps=8` a request
  arriving mid-burst measured 53.6 ms p50 wall time versus 68.3 ms with the signal disabled
  (worst round 329 ms vs 676 ms): the burst stops after its active round instead of running the
  budget out. `--no-decode-burst-yield` and `--no-decode-burst-yield-on-arrival` are the ablation
  flags. In online streaming, generated tokens are delivered together at the end of each burst.

The policies can be changed through `LLM(...)` config fields or the equivalent
`nanovllm-serve` / `benchmarks/bench.py` flags: `latency_aware_scheduling`,
`cache_affinity_admission`, `aging_fairness`, `recompute_aware_preemption`,
`slo_aware_scheduling`, `admission_window`, `aging_timeout_ms`,
`prefill_reserve_tokens`, `max_prefill_chunk_tokens`, and
`queue_depth_for_full_prefill`. Set `default_ttft_slo_ms` (or the server flag
`--default-ttft-slo-ms`) to change the request target used when a caller omits
`ttft_slo_ms`. Set `default_tpot_slo_ms` or pass `tpot_slo_ms` on an individual request;
`--no-tpot-aware-scheduling` disables TPOT prioritization, and
`tpot_decode_ms_fallback` controls the estimate before request samples are available.
Set `max_decode_steps` (1–16) to bound each decode burst; `multi_step_decode=False`
or `--no-multi-step-decode` provides the single-step ablation, and `decode_burst_yield=False`
or `--no-decode-burst-yield` keeps a burst running its full round budget past newly arrived
prefills (the queue-check path). The benchmark accepts
`--max-decode-steps`, `--default-tpot-slo-ms`, and both `--no-*` ablations; the continuous
arrival harness includes `without_tpot_aware`, `without_multi_step_decode`, and
`without_decode_burst_yield` cases.
Admission can be tuned with `--max-queued-requests`,
`--max-deferred-requests`, `--max-admission-wait-ms`, `--admission-work-budget-ms`,
`--admission-target-ttft-ms`, and `--admission-soft-pressure`. Use
`--no-dynamic-admission` to disable predictive defer/reject decisions while retaining the
hard queue limit. Use `--no-admission-prefix-cache-awareness` to keep predictive admission
while estimating every prompt as a cache miss. `/health` reports accept/defer/reject counts,
accepted-request estimates, and actual prefix-hit counters. Successful JSON replies include the
admission decision, estimated prefill/cached tokens, and actual prefix-hit tokens. SSE replies
put the estimates in response headers and the same metadata in the final event.

Run a continuous-arrival ablation with one identical request trace for each variant:

```bash
python benchmarks/scheduling_ablation.py --num-seqs 128 --arrival-rate 16 \
  --arrival-mode poisson --shared-prefix-len 512
```

The default variants are `baseline` (all policies off), `all_on`, and one run with
each policy removed from `all_on`. This includes `without_prefix_feature_cache`, which keeps
cache-affinity scheduling enabled but reparses each Top-W feature query; it isolates the cost
and reuse benefit of generation-tagged lazy features, and `without_decode_burst_yield`, which
keeps multi-step decode on but lets a burst run its full round budget past arrived prefills.
Use `--variants baseline all_on` for a quick A/B run,
or `--arrival-mode burst` to compare burst admission with continuously offered load. Set
`--min-request-ttft-slo-ms` and `--max-request-ttft-slo-ms` to give requests reproducible
individual TTFT targets. The report compares throughput, TTFT p50/p99, queue wait, TPOT,
E2E, fixed and per-request SLO attainment, adaptive prefill quotas, actual prefix-cache hits,
aging promotions, recompute/swap counts, and estimated preemption cost. It also records
lazy-feature parses/reuses, stale-generation reparses, KV-generation mutations, LRU
evictions, deferred-free references queued/committed with peak unique-block and reference
counts, and decode-burst rounds/yields with the decode slots given up to prefill.
The JSON includes per-policy deltas from `baseline` and paired deltas from `all_on`, so each
`without_*` run shows the effect of removing that policy while holding the others on.
Raw per-request data and the shared arrival trace are saved as JSON under `results/`.
`benchmarks/bench.py` also accepts `--no-latency-aware-scheduling`,
`--no-cache-affinity-admission`, `--no-aging-fairness`,
`--no-prefix-feature-cache`, `--no-recompute-aware-preemption`, `--no-slo-aware-scheduling`, and
`--no-mixed-cudagraph` for manual single-run comparisons. Mixed graph capture is lazy and
retains at most four batch shapes by default; `--mixed-cudagraph-max-graphs` changes that
bound. Rolling and Gemma-2 split batches copy their ring/full-pool tables and slot mappings
through the graph's static inputs. sparse24 keeps the configuration enabled but uses an
observable eager fallback because the custom Triton path's graph replay is not currently
trusted. Batches
above 4096 tokens stay eager by default to bound capture memory;
`--mixed-cudagraph-max-tokens` changes that threshold. `LLMEngine.collect_metrics()` reports
parse/reuse counts and rates, stale-generation reparses, KV-generation mutations, cache
evictions, deferred-free lifecycle counters, mixed graph captures/replays/fallbacks, and
decode graph batch shapes/replays/fallbacks, plus per-request prefix-hit tokens. The
`context_concurrency.py` JSON and CSV also retain decode/mixed graph replay and eager-fallback
counts for each workload point.

Compare the three admission variants (off / on without cache awareness / on with cache
awareness) against one shared arrival trace with:

```bash
python benchmarks/admission_ablation.py --num-seqs 128 --arrival-rate 16 \
  --arrival-mode poisson --shared-prefix-len 512
```

Each variant gets a fresh engine and the same trace. It reports accepted/deferred/rejected
requests, per-arrival accept/reject rates, throughput, TTFT/TPOT percentiles, and joint SLO
goodput (completed requests and output tokens meeting both targets), plus the per-request
cache-hit estimate vs the tokens actually reused. It stores the exact workload fingerprint and
admission counters in `results/admission_ablation_*.json`. Compare before/after reports with:

```bash
python benchmarks/compare_ablation.py results/before.json results/after.json \
  --variant dynamic_on_cache_aware
```

The comparison checks that prompts, sampling parameters, arrival times, seed, and SLO targets
match before reporting deltas. Pick a load that actually breaks
the target: at 8 req/s with a 2 s target every request already meets it and admission can only
look harmful, whereas at 24 req/s / 128 requests without admission only 26.6% of requests get a
first token within 2 s (p99 4.7 s) and admission trades that for a 15.6% accept rate at a 95%
served-request target rate.

Two standalone checks make the estimate-vs-measured comparison easy to read:

```bash
python benchmarks/prefix_cache_probe.py       # in-process: scheduler estimate vs committed reuse
python benchmarks/prefix_cache_verify.py      # over HTTP against nanovllm-serve
python benchmarks/prefix_estimate_check.py    # three requests, estimate and actual per response
```

Measure how a request that arrives *during* a decode burst is handled (this is the path the
trace-driven ablation cannot reproduce, because it only submits between engine steps):

```bash
python benchmarks/decode_burst_server_probe.py --base-url http://127.0.0.1:8000 \
  --prefix-sentences 5 --first-output 128 --late-output 8 --late-delay-ms 300 --rounds 8
python benchmarks/decode_burst_arrival_probe.py    # in-process trace variant
```

### Current scope and known limits

Admission is predictive and bounded, but its model is a heuristic: prompt work is charged net of a
block-aligned prefix-cache prediction that matched the engine's committed reuse exactly in every
probe run so far (see `benchmarks/prefix_cache_probe.py` and `benchmarks/prefix_cache_verify.py`),
while the accept/defer thresholds themselves (`admission_work_budget_ms`,
`admission_soft_pressure`) are not calibrated per model or arrival rate — a 2026-10-06 sweep on the
RTX 5060 Ti (64-request Poisson traces) lost 12% throughput at 4 req/s, 12–54% at 8 req/s, and 45%
at 16 req/s versus disabling admission, with the cache-aware variant consistently better than the
cache-blind one (up to +51%), so treat the default thresholds as unvalidated on new hardware. A single
global output-length ratio is not conditioned on request content. Throughput fallbacks may
misestimate a model until online samples arrive.
The fixed hard queue cap remains as a safety bound. Conversation
history is held in host memory only and is lost when the server exits; GPU KV is not retained
between turns. Summary compaction is lossy and costs an additional model inference. Multi-step
decode is limited to pure non-speculative decode windows; `decode_burst_yield` (default on) ends a
burst early once new requests are waiting, so the added delay is bounded by the rounds already
executed, and several token IDs may arrive in one SSE text chunk. TPOT
control uses a request-local EWMA with a configurable fallback, so it is a heuristic rather than
a hard real-time guarantee. Run the
same-trace scheduling ablation on the target GPU before drawing conclusions.
The current implementation composes FP8 KV, graph-compatible weight quantization, rolling
cache, KV swap and CUDA Graphs for supported Mistral and Gemma-2 sliding-window models. It does
not make every Cartesian-product combination valid: rolling cache depends on sliding-window
semantics, Gemma-2 split rolling disables speculative decoding, and sparse24 stays quantized but
falls back to eager model execution under CUDA Graphs. These combined paths still need a
same-trace GPU run on the target card; archived combination reports below are earlier
measurements and do not validate the new FP8/soft-cap, split-pool swap, or rolling/split graph
combinations. Use `benchmarks/context_concurrency.py` to exercise a combination, for example:

```bash
python benchmarks/context_concurrency.py --model ~/huggingface/Mistral-7B-v0.1 \
  --kv-cache-dtype fp8_e4m3 --quantization int4 --rolling-cache \
  --context-lengths 4096,8192 --concurrency 1,2,4 --repeats 1
```

KV swap only runs when decode KV pressure triggers preemption. For a swap-path run, raise
concurrency or context length until the JSON reports `swaps > 0`; otherwise that run measures
the enabled policy without exercising a transfer.

KV swap is on by default; `--no-swap-kv` disables it. `--enforce-eager` disables CUDA Graphs,
and `--no-mixed-cudagraph` disables only the mixed-batch graph family. Gemma-2 split rolling
supports FP8 KV and both decode/mixed graphs, but still requires speculative decoding off.
sparse24 falls back to eager even when graph capture is otherwise enabled. These are code paths
available for measurement, not claims that every combination has been verified on the target GPU.

Combination boundaries: KV swap is active only for TP=1 and otherwise uses recompute; PD mode
still requires `kv_cache_dtype="auto"` and does not support rolling cache. Mixed graphs cover MHA
paths up to the configured token cap; MLA and spec-mixed batches stay eager. Mistral rolling with
n-gram verification also uses eager verification, while its ordinary decode and mixed batches can
use graphs.

The scheduler policies and admission controller have ablation harnesses, but the performance
table below predates those changes. It records earlier single-GPU runs and does not establish
that the newer policies improve throughput or latency; use
`benchmarks/scheduling_ablation.py` and `benchmarks/admission_ablation.py` to measure them on
the current checkout.

The CUDA C GEMM in `benchmarks/_cuda_gemm_dev.cu` is a standalone learning/measurement kernel;
the inference engine does not call it. Mixtral and PP/DP/EP are not implemented. TP has code,
but multi-GPU TP behavior has not been benchmarked
on the single-GPU development machine. PD is experimental and requires two visible GPUs.

## Benchmark

### Concurrent and long-context capacity sweep

Run a simultaneous burst for each prompt-length/concurrency pair. The script loads one engine,
uses independent deterministic synthetic token prompts, and writes a full JSON report plus a
CSV summary under `results/`:

```bash
python benchmarks/context_concurrency.py --model ~/huggingface/Qwen3-0.6B \
  --context-lengths 512,1024,2048,4096 --concurrency 1,2,4,8,16 \
  --repeats 3 --max-output-tokens 32
```

The JSON report keeps per-request records and includes acceptance/failure counts; TTFT, average
TPOT, queue wait and E2E avg/p50/p90/p99; TTFT/TPOT SLO targets and attainment; input/output/total
token throughput and request throughput; admission decisions, waits and predicted pressure;
preemptions, KV swaps, recompute and prefix-cache counters; peak KV block occupancy; actual cache
tensor bytes; and CUDA allocated/reserved-memory peaks. The CSV summarizes the same matrix points.
Its KV budget is measured from live MHA or MLA cache tensors and block pools. It estimates
capacity per pool: full-history pools reserve prompt plus output tokens; rolling pools account for
full-prompt allocation during prefill and the bounded ring capacity during generation. The report
excludes host-swapped KV and does not mean that the GPU can run that many requests at the same
latency.
Admission control is off by default to expose scheduler capacity; add `--dynamic-admission` to
include the online accept/defer/reject policy.

### FP8 KV calibration

Create a calibration token-ID file from real text and compare FP8 scales against an auto-dtype KV
baseline in the same run:

```bash
python benchmarks/kv_fp8_calibrate.py --model ~/huggingface/Qwen3-0.6B \
  --calibration-file calibration.jsonl --eval-file heldout.jsonl \
  --margins 0.9,1.0,1.1,1.25 --output results/kv_eval.json
```

Text inputs may be plain text or JSONL records with `prompt` or `text`. With no input files the
script uses a small built-in calibration and evaluation corpus, repeating each prompt to 1024
tokens so the default run exercises a longer KV history. For supplied corpora, original prompt
lengths are preserved unless `--calibration-context-length` or `--eval-context-length` is set.
It writes `results/kv_eval.tokens.json` for `--kv-calibration-path`, a JSON report with per-layer
scales/ranges and per-prompt results, and a CSV summary. Precision metrics are measured from the
first decode logits only when the baseline and FP8 runs emitted the same first token: KL divergence,
top-1 agreement, top-5 overlap, cosine similarity, and max/mean/RMSE logit error. Held-out
activation range utilization above 1 means an observed maximum is beyond that layer's E4M3 scale
limit and will be clamped; it is not an elementwise saturation percentage. Gemma-2's attention
logit soft-cap is implemented in the FP8 paged decode and varlen attention kernels; Gemma-2 FP8
KV calibration and end-to-end combination performance still need target-GPU verification.

Use the generated calibration file in a throughput run with:

```bash
python benchmarks/context_concurrency.py --model ~/huggingface/Qwen3-0.6B \
  --kv-cache-dtype fp8_e4m3 --kv-calibration-path results/kv_eval.tokens.json \
  --context-lengths 512,1024,2048,4096 --concurrency 1,2,4,8
```

The generic FP8 configuration still falls back to its deterministic random-token calibration
when `kv_calibration_path` is empty. Real, representative calibration text and a separate held-out
evaluation corpus give a more useful precision check.

**Archived test configuration (2026-08/09):** RTX 5060 Ti 16GB (sm_120, 36 SM) · WSL2 · torch 2.8.0+cu128 ·
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

**Archived comparison note:** in the recorded vLLM 0.10.2 run, its tested FP8 KV path did not run
on this sm_120 setup; nano-vllm's custom Triton paged kernels did. In that workload the cache
capacity was ×1.9, KL 0.0073, and top-1 was 100% on aligned logits. Single-engine peak in the
archived run: **5,825 tok/s** (FP8 weights, Qwen3-0.6B, bs=256).

Honest boundaries: these are single-card WSL2 numbers for a 0.6B model; decode single-step
latencies are at parity with vLLM fp16, and several features (e.g. pure-int4 large batch,
2:4 sparsity, EAGLE γ=4, KV swap on this machine) are *documented losses* — see
[`INTERVIEW.md`](INTERVIEW.md) §10 (基准档案) for the full tables, conditions, and what to
re-run after changing environment.

## Documentation

* [`CLAUDE.md`](CLAUDE.md) — architecture deep-dive (offline/online request lifecycle, Context contract, KV/paging/offload, scheduling, quantization, speculative decoding, PD) and dev-machine gotchas.
* [`INTERVIEW.md`](INTERVIEW.md) — **master doc** (Chinese): learning route with per-feature code pointers, interview narrative, 18 pitfall stories, deep-dive Q&A, benchmark archive, and a symbol-based code map. (Consolidated from the former BENCHMARKS.md / LEARNING.md / INTERVIEW.md.)
* [`AGENTS.md`](AGENTS.md) — contribution guidelines (style, commit conventions, testing).
* `benchmarks/_stage2b_ext_report.md` — evidence tables for the stage-2 MLA / rolling-ring / fp8-KV-combination work.

## Tests

```bash
python -m pytest tests/ -q     # pure-Python coverage; no GPU required
```

`tests/test_prefix_feature_lifecycle.py` covers lazy feature reuse and invalidation,
uncached-first/LRU allocation, the free-prefix cap, shared deferred-free commit, swap release,
non-sharing cache modes, and metrics reset without changing live state.

## Repository Layout

```
nanovllm/
├─ engine/    LLMEngine · Scheduler · BlockManager · Sequence · ModelRunner · KV transfer
├─ layers/    attention (+ MLA) · linear (quant kernels) · moe · layernorm · rope · embed_head · sampler · medusa · eagle
├─ models/    per-family modules + model_type registry
├─ server.py  OpenAI-style HTTP/SSE service and in-memory conversations
└─ utils/     weight loader · Context singleton (per-step data contract)
tests/        pytest suite
benchmarks/   bench.py · context_concurrency.py · kv_fp8_calibrate.py · scheduling_ablation.py · profiler.py · probes & reports (see INTERVIEW.md §1.8)
```

## License

[MIT](LICENSE)

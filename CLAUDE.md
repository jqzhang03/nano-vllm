# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

nano-vllm is a from-scratch reimplementation of vLLM's offline inference stack (~1,200 lines of Python) built on raw PyTorch + Triton + flash-attn. It supports **Qwen3-style causal LMs only** (`nanovllm/models/qwen3.py`). The public API mirrors vLLM: `LLM(path, ...)` + `SamplingParams`, and `LLM.generate(prompts, sampling_params)` returns a list of `{"text": ..., "token_ids": [...]}` dicts in input order. Root `example.py` and `bench.py` are the runnable demos.

Hard runtime requirements (GPU): CUDA, `flash-attn`, `triton`, and NCCL. There is **no test suite and no linter**.

## Common commands

```bash
pip install -e .          # editable install (the only build step)
python example.py         # end-to-end inference demo (2 prompts)
python bench.py           # throughput benchmark (256 seqs)
python benchmarks/bench.py --num-seqs 256                  # full metrics: TTFT/TPOT/E2E/p50/p99/SLO (+ JSON in results/)
python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512   # prefix-cache workload
python benchmarks/compare_workload.py --tag small && python benchmarks/compare_nanovllm.py --workload results/compare_workload_small.json --output results/compare_nanovllm_small_fp16.json && python benchmarks/compare_merge.py results/compare_*.json   # vLLM comparison (see BENCHMARKS.md §5; runs in the isolated vllm-compare env)
python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64  # prefill/decode torch.profiler breakdown
```

See `BENCHMARKS.md` for metric definitions and interpretation. Timing instrumentation lives on `Sequence` (`t_submitted`/`t_first_token`/`t_completed`, driver-side only) and is exported via `LLMEngine.collect_metrics()`; `benchmarks/bench.py` consumes it. `benchmarks/run_in_wsl.sh` drives any python command from Windows into the WSL env (it sets `PYTHONPATH` so the workspace copy of `nanovllm` wins over the conda env's editable install, which may point at another clone such as `~/AI/nano-vllm`).

`example.py` / `bench.py` hardcode the model path `~/huggingface/Qwen3-0.6B/`; download it with:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
```

## Runtime environment notes (this dev machine)

This repo is developed on Windows + WSL2 (Ubuntu), conda env `nanovllm`, Python 3.12, **RTX 5060 Ti 16GB (Blackwell, sm_120)**. These gotchas are non-obvious and already cost setup time:

- Blackwell needs a CUDA 12.8+ torch build. Install with `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` (cu121/cu124 wheels have no sm_120 kernels).
- **flash-attn on PyPI ships sdists only — every `pip install flash-attn` triggers a source build (fails without nvcc).** Use the prebuilt wheels from the Dao-AILab GitHub releases, matching the installed torch minor version, `cp312`, and `cxx11abiFALSE` (pytorch.org wheels are cxx11abiFALSE), e.g. `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`.
- WSL's `/tmp` is a small tmpfs (half of WSL RAM). `export TMPDIR=$HOME/pip_tmp` (a dir on the root ext4) before any `pip install`, or the ~3GB wheel unpack fills `/tmp`, raises `OSError: [Errno 28]`, and gets OOM-killed.
- First run of any script shows harmless `_POSIX_C_SOURCE redefined` gcc warnings — these are triton JIT-compiling its launcher, not errors.
- The tqdm `Prefill/Decode tok/s` in `example.py` are low because it forces `enforce_eager=True`; `bench.py` (CUDA-graph decode, 256 seqs) is the real throughput number.

## Architecture

### Request lifecycle

`LLM.generate` → `Scheduler.schedule()` returns a prefill or decode batch → `ModelRunner.run(seqs, is_prefill)` packs tensors → `Qwen3ForCausalLM` forward → `Sampler` returns one token per sequence → `Scheduler.postprocess` appends tokens / finalizes. Loop until `scheduler.is_finished()`.

### The `Context` singleton — the per-step data-flow contract

`nanovllm/utils/context.py` holds a module-global `Context` dataclass. `ModelRunner.prepare_prefill`/`prepare_decode` build the per-step GPU tensors (`cu_seqlens_q/k`, `max_seqlen_q/k`, `slot_mapping`, `context_lens`, `block_tables`) and call `set_context(...)`; the `Attention` kernel and `ParallelLMHead` read them via `get_context()`; `run_model` reads them for the CUDA-graph path; `reset_context()` is called after every `run()`. **When adding a kernel that needs per-step tensors, extend `Context` + `set_context` — do not introduce new module-level state.** Keep this interface stable; it's the contract that lets attention kernels stay stateless.

### KV cache & paged attention

- `ModelRunner.allocate_kv_cache()` allocates one big `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]` tensor and binds each layer's `k_cache`/`v_cache` by matching modules that have those attributes. `num_kvcache_blocks` is derived from a free-VRAM heuristic (`total*util - used - warmup_peak + current // block_bytes`) and asserts `> 0`.
- Block size = `kvcache_block_size` (default 256, must be a multiple of 256). `Config` asserts this.
- `Attention.forward` writes K/V into the cache with a custom triton kernel (`store_kvcache`, using `context.slot_mapping`), then calls flash-attn: `flash_attn_varlen_func` for prefill (paged via `block_table`), `flash_attn_with_kvcache` for decode (paged via `block_table` + `cache_seqlens`).
- **KV cache quantization**: `kv_cache_dtype="fp8_e4m3"` stores K/V as FP8(E4M3) (1 byte/elem, ~2× capacity) with per-layer static scales calibrated on random-token prefill; decode/verify read the cache with the custom fp8 Triton attention kernels (`paged_decode_attention_fp8` / `paged_varlen_attention_fp8`) that load fp8 directly and dequantize in-register via hardware cvt.

### Weight quantization & sparsity (`--quantization none|w8a8|int4|awq|sparse24`)

- All paths live in `nanovllm/layers/linear.py`: `LinearBase.quantize_w8a8` (per-group 128 int8 + per-token int8 + SmoothQuant folding, Triton int8 GEMM), and the shared `WeightQuantMixin` (used by `LinearBase` **and** `ParallelLMHead`) providing `quantize_int4` (per-group 128 symmetric int4, packed `[N, K//2]` int8 along K, Triton 2-dot dequant GEMM with M-adaptive tiles) and `quantize_sparse24` (2:4 magnitude pruning, packed `v [N,K//2] bf16` + `idx [N,K//4] uint8`, Triton 4-way-split sparse GEMM). `ModelRunner` applies them after weight loading (`quantize_int4_weights` / `quantize_awq_weights` / `prune_sparse24`), before warmup/graph capture. `quantize_lm_head` defaults False; Qwen3-0.6B ties embeddings so the guard skips lm_head anyway.
- **AWQ** (`benchmarks/awq_calibrate.py`): real-text calibration → per-layer α search (`s=(mean|X|/w_col)^α`, α∈{0..1}, objective = quantized output error on the calibration batch) → scales file `results/awq_scales.pt`; inference folds `W'=W·s, X'=X/s` (**weight multiply, activation divide** — the paper direction; the reverse collapses group scales). Engine falls back to inline random-token calibration when `awq_scales_path` is empty.
- **int4 dual-path mode** (default, `int4_dense_path=False` to disable): `quantize_int4` also saves a bf16 dequantized copy `w_deq`; `_int4_forward` routes by shape — `M≤128 and N≥2048` → int4 kernel, otherwise `F.linear(x, w_deq)` (cuBLAS). Both paths share the same q/scale (mathematically identical, verified ~bf16 noise). Recovers the large-M loss (bs=256 0.64×→1.06×, TTFT) at the cost of weights memory 0.85→1.73 GB.
- Honest findings (see BENCHMARKS.md §10): int4 ppl 3.32→4.38 (RTN) / 3.76 (AWQ); throughput beats fp16 in dual-path mode at both batch sizes; sparse24 kernel is bit-exact but one-shot 2:4 magnitude pruning destroys accuracy on 0.6B (KL 8.5) and torch's cuSPARSELt/CUTLASS paths are unusable on sm_120 (per-call overhead / sm_8x-only).


### Prefix caching & chunked prefill

- `BlockManager` owns a fixed block pool and a `hash_to_block_id` map. Block hashes are a chain: `xxhash.xxh64` over the block's token bytes plus the previous block's hash as 8-byte little-endian (`compute_hash`). `can_allocate` walks a sequence's blocks and returns how many already exist in the cache; `allocate` shares them via refcounts; `hash_blocks` publishes new hashes after a step.
- **Partial blocks are cached and shared too, with copy-on-write safety**: `hash_blocks` publishes the last partial block's hash (ceiling `end`); `can_allocate` checks it; before a writer (decode append or a prefill tail crossing into a shared block) touches a shared block, `BlockManager.cow_block(seq, write_start)` swaps in a fresh block and records `(old, new)` in `Scheduler.cow_pairs`, which `LLMEngine.step()` executes on GPU via `ModelRunner.cow_block` (`kv_cache[:, :, new] = kv_cache[:, :, old]`) **before** `run()`. Stale hash entries are deleted with a guard (`hash_to_block_id.get(old_hash) == block_id`) — unguarded deletion hits KeyError when two identical-content blocks (e.g. a COW copy) share a hash. `allocate`/`can_append`/`can_allocate` reserve free blocks for COW; `num_cached_tokens` counts the actual cached tokens (`(n-1)*block_size + len(last_block.token_ids)`), and the scheduler derives `num_tokens` from it (never `num_cached_blocks * block_size`, which breaks on partial blocks).
- `Sequence.num_cached_tokens` tracks how many tokens are already resident in the cache. During prefill, if the key span exceeds the query span (prefix reused), `prepare_prefill` builds `block_tables` and `Attention` uses the cache tensors directly as K/V.
- Prefill is chunked when a sequence exceeds the remaining `max_num_batched_tokens`; **only the first scheduled sequence may be split** (see the guard in `Scheduler.schedule`).

### Scheduler

Three queues WAITING/RUNNING/FINISHED. `schedule()` returns `(seqs, kind)` with kind ∈ `prefill | decode | mixed | spec`: when both queues are non-empty it builds a **mixed batch** (prefill rows first, decode rows last, sharing the `max_num_batched_tokens` budget — vLLM V1-style), so early prefills start decoding immediately instead of waiting for all prefills; pure phases remain when one queue is empty. Prefill fills the batch up to `max_num_batched_tokens`/`max_num_seqs`; decode runs one token per sequence; if KV blocks run out, the scheduler **preempts** (pops from the running deque, deallocates its blocks, pushes it to the front of waiting). `postprocess` branches per-seq on `seq.is_prefill` (mixed batches contain both), appends tokens, checks EOS/`max_tokens`, and re-hashes blocks. The engine's `step()` returns `(outputs, kind, n_prefill, n_decode)`; `ModelRunner.run(seqs, kind)` dispatches to `prepare_prefill`/`prepare_decode`/`prepare_mixed`/`prepare_spec` (mixed sets `Context.is_mixed` with prefill-group `cu_seqlens` + decode-group `context_lens`/`block_tables` + `n_prefill_tokens` split); `Attention.forward` routes mixed batches (prefill rows via `flash_attn_varlen_func` — cache-shaped K/V when chunked seqs need their own earlier chunks — decode rows via flash-attn kvcache or the fp8 kernel) and `ParallelLMHead` gathers the last row per prefill seq. Only pure decode steps use the CUDA-graph path.

### Speculative decoding (n-gram / prompt-lookup / Medusa, `--speculative ngram|medusa`)

`nanovllm/engine/ngram.py` holds the pure functions (`find_ngram_draft` — last-occurrence window search with 4→1 fallback, EOS truncation; `verify_drafts` — point-mass-draft acceptance, output = target sample always, so the output distribution equals the plain sampler's exactly). In spec mode `schedule()` computes drafts for all running seqs first (medusa mode: keeps engine-provided drafts, n-gram fallback only when unset); if any draft is non-empty the running rows become **verify rows** (kind `spec`, or `mixed` with waiting prefills — the mixed forward then runs the whole batch through varlen with cache-shaped K/V, flagged `Context.is_spec`/`n_prefill_rows`; `ParallelLMHead` keeps ALL rows of verify seqs). A verify row is a chunked prefill with prefix reuse: query = `[last_token, drafts...]` (γ+1 tokens at positions len-1..), num_cached = len-1, cache-shaped K/V + block tables. Logits at position len-1+i predict len+i → sample s_i verifies draft d_i; last row is the bonus. `LLMEngine._verify` runs acceptance and `postprocess_spec` commits only the accepted tokens: `num_cached = num_tokens`, hash range `[num_tokens-n_acc-1, num_tokens-1)` so rejected drafts never enter the prefix-cache hash; stale KV slots are overwritten next step (no rollback). Write spans may cross block boundaries — `BlockManager.can_append_spec/may_append_spec` handle new-block + COW accounting for spans. All-draft-empty steps fall back to plain decode (CUDA graph intact). **Pure-spec steps are CUDA-graphed too** (`capture_spec_graph`): capacity family rows [8..256] × two strides (γ_max+1 and 3 — lower-γ steps use the tighter stride-3 family to cut padding waste), captured with all rows full-length (flash varlen grid baked by capacity), replayed with real rows + zero-length padding rows (trailing duplicate cu_seqlens; bit-exact, verified by `benchmarks/_graph_pad_probe.py`); `max_seqlen_q`/`max_seqlen_k` are baked capture-time scalars (key loop is driven by cu_seqlens, so the baked upper bound costs nothing); the LM head runs outside the graph on the real-row slice. fp8 KV verify stays eager (its per-layer full-cache dequant is GPU-side work graphs can't remove — see BENCHMARKS.md §9). **Medusa mode** (`nanovllm/layers/medusa.py`): γ+1 small MLP heads (hidden→256→vocab) on the last-layer hidden; head_k(h_t) predicts token_{t+k+1} (self-distilled training via `benchmarks/medusa_train.py`, ~7 min — must `llm.exit()` before training, the engine's 14GB makes allocator ~60× slower); after each verify step `LLMEngine._medusa_drafts` picks the new t_last's hidden row (row n_acc, or row γ_i with head shift 1 on full acceptance — bonus has no hidden) and batch-runs the heads (capture n_rows_list BEFORE postprocess — it zeroes num_scheduled_tokens). Benchmarks and honest performance findings in BENCHMARKS.md §9/§9b (ngram wins on repetitive content at small batch; Medusa's α is capped by the 0.6B model's ~35% top-1 predictability).

### CUDA-graph decode path

`capture_cudagraph` captures decode forward passes for batch sizes `[1,2,4,8] + range(16, ≤512, 16)` sharing one memory pool. The non-prefill `run_model` path selects the smallest captured size ≥ the live batch, copies inputs into static `graph_vars` tensors, replays, and slices `outputs[:bs]`. This path is used only when `not enforce_eager` **and** `bs ≤ 512`; prefill is never graphed. Note `enforce_eager` controls **only** CUDA-graph capture — see torch.compile below.

### Tensor parallelism

NCCL is initialized unconditionally (`tcp://localhost:2333`, even at TP=1). For TP>1: rank 0 is the driver; other ranks enter `ModelRunner.loop()`, reading pickled `(method_name, args)` from a 1 MB `SharedMemory("nanovllm")` signaled by `Event`s; `ModelRunner.call()` writes to shared memory then dispatches locally. Weights are sharded at load time via each parameter's bound `weight_loader` (`ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`). `RowParallelLinear` and `VocabParallelEmbedding` `all_reduce`; `ParallelLMHead` gathers the vocab partition. Sequences cross the process boundary via custom `__getstate__`/`__setstate__` (full token_ids in prefill, only `last_token` in decode).

### torch.compile — fires even in "eager" mode

`@torch.compile` is applied to `RMSNorm` (both variants), `SiluAndMul`, `RotaryEmbedding.forward`, and `Sampler.forward`. These compile **regardless of `enforce_eager`** (that flag only disables CUDA-graph capture), so the first forward pass always does triton JIT compilation. `Sampler` samples via the exponential-noise (Gumbel) trick, and `SamplingParams` forbids greedy (`temperature > 1e-10`).

### Weight loading

`nanovllm/utils/loader.py` globs `*.safetensors`, remaps HF names through `Qwen3ForCausalLM.packed_modules_mapping` (e.g. `q_proj → (qkv_proj, "q")`, `gate_proj → (gate_up_proj, 0)`), then dispatches to the parameter's `weight_loader(param, tensor, shard_id)`. Parallel layers attach their loader in `__init__`. Adding a new model means replicating this packed-mapping + weight_loader convention.

## Contribution guidance

See `AGENTS.md` for coding style, commit/PR conventions, and architecture notes. Keep new code consistent with the existing heavy-comment style.

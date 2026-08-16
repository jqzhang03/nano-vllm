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
python benchmarks/bench.py --num-seqs 256 --compare-vllm   # side-by-side vs real vLLM (needs pip install vllm)
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

### Prefix caching & chunked prefill

- `BlockManager` owns a fixed block pool and a `hash_to_block_id` map. Block hashes are a chain: `xxhash.xxh64` over the block's token bytes plus the previous block's hash as 8-byte little-endian (`compute_hash`). `can_allocate` walks a sequence's blocks and returns how many already exist in the cache; `allocate` shares them via refcounts; `hash_blocks` publishes new hashes after a step.
- `Sequence.num_cached_tokens` tracks how many tokens are already resident in the cache. During prefill, if the key span exceeds the query span (prefix reused), `prepare_prefill` builds `block_tables` and `Attention` uses the cache tensors directly as K/V.
- Prefill is chunked when a sequence exceeds the remaining `max_num_batched_tokens`; **only the first scheduled sequence may be split** (see the guard in `Scheduler.schedule`).

### Scheduler

Three queues WAITING/RUNNING/FINISHED. Prefill fills the batch up to `max_num_batched_tokens`/`max_num_seqs`. Decode runs one token per sequence per step; if KV blocks run out, the scheduler **preempts** (pops from the running deque, deallocates its blocks, pushes it to the front of waiting). `postprocess` appends tokens, checks EOS/`max_tokens`, and re-hashes blocks.

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

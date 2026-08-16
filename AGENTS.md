# Repository Guidelines

## Project Structure & Module Organization

All source code lives under `nanovllm/`:

- `engine/` – orchestration: `LLMEngine`, `Scheduler`, `BlockManager`, `Sequence`, `ModelRunner`
- `layers/` – model operations: attention, linear, layernorm, rotary embedding, sampler
- `models/` – model definitions (`Qwen3ForCausalLM`)
- `utils/` – weight loading and the global inference context

Root-level `example.py` and `bench.py` are runnable demos; `assets/` holds images; `pyproject.toml` defines packaging and dependencies. `benchmarks/` holds the performance tooling (see `BENCHMARKS.md`): `bench.py` (throughput/latency/SLO/vLLM comparison), `profiler.py` (torch.profiler prefill/decode breakdown), plus dev scripts to drive runs from Windows into WSL. There is no separate tests directory yet.

## Build, Test, and Development Commands

```bash
pip install -e .        # install the package in editable mode
python example.py       # end-to-end inference; expects a local Qwen3-0.6B checkpoint
python bench.py         # throughput benchmark (256 sequences)
python benchmarks/bench.py --num-seqs 256            # full metrics: TTFT/TPOT/E2E/p50/p99/SLO
python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512   # prefix-cache workload
python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64  # prefill/decode breakdown
```

`benchmarks/bench.py` consumes the timing instrumentation on `Sequence` (`t_submitted`/`t_first_token`/`t_completed`, driver-side only) exported through `LLMEngine.collect_metrics()`; keep those fields when touching the engine. Never name a script `profile.py` under `benchmarks/` — it shadows the stdlib `profile` module that torch's `cProfile` import chain needs.

Requires Python 3.10–3.12 and an NVIDIA GPU; `flash-attn`, `triton`, and NCCL are hard dependencies. There is no build step beyond `pip install`.

## Coding Style & Naming Conventions

- Python, 4-space indentation, type hints on all public signatures.
- Prefer dataclasses with `slots=True` for config-like objects (`Config`, `SamplingParams`, `Context`).
- Name modules after their layer (`engine/`, `layers/`) and classes after the concept (`Scheduler`, `BlockManager`, `Sequence`).
- No linter or formatter is configured; keep new code consistent with surrounding style.

## Testing Guidelines

No automated test suite exists today. When adding tests:

- Place them under `tests/` with `test_*.py` naming.
- Run with `pytest`.
- Scheduler and block-manager logic is pure Python and testable without a GPU; GPU paths (`ModelRunner`, attention kernels) require CUDA hardware.

## Commit & Pull Request Guidelines

Git history uses short imperative summaries, often `fix(scope): message` (e.g., `fix(model_runner): correct seqlen_k to chunk boundary`). Changes land via pull requests on branches named after the change (e.g., `fix/decoding-positions`). PR descriptions should state the problem, the change, and any benchmark impact; link related issues when present.

## Architecture Notes for Contributors

Inference runs as: `LLM.generate` → scheduler picks prefill/decode batches → `ModelRunner` packs tensors → `Qwen3ForCausalLM` forward → sampler returns tokens. The `Context` singleton in `nanovllm/utils/context.py` passes per-step tensors (slot mappings, block tables) from the runner into attention kernels; keep this interface stable when modifying layers.

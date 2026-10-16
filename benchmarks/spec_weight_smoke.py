"""Check whether local Medusa/EAGLE weights load and generate with a target model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone

import torch

from nanovllm import LLM, SamplingParams


def _file_info(path: str) -> dict[str, object]:
    weight_path = Path(path).expanduser()
    return {
        "path": str(weight_path),
        "exists": weight_path.is_file(),
        "size_bytes": weight_path.stat().st_size if weight_path.is_file() else None,
    }


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, default=str)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--medusa-path", default="results/medusa_heads.pt")
    parser.add_argument("--eagle-path", default="results/eagle_layer.pt")
    parser.add_argument("--max-draft-len", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.model = os.path.expanduser(args.model)
    args.medusa_path = os.path.expanduser(args.medusa_path)
    args.eagle_path = os.path.expanduser(args.eagle_path)
    if args.max_draft_len < 1 or args.max_tokens < 1:
        parser.error("draft length and output length must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this compatibility smoke")

    output_path = Path(args.output).expanduser()
    payload: dict[str, object] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "config": {
            "max_draft_len": args.max_draft_len,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "medusa_weights": _file_info(args.medusa_path),
            "eagle_weights": _file_info(args.eagle_path),
        },
        "status": "running",
        "runs": [],
    }
    _write_checkpoint(output_path, payload)
    for mode, weight_path in (("medusa", args.medusa_path),
                              ("eagle", args.eagle_path)):
        result: dict[str, object] = {
            "mode": mode,
            "weight_path": weight_path,
            "status": "running",
        }
        started = time.perf_counter()
        llm = None
        try:
            llm = LLM(
                args.model,
                speculative=mode,
                medusa_path=weight_path if mode == "medusa" else "",
                eagle_path=weight_path if mode == "eagle" else "",
                max_draft_len=args.max_draft_len,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            prompt_tokens = llm.tokenizer.encode(
                "Answer briefly: what is 2 plus 2?")
            llm.model_runner.sampler.forward = lambda logits, _temps: logits.argmax(dim=-1)
            torch.manual_seed(20261016)
            torch.cuda.manual_seed_all(20261016)
            outputs = llm.generate(
                [prompt_tokens],
                SamplingParams(temperature=1.0, ignore_eos=True,
                               max_tokens=args.max_tokens),
                use_tqdm=False,
            )
            tokens = outputs[0]["token_ids"]
            result.update({
                "status": "passed",
                "output_tokens": len(tokens),
                "output_token_fingerprint": hashlib.sha256(json.dumps(
                    tokens, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "engine_metrics": llm.collect_metrics(),
            })
        except Exception as exc:
            result.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })
        finally:
            result["elapsed_seconds"] = time.perf_counter() - started
            if llm is not None:
                try:
                    llm.exit()
                except Exception as exc:
                    result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                del llm
            gc.collect()
            torch.cuda.empty_cache()
        payload["runs"].append(result)
        _write_checkpoint(output_path, payload)
        print(f"{mode}: {result['status']} ({result['elapsed_seconds']:.1f}s)",
              flush=True)

    runs = payload["runs"]
    payload["status"] = ("complete" if all(row["status"] == "passed" for row in runs)
                         else "completed_with_failures")
    _write_checkpoint(output_path, payload)
    print(f"results -> {output_path}")


if __name__ == "__main__":
    main()

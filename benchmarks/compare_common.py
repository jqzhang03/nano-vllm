"""Shared helpers for the nano-vllm vs real-vLLM comparison.

Pure stdlib so it runs in ANY env (nano-vllm env, vllm env, or Windows).
The latency formulas below deliberately mirror `bench.make_summary`:
  TTFT = t_first_token - t_submitted
  TPOT = (t_completed - t_first_token) / (completion_tokens - 1)
so both engines are measured with identical definitions.
"""
from __future__ import annotations

import statistics
from typing import Any, Iterable


def percentile(sorted_vals: list[float], q: float) -> float:
    return sorted_vals[min(int(len(sorted_vals) * q), len(sorted_vals) - 1)]


def summarize(values: Iterable[float]) -> dict[str, float | None]:
    """avg/p50/p99/min/max/count over non-None values (same as bench.summarize)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"avg": None, "p50": None, "p99": None, "min": None, "max": None, "count": 0}
    return {
        "avg": statistics.fmean(vals),
        "p50": percentile(vals, 0.50),
        "p99": percentile(vals, 0.99),
        "min": vals[0],
        "max": vals[-1],
        "count": len(vals),
    }


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: per-request dicts with t_submitted/t_first_token/t_completed/completion_tokens.

    Returns {ttft, tpot, e2e} summaries in seconds. Missing fields are skipped
    (count reflects how many requests contributed).
    """
    ttfts = [
        r["t_first_token"] - r["t_submitted"]
        for r in rows
        if r.get("t_submitted") is not None and r.get("t_first_token") is not None
    ]
    tpots = [
        (r["t_completed"] - r["t_first_token"]) / (r["completion_tokens"] - 1)
        for r in rows
        if (r.get("t_completed") is not None and r.get("t_first_token") is not None
            and r.get("completion_tokens", 0) > 1)
    ]
    e2es = [
        r["t_completed"] - r["t_submitted"]
        for r in rows
        if r.get("t_completed") is not None and r.get("t_submitted") is not None
    ]
    return {"ttft": summarize(ttfts), "tpot": summarize(tpots), "e2e": summarize(e2es)}


def slo_attainment(rows: list[dict[str, Any]], slo_ttft_ms: float = 500.0,
                   slo_tpot_ms: float = 10.0) -> dict[str, float]:
    """Fraction of requests meeting TTFT/TPOT SLOs (same definitions as bench.py)."""
    n = len(rows)
    ttft_ok = sum(
        1 for r in rows
        if r.get("t_first_token") is not None and r.get("t_submitted") is not None
        and (r["t_first_token"] - r["t_submitted"]) * 1000 < slo_ttft_ms
    )
    tpot_ok = sum(
        1 for r in rows
        if (r.get("t_completed") is not None and r.get("t_first_token") is not None
            and r.get("completion_tokens", 0) > 1
            and (r["t_completed"] - r["t_first_token"]) / (r["completion_tokens"] - 1) * 1000 < slo_tpot_ms)
    )
    return {"ttft_ok": 100.0 * ttft_ok / max(1, n), "tpot_ok": 100.0 * tpot_ok / max(1, n)}

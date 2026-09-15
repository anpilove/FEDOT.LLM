"""Phase 0: stock vs stock. How noisy is DEV AUC without a patch."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path

from fedotllm.agents.evolve.evaluation.eval import run_stock
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import ScoreResult


def calibrate_stock(
    *,
    checkout: Path,
    task_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    workspace: Path | None = None,
) -> dict:
    rows: list[dict] = []
    journal = None if workspace is None else workspace / "noise.jsonl"
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        os.environ["EVOLVE_AGENT_SEED"] = str(seed)
        for task_id in task_ids:
            result = run_stock(task_id, checkout=checkout)
            row = _row(result, seed)
            rows.append(row)
            if journal is not None:
                append_journal(journal, {"event": "stock", **row})
    return {"rows": rows, "by_task": _summarize(rows), "seeds": list(seeds), "task_ids": list(task_ids)}


def _row(result: ScoreResult, seed: int) -> dict:
    return {
        "task_id": result.task_id,
        "seed": seed,
        "status": result.status,
        "score": result.score,
        "duration_s": result.duration_s,
        "detail": (result.detail or "")[:240],
    }


def _summarize(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    out: dict[str, dict] = {}
    for task_id, items in grouped.items():
        ok = [float(item["score"]) for item in items if item["status"] == "ok" and _finite(item["score"])]
        statuses = {item["status"] for item in items}
        stats: dict = {
            "n": len(items),
            "n_ok": len(ok),
            "statuses": sorted(statuses),
        }
        if ok:
            mean = sum(ok) / len(ok)
            denom = max(len(ok) - 1, 1)
            var = sum((x - mean) ** 2 for x in ok) / denom
            stats.update(
                {
                    "mean": mean,
                    "std": math.sqrt(var),
                    "min": min(ok),
                    "max": max(ok),
                    "range": max(ok) - min(ok),
                }
            )
        out[task_id] = stats
    return out


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))

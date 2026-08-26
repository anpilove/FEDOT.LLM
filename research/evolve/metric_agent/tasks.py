from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from research.evolve.metric_agent.guard import repo_root
from research.evolve.metric_agent.types import TaskSpec

CASES_PATH = repo_root() / "data" / "cases.json"
DEFAULT_REGRESS = ("fast_ica->lgbm",)


@lru_cache(maxsize=1)
def _blob() -> dict:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def load_task(task_id: str) -> TaskSpec:
    blob = _blob()
    for raw in blob["cases"]:
        if raw["id"] == task_id:
            return _spec(raw, blob.get("metric", "holdout_roc_auc"))
    raise KeyError(task_id)


def all_tasks() -> list[TaskSpec]:
    blob = _blob()
    metric = blob.get("metric", "holdout_roc_auc")
    return [_spec(raw, metric) for raw in blob["cases"]]


def list_task_metadata() -> list[dict]:
    """Harness-only. Not an LLM tool — the case catalog is our hidden exam."""

    out = []
    for spec in all_tasks():
        out.append(
            {
                "task_id": spec.task_id,
                "nodes": list(spec.nodes),
                "metric": spec.metric,
                "sentinel": spec.sentinel,
                "min_delta": spec.min_delta,
            }
        )
    return out


def _spec(raw: dict, metric: str) -> TaskSpec:
    kind = raw["kind"]
    nodes = tuple(raw.get("nodes") or ())
    if kind == "seq":
        nodes = tuple(raw["nodes"])
    regress = DEFAULT_REGRESS if raw.get("role") == "fail" else ()
    return TaskSpec(
        task_id=raw["id"],
        kind=kind,
        nodes=nodes,
        left=raw.get("left"),
        right=raw.get("right"),
        join=raw.get("join"),
        tail=tuple(raw.get("tail") or ()),
        role=raw.get("role") or "",
        metric=metric,
        must_not_regress=regress,
    )


def hidden_exam() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Harness-only ids. Never send these strings to the LLM."""

    lift = tuple(
        part.strip()
        for part in os.environ.get("METRIC_AGENT_LIFT", "pca->catboost").split(",")
        if part.strip()
    )
    protect = tuple(
        part.strip()
        for part in os.environ.get("METRIC_AGENT_PROTECT", "catboost,fast_ica->lgbm").split(",")
        if part.strip()
    )
    return lift, protect

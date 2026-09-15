"""Pre-registered full OpenML tasks for the hour-long Fedot quality path.

Hunt never imports this module. The frozen PipelineBuilder suite in tasks.py
stays for coverage and causal probes only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("quality_registry.json")


@dataclass(frozen=True)
class QualityDataset:
    task_id: str
    source: str
    openml_task: int
    openml_data: int
    name: str
    problem: str
    metric: str
    higher_is_better: bool
    min_delta: float
    fold: int
    repeat: int
    why_full: str


@dataclass(frozen=True)
class QualityRegistry:
    identity: str
    api: str
    preset: str
    with_tuning: bool
    timeout_seconds: int
    seed: int
    require_search_ran: bool
    portfolio: bool
    cpu_quota: int
    n_jobs_per_job: int
    starter_task_ids: tuple[str, ...]
    datasets: tuple[QualityDataset, ...]


def _load_raw() -> dict:
    return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def load_quality_registry() -> QualityRegistry:
    raw = _load_raw()
    datasets = tuple(
        QualityDataset(
            task_id=str(row["task_id"]),
            source=str(row["source"]),
            openml_task=int(row["openml_task"]),
            openml_data=int(row["openml_data"]),
            name=str(row["name"]),
            problem=str(row["problem"]),
            metric=str(row["metric"]),
            higher_is_better=bool(row["higher_is_better"]),
            min_delta=float(row["min_delta"]),
            fold=int(row["fold"]),
            repeat=int(row["repeat"]),
            why_full=str(row["why_full"]),
        )
        for row in raw["datasets"]
    )
    if not datasets:
        raise ValueError("quality registry has no datasets")
    ids = [item.task_id for item in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError("quality registry task_id values must be unique")
    registry = QualityRegistry(
        identity=str(raw["identity"]),
        api=str(raw["api"]),
        preset=str(raw["preset"]),
        with_tuning=bool(raw["with_tuning"]),
        timeout_seconds=int(raw["timeout_seconds"]),
        seed=int(raw["seed"]),
        require_search_ran=bool(raw["require_search_ran"]),
        portfolio=bool(raw["portfolio"]),
        cpu_quota=int(raw["cpu_quota"]),
        n_jobs_per_job=int(raw["n_jobs_per_job"]),
        starter_task_ids=tuple(str(item) for item in raw.get("starter_task_ids") or ids[:3]),
        datasets=datasets,
    )
    missing = [task_id for task_id in registry.starter_task_ids if task_id not in set(ids)]
    if missing:
        raise ValueError("starter_task_ids not in registry: " + ",".join(missing))
    return registry


def get_dataset(task_id: str, registry: QualityRegistry | None = None) -> QualityDataset:
    registry = registry or load_quality_registry()
    for item in registry.datasets:
        if item.task_id == task_id:
            return item
    raise KeyError(task_id)


def list_quality_task_ids(registry: QualityRegistry | None = None) -> tuple[str, ...]:
    registry = registry or load_quality_registry()
    return tuple(item.task_id for item in registry.datasets)


def list_starter_task_ids(registry: QualityRegistry | None = None) -> tuple[str, ...]:
    """Pre-registered full OpenML tasks. Default is the whole frozen registry."""

    registry = registry or load_quality_registry()
    return registry.starter_task_ids or list_quality_task_ids(registry)


def job_spec(
    dataset: QualityDataset,
    *,
    n_jobs: int | None = None,
    registry: QualityRegistry | None = None,
) -> dict:
    """Fixed pair identity. Dataset choice is independent of later scores."""

    registry = registry or load_quality_registry()
    jobs = int(n_jobs if n_jobs is not None else registry.n_jobs_per_job)
    return {
        "kind": "quality",
        "api": "fedot",
        "preset": registry.preset,
        "with_tuning": registry.with_tuning,
        "timeout_seconds": registry.timeout_seconds,
        "timeout_minutes": registry.timeout_seconds / 60.0,
        "seed": registry.seed,
        "n_jobs": jobs,
        "cpu_limit": jobs,
        "require_search_ran": registry.require_search_ran,
        "portfolio": False,
        "source_dataset": dataset.task_id,
        "openml_task": dataset.openml_task,
        "openml_data": dataset.openml_data,
        "split": f"openml_repeat{dataset.repeat}_fold{dataset.fold}",
        "fold": dataset.fold,
        "repeat": dataset.repeat,
        "problem": dataset.problem,
        "metric": dataset.metric,
        "higher_is_better": dataset.higher_is_better,
        "min_delta": dataset.min_delta,
        "runs": ["stock", "patch"],
    }

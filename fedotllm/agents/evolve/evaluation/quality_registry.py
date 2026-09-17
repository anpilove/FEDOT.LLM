"""Pre-registered full OpenML + public TS tasks for the hour-long Fedot path.

Hunt never imports this module. The frozen PipelineBuilder suite in tasks.py
stays for coverage and causal probes only.

Tabular list is frozen before scores: OpenML-CC18 study 99 official order,
first 20 classification tasks, plus already-registered AMLB/CC18 tasks
outside that prefix. Official repeat=0 fold=0. TS rows are full public
FEDOT example series (beer, australia, salaries), not toy CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("quality_registry.json")
_MIN_DATASETS = 16
_ALLOWED_SOURCES = frozenset({"openml_task", "fedot_public_ts"})
_ALLOWED_PROBLEMS = frozenset({"classification", "ts_forecasting"})
_ALLOWED_TS_DATASETS = frozenset({"beer", "australia", "salaries"})
_TS_PATH_MARKERS = (
    "ts_implementations",
    "ts_transformations",
    "/ts/",
    "time_series",
)
_TABULAR_PATH_MARKERS = (
    "boostings_implementations",
    "sklearn_eval",
    "models/sklearn",
)


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
    forecast_horizon: int = 0
    ts_dataset: str = ""


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
    datasets: tuple[QualityDataset, ...]
    selection_study: int
    selection_rule: str


def _load_raw() -> dict:
    return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def load_quality_registry() -> QualityRegistry:
    raw = _load_raw()
    datasets = tuple(
        QualityDataset(
            task_id=str(row["task_id"]),
            source=str(row["source"]),
            openml_task=int(row.get("openml_task") or 0),
            openml_data=int(row.get("openml_data") or 0),
            name=str(row["name"]),
            problem=str(row["problem"]),
            metric=str(row["metric"]),
            higher_is_better=bool(row["higher_is_better"]),
            min_delta=float(row["min_delta"]),
            fold=int(row["fold"]),
            repeat=int(row["repeat"]),
            why_full=str(row["why_full"]),
            forecast_horizon=int(row.get("forecast_horizon") or 0),
            ts_dataset=str(row.get("ts_dataset") or ""),
        )
        for row in raw["datasets"]
    )
    if len(datasets) < _MIN_DATASETS:
        raise ValueError(f"quality registry must list at least {_MIN_DATASETS} full tasks")
    ids = [item.task_id for item in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError("quality registry task_id values must be unique")
    openml_tasks = [item.openml_task for item in datasets if item.source == "openml_task"]
    if len(openml_tasks) != len(set(openml_tasks)):
        raise ValueError("quality registry openml_task values must be unique")
    for item in datasets:
        if item.source not in _ALLOWED_SOURCES:
            raise ValueError(f"{item.task_id}: source must be openml_task or fedot_public_ts")
        if item.problem not in _ALLOWED_PROBLEMS:
            raise ValueError(f"{item.task_id}: problem must be classification or ts_forecasting")
        if item.fold != 0 or item.repeat != 0:
            raise ValueError(f"{item.task_id}: official fold is repeat=0 fold=0")
        if item.source == "openml_task":
            if item.problem != "classification":
                raise ValueError(f"{item.task_id}: openml_task rows are classification")
            if item.openml_task <= 0 or item.openml_data <= 0:
                raise ValueError(f"{item.task_id}: openml_task/openml_data must be positive")
        elif item.source == "fedot_public_ts":
            if item.problem != "ts_forecasting":
                raise ValueError(f"{item.task_id}: fedot_public_ts rows are ts_forecasting")
            if item.ts_dataset not in _ALLOWED_TS_DATASETS:
                raise ValueError(f"{item.task_id}: unknown public TS dataset")
            if item.forecast_horizon <= 0:
                raise ValueError(f"{item.task_id}: forecast_horizon must be positive")
    selection = raw.get("selection") or {}
    prefix = tuple(int(task) for task in selection.get("cc18_prefix_tasks") or ())
    extra = tuple(int(task) for task in selection.get("extra_openml_tasks") or ())
    registered = set(openml_tasks)
    missing_prefix = [task for task in prefix if task not in registered]
    missing_extra = [task for task in extra if task not in registered]
    if missing_prefix or missing_extra:
        raise ValueError(
            "quality registry missing frozen selection tasks: "
            + ",".join(str(task) for task in (*missing_prefix, *missing_extra))
        )
    expected_ts = tuple(str(name) for name in selection.get("fedot_public_ts") or ())
    registered_ts = {item.ts_dataset for item in datasets if item.source == "fedot_public_ts"}
    missing_ts = [name for name in expected_ts if name not in registered_ts]
    if missing_ts:
        raise ValueError("quality registry missing public TS datasets: " + ",".join(missing_ts))
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
        datasets=datasets,
        selection_study=int(selection.get("study") or 99),
        selection_rule=str(selection.get("rule") or ""),
    )
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


def is_ts_dataset(dataset: QualityDataset) -> bool:
    return dataset.problem == "ts_forecasting"


def patch_task_family(file_path: str = "", hint: str = "") -> str:
    """ts / tabular / both. Path-based; unknown shared code runs both families."""

    blob = f"{file_path} {hint}".replace("\\", "/").lower()
    ts_hit = any(marker in blob for marker in _TS_PATH_MARKERS)
    tabular_hit = any(marker in blob for marker in _TABULAR_PATH_MARKERS)
    if ts_hit and not tabular_hit:
        return "ts"
    if tabular_hit and not ts_hit:
        return "tabular"
    return "both"


def select_task_ids_for_job(
    job: dict | None = None,
    *,
    file_path: str = "",
    hint: str = "",
    requested: tuple[str, ...] | None = None,
    registry: QualityRegistry | None = None,
) -> tuple[str, ...]:
    """Pick applicable registry tasks. Inapplicable ids are omitted, not scored."""

    registry = registry or load_quality_registry()
    job = job or {}
    family = patch_task_family(
        str(job.get("file_path") or file_path),
        str(job.get("hint") or hint),
    )
    pool = requested
    if pool is None:
        raw = job.get("task_ids")
        if raw:
            pool = tuple(str(item) for item in raw if str(item))
    if not pool:
        pool = list_quality_task_ids(registry)
    known = {item.task_id: item for item in registry.datasets}
    chosen: list[str] = []
    for task_id in pool:
        dataset = known.get(task_id)
        if dataset is None:
            continue
        ts = is_ts_dataset(dataset)
        if family == "ts" and not ts:
            continue
        if family == "tabular" and ts:
            continue
        chosen.append(task_id)
    return tuple(chosen)


def job_spec(
    dataset: QualityDataset,
    *,
    n_jobs: int | None = None,
    registry: QualityRegistry | None = None,
) -> dict:
    """Fixed pair identity. Dataset choice is independent of later scores."""

    registry = registry or load_quality_registry()
    jobs = int(n_jobs if n_jobs is not None else registry.n_jobs_per_job)
    ts = is_ts_dataset(dataset)
    split = (
        f"ts_holdout_last_{dataset.forecast_horizon}"
        if ts
        else f"openml_repeat{dataset.repeat}_fold{dataset.fold}"
    )
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
        "openml_task": dataset.openml_task or None,
        "openml_data": dataset.openml_data or None,
        "ts_dataset": dataset.ts_dataset or None,
        "forecast_horizon": dataset.forecast_horizon or None,
        "split": split,
        "fold": dataset.fold,
        "repeat": dataset.repeat,
        "problem": dataset.problem,
        "metric": dataset.metric,
        "higher_is_better": dataset.higher_is_better,
        "min_delta": dataset.min_delta,
        "runs": ["stock", "patch"],
    }

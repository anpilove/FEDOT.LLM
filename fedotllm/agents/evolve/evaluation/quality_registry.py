"""Pre-registered classification, regression, and forecasting quality tasks.

Hunt never imports this module. The frozen PipelineBuilder suite in tasks.py
stays for coverage and causal probes only.

Classification list is frozen before scores: OpenML-CC18 study 99 official
order, first 20 tasks, plus four previously registered tasks. Regression uses
eight OpenML-CTR23 tasks; forecasting uses three public FEDOT series and five
checksum-pinned FEDOT real-case series. OpenML uses repeat=0 fold=0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("quality_registry.json")
_EXPECTED_PROBLEM_COUNTS = {
    "classification": 24,
    "regression": 8,
    "ts_forecasting": 8,
}
_ALLOWED_SOURCES = frozenset({"openml_task", "fedot_public_ts", "fedot_local_ts"})
_ALLOWED_PROBLEMS = frozenset({"classification", "regression", "ts_forecasting"})
TUNING_EVALUATION_POLICIES = frozenset({"legacy", "isolated-objective-process-v1"})
_ALLOWED_TS_DATASETS = frozenset({"beer", "australia", "salaries"})
_ALLOWED_LOCAL_TS_DATASETS = frozenset(
    {"temperature", "traffic", "economic", "arctic", "metocean"}
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
    data_path: str = ""
    target: str = ""
    data_sha256: str = ""
    delta_mode: str = "absolute"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityRegistry:
    identity: str
    api: str
    preset: str
    with_tuning: bool
    timeout_seconds: int
    fedot_search_budget_seconds: int
    seed: int
    require_search_ran: bool
    portfolio: bool
    cpu_quota: int
    n_jobs_per_job: int
    datasets: tuple[QualityDataset, ...]
    selection_study: int
    selection_rule: str
    watchdog_seconds: int = 10800
    fedot_cache_policy: str = "isolated-worker-v1"
    fedot_timeout_policy: str = "legacy"
    final_fit_reserve_seconds: int = 1500
    runtime_paths_policy: str = "isolated-worker-v1"
    max_pipeline_fit_policy: str = "per-cv-fold-v1"
    max_pipeline_evaluation_policy: str = "per-candidate-evaluation-v1"
    composer_options: tuple[tuple[str, int], ...] = ()
    candidate_dispatcher: str = "golem-joblib-v1"
    tuning_evaluation_policy: str = "legacy"


def _load_raw() -> dict:
    return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def load_quality_registry(path: Path | str | None = None) -> QualityRegistry:
    """Load a frozen campaign registry without changing process-global routing."""
    raw = (
        _load_raw()
        if path is None
        else json.loads(Path(path).read_text(encoding="utf-8"))
    )
    raw_composer_options = raw.get("composer_options", {})
    if not isinstance(raw_composer_options, dict):
        raise ValueError("composer_options must be an object")
    allowed_composer_options = {"pop_size", "num_of_generations"}
    composer_options = []
    for key, value in raw_composer_options.items():
        if key not in allowed_composer_options:
            raise ValueError(f"unsupported FEDOT composer option: {key}={value!r}")
        if type(value) is not int or value <= 0:
            raise ValueError(f"composer_options.{key} must be a positive integer")
        composer_options.append((key, value))
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
            data_path=str(row.get("data_path") or ""),
            target=str(row.get("target") or ""),
            data_sha256=str(row.get("data_sha256") or ""),
            delta_mode=str(row.get("delta_mode") or "absolute"),
            tags=tuple(str(tag) for tag in row.get("tags", ()) if str(tag)),
        )
        for row in raw["datasets"]
    )
    problem_counts = {
        problem: sum(item.problem == problem for item in datasets)
        for problem in _EXPECTED_PROBLEM_COUNTS
    }
    if problem_counts != _EXPECTED_PROBLEM_COUNTS:
        raise ValueError(
            "quality registry requires exactly 24 classification, 8 regression, and 8 "
            f"forecasting tasks; found {problem_counts}"
        )
    ids = [item.task_id for item in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError("quality registry task_id values must be unique")
    openml_tasks = [
        item.openml_task for item in datasets if item.source == "openml_task"
    ]
    if len(openml_tasks) != len(set(openml_tasks)):
        raise ValueError("quality registry openml_task values must be unique")
    for item in datasets:
        if item.source not in _ALLOWED_SOURCES:
            raise ValueError(
                f"{item.task_id}: unsupported quality dataset source {item.source}"
            )
        if item.problem not in _ALLOWED_PROBLEMS:
            raise ValueError(
                f"{item.task_id}: problem must be classification, regression, or ts_forecasting"
            )
        if item.fold != 0 or item.repeat != 0:
            raise ValueError(f"{item.task_id}: official fold is repeat=0 fold=0")
        if item.source == "openml_task":
            if item.problem not in {"classification", "regression"}:
                raise ValueError(
                    f"{item.task_id}: OpenML quality rows must be supervised tabular tasks"
                )
            if item.openml_task <= 0 or item.openml_data <= 0:
                raise ValueError(
                    f"{item.task_id}: openml_task/openml_data must be positive"
                )
        elif item.source == "fedot_public_ts":
            if item.problem != "ts_forecasting":
                raise ValueError(
                    f"{item.task_id}: fedot_public_ts rows are ts_forecasting"
                )
            if item.ts_dataset not in _ALLOWED_TS_DATASETS:
                raise ValueError(f"{item.task_id}: unknown public TS dataset")
            if len(item.data_sha256) != 64:
                raise ValueError(f"{item.task_id}: public TS data sha256 is required")
            if item.forecast_horizon <= 0:
                raise ValueError(f"{item.task_id}: forecast_horizon must be positive")
        elif item.source == "fedot_local_ts":
            if item.problem != "ts_forecasting":
                raise ValueError(
                    f"{item.task_id}: fedot_local_ts rows are ts_forecasting"
                )
            if item.ts_dataset not in _ALLOWED_LOCAL_TS_DATASETS:
                raise ValueError(f"{item.task_id}: unknown local TS dataset")
            if not item.data_path or not item.target or len(item.data_sha256) != 64:
                raise ValueError(
                    f"{item.task_id}: local TS path, target, and sha256 are required"
                )
            if item.forecast_horizon <= 0:
                raise ValueError(f"{item.task_id}: forecast_horizon must be positive")
        if item.delta_mode not in {"absolute", "relative"}:
            raise ValueError(
                f"{item.task_id}: unsupported delta mode {item.delta_mode}"
            )
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
    registered_ts = {
        item.ts_dataset for item in datasets if item.source == "fedot_public_ts"
    }
    missing_ts = [name for name in expected_ts if name not in registered_ts]
    if missing_ts:
        raise ValueError(
            "quality registry missing public TS datasets: " + ",".join(missing_ts)
        )
    expected_local_ts = tuple(
        str(name) for name in selection.get("fedot_local_ts") or ()
    )
    registered_local_ts = {
        item.ts_dataset for item in datasets if item.source == "fedot_local_ts"
    }
    missing_local_ts = [
        name for name in expected_local_ts if name not in registered_local_ts
    ]
    if missing_local_ts:
        raise ValueError(
            "quality registry missing local TS datasets: " + ",".join(missing_local_ts)
        )
    expected_regression = tuple(
        int(task) for task in selection.get("ctr23_regression_tasks") or ()
    )
    registered_regression = {
        item.openml_task for item in datasets if item.problem == "regression"
    }
    missing_regression = [
        task for task in expected_regression if task not in registered_regression
    ]
    if missing_regression:
        raise ValueError(
            "quality registry missing CTR23 regression tasks: "
            + ",".join(map(str, missing_regression))
        )
    registry = QualityRegistry(
        identity=str(raw["identity"]),
        api=str(raw["api"]),
        preset=str(raw["preset"]),
        with_tuning=bool(raw["with_tuning"]),
        timeout_seconds=int(raw["timeout_seconds"]),
        fedot_search_budget_seconds=int(
            raw.get("fedot_search_budget_seconds", raw["timeout_seconds"])
        ),
        seed=int(raw["seed"]),
        require_search_ran=bool(raw["require_search_ran"]),
        portfolio=bool(raw["portfolio"]),
        cpu_quota=int(raw["cpu_quota"]),
        n_jobs_per_job=int(raw["n_jobs_per_job"]),
        datasets=datasets,
        selection_study=int(selection.get("study") or 99),
        selection_rule=str(selection.get("rule") or ""),
        watchdog_seconds=int(raw.get("watchdog_seconds", 10800)),
        fedot_cache_policy=str(raw.get("fedot_cache_policy", "isolated-worker-v1")),
        fedot_timeout_policy=str(raw.get("fedot_timeout_policy", "legacy")),
        final_fit_reserve_seconds=int(raw.get("final_fit_reserve_seconds", 1500)),
        runtime_paths_policy=str(raw.get("runtime_paths_policy", "isolated-worker-v1")),
        max_pipeline_fit_policy=str(
            raw.get("max_pipeline_fit_policy", "per-cv-fold-v1")
        ),
        max_pipeline_evaluation_policy=str(
            raw.get(
                "max_pipeline_evaluation_policy",
                "per-cv-candidate-v1"
                if raw.get("fedot_timeout_policy", "legacy") == "return_best"
                else "per-candidate-evaluation-v1",
            )
        ),
        composer_options=tuple(sorted(composer_options)),
        candidate_dispatcher=str(raw.get("candidate_dispatcher", "golem-joblib-v1")),
        tuning_evaluation_policy=str(raw.get("tuning_evaluation_policy", "legacy")),
    )
    if registry.tuning_evaluation_policy not in TUNING_EVALUATION_POLICIES:
        raise ValueError("unsupported tuning evaluation policy")
    if registry.fedot_cache_policy not in {"isolated-worker-v1", "disabled-sqlite-v1"}:
        raise ValueError("unsupported FEDOT cache policy")
    if registry.runtime_paths_policy != "isolated-worker-v1":
        raise ValueError("unsupported worker runtime paths policy")
    if registry.fedot_timeout_policy not in {"legacy", "return_best"}:
        raise ValueError("unsupported FEDOT timeout policy")
    if registry.final_fit_reserve_seconds < 0:
        raise ValueError("final_fit_reserve_seconds must be non-negative")
    if registry.max_pipeline_fit_policy != "per-cv-fold-v1":
        raise ValueError("unsupported per-pipeline fit time policy")
    expected_evaluation_policy = (
        "per-cv-candidate-v1"
        if registry.fedot_timeout_policy == "return_best"
        else "per-candidate-evaluation-v1"
    )
    if registry.max_pipeline_evaluation_policy != expected_evaluation_policy:
        raise ValueError(
            "unsupported total cross-validation candidate evaluation budget policy"
        )
    if registry.candidate_dispatcher not in {
        "golem-joblib-v1",
        "isolated-candidate-process-v1",
    }:
        raise ValueError("unsupported candidate evaluation dispatcher")
    if registry.timeout_seconds <= 0:
        raise ValueError("quality timeout must be positive")
    if not 0 < registry.fedot_search_budget_seconds < registry.timeout_seconds:
        raise ValueError(
            "FEDOT search budget must be positive and leave time before the side deadline"
        )
    for key, value in registry.composer_options:
        if key not in allowed_composer_options or value <= 0:
            raise ValueError(f"unsupported FEDOT composer option: {key}={value}")
    return registry


def get_dataset(
    task_id: str, registry: QualityRegistry | None = None
) -> QualityDataset:
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
    """Compatibility view of the richer classification/regression/TS scope."""

    from fedotllm.agents.evolve.evaluation.quality_scope import analyze_quality_scope

    scope = analyze_quality_scope({"file_path": file_path, "hint": hint})[
        "problem_families"
    ]
    if scope == ["ts_forecasting"]:
        return "ts"
    if "ts_forecasting" not in scope:
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
    from fedotllm.agents.evolve.evaluation.quality_scope import analyze_quality_scope

    impact_job = dict(job)
    if file_path and not impact_job.get("file_path"):
        impact_job["file_path"] = file_path
    if hint and not impact_job.get("hint"):
        impact_job["hint"] = hint
    scope = analyze_quality_scope(impact_job, registry)
    families = set(scope["problem_families"])
    traits = set(scope.get("dataset_traits", ()))
    pool = requested
    if pool is None:
        raw = job.get("task_ids")
        if raw is not None:
            pool = tuple(str(item) for item in raw if str(item))
    if pool is None:
        pool = list_quality_task_ids(registry)
    known = {item.task_id: item for item in registry.datasets}
    chosen: list[str] = []
    for task_id in pool:
        dataset = known.get(task_id)
        if dataset is None:
            continue
        if dataset.problem not in families:
            continue
        if traits and not traits.issubset(set(dataset.tags)):
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
    cv_folds = 3 if ts else 5
    max_pipeline_evaluation_time_seconds = (
        registry.fedot_search_budget_seconds // cv_folds
    )
    max_pipeline_fit_time_seconds = (
        max_pipeline_evaluation_time_seconds // cv_folds
        if registry.fedot_timeout_policy == "return_best"
        else max_pipeline_evaluation_time_seconds
    )
    if max_pipeline_fit_time_seconds <= 0:
        raise ValueError(
            "quality timeout must allow a positive per-fold pipeline fit limit"
        )
    split = (
        f"ts_holdout_last_{dataset.forecast_horizon}"
        if ts
        else f"openml_repeat{dataset.repeat}_fold{dataset.fold}"
    )
    return {
        "kind": "quality",
        "registry_identity": registry.identity,
        "api": "fedot",
        "preset": registry.preset,
        "composer_options": dict(registry.composer_options),
        "candidate_dispatcher": registry.candidate_dispatcher,
        "tuning_evaluation_policy": registry.tuning_evaluation_policy,
        "with_tuning": registry.with_tuning,
        "timeout_seconds": registry.timeout_seconds,
        "fedot_search_budget_seconds": registry.fedot_search_budget_seconds,
        "watchdog_seconds": registry.watchdog_seconds,
        "fedot_cache_policy": registry.fedot_cache_policy,
        "fedot_timeout_policy": registry.fedot_timeout_policy,
        "final_fit_reserve_seconds": registry.final_fit_reserve_seconds,
        "runtime_paths_policy": registry.runtime_paths_policy,
        "timeout_minutes": registry.fedot_search_budget_seconds / 60.0,
        "cv_folds": cv_folds,
        "max_pipeline_fit_time_seconds": max_pipeline_fit_time_seconds,
        "max_pipeline_fit_policy": registry.max_pipeline_fit_policy,
        "max_pipeline_evaluation_time_seconds": max_pipeline_evaluation_time_seconds,
        "max_pipeline_evaluation_policy": registry.max_pipeline_evaluation_policy,
        "seed": registry.seed,
        "n_jobs": jobs,
        "cpu_limit": jobs,
        "require_search_ran": registry.require_search_ran,
        "portfolio": False,
        "source_dataset": dataset.task_id,
        "openml_task": dataset.openml_task or None,
        "openml_data": dataset.openml_data or None,
        "data_path": dataset.data_path or None,
        "target": dataset.target or None,
        "data_sha256": dataset.data_sha256 or None,
        "ts_dataset": dataset.ts_dataset or None,
        "forecast_horizon": dataset.forecast_horizon or None,
        "split": split,
        "fold": dataset.fold,
        "repeat": dataset.repeat,
        "problem": dataset.problem,
        "metric": dataset.metric,
        "higher_is_better": dataset.higher_is_better,
        "min_delta": dataset.min_delta,
        "delta_mode": dataset.delta_mode,
        "runs": ["stock", "patch"],
    }

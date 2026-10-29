from __future__ import annotations

import os
from dataclasses import replace

from fedotllm.agents.evolve.types import TaskSpec

# Ordinary working pipelines for stock vs patched quality.
_TASKS: dict[str, TaskSpec] = {}

# Profile a diverse prefix by default.  One tree-model workload mostly exposes
# orchestration wrappers and leaves Scout blind to concrete metric-bearing
# preprocessing and forecasting implementations.
DEFAULT_COVERAGE_TASKS = 12

_DEFAULT_SUITE = (
    "imputation->scaling->logit",
    "normalization->knn",
    "poly_features->logit",
    "scaling->lasso",
    "poly_features->ridge",
    "smoothing->lagged->ridge",
    "gaussian_filter->lagged->ridge",
    "pca->catboost",
    "fast_ica->lgbm",
    "catboost",
    "lgbm",
    "rf",
    "cancer",
    "kc2",
    "cancer->lgbm",
    "kc2->lgbm",
    "cholesterol",
    "river",
    "metocean",
    "temperature",
    "traffic",
    "economic",
    "arctic",
    "nemo",
    "ar->ridge",
    "ar-shifted-index->ridge",
)


def _put(**kwargs) -> TaskSpec:
    spec = TaskSpec(**kwargs)
    _TASKS[spec.task_id] = spec
    return spec


def _seq(task_id: str, *nodes: str, **kwargs) -> TaskSpec:
    return _put(task_id=task_id, kind="seq", nodes=nodes, **kwargs)


def _rmse(task_id: str, *nodes: str, **kwargs) -> TaskSpec:
    return _seq(
        task_id,
        *nodes,
        metric="holdout_rmse",
        higher_is_better=False,
        sentinel=1e9,
        min_delta=0.01,
        min_delta_mode="relative",
        **kwargs,
    )


_seq("catboost", "catboost")
_seq("lgbm", "lgbm")
_seq("rf", "rf")
_seq(
    "imputation->scaling->logit",
    "simple_imputation",
    "scaling",
    "logit",
)
_seq(
    "normalization->knn",
    "normalization",
    "knn",
    dataset="cancer",
    train_file="cancer/cancer_train.csv",
    test_file="cancer/cancer_test.csv",
    drop=("Unnamed: 0",),
)
_seq(
    "poly_features->logit",
    "poly_features",
    "logit",
    dataset="cancer",
    train_file="cancer/cancer_train.csv",
    test_file="cancer/cancer_test.csv",
    drop=("Unnamed: 0",),
)
_rmse(
    "scaling->lasso",
    "scaling",
    "lasso",
    dataset="cholesterol",
    problem="regression",
    train_file="cholesterol/cholesterol.csv",
)
_rmse(
    "poly_features->ridge",
    "poly_features",
    "ridge",
    dataset="cholesterol",
    problem="regression",
    train_file="cholesterol/cholesterol.csv",
)
_rmse(
    "smoothing->lagged->ridge",
    "smoothing",
    "lagged",
    "ridge",
    dataset="temperature",
    problem="ts",
    train_file="time_series/temperature.csv",
    target="value",
    forecast_horizon=24,
    history_size=2000,
)
_rmse(
    "gaussian_filter->lagged->ridge",
    "gaussian_filter",
    "lagged",
    "ridge",
    dataset="temperature",
    problem="ts",
    train_file="time_series/temperature.csv",
    target="value",
    forecast_horizon=24,
    history_size=2000,
)
_seq(
    "pca->catboost",
    "pca",
    "catboost",
)
_seq("fast_ica->lgbm", "fast_ica", "lgbm")
_seq(
    "cancer",
    "rf",
    dataset="cancer",
    train_file="cancer/cancer_train.csv",
    test_file="cancer/cancer_test.csv",
    drop=("Unnamed: 0",),
)
_seq(
    "kc2",
    "rf",
    dataset="kc2",
    train_file="kc2/kc2.csv",
    target="problems",
)
_seq(
    "cancer->lgbm",
    "lgbm",
    dataset="cancer",
    train_file="cancer/cancer_train.csv",
    test_file="cancer/cancer_test.csv",
    drop=("Unnamed: 0",),
)
_seq(
    "kc2->lgbm",
    "lgbm",
    dataset="kc2",
    train_file="kc2/kc2.csv",
    target="problems",
)
_rmse(
    "cholesterol",
    "ridge",
    dataset="cholesterol",
    problem="regression",
    train_file="cholesterol/cholesterol.csv",
)
_rmse(
    "river",
    "ridge",
    dataset="river",
    problem="regression",
    train_file="river_levels/station_levels.csv",
    target="level_station_2",
    drop=("date",),
)
_rmse(
    "metocean",
    "lagged",
    "ridge",
    dataset="metocean",
    problem="ts",
    train_file="metocean/metocean_data_train.csv",
    target="sea_height",
    forecast_horizon=24,
    history_size=2000,
)
_rmse(
    "temperature",
    "lagged",
    "ridge",
    dataset="temperature",
    problem="ts",
    train_file="time_series/temperature.csv",
    target="value",
    forecast_horizon=24,
    history_size=2000,
)
_rmse(
    "traffic",
    "lagged",
    "ridge",
    dataset="traffic",
    problem="ts",
    train_file="time_series/traffic.csv",
    target="value",
    forecast_horizon=24,
    history_size=2000,
)
_rmse(
    "economic",
    "lagged",
    "ridge",
    dataset="economic",
    problem="ts",
    train_file="time_series/economic_data.csv",
    target="value",
    forecast_horizon=12,
)
_rmse(
    "arctic",
    "lagged",
    "ridge",
    dataset="arctic",
    problem="ts",
    train_file="arctic/topaz_multi_ts.csv",
    target="61_91",
    forecast_horizon=14,
)
_rmse(
    "nemo",
    "lagged",
    "ridge",
    dataset="nemo",
    problem="ts",
    train_file="nemo/sea_surface_height.csv",
    target="sea_level",
    forecast_horizon=14,
)


for _task_name, _offset in (("ar->ridge", 0), ("ar-shifted-index->ridge", 1000)):
    _rmse(
        _task_name, "ar", "ridge", dataset="temperature", problem="ts",
        train_file="time_series/temperature.csv", target="value",
        forecast_horizon=24, history_size=2000, index_offset=_offset,
    )


def load_task(task_id: str) -> TaskSpec:
    spec = _TASKS.get(task_id)
    if spec is None:
        raise KeyError(task_id)
    return spec


def workload_on_dataset(dataset_task: str, pipeline_task: str | None = None) -> TaskSpec:
    """Keep the dataset contract, transplant only a registered pipeline."""
    data = load_task(dataset_task)
    if pipeline_task is None:
        return data
    pipeline = load_task(pipeline_task)
    if data.problem != pipeline.problem or data.metric != pipeline.metric:
        raise ValueError("pipeline and dataset have incompatible task types")
    return replace(data, **{
        name: getattr(pipeline, name)
        for name in ("kind", "nodes", "left", "right", "join", "tail", "index_offset")
    })


def all_tasks() -> list[TaskSpec]:
    return list(_TASKS.values())


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.environ.get(name, default).split(",") if part.strip())


def quality_suite() -> tuple[str, ...]:
    """Frozen PipelineBuilder workloads for coverage/causal probes, not quality."""

    return _csv_env("EVOLVE_AGENT_SUITE", ",".join(_DEFAULT_SUITE))


def coverage_task_limit() -> int:
    """Number of suite-prefix workloads profiled for exact executed lines."""

    raw = os.environ.get("EVOLVE_AGENT_COVERAGE_TASKS", str(DEFAULT_COVERAGE_TASKS))
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_COVERAGE_TASKS
    return max(1, min(value, len(quality_suite())))


def hidden_exam() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Quality suite used as both lift and protect.

    FINAL evaluates the same workloads on their disjoint FINAL partitions, so
    ``final_exam`` is the same pair.
    """

    suite = quality_suite()
    return suite, suite


final_exam = hidden_exam

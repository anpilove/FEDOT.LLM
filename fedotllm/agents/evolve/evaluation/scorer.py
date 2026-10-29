"""Frozen holdout scorer. Not shown to the LLM. Runs inside a subprocess."""

from __future__ import annotations

import logging
import os
import traceback as tb_mod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from fedotllm.agents.evolve.evaluation.tasks import load_task, workload_on_dataset
from fedotllm.agents.evolve.evaluation.independent_data import PROTOCOL as INDEPENDENT_DATA_PROTOCOL
from fedotllm.agents.evolve.types import TaskSpec


# The data partition is part of the frozen benchmark contract.  Confirmation
# seeds vary stochastic model/operation behaviour only; reusing them here would
# move rows between SHADOW/DEV/FINAL and leak FINAL rows into another DEV seed.
FROZEN_SPLIT_SEED = 42

_SCORING_DROP = frozenset({"ID", "target"})


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true, y_score = _metric_arrays(y_true, y_score)
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("ROC AUC requires binary 0/1 targets")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    left, right = _metric_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _metric_arrays(target, prediction) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(target, dtype=float).ravel()
    right = np.asarray(prediction, dtype=float).ravel()
    if left.size == 0 or left.size != right.size:
        raise ValueError("metric requires nonempty, equal-length targets and predictions")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("metric requires finite targets and predictions")
    return left, right


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_scoring_split() -> Split:
    import pandas as pd

    train = pd.read_csv(_case_csv("scoring/scoring_train.csv"))
    test = pd.read_csv(_case_csv("scoring/scoring_test.csv"))
    feature_cols = [c for c in train.columns if c not in _SCORING_DROP]
    return Split(
        X_train=np.array(train[feature_cols], dtype=float, copy=True),
        y_train=np.array(train["target"], dtype=int, copy=True),
        X_test=np.array(test[feature_cols], dtype=float, copy=True),
        y_test=np.array(test["target"], dtype=int, copy=True),
    )


def _cases_root() -> Path:
    import fedot

    root = Path(fedot.__file__).resolve().parents[1] / "examples" / "real_cases" / "data"
    if root.is_dir():
        return root
    env = os.environ.get("FEDOTLLM_REPO_PATH")
    if env:
        alt = Path(env) / "examples" / "real_cases" / "data"
        if alt.is_dir():
            return alt
    raise FileNotFoundError("FEDOT examples/real_cases/data not found")


def _case_csv(rel: str) -> Path:
    path = _cases_root() / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@contextmanager
def _production_session() -> Iterator[None]:
    pytest_key = "PYTEST_CURRENT_TEST"
    saved_pytest = os.environ.pop(pytest_key, None)
    prev_log = logging.Logger._log

    def _log_without_golem_kwargs(self, level, msg, args, **kwargs):
        kwargs.pop("raise_if_test", None)
        kwargs.pop("exc", None)
        return prev_log(self, level, msg, args, **kwargs)

    logging.Logger._log = _log_without_golem_kwargs  # type: ignore[method-assign]
    try:
        yield
    finally:
        logging.Logger._log = prev_log  # type: ignore[method-assign]
        if saved_pytest is not None:
            os.environ[pytest_key] = saved_pytest


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _table_xy(frame, spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    drop = set(spec.drop) | {spec.target}
    feat_cols = [col for col in frame.columns if col not in drop]
    features = frame[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    raw = frame[spec.target]
    if spec.problem == "classification" and raw.dtype == object:
        labels = (raw.astype(str).str.lower() == "yes").astype(int).to_numpy()
    elif spec.problem == "classification":
        labels = np.array(raw, dtype=int, copy=True)
    else:
        labels = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    return features, labels


def _tabular_input(features: np.ndarray, target: np.ndarray, *, problem: str):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    task_type = TaskTypesEnum.regression if problem == "regression" else TaskTypesEnum.classification
    y = np.asarray(target)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return InputData(
        idx=np.arange(len(target)),
        features=np.asarray(features),
        target=y,
        task=Task(task_type),
        data_type=DataTypesEnum.table,
    )


def _text_input(features: np.ndarray, target: np.ndarray):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    y = np.asarray(target).reshape(-1, 1)
    return InputData(
        idx=np.arange(len(target)),
        features=np.asarray(features),
        target=y,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.text,
    )


def _split_indices(target: np.ndarray, *, test_size: float, seed: int, problem: str):
    from sklearn.model_selection import train_test_split

    stratify = target if problem in {"classification", "text"} and len(np.unique(target)) > 1 else None
    indices = np.arange(len(target))
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )


def _shadow_train_split(features, target, *, seed: int, problem: str):
    """Carve a validation view only from training rows.

    The shadow view is deliberately disjoint from both frozen DEV and FINAL.
    It is used after an apparent DEV improvement to reject changes that merely
    exploit the single DEV sample.  FINAL rows never participate in it.
    """

    train_idx, shadow_idx = _split_indices(
        target, test_size=0.2, seed=seed + 1, problem=problem
    )
    return (
        features[train_idx],
        target[train_idx],
        features[shadow_idx],
        target[shadow_idx],
    )


def _single_csv_split(features, target, *, seed: int, problem: str, split_name: str):
    """Deterministic train/SHADOW/DEV/FINAL split without holdout leakage."""

    if split_name not in {"shadow", "dev", "final"}:
        raise ValueError(f"unknown split {split_name!r}")

    train_idx, holdout_idx = _split_indices(
        target, test_size=0.4, seed=seed, problem=problem
    )
    if split_name == "shadow":
        return _shadow_train_split(
            features[train_idx],
            target[train_idx],
            seed=seed,
            problem=problem,
        )
    holdout_target = target[holdout_idx]
    dev_rel, final_rel = _split_indices(
        holdout_target, test_size=0.5, seed=seed, problem=problem
    )
    eval_idx = holdout_idx[dev_rel if split_name == "dev" else final_rel]
    return features[train_idx], target[train_idx], features[eval_idx], target[eval_idx]


def _separate_test_split(features, target, *, seed: int, problem: str, split_name: str):
    """Split a provided test CSV into disjoint 50/50 DEV and FINAL."""

    if split_name not in {"dev", "final"}:
        raise ValueError(f"separate test supports only DEV/FINAL, got {split_name!r}")

    dev_idx, final_idx = _split_indices(target, test_size=0.5, seed=seed, problem=problem)
    idx = dev_idx if split_name == "dev" else final_idx
    return features[idx], target[idx]


def _make_split(
    spec: TaskSpec,
    split_name: str = "dev",
    *,
    split_seed: int = FROZEN_SPLIT_SEED,
):
    import pandas as pd

    if spec.dataset == "scoring":
        split = load_scoring_split()
        if split_name == "shadow":
            x_train, y_train, x_test, y_test = _shadow_train_split(
                split.X_train,
                split.y_train,
                seed=split_seed,
                problem="classification",
            )
            train = _tabular_input(x_train, y_train, problem="classification")
            test = _tabular_input(x_test, y_test, problem="classification")
            return train, test, int(x_train.shape[0])
        x_test, y_test = _separate_test_split(
            split.X_test,
            split.y_test,
            seed=split_seed,
            problem="classification",
            split_name=split_name,
        )
        train = _tabular_input(split.X_train, split.y_train, problem="classification")
        test = _tabular_input(x_test, y_test, problem="classification")
        return train, test, int(split.X_train.shape[0])

    if spec.problem == "ts":
        return _ts_split(spec, split_name=split_name)

    if spec.problem == "text":
        frame = pd.read_csv(_case_csv(spec.train_file))
        texts = frame["text"].astype(str).to_numpy()
        labels = np.array(frame[spec.target], dtype=int, copy=True)
        x_train, y_train, x_test, y_test = _single_csv_split(
            texts, labels, seed=split_seed, problem="text", split_name=split_name
        )
        return _text_input(x_train, y_train), _text_input(x_test, y_test), int(len(y_train))

    train_df = pd.read_csv(_case_csv(spec.train_file))
    x_train, y_train = _table_xy(train_df, spec)
    if spec.test_file:
        test_df = pd.read_csv(_case_csv(spec.test_file))
        x_test_all, y_test_all = _table_xy(test_df, spec)
        if split_name == "shadow":
            x_train, y_train, x_test, y_test = _shadow_train_split(
                x_train,
                y_train,
                seed=split_seed,
                problem=spec.problem,
            )
        else:
            x_test, y_test = _separate_test_split(
                x_test_all,
                y_test_all,
                seed=split_seed,
                problem=spec.problem,
                split_name=split_name,
            )
    else:
        x_train, y_train, x_test, y_test = _single_csv_split(
            x_train,
            y_train,
            seed=split_seed,
            problem=spec.problem,
            split_name=split_name,
        )
    return (
        _tabular_input(x_train, y_train, problem=spec.problem),
        _tabular_input(x_test, y_test, problem=spec.problem),
        int(len(y_train)),
    )


def _make_independent_split(spec: TaskSpec, split_name: str, *, fold=None):
    import pandas as pd
    from fedotllm.agents.evolve.evaluation.independent_data import (
        _digest, evidence, table_split, ts_indices,
    )

    if spec.problem == "ts":
        from fedot.core.data.data import InputData
        from fedot.core.data.data_split import train_test_data_setup
        from fedot.core.repository.dataset_types import DataTypesEnum
        from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

        frame = pd.read_csv(_case_csv(spec.train_file))
        series = np.asarray(pd.to_numeric(frame[spec.target], errors="coerce"), dtype=float)
        series = series[np.isfinite(series)]
        if spec.history_size:
            series = series[-spec.history_size:]
        horizon = spec.forecast_horizon or 12
        train_ids, test_ids = ts_indices(len(series), horizon, split=split_name, fold=fold)
        selected = series[:int(test_ids[-1]) + 1]
        data = InputData(
            idx=np.arange(len(selected)), features=selected, target=selected,
            task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon)),
            data_type=DataTypesEnum.ts,
        )
        train, test = train_test_data_setup(data)
        _assert_frozen_ts_boundary(selected, horizon, train, test)
        metadata = evidence(
            data_hash=_digest(series.tolist()), train=train_ids, namespace="train",
            validation=test_ids, core=np.arange(len(series) - 3 * horizon),
            split=split_name, fold=fold, kind="rolling_origin",
        )
        return train, test, len(train_ids), metadata

    external = None
    if spec.dataset == "scoring":
        data = load_scoring_split()
        features, target = data.X_train, data.y_train
        external = (data.X_test, data.y_test)
    else:
        frame = pd.read_csv(_case_csv(spec.train_file))
        if spec.problem == "text":
            features = frame["text"].astype(str).to_numpy()
            target = np.asarray(frame[spec.target], dtype=int)
        else:
            features, target = _table_xy(frame, spec)
        if spec.test_file:
            external = _table_xy(pd.read_csv(_case_csv(spec.test_file)), spec)
    x_train, y_train, x_test, y_test, metadata = table_split(
        features, target, problem=spec.problem, split=split_name, fold=fold, external=external,
    )
    if spec.problem == "text":
        train, test = _text_input(x_train, y_train), _text_input(x_test, y_test)
    else:
        train = _tabular_input(x_train, y_train, problem=spec.problem)
        test = _tabular_input(x_test, y_test, problem=spec.problem)
    return train, test, len(y_train), metadata


def _ts_split(spec: TaskSpec, *, split_name: str = "dev"):
    import pandas as pd
    from fedot.core.data.data import InputData
    from fedot.core.data.data_split import train_test_data_setup
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

    if split_name not in {"shadow", "dev", "final"}:
        raise ValueError(f"unknown split {split_name!r}")
    frame = pd.read_csv(_case_csv(spec.train_file))
    series = np.asarray(pd.to_numeric(frame[spec.target], errors="coerce"), dtype=float)
    series = series[np.isfinite(series)]
    if spec.history_size and series.size > spec.history_size:
        series = series[-spec.history_size :]
    horizon = spec.forecast_horizon or 12
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))
    if split_name == "shadow":
        if series.size <= 3 * horizon:
            raise ValueError("time series is too short for SHADOW, DEV and FINAL windows")
        selected = series[: -2 * horizon]
    elif split_name == "dev":
        if series.size <= 2 * horizon:
            raise ValueError("time series is too short for disjoint DEV and FINAL windows")
        selected = series[:-horizon]
    else:
        selected = series
    data = InputData(
        idx=np.arange(selected.size),
        features=selected,
        target=selected,
        task=task,
        data_type=DataTypesEnum.ts,
    )
    train, test = train_test_data_setup(data)
    _assert_frozen_ts_boundary(selected, horizon, train, test)
    return train, test, int(len(np.asarray(train.target).ravel()))


def _assert_frozen_ts_boundary(selected, horizon: int, train, test) -> None:
    """Prevent source patches from improving scores by training on holdout targets."""

    series = np.asarray(selected).ravel()
    expected_train = series[:-horizon]
    expected_test = series[-horizon:]
    actual_train = np.asarray(train.target).ravel()
    actual_test = np.asarray(test.target).ravel()
    expected_train_idx = np.arange(expected_train.size)
    expected_test_idx = np.arange(expected_train.size, series.size)
    train_idx = np.asarray(train.idx).ravel()
    test_idx = np.asarray(test.idx).ravel()
    if not (
        np.array_equal(actual_train, expected_train)
        and np.array_equal(actual_test, expected_test)
        and np.array_equal(train_idx, expected_train_idx)
        and np.array_equal(test_idx, expected_test_idx)
    ):
        raise RuntimeError(
            "protected evaluation split contract violated: training rows must "
            "not include DEV/FINAL target indexes"
        )


def _predict_score(spec: TaskSpec, pipeline, test, observations: dict | None = None) -> float:
    if spec.metric == "holdout_roc_auc":
        pred = pipeline.predict(test, output_mode="probs")
        arr = np.asarray(pred.predict)
        scores = arr[:, -1] if arr.ndim == 2 else arr.ravel()
        if observations is not None:
            observations.update(target=np.asarray(test.target).ravel().tolist(), prediction=scores.tolist())
        return float(roc_auc_score(np.asarray(test.target), scores))
    pred = pipeline.predict(test)
    if observations is not None:
        observations.update(target=np.asarray(test.target).ravel().tolist(),
                            prediction=np.asarray(pred.predict).ravel().tolist())
    return float(rmse_score(np.asarray(test.target), np.asarray(pred.predict)))


def score_task(
    task_id: str,
    seed: int = 42,
    split_name: str = "dev",
    task_override: dict | None = None,
) -> dict:
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

    _set_seed(seed)
    spec = load_task(task_id)
    override = dict(task_override or {})
    if override.get("pipeline_task") is not None:
        spec = workload_on_dataset(task_id, override["pipeline_task"])
    if "index_offset" in override:
        spec = replace(spec, index_offset=int(override["index_offset"]))
    history_size = override.get("history_size")
    if history_size is not None:
        spec = replace(spec, history_size=max(0, int(history_size)))
    operation_params = override.get("operation_params") or {}
    if not isinstance(operation_params, dict):
        raise TypeError("operation_params override must be a mapping")
    traceback_text = ""
    detail = "not_run"
    automl = spec.sentinel
    status = "ok"
    n_train = 0
    data_evidence = {}
    metric_observations = {}
    with _production_session():
        try:
            if split_name not in {"shadow", "dev", "final"}:
                raise ValueError(f"unknown split {split_name!r}")
            protocol = override.get("evaluation_protocol")
            if protocol is None:
                train, test, n_train = _make_split(spec, split_name)
            elif protocol == INDEPENDENT_DATA_PROTOCOL:
                train, test, n_train, data_evidence = _make_independent_split(
                    spec, split_name, fold=override.get("resample_fold"),
                )
            else:
                raise ValueError(f"unknown evaluation protocol: {protocol}")
            if spec.index_offset:
                if spec.problem != "ts":
                    raise ValueError("index offsets are restricted to time series")
                train.idx = np.asarray(train.idx) + spec.index_offset
                test.idx = np.asarray(test.idx) + spec.index_offset
            builder = PipelineBuilder()
            if spec.kind == "seq":
                for node in spec.nodes:
                    params = operation_params.get(node)
                    if params is not None and not isinstance(params, dict):
                        raise TypeError(f"parameters for {node!r} must be a mapping")
                    builder = (
                        builder.add_node(node, params=dict(params))
                        if params
                        else builder.add_node(node)
                    )
            elif spec.kind == "branch":
                builder = builder.add_branch(spec.left, spec.right).join_branches(spec.join)
            elif spec.kind == "branch_then":
                builder = builder.add_branch(spec.left, spec.right).join_branches(spec.join)
                for node in spec.tail:
                    builder = builder.add_node(node)
            else:
                return {
                    "task_id": task_id,
                    "status": "invalid",
                    "score": float("nan"),
                    "traceback": "",
                    "detail": f"unknown kind {spec.kind}",
                    "n_train": n_train,
                }
            pipeline = builder.build()
            _set_seed(seed)
            pipeline.fit(train)
            automl = (
                _predict_score(spec, pipeline, test, observations=metric_observations)
                if data_evidence else _predict_score(spec, pipeline, test)
            )
            detail = (
                f"ok split={split_name} split_seed={FROZEN_SPLIT_SEED} "
                f"model_seed={seed}"
            )
            if not np.isfinite(automl):
                status = "invalid"
            else:
                status = "ok"
        except Exception as exc:
            traceback_text = tb_mod.format_exc()
            detail = f"{type(exc).__name__}: {exc}"
            automl = spec.sentinel
            status = "crash"

    return {
        "task_id": task_id,
        "status": status,
        "score": automl,
        "traceback": traceback_text,
        "detail": detail,
        "n_train": n_train,
        "model_seed": seed,
        "split_seed": FROZEN_SPLIT_SEED,
        "data_evidence": data_evidence,
        "metric_observations": metric_observations,
    }

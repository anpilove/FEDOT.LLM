"""Frozen holdout scorer. Not shown to the LLM. Runs inside a subprocess."""

from __future__ import annotations

import logging
import os
import traceback as tb_mod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from research.evolve.metric_agent.tasks import load_task

_SCORING_DROP = frozenset({"ID", "target"})
_SCORING_CACHE = Path(os.environ.get("FEDOT_SCORING_CACHE", "/tmp/fedot-official-scoring"))
_SCORING_URLS = {
    "scoring_train.csv": (
        "https://raw.githubusercontent.com/aimclub/FEDOT/master/"
        "examples/real_cases/data/scoring/scoring_train.csv"
    ),
    "scoring_test.csv": (
        "https://raw.githubusercontent.com/aimclub/FEDOT/master/"
        "examples/real_cases/data/scoring/scoring_test.csv"
    ),
}


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _official_scoring_csv(name: str) -> Path:
    import urllib.request

    _SCORING_CACHE.mkdir(parents=True, exist_ok=True)
    path = _SCORING_CACHE / name
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(_SCORING_URLS[name], path)
    return path


def load_scoring_split() -> Split:
    import pandas as pd

    train = pd.read_csv(_official_scoring_csv("scoring_train.csv"))
    test = pd.read_csv(_official_scoring_csv("scoring_test.csv"))
    feature_cols = [c for c in train.columns if c not in _SCORING_DROP]
    return Split(
        X_train=np.array(train[feature_cols], dtype=float, copy=True),
        y_train=np.array(train["target"], dtype=int, copy=True),
        X_test=np.array(test[feature_cols], dtype=float, copy=True),
        y_test=np.array(test["target"], dtype=int, copy=True),
    )


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


def _input_pair():
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    split = load_scoring_split()
    task = Task(TaskTypesEnum.classification)
    train = InputData(
        idx=np.arange(len(split.y_train)),
        features=np.asarray(split.X_train),
        target=np.asarray(split.y_train).reshape(-1, 1),
        task=task,
        data_type=DataTypesEnum.table,
    )
    test = InputData(
        idx=np.arange(len(split.y_test)),
        features=np.asarray(split.X_test),
        target=np.asarray(split.y_test).reshape(-1, 1),
        task=task,
        data_type=DataTypesEnum.table,
    )
    return split, train, test


def score_task(task_id: str) -> dict:
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

    spec = load_task(task_id)
    split, train, test = _input_pair()
    traceback_text = ""
    detail = "not_run"
    automl = 0.5
    status = "ok"
    with _production_session():
        builder = PipelineBuilder()
        if spec.kind == "seq":
            for node in spec.nodes:
                builder = builder.add_node(node)
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
                "n_train": int(split.X_train.shape[0]),
            }
        pipeline = builder.build()
        try:
            pipeline.fit(train)
            pred = pipeline.predict(test, output_mode="probs")
            arr = np.asarray(pred.predict)
            scores = arr[:, -1] if arr.ndim == 2 else arr.ravel()
            automl = float(roc_auc_score(split.y_test, scores))
            detail = "ok"
            if automl != automl:
                status = "invalid"
            else:
                status = "ok"
        except Exception as exc:
            traceback_text = tb_mod.format_exc()
            detail = f"{type(exc).__name__}: {exc}"
            automl = 0.5
            status = "crash"

    return {
        "task_id": task_id,
        "status": status,
        "score": automl,
        "traceback": traceback_text,
        "detail": detail,
        "n_train": int(split.X_train.shape[0]),
    }

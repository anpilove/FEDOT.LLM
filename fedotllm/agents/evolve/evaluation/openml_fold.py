"""Load a pre-registered OpenML task fold. Not used by hunt."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from fedotllm.agents.evolve.evaluation.quality_registry import QualityDataset

_OPENML_JSON = "https://www.openml.org/api/v1/json"
_OPENML_SPLITS = "https://www.openml.org/api_splits/get/{task}/{task}"


def cache_root() -> Path:
    return Path(os.environ.get("EVOLVE_QUALITY_DATA_CACHE", "/tmp/evolve-fedot-quality-data"))


def load_registered_fold(dataset: QualityDataset) -> dict:
    """Return train/test frames for the locked official fold."""

    try:
        import openml

        task = openml.tasks.get_task(dataset.openml_task)
        features, target = task.get_X_and_y(dataset_format="dataframe")
        train_idx, test_idx = task.get_train_test_split_indices(
            fold=dataset.fold, repeat=dataset.repeat
        )
        x_train = features.iloc[list(train_idx)].reset_index(drop=True)
        y_train = target.iloc[list(train_idx)].reset_index(drop=True)
        x_test = features.iloc[list(test_idx)].reset_index(drop=True)
        y_test = target.iloc[list(test_idx)].reset_index(drop=True)
        return {
            "X_train": x_train,
            "y_train": y_train,
            "X_test": x_test,
            "y_test": y_test,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "loader": "openml.tasks.get_task",
        }
    except Exception:
        return _load_sklearn_and_splits(dataset)


def _load_sklearn_and_splits(dataset: QualityDataset) -> dict:
    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(
        data_id=dataset.openml_data,
        as_frame=True,
        parser="auto",
    )
    features = bundle.data
    target = bundle.target
    train_idx, test_idx = _official_indices(dataset, n_rows=len(features))
    return {
        "X_train": features.iloc[train_idx].reset_index(drop=True),
        "y_train": target.iloc[train_idx].reset_index(drop=True),
        "X_test": features.iloc[test_idx].reset_index(drop=True),
        "y_test": target.iloc[test_idx].reset_index(drop=True),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "loader": "sklearn.fetch_openml+openml_splits",
    }


def _official_indices(dataset: QualityDataset, *, n_rows: int) -> tuple[list[int], list[int]]:
    cached = cache_root() / f"task-{dataset.openml_task}-r{dataset.repeat}-f{dataset.fold}.json"
    if cached.is_file():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        return list(payload["train"]), list(payload["test"])
    train_idx, test_idx = _download_split_indices(dataset, n_rows=n_rows)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps({"train": train_idx, "test": test_idx}),
        encoding="utf-8",
    )
    return train_idx, test_idx


def _download_split_indices(dataset: QualityDataset, *, n_rows: int) -> tuple[list[int], list[int]]:
    url = _OPENML_SPLITS.format(task=dataset.openml_task)
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    train: list[int] = []
    test: list[int] = []
    in_data = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        parsed = _parse_split_row(parts)
        if parsed is None:
            continue
        kind, rowid, repeat, fold = parsed
        if repeat != dataset.repeat or fold != dataset.fold:
            continue
        index = rowid - 1 if rowid >= 1 else rowid
        if index < 0 or index >= n_rows:
            continue
        if kind == "TRAIN":
            train.append(index)
        elif kind == "TEST":
            test.append(index)
    if not train or not test:
        raise RuntimeError(
            f"official OpenML split missing for task {dataset.openml_task} "
            f"repeat={dataset.repeat} fold={dataset.fold}"
        )
    return train, test


def _parse_split_row(parts: list[str]) -> tuple[str, int, int, int] | None:
    """Accept official OpenML order type,rowid,repeat,fold and the reverse."""

    head = parts[0].strip().strip("'").upper()
    tail = parts[3].strip().strip("'").upper()
    try:
        if head in {"TRAIN", "TEST"}:
            return head, int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
        if tail in {"TRAIN", "TEST"}:
            return tail, int(float(parts[2])), int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None
    return None


def describe_task(task_id: int) -> dict:
    url = f"{_OPENML_JSON}/task/{int(task_id)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))

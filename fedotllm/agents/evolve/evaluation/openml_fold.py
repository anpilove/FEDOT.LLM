"""Load a pre-registered OpenML fold or public FEDOT TS series. Not used by hunt."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from fedotllm.agents.evolve.evaluation.quality_registry import QualityDataset

_OPENML_JSON = "https://www.openml.org/api/v1/json"
_OPENML_SPLITS = "https://www.openml.org/api_splits/get/{task}/{task}"
_FEDOT_TS_RAW = "https://raw.githubusercontent.com/aimclub/FEDOT/master/examples/data/ts/{name}.csv"
_FEDOT_TS_FILES = {
    "beer": "beer.csv",
    "australia": "australia.csv",
    "salaries": "salaries.csv",
}


def cache_root() -> Path:
    return Path(os.environ.get("EVOLVE_QUALITY_DATA_CACHE", "/tmp/evolve-fedot-quality-data"))


def load_registered_fold(dataset: QualityDataset) -> dict:
    """Return train/test frames for the locked official fold or TS holdout."""

    if dataset.source == "fedot_public_ts":
        return load_fedot_public_ts(dataset)
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
        # Older caches were written with rowid shifted by one (row 0 duplicated,
        # last row dropped). Only trust files that record the 0-based origin.
        if payload.get("rowid_base") == 0:
            return list(payload["train"]), list(payload["test"])
    train_idx, test_idx = _download_split_indices(dataset, n_rows=n_rows)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps({"train": train_idx, "test": test_idx, "rowid_base": 0}),
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
        # OpenML split files use 0-based rowid (same as openml-python).
        index = rowid
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


def _fedot_checkout_ts_csv(name: str) -> Path | None:
    rel = Path("examples") / "data" / "ts" / _FEDOT_TS_FILES[name]
    raw = os.environ.get("FEDOTLLM_REPO_PATH")
    if raw:
        local = Path(raw) / rel
        if local.is_file():
            return local
    raw_cache = os.environ.get("FEDOTLLM_REPO_CACHE")
    if raw_cache:
        local = Path(raw_cache) / rel
        if local.is_file():
            return local
    return None


def load_fedot_public_ts(dataset: QualityDataset) -> dict:
    """Full public FEDOT example series, last-horizon holdout. Not toy CSV."""

    import numpy as np
    import pandas as pd

    name = dataset.ts_dataset or dataset.name
    if name not in _FEDOT_TS_FILES:
        raise ValueError(f"unsupported public TS dataset: {name}")
    horizon = int(dataset.forecast_horizon)
    if horizon <= 0:
        raise ValueError(f"{dataset.task_id}: forecast_horizon must be positive")
    cached = cache_root() / f"fedot-ts-{name}.csv"
    source = _fedot_checkout_ts_csv(name)
    if source is None:
        if not cached.is_file():
            url = _FEDOT_TS_RAW.format(name=name)
            with urllib.request.urlopen(url, timeout=60) as response:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(response.read())
        source = cached
    frame = pd.read_csv(source)
    if "value" not in frame.columns:
        raise RuntimeError(f"{name}: public TS CSV has no value column")
    series = np.asarray(pd.to_numeric(frame["value"], errors="coerce"), dtype=float)
    series = series[np.isfinite(series)]
    if series.size <= horizon:
        raise RuntimeError(f"{name}: series shorter than horizon {horizon}")
    train = series[:-horizon]
    test = series[-horizon:]
    return {
        "X_train": train,
        "y_train": train,
        "X_test": train,
        "y_test": test,
        "n_train": int(train.size),
        "n_test": int(test.size),
        "forecast_horizon": horizon,
        "loader": "fedot.examples.data.ts",
    }



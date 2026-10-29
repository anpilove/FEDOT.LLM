"""Load a pre-registered OpenML fold or public FEDOT TS series. Not used by hunt."""

from __future__ import annotations

import json
import hashlib
import os
import pickle
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

from fedotllm.agents.evolve.evaluation.quality_registry import QualityDataset

_OPENML_JSON = "https://www.openml.org/api/v1/json"
_OPENML_SPLITS = "https://www.openml.org/api_splits/get/{task}/{task}"
_FEDOT_TS_RAW = (
    "https://raw.githubusercontent.com/aimclub/FEDOT/master/examples/data/ts/{name}.csv"
)
_FEDOT_TS_FILES = {
    "beer": "beer.csv",
    "australia": "australia.csv",
    "salaries": "salaries.csv",
}


def cache_root() -> Path:
    return Path(
        os.environ.get("EVOLVE_QUALITY_DATA_CACHE", "/tmp/evolve-fedot-quality-data")
    )


def _atomic_cache_write(path: Path, payload: bytes) -> None:
    """Publish complete cache contents so concurrent workers never read a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_registered_fold(dataset: QualityDataset) -> dict:
    """Return train/test frames for the locked official fold or TS holdout."""

    with _measurement_network_policy():
        return _load_registered_fold(dataset)


def _explicit_cache(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    path = Path(raw)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.is_dir()
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ValueError(
            f"{name}: existing absolute nonredirected cache required: {path}"
        )
    return str(path)


@contextmanager
def _measurement_network_policy():
    """Only cached inputs are allowed in an explicitly offline measurement.

    This scope runs in one physical worker during fold loading. It restores
    the APIs even when a cache miss fails, before any ML threads are started.
    """
    if os.environ.get("EVOLVE_QUALITY_DATA_OFFLINE") != "1":
        yield
        return
    import socket
    from unittest.mock import patch

    def unavailable(*args, **kwargs):
        raise RuntimeError(
            "offline measurement: dataset assets must be prepared and pinned before training"
        )

    with (
        patch.object(urllib.request, "urlopen", unavailable),
        patch.object(socket.socket, "connect", unavailable),
        patch.object(socket.socket, "connect_ex", unavailable),
        patch.object(socket, "getaddrinfo", unavailable),
        patch.object(socket, "create_connection", unavailable),
    ):
        yield


def _prepared_cache_file(path: Path) -> Path:
    if (
        ".." in path.parts
        or not path.is_absolute()
        or not path.is_file()
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ValueError(f"prepared regular OpenML cache file required: {path}")
    return path


def _read_prepared_pickle(path: Path):
    # These are the already prepared, hash-bound caches that OpenML itself
    # deserializes. Never call its repair/rebuild paths during measurement.
    with _prepared_cache_file(path).open("rb") as stream:
        return pickle.load(stream)


def _prepared_task_target(dataset: QualityDataset, server_cache: Path) -> str:
    path = server_cache / "tasks" / str(dataset.openml_task) / "task.xml"
    task = ET.parse(_prepared_cache_file(path)).getroot()
    namespace = "{http://openml.org/openml}"
    if task.tag != namespace + "task":
        raise ValueError("prepared OpenML task requires its official namespace")

    def one_text(parent, name):
        elements = parent.findall(namespace + name)
        if (
            len(elements) != 1
            or len(elements[0])
            or not elements[0].text
            or not elements[0].text.strip()
        ):
            raise ValueError(f"prepared OpenML task requires one explicit {name}")
        return elements[0].text.strip()

    if int(one_text(task, "task_id")) != dataset.openml_task:
        raise ValueError("prepared OpenML task ID differs from the registry")
    expected_type = {"classification": 1, "regression": 2}[dataset.problem]
    if int(one_text(task, "task_type_id")) != expected_type:
        raise ValueError("prepared OpenML task type differs from the registry")
    sources = [
        entry
        for entry in task.findall(namespace + "input")
        if entry.get("name") == "source_data"
    ]
    if len(sources) != 1:
        raise ValueError("prepared OpenML task requires one source_data input")
    data = sources[0].findall(namespace + "data_set")
    if len(data) != 1:
        raise ValueError("prepared OpenML task requires one source data_set")
    if int(one_text(data[0], "data_set_id")) != dataset.openml_data:
        raise ValueError("prepared OpenML task/data mapping differs from the registry")
    return one_text(data[0], "target_feature")


def _prepared_openml_cache(dataset: QualityDataset, server_cache: Path) -> Path:
    import numpy as np
    import pandas as pd
    from scipy import sparse

    # get_task reads classification metadata before returning its dataset ID.
    # Verify the same official fields as OpenML's task XML parser first.
    target = _prepared_task_target(dataset, server_cache)
    directory = server_cache / "datasets" / str(dataset.openml_data)
    if dataset.problem == "classification":
        _prepared_cache_file(directory / "features.xml")
        features = _read_prepared_pickle(directory / "features.xml.pkl")
        if not isinstance(features, dict):
            raise ValueError("prepared OpenML feature cache must be a dictionary")
        target_features = [
            feature
            for feature in features.values()
            if getattr(feature, "name", None) == target
        ]
        if len(target_features) != 1:
            raise ValueError(
                "prepared OpenML feature cache must identify the task target"
            )
        target_type = getattr(target_features[0], "data_type", None)
        if (
            not isinstance(target_type, str)
            or not target_type
            or target_type == "string"
        ):
            # OpenML resolves string class labels by loading a fresh dataset
            # instance inside get_task, before its cache pointer can be bound.
            raise ValueError(
                "prepared OpenML classification target requires readonly metadata"
            )
    split = server_cache / "tasks" / str(dataset.openml_task) / "datasplits.pkl.py3"
    _read_prepared_pickle(split)
    path = directory / f"dataset_{dataset.openml_data}.pkl.py3"
    cached = _read_prepared_pickle(path)
    if not isinstance(cached, (tuple, list)) or len(cached) != 3:
        raise ValueError(
            "prepared OpenML dataset cache must contain data and column metadata"
        )
    values, categorical, names = cached
    if not isinstance(values, pd.DataFrame) and not sparse.issparse(values):
        raise ValueError(
            "prepared OpenML dataset cache requires a current DataFrame or sparse matrix"
        )
    if (
        not isinstance(names, (list, tuple))
        or not all(isinstance(name, str) for name in names)
        or not isinstance(categorical, (list, tuple, np.ndarray))
        or len(names) != values.shape[1]
        or len(categorical) != values.shape[1]
        or not all(isinstance(flag, (bool, np.bool_)) for flag in categorical)
        or (isinstance(values, pd.DataFrame) and list(values.columns) != list(names))
    ):
        raise ValueError(
            "prepared OpenML dataset cache has inconsistent column metadata"
        )
    return path


def _offline_openml_data(task, cache_path: Path):
    # OpenML 0.15.1's metadata-only get_X_and_y forgets the cached pickle
    # pointer and rewrites it even when the bytes are already present. Bind
    # only this new dataset instance to the validated prepared representation.
    dataset = task.get_dataset(download_data=False)
    target = getattr(task, "target_name", None)
    if not isinstance(target, str) or not target:
        raise ValueError("registered OpenML task requires its explicit target feature")
    dataset.cache_format = "pickle"
    dataset.data_pickle_file = str(cache_path)
    dataset.data_file = None
    dataset.parquet_file = None

    def forbidden_writer(*args, **kwargs):
        raise RuntimeError(
            "offline OpenML measurement cannot rebuild or download cache files"
        )

    dataset._cache_compressed_file_from_file = forbidden_writer
    dataset._download_data = forbidden_writer
    features, target, _, _ = dataset.get_data(dataset_format="dataframe", target=target)
    return features, target


def _load_registered_fold(dataset: QualityDataset) -> dict:
    if dataset.source == "fedot_public_ts":
        return load_fedot_public_ts(dataset)
    if dataset.source == "fedot_local_ts":
        return load_fedot_local_ts(dataset)
    offline = os.environ.get("EVOLVE_QUALITY_DATA_OFFLINE") == "1"
    prepared_cache = None
    try:
        import openml

        explicit_cache = _explicit_cache("EVOLVE_OPENML_CACHE")
        if explicit_cache is not None:
            setter = getattr(
                openml.config, "set_root_cache_directory", None
            ) or getattr(openml.config, "set_cache_directory", None)
            if not callable(setter):
                raise RuntimeError("unsupported OpenML cache configuration interface")
            setter(explicit_cache)
            server_cache = Path(openml.config.get_cache_directory())
            if (
                not server_cache.is_absolute()
                or ".." in server_cache.parts
                or not server_cache.is_relative_to(Path(explicit_cache))
                or not server_cache.is_dir()
                or any(
                    part.is_symlink() for part in (server_cache, *server_cache.parents)
                )
            ):
                raise RuntimeError(
                    f"OpenML server cache outside checked root or missing: {server_cache}"
                )
            if offline and server_cache != Path(explicit_cache) / "org/openml/www":
                raise RuntimeError(
                    f"OpenML server cache differs from pinned production namespace: {server_cache}"
                )
        if offline:
            if explicit_cache is None:
                raise ValueError(
                    "offline OpenML measurement requires its explicit prepared cache"
                )
            prepared_cache = _prepared_openml_cache(dataset, server_cache)
        task = openml.tasks.get_task(dataset.openml_task)
    except Exception as exc:
        raise RuntimeError(
            f"{dataset.task_id}: could not verify OpenML task/data mapping for "
            f"task {dataset.openml_task} and data {dataset.openml_data}"
        ) from exc

    task_data = getattr(task, "dataset_id", None)
    if task_data is None or int(task_data) != dataset.openml_data:
        raise RuntimeError(
            f"{dataset.task_id}: OpenML task/data ID mismatch: task "
            f"{dataset.openml_task} references data {task_data}, registry pins "
            f"{dataset.openml_data}"
        )

    try:
        if offline:
            features, target = _offline_openml_data(task, prepared_cache)
        else:
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
    except Exception as exc:
        if os.environ.get("EVOLVE_QUALITY_DATA_OFFLINE") == "1":
            raise RuntimeError(
                f"{dataset.task_id}: offline OpenML cache load failed; "
                "unbound sklearn/split fallback is disabled"
            ) from exc
        return _load_sklearn_and_splits(dataset)


def _load_sklearn_and_splits(dataset: QualityDataset) -> dict:
    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(
        data_id=dataset.openml_data,
        as_frame=True,
        parser="auto",
        data_home=_explicit_cache("SKLEARN_DATA"),
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


def _official_indices(
    dataset: QualityDataset, *, n_rows: int
) -> tuple[list[int], list[int]]:
    cached = (
        cache_root()
        / f"task-{dataset.openml_task}-r{dataset.repeat}-f{dataset.fold}.json"
    )
    if cached.is_file():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        # Older caches were written with rowid shifted by one (row 0 duplicated,
        # last row dropped). Only trust complete files that record the 0-based origin.
        if (
            payload.get("rowid_base") == 0
            and isinstance(payload.get("train"), list)
            and isinstance(payload.get("test"), list)
        ):
            return list(payload["train"]), list(payload["test"])
    train_idx, test_idx = _download_split_indices(dataset, n_rows=n_rows)
    _atomic_cache_write(
        cached,
        json.dumps(
            {"train": train_idx, "test": test_idx, "rowid_base": 0},
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    return train_idx, test_idx


def _download_split_indices(
    dataset: QualityDataset, *, n_rows: int
) -> tuple[list[int], list[int]]:
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
            return (
                head,
                int(float(parts[1])),
                int(float(parts[2])),
                int(float(parts[3])),
            )
        if tail in {"TRAIN", "TEST"}:
            return (
                tail,
                int(float(parts[2])),
                int(float(parts[0])),
                int(float(parts[1])),
            )
    except ValueError:
        return None
    return None


def _offline_fedot_asset(relative: Path) -> Path:
    root = _explicit_cache("FEDOTLLM_REPO_CACHE")
    if root is None or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "FEDOTLLM_REPO_CACHE: explicit prepared data checkout required"
        )
    source = Path(root) / relative
    if not source.is_file() or any(
        part.is_symlink() for part in (source, *source.parents)
    ):
        raise FileNotFoundError(
            f"FEDOTLLM_REPO_CACHE: prepared regular data asset required: {source}"
        )
    return source


def _fedot_checkout_ts_csv(name: str) -> Path | None:
    rel = Path("examples") / "data" / "ts" / _FEDOT_TS_FILES[name]
    if os.environ.get("EVOLVE_QUALITY_DATA_OFFLINE") == "1":
        return _offline_fedot_asset(rel)
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
        if (
            not cached.is_file()
            or hashlib.sha256(cached.read_bytes()).hexdigest() != dataset.data_sha256
        ):
            url = _FEDOT_TS_RAW.format(name=name)
            with urllib.request.urlopen(url, timeout=60) as response:
                _atomic_cache_write(cached, response.read())
        source = cached
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != dataset.data_sha256:
        raise RuntimeError(
            f"{dataset.task_id}: FEDOT data checksum mismatch for {name}.csv: {digest}"
        )
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


def load_fedot_local_ts(dataset: QualityDataset) -> dict:
    """Load a checksum-pinned FEDOT real-case series with a chronological holdout."""

    import numpy as np
    import pandas as pd

    rel = Path("examples") / "real_cases" / "data" / dataset.data_path
    source = None
    if os.environ.get("EVOLVE_QUALITY_DATA_OFFLINE") == "1":
        source = _offline_fedot_asset(rel)
    else:
        for env_name in ("FEDOTLLM_REPO_PATH", "FEDOTLLM_REPO_CACHE"):
            raw = os.environ.get(env_name)
            if raw:
                candidate = (Path(raw) / rel).resolve()
                if candidate.is_file():
                    source = candidate
                    break
    if source is None:
        raise FileNotFoundError(
            f"{dataset.task_id}: FEDOT data asset not found: {dataset.data_path}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != dataset.data_sha256:
        raise RuntimeError(
            f"{dataset.task_id}: FEDOT data checksum mismatch for {dataset.data_path}: {digest}"
        )
    frame = pd.read_csv(source)
    if dataset.target not in frame.columns:
        raise RuntimeError(
            f"{dataset.task_id}: target column {dataset.target!r} is missing"
        )
    frame = _sort_local_ts_chronologically(frame, dataset.task_id)
    values = pd.to_numeric(frame[dataset.target], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    horizon = int(dataset.forecast_horizon)
    if horizon <= 0 or values.size <= horizon:
        raise RuntimeError(
            f"{dataset.task_id}: series has {values.size} valid values for horizon {horizon}"
        )
    train = values[:-horizon]
    test = values[-horizon:]
    return {
        "X_train": train,
        "y_train": train,
        "X_test": train,
        "y_test": test,
        "n_train": int(train.size),
        "n_test": int(test.size),
        "forecast_horizon": horizon,
        "loader": "fedot.examples.real_cases.data",
    }


def _sort_local_ts_chronologically(frame, task_id: str):
    """Sort by a recognized time column; preserve row order when none exists.

    FEDOT's metocean example encodes timestamps as ``YYYYMMDDH`` (the final
    one or two digits are the hour), so parsing numeric values as magnitudes
    would put each new day before the previous day's late hours.
    """

    import numpy as np
    import pandas as pd

    column = next(
        (
            name
            for name in frame.columns
            if str(name).strip().lower() in {"datetime", "timestamp", "date", "time"}
        ),
        None,
    )
    if column is None:
        return frame
    raw = frame[column]
    text = raw.astype(str).str.strip().str.removesuffix(".0")
    if bool(text.str.fullmatch(r"\d{9,10}").all()):
        day = pd.to_datetime(text.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        hour = pd.to_numeric(text.str.slice(8), errors="coerce")
        if day.notna().all() and hour.notna().all() and bool(hour.between(0, 23).all()):
            timestamps = day + pd.to_timedelta(hour, unit="h")
        else:
            timestamps = pd.to_datetime(raw, errors="coerce")
    else:
        timestamps = pd.to_datetime(raw, errors="coerce")
    if timestamps.isna().any():
        raise RuntimeError(
            f"{task_id}: cannot parse all timestamps in column {column!r}"
        )
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    return frame.iloc[order].reset_index(drop=True)

"""Dataset preparation and measurement must use the same pinned cache roots."""

import socket
import sys
from dataclasses import replace
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import urllib.request

import pandas as pd
import pytest

from fedotllm.agents.evolve.evaluation import openml_fold
from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset


def test_registered_openml_loader_binds_explicit_cache_before_loading(
    tmp_path, monkeypatch
):
    dataset = get_dataset("openml-10101-blood-transfusion")
    observed = []
    module = ModuleType("openml")
    module.config = SimpleNamespace(
        set_cache_directory=lambda path: observed.append(("cache", path))
    )
    module.config.get_cache_directory = lambda: str(tmp_path)
    task = SimpleNamespace(
        dataset_id=dataset.openml_data,
        get_X_and_y=lambda **kwargs: (pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1])),
        get_train_test_split_indices=lambda **kwargs: ([0], [1]),
    )
    module.tasks = SimpleNamespace(
        get_task=lambda value: observed.append(("task", value)) or task
    )
    monkeypatch.setitem(sys.modules, "openml", module)
    monkeypatch.setenv("EVOLVE_OPENML_CACHE", str(tmp_path))
    result = openml_fold.load_registered_fold(dataset)
    assert observed == [("cache", str(tmp_path)), ("task", dataset.openml_task)]
    assert (result["n_train"], result["n_test"]) == (1, 1)


def test_current_openml_root_setter_preserves_server_namespace(tmp_path, monkeypatch):
    dataset = get_dataset("openml-31-credit-g")
    namespace = tmp_path / "org/openml/www"
    namespace.mkdir(parents=True)
    observed = []
    module = ModuleType("openml")
    module.config = SimpleNamespace(
        set_root_cache_directory=lambda path: observed.append(path),
        get_cache_directory=lambda: str(namespace),
    )
    module.tasks = SimpleNamespace(
        get_task=lambda value: SimpleNamespace(
            dataset_id=dataset.openml_data,
            get_X_and_y=lambda **kwargs: (
                pd.DataFrame({"x": [1, 2]}),
                pd.Series([0, 1]),
            ),
            get_train_test_split_indices=lambda **kwargs: ([0], [1]),
        )
    )
    monkeypatch.setitem(sys.modules, "openml", module)
    monkeypatch.setenv("EVOLVE_OPENML_CACHE", str(tmp_path))
    assert openml_fold.load_registered_fold(dataset)["n_test"] == 1
    assert observed == [str(tmp_path)]


def test_sklearn_fallback_receives_explicit_cache(tmp_path, monkeypatch):
    import sklearn.datasets

    observed = []

    def fetch(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(
            data=pd.DataFrame({"x": [1, 2]}), target=pd.Series([0, 1])
        )

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", fetch)
    monkeypatch.setattr(
        openml_fold, "_official_indices", lambda *args, **kwargs: ([0], [1])
    )
    monkeypatch.setenv("SKLEARN_DATA", str(tmp_path))
    openml_fold._load_sklearn_and_splits(get_dataset("openml-31-credit-g"))
    assert observed[0]["data_home"] == str(tmp_path)


@pytest.mark.parametrize("api", ["urllib", "socket", "requests"])
def test_offline_fold_cannot_silently_fetch_missing_assets(monkeypatch, api):
    reached_network = []
    module = ModuleType("openml")

    def fetch(value):
        if api == "urllib":
            urllib.request.urlopen("https://example.invalid/missing")
        elif api == "socket":
            with socket.socket() as stream:
                stream.connect(("127.0.0.1", 9))
        else:
            import requests

            requests.get("https://example.invalid/missing", timeout=1)

    module.tasks = SimpleNamespace(get_task=fetch)
    monkeypatch.setitem(sys.modules, "openml", module)
    monkeypatch.setenv("EVOLVE_QUALITY_DATA_OFFLINE", "1")
    monkeypatch.delenv("EVOLVE_OPENML_CACHE", raising=False)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: reached_network.append("urllib"),
    )
    original_socket_connect = socket.socket.connect
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *args, **kwargs: reached_network.append("socket"),
    )
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *args, **kwargs: reached_network.append("dns")
    )
    bound_connect = socket.socket.connect
    with pytest.raises(RuntimeError, match="offline measurement"):
        with openml_fold._measurement_network_policy():
            fetch(None)
    assert reached_network == []
    assert socket.socket.connect is bound_connect
    assert bound_connect is not original_socket_connect


def test_cache_root_rejects_parent_symlink_before_openml_access(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("EVOLVE_OPENML_CACHE", str(link / "cache"))
    module = ModuleType("openml")
    module.tasks = SimpleNamespace(
        get_task=lambda value: pytest.fail("cache checked before loading")
    )
    module.config = SimpleNamespace(
        set_cache_directory=lambda path: pytest.fail("redirected cache")
    )
    monkeypatch.setitem(sys.modules, "openml", module)
    with pytest.raises(RuntimeError, match="could not verify OpenML"):
        openml_fold.load_registered_fold(get_dataset("openml-31-credit-g"))


def test_offline_openml_parse_failure_never_uses_unpinned_fallback(
    tmp_path, monkeypatch
):
    dataset = get_dataset("openml-31-credit-g")

    def broken(*args, **kwargs):
        raise ValueError("cached data cannot be parsed")

    namespace = tmp_path / "org/openml/www"
    namespace.mkdir(parents=True)
    module = ModuleType("openml")
    module.config = SimpleNamespace(
        set_root_cache_directory=lambda path: None,
        get_cache_directory=lambda: str(namespace),
    )
    module.tasks = SimpleNamespace(
        get_task=lambda value: SimpleNamespace(dataset_id=dataset.openml_data)
    )
    monkeypatch.setitem(sys.modules, "openml", module)
    monkeypatch.setenv("EVOLVE_QUALITY_DATA_OFFLINE", "1")
    monkeypatch.setenv("EVOLVE_OPENML_CACHE", str(tmp_path))
    monkeypatch.setattr(
        openml_fold, "_prepared_openml_cache", lambda *args: tmp_path / "checked.pkl"
    )
    monkeypatch.setattr(openml_fold, "_offline_openml_data", broken)
    monkeypatch.setattr(
        openml_fold,
        "_load_sklearn_and_splits",
        lambda dataset: {"loader": "unbound fallback"},
    )
    with pytest.raises(RuntimeError, match="offline.*fallback"):
        openml_fold.load_registered_fold(dataset)


def test_offline_loader_rejects_different_server_namespace(tmp_path, monkeypatch):
    namespace = tmp_path / "org/openml/test"
    namespace.mkdir(parents=True)
    module = ModuleType("openml")
    module.config = SimpleNamespace(
        set_root_cache_directory=lambda path: None,
        get_cache_directory=lambda: str(namespace),
    )
    module.tasks = SimpleNamespace(
        get_task=lambda value: pytest.fail("wrong server namespace")
    )
    monkeypatch.setitem(sys.modules, "openml", module)
    monkeypatch.setenv("EVOLVE_OPENML_CACHE", str(tmp_path))
    monkeypatch.setenv("EVOLVE_QUALITY_DATA_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="could not verify OpenML"):
        openml_fold.load_registered_fold(get_dataset("openml-31-credit-g"))


@pytest.mark.parametrize("family", ["public", "local"])
@pytest.mark.parametrize("offline", [True, False])
def test_ts_loader_uses_approved_data_checkout_only_when_offline(
    tmp_path, monkeypatch, family, offline
):
    dataset = get_dataset(
        "fedot-ts-beer" if family == "public" else "fedot-ts-temperature"
    )
    dataset = replace(dataset, forecast_horizon=1)
    relative = (
        Path("examples/data/ts/beer.csv")
        if family == "public"
        else Path("examples/real_cases/data") / dataset.data_path
    )
    payload = f"{'value' if family == 'public' else dataset.target}\n1\n2\n3\n".encode()
    approved, runtime = tmp_path / "approved", tmp_path / "runtime"
    for root in (approved, runtime):
        source = root / relative
        source.parent.mkdir(parents=True)
        source.write_bytes(payload)
    dataset = replace(dataset, data_sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv("FEDOTLLM_REPO_CACHE", str(approved))
    monkeypatch.setenv("FEDOTLLM_REPO_PATH", str(runtime))
    monkeypatch.setenv("EVOLVE_QUALITY_DATA_OFFLINE", "1" if offline else "0")
    read_paths = []
    real_read = pd.read_csv

    def read_csv(path, *args, **kwargs):
        read_paths.append(Path(path))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)
    result = openml_fold.load_registered_fold(dataset)
    assert result["y_test"].tolist() == [3.0]
    assert read_paths == [(approved if offline else runtime) / relative]


@pytest.mark.parametrize("family", ["public", "local"])
@pytest.mark.parametrize("bad_root", ["missing", "relative", "symlink"])
def test_offline_ts_loader_rejects_unapproved_data_root_before_runtime_fallback(
    tmp_path, monkeypatch, family, bad_root
):
    dataset = get_dataset(
        "fedot-ts-beer" if family == "public" else "fedot-ts-temperature"
    )
    relative = (
        Path("examples/data/ts/beer.csv")
        if family == "public"
        else Path("examples/real_cases/data") / dataset.data_path
    )
    payload = f"{'value' if family == 'public' else dataset.target}\n1\n2\n3\n".encode()
    runtime = tmp_path / "runtime"
    source = runtime / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    dataset = replace(
        dataset, forecast_horizon=1, data_sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setenv("FEDOTLLM_REPO_PATH", str(runtime))
    monkeypatch.setenv("EVOLVE_QUALITY_DATA_OFFLINE", "1")
    if bad_root == "missing":
        monkeypatch.delenv("FEDOTLLM_REPO_CACHE", raising=False)
    elif bad_root == "relative":
        monkeypatch.setenv("FEDOTLLM_REPO_CACHE", "unapproved-relative-root")
    else:
        link = tmp_path / "link"
        link.symlink_to(runtime, target_is_directory=True)
        monkeypatch.setenv("FEDOTLLM_REPO_CACHE", str(link))
    with pytest.raises((ValueError, FileNotFoundError), match="FEDOTLLM_REPO_CACHE"):
        openml_fold.load_registered_fold(dataset)

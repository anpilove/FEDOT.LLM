"""A prepared OpenML cache must remain read-only in concurrent ML workers."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from fedotllm.agents.evolve.evaluation import openml_fold as loader
from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset


def task_xml(dataset, *, target="target"):
    # Same namespace/nesting as OpenML 0.15.1's _create_task_from_xml.
    task_type = 1 if dataset.problem == "classification" else 2
    return f"""<oml:task xmlns:oml="http://openml.org/openml">
      <oml:task_id>{dataset.openml_task}</oml:task_id>
      <oml:task_type_id>{task_type}</oml:task_type_id>
      <oml:input name="source_data"><oml:data_set>
        <oml:data_set_id>{dataset.openml_data}</oml:data_set_id>
        <oml:target_feature>{target}</oml:target_feature>
      </oml:data_set></oml:input>
    </oml:task>"""


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "openml"
    namespace = root / "org/openml/www"
    dataset = get_dataset("openml-regression-airfoil-self-noise")
    directory = namespace / f"datasets/{dataset.openml_data}"
    directory.mkdir(parents=True)
    task_directory = namespace / f"tasks/{dataset.openml_task}"
    task_directory.mkdir(parents=True)
    (task_directory / "task.xml").write_text(task_xml(dataset))
    (task_directory / "datasplits.pkl.py3").write_bytes(
        pickle.dumps({"prepared": True})
    )
    data_path = directory / f"dataset_{dataset.openml_data}.pkl.py3"
    frame = pd.DataFrame(
        {
            "category": pd.Categorical(["a", "b", "a"]),
            "number": [1, 2, 3],
            "target": [10.0, 20.0, 30.0],
        }
    )
    payload = (frame, [True, False, False], list(frame.columns))
    data_path.write_bytes(pickle.dumps(payload))
    os.utime(data_path, ns=(1_000_000_000_000_000_000,) * 2)
    state = {"writes": [], "task_calls": 0, "get_dataset_kwargs": []}

    class Dataset:
        # OpenML 0.15.1 default metadata-only construction forgets this pointer
        # even when the pickle exists. The writer can reproduce identical bytes.
        cache_format = "pickle"
        data_pickle_file = None
        data_file = None
        parquet_file = None

        def _download_data(self):
            return None

        def _cache_compressed_file_from_file(self, *args):
            state["writes"].append(str(data_path))
            data_path.write_bytes(pickle.dumps(payload))

        def get_data(self, *, dataset_format, target):
            assert dataset_format == "dataframe" and target == state.get(
                "target_name", "target"
            )
            if state.get("force_writer"):
                getattr(self, state["force_writer"])(data_path)
            if self.data_pickle_file is None:
                self._cache_compressed_file_from_file(data_path)
            with Path(self.data_pickle_file or data_path).open("rb") as stream:
                values, categorical, names = pickle.load(stream)
            return (
                values.drop(columns=[target]),
                values[target],
                categorical[:-1],
                names[:-1],
            )

    class Task:
        dataset_id = dataset.openml_data
        target_name = "target"

        def get_dataset(self, **kwargs):
            state["get_dataset_kwargs"].append(kwargs)
            return Dataset()

        def get_X_and_y(self, *, dataset_format):
            values, target, *_ = self.get_dataset().get_data(
                dataset_format=dataset_format, target=self.target_name
            )
            return values, target

        def get_train_test_split_indices(self, **kwargs):
            return [0, 1], [2]

    module = ModuleType("openml")
    module.config = SimpleNamespace(
        set_root_cache_directory=lambda path: None,
        get_cache_directory=lambda: str(namespace),
    )

    def get_task(task_id):
        state["task_calls"] += 1
        return Task()

    module.tasks = SimpleNamespace(get_task=get_task)
    monkeypatch.setitem(sys.modules, "openml", module)
    monkeypatch.setenv("EVOLVE_OPENML_CACHE", str(root))
    monkeypatch.setenv("EVOLVE_QUALITY_DATA_OFFLINE", "1")
    return dataset, data_path, payload, state, module, namespace


def snapshot(path):
    return path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepared_offline_data_never_rewrites_identical_pickle(cache):
    dataset, path, _, state, _, _ = cache
    before = snapshot(path)
    result = loader.load_registered_fold(dataset)
    assert snapshot(path) == before
    assert state["writes"] == []
    assert state["get_dataset_kwargs"] == [{"download_data": False}]
    assert result["loader"] == "openml.tasks.get_task"
    assert result["X_train"]["category"].dtype.name == "category"
    assert result["y_train"].tolist() == [10.0, 20.0]


def test_concurrent_fold_reads_preserve_shared_cache_mtime_and_hash(cache):
    dataset, path, _, state, _, _ = cache
    before = snapshot(path)
    # Each real worker has its own Python process. Test the data path in threads
    # directly to avoid nesting the worker-only global network-API scope.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: loader._load_registered_fold(dataset), range(8))
        )
    assert all(row["y_test"].tolist() == [30.0] for row in results)
    assert snapshot(path) == before
    assert state["writes"] == []


@pytest.mark.parametrize("kind", ["missing", "corrupt", "legacy-array"])
def test_unprepared_data_fails_before_openml_can_create_or_replace_it(cache, kind):
    dataset, path, payload, state, _, _ = cache
    if kind == "missing":
        path.unlink()
        before = None
    else:
        if kind == "corrupt":
            path.write_bytes(b"not a pickle")
        else:
            path.write_bytes(
                pickle.dumps((payload[0].to_numpy(), payload[1], payload[2]))
            )
        before = path.read_bytes()
    with pytest.raises(RuntimeError, match="OpenML"):
        loader.load_registered_fold(dataset)
    assert state["writes"] == []
    assert path.read_bytes() == before if before is not None else not path.exists()


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_classifier_feature_pickle_is_checked_before_get_task(cache, monkeypatch, kind):
    _, _, _, state, module, namespace = cache
    dataset = get_dataset("openml-31-credit-g")
    directory = namespace / f"datasets/{dataset.openml_data}"
    directory.mkdir()
    tasks = namespace / f"tasks/{dataset.openml_task}"
    tasks.mkdir()
    (tasks / "task.xml").write_text(task_xml(dataset))
    (directory / "features.xml").write_text("<features/>")
    feature_path = directory / "features.xml.pkl"
    if kind == "corrupt":
        feature_path.write_bytes(b"broken")

    def forbidden_task(task_id):
        state["task_calls"] += 1
        feature_path.write_bytes(pickle.dumps({"rewritten": True}))
        raise ValueError("upstream would write feature cache before loading data")

    module.tasks.get_task = forbidden_task
    before = feature_path.read_bytes() if feature_path.exists() else None
    with pytest.raises(RuntimeError, match="OpenML"):
        loader.load_registered_fold(dataset)
    assert state["task_calls"] == 0
    assert (
        feature_path.read_bytes() == before
        if before is not None
        else not feature_path.exists()
    )


@pytest.mark.parametrize(
    "method", ["_download_data", "_cache_compressed_file_from_file"]
)
def test_instance_writer_paths_fail_closed_without_touching_shared_cache(cache, method):
    dataset, path, _, state, _, _ = cache
    before = snapshot(path)
    state["force_writer"] = method
    with pytest.raises(RuntimeError, match="offline OpenML"):
        loader.load_registered_fold(dataset)
    assert state["writes"] == []
    assert snapshot(path) == before


def test_current_sparse_dataset_cache_is_supported(cache):
    from scipy import sparse

    dataset, path, payload, _, _, namespace = cache
    path.write_bytes(
        pickle.dumps(
            (
                sparse.csr_matrix([[1, 2, 3], [4, 5, 6]]),
                [False, False, False],
                payload[2],
            )
        )
    )
    before = snapshot(path)
    assert loader._prepared_openml_cache(dataset, namespace) == path
    assert snapshot(path) == before


def test_symlink_data_cache_is_rejected_before_upstream_access(cache):
    dataset, path, _, state, _, _ = cache
    original = path.with_suffix(".original")
    path.rename(original)
    path.symlink_to(original)
    before = snapshot(original)
    with pytest.raises(RuntimeError, match="OpenML"):
        loader.load_registered_fold(dataset)
    assert state["task_calls"] == 0 and state["writes"] == []
    assert snapshot(original) == before


@pytest.mark.parametrize("target_type", ["nominal", "string"])
def test_only_string_target_is_rejected_before_class_label_resolution(
    cache, target_type
):
    _, _, payload, state, module, namespace = cache
    dataset = get_dataset("openml-31-credit-g")
    directory = namespace / "datasets/31"
    directory.mkdir()
    (directory / "features.xml").write_text("<features/>")
    features = {
        0: SimpleNamespace(name="other_text", data_type="string"),
        1: SimpleNamespace(name="target", data_type=target_type),
    }
    (directory / "features.xml.pkl").write_bytes(pickle.dumps(features))
    data_path = directory / "dataset_31.pkl.py3"
    data_path.write_bytes(pickle.dumps(payload))
    tasks = namespace / "tasks/31"
    tasks.mkdir()
    (tasks / "task.xml").write_text(task_xml(dataset))
    (tasks / "datasplits.pkl.py3").write_bytes(pickle.dumps({"prepared": True}))
    # Reuse the fixture's native-equivalent instance reader, changing task ID.
    original = module.tasks.get_task

    def get_task(task_id):
        task = original(task_id)
        task.dataset_id = 31
        return task

    module.tasks.get_task = get_task
    before = snapshot(data_path)
    if target_type == "string":
        with pytest.raises(RuntimeError, match="OpenML"):
            loader.load_registered_fold(dataset)
        assert state["task_calls"] == 0
    else:
        assert loader.load_registered_fold(dataset)["y_test"].tolist() == [30.0]
        assert state["task_calls"] == 1
    assert snapshot(data_path) == before and state["writes"] == []


def test_exact_public_task31_xml_matches_pinned_server_cache_and_loads_readonly(cache):
    _, _, payload, state, module, namespace = cache
    dataset = get_dataset("openml-31-credit-g")
    source = Path(__file__).parents[2] / "fixtures/openml/task_31.xml"
    raw = source.read_bytes()
    # This is also the exact actual v5 manifest task-metadata SHA.
    assert (
        hashlib.sha256(raw).hexdigest()
        == "151559e7e278221b2ae343457a8c1551dfcde2c90632945bf8f0f0167d40667d"
    )
    task_directory = namespace / "tasks/31"
    task_directory.mkdir()
    (task_directory / "task.xml").write_bytes(raw)
    (task_directory / "datasplits.pkl.py3").write_bytes(
        pickle.dumps({"prepared": True})
    )
    directory = namespace / "datasets/31"
    directory.mkdir()
    (directory / "features.xml").write_text("<features/>")
    (directory / "features.xml.pkl").write_bytes(
        pickle.dumps({0: SimpleNamespace(name="class", data_type="nominal")})
    )
    path = directory / "dataset_31.pkl.py3"
    values = payload[0].rename(columns={"target": "class"})
    path.write_bytes(pickle.dumps((values, payload[1], list(values.columns))))
    state["target_name"] = "class"
    original = module.tasks.get_task

    def get_task(task_id):
        task = original(task_id)
        task.dataset_id = 31
        task.target_name = "class"
        return task

    module.tasks.get_task = get_task
    before = snapshot(path)
    result = loader.load_registered_fold(dataset)
    assert result["y_test"].tolist() == [30.0]
    assert state["task_calls"] == 1 and state["writes"] == []
    assert snapshot(path) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "task-id",
        "data-id",
        "task-type",
        "duplicate-source",
        "duplicate-target",
        "missing-target",
        "wrong-namespace",
    ],
)
def test_cached_task_mapping_is_rejected_before_upstream_access(cache, mutation):
    dataset, path, _, state, _, namespace = cache
    task_path = namespace / f"tasks/{dataset.openml_task}/task.xml"
    tree = ET.fromstring(task_xml(dataset))
    ns = "{http://openml.org/openml}"
    source = tree.find(f"{ns}input")
    data = source.find(f"{ns}data_set")
    if mutation == "task-id":
        tree.find(f"{ns}task_id").text = str(dataset.openml_task + 1)
    elif mutation == "data-id":
        data.find(f"{ns}data_set_id").text = str(dataset.openml_data + 1)
    elif mutation == "task-type":
        tree.find(f"{ns}task_type_id").text = "1"
    elif mutation == "duplicate-source":
        tree.append(ET.fromstring(ET.tostring(source)))
    elif mutation == "duplicate-target":
        ET.SubElement(data, ns + "target_feature").text = "other"
    elif mutation == "missing-target":
        data.remove(data.find(f"{ns}target_feature"))
    else:
        tree.tag = "{https://unapproved.invalid}task"
    task_path.write_bytes(ET.tostring(tree))
    before = snapshot(path)
    with pytest.raises(RuntimeError, match="OpenML"):
        loader.load_registered_fold(dataset)
    assert state["task_calls"] == 0 and state["writes"] == []
    assert snapshot(path) == before

"""Protocol boundaries: fresh physical identity and explicit tuning policy."""

import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fedotllm.agents.evolve.evaluation import _fedot_quality_worker as worker
from fedotllm.agents.evolve.evaluation import quality_registry as registry_module


@pytest.mark.parametrize("problem", ["classification", "regression", "ts_forecasting"])
def test_registered_quality_jobs_require_process_bounded_tuning(problem):
    registry = registry_module.load_quality_registry()
    dataset = next(item for item in registry.datasets if item.problem == problem)

    spec = registry_module.job_spec(dataset, registry=registry)

    assert spec.get("tuning_evaluation_policy") == "isolated-objective-process-v1"


def test_unknown_tuning_policy_cannot_silently_fall_back_to_threads(monkeypatch):
    raw = registry_module._load_raw()
    raw["tuning_evaluation_policy"] = "unbounded-unknown-policy"
    monkeypatch.setattr(registry_module, "_load_raw", lambda: raw)

    with pytest.raises(ValueError, match="tuning"):
        registry_module.load_quality_registry()


def test_explicit_frozen_registries_do_not_change_default_or_each_other(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    original = registry_module.load_quality_registry()
    paths = []
    for name in ("campaign-one", "campaign-two"):
        raw = registry_module._load_raw()
        raw["identity"] = name
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(raw))
        paths.append(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        loaded = list(pool.map(registry_module.load_quality_registry, paths))

    assert [item.identity for item in loaded] == ["campaign-one", "campaign-two"]
    assert registry_module.load_quality_registry().identity == original.identity


def test_failed_worker_echoes_physical_invocation_in_payload_and_stage(tmp_path):
    out = tmp_path / "result.json"
    invocation = "0123456789abcdef0123456789abcdef"
    spec = {
        "worker_invocation_id": invocation,
        "attempt_id": "proposal:dataset:patch:1",
        "source_dataset": "dataset",
        "side": "patch",
        "attempt": 1,
        "seed": 42,
        "composer_options": {"unsupported_option": 1},
    }

    payload, _ = worker._evaluate_worker(spec, out)

    assert payload["status"] == "crash"
    assert payload["failed_stage"] == "validate_spec"
    assert payload.get("worker_invocation_id") == invocation
    stage = json.loads(out.with_suffix(".stages.json").read_text())
    assert stage.get("worker_invocation_id") == invocation


def test_worker_rejects_unknown_tuning_policy_before_runtime_configuration(tmp_path, monkeypatch):
    def unexpected_configuration(out):
        raise AssertionError("invalid protocol reached runtime configuration")

    monkeypatch.setattr(worker, "configure_worker_runtime", unexpected_configuration)
    payload, _ = worker._evaluate_worker(
        {"source_dataset": "dataset", "seed": 42,
         "tuning_evaluation_policy": "unknown"},
        tmp_path / "result.json",
    )

    assert payload["status"] == "crash"
    assert payload["failed_stage"] == "validate_spec"
    assert "tuning" in payload["detail"]


@pytest.mark.parametrize("fail_fit", [False, True])
def test_worker_owns_tuning_guard_for_fit_and_preserves_its_diagnostics(
    tmp_path, monkeypatch, fail_fit,
):
    from fedotllm.agents.evolve.evaluation import tuning_isolation

    state = {"active": False, "cleaned": False}

    class Guard:
        def __enter__(self):
            state["active"] = True
            return self

        def __exit__(self, *args):
            state.update(active=False, cleaned=True)

        def diagnostics(self):
            return {"policy": "isolated-objective-process-v1",
                    "cleanup_verified": state["cleaned"], "evaluations": 2}

    class Model:
        def __init__(self, **kwargs):
            self.api_composer = SimpleNamespace(was_optimised=True, was_tuned=True)
            self.history = None

        def fit(self, **kwargs):
            assert state["active"], "tuning evaluation was not process guarded"
            if fail_fit:
                raise ValueError("controlled fit failure")
            logging.info("Pipeline composition started")

        def predict(self, values):
            self.prediction = values

        def get_metrics(self, **kwargs):
            return {"rmse": 0.125}

    module = ModuleType("fedot.api.main")
    module.Fedot = Model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(tuning_isolation, "install_tuning_isolation", lambda *args: Guard())
    monkeypatch.setattr(worker, "configure_worker_runtime", lambda out: {"temp_dir": str(tmp_path)})
    monkeypatch.setattr(worker, "cache_policy_parameters", lambda *args: ({}, ()))
    dataset = SimpleNamespace(task_id="fixture", problem="regression", metric="rmse")
    fold = {"X_train": [1], "y_train": [1], "X_test": [2], "y_test": [2],
            "n_train": 1, "n_test": 1, "loader": "fixture"}
    monkeypatch.setattr(worker, "load_worker_fold", lambda spec: (dataset, fold))
    spec = {"source_dataset": "fixture", "timeout_seconds": 3600,
            "fedot_search_budget_seconds": 2700, "timeout_minutes": 45,
            "fedot_timeout_policy": "return_best", "final_fit_reserve_seconds": 300,
            "max_pipeline_evaluation_policy": "per-cv-candidate-v1",
            "seed": 42, "n_jobs": 4, "preset": "best_quality", "with_tuning": True,
            "tuning_evaluation_policy": "isolated-objective-process-v1"}

    payload, _ = worker._evaluate_worker(spec, tmp_path / "result.json")

    assert payload["status"] == ("crash" if fail_fit else "ok"), payload.get("detail")
    diagnostics = payload["metric_observations"].get("tuning_isolation", {})
    assert diagnostics.get("policy") == "isolated-objective-process-v1"
    assert diagnostics.get("cleanup_verified") is True
    assert diagnostics.get("evaluations") == 2
    assert state == {"active": False, "cleaned": True}

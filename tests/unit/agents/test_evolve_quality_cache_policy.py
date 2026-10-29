"""Worker protocol checks without training or external dataset access."""

import json
import logging
import sys
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from fedotllm.agents.evolve.evaluation import _fedot_quality_worker as worker
from fedotllm.agents.evolve.evaluation import quality_registry as registry_module
from fedotllm.agents.evolve.evaluation.fedot_quality import stock_cache_identity
from fedotllm.agents.evolve.evaluation.smoke_data import smoke_registry


def run_fixture(
    tmp_path,
    monkeypatch,
    *,
    variant="operations",
    problem="regression",
    policy="disabled-sqlite-v1",
    enabled_at=None,
    missing_attribute=False,
):
    """Simulate the two public FEDOT cache APIs and their disk side effects."""
    enum = Enum("TaskTypesEnum", ["classification", "regression", "ts_forecasting"])
    flag_attrs = (
        {
            "use_operations_cache": "operations_cache",
            "use_preprocessing_cache": "preprocessing_cache",
            "use_predictions_cache": "predictions_cache",
        }
        if variant == "operations"
        else {
            "use_pipelines_cache": "pipelines_cache",
            "use_preprocessing_cache": "preprocessing_cache",
        }
    )

    class ParamsRepository:
        @staticmethod
        def default_params_for_task(task):
            assert task in enum
            return {key: True for key in flag_attrs}

    class FixtureFedot:
        def __init__(self, **kwargs):
            self.cache_dir = Path(kwargs["cache_dir"])
            self.params = {key: kwargs.get(key, True) for key in flag_attrs}
            self.api_composer = SimpleNamespace(was_optimised=True, was_tuned=False)
            self.history = None
            for flag, attr in flag_attrs.items():
                active = self.params[flag] or enabled_at == "constructor"
                setattr(self.api_composer, attr, object() if active else None)
                if active:
                    (self.cache_dir / f"{attr}.sqlite").write_bytes(b"fixture cache")
            if missing_attribute:
                delattr(self.api_composer, next(iter(flag_attrs.values())))

        def fit(self, **kwargs):
            logging.info("Pipeline composition started")
            if enabled_at == "fit":
                setattr(self.api_composer, next(iter(flag_attrs.values())), object())
                (self.cache_dir / "unexpected.sqlite").write_bytes(b"fixture cache")

        def predict(self, values):
            self.prediction = values

        predict_proba = predict

        def forecast(self):
            self.prediction = [1]

        def get_metrics(self, **kwargs):
            return {"rmse": 0.125, "roc_auc": 0.8, "mae": 0.2}

    for module_name, attrs in (
        ("fedot.api.main", {"Fedot": FixtureFedot}),
        (
            "fedot.api.api_utils.api_params_repository",
            {"ApiParamsRepository": ParamsRepository},
        ),
        (
            "fedot.core.repository.tasks",
            {"TaskTypesEnum": enum, "TsForecastingParams": lambda **kwargs: kwargs},
        ),
    ):
        module = ModuleType(module_name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module_name, module)
    metric = {
        "classification": "roc_auc",
        "regression": "rmse",
        "ts_forecasting": "mae",
    }[problem]
    dataset = SimpleNamespace(
        task_id="fixture", problem=problem, metric=metric, forecast_horizon=1
    )
    fold = {
        "X_train": [1],
        "y_train": [1],
        "X_test": [1],
        "y_test": [1],
        "n_train": 1,
        "n_test": 1,
        "loader": "fixture",
    }
    monkeypatch.setattr(worker, "load_worker_fold", lambda spec: (dataset, fold))
    monkeypatch.setattr(
        worker, "configure_worker_runtime", lambda out: {"temp_dir": str(tmp_path)}
    )
    spec = tmp_path / "spec.json"
    cv_folds = 3 if problem == "ts_forecasting" else 5
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "cv_folds": cv_folds,
                "max_pipeline_fit_time_seconds": 2700 // cv_folds,
                "seed": 42,
                "n_jobs": 4,
                "preset": "best_quality",
                "with_tuning": False,
                "fedot_cache_policy": policy,
            }
        )
    )
    out = tmp_path / "payload.json"
    root_log = logging.getLogger()
    before_handlers, before_level = list(root_log.handlers), root_log.level
    try:
        rc = worker.main(["--spec-json", str(spec), "--out", str(out)])
    finally:
        root_log.handlers[:] = before_handlers
        root_log.setLevel(before_level)
    return (
        rc,
        json.loads(out.read_text()),
        list(out.with_suffix(".fedot-cache").glob("*")),
    )


@pytest.mark.parametrize("problem", ["classification", "regression", "ts_forecasting"])
@pytest.mark.parametrize("variant", ["operations", "pipelines"])
def test_disabled_policy_creates_no_sqlite_cache_on_supported_task_and_runtime(
    tmp_path, monkeypatch, variant, problem
):
    # Missing any public disable flag creates a database and breaks the new
    # protocol, even if fit and the final metric otherwise succeed.
    rc, payload, caches = run_fixture(
        tmp_path, monkeypatch, variant=variant, problem=problem
    )
    assert rc == 0, payload.get("detail")
    assert payload["status"] == "ok"
    assert caches == []
    observations = payload["metric_observations"]
    assert observations["fedot_cache_policy"] == "disabled-sqlite-v1"
    assert observations["fedot_cache_flags"] == (
        {
            "use_operations_cache": False,
            "use_preprocessing_cache": False,
            "use_predictions_cache": False,
        }
        if variant == "operations"
        else {"use_pipelines_cache": False, "use_preprocessing_cache": False}
    )


@pytest.mark.parametrize("enabled_at", ["constructor", "fit"])
def test_disabled_policy_rejects_runtime_that_keeps_or_reenables_a_cache(
    tmp_path, monkeypatch, enabled_at
):
    # Accepting a runtime that ignores its flags would report valid metrics
    # while retaining the storage growth that this protocol promises to avoid.
    rc, payload, _ = run_fixture(tmp_path, monkeypatch, enabled_at=enabled_at)
    assert rc == 2
    assert payload["status"] == "crash"
    assert "cache policy" in payload["detail"].lower()
    assert "active" in payload["detail"].lower()
    assert ("before fit" if enabled_at == "constructor" else "after fit") in payload[
        "detail"
    ].lower()
    assert payload["score"] is None
    assert payload["metric_observations"]["metric_outcome"] == "not_measured"
    assert payload["metric_observations"]["metric_valid"] is False


def test_disabled_policy_rejects_missing_expected_composer_attribute(
    tmp_path, monkeypatch
):
    # Treating a missing cache attribute as None falsely accepts an unknown API.
    rc, payload, _ = run_fixture(tmp_path, monkeypatch, missing_attribute=True)
    assert rc == 2
    assert "cache policy" in payload["detail"].lower()
    assert "missing" in payload["detail"].lower()


def test_explicit_legacy_policy_preserves_its_cache_behavior(tmp_path, monkeypatch):
    rc, payload, caches = run_fixture(
        tmp_path, monkeypatch, policy="isolated-worker-v1"
    )
    assert rc == 0, payload.get("detail")
    assert payload["metric_observations"]["fedot_cache_policy"] == "isolated-worker-v1"
    assert len(caches) == 3


def test_new_production_and_smoke_protocols_cannot_reuse_legacy_stock_metrics():
    # Updating the worker while leaving its registry frozen would incorrectly
    # accept a baseline measured under the previous cache/search protocol.
    for registry in (registry_module.load_quality_registry(), smoke_registry()):
        assert registry.fedot_cache_policy == "disabled-sqlite-v1"
        legacy = replace(registry, fedot_cache_policy="isolated-worker-v1")
        current_key = stock_cache_identity(
            registry.datasets[0], registry, worker_runtime="fixture"
        )
        previous_key = stock_cache_identity(
            registry.datasets[0], legacy, worker_runtime="fixture"
        )
        assert current_key != previous_key


def test_registry_loader_accepts_explicit_legacy_policy(monkeypatch):
    raw = registry_module._load_raw()
    raw["fedot_cache_policy"] = "isolated-worker-v1"
    monkeypatch.setattr(registry_module, "_load_raw", lambda: raw)
    assert (
        registry_module.load_quality_registry().fedot_cache_policy
        == "isolated-worker-v1"
    )

import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

from fedotllm.agents.evolve.commands.reliability_report import render_reliability_report
from fedotllm.agents.evolve.evaluation import _fedot_quality_worker as worker
from fedotllm.agents.evolve.evaluation.return_best_policy import (
    signature_compatibility,
    worker_action,
)


class _ReturnBestFedot:
    def __init__(
        self, timeout_policy, final_fit_reserve_seconds, cv_folds,
        max_pipeline_fit_time, max_pipeline_evaluation_time, **kwargs,
    ):
        del timeout_policy, final_fit_reserve_seconds, cv_folds
        del max_pipeline_fit_time, max_pipeline_evaluation_time, kwargs


class _StockFedot:
    def __init__(self, problem, timeout=None, n_jobs=-1, **composer_tuner_params):
        raise AssertionError("stock pin must not be constructed")


def test_installed_fedot_signature_is_not_return_best():
    from fedot.api.main import Fedot

    report = signature_compatibility(Fedot)

    assert report["compatible"] is False
    assert report["action"] == "invalid_before_fit"
    assert "timeout_policy" in report["missing"]
    assert "timeout_policy" not in inspect.signature(Fedot.__init__).parameters


def test_explicit_parameters_are_return_best_and_legacy_does_not_claim_them():
    assert signature_compatibility(_ReturnBestFedot)["action"] == "pass_worker_kwargs"
    legacy = worker_action("legacy", _StockFedot)
    assert legacy["action"] == "legacy_kwargs_unchanged"
    assert legacy["compatible"] is False


def test_worker_rejects_return_best_pin_before_fit(tmp_path: Path, monkeypatch):
    called = {"fit": False}

    class Stock:
        def __init__(self, problem, timeout=None, n_jobs=-1, **composer_tuner_params):
            called["fit"] = True
            raise AssertionError(composer_tuner_params)

    module = ModuleType("fedot.api.main")
    module.Fedot = Stock
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "source_dataset": "fixture",
        "fedot_timeout_policy": "return_best",
        "seed": 1,
    }))
    out = tmp_path / "payload.json"

    rc = worker.main(["--spec-json", str(spec), "--out", str(out)])
    payload = json.loads(out.read_text())

    assert rc == 2
    assert called["fit"] is False
    assert payload["status"] == "invalid"
    assert payload["outcome"] == "invalid"
    assert payload["failure_kind"] == "return_best_incompatible"
    assert payload["score"] is None
    assert "timeout_policy" in payload["detail"]


def test_reliability_report_requires_measured_evidence():
    evidence = {
        "doctor": "preflight return_best/dask/path",
        "worker_invalid": "exceptions publish invalid",
        "checkout_contract": "disjoint checkout",
        "return_best_policy": "legacy_reject before fit",
        "dev_ne_final": "disjoint splits",
        "budget": "search budget below the side deadline",
        "recovery": "unfinished attempts stay crash",
        "smoke_aa": {
            "purpose": "smoke_aa_measurement_noise",
            "quality_claim": False,
            "values": [12, 12],
            "abs_delta": 0,
            "identical": True,
        },
    }

    text = render_reliability_report(evidence)

    assert "quality KEEP" in text
    assert "abs_delta: 0" in text
    assert "quality_claim: False" in text

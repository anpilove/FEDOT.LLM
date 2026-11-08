import numpy as np

from fedotllm.agents.evolve.controller.confirmation import _deterministic_patch_failure
from fedotllm.agents.evolve.controller.feedback import (
    _blocking_lift_crash_ids,
    _dev_feedback,
)
from fedotllm.agents.evolve.controller.metric_study import judge_pair
from fedotllm.agents.evolve.evaluation.independent_data import PROTOCOL
from fedotllm.agents.evolve.types import Decision, ScoreResult


def _row(task_id: str, status: str, kind: str | None = None, **kwargs) -> ScoreResult:
    observations = {} if kind is None else {"failure_kind": kind}
    return ScoreResult(task_id, status, kwargs.pop("score", float("nan")), metric_observations=observations, **kwargs)


def _evidence() -> dict:
    return {"protocol": PROTOCOL, "validation_ids": ["train:1"]}


def test_confirmation_does_not_retry_a_candidate_invalid():
    stock = {"rf": _row("rf", "ok", score=0.8)}
    patched = {"rf": _row("rf", "invalid", "fedot_import_error")}

    reason, infrastructure = _deterministic_patch_failure(stock, patched, ("rf",))

    assert infrastructure is False
    assert reason == "regression rf crashed after patch"


def test_confirmation_keeps_a_pin_mismatch_as_infrastructure():
    stock = {"rf": _row("rf", "ok", score=0.8)}
    patched = {"rf": _row("rf", "invalid", "return_best_incompatible")}

    reason, infrastructure = _deterministic_patch_failure(stock, patched, ("rf",))

    assert infrastructure is True
    assert reason == "regression rf timeout_or_invalid"


def test_judge_pair_rejects_candidate_invalid_as_a_regression():
    before = _row("target", "ok", score=0.7, data_evidence=_evidence())
    after = _row("target", "invalid", "fedot_fit_error", data_evidence=_evidence())

    passed, result = judge_pair({"target": before}, {"target": after}, "target")

    assert passed is False
    assert result["reason"] == "regression:target:crash"


def test_feedback_reports_an_invalid_failure_path_change():
    stock = {
        "rf": _row(
            "rf", "invalid", "fedot_fit_error", detail="old",
            traceback='File "/tmp/fedot/a.py", line 3, in fit',
        )
    }
    patched = {
        "rf": _row(
            "rf", "invalid", "fedot_fit_error", detail="new",
            traceback='File "/tmp/fedot/b.py", line 4, in predict',
        )
    }

    text = _dev_feedback(
        Decision(False, "regression rf crashed after patch", None),
        patched,
        stock=stock,
    )

    assert "Changed failure path for rf" in text
    assert "fedot/b.py:predict" in text


def test_frozen_splits_record_dev_and_final_without_a_keep():
    from fedotllm.agents.evolve.evaluation.independent_data import frozen_split_outcomes

    report = frozen_split_outcomes(np.tile([0, 1], 40), problem="classification")

    assert report["keep"] is None
    assert report["quality_claim"] is False
    assert report["disjoint"] is True
    assert report["dev"]["outcome"] == "reserved"
    assert report["final"]["outcome"] == "reserved"
    assert report["dev"]["count"] > 0
    assert report["final"]["count"] > 0
    assert report["dev"]["quality_claim"] is False
    assert report["final"]["quality_claim"] is False


def test_blocking_lifts_include_candidate_invalid_when_none_are_healthy():
    stock = {
        "broken": _row("broken", "invalid", "worker_error"),
        "pin": _row("pin", "invalid", "return_best_incompatible"),
    }

    assert _blocking_lift_crash_ids(stock, ("broken", "pin")) == ("broken",)
    assert _blocking_lift_crash_ids(
        {"healthy": _row("healthy", "ok", score=0.8), "broken": stock["broken"]},
        ("healthy", "broken"),
    ) == ()

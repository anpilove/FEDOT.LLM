from fedotllm.agents.evolve.evaluation.affected_eval import _normalized_delta
from fedotllm.agents.evolve.evaluation.compare import compare_pack, patched_broke
from fedotllm.agents.evolve.types import ScoreResult


def _row(task_id: str, status: str, score: float, kind: str | None = None) -> ScoreResult:
    observations = {} if kind is None else {"failure_kind": kind}
    return ScoreResult(task_id, status, score, metric_observations=observations)


def test_candidate_invalid_on_a_measured_stock_is_a_broken_patch():
    stock = _row("catboost", "ok", 0.8)
    patched = _row("catboost", "invalid", float("nan"), "fedot_import_error")

    decision = compare_pack(
        {"catboost": stock},
        {"catboost": patched},
        lift_ids=("catboost",),
        protect_ids=("catboost",),
    )

    assert patched_broke(stock, patched)
    assert decision.keep is False
    assert decision.infrastructure_error is False
    assert "crashed after patch" in decision.reason


def test_return_best_mismatch_stays_infrastructure():
    stock = _row("catboost", "ok", 0.8)
    patched = _row("catboost", "invalid", float("nan"), "return_best_incompatible")

    decision = compare_pack(
        {"catboost": stock},
        {"catboost": patched},
        lift_ids=("catboost",),
        protect_ids=("catboost",),
    )

    assert patched_broke(stock, patched) is False
    assert decision.infrastructure_error is True
    assert decision.reason == "timeout_or_invalid"


def test_protect_invalid_fit_is_a_regression_not_infrastructure():
    stock = {
        "catboost": _row("catboost", "ok", 0.80),
        "rf": _row("rf", "ok", 0.70),
    }
    patched = {
        "catboost": _row("catboost", "ok", 0.84),
        "rf": _row("rf", "invalid", float("nan"), "fedot_fit_error"),
    }

    decision = compare_pack(
        stock, patched, lift_ids=("catboost",), protect_ids=("catboost", "rf"),
    )

    assert decision.keep is False
    assert decision.infrastructure_error is False
    assert decision.reason == "regression rf crashed after patch"


def test_one_invalid_lift_does_not_keep_the_rest_of_the_pack():
    stock = {
        "catboost": _row("catboost", "ok", 0.80),
        "rf": _row("rf", "ok", 0.70),
    }
    crashed = {
        "catboost": _row("catboost", "ok", 0.90),
        "rf": _row("rf", "invalid", float("nan"), "worker_error"),
    }
    unanswered = {
        "catboost": _row("catboost", "ok", 0.90),
        "rf": _row("rf", "invalid", float("nan"), "memory_error"),
    }

    crash = compare_pack(stock, crashed, lift_ids=("catboost", "rf"), protect_ids=("catboost",))
    missing = compare_pack(stock, unanswered, lift_ids=("catboost", "rf"), protect_ids=("catboost",))

    assert crash.keep is False
    assert crash.infrastructure_error is False
    assert "rf" in crash.reason
    assert missing.keep is False
    assert missing.infrastructure_error is True
    assert missing.reason == "timeout_or_invalid"


def test_affected_metric_treats_candidate_invalid_as_regressed():
    label, delta = _normalized_delta(
        "rf",
        _row("rf", "ok", 0.7),
        _row("rf", "invalid", float("nan"), "scoring_error"),
    )

    assert label == "regressed"
    assert delta is None

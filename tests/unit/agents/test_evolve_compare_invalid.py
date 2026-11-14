import pytest

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


def test_shared_protect_crash_does_not_block_a_neutral_patch():
    stock = {
        "catboost": _row("catboost", "ok", 0.80),
        "pca->catboost": ScoreResult(
            "pca->catboost", "crash", 0.5, detail="IndexError: index 1 is out of bounds"
        ),
    }
    patched = {
        "catboost": _row("catboost", "ok", 0.80),
        "pca->catboost": ScoreResult(
            "pca->catboost", "crash", 0.5, detail="IndexError: index 1 is out of bounds"
        ),
    }

    decision = compare_pack(
        stock,
        patched,
        lift_ids=("catboost",),
        protect_ids=("catboost", "pca->catboost"),
    )

    assert decision.infrastructure_error is False
    assert "timeout_or_invalid" not in decision.reason
    assert decision.reason.startswith("target_delta")


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


def test_evidence_signal_does_not_keep_when_another_lift_is_invalid():
    stock = {
        "cholesterol": _row("cholesterol", "ok", 50.0),
        "rf": _row("rf", "ok", 0.70),
    }
    patched = {
        "cholesterol": _row("cholesterol", "ok", 49.8),
        "rf": _row("rf", "invalid", float("nan"), "memory_error"),
    }

    decision = compare_pack(
        stock,
        patched,
        lift_ids=("cholesterol", "rf"),
        protect_ids=("cholesterol", "rf"),
        evidence_only=True,
    )

    assert decision.keep is False
    assert decision.reason == "timeout_or_invalid"
    assert decision.infrastructure_error is True


def test_missing_requested_lift_refuses_keep_without_raising():
    stock = {"catboost": _row("catboost", "ok", 0.80)}
    patched = {"catboost": _row("catboost", "ok", 0.95)}

    decision = compare_pack(
        stock,
        patched,
        lift_ids=("catboost", "rf"),
        protect_ids=("catboost",),
    )

    assert decision.keep is False
    assert decision.infrastructure_error is True
    assert decision.reason == "missing_measurement rf"
    assert decision.regression_deltas["rf"] is None


def test_missing_requested_protect_refuses_keep_without_raising():
    stock = {"catboost": _row("catboost", "ok", 0.80)}
    patched = {
        "catboost": _row("catboost", "ok", 0.95),
        "rf": _row("rf", "ok", 0.70),
    }

    decision = compare_pack(
        stock,
        patched,
        lift_ids=("catboost",),
        protect_ids=("catboost", "rf"),
    )

    assert decision.keep is False
    assert decision.infrastructure_error is True
    assert decision.reason == "missing_measurement rf"
    assert decision.target_delta == pytest.approx(0.15)


def test_gain_beside_a_patch_map_gap_does_not_keep():
    stock = {
        "catboost": _row("catboost", "ok", 0.80),
        "rf": _row("rf", "ok", 0.70),
    }
    patched = {"catboost": _row("catboost", "ok", 0.95)}

    decision = compare_pack(
        stock,
        patched,
        lift_ids=("catboost", "rf"),
        protect_ids=("catboost", "rf"),
    )

    assert decision.keep is False
    assert decision.infrastructure_error is True
    assert decision.reason == "missing_measurement rf"


def test_stock_crash_is_not_a_measured_baseline():
    stock = {"catboost": _row("catboost", "crash", 0.5)}
    patched = {"catboost": _row("catboost", "ok", 0.90)}

    decision = compare_pack(
        stock,
        patched,
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        evidence_only=True,
    )

    assert decision.keep is False
    assert decision.reason == "timeout_or_invalid"
    assert decision.infrastructure_error is True


def test_evidence_confirmation_counts_a_crashed_seed(tmp_path):
    from fedotllm.agents.evolve.controller.confirmation import confirm_dev
    from fedotllm.agents.evolve.evaluation.judge import verdict

    def stock(exam_ids, *, seed=42, **_kwargs):
        status = "crash" if seed == 44 else "ok"
        score = 0.5 if status == "crash" else 0.80
        return {task_id: ScoreResult(task_id, status, score) for task_id in exam_ids}

    def patched(exam_ids, **_kwargs):
        return {task_id: ScoreResult(task_id, "ok", 0.90) for task_id in exam_ids}

    confirmed, detail = confirm_dev(
        tmp_path,
        tmp_path,
        ("catboost",),
        ("catboost",),
        ("catboost",),
        evidence_only=True,
        measure_stock_fn=stock,
        measure_patched_fn=patched,
        verdict_fn=verdict,
    )

    assert confirmed is False
    assert detail["seed_confirmed"] is False
    assert detail["infrastructure_failures"] == 1
    assert detail["improved_seeds"] == 2


def test_affected_metric_treats_candidate_invalid_as_regressed():
    label, delta = _normalized_delta(
        "rf",
        _row("rf", "ok", 0.7),
        _row("rf", "invalid", float("nan"), "scoring_error"),
    )

    assert label == "regressed"
    assert delta is None

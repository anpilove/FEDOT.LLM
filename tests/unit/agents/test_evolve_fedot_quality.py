from __future__ import annotations

from fedotllm.agents.evolve.evaluation.fedot_quality import decision_from_quality_jobs
from fedotllm.agents.evolve.evaluation.quality_registry import (
    get_dataset,
    job_spec,
    list_quality_task_ids,
    list_starter_task_ids,
    load_quality_registry,
)
from fedotllm.agents.evolve.controller.quality_queue import queue_priority
from fedotllm.agents.evolve.types import ScoreResult


def test_quality_registry_is_full_openml_and_fixed_before_scores():
    registry = load_quality_registry()
    ids = list_quality_task_ids(registry)
    assert registry.timeout_seconds == 3600
    assert registry.preset == "best_quality"
    assert registry.with_tuning is True
    assert registry.portfolio is False
    assert registry.api == "fedot"
    assert registry.cpu_quota == 32
    assert registry.n_jobs_per_job == 8
    assert ids == (
        "openml-31-credit-g",
        "openml-10101-blood-transfusion",
        "openml-37-diabetes",
        "openml-3917-kc1",
        "openml-53-vehicle",
        "openml-9952-phoneme",
        "openml-3-kr-vs-kp",
        "openml-146818-australian",
    )
    assert list_starter_task_ids(registry) == ids
    assert len(ids) >= 8
    for dataset in registry.datasets:
        assert dataset.source == "openml_task"
        assert dataset.fold == 0
        assert dataset.repeat == 0
        spec = job_spec(dataset, registry=registry)
        assert spec["runs"] == ["stock", "patch"]
        assert spec["timeout_minutes"] == 60.0
        assert spec["require_search_ran"] is True
        assert spec["portfolio"] is False
        assert dataset.openml_task > 0


def test_toy_metric_never_vetoes_the_hour_queue():
    assert queue_priority(probe_status="no_change", toy_metric_moved=False) == "normal"
    assert queue_priority(probe_status="changed", toy_metric_moved=False) == "high"
    assert queue_priority(probe_status="no_change", toy_metric_moved=True) == "high"


def _pair(task_id: str, stock: float, patched: float, *, search=True, status="ok"):
    spec = job_spec(get_dataset(task_id))
    stock_result = ScoreResult(
        task_id, status, stock, metric_observations={"search_ran": search}
    )
    patched_result = ScoreResult(
        task_id, status, patched, metric_observations={"search_ran": search}
    )
    return {
        "spec": spec,
        "stock": {"status": status, "score": stock},
        "patched": {"status": status, "score": patched},
        "stock_result": stock_result,
        "patched_result": patched_result,
        "search_ran": search,
    }


def test_composing_skip_is_not_a_quality_measurement():
    rows = [_pair("openml-31-credit-g", 0.70, 0.80, search=False, status="invalid")]
    decision = decision_from_quality_jobs(rows)
    assert decision.keep is False
    assert decision.infrastructure_error is True
    assert "composing_did_not_start" in decision.reason


def test_quality_keep_requires_gain_without_regression():
    rows = [
        _pair("openml-31-credit-g", 0.70, 0.82),
        _pair("openml-10101-blood-transfusion", 0.65, 0.66),
        _pair("openml-37-diabetes", 0.72, 0.73),
        _pair("openml-3917-kc1", 0.80, 0.80),
    ]
    decision = decision_from_quality_jobs(rows)
    assert decision.keep is True
    assert decision.reason == "quality_improved"


def test_quality_regression_drops():
    rows = [
        _pair("openml-31-credit-g", 0.80, 0.82),
        _pair("openml-10101-blood-transfusion", 0.70, 0.50),
    ]
    decision = decision_from_quality_jobs(rows)
    assert decision.keep is False
    assert "regression" in decision.reason

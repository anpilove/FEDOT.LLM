from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fedotllm.agents.evolve.evaluation import fedot_quality
from fedotllm.agents.evolve.evaluation.fedot_quality import decision_from_quality_jobs
from fedotllm.agents.evolve.evaluation.quality_registry import (
    get_dataset,
    job_spec,
    list_quality_task_ids,
    load_quality_registry,
    select_task_ids_for_job,
)
from fedotllm.agents.evolve.controller.quality_queue import queue_priority
from fedotllm.agents.evolve.execution.process import WorkerOutcome
from fedotllm.agents.evolve.types import ScoreResult, TestResult


def test_evaluation_runtime_identity_records_giotto_ph_version(monkeypatch):
    def package_version(name: str) -> str:
        if name == "giotto-ph":
            return "0.2.4"
        return "test-version"

    monkeypatch.setattr(fedot_quality, "version", package_version)

    assert fedot_quality._runtime_versions()["giotto-ph"] == "0.2.4"


def _task_ids(registry, problem: str) -> tuple[str, ...]:
    return tuple(item.task_id for item in registry.datasets if item.problem == problem)


def test_quality_registry_is_full_openml_and_fixed_before_scores():
    registry = load_quality_registry()
    ids = list_quality_task_ids(registry)
    tabular = _task_ids(registry, "classification")
    regression = _task_ids(registry, "regression")
    ts_ids = _task_ids(registry, "ts_forecasting")
    assert registry.timeout_seconds == 5400
    assert registry.fedot_search_budget_seconds == 4500
    assert registry.fedot_timeout_policy == "return_best"
    assert registry.final_fit_reserve_seconds == 1500
    assert registry.preset == "best_quality"
    assert registry.with_tuning is True
    assert registry.portfolio is False
    assert registry.api == "fedot"
    assert registry.cpu_quota == 28
    assert registry.n_jobs_per_job == 4
    assert job_spec(get_dataset("fedot-ts-beer", registry), registry=registry)[
        "max_pipeline_fit_time_seconds"
    ] == 500
    ts_spec = job_spec(get_dataset("fedot-ts-beer", registry), registry=registry)
    assert ts_spec["max_pipeline_evaluation_time_seconds"] == 1500
    assert ts_spec["timeout_seconds"] == 5400
    assert ts_spec["fedot_search_budget_seconds"] == 4500
    assert ts_spec["fedot_timeout_policy"] == "return_best"
    assert ts_spec["final_fit_reserve_seconds"] == 1500
    assert ts_spec["timeout_minutes"] == 75.0
    assert job_spec(get_dataset("openml-31-credit-g", registry), registry=registry)[
        "max_pipeline_fit_time_seconds"
    ] == 180
    assert job_spec(get_dataset("openml-31-credit-g", registry), registry=registry)[
        "max_pipeline_evaluation_time_seconds"
    ] == 900
    assert tabular == (
        "openml-31-credit-g",
        "openml-10101-blood-transfusion",
        "openml-37-diabetes",
        "openml-3917-kc1",
        "openml-53-vehicle",
        "openml-9952-phoneme",
        "openml-3-kr-vs-kp",
        "openml-146818-australian",
        "openml-6-letter",
        "openml-11-balance-scale",
        "openml-12-mfeat-factors",
        "openml-14-mfeat-fourier",
        "openml-15-breast-w",
        "openml-16-mfeat-karhunen",
        "openml-18-mfeat-morphological",
        "openml-22-mfeat-zernike",
        "openml-23-cmc",
        "openml-28-optdigits",
        "openml-29-credit-approval",
        "openml-32-pendigits",
        "openml-43-spambase",
        "openml-45-splice",
        "openml-49-tic-tac-toe",
        "openml-219-electricity",
    )
    assert regression == (
        "openml-regression-moneyball",
        "openml-regression-abalone",
        "openml-regression-airfoil-self-noise",
        "openml-regression-concrete-strength",
        "openml-regression-energy-efficiency",
        "openml-regression-forest-fires",
        "openml-regression-protein",
        "openml-regression-superconductivity",
    )
    assert ts_ids == (
        "fedot-ts-beer",
        "fedot-ts-australia",
        "fedot-ts-salaries",
        "fedot-ts-temperature",
        "fedot-ts-traffic",
        "fedot-ts-economic",
        "fedot-ts-arctic",
        "fedot-ts-metocean",
    )
    assert ids == tabular + regression + ts_ids
    assert len(tabular) == 24
    assert len(regression) == len(ts_ids) == 8
    assert len(ids) == 40
    registry_json = json.loads(
        Path("fedotllm/agents/evolve/evaluation/quality_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry_json["starter_task_ids"] == list(ids)
    assert registry_json["selection"]["regression_suite"] == "OpenML-CTR23"
    assert registry.selection_study == 99
    registered_tasks = {
        dataset.openml_task for dataset in registry.datasets if dataset.source == "openml_task"
    }
    assert {
        3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49, 53, 219
    } <= registered_tasks
    assert {10101, 3917, 9952, 146818} <= registered_tasks
    for dataset in registry.datasets:
        assert dataset.fold == 0
        assert dataset.repeat == 0
        spec = job_spec(dataset, registry=registry)
        assert spec["runs"] == ["stock", "patch"]
        assert spec["timeout_minutes"] == 75.0
        assert spec["require_search_ran"] is False
        assert spec["portfolio"] is False
        if dataset.source == "openml_task":
            assert dataset.problem in {"classification", "regression"}
            assert dataset.openml_task > 0
            if dataset.problem == "regression":
                assert dataset.metric == "rmse"
                assert dataset.delta_mode == "relative"
        else:
            assert dataset.source in {"fedot_public_ts", "fedot_local_ts"}
            assert dataset.problem == "ts_forecasting"
            assert dataset.forecast_horizon > 0
            assert len(dataset.data_sha256) == 64
            assert spec["forecast_horizon"] == dataset.forecast_horizon
            if dataset.source == "fedot_local_ts":
                assert dataset.delta_mode == "relative"
                assert dataset.min_delta == 0.01


def test_toy_metric_never_vetoes_the_hour_queue():
    from fedotllm.agents.evolve.controller.quality_queue import (
        cheap_screen_not_quality_verdict,
    )

    assert queue_priority(probe_status="no_change", toy_metric_moved=False) == "normal"
    assert queue_priority(probe_status="changed", toy_metric_moved=False) == "high"
    assert queue_priority(probe_status="no_change", toy_metric_moved=True) == "high"
    assert cheap_screen_not_quality_verdict(probe_status="no_change", target_delta=0.0)
    assert cheap_screen_not_quality_verdict(
        reason="no_affected_metric_signal", target_delta=0.0
    )
    assert cheap_screen_not_quality_verdict(
        reason="target_delta 0.0000 < 0.01", target_delta=0.0
    )
    assert not cheap_screen_not_quality_verdict(
        reason="no_quality_gain", target_delta=0.0, stage="quality"
    )


def test_historical_delta0_and_no_change_stay_hour_queue_eligible(tmp_path):
    from fedotllm.agents.evolve.storage.replay import (
        rejected_probe_hashes_from_findings,
        tried_patch_hashes_from_findings,
    )

    path = tmp_path / "findings.jsonl"
    rows = [
        {
            "record_type": "finding",
            "source_hash": "source",
            "evaluation_protocol_hash": "protocol",
            "patch_hash": "probe-zero",
            "behavior_probe": {"status": "no_change", "code": "print(1)"},
            "dev": {
                "reason": "behavior_probe_no_change",
                "patched": None,
                "target_delta": None,
            },
        },
        {
            "record_type": "finding",
            "source_hash": "source",
            "evaluation_protocol_hash": "protocol",
            "patch_hash": "toy-zero",
            "behavior_probe": {"status": "changed", "code": "print(2)"},
            "dev": {
                "reason": "no_affected_metric_signal",
                "patched": {"lgbm": {"status": "ok", "score": 0.86}},
                "target_delta": 0.0,
                "stage": "dev",
            },
        },
        {
            "record_type": "finding",
            "source_hash": "source",
            "evaluation_protocol_hash": "protocol",
            "patch_hash": "hour-zero",
            "dev": {
                "reason": "no_quality_gain",
                "target_delta": 0.0,
                "stage": "quality",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    scope = {"source_hash": "source", "evaluation_protocol_hash": "protocol"}
    assert tried_patch_hashes_from_findings(path, **scope) == {"hour-zero"}
    assert rejected_probe_hashes_from_findings(path, **scope) == {}


def test_campaign_enqueues_no_change_and_bit_identical_toy(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.types import EvolveRunPolicy, PatchCandidate, PatchEdit, MatchSite

    source = tmp_path / "source"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "fedot" / "a.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    lead = MatchSite("execution", "fedot/a.py", 1, "executed symbol value")
    stock = {
        "catboost": ScoreResult("catboost", "ok", 0.86),
        "rf": ScoreResult("rf", "ok", 0.80),
    }
    monkeypatch.setattr(loop, "scout", lambda *_a, **_k: [lead])
    monkeypatch.setattr(
        loop,
        "fix_lead",
        lambda *_a, **_k: PatchCandidate(
            "toy-delta0",
            edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
            file_path="fedot/a.py",
            behavior_probe="print(1)",
        ),
    )
    monkeypatch.setattr(
        loop,
        "compare_behavior_probe",
        lambda *_a, **_k: {"status": "no_change", "code": "print(1)"},
    )
    monkeypatch.setattr(loop, "measure_stock", lambda *_a, **_k: stock)
    monkeypatch.setattr(loop, "measure_patched", lambda *_a, **_k: stock)
    monkeypatch.setattr(
        loop, "measure_fedot_tests", lambda *_a, **_k: TestResult("passed", 0)
    )
    monkeypatch.setattr(
        loop, "_candidate_confirmation_scope", lambda *_a, **_k: ("catboost",)
    )
    monkeypatch.setattr(loop, "snapshot_diff", lambda *_a, **_k: "diff")
    workspace = tmp_path / "work"
    decision = loop.run_once(
        checkout=source,
        workspace=workspace,
        inference=object(),
        lift_ids=("catboost", "rf"),
        protect_ids=("catboost", "rf"),
        max_leads=1,
        max_revisions=1,
        policy=EvolveRunPolicy(
            verify_manifest=False,
            confirm_and_ablate=False,
            confirm_small_signals=False,
            evaluate_final=False,
            fedot_quality_jobs=True,
        ),
    )
    assert decision.reason == "queued_for_fedot_quality"
    job = json.loads((workspace / "quality_queue" / "toy-delta0.json").read_text())
    assert job["status"] == "queued"
    assert job["probe_status"] == "no_change"
    assert job["toy_metric"] == "no_veto"
    assert job["priority"] == "normal"
    assert job["hypothesis"]["id"].startswith("h-1-")
    assert job["quality_scope"]["confidence"] == "fallback"
    queued_event = next(
        json.loads(line)
        for line in (workspace / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event") == "quality_queue"
    )
    assert queued_event["hypothesis"] == job["hypothesis"]
    assert queued_event["quality_scope"] == job["quality_scope"]
    assert queued_event["task_ids"] == job["task_ids"]
    summary = json.loads((workspace / "campaign_summary.json").read_text())
    assert summary["quality_queue"]["status_counts"] == {"queued": 1}
    assert summary["quality_queue"]["pending"] == [
        {
            "candidate_id": "toy-delta0",
            "status": "queued",
            "hypothesis_id": job["hypothesis"]["id"],
            "task_count": len(job["task_ids"]),
        }
    ]


def test_enqueue_writes_campaign_root_not_run_subdir(tmp_path):
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import PatchCandidate

    run = tmp_path / "runs" / "20260916-run"
    run.mkdir(parents=True)
    candidate = PatchCandidate(candidate_id="abc123def456", file_path="fedot/x.py")
    job_path = enqueue_quality_job(
        run,
        candidate=candidate,
        patch_text="diff --git a/fedot/x.py b/fedot/x.py\n",
        hint="technically_valid_metric_unclear",
        priority="normal",
        probe_status="no_change",
        toy_metric="no_veto",
    )
    assert job_path == tmp_path / "quality_queue" / "abc123def456.json"
    assert (tmp_path / "quality_queue" / "abc123def456.patch").is_file()
    assert not (run / "quality_queue").exists()


def _pair(task_id: str, stock: float, patched: float, *, search=True, status="ok"):
    spec = job_spec(get_dataset(task_id))
    stock_result = ScoreResult(
        task_id, status, stock,
        metric_observations={"metric_valid": True, "search_ran": search},
    )
    patched_result = ScoreResult(
        task_id, status, patched,
        metric_observations={"metric_valid": True, "search_ran": search},
    )
    return {
        "spec": spec,
        "stock": {
            "status": status,
            "score": stock,
            "metric_observations": {"metric_valid": True},
        },
        "patched": {
            "status": status,
            "score": patched,
            "metric_observations": {"metric_valid": True},
        },
        "stock_result": stock_result,
        "patched_result": patched_result,
        "search_ran": search,
    }


def test_composing_skip_with_valid_paired_metrics_is_scored():
    row = _pair("openml-31-credit-g", 0.70, 0.80, search=False, status="ok")
    rows = [row]

    decision = decision_from_quality_jobs(rows)
    assert decision.keep is True
    assert decision.infrastructure_error is False
    assert decision.quality_failure is None
    assert decision.reason == "quality_improved"


def test_no_search_does_not_mask_a_measured_regression():
    rows = [_pair("openml-31-credit-g", 0.80, 0.70, search=False, status="ok")]

    decision = decision_from_quality_jobs(rows)

    assert decision.keep is False
    assert decision.infrastructure_error is False
    assert decision.quality_failure is None
    assert decision.reason.startswith("regression openml-31-credit-g")


def test_worker_metric_is_valid_even_when_composition_is_skipped():
    from fedotllm.agents.evolve.evaluation.fedot_quality import _score_from_payload

    result = _score_from_payload(
        "openml-31-credit-g",
        {
            "status": "ok",
            "score": 0.73,
            "metric_observations": {
                "metric_valid": True,
                "search_ran": False,
                "search_detail": "composing_did_not_start",
            },
        },
        env_hash="test-env",
        cmd="worker",
    )

    assert result.status == "ok"
    assert result.score == 0.73
    assert result.metric_observations["search_ran"] is False
    assert result.metric_observations["metric_valid"] is True


@pytest.mark.parametrize("side", ["stock_result", "patched_result"])
@pytest.mark.parametrize("invalid_observations", [{}, {"metric_valid": False}])
def test_quality_decision_refuses_unattested_stock_or_patch_metrics(
    side, invalid_observations
):
    row = _pair("openml-31-credit-g", 0.70, 0.80)
    row[side].metric_observations = dict(invalid_observations)

    decision = decision_from_quality_jobs([row])

    assert decision.keep is None
    assert decision.quality_outcome == "not_evaluated"
    assert decision.unresolved_task_ids == ("openml-31-credit-g",)
    assert "metric_valid" in decision.reason


def test_timeout_remains_infrastructure_error():
    rows = [_pair("openml-31-credit-g", 0.70, 0.80, search=False, status="timeout")]

    decision = decision_from_quality_jobs(rows)

    assert decision.keep is None
    assert decision.infrastructure_error is True
    assert decision.quality_failure is None
    assert "timeout" in decision.reason


def test_fedot_internal_fit_budget_timeout_is_not_misclassified_as_infrastructure():
    row = _pair("fedot-ts-traffic", 0.70, 0.80)
    row["patched_result"] = ScoreResult(
        "fedot-ts-traffic", "timeout", float("nan"),
        detail="FEDOT return_best budget exhausted during final_refit",
        metric_observations={
            "failure_kind": "fedot_budget_timeout",
            "failed_stage": "fit",
            "metric_valid": False,
            "metric_outcome": "not_measured",
            "fedot_budget_phase": "final_refit",
        },
    )

    decision = decision_from_quality_jobs([row])

    assert decision.keep is None
    assert decision.infrastructure_error is False
    assert decision.quality_failure == "fedot_budget_timeout"
    assert decision.quality_outcome == "not_evaluated"
    assert decision.unresolved_task_ids == ("fedot-ts-traffic",)


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


def _quality_rows_for_deltas(deltas, *, invalid_index=None):
    from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry

    registry = load_quality_registry()
    task_ids = [
        dataset.task_id
        for dataset in registry.datasets
        if dataset.problem == "classification"
    ][: len(deltas)]
    rows = []
    for index, (task_id, delta) in enumerate(zip(task_ids, deltas, strict=True)):
        spec = job_spec(get_dataset(task_id, registry), registry=registry)
        stock = ScoreResult(task_id, "ok", 0.70,
                            metric_observations={"metric_valid": True, "search_ran": True})
        patched = ScoreResult(
            task_id,
            "invalid" if index == invalid_index else "ok",
            float("nan") if index == invalid_index else 0.70 + delta,
            metric_observations={
                "metric_valid": True,
                "search_ran": True,
                **({"failure_kind": "fedot_fit_error", "failed_stage": "fit"}
                   if index == invalid_index else {}),
            },
        )
        rows.append(
            {
                "spec": spec,
                "stock": asdict(stock),
                "patched": asdict(patched),
                "stock_result": stock,
                "patched_result": patched,
            }
        )
    return rows


def test_quality_verdict_uses_majority_for_broad_and_two_wins_for_local():
    broad = decision_from_quality_jobs(_quality_rows_for_deltas([0.02] * 5 + [0.0] * 3))
    local = decision_from_quality_jobs(_quality_rows_for_deltas([0.02, 0.02] + [0.0] * 6))

    assert broad.keep is True
    assert broad.quality_outcome == "broad_positive"
    assert broad.improved_task_count == 5
    assert broad.target_delta is None
    assert local.keep is True
    assert local.quality_outcome == "local_positive"
    assert local.improved_task_count == 2
    assert local.target_delta is None


def test_any_real_loss_is_visible_even_below_gain_threshold():
    deltas = [0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, -0.001]
    decision = decision_from_quality_jobs(_quality_rows_for_deltas(deltas))

    assert decision.keep is False
    assert decision.quality_outcome == "regression"
    assert decision.regressed_task_count == 1
    assert min(decision.regression_deltas.values()) < 0
    assert decision.target_delta is None


def test_one_gain_is_insufficient_and_one_invalid_pair_is_not_evaluated():
    insufficient = decision_from_quality_jobs(
        _quality_rows_for_deltas([0.02] + [0.0] * 7)
    )
    incomplete = decision_from_quality_jobs(
        _quality_rows_for_deltas([0.02] * 7 + [0.0], invalid_index=7)
    )

    assert insufficient.keep is False
    assert insufficient.quality_outcome == "insufficient_effect"
    assert incomplete.quality_outcome == "not_evaluated"
    assert incomplete.quality_failure == "candidate_error"
    assert incomplete.keep is None
    assert incomplete.improved_task_count == 7


def test_patch_scoring_failure_is_unresolved_evaluation_error_not_candidate_error():
    rows = _quality_rows_for_deltas([0.02] * 7 + [0.0], invalid_index=7)
    rows[-1]["patched_result"].metric_observations.update(
        failure_kind="scoring_error", failed_stage="metrics"
    )

    decision = decision_from_quality_jobs(rows)

    assert decision.keep is None
    assert decision.quality_outcome == "not_evaluated"
    assert decision.quality_failure == "evaluation_error"
    assert decision.infrastructure_error is False
    assert "failed_stage=metrics" in decision.reason


def test_queue_patch_strips_stock_prefix(tmp_path):
    from fedotllm.agents.evolve.controller.quality_executor import apply_queue_patch

    checkout = tmp_path / "exp"
    target = checkout / "fedot" / "core" / "data" / "merge" / "data_merger.py"
    target.parent.mkdir(parents=True)
    target.write_text("common_predicts = [output.predict[:predict_len] for output in self.outputs]\n", encoding="utf-8")
    assert apply_queue_patch(
        checkout,
        candidate_id="tail-merge",
        file_path="fedot/core/data/merge/data_merger.py",
        patch_text=(
            "--- stock/fedot/core/data/merge/data_merger.py\n"
            "+++ patched/fedot/core/data/merge/data_merger.py\n"
            "@@\n"
            "-common_predicts = [output.predict[:predict_len] for output in self.outputs]\n"
            "+common_predicts = [output.predict[-predict_len:] for output in self.outputs]\n"
        ),
    )
    assert "predict[-predict_len:]" in target.read_text(encoding="utf-8")


def test_drain_runs_queued_candidates_without_separate_stock_wave(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot" / "x.py").write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate(candidate_id="lgbm-toy", file_path="fedot/x.py"),
        patch_text=(
            "--- a/fedot/x.py\n"
            "+++ b/fedot/x.py\n"
            "@@\n"
            "-value = 1\n"
            "+value = 2\n"
        ),
        hint="lgbm callback discarded; toy delta 0 is not a veto",
        priority="normal",
        probe_status="no_change",
        toy_metric="no_veto",
    )
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate(candidate_id="svd-hint", file_path="fedot/x.py"),
        patch_text=(
            "--- a/fedot/x.py\n"
            "+++ b/fedot/x.py\n"
            "@@\n"
            "-value = 1\n"
            "+value = 3\n"
        ),
        hint="svd",
        priority="high",
        probe_status="changed",
        toy_metric="early_gain",
    )

    calls: list[str] = []

    def fake_measure(source_checkout, experiment, **kwargs):
        calls.append("patch:" + experiment.name)
        return Decision(True, "quality_improved", 0.02, stage="quality")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        fake_measure,
    )
    journal_events = []
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda _path, event: journal_events.append(event),
    )

    payload = drain_quality_queue(
        source,
        workspace,
        task_ids=list_quality_task_ids(),
    )
    # This fake quality evaluator emits no per-side results. The drain must
    # not run a separate stock-only wave ahead of it.
    assert payload["stock_ok"] is False
    assert payload["task_ids"] == []
    assert [job["candidate_id"] for job in payload["jobs"]] == ["svd-hint", "lgbm-toy"]
    assert all(job["keep"] for job in payload["jobs"])
    # Execution is concurrent under the shared proposal scheduler; the returned
    # payload above still preserves queue priority, while call order is incidental.
    assert set(calls) == {"patch:svd-hint", "patch:lgbm-toy"}
    lgbm = json.loads(
        (workspace / "quality_queue" / "lgbm-toy.json").read_text(encoding="utf-8")
    )
    assert lgbm["status"] == "keep"
    assert lgbm["probe_status"] == "no_change"


def test_drain_keeps_candidate_failure_separate_from_drop_and_infrastructure(
    tmp_path, monkeypatch
):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    target = source / "fedot" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate("patched-invalid", file_path="fedot/x.py"),
        patch_text="--- a/fedot/x.py\n+++ b/fedot/x.py\n@@\n-value = 1\n+value = 2\n",
        hint="patched side invalid",
        priority="normal",
    )
    journal_events = []
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        lambda *_args, **_kwargs: Decision(
            False,
            "patched_quality_failed:openml-31-credit-g:invalid:search_ran=False",
            None,
            stage="quality",
            quality_failure="candidate_error",
        ),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda _path, event: journal_events.append(event),
    )

    payload = drain_quality_queue(
        source, workspace, task_ids=["openml-31-credit-g"]
    )

    proposal = json.loads(
        (workspace / "quality_queue" / "patched-invalid.json").read_text(encoding="utf-8")
    )
    assert proposal["status"] == "candidate_error"
    assert proposal["infrastructure_error"] is False
    assert payload["jobs"][0]["status"] == "candidate_error"
    assert "автоматически повторять" in proposal["quality_message"]
    disposition = next(
        event for event in journal_events if event["event"] == "quality_disposition"
    )
    assert disposition["quality_failure"] == "candidate_error"
    assert disposition["infrastructure_error"] is False


def test_stock_cache_requires_exact_compute_profile_and_complete_identity(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    cache = tmp_path / "stock_cache"
    fq.bind_stock_cache(cache)
    registry = load_quality_registry()
    dataset = get_dataset("openml-31-credit-g", registry)
    checkout = tmp_path / "FEDOT"
    checkout.mkdir()
    wrong_profile = fq.stock_cache_identity(
        dataset,
        registry,
        n_jobs=10,
        source_sha256=fq._source_sha256(str(checkout.resolve())),
    )
    leftover = cache / f"{registry.identity}__{dataset.task_id}__n10__s{registry.seed}.json"
    leftover.write_text(json.dumps({"status": "ok", "score": 0.77, "identity": wrong_profile}))
    called = []
    monkeypatch.setattr(
        fq,
        "run_worker",
        lambda *_args, **_kwargs: (called.append(True) or WorkerOutcome(returncode=0, stdout="", stderr="")),
    )
    result = fq.run_quality_side(
        dataset,
        checkout=checkout,
        side="stock",
        n_jobs=8,
        registry=registry,
    )
    assert called == [True]
    assert result.metric_observations.get("cache_hit") is not True

    exact_path = fq._stock_cache_path(dataset, registry, 8)
    exact_identity = fq.stock_cache_identity(
        dataset,
        registry,
        n_jobs=8,
        source_sha256=fq._source_sha256(str(checkout.resolve())),
    )
    exact_path.write_text(
        json.dumps({"status": "ok", "score": 0.77,
                    "metric_observations": {"metric_valid": True},
                    "identity": exact_identity}),
        encoding="utf-8",
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("exact compatible baseline cache must not launch FEDOT")

    monkeypatch.setattr(fq, "run_worker", boom)
    hit = fq.run_quality_side(
        dataset,
        checkout=checkout,
        side="stock",
        n_jobs=8,
        registry=registry,
    )
    assert hit.score == 0.77
    assert hit.metric_observations["cache_hit"] is True


@pytest.mark.parametrize("metric_observations", [{}, {"metric_valid": False}])
def test_stock_cache_rejects_success_score_without_valid_metric_attestation(
    tmp_path, metric_observations
):
    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    cache = tmp_path / "stock_cache"
    fq.bind_stock_cache(cache)
    registry = load_quality_registry()
    dataset = get_dataset("openml-31-credit-g", registry)
    checkout = tmp_path / "FEDOT"
    checkout.mkdir()
    source_sha256 = fq._source_sha256(str(checkout.resolve()))
    identity = fq.stock_cache_identity(
        dataset, registry, n_jobs=8, source_sha256=source_sha256
    )
    path = fq._stock_cache_path(dataset, registry, 8)
    path.write_text(json.dumps({
        "task_id": dataset.task_id,
        "status": "ok",
        "score": 0.77,
        "metric_observations": metric_observations,
        "identity": identity,
    }))

    result = fq._load_stock_cache(
        dataset, registry, 8, env_hash="test", source_sha256=source_sha256
    )

    assert result is None


def test_quality_worker_receives_task_deadline_separate_from_watchdog(tmp_path, monkeypatch):
    """The per-side limit must not be replaced by the longer guardian cap."""
    import sys

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    registry = load_quality_registry()
    dataset = get_dataset("fedot-ts-beer", registry)
    checkout = tmp_path / "FEDOT"
    checkout.mkdir()
    worker_tmp = tmp_path / "worker-tmp"
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(worker_tmp))
    monkeypatch.setattr(fq, "_fedot_commit", lambda _checkout: "fedot-commit")
    monkeypatch.setattr(fq, "_source_sha256", lambda _checkout: "source-sha")
    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: sys.executable)
    monkeypatch.setattr(fq, "interpreter_identity", lambda *_args, **_kwargs: "runtime")
    monkeypatch.setattr(fq, "clean_subprocess_env", lambda *_args, **_kwargs: {})
    received = {}

    def timed_out_worker(_cmd, *, timeout, watchdog_timeout, log_path, **_kwargs):
        received.update(timeout=timeout, watchdog_timeout=watchdog_timeout)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("worker timed out", encoding="utf-8")
        output_path = Path(_cmd[_cmd.index("--out") + 1])
        output_path.with_suffix(".stages.json").write_text(json.dumps({
            "event": "heartbeat",
            "updated_at": "2026-10-14T22:10:41Z",
            "active_stage": "fit",
            "last_worker_output_line": "TunerInterface final_check",
            "task_id": "fedot-ts-beer",
            "side": "patch",
            "attempt": 1,
            "attempt_id": "quality:fedot-ts-beer:patch:1",
        }), encoding="utf-8")
        return WorkerOutcome(
            returncode=None, stdout="worker timed out", stderr="",
            timed_out=True, timeout_kind="task_timeout",
        )

    monkeypatch.setattr(fq, "run_worker", timed_out_worker)
    result = fq.run_quality_side(
        dataset, checkout=checkout, side="patch", registry=registry,
    )

    assert 0 < received["timeout"] <= 5400.0
    assert received["watchdog_timeout"] == 10800.0
    assert result.status == "timeout"
    assert result.metric_observations["timeout_seconds"] == 5400
    assert result.metric_observations["watchdog_seconds"] == 10800
    timeout_manifest = Path(result.metric_observations["timeout_manifest_path"])
    assert timeout_manifest.is_file()
    manifest = json.loads(timeout_manifest.read_text(encoding="utf-8"))
    assert manifest["event"] == "quality_worker_parent_timeout"
    assert manifest["task_id"] == "fedot-ts-beer"
    assert manifest["side"] == "patch"
    assert manifest["status"] == "timeout"
    assert manifest["active_stage"] == "fit"
    assert manifest["last_worker_output_line"] == "TunerInterface final_check"
    assert manifest["worker_log_path"] == result.metric_observations["worker_log_path"]
    assert Path(manifest["stage_status_path"]).is_file()


@pytest.mark.parametrize(
    ("identity_field", "wrong_value"),
    [
        ("task_id", "openml-37-diabetes"),
        ("side", "stock"),
        ("attempt", 3),
        ("attempt_id", "proposal-1:openml-31-credit-g:patch:3"),
        ("registry_identity", "another-registry"),
        ("evaluation_identity", {"source_sha256": "stale-source", "n_jobs": 4}),
    ],
)
def test_successful_worker_payload_with_mismatched_identity_is_rejected(
    tmp_path, monkeypatch, identity_field, wrong_value
):
    from fedotllm.agents.evolve.evaluation import fedot_quality as quality

    registry = load_quality_registry()
    dataset = get_dataset("openml-31-credit-g", registry)
    checkout = tmp_path / "FEDOT"
    checkout.mkdir()
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "worker-artifacts"))
    monkeypatch.setattr(quality, "_fedot_commit", lambda _checkout: "fedot-commit")
    monkeypatch.setattr(quality, "_source_sha256", lambda _checkout: "source-sha")
    monkeypatch.setattr(quality, "interpreter_identity", lambda *_args, **_kwargs: "runtime")
    monkeypatch.setattr(quality, "clean_subprocess_env", lambda *_args, **_kwargs: {})
    evaluation_identity = {"source_sha256": "current-source", "n_jobs": 4}

    def worker(command, **_kwargs):
        spec_path = Path(command[command.index("--spec-json") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        payload = {
            "task_id": spec["source_dataset"],
            "status": "ok",
            "score": 0.9,
            "duration_s": 1.0,
            "proposal_id": spec["proposal_id"],
            "attempt_id": spec["attempt_id"],
            "worker_invocation_id": spec["worker_invocation_id"],
            "source_dataset": spec["source_dataset"],
            "side": spec["side"],
            "attempt": spec["attempt"],
            "registry_identity": spec["registry_identity"],
            "evaluation_identity": spec["evaluation_identity"],
            "metric_observations": {"metric_valid": True},
        }
        payload[identity_field] = wrong_value
        output_path = Path(command[command.index("--out") + 1])
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return WorkerOutcome(returncode=0, stdout="worker complete", stderr="")

    monkeypatch.setattr(quality, "run_worker", worker)
    result = quality.run_quality_side(
        dataset,
        checkout=checkout,
        side="patch",
        registry=registry,
        proposal_id="proposal-1",
        attempt=2,
        evaluation_identity=evaluation_identity,
    )

    assert result.status == "invalid"
    assert result.score != result.score  # NaN: the reported metric must not be used.
    assert result.metric_observations["failure_kind"] == "worker_identity_mismatch"
    assert identity_field in result.metric_observations["identity_mismatch_fields"]


@pytest.mark.parametrize("timed_out", [False, True])
def test_concurrent_stock_cache_misses_share_one_baseline_training(tmp_path, monkeypatch, timed_out):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    cache = tmp_path / "stock-cache"
    fq.bind_stock_cache(cache)
    worker_tmp = tmp_path / "worker-tmp"
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(worker_tmp))
    registry = load_quality_registry()
    dataset = get_dataset("openml-31-credit-g", registry)
    checkout = tmp_path / "FEDOT"
    checkout.mkdir()
    monkeypatch.setattr(fq, "_fedot_commit", lambda _checkout: "commit-a")
    monkeypatch.setattr(fq, "_source_sha256", lambda _checkout: "source-a")
    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: "/python")
    monkeypatch.setattr(fq, "interpreter_identity", lambda *_args, **_kwargs: "runtime-a")
    monkeypatch.setattr(fq, "clean_subprocess_env", lambda *_args, **_kwargs: {"PATH": ""})

    calls = 0
    calls_lock = threading.Lock()

    def fake_worker(cmd, **_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        if timed_out:
            return WorkerOutcome(returncode=None, stdout="worker timed out", stderr="", timed_out=True)
        spec_path = Path(cmd[cmd.index("--spec-json") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_text(json.dumps({
            "task_id": dataset.task_id,
            "side": spec["side"],
            "attempt": spec["attempt"],
            "attempt_id": spec["attempt_id"],
            "worker_invocation_id": spec["worker_invocation_id"],
            "registry_identity": spec["registry_identity"],
            "evaluation_identity": spec.get("evaluation_identity"),
            "status": "ok",
            "score": 0.75,
            "duration_s": 0.05,
            "seed": registry.seed,
            "metric_observations": {"metric_valid": True},
        }))
        return WorkerOutcome(returncode=0, stdout="worker complete", stderr="")

    monkeypatch.setattr(fq, "run_worker", fake_worker)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(
            fq.run_quality_side,
            dataset,
            checkout=checkout,
            side="stock",
            n_jobs=4,
            registry=registry,
        ) for _ in range(2)]
        results = [future.result(timeout=5) for future in futures]

    assert calls == 1
    expected_status = "timeout" if timed_out else "ok"
    assert [result.status for result in results] == [expected_status, expected_status]
    if timed_out:
        assert all(not result.metric_observations.get("cache_hit") for result in results)
    else:
        assert [result.score for result in results] == [0.75, 0.75]
    assert sum(bool(result.metric_observations.get("singleflight_shared")) for result in results) == 1


def test_stock_singleflight_reports_waiters_until_shared_fit_finishes(tmp_path):
    from threading import Event, Thread

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq
    from fedotllm.agents.evolve.types import ScoreResult

    cache_path = tmp_path / "stock-cache.json"
    owner_entered = Event()
    release_owner = Event()
    waiter_observed = Event()
    waiting_events = []

    def compute():
        owner_entered.set()
        assert release_owner.wait(timeout=2)
        return ScoreResult(
            task_id="openml-31-credit-g",
            status="ok",
            score=0.75,
            metric_observations={"metric_valid": True},
        )

    owner_result = []
    follower_result = []
    follower_errors = []
    owner = Thread(target=lambda: owner_result.append(fq._join_stock_flight(
        cache_path,
        identity_key="wait-observability-test",
        cancellation_event=None,
        compute=compute,
    )))
    owner.start()
    assert owner_entered.wait(timeout=2)

    def on_waiting(wait_kind, elapsed_seconds):
        waiting_events.append((wait_kind, elapsed_seconds))
        waiter_observed.set()

    def join_follower():
        try:
            follower_result.append(fq._join_stock_flight(
                cache_path,
                identity_key="wait-observability-test",
                cancellation_event=None,
                compute=lambda: pytest.fail("singleflight follower must not compute"),
                on_waiting=on_waiting,
            ))
        except BaseException as exc:
            follower_errors.append(exc)

    follower = Thread(target=join_follower)
    follower.start()
    assert waiter_observed.wait(timeout=2)
    release_owner.set()
    owner.join(timeout=2)
    follower.join(timeout=2)

    assert waiter_observed.is_set()
    assert follower_errors == []
    assert waiting_events == [("shared_stock_fit", 0.0)]
    assert owner_result[0].score == follower_result[0].score == 0.75


def test_stock_cache_file_lock_wait_is_bounded_by_side_deadline(tmp_path):
    import fcntl
    import time
    from threading import Event, Thread

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    cache_path = tmp_path / "stock-cache.json"
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_has_lock = Event()
    release_owner = Event()

    def hold_lock():
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            owner_has_lock.set()
            assert release_owner.wait(timeout=2)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    owner = Thread(target=hold_lock)
    owner.start()
    assert owner_has_lock.wait(timeout=2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="stock cache file lock"):
            with fq._stock_cache_singleflight(cache_path, deadline=started + 0.05):
                pytest.fail("lock owner must prevent entering the cache section")
        assert time.monotonic() - started < 1.0
    finally:
        release_owner.set()
        owner.join(timeout=2)
    assert not owner.is_alive()


def test_shared_stock_wait_is_bounded_by_side_deadline(tmp_path):
    import time
    from threading import Event, Thread

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq
    from fedotllm.agents.evolve.types import ScoreResult

    cache_path = tmp_path / "stock-cache.json"
    owner_entered = Event()
    release_owner = Event()

    def compute():
        owner_entered.set()
        assert release_owner.wait(timeout=2)
        return ScoreResult(task_id="openml-31-credit-g", status="ok", score=0.75)

    owner = Thread(target=lambda: fq._join_stock_flight(
        cache_path,
        identity_key="bounded-shared-wait-test",
        cancellation_event=None,
        compute=compute,
    ))
    owner.start()
    assert owner_entered.wait(timeout=2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="shared stock fit"):
            fq._join_stock_flight(
                cache_path,
                identity_key="bounded-shared-wait-test",
                cancellation_event=None,
                compute=lambda: pytest.fail("singleflight follower must not compute"),
                deadline=started + 0.05,
            )
        assert time.monotonic() - started < 1.0
    finally:
        release_owner.set()
        owner.join(timeout=2)
    assert not owner.is_alive()


def test_stock_side_records_timeout_when_cache_lock_outlives_side_budget(
    tmp_path, monkeypatch
):
    import fcntl
    import time
    from threading import Event, Thread

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    monkeypatch.setenv("EVOLVE_QUALITY_STOCK_CACHE", str(tmp_path / "stock-cache"))
    registry = load_quality_registry()
    dataset = get_dataset("openml-31-credit-g", registry)
    cache_path = fq._stock_cache_path(dataset, registry, 4)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_has_lock = Event()
    release_owner = Event()

    def hold_lock():
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            owner_has_lock.set()
            assert release_owner.wait(timeout=2)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    monkeypatch.setattr(fq, "_fedot_commit", lambda _checkout: "fedot-test")
    monkeypatch.setattr(fq, "_source_sha256", lambda _checkout: "source-test")
    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: "/python")
    monkeypatch.setattr(fq, "interpreter_identity", lambda *_args, **_kwargs: {})
    owner = Thread(target=hold_lock)
    owner.start()
    assert owner_has_lock.wait(timeout=2)
    try:
        result = fq.run_quality_side(
            dataset,
            checkout=tmp_path / "checkout",
            side="stock",
            n_jobs=4,
            registry=registry,
            _side_started=time.monotonic() - registry.timeout_seconds + 0.05,
        )
    finally:
        release_owner.set()
        owner.join(timeout=2)

    assert not owner.is_alive()
    assert result.status == "timeout"
    assert result.metric_observations["timeout_kind"] == "stock cache file lock"
    assert result.metric_observations["worker_spawned"] is False
    assert result.metric_observations["timeout_limit_seconds"] == registry.timeout_seconds


def test_stock_warm_does_not_skip_identity_mismatch(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    cache = tmp_path / "stock_cache"
    fq.bind_stock_cache(cache)
    registry = load_quality_registry()
    dataset = get_dataset("openml-31-credit-g", registry)
    path = fq._stock_cache_path(dataset, registry, 8)
    path.write_text(
        json.dumps(
            {
                "task_id": dataset.task_id,
                "status": "ok",
                "score": 0.11,
                "seed": 42,
                "metric_observations": {
                    "search_ran": True,
                    "preset": "best_quality",
                    "timeout_seconds": 60,
                },
            }
        ),
        encoding="utf-8",
    )
    called = []

    def fake_run(*_args, **_kwargs):
        called.append(True)
        return WorkerOutcome(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(fq, "run_worker", fake_run)
    result = fq.run_quality_side(
        dataset,
        checkout=tmp_path / "FEDOT",
        side="stock",
        n_jobs=8,
        registry=registry,
    )
    assert called == [True]
    assert result.metric_observations.get("cache_hit") is not True


def test_stock_cache_identity_binds_checksum_pinned_ts_data():
    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    registry = load_quality_registry()
    dataset = get_dataset("fedot-ts-beer", registry)
    expected = fq.stock_cache_identity(dataset, registry)
    assert fq._identity_matches({"identity": expected}, expected)
    stale = {**expected, "data_sha256": "0" * 64}
    assert not fq._identity_matches({"identity": stale}, expected)


def test_quality_job_selects_tabular_or_ts_by_patch_path():
    registry = load_quality_registry()
    tabular = _task_ids(registry, "classification") + _task_ids(registry, "regression")
    regression = _task_ids(registry, "regression")
    classification = _task_ids(registry, "classification")
    ts_ids = _task_ids(registry, "ts_forecasting")
    assert select_task_ids_for_job(
        {"file_path": "fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py"}
    ) == tabular
    assert select_task_ids_for_job(
        {
            "file_path": (
                "fedot/core/operations/evaluation/operation_implementations/"
                "models/ts_implementations/statsmodels.py"
            )
        }
    ) == ts_ids
    assert select_task_ids_for_job(
        {
            "file_path": (
                "fedot/core/operations/evaluation/operation_implementations/"
                "data_operations/ts_transformations.py"
            )
        }
    ) == ts_ids
    assert select_task_ids_for_job(
        {"file_path": "fedot/core/operations/evaluation/operation_implementations/models/regression/ridge.py"}
    ) == regression
    assert select_task_ids_for_job(
        {"file_path": "fedot/core/operations/evaluation/operation_implementations/models/classification/logit.py"}
    ) == classification
    both = select_task_ids_for_job({"file_path": "fedot/core/data/data_split.py"})
    assert both == tabular + ts_ids


def test_inapplicable_skip_remains_unresolved_for_that_required_dataset():
    decision = decision_from_quality_jobs(
        [{"status": "skipped", "spec": {"source_dataset": "openml-31-credit-g"}}]
    )
    assert decision.keep is None
    assert decision.infrastructure_error is False
    assert decision.quality_outcome == "not_evaluated"
    assert decision.unresolved_task_ids == ("openml-31-credit-g",)
    assert decision.reason == "required_dataset_skipped:openml-31-credit-g"


def test_fedot_public_ts_loader_holdout_dry_run(tmp_path, monkeypatch):
    import hashlib
    from dataclasses import replace

    from fedotllm.agents.evolve.evaluation.openml_fold import load_fedot_public_ts

    checkout = tmp_path / "FEDOT"
    ts_dir = checkout / "examples" / "data" / "ts"
    ts_dir.mkdir(parents=True)
    rows = ["idx,value"] + [f"1956-{index:02d},{100 + index}" for index in range(1, 25)]
    source = ts_dir / "beer.csv"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("FEDOTLLM_REPO_PATH", str(checkout))
    dataset = replace(
        get_dataset("fedot-ts-beer"),
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    fold = load_fedot_public_ts(dataset)
    assert fold["n_train"] == 12
    assert fold["n_test"] == 12
    assert fold["forecast_horizon"] == 12
    assert fold["loader"] == "fedot.examples.data.ts"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_fedot_public_ts(replace(dataset, data_sha256="0" * 64))


def test_official_split_recovers_from_interrupted_shared_cache_write(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation import openml_fold

    monkeypatch.setenv("EVOLVE_QUALITY_DATA_CACHE", str(tmp_path / "shared-cache"))
    dataset = get_dataset("openml-31-credit-g")
    cached = openml_fold.cache_root() / (
        f"task-{dataset.openml_task}-r{dataset.repeat}-f{dataset.fold}.json"
    )
    cached.parent.mkdir(parents=True)
    # A concurrent worker can observe a directly-written shared cache after
    # its truncate but before its JSON payload has reached disk.
    cached.write_text('{"train": [0,', encoding="utf-8")
    monkeypatch.setattr(
        openml_fold, "_download_split_indices", lambda _dataset, *, n_rows: ([0, 1], [2])
    )

    assert openml_fold._official_indices(dataset, n_rows=3) == ([0, 1], [2])


def test_openml_task_loader_rejects_data_id_mismatch(monkeypatch):
    import pandas as pd

    from fedotllm.agents.evolve.evaluation import openml_fold

    dataset = get_dataset("openml-31-credit-g")
    task = SimpleNamespace(
        dataset_id=dataset.openml_data + 1,
        get_X_and_y=lambda **_kwargs: (pd.DataFrame({"feature": [1, 2]}), pd.Series([0, 1])),
        get_train_test_split_indices=lambda **_kwargs: ([0], [1]),
    )
    fake_openml = ModuleType("openml")
    fake_openml.tasks = SimpleNamespace(get_task=lambda _task_id: task)
    monkeypatch.setitem(sys.modules, "openml", fake_openml)
    monkeypatch.setattr(
        openml_fold,
        "_load_sklearn_and_splits",
        lambda _dataset: pytest.fail("data mismatch must not fall back to another loader"),
    )

    with pytest.raises(RuntimeError, match="OpenML task/data ID mismatch"):
        openml_fold.load_registered_fold(dataset)


def test_openml_task_loader_does_not_fallback_without_verified_task_metadata(monkeypatch):
    from fedotllm.agents.evolve.evaluation import openml_fold

    dataset = get_dataset("openml-31-credit-g")
    fake_openml = ModuleType("openml")

    def fail_get_task(_task_id):
        raise TimeoutError("metadata service unavailable")

    fake_openml.tasks = SimpleNamespace(get_task=fail_get_task)
    monkeypatch.setitem(sys.modules, "openml", fake_openml)
    monkeypatch.setattr(
        openml_fold,
        "_load_sklearn_and_splits",
        lambda _dataset: pytest.fail("fallback requires a verified task/data mapping"),
    )

    with pytest.raises(RuntimeError, match="could not verify OpenML task/data mapping"):
        openml_fold.load_registered_fold(dataset)


def test_public_ts_loader_recovers_from_interrupted_shared_cache_write(tmp_path, monkeypatch):
    import hashlib
    from dataclasses import replace
    from io import BytesIO

    from fedotllm.agents.evolve.evaluation import openml_fold

    cache = tmp_path / "shared-cache"
    monkeypatch.setenv("EVOLVE_QUALITY_DATA_CACHE", str(cache))
    monkeypatch.delenv("FEDOTLLM_REPO_PATH", raising=False)
    monkeypatch.delenv("FEDOTLLM_REPO_CACHE", raising=False)
    payload = b"idx,value\n0,10\n1,11\n2,12\n3,13\n"
    dataset = replace(
        get_dataset("fedot-ts-beer"),
        forecast_horizon=1,
        data_sha256=hashlib.sha256(payload).hexdigest(),
    )
    cached = cache / "fedot-ts-beer.csv"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(payload[:10])
    monkeypatch.setattr(
        openml_fold.urllib.request,
        "urlopen",
        lambda _url, timeout: BytesIO(payload),
    )

    fold = openml_fold.load_fedot_public_ts(dataset)

    assert fold["n_train"] == 3
    assert fold["n_test"] == 1
    assert cached.read_bytes() == payload


def test_fedot_local_ts_loader_uses_manifest_asset_and_last_horizon(tmp_path, monkeypatch):
    import hashlib
    from dataclasses import replace

    from fedotllm.agents.evolve.evaluation.openml_fold import load_fedot_local_ts

    checkout = tmp_path / "FEDOT"
    data_dir = checkout / "examples" / "real_cases" / "data" / "time_series"
    data_dir.mkdir(parents=True)
    (data_dir / "temperature.csv").write_text(
        "datetime,value,series_id\n"
        "2020-01-02,11,temp\n2020-01-01,10,temp\n"
        "2020-01-04,13,temp\n2020-01-03,12,temp\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEDOTLLM_REPO_PATH", str(checkout))
    source = data_dir / "temperature.csv"
    dataset = replace(
        get_dataset("fedot-ts-temperature"),
        forecast_horizon=2,
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    fold = load_fedot_local_ts(dataset)
    assert fold["n_train"] == 2
    assert fold["n_test"] == 2
    assert list(fold["X_train"]) == [10.0, 11.0]
    assert list(fold["y_test"]) == [12.0, 13.0]
    assert fold["loader"] == "fedot.examples.real_cases.data"


def test_quality_worker_environment_can_read_fedot_data_from_separate_source(tmp_path, monkeypatch):
    import subprocess
    import sys

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    checkout = tmp_path / "FEDOT-runtime"
    checkout.mkdir()
    dataset_source = tmp_path / "FEDOT-data-source"
    data_dir = dataset_source / "examples" / "data" / "ts"
    data_dir.mkdir(parents=True)
    (data_dir / "beer.csv").write_text("value\n1\n2\n", encoding="utf-8")
    monkeypatch.delenv("FEDOTLLM_REPO_PATH", raising=False)
    monkeypatch.setenv("FEDOTLLM_REPO_CACHE", str(dataset_source))
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "quality-tmp"))

    code = """
import hashlib, os
from pathlib import Path
from fedotllm.agents.evolve.evaluation.openml_fold import _fedot_checkout_ts_csv
source = _fedot_checkout_ts_csv('beer')
assert source == Path(os.environ['FEDOTLLM_REPO_CACHE']) / 'examples/data/ts/beer.csv'
assert source.read_text() == 'value\\n1\\n2\\n'
"""

    def run_worker(_cmd, *, cwd, env, timeout, watchdog_timeout, log_path, resource_check):
        resource_check()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake worker diagnostics", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        return WorkerOutcome(returncode=0, stdout=result.stdout, stderr=result.stderr)

    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: sys.executable)
    monkeypatch.setattr(fq, "run_worker", run_worker)
    dataset = get_dataset("fedot-ts-temperature")
    result = fq.run_quality_side(
        dataset,
        checkout=checkout,
        side="patch",
        registry=load_quality_registry(),
    )

    assert result.status == "invalid"  # fake worker intentionally writes no score file
    worker_log = Path(result.metric_observations["worker_log_path"])
    assert worker_log.is_file()
    assert worker_log.read_text(encoding="utf-8") == "fake worker diagnostics"


def test_quality_worker_environment_loads_local_ts_from_fedot_cache(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    import sys

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    checkout = tmp_path / "FEDOT-runtime"
    checkout.mkdir()
    dataset_source = tmp_path / "FEDOT-data-source"
    data_dir = dataset_source / "examples" / "real_cases" / "data" / "time_series"
    data_dir.mkdir(parents=True)
    source = data_dir / "temperature.csv"
    source.write_text(
        "datetime,value,series_id\n"
        "2020-01-02,11,temp\n2020-01-01,10,temp\n"
        "2020-01-04,13,temp\n2020-01-03,12,temp\n",
        encoding="utf-8",
    )
    dataset_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.delenv("FEDOTLLM_REPO_PATH", raising=False)
    monkeypatch.setenv("FEDOTLLM_REPO_CACHE", str(dataset_source))
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "quality-tmp"))

    code = f"""
from dataclasses import replace
from fedotllm.agents.evolve.evaluation.openml_fold import load_fedot_local_ts
from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset
dataset = replace(get_dataset('fedot-ts-temperature'), forecast_horizon=2,
                  data_sha256='{dataset_sha}')
fold = load_fedot_local_ts(dataset)
assert fold['n_train'] == 2
assert fold['n_test'] == 2
assert list(fold['X_train']) == [10.0, 11.0]
assert list(fold['y_test']) == [12.0, 13.0]
"""

    def run_worker(_cmd, *, cwd, env, timeout, watchdog_timeout, log_path, resource_check):
        resource_check()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("worker reached local TS loader", encoding="utf-8")
        missing_cache_env = dict(env)
        missing_cache_env.pop("FEDOTLLM_REPO_CACHE", None)
        without_cache = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=missing_cache_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        assert without_cache.returncode != 0
        assert "FEDOT data asset not found" in without_cache.stderr
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        return WorkerOutcome(returncode=0, stdout=result.stdout, stderr=result.stderr)

    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: sys.executable)
    monkeypatch.setattr(fq, "run_worker", run_worker)
    result = fq.run_quality_side(
        get_dataset("fedot-ts-temperature"),
        checkout=checkout,
        side="patch",
        registry=load_quality_registry(),
    )

    assert result.status == "invalid"  # fake worker intentionally writes no score file
    assert result.metric_observations["worker_log_path"]


def test_quality_cache_path_reaches_real_worker_popen(tmp_path, monkeypatch):
    import hashlib
    import sys

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq
    from fedotllm.agents.evolve.execution.process import run_worker as real_run_worker

    checkout = tmp_path / "FEDOT-runtime"
    checkout.mkdir()
    dataset_source = tmp_path / "FEDOT-data-source"
    data_dir = dataset_source / "examples" / "real_cases" / "data" / "time_series"
    data_dir.mkdir(parents=True)
    source = data_dir / "temperature.csv"
    source.write_text(
        "datetime,value,series_id\n"
        "2020-01-02,11,temp\n2020-01-01,10,temp\n"
        "2020-01-04,13,temp\n2020-01-03,12,temp\n",
        encoding="utf-8",
    )
    dataset_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.delenv("FEDOTLLM_REPO_PATH", raising=False)
    monkeypatch.setenv("FEDOTLLM_REPO_CACHE", str(dataset_source))
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "quality-tmp"))
    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: sys.executable)

    child_code = f"""
import json
import sys
from dataclasses import replace
from pathlib import Path
from fedotllm.agents.evolve.evaluation.openml_fold import load_fedot_local_ts
from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset

spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
dataset = replace(get_dataset('fedot-ts-temperature'), forecast_horizon=2,
                  data_sha256='{dataset_sha}')
fold = load_fedot_local_ts(dataset)
assert fold['n_train'] == 2
assert fold['n_test'] == 2
assert list(fold['X_train']) == [10.0, 11.0]
assert list(fold['y_test']) == [12.0, 13.0]
result = {{key: spec.get(key) for key in (
    'side', 'attempt', 'attempt_id', 'registry_identity',
    'evaluation_identity', 'worker_invocation_id',
)}}
result['task_id'] = spec['source_dataset']
result.update(status='ok', score=1.25, duration_s=0.01,
              metric_observations={{'metric_valid': True}})
Path(sys.argv[2]).write_text(json.dumps(result), encoding='utf-8')
"""

    def invoke_worker(command, *, remove_cache=False, **kwargs):
        spec_path = Path(command[command.index("--spec-json") + 1])
        out_path = Path(command[command.index("--out") + 1])
        env = dict(kwargs.pop("env"))
        if remove_cache:
            env.pop("FEDOTLLM_REPO_CACHE", None)
        child_command = [
            sys.executable, "-c", child_code, str(spec_path), str(out_path),
        ]
        return real_run_worker(child_command, env=env, **kwargs)

    monkeypatch.setattr(
        fq, "run_worker", lambda command, **kwargs: invoke_worker(
            command, remove_cache=True, **kwargs
        ),
    )
    without_cache = fq.run_quality_side(
        get_dataset("fedot-ts-temperature"), checkout=checkout, side="patch",
        registry=load_quality_registry(), proposal_id="cache-popen-test",
    )
    assert without_cache.status == "invalid"
    assert "FEDOT data asset not found" in without_cache.log_tail

    monkeypatch.setattr(
        fq, "run_worker", lambda command, **kwargs: invoke_worker(command, **kwargs),
    )
    with_cache = fq.run_quality_side(
        get_dataset("fedot-ts-temperature"), checkout=checkout, side="patch",
        registry=load_quality_registry(), proposal_id="cache-popen-test",
    )
    assert with_cache.status == "ok"
    assert with_cache.score == 1.25
    assert with_cache.metric_observations["metric_valid"] is True


def test_quality_worker_passes_opt_in_line_trace_controls(tmp_path, monkeypatch):
    import sys

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq
    from fedotllm.agents.evolve.execution.process import WorkerOutcome

    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "quality-tmp"))
    monkeypatch.setenv("EVOLVE_TRACE_PYTHON_LINES", "1")
    monkeypatch.setenv("EVOLVE_TRACE_LINE_INTERVAL_SECONDS", "1")

    captured = {}

    def run_worker(_cmd, *, cwd, env, timeout, watchdog_timeout, log_path, resource_check):
        captured.update(env)
        resource_check()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("worker reached launch boundary", encoding="utf-8")
        return WorkerOutcome(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: sys.executable)
    monkeypatch.setattr(fq, "run_worker", run_worker)
    result = fq.run_quality_side(
        get_dataset("fedot-ts-temperature"),
        checkout=tmp_path / "FEDOT",
        side="patch",
        registry=load_quality_registry(),
    )

    assert captured["EVOLVE_TRACE_PYTHON_LINES"] == "1"
    assert captured["EVOLVE_TRACE_LINE_INTERVAL_SECONDS"] == "1"
    assert result.status == "invalid"  # stub intentionally writes no metric result


def test_quality_worker_passes_opt_in_operation_fit_trace_controls(tmp_path, monkeypatch):
    import sys

    from fedotllm.agents.evolve.evaluation import fedot_quality as fq
    from fedotllm.agents.evolve.execution.process import WorkerOutcome

    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "quality-tmp"))
    trace_controls = {
        "FEDOT_TRACE_OPERATIONS": "1",
        "FEDOT_TRACE_OPERATIONS_SLOW_SECONDS": "30",
        "FEDOT_TRACE_OPERATIONS_ALL": "1",
        "FEDOT_TRACE_OPERATIONS_MAX_EVENTS": "123",
        "FEDOT_TRACE_RIPSER": "1",
        "FEDOT_TRACE_RIPSER_SLOW_SECONDS": "7",
        "FEDOT_TRACE_RIPSER_ALL": "1",
        "FEDOT_TRACE_RIPSER_MAX_EVENTS": "456",
    }
    for key, value in trace_controls.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("FEDOT_TRACE_OPERATIONS_UNUSED", raising=False)
    captured = {}

    def run_worker(_cmd, *, cwd, env, timeout, watchdog_timeout, log_path, resource_check):
        captured.update(env)
        resource_check()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("worker reached launch boundary", encoding="utf-8")
        return WorkerOutcome(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(fq, "fedot_python", lambda _checkout: sys.executable)
    monkeypatch.setattr(fq, "run_worker", run_worker)
    result = fq.run_quality_side(
        get_dataset("fedot-ts-temperature"),
        checkout=tmp_path / "FEDOT",
        side="patch",
        registry=load_quality_registry(),
    )

    for key, value in trace_controls.items():
        assert captured[key] == value
    assert captured["FEDOT_EVOLVE_JOB_ID"] == "quality:fedot-ts-temperature:patch:1"
    assert "FEDOT_TRACE_OPERATIONS_UNUSED" not in captured
    assert result.status == "invalid"  # stub intentionally writes no metric result


def test_fedot_local_ts_loader_orders_metocean_encoded_timestamps(tmp_path, monkeypatch):
    import hashlib
    from dataclasses import replace

    from fedotllm.agents.evolve.evaluation.openml_fold import load_fedot_local_ts

    checkout = tmp_path / "FEDOT"
    data_dir = checkout / "examples" / "real_cases" / "data" / "metocean"
    data_dir.mkdir(parents=True)
    source = data_dir / "metocean_data_train.csv"
    source.write_text(
        "datetime,sea_height\n"
        "200906180,30\n200906172,10\n200906179,20\n200906181,40\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEDOTLLM_REPO_PATH", str(checkout))
    dataset = replace(
        get_dataset("fedot-ts-metocean"),
        forecast_horizon=2,
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    fold = load_fedot_local_ts(dataset)
    assert list(fold["X_train"]) == [10.0, 20.0]
    assert list(fold["y_test"]) == [30.0, 40.0]


def test_worker_scores_regression_rmse_from_prediction():
    import numpy as np

    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import _score_regression

    actual = np.array([0.0, 2.0, 4.0])
    predicted = np.array([0.0, 1.0, 5.0])
    assert _score_regression(actual, predicted, "rmse") == pytest.approx(0.816496580927726)


def test_worker_uses_regression_fallback_when_fedot_metric_fails():
    import numpy as np

    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import _holdout_metrics

    class Model:
        prediction = np.array([0.0, 3.0])

        def get_metrics(self, **_kwargs):
            raise ValueError("metric unavailable")

    class Dataset:
        problem = "regression"
        metric = "rmse"

    score, metrics = _holdout_metrics(
        Model(), Dataset(), {"y_test": np.array([0.0, 1.0])}
    )
    assert score == pytest.approx(2**0.5)
    assert metrics["fallback"] == "sklearn_from_prediction"


def test_regression_quality_delta_is_relative_and_scale_independent():
    rows = []
    for task_id, stock, patched in (
        ("openml-regression-moneyball", 100.0, 98.0),
        ("openml-regression-abalone", 10.0, 9.8),
        ("fedot-ts-temperature", 1000.0, 980.0),
    ):
        rows.append(_pair(task_id, stock, patched))
    decision = decision_from_quality_jobs(rows)
    assert decision.keep is True
    assert decision.quality_outcome == "broad_positive"
    assert decision.target_delta is None
    assert all(value == pytest.approx(0.02) for value in decision.regression_deltas.values())


def test_quality_journal_keeps_hypothesis_scope_and_per_task_scores(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller import quality_executor
    from fedotllm.agents.evolve.types import PatchCandidate

    monkeypatch.setattr(
        quality_executor,
        "run_quality_jobs",
        lambda **_kwargs: [_pair("openml-regression-abalone", 10.0, 9.7)],
    )
    journal = tmp_path / "quality_jobs.jsonl"
    hypothesis = {
        "id": "h-42",
        "claim": "Avoid redundant feature transforms",
        "expected_effect": "Lower RMSE on regression",
    }
    scope = {
        "problem_families": ["regression"],
        "changed_files": ["fedot/core/operations/evaluation/regression.py"],
        "reasons": ["path:fedot/core/operations/evaluation/regression.py"],
        "confidence": "direct",
    }
    decision = quality_executor.measure_fedot_quality(
        tmp_path / "stock",
        tmp_path / "patched",
        journal=journal,
        candidate=PatchCandidate("candidate-42", "fedot/core/operations/evaluation/regression.py"),
        task_ids=("openml-regression-abalone",),
        hypothesis=hypothesis,
        quality_scope=scope,
    )
    row = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert decision.keep
    assert row["candidate"] == "candidate-42"
    assert row["hypothesis"] == hypothesis
    assert row["quality_scope"] == scope
    assert row["task_ids"] == ["openml-regression-abalone"]
    assert row["jobs"][0]["stock_score"] == 10.0
    assert row["jobs"][0]["patched_score"] == 9.7
    assert row["jobs"][0]["quality_delta"] == pytest.approx(0.03)


def test_quality_journal_links_each_side_to_worker_log_and_search_observations(
    tmp_path, monkeypatch
):
    from fedotllm.agents.evolve.controller import quality_executor
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths
    from fedotllm.agents.evolve.types import PatchCandidate

    pair = _pair("openml-regression-abalone", 10.0, 9.7)
    pair["stock_result"].metric_observations.update(
        {
            "search_detail": "search completed",
            "was_optimised": True,
            "worker_log_path": "/campaign/tmp/quality-log-abalone-stock.log",
        }
    )
    pair["patched_result"].metric_observations.update(
        {
            "search_detail": "search completed",
            "was_optimised": False,
            "worker_log_path": "/campaign/tmp/quality-log-abalone-patch.log",
            "failure_kind": "scoring_error",
            "failed_stage": "metrics",
            "stage_status_path": "/campaign/tmp/quality-out-abalone-patch.stages.json",
            "worker_return_code": 2,
        }
    )
    journal = tmp_path / "quality_jobs.jsonl"
    worker_root = tmp_path / "worker_artifacts"
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(worker_root))

    def fake_jobs(**kwargs):
        emit = kwargs["on_side_event"]
        spec = pair["spec"]
        kwargs["on_side_wait"](spec, "stock", "shared_stock_fit", 12.0, 1)
        emit(spec, "stock", "started", None, 1)
        emit(spec, "stock", "finished", pair["stock_result"], 1)
        saved = [json.loads(line) for line in journal.read_text().splitlines()]
        assert saved[-1]["event"] == "quality_task_side_finished"
        assert saved[-1]["side"] == "stock"
        emit(spec, "patch", "started", None, 1)
        emit(spec, "patch", "finished", pair["patched_result"], 1)
        return [pair]

    monkeypatch.setattr(quality_executor, "run_quality_jobs", fake_jobs)

    quality_executor.measure_fedot_quality(
        tmp_path / "stock",
        tmp_path / "patched",
        journal=journal,
        candidate=PatchCandidate("candidate-log-links"),
        task_ids=("openml-regression-abalone",),
    )

    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    side_events = [
        event for event in events
        if event["event"].startswith("quality_task_side_")
        and event["event"] != "quality_task_side_waiting_stock_cache"
    ]
    wait_event = next(
        event for event in events if event["event"] == "quality_task_side_waiting_stock_cache"
    )
    assert [event["event"] for event in side_events] == [
        "quality_task_side_started",
        "quality_task_side_finished",
        "quality_task_side_started",
        "quality_task_side_finished",
    ]
    assert [event["side"] for event in side_events] == ["stock", "stock", "patch", "patch"]
    assert all(event["n_jobs_per_side"] == 4 for event in side_events)
    assert all(event["cpu_quota"] == 28 for event in side_events)
    assert all(event["dataset_pair_slots"] == 7 for event in side_events)
    assert side_events[1]["attempt_id"] == "candidate-log-links:openml-regression-abalone:stock:1"
    expected_stock_paths = worker_artifact_paths(
        "openml-regression-abalone", "stock", "candidate-log-links", 1,
        root=worker_root,
        worker_invocation_id=side_events[0]["worker_invocation_id"],
    )
    assert side_events[0]["worker_paths"] == expected_stock_paths
    assert side_events[0]["artifact_state"] == "expected"
    assert side_events[0]["candidate_revision_id"] == "candidate-log-links"
    assert side_events[0]["proposal_id"] == "candidate-log-links"
    assert side_events[1]["result"]["worker_log_path"].endswith("stock.log")
    assert side_events[1]["worker_paths"]["worker_log_path"].endswith("stock.log")
    assert side_events[3]["result"]["metric_observations"]["failure_kind"] == "scoring_error"
    assert side_events[3]["result"]["metric_observations"]["failed_stage"] == "metrics"
    assert side_events[3]["result"]["metric_observations"]["stage_status_path"].endswith(".stages.json")
    assert side_events[3]["result"]["metric_observations"]["worker_return_code"] == 2
    assert wait_event["wait_kind"] == "shared_stock_fit"
    assert wait_event["wait_elapsed_seconds"] == 12.0
    assert wait_event["pair_slot_held_during_wait"] is False
    assert wait_event["campaign_pair_cap"] == 7
    assert wait_event["scheduler_active_pair_slots"] is None
    row = next(event for event in events if event["event"] == "fedot_quality")
    job = row["jobs"][0]
    assert job["stock_observations"]["search_detail"] == "search completed"
    assert job["stock_observations"]["worker_log_path"].endswith("stock.log")
    assert job["patched_observations"]["was_optimised"] is False
    assert job["patched_observations"]["worker_log_path"].endswith("patch.log")


def test_drain_does_not_emit_obsolete_stock_warm_event(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller import quality_executor
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    target = source / "fedot" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    patch = "--- a/fedot/x.py\n+++ b/fedot/x.py\n@@\n-value = 1\n+value = 2\n"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate("candidate-stock-log", file_path="fedot/x.py"),
        patch_text=patch,
        hint="test stock diagnostics",
        priority="normal",
    )

    monkeypatch.setattr(
        quality_executor,
        "measure_fedot_quality",
        lambda *_args, **_kwargs: Decision(
            False, "composing_did_not_start", None, infrastructure_error=True
        ),
    )
    events = []
    monkeypatch.setattr(
        quality_executor, "append_journal", lambda _path, row: events.append(row)
    )

    quality_executor.drain_quality_queue(
        source,
        workspace,
        task_ids=("fedot-ts-temperature",),
    )

    assert not any(row["event"] == "fedot_quality_stock" for row in events)
    assert any(row["event"] == "quality_disposition" for row in events)


def test_hourly_summary_keeps_per_side_search_details_and_log_links():
    from research.evolve.scripts.sync_hour_verdicts import _dataset_rows

    rows = _dataset_rows(
        [
            {
                "spec": {"source_dataset": "fedot-ts-temperature", "metric": "rmse"},
                "stock_status": "ok",
                "patched_status": "invalid",
                "stock_score": 12.0,
                "patched_score": None,
                "stock_detail": "",
                "patched_detail": "search did not start",
                "stock_observations": {
                    "search_ran": True,
                    "was_optimised": True,
                    "worker_log_path": "/snapshot/tmp/temperature-stock.log",
                },
                "patched_observations": {
                    "search_ran": False,
                    "search_detail": "composing_did_not_start",
                    "was_optimised": False,
                    "worker_log_path": "/snapshot/tmp/temperature-patch.log",
                },
            }
        ]
    )

    assert rows[0]["stock_search_ran"] is True
    assert rows[0]["patched_search_detail"] == "composing_did_not_start"
    assert rows[0]["patched_was_optimised"] is False
    assert rows[0]["stock_worker_log_path"].endswith("temperature-stock.log")
    assert rows[0]["patched_worker_log_path"].endswith("temperature-patch.log")
    assert rows[0]["patched_detail"] == "search did not start"


def test_hourly_candidate_report_shows_task_metrics_and_disposition(tmp_path):
    from research.evolve.scripts.sync_hour_verdicts import (
        _queue_meta,
        render_md,
        summarize,
        write_verdict_files,
    )

    queue = tmp_path / "quality_queue"
    queue.mkdir()
    proposal = {
        "candidate_id": "candidate-report-1",
        "file_path": "fedot/core/operations/evaluation/regression.py",
        "status": "drop",
        "reason": "no_quality_gain",
        "hypothesis": {
            "id": "h-17",
            "claim": "Avoid redundant regression transforms",
            "expected_effect": "Lower RMSE",
        },
        "quality_scope": {
            "problem_families": ["regression"],
            "reasons": ["path:regression.py"],
        },
        "task_ids": ["reg-a", "reg-b"],
        "quality_message": "Не принимаем: прироста нет на двух задачах.",
    }
    (queue / "candidate-report-1.json").write_text(
        json.dumps(proposal), encoding="utf-8"
    )

    journal = [
        {
            "event": "fedot_quality",
            "candidate": "candidate-report-1",
            "decision": {"keep": False, "reason": "no_quality_gain"},
            "jobs": [
                {
                    "spec": {
                        "source_dataset": "reg-a",
                        "metric": "rmse",
                        "higher_is_better": False,
                    },
                    "stock_status": "ok",
                    "patched_status": "ok",
                    "stock_score": 10.0,
                    "patched_score": 9.8,
                },
                {
                    "spec": {
                        "source_dataset": "reg-b",
                        "metric": "rmse",
                        "higher_is_better": False,
                    },
                    "stock_status": "ok",
                    "patched_status": "ok",
                    "stock_score": 5.0,
                    "patched_score": 5.1,
                },
            ],
        }
    ]
    summary = summarize(journal, queue)
    candidate = summary["candidates"]["candidate-report-1"]
    assert candidate["hypothesis"] == proposal["hypothesis"]
    assert candidate["quality_scope"] == proposal["quality_scope"]
    assert candidate["task_ids"] == proposal["task_ids"]
    assert candidate["quality_message"] == proposal["quality_message"]
    assert "Avoid redundant regression transforms" in render_md(summary)
    assert "Не принимаем: прироста нет на двух задачах." in render_md(summary)
    assert "| `reg-a` | rmse | ok | ok | 10 | 9.8 | 0.2 |" in render_md(summary)
    assert "| `reg-b` | rmse | ok | ok | 5 | 5.1 | -0.1 |" in render_md(summary)

    output = tmp_path / "verdicts"
    output.mkdir()
    verdict = write_verdict_files(summary, [], _queue_meta(queue), output)[0]
    payload = json.loads(verdict.read_text(encoding="utf-8"))
    assert payload["hypothesis"] == proposal["hypothesis"]
    assert payload["quality_scope"] == proposal["quality_scope"]
    assert payload["task_ids"] == proposal["task_ids"]
    assert payload["quality_message"] == proposal["quality_message"]


def test_drain_runs_ts_tasks_for_ts_patch_and_keeps_toy_delta_zero(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    (source / "fedot" / "core" / "operations" / "evaluation" / "operation_implementations" / "models" / "ts_implementations").mkdir(parents=True)
    target = (
        source
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "models"
        / "ts_implementations"
        / "statsmodels.py"
    )
    target.write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    rel = "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate(candidate_id="ar-toy-zero", file_path=rel),
        patch_text=(
            f"--- a/{rel}\n"
            f"+++ b/{rel}\n"
            "@@\n"
            "-value = 1\n"
            "+value = 2\n"
        ),
        hint="hunt_enqueue_no_toy_veto",
        priority="normal",
        probe_status="no_change",
        toy_metric="no_veto",
        task_ids=list_quality_task_ids(),
    )
    seen: list[tuple[str, ...]] = []

    def fake_measure(_source, _experiment, **kwargs):
        seen.append(("patch", kwargs["task_ids"]))
        return Decision(True, "quality_improved", 0.6, stage="quality")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        fake_measure,
    )
    journal_events = []
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda _path, event: journal_events.append(event),
    )
    payload = drain_quality_queue(source, workspace)
    assert payload["jobs"][0]["keep"] is True
    assert "Принимаем" in payload["jobs"][0]["quality_message"]
    assert payload["jobs"][0]["task_ids"] == list(_task_ids(load_quality_registry(), "ts_forecasting"))
    assert seen == [("patch", _task_ids(load_quality_registry(), "ts_forecasting"))]
    job = json.loads((workspace / "quality_queue" / "ar-toy-zero.json").read_text(encoding="utf-8"))
    assert job["status"] == "keep"
    assert job["probe_status"] == "no_change"
    assert "quality_message" in job
    disposition = next(event for event in journal_events if event["event"] == "quality_disposition")
    assert disposition["status"] == "keep"
    assert disposition["task_ids"] == job["task_ids"]
    assert "Принимаем" in disposition["quality_message"]


def test_old_skipped_entry_is_rechecked_on_its_required_datasets(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    rel = (
        "fedot/core/operations/evaluation/operation_implementations/"
        "models/ts_implementations/statsmodels.py"
    )
    target = source / rel
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate(candidate_id="ts-only", file_path=rel),
        patch_text=f"--- a/{rel}\n+++ b/{rel}\n@@\n-value = 1\n+value = 2\n",
        hint="ts family against tabular-only pool",
        priority="normal",
        task_ids=_task_ids(load_quality_registry(), "classification"),
    )
    queued_path = workspace / "quality_queue" / "ts-only.json"
    queued_item = json.loads(queued_path.read_text())
    queued_item["status"] = "skipped"
    queued_item["task_ids"] = []
    queued_path.write_text(json.dumps(queued_item))

    journal_events = []
    measured_task_ids = []
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda _path, event: journal_events.append(event),
    )

    def measure(_source, _tree, *, task_ids, **_kwargs):
        measured_task_ids.extend(task_ids)
        return Decision(False, "no_quality_gain", None, stage="quality")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        measure,
    )
    payload = drain_quality_queue(
        source,
        workspace,
        task_ids=_task_ids(load_quality_registry(), "classification"),
    )
    required_ts_ids = _task_ids(load_quality_registry(), "ts_forecasting")
    assert measured_task_ids == list(required_ts_ids)
    assert payload["jobs"][0]["status"] == "drop"
    assert payload["jobs"][0]["reason"] == "no_quality_gain"
    assert sum(event.get("event") == "quality_disposition" for event in journal_events) == 1

    resumed = drain_quality_queue(
        source,
        workspace,
        task_ids=_task_ids(load_quality_registry(), "classification"),
    )
    assert resumed["jobs"] == []
    assert sum(event.get("event") == "quality_disposition" for event in journal_events) == 1


def test_drain_picks_up_jobs_enqueued_after_start(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot" / "x.py").write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    patch = "--- a/fedot/x.py\n+++ b/fedot/x.py\n@@\n-value = 1\n+value = 2\n"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate(candidate_id="first-job", file_path="fedot/x.py"),
        patch_text=patch,
        hint="already queued",
        priority="high",
    )

    def fake_measure(_source, experiment, **_kwargs):
        if experiment.name == "first-job":
            enqueue_quality_job(
                workspace,
                candidate=PatchCandidate(candidate_id="late-job", file_path="fedot/x.py"),
                patch_text=patch,
                hint="arrived after drain started",
                priority="normal",
            )
        return Decision(True, "quality_improved", 0.02, stage="quality")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        fake_measure,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda *a, **k: None,
    )
    payload = drain_quality_queue(source, workspace)
    assert [job["candidate_id"] for job in payload["jobs"]] == ["first-job", "late-job"]
    assert all(job["keep"] for job in payload["jobs"])


def test_drain_evaluates_multiple_proposals_concurrently_with_shared_budget(tmp_path, monkeypatch):
    import threading

    from fedotllm.agents.evolve.controller.quality_executor import (
        drain_quality_queue,
        write_job_status as original_write_status,
    )
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    ts_path = "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
    cls_path = "fedot/core/data/data_split.py"
    for rel in (ts_path, cls_path):
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate("candidate-a", file_path=ts_path),
        patch_text=f"--- a/{ts_path}\n+++ b/{ts_path}\n@@\n-value = 1\n+value = 2\n",
        hint="first",
        priority="normal",
        task_ids=("fedot-ts-beer",),
    )
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate("candidate-b", file_path=cls_path),
        patch_text=f"--- a/{cls_path}\n+++ b/{cls_path}\n@@\n-value = 1\n+value = 2\n",
        hint="second",
        priority="normal",
        task_ids=("openml-31-credit-g",),
    )

    events = []
    scheduler_ids = set()
    events_lock = threading.Lock()
    both_started = threading.Barrier(2)
    journal_events = []

    def fake_status(job_path, status, extra=None):
        if status == "running":
            candidate_id = json.loads(job_path.read_text())["candidate_id"]
            with events_lock:
                events.append(("running", candidate_id))
        original_write_status(job_path, status, extra)

    def fake_measure(_source, _experiment, *, candidate=None, task_ids=None,
                     n_jobs=None, cpu_quota=None, pair_scheduler=None, **_kwargs):
        assert n_jobs == 4
        assert cpu_quota == 8
        assert pair_scheduler.max_pairs == 2
        scheduler_ids.add(id(pair_scheduler))
        both_started.wait(timeout=3)
        with events_lock:
            events.append(("patch", candidate.candidate_id, task_ids))
        return Decision(True, "quality_improved", 0.1, stage="quality")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.write_job_status", fake_status
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        fake_measure,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda _path, event: journal_events.append(event),
    )

    result = drain_quality_queue(source, workspace, n_jobs=4, cpu_quota=8)

    assert set(events[:2]) == {("running", "candidate-a"), ("running", "candidate-b")}
    assert set(events[2:]) == {
        ("patch", "candidate-a", ("fedot-ts-beer",)),
        ("patch", "candidate-b", ("openml-31-credit-g",)),
    }
    assert len(scheduler_ids) == 1
    assert [job["candidate_id"] for job in result["jobs"]] == ["candidate-a", "candidate-b"]
    scheduler_event = next(
        event for event in journal_events if event["event"] == "quality_scheduler_batch_started"
    )
    assert scheduler_event["parallel_proposals"] == 2
    assert scheduler_event["dataset_pair_slots"] == 2
    assert scheduler_event["cpu_slots"] == 8
    candidate_events = {
        event["candidate"]: event
        for event in journal_events
        if event["event"] == "quality_candidate_started"
    }
    assert set(candidate_events) == {"candidate-a", "candidate-b"}
    assert all(event["cpu_quota"] == 8 for event in candidate_events.values())
    assert all(event["n_jobs_per_side"] == 4 for event in candidate_events.values())
    assert all(event["dataset_pair_slots"] == 2 for event in candidate_events.values())
    assert all(event["resource_budget_scope"] == "campaign-shared" for event in candidate_events.values())


def test_drain_marks_executor_crash_failed_not_running(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import PatchCandidate

    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot" / "x.py").write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    enqueue_quality_job(
        workspace,
        candidate=PatchCandidate(candidate_id="crash-job", file_path="fedot/x.py"),
        patch_text="--- a/fedot/x.py\n+++ b/fedot/x.py\n@@\n-value = 1\n+value = 2\n",
        hint="executor crash should not stick in running",
        priority="normal",
    )


    def boom(*_args, **_kwargs):
        raise RuntimeError("fedot worker died")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        boom,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda *a, **k: None,
    )

    payload = drain_quality_queue(source, workspace, task_ids=list_quality_task_ids())
    job = json.loads((workspace / "quality_queue" / "crash-job.json").read_text(encoding="utf-8"))
    assert job["status"] == "failed"
    assert job["status"] != "running"
    assert "не завершена" in job["quality_message"]
    assert "не считать" in job["quality_message"]
    assert payload["jobs"][0]["status"] == "failed"
    assert "executor_error" in payload["jobs"][0]["reason"]
    assert "не завершена" in payload["jobs"][0]["quality_message"]


def test_quality_regression_drops():
    rows = [
        _pair("openml-31-credit-g", 0.80, 0.82),
        _pair("openml-10101-blood-transfusion", 0.70, 0.50),
    ]
    decision = decision_from_quality_jobs(rows)
    assert decision.keep is False
    assert "regression" in decision.reason


def test_worker_scores_roc_auc_from_predict_proba_without_labels():
    import numpy as np

    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import (
        _score_classification,
    )

    y = np.array([1, 0, 1, 0])
    proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])
    score = _score_classification(y, proba, "roc_auc")
    assert score == 1.0


def test_worker_classification_predict_path_skips_labels():
    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import _predict_holdout

    class Dataset:
        problem = "classification"

    class Model:
        def __init__(self):
            self.calls = []

        def predict(self, features):
            self.calls.append(("predict", features))
            raise AssertionError("labels predict must not run for classification")

        def predict_proba(self, features):
            self.calls.append(("predict_proba", features))
            return features

    model = Model()
    path = _predict_holdout(model, Dataset(), {"X_test": "fold-x"})
    assert path == "predict_proba"
    assert model.calls == [("predict_proba", "fold-x")]


def test_quality_resume_recovers_finished_sides_and_interrupted_attempt(tmp_path):
    import json

    from fedotllm.agents.evolve.controller.quality_executor import _recover_side_state

    journal = tmp_path / "quality_jobs.jsonl"
    events = [
        {
            "event": "quality_task_side_started",
            "candidate": "candidate-a",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "registry_identity": "registry-v1",
        },
        {
            "event": "quality_task_side_finished",
            "candidate": "candidate-a",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "registry_identity": "registry-v1",
                "result": {
                    "status": "ok",
                    "score": 0.61,
                    "seed": 42,
                    "cmd": "python worker.py",
                    "metric_observations": {
                        "metric_valid": True,
                        "worker_log_path": "/tmp/beer-stock.log",
                    },
            },
        },
        {
            "event": "quality_task_side_started",
            "candidate": "candidate-a",
            "task_id": "fedot-ts-beer",
            "side": "patch",
            "attempt": 1,
            "registry_identity": "registry-v1",
        },
    ]
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))

    results, attempts = _recover_side_state(
        journal,
        candidate_id="candidate-a",
        task_ids=("fedot-ts-beer",),
        registry_identity="registry-v1",
    )

    assert results[("fedot-ts-beer", "stock")].score == 0.61
    assert results[("fedot-ts-beer", "stock")].cmd == "python worker.py"
    assert results[("fedot-ts-beer", "patch")].status == "crash"
    assert "interrupted" in results[("fedot-ts-beer", "patch")].detail
    assert attempts == {("fedot-ts-beer", "stock"): 1, ("fedot-ts-beer", "patch"): 1}


def test_quality_queue_lock_rejects_concurrent_drain(tmp_path):
    from fedotllm.agents.evolve.controller.quality_executor import (
        QualityQueueAlreadyRunning,
        quality_queue_lock,
    )

    workspace = tmp_path / "campaign"
    workspace.mkdir()
    with quality_queue_lock(workspace):
        try:
            with quality_queue_lock(workspace):
                raise AssertionError("second controller unexpectedly acquired the lock")
        except QualityQueueAlreadyRunning:
            pass


def test_enqueue_freezes_revision_and_never_overwrites_changed_patch(tmp_path):
    from fedotllm.agents.evolve.controller.quality_queue import (
        enqueue_quality_job,
        write_job_status,
    )
    from fedotllm.agents.evolve.types import PatchCandidate

    workspace = tmp_path / "campaign"
    candidate = PatchCandidate("proposal-1", file_path="fedot/model.py")
    first_patch = "--- a/fedot/model.py\n+++ b/fedot/model.py\n@@\n-value = 1\n+value = 2\n"
    first_path = enqueue_quality_job(
        workspace,
        candidate=candidate,
        patch_text=first_patch,
        hint="candidate title",
        priority="high",
        hypothesis={"claim": "Improve model quality"},
    )
    frozen = json.loads(first_path.read_text(encoding="utf-8"))
    write_job_status(first_path, "drop", {"reason": "measured"})

    assert enqueue_quality_job(
        workspace,
        candidate=candidate,
        patch_text=first_patch,
        hint="repeated enqueue",
        priority="low",
        hypothesis={"claim": "Different text must not rewrite a frozen revision"},
    ) == first_path
    assert json.loads(first_path.read_text(encoding="utf-8"))["status"] == "drop"

    second_path = enqueue_quality_job(
        workspace,
        candidate=candidate,
        patch_text=first_patch.replace("value = 2", "value = 3"),
        hint="candidate title",
        priority="high",
        hypothesis={"claim": "Improve model quality"},
    )
    revised = json.loads(second_path.read_text(encoding="utf-8"))

    assert second_path != first_path
    assert frozen["patch_sha256"] != revised["patch_sha256"]
    assert revised["candidate_revision_id"] == revised["candidate_id"]
    assert revised["proposal_id"] == "proposal-1"
    assert revised["quality_scope_frozen"] is True
    assert revised["registry_identity"]
    assert json.loads(first_path.read_text(encoding="utf-8"))["patch_sha256"] == frozen["patch_sha256"]


def test_quality_pair_reuses_finished_sides_after_restart(monkeypatch):
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset
    from fedotllm.agents.evolve.types import ScoreResult

    task_id = "fedot-ts-beer"
    stock = ScoreResult(task_id=task_id, status="ok", score=0.81, cmd="cached stock")
    patched = ScoreResult(task_id=task_id, status="ok", score=0.83, cmd="cached patch")

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("completed task sides must be restored from the journal")

    monkeypatch.setattr(quality, "run_quality_side", should_not_run)
    events = []
    result = quality.run_quality_pair(
        get_dataset(task_id),
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
        prior_side_results={(task_id, "stock"): stock, (task_id, "patch"): patched},
        prior_attempt_counts={(task_id, "stock"): 1, (task_id, "patch"): 1},
        on_side_event=lambda _spec, side, phase, score, attempt: events.append(
            (side, phase, score.score, attempt)
        ),
    )

    assert result["stock_result"].score == 0.81
    assert result["patched_result"].score == 0.83
    assert events == [
        ("stock", "reused", 0.81, 1),
        ("patch", "reused", 0.83, 1),
    ]


def test_quality_pair_does_not_retry_timeout_or_deterministic_patch_crash(monkeypatch):
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset
    from fedotllm.agents.evolve.types import ScoreResult

    task_id = "fedot-ts-beer"
    calls = []

    def fake_side(_dataset, *, side, **_kwargs):
        calls.append(side)
        if side == "stock":
            return ScoreResult(task_id=task_id, status="timeout", score=float("nan"))
        return ScoreResult(
            task_id=task_id,
            status="crash",
            score=float("nan"),
            detail="patched code raised ValueError on this dataset",
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    result = quality.run_quality_pair(
        get_dataset(task_id),
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
    )

    assert calls == ["stock", "patch"]
    assert result["stock_result"].status == "timeout"
    assert result["patched_result"].status == "crash"


def test_quality_pair_never_retries_timeout_with_transient_marker(monkeypatch):
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset
    from fedotllm.agents.evolve.types import ScoreResult

    task_id = "fedot-ts-beer"
    calls = []

    def fake_side(_dataset, *, side, **_kwargs):
        calls.append(side)
        if side == "stock":
            return ScoreResult(
                task_id=task_id,
                status="timeout",
                score=float("nan"),
                detail="task_timeout after 3600s",
                log_tail="ConnectionError: transient backend reset",
            )
        return ScoreResult(task_id=task_id, status="ok", score=0.75)

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    result = quality.run_quality_pair(
        get_dataset(task_id),
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
    )

    assert calls == ["stock", "patch"]
    assert result["stock_result"].status == "timeout"
    assert result["patched_result"].score == 0.75


def test_quality_resume_reuses_finished_timeout_and_runs_opposite_side(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.controller.quality_executor import _recover_side_state
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset, load_quality_registry
    from fedotllm.agents.evolve.types import ScoreResult

    task_id = "fedot-ts-beer"
    journal = tmp_path / "quality_jobs.jsonl"
    journal.write_text(json.dumps({
        "event": "quality_task_side_started",
        "candidate": "candidate-timeout",
        "task_id": task_id,
        "side": "stock",
        "attempt": 1,
        "registry_identity": load_quality_registry().identity,
    }) + "\n" + json.dumps({
        "event": "quality_task_side_finished",
        "candidate": "candidate-timeout",
        "task_id": task_id,
        "side": "stock",
        "attempt": 1,
        "registry_identity": load_quality_registry().identity,
        "result": {"status": "timeout", "detail": "timeout after 3600s"},
    }) + "\n", encoding="utf-8")
    prior_results, attempts = _recover_side_state(
        journal,
        candidate_id="candidate-timeout",
        task_ids=(task_id,),
        registry_identity=load_quality_registry().identity,
    )

    calls = []

    def run_side(_dataset, *, side, **_kwargs):
        calls.append(side)
        assert side == "patch", "the completed stock timeout must be reused on resume"
        return ScoreResult(task_id=task_id, status="ok", score=0.75)

    monkeypatch.setattr(quality, "run_quality_side", run_side)
    events = []
    result = quality.run_quality_pair(
        get_dataset(task_id),
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
        prior_side_results=prior_results,
        prior_attempt_counts=attempts,
        on_side_event=lambda _spec, side, phase, score, attempt: events.append(
            (side, phase, score.status if score else None, attempt)
        ),
    )

    assert calls == ["patch"]
    assert result["stock_result"].status == "timeout"
    assert result["patched_result"].score == 0.75
    assert events == [
        ("stock", "reused", "timeout", 1),
        ("patch", "started", None, 1),
        ("patch", "finished", "ok", 1),
    ]


def test_quality_jobs_cap_concurrent_datasets_by_cpu_quota(monkeypatch):
    import threading
    import time

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry

    guard = threading.Lock()
    active = 0
    maximum = 0

    def fake_pair(task_id, **_kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return {"spec": {"source_dataset": task_id}}

    monkeypatch.setattr(quality, "run_quality_pair", fake_pair)
    ids = tuple(f"task-{index}" for index in range(8))
    rows = quality.run_quality_jobs(
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
        task_ids=ids,
        n_jobs=4,
        cpu_quota=28,
        registry=load_quality_registry(),
    )

    assert maximum == 7
    assert [row["spec"]["source_dataset"] for row in rows] == list(ids)


def test_drain_pauses_after_unresolved_infrastructure_failure(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import Decision, PatchCandidate

    source = tmp_path / "FEDOT"
    rel = "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
    target = source / rel
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    patch = f"--- a/{rel}\n+++ b/{rel}\n@@\n-value = 1\n+value = 2\n"
    for candidate_id in ("candidate-a", "candidate-b"):
        enqueue_quality_job(
            workspace,
            candidate=PatchCandidate(candidate_id, file_path=rel),
            patch_text=patch,
            hint="timeout gate",
            priority="normal",
            task_ids=("fedot-ts-beer",),
        )

    evaluated = []

    def fake_measure(_source, _experiment, *, candidate=None, task_ids=None, **_kwargs):
        evaluated.append(candidate.candidate_id)
        return Decision(
            keep=None,
            reason="quality fedot-ts-beer timeout after retry",
            target_delta=None,
            stage="quality",
            infrastructure_error=True,
            quality_outcome="not_evaluated",
            unresolved_task_ids=("fedot-ts-beer",),
        )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        fake_measure,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda *_args, **_kwargs: None,
    )

    result = drain_quality_queue(source, workspace, n_jobs=4, cpu_quota=4)
    first = json.loads((workspace / "quality_queue" / "candidate-a.json").read_text())
    second = json.loads((workspace / "quality_queue" / "candidate-b.json").read_text())
    assert evaluated == ["candidate-a"]
    assert first["status"] == "infrastructure_error"
    assert second["status"] == "queued"
    assert result["paused_for_retry"] is True


def test_quality_job_cli_loads_registry_for_stock_only(tmp_path, monkeypatch):
    from argparse import Namespace

    from fedotllm.agents.evolve.__main__ import _cmd_quality_job
    from fedotllm.agents.evolve.evaluation import fedot_quality, quality_registry
    from fedotllm.agents.evolve.execution import checkout

    monkeypatch.setattr(fedot_quality, "bind_stock_cache", lambda _path: tmp_path / "cache")
    monkeypatch.setattr(
        fedot_quality,
        "run_quality_stock_jobs",
        lambda **_kwargs: [
            {
                "spec": {"source_dataset": "fedot-ts-beer"},
                "search_ran": True,
                "stock": {"status": "ok", "score": 1.0},
            }
        ],
    )
    monkeypatch.setattr(quality_registry, "load_quality_registry", lambda: object())
    monkeypatch.setattr(
        quality_registry,
        "list_quality_task_ids",
        lambda _registry: ("fedot-ts-beer",),
    )
    monkeypatch.setattr(checkout, "resolve_fedot_src", lambda: tmp_path / "FEDOT")
    args = Namespace(
        patch_checkout=None,
        stock_only=True,
        workspace=tmp_path / "quality-job",
        stock_cache=None,
        tasks="fedot-ts-beer",
        n_jobs=4,
        cpu_quota=28,
    )

    assert _cmd_quality_job(args, parser=object()) == 0
    journal = (args.workspace / "quality_jobs.jsonl").read_text(encoding="utf-8")
    assert '"event": "fedot_quality_stock"' in journal

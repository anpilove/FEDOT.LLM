from __future__ import annotations

import json

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


def _task_ids(registry, problem: str) -> tuple[str, ...]:
    return tuple(item.task_id for item in registry.datasets if item.problem == problem)


def test_quality_registry_is_full_openml_and_fixed_before_scores():
    registry = load_quality_registry()
    ids = list_quality_task_ids(registry)
    tabular = _task_ids(registry, "classification")
    ts_ids = _task_ids(registry, "ts_forecasting")
    assert registry.timeout_seconds == 3600
    assert registry.preset == "best_quality"
    assert registry.with_tuning is True
    assert registry.portfolio is False
    assert registry.api == "fedot"
    assert registry.cpu_quota == 32
    assert registry.n_jobs_per_job == 8
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
    assert ts_ids == ("fedot-ts-beer", "fedot-ts-australia", "fedot-ts-salaries")
    assert ids == tabular + ts_ids
    assert len(tabular) == 24
    assert len(ids) >= 16
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
        assert spec["timeout_minutes"] == 60.0
        assert spec["require_search_ran"] is True
        assert spec["portfolio"] is False
        if dataset.source == "openml_task":
            assert dataset.problem == "classification"
            assert dataset.openml_task > 0
        else:
            assert dataset.source == "fedot_public_ts"
            assert dataset.problem == "ts_forecasting"
            assert dataset.forecast_horizon > 0
            assert spec["forecast_horizon"] == dataset.forecast_horizon


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


def test_drain_runs_stock_then_queued_patches_including_toy_delta_zero(tmp_path, monkeypatch):
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

    def fake_stock(**kwargs):
        calls.append("stock:" + ",".join(kwargs["task_ids"]))
        return [
            {
                "spec": {"source_dataset": task_id},
                "stock": {"status": "ok", "score": 0.7},
                "search_ran": True,
            }
            for task_id in kwargs["task_ids"]
        ]

    def fake_measure(source_checkout, experiment, **kwargs):
        calls.append("patch:" + experiment.name)
        return Decision(True, "quality_improved", 0.02, stage="quality")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.run_quality_stock_jobs",
        fake_stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        fake_measure,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda *a, **k: None,
    )

    payload = drain_quality_queue(
        source,
        workspace,
        task_ids=list_quality_task_ids(),
    )
    assert payload["stock_ok"] is True
    assert payload["task_ids"] == list(list_quality_task_ids())
    assert [job["candidate_id"] for job in payload["jobs"]] == ["svd-hint", "lgbm-toy"]
    assert all(job["keep"] for job in payload["jobs"])
    assert calls[0].startswith("stock:openml-31-credit-g")
    assert "patch:svd-hint" in calls
    assert "patch:lgbm-toy" in calls
    lgbm = json.loads((workspace / "quality_queue" / "lgbm-toy.json").read_text(encoding="utf-8"))
    assert lgbm["status"] == "keep"
    assert lgbm["probe_status"] == "no_change"


def test_stock_warm_skips_matching_cache_identity_including_leftover_njobs(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation import fedot_quality as fq

    cache = tmp_path / "stock_cache"
    fq.bind_stock_cache(cache)
    registry = load_quality_registry()
    dataset = get_dataset("openml-31-credit-g", registry)
    leftover = cache / f"{registry.identity}__{dataset.task_id}__n10__s{registry.seed}.json"
    leftover.write_text(
        json.dumps(
            {
                "task_id": dataset.task_id,
                "status": "ok",
                "score": 0.77,
                "seed": 42,
                "metric_observations": {
                    "search_ran": True,
                    "preset": "best_quality",
                    "timeout_seconds": 3600,
                },
            }
        ),
        encoding="utf-8",
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("matching stock cache must not launch Fedot")

    monkeypatch.setattr(fq, "run_worker", boom)
    result = fq.run_quality_side(
        dataset,
        checkout=tmp_path / "FEDOT",
        side="stock",
        n_jobs=8,
        registry=registry,
    )
    assert result.score == 0.77
    assert result.metric_observations["cache_hit"] is True
    rows = fq.run_quality_stock_jobs(
        stock_checkout=tmp_path / "FEDOT",
        task_ids=(dataset.task_id,),
        n_jobs=8,
        cpu_quota=8,
        registry=registry,
    )
    assert rows[0]["cache_hit"] is True
    assert rows[0]["search_ran"] is True


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


def test_quality_job_selects_tabular_or_ts_by_patch_path():
    registry = load_quality_registry()
    tabular = _task_ids(registry, "classification")
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
    both = select_task_ids_for_job({"file_path": "fedot/core/data/data_split.py"})
    assert both == tabular + ts_ids


def test_inapplicable_skip_is_not_a_quality_verdict():
    decision = decision_from_quality_jobs(
        [{"status": "skipped", "spec": {"source_dataset": "openml-31-credit-g"}}]
    )
    assert decision.keep is False
    assert decision.infrastructure_error is False
    assert decision.reason == "no_applicable_quality_tasks"


def test_fedot_public_ts_loader_holdout_dry_run(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation.openml_fold import load_fedot_public_ts

    checkout = tmp_path / "FEDOT"
    ts_dir = checkout / "examples" / "data" / "ts"
    ts_dir.mkdir(parents=True)
    rows = ["idx,value"] + [f"1956-{index:02d},{100 + index}" for index in range(1, 25)]
    (ts_dir / "beer.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("FEDOTLLM_REPO_PATH", str(checkout))
    fold = load_fedot_public_ts(get_dataset("fedot-ts-beer"))
    assert fold["n_train"] == 12
    assert fold["n_test"] == 12
    assert fold["forecast_horizon"] == 12
    assert fold["loader"] == "fedot.examples.data.ts"


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

    def fake_stock(**kwargs):
        seen.append(("stock", kwargs["task_ids"]))
        return [
            {
                "spec": {"source_dataset": task_id},
                "stock": {"status": "ok", "score": 1.2},
                "search_ran": True,
            }
            for task_id in kwargs["task_ids"]
        ]

    def fake_measure(_source, _experiment, **kwargs):
        seen.append(("patch", kwargs["task_ids"]))
        return Decision(True, "quality_improved", 0.6, stage="quality")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.run_quality_stock_jobs",
        fake_stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.measure_fedot_quality",
        fake_measure,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda *a, **k: None,
    )
    payload = drain_quality_queue(source, workspace)
    assert payload["jobs"][0]["keep"] is True
    assert payload["jobs"][0]["task_ids"] == list(_task_ids(load_quality_registry(), "ts_forecasting"))
    assert seen[0] == ("stock", _task_ids(load_quality_registry(), "ts_forecasting"))
    assert seen[1][0] == "patch"
    assert seen[1][1] == _task_ids(load_quality_registry(), "ts_forecasting")
    job = json.loads((workspace / "quality_queue" / "ar-toy-zero.json").read_text(encoding="utf-8"))
    assert job["status"] == "keep"
    assert job["probe_status"] == "no_change"


def test_drain_does_not_rewarm_pool_when_no_applicable_tasks(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import PatchCandidate

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
    stock_calls: list[tuple[str, ...]] = []

    def fake_stock(**kwargs):
        stock_calls.append(tuple(kwargs["task_ids"]))
        return []

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.run_quality_stock_jobs",
        fake_stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.append_journal",
        lambda *a, **k: None,
    )
    payload = drain_quality_queue(
        source,
        workspace,
        task_ids=_task_ids(load_quality_registry(), "classification"),
    )
    assert stock_calls == []
    assert payload["task_ids"] == []
    assert payload["jobs"][0]["status"] == "skipped"
    assert payload["jobs"][0]["reason"] == "inapplicable_tasks"


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

    def fake_stock(**kwargs):
        return [
            {
                "spec": {"source_dataset": task_id},
                "stock": {"status": "ok", "score": 0.7},
                "search_ran": True,
                "cache_hit": True,
            }
            for task_id in kwargs["task_ids"]
        ]

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
        "fedotllm.agents.evolve.controller.quality_executor.run_quality_stock_jobs",
        fake_stock,
    )
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

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.quality_executor.run_quality_stock_jobs",
        lambda **kwargs: [
            {
                "spec": {"source_dataset": task_id},
                "stock": {"status": "ok", "score": 0.7},
                "search_ran": True,
            }
            for task_id in kwargs["task_ids"]
        ],
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
    assert payload["jobs"][0]["status"] == "failed"
    assert "executor_error" in payload["jobs"][0]["reason"]


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

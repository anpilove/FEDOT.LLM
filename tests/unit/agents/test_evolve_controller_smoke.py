"""Acceptance uses the actual queue and worker protocol, with isolated small data."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fedotllm.agents.evolve.controller import quality_queue
from fedotllm.agents.evolve.evaluation import fedot_quality


def test_worker_artifact_validation_requires_nonempty_logs_and_json_inside_workspace(tmp_path):
    from fedotllm.agents.evolve.controller.smoke import validate_worker_artifacts

    workspace = tmp_path / "smoke"
    artifacts = workspace / "worker_artifacts"
    artifacts.mkdir(parents=True)
    log = artifacts / "quality-log-fixture.log"
    stage = artifacts / "quality-out-fixture.stages.json"
    log.write_text("worker started\n", encoding="utf-8")
    identity = {"proposal_id":"proposal-1", "task_id":"fixture", "side":"stock",
                "attempt":1, "attempt_id":"proposal-1:fixture:stock:1"}
    stage.write_text(json.dumps({"event":"finished", "last_stage":"metrics", **identity}),
                     encoding="utf-8")
    event = {
        "event":"quality_task_side_finished", "candidate":"proposal-1",
        **identity, "artifact_state":"finished",
        "worker_paths":{"worker_log_path":str(log), "stage_status_path":str(stage)},
        "result":{"status":"ok", "worker_log_path":str(log),
                  "metric_observations":{"stage_status_path":str(stage)}},
    }

    report = validate_worker_artifacts(workspace, [event])

    assert report["complete"]
    assert report["checked_workers"] == 1
    assert report["issues"] == []


@pytest.mark.parametrize("missing", ["worker_log", "stage_status"])
def test_worker_artifact_validation_reports_missing_required_file(tmp_path, missing):
    from fedotllm.agents.evolve.controller.smoke import validate_worker_artifacts

    workspace = tmp_path / "smoke"
    artifacts = workspace / "worker_artifacts"
    artifacts.mkdir(parents=True)
    log = artifacts / "quality-log-fixture.log"
    stage = artifacts / "quality-out-fixture.stages.json"
    log.write_text("worker started\n", encoding="utf-8")
    stage.write_text('{"event":"finished"}', encoding="utf-8")
    (artifacts / ("quality-log-fixture.log" if missing == "worker_log"
                  else "quality-out-fixture.stages.json")).unlink()
    event = {
        "event":"quality_task_side_finished", "candidate":"proposal-1",
        "task_id":"fixture", "side":"stock", "artifact_state":"finished",
        "worker_paths":{"worker_log_path":str(log), "stage_status_path":str(stage)},
        "result":{"status":"ok", "worker_log_path":str(log),
                  "metric_observations":{"stage_status_path":str(stage)}},
    }

    report = validate_worker_artifacts(workspace, [event])

    assert not report["complete"]
    assert any(issue["artifact"] == missing for issue in report["issues"])


def test_worker_artifact_validation_requires_event_identity_in_stage_sidecar(tmp_path):
    from fedotllm.agents.evolve.controller.smoke import validate_worker_artifacts

    workspace = tmp_path / "smoke"
    artifacts = workspace / "worker_artifacts"
    artifacts.mkdir(parents=True)
    log = artifacts / "worker.log"
    stage = artifacts / "worker.stages.json"
    log.write_text("worker started\n", encoding="utf-8")
    identity = {"proposal_id":"proposal-1", "task_id":"fixture", "side":"stock",
                "attempt":1, "attempt_id":"proposal-1:fixture:stock:1"}
    stage_identity = dict(identity)
    stage_identity.pop("task_id")
    stage.write_text(json.dumps({"event":"end", "last_stage":"metrics", **stage_identity}),
                     encoding="utf-8")

    report = validate_worker_artifacts(workspace, [{
        "event":"quality_task_side_finished", "candidate":"proposal-1",
        **identity,
        "worker_paths":{"worker_log_path":str(log), "stage_status_path":str(stage)},
        "result":{"status":"ok", "metric_observations":{"metric_valid":True}},
    }])

    assert not report["complete"]
    assert any(issue["artifact"] == "stage_status"
               and issue["reason"] == "identity_missing:task_id"
               for issue in report["issues"])


def test_worker_artifact_validation_rejects_event_without_attempt_identity(tmp_path):
    from fedotllm.agents.evolve.controller.smoke import validate_worker_artifacts

    workspace = tmp_path / "smoke"
    artifacts = workspace / "worker_artifacts"
    artifacts.mkdir(parents=True)
    log = artifacts / "worker.log"
    stage = artifacts / "worker.stages.json"
    log.write_text("worker started\n", encoding="utf-8")
    stage.write_text(json.dumps({"event":"end", "last_stage":"metrics",
                                 "proposal_id":"proposal-1", "task_id":"fixture",
                                 "side":"stock", "attempt":1,
                                 "attempt_id":"proposal-1:fixture:stock:1"}), encoding="utf-8")

    report = validate_worker_artifacts(workspace, [{
        "event":"quality_task_side_finished", "candidate":"proposal-1",
        "task_id":"fixture", "side":"stock", "attempt":1,
        "worker_paths":{"worker_log_path":str(log), "stage_status_path":str(stage)},
        "result":{"status":"ok", "metric_observations":{"metric_valid":True}},
    }])

    assert not report["complete"]
    assert any(issue["artifact"] == "event"
               and issue["reason"] == "identity_missing:attempt_id"
               for issue in report["issues"])


def test_worker_artifact_validation_does_not_claim_cache_hits_are_new_workers(tmp_path):
    from fedotllm.agents.evolve.controller.smoke import validate_worker_artifacts

    report = validate_worker_artifacts(tmp_path / "smoke", [{
        "event":"quality_task_side_finished", "candidate":"proposal-1",
        "task_id":"fixture", "side":"stock", "artifact_state":"stock_cache_hit",
        "worker_paths":{"worker_log_path":"/absent/log", "stage_status_path":"/absent/stage"},
        "result":{"status":"ok", "metric_observations":{"cache_hit":True}},
    }])

    assert not report["complete"]
    assert report["issues"] == []
    assert report["checked_workers"] == 0
    assert report["reused_results"] == 1


def test_worker_artifact_validation_does_not_require_artifacts_when_waiter_never_spawned(tmp_path):
    from fedotllm.agents.evolve.controller.smoke import validate_worker_artifacts

    workspace = tmp_path / "smoke"
    artifacts = workspace / "worker_artifacts"
    artifacts.mkdir(parents=True)
    report = validate_worker_artifacts(workspace, [{
        "event":"quality_task_side_finished", "candidate":"proposal-1",
        "task_id":"fixture", "side":"stock", "attempt":1,
        "artifact_state":"finished",
        "result":{"status":"timeout", "detail":"shared stock fit wait exceeded deadline",
                  "metric_observations":{"worker_spawned":False,
                                         "wait_kind":"shared stock fit"}},
    }])

    assert report["issues"] == []
    assert report["checked_workers"] == 0
    assert report["waited_without_worker"] == 1


def test_worker_artifact_validation_rejects_empty_log_invalid_stage_and_escape(tmp_path):
    from fedotllm.agents.evolve.controller.smoke import validate_worker_artifacts

    workspace = tmp_path / "smoke"
    artifacts = workspace / "worker_artifacts"
    artifacts.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("outside evidence\n", encoding="utf-8")
    empty_log = artifacts / "empty.log"
    empty_log.touch()
    bad_stage = artifacts / "bad.stages.json"
    bad_stage.write_text("not-json", encoding="utf-8")
    (artifacts / "escape.log").symlink_to(outside)
    events = [
        {"event":"quality_task_side_finished", "candidate":"proposal-1", "task_id":"fixture",
         "side":"stock", "attempt":1, "artifact_state":"finished",
         "worker_paths":{"worker_log_path":str(empty_log), "stage_status_path":str(bad_stage)},
         "result":{"status":"ok", "worker_log_path":str(empty_log),
                   "metric_observations":{"stage_status_path":str(bad_stage)}}},
        {"event":"quality_task_side_finished", "candidate":"proposal-1", "task_id":"fixture",
         "side":"patch", "attempt":1, "artifact_state":"finished",
         "worker_paths":{"worker_log_path":str(artifacts / "escape.log"),
                         "stage_status_path":str(bad_stage)},
         "result":{"status":"crash", "worker_log_path":str(artifacts / "escape.log"),
                   "metric_observations":{"stage_status_path":str(bad_stage)}}},
    ]

    report = validate_worker_artifacts(workspace, events)

    assert not report["complete"]
    reasons = {(issue["artifact"], issue["reason"]) for issue in report["issues"]}
    assert ("worker_log", "empty_file") in reasons
    assert ("worker_log", "outside_worker_artifacts") in reasons
    assert ("stage_status", "invalid_json:JSONDecodeError") in reasons


def test_smoke_data_is_small_repeatable_and_separate_from_quality_registry():
    from fedotllm.agents.evolve.evaluation.smoke_data import load_smoke_fold, smoke_registry
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset, load_quality_registry
    from fedotllm.agents.evolve.evaluation.quality_registry import job_spec

    registry = smoke_registry()
    assert registry.timeout_seconds == 180
    assert registry.watchdog_seconds == 600
    assert registry.n_jobs_per_job == 4
    assert not registry.with_tuning
    assert dict(registry.composer_options) == {"pop_size":4, "num_of_generations":2}
    assert {row.problem for row in registry.datasets} == {
        "classification", "regression", "ts_forecasting"}
    for dataset in registry.datasets:
        assert job_spec(dataset, registry=registry)["composer_options"] == {
            "pop_size":4, "num_of_generations":2}
        first = load_smoke_fold(dataset)
        second = load_smoke_fold(dataset)
        assert 0 < first["n_train"] <= 128
        assert 0 < first["n_test"] <= 32
        np.testing.assert_array_equal(first["X_train"], second["X_train"])
        np.testing.assert_array_equal(first["y_test"], second["y_test"])
        with pytest.raises(KeyError):
            get_dataset(dataset.task_id, load_quality_registry())
    full = load_quality_registry()
    assert full.timeout_seconds == 5400
    assert full.fedot_search_budget_seconds == 4500
    assert full.watchdog_seconds == 10800
    assert len(full.datasets) == 40


@pytest.mark.parametrize("value", [True, 2.5, "4"])
def test_quality_registry_rejects_coerced_composer_option_values(monkeypatch, value):
    from fedotllm.agents.evolve.evaluation import quality_registry

    raw = quality_registry._load_raw()
    raw["composer_options"] = {"pop_size":value}
    monkeypatch.setattr(quality_registry, "_load_raw", lambda: raw)

    with pytest.raises(ValueError, match="composer_options"):
        quality_registry.load_quality_registry()


@pytest.mark.parametrize("value", [[], "", 0, None])
def test_quality_registry_rejects_falsey_non_object_composer_options(monkeypatch, value):
    from fedotllm.agents.evolve.evaluation import quality_registry

    raw = quality_registry._load_raw()
    raw["composer_options"] = value
    monkeypatch.setattr(quality_registry, "_load_raw", lambda: raw)

    with pytest.raises(ValueError, match="composer_options"):
        quality_registry.load_quality_registry()


def test_quality_registry_pins_candidate_dispatcher_into_each_worker_spec(monkeypatch):
    from fedotllm.agents.evolve.evaluation import quality_registry

    raw = quality_registry._load_raw()
    raw["identity"] = "isolated-candidate-test-v1"
    raw["candidate_dispatcher"] = "isolated-candidate-process-v1"
    monkeypatch.setattr(quality_registry, "_load_raw", lambda: raw)

    registry = quality_registry.load_quality_registry()
    dataset = registry.datasets[0]
    spec = quality_registry.job_spec(dataset, registry=registry)

    assert registry.candidate_dispatcher == "isolated-candidate-process-v1"
    assert spec["candidate_dispatcher"] == "isolated-candidate-process-v1"
    assert spec["n_jobs"] == 4
    assert spec["fedot_search_budget_seconds"] == 4500
    assert spec["timeout_seconds"] == 5400
    assert spec["watchdog_seconds"] == 10800


def test_quality_registry_rejects_unknown_candidate_dispatcher(monkeypatch):
    from fedotllm.agents.evolve.evaluation import quality_registry

    raw = quality_registry._load_raw()
    raw["candidate_dispatcher"] = "unbounded-retry-v0"
    monkeypatch.setattr(quality_registry, "_load_raw", lambda: raw)

    with pytest.raises(ValueError, match="candidate evaluation dispatcher"):
        quality_registry.load_quality_registry()


PROTOCOL_WORKER = '''\
import json
from pathlib import Path
import sys

spec = json.loads(Path(sys.argv[sys.argv.index("--spec-json") + 1]).read_text())
out = Path(sys.argv[sys.argv.index("--out") + 1])
print("fixture worker started", flush=True)
broken = "raise RuntimeError" in (Path(__file__).parent / "fedot/__init__.py").read_text()
unexpected_stock = False
leak_sqlite = False
if leak_sqlite:
    out.with_suffix('.unexpected.sqlite').write_bytes(b'SQLite format 3\\x00')
if unexpected_stock and spec["source_dataset"] == "smoke-classification" and spec["side"] == "stock":
    flag = out.parent / "unexpected-failure-used"
    if not flag.exists():
        flag.touch()
        out.write_text(json.dumps({"status":"crash", "score":None, "detail":"unexpected GOLEM failure"}))
        raise SystemExit(0)
payload = {"task_id":spec["source_dataset"], "side":spec["side"],
           "attempt":spec["attempt"], "attempt_id":spec["attempt_id"],
           "worker_invocation_id":spec["worker_invocation_id"],
           "registry_identity":spec["registry_identity"],
           "evaluation_identity":spec.get("evaluation_identity"),
           "status":"crash" if broken else "ok",
           "score":None if broken else 0.75, "n_train":128, "n_test":32,
           "detail":"intentional controller smoke import failure" if broken else "",
           "failure_kind":"fedot_import_error" if broken else None,
           "failed_stage":"import" if broken else "",
           "last_stage":"import" if broken else "metrics",
           "metric_observations":{"metric_valid":not broken,
              "fedot_cache_policy":"disabled-sqlite-v1",
              "fedot_cache_flags":{"use_preprocessing_cache":False,"use_operations_cache":False,"use_predictions_cache":False}}}
out.write_text(json.dumps(payload))
stage = {"task_id":spec["source_dataset"], "side":spec["side"],
         "attempt":spec["attempt"], "attempt_id":spec["attempt_id"],
           "worker_invocation_id":spec["worker_invocation_id"],
         "proposal_id":spec["proposal_id"], "event":"failed" if broken else "end",
         "last_stage":"import" if broken else "metrics"}
out.with_suffix('.stages.json').write_text(json.dumps(stage))
'''


def test_smoke_advances_after_failed_proposal_and_recovers_without_refitting(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.smoke import run_smoke

    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/__init__.py").write_text('"""FEDOT fixture."""\n')
    (source / "smoke_protocol_fixture.py").write_text(PROTOCOL_WORKER)
    original = (source / "fedot/__init__.py").read_bytes()
    monkeypatch.setenv("FEDOTLLM_REPO_PYTHON", sys.executable)
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setattr(fedot_quality, "WORKER_MODULE", "smoke_protocol_fixture")
    workspace = tmp_path / "runs" / "isolated-smoke"
    report = run_smoke(source, workspace)
    assert report["passed"], report
    assert report["purpose"] == "controller_acceptance"
    assert report["failed_proposal_did_not_block"]
    assert report["recovery_without_refitting"]
    assert report["worker_artifacts_complete"]
    assert report["worker_artifacts_checked"] > 0
    assert report["worker_artifact_issues"] == []
    assert len(report["metrics"]) == 3
    assert report["queue_counts"] == {"candidate_error": 1, "drop": 1}
    assert (source / "fedot/__init__.py").read_bytes() == original
    assert not (tmp_path / "quality_queue").exists()
    assert not (tmp_path / "runs/quality_queue").exists()
    events = [json.loads(line) for line in (workspace / "check/quality_jobs.jsonl").read_text().splitlines()]
    assert len([row for row in events if row["event"] == "quality_task_side_reused"
                and row["candidate"] == "smoke-02-control"]) == 6
    with pytest.raises(FileExistsError):
        run_smoke(source, workspace)


def test_smoke_does_not_hide_unexpected_stock_failure(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.smoke import run_smoke

    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/__init__.py").write_text('"""FEDOT fixture."""\n')
    (source / "smoke_protocol_fixture.py").write_text(
        PROTOCOL_WORKER.replace("unexpected_stock = False", "unexpected_stock = True"))
    monkeypatch.setenv("FEDOTLLM_REPO_PYTHON", sys.executable)
    monkeypatch.setattr(fedot_quality, "WORKER_MODULE", "smoke_protocol_fixture")
    report = run_smoke(source, tmp_path / "smoke")
    assert not report["passed"]
    assert report["unexpected_failures"]


def test_smoke_rejects_sqlite_cache_created_despite_disabled_policy(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller.smoke import run_smoke
    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/__init__.py").write_text('"""FEDOT fixture."""\n')
    (source / "smoke_protocol_fixture.py").write_text(
        PROTOCOL_WORKER.replace("leak_sqlite = False", "leak_sqlite = True"))
    monkeypatch.setenv("FEDOTLLM_REPO_PYTHON", sys.executable)
    monkeypatch.setattr(fedot_quality, "WORKER_MODULE", "smoke_protocol_fixture")
    report = run_smoke(source, tmp_path / "smoke")
    assert not report["passed"]
    assert report["sqlite_cache_files"]


def test_real_golem_default_output_is_isolated_before_fedot_import(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import configure_worker_runtime
    import golem.core.paths as paths

    monkeypatch.setattr(paths, "default_data_dir", paths.default_data_dir)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    runtime = configure_worker_runtime(tmp_path / "attempt.json")
    assert paths.default_data_dir() == runtime["golem_data_dir"]
    assert runtime["golem_data_dir"] != "/tmp/GOLEM"
    assert os.environ["TMPDIR"] == runtime["temp_dir"]
    assert Path(runtime["golem_data_dir"]).is_dir()


def test_worker_runtime_path_is_unique_when_output_path_is_reused(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import configure_worker_runtime
    import golem.core.paths as paths

    monkeypatch.setattr(paths, "default_data_dir", paths.default_data_dir)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    out_path = tmp_path / "attempt.json"

    first = configure_worker_runtime(out_path)
    second = configure_worker_runtime(out_path)

    assert first["temp_dir"] != second["temp_dir"]
    assert Path(first["temp_dir"]).is_dir()
    assert Path(second["temp_dir"]).is_dir()


def test_controller_smoke_reserves_only_time_within_its_search_budget():
    from fedotllm.agents.evolve.evaluation.smoke_data import SMOKE_IDENTITY, smoke_registry

    registry = smoke_registry()

    assert registry.final_fit_reserve_seconds == 0
    assert 0 <= registry.final_fit_reserve_seconds < registry.fedot_search_budget_seconds
    assert SMOKE_IDENTITY == "controller-smoke-v5-fit-reserve-disabled-sqlite"


@pytest.mark.parametrize("problem", ["classification", "regression", "ts_forecasting"])
def test_registered_smoke_profile_reaches_fit_and_measures_holdout(
    tmp_path, monkeypatch, problem,
):
    from enum import Enum
    from types import ModuleType, SimpleNamespace

    from fedotllm.agents.evolve.evaluation import _fedot_quality_worker as worker
    from fedotllm.agents.evolve.evaluation.quality_registry import job_spec
    from fedotllm.agents.evolve.evaluation.smoke_data import smoke_registry

    task_types = Enum("TaskTypesEnum", ["classification", "regression", "ts_forecasting"])

    class ParamsRepository:
        @staticmethod
        def default_params_for_task(task):
            return {"use_operations_cache": True, "use_preprocessing_cache": True}

    class SmokeFedot:
        def __init__(self, **kwargs):
            self.params = kwargs
            self.api_composer = SimpleNamespace(
                operations_cache=None, preprocessing_cache=None,
                was_optimised=False, was_tuned=False,
            )
            self.history = None
            self.fitted = False

        def fit(self, *, features, target=None):
            assert len(features) == 128
            assert (target is None) == (problem == "ts_forecasting")
            self.fitted = True

        def predict(self, features):
            assert self.fitted

        predict_proba = predict

        def forecast(self):
            assert self.fitted

        def get_metrics(self, *, target, metric_names):
            assert self.fitted
            assert len(target) == (12 if problem == "ts_forecasting" else 32)
            return {metric_names[0]: 0.75}

    for module_name, attributes in (
        ("fedot.api.main", {"Fedot": SmokeFedot}),
        ("fedot.api.api_utils.api_params_repository", {"ApiParamsRepository": ParamsRepository}),
        ("fedot.core.repository.tasks", {
            "TaskTypesEnum": task_types, "TsForecastingParams": lambda **kwargs: kwargs,
        }),
    ):
        module = ModuleType(module_name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(worker, "configure_worker_runtime", lambda out: {"temp_dir": str(tmp_path)})
    registry = smoke_registry()
    dataset = next(row for row in registry.datasets if row.problem == problem)

    payload, _ = worker._evaluate_worker(job_spec(dataset, registry=registry), tmp_path / "result.json")

    assert payload["status"] == "ok", payload.get("detail")
    assert payload["score"] == 0.75
    assert payload["stage_timings"]["configure_fit_budget"]["status"] == "end"
    assert payload["stage_timings"]["fit"]["status"] == "end"
    assert payload["metric_observations"]["max_pipeline_fit_time_seconds"] == (
        50 if problem == "ts_forecasting" else 30
    )
    assert payload["metric_observations"]["max_pipeline_evaluation_policy"] == (
        "per-candidate-evaluation-v1"
    )


def test_smoke_cannot_be_drained_as_a_full_quality_run(tmp_path):
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.evaluation.smoke_data import smoke_registry
    from fedotllm.agents.evolve.types import PatchCandidate

    workspace = tmp_path / "check"
    path = quality_queue.enqueue_quality_job(
        workspace, candidate=PatchCandidate("smoke", "fedot/__init__.py"),
        patch_text="fixture", hint="smoke", priority="high", registry=smoke_registry())
    result = drain_quality_queue(tmp_path, workspace, continue_on_error=True)
    assert result["jobs"][0]["reason"] == "registry_mismatch"
    assert json.loads(path.read_text())["status"] == "blocked"


def test_worker_rejects_changed_smoke_profile():
    from fedotllm.agents.evolve.evaluation.smoke_data import smoke_registry
    from fedotllm.agents.evolve.evaluation.quality_registry import job_spec
    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import load_worker_fold

    registry = smoke_registry()
    spec = job_spec(registry.datasets[0], registry=registry)
    dataset, fold = load_worker_fold(spec)
    assert dataset.task_id == registry.datasets[0].task_id
    assert fold["n_train"] == 128
    for key, value in {"seed":43, "timeout_seconds":3600, "source_dataset":"openml-31"}.items():
        with pytest.raises((ValueError, KeyError)):
            load_worker_fold({**spec, key:value})


@pytest.mark.parametrize("via_symlink", [False, True])
def test_smoke_rejects_physical_workspace_inside_immutable_source(tmp_path, via_symlink):
    from fedotllm.agents.evolve.controller.smoke import run_smoke

    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    original = b'"""FEDOT fixture."""\n'
    (source / "fedot/__init__.py").write_bytes(original)
    alias = tmp_path / "alias"
    if via_symlink:
        alias.symlink_to(source, target_is_directory=True)
        workspace = alias / "smoke"
    else:
        alias.mkdir()
        workspace = alias / ".." / "FEDOT" / "smoke"
    with pytest.raises(ValueError, match="disjoint"):
        run_smoke(source, workspace)
    assert not (source / "smoke").exists()
    assert (source / "fedot/__init__.py").read_bytes() == original


@pytest.mark.parametrize("marker", ["service_manifest.json", "launch_audit.json"])
def test_server_service_root_under_runs_keeps_its_own_queue(tmp_path, marker):
    service = tmp_path / "evolve/runs/full-campaign"
    service.mkdir(parents=True)
    (service / marker).write_text("{}")
    assert quality_queue.campaign_workspace(service) == service
    hunt = service / "runs/hunt-01"
    hunt.mkdir(parents=True)
    assert quality_queue.campaign_workspace(hunt) == service

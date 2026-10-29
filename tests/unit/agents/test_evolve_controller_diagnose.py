"""Read-only run diagnosis classifies saved evidence without recovery actions."""

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from fedotllm.agents.evolve.controller.diagnose import diagnose_run


NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_fresh_heartbeat_without_worker_log_is_unconfirmed(tmp_path):
    workspace = tmp_path / "run"
    status_path = workspace / "logs/service_status.json"
    _write(status_path, {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain", "log": str(workspace / "logs/drain.log")},
        "pending_proposals": ["proposal-7"],
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_started", "candidate": "proposal-7",
        "task_id": "traffic", "side": "patch", "attempt": 2,
        "attempt_id": "proposal-7:traffic:patch:2", "ts": "2026-10-05T11:55:00+00:00",
    }) + "\n")
    stage_path = workspace / "worker_artifacts/quality-out-traffic-patch-abc.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-7", "attempt_id": "proposal-7:traffic:patch:2",
        "source_dataset": "traffic", "side": "patch", "attempt": 2,
        "active_stage": "fit", "last_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:58:30+00:00", "stage_elapsed_seconds": 600,
        "last_fedot_log_line": "generation 8 fitness evaluated",
        "cpu_accounting": {
            "source": "resource.getrusage",
            "scope": "worker_and_reaped_children",
            "children_supported": True,
            "accounted_cpu_core_equivalents": 2.25,
        },
        "worker_log_path": str(workspace / "worker_artifacts/quality-log-traffic-patch-abc.log"),
    })

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "progress_unconfirmed"
    assert report["supervisor_state"] == "heartbeat_fresh"
    assert report["worker_checks"][0]["state"] == "heartbeat_fresh"
    assert report["worker_checks"][0]["progress_state"] == "stage_heartbeat_only"
    assert report["worker_checks"][0]["stage"] == "fit"
    assert report["worker_checks"][0]["proposal_id"] == "proposal-7"
    assert report["worker_checks"][0]["task_id"] == "traffic"
    assert report["worker_checks"][0]["side"] == "patch"
    assert report["worker_checks"][0]["last_fedot_log_line"] == "generation 8 fitness evaluated"
    assert report["worker_checks"][0]["cpu_accounting"]["scope"] == "worker_and_reaped_children"
    assert report["worker_checks"][0]["cpu_accounting"]["accounted_cpu_core_equivalents"] == 2.25
    assert report["automatic_action"].startswith("none")


def test_diagnose_preserves_compute_pool_counts_separately_from_pair_callbacks(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running",
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    attempt_id = "proposal-7:traffic:stock:1"
    _write(workspace / "worker_artifacts/quality-out-traffic-stock-a1.stages.json", {
        "proposal_id": "proposal-7", "attempt_id": attempt_id,
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "last_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:59:30+00:00", "stage_elapsed_seconds": 100,
        "worker_log_path": str(workspace / "worker_artifacts/quality-log-traffic-stock-a1.log"),
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_worker_spawned", "candidate": "proposal-7",
        "proposal_id": "proposal-7", "task_id": "traffic", "side": "stock",
        "attempt": 1, "attempt_id": attempt_id, "campaign_pair_cap": 4,
        "scheduler_active_pair_slots": 0, "scheduler_queued_pair_tasks": 0,
        "scheduler_active_compute_workers": 4, "scheduler_peak_compute_workers": 4,
        "ts": "2026-10-05T11:58:00+00:00",
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    worker_check = report["worker_checks"][0]
    assert worker_check["scheduler_active_pair_slots"] == 0
    assert worker_check["scheduler_active_pair_callbacks"] == 0
    assert worker_check["scheduler_active_compute_workers"] == 4
    assert worker_check["scheduler_peak_compute_workers"] == 4


def test_hyperopt_completed_trial_is_reported_as_fedot_progress(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "heartbeat_at": "2026-10-14T20:51:32.362791+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    attempt_id = "proposal-7:fedot-ts-traffic:stock:1"
    worker_log = workspace / "worker_artifacts/quality-log-traffic-stock-run1.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text(
        "2026-10-14 20:46:39,074 - TPE using 1/1 trials with best loss 2847.547359\n"
    )
    _write(workspace / "worker_artifacts/quality-out-traffic-stock-run1.stages.json", {
        "proposal_id": "proposal-7", "attempt_id": attempt_id,
        "source_dataset": "fedot-ts-traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "last_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-14T20:51:41.332502+00:00",
        "stage_elapsed_seconds": 4681.0, "worker_log_path": str(worker_log),
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_worker_spawned", "candidate": "proposal-7",
        "proposal_id": "proposal-7", "task_id": "fedot-ts-traffic", "side": "stock",
        "attempt": 1, "attempt_id": attempt_id, "ts": "2026-10-14T19:34:00+00:00",
        "worker_paths": {"worker_log_path": str(worker_log)},
    }) + "\n")

    report = diagnose_run(
        workspace, now=datetime.fromisoformat("2026-10-14T20:52:30+00:00")
    )

    worker_check = report["worker_checks"][0]
    assert worker_check["evaluation_state"] == "progressing"
    assert worker_check["progress_state"] == "recent_fedot_output"
    assert worker_check["raw_optimizer_event_kind"] == "tuning_trial_completed"
    assert worker_check["raw_tuning_trials_completed"] == 1
    assert worker_check["raw_tuning_trials_total"] == 1


def test_diagnose_resolves_remote_worker_artifacts_inside_snapshot_directory(tmp_path):
    workspace = tmp_path / "snapshot"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "heartbeat_at": "2026-10-14T20:51:32.362791+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    attempt_id = "proposal-7:fedot-ts-traffic:stock:1"
    worker_name = "quality-log-fedot-ts-traffic-stock-run1.log"
    worker_log = workspace / "worker_artifacts" / worker_name
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text(
        "2026-10-14 20:46:39,074 - TPE using 1/1 trials with best loss 2847.547359\n"
    )
    stack_name = "quality-out-fedot-ts-traffic-stock-run1.stacks.txt"
    (workspace / "worker_artifacts" / stack_name).write_text(
        "--- stack dump at 2026-10-14T20:52:18+00:00 ---\n"
        "Thread 0x123 (most recent call first):\n"
        '  File "/fedot/core/pipeline.py", line 10, in fit\n'
    )
    _write(workspace / "worker_artifacts/quality-out-fedot-ts-traffic-stock-run1.stages.json", {
        "proposal_id": "proposal-7", "attempt_id": attempt_id,
        "source_dataset": "fedot-ts-traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-14T20:51:41.332502+00:00",
        "stage_elapsed_seconds": 4681.0,
        "worker_log_path": f"/remote/run/worker_artifacts/{worker_name}",
        "stack_trace_path": f"/remote/run/worker_artifacts/{stack_name}",
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_worker_spawned", "candidate": "proposal-7",
        "proposal_id": "proposal-7", "task_id": "fedot-ts-traffic", "side": "stock",
        "attempt": 1, "attempt_id": attempt_id, "ts": "2026-10-14T19:34:00+00:00",
        "worker_paths": {
            "worker_log_path": f"/remote/run/worker_artifacts/{worker_name}",
            "stack_trace_path": f"/remote/run/worker_artifacts/{stack_name}",
        },
    }) + "\n")

    report = diagnose_run(
        workspace, now=datetime.fromisoformat("2026-10-14T20:52:30+00:00")
    )

    worker_check = report["worker_checks"][0]
    assert worker_check["worker_log"] == str(worker_log)
    assert worker_check["raw_optimizer_event_kind"] == "tuning_trial_completed"
    assert worker_check["evaluation_state"] == "progressing"
    assert worker_check["stack_trace_artifact_state"] == "present"


def test_diagnose_reads_flattened_child_manifest_only_for_exact_attempt(tmp_path):
    workspace = tmp_path / "snapshot"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "heartbeat_at": "2026-10-14T20:51:32+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    attempt_id = "proposal-7:fedot-ts-traffic:stock:1"
    stack_name = "process-50086-child.stacks.txt"
    (workspace / "worker_artifacts" / stack_name).parent.mkdir(parents=True)
    (workspace / "worker_artifacts" / stack_name).write_text(
        "--- stack dump at 2026-10-14T20:52:18+00:00 ---\n"
        "Thread 0x123 (most recent call first):\n"
        '  File "/sklearn/tree/_classes.py", line 100, in _fit\n'
    )
    remote_manifest = "/remote/run/worker_artifacts/quality-out-traffic-stock-x.process-traces/manifest.jsonl"
    (workspace / "worker_artifacts/manifest.jsonl").write_text(json.dumps({
        "attempt_id": attempt_id, "pid": 50086, "owner_pid": 50000,
        "stack_trace_path": f"/remote/run/worker_artifacts/quality-out-traffic-stock-x.process-traces/{stack_name}",
        "started_at": "2026-10-14T20:50:00+00:00", "capture_method": "python_thread",
    }) + "\n" + json.dumps({
        "attempt_id": "other-proposal:other-task:stock:1", "pid": 777,
        "stack_trace_path": f"/remote/run/worker_artifacts/quality-out-other.process-traces/{stack_name}",
    }) + "\n")
    attempt_root = workspace / "worker_artifacts/quality-out-traffic-stock-x.stages.json"
    _write(attempt_root, {
        "proposal_id": "proposal-7", "attempt_id": attempt_id,
        "source_dataset": "fedot-ts-traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-14T20:51:41+00:00", "stage_elapsed_seconds": 4681,
        "process_trace_manifest_path": remote_manifest,
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_worker_spawned", "candidate": "proposal-7",
        "proposal_id": "proposal-7", "task_id": "fedot-ts-traffic", "side": "stock",
        "attempt": 1, "attempt_id": attempt_id, "ts": "2026-10-14T19:34:00+00:00",
        "worker_paths": {"process_trace_manifest_path": remote_manifest},
    }) + "\n")

    report = diagnose_run(
        workspace, now=datetime.fromisoformat("2026-10-14T20:52:30+00:00")
    )

    trace = report["worker_checks"][0]["child_trace_evidence"]
    assert trace["manifest_state"] == "present"
    assert len(trace["processes"]) == 1
    assert trace["processes"][0]["attempt_id"] == attempt_id
    assert "_fit" in trace["processes"][0]["stack_trace_tail"]


def test_spawned_stock_owner_keeps_stage_and_stack_after_wait_heartbeat(tmp_path):
    """Legacy owner wait telemetry cannot conceal a spawned worker's evidence."""
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    attempt_id = "proposal-owner:fedot-ts-beer:stock:1"
    stack_path = workspace / "quality-out-beer.stacks.txt"
    stack_path.write_text(
        'Timeout (0:00:20)!\nThread 0x0000 (most recent call first):\n'
        '  File "/fedot/api/main.py", line 100, in fit\n',
    )
    _write(workspace / "worker_artifacts/quality-out-beer.stages.json", {
        "proposal_id": "proposal-owner", "attempt_id": attempt_id,
        "source_dataset": "fedot-ts-beer", "side": "stock", "attempt": 1,
        "active_stage": "fit", "last_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T09:00:00+00:00",
        "stack_trace_path": str(stack_path),
    })
    common = {
        "candidate": "proposal-owner", "proposal_id": "proposal-owner",
        "task_id": "fedot-ts-beer", "side": "stock", "attempt": 1,
        "attempt_id": attempt_id, "stock_flight_id": "shared-beer",
        "worker_paths": {"stack_trace_path": str(stack_path)},
    }
    events = [
        {**common, "event": "quality_task_worker_spawned", "worker_pid": 123,
         "ts": "2026-10-05T08:59:00+00:00"},
        {**common, "event": "quality_task_side_waiting_stock_cache",
         "wait_kind": "shared_stock_fit", "wait_elapsed_seconds": 100,
         "ts": "2026-10-05T11:59:30+00:00"},
        {**common, "candidate": "proposal-subscriber", "proposal_id": "proposal-subscriber",
         "attempt_id": "proposal-subscriber:fedot-ts-beer:stock:1",
         "event": "quality_task_side_waiting_stock_cache", "wait_kind": "shared_stock_fit",
         "wait_elapsed_seconds": 100, "ts": "2026-10-05T11:59:30+00:00"},
    ]
    (workspace / "quality_jobs.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    report = diagnose_run(workspace, now=NOW)
    checks = {row["proposal_id"]: row for row in report["worker_checks"]}
    owner = checks["proposal-owner"]
    assert owner["worker_launch_state"] == "spawned"
    assert owner["stage"] == "fit"
    assert owner["stack_trace_artifact_state"] == "present"
    assert owner["stack_dump_count"] == 1
    assert owner["state"] == "stale_worker", "spawn alone must not prove current activity"
    assert checks["proposal-subscriber"]["worker_launch_state"] == "not_spawned_waiting"


def test_shared_stock_cache_wait_is_explained_without_claiming_worker_started(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running",
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_waiting_stock_cache",
        "candidate": "proposal-2",
        "proposal_id": "proposal-2",
        "task_id": "fedot-ts-traffic",
        "side": "stock",
        "attempt": 1,
        "wait_kind": "shared_stock_fit",
        "wait_elapsed_seconds": 75.0,
        "artifact_state": "waiting_stock_cache",
        "worker_paths": {"worker_log_path": "/run/not-started.log"},
        "ts": "2026-10-05T11:59:30+00:00",
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "waiting_on_shared_stock"
    assert report["worker_checks"][0]["state"] == "waiting_on_shared_stock"
    assert report["worker_checks"][0]["wait_kind"] == "shared_stock_fit"
    assert report["worker_checks"][0]["wait_elapsed_seconds"] == 75.0
    assert report["worker_checks"][0]["worker_launch_state"] == "not_spawned_waiting"


def test_scheduler_queue_wait_is_not_reported_as_fedot_fit(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running",
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_waiting_stock_cache",
        "candidate": "proposal-2",
        "proposal_id": "proposal-2",
        "task_id": "fedot-ts-traffic",
        "side": "stock",
        "attempt": 1,
        "wait_kind": "scheduler_queue",
        "wait_elapsed_seconds": 120.0,
        "artifact_state": "waiting_stock_cache",
        "worker_paths": {"worker_log_path": "/run/not-started.log"},
        "ts": "2026-10-05T11:59:30+00:00",
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "waiting_in_scheduler_queue"
    assert report["worker_checks"][0]["state"] == "waiting_in_scheduler_queue"
    assert report["worker_checks"][0]["evaluation_state"] == "waiting_for_scheduler_slot"
    assert report["worker_checks"][0]["wait_kind"] == "scheduler_queue"
    assert report["worker_checks"][0]["worker_launch_state"] == "not_spawned_waiting"


def test_stack_trace_artifact_is_linked_to_the_active_attempt(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running",
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    attempt_id = "proposal-2:fedot-ts-beer:stock:1"
    stack_path = workspace / "quality-out-beer.stacks.txt"
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    stack_path.write_text(
        "periodic stack sampler started\n--- stack dump at 2026-10-05T11:59:20+00:00 ---\n"
        "Thread 0x0000 (most recent call first):\nlegacy snapshot\n"
        "Timeout (0:00:20)!\n"
        "Thread 0x0000 (most recent call first):\n"
        "Thread 0x0000 (most recent call first):\nmodel.fit\n",
        encoding="utf-8",
    )
    stage_path = workspace / "worker_artifacts/quality-out-beer.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-2", "attempt_id": attempt_id,
        "source_dataset": "fedot-ts-beer", "side": "stock", "attempt": 1,
        "active_stage": "fit", "last_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:59:40+00:00",
        "stack_trace_path": "/remote/run/quality-out-beer.stacks.txt",
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_worker_spawned", "candidate": "proposal-2",
        "proposal_id": "proposal-2", "task_id": "fedot-ts-beer", "side": "stock",
        "attempt": 1, "attempt_id": attempt_id,
        "worker_paths": {"stack_trace_path": "/remote/run/quality-out-beer.stacks.txt"},
        "ts": "2026-10-05T11:58:00+00:00",
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    worker_check = report["worker_checks"][0]
    assert worker_check["stack_trace_path"] == "/remote/run/quality-out-beer.stacks.txt"
    assert worker_check["stack_trace_artifact_state"] == "present"
    assert worker_check["stack_dump_count"] == 2
    assert worker_check["last_stack_dump_at"] == "2026-10-05T11:59:20+00:00"
    assert worker_check["last_faulthandler_timeout"] == "0:00:20"


def test_shared_stock_wait_older_than_threshold_is_flagged_not_interrupted(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running",
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_waiting_stock_cache",
        "candidate": "proposal-2",
        "task_id": "fedot-ts-traffic",
        "side": "stock",
        "attempt": 1,
        "wait_kind": "shared_stock_fit",
        "wait_elapsed_seconds": 3601.0,
        "ts": "2026-10-05T11:59:30+00:00",
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "stale_shared_stock_wait"
    assert report["worker_checks"][0]["state"] == "stale_shared_stock_wait"
    assert report["automatic_action"].startswith("none")


def test_terminal_service_does_not_report_last_stock_wait_as_active(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "completed",
        "child_running": False,
        "active_child": None,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_waiting_stock_cache",
        "candidate": "proposal-2",
        "task_id": "fedot-ts-traffic",
        "side": "stock",
        "attempt": 1,
        "wait_kind": "shared_stock_fit",
        "wait_elapsed_seconds": 30.0,
        "ts": "2026-10-05T11:00:00+00:00",
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "terminal"
    assert report["worker_checks"][0]["state"] == "incomplete_after_service_exit"
    assert report["worker_checks"][0]["evaluation_state"] == "interrupted_wait_after_service_exit"
    assert report["worker_checks"][0]["progress_state"] == "last_observed_wait_before_service_exit"


def test_old_wait_event_age_is_added_to_reported_cache_wait(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running",
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_waiting_stock_cache",
        "candidate": "proposal-2",
        "task_id": "fedot-ts-traffic",
        "side": "stock",
        "attempt": 1,
        "wait_kind": "shared_stock_fit",
        "wait_elapsed_seconds": 30.0,
        "ts": "2026-10-05T11:00:00+00:00",
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "stale_shared_stock_wait"
    assert report["worker_checks"][0]["wait_elapsed_seconds"] == 30.0
    assert report["worker_checks"][0]["wait_elapsed_seconds_at_snapshot"] == 3630.0


def test_terminal_dataset_check_does_not_call_unattested_score_measured(tmp_path):
    from fedotllm.agents.evolve.controller.diagnose import _terminal_dataset_checks

    events = [{
        "event": "quality_task_side_finished",
        "candidate": "proposal-3",
        "task_id": "fedot-ts-traffic",
        "side": "stock",
        "attempt": 1,
        "result": {"status": "ok", "score": 0.82, "metric_observations": {}},
    }]
    checks, _scope_complete = _terminal_dataset_checks(tmp_path, events, [])

    stock_check = next(row for row in checks if row["side"] == "stock")
    assert stock_check["metric_state"] == "unverified"
    assert stock_check["metric_valid"] is False
    assert stock_check["score"] == 0.82


def test_terminal_stock_cache_wait_timeout_is_reported_without_worker_log(tmp_path):
    from fedotllm.agents.evolve.controller.diagnose import _terminal_dataset_checks

    events = [{
        "event": "quality_task_side_finished",
        "candidate": "proposal-4",
        "task_id": "fedot-ts-traffic",
        "side": "stock",
        "attempt": 1,
        "worker_spawned": False,
        "artifact_state": "timed_out_waiting_stock_cache",
        "wait_kind": "stock cache file lock",
        "result": {
            "status": "timeout",
            "score": None,
            "detail": "stock cache file lock wait exceeded side deadline",
            "metric_observations": {
                "failure_kind": "timeout",
                "timeout_kind": "stock cache file lock",
                "wait_kind": "stock cache file lock",
                "worker_spawned": False,
                "side_elapsed_seconds": 3600.0,
                "timeout_limit_seconds": 3600,
            },
        },
    }]

    checks, _scope_complete = _terminal_dataset_checks(tmp_path, events, [])

    stock_check = next(row for row in checks if row["side"] == "stock")
    assert stock_check["metric_state"] == "missing"
    assert stock_check["timeout_kind"] == "stock cache file lock"
    assert stock_check["wait_kind"] == "stock cache file lock"
    assert stock_check["worker_spawned"] is False
    assert stock_check["side_elapsed_seconds"] == 3600.0
    assert stock_check["timeout_limit_seconds"] == 3600
    assert stock_check["worker_log"] is None


def test_spawned_event_keeps_exact_worker_paths_before_sidecar_exists(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    log_path = workspace / "worker_artifacts/quality-log-traffic-stock-attempt2.log"
    stage_path = workspace / "worker_artifacts/quality-out-traffic-stock-attempt2.stages.json"
    journal = workspace / "quality_jobs.jsonl"
    started_event = {
        "event": "quality_task_side_started", "candidate": "revision-2",
        "candidate_revision_id": "revision-2", "proposal_id": "proposal-7",
        "patch_revision": "patch-2", "task_id": "traffic", "side": "stock",
        "attempt": 2, "attempt_id": "revision-2:traffic:stock:2",
        "ts": "2026-10-05T11:58:00+00:00", "artifact_state": "expected",
        "worker_paths": {"worker_log_path": str(log_path),
                          "stage_status_path": str(stage_path)},
    }
    spawned_event = {
        **started_event,
        "event": "quality_task_worker_spawned",
        "ts": "2026-10-05T11:58:01+00:00",
        "artifact_state": "spawned",
    }
    journal.write_text("\n".join(json.dumps(row) for row in (started_event, spawned_event)) + "\n")

    report = diagnose_run(workspace, now=NOW)

    worker = report["worker_checks"][0]
    assert worker["state"] == "unknown"
    assert worker["proposal_id"] == "proposal-7"
    assert worker["candidate_revision_id"] == "revision-2"
    assert worker["patch_revision"] == "patch-2"
    assert worker["worker_log"] == str(log_path)
    assert worker["stage_file"] == str(stage_path)
    assert worker["artifact_state"] == "spawned"
    assert worker["worker_launch_state"] == "spawned"


def test_diagnose_resolves_remote_worker_paths_inside_flat_local_snapshot(tmp_path):
    workspace = tmp_path / "full_run_snapshot"
    _write(workspace / "service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    remote_root = Path(
        "/home/anpilov.kirill2/FEDOT.LLM/evolve/runs/run/worker_artifacts"
    )
    remote_log = remote_root / "quality-log-traffic-stock-abc.log"
    remote_stage = remote_root / "quality-out-traffic-stock-abc.stages.json"
    local_log = workspace / remote_log.name
    local_stage = workspace / remote_stage.name
    local_log.write_text("EvoGraphOptimizer - Generation num: 2 size: 4\n")
    _write(local_stage, {
        "proposal_id": "proposal-12", "attempt_id": "proposal-12:traffic:stock:1",
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:58:30+00:00", "stage_elapsed_seconds": 600,
        "last_fedot_log_line": "Generation num: 2 size: 4",
        "last_optimizer_event_at": "2026-10-05T11:59:00+00:00",
        "worker_log_path": str(remote_log),
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_worker_spawned", "candidate": "proposal-12",
        "proposal_id": "proposal-12", "task_id": "traffic", "side": "stock",
        "attempt": 1, "attempt_id": "proposal-12:traffic:stock:1",
        "ts": "2026-10-05T11:55:00+00:00", "artifact_state": "spawned",
        "worker_paths": {"worker_log_path": str(remote_log),
                          "stage_status_path": str(remote_stage)},
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    worker = report["worker_checks"][0]
    assert report["state"] == "progressing"
    assert worker["worker_log"] == str(local_log)
    assert worker["stage_file"] == str(local_stage)
    assert worker["worker_log_remote_path"] == str(remote_log)


@pytest.mark.parametrize("layout", ["nested", "copied_invocation", "flat"])
@pytest.mark.parametrize("explicit_stage_path", [False, True])
def test_diagnose_reads_only_the_named_worker_invocation(
    tmp_path, layout, explicit_stage_path,
):
    """A saved invocation keeps stage, log and stack evidence after SCP."""
    workspace = tmp_path / "snapshot"
    _write(workspace / "service_status.json", {
        "state": "running", "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    invocation = "a" * 32
    remote = Path("/remote/run/worker_artifacts") / invocation
    local = {
        "nested": workspace / "worker_artifacts" / invocation,
        "copied_invocation": workspace / invocation,
        "flat": workspace,
    }[layout]
    identity = {
        "proposal_id": "proposal-12", "attempt_id": "proposal-12:traffic:stock:1",
        "task_id": "traffic", "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "worker_invocation_id": invocation, "registry_identity": "registry-current",
        "evaluation_identity": {"preset": "current"},
    }
    stage_name = "quality-out-traffic-stock-exact.stages.json"
    log_name = "quality-log-traffic-stock-exact.log"
    stack_name = "quality-out-traffic-stock-exact.stacks.txt"
    _write(local / stage_name, {
        **identity, "active_stage": "fit", "updated_at": "2026-10-05T11:59:30+00:00",
        "worker_log_path": str(remote / log_name),
        "stack_trace_path": str(remote / stack_name),
    })
    (local / log_name).write_text("2026-10-05 11:59:40 - Generation num: 3 size: 4\n")
    (local / stack_name).write_text(
        "--- stack dump at 2026-10-05T11:59:45+00:00 ---\n"
        "Thread 0x123 (most recent call first):\n"
        '  File "/fedot/core/pipeline.py", line 10, in fit\n'
    )
    paths = {"worker_invocation_id": invocation, "worker_log_path": str(remote / log_name)}
    if explicit_stage_path:
        paths["stage_status_path"] = str(remote / stage_name)
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        **identity, "event": "quality_task_worker_spawned", "candidate": "proposal-12",
        "ts": "2026-10-05T11:55:00+00:00", "worker_paths": paths,
    }) + "\n")
    # Even a newer sidecar cannot replace the exact journaled artifact path.
    _write(workspace / "worker_artifacts/quality-out-traffic-stock-foreign.stages.json", {
        **identity, "worker_invocation_id": invocation if explicit_stage_path else "b" * 32,
        "active_stage": "predict", "updated_at": "2026-10-05T11:59:55+00:00",
    })

    report = diagnose_run(workspace, now=NOW)

    worker = report["worker_checks"][0]
    assert worker["stage"] == "fit"
    assert worker["state"] == "heartbeat_fresh"
    assert worker["stage_file"] == str(local / stage_name)
    assert worker["worker_log"] == str(local / log_name)
    assert worker["raw_optimizer_generation"] == 3
    assert worker["stack_trace_artifact_state"] == "present"


@pytest.mark.parametrize("mismatched_field", [
    "worker_invocation_id", "registry_identity", "evaluation_identity",
])
def test_diagnose_rejects_stage_from_another_execution_identity(tmp_path, mismatched_field):
    """Matching attempt IDs cannot authorize another invocation or evaluation."""
    workspace = tmp_path / "snapshot"
    _write(workspace / "service_status.json", {
        "state": "running", "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    identity = {
        "proposal_id": "proposal-12", "attempt_id": "proposal-12:traffic:stock:1",
        "task_id": "traffic", "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "worker_invocation_id": "a" * 32, "registry_identity": "registry-current",
        "evaluation_identity": {"preset": "current"},
    }
    stage_path = workspace / "quality-out-traffic-stock-exact.stages.json"
    _write(stage_path, {
        **identity, mismatched_field: {"preset": "previous"}
        if mismatched_field == "evaluation_identity" else "b" * 32,
        "active_stage": "fit", "updated_at": "2026-10-05T11:59:30+00:00",
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        **identity, "event": "quality_task_worker_spawned", "candidate": "proposal-12",
        "ts": "2026-10-05T11:55:00+00:00",
        "worker_paths": {"stage_status_path": str(stage_path)},
    }) + "\n")

    worker = diagnose_run(workspace, now=NOW)["worker_checks"][0]

    assert worker["stage"] is None
    assert worker["state"] == "unknown"


def test_recent_optimizer_event_and_existing_log_are_reported_as_progress(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_started", "candidate": "proposal-10",
        "task_id": "traffic", "side": "stock", "attempt": 1,
        "attempt_id": "proposal-10:traffic:stock:1", "ts": "2026-10-05T11:55:00+00:00",
    }) + "\n")
    worker_log = workspace / "worker_artifacts/quality-log-traffic-stock-run2.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text("EvoGraphOptimizer - Generation num: 2 size: 4\n")
    stage_path = workspace / "worker_artifacts/quality-out-traffic-stock-run2.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-10", "attempt_id": "proposal-10:traffic:stock:1",
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:58:30+00:00", "stage_elapsed_seconds": 600,
        "last_fedot_log_line": "Generation num: 2 size: 4",
        "last_fedot_log_at": "2026-10-05T11:59:00+00:00",
        "last_optimizer_event_at": "2026-10-05T11:59:00+00:00",
        "worker_log_path": str(worker_log),
    })

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "progressing"
    assert report["worker_checks"][0]["progress_state"] == "recent_fedot_output"
    assert report["worker_checks"][0]["progress_evidence_source"] == "sidecar_optimizer_event_claim"


def test_fresh_stage_heartbeat_with_stale_fedot_log_is_stale_evaluation(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_started", "candidate": "proposal-9",
        "task_id": "traffic", "side": "stock", "attempt": 1,
        "attempt_id": "proposal-9:traffic:stock:1", "ts": "2026-10-05T11:55:00+00:00",
    }) + "\n")
    worker_log = workspace / "worker_artifacts/quality-log-traffic-stock-run1.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text("2026-10-05 09:59:00,000 - EvoGraphOptimizer - Generation num: 1 size: 4\n")
    stage_path = workspace / "worker_artifacts/quality-out-traffic-stock-run1.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-9", "attempt_id": "proposal-9:traffic:stock:1",
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:58:30+00:00", "stage_elapsed_seconds": 600,
        "last_fedot_log_line": "Generation num: 1 size: 4",
        "last_fedot_log_at": "2026-10-05T09:59:00+00:00",
        "worker_log_path": str(worker_log),
    })

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "stale_evaluation"
    assert report["worker_checks"][0]["state"] == "heartbeat_fresh"
    assert report["worker_checks"][0]["progress_state"] == "log_quiet"
    assert report["worker_checks"][0]["evaluation_state"] == "stale_evaluation"
    assert report["worker_checks"][0]["fedot_log_age_seconds"] == 7260
    assert report["worker_checks"][0]["sidecar_fedot_log_age_seconds"] == 7260
    assert report["worker_checks"][0]["raw_fedot_log_age_seconds"] == 7260


def test_fresh_worker_heartbeat_with_stale_fit_log_is_stale_evaluation_signal(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    worker_log = workspace / "worker_artifacts/quality-log-traffic-stock-attempt1.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text("Generation num: 1 size: 4\n")
    stage_path = workspace / "worker_artifacts/quality-out-traffic-stock-attempt1.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-11", "attempt_id": "proposal-11:traffic:stock:1",
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:59:30+00:00", "stage_elapsed_seconds": 7200,
        "last_fedot_log_line": "Generation num: 1 size: 4",
        "last_fedot_log_at": "2026-10-05T09:59:00+00:00",
        "last_optimizer_event_at": "2026-10-05T09:59:00+00:00",
        "fedot_phase": "post_search_fit",
        "fedot_phase_at": "2026-10-05T10:20:00+00:00",
        "fedot_phase_line": "Model generation finished",
        "worker_log_path": str(worker_log),
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_started", "candidate": "proposal-11",
        "task_id": "traffic", "side": "stock", "attempt": 1,
        "attempt_id": "proposal-11:traffic:stock:1", "ts": "2026-10-05T09:59:00+00:00",
        "worker_paths": {"worker_log_path": str(worker_log), "stage_status_path": str(stage_path)},
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    worker = report["worker_checks"][0]
    assert report["state"] == "stale_evaluation"
    assert worker["state"] == "heartbeat_fresh"
    assert worker["evaluation_state"] == "stale_evaluation"
    assert worker["stage"] == "fit"
    assert worker["fedot_phase"] == "post_search_fit"
    assert worker["fedot_phase_at"] == "2026-10-05T10:20:00+00:00"
    assert worker["fedot_phase_line"] == "Model generation finished"
    assert worker["fedot_log_age_seconds"] == 7260
    assert worker["last_fedot_log_line"] == "Generation num: 1 size: 4"
    assert worker["sidecar_last_fedot_log_line"] == "Generation num: 1 size: 4"
    assert worker["raw_last_fedot_log_line"] is None
    assert report["automatic_action"].startswith("none")


def test_raw_worker_log_supersedes_older_sidecar_claim_without_claiming_new_generation(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    worker_log = workspace / "worker_artifacts/quality-log-traffic-stock-attempt1.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text(
        "2026-10-05 11:30:00,000 - EvoGraphOptimizer - Generation num: 1 size: 4\n"
        "2026-10-05 11:59:00,000 - warnings - optional dependency notice\n"
        "EvoGraphOptimizer - Generation num: 1 size: 4\n"
    )
    stage_path = workspace / "worker_artifacts/quality-out-traffic-stock-attempt1.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-raw", "attempt_id": "proposal-raw:traffic:stock:1",
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:59:30+00:00", "stage_elapsed_seconds": 7200,
        "last_fedot_log_line": "MultiprocessingDispatcher - Number of used CPU's: 4",
        "last_fedot_log_at": "2026-10-05T09:59:00+00:00",
        "last_optimizer_event_at": "2026-10-05T09:59:00+00:00",
        "worker_log_path": str(worker_log),
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_started", "candidate": "proposal-raw",
        "proposal_id": "proposal-raw", "task_id": "traffic", "side": "stock",
        "attempt": 1, "attempt_id": "proposal-raw:traffic:stock:1",
        "ts": "2026-10-05T09:59:00+00:00",
        "worker_paths": {"worker_log_path": str(worker_log), "stage_status_path": str(stage_path)},
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    worker = report["worker_checks"][0]
    assert worker["sidecar_fedot_log_age_seconds"] == 7260
    assert worker["raw_fedot_log_age_seconds"] == 1800
    assert worker["fedot_log_age_seconds"] == 1800
    assert worker["raw_optimizer_event_at"] == "2026-10-05T11:30:00+00:00"
    assert worker["raw_optimizer_event_kind"] == "generation"
    assert worker["raw_last_fedot_log_line"].endswith("optional dependency notice") is False
    assert worker["raw_last_timestamped_log_line"].endswith("optional dependency notice")
    assert worker["sidecar_last_fedot_log_line"].startswith("MultiprocessingDispatcher")
    assert worker["last_fedot_log_line"].endswith("Generation num: 1 size: 4")
    assert worker["evaluation_state"] == "progressing"
    assert worker["raw_optimizer_generation"] == 1


def test_window_selector_log_is_fit_activity_not_optimizer_progress(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:30+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    worker_log = workspace / "worker_artifacts/quality-log-traffic-stock-attempt1.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text(
        "2026-10-05 11:59:00,000 - SparseLaggedTransformationImplementation - "
        "Window size of lagged transformation was changed by WindowSizeSelector from 0 to 614\n"
    )
    stage_path = workspace / "worker_artifacts/quality-out-traffic-stock-attempt1.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-fit", "attempt_id": "proposal-fit:traffic:stock:1",
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:59:30+00:00", "stage_elapsed_seconds": 600,
        "worker_log_path": str(worker_log),
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_side_started", "candidate": "proposal-fit",
        "proposal_id": "proposal-fit", "task_id": "traffic", "side": "stock",
        "attempt": 1, "attempt_id": "proposal-fit:traffic:stock:1",
        "ts": "2026-10-05T11:55:00+00:00",
        "worker_paths": {"worker_log_path": str(worker_log), "stage_status_path": str(stage_path)},
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    worker = report["worker_checks"][0]
    assert report["state"] == "fit_activity_observed"
    assert worker["progress_state"] == "recent_fedot_output"
    assert worker["evaluation_state"] == "fit_activity_observed"
    assert worker["raw_optimizer_event_kind"] == "fit_operation"
    assert worker["raw_optimizer_generation"] is None


def test_diagnose_reports_stale_supervisor_and_exact_stale_worker(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T10:00:00+00:00",
        "active_child": {"kind": "quality-drain", "log": str(workspace / "logs/drain.log")},
    })
    with (workspace / "quality_jobs.jsonl").open("w") as handle:
        handle.write(json.dumps({
            "event": "quality_task_side_started", "candidate": "proposal-7",
            "task_id": "traffic", "side": "patch", "attempt": 2,
            "attempt_id": "proposal-7:traffic:patch:2", "ts": "2026-10-05T10:00:01+00:00",
        }) + "\n")
    stage_path = workspace / "worker_artifacts/quality-out-traffic-patch-abc.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-7", "attempt_id": "proposal-7:traffic:patch:2",
        "source_dataset": "traffic", "side": "patch", "attempt": 2,
        "active_stage": "fit", "event": "heartbeat", "updated_at": "2026-10-05T10:00:10+00:00",
        "worker_log_path": str(workspace / "worker_artifacts/quality-log-traffic-patch-abc.log"),
    })

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "stale_supervisor"
    assert report["supervisor_state"] == "stale_supervisor"
    assert report["worker_checks"][0]["state"] == "stale_worker"
    assert report["worker_checks"][0]["stage"] == "fit"
    assert report["worker_checks"][0]["worker_log"].endswith("quality-log-traffic-patch-abc.log")


def test_diagnose_marks_terminal_without_claiming_measurement(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "storage_error", "completion": "incomplete",
        "child_running": False, "active_child": None,
    })
    (workspace / "quality_jobs.jsonl").write_text("")

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "terminal"
    assert report["service_state"] == "storage_error"
    assert "Heartbeat показывает только активность" in report["interpretation"]


def test_diagnose_uses_latest_attempt_after_a_retry(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00", "active_child": {"kind": "quality-drain"},
    })
    events = [
        {"event": "quality_task_side_started", "candidate": "p", "task_id": "d",
         "side": "patch", "attempt": 1, "attempt_id": "p:d:patch:1", "ts": "2026-10-05T11:00:00+00:00"},
        {"event": "quality_task_side_started", "candidate": "p", "task_id": "d",
         "side": "patch", "attempt": 2, "attempt_id": "p:d:patch:2", "ts": "2026-10-05T11:30:00+00:00"},
        {"event": "quality_task_side_finished", "candidate": "p", "task_id": "d",
         "side": "patch", "attempt": 2, "attempt_id": "p:d:patch:2", "ts": "2026-10-05T11:45:00+00:00"},
    ]
    (workspace / "quality_jobs.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "quality_jobs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "unknown"
    assert report["worker_checks"] == []


def test_terminal_report_lists_each_missing_or_failed_dataset_side(tmp_path):
    workspace = tmp_path / "finished-run"
    _write(workspace / "logs/service_status.json", {
        "state": "completed", "completion": "incomplete",
        "child_running": False, "active_child": None,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
    })
    (workspace / "quality_queue").mkdir(parents=True)
    _write(workspace / "quality_queue/proposal-a.json", {
        "candidate_id": "proposal-a", "proposal_id": "proposal-a",
        "task_ids": ["traffic", "metocean"],
    })
    events = [
        {
            "event": "quality_task_side_finished", "candidate": "proposal-a",
            "proposal_id": "proposal-a", "task_id": "traffic", "side": "stock",
            "attempt": 1, "attempt_id": "proposal-a:traffic:stock:1",
            "ts": "2026-10-05T10:00:00+00:00",
            "result": {"status": "ok", "score": 0.8, "duration_s": 12,
                       "metric_observations": {"metric_valid": True,
                                                "worker_log_path": "/remote/traffic-stock.log"}},
        },
        {
            "event": "quality_task_side_finished", "candidate": "proposal-a",
            "proposal_id": "proposal-a", "task_id": "traffic", "side": "patch",
            "attempt": 1, "attempt_id": "proposal-a:traffic:patch:1",
            "ts": "2026-10-05T10:01:00+00:00",
            "result": {"status": "timeout", "score": None, "duration_s": 3600,
                       "detail": "task_timeout after 3600s",
                       "metric_observations": {"failure_kind": "timeout",
                                                "timeout_kind": "task_timeout",
                                                "failed_stage": "fit",
                                                "worker_log_path": "/remote/traffic-patch.log",
                                                "stage_status_path": "/remote/traffic-patch.stages.json"}},
        },
        {
            "event": "quality_task_side_finished", "candidate": "proposal-a",
            "proposal_id": "proposal-a", "task_id": "metocean", "side": "stock",
            "attempt": 1, "attempt_id": "proposal-a:metocean:stock:1",
            "ts": "2026-10-05T10:02:00+00:00",
            "result": {"status": "ok", "score": 0.7, "duration_s": 15,
                       "metric_observations": {"metric_valid": True}},
        },
    ]
    (workspace / "quality_jobs.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    report = diagnose_run(workspace, now=NOW)

    checks = {(row["task_id"], row["side"]): row for row in report["dataset_checks"]}
    assert checks[("traffic", "stock")]["metric_state"] == "measured"
    assert checks[("traffic", "stock")]["score"] == 0.8
    assert checks[("traffic", "patch")]["metric_state"] == "missing"
    assert checks[("traffic", "patch")]["result_status"] == "timeout"
    assert checks[("traffic", "patch")]["failure_kind"] == "timeout"
    assert checks[("traffic", "patch")]["stage"] == "fit"
    assert checks[("traffic", "patch")]["worker_log"] == "/remote/traffic-patch.log"
    assert checks[("metocean", "stock")]["metric_state"] == "measured"
    assert checks[("metocean", "patch")]["result_status"] == "no_terminal_side_event"
    assert checks[("metocean", "patch")]["metric_state"] == "missing"


def test_fit_without_optimizer_marker_is_reported_as_unconfirmed_stale(tmp_path):
    workspace = tmp_path / "run"
    _write(workspace / "logs/service_status.json", {
        "state": "running", "child_running": True,
        "heartbeat_at": "2026-10-05T11:59:00+00:00",
        "active_child": {"kind": "quality-drain"},
    })
    worker_log = workspace / "worker_artifacts/quality-log-traffic-stock-attempt1.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text("worker fit profile cv_folds=3\n")
    stage_path = workspace / "worker_artifacts/quality-out-traffic-stock-attempt1.stages.json"
    _write(stage_path, {
        "proposal_id": "proposal-13", "attempt_id": "proposal-13:traffic:stock:1",
        "source_dataset": "traffic", "side": "stock", "attempt": 1,
        "active_stage": "fit", "event": "heartbeat",
        "updated_at": "2026-10-05T11:59:30+00:00", "stage_elapsed_seconds": 7200,
        "worker_log_path": str(worker_log),
    })
    (workspace / "quality_jobs.jsonl").write_text(json.dumps({
        "event": "quality_task_worker_spawned", "candidate": "proposal-13",
        "task_id": "traffic", "side": "stock", "attempt": 1,
        "attempt_id": "proposal-13:traffic:stock:1", "ts": "2026-10-05T09:59:00+00:00",
        "worker_paths": {"worker_log_path": str(worker_log), "stage_status_path": str(stage_path)},
    }) + "\n")

    report = diagnose_run(workspace, now=NOW)

    assert report["state"] == "stale_evaluation_unconfirmed"
    assert report["worker_checks"][0]["evaluation_state"] == "fit_without_optimizer_evidence"

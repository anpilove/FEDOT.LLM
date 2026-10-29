"""Service acceptance at the named subprocess boundary; no FEDOT or LLM calls."""

import importlib
import json
import sys
import threading
import time

import pytest

from fedotllm.agents.evolve.execution.process import run_worker as real_run_worker


@pytest.fixture
def service():
    name = "fedotllm.agents.evolve.controller.supervisor"
    assert importlib.util.find_spec(name) is not None, "unattended service is missing"
    return importlib.import_module(name)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    (path / "fedot").mkdir(parents=True)
    (path / "fedot" / "model.py").write_text("VERSION = 1\n")
    return path


def enqueue(workspace, proposal="a"):
    directory = workspace / "quality_queue"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{proposal}.json").write_text(
        json.dumps(
            {
                "candidate_id": proposal,
                "proposal_id": proposal,
                "status": "queued",
                "task_ids": ["example"],
            }
        )
    )


@pytest.fixture
def named_child(tmp_path, monkeypatch, service):
    path = tmp_path / "quality_service_fixture.py"
    path.write_text("""import json, os, sys, time
from pathlib import Path
workspace = Path(sys.argv[1]); kind = sys.argv[2]
def save(path, item):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(item))
    temporary.replace(path)
with (workspace / "credential_flags.jsonl").open("a") as audit:
    audit.write(json.dumps({"kind": kind, "key_present": bool(os.environ.get("FEDOTLLM_LLM_API_KEY"))}) + "\\n")
if kind == "run":
    queue = workspace / "quality_queue"; queue.mkdir(exist_ok=True)
    item = {"candidate_id": "generated", "proposal_id": "generated", "status": "queued", "task_ids": ["example"]}
    save(queue / "generated.json", item)
    raise SystemExit(0)
if (workspace / "crash_once").exists():
    (workspace / "crash_once").unlink(); raise SystemExit(17)
if (workspace / "sleep_child").exists():
    time.sleep(30)
if (workspace / "hang_forever").exists():
    raise SystemExit(17)
if (workspace / "skip_scope").exists():
    for path in (workspace / "quality_queue").glob("*.json"):
        item = json.loads(path.read_text())
        if item["status"] == "queued":
            item.update(status="skipped", reason="inapplicable_tasks", task_ids=[])
            save(path, item)
    raise SystemExit(3)
if (workspace / "disk_full").exists():
    print("OSError: [Errno 28] No space left on device", flush=True)
    raise SystemExit(1)
if (workspace / "progress_once").exists():
    (workspace / "progress_once").unlink()
    with (workspace / "quality_jobs.jsonl").open("a") as journal:
        journal.write(json.dumps({"event": "quality_task_side_finished", "candidate": "a", "task_id": "example", "side": "stock", "result": {"status": "ok", "score": 1.0, "metric_observations": {"metric_valid": True}}}) + "\\n")
    raise SystemExit(17)
for path in (workspace / "quality_queue").glob("*.json"):
    item = json.loads(path.read_text())
    if item["status"] != "queued": continue
    terminal_status = workspace / "terminal_status"
    item["status"] = terminal_status.read_text() if terminal_status.exists() else ("timeout" if (workspace / "error_outcome").exists() else "keep")
    save(path, item)
    if item["status"] == "keep":
        with (workspace / "quality_jobs.jsonl").open("a") as journal:
            for side in ["stock", "patch"]:
                journal.write(json.dumps({"event": "quality_task_side_finished", "candidate": item["candidate_id"], "task_id": "example", "side": side, "result": {"status": "ok", "score": 1.0, "metric_observations": {"metric_valid": True}}}) + "\\n")
""")

    def boundary(command, **kwargs):
        kind = command[3]
        expected_module = "fedotllm.agents.evolve.controller" if kind == "drain" else "fedotllm.agents.evolve"
        assert command[:3] == [sys.executable, "-m", expected_module]
        if kind == "drain":
            kind = "quality-drain"
        assert kind in {"run", "quality-drain"}
        if kind == "quality-drain":
            assert "--continue-on-error" in command
        else:
            assert "--unlimited-llm-budget" not in command
        workspace = command[command.index("--workspace") + 1]
        return real_run_worker([sys.executable, str(path), workspace, kind], **kwargs)

    monkeypatch.setattr(service, "run_worker", boundary)
    monkeypatch.setattr(service, "fedot_python", lambda _: sys.executable)
    return path


def configuration(service, **overrides):
    values = dict(
        duration_seconds=2,
        poll_seconds=0.02,
        controller_timeout_seconds=1,
        target_proposals=1,
        max_controller_restarts=2,
    )
    values.update(overrides)
    return service.QualityServiceConfig(**values)


def test_disk_full_stops_without_retry_and_latches_across_service_resume(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "disk_full").touch()
    config = configuration(service)
    for _ in range(2):
        result = service.run_quality_service(source, workspace, config=config)
        assert result["state"] == "storage_error"
        assert result["completion"] == "incomplete"
        assert result["controller_launches"] == 1
        assert result["child_running"] is False
        assert result["active_child"] is None
    assert json.loads((workspace / "quality_queue/a.json").read_text())["status"] == "queued"
    assert len(list((workspace / "logs").glob("controller-*.log"))) == 1


def test_unsaved_storage_failure_in_final_launch_log_prevents_resume(
    service, source, tmp_path, named_child, monkeypatch
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "sleep_child").touch()
    original = service._write_durable
    def failing_storage(path, payload):
        if path.name == "service_status.json" and (
            payload.get("child_running") or payload.get("state") == "storage_error"
        ):
            raise OSError(28, "No space left on device")
        return original(path, payload)
    monkeypatch.setattr(service, "_write_durable", failing_storage)
    config = configuration(service, controller_timeout_seconds=30)
    failure = service.run_quality_service(source, workspace, config=config)
    assert failure["state"] == "storage_error"
    # This is the fixed CLI's final record after an unsavable status; the previous
    # durable ledger remains running. Recovery must consult this artifact.
    (workspace / "logs/service-launch.log").write_text(json.dumps(failure, indent=2) + '\n')
    monkeypatch.setattr(service, "_write_durable", original)
    resumed = service.run_quality_service(source, workspace, config=config)
    assert resumed["state"] == "storage_error"
    assert resumed["controller_launches"] == 1
    assert len(list((workspace / "logs").glob("controller-*.log"))) == 1


def test_restarts_crashed_controller_and_records_measured_completion(
    service, source, tmp_path, named_child
):
    # Missing bounded restart would leave this real queued improvement unresolved.
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "crash_once").touch()
    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )
    assert result["state"] == "completed"
    assert result["completion"] == "all_measured"
    assert result["controller_restarts"] == 1
    assert len(list((workspace / "logs").glob("controller-*.log"))) == 2
    events = [json.loads(line) for line in (workspace / "logs/service_events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events].count("child_started") == 2
    assert [row["event"] for row in events].count("child_finished") == 2
    assert events[-1]["event"] == "service_finished"
    assert all(row.get("log") for row in events if row["event"] == "child_started")


def test_hundred_improvements_finish_with_complete_paired_results(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    for index in range(100):
        enqueue(workspace, proposal=f"improvement-{index:03d}")
    (workspace / "crash_once").touch()
    result = service.run_quality_service(
        source, workspace, config=configuration(service, duration_seconds=10, target_proposals=100)
    )
    assert result["completion"] == "all_measured"
    assert result["measured_proposals"] == 100
    assert result["task_finished_events"] == 200
    assert result["pending_proposals"] == []
    assert result["controller_launches"] == 2


def test_waits_for_late_queue_arrival(service, source, tmp_path, named_child):
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    producer = threading.Thread(target=lambda: (time.sleep(0.12), enqueue(workspace)))
    producer.start()
    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )
    producer.join(timeout=1)
    assert result["terminal_proposals"] == 1
    assert result["completion"] == "all_measured"


def test_duration_bounds_idle_service(service, source, tmp_path, named_child):
    start = time.monotonic()
    result = service.run_quality_service(
        source,
        tmp_path / "campaign",
        config=configuration(service, duration_seconds=0.12),
    )
    assert time.monotonic() - start < 1
    assert result["state"] == "duration_exhausted"
    assert result["terminal_proposals"] == 0


def test_errors_are_terminal_without_claiming_quality_measurement(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "error_outcome").touch()
    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )
    assert result["state"] == "completed"
    assert result["completion"] == "terminal_with_errors"
    assert result["measured_proposals"] == 0


@pytest.mark.parametrize("status", ["fedot_budget_timeout", "unverified_metric"])
def test_quality_failure_finishes_service_after_the_controller_child_exits(
    service, source, tmp_path, named_child, status
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "terminal_status").write_text(status)

    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )

    assert result["state"] == "completed"
    assert result["completion"] == "terminal_with_errors"
    assert result["terminal_proposals"] == 1
    assert result["measured_proposals"] == 0
    assert result["pending_proposals"] == []
    assert result["queue_counts"] == {status: 1}
    assert result["controller_launches"] == 1
    assert result["children"][0]["returncode"] == 0
    saved = json.loads((workspace / "logs/service_status.json").read_text())
    assert saved["state"] == "completed"
    assert saved["child_running"] is False


def test_skipped_scope_is_pending_until_its_datasets_are_rechecked(service, tmp_path):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    path = workspace / "quality_queue" / "a.json"
    item = json.loads(path.read_text())
    item.update(status="skipped", reason="inapplicable_tasks", task_ids=[])
    path.write_text(json.dumps(item))

    result = service._progress(workspace)

    assert result["terminal_proposals"] == 0
    assert result["measured_proposals"] == 0
    assert result["skipped_proposals"] == 1
    assert result["pending_proposals"] == ["a"]


def test_queued_proposal_that_becomes_skipped_does_not_complete_the_service(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "skip_scope").touch()

    result = service.run_quality_service(
        source, workspace, config=configuration(service, max_controller_restarts=0)
    )

    assert result["state"] == "restart_exhausted"
    assert result.get("completion") != "all_measured"
    assert result["terminal_proposals"] == 0
    assert result["skipped_proposals"] == 1
    assert result["pending_proposals"] == ["a"]
    assert result["controller_launches"] == 1
    assert result["children"][0]["returncode"] == 3


def test_mixed_skipped_and_measured_revisions_are_terminal_but_not_fully_measured(
    service, tmp_path
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    enqueue(workspace, proposal="a-r2")
    first_path = workspace / "quality_queue" / "a.json"
    second_path = workspace / "quality_queue" / "a-r2.json"
    first = json.loads(first_path.read_text())
    second = json.loads(second_path.read_text())
    first.update(proposal_id="a", status="skipped", reason="inapplicable_tasks", task_ids=[])
    second.update(proposal_id="a", status="keep")
    first_path.write_text(json.dumps(first))
    second_path.write_text(json.dumps(second))
    (workspace / "quality_jobs.jsonl").write_text("\n".join(
        json.dumps({
            "event": "quality_task_side_finished",
            "candidate": "a-r2",
            "task_id": "example",
            "side": side,
            "result": {
                "status": "ok",
                "score": 0.9,
                "metric_observations": {"metric_valid": True},
            },
        })
        for side in ("stock", "patch")
    ) + "\n")

    progress = service._progress(workspace)

    assert progress["terminal_proposals"] == 0
    assert progress["measured_proposals"] == 0
    assert progress["skipped_proposals"] == 0
    assert progress["pending_proposals"] == ["a"]


def test_finite_scores_without_metric_attestation_are_not_measured(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    path = workspace / "quality_queue" / "a.json"
    item = json.loads(path.read_text())
    item["status"] = "keep"
    path.write_text(json.dumps(item))
    events = workspace / "quality_jobs.jsonl"
    events.write_text("\n".join(
        json.dumps({
            "event": "quality_task_side_finished",
            "candidate": "a",
            "task_id": "example",
            "side": side,
            "result": {"status": "ok", "score": 0.9},
        })
        for side in ("stock", "patch")
    ) + "\n")

    result = service.run_quality_service(source, workspace, config=configuration(service))

    assert result["state"] == "completed"
    assert result["completion"] == "terminal_with_errors"
    assert result["terminal_proposals"] == 1
    assert result["measured_proposals"] == 0
    assert result["controller_launches"] == 0


def test_controller_retry_budget_is_finite(service, source, tmp_path, named_child):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "hang_forever").touch()
    result = service.run_quality_service(
        source, workspace, config=configuration(service, max_controller_restarts=1)
    )
    assert result["state"] == "restart_exhausted"
    assert len(list((workspace / "logs").glob("controller-*.log"))) == 2
    events = [json.loads(line) for line in (workspace / "logs/service_events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events].count("child_started") == 2
    assert [row["event"] for row in events].count("child_finished") == 2
    assert events[-1]["state"] == "restart_exhausted"


def test_records_child_exception_with_the_exact_log_path(service, source, tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    enqueue(workspace)

    def fail_child(*args, **kwargs):
        raise RuntimeError("fixture child launch failure")

    monkeypatch.setattr(service, "run_worker", fail_child)
    result = service.run_quality_service(
        source, workspace, config=configuration(service, max_controller_restarts=0)
    )

    assert result["state"] == "artifact_error"
    events = [json.loads(line) for line in (workspace / "logs/service_events.jsonl").read_text().splitlines()]
    interrupted = [row for row in events if row["event"] == "child_interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["log"].rsplit("/", 1)[-1].startswith("controller-0001-")
    assert interrupted[0]["reason"].endswith("fixture child launch failure")
    assert events[-1]["event"] == "service_failed"
    assert events[-1]["state"] == "artifact_error"


def test_refuses_changed_frozen_config(service, source, tmp_path, named_child):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    service.run_quality_service(source, workspace, config=configuration(service))
    with pytest.raises(ValueError, match="frozen"):
        service.run_quality_service(
            source, workspace, config=configuration(service, target_proposals=2)
        )


def test_refuses_changed_source(service, source, tmp_path, named_child):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    service.run_quality_service(source, workspace, config=configuration(service))
    (source / "fedot" / "model.py").write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="frozen"):
        service.run_quality_service(source, workspace, config=configuration(service))


def test_corrupt_journal_never_claims_completion(
    service, source, tmp_path, named_child, capsys
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "quality_jobs.jsonl").write_text('{"broken":\n')
    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )
    assert result["state"] == "artifact_error"
    assert "quality_jobs.jsonl:1" in capsys.readouterr().err
    assert not list((workspace / "logs").glob("controller-*.log"))


def test_hunt_conservative_budget_persists_across_resume(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    config = configuration(
        service,
        hunt=True,
        target_proposals=2,
        duration_seconds=0.4,
        llm_max_cost_usd=0.25,
        hunt_total_cost_usd=0.25,
    )
    result = service.run_quality_service(source, workspace, config=config)
    assert result["hunt_campaigns"] == 1
    assert result["hunt_reserved_cost_usd"] == 0.25
    service.run_quality_service(source, workspace, config=config)
    assert len(list((workspace / "logs").glob("hunt-*.log"))) == 1


def test_exclusive_service_lock(service, source, tmp_path, named_child):
    import fcntl

    workspace = tmp_path / "campaign"
    workspace.mkdir()
    with (workspace / ".quality_service.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already"):
            service.run_quality_service(
                source, workspace, config=configuration(service)
            )


def test_service_duration_caps_a_blocked_controller(
    service, source, tmp_path, named_child
):
    # A blocking quality command must not extend a short remaining service lease.
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "sleep_child").touch()
    start = time.monotonic()
    result = service.run_quality_service(
        source,
        workspace,
        config=configuration(
            service,
            duration_seconds=0.15,
            controller_timeout_seconds=30,
            max_controller_restarts=0,
        ),
    )
    assert time.monotonic() - start < 2
    assert result["state"] == "duration_exhausted"
    assert result["last_child"]["timed_out"] is True


def test_storage_failure_cancels_owned_child_and_reports_stderr(
    service, source, tmp_path, named_child, monkeypatch, capsys
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "sleep_child").touch()
    original = service._write_durable

    def failing_status(path, payload):
        if path.name == "service_status.json" and payload.get("child_running"):
            raise OSError(28, "No space left on device")
        return original(path, payload)

    monkeypatch.setattr(service, "_write_durable", failing_status)
    start = time.monotonic()
    result = service.run_quality_service(
        source, workspace, config=configuration(service, controller_timeout_seconds=30)
    )
    assert time.monotonic() - start < 2
    assert result["state"] == "storage_error"
    assert "No space left on device" in capsys.readouterr().err
    assert (
        json.loads((workspace / "quality_queue" / "a.json").read_text())["status"]
        == "queued"
    )


def test_failure_is_visible_when_even_error_status_cannot_be_saved(
    service, source, tmp_path, monkeypatch, capsys
):
    def failing_storage(path, payload):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(service, "_write_durable", failing_storage)
    result = service.run_quality_service(
        source, tmp_path / "campaign", config=configuration(service)
    )
    assert result["state"] == "storage_error"
    assert "could not persist failure" in capsys.readouterr().err


def test_resume_charges_interrupted_controller_before_launch(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    config = configuration(service, max_controller_restarts=0)
    service.run_quality_service(source, workspace, config=config)
    state_path = workspace / "logs" / "service_status.json"
    state = json.loads(state_path.read_text())
    state.update(
        state="running", child_running=True, active_child={"kind": "quality-drain"}
    )
    state_path.write_text(json.dumps(state))
    enqueue(workspace, "late")
    result = service.run_quality_service(source, workspace, config=config)
    assert result["state"] == "restart_exhausted"
    assert result["controller_restarts"] == 1
    assert len(list((workspace / "logs").glob("controller-*.log"))) == 1


def test_distinct_proposals_do_not_count_revisions_as_new_improvements(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    enqueue(workspace, "a-r2")
    path = workspace / "quality_queue" / "a-r2.json"
    revision = json.loads(path.read_text())
    revision["proposal_id"] = "a"
    path.write_text(json.dumps(revision))
    result = service.run_quality_service(
        source,
        workspace,
        config=configuration(service, target_proposals=2, duration_seconds=0.25),
    )
    assert result["state"] == "completed"
    assert result["total_proposals"] == 1
    assert result["terminal_proposals"] == 1


def test_frozen_required_tasks_prevent_partial_metrics_claim(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    path = workspace / "quality_queue" / "a.json"
    item = json.loads(path.read_text())
    item["required_task_ids"] = ["example", "missing"]
    path.write_text(json.dumps(item))
    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )
    assert result["completion"] == "terminal_with_errors"
    assert result["measured_proposals"] == 0


def test_new_durable_task_progress_resets_controller_failure_streak(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "progress_once").touch()
    result = service.run_quality_service(
        source, workspace, config=configuration(service, max_controller_restarts=0)
    )
    assert result["state"] == "completed"
    assert result["total_controller_restarts"] == 1
    assert result["controller_restarts"] == 0


def test_preserves_every_owned_child_launch_outcome(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    (workspace / "crash_once").touch()
    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )
    assert [child["returncode"] for child in result["children"]] == [17, 0]
    saved = json.loads((workspace / "logs" / "service_status.json").read_text())
    assert [child["returncode"] for child in saved["children"]] == [17, 0]


def test_hunt_campaign_count_is_bounded_even_without_new_improvements(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    config = configuration(
        service,
        hunt=True,
        target_proposals=2,
        duration_seconds=0.7,
        max_hunt_campaigns=2,
        hunt_total_cost_usd=2,
    )
    result = service.run_quality_service(source, workspace, config=config)
    assert result["hunt_campaigns"] == 2
    assert result["total_proposals"] == 1
    assert len(list((workspace / "logs").glob("hunt-*.log"))) == 2


def test_missing_budget_ledger_is_never_recreated_with_fresh_hunt_budget(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    config = configuration(
        service,
        hunt=True,
        target_proposals=2,
        duration_seconds=0.4,
        llm_max_cost_usd=0.25,
        hunt_total_cost_usd=0.25,
    )
    service.run_quality_service(source, workspace, config=config)
    ledger = workspace / "logs" / "service_status.json"
    ledger.unlink()
    first = service.run_quality_service(source, workspace, config=config)
    second = service.run_quality_service(source, workspace, config=config)
    assert first["state"] == second["state"] == "artifact_error"
    assert not ledger.exists()
    assert len(list((workspace / "logs").glob("hunt-*.log"))) == 1


def test_malformed_budget_ledger_is_preserved_for_audit(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    config = configuration(service)
    service.run_quality_service(source, workspace, config=config)
    ledger = workspace / "logs" / "service_status.json"
    ledger.write_text('{"state": "running"}')
    result = service.run_quality_service(source, workspace, config=config)
    assert result["state"] == "artifact_error"
    assert ledger.read_text() == '{"state": "running"}'
    assert len(list((workspace / "logs").glob("controller-*.log"))) == 1


def test_hunt_credentials_are_available_only_to_hunt_child(
    service, source, tmp_path, named_child, monkeypatch
):
    monkeypatch.setenv("FEDOTLLM_LLM_API_KEY", "test-only-provider-token")
    workspace = tmp_path / "campaign"
    result = service.run_quality_service(
        source, workspace, config=configuration(service, hunt=True)
    )
    assert result["completion"] == "all_measured"
    flags = [
        json.loads(line)
        for line in (workspace / "credential_flags.jsonl").read_text().splitlines()
    ]
    assert flags == [
        {"kind": "run", "key_present": True},
        {"kind": "quality-drain", "key_present": False},
    ]
    assert (
        "test-only-provider-token"
        not in (workspace / "logs" / "service_status.json").read_text()
    )


def test_blocked_proposal_is_terminal_error_without_automatic_reexecution(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    path = workspace / "quality_queue" / "a.json"
    item = json.loads(path.read_text())
    item["status"] = "blocked"
    path.write_text(json.dumps(item))
    result = service.run_quality_service(
        source, workspace, config=configuration(service)
    )
    assert result["completion"] == "terminal_with_errors"
    assert not list((workspace / "logs").glob("controller-*.log"))


def test_replayed_or_reused_side_events_do_not_count_as_new_task_progress(
    service, tmp_path
):
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    finished = {
        "event": "quality_task_side_finished",
        "candidate": "a",
        "task_id": "example",
        "side": "stock",
        "attempt": 1,
        "result": {"status": "ok", "score": 1.0},
    }
    reused = dict(finished, event="quality_task_side_reused", candidate="b")
    (workspace / "quality_jobs.jsonl").write_text(
        "\n".join(json.dumps(item) for item in [finished, finished, reused]) + "\n"
    )
    assert service._progress(workspace)["task_finished_events"] == 1


def test_existing_finite_queue_completes_without_hunt_target_gate(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    enqueue(workspace)
    result = service.run_quality_service(
        source, workspace, config=configuration(service, target_proposals=100)
    )
    assert result["state"] == "completed"
    assert result["terminal_proposals"] == 1


def test_exhausted_hunt_budget_returns_explicit_incomplete_result(
    service, source, tmp_path, named_child
):
    workspace = tmp_path / "campaign"
    config = configuration(
        service,
        hunt=True,
        target_proposals=2,
        duration_seconds=30,
        hunt_total_cost_usd=0.25,
    )
    start = time.monotonic()
    result = service.run_quality_service(source, workspace, config=config)
    assert time.monotonic() - start < 2
    assert result["state"] == "hunt_budget_exhausted"
    assert result["completion"] == "incomplete"


def test_progress_reads_saved_side_snapshot_even_when_journal_append_never_happened(
    service, tmp_path
):
    from fedotllm.agents.evolve.storage.quality_events import save_quality_event

    workspace = tmp_path / "campaign"
    workspace.mkdir()
    save_quality_event(
        workspace / "quality_jobs.jsonl",
        {
            "event": "quality_task_side_finished",
            "candidate": "a",
            "task_id": "example",
            "side": "stock",
            "attempt": 1,
            "result": {"status": "ok", "score": 1.0},
        },
    )
    assert service._progress(workspace)["task_finished_events"] == 1


def test_only_active_last_incomplete_journal_row_is_temporarily_ignored(
    service, tmp_path
):
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    journal = workspace / "quality_jobs.jsonl"
    journal.write_text('{"event":"quality_task_side_started"}\n{"event":')
    assert service._progress(workspace, child_running=True)["task_finished_events"] == 0
    with pytest.raises(ValueError, match="quality_jobs.jsonl:2"):
        service._progress(workspace)
    journal.write_text('{"event":\n{"event":"quality_task_side_started"}\n')
    with pytest.raises(ValueError, match="quality_jobs.jsonl:1"):
        service._progress(workspace, child_running=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"duration_seconds": float("inf")},
        {"poll_seconds": 0},
        {"target_proposals": -1},
        {"llm_max_cost_usd": 0},
        {"max_controller_restarts": -1},
        {"max_hunt_campaigns": 1.5},
    ],
)
def test_invalid_limits_are_rejected_before_any_launch(
    service, source, tmp_path, changes
):
    workspace = tmp_path / "campaign"
    with pytest.raises(ValueError):
        service.run_quality_service(
            source, workspace, config=configuration(service, **changes)
        )
    assert not workspace.exists()

"""Regression checks for unattended queue completion and durable recovery."""

import json
import math
import os
import shutil
import threading
from pathlib import Path
import sys
from dataclasses import replace

import pytest

from fedotllm.agents.evolve.controller import quality_executor as executor
from fedotllm.agents.evolve.controller import quality_queue as queue
from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry
from fedotllm.agents.evolve.storage import journal as storage
from fedotllm.agents.evolve.types import Decision, PatchCandidate


QUALITY_PROTOCOL_FIXTURE = """\
import json
from pathlib import Path
import sys
import time

spec = json.loads(Path(sys.argv[sys.argv.index("--spec-json") + 1]).read_text())
out = Path(sys.argv[sys.argv.index("--out") + 1])
root = Path(__file__).parent
mode = (root / "fixture-mode").read_text()
print("fixture worker entered", flush=True)
with (root / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps({"dataset": spec["source_dataset"], "side": spec["side"]}) + "\\n")
if mode == "stage_corrupt":
    out.with_suffix(".stages.json").write_text('{"last_stage":')
    time.sleep(30)
elif mode == "mixed" and spec["source_dataset"] == "fedot-ts-australia":
    print("fixture import failure", flush=True)
    raise SystemExit(7)
elif mode == "mixed" and spec["source_dataset"] == "fedot-ts-beer":
    out.write_text('{"status":')
elif mode == "mixed" and spec["source_dataset"] == "fedot-ts-arctic":
    time.sleep(30)
elif mode == "mixed":
    payload = {"status": "ok", "score": 0.75,
        "metric_observations": {"metric_valid": True}}
    payload.update({key: spec.get(key) for key in (
        "side", "attempt", "attempt_id", "registry_identity", "evaluation_identity", "worker_invocation_id")})
    payload["task_id"] = spec["source_dataset"]
    out.write_text(json.dumps(payload))
else:
    raw = (root / "fixture-payload").read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        out.write_bytes(raw)
    else:
        if isinstance(payload, dict):
            payload.update({key: spec.get(key) for key in (
                "side", "attempt", "attempt_id", "registry_identity", "evaluation_identity", "worker_invocation_id")})
            payload["task_id"] = spec["source_dataset"]
        out.write_text(json.dumps(payload))
"""


def _protocol_fixture(
    tmp_path, monkeypatch, *, mode="payload", payload=b"{}", watchdog=5
):
    from fedotllm.agents.evolve.evaluation import fedot_quality as quality

    (tmp_path / "quality_protocol_fixture.py").write_text(QUALITY_PROTOCOL_FIXTURE)
    (tmp_path / "fixture-mode").write_text(mode)
    (tmp_path / "fixture-payload").write_bytes(payload)
    monkeypatch.setenv("FEDOTLLM_REPO_PYTHON", sys.executable)
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EVOLVE_QUALITY_STOCK_CACHE", str(tmp_path / "stock_cache"))
    monkeypatch.setattr(quality, "WORKER_MODULE", "quality_protocol_fixture")
    return quality, replace(load_quality_registry(), watchdog_seconds=watchdog)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"status":',
        b"[]",
        b"null",
        b"\xff",
        b'{"status":"ok","score":0.9,"duration_s":"bad"}',
        b'{"status":"ok","score":0.9,"metric_observations":{}}',
        b'{"status":"ok","score":0.9,"metric_observations":{"metric_valid":false}}',
        b'{"status":"ok","score":0.9,"metric_observations":7}',
        b'{"status":"ok","score":0.9,"duration_s":-1}',
        b'{"status":"ok","score":0.9,"duration_s":1e999}',
        b'{"status":"unexpected","score":0.9}',
        b'{"status":"ok","score":true}',
        b'{"status":"ok","score":[]}',
        b'{"status":"ok","score":"bad"}',
        b'{"status":"ok","score":1e999}',
        b'{"status":"ok","score":null}',
        b"{}",
    ],
)
def test_malformed_real_worker_result_becomes_invalid_with_saved_log(
    tmp_path, monkeypatch, payload
):
    quality, registry = _protocol_fixture(tmp_path, monkeypatch, payload=payload)
    result = quality.run_quality_side(
        "fedot-ts-beer", checkout=tmp_path, side="patch", registry=registry
    )
    assert result.status == "invalid"
    assert "worker result" in result.detail
    assert "quality-out-" in result.detail
    assert "fixture worker entered" in result.log_tail
    assert Path(result.metric_observations["worker_log_path"]).is_file()
    saved = Path(result.metric_observations["worker_result_path"]).read_bytes()
    try:
        original = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        assert saved == payload
    else:
        saved_payload = json.loads(saved)
        if isinstance(original, dict):
            assert {key: saved_payload.get(key) for key in original} == original
        else:
            assert saved_payload == original


def test_malformed_stock_cache_is_recomputed_instead_of_stopping_measurement(
    tmp_path, monkeypatch
):
    quality, registry = _protocol_fixture(
        tmp_path,
        monkeypatch,
        payload=b'{"status":"ok","score":0.75,"metric_observations":{"metric_valid":true}}',
    )
    first = quality.run_quality_side(
        "fedot-ts-beer", checkout=tmp_path, side="stock", registry=registry
    )
    assert first.status == "ok"
    cache = next((tmp_path / "stock_cache").glob("*.json"))
    payload = json.loads(cache.read_text())
    payload["duration_s"] = "broken metadata"
    cache.write_text(json.dumps(payload))
    next_result = quality.run_quality_side(
        "fedot-ts-beer", checkout=tmp_path, side="stock", registry=registry
    )
    assert next_result.status == "ok"
    assert not next_result.metric_observations.get("cache_hit")
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2


def test_successful_worker_result_survives_until_the_side_event_is_persisted(
    tmp_path, monkeypatch
):
    quality, registry = _protocol_fixture(
        tmp_path,
        monkeypatch,
        payload=b'{"status":"ok","score":0.75,"duration_s":0.2,"metric_observations":{"metric_valid":true}}',
    )
    result = quality.run_quality_side(
        "fedot-ts-beer",
        checkout=tmp_path,
        side="patch",
        registry=registry,
        proposal_id="candidate-a",
        attempt=1,
    )

    result_path = Path(result.metric_observations["worker_result_path"])
    assert result.status == "ok"
    assert result_path.is_file()


def test_quality_job_retains_final_worker_result_and_spec_for_audit(
    tmp_path, monkeypatch
):
    quality, registry = _protocol_fixture(
        tmp_path,
        monkeypatch,
        payload=b'{"status":"ok","score":0.75,"duration_s":0.2,"metric_observations":{"metric_valid":true}}',
    )
    source, _workspace = _enqueue(tmp_path)
    (source / "quality_protocol_fixture.py").write_text(QUALITY_PROTOCOL_FIXTURE)
    (source / "fixture-mode").write_text("payload")
    (source / "fixture-payload").write_bytes(
        b'{"status":"ok","score":0.75,"duration_s":0.2,"metric_observations":{"metric_valid":true}}'
    )
    finished = []
    rows = quality.run_quality_jobs(
        stock_checkout=source,
        patch_checkout=source,
        task_ids=("fedot-ts-beer",),
        n_jobs=1,
        cpu_quota=1,
        registry=registry,
        on_side_event=lambda spec, side, event, result, attempt: finished.append(
            (spec, side, event, result)
        )
        if event == "finished"
        else None,
    )

    assert len(rows) == 1
    assert len(finished) == 2
    for spec, _side, _event, result in finished:
        assert Path(result.metric_observations["worker_result_path"]).is_file()
        paths = result.metric_observations["worker_paths"]
        assert Path(paths["worker_spec_path"]).is_file()


def test_quality_journal_distinguishes_worker_request_from_actual_spawn(
    tmp_path, monkeypatch
):
    quality, registry = _protocol_fixture(
        tmp_path,
        monkeypatch,
        payload=b'{"status":"ok","score":0.75,"duration_s":0.2,"metric_observations":{"metric_valid":true}}',
    )
    source, workspace = _enqueue(tmp_path)
    (source / "quality_protocol_fixture.py").write_text(QUALITY_PROTOCOL_FIXTURE)
    (source / "fixture-mode").write_text("payload")
    (source / "fixture-payload").write_bytes(
        b'{"status":"ok","score":0.75,"duration_s":0.2,"metric_observations":{"metric_valid":true}}'
    )

    executor.measure_fedot_quality(
        source,
        source,
        journal=workspace / "quality_jobs.jsonl",
        candidate=PatchCandidate("improvement-0"),
        task_ids=("fedot-ts-beer",),
        registry=registry,
    )

    events = storage.read_jsonl(workspace / "quality_jobs.jsonl", strict=True)
    lifecycle = [event for event in events if event.get("task_id") == "fedot-ts-beer"]
    for side in ("stock", "patch"):
        side_events = [event for event in lifecycle if event.get("side") == side]
        assert [event["event"] for event in side_events] == [
            "quality_task_side_started",
            "quality_task_worker_spawned",
            "quality_task_side_finished",
        ]
        assert side_events[0]["artifact_state"] == "expected"
        assert side_events[1]["artifact_state"] == "spawned"
        assert type(side_events[1].get("worker_pid")) is int
        assert side_events[1]["worker_pid"] > 0
        assert Path(side_events[1]["worker_log_path"]).is_file()
        assert side_events[2]["result"]["status"] == "ok"
        result_path = Path(side_events[2]["worker_result_path"])
        assert side_events[2]["result"]["metric_observations"][
            "worker_result_path"
        ] == str(result_path)
        assert result_path.is_file()


def test_stock_cache_hit_keeps_owner_artifacts_separate_from_current_attempt(
    tmp_path, monkeypatch
):
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset, job_spec
    from fedotllm.agents.evolve.types import ScoreResult

    source, workspace = _enqueue(tmp_path)
    registry = load_quality_registry()
    spec = job_spec(get_dataset("fedot-ts-beer", registry), registry=registry)
    owner_log = tmp_path / "prior-attempt.log"
    owner_result = tmp_path / "prior-attempt.json"
    owner_stage = tmp_path / "prior-attempt.stages.json"
    owner_log.write_text("cached baseline artifact")
    stock_result = ScoreResult(
        "fedot-ts-beer",
        "ok",
        0.9,
        metric_observations={
            "cache_hit": True,
            "worker_log_path": str(owner_log),
            "worker_result_path": str(owner_result),
            "stage_status_path": str(owner_stage),
        },
    )
    patched_result = ScoreResult("fedot-ts-beer", "ok", 0.92)

    def fake_jobs(*, on_side_event, **_kwargs):
        for side, result in (("stock", stock_result), ("patch", patched_result)):
            on_side_event(spec, side, "started", None, 1)
            if side == "patch":
                on_side_event(spec, side, "spawned", None, 1)
            on_side_event(spec, side, "finished", result, 1)
        return []

    monkeypatch.setattr(executor, "run_quality_jobs", fake_jobs)
    monkeypatch.setattr(
        executor,
        "decision_from_quality_jobs",
        lambda _rows: Decision(False, "drop", 0.02),
    )

    executor.measure_fedot_quality(
        source,
        source,
        journal=workspace / "quality_jobs.jsonl",
        candidate=PatchCandidate("improvement-0"),
        task_ids=("fedot-ts-beer",),
        registry=registry,
    )

    events = storage.read_jsonl(workspace / "quality_jobs.jsonl", strict=True)
    stock_finished = next(
        event
        for event in events
        if event.get("event") == "quality_task_side_finished"
        and event.get("side") == "stock"
    )
    assert stock_finished["artifact_state"] == "stock_cache_hit"
    assert stock_finished["worker_paths"]["worker_log_path"] != str(owner_log)
    assert stock_finished["shared_worker_paths"]["worker_log_path"] == str(owner_log)


def test_corrupt_stage_snapshot_preserves_real_worker_timeout(tmp_path, monkeypatch):
    quality, registry = _protocol_fixture(
        tmp_path, monkeypatch, mode="stage_corrupt", watchdog=0.3
    )
    result = quality.run_quality_side(
        "fedot-ts-beer", checkout=tmp_path, side="patch", registry=registry
    )
    assert result.status == "timeout"
    assert "stage_status_error" in result.metric_observations
    assert Path(result.metric_observations["stage_status_path"]).is_file()


def test_real_worker_failure_and_timeout_do_not_prevent_the_next_dataset_pair(
    tmp_path, monkeypatch
):
    quality, registry = _protocol_fixture(
        tmp_path, monkeypatch, mode="mixed", watchdog=0.3
    )
    events = []
    results = quality.run_quality_jobs(
        stock_checkout=tmp_path,
        patch_checkout=tmp_path,
        task_ids=(
            "fedot-ts-beer",
            "fedot-ts-arctic",
            "fedot-ts-australia",
            "fedot-ts-traffic",
        ),
        n_jobs=4,
        cpu_quota=4,
        registry=registry,
        on_side_event=lambda spec, side, event, result, attempt: events.append(
            (
                spec["source_dataset"],
                side,
                event,
                result.status if result else None,
                attempt,
            )
        ),
    )
    assert [row["stock_result"].status for row in results] == [
        "invalid",
        "timeout",
        "invalid",
        "ok",
    ]
    assert [row["patched_result"].status for row in results] == [
        "invalid",
        "timeout",
        "invalid",
        "ok",
    ]
    assert "worker exit 7" in results[2]["stock_result"].detail
    assert "fixture import failure" in results[2]["stock_result"].log_tail
    finished = [event for event in events if event[2] == "finished"]
    assert len(finished) == 8
    assert max(event[4] for event in finished) == 1
    spawned = [event for event in events if event[2] == "spawned"]
    assert len(spawned) == 8
    assert {(event[0], event[1]) for event in spawned} == {
        (dataset, side)
        for dataset in (
            "fedot-ts-beer",
            "fedot-ts-arctic",
            "fedot-ts-australia",
            "fedot-ts-traffic",
        )
        for side in ("stock", "patch")
    }
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 8
    assert calls[-1] == {"dataset": "fedot-ts-traffic", "side": "patch"}


def test_failed_append_rolls_back_only_its_partial_bytes(tmp_path, monkeypatch):
    path = tmp_path / "quality_jobs.jsonl"
    storage.append_journal(path, {"event": "previous"})
    original = path.read_bytes()
    real_write = os.write
    calls = []

    def failing_write(fd, data):
        calls.append(True)
        if len(calls) == 1:
            return real_write(fd, data[:7])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(storage.os, "write", failing_write)
    with pytest.raises(OSError, match="No space"):
        storage.append_journal(path, {"event": "new"})
    assert path.read_bytes() == original


def test_paired_workers_use_the_stock_python_even_when_copy_has_no_venv(
    tmp_path, monkeypatch
):
    from fedotllm.agents.evolve.evaluation import fedot_quality as quality
    from fedotllm.agents.evolve.types import ScoreResult

    source = tmp_path / "source"
    python = source / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    patch = tmp_path / "experiment"
    patch.mkdir()
    launched = []

    def side(dataset, *, python_executable=None, **_kwargs):
        launched.append(python_executable)
        return ScoreResult(dataset.task_id, "ok", 1.0)

    monkeypatch.delenv("FEDOTLLM_REPO_PYTHON", raising=False)
    monkeypatch.setattr(quality, "run_quality_side", side)
    quality.run_quality_pair(
        "fedot-ts-beer", stock_checkout=source, patch_checkout=patch
    )
    assert launched == [str(python), str(python)]


def test_nonzero_worker_exit_cannot_return_a_successful_metric(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation import fedot_quality as quality
    from fedotllm.agents.evolve.execution.process import WorkerOutcome

    def failed_worker(command, **_kwargs):
        output = Path(command[command.index("--out") + 1])
        spec = json.loads(Path(command[command.index("--spec-json") + 1]).read_text())
        payload = {
            key: spec.get(key)
            for key in (
                "side",
                "attempt",
                "attempt_id",
                "registry_identity",
                "evaluation_identity",
                "worker_invocation_id",
            )
        }
        payload.update(task_id=spec["source_dataset"], status="ok", score=0.9)
        output.write_text(json.dumps(payload))
        return WorkerOutcome(7, "failed after writing output", "")

    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setattr(quality, "run_worker", failed_worker)
    result = quality.run_quality_side("fedot-ts-beer", checkout=tmp_path, side="patch")
    assert result.status != "ok"
    assert "7" in result.detail


def _enqueue(tmp_path, count=1):
    source = tmp_path / "FEDOT"
    rel = "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
    target = source / rel
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n")
    workspace = tmp_path / "campaign"
    for index in range(count):
        queue.enqueue_quality_job(
            workspace,
            candidate=PatchCandidate(f"improvement-{index}", file_path=rel),
            patch_text=f"--- a/{rel}\n+++ b/{rel}\n@@\n-value = 1\n+value = 2\n",
            hint="forecasting improvement",
            priority="normal",
        )
    return source, workspace


def test_status_write_failure_preserves_previous_proposal(tmp_path, monkeypatch):
    path = tmp_path / "proposal.json"
    path.write_text('{"status": "queued", "candidate_id": "a"}')
    original = path.read_bytes()

    def no_space(_self, _target):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", no_space)
    with pytest.raises(OSError):
        queue.write_job_status(path, "running")
    assert path.read_bytes() == original


def test_unreadable_proposal_cannot_silently_disappear(tmp_path):
    directory = tmp_path / "quality_queue"
    directory.mkdir()
    (directory / "broken.json").write_text('{"candidate_id":')
    with pytest.raises(ValueError, match="broken.json"):
        queue.list_queued_jobs(tmp_path)


def test_corrupt_quality_journal_blocks_recovery_with_exact_line(tmp_path):
    path = tmp_path / "quality_jobs.jsonl"
    path.write_text('{"event":"started"}\n{"event":\n')
    with pytest.raises(ValueError, match="quality_jobs.jsonl:2"):
        executor._recover_side_state(
            path, candidate_id="a", task_ids=("fedot-ts-beer",), registry_identity="v1"
        )


def test_new_attempt_does_not_reuse_result_from_previous_attempt(tmp_path):
    path = tmp_path / "quality_jobs.jsonl"
    common = dict(
        candidate="a", task_id="fedot-ts-beer", side="stock", registry_identity="v1"
    )
    storage.append_journal(
        path,
        {
            **common,
            "event": "quality_task_side_finished",
            "attempt": 1,
            "result": {"status": "ok", "score": 0.9},
        },
    )
    storage.append_journal(
        path, {**common, "event": "quality_task_side_started", "attempt": 2}
    )
    results, attempts = executor._recover_side_state(
        path, candidate_id="a", task_ids=("fedot-ts-beer",), registry_identity="v1"
    )
    assert attempts[("fedot-ts-beer", "stock")] == 2
    assert results[("fedot-ts-beer", "stock")].status == "crash"


def test_unfinished_spawned_event_recovers_as_interrupted_attempt(tmp_path):
    path = tmp_path / "quality_jobs.jsonl"
    storage.append_journal(
        path,
        {
            "event": "quality_task_side_started",
            "candidate": "a",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "attempt_id": "a:fedot-ts-beer:stock:1",
            "registry_identity": "v1",
        },
    )
    storage.append_journal(
        path,
        {
            "event": "quality_task_worker_spawned",
            "candidate": "a",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "attempt_id": "a:fedot-ts-beer:stock:1",
            "registry_identity": "v1",
        },
    )
    torn_tail = b'{"event":"quality_task_side_finished","candidate":"a"'
    with path.open("ab") as stream:
        stream.write(torn_tail)

    results, attempts = executor._recover_side_state(
        path, candidate_id="a", task_ids=("fedot-ts-beer",), registry_identity="v1"
    )

    result = results[("fedot-ts-beer", "stock")]
    assert result.status == "crash"
    assert result.detail == "controller interrupted while this task side was running"
    assert attempts[("fedot-ts-beer", "stock")] == 1
    events = executor.read_quality_events(path)
    finished = [
        row for row in events if row.get("event") == "quality_task_side_finished"
    ]
    assert finished == []
    interrupted = [
        row for row in events if row.get("event") == "quality_task_side_interrupted"
    ]
    assert len(interrupted) == 1
    assert interrupted[0]["attempt_id"] == "a:fedot-ts-beer:stock:1"
    assert (
        interrupted[0]["result"]["metric_observations"]["failure_kind"]
        == "controller_interrupted"
    )
    assert storage.read_jsonl(path, strict=True)
    quarantine_files = list(tmp_path.glob("quality_jobs.jsonl.incomplete-tail.*.bin"))
    assert len(quarantine_files) == 1
    assert quarantine_files[0].read_bytes() == torn_tail

    resumed_results, resumed_attempts = executor._recover_side_state(
        path, candidate_id="a", task_ids=("fedot-ts-beer",), registry_identity="v1"
    )
    assert resumed_results[("fedot-ts-beer", "stock")].status == "crash"
    assert resumed_attempts[("fedot-ts-beer", "stock")] == 1
    assert (
        len(
            [
                row
                for row in executor.read_quality_events(path)
                if row.get("event") == "quality_task_side_interrupted"
            ]
        )
        == 1
    )


def test_diagnose_reports_interrupted_attempt_without_metric(tmp_path):
    from datetime import datetime
    from fedotllm.agents.evolve.controller.diagnose import diagnose_run

    workspace = tmp_path / "run"
    (workspace / "quality_queue").mkdir(parents=True)
    (workspace / "logs").mkdir()
    journal = workspace / "quality_jobs.jsonl"
    storage.append_journal(
        journal,
        {
            "event": "quality_task_side_interrupted",
            "candidate": "a",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "attempt_id": "a:fedot-ts-beer:stock:1",
            "registry_identity": "v1",
            "result": {
                "status": "crash",
                "metric_observations": {
                    "failure_kind": "controller_interrupted",
                    "metric_valid": False,
                },
            },
        },
    )
    (workspace / "logs/service_status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "child_running": False,
                "active_child": None,
                "heartbeat_at": "2026-10-05T11:59:00+00:00",
            }
        )
    )
    (workspace / "quality_queue/candidate-a.json").write_text(
        json.dumps(
            {
                "candidate_id": "a",
                "proposal_id": "a",
                "task_ids": ["fedot-ts-beer"],
            }
        )
    )
    (workspace / "quality_jobs.jsonl").write_bytes(journal.read_bytes())
    report = diagnose_run(
        workspace, now=datetime.fromisoformat("2026-10-05T12:00:00+00:00")
    )
    assert report["worker_checks"] == []
    assert report["dataset_checks"][0]["result_status"] == "crash"
    assert report["dataset_checks"][0]["metric_state"] == "missing"


def test_recovers_durable_worker_result_when_finished_event_was_not_written(
    tmp_path, monkeypatch
):
    path = tmp_path / "quality_jobs.jsonl"
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths

    worker_paths = worker_artifact_paths(
        "fedot-ts-beer", "stock", "proposal-r2", 1, root=tmp_path
    )
    result_path = Path(worker_paths["worker_result_path"])
    stage_path = Path(worker_paths["stage_status_path"])
    identity = {
        "proposal_id": "proposal-r2",
        "attempt_id": "proposal-r2:fedot-ts-beer:stock:1",
        "task_id": "fedot-ts-beer",
        "source_dataset": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "registry_identity": "v1",
        "evaluation_identity": {"n_jobs": 4, "source_sha256": "current-source"},
        "n_jobs": 4,
        "timeout_seconds": 3600,
        "fedot_search_budget_seconds": 2700,
    }
    common = {
        "candidate": "proposal-r2",
        "candidate_revision_id": "proposal-r2",
        "proposal_id": "proposal-r2",
        "task_id": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "attempt_id": identity["attempt_id"],
        "registry_identity": "v1",
        "evaluation_identity": identity["evaluation_identity"],
        "spec": {
            "n_jobs": 4,
            "timeout_seconds": 3600,
            "fedot_search_budget_seconds": 2700,
        },
        "worker_result_path": str(result_path),
        "stage_status_path": str(stage_path),
        "worker_log_path": worker_paths["worker_log_path"],
        "worker_paths": worker_paths,
    }
    storage.append_journal(path, {**common, "event": "quality_task_side_started"})
    storage.append_journal(path, {**common, "event": "quality_task_worker_spawned"})
    torn_tail = b'{"event":"quality_task_side_finished","candidate":"proposal-r2"'
    with path.open("ab") as stream:
        stream.write(torn_tail)
    result_path.write_text(
        json.dumps(
            {
                "task_id": "fedot-ts-beer",
                "status": "ok",
                "score": 0.91,
                "duration_s": 42.0,
                "metric_observations": {"metric_valid": True, "search_ran": True},
                "proposal_id": "proposal-r2",
                "attempt_id": identity["attempt_id"],
                "source_dataset": "fedot-ts-beer",
                "side": "stock",
                "attempt": 1,
                "registry_identity": "v1",
                "evaluation_identity": identity["evaluation_identity"],
            }
        )
    )
    stage_path.write_text(json.dumps({**identity, "event": "worker_finished"}))

    results, attempts = executor._recover_side_state(
        path,
        candidate_id="proposal-r2",
        task_ids=("fedot-ts-beer",),
        registry_identity="v1",
        evaluation_identity=identity["evaluation_identity"],
    )

    result = results[("fedot-ts-beer", "stock")]
    assert result.status == "ok"
    assert result.score == pytest.approx(0.91)
    assert result.metric_observations["recovered_worker_result"] is True
    assert attempts[("fedot-ts-beer", "stock")] == 1
    events = executor.read_quality_events(path)
    finished = [
        row for row in events if row.get("event") == "quality_task_side_finished"
    ]
    assert len(finished) == 1
    assert finished[0]["artifact_state"] == "recovered_worker_result"
    assert finished[0]["result"]["score"] == pytest.approx(0.91)
    assert storage.read_jsonl(path, strict=True)
    quarantine_files = list(tmp_path.glob("quality_jobs.jsonl.incomplete-tail.*.bin"))
    assert len(quarantine_files) == 1
    assert quarantine_files[0].read_bytes() == torn_tail

    resumed_results, resumed_attempts = executor._recover_side_state(
        path,
        candidate_id="proposal-r2",
        task_ids=("fedot-ts-beer",),
        registry_identity="v1",
        evaluation_identity=identity["evaluation_identity"],
    )
    assert resumed_results[("fedot-ts-beer", "stock")].score == pytest.approx(0.91)
    assert resumed_attempts[("fedot-ts-beer", "stock")] == 1

    from fedotllm.agents.evolve.evaluation import fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset
    from fedotllm.agents.evolve.types import ScoreResult

    calls = []

    def run_side(_dataset, *, side, **_kwargs):
        calls.append(side)
        assert side == "patch", "recovered stock metric must avoid a second stock fit"
        return ScoreResult(task_id="fedot-ts-beer", status="ok", score=0.92)

    monkeypatch.setattr(quality, "run_quality_side", run_side)
    pair = quality.run_quality_pair(
        get_dataset("fedot-ts-beer"),
        stock_checkout=tmp_path / "stock",
        patch_checkout=tmp_path / "patch",
        prior_side_results=resumed_results,
        prior_attempt_counts=resumed_attempts,
    )
    assert pair["stock_result"].score == pytest.approx(0.91)
    assert pair["patched_result"].score == pytest.approx(0.92)
    assert calls == ["patch"]


def test_recovers_parent_timeout_manifest_without_inventing_a_metric(tmp_path):
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths

    journal = tmp_path / "quality_jobs.jsonl"
    paths = worker_artifact_paths(
        "fedot-ts-beer", "patch", "proposal-r3", 2, root=tmp_path
    )
    evaluation_identity = {"n_jobs": 4, "source_sha256": "source-v11"}
    identity = {
        "candidate": "proposal-r3",
        "task_id": "fedot-ts-beer",
        "side": "patch",
        "attempt": 2,
        "attempt_id": "proposal-r3:fedot-ts-beer:patch:2",
        "registry_identity": "registry-v11",
        "evaluation_identity": evaluation_identity,
    }
    common = {
        **identity,
        "candidate_revision_id": "proposal-r3",
        "proposal_id": "proposal-r3",
        "spec": {
            "n_jobs": 4,
            "timeout_seconds": 5400,
            "fedot_search_budget_seconds": 4500,
        },
        "worker_paths": paths,
        "worker_result_path": paths["worker_result_path"],
        "stage_status_path": paths["stage_status_path"],
        "worker_log_path": paths["worker_log_path"],
    }
    storage.append_journal(journal, {**common, "event": "quality_task_side_started"})
    timeout_path = Path(paths["timeout_manifest_path"])
    timeout_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event": "quality_worker_parent_timeout",
                **{
                    key: identity[key]
                    for key in (
                        "task_id",
                        "side",
                        "attempt",
                        "attempt_id",
                        "registry_identity",
                        "evaluation_identity",
                    )
                },
                "source_dataset": "fedot-ts-beer",
                "proposal_id": "proposal-r3",
                "status": "timeout",
                "failure_kind": "timeout",
                "metric_outcome": "not_measured",
                "metric_valid": False,
                "timeout_kind": "task_timeout",
                "detail": "task_timeout after 5400s",
                "side_elapsed_seconds": 5400.0,
                "worker_elapsed_seconds": 5350.0,
                "worker_result_path": paths["worker_result_path"],
                "worker_log_path": paths["worker_log_path"],
                "stage_status_path": paths["stage_status_path"],
                "stage_observations": {
                    "active_stage": "fit",
                    "stage_updated_at": "2026-10-15T01:00:00Z",
                    "metric_outcome": "measured",
                    "metric_valid": True,
                    "task_id": "another-dataset",
                    "side": "stock",
                    "recovered_parent_timeout_manifest": False,
                },
                "log_tail": "last worker line",
            }
        )
    )

    results, attempts = executor._recover_side_state(
        journal,
        candidate_id="proposal-r3",
        task_ids=("fedot-ts-beer",),
        registry_identity="registry-v11",
        evaluation_identity=evaluation_identity,
    )

    recovered = results[("fedot-ts-beer", "patch")]
    assert recovered.status == "timeout"
    assert math.isnan(recovered.score)
    assert recovered.metric_observations["metric_outcome"] == "not_measured"
    assert recovered.metric_observations["metric_valid"] is False
    assert recovered.metric_observations["recovered_parent_timeout_manifest"] is True
    assert recovered.metric_observations["task_id"] == "fedot-ts-beer"
    assert recovered.metric_observations["side"] == "patch"
    assert recovered.metric_observations["active_stage"] == "fit"
    assert attempts[("fedot-ts-beer", "patch")] == 2
    events = executor.read_quality_events(journal)
    terminal = [
        row for row in events if row.get("event") == "quality_task_side_finished"
    ]
    assert len(terminal) == 1
    assert terminal[0]["result"]["status"] == "timeout"
    assert terminal[0]["result"]["score"] is None
    assert storage.read_jsonl(journal, strict=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("side_elapsed_seconds", None),
        ("side_elapsed_seconds", "5400"),
        ("side_elapsed_seconds", True),
        ("worker_elapsed_seconds", None),
        ("worker_elapsed_seconds", "5350"),
        ("worker_elapsed_seconds", False),
    ],
)
def test_parent_timeout_manifest_rejects_invalid_schema_and_elapsed_fields(
    tmp_path, field, value
):
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths

    journal = tmp_path / "quality_jobs.jsonl"
    paths = worker_artifact_paths(
        "fedot-ts-beer", "patch", "proposal-r3", 2, root=tmp_path
    )
    evaluation_identity = {"n_jobs": 4, "source_sha256": "source-v11"}
    identity = {
        "candidate": "proposal-r3",
        "task_id": "fedot-ts-beer",
        "side": "patch",
        "attempt": 2,
        "attempt_id": "proposal-r3:fedot-ts-beer:patch:2",
        "registry_identity": "registry-v11",
        "evaluation_identity": evaluation_identity,
    }
    common = {
        **identity,
        "candidate_revision_id": "proposal-r3",
        "proposal_id": "proposal-r3",
        "spec": {
            "n_jobs": 4,
            "timeout_seconds": 5400,
            "fedot_search_budget_seconds": 4500,
        },
        "worker_paths": paths,
        "worker_result_path": paths["worker_result_path"],
        "stage_status_path": paths["stage_status_path"],
        "worker_log_path": paths["worker_log_path"],
    }
    storage.append_journal(journal, {**common, "event": "quality_task_side_started"})
    timeout_payload = {
        "schema_version": 1,
        "event": "quality_worker_parent_timeout",
        **{
            key: identity[key]
            for key in (
                "task_id",
                "side",
                "attempt",
                "attempt_id",
                "registry_identity",
                "evaluation_identity",
            )
        },
        "source_dataset": "fedot-ts-beer",
        "proposal_id": "proposal-r3",
        "status": "timeout",
        "failure_kind": "timeout",
        "metric_outcome": "not_measured",
        "metric_valid": False,
        "timeout_kind": "task_timeout",
        "detail": "task_timeout after 5400s",
        "side_elapsed_seconds": 5400.0,
        "worker_elapsed_seconds": 5350.0,
        "worker_result_path": paths["worker_result_path"],
        "worker_log_path": paths["worker_log_path"],
        "stage_status_path": paths["stage_status_path"],
        "stage_observations": {"active_stage": "fit"},
    }
    timeout_payload[field] = value
    Path(paths["timeout_manifest_path"]).write_text(json.dumps(timeout_payload))

    results, _attempts = executor._recover_side_state(
        journal,
        candidate_id="proposal-r3",
        task_ids=("fedot-ts-beer",),
        registry_identity="registry-v11",
        evaluation_identity=evaluation_identity,
    )

    rejected = results[("fedot-ts-beer", "patch")]
    assert rejected.status == "crash"
    assert rejected.metric_observations["failure_kind"] == "controller_interrupted"
    assert rejected.metric_observations.get("metric_outcome") != "measured"


def test_recovers_worker_fedot_budget_timeout_without_erasing_phase(tmp_path):
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths

    journal = tmp_path / "quality_jobs.jsonl"
    paths = worker_artifact_paths(
        "fedot-ts-beer", "stock", "proposal-r4", 1, root=tmp_path
    )
    evaluation_identity = {"n_jobs": 4, "source_sha256": "source-v11"}
    identity = {
        "proposal_id": "proposal-r4",
        "task_id": "fedot-ts-beer",
        "source_dataset": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "attempt_id": "proposal-r4:fedot-ts-beer:stock:1",
        "registry_identity": "registry-v11",
        "evaluation_identity": evaluation_identity,
    }
    spec = {
        "n_jobs": 4,
        "timeout_seconds": 5400,
        "fedot_search_budget_seconds": 4500,
        "preset": "best_quality",
        "seed": 42,
        "cv_folds": 3,
        "max_pipeline_fit_time_seconds": 500,
        "max_pipeline_fit_policy": "per-cv-fold-v1",
        "max_pipeline_evaluation_time_seconds": 1500,
        "max_pipeline_evaluation_policy": "per-cv-candidate-v1",
    }
    started = {
        **identity,
        "candidate": "proposal-r4",
        "candidate_revision_id": "proposal-r4",
        "spec": spec,
        "worker_paths": paths,
        "worker_result_path": paths["worker_result_path"],
        "stage_status_path": paths["stage_status_path"],
        "worker_log_path": paths["worker_log_path"],
    }
    storage.append_journal(journal, {**started, "event": "quality_task_side_started"})
    storage.append_journal(journal, {**started, "event": "quality_task_worker_spawned"})

    result_path = Path(paths["worker_result_path"])
    stage_path = Path(paths["stage_status_path"])
    result_path.write_text(
        json.dumps(
            {
                **identity,
                "status": "timeout",
                "failure_kind": "fedot_budget_timeout",
                "failed_stage": "fit",
                "last_stage": "fit",
                "score": None,
                "duration_s": 4501.0,
                "detail": "FedotBudgetExceeded during final_refit",
                "traceback": "typed budget timeout traceback",
                "log_tail": "Final refit exceeded reserve",
                "stage_status_path": str(stage_path),
                "metric_observations": {
                    "failure_kind": "fedot_budget_timeout",
                    "metric_outcome": "not_measured",
                    "metric_valid": False,
                    "fedot_budget_phase": "final_refit",
                    "fedot_budget_seconds": 1500.0,
                    "fedot_budget_elapsed_seconds": 1501.0,
                },
            }
        )
    )
    stage_path.write_text(
        json.dumps(
            {
                **identity,
                "event": "end",
                "last_stage": "saved_payload",
                "active_stage": None,
            }
        )
    )

    results, attempts = executor._recover_side_state(
        journal,
        candidate_id="proposal-r4",
        task_ids=("fedot-ts-beer",),
        registry_identity="registry-v11",
        evaluation_identity=evaluation_identity,
    )

    recovered = results[("fedot-ts-beer", "stock")]
    assert recovered.status == "timeout"
    assert math.isnan(recovered.score)
    assert recovered.metric_observations["failure_kind"] == "fedot_budget_timeout"
    assert recovered.metric_observations["metric_outcome"] == "not_measured"
    assert recovered.metric_observations["fedot_budget_phase"] == "final_refit"
    assert recovered.metric_observations["recovered_worker_result"] is True
    assert recovered.traceback == "typed budget timeout traceback"
    assert attempts[("fedot-ts-beer", "stock")] == 1

    terminal = [
        row
        for row in executor.read_quality_events(journal)
        if row.get("event") == "quality_task_side_finished"
    ]
    assert len(terminal) == 1
    assert terminal[0]["artifact_state"] == "recovered_worker_result"
    assert terminal[0]["result"]["score"] is None
    assert (
        terminal[0]["result"]["metric_observations"]["fedot_budget_phase"]
        == "final_refit"
    )
    assert storage.read_jsonl(journal, strict=True)


@pytest.mark.parametrize(
    ("event_override", "stage_override", "payload_override"),
    [
        ({"attempt_id": "another-proposal:fedot-ts-beer:stock:1"}, {}, {}),
        ({"attempt": True}, {}, {}),
        ({}, {"attempt_id": "another-proposal:fedot-ts-beer:stock:1"}, {}),
        ({}, {"attempt": True}, {}),
        ({}, {"side": "patch"}, {}),
        ({}, {"registry_identity": "other-registry"}, {}),
        ({}, {}, {"evaluation_identity": {"n_jobs": 4, "source_sha256": "old-source"}}),
        ({}, {}, {"attempt": True}),
    ],
)
def test_recovery_rejects_worker_result_with_mismatched_identity(
    tmp_path, event_override, stage_override, payload_override
):
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths

    path = tmp_path / "quality_jobs.jsonl"
    worker_paths = worker_artifact_paths(
        "fedot-ts-beer", "stock", "proposal-r2", 1, root=tmp_path
    )
    expected_identity = {
        "proposal_id": "proposal-r2",
        "attempt_id": "proposal-r2:fedot-ts-beer:stock:1",
        "task_id": "fedot-ts-beer",
        "source_dataset": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "registry_identity": "v1",
        "evaluation_identity": {"n_jobs": 4, "source_sha256": "current-source"},
        "n_jobs": 4,
        "timeout_seconds": 3600,
    }
    event = {
        "candidate": "proposal-r2",
        "task_id": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "attempt_id": expected_identity["attempt_id"],
        "registry_identity": "v1",
        "evaluation_identity": expected_identity["evaluation_identity"],
        "spec": {"n_jobs": 4, "timeout_seconds": 3600},
        "worker_paths": worker_paths,
    }
    for phase in ("quality_task_side_started", "quality_task_worker_spawned"):
        storage.append_journal(path, {**event, **event_override, "event": phase})
    payload = {
        "task_id": "fedot-ts-beer",
        "status": "ok",
        "score": 0.91,
        "duration_s": 42.0,
        "metric_observations": {"metric_valid": True},
        "proposal_id": "proposal-r2",
        "attempt_id": expected_identity["attempt_id"],
        "source_dataset": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "registry_identity": "v1",
        "evaluation_identity": expected_identity["evaluation_identity"],
    }
    Path(worker_paths["worker_result_path"]).write_text(
        json.dumps({**payload, **payload_override})
    )
    Path(worker_paths["stage_status_path"]).write_text(
        json.dumps(
            {
                **expected_identity,
                **stage_override,
                "event": "worker_finished",
            }
        )
    )

    results, attempts = executor._recover_side_state(
        path,
        candidate_id="proposal-r2",
        task_ids=("fedot-ts-beer",),
        registry_identity="v1",
        evaluation_identity=expected_identity["evaluation_identity"],
    )

    assert results[("fedot-ts-beer", "stock")].status == "crash"
    assert attempts[("fedot-ts-beer", "stock")] == 1
    events = executor.read_quality_events(path)
    assert (
        len(
            [
                row
                for row in events
                if row.get("event") == "quality_task_side_interrupted"
            ]
        )
        == 1
    )
    assert not [
        row for row in events if row.get("event") == "quality_task_side_finished"
    ]


def test_recovery_rejects_worker_artifacts_under_symlinked_parent(tmp_path):
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths

    actual_root = tmp_path / "actual-artifacts"
    actual_root.mkdir()
    linked_root = tmp_path / "linked-artifacts"
    linked_root.symlink_to(actual_root, target_is_directory=True)
    worker_paths = worker_artifact_paths(
        "fedot-ts-beer", "stock", "proposal-r2", 1, root=linked_root
    )
    evaluation_identity = {"n_jobs": 4, "source_sha256": "current-source"}
    identity = {
        "proposal_id": "proposal-r2",
        "attempt_id": "proposal-r2:fedot-ts-beer:stock:1",
        "task_id": "fedot-ts-beer",
        "source_dataset": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "registry_identity": "v1",
        "evaluation_identity": evaluation_identity,
        "n_jobs": 4,
        "timeout_seconds": 3600,
    }
    result_path = Path(worker_paths["worker_result_path"])
    stage_path = Path(worker_paths["stage_status_path"])
    result_path.write_text(
        json.dumps(
            {
                **identity,
                "status": "ok",
                "score": 0.91,
                "duration_s": 42.0,
                "metric_observations": {"metric_valid": True},
            }
        )
    )
    stage_path.write_text(json.dumps(identity))

    recovered = executor._recover_durable_worker_result(
        {
            "candidate": "proposal-r2",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "attempt_id": identity["attempt_id"],
            "registry_identity": "v1",
            "evaluation_identity": evaluation_identity,
            "spec": {"n_jobs": 4, "timeout_seconds": 3600},
            "worker_paths": worker_paths,
        },
        candidate_id="proposal-r2",
        task_id="fedot-ts-beer",
        side="stock",
        attempt=1,
        registry_identity="v1",
    )

    assert recovered is None


def test_partial_dataset_check_cannot_accept_whole_improvement(tmp_path, monkeypatch):
    source, workspace = _enqueue(tmp_path)
    monkeypatch.setattr(
        executor,
        "measure_fedot_quality",
        lambda *_a, **_k: Decision(
            True,
            "quality_improved",
            0.1,
            stage="quality",
            quality_outcome="broad_positive",
        ),
    )
    executor.drain_quality_queue(source, workspace, task_ids=("fedot-ts-beer",))
    saved = json.loads((workspace / "quality_queue/improvement-0.json").read_text())
    assert saved["status"] == "partial"
    assert saved["keep"] is None
    assert len(saved["required_task_ids"]) == 8
    assert saved["task_ids"] == saved["required_task_ids"]
    assert saved["evaluated_task_ids"] == ["fedot-ts-beer"]
    events = storage.read_jsonl(workspace / "quality_jobs.jsonl", strict=True)
    disposition = next(
        event for event in events if event.get("event") == "quality_disposition"
    )
    assert "keep" in disposition and disposition["keep"] is None


def test_skipped_required_dataset_result_keeps_full_check_retryable(
    tmp_path, monkeypatch
):
    source, workspace = _enqueue(tmp_path)
    registry = load_quality_registry()
    all_ids = tuple(item.task_id for item in registry.datasets)
    monkeypatch.setattr(
        executor,
        "measure_fedot_quality",
        lambda *_a, **_k: Decision(
            None,
            "required_dataset_skipped:fedot-ts-beer",
            None,
            stage="quality",
            quality_outcome="not_evaluated",
            unresolved_task_ids=("fedot-ts-beer",),
        ),
    )

    executor.drain_quality_queue(source, workspace, task_ids=all_ids)

    saved = json.loads((workspace / "quality_queue/improvement-0.json").read_text())
    assert saved["status"] == "partial"
    assert saved["keep"] is None
    assert saved["unresolved_task_ids"] == ["fedot-ts-beer"]
    assert saved["evaluated_task_ids"] == saved["required_task_ids"]


@pytest.mark.parametrize("failure", ["executor", "missing_patch", "patch_rejected"])
def test_unmeasured_proposal_failure_has_no_quality_verdict(
    tmp_path, monkeypatch, failure
):
    source, workspace = _enqueue(tmp_path)
    if failure == "executor":

        def fail_measurement(*_args, **_kwargs):
            raise RuntimeError("fixture worker orchestration failed")

        monkeypatch.setattr(executor, "measure_fedot_quality", fail_measurement)
    elif failure == "missing_patch":
        (workspace / "quality_queue/improvement-0.patch").unlink()
    else:
        monkeypatch.setattr(
            executor, "apply_queue_patch", lambda *_args, **_kwargs: False
        )

    result = executor.drain_quality_queue(source, workspace, continue_on_error=True)
    item = result["jobs"][0]
    saved = json.loads((workspace / "quality_queue/improvement-0.json").read_text())

    assert item["status"] in {"failed", "apply_failed"}
    assert "keep" in item and item["keep"] is None
    assert item["quality_outcome"] == "not_evaluated"
    assert "keep" in saved and saved["keep"] is None
    assert saved["quality_outcome"] == "not_evaluated"
    events = storage.read_jsonl(workspace / "quality_jobs.jsonl", strict=True)
    dispositions = [
        event for event in events if event.get("event") == "quality_disposition"
    ]
    assert len(dispositions) == 1
    assert dispositions[0]["status"] == item["status"]
    assert "keep" in dispositions[0] and dispositions[0]["keep"] is None
    assert dispositions[0]["quality_outcome"] == "not_evaluated"


def test_unattended_queue_continues_after_dataset_timeout(tmp_path, monkeypatch):
    source, workspace = _enqueue(tmp_path, 3)
    measured = []

    def measure(_source, _experiment, *, candidate, **_kwargs):
        measured.append(candidate.candidate_id)
        if candidate.candidate_id == "improvement-0":
            return Decision(
                None,
                "timeout after retry",
                None,
                stage="quality",
                infrastructure_error=True,
                quality_outcome="not_evaluated",
                unresolved_task_ids=("fedot-ts-beer",),
            )
        return Decision(False, "no improvement", 0.0, stage="quality")

    monkeypatch.setattr(executor, "measure_fedot_quality", measure)
    result = executor.drain_quality_queue(source, workspace, continue_on_error=True)
    # The shared scheduler intentionally starts proposal jobs concurrently;
    # invocation order is therefore nondeterministic, while result order stays stable.
    assert set(measured) == {"improvement-0", "improvement-1", "improvement-2"}
    assert [job["status"] for job in result["jobs"]] == [
        "infrastructure_error",
        "drop",
        "drop",
    ]
    timed_out = result["jobs"][0]
    assert timed_out["keep"] is None
    saved = json.loads((workspace / "quality_queue/improvement-0.json").read_text())
    assert saved["keep"] is None
    events = storage.read_jsonl(workspace / "quality_jobs.jsonl", strict=True)
    disposition = next(
        event
        for event in events
        if event.get("event") == "quality_disposition"
        and event.get("candidate") == "improvement-0"
    )
    assert "keep" in disposition and disposition["keep"] is None
    assert result["paused_for_retry"] is False


def test_queue_registers_every_initial_proposal_before_pair_dispatch(
    tmp_path, monkeypatch
):
    source, workspace = _enqueue(tmp_path, 3)
    started: list[str] = []
    group_ids: set[int] = set()

    def measure(
        _source, _experiment, *, candidate, pair_scheduler, pair_group_id, **_kwargs
    ):
        assert pair_group_id not in group_ids
        group_ids.add(pair_group_id)
        pair_scheduler.run_registered_group(
            pair_group_id,
            (candidate.candidate_id,),
            lambda task_id: started.append(task_id) or task_id,
        )
        return Decision(False, "no improvement", 0.0, stage="quality")

    monkeypatch.setattr(executor, "measure_fedot_quality", measure)
    result = executor.drain_quality_queue(
        source, workspace, continue_on_error=True, n_jobs=4, cpu_quota=8
    )

    assert group_ids == {0, 1, 2}
    assert set(started) == {"improvement-0", "improvement-1", "improvement-2"}
    assert set(started[:2]) == {"improvement-0", "improvement-1"}
    assert [job["status"] for job in result["jobs"]] == ["drop", "drop", "drop"]


def test_queue_overlaps_proposals_and_refills_freed_slot_while_peer_waits(
    tmp_path, monkeypatch
):
    source, workspace = _enqueue(tmp_path, 3)
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    replacement_started = threading.Event()
    lock = threading.Lock()
    active: set[str] = set()
    peak_active = 0
    replacement_active: set[str] = set()

    def measure(
        _source, _experiment, *, candidate, pair_scheduler, pair_group_id, **_kwargs
    ):
        nonlocal peak_active
        candidate_id = candidate.candidate_id
        with lock:
            active.add(candidate_id)
            peak_active = max(peak_active, len(active))
        try:
            pair_scheduler.run_registered_group(
                pair_group_id, (candidate_id,), lambda task_id: task_id
            )
            if candidate_id == "improvement-0":
                first_started.set()
                assert second_started.wait(3), (
                    "second proposal did not overlap the first"
                )
                assert release_first.wait(3), "test did not release the first proposal"
            elif candidate_id == "improvement-1":
                second_started.set()
                assert release_second.wait(3), "test did not release the blocked peer"
            elif candidate_id == "improvement-2":
                with lock:
                    replacement_active.update(active)
                replacement_started.set()
            return Decision(False, "no improvement", 0.0, stage="quality")
        finally:
            with lock:
                active.discard(candidate_id)

    monkeypatch.setattr(executor, "measure_fedot_quality", measure)
    result_holder = []
    errors = []

    def run_queue():
        try:
            result_holder.append(
                executor.drain_quality_queue(
                    source, workspace, continue_on_error=True, n_jobs=4, cpu_quota=8
                )
            )
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run_queue, daemon=True)
    runner.start()
    try:
        assert first_started.wait(2), "first proposal did not start"
        assert second_started.wait(2), "second proposal did not overlap the first"
        release_first.set()
        assert replacement_started.wait(2), (
            "queue did not refill the freed proposal slot"
        )
        with lock:
            assert replacement_active == {"improvement-1", "improvement-2"}
            assert peak_active == 2
        release_second.set()
        runner.join(timeout=4)
        assert not runner.is_alive(), (
            "queue did not drain after all barriers were released"
        )
        assert not errors
        assert len(result_holder) == 1
        assert [job["candidate_id"] for job in result_holder[0]["jobs"]] == [
            "improvement-0",
            "improvement-1",
            "improvement-2",
        ]
        journal_events = [
            json.loads(line)
            for line in (workspace / "quality_jobs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        batch = next(
            event
            for event in journal_events
            if event.get("event") == "quality_scheduler_batch_started"
        )
        assert batch["parallel_proposals"] == 2
        assert batch["dataset_pair_slots"] == 2
        assert batch["cpu_slots"] == 8
        assert batch["cpu_quota"] == 8
        assert batch["n_jobs_per_side"] == 4
    finally:
        release_first.set()
        release_second.set()
        runner.join(timeout=4)


def test_queue_shuts_down_executors_if_initial_proposal_submit_fails(
    tmp_path, monkeypatch
):
    source, workspace = _enqueue(tmp_path, 1)
    shutdowns = {"proposal": 0, "pairs": 0}

    class FailingProposalExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, *_args, **_kwargs):
            raise RuntimeError("executor submit failed")

        def shutdown(self, **_kwargs):
            shutdowns["proposal"] += 1

    real_scheduler = executor.SharedPairScheduler

    class ObservedScheduler(real_scheduler):
        def shutdown(self, **kwargs):
            shutdowns["pairs"] += 1
            return super().shutdown(**kwargs)

    monkeypatch.setattr(executor, "ThreadPoolExecutor", FailingProposalExecutor)
    monkeypatch.setattr(executor, "SharedPairScheduler", ObservedScheduler)

    with pytest.raises(RuntimeError, match="executor submit failed"):
        executor.drain_quality_queue(source, workspace, n_jobs=4, cpu_quota=4)

    assert shutdowns == {"proposal": 1, "pairs": 1}


def test_unattended_queue_stops_when_results_cannot_be_saved(tmp_path, monkeypatch):
    source, workspace = _enqueue(tmp_path, 3)
    measured = []

    def measure(_source, _experiment, *, candidate, **_kwargs):
        measured.append(candidate.candidate_id)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(executor, "measure_fedot_quality", measure)
    with pytest.raises(OSError, match="No space left"):
        executor.drain_quality_queue(
            source, workspace, continue_on_error=True, n_jobs=4, cpu_quota=8
        )
    assert set(measured) == {"improvement-0", "improvement-1"}
    third = json.loads((workspace / "quality_queue/improvement-2.json").read_text())
    assert third["status"] == "queued"


@pytest.mark.parametrize(
    "error_text",
    [
        "[Errno 28] No space left on device",
        "[Errno 122] Disk quota exceeded",
    ],
)
def test_nested_checkout_storage_failure_stops_drain_without_failing_proposal(
    tmp_path, monkeypatch, error_text
):
    from fedotllm.agents.evolve.execution.storage_guard import StorageCapacityError

    source, workspace = _enqueue(tmp_path, 3)
    destination = (
        workspace / "experiments/quality-drain/improvement-0/.ruff_cache/CACHEDIR.TAG"
    )
    original = shutil.Error(
        [(str(source / ".ruff_cache/CACHEDIR.TAG"), str(destination), error_text)]
    )

    def failing_checkout(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(executor, "create_experiment_checkout", failing_checkout)
    with pytest.raises(StorageCapacityError) as caught:
        executor.drain_quality_queue(
            source, workspace, continue_on_error=True, n_jobs=4, cpu_quota=8
        )
    assert caught.value.errno == 28
    assert caught.value.filename == str(destination)
    assert any(
        candidate in str(caught.value)
        for candidate in ("improvement-0", "improvement-1")
    )
    assert caught.value.__cause__ is original
    for name in ("improvement-0", "improvement-1", "improvement-2"):
        proposal = json.loads((workspace / f"quality_queue/{name}.json").read_text())
        assert proposal["status"] == "queued"
        assert proposal.get("keep") is None
    events = storage.read_jsonl(workspace / "quality_jobs.jsonl", strict=True)
    started_candidates = {
        event.get("candidate")
        for event in events
        if event["event"] == "quality_candidate_started"
    }
    assert started_candidates == {"improvement-0", "improvement-1"}
    assert events[0]["event"] == "quality_scheduler_batch_started"


def test_candidate_start_event_reports_effective_pair_cap_not_cpu_derived_capacity(
    tmp_path, monkeypatch
):
    source, workspace = _enqueue(tmp_path, 1)
    journal_events = []
    monkeypatch.setattr(
        executor,
        "measure_fedot_quality",
        lambda *_args, **_kwargs: Decision(
            True, "quality_improved", 0.1, stage="quality"
        ),
    )
    monkeypatch.setattr(
        executor, "append_journal", lambda _path, event: journal_events.append(event)
    )

    executor.drain_quality_queue(
        source,
        workspace,
        continue_on_error=True,
        n_jobs=4,
        cpu_quota=32,
        max_pair_slots=4,
    )

    batch = next(
        event
        for event in journal_events
        if event["event"] == "quality_scheduler_batch_started"
    )
    candidate = next(
        event
        for event in journal_events
        if event["event"] == "quality_candidate_started"
    )
    assert batch["dataset_pair_slots"] == 4
    assert candidate["dataset_pair_slots"] == batch["dataset_pair_slots"]


def test_side_start_and_worker_events_report_effective_scheduler_pair_cap(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    source, workspace = _enqueue(tmp_path, 1)
    experiment = tmp_path / "experiment"
    (experiment / "fedot").mkdir(parents=True)
    journal_events = []

    monkeypatch.setattr(executor, "source_fingerprint", lambda _path: "pinned-source")
    monkeypatch.setattr(
        executor, "interpreter_identity", lambda _path: {"python": "pinned"}
    )
    monkeypatch.setattr(
        executor, "append_journal", lambda _path, event: journal_events.append(event)
    )
    monkeypatch.setattr(
        executor,
        "decision_from_quality_jobs",
        lambda _rows: Decision(True, "quality_improved", 0.1, stage="quality"),
    )

    def fake_run_quality_jobs(**kwargs):
        spec = {
            "source_dataset": "fedot-ts-beer",
            "stock_flight_id": "flight-1",
            "scheduler_queue_wait_seconds": 23.5,
        }
        kwargs["on_side_event"](spec, "stock", "started", None, 1)
        kwargs["on_side_event"](spec, "stock", "spawned", None, 1)
        return []

    monkeypatch.setattr(executor, "run_quality_jobs", fake_run_quality_jobs)
    scheduler = SimpleNamespace(
        max_pairs=4,
        active_pair_slots=1,
        queued_pair_tasks=0,
        active_compute_workers=1,
        peak_compute_workers=4,
    )

    executor.measure_fedot_quality(
        source,
        experiment,
        candidate=PatchCandidate("candidate", "fedot/example.py"),
        task_ids=("fedot-ts-beer",),
        n_jobs=4,
        cpu_quota=32,
        pair_scheduler=scheduler,
        journal=workspace / "quality_jobs.jsonl",
    )

    events = {event["event"]: event for event in journal_events}
    assert events["quality_task_side_started"]["dataset_pair_slots"] == 4
    assert events["quality_task_worker_spawned"]["dataset_pair_slots"] == 4
    assert events["quality_task_worker_spawned"]["scheduler_queue_wait_seconds"] == 23.5


def test_registry_profile_bounds_quality_worker_and_enters_cache_identity():
    from fedotllm.agents.evolve.evaluation.fedot_quality import stock_cache_identity
    from fedotllm.agents.evolve.evaluation.quality_registry import job_spec

    registry = load_quality_registry()
    dataset = next(
        item for item in registry.datasets if item.problem == "ts_forecasting"
    )
    spec = job_spec(dataset, registry=registry)
    assert spec["watchdog_seconds"] == 10800
    assert spec["max_pipeline_fit_policy"] == "per-cv-fold-v1"
    assert spec["cv_folds"] == 3
    assert spec["timeout_seconds"] == 5400
    assert spec["fedot_search_budget_seconds"] == 4500
    assert spec["timeout_minutes"] == 75.0
    assert spec["max_pipeline_fit_time_seconds"] == 500
    assert spec["max_pipeline_evaluation_time_seconds"] == 1500
    assert spec["max_pipeline_evaluation_policy"] == "per-cv-candidate-v1"
    identity = stock_cache_identity(dataset, registry)
    assert identity["watchdog_seconds"] == 10800
    assert identity["max_pipeline_fit_policy"] == "per-cv-fold-v1"
    assert identity["fedot_search_budget_seconds"] == 4500
    assert identity["max_pipeline_fit_time_seconds"] == 500


def test_different_registry_creates_new_check_without_resetting_old_result(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    source, workspace = _enqueue(tmp_path)
    path = workspace / "quality_queue/improvement-0.json"
    proposal = json.loads(path.read_text())
    queue.write_job_status(path, "drop", {"reason": "measured"})
    registry = load_quality_registry()
    monkeypatch.setattr(
        queue,
        "load_quality_registry",
        lambda: replace(registry, identity="new-registry"),
    )
    revised = queue.enqueue_quality_job(
        workspace,
        candidate=PatchCandidate("improvement-0", file_path=proposal["file_path"]),
        patch_text=path.with_suffix(".patch").read_text(),
        hint="same patch",
        priority="normal",
    )
    assert revised != path
    assert json.loads(path.read_text())["status"] == "drop"
    assert json.loads(revised.read_text())["registry_identity"] == "new-registry"


def test_stale_registry_is_blocked_before_measurement(tmp_path, monkeypatch):
    source, workspace = _enqueue(tmp_path)
    path = workspace / "quality_queue/improvement-0.json"
    proposal = json.loads(path.read_text())
    proposal["registry_identity"] = "obsolete-registry"
    path.write_text(json.dumps(proposal))
    measured = []
    monkeypatch.setattr(
        executor, "measure_fedot_quality", lambda *_a, **_k: measured.append(True)
    )
    result = executor.drain_quality_queue(source, workspace, continue_on_error=True)
    assert measured == []
    assert result["jobs"][0]["status"] == "blocked"
    assert result["jobs"][0]["reason"] == "registry_mismatch"


def test_changed_patch_bytes_cannot_be_measured_as_frozen_revision(
    tmp_path, monkeypatch
):
    source, workspace = _enqueue(tmp_path)
    path = workspace / "quality_queue/improvement-0.patch"
    path.write_text(path.read_text().replace("value = 2", "value = 3"))
    measured = []
    monkeypatch.setattr(
        executor, "measure_fedot_quality", lambda *_a, **_k: measured.append(True)
    )
    result = executor.drain_quality_queue(source, workspace, continue_on_error=True)
    assert measured == []
    assert result["jobs"][0]["reason"] == "patch_hash_mismatch"


def test_resume_rejects_results_from_different_measurement_identity(tmp_path):
    path = tmp_path / "quality_jobs.jsonl"
    storage.append_journal(
        path,
        {
            "event": "quality_task_side_finished",
            "candidate": "a",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "registry_identity": "v1",
            "evaluation_identity": {"n_jobs": 1},
            "result": {"status": "ok", "score": 0.9},
        },
    )
    results, attempts = executor._recover_side_state(
        path,
        candidate_id="a",
        task_ids=("fedot-ts-beer",),
        registry_identity="v1",
        evaluation_identity={"n_jobs": 4},
    )
    assert results == {}
    assert attempts == {}


def test_resume_uses_durable_side_event_when_jsonl_tail_is_truncated(tmp_path):
    from fedotllm.agents.evolve.storage.quality_events import save_quality_event

    path = tmp_path / "quality_jobs.jsonl"
    durable = save_quality_event(
        path,
        {
            "event": "quality_task_side_finished",
            "candidate": "proposal-r2",
            "proposal_id": "proposal",
            "candidate_revision_id": "proposal-r2",
            "task_id": "fedot-ts-beer",
            "side": "stock",
            "attempt": 1,
            "registry_identity": "v1",
            "evaluation_identity": {"n_jobs": 4},
            "result": {
                "status": "ok",
                "score": 0.91,
                "duration_s": 12.0,
                "metric_observations": {"metric_valid": True},
            },
        },
    )
    path.write_bytes(
        (json.dumps(durable) + "\n").encode("utf-8") + b'{"event":"unfinished'
    )

    results, attempts = executor._recover_side_state(
        path,
        candidate_id="proposal-r2",
        task_ids=("fedot-ts-beer",),
        registry_identity="v1",
        evaluation_identity={"n_jobs": 4},
    )

    assert results[("fedot-ts-beer", "stock")].score == 0.91
    assert attempts[("fedot-ts-beer", "stock")] == 1


def test_resume_still_rejects_corruption_before_jsonl_tail(tmp_path):
    path = tmp_path / "quality_jobs.jsonl"
    path.write_bytes(b'{"event": malformed}\n{"event":"unfinished')
    original = path.read_bytes()

    with pytest.raises(storage.JournalCorruptionError):
        executor._recover_side_state(
            path,
            candidate_id="proposal-r2",
            task_ids=("fedot-ts-beer",),
            registry_identity="v1",
        )
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("quality_jobs.jsonl.incomplete-tail.*.bin"))


def test_resume_terminates_complete_json_object_without_final_newline(tmp_path):
    path = tmp_path / "quality_jobs.jsonl"
    event = {
        "event": "quality_task_side_finished",
        "candidate": "candidate-a",
        "task_id": "fedot-ts-beer",
        "side": "stock",
        "attempt": 1,
        "registry_identity": "registry-v1",
        "result": {
            "status": "ok",
            "score": 0.61,
            "metric_observations": {"metric_valid": True},
        },
    }
    encoded = json.dumps(event).encode("utf-8")
    path.write_bytes(encoded)

    results, _attempts = executor._recover_side_state(
        path,
        candidate_id="candidate-a",
        task_ids=("fedot-ts-beer",),
        registry_identity="registry-v1",
    )

    assert path.read_bytes().startswith(encoded + b"\n")
    assert storage.read_jsonl(path, strict=True)
    assert results[("fedot-ts-beer", "stock")].score == 0.61
    assert not list(tmp_path.glob("quality_jobs.jsonl.incomplete-tail.*.bin"))


def test_quality_journal_delta_is_missing_without_metric_attestation():
    from fedotllm.agents.evolve.controller.quality_executor import _quality_delta

    spec = {"higher_is_better": False, "delta_mode": "relative"}
    stock = {"status": "ok", "score": 100.0, "metric_observations": {}}
    patched = {
        "status": "ok",
        "score": 90.0,
        "metric_observations": {"metric_valid": True},
    }

    assert _quality_delta(spec, stock, patched) is None


def test_recovery_contains_malformed_metric_observation_payload(tmp_path):
    path = tmp_path / "quality_jobs.jsonl"
    storage.append_journal(
        path,
        {
            "event": "quality_task_side_finished",
            "candidate": "candidate-invalid",
            "task_id": "fedot-ts-traffic",
            "side": "stock",
            "attempt": 1,
            "registry_identity": "registry-v1",
            "result": {
                "status": "ok",
                "score": 0.8,
                "metric_observations": ["malformed"],
            },
        },
    )

    results, _attempts = executor._recover_side_state(
        path,
        candidate_id="candidate-invalid",
        task_ids=("fedot-ts-traffic",),
        registry_identity="registry-v1",
    )

    recovered = results[("fedot-ts-traffic", "stock")]
    assert recovered.status == "invalid"
    assert recovered.metric_observations["failure_kind"] == "unverified_metric"


def test_shared_stock_wait_is_durably_written_to_quality_journal(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller import quality_executor
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        get_dataset,
        job_spec,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import PatchCandidate

    registry = load_quality_registry()
    spec = job_spec(get_dataset("fedot-ts-traffic", registry), registry=registry)
    spec["stock_flight_id"] = "flight-traffic-6a7b"

    def fake_jobs(**kwargs):
        kwargs["on_side_wait"](spec, "stock", "shared_stock_fit", 60.0, 1)
        return []

    monkeypatch.setattr(quality_executor, "run_quality_jobs", fake_jobs)
    journal = tmp_path / "quality_jobs.jsonl"
    quality_executor.measure_fedot_quality(
        tmp_path / "stock",
        tmp_path / "patch",
        journal=journal,
        candidate=PatchCandidate("candidate-wait"),
        task_ids=("fedot-ts-traffic",),
        registry=registry,
    )

    waiting = [
        event
        for event in quality_executor.read_quality_events(journal)
        if event.get("event") == "quality_task_side_waiting_stock_cache"
    ]
    assert len(waiting) == 1
    assert waiting[0]["task_id"] == "fedot-ts-traffic"
    assert waiting[0]["wait_kind"] == "shared_stock_fit"
    assert waiting[0]["wait_elapsed_seconds"] == 60.0
    assert waiting[0]["stock_flight_id"] == "flight-traffic-6a7b"


def test_finished_side_survives_journal_append_failure(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset, job_spec
    from fedotllm.agents.evolve.types import ScoreResult

    source, workspace = _enqueue(tmp_path)
    workspace.mkdir(exist_ok=True)
    path = workspace / "quality_jobs.jsonl"
    original_append = executor.append_journal
    registry = load_quality_registry()

    def fail_finished(destination, event):
        if event["event"] == "quality_task_side_finished":
            raise OSError(28, "No space left on device")
        return original_append(destination, event)

    def run_jobs(*_args, on_side_event, **_kwargs):
        spec = job_spec(get_dataset("fedot-ts-beer"), registry=registry)
        on_side_event(spec, "stock", "started", None, 1)
        on_side_event(
            spec,
            "stock",
            "finished",
            ScoreResult(
                "fedot-ts-beer",
                "ok",
                0.9,
                metric_observations={"metric_valid": True},
            ),
            1,
        )
        raise AssertionError("append failure must propagate")

    monkeypatch.setattr(executor, "append_journal", fail_finished)
    monkeypatch.setattr(executor, "run_quality_jobs", run_jobs)
    with pytest.raises(OSError, match="No space left"):
        executor.measure_fedot_quality(
            source,
            source,
            journal=path,
            candidate=PatchCandidate("a"),
            task_ids=("fedot-ts-beer",),
        )
    results, attempts = executor._recover_side_state(
        path,
        candidate_id="a",
        task_ids=("fedot-ts-beer",),
        registry_identity=registry.identity,
    )
    assert results[("fedot-ts-beer", "stock")].score == 0.9
    assert attempts[("fedot-ts-beer", "stock")] == 1


def test_legacy_narrow_scope_requires_new_check_instead_of_partial_keep(
    tmp_path, monkeypatch
):
    source, workspace = _enqueue(tmp_path)
    path = workspace / "quality_queue/improvement-0.json"
    proposal = json.loads(path.read_text())
    proposal.pop("scope_policy_version")
    proposal.pop("required_task_ids")
    proposal["changed_files"] = ["fedot/core/data/data.py"]
    proposal["changed_code_tags"] = ["categorical"]
    proposal["hypothesis"] = {"claim": "Preserve categorical features"}
    registry = load_quality_registry()
    proposal["task_ids"] = [
        item.task_id for item in registry.datasets if "categorical" in item.tags
    ]
    proposal["quality_scope"] = {
        "problem_families": ["classification", "regression", "ts_forecasting"],
        "dataset_traits": ["categorical"],
    }
    path.write_text(json.dumps(proposal))
    measured = []
    monkeypatch.setattr(
        executor, "measure_fedot_quality", lambda *_a, **_k: measured.append(True)
    )
    result = executor.drain_quality_queue(source, workspace, continue_on_error=True)
    assert measured == []
    assert result["jobs"][0]["reason"] == "routing_scope_changed"

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "research/evolve/scripts/sync_hour_verdicts.py"
SPEC = importlib.util.spec_from_file_location("sync_hour_verdicts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sync_hour_verdicts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync_hour_verdicts)


def _write_proposal(path: Path, candidate: str, status: str, reason: str, keep: bool):
    path.write_text(
        json.dumps(
            {
                "candidate_id": candidate,
                "status": status,
                "reason": reason,
                "keep": keep,
                "hypothesis": {"claim": f"Claim for {candidate}"},
                "task_ids": ["task-1"],
            }
        ),
        encoding="utf-8",
    )


def test_quality_rollup_does_not_count_infrastructure_failures_as_drops(tmp_path):
    queue = tmp_path / "quality_queue"
    queue.mkdir()
    _write_proposal(queue / "infra.json", "infra", "infrastructure_error", "worker_crash", False)
    _write_proposal(queue / "drop.json", "drop", "drop", "no_quality_gain", False)
    _write_proposal(queue / "keep.json", "keep", "keep", "quality_improved", True)
    journal = [
        {
            "event": "fedot_quality",
            "candidate": "infra",
            "decision": {"keep": False, "infrastructure_error": True, "reason": "worker_crash"},
            "jobs": [],
        },
        {
            "event": "fedot_quality",
            "candidate": "drop",
            "decision": {"keep": False, "infrastructure_error": False, "reason": "no_quality_gain"},
            "jobs": [],
        },
        {
            "event": "fedot_quality",
            "candidate": "keep",
            "decision": {"keep": True, "infrastructure_error": False, "reason": "quality_improved"},
            "jobs": [],
        },
    ]

    summary = sync_hour_verdicts.summarize(journal, queue)

    assert summary["keep_true"] == ["keep"]
    assert summary["keep_false_count"] == 1
    assert summary["candidates"]["infra"]["status"] == "infrastructure_error"
    assert summary["candidates"]["drop"]["status"] == "drop"
    report = sync_hour_verdicts.render_md(summary)
    assert "**Quality drops:** 1" in report
    assert "**Infrastructure errors (not quality drops):** 1" in report


def test_quality_rollup_keeps_candidate_and_evaluation_errors_out_of_drop_count(tmp_path):
    queue = tmp_path / "quality_queue"
    queue.mkdir()
    for candidate, failure in (
        ("candidate", "candidate_error"),
        ("evaluation", "evaluation_error"),
    ):
        (queue / f"{candidate}.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate,
                    "status": failure,
                    "quality_failure": failure,
                    "keep": False,
                    "reason": f"{failure}:task-1",
                    "task_ids": ["task-1"],
                }
            ),
            encoding="utf-8",
        )
    journal = [
        {
            "event": "fedot_quality",
            "candidate": candidate,
            "decision": {
                "keep": False,
                "infrastructure_error": False,
                "quality_failure": failure,
                "reason": f"{failure}:task-1",
            },
            "jobs": [],
        }
        for candidate, failure in (
            ("candidate", "candidate_error"),
            ("evaluation", "evaluation_error"),
        )
    ]

    summary = sync_hour_verdicts.summarize(journal, queue)

    assert summary["keep_false_count"] == 0
    assert summary["quality_failure_counts"] == {
        "candidate_error": 1,
        "evaluation_error": 1,
    }
    assert summary["candidates"]["candidate"]["status"] == "candidate_error"
    report = sync_hour_verdicts.render_md(summary)
    assert "**Quality drops:** 0" in report
    assert "**Candidate-side errors:** 1" in report
    assert "**Evaluation/baseline errors:** 1" in report


def test_orphan_journal_rows_are_not_assigned_by_matching_reason(tmp_path):
    queue = tmp_path / "quality_queue"
    queue.mkdir()
    _write_proposal(
        queue / "candidate.json",
        "candidate",
        "infrastructure_error",
        "composing_did_not_start:task-1",
        False,
    )
    orphan = {
        "event": "fedot_quality",
        "candidate": None,
        "decision": {"keep": False, "infrastructure_error": True, "reason": "composing_did_not_start:task-1"},
        "task_ids": ["task-1"],
        "jobs": [
            {
                "spec": {"source_dataset": "task-1", "metric": "roc_auc", "higher_is_better": True},
                "stock_status": "ok",
                "patched_status": "invalid",
                "stock_score": 0.8,
                "patched_score": None,
            }
        ],
    }

    summary = sync_hour_verdicts.summarize([orphan], queue)

    assert summary["candidates"]["candidate"]["jobs"] == []
    assert len(summary["unattributed_journal_rows"]) == 1
    assert summary["unattributed_journal_rows"][0]["jobs"][0]["spec"]["source_dataset"] == "task-1"
    assert "not assigned to any candidate" in sync_hour_verdicts.render_md(summary)
    assert sync_hour_verdicts._verdict_numbers([orphan], sync_hour_verdicts._queue_meta(queue)) == {}

    details = tmp_path / "details.jsonl"
    sync_hour_verdicts.write_detail_jsonl(summary, details)
    detail_rows = [json.loads(line) for line in details.read_text(encoding="utf-8").splitlines()]
    detail_row = detail_rows[-1]
    assert detail_row["candidate_id"] is None
    assert detail_row["unattributed"] is True
    assert detail_row["datasets"][0]["task_id"] == "task-1"

    rollup = tmp_path / "rollup.json"
    sync_hour_verdicts.write_campaign_rollup(summary, rollup)
    assert json.loads(rollup.read_text(encoding="utf-8"))["unattributed_journal_row_count"] == 1


def test_live_report_builds_per_dataset_progress_from_side_events(tmp_path):
    queue = tmp_path / "quality_queue"
    queue.mkdir()
    (queue / "candidate-live.json").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-live",
                "status": "running",
                "file_path": "fedot/core/ts/model.py",
                "hypothesis": {"claim": "Improve forecasts"},
                "task_ids": ["task-1", "task-2"],
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "source_dataset": "task-1",
        "metric": "mae",
        "higher_is_better": False,
    }
    journal = [
        {
            "event": "quality_task_side_started",
            "candidate": "candidate-live",
            "task_id": "task-1",
            "side": "stock",
            "attempt": 1,
            "spec": spec,
            "ts": "2026-09-28T01:00:00+00:00",
        },
        {
            "event": "quality_task_side_finished",
            "candidate": "candidate-live",
            "task_id": "task-1",
            "side": "stock",
            "attempt": 1,
            "spec": spec,
            "result": {"status": "ok", "score": 10.0, "worker_log_path": "/tmp/stock.log"},
            "ts": "2026-09-28T01:02:00+00:00",
        },
        {
            "event": "quality_task_side_finished",
            "candidate": "candidate-live",
            "task_id": "task-1",
            "side": "patch",
            "attempt": 1,
            "spec": spec,
            "result": {"status": "ok", "score": 9.7, "worker_log_path": "/tmp/patch.log"},
            "ts": "2026-09-28T01:05:00+00:00",
        },
        {
            "event": "quality_task_side_started",
            "candidate": "candidate-live",
            "task_id": "task-2",
            "side": "stock",
            "attempt": 1,
            "spec": {**spec, "source_dataset": "task-2"},
            "ts": "2026-09-28T01:06:00+00:00",
        },
    ]

    summary = sync_hour_verdicts.summarize(journal, queue)

    candidate = summary["candidates"]["candidate-live"]
    assert candidate["planned_task_count"] == 2
    assert candidate["completed_task_pair_count"] == 1
    assert candidate["active_task_ids"] == ["task-2"]
    by_task = {row["task_id"]: row for row in candidate["datasets"]}
    assert by_task["task-1"]["stock_status"] == "ok"
    assert by_task["task-1"]["patched_status"] == "ok"
    assert round(by_task["task-1"]["delta"], 6) == 0.3
    assert by_task["task-2"]["stock_status"] == "running"
    report = sync_hour_verdicts.render_md(summary)
    assert "1/2" in report
    assert "task-2" in report


def test_live_report_counts_reused_cached_side_as_finished(tmp_path):
    queue = tmp_path / "quality_queue"
    queue.mkdir()
    (queue / "candidate-cache.json").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-cache",
                "status": "running",
                "task_ids": ["task-cache"],
            }
        ),
        encoding="utf-8",
    )
    spec = {"source_dataset": "task-cache", "metric": "rmse", "higher_is_better": False}
    journal = [
        {
            "event": "quality_task_side_reused",
            "candidate": "candidate-cache",
            "task_id": "task-cache",
            "side": "stock",
            "attempt": 1,
            "spec": spec,
            "result": {"status": "ok", "score": 3.0},
        },
        {
            "event": "quality_task_side_finished",
            "candidate": "candidate-cache",
            "task_id": "task-cache",
            "side": "patch",
            "attempt": 1,
            "spec": spec,
            "result": {"status": "ok", "score": 2.8},
        },
    ]

    summary = sync_hour_verdicts.summarize(journal, queue)
    candidate = summary["candidates"]["candidate-cache"]

    assert candidate["completed_task_pair_count"] == 1
    assert candidate["measured_task_pair_count"] == 1
    assert abs(candidate["datasets"][0]["delta"] - 0.2) < 1e-9


def test_history_index_joins_only_exact_candidate_and_task_ids(tmp_path):
    import hashlib
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "research/evolve/scripts/index_quality_history.py"
    assert script.is_file(), "history index importer should be available"
    spec = importlib.util.spec_from_file_location("index_quality_history", script)
    assert spec is not None and spec.loader is not None
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)

    run = tmp_path / "old-run"
    queue = run / "quality_queue"
    queue.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": "run-exact", "registry_identity": "registry-v1"}),
        encoding="utf-8",
    )
    proposal = {
        "candidate_id": "candidate-1",
        "task_ids": ["task-1"],
        "status": "drop",
    }
    proposal_path = queue / "candidate-1.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    rows = [
        {"event": "fedot_quality_task_finished", "candidate": "candidate-1", "task_id": "task-1"},
        {"event": "fedot_quality_task_finished", "task_id": "task-1"},
        {"event": "fedot_quality_task_finished", "candidate": "candidate-1", "task_id": "task-unknown"},
        {"event": "fedot_quality_task_finished", "candidate": "other-candidate", "task_id": "task-1"},
    ]
    journal_path = run / "quality_jobs.jsonl"
    journal_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (proposal_path, journal_path)
    }

    output = tmp_path / "history.jsonl"
    summary = importer.build_quality_history_index([run], output)
    imported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    proposal = next(row for row in imported if row["record_type"] == "proposal")
    unresolved = [row for row in imported if row["record_type"] == "unresolved_event"]

    assert summary == {"proposals": 1, "linked_events": 1, "unresolved_events": 3, "malformed_records": 0}
    assert proposal["run_id"] == "run-exact"
    assert [row["task_ids"] for row in proposal["journal_events"]] == [["task-1"]]
    assert {row["reason"] for row in unresolved} == {
        "missing_candidate_id",
        "task_not_in_frozen_scope",
        "candidate_not_in_run_queue",
    }
    assert all("journal_events" not in row for row in unresolved)
    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_hashes
    }


def test_history_index_cli_discovers_runs_under_campaign_root(tmp_path, monkeypatch):
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[2] / "research/evolve/scripts/index_quality_history.py"
    spec = importlib.util.spec_from_file_location("index_quality_history_cli", script)
    assert spec is not None and spec.loader is not None
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    run = tmp_path / "campaigns" / "run-a"
    queue = run / "quality_queue"
    queue.mkdir(parents=True)
    (queue / "candidate-a.json").write_text(
        json.dumps({"candidate_id": "candidate-a", "task_ids": ["task-a"]}),
        encoding="utf-8",
    )
    output = tmp_path / "history.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["index_quality_history.py", "--campaign-root", str(tmp_path / "campaigns"), "--output", str(output)],
    )

    assert importer.main() == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-a"


def test_report_shows_worker_log_activity_without_claiming_process_is_live(tmp_path):
    summary = sync_hour_verdicts.summarize([], tmp_path / "quality_queue")
    sync_hour_verdicts.attach_run_manifest(
        summary,
        {
            "last_confirmed_server_snapshot_utc": "2026-09-27T23:07:31Z",
            "current_stage": "stock_baseline",
            "latest_artifact_observation": {
                "observed_at_utc": "2026-09-27T23:07:31Z",
                "observed_worker_tasks": ["fedot-ts-traffic", "fedot-ts-metocean"],
                "latest_worker_log_update_utc": "2026-09-27T22:24:29Z",
                "process_state": "not_checked_by_user_instruction",
            },
        },
    )

    report = sync_hour_verdicts.render_md(summary)

    assert "Последний наблюдавшийся этап: `stock_baseline`" in report
    assert "fedot-ts-traffic, fedot-ts-metocean" in report
    assert "Живость процесса этим снимком не подтверждена" in report


def test_report_recalculates_frozen_outcome_without_mixing_metric_families():
    rows = [
        {
            "task_id": f"task-{index}",
            "metric": "rmse" if index < 4 else "roc_auc",
            "stock_status": "ok",
            "patched_status": "ok",
            "stock_score": 1.0,
            "patched_score": 1.0 + (0.02 if index < 5 else 0.0),
            "delta": 0.02 if index < 5 else 0.0,
            "min_delta": 0.01,
        }
        for index in range(8)
    ]
    broad = sync_hour_verdicts.derive_quality_outcome(rows, [row["task_id"] for row in rows])
    assert broad["outcome"] == "broad_positive"
    assert broad["wins"] == 5

    rows[6]["patched_score"] = 1.001
    rows[6]["delta"] = -0.001
    regression = sync_hour_verdicts.derive_quality_outcome(rows, [row["task_id"] for row in rows])
    assert regression["outcome"] == "regression"
    assert regression["losses"] == 1

    rows[6]["patched_status"] = "timeout"
    incomplete = sync_hour_verdicts.derive_quality_outcome(rows, [row["task_id"] for row in rows])
    assert incomplete["outcome"] == "not_evaluated"
    assert incomplete["unresolved_task_ids"] == ["task-6"]

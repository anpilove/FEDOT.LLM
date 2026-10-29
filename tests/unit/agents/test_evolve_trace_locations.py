"""Saved stack evidence reports locations without diagnosing deadlock from repetition."""

import json

from fedotllm.agents.evolve.controller.diagnose import (
    _stack_trace_evidence,
    diagnose_run,
)


def test_stack_report_selects_latest_dump_and_preserves_compute_and_wait_threads(
    tmp_path,
):
    path = tmp_path / "worker.stacks.txt"
    path.write_text("""Timeout (0:01:00)!
Thread 0x111 (most recent call first):
  File "/fedot/old.py", line 9 in old_operation

Timeout (0:01:00)!
Thread 0x222 (most recent call first):
  File "/statsmodels/base/model.py", line 578 in fit
  File "/fedot/core/pipelines/pipeline.py", line 220 in _fit

Thread 0x111 (most recent call first):
  File "/joblib/parallel.py", line 1762 in _retrieve
  File "/fedot/api/main.py", line 279 in fit
""")

    report = _stack_trace_evidence(path)

    assert report["stack_dump_count"] == 2
    assert report["stack_threads"] == [
        {
            "thread": "0x222",
            "frames": [
                {"file": "/statsmodels/base/model.py", "line": 578, "function": "fit"},
                {
                    "file": "/fedot/core/pipelines/pipeline.py",
                    "line": 220,
                    "function": "_fit",
                },
            ],
        },
        {
            "thread": "0x111",
            "frames": [
                {"file": "/joblib/parallel.py", "line": 1762, "function": "_retrieve"},
                {"file": "/fedot/api/main.py", "line": 279, "function": "fit"},
            ],
        },
    ]
    assert report["stack_frame_scope"] == "python_frames_in_this_process"
    assert report["last_stack_dump_at"] is None


def test_repeated_stacks_are_evidence_not_a_deadlock_verdict(tmp_path):
    path = tmp_path / "worker.stacks.txt"
    first = 'Timeout (0:01:00)!\nThread 0x123 (most recent call first):\n  File "/fedot/search.py", line 42 in search\n'
    same = 'Timeout (0:01:00)!\nThread 0x123 (most recent call first):\n  File "/scipy/optimize.py", line 100 in minimize\n'
    path.write_text(first + same + same)

    report = _stack_trace_evidence(path)

    assert report["repeated_stack_snapshots"] == 2
    assert report["stack_repetition_proves_hang"] is False
    assert report["stack_threads"][0]["frames"][0]["function"] == "minimize"


def test_incomplete_stack_frame_does_not_hide_prior_valid_frames(tmp_path):
    path = tmp_path / "worker.stacks.txt"
    path.write_text(
        'Timeout (0:01:00)!\nThread 0x123 (most recent call first):\n  File "/fedot/model.py", line 37 in fit\n  File "/partial'
    )

    report = _stack_trace_evidence(path)

    assert report["stack_threads"][0]["frames"] == [
        {"file": "/fedot/model.py", "line": 37, "function": "fit"},
    ]
    assert report["stack_snapshot_completeness"] == "unknown"


def test_diagnosis_reads_only_manifest_children_of_the_active_attempt(tmp_path):
    attempt = "proposal:beer:stock:1"
    traces = tmp_path / "quality-out-beer.process-traces"
    traces.mkdir()
    stack = traces / "child-21.stacks.txt"
    stack.write_text(
        'Timeout (0:00:15)!\nThread 0xabc (most recent call first):\n  File "/fedot/operation.py", line 50 in fit\n'
    )
    (traces / "unlisted.stacks.txt").write_text("must not be discovered")
    manifest = traces / "manifest.jsonl"
    entry = {
        "attempt_id": attempt,
        "pid": 21,
        "ppid": 20,
        "started_at": "2026-10-09T10:00:00+00:00",
        "stack_trace_path": "/remote/quality-out-beer.process-traces/child-21.stacks.txt",
    }
    manifest.write_text(
        json.dumps(entry)
        + "\n"
        + json.dumps({**entry, "attempt_id": "other"})
        + '\n{"incomplete":'
    )
    (tmp_path / "quality_jobs.jsonl").write_text(
        json.dumps(
            {
                "event": "quality_task_worker_spawned",
                "attempt_id": attempt,
                "candidate": "proposal",
                "task_id": "beer",
                "side": "stock",
                "attempt": 1,
                "worker_paths": {
                    "process_trace_manifest_path": "/remote/quality-out-beer.process-traces/manifest.jsonl"
                },
            }
        )
        + "\n"
    )

    report = diagnose_run(tmp_path)
    evidence = report["worker_checks"][0]["child_trace_evidence"]

    assert evidence["manifest_state"] == "present"
    assert evidence["invalid_manifest_rows"] == 1
    assert evidence["coverage"] == "observed_python_children_only"
    assert len(evidence["processes"]) == 1
    assert evidence["processes"][0]["pid"] == 21
    assert evidence["processes"][0]["stack_threads"][0]["frames"][0] == {
        "file": "/fedot/operation.py",
        "line": 50,
        "function": "fit",
    }


def test_signal_stream_does_not_invent_dump_counts_or_wall_clock_times(tmp_path):
    path = tmp_path / "signal.stacks.txt"
    path.write_text(
        'Thread 0x123 (most recent call first):\n  File "/fedot/model.py", line 37 in fit\n'
    )

    report = _stack_trace_evidence(path)

    assert report["stack_threads"][0]["frames"][0]["line"] == 37
    assert report["stack_dump_count"] == 0
    assert report["repeated_stack_snapshots"] == 0
    assert report["stack_snapshot_boundary"] == "unmarked_stream"
    assert report["last_stack_dump_at"] is None


def test_controller_stacks_remain_visible_without_service_status(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    stack = logs / "controller.stacks.txt"
    stack.write_text(
        "--- stack dump at 2026-10-09T10:00:00+00:00 ---\n"
        "Thread 0xabc (most recent call first):\n"
        '  File "/controller/queue.py", line 89 in wait_for_result\n'
        "--- stack dump complete ---\n"
    )
    (tmp_path / "quality_jobs.jsonl").write_text("")

    report = diagnose_run(tmp_path)

    assert report["state"] == "unknown"
    assert report["controller_stack_evidence"]["stack_threads"][0]["frames"][0] == {
        "file": "/controller/queue.py",
        "line": 89,
        "function": "wait_for_result",
    }
    assert (
        report["controller_stack_evidence"]["stack_snapshot_completeness"] == "complete"
    )

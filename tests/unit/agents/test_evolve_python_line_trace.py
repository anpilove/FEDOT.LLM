import builtins
import importlib.util
import json
import threading
import time

import pytest

from fedotllm.agents.evolve.evaluation import python_line_trace
from fedotllm.agents.evolve.evaluation.python_line_trace import python_line_samples


def test_line_samples_pin_a_blocked_callsite_to_fedot_source(tmp_path, monkeypatch):
    fedot_root = tmp_path / "FEDOT"
    package = fedot_root / "fedot"
    package.mkdir(parents=True)
    source_file = package / "slow_operation.py"
    source_file.write_text(
        "import time\n"
        "def blocked(release):\n"
        "    while not release.is_set():\n"
        "        time.sleep(0.005)\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("slow_operation", source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "trace"
    release = threading.Event()
    with python_line_samples(
        fedot_root,
        output,
        interval_seconds=0.02,
        context={
            "candidate_id": "candidate-1",
            "evaluation_id": "eval-1",
            "fold_id": 2,
        },
    ) as status:

        def blocked_with_identity():
            module.blocked(release)

        worker = threading.Thread(target=blocked_with_identity, name="fedot-blocked")
        worker.start()
        time.sleep(0.1)
        release.set()
        worker.join(timeout=1)

    assert status["enabled"] is True
    records = [
        json.loads(line)
        for line in (output / f"python-lines-{status['pid']}.jsonl")
        .read_text()
        .splitlines()
    ]
    blocked_lines = [
        record
        for record in records
        if any(
            frame["function"] == "blocked" and frame["filename"] == str(source_file)
            for frame in record["fedot_frames"]
        )
    ]
    assert blocked_lines, "the sampled trace did not retain the FEDOT call stack"
    assert any(
        frame["function"] == "blocked" and frame["lineno"] == 4
        for record in blocked_lines
        for frame in record["fedot_frames"]
    ), "the trace did not identify the blocked Python call line"
    assert any(
        record.get("candidate_id") == "candidate-1"
        and record.get("evaluation_id") == "eval-1"
        and record.get("fold_id") == 2
        for record in blocked_lines
    ), "the line sample lost candidate/fold correlation context"


def test_line_sampling_is_a_noop_without_explicit_source_and_output(tmp_path):
    with python_line_samples(None, None, interval_seconds=0.02) as status:
        assert status["enabled"] is False
    assert list(tmp_path.iterdir()) == []


def test_line_sampling_does_not_swallow_worker_exceptions(tmp_path):
    source = tmp_path / "FEDOT"
    source.mkdir()
    with pytest.raises(RuntimeError, match="worker failure"):
        with python_line_samples(source, tmp_path / "trace", interval_seconds=0.02):
            raise RuntimeError("worker failure")


def test_line_snapshot_never_imports_fedot_context(monkeypatch):
    lock = threading.RLock()
    states = {
        17: {
            "thread_name": "fedot-import",
            "frames": [
                {
                    "frame_id": 1,
                    "filename": "/fedot/api/main.py",
                    "function": "<module>",
                    "lineno": 10,
                }
            ],
        }
    }

    original_import = builtins.__import__

    def reject_fedot_context_import(name, *args, **kwargs):
        if name == "fedot.utilities.evaluation_trace":
            raise AssertionError(
                "line sampler must not import FEDOT from a sampling thread"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_fedot_context_import)
    snapshots = python_line_trace._sample_line_snapshots(
        states,
        lock,
        timestamp_utc="now",
        monotonic_ns=1,
        pid=2,
        context={"proposal_id": "candidate-1", "task_id": "beer", "fold": 2},
    )
    lock_acquired = lock.acquire(timeout=0.2)
    if lock_acquired:
        lock.release()

    assert lock_acquired, "line sampling left its state lock held"
    assert snapshots[0]["proposal_id"] == "candidate-1"
    assert snapshots[0]["task_id"] == "beer"
    assert snapshots[0]["fold"] == 2

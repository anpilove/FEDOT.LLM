"""Opt-in controller stacks locate drain waits without inspecting processes."""

import inspect
import json
from pathlib import Path
import re
import threading
import time

import pytest

from fedotllm.agents.evolve.controller import quality_executor as executor


def test_controller_trace_captures_controlled_stall_in_real_drain_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_PROCESSES", "1")
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", "0.02")
    result = {"jobs": [], "fixture_completed": True}

    def controlled_result_persistence_stall(_source, _workspace, **_kwargs):
        time.sleep(0.09)
        return result

    monkeypatch.setattr(executor, "_drain_quality_queue_locked", controlled_result_persistence_stall)
    workspace = tmp_path / "run"
    assert executor.drain_quality_queue(tmp_path / "FEDOT", workspace) is result
    stack_path = workspace / "logs/controller.stacks.txt"
    assert stack_path.is_file(), "opted-in drain did not save its own thread stacks"
    text = stack_path.read_text()
    assert "--- stack dump at " in text
    expected_file = str(Path(__file__).resolve())
    expected_line = inspect.getsourcelines(controlled_result_persistence_stall)[1] + 1
    assert re.search(
        rf'File "{re.escape(expected_file)}", line {expected_line} in controlled_result_persistence_stall',
        text,
    )
    metadata = json.loads((workspace / "logs/controller.trace.json").read_text())
    assert metadata["scope"] == "controller_process_threads"
    assert metadata["stack_trace_path"] == str(stack_path)
    assert metadata["state"] == "stopped"
    assert metadata["enabled"] is True
    before = stack_path.stat().st_size
    time.sleep(0.05)
    assert stack_path.stat().st_size == before, "controller sampler remained active after drain"


def test_controller_trace_disabled_by_default_creates_no_trace_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLVE_QUALITY_TRACE_PROCESSES", raising=False)
    result = {"jobs": []}
    monkeypatch.setattr(executor, "_drain_quality_queue_locked", lambda *_args, **_kwargs: result)
    assert executor.drain_quality_queue(tmp_path / "FEDOT", tmp_path / "run") is result
    assert not (tmp_path / "run/logs/controller.stacks.txt").exists()
    assert not (tmp_path / "run/logs/controller.trace.json").exists()


def test_concurrent_drains_skip_competing_sampler_without_serializing_work(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_PROCESSES", "1")
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", "0.02")
    entered, release = threading.Event(), threading.Event()
    first_workspace = tmp_path / "first"
    results, errors = [], []

    def controlled_drain(_source, workspace, **_kwargs):
        if workspace == first_workspace:
            entered.set()
            assert release.wait(3), "test did not release the first drain"
        return {"workspace": str(workspace)}

    monkeypatch.setattr(executor, "_drain_quality_queue_locked", controlled_drain)

    def first_drain():
        try:
            results.append(executor.drain_quality_queue(tmp_path / "FEDOT", first_workspace))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=first_drain, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        second_workspace = tmp_path / "second"
        second_result = executor.drain_quality_queue(tmp_path / "FEDOT", second_workspace)
        assert second_result == {"workspace": str(second_workspace)}
        assert thread.is_alive(), "the second drain waited for the first to finish"
        second_metadata = json.loads((second_workspace / "logs/controller.trace.json").read_text())
        assert second_metadata["state"] == "skipped_process_sampler_busy"
        assert second_metadata["enabled"] is False
        assert not (second_workspace / "logs/controller.stacks.txt").exists()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not errors
    assert results == [{"workspace": str(first_workspace)}]


@pytest.mark.parametrize("interval", ["not-a-number", "0", "nan", "inf"])
def test_invalid_trace_interval_does_not_change_drain_result(tmp_path, monkeypatch, interval):
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_PROCESSES", "1")
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", interval)
    result = {"jobs": []}
    monkeypatch.setattr(executor, "_drain_quality_queue_locked", lambda *_args, **_kwargs: result)
    assert executor.drain_quality_queue(tmp_path / "FEDOT", tmp_path / "run") is result
    metadata = json.loads((tmp_path / "run/logs/controller.trace.json").read_text())
    assert metadata["enabled"] is False
    assert metadata["error"]


def test_trace_storage_failure_does_not_change_drain_result(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_PROCESSES", "1")
    workspace = tmp_path / "run"
    workspace.mkdir()
    (workspace / "logs").write_text("existing file")
    result = {"jobs": []}
    monkeypatch.setattr(executor, "_drain_quality_queue_locked", lambda *_args, **_kwargs: result)
    assert executor.drain_quality_queue(tmp_path / "FEDOT", workspace) is result


def test_controller_trace_preserves_drain_failure_and_releases_sampler(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_PROCESSES", "1")
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", "0.02")

    def failing_drain(*_args, **_kwargs):
        raise ValueError("controlled original failure")

    monkeypatch.setattr(executor, "_drain_quality_queue_locked", failing_drain)
    with pytest.raises(ValueError, match="controlled original failure"):
        executor.drain_quality_queue(tmp_path / "FEDOT", tmp_path / "failed")
    monkeypatch.setattr(executor, "_drain_quality_queue_locked", lambda *_args, **_kwargs: {"jobs": []})
    executor.drain_quality_queue(tmp_path / "FEDOT", tmp_path / "next")
    assert json.loads((tmp_path / "next/logs/controller.trace.json").read_text())["enabled"] is True

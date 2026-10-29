"""Storage faults must stop owned work while durable evidence still fits."""
import errno
import json
import sys
import time
from types import SimpleNamespace

import pytest

from fedotllm.agents.evolve.execution.process import run_worker


def test_storage_reserve_refuses_low_bytes_and_inodes_without_writing(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.execution import storage_guard
    for available, inodes in [(4095, 5000), (8192, 9)]:
        monkeypatch.setattr(storage_guard.os, "statvfs", lambda _: SimpleNamespace(
            f_bavail=available, f_frsize=1, f_favail=inodes))
        with pytest.raises(storage_guard.StorageCapacityError) as failure:
            storage_guard.require_storage(tmp_path, minimum_bytes=4096, minimum_inodes=10)
        assert failure.value.errno == errno.ENOSPC
        assert str(tmp_path) in str(failure.value)
        assert not list(tmp_path.iterdir())


def test_storage_guard_interrupts_owned_child_before_watchdog(tmp_path):
    from fedotllm.agents.evolve.execution.storage_guard import StorageCapacityError
    child = tmp_path / "storage_fixture.py"
    marker = tmp_path / "finished"
    started = tmp_path / "started"
    child.write_text("import pathlib, sys, time\npathlib.Path(sys.argv[2]).touch()\n"
                     "time.sleep(2)\npathlib.Path(sys.argv[1]).touch()\n")
    checks = []
    def guard():
        checks.append(True)
        if started.exists():
            raise StorageCapacityError(errno.ENOSPC, "reserve exhausted", str(tmp_path))
    before = time.monotonic()
    with pytest.raises(StorageCapacityError):
        run_worker([sys.executable, str(child), str(marker), str(started)], cwd=tmp_path,
                   env={}, timeout=30, resource_check=guard)
    assert time.monotonic() - before < 1.5
    assert started.exists()
    assert len(checks) >= 2
    assert not marker.exists()


def test_low_capacity_launcher_never_creates_workspace(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller import launch
    from fedotllm.agents.evolve.execution import storage_guard
    from fedotllm.agents.evolve.execution.guard import repo_root
    from fedotllm.agents.evolve.controller.release import controller_version
    source = tmp_path / "source"
    (source / "fedot").mkdir(parents=True)
    monkeypatch.setattr(storage_guard.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=40960, f_frsize=1, f_favail=50000))
    with pytest.raises(storage_guard.StorageCapacityError):
        launch.main(["--mode", "launch", "--controller-root", str(repo_root()),
                     "--fedot-source", str(source), "--python", sys.executable,
                     "--workspace", str(tmp_path / "new-run"),
                     "--expected-controller-sha256", controller_version()])
    assert not (tmp_path / "new-run").exists()


@pytest.mark.parametrize("message", ["OSError: [Errno 28] No space left on device",
                                     "OSError: [Errno 122] Disk quota exceeded"])
def test_structured_worker_storage_failure_stops_without_stdout_marker(tmp_path, monkeypatch, message):
    from fedotllm.agents.evolve.evaluation import fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry
    from fedotllm.agents.evolve.execution.storage_guard import StorageCapacityError
    source = tmp_path / "source"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/__init__.py").write_text("VERSION=1\n")
    fixture = tmp_path / "storage_payload_fixture.py"
    fixture.write_text("import json, pathlib, sys\n"
                       "spec=json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
                       "identity={k:spec.get(k) for k in ('side','attempt','attempt_id',"
                       "'registry_identity','evaluation_identity','worker_invocation_id')}\n"
                       "identity['task_id']=spec['source_dataset']\n"
                       "pathlib.Path(sys.argv[1]).write_text("
                       "json.dumps({**identity,'status':'crash', 'score':None, 'detail':sys.argv[2], "
                       "'traceback':sys.argv[2]}))\nraise SystemExit(2)\n")
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EVOLVE_QUALITY_STOCK_CACHE", str(tmp_path / "stock-cache"))
    def boundary(command, **kwargs):
        output = command[command.index("--out") + 1]
        return run_worker([sys.executable, str(fixture), output, message, command[command.index("--spec-json") + 1]], **kwargs)
    monkeypatch.setattr(quality, "run_worker", boundary)
    dataset = load_quality_registry().datasets[0].task_id
    with pytest.raises(StorageCapacityError):
        quality.run_quality_side(dataset, checkout=source, side="patch", python_executable=sys.executable)
    results = list((tmp_path / "artifacts").glob("*/quality-out-*.json"))
    assert len(results) == 1
    assert message in results[0].read_text()


def test_previous_run_with_stale_running_status_requires_terminal_log_and_free_locks(tmp_path):
    import fcntl
    from fedotllm.agents.evolve.controller.launch import previous_run_guard
    run = tmp_path / "old"
    (run / "logs").mkdir(parents=True)
    for name in (".quality_service.lock", ".quality_queue.lock"):
        (run / name).touch()
    (run / "logs/service_status.json").write_text(json.dumps({"state":"running", "child_running":True}))
    (run / "logs/service-launch.log").write_text('traceback\n' + json.dumps({
        "state":"artifact_error", "child_running":False, "active_child":None}, indent=2) + '\n')
    with previous_run_guard(run) as evidence:
        assert evidence["state"] == "artifact_error"
    with (run / ".quality_queue.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active"):
            with previous_run_guard(run):
                pass
    (run / "logs/service-launch.log").write_text('no terminal evidence\n')
    with pytest.raises(ValueError, match="terminal"):
        with previous_run_guard(run):
            pass

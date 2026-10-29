"""Safe observational stacks avoid native watchdog timer/import interference."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time


def test_sampler_repeats_exact_frames_without_source_or_locals(tmp_path):
    from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples

    path = tmp_path / "stacks.txt"
    secret_value = "never_capture_this_local_secret"

    def controlled_python_wait():
        threading.Event().wait(0.08)

    with periodic_stack_samples(path, interval_seconds=0.015) as status:
        controlled_python_wait()
    text = path.read_text()
    line = inspect.getsourcelines(controlled_python_wait)[1] + 1
    assert f'File "{__file__}", line {line} in controlled_python_wait' in text
    assert text.count("--- stack dump at ") >= 2
    assert text.count("--- stack dump complete ---") >= 2
    assert secret_value not in text
    assert "secret_value" not in text
    assert status["dump_count"] >= 2
    assert status["last_dump_at"]
    assert status["capture_method"] == "python_thread"
    assert "GIL" in status["limitations"]
    size = path.stat().st_size
    time.sleep(0.04)
    assert path.stat().st_size == size


def _run_fixture(tmp_path, name, body):
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(body))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    return subprocess.run([sys.executable, str(script), str(tmp_path / f"{name}.stacks.txt")],
                          env=env, capture_output=True, text=True, timeout=10)


def test_sampler_preserves_native_sleep_duration(tmp_path):
    result = _run_fixture(tmp_path, "native_sleep", """
        import ctypes, json, sys, time
        from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples
        sleep = ctypes.PyDLL(None).sleep
        sleep.argtypes = [ctypes.c_uint]
        sleep.restype = ctypes.c_uint
        started = time.monotonic()
        with periodic_stack_samples(sys.argv[1], interval_seconds=0.02) as status:
            sleep(1)
        print(json.dumps({"elapsed": time.monotonic() - started, "status": status}))
    """)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert 0.95 <= payload["elapsed"] < 4
    assert "GIL" in payload["status"]["limitations"]


def test_sampler_does_not_block_joblib_import_or_process_exit(tmp_path):
    result = _run_fixture(tmp_path, "joblib_import", """
        import sys, time
        from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples
        with periodic_stack_samples(sys.argv[1], interval_seconds=0.005):
            import joblib
            time.sleep(0.03)
        print("clean_exit")
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean_exit"


def test_sampler_storage_failure_is_observational(tmp_path):
    from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples

    blocker = tmp_path / "file"
    blocker.write_text("preserve")
    completed = False
    with periodic_stack_samples(blocker / "stacks.txt", interval_seconds=0.01) as status:
        completed = True
    assert completed
    assert status["enabled"] is False
    assert status["error"]
    assert blocker.read_text() == "preserve"


def test_slow_diagnostic_write_does_not_block_context_exit_or_close_writer_fd(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation import stack_sampler as sampler

    entered, release, wrote = threading.Event(), threading.Event(), threading.Event()
    original_write = sampler._write_snapshot

    def controlled_slow_write(fd, payload):
        entered.set()
        assert release.wait(3), "test did not release diagnostic write"
        original_write(fd, payload)
        wrote.set()

    monkeypatch.setattr(sampler, "_write_snapshot", controlled_slow_write)
    path = tmp_path / "slow.stacks.txt"
    started = time.monotonic()
    try:
        with sampler.periodic_stack_samples(path, interval_seconds=0.01) as status:
            assert entered.wait(2)
        assert time.monotonic() - started < 1
        assert status["stop_pending"] is True
        release.set()
        assert wrote.wait(2), "context exit closed the still-active writer's descriptor"
    finally:
        release.set()

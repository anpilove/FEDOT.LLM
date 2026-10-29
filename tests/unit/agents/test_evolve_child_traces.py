"""Real compute-child diagnostics without FEDOT training or process scans."""

import json
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import subprocess
import re
import sys

import pytest


_FIXTURE = """
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from fedotllm.agents.evolve.evaluation.child_traces import traced_children
from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples


def active_compute_line():
    import numpy
    if os.environ.get("EVOLVE_TRACE_PYTHON_LINES") == "1":
        from fedot.fixture import active_compute
        active_compute()
    time.sleep(0.4)
    return os.getpid()


def native_compute_line(directory):
    native_sleep = ctypes.PyDLL(None).sleep
    native_sleep.argtypes = [ctypes.c_uint]
    native_sleep.restype = ctypes.c_uint
    Path(directory, "native.started").touch()
    started_at = time.monotonic()
    native_sleep(1)
    Path(directory, "native.elapsed").write_text(str(time.monotonic() - started_at))
    Path(directory, "native.finished").touch()
    time.sleep(0.15)


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    mode = sys.argv[2]
    with traced_children(directory, attempt_id="proposal:traffic:patch:2", interval_seconds=0.03) as status:
        assert status["enabled"], status
        if mode == "manifest-error":
            (directory / "manifest.jsonl").mkdir()
        parent_sampler = periodic_stack_samples(directory / "parent.stacks.txt", interval_seconds=0.03)
        parent_sampler.__enter__()
        try:
            if mode == "loky":
                from joblib import Parallel, delayed
                pids = Parallel(n_jobs=2)(delayed(active_compute_line)() for _ in range(2))
            else:
                context = multiprocessing.get_context("spawn" if mode in {"native", "manifest-error"} else mode)
                target = native_compute_line if mode == "native" else active_compute_line
                args = (str(directory),) if mode == "native" else ()
                child = context.Process(target=target, args=args)
                child.start()
                pids = [child.pid]
                child.join(15)
                if child.is_alive():
                    child.terminate()
                    child.join(5)
                    raise AssertionError("fixture compute child did not exit")
                assert child.exitcode == 0, child.exitcode
        finally:
            parent_sampler.__exit__(None, None, None)
    print(json.dumps({"parent_pid": os.getpid(), "compute_pids": pids}))
"""


def _run_compute_fixture(tmp_path, mode, *, python_line_trace=False):
    script = tmp_path / "compute_fixture.py"
    script.write_text(_FIXTURE, encoding="utf-8")
    directory = tmp_path / "trace"
    if python_line_trace:
        fedot_root = tmp_path / "FEDOT"
        package = fedot_root / "fedot"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "fixture.py").write_text(
            "import time\n"
            "def active_compute():\n"
            "    deadline = time.monotonic() + 0.4\n"
            "    while time.monotonic() < deadline:\n"
            "        time.sleep(0.005)\n",
            encoding="utf-8",
        )
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = str(repo_root)
    env["OPENBLAS_NUM_THREADS"] = "1"
    if python_line_trace:
        env.update(
            {
                "EVOLVE_TRACE_PYTHON_LINES": "1",
                "EVOLVE_TRACE_LINE_INTERVAL_SECONDS": "0.03",
                "FEDOTLLM_REPO_PATH": str(fedot_root),
                "PYTHONPATH": os.pathsep.join((str(fedot_root), env["PYTHONPATH"])),
            }
        )
    completed = subprocess.run(
        [sys.executable, str(script), str(directory), mode],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    payload["stderr"] = completed.stderr
    manifest = (
        [
            json.loads(line)
            for line in (directory / "manifest.jsonl").read_text().splitlines()
        ]
        if (directory / "manifest.jsonl").is_file()
        else []
    )
    return directory, payload, manifest


def test_disabled_context_leaves_environment_untouched():
    from fedotllm.agents.evolve.evaluation.child_traces import traced_children

    before = dict(os.environ)
    with traced_children(None, attempt_id="disabled") as status:
        assert status["enabled"] is False
        assert dict(os.environ) == before
    assert dict(os.environ) == before


def test_context_restores_existing_environment_after_exception(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.evaluation.child_traces import traced_children

    monkeypatch.setenv("EVOLVE_TRACE_ATTEMPT_ID", "outer-attempt")
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="fixture"):
        with traced_children(
            tmp_path, attempt_id="inner-attempt", interval_seconds=0.03
        ) as status:
            assert status["enabled"] is True
            assert os.environ["EVOLVE_TRACE_ATTEMPT_ID"] == "inner-attempt"
            assert os.environ["PYTHONPATH"].split(os.pathsep)[-1] == "/existing/path"
            raise RuntimeError("fixture")
    assert dict(os.environ) == before
    assert not (tmp_path / "manifest.jsonl").exists(), (
        "root must use its existing sampler"
    )


@pytest.mark.parametrize("mode", ["loky", "spawn", "fork", "forkserver"])
def test_compute_children_have_own_active_line_while_parent_waits(tmp_path, mode):
    if mode != "loky" and mode not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{mode} unavailable")
    directory, payload, manifest = _run_compute_fixture(tmp_path, mode)
    if mode == "loky":
        assert len(set(payload["compute_pids"])) == 2

    compute_records = [
        record for record in manifest if record["pid"] in payload["compute_pids"]
    ]
    assert {record["pid"] for record in compute_records} == set(payload["compute_pids"])
    for record in compute_records:
        assert record["pid"] != payload["parent_pid"]
        assert record["attempt_id"] == "proposal:traffic:patch:2"
        assert record["ppid"] > 0
        assert record["started_at"]
        assert json.loads(Path(record["metadata_path"]).read_text()) == record
        assert record["coverage"] == "armed"
        stack = Path(record["stack_trace_path"]).read_text()
        assert "--- stack dump at " in stack
        assert "active_compute_line" in stack
        sleep_line = _FIXTURE.splitlines().index("    time.sleep(0.4)") + 1
        assert f"line {sleep_line} in active_compute_line" in stack
        assert str(tmp_path / "compute_fixture.py") in stack
    available = [
        record["stack_trace_path"] for record in manifest if record["stack_trace_path"]
    ]
    assert len(set(available)) == len(available)
    parent_stack = (directory / "parent.stacks.txt").read_text()
    assert "--- stack dump at " in parent_stack
    assert "active_compute_line" not in parent_stack
    if mode == "loky":
        assert "_retrieve" in parent_stack
        assert "parallel.py" in parent_stack
    else:
        assert "join" in parent_stack


def test_native_call_keeps_duration_and_documents_gil_sampling_limit(tmp_path):
    directory, payload, manifest = _run_compute_fixture(tmp_path, "native")
    assert float((directory / "native.elapsed").read_text()) >= 0.85, (
        "diagnostic timer interrupted native task"
    )
    record = next(
        record for record in manifest if record["pid"] in payload["compute_pids"]
    )
    assert "native_call_holding_gil" in record["limitations"]
    assert record["capture_method"] == "python_thread"
    stack = Path(record["stack_trace_path"]).read_text()
    snapshot_times = [
        datetime.fromisoformat(value).timestamp()
        for value in re.findall(r"--- stack dump at ([^ ]+) ---", stack)
    ]
    assert snapshot_times, (
        "sampler did not resume after the native call released the GIL"
    )
    assert not any(
        (directory / "native.started").stat().st_mtime + 0.15
        < snapshot
        < (directory / "native.finished").stat().st_mtime - 0.15
        for snapshot in snapshot_times
    ), "Python sampler claimed to run with the GIL held"


def test_loky_child_records_exact_fedot_python_line(tmp_path):
    directory, payload, manifest = _run_compute_fixture(
        tmp_path,
        "loky",
        python_line_trace=True,
    )
    record = next(
        record for record in manifest if record["pid"] in payload["compute_pids"]
    )
    line_trace = record.get("python_line_trace_path")
    assert line_trace, "child manifest omitted the Python line trace path"
    samples = [json.loads(line) for line in Path(line_trace).read_text().splitlines()]
    blocked = [
        sample
        for sample in samples
        if any(
            frame["function"] == "active_compute"
            and frame["filename"].endswith("/fedot/fixture.py")
            for frame in sample["fedot_frames"]
        )
    ]
    assert blocked, "compute child did not record frames from the FEDOT source tree"
    assert any(
        frame["lineno"] == 5
        for sample in blocked
        for frame in sample["fedot_frames"]
        if frame["function"] == "active_compute"
    ), "trace did not identify the active source line"


def test_child_manifest_failure_does_not_interrupt_compute(tmp_path):
    directory, payload, manifest = _run_compute_fixture(tmp_path, "manifest-error")
    assert payload["compute_pids"]
    assert "diagnostic warning" in payload["stderr"]
    assert manifest == []
    assert (directory / "manifest.jsonl").is_dir()


def test_unwritable_diagnostic_directory_does_not_block_task(tmp_path, capsys):
    from fedotllm.agents.evolve.evaluation.child_traces import traced_children

    blocker = tmp_path / "file"
    blocker.write_text("preserve")
    before = dict(os.environ)
    with traced_children(blocker / "trace", attempt_id="fixture") as status:
        assert status["enabled"] is False
        assert "error" in status
        assert sum([1, 2, 3]) == 6
    assert blocker.read_text() == "preserve"
    assert dict(os.environ) == before
    assert "diagnostic" in capsys.readouterr().err


def _run_startup_fixture(tmp_path, hook_source, *, bootstrap, tracing):
    repo = Path(__file__).resolve().parents[3]
    hook_directory = tmp_path / "original-hook"
    hook_directory.mkdir(exist_ok=True)
    (hook_directory / "sitecustomize.py").write_text(hook_source, encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EVOLVE_TRACE_")
    }
    paths = [str(hook_directory), str(repo)]
    if bootstrap:
        paths.insert(
            0, str(repo / "fedotllm/agents/evolve/evaluation/_trace_bootstrap")
        )
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if tracing:
        env.update(
            {
                "EVOLVE_TRACE_DIRECTORY": str(tmp_path / "trace"),
                "EVOLVE_TRACE_ATTEMPT_ID": "startup-fixture",
                "EVOLVE_TRACE_OWNER_PID": "999999999",
                "EVOLVE_TRACE_INTERVAL_SECONDS": "0.03",
            }
        )
    return subprocess.run(
        [sys.executable, "-c", "import sys; print('startup-complete')"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("tracing", [False, True])
def test_bootstrap_preserves_existing_sitecustomize_and_self_import(tmp_path, tracing):
    marker = tmp_path / "hook-marker.txt"
    hook_source = (
        "import sitecustomize\n"
        "assert sitecustomize.__file__ == __file__\n"
        "from pathlib import Path\n"
        f"with Path({str(marker)!r}).open('a') as stream:\n"
        "    stream.write('original-hook-ran\\n')\n"
    )
    completed = _run_startup_fixture(
        tmp_path, hook_source, bootstrap=True, tracing=tracing
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert marker.read_text() == "original-hook-ran\n"


@pytest.mark.parametrize(
    "hook_source",
    [
        "raise RuntimeError('original-hook-fixture-error')\n",
        "raise ImportError('original-hook-fixture-error', name='missing_original_dependency')\n",
    ],
)
@pytest.mark.parametrize("tracing", [False, True])
def test_bootstrap_preserves_standard_sitecustomize_exception_reporting(
    tmp_path, hook_source, tracing
):
    baseline = _run_startup_fixture(
        tmp_path, hook_source, bootstrap=False, tracing=tracing
    )
    wrapped = _run_startup_fixture(
        tmp_path, hook_source, bootstrap=True, tracing=tracing
    )
    assert wrapped.returncode == baseline.returncode
    assert wrapped.stdout == baseline.stdout
    assert wrapped.stderr == baseline.stderr
    assert "original-hook-fixture-error" in wrapped.stderr

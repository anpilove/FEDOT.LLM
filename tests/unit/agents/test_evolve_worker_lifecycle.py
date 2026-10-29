"""Real local process fixtures; observe only files belonging to these fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from fedotllm.agents.evolve.execution import process as worker_process
from fedotllm.agents.evolve.execution import _worker_lifetime_guard as worker_guard
from fedotllm.agents.evolve.execution.process import run_worker


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX worker groups")
REPO_ROOT = Path(__file__).resolve().parents[3]


WORKER_FIXTURE = '''\
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1])
mode = sys.argv[2]
deadline = time.monotonic() + min(30.0, float(os.environ.get("EVOLVE_FIXTURE_LIFETIME_SECONDS", "30")))
if mode == "child":
    (root / "child.pid").write_text(str(os.getpid()))
    while time.monotonic() < deadline:
        with (root / "heartbeat").open("a") as handle:
            handle.write(".")
        time.sleep(0.02)
else:
    (root / "worker.pid").write_text(str(os.getpid()))
    subprocess.Popen(
        [sys.executable, __file__, str(root), "child"],
        start_new_session=mode == "escaped",
    )
    while not (root / "heartbeat").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if mode == "invalid":
        os.write(1, bytes([255]))
    (root / "ready").write_text("ready")
    print("worker started", flush=True)
    if mode in ("finish_ok", "finish_crash"):
        raise SystemExit(0 if mode == "finish_ok" else 7)
    while time.monotonic() < deadline:
        time.sleep(0.02)
'''


def test_guard_treats_darwin_permission_race_as_clean_only_if_group_is_gone(monkeypatch):
    calls = []

    def killpg(pgid, sig):
        calls.append((pgid, sig))
        if len(calls) == 1:
            raise PermissionError("fixture race after worker exit")
        raise ProcessLookupError("fixture group already gone")

    monkeypatch.setattr(worker_guard.os, "killpg", killpg)
    assert worker_guard.terminate_owned_worker_group(43210) == 0
    assert calls == [(43210, signal.SIGKILL), (43210, 0)]


def test_guard_reports_permission_denial_when_owned_group_still_exists(monkeypatch):
    calls = []

    def killpg(pgid, sig):
        calls.append((pgid, sig))
        if sig == signal.SIGKILL:
            raise PermissionError("fixture cannot kill group")

    monkeypatch.setattr(worker_guard.os, "killpg", killpg)
    assert worker_guard.terminate_owned_worker_group(43211) == 1
    assert calls == [(43211, signal.SIGKILL), (43211, 0)]


CONTROLLER_FIXTURE = '''\
import json
import os
from pathlib import Path
import sys
import threading
from fedotllm.agents.evolve.execution.process import run_worker

root = Path(sys.argv[1])
try:
    outcome = run_worker(
        [sys.executable, str(root / "worker_fixture.py"), str(root), sys.argv[2]],
        cwd=root, env=dict(os.environ), timeout=float(sys.argv[3]),
        watchdog_timeout=float(sys.argv[6]) if len(sys.argv) > 6 else None,
        cancellation_event=threading.Event() if sys.argv[4] == "poll" else None,
        log_path=root / "worker.log" if sys.argv[5] == "log" else None,
    )
except KeyboardInterrupt:
    (root / "interrupted").write_text("interrupted")
else:
    (root / "outcome.json").write_text(json.dumps(outcome.__dict__))
'''


def _start_controller(root, *, mode="same", timeout=30, cancellable=False, use_log=False,
                      watchdog_timeout=None):
    (root / "worker_fixture.py").write_text(WORKER_FIXTURE)
    controller_script = root / "controller_fixture.py"
    controller_script.write_text(CONTROLLER_FIXTURE)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.Popen(
        [sys.executable, str(controller_script), str(root), mode, str(timeout),
         "poll" if cancellable else "direct", "log" if use_log else "capture",
         *([str(watchdog_timeout)] if watchdog_timeout is not None else [])],
        # Only fixture-owned file logs are used by log-mode lifecycle checks.
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )


def _wait_file(path, controller):
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        if controller.poll() is not None:
            pytest.fail(f"fixture controller exited early: {controller.communicate()}")
        time.sleep(0.02)
    assert path.exists(), f"fixture never became ready: {path}"


def _kill_known_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin may reject a group whose leader has already disappeared.
        # Signal only this fixture's known leader; never discover other PIDs.
        try:
            os.kill(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _cleanup_fixture(root, controller, *, escaped=False):
    # Every identifier comes from a process launched by this test itself.
    _kill_known_group(controller.pid)
    # Clean the escaped child first: its independent group may retain output
    # pipes even when Darwin rejects signaling an already-gone worker group.
    for name in ("child.pid", "worker.pid") if escaped else ("worker.pid",):
        path = root / name
        if path.exists():
            _kill_known_group(int(path.read_text()))
    controller.communicate(timeout=3)


def _assert_heartbeat_stopped(root):
    # A live fixture changes this file every 20ms. A killed zombie cannot do so.
    time.sleep(0.8)
    size = (root / "heartbeat").stat().st_size
    time.sleep(0.2)
    assert (root / "heartbeat").stat().st_size == size


def test_timeout_does_not_wait_forever_for_escaped_descendants_with_open_pipes(tmp_path):
    # Removing the bounded post-kill communicate would hang on the escaped child.
    controller = _start_controller(tmp_path, mode="escaped", timeout=0.3)
    try:
        _wait_file(tmp_path / "ready", controller)
        controller.communicate(timeout=5)
        outcome = json.loads((tmp_path / "outcome.json").read_text())
        assert outcome["timed_out"] is True
        assert outcome["returncode"] is None
        assert "worker started" in outcome["stdout"]
    finally:
        _cleanup_fixture(tmp_path, controller, escaped=True)


SNIPPET_CONTROLLER_FIXTURE = '''\
import json
from pathlib import Path
import sys
from fedotllm.agents.evolve.execution import run_code

root = Path(sys.argv[1])
run_code.WORKER_MODULE = "snippet_fixture"
run_code.fedot_python = lambda checkout: sys.executable
try:
    result = run_code.run_fedot_snippet(root, "print(1)", timeout_s=float(sys.argv[2]))
except KeyboardInterrupt:
    (root / "interrupted").write_text("interrupted")
else:
    (root / "outcome.json").write_text(json.dumps(result.__dict__))
'''


def _start_snippet_controller(root, *, escaped=False, timeout=30):
    (root / "worker_fixture.py").write_text(WORKER_FIXTURE)
    (root / "mode").write_text("escaped" if escaped else "same")
    (root / "snippet_fixture.py").write_text(
        "import runpy, sys\nfrom pathlib import Path\n"
        "root = Path(sys.argv[sys.argv.index('--checkout') + 1])\n"
        "sys.argv = [str(root / 'worker_fixture.py'), str(root), (root / 'mode').read_text()]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    script = root / "snippet_controller_fixture.py"
    script.write_text(SNIPPET_CONTROLLER_FIXTURE)
    return subprocess.Popen(
        [sys.executable, str(script), str(root), str(timeout)],
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )


def test_snippet_timeout_returns_even_when_escaped_child_holds_output_pipe(tmp_path):
    controller = _start_snippet_controller(tmp_path, escaped=True, timeout=0.3)
    try:
        _wait_file(tmp_path / "ready", controller)
        controller.communicate(timeout=5)
        result = json.loads((tmp_path / "outcome.json").read_text())
        assert result["status"] == "timeout"
        assert result["exit_code"] is None
        assert "worker started" in result["stdout"]
    finally:
        _cleanup_fixture(tmp_path, controller, escaped=True)


@pytest.mark.parametrize("interrupt", [signal.SIGINT, signal.SIGKILL])
def test_snippet_controller_interruption_stops_owned_worker_descendants(tmp_path, interrupt):
    controller = _start_snippet_controller(tmp_path)
    try:
        _wait_file(tmp_path / "ready", controller)
        time.sleep(0.1)
        os.kill(controller.pid, interrupt)
        controller.wait(timeout=5)
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller)


def test_interpreter_probe_timeout_returns_when_escaped_child_holds_output_pipe(tmp_path):
    # Runtime identity runs before ML's watchdog; it needs bounded cleanup too.
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    interpreter = tmp_path / "interpreter_fixture"
    interpreter.write_text(
        f"#!{sys.executable}\nimport runpy, sys\nfrom pathlib import Path\n"
        "root = Path(__file__).parent\n"
        "sys.argv = [str(root / 'worker_fixture.py'), str(root), 'escaped']\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    interpreter.chmod(0o700)
    driver = tmp_path / "identity_controller_fixture.py"
    driver.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "from fedotllm.agents.evolve.execution.process import _interpreter_identity\n"
        "root = Path(sys.argv[1])\n"
        "identity = _interpreter_identity(str(root / 'interpreter_fixture'))\n"
        "(root / 'outcome.json').write_text(json.dumps({'identity': identity}))\n"
    )
    controller = subprocess.Popen(
        [sys.executable, str(driver), str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        _wait_file(tmp_path / "ready", controller)
        controller.communicate(timeout=15)
        result = json.loads((tmp_path / "outcome.json").read_text())
        assert result["identity"].startswith(str(interpreter))
    finally:
        _cleanup_fixture(tmp_path, controller, escaped=True)


def test_controller_sigkill_terminates_owned_worker_descendants(tmp_path):
    # Without a separate pipe guardian the child's heartbeat continues forever.
    controller = _start_controller(tmp_path)
    try:
        _wait_file(tmp_path / "ready", controller)
        time.sleep(0.1)
        os.kill(controller.pid, signal.SIGKILL)
        controller.wait(timeout=2)
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller)


def test_keyboard_interrupt_cleans_owned_worker_descendants(tmp_path):
    # A BaseException handler must kill children before propagating interruption.
    controller = _start_controller(tmp_path)
    try:
        _wait_file(tmp_path / "ready", controller)
        os.kill(controller.pid, signal.SIGINT)
        controller.wait(timeout=5)
        assert (tmp_path / "interrupted").exists()
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller)


def test_invalid_worker_output_does_not_replace_keyboard_interrupt(tmp_path):
    # Decoding captured bytes during cleanup must not mask the user's interrupt.
    controller = _start_controller(tmp_path, mode="invalid")
    try:
        _wait_file(tmp_path / "ready", controller)
        os.kill(controller.pid, signal.SIGINT)
        controller.wait(timeout=5)
        assert (tmp_path / "interrupted").exists()
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller)


@pytest.mark.parametrize("cancellable", [False, True])
def test_timeout_kills_children_in_the_owned_group(tmp_path, cancellable):
    controller = _start_controller(tmp_path, timeout=0.3, cancellable=cancellable)
    try:
        _wait_file(tmp_path / "ready", controller)
        controller.communicate(timeout=5)
        assert json.loads((tmp_path / "outcome.json").read_text())["timed_out"]
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller)


def test_timeout_kills_loky_workers_from_owned_quality_side(tmp_path):
    fixture = tmp_path / "loky_fixture.py"
    fixture.write_text(
        "from joblib import Parallel, delayed\n"
        "from pathlib import Path\n"
        "import sys, time\n"
        "root = Path(sys.argv[1])\n"
        "def worker(slot):\n"
        "    heartbeat = root / f'heartbeat-{slot}'\n"
        "    while True:\n"
        "        with heartbeat.open('a') as stream: stream.write('.')\n"
        "        time.sleep(0.02)\n"
        "Parallel(n_jobs=2)(delayed(worker)(slot) for slot in range(2))\n",
        encoding="utf-8",
    )

    outcome = run_worker(
        [sys.executable, str(fixture), str(tmp_path)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        timeout=3,
        watchdog_timeout=10,
        log_path=tmp_path / "loky-worker.log",
    )

    assert outcome.timed_out is True
    heartbeat_paths = [tmp_path / "heartbeat-0", tmp_path / "heartbeat-1"]
    assert all(path.exists() and path.stat().st_size > 0 for path in heartbeat_paths)
    sizes = [path.stat().st_size for path in heartbeat_paths]
    time.sleep(0.2)
    assert [path.stat().st_size for path in heartbeat_paths] == sizes


@pytest.mark.parametrize("mode,returncode", [("finish_ok", 0), ("finish_crash", 7)])
def test_completed_log_worker_cleans_remaining_owned_group(tmp_path, mode, returncode):
    # File redirects let communicate reap just the direct parent. Disarming
    # the guardian without clearing the group leaves that parent's child alive.
    controller = _start_controller(tmp_path, mode=mode, use_log=True)
    try:
        controller.communicate(timeout=5)
        outcome = json.loads((tmp_path / "outcome.json").read_text())
        assert outcome["returncode"] == returncode
        assert outcome["timed_out"] is False
        assert "worker started" in outcome["stdout"]
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller)


def test_permission_denied_direct_cleanup_falls_back_to_guardian(tmp_path, monkeypatch):
    """Darwin may deny signaling a group after its direct leader exits.

    The independent guardian still owns the cleanup responsibility; a denied
    controller signal must not turn a completed FEDOT result into a queue error.
    """
    script = tmp_path / "worker_fixture.py"
    script.write_text(WORKER_FIXTURE)
    real_killpg = os.killpg

    def deny_controller_signal(pgid, signum):
        raise PermissionError("fixture: controller cannot signal exited worker group")

    monkeypatch.setattr(worker_process.os, "killpg", deny_controller_signal)
    try:
        outcome = run_worker(
            [sys.executable, str(script), str(tmp_path), "finish_ok"],
            cwd=tmp_path, env=dict(os.environ), timeout=5,
            log_path=tmp_path / "worker.log",
        )
        assert outcome.returncode == 0
        assert outcome.timed_out is False
        assert "worker started" in outcome.stdout
        _assert_heartbeat_stopped(tmp_path)
    finally:
        monkeypatch.setattr(worker_process.os, "killpg", real_killpg)
        worker_path = tmp_path / "child.pid"
        if worker_path.exists():
            _kill_known_group(int(worker_path.read_text()))


@pytest.mark.parametrize("cancellable", [False, True])
def test_normal_completion_preserves_stdout_stderr_and_returncode(tmp_path, cancellable):
    script = tmp_path / "output_fixture.py"
    script.write_text('import sys\nprint("output")\nprint("error", file=sys.stderr)\nsys.exit(7)\n')
    outcome = run_worker([sys.executable, str(script)], cwd=tmp_path,
                         env=dict(os.environ), timeout=5,
                         cancellation_event=threading.Event() if cancellable else None)
    assert outcome.returncode == 7
    assert outcome.stdout == "output\n"
    assert outcome.stderr == "error\n"
    assert outcome.timed_out is False
    assert outcome.cancelled is False


def test_parent_monitored_task_timeout_is_distinct_and_kills_same_session_child(tmp_path):
    """The task deadline must fire before, and independently of, its guardian cap."""
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    started = time.monotonic()
    outcome = run_worker(
        [sys.executable, str(tmp_path / "worker_fixture.py"), str(tmp_path), "same"],
        cwd=tmp_path, env=dict(os.environ), timeout=0.2, watchdog_timeout=2.0,
        log_path=tmp_path / "worker.log",
    )
    assert time.monotonic() - started < 1.5
    assert outcome.timed_out is True
    assert outcome.timeout_kind == "task_timeout"
    assert outcome.returncode is None
    assert "worker started" in outcome.stdout
    _assert_heartbeat_stopped(tmp_path)


def test_blocked_resource_check_cannot_extend_task_deadline(tmp_path):
    """A stuck storage probe must not defer the worker's wall-clock deadline."""
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    release_check = threading.Event()
    unblock_later = threading.Timer(1.0, release_check.set)
    unblock_later.start()
    started = time.monotonic()
    try:
        outcome = run_worker(
            [sys.executable, str(tmp_path / "worker_fixture.py"), str(tmp_path), "same"],
            cwd=tmp_path, env=dict(os.environ), timeout=0.2, watchdog_timeout=2.0,
            resource_check=lambda: release_check.wait(timeout=2.0),
            log_path=tmp_path / "worker.log",
        )
        elapsed = time.monotonic() - started
        assert elapsed < 0.9
        assert outcome.timed_out is True
        assert outcome.timeout_kind == "task_timeout"
        assert not (tmp_path / "ready").exists()
    finally:
        release_check.set()
        unblock_later.cancel()


def test_blocked_periodic_resource_check_cannot_extend_task_deadline(tmp_path, monkeypatch):
    """Periodic storage checks must stay off the deadline-monitoring thread."""
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    release_check = threading.Event()
    calls = 0

    def check_resource():
        nonlocal calls
        calls += 1
        if calls > 1:
            release_check.wait(timeout=2.0)

    monkeypatch.setattr(worker_process, "_RESOURCE_CHECK_INTERVAL", 0.05)
    started = time.monotonic()
    try:
        outcome = run_worker(
            [sys.executable, str(tmp_path / "worker_fixture.py"), str(tmp_path), "same"],
            cwd=tmp_path, env=dict(os.environ), timeout=0.35, watchdog_timeout=2.0,
            resource_check=check_resource,
            log_path=tmp_path / "worker.log",
        )
        elapsed = time.monotonic() - started
        assert calls >= 2
        assert elapsed < 1.2
        assert outcome.timed_out is True
        assert outcome.timeout_kind == "task_timeout"
        assert "worker started" in outcome.stdout
        _assert_heartbeat_stopped(tmp_path)
    finally:
        release_check.set()


def test_repeated_blocked_resource_checks_do_not_create_unbounded_threads(tmp_path):
    """A stuck storage probe must not add one permanent daemon thread per retry."""
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    release_check = threading.Event()
    callback_started = threading.Event()
    callback_calls = 0

    def blocked_check():
        nonlocal callback_calls
        callback_calls += 1
        callback_started.set()
        release_check.wait()

    before = sum(thread.name == "evolve-resource-check" for thread in threading.enumerate())
    try:
        first = run_worker(
            [sys.executable, str(tmp_path / "worker_fixture.py"), str(tmp_path), "same"],
            cwd=tmp_path, env=dict(os.environ), timeout=0.15, watchdog_timeout=2.0,
            resource_check=blocked_check,
        )
        assert first.timed_out is True
        assert callback_started.wait(timeout=1)

        for _ in range(3):
            outcome = run_worker(
                [sys.executable, str(tmp_path / "worker_fixture.py"), str(tmp_path), "same"],
                cwd=tmp_path, env=dict(os.environ), timeout=0.08, watchdog_timeout=2.0,
                resource_check=blocked_check,
            )
            assert outcome.timed_out is True

        after = sum(thread.name == "evolve-resource-check" for thread in threading.enumerate())
        assert after - before <= 1
        assert callback_calls == 1
    finally:
        release_check.set()


def test_earlier_guardian_watchdog_is_reported_when_escaped_child_holds_pipe(tmp_path):
    """Report the deadline that fired first even if task cleanup finishes later."""
    controller = _start_controller(
        tmp_path,
        mode="escaped",
        timeout=0.8,
        watchdog_timeout=0.2,
    )
    try:
        _wait_file(tmp_path / "ready", controller)
        controller.communicate(timeout=8)
        outcome = json.loads((tmp_path / "outcome.json").read_text())
        assert outcome["timed_out"] is True
        assert outcome["timeout_kind"] == "watchdog_timeout"
    finally:
        _cleanup_fixture(tmp_path, controller, escaped=True)


def test_abrupt_controller_exit_lets_guardian_kill_owned_worker_group(tmp_path):
    """The fallback guardian must clean its worker when the controller is killed."""
    controller = _start_controller(
        tmp_path, mode="same", timeout=30, use_log=True, watchdog_timeout=3.0
    )
    try:
        _wait_file(tmp_path / "ready", controller)
        os.kill(controller.pid, signal.SIGKILL)
        controller.wait(timeout=2)
        assert controller.returncode == -signal.SIGKILL
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller)


def test_guardian_watchdog_fires_if_parent_stops_monitoring_worker(tmp_path):
    """Exercise the independent cap while the controller deliberately does not poll."""
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    worker = subprocess.Popen(
        [sys.executable, str(tmp_path / "worker_fixture.py"), str(tmp_path), "same"],
        cwd=tmp_path, env=dict(os.environ), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    guardian = None
    parent_write = None
    try:
        _wait_file(tmp_path / "ready", worker)
        guardian, parent_write = worker_process._start_lifetime_guard(
            worker, watchdog_timeout=0.6
        )
        # Intentionally leave the parent pipe open and do not call communicate:
        # only the fixed guardian's own deadline can stop this worker group.
        _assert_heartbeat_stopped(tmp_path)
        assert guardian.wait(timeout=2) == worker_process._GUARDIAN_WATCHDOG_EXIT
        assert worker.poll() is not None
    finally:
        if parent_write is not None:
            os.close(parent_write)
        if guardian is not None and guardian.poll() is None:
            guardian.kill()
            guardian.wait(timeout=2)
        if worker.poll() is None:
            _kill_known_group(worker.pid)
            worker.wait(timeout=2)


def test_log_outcome_reads_only_a_bounded_tail(tmp_path):
    # Reading the entire file would return the 2MiB prefix as well as the tail.
    script = tmp_path / "log_fixture.py"
    script.write_text('import sys\nsys.stdout.write("x" * (2 * 1024 * 1024) + "FINAL\\n")\n')
    log = tmp_path / "worker.log"
    outcome = run_worker([sys.executable, str(script)], cwd=tmp_path,
                         env=dict(os.environ), timeout=5, log_path=log)
    assert log.stat().st_size == 2 * 1024 * 1024 + 6
    assert len(outcome.stdout.encode("utf-8")) <= 1024 * 1024
    assert outcome.stdout.endswith("FINAL\n")
    assert outcome.stderr == ""


def test_guardian_spawn_failure_cleans_already_launched_worker(tmp_path, monkeypatch):
    # A launch failure must not leave the real worker/child group running.
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    real_popen = subprocess.Popen
    worker = None

    def fail_guardian_spawn(cmd, **kwargs):
        nonlocal worker
        if "--worker-pgid" in cmd:
            _wait_file(tmp_path / "ready", worker)
            raise OSError("guardian launch denied")
        worker = real_popen(cmd, **kwargs)
        return worker

    monkeypatch.setattr(worker_process.subprocess, "Popen", fail_guardian_spawn)
    try:
        with pytest.raises(OSError, match="guardian launch denied"):
            run_worker([sys.executable, str(tmp_path / "worker_fixture.py"),
                        str(tmp_path), "same"], cwd=tmp_path,
                       env=dict(os.environ), timeout=5)
        _assert_heartbeat_stopped(tmp_path)
    finally:
        if worker is not None:
            if worker.poll() is None:
                _kill_known_group(worker.pid)
            worker.communicate(timeout=3)


def test_cleanup_error_does_not_replace_original_interrupt(tmp_path, monkeypatch):
    # PermissionError from direct group cleanup must not mask KeyboardInterrupt;
    # the real independent guardian still receives EOF and kills the group.
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    real_start_guard = worker_process._start_lifetime_guard
    worker = None

    def interrupt_after_guard_ready(proc, *, watchdog_timeout=None):
        nonlocal worker
        worker = proc
        guardian, parent_write = real_start_guard(proc, watchdog_timeout=watchdog_timeout)
        communicate = proc.communicate

        def first_call_interrupt(*args, **kwargs):
            proc.communicate = communicate
            _wait_file(tmp_path / "ready", proc)
            raise KeyboardInterrupt("operator interrupted")

        proc.communicate = first_call_interrupt
        return guardian, parent_write

    def permission_denied(pgid, signum):
        raise PermissionError("fixture controller cannot signal owned group")

    monkeypatch.setattr(worker_process, "_start_lifetime_guard", interrupt_after_guard_ready)
    # Only the controller's os.killpg is replaced; the guardian is a real process.
    real_killpg = os.killpg
    monkeypatch.setattr(worker_process.os, "killpg", permission_denied)
    try:
        with pytest.raises(KeyboardInterrupt, match="operator interrupted"):
            run_worker([sys.executable, str(tmp_path / "worker_fixture.py"),
                        str(tmp_path), "same"], cwd=tmp_path,
                       env=dict(os.environ), timeout=5)
        _assert_heartbeat_stopped(tmp_path)
    finally:
        if worker is not None:
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    real_killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            worker.communicate(timeout=3)


def test_guardian_pipe_creation_failure_closes_already_created_descriptors(tmp_path, monkeypatch):
    # Failing the second pipe must close both ends of the first one.
    script = tmp_path / "sleep_fixture.py"
    script.write_text("import time\ntime.sleep(60)\n")
    real_pipe = os.pipe
    created = []

    def fail_second_pipe():
        if created:
            raise OSError("fixture readiness pipe unavailable")
        descriptors = real_pipe()
        created.extend(descriptors)
        return descriptors

    real_start_guard = worker_process._start_lifetime_guard

    def fail_guard_pipe_creation(proc, *, watchdog_timeout=None):
        # Popen creates its own output/error pipes: patch only after the real
        # worker has launched, so the failure belongs to guardian setup.
        monkeypatch.setattr(worker_process.os, "pipe", fail_second_pipe)
        return real_start_guard(proc, watchdog_timeout=watchdog_timeout)

    monkeypatch.setattr(worker_process, "_start_lifetime_guard", fail_guard_pipe_creation)
    with pytest.raises(OSError, match="fixture readiness pipe unavailable"):
        run_worker([sys.executable, str(script)], cwd=tmp_path,
                   env=dict(os.environ), timeout=5, log_path=tmp_path / "worker.log")
    try:
        for descriptor in created:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in created:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("ready", [False, True])
def test_guardian_failure_keeps_original_error_when_cleanup_fails(tmp_path, monkeypatch, ready):
    # An error while stopping a not-ready guardian must not replace the reason
    # worker startup failed. Both fixture processes are independently cleaned.
    worker_script = tmp_path / "sleep_fixture.py"
    worker_script.write_text("import time\ntime.sleep(60)\n")
    guard_script = tmp_path / "failed_guard_fixture.py"
    guard_script.write_text(
        "import os, sys, time\n"
        + ("os.write(int(sys.argv[sys.argv.index('--ready-fd') + 1]), b'R')\n" if ready else "")
        +
        "os.close(int(sys.argv[sys.argv.index('--ready-fd') + 1]))\n"
        "time.sleep(60)\n"
    )
    real_popen = subprocess.Popen
    launched = []

    def start_fixture(cmd, **kwargs):
        is_guard = "--worker-pgid" in cmd
        if is_guard:
            cmd = [sys.executable, str(guard_script), *cmd[3:]]
        proc = real_popen(cmd, **kwargs)
        launched.append(proc)
        if is_guard:
            def fail_guard_cleanup():
                raise PermissionError("fixture guardian cleanup denied")
            proc.kill = fail_guard_cleanup
        elif ready:
            real_communicate = proc.communicate
            def first_call_interrupt(*args, **kwargs):
                proc.communicate = real_communicate
                raise KeyboardInterrupt("fixture operator interrupted")
            proc.communicate = first_call_interrupt
        return proc

    monkeypatch.setattr(worker_process.subprocess, "Popen", start_fixture)
    try:
        expected = KeyboardInterrupt if ready else RuntimeError
        detail = "fixture operator interrupted" if ready else "guardian exited before readiness"
        with pytest.raises(expected, match=detail):
            run_worker([sys.executable, str(worker_script)], cwd=tmp_path,
                       env=dict(os.environ), timeout=5)
    finally:
        for proc in launched:
            if proc.poll() is None:
                _kill_known_group(proc.pid)
            proc.wait(timeout=3)


@pytest.mark.parametrize("timeout", [None, 30])
@pytest.mark.parametrize("use_log", [False, True])
def test_cancellation_event_stops_owned_group_without_reporting_timeout(tmp_path, timeout, use_log):
    # Ignoring the event would wait the worker's full timeout and keep its child
    # running; cancellation is a separate outcome from an ML execution timeout.
    (tmp_path / "worker_fixture.py").write_text(WORKER_FIXTURE)
    cancellation = threading.Event()

    def cancel_when_ready():
        deadline = time.monotonic() + 3
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancellation.set()

    setter = threading.Thread(target=cancel_when_ready)
    setter.start()
    started = time.monotonic()
    try:
        outcome = run_worker([sys.executable, str(tmp_path / "worker_fixture.py"),
                              str(tmp_path), "same"], cwd=tmp_path,
                             env=dict(os.environ), timeout=timeout,
                             log_path=tmp_path / "worker.log" if use_log else None,
                             cancellation_event=cancellation)
        assert time.monotonic() - started < 5
        assert outcome.cancelled is True
        assert outcome.timed_out is False
        assert outcome.returncode is None
        assert "worker started" in outcome.stdout
        _assert_heartbeat_stopped(tmp_path)
    finally:
        cancellation.set()
        setter.join(timeout=4)
        path = tmp_path / "worker.pid"
        if path.exists() and not (tmp_path / "ready").exists():
            _kill_known_group(int(path.read_text()))


def test_escaped_fixture_has_a_deadline_even_if_controller_and_test_owner_die(tmp_path, monkeypatch):
    monkeypatch.setenv('EVOLVE_FIXTURE_LIFETIME_SECONDS', '0.6')
    controller = _start_controller(tmp_path, mode='escaped')
    try:
        _wait_file(tmp_path / 'ready', controller)
        _kill_known_group(controller.pid)
        _assert_heartbeat_stopped(tmp_path)
    finally:
        _cleanup_fixture(tmp_path, controller, escaped=True)

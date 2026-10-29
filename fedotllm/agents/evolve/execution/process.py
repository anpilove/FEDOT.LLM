"""Shared subprocess policy for every EvolveAgent execution boundary.

The generated candidate is trusted to run local FEDOT code, but it must not
inherit API tokens or accidentally use a different Python than ``doctor``.
This module is deliberately small so evaluator, pytest, import and snippet
workers all share exactly the same interpreter and environment policy.
"""

from __future__ import annotations

import os
import queue
import dataclasses
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class WorkerOutcome:
    """Result of one worker subprocess; ``timed_out`` marks its deadline expiry."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    timeout_kind: str | None = None


_CLEANUP_TIMEOUT = 2.0
_LOG_TAIL_BYTES = 1024 * 1024
_GUARDIAN_WATCHDOG_EXIT = 124
_RESOURCE_CHECK_INTERVAL = 1.0
_RESOURCE_CHECK_QUEUE_CAPACITY = 7
_resource_check_requests: queue.Queue[tuple[Future, Callable[[], object]]] = (
    queue.Queue(maxsize=_RESOURCE_CHECK_QUEUE_CAPACITY)
)
_resource_check_thread: threading.Thread | None = None
_resource_check_thread_lock = threading.Lock()


def _resource_check_loop() -> None:
    """Serialize storage probes so a blocked filesystem call cannot leak threads."""
    while True:
        future, callback = _resource_check_requests.get()
        if not future.set_running_or_notify_cancel():
            continue
        try:
            callback()
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(None)


def _ensure_resource_check_thread() -> None:
    global _resource_check_thread
    with _resource_check_thread_lock:
        if _resource_check_thread is None or not _resource_check_thread.is_alive():
            _resource_check_thread = threading.Thread(
                target=_resource_check_loop,
                name="evolve-resource-check",
                daemon=True,
            )
            _resource_check_thread.start()


class _AsyncResourceCheck:
    """Run a potentially blocking resource probe on the shared bounded worker."""

    def __init__(self, callback: Callable[[], object]):
        self.callback = callback
        self.future: Future | None = None
        self.pending = False

    def start(self) -> None:
        if self.pending:
            return
        self.pending = True
        self.future = Future()
        _ensure_resource_check_thread()
        try:
            _resource_check_requests.put_nowait((self.future, self.callback))
        except queue.Full:
            self.future.set_exception(
                RuntimeError(
                    "storage probe queue is full; an earlier probe is still blocked"
                )
            )

    def poll(self) -> bool:
        if not self.pending or self.future is None or not self.future.done():
            return False
        self.pending = False
        self.future.result()
        return True

    def cancel(self) -> None:
        if self.future is not None and not self.future.done():
            self.future.cancel()
        self.pending = False


def _kill_worker_group(proc: subprocess.Popen) -> bool:
    """Terminate the worker session, returning false when guardian fallback is needed."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        # Darwin can deny signaling a group after its leader has exited. The
        # independent lifetime guardian still owns this exact PGID and will
        # perform cleanup when its control pipe closes without "complete".
        return False
    return True


def _bounded_worker_output(proc: subprocess.Popen) -> tuple[str, str]:
    """Reap within a deadline even if an escaped descendant retains a pipe."""
    try:
        out, err = proc.communicate(timeout=_CLEANUP_TIMEOUT)
        return out or "", err or ""
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired carries the complete partial capture, possibly bytes
        # even for text=True. Do not append the earlier timeout's duplicate data.
        def _text(value):
            return (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value or ""
            )

        return _text(exc.stdout), _text(exc.stderr)
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        try:
            proc.wait(timeout=_CLEANUP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass


def _start_lifetime_guard(
    proc: subprocess.Popen,
    *,
    watchdog_timeout: float | None = None,
) -> tuple[subprocess.Popen, int]:
    """Bind a fixed guardian to this worker and a controller-owned pipe."""
    if watchdog_timeout is not None and watchdog_timeout <= 0:
        raise ValueError("watchdog_timeout must be positive")
    parent_read = parent_write = ready_read = ready_write = -1
    guardian = None
    try:
        parent_read, parent_write = os.pipe()
        ready_read, ready_write = os.pipe()
        command = [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("_worker_lifetime_guard.py")),
            "--worker-pgid",
            str(proc.pid),
            "--parent-fd",
            str(parent_read),
            "--ready-fd",
            str(ready_write),
        ]
        if watchdog_timeout is not None:
            command.extend(("--watchdog-timeout-seconds", str(watchdog_timeout)))
        guardian = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=(parent_read, ready_write),
            start_new_session=True,
        )
        os.close(parent_read)
        parent_read = -1
        os.close(ready_write)
        ready_write = -1
        if not select.select([ready_read], [], [], _CLEANUP_TIMEOUT)[0]:
            raise RuntimeError("worker lifetime guardian did not become ready")
        if os.read(ready_read, 1) != b"R":
            raise RuntimeError("worker lifetime guardian exited before readiness")
        return guardian, parent_write
    except BaseException as original:
        if parent_write != -1:
            os.close(parent_write)  # EOF tells any started guardian to clean up.
        if guardian is not None:
            try:
                _finish_lifetime_guard(guardian, None, completed=False)
            except Exception as cleanup_error:
                if hasattr(original, "add_note"):
                    original.add_note(
                        f"Guardian {guardian.pid} cleanup failed: {cleanup_error}"
                    )
        raise
    finally:
        for descriptor in (parent_read, ready_read, ready_write):
            if descriptor != -1:
                os.close(descriptor)


def _finish_lifetime_guard(
    guardian: subprocess.Popen,
    parent_write: int | None,
    *,
    completed: bool,
) -> int | None:
    """Disarm on normal completion; otherwise EOF requests group termination."""
    if parent_write is not None:
        try:
            if completed:
                try:
                    os.write(parent_write, b"complete\n")
                except BrokenPipeError:
                    pass
        finally:
            os.close(parent_write)
    stderr = ""
    try:
        _, stderr_bytes = guardian.communicate(timeout=_CLEANUP_TIMEOUT)
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired:
        guardian.kill()
        try:
            _, stderr_bytes = guardian.communicate(timeout=_CLEANUP_TIMEOUT)
            stderr = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
        except subprocess.TimeoutExpired:
            pass
    if guardian.returncode not in {0, _GUARDIAN_WATCHDOG_EXIT}:
        raise RuntimeError(
            f"worker lifetime guardian {guardian.pid} failed with exit code "
            f"{guardian.returncode} while cleaning process group"
            + (f": {stderr}" if stderr else "")
        )
    return guardian.returncode


def run_worker(
    cmd: list[str],
    *,
    cwd: Path | str,
    env: dict[str, str],
    timeout: float | None,
    watchdog_timeout: float | None = None,
    log_path: Path | None = None,
    cancellation_event: threading.Event | None = None,
    resource_check: Callable[[], object] | None = None,
    on_spawned: Callable[[int], object] | None = None,
) -> WorkerOutcome:
    """Run in a new session; kill its owned group on timeout or interruption.

    ``subprocess.run(timeout=...)`` only kills the direct child; FEDOT workers
    fork joblib/multiprocessing children that would otherwise keep burning the
    CPU quota as orphans after a wall-clock timeout.

    A fixed stdlib guardian also kills that group if the controller dies after
    guardian readiness. Descendants that deliberately create another session
    cannot be killed through this group; their pipes never block cleanup forever.

    With ``log_path`` both streams are written to that file (useful for hour-long
    jobs the operator wants to ``tail``). Only its last 1 MiB is read and returned
    as ``stdout``, decoding UTF-8 with replacement at a truncated character.
    ``timeout`` is the task deadline observed by this parent. ``watchdog_timeout``
    is an independent hard cap enforced by the fixed guardian if this parent stops
    monitoring. ``cancellation_event`` requests orderly group cleanup, checked at most every
    0.2 seconds while communicating; its result is ``cancelled``, not timed out.
    """

    resource_monitors: list[_AsyncResourceCheck] = []

    def _spawn(stdout, stderr) -> WorkerOutcome:
        started_at = time.monotonic()
        deadline = None if timeout is None else started_at + timeout
        resource_monitor = (
            _AsyncResourceCheck(resource_check) if resource_check is not None else None
        )
        if resource_monitor is not None:
            resource_monitors.append(resource_monitor)
            resource_monitor.start()
            while not resource_monitor.poll():
                if cancellation_event is not None and cancellation_event.is_set():
                    return WorkerOutcome(None, "", "", cancelled=True)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return WorkerOutcome(
                        None,
                        "",
                        "",
                        timed_out=True,
                        timeout_kind="task_timeout",
                    )
                time.sleep(0.05 if remaining is None else min(0.05, remaining))
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        guardian = None
        parent_write = None
        completed = False
        try:
            guardian, parent_write = _start_lifetime_guard(
                proc,
                watchdog_timeout=watchdog_timeout,
            )
            if on_spawned is not None:
                on_spawned(proc.pid)
            next_resource_check = time.monotonic() + _RESOURCE_CHECK_INTERVAL
            outcome = None
            while True:
                now = time.monotonic()
                if resource_monitor is not None:
                    if now >= next_resource_check:
                        resource_monitor.start()
                        next_resource_check = now + _RESOURCE_CHECK_INTERVAL
                    resource_monitor.poll()
                if cancellation_event is not None and cancellation_event.is_set():
                    completed = _kill_worker_group(proc)
                    out, err = _bounded_worker_output(proc)
                    outcome = WorkerOutcome(None, out, err, cancelled=True)
                    break
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                poll_timeout = remaining
                if cancellation_event is not None or resource_check is not None:
                    poll_timeout = 0.2 if remaining is None else min(0.2, remaining)
                try:
                    out, err = proc.communicate(timeout=poll_timeout)
                    break
                except subprocess.TimeoutExpired:
                    if (
                        cancellation_event is not None or resource_check is not None
                    ) and (deadline is None or time.monotonic() < deadline):
                        continue
                    completed = _kill_worker_group(proc)
                    out, err = _bounded_worker_output(proc)
                    outcome = WorkerOutcome(
                        None,
                        out,
                        err,
                        timed_out=True,
                        timeout_kind="task_timeout",
                    )
                    break
            # File redirects let communicate return while same-group children
            # remain alive. Clear those owned descendants before disarming.
            if outcome is None:
                completed = _kill_worker_group(proc)
                outcome = WorkerOutcome(proc.returncode, out or "", err or "")
        except BaseException as original:
            # Cleanup must also cover guardian startup failure and interrupts.
            # Preserve the original exception even when cleanup itself fails.
            try:
                _kill_worker_group(proc)
            except Exception as cleanup_error:
                if hasattr(original, "add_note"):
                    original.add_note(
                        f"Worker group {proc.pid} cleanup failed: {cleanup_error}"
                    )
            try:
                _bounded_worker_output(proc)
            except Exception as cleanup_error:
                if hasattr(original, "add_note"):
                    original.add_note(
                        f"Worker {proc.pid} output cleanup failed: {cleanup_error}"
                    )
            raise
        finally:
            if guardian is not None:
                original = sys.exc_info()[1]
                try:
                    guardian_status = _finish_lifetime_guard(
                        guardian,
                        parent_write,
                        completed=completed,
                    )
                    if (
                        guardian_status == _GUARDIAN_WATCHDOG_EXIT
                        and outcome is not None
                    ):
                        outcome = dataclasses.replace(
                            outcome,
                            returncode=None,
                            timed_out=True,
                            timeout_kind="watchdog_timeout",
                        )
                except Exception as cleanup_error:
                    if original is None:
                        raise
                    if hasattr(original, "add_note"):
                        original.add_note(
                            f"Guardian {guardian.pid} cleanup failed: {cleanup_error}"
                        )
        if outcome is None:
            raise RuntimeError("worker ended without producing an outcome")
        return outcome

    try:
        if log_path is None:
            return _spawn(subprocess.PIPE, subprocess.PIPE)
        with log_path.open("w", encoding="utf-8") as handle:
            outcome = _spawn(handle, subprocess.STDOUT)
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
            text = handle.read(_LOG_TAIL_BYTES).decode("utf-8", errors="replace")
        return WorkerOutcome(
            outcome.returncode,
            text,
            "",
            timed_out=outcome.timed_out,
            cancelled=outcome.cancelled,
            timeout_kind=outcome.timeout_kind,
        )
    finally:
        for monitor in resource_monitors:
            monitor.cancel()


_SAFE_ENV_NAMES = {
    "HOME",
    "PATH",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "MPLBACKEND",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "SYSTEMROOT",
    "WINDIR",
}

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def thread_environment() -> dict[str, str]:
    """Bound small evaluator workloads; preserve explicit user overrides."""
    return {name: os.environ.get(name, "1") for name in _THREAD_ENV_NAMES}


def fedot_python(checkout: Path) -> str:
    """Return the one interpreter used by every FEDOT subprocess."""

    configured = os.environ.get("FEDOTLLM_REPO_PYTHON")
    if configured:
        path = Path(os.path.abspath(Path(configured).expanduser()))
        if not path.is_file():
            raise FileNotFoundError(f"FEDOTLLM_REPO_PYTHON does not exist: {path}")
        return str(path)
    for candidate in (
        checkout / ".venv" / "bin" / "python",
        checkout.parent / ".venv-fedot" / "bin" / "python",
    ):
        if candidate.is_file():
            # Resolving a venv symlink loses pyvenv.cfg and installed packages.
            return str(Path(os.path.abspath(candidate)))
    if Path(sys.executable).is_file():
        return os.path.abspath(sys.executable)
    discovered = shutil.which("python3") or shutil.which("python")
    if discovered:
        return os.path.abspath(discovered)
    raise FileNotFoundError("no Python interpreter available for FEDOT subprocesses")


@lru_cache(maxsize=8)
def _interpreter_identity(executable: str) -> str:
    """Fingerprint the selected interpreter, not merely the controller Python."""

    env = {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_ENV_NAMES or name.startswith("LC_")
    }
    env["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [executable, str(Path(__file__).with_name("_interpreter_probe.py"))],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        path = Path(executable)
        try:
            stat = path.stat()
            return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            return executable


def interpreter_identity(checkout: Path, executable: str | None = None) -> str:
    return _interpreter_identity(executable or fedot_python(checkout))


def clean_subprocess_env(
    checkout: Path,
    *,
    repo_root: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal environment without credentials.

    ``repo_root`` is appended only when a worker is launched with ``python -m``;
    the checkout remains first so ``import fedot`` always resolves to the
    experiment rather than an installed package.
    """

    env = {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_ENV_NAMES or name.startswith("LC_")
    }
    python_paths = [str(checkout.resolve())]
    if repo_root is not None:
        python_paths.append(str(repo_root.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(thread_environment())
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env

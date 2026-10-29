"""Opt-in Python thread stacks from this quality worker's compute children.

The static startup hook covers loky/spawn interpreters; multiprocessing's
after-fork callback starts a fresh sampler for inherited runtimes. No process
discovery, signals, backend changes, or FEDOT edits are needed.
Raw ``os.fork`` children outside multiprocessing, ordinary Python subprocess
scripts, fresh Process subclasses overriding ``run``, and children launched
with ``-I``/``-S`` or overridden startup environments need a separate hook.
Reuse of a compute pool across attempts is unsupported: this context belongs to
one isolated worker/attempt. Native C/C++ frames are unavailable; a sampler
cannot run while a native call holds the GIL. A Python
frame identifies a call site, not proof of useful work. C watchdogs and signals
are intentionally avoided because they can deadlock or interrupt native calls.
"""

import atexit
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import uuid


_ENV_NAMES = (
    "EVOLVE_TRACE_DIRECTORY",
    "EVOLVE_TRACE_ATTEMPT_ID",
    "EVOLVE_TRACE_OWNER_PID",
    "EVOLVE_TRACE_INTERVAL_SECONDS",
    "PYTHONPATH",
)
_BOOTSTRAP_DIRECTORY = Path(__file__).with_name("_trace_bootstrap")
_MANIFEST_RECORD_LIMIT = 4096


def _warn(exc):
    # The exception message may contain an artifact path, never environment or
    # task inputs. Warnings cannot turn observational diagnostics into failure.
    try:
        print(f"child stack diagnostic warning: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
    except Exception:
        pass


class _TraceState:
    def __init__(self):
        self.pid = os.getpid()
        self.context = None
        self.line_trace_status = None
        self.armed = False
        self.registered = False
        self.finalizer = None
        self.attempted = False
        self.run_hook_installed = False

    def close(self):
        if self.pid != os.getpid():
            return
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None
        self.armed = False


_STATE = _TraceState()  # register_after_fork keeps only a weak reference.


def _after_fork(state):
    install_child_trace_from_environment(source="multiprocessing_after_fork")


def _register_lifecycle():
    if not _STATE.registered:
        from multiprocessing.util import register_after_fork

        register_after_fork(_STATE, _after_fork)
        atexit.register(_close_child_trace)
        _STATE.registered = True


def _close_child_trace():
    try:
        _STATE.close()
    except Exception as exc:
        _warn(exc)


def _defer_until_process_run():
    # Standard multiprocessing/loky target unpickling and startup imports finish
    # before BaseProcess.run; resource helper scripts need no compute sampler.
    if _STATE.run_hook_installed:
        return
    from multiprocessing.process import BaseProcess

    original_run = BaseProcess.run

    def run_with_child_trace(process):
        install_child_trace_from_environment(source="process_run")
        try:
            return original_run(process)
        finally:
            _close_child_trace()

    BaseProcess.run = run_with_child_trace
    _STATE.run_hook_installed = True


def _append_registration(root, record):
    serialized = json.dumps(record, sort_keys=True, ensure_ascii=True)
    manifest_record = (serialized + "\n").encode("utf-8")
    if len(manifest_record) >= _MANIFEST_RECORD_LIMIT:
        raise ValueError("child trace manifest record exceeds 4095 bytes")
    with Path(record["metadata_path"]).open("x", encoding="utf-8") as metadata:
        metadata.write(serialized + "\n")
    descriptor = os.open(root / "manifest.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        if os.write(descriptor, manifest_record) != len(manifest_record):
            raise OSError("incomplete child trace manifest append")
    finally:
        os.close(descriptor)


def install_child_trace_from_environment(*, source="sitecustomize"):
    """Install once per child PID; diagnostic failures never fail its task.

    Only the explicit private trace settings are read. Metadata excludes
    arguments, local variables, credentials, and unrelated environment values.
    """
    global _STATE
    directory = os.environ.get("EVOLVE_TRACE_DIRECTORY")
    if not directory:
        return
    try:
        _register_lifecycle()
        pid = os.getpid()
        owner_pid = int(os.environ["EVOLVE_TRACE_OWNER_PID"])
        if pid == owner_pid or (_STATE.pid == pid and _STATE.attempted):
            return
        if source == "sitecustomize" and _STATE.pid == pid:
            _defer_until_process_run()
            return
        if source == "multiprocessing_after_fork" and _STATE.pid == pid:
            return
        interval = float(os.environ["EVOLVE_TRACE_INTERVAL_SECONDS"])
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("child trace interval must be finite and positive")
        inherited_runtime = _STATE.pid != pid
        if inherited_runtime:
            # Never exit/join an inherited sampler: its thread vanished at fork.
            # stack_sampler also guards generator finalization by owner PID.
            _STATE = _TraceState()
            _register_lifecycle()
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        stem = f"process-{pid}-{token}"
        stack_path = root / f"{stem}.stacks.txt"
        metadata_path = root / f"{stem}.json"
        record = {
            "schema_version": 1,
            "pid": pid,
            "ppid": os.getppid(),
            "owner_pid": owner_pid,
            "attempt_id": os.environ["EVOLVE_TRACE_ATTEMPT_ID"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "interval_seconds": interval,
            "source": source,
            "scope": "python_process_threads",
            "capture_method": "python_thread",
            "coverage": "armed",
            "limitations": ["native_call_holding_gil", "native_c_frames"],
            "stack_trace_path": str(stack_path),
            "metadata_path": str(metadata_path),
        }
        _STATE.attempted = True
        from fedotllm.agents.evolve.evaluation.python_line_trace import python_line_samples
        from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples

        _STATE.context = ExitStack()
        sampler_status = _STATE.context.enter_context(
            periodic_stack_samples(stack_path, interval_seconds=interval,
                                   scope="python_process_threads")
        )
        if os.environ.get("EVOLVE_TRACE_PYTHON_LINES") == "1":
            try:
                line_interval = float(os.environ.get("EVOLVE_TRACE_LINE_INTERVAL_SECONDS", "1"))
            except ValueError:
                line_interval = 1.0
            _STATE.line_trace_status = _STATE.context.enter_context(
                python_line_samples(
                    os.environ.get("FEDOTLLM_REPO_PATH"), root,
                    interval_seconds=line_interval,
                )
            )
            record.update(
                python_line_trace_enabled=bool(_STATE.line_trace_status.get("enabled")),
                python_line_trace_path=_STATE.line_trace_status.get("output_path"),
            )
        if not sampler_status["enabled"]:
            record.update(coverage="unavailable", reason=sampler_status.get("error", "sampler_unavailable"),
                          stack_trace_path=None)
            _STATE.close()
            _append_registration(root, record)
            return
        _STATE.armed = True
        _append_registration(root, record)
        from multiprocessing.util import Finalize

        _STATE.finalizer = Finalize(None, _close_child_trace, exitpriority=10)
    except Exception as exc:
        _close_child_trace()
        _warn(exc)


@contextmanager
def traced_children(directory: Path | None, *, attempt_id: str, interval_seconds: float = 60):
    """Give future compute children their own sampler; restore parent env.

    The caller keeps its existing root sampler. This context belongs in the
    isolated quality worker, not in a multithreaded controller. Child processes
    own their diagnostics until they exit, including reusable loky workers.
    """
    status = {
        "enabled": False,
        "directory": str(directory) if directory is not None else None,
        "attempt_id": attempt_id,
        "interval_seconds": interval_seconds,
        "scope": "descendant_python_process_threads",
        "capture_method": "python_thread",
        "unsupported": ["raw_os_fork", "ordinary_python_subprocess",
                        "python_isolated_or_no_site", "child_environment_override",
                        "custom_process_run_override", "reused_pool_across_attempts",
                        "native_call_holding_gil", "native_c_frames"],
    }
    if directory is None:
        yield status
        return
    previous = {name: os.environ.get(name) for name in _ENV_NAMES}
    try:
        try:
            if not math.isfinite(interval_seconds) or interval_seconds <= 0:
                raise ValueError("child trace interval must be finite and positive")
            root = Path(directory).resolve()
            root.mkdir(parents=True, exist_ok=True)
            os.environ.update({
                "EVOLVE_TRACE_DIRECTORY": str(root),
                "EVOLVE_TRACE_ATTEMPT_ID": str(attempt_id),
                "EVOLVE_TRACE_OWNER_PID": str(os.getpid()),
                "EVOLVE_TRACE_INTERVAL_SECONDS": str(interval_seconds),
                "PYTHONPATH": os.pathsep.join(
                    [str(_BOOTSTRAP_DIRECTORY.resolve())]
                    + ([previous["PYTHONPATH"]] if previous["PYTHONPATH"] else [])
                ),
            })
            install_child_trace_from_environment()
            status.update({
                "enabled": True,
                "directory": str(root),
                "manifest_path": str(root / "manifest.jsonl"),
                "owner_pid": os.getpid(),
            })
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            _warn(exc)
        yield status
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

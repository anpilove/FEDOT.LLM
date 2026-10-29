"""Opt-in sampled Python source lines for diagnosing stalled FEDOT calls.

This records the active FEDOT Python frames periodically rather than writing
every interpreter line event. It stays lightweight enough to leave enabled for
one bounded diagnostic run and still retains the exact line across a blocking
native call. Native C/C++ frames remain unavailable.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterator


def _sample_line_snapshots(
    thread_states: dict[int, dict],
    lock: threading.RLock,
    *,
    timestamp_utc: str,
    monotonic_ns: int,
    pid: int,
    context: dict | None = None,
) -> list[dict]:
    """Copy trace state under lock, then enrich it without holding that lock.

    The sampler must not import FEDOT to look up thread context: that import
    can race the worker's FEDOT import and trigger CPython's module-lock
    deadlock detector. The worker passes immutable attempt context at setup.
    """
    with lock:
        states = [
            (
                thread_id,
                state["thread_name"],
                [dict(frame) for frame in state["frames"]],
            )
            for thread_id, state in thread_states.items()
            if state["frames"]
        ]

    snapshots = []
    for thread_id, thread_name, frames in states:
        snapshots.append(
            {
                "event": "fedot_python_line_sample",
                "timestamp_utc": timestamp_utc,
                "monotonic_ns": monotonic_ns,
                "pid": pid,
                "ppid": os.getppid(),
                "run_id": os.environ.get("EVOLVE_TRACE_ATTEMPT_ID"),
                "thread_id": thread_id,
                "thread_name": thread_name,
                **(context or {}),
                "fedot_frames": [
                    {key: value for key, value in frame.items() if key != "frame_id"}
                    for frame in frames
                ],
            }
        )
    return snapshots


@contextmanager
def python_line_samples(
    source_root: Path | str | None,
    output_dir: Path | str | None,
    *,
    interval_seconds: float = 1.0,
    context: dict | None = None,
) -> Iterator[dict]:
    """Sample active Python frames under ``source_root`` into per-PID JSONL.

    The trace hook only updates in-memory locations; a separate thread writes
    periodic snapshots. No function arguments, locals, or source text are
    captured. Errors in this optional diagnostic are fail-open.
    """
    status = {
        "enabled": False,
        "source_root": str(source_root) if source_root is not None else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "interval_seconds": interval_seconds,
        "scope": "active_python_frames_under_fedot_source",
        "capture_method": "sys.settrace_and_threading.settrace_with_periodic_snapshots",
        "limitations": ["native_c_frames", "threads_created_before_tracing"],
    }
    if source_root is None or output_dir is None:
        yield status
        return
    try:
        interval = float(interval_seconds)
        if not 0.01 <= interval <= 60:
            raise ValueError("line trace interval must be between 0.01 and 60 seconds")
        root = Path(source_root).resolve(strict=True)
        output = Path(output_dir).resolve(strict=False)
        if not root.is_dir():
            raise ValueError("FEDOT source root is not a directory")
        output.mkdir(parents=True, exist_ok=True)
        path_prefix = str(root) + os.sep
        pid = os.getpid()
        thread_states: dict[int, dict] = {}
        lock = threading.RLock()
        stop = threading.Event()
        filename = output / f"python-lines-{pid}.jsonl"
        previous_trace = sys.gettrace()
        previous_thread_trace = threading.gettrace()
        code_path_cache: dict[object, bool] = {}

        def is_fedot_frame(frame) -> bool:
            code = frame.f_code
            inside = code_path_cache.get(code)
            if inside is None:
                source = os.path.realpath(code.co_filename)
                inside = source == str(root) or source.startswith(path_prefix)
                code_path_cache[code] = inside
            return inside

        def trace(frame, event, _arg):
            thread_id = threading.get_ident()
            if event == "call":
                if not is_fedot_frame(frame):
                    return None
                with lock:
                    state = thread_states.setdefault(
                        thread_id,
                        {
                            "thread_name": threading.current_thread().name,
                            "frames": [],
                        },
                    )
                    state["frames"].append(
                        {
                            "frame_id": id(frame),
                            "filename": os.path.realpath(frame.f_code.co_filename),
                            "function": frame.f_code.co_name,
                            "lineno": frame.f_lineno,
                        }
                    )
                return trace

            with lock:
                state = thread_states.get(thread_id)
                if state is None:
                    return None
                frames = state["frames"]
                frame_id = id(frame)
                index = next(
                    (i for i, row in enumerate(frames) if row["frame_id"] == frame_id),
                    None,
                )
                if index is None:
                    return trace if is_fedot_frame(frame) else None
                if event == "return":
                    frames.pop(index)
                    if not frames:
                        thread_states.pop(thread_id, None)
                        return None
                else:
                    frames[index]["lineno"] = frame.f_lineno
                    frames[index]["last_event"] = event
                    frames[index]["last_line_at_monotonic_ns"] = time.monotonic_ns()
            return trace

        def sample() -> None:
            while not stop.wait(interval):
                now = datetime.now(timezone.utc).isoformat()
                snapshots = _sample_line_snapshots(
                    thread_states,
                    lock,
                    timestamp_utc=now,
                    monotonic_ns=time.monotonic_ns(),
                    pid=pid,
                    context=context,
                )
                if not snapshots:
                    continue
                try:
                    with filename.open("a", encoding="utf-8") as stream:
                        for snapshot in snapshots:
                            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
                        stream.flush()
                except Exception as exc:
                    status["write_error"] = f"{type(exc).__name__}: {exc}"

        sampler = threading.Thread(
            target=sample, name=f"fedot-line-sampler-{pid}", daemon=True
        )
        try:
            sys.settrace(trace)
            threading.settrace(trace)
            sampler.start()
            status.update(enabled=True, output_path=str(filename), pid=pid)
        except Exception as exc:
            sys.settrace(previous_trace)
            threading.settrace(previous_thread_trace)
            status["error"] = f"{type(exc).__name__}: {exc}"
            yield status
            return

        try:
            yield status
        finally:
            stop.set()
            sampler.join(timeout=max(1.0, interval * 2))
            sys.settrace(previous_trace)
            threading.settrace(previous_thread_trace)
    except Exception as exc:
        if status["enabled"]:
            raise
        status["error"] = f"{type(exc).__name__}: {exc}"
        yield status


def python_line_samples_from_environment(
    *, output_dir=None, interval_seconds=None, context=None
):
    """Create the opt-in sampler from the worker's diagnostic environment."""
    if os.environ.get("EVOLVE_TRACE_PYTHON_LINES") != "1":
        return python_line_samples(None, None)
    source_root = os.environ.get("FEDOTLLM_REPO_PATH")
    if output_dir is None:
        output_dir = os.environ.get("EVOLVE_TRACE_DIRECTORY")
    if interval_seconds is None:
        try:
            interval_seconds = float(
                os.environ.get("EVOLVE_TRACE_INTERVAL_SECONDS", "1")
            )
        except ValueError:
            interval_seconds = 1.0
    return python_line_samples(
        source_root, output_dir, interval_seconds=interval_seconds, context=context
    )

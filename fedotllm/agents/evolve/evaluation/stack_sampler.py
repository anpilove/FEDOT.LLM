"""Observational Python stacks without signals or native watchdog threads.

The sampler needs the GIL. A native call that holds the GIL prevents capture
until that call returns; missing samples never prove a stopped process. Frames
contain only file, function and line, never source text, locals or arguments.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading


_START_WAIT_SECONDS = 0.1
_STOP_JOIN_SECONDS = 0.1
_MAX_FRAMES_PER_THREAD = 100
_LIMITATIONS = (
    "Requires the GIL; native code holding it prevents samples until it returns. "
    "Python call sites only, no native frames; at most 100 frames per thread. "
    "Repeated frames are observations, not proof of a hang."
)


def _write_snapshot(descriptor, payload):
    """Use an unbuffered descriptor, avoiding inherited Python file locks."""
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("stack artifact write made no progress")
        remaining = remaining[written:]


def _capture_frames():
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [f"--- stack dump at {timestamp} ---\n"]
    frames = sys._current_frames()
    try:
        for thread_id, frame in sorted(frames.items()):
            lines.append(f"Thread {hex(thread_id)} (most recent call first):\n")
            depth = 0
            while frame is not None and depth < _MAX_FRAMES_PER_THREAD:
                code = frame.f_code
                filename = json.dumps(code.co_filename, ensure_ascii=False)
                lines.append(f"  File {filename}, line {frame.f_lineno} in {code.co_name}\n")
                frame = frame.f_back
                depth += 1
            if frame is not None:
                lines.append("  ... Python stack truncated after 100 frames ...\n")
    finally:
        # Our own frame is also in this snapshot. Release frame references
        # promptly rather than retaining observed tasks through a reference cycle.
        frames.clear()
        frame = None
    lines.append("--- stack dump complete ---\n")
    return timestamp, "".join(lines).encode("utf-8", errors="replace")


@contextmanager
def periodic_stack_samples(path, *, interval_seconds=15.0, scope="python_process_threads"):
    """Sample all Python threads; diagnostics never control the observed work.

    The daemon owns its descriptor and closes it itself. Context exit waits
    briefly, then leaves a blocked writer to finish without closing its live
    descriptor. A forked child never stops or joins the inherited thread state.
    """
    artifact = Path(path) if path else None
    owner_pid = os.getpid()
    status = {
        "enabled": False,
        "path": str(artifact) if artifact is not None else None,
        "interval_seconds": interval_seconds,
        "scope": scope,
        "capture_method": "python_thread",
        "limitations": _LIMITATIONS,
        "dump_count": 0,
        "last_dump_at": None,
        "stop_pending": False,
    }
    try:
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("stack sample interval must be finite and positive")
        status["interval_seconds"] = interval
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        yield status
        return
    if artifact is None:
        yield status
        return

    stop, ready = threading.Event(), threading.Event()

    def sample():
        descriptor = None
        try:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(artifact, flags, 0o600)
            status["enabled"] = True
            ready.set()
            while not stop.wait(interval):
                timestamp, payload = _capture_frames()
                _write_snapshot(descriptor, payload)
                status["dump_count"] += 1
                status["last_dump_at"] = timestamp
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            ready.set()
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    status["error"] = f"{type(exc).__name__}: {exc}"
            status["stop_pending"] = False

    thread = threading.Thread(target=sample, name="evolve-stack-sampler", daemon=True)
    try:
        thread.start()
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        yield status
        return
    ready.wait(_START_WAIT_SECONDS)
    try:
        yield status
    finally:
        if os.getpid() == owner_pid:
            stop.set()
            thread.join(_STOP_JOIN_SECONDS)
            status["stop_pending"] = thread.is_alive()

"""Fixed subprocess entry point for one controller-owned tuning evaluation.

The small supervisor never imports FEDOT/native numeric code. It owns the
compute session, watches the caller's lifetime pipe, and kills/reaps the session
before publishing a result. Only the compute child unpickles private input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

MAX_RESULT_BYTES = 65536


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def event(directory, name, **fields):
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"event": name, "monotonic": time.monotonic(), **fields},
                allow_nan=False,
            )
            + "\n"
        )


def read_json(path):
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError("oversized result")
    return json.loads(path.read_text(encoding="utf-8"))


def execute(directory, start_fd):
    if os.read(start_fd, 1) != b"R":
        raise RuntimeError("compute guardian was not armed")
    os.close(start_fd)
    from fedotllm.agents.evolve.evaluation._tuning_process_group_guard import (
        install_process_group_guard,
    )

    containment = install_process_group_guard()
    atomic_json(directory / "containment.json", containment)
    # Import-time GOLEM bindings must be private before cloudpickle imports
    # FEDOT classes. TMPDIR alone does not affect GOLEM on Darwin.
    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import (
        configure_worker_runtime,
    )

    runtime = configure_worker_runtime(directory / "compute.json")
    atomic_json(directory / "runtime.json", runtime)
    import cloudpickle

    with (directory / "input.pkl").open("rb") as stream:
        evaluator, graph, rng = cloudpickle.load(stream)
    import random

    numpy = None
    if rng["numpy"] is not None:
        import numpy

        numpy.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    producer = evaluator._data_producer
    objective = evaluator._objective
    state = {
        "started": 0,
        "completed": 0,
        "exhausted": False,
        "invalid": False,
        "max_fold_seconds": 0.0,
    }

    def folds():
        for fold in producer():
            index = state["started"]
            state["started"] += 1
            started = time.monotonic()
            atomic_json(
                directory / "progress.json", {"fold": index, "started": started}
            )
            event(directory, "fold_started", fold=index)
            yield fold
            elapsed = time.monotonic() - started
            state["max_fold_seconds"] = max(state["max_fold_seconds"], elapsed)
            event(directory, "fold_complete", fold=index, elapsed_seconds=elapsed)
        state["exhausted"] = True
        atomic_json(directory / "progress.json", {"fold": None, "started": None})

    def checked_objective(*args, **kwargs):
        fitness = objective(*args, **kwargs)
        if (
            not fitness.valid
            or not fitness.values
            or not all(math.isfinite(float(value)) for value in fitness.values)
        ):
            state["invalid"] = True
        else:
            state["completed"] += 1
        return fitness

    checked_objective.is_multi_objective = getattr(
        objective, "is_multi_objective", False
    )
    evaluator._data_producer = folds
    evaluator._objective = checked_objective
    try:
        fitness = evaluator.evaluate(graph)
        if (
            state["invalid"]
            or not fitness.valid
            or not fitness.values
            or not all(math.isfinite(float(v)) for v in fitness.values)
        ):
            status = "invalid_fitness"
        elif (
            not state["exhausted"]
            or not state["started"]
            or state["completed"] != state["started"]
        ):
            status = "incomplete_cv"
        else:
            status = "complete"
        result = {
            "status": status,
            "values": list(map(float, fitness.values))
            if status == "complete"
            else None,
            "completed_folds": state["completed"],
        }
    except BaseException as exc:
        result = {
            "status": "evaluation_error",
            "values": None,
            "completed_folds": state["completed"],
            "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
        }
    numpy_state = None
    if numpy is not None:
        numpy_state = list(numpy.random.get_state())
        numpy_state[1] = numpy_state[1].tolist()
    result.update(
        max_fold_seconds=state["max_fold_seconds"],
        finished_at=time.monotonic(),
        rng_state={"python": random.getstate(), "numpy": numpy_state},
    )
    if len(json.dumps(result)) > MAX_RESULT_BYTES:
        result = {"status": "invalid_result", "values": None, "completed_folds": 0}
    atomic_json(directory / "result.json", result)
    # Native libraries can leave non-daemon Python threads behind. The supervisor
    # performs process-group cleanup even after a successful metric transfer.
    os._exit(0)


def cleanup(process, before_reap=None):
    # Keep the leader unreaped until after signaling its group: no PGID reuse race.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin reports EPERM for a zombie-only group. Reap our leader and
        # verify disappearance below; a surviving group is still failure.
        pass
    if before_reap is not None:
        before_reap()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        return False
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        # Linux supervisor is a subreaper; reclaim only this owned process group.
        try:
            while os.waitpid(-process.pid, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Darwin can retain an orphaned zombie group briefly. Permission
            # failure is not proof of disappearance; retry only our owned PGID.
            pass
        time.sleep(0.02)
    return False


def supervise(directory, parent_fd, deadline, deadline_kind, fold_seconds):
    if sys.platform.startswith("linux"):
        import ctypes

        if (
            ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0
        ):  # PR_SET_CHILD_SUBREAPER
            raise OSError("cannot enable owned descendant reaping")
    result = {"status": "supervisor_error", "values": None, "completed_folds": 0}
    start_read, start_write = os.pipe()
    guardian = None
    guard_write = None
    with (directory / "child.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "execute",
                str(directory),
                "--start-fd",
                str(start_read),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            pass_fds=(start_read,),
            close_fds=True,
        )
    os.close(start_read)
    exit_queue = None
    try:
        guard_read, guard_write = os.pipe()
        ready_read, ready_write = os.pipe()
        guard_script = (
            Path(__file__).resolve().parent.parent
            / "execution"
            / "_worker_lifetime_guard.py"
        )
        try:
            with (directory / "guardian.log").open("ab") as log:
                guardian = subprocess.Popen(
                    [
                        sys.executable,
                        str(guard_script),
                        "--worker-pgid",
                        str(process.pid),
                        "--parent-fd",
                        str(guard_read),
                        "--ready-fd",
                        str(ready_write),
                        "--watchdog-timeout-seconds",
                        str(max(0.01, deadline - time.monotonic()) + 4),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=(guard_read, ready_write),
                    close_fds=True,
                )
            if not select.select(
                [ready_read], [], [], min(2, max(0, deadline - time.monotonic()))
            )[0]:
                raise RuntimeError("compute lifetime guardian readiness timeout")
            if os.read(ready_read, 1) != b"R":
                raise RuntimeError("compute lifetime guardian failed to arm")
        finally:
            os.close(guard_read)
            os.close(ready_read)
            os.close(ready_write)
        os.write(start_write, b"R")
        os.close(start_write)
        start_write = None
        if not hasattr(os, "waitid"):
            exit_queue = select.kqueue()
            exit_queue.control(
                [
                    select.kevent(
                        process.pid,
                        filter=select.KQ_FILTER_PROC,
                        flags=select.KQ_EV_ADD,
                        fflags=select.KQ_NOTE_EXIT,
                    )
                ],
                0,
                0,
            )
        event(directory, "worker_started", worker_pid=process.pid)
        atomic_json(
            directory / "worker.json",
            {"worker_pid": process.pid, "guardian_pid": guardian.pid},
        )
        while True:
            if guardian.poll() is not None:
                result["status"] = "guardian_error"
                break
            if select.select([parent_fd], [], [], 0)[0] and not os.read(parent_fd, 1):
                result["status"] = "parent_exited"
                break
            now = time.monotonic()
            if now >= deadline:
                result["status"] = deadline_kind
                break
            progress = directory / "progress.json"
            if progress.exists():
                current = read_json(progress)
                if (
                    current.get("started") is not None
                    and now - current["started"] >= fold_seconds
                ):
                    result["status"] = "fold_timeout"
                    break
            output = directory / "result.json"
            if output.exists():
                result = read_json(output)
                break
            # waitid(WNOWAIT) observes exit without reaping the PGID anchor.
            if hasattr(os, "waitid"):
                exited = os.waitid(
                    os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
                if exited is not None:
                    # File creation precedes exit, but may race the earlier check.
                    result = (
                        read_json(output)
                        if output.exists()
                        else {**result, "status": "child_crash"}
                    )
                    break
            elif exit_queue.control(None, 1, 0):
                result = (
                    read_json(output)
                    if output.exists()
                    else {**result, "status": "child_crash"}
                )
                break
            time.sleep(0.02)
    except BaseException as exc:
        result = {
            **result,
            "status": "supervisor_error",
            "values": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        fold_elapsed = result.pop("max_fold_seconds", None)
        finished_at = result.pop("finished_at", None)
        if result["status"] == "complete":
            if finished_at is None or fold_elapsed is None:
                result.update(status="invalid_result", values=None)
            elif finished_at >= deadline:
                result.update(status=deadline_kind, values=None)
            elif fold_elapsed >= fold_seconds:
                result.update(status="fold_timeout", values=None)

        def disarm_guardian():
            # Group already received SIGKILL, but leader is still unreaped. The
            # guard must exit before releasing that PID/PGID reuse anchor.
            if guardian is None:
                return
            try:
                os.write(guard_write, b"complete\n")
            except BrokenPipeError:
                pass
            try:
                guardian.wait(timeout=2)
            except subprocess.TimeoutExpired:
                guardian.kill()
                guardian.wait(timeout=2)
                raise RuntimeError("compute guardian failed to disarm")

        verified = cleanup(process, before_reap=disarm_guardian)
        if start_write is not None:
            os.close(start_write)
        if guard_write is not None:
            os.close(guard_write)
        containment_path = directory / "containment.json"
        if containment_path.exists():
            result["containment"] = read_json(containment_path)
        elif result["status"] == "child_crash":
            result["status"] = "bootstrap_error"
        result.update(worker_pid=process.pid, cleanup_verified=verified)
        if not verified:
            result.update(status="cleanup_failed", values=None)
        event(directory, "cleanup", verified=verified, worker_pid=process.pid)
        event(
            directory,
            "evaluation_finished",
            status=result["status"],
            completed_folds=result.get("completed_folds", 0),
        )
        atomic_json(directory / "status.json", result)
        os.close(parent_fd)
        if exit_queue is not None:
            exit_queue.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["supervise", "execute"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parent-fd", type=int)
    parser.add_argument("--start-fd", type=int)
    parser.add_argument("--deadline", type=float)
    parser.add_argument(
        "--deadline-kind", choices=["tuning_timeout", "candidate_timeout"]
    )
    parser.add_argument("--fold-seconds", type=float)
    args = parser.parse_args()
    if args.mode == "execute":
        try:
            execute(args.directory, args.start_fd)
        except BaseException as exc:
            atomic_json(
                args.directory / "result.json",
                {
                    "status": "bootstrap_error",
                    "values": None,
                    "completed_folds": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
                },
            )
            raise
    else:
        supervise(
            args.directory,
            args.parent_fd,
            args.deadline,
            args.deadline_kind,
            args.fold_seconds,
        )


if __name__ == "__main__":
    main()

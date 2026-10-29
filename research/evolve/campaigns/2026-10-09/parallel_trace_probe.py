#!/usr/bin/env python3
"""LOCAL instrumentation fixture, not a FEDOT reproduction; default reads only.

Explicit launch creates four bounded synchronized child processes and retains
their logs, production sampled stacks, metadata, and a hashed launch audit.
No SSH, data loading, FEDOT training, process discovery, or old-run changes.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
CASES = ("cpu_loop", "wait", "native_gil", "raised_exception")
PURPOSE = "instrumentation_fixture_not_fedot_reproduction"
STACK_INTERVAL_SECONDS = 0.1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fixture_cpu_loop() -> None:
    deadline = time.monotonic() + 1.0
    value = 1
    while time.monotonic() < deadline:
        value = (value * 37 + 17) % 1000003
    print(f"bounded CPU loop result={value}", flush=True)


def fixture_wait() -> None:
    time.sleep(1.0)


def fixture_native_gil() -> dict:
    # Observe this wrapper before the call. A Python sampler cannot execute
    # while PyDLL holds the GIL; this sample does not reveal native-call frames.
    time.sleep(0.2)
    blocking_sleep = ctypes.PyDLL(None).sleep
    blocking_sleep.argtypes = [ctypes.c_uint]
    blocking_sleep.restype = ctypes.c_uint
    started = time.monotonic()
    blocking_sleep(1)
    return {
        "native_call_elapsed_seconds": time.monotonic() - started,
        "native_inside_call_line_guaranteed": False,
        "sampling_gap_expected": True,
        "native_observation": "Python wrapper sampled before GIL hold; inside-call Python sampling unavailable",
    }


def fixture_raised_exception() -> None:
    time.sleep(1.0)
    raise RuntimeError("intentional diagnostic exception")


FUNCTIONS = {
    "cpu_loop": fixture_cpu_loop,
    "wait": fixture_wait,
    "native_gil": fixture_native_gil,
    "raised_exception": fixture_raised_exception,
}


def child(case: str, workspace: str, ready, release) -> None:
    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import (
        _periodic_stack_dumps,
    )

    directory = Path(workspace) / case
    log_path = directory / "worker.log"
    stack_path = directory / "worker.stacks.txt"
    record = {
        "case": case,
        "pid": os.getpid(),
        "attempt_id": f"parallel-trace:{case}:1",
        "log_path": str(log_path),
        "stack_trace_path": str(stack_path),
        "expected_function": FUNCTIONS[case].__name__,
        "purpose": PURPOSE,
    }
    returncode = 0
    diagnostics = {}
    with log_path.open("w", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(f"ready case={case} pid={os.getpid()}", flush=True)
            ready.put(case)
            if not release.wait(timeout=3):
                raise RuntimeError("parent did not release synchronized fixture")
            record["started_at"] = datetime.now(timezone.utc).isoformat()
            record["started_monotonic"] = time.monotonic()
            try:
                with _periodic_stack_dumps(
                    stack_path, interval_seconds=STACK_INTERVAL_SECONDS
                ) as diagnostics:
                    observations = FUNCTIONS[case]()
                    if observations:
                        record.update(observations)
            except BaseException:
                returncode = 3
                traceback.print_exc()
            finally:
                record["finished_monotonic"] = time.monotonic()
                record["finished_at"] = datetime.now(timezone.utc).isoformat()
                record["elapsed_seconds"] = (
                    record["finished_monotonic"] - record["started_monotonic"]
                )
                record["stack_diagnostics"] = diagnostics
                record["returncode"] = returncode
                print(f"finished case={case} returncode={returncode}", flush=True)
                write_json(directory / "metadata.json", record)
    raise SystemExit(returncode)


def validate_workspace(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise ValueError("workspace must be a new absolute non-symlink path")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("workspace parent must be an existing directory")
    workspace = parent / path.name
    if (
        ROOT / ".repo_cache" in workspace.parents
        or ROOT / "fedotllm" in workspace.parents
    ):
        raise ValueError("workspace must be separate from source checkouts")
    return workspace


def launch(workspace: Path, arguments: list[str]) -> dict:
    script = Path(__file__).resolve()
    worker_source = ROOT / "fedotllm/agents/evolve/evaluation/_fedot_quality_worker.py"
    report = {
        "mode": "launch",
        "purpose": PURPOSE,
        "passed": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script),
        "script_sha256": sha256(script),
        "worker_source_sha256": sha256(worker_source),
        "arguments": arguments,
        "affected_paths": [str(workspace)],
        "workspace": str(workspace),
        "stack_interval_seconds": STACK_INTERVAL_SECONDS,
        "workers": [],
    }
    workspace.mkdir(exist_ok=False)
    write_json(workspace / "launch_audit.json", report)
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    release = context.Event()
    children = []
    try:
        for case in CASES:
            (workspace / case).mkdir()
            process = context.Process(
                target=child, args=(case, str(workspace), ready, release), name=case
            )
            process.start()
            children.append((case, process))
        ready_cases = {ready.get(timeout=3) for _ in CASES}
        if ready_cases != set(CASES):
            raise RuntimeError(
                "not all four fixture processes reached the start barrier"
            )
        release.set()
        deadline = time.monotonic() + 4
        for case, process in children:
            process.join(timeout=max(0, deadline - time.monotonic()))
            if process.is_alive():
                raise RuntimeError(f"bounded fixture exceeded deadline: {case}")
            record = json.loads((workspace / case / "metadata.json").read_text())
            stacks = Path(record["stack_trace_path"]).read_text()
            expected = record["expected_function"]
            frame = rf'File "{re.escape(str(script))}", line [1-9][0-9]* in {re.escape(expected)}'
            record.update(
                stack_dump_count=stacks.count("--- stack dump at "),
                expected_function_present=f"in {expected}" in stacks,
                source_line_present=bool(re.search(frame, stacks)),
                actual_returncode=process.exitcode,
            )
            report["workers"].append(record)
        records = report["workers"]
        report["overlap_seconds"] = min(
            row["finished_monotonic"] for row in records
        ) - max(row["started_monotonic"] for row in records)
        report["passed"] = report["overlap_seconds"] > 0.1 and all(
            row["stack_dump_count"] >= 1
            and row["expected_function_present"]
            and row["source_line_present"]
            and row["actual_returncode"]
            == row["returncode"]
            == (3 if row["case"] == "raised_exception" else 0)
            and (
                row["case"] != "native_gil" or row["native_call_elapsed_seconds"] >= 1.0
            )
            and row["elapsed_seconds"] < 10
            for row in records
        )
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for _, process in children:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        ready.close()
        ready.join_thread()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(workspace / "launch_audit.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "launch"), default="check")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        workspace = validate_workspace(args.workspace)
        if args.mode == "check":
            print(
                json.dumps(
                    {
                        "mode": "check",
                        "purpose": PURPOSE,
                        "workspace": str(workspace),
                        "script_sha256": sha256(Path(__file__).resolve()),
                        "cases": list(CASES),
                        "processes_started": False,
                    }
                )
            )
            return 0
        report = launch(workspace, list(sys.argv[1:] if argv is None else argv))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

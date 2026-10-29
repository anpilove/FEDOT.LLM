"""Validate or launch one fixed quality service release in a new workspace.

No SSH, process discovery, source modification, cleanup or arbitrary commands.
The default check mode reads files only; launch mode creates explicitly listed
    controller artifacts and starts the fixed controller CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import signal
import sys
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True


@contextmanager
def previous_run_guard(run: Path, *, allow_stale: bool = False,
                       stale_after_seconds: float = 900):
    """Require a terminal run or explicitly acknowledge a stale, lock-free run."""
    import fcntl
    from fedotllm.agents.evolve.execution.storage_guard import terminal_service_record
    if not run.is_absolute() or run.is_symlink() or not run.is_dir():
        raise ValueError("previous-run must be an existing absolute run directory")
    with ExitStack() as stack:
        for name in (".quality_service.lock", ".quality_queue.lock"):
            path = run / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing previous run lock: {path}")
            lock = stack.enter_context(path.open("r"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"previous run is active: {run}") from exc
        state = json.loads((run / "logs/service_status.json").read_text())
        terminal_states = {"artifact_error", "storage_error", "completed", "duration_exhausted",
                           "restart_exhausted", "hunt_budget_exhausted", "hunt_campaigns_exhausted"}
        if state.get("state") not in terminal_states or state.get("child_running") is not False:
            fallback = terminal_service_record(run / "logs/service-launch.log")
            if (fallback is not None and fallback.get("state") in terminal_states
                    and fallback.get("child_running") is False and fallback.get("active_child") is None):
                state = fallback
        terminal = (state.get("state") in terminal_states and state.get("child_running") is False
                    and state.get("active_child") is None)
        if terminal:
            yield {"path":str(run), "state":state["state"],
                   "locks":"both acquired nonblocking; no active owner"}
            return
        heartbeat = state.get("heartbeat_at")
        try:
            parsed = datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            age = None
        stale = (
            allow_stale
            and state.get("state") in {"running", "starting"}
            and state.get("child_running") is True
            and isinstance(state.get("active_child"), dict)
            and age is not None and age > stale_after_seconds
        )
        if not stale:
            raise ValueError(f"previous run has no terminal lifecycle evidence: {run}")
        yield {
            "path": str(run),
            "state": "stale_unconfirmed",
            "saved_state": state.get("state"),
            "locks": "both acquired nonblocking; no active owner at check time",
            "heartbeat_at": heartbeat,
            "heartbeat_age_seconds": age,
            "active_child_log": state.get("active_child", {}).get("log"),
            "terminal_outcome": "unknown; previous run artifacts preserved",
        }


def _read_proposal_seed(directory, source, source_hash, parser):
    from fedotllm.agents.evolve.controller.quality_executor import _parse_unified_hunks
    from fedotllm.agents.evolve.types import PatchCandidate
    from fedotllm.agents.evolve.execution.patch import apply_patch

    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        parser.error("proposals-from must be an existing absolute queue directory")
    proposals = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("._"):
            continue
        if path.is_symlink():
            parser.error(f"unexpected proposal symlink: {path}")
        record_bytes = path.read_bytes()
        record = json.loads(record_bytes)
        if not isinstance(record, dict):
            parser.error(f"invalid proposal: {path}")
        if record.get("status", "queued") not in {"queued", "running", "partial"}:
            continue
        identity = str(record.get("candidate_id") or path.stem)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identity):
            parser.error(f"invalid proposal id: {path}")
        if record.get("source_sha256") not in {None, "", source_hash}:
            parser.error(f"proposal source mismatch: {path}")
        patch_path = directory / Path(str(record.get("patch") or path.with_suffix(".patch").name)).name
        if patch_path.is_symlink() or not patch_path.is_file():
            parser.error(f"missing or redirected proposal patch: {patch_path}")
        patch = patch_path.read_text(encoding="utf-8")
        patch_hash = hashlib.sha256(patch.encode()).hexdigest()
        if record.get("patch_sha256") not in {None, "", patch_hash}:
            parser.error(f"patch hash mismatch: {patch_path}")
        if any(line.startswith("@@") and not re.fullmatch(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*", line)
               for line in patch.splitlines()):
            parser.error(f"invalid FEDOT proposal patch: malformed hunk: {patch_path}")
        edits = _parse_unified_hunks(patch)
        if not edits or any(not edit.file_path.startswith("fedot/") or ".." in Path(edit.file_path).parts for edit in edits):
            parser.error(f"invalid FEDOT proposal patch: {patch_path}")
        hypothesis = record.get("hypothesis") or {}
        if not isinstance(hypothesis, dict):
            parser.error(f"invalid proposal hypothesis: {path}")
        proposal = PatchCandidate(identity, str(record.get("file_path") or edits[0].file_path), edits=edits)
        diagnostics = []
        try:
            applicable = apply_patch(source, proposal, diagnostics=diagnostics, dry_run=True)
        except PermissionError as exc:
            parser.error(f"invalid FEDOT proposal patch: {patch_path}: {exc}")
        if not applicable:
            parser.error(f"invalid FEDOT proposal patch: {patch_path}: {diagnostics}")
        proposals.append((proposal, patch, record, {"proposal_path":str(path),
                          "proposal_sha256":hashlib.sha256(record_bytes).hexdigest(),
                          "patch_path":str(patch_path), "patch_sha256":patch_hash}))
    if not proposals:
        parser.error("proposals-from has no pending proposed improvements")
    if len({row[0].candidate_id for row in proposals}) != len(proposals):
        parser.error("duplicate proposal ids in proposals-from")
    return proposals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "launch"), default="check")
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--fedot-source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--duration-seconds", type=float, default=604800)
    parser.add_argument("--target-proposals", type=int, default=100)
    parser.add_argument("--hunt", action="store_true")
    parser.add_argument("--llm-max-cost-usd", type=float, default=0.25)
    parser.add_argument("--hunt-total-cost-usd", type=float, default=10)
    parser.add_argument("--presets", default="fedotllm:openrouter")
    parser.add_argument("--data-cache", type=Path, default=None)
    parser.add_argument("--proposals-from", type=Path, default=None)
    parser.add_argument("--previous-run", type=Path, default=None)
    parser.add_argument("--allow-stale-previous-run", action="store_true",
                        help="allow a new isolated run only when the previous run is stale and both locks are free")
    args = parser.parse_args(argv)
    for name in ("controller_root", "fedot_source", "python", "workspace"):
        if not getattr(args, name).is_absolute():
            parser.error(f"{name} must be an absolute path")
    # Full-library installs expose fedotllm -> current/fedotllm. Pin the
    # physical release so an update of current cannot change this launch.
    controller = (args.controller_root / "fedotllm").resolve().parent
    source = args.fedot_source.resolve()
    workspace = args.workspace.resolve()
    # Keep the venv executable's symlink path so Python finds pyvenv.cfg.
    python = str(args.python.absolute())
    if not (controller / "fedotllm/agents/evolve/controller").is_dir():
        parser.error("controller release is missing")
    if not (source / "fedot").is_dir() or not args.python.is_file():
        parser.error("FEDOT source or selected Python is missing")
    if workspace.exists() or args.workspace.is_symlink():
        parser.error(f"workspace already exists; preserving old run: {workspace}")
    if not workspace.parent.is_dir():
        parser.error(f"workspace parent does not exist: {workspace.parent}")
    from fedotllm.agents.evolve.execution.storage_guard import require_storage, LAUNCH_RESERVE_BYTES
    storage = require_storage(workspace.parent, minimum_bytes=LAUNCH_RESERVE_BYTES)
    if (workspace == controller or workspace in controller.parents or controller in workspace.parents
            or workspace == source or workspace in source.parents or source in workspace.parents):
        parser.error("run artifacts must be separate from FEDOT source and controller release")
    sys.path.insert(0, str(controller))
    from fedotllm.agents.evolve.controller.release import controller_version
    from fedotllm.agents.evolve.controller.supervisor import QualityServiceConfig
    from fedotllm.agents.evolve.execution.checkout import source_fingerprint
    from fedotllm.agents.evolve.execution.process import clean_subprocess_env
    from fedotllm.agents.evolve.storage.journal import write_json_atomic
    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job

    version = controller_version(controller)
    if version != args.expected_controller_sha256:
        parser.error(f"controller hash mismatch: expected {args.expected_controller_sha256}, got {version}")
    config = QualityServiceConfig(duration_seconds=args.duration_seconds,
                                  target_proposals=args.target_proposals, hunt=args.hunt,
                                  llm_max_cost_usd=args.llm_max_cost_usd,
                                  hunt_total_cost_usd=args.hunt_total_cost_usd, presets=args.presets)
    config.validate()
    source_hash = source_fingerprint(source)
    initial_proposals = (_read_proposal_seed(args.proposals_from, source, source_hash, parser)
                         if args.proposals_from is not None else [])
    command = [python, "-m", "fedotllm.agents.evolve.controller", "serve",
               "--fedot", str(source), "--workspace", str(workspace),
               "--stock-cache", str(workspace / "stock_cache"),
               "--duration-seconds", str(config.duration_seconds),
               "--target-proposals", str(config.target_proposals),
               "--llm-max-cost-usd", str(config.llm_max_cost_usd),
               "--hunt-total-cost-usd", str(config.hunt_total_cost_usd),
               "--presets", config.presets]
    if args.hunt:
        command.append("--hunt")
    env = clean_subprocess_env(source, repo_root=controller, extra={
        "FEDOTLLM_REPO_PATH": str(source), "FEDOTLLM_REPO_PYTHON": python,
        "EVOLVE_AGENT_TMP": str(workspace / "worker_artifacts"),
        "PYTHONUNBUFFERED": "1",
    })
    if args.data_cache is not None:
        if not args.data_cache.is_absolute() or not args.data_cache.is_dir():
            parser.error("data-cache must be an existing absolute directory")
        env["EVOLVE_QUALITY_DATA_CACHE"] = str(args.data_cache.resolve())
    if args.hunt:
        if not os.environ.get("FEDOTLLM_LLM_API_KEY"):
            parser.error("hunt requires FEDOTLLM_LLM_API_KEY in the interactive environment")
        env["FEDOTLLM_LLM_API_KEY"] = os.environ["FEDOTLLM_LLM_API_KEY"]
    stamp = datetime.now(timezone.utc).isoformat()
    audit = {
        "mode": args.mode, "time": stamp, "command": command,
        "controller_sha256": version,
        "controller_root": str(controller),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fedot_source_sha256": source_hash,
        "initial_proposal_count":len(initial_proposals),
        "initial_proposals":[row[3] for row in initial_proposals],
        "profile": {"n_jobs": 4, "max_parallel_pairs": 7,
                    "side_timeout_seconds": 5400, "fedot_search_budget_seconds": 4500,
                    "watchdog_seconds": 10800},
        "affected_paths": [str(workspace)],
        "storage_preflight":storage,
        "result": "checked; no process launched",
    }
    with ExitStack() as guards:
        if args.previous_run is not None:
            if args.previous_run.parent.resolve() != workspace.parent:
                parser.error("previous-run must belong to the same run directory")
            audit["previous_run"] = guards.enter_context(previous_run_guard(
                args.previous_run, allow_stale=args.allow_stale_previous_run,
            ))
        elif args.allow_stale_previous_run:
            parser.error("allow-stale-previous-run requires previous-run")
        if args.mode == "launch":
            # mkdir(exist_ok=False) also closes the check/create race.
            workspace.mkdir()
            logs = workspace / "logs"
            logs.mkdir()
            audit_path = workspace / "launch_audit.json"
            audit["result"] = "launch reserved"
            write_json_atomic(audit_path, audit)
            try:
                for proposal, patch, record, _origin in initial_proposals:
                    enqueue_quality_job(
                        workspace, candidate=proposal, patch_text=patch,
                        hint=str(record.get("hint") or record.get("title") or proposal.candidate_id),
                        priority=str(record.get("priority") or "normal"),
                        hypothesis=record.get("hypothesis") or {}, source_root=source,
                    )
                audit["result"] = "queue seeded; launching service"
                write_json_atomic(audit_path, audit)
                with (logs / "service-launch.log").open("x") as output:
                    proc = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                audit.update(result="spawned; progress must be checked in service_status.json", pid=proc.pid)
            except (OSError, ValueError) as exc:
                audit.update(result="launch failed", error=f"{type(exc).__name__}: {exc}")
                write_json_atomic(audit_path, audit)
                raise
            try:
                write_json_atomic(audit_path, audit)
            except OSError as original:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                    if hasattr(original, "add_note"):
                        original.add_note(f"owned service launch cleanup failed: {cleanup_error}")
                raise
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from research.evolve.metric_agent.guard import repo_root
from research.evolve.metric_agent.tasks import load_task
from research.evolve.metric_agent.types import ScoreResult

WORKER = Path(__file__).resolve().parent / "_worker.py"


def run_stock(task_id: str, *, checkout: Path, timeout_s: float | None = None) -> ScoreResult:
    return _run(task_id, checkout=checkout, timeout_s=timeout_s)


def run_patched(task_id: str, *, checkout: Path, timeout_s: float | None = None) -> ScoreResult:
    return _run(task_id, checkout=checkout, timeout_s=timeout_s)


def _run(task_id: str, *, checkout: Path, timeout_s: float | None) -> ScoreResult:
    spec = load_task(task_id)
    limit = timeout_s if timeout_s is not None else spec.timeout_s
    safe = task_id.replace("/", "_").replace(">", "_")
    out = Path(os.environ.get("METRIC_AGENT_TMP", "/tmp/metric-agent")) / f"{safe}-{uuid.uuid4().hex[:8]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    root = str(repo_root())
    fedot = str(checkout.resolve())
    env["PYTHONPATH"] = os.pathsep.join([fedot, root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, str(WORKER), "--task", task_id, "--out", str(out)]
    cmd_s = " ".join(cmd)
    env_hash = _env_hash(checkout)
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=limit,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ScoreResult(
            task_id=task_id,
            status="timeout",
            score=float("nan"),
            traceback="",
            detail=f"timeout after {limit}s",
            duration_s=float(limit),
            env_hash=env_hash,
            cmd=cmd_s,
        )
    log_tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:]
    if not out.exists():
        return ScoreResult(
            task_id=task_id,
            status="invalid",
            score=float("nan"),
            traceback=(proc.stderr or "")[-4000:],
            detail=f"worker exit {proc.returncode}",
            duration_s=0.0,
            env_hash=env_hash,
            cmd=cmd_s,
            log_tail=log_tail,
        )
    payload = json.loads(out.read_text(encoding="utf-8"))
    return ScoreResult(
        task_id=payload["task_id"],
        status=payload["status"],
        score=float(payload["score"]),
        traceback=payload.get("traceback") or "",
        detail=payload.get("detail") or "",
        duration_s=float(payload.get("duration_s") or 0.0),
        n_train=int(payload.get("n_train") or 0),
        env_hash=env_hash,
        cmd=cmd_s,
        log_tail=log_tail,
    )


def _env_hash(checkout: Path) -> str:
    marker = checkout / "fedot" / "__init__.py"
    raw = marker.read_bytes() if marker.is_file() else str(checkout).encode()
    return hashlib.sha256(raw).hexdigest()[:16]

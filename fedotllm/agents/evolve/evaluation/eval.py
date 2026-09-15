from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.execution.checkout import source_commit, source_fingerprint
from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.execution.process import (
    clean_subprocess_env,
    fedot_python,
    interpreter_identity,
    thread_environment,
)
from fedotllm.agents.evolve.protocol import score_protocol_fingerprint
from fedotllm.agents.evolve.evaluation.tasks import load_task
from fedotllm.agents.evolve.types import ScoreResult

WORKER_MODULE = "fedotllm.agents.evolve.evaluation._worker"


def run_stock(
    task_id: str,
    *,
    checkout: Path,
    timeout_s: float | None = None,
    split: str = "dev",
    seed: int | None = None,
    collect_coverage: bool = False,
    task_override: dict | None = None,
) -> ScoreResult:
    seed_value = int(seed if seed is not None else os.environ.get("EVOLVE_AGENT_SEED", "42"))
    fingerprint = _env_hash(
        checkout, task_id, split=split, task_override=task_override
    )
    cache_root = Path(
        os.environ.get("EVOLVE_AGENT_BASELINE_CACHE", "/tmp/evolve-agent-baseline-cache")
    )
    cache_key = hashlib.sha256(
        f"{fingerprint}|{task_id}|{split}|{seed_value}|coverage={int(collect_coverage)}".encode()
    ).hexdigest()
    cache_file = cache_root / f"{cache_key}.json"
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("env_hash") == fingerprint:
            payload["coverage"] = tuple(payload.get("coverage") or ())
            payload["dataflow"] = tuple(payload.get("dataflow") or ())
            return ScoreResult(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    result = _run(
        task_id,
        checkout=checkout,
        timeout_s=timeout_s,
        split=split,
        seed=seed_value,
        collect_coverage=collect_coverage,
        task_override=task_override,
    )
    if result.status in {"ok", "crash"}:
        cache_root.mkdir(parents=True, exist_ok=True)
        temp = cache_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        temp.write_text(json.dumps(asdict(result), default=str), encoding="utf-8")
        temp.replace(cache_file)
    return result


def run_patched(
    task_id: str,
    *,
    checkout: Path,
    timeout_s: float | None = None,
    split: str = "dev",
    seed: int | None = None,
    collect_coverage: bool = False,
    task_override: dict | None = None,
) -> ScoreResult:
    seed_value = int(seed if seed is not None else os.environ.get("EVOLVE_AGENT_SEED", "42"))
    fingerprint = _env_hash(
        checkout, task_id, split=split, task_override=task_override
    )
    cache_root = Path(
        os.environ.get("EVOLVE_AGENT_SCORE_CACHE", "/tmp/evolve-agent-score-cache")
    )
    cache_key = hashlib.sha256(
        f"{fingerprint}|{task_id}|{split}|{seed_value}|coverage={int(collect_coverage)}".encode()
    ).hexdigest()
    cache_file = cache_root / f"{cache_key}.json"
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("env_hash") == fingerprint:
            payload["coverage"] = tuple(payload.get("coverage") or ())
            payload["dataflow"] = tuple(payload.get("dataflow") or ())
            return ScoreResult(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    result = _run(
        task_id,
        checkout=checkout,
        timeout_s=timeout_s,
        split=split,
        seed=seed_value,
        collect_coverage=collect_coverage,
        task_override=task_override,
    )
    if result.status in {"ok", "crash"}:
        cache_root.mkdir(parents=True, exist_ok=True)
        temp = cache_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        temp.write_text(json.dumps(asdict(result), default=str), encoding="utf-8")
        temp.replace(cache_file)
    return result


def _run(
    task_id: str,
    *,
    checkout: Path,
    timeout_s: float | None,
    split: str = "dev",
    seed: int | None = None,
    collect_coverage: bool = False,
    task_override: dict | None = None,
) -> ScoreResult:
    spec = load_task(task_id)
    limit = timeout_s if timeout_s is not None else spec.timeout_s
    safe = task_id.replace("/", "_").replace(">", "_")
    out = Path(os.environ.get("EVOLVE_AGENT_TMP", "/tmp/evolve-agent")) / f"{safe}-{uuid.uuid4().hex[:8]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    root = str(repo_root())
    env = clean_subprocess_env(checkout, repo_root=repo_root())
    python = fedot_python(checkout)
    seed = int(seed if seed is not None else os.environ.get("EVOLVE_AGENT_SEED", "42"))
    cmd = [
        python,
        "-m",
        WORKER_MODULE,
        "--task",
        task_id,
        "--out",
        str(out),
        "--seed",
        str(seed),
        "--split",
        split,
        "--checkout",
        str(checkout.resolve()),
    ]
    if collect_coverage:
        cmd.append("--coverage")
    if task_override:
        cmd.extend(
            [
                "--task-override-json",
                json.dumps(task_override, sort_keys=True, separators=(",", ":")),
            ]
        )
    cmd_s = " ".join(cmd)
    env_hash = _env_hash(
        checkout, task_id, split=split, task_override=task_override
    )
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
        out.unlink(missing_ok=True)
        return ScoreResult(
            task_id=task_id,
            status="timeout",
            score=float("nan"),
            traceback="",
            detail=f"timeout after {limit}s",
            duration_s=float(limit),
            env_hash=env_hash,
            cmd=cmd_s,
            seed=seed,
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
            seed=seed,
        )
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
        return ScoreResult(
            task_id=str(payload["task_id"]),
            status=payload["status"],
            score=float(payload["score"]),
            traceback=payload.get("traceback") or "",
            detail=payload.get("detail") or "",
            duration_s=float(payload.get("duration_s") or 0.0),
            n_train=int(payload.get("n_train") or 0),
            env_hash=env_hash,
            cmd=cmd_s,
            log_tail=log_tail,
            seed=int(payload.get("seed") or seed),
            coverage=tuple(payload.get("coverage") or ()),
            dataflow=tuple(payload.get("dataflow") or ()),
            data_evidence=payload.get("data_evidence") or {},
            metric_observations=payload.get("metric_observations") or {},
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return ScoreResult(
            task_id=task_id,
            status="invalid",
            score=float("nan"),
            traceback="",
            detail=f"malformed worker output: {type(exc).__name__}: {exc}",
            duration_s=0.0,
            env_hash=env_hash,
            cmd=cmd_s,
            log_tail=log_tail,
            seed=seed,
        )
    finally:
        out.unlink(missing_ok=True)


def _env_hash(
    checkout: Path,
    task_id: str | None = None,
    *,
    split: str = "dev",
    task_override: dict | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(source_fingerprint(checkout).encode())
    digest.update(source_commit(checkout).encode())
    digest.update(interpreter_identity(checkout).encode())
    digest.update(json.dumps(thread_environment(), sort_keys=True).encode())
    digest.update(split.encode())
    digest.update(score_protocol_fingerprint().encode())
    digest.update(
        json.dumps(task_override or {}, sort_keys=True, default=str).encode()
    )
    lock = repo_root() / "uv.lock"
    if lock.is_file():
        digest.update(lock.read_bytes())
    if task_id:
        spec = load_task(task_id)
        digest.update(json.dumps(asdict(spec), sort_keys=True, default=str).encode())
        data_root = checkout / "examples" / "real_cases" / "data"
        names = [spec.train_file, spec.test_file]
        if spec.dataset == "scoring":
            names.extend(("scoring/scoring_train.csv", "scoring/scoring_test.csv"))
        for name in sorted(set(item for item in names if item)):
            path = data_root / name
            digest.update(name.encode())
            digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()[:16]

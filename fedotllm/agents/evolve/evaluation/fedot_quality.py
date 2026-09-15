"""Paired Fedot(best_quality, timeout=60 min) jobs on the quality registry."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.evaluation.quality_registry import (
    QualityDataset,
    QualityRegistry,
    get_dataset,
    job_spec,
    list_quality_task_ids,
    load_quality_registry,
)
from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.execution.process import (
    clean_subprocess_env,
    fedot_python,
)
from fedotllm.agents.evolve.types import Decision, ScoreResult

WORKER_MODULE = "fedotllm.agents.evolve.evaluation._fedot_quality_worker"


def _stock_cache_root() -> Path:
    return Path(os.environ.get("EVOLVE_QUALITY_STOCK_CACHE", "/tmp/evolve-fedot-quality-stock"))


def _stock_cache_path(dataset: QualityDataset, registry: QualityRegistry, n_jobs: int) -> Path:
    name = f"{registry.identity}__{dataset.task_id}__n{n_jobs}__s{registry.seed}.json"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return _stock_cache_root() / safe


def _load_stock_cache(
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
    *,
    env_hash: str,
) -> ScoreResult | None:
    path = _stock_cache_path(dataset, registry, n_jobs)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    observations = payload.get("metric_observations") or {}
    if payload.get("status") != "ok" or not observations.get("search_ran"):
        return None
    return _score_from_payload(
        dataset.task_id,
        payload,
        env_hash=env_hash,
        cmd=str(payload.get("cmd") or "stock-cache"),
    )


def _store_stock_cache(
    result: ScoreResult,
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
) -> None:
    if result.status != "ok" or not (result.metric_observations or {}).get("search_ran"):
        return
    root = _stock_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": result.task_id,
        "status": result.status,
        "score": result.score,
        "traceback": result.traceback,
        "detail": result.detail,
        "duration_s": result.duration_s,
        "n_train": result.n_train,
        "seed": result.seed,
        "cmd": result.cmd,
        "log_tail": result.log_tail,
        "metric_observations": result.metric_observations,
    }
    _stock_cache_path(dataset, registry, n_jobs).write_text(
        json.dumps(payload, default=str),
        encoding="utf-8",
    )


def _score_from_payload(task_id: str, payload: dict, *, env_hash: str, cmd: str) -> ScoreResult:
    observations = payload.get("metric_observations") or {}
    status = str(payload.get("status") or "invalid")
    if observations.get("search_ran") is False:
        status = "invalid"
    return ScoreResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        score=float(payload.get("score", float("nan"))),
        traceback=str(payload.get("traceback") or ""),
        detail=str(payload.get("detail") or ""),
        duration_s=float(payload.get("duration_s") or 0.0),
        n_train=int(payload.get("n_train") or 0),
        env_hash=env_hash,
        cmd=cmd,
        log_tail=str(payload.get("log_tail") or ""),
        seed=int(payload.get("seed") or 42),
        metric_observations=dict(observations),
    )


def run_quality_side(
    dataset: QualityDataset | str,
    *,
    checkout: Path,
    side: str,
    n_jobs: int | None = None,
    registry: QualityRegistry | None = None,
) -> ScoreResult:
    registry = registry or load_quality_registry()
    if isinstance(dataset, str):
        dataset = get_dataset(dataset, registry)
    spec = job_spec(dataset, n_jobs=n_jobs, registry=registry)
    spec["side"] = side
    env_hash = f"{registry.identity}|{dataset.task_id}|{side}|{spec['n_jobs']}"
    if side == "stock":
        cached = _load_stock_cache(dataset, registry, spec["n_jobs"], env_hash=env_hash)
        if cached is not None:
            return cached
    tmp = Path(os.environ.get("EVOLVE_AGENT_TMP", "/tmp/evolve-agent"))
    tmp.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    spec_path = tmp / f"quality-spec-{dataset.task_id}-{side}-{token}.json"
    out_path = tmp / f"quality-out-{dataset.task_id}-{side}-{token}.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    python = fedot_python(checkout)
    cmd = [
        python,
        "-m",
        WORKER_MODULE,
        "--spec-json",
        str(spec_path),
        "--out",
        str(out_path),
    ]
    env = clean_subprocess_env(checkout, repo_root=repo_root())
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    wall = float(registry.timeout_seconds) + 300.0
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=wall,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ScoreResult(
            task_id=dataset.task_id,
            status="timeout",
            score=float("nan"),
            detail=f"timeout after {wall}s",
            duration_s=wall,
            env_hash=env_hash,
            cmd=" ".join(cmd),
            seed=registry.seed,
            metric_observations={"search_ran": False, "search_detail": "wall_timeout"},
        )
    if not out_path.is_file():
        return ScoreResult(
            task_id=dataset.task_id,
            status="invalid",
            score=float("nan"),
            traceback=(proc.stderr or "")[-4000:],
            detail=f"worker exit {proc.returncode}",
            env_hash=env_hash,
            cmd=" ".join(cmd),
            log_tail=((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:],
            seed=registry.seed,
            metric_observations={"search_ran": False},
        )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["log_tail"] = (
        str(payload.get("log_tail") or "")
        + "\n"
        + ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-2000:]
    )
    result = _score_from_payload(
        dataset.task_id, payload, env_hash=env_hash, cmd=" ".join(cmd)
    )
    if side == "stock":
        _store_stock_cache(result, dataset, registry, spec["n_jobs"])
    return result


def run_quality_pair(
    dataset: QualityDataset | str,
    *,
    stock_checkout: Path,
    patch_checkout: Path,
    n_jobs: int | None = None,
    registry: QualityRegistry | None = None,
) -> dict:
    """One indivisible stock-then-patch job on the same host."""

    registry = registry or load_quality_registry()
    if isinstance(dataset, str):
        dataset = get_dataset(dataset, registry)
    spec = job_spec(dataset, n_jobs=n_jobs, registry=registry)
    stock = run_quality_side(
        dataset, checkout=stock_checkout, side="stock", n_jobs=n_jobs, registry=registry
    )
    patched = run_quality_side(
        dataset, checkout=patch_checkout, side="patch", n_jobs=n_jobs, registry=registry
    )
    return {
        "spec": spec,
        "stock": asdict(stock),
        "patched": asdict(patched),
        "stock_result": stock,
        "patched_result": patched,
        "search_ran": bool(
            (stock.metric_observations or {}).get("search_ran")
            and (patched.metric_observations or {}).get("search_ran")
        ),
    }


def run_quality_stock_jobs(
    *,
    stock_checkout: Path,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    registry: QualityRegistry | None = None,
) -> list[dict]:
    """Stock-only baselines on the pre-registered set (cache for later pairs)."""

    registry = registry or load_quality_registry()
    ids = task_ids or list_quality_task_ids(registry)
    jobs = int(n_jobs if n_jobs is not None else registry.n_jobs_per_job)
    quota = int(cpu_quota if cpu_quota is not None else registry.cpu_quota)
    parallel = max(1, min(len(ids), quota // max(1, jobs)))

    def one(task_id: str) -> dict:
        dataset = get_dataset(task_id, registry)
        spec = job_spec(dataset, n_jobs=jobs, registry=registry)
        stock = run_quality_side(
            dataset,
            checkout=stock_checkout,
            side="stock",
            n_jobs=jobs,
            registry=registry,
        )
        return {
            "spec": spec,
            "stock": asdict(stock),
            "search_ran": bool((stock.metric_observations or {}).get("search_ran")),
        }

    if parallel <= 1 or len(ids) == 1:
        return [one(task_id) for task_id in ids]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(one, task_id): task_id for task_id in ids}
        by_id = {futures[future]: future.result() for future in as_completed(futures)}
        return [by_id[task_id] for task_id in ids]


def run_quality_jobs(
    *,
    stock_checkout: Path,
    patch_checkout: Path,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    registry: QualityRegistry | None = None,
) -> list[dict]:
    """Run 1–N registered pairs. Parallelism is datasets × n_jobs under cpu_quota."""

    registry = registry or load_quality_registry()
    ids = task_ids or list_quality_task_ids(registry)
    jobs = int(n_jobs if n_jobs is not None else registry.n_jobs_per_job)
    quota = int(cpu_quota if cpu_quota is not None else registry.cpu_quota)
    parallel = max(1, min(len(ids), quota // max(1, jobs)))
    rows: list[dict] = []
    if parallel <= 1 or len(ids) == 1:
        return [
            run_quality_pair(
                task_id,
                stock_checkout=stock_checkout,
                patch_checkout=patch_checkout,
                n_jobs=jobs,
                registry=registry,
            )
            for task_id in ids
        ]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                run_quality_pair,
                task_id,
                stock_checkout=stock_checkout,
                patch_checkout=patch_checkout,
                n_jobs=jobs,
                registry=registry,
            ): task_id
            for task_id in ids
        }
        by_id = {futures[future]: future.result() for future in as_completed(futures)}
        rows = [by_id[task_id] for task_id in ids]
    return rows


def decision_from_quality_jobs(rows: list[dict]) -> Decision:
    if not rows:
        return Decision(
            keep=False,
            reason="empty_quality_jobs",
            target_delta=None,
            stage="quality",
            infrastructure_error=True,
        )
    stock = {row["spec"]["source_dataset"]: row["stock_result"] for row in rows}
    patched = {row["spec"]["source_dataset"]: row["patched_result"] for row in rows}
    for task_id, result in (*stock.items(), *patched.items()):
        if result.status in {"timeout", "crash"}:
            return Decision(
                keep=False,
                reason=f"quality {task_id} {result.status}",
                target_delta=None,
                stage="quality",
                infrastructure_error=result.status == "timeout",
            )
        if not (result.metric_observations or {}).get("search_ran"):
            return Decision(
                keep=False,
                reason=f"composing_did_not_start:{task_id}",
                target_delta=None,
                stage="quality",
                infrastructure_error=True,
            )
        if result.status != "ok":
            return Decision(
                keep=False,
                reason=f"quality {task_id} {result.status}",
                target_delta=None,
                stage="quality",
                infrastructure_error=True,
            )
    deltas: dict[str, float | None] = {}
    improved = 0
    for row in rows:
        spec = row["spec"]
        task_id = spec["source_dataset"]
        before = float(stock[task_id].score)
        after = float(patched[task_id].score)
        delta = (after - before) if spec["higher_is_better"] else (before - after)
        deltas[task_id] = delta
        if delta < -float(spec["min_delta"]):
            return Decision(
                keep=False,
                reason=f"regression {task_id} delta {delta:.4f}",
                target_delta=delta,
                regression_deltas=deltas,
                stage="quality",
            )
        if delta >= float(spec["min_delta"]):
            improved += 1
    mean_delta = sum(value or 0.0 for value in deltas.values()) / len(deltas)
    keep = improved > 0
    return Decision(
        keep=keep,
        reason="quality_improved" if keep else "no_quality_gain",
        target_delta=mean_delta,
        regression_deltas=deltas,
        stage="quality",
        final_keep=keep,
    )

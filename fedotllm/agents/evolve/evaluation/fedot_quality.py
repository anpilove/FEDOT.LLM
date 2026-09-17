"""Paired Fedot(best_quality, timeout=60 min) jobs on the quality registry."""

from __future__ import annotations

import json
import os
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
    run_worker,
)
from fedotllm.agents.evolve.storage.journal import write_json_atomic
from fedotllm.agents.evolve.types import Decision, ScoreResult

WORKER_MODULE = "fedotllm.agents.evolve.evaluation._fedot_quality_worker"
STOCK_CACHE_ENV = "EVOLVE_QUALITY_STOCK_CACHE"
DEFAULT_STOCK_CACHE = Path("/tmp/evolve-fedot-quality-stock")


def bind_stock_cache(path: Path | str | None) -> Path:
    """Pin local and cluster jobs to the same cache directory."""

    if path is None:
        return _stock_cache_root()
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ[STOCK_CACHE_ENV] = str(resolved)
    return resolved


def _stock_cache_root() -> Path:
    return Path(os.environ.get(STOCK_CACHE_ENV, str(DEFAULT_STOCK_CACHE)))


def _safe_cache_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def stock_cache_identity(
    dataset: QualityDataset,
    registry: QualityRegistry,
    *,
    fedot_commit: str = "",
) -> dict:
    """Skip-warm key: dataset, official fold, seed, timeout, preset, FEDOT commit."""

    return {
        "task_id": dataset.task_id,
        "fold": int(dataset.fold),
        "repeat": int(dataset.repeat),
        "seed": int(registry.seed),
        "timeout_seconds": int(registry.timeout_seconds),
        "preset": str(registry.preset),
        "registry_identity": str(registry.identity),
        "fedot_commit": str(fedot_commit or ""),
    }


def _fedot_commit(checkout: Path | None) -> str:
    if checkout is None:
        return ""
    try:
        from fedotllm.agents.evolve.execution.checkout import source_commit

        return source_commit(checkout) or ""
    except OSError:
        return ""


def _stock_cache_path(dataset: QualityDataset, registry: QualityRegistry, n_jobs: int) -> Path:
    name = f"{registry.identity}__{dataset.task_id}__n{n_jobs}__s{registry.seed}.json"
    return _stock_cache_root() / _safe_cache_name(name)


def _iter_stock_cache_paths(
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
) -> list[Path]:
    """Exact n_jobs first, then leftover files with the same identity and another n_jobs."""

    root = _stock_cache_root()
    exact = _stock_cache_path(dataset, registry, n_jobs)
    paths: list[Path] = []
    if exact.is_file():
        paths.append(exact)
    prefix = _safe_cache_name(f"{registry.identity}__{dataset.task_id}__")
    if root.is_dir():
        for path in sorted(root.glob(f"{prefix}n*__s{registry.seed}.json")):
            if path not in paths and path.is_file():
                paths.append(path)
    return paths


def _field_matches(payload: dict, key: str, expected) -> bool:
    if key not in payload or payload[key] in (None, ""):
        return True
    value = payload[key]
    if isinstance(expected, int):
        try:
            return int(value) == expected
        except (TypeError, ValueError):
            return False
    return value == expected


def _identity_matches(payload: dict, expected: dict) -> bool:
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    blob = {**payload, **identity}
    observations = payload.get("metric_observations") or {}
    if not _field_matches(blob, "task_id", expected["task_id"]):
        return False
    if not _field_matches(blob, "fold", expected["fold"]):
        return False
    if not _field_matches(blob, "repeat", expected["repeat"]):
        return False
    if not _field_matches(blob, "seed", expected["seed"]):
        return False
    if not _field_matches(blob, "timeout_seconds", expected["timeout_seconds"]):
        return False
    if not _field_matches(observations, "timeout_seconds", expected["timeout_seconds"]):
        return False
    if not _field_matches(blob, "preset", expected["preset"]):
        return False
    if not _field_matches(observations, "preset", expected["preset"]):
        return False
    if not _field_matches(blob, "registry_identity", expected["registry_identity"]):
        return False
    stored_commit = str(blob.get("fedot_commit") or "")
    expected_commit = str(expected.get("fedot_commit") or "")
    if stored_commit and expected_commit and stored_commit != expected_commit:
        return False
    return True


def _load_stock_cache(
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
    *,
    env_hash: str,
    fedot_commit: str = "",
) -> ScoreResult | None:
    expected = stock_cache_identity(dataset, registry, fedot_commit=fedot_commit)
    for path in _iter_stock_cache_paths(dataset, registry, n_jobs):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        observations = payload.get("metric_observations") or {}
        if payload.get("status") != "ok" or not observations.get("search_ran"):
            continue
        if not _identity_matches(payload, expected):
            continue
        result = _score_from_payload(
            dataset.task_id,
            payload,
            env_hash=env_hash,
            cmd=str(payload.get("cmd") or "stock-cache"),
        )
        observations = dict(result.metric_observations or {})
        observations["cache_hit"] = True
        observations["cache_path"] = str(path)
        result.metric_observations = observations
        return result
    return None


def _store_stock_cache(
    result: ScoreResult,
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
    *,
    fedot_commit: str = "",
) -> None:
    if result.status != "ok" or not (result.metric_observations or {}).get("search_ran"):
        return
    root = _stock_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    identity = stock_cache_identity(dataset, registry, fedot_commit=fedot_commit)
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
        "identity": identity,
        "fold": identity["fold"],
        "repeat": identity["repeat"],
        "timeout_seconds": identity["timeout_seconds"],
        "preset": identity["preset"],
        "registry_identity": identity["registry_identity"],
        "fedot_commit": identity["fedot_commit"],
    }
    write_json_atomic(_stock_cache_path(dataset, registry, n_jobs), payload)


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
    fedot_commit = _fedot_commit(checkout)
    cache_path = (
        _stock_cache_path(dataset, registry, spec["n_jobs"]) if side == "stock" else None
    )
    if side == "stock":
        cached = _load_stock_cache(
            dataset,
            registry,
            spec["n_jobs"],
            env_hash=env_hash,
            fedot_commit=fedot_commit,
        )
        if cached is not None:
            hit_path = (cached.metric_observations or {}).get("cache_path") or cache_path
            print(f"quality stock cache-hit {dataset.task_id} {hit_path}", flush=True)
            return cached
    tmp = Path(os.environ.get("EVOLVE_AGENT_TMP", "/tmp/evolve-agent"))
    tmp.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    spec_path = tmp / f"quality-spec-{dataset.task_id}-{side}-{token}.json"
    out_path = tmp / f"quality-out-{dataset.task_id}-{side}-{token}.json"
    log_path = tmp / f"quality-log-{dataset.task_id}-{side}-{token}.log"
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
    extra = {
        key: os.environ[key]
        for key in (
            "EVOLVE_QUALITY_STOCK_CACHE",
            "EVOLVE_QUALITY_DATA_CACHE",
            "EVOLVE_AGENT_TMP",
        )
        if os.environ.get(key)
    }
    env = clean_subprocess_env(checkout, repo_root=repo_root(), extra=extra)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    wall = float(registry.timeout_seconds) + 300.0
    print(
        f"quality {side} start {dataset.task_id} "
        f"timeout_s={spec['timeout_seconds']} n_jobs={spec['n_jobs']} "
        f"preset={spec['preset']} seed={spec['seed']} "
        f"cache={cache_path} log={log_path}",
        flush=True,
    )
    try:
        proc = run_worker(cmd, cwd=repo_root(), env=env, timeout=wall, log_path=log_path)
        log_text = proc.stdout
        if proc.timed_out:
            return ScoreResult(
                task_id=dataset.task_id,
                status="timeout",
                score=float("nan"),
                detail=f"timeout after {wall}s",
                duration_s=wall,
                env_hash=env_hash,
                cmd=" ".join(cmd),
                seed=registry.seed,
                log_tail=log_text[-4000:],
                metric_observations={"search_ran": False, "search_detail": "wall_timeout"},
            )
        if not out_path.is_file():
            return ScoreResult(
                task_id=dataset.task_id,
                status="invalid",
                score=float("nan"),
                traceback=log_text[-4000:],
                detail=f"worker exit {proc.returncode}",
                env_hash=env_hash,
                cmd=" ".join(cmd),
                log_tail=log_text[-4000:],
                seed=registry.seed,
                metric_observations={"search_ran": False},
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        spec_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
    payload["log_tail"] = (str(payload.get("log_tail") or "") + "\n" + log_text)[-2000:]
    result = _score_from_payload(
        dataset.task_id, payload, env_hash=env_hash, cmd=" ".join(cmd)
    )
    if side == "stock":
        _store_stock_cache(
            result,
            dataset,
            registry,
            spec["n_jobs"],
            fedot_commit=fedot_commit,
        )
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

    def one(task_id: str, *, registry: QualityRegistry, n_jobs: int) -> dict:
        dataset = get_dataset(task_id, registry)
        spec = job_spec(dataset, n_jobs=n_jobs, registry=registry)
        stock = run_quality_side(
            dataset,
            checkout=stock_checkout,
            side="stock",
            n_jobs=n_jobs,
            registry=registry,
        )
        return {
            "spec": spec,
            "stock": asdict(stock),
            "search_ran": bool((stock.metric_observations or {}).get("search_ran")),
            "cache_hit": bool((stock.metric_observations or {}).get("cache_hit")),
        }

    return _run_over_tasks(
        one, task_ids=task_ids, n_jobs=n_jobs, cpu_quota=cpu_quota, registry=registry
    )


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

    def one(task_id: str, *, registry: QualityRegistry, n_jobs: int) -> dict:
        return run_quality_pair(
            task_id,
            stock_checkout=stock_checkout,
            patch_checkout=patch_checkout,
            n_jobs=n_jobs,
            registry=registry,
        )

    return _run_over_tasks(
        one, task_ids=task_ids, n_jobs=n_jobs, cpu_quota=cpu_quota, registry=registry
    )


def _run_over_tasks(
    one,
    *,
    task_ids: tuple[str, ...] | None,
    n_jobs: int | None,
    cpu_quota: int | None,
    registry: QualityRegistry | None,
) -> list[dict]:
    """Run ``one(task_id)`` per task, ``cpu_quota // n_jobs`` datasets at a time; keep order."""

    registry = registry or load_quality_registry()
    ids = task_ids or list_quality_task_ids(registry)
    jobs = int(n_jobs if n_jobs is not None else registry.n_jobs_per_job)
    quota = int(cpu_quota if cpu_quota is not None else registry.cpu_quota)
    parallel = max(1, min(len(ids), quota // max(1, jobs)))
    if parallel <= 1 or len(ids) == 1:
        return [one(task_id, registry=registry, n_jobs=jobs) for task_id in ids]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(one, task_id, registry=registry, n_jobs=jobs): task_id
            for task_id in ids
        }
        by_id = {futures[future]: future.result() for future in as_completed(futures)}
    return [by_id[task_id] for task_id in ids]


def decision_from_quality_jobs(rows: list[dict]) -> Decision:
    rows = [
        row
        for row in rows
        if str(row.get("status") or "") not in {"skipped", "inapplicable"}
        and str((row.get("spec") or {}).get("status") or "") != "skipped"
    ]
    if not rows:
        return Decision(
            keep=False,
            reason="no_applicable_quality_tasks",
            target_delta=None,
            stage="quality",
            infrastructure_error=False,
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

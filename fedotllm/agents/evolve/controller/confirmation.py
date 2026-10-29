"""Technical screen and DEV confirmation. Short-horizon scores never DROP quality."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.types import Decision, ScoreResult

ScoreRunner = Callable[..., dict[str, ScoreResult]]
VerdictRunner = Callable[..., Decision]

_METRIC_EPS = 1e-12


def affected_metric_moved(
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult],
    affected_ids: tuple[str, ...],
    *,
    eps: float = _METRIC_EPS,
) -> bool:
    """True when any affected workload's numeric score changed at all."""

    for task_id in affected_ids:
        before = stock.get(task_id)
        after = patched.get(task_id)
        if before is None or after is None:
            continue
        if before.status != "ok" or after.status != "ok":
            continue
        try:
            if abs(float(after.score) - float(before.score)) > eps:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _deterministic_patch_failure(
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult],
    exam_ids: tuple[str, ...],
) -> tuple[str | None, bool]:
    """Return (reason, infrastructure) when stock ran and the patch did not."""

    for task_id in exam_ids:
        before = stock[task_id]
        after = patched[task_id]
        if before.status != "ok":
            continue
        if after.status == "crash":
            return f"regression {task_id} crashed after patch", False
        if after.status in {"timeout", "invalid"}:
            return f"regression {task_id} timeout_or_invalid", True
    return None, False


def quick_quality_screen(
    experiment: Path,
    exam_ids: tuple[str, ...],
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    stock_dev: dict[str, ScoreResult],
    *,
    seed: int = 42,
    measure_patched_fn: ScoreRunner,
    verdict_fn: VerdictRunner,
) -> tuple[bool, dict]:
    """Infrastructure-only gate. Short-horizon scores never DROP a candidate.

    Reject only a deterministic launch failure: stock succeeded and the patch
    crashed, timed out, or returned an invalid score. A missing or negative
    short-horizon delta only annotates queue priority. Quality KEEP/DROP is
    reserved for the closed FINAL batch; this stage does not open SHADOW.
    """

    local_lift = tuple(task_id for task_id in lift_ids if task_id in exam_ids) or exam_ids
    local_protect = (
        tuple(task_id for task_id in protect_ids if task_id in exam_ids) or exam_ids
    )
    dev_stock = {task_id: stock_dev[task_id] for task_id in exam_ids}
    dev_patched = measure_patched_fn(
        exam_ids,
        checkout=experiment,
        split="dev",
        seed=seed,
    )
    failure, infrastructure = _deterministic_patch_failure(
        dev_stock, dev_patched, exam_ids
    )
    dev = verdict_fn(
        dev_stock,
        dev_patched,
        lift_ids=local_lift,
        protect_ids=local_protect,
        evidence_only=True,
    )
    payload = {
        "target_delta": dev.target_delta,
        "dev": asdict(dev),
        "shadow": {"evaluated": False},
        "queue_priority": "high" if dev.keep else "normal",
    }
    if failure:
        payload.update(
            {
                "passed": False,
                "reason": failure,
                "infrastructure_error": infrastructure,
            }
        )
        return False, payload
    payload.update(
        {
            "passed": True,
            "reason": "early_gain" if dev.keep else "technically_valid",
            "infrastructure_error": False,
        }
    )
    return True, payload


def confirm_dev(
    source: Path,
    experiment: Path,
    exam_ids: tuple[str, ...],
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    *,
    seeds: tuple[int, ...] = (42, 43, 44),
    evidence_only: bool = False,
    measure_stock_fn: ScoreRunner,
    measure_patched_fn: ScoreRunner,
    verdict_fn: VerdictRunner,
) -> tuple[bool, dict]:
    """Confirm a DEV gain across model seeds and an independent data view.

    Model seeds alone do not protect deterministic estimators from fitting a
    lucky DEV partition.  SHADOW is carved only from training rows, so it adds
    data-sample evidence without exposing FINAL.
    """

    rows: list[dict] = []
    improved = 0
    regressed = 0
    infrastructure = 0
    for seed in seeds:
        stock = measure_stock_fn(exam_ids, checkout=source, split="dev", seed=seed)
        patched = measure_patched_fn(exam_ids, checkout=experiment, split="dev", seed=seed)
        invalid = any(
            result.status in {"timeout", "invalid"}
            for result in (*stock.values(), *patched.values())
        )
        decision = verdict_fn(
            stock,
            patched,
            lift_ids=lift_ids,
            protect_ids=protect_ids,
            evidence_only=evidence_only,
        )
        if invalid:
            infrastructure += 1
        elif decision.reason.startswith("regression"):
            regressed += 1
        elif decision.keep:
            improved += 1
        rows.append(
            {
                "seed": seed,
                "keep": decision.keep,
                "reason": decision.reason,
                "target_delta": decision.target_delta,
            }
        )
    seed_confirmed = improved >= 2 and regressed == 0 and infrastructure == 0
    shadow_row: dict = {"evaluated": False, "keep": False, "reason": "dev_seed_gate_failed"}
    if seed_confirmed and seeds:
        shadow_seed = seeds[0]
        shadow_stock = measure_stock_fn(
            exam_ids, checkout=source, split="shadow", seed=shadow_seed
        )
        shadow_patched = measure_patched_fn(
            exam_ids, checkout=experiment, split="shadow", seed=shadow_seed
        )
        shadow_invalid = any(
            result.status in {"timeout", "invalid"}
            for result in (*shadow_stock.values(), *shadow_patched.values())
        )
        shadow_decision = verdict_fn(
            shadow_stock,
            shadow_patched,
            lift_ids=lift_ids,
            protect_ids=protect_ids,
            evidence_only=evidence_only,
        )
        if shadow_invalid:
            infrastructure += 1
        shadow_row = {
            "evaluated": True,
            "seed": shadow_seed,
            "keep": shadow_decision.keep and not shadow_invalid,
            "reason": shadow_decision.reason,
            "target_delta": shadow_decision.target_delta,
        }
    confirmed = seed_confirmed and bool(shadow_row["keep"])
    return confirmed, {
        "confirmed": confirmed,
        "mode": "metric_signal" if evidence_only else "practical_keep",
        "seed_confirmed": seed_confirmed,
        "improved_seeds": improved,
        "regressed_task_seed_pairs": regressed,
        "infrastructure_failures": infrastructure,
        "seeds": rows,
        "shadow": shadow_row,
    }

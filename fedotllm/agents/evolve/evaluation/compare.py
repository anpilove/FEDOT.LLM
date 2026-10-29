from __future__ import annotations

import math

from fedotllm.agents.evolve.evaluation.tasks import load_task
from fedotllm.agents.evolve.types import Decision, ScoreResult, TaskSpec


def _numeric(result: ScoreResult, sentinel: float) -> float | None:
    if result.status == "crash":
        return sentinel
    if result.status != "ok" or not math.isfinite(result.score):
        return None
    return result.score


def _patched_broke(stock: ScoreResult, patched: ScoreResult) -> bool:
    return stock.status == "ok" and patched.status == "crash"


def _spec(
    task_id: str,
    *,
    higher_is_better: bool,
    min_delta: float,
    sentinel: float,
) -> tuple[bool, float, float, str]:
    try:
        spec: TaskSpec = load_task(task_id)
        return spec.higher_is_better, spec.min_delta, spec.sentinel, spec.min_delta_mode
    except KeyError:
        return higher_is_better, min_delta, sentinel, "absolute"


def _threshold(stock_value: float, min_delta: float, mode: str) -> float:
    if mode == "relative":
        return min_delta * max(abs(stock_value), 1e-12)
    return min_delta


def _signal_threshold(stock_value: float) -> float:
    """Numerical floor for evidence discovery, not the practical KEEP bar.

    A candidate below the task's declared effect-size threshold can still be
    causally useful feedback for the fixer.  The floor only removes floating
    point dust; multi-seed DEV, SHADOW and the ordinary protect thresholds are
    responsible for rejecting noise and regressions.
    """

    return max(abs(stock_value) * 1e-9, 1e-12)


def compare_pack(
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult],
    *,
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    higher_is_better: bool = True,
    min_delta: float = 0.01,
    sentinel: float = 0.5,
    evidence_only: bool = False,
) -> Decision:
    """Compare a workload pack using either practical or evidence thresholds.

    ``evidence_only`` is used only to investigate a small positive DEV signal.
    It does not turn that signal into a production KEEP: protect tasks retain
    their normal task thresholds and the controller still requires multi-seed
    DEV plus SHADOW before preserving the candidate as promising evidence.
    """

    lift_deltas: dict[str, float | None] = {}
    lift_ok = False
    best: float | None = None
    for task_id in lift_ids:
        hib, floor, sent, mode = _spec(
            task_id, higher_is_better=higher_is_better, min_delta=min_delta, sentinel=sentinel
        )
        if _patched_broke(stock[task_id], patched[task_id]):
            lift_deltas[task_id] = None
            continue
        a = _numeric(stock[task_id], sent)
        b = _numeric(patched[task_id], sent)
        if a is None or b is None:
            lift_deltas[task_id] = None
            continue
        delta = (b - a) if hib else (a - b)
        lift_deltas[task_id] = delta
        if best is None or delta > best:
            best = delta
        lift_need = _signal_threshold(a) if evidence_only else _threshold(a, floor, mode)
        if delta >= lift_need:
            lift_ok = True
    regressions: dict[str, float | None] = dict(lift_deltas)
    if best is None:
        return Decision(
            keep=False,
            reason="timeout_or_invalid",
            target_delta=None,
            regression_deltas=regressions,
            infrastructure_error=True,
        )
    for task_id in protect_ids:
        hib, floor, sent, mode = _spec(
            task_id, higher_is_better=higher_is_better, min_delta=min_delta, sentinel=sentinel
        )
        if _patched_broke(stock[task_id], patched[task_id]):
            regressions[task_id] = None
            return Decision(
                keep=False,
                reason=f"regression {task_id} crashed after patch",
                target_delta=best,
                regression_deltas=regressions,
            )
        a = _numeric(stock[task_id], sent)
        b = _numeric(patched[task_id], sent)
        if a is None or b is None:
            regressions[task_id] = None
            return Decision(
                keep=False,
                reason=f"regression {task_id} timeout_or_invalid",
                target_delta=best,
                regression_deltas=regressions,
                infrastructure_error=True,
            )
        d = (b - a) if hib else (a - b)
        regressions[task_id] = d
        need = _threshold(a, floor, mode)
        if d < -need:
            return Decision(
                keep=False,
                reason=f"regression {task_id} delta {d:.4f}",
                target_delta=best,
                regression_deltas=regressions,
            )
    if not lift_ok:
        threshold_label = "numerical signal floor" if evidence_only else "per-task threshold"
        return Decision(
            keep=False,
            reason=f"target_delta {best:.4f} below {threshold_label}",
            target_delta=best,
            regression_deltas=regressions,
        )
    return Decision(
        keep=True,
        reason="metric_signal" if evidence_only else "keep",
        target_delta=best,
        regression_deltas=regressions,
    )

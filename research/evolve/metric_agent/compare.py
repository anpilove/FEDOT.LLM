from __future__ import annotations

from research.evolve.metric_agent.types import Decision, ScoreResult


def _numeric(result: ScoreResult, sentinel: float) -> float | None:
    if result.status == "crash":
        return sentinel
    if result.status in {"timeout", "invalid"}:
        return None
    return result.score


def compare(
    target_stock: ScoreResult,
    target_patched: ScoreResult,
    other_pairs: list[tuple[ScoreResult, ScoreResult]],
    *,
    higher_is_better: bool = True,
    min_delta: float = 0.01,
    sentinel: float = 0.5,
) -> Decision:
    s = _numeric(target_stock, sentinel)
    p = _numeric(target_patched, sentinel)
    if s is None or p is None:
        return Decision(
            keep=False,
            reason="timeout_or_invalid",
            target_delta=None,
            regression_deltas={},
        )
    delta = (p - s) if higher_is_better else (s - p)
    regressions: dict[str, float | None] = {}
    if delta < min_delta:
        return Decision(
            keep=False,
            reason=f"target_delta {delta:.4f} < {min_delta}",
            target_delta=delta,
            regression_deltas=regressions,
        )
    for stock_r, patched_r in other_pairs:
        a = _numeric(stock_r, sentinel)
        b = _numeric(patched_r, sentinel)
        if a is None or b is None:
            regressions[patched_r.task_id] = None
            return Decision(
                keep=False,
                reason=f"regression {patched_r.task_id} timeout_or_invalid",
                target_delta=delta,
                regression_deltas=regressions,
            )
        d = (b - a) if higher_is_better else (a - b)
        regressions[patched_r.task_id] = d
        if d < -min_delta:
            return Decision(
                keep=False,
                reason=f"regression {patched_r.task_id} delta {d:.4f}",
                target_delta=delta,
                regression_deltas=regressions,
            )
    return Decision(
        keep=True,
        reason="keep",
        target_delta=delta,
        regression_deltas=regressions,
    )


def compare_pack(
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult],
    *,
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    higher_is_better: bool = True,
    min_delta: float = 0.01,
    sentinel: float = 0.5,
) -> Decision:
    """Hidden exam: at least one lift task improves; protect tasks must not drop."""

    lift_deltas: dict[str, float | None] = {}
    best: float | None = None
    for task_id in lift_ids:
        a = _numeric(stock[task_id], sentinel)
        b = _numeric(patched[task_id], sentinel)
        if a is None or b is None:
            lift_deltas[task_id] = None
            continue
        delta = (b - a) if higher_is_better else (a - b)
        lift_deltas[task_id] = delta
        if best is None or delta > best:
            best = delta
    regressions: dict[str, float | None] = dict(lift_deltas)
    if best is None:
        return Decision(keep=False, reason="timeout_or_invalid", target_delta=None, regression_deltas=regressions)
    for task_id in protect_ids:
        a = _numeric(stock[task_id], sentinel)
        b = _numeric(patched[task_id], sentinel)
        if a is None or b is None:
            regressions[task_id] = None
            return Decision(
                keep=False,
                reason=f"regression {task_id} timeout_or_invalid",
                target_delta=best,
                regression_deltas=regressions,
            )
        d = (b - a) if higher_is_better else (a - b)
        regressions[task_id] = d
        if d < -min_delta:
            return Decision(
                keep=False,
                reason=f"regression {task_id} delta {d:.4f}",
                target_delta=best,
                regression_deltas=regressions,
            )
    if best < min_delta:
        return Decision(
            keep=False,
            reason=f"target_delta {best:.4f} < {min_delta}",
            target_delta=best,
            regression_deltas=regressions,
        )
    return Decision(keep=True, reason="keep", target_delta=best, regression_deltas=regressions)

from __future__ import annotations

import math

from fedotllm.agents.evolve.evaluation.tasks import load_task
from fedotllm.agents.evolve.types import Decision, ScoreResult, TaskSpec

# Worker exceptions that belong to the candidate. A pin mismatch or a
# budget timeout is not one of these: those stay unresolved infrastructure.
CANDIDATE_FAILURES = frozenset({
    "fedot_import_error",
    "fedot_fit_error",
    "scoring_error",
    "worker_error",
})


def _failure_kind(result: ScoreResult) -> str | None:
    kind = (result.metric_observations or {}).get("failure_kind")
    return kind if isinstance(kind, str) else None


def _numeric(result: ScoreResult) -> float | None:
    # A crash has no fitness. GOLEM drops an individual that was not evaluated
    # and never treats an invalid fitness as a measured value.
    if result.status != "ok" or not math.isfinite(result.score):
        return None
    return result.score


def candidate_failed(result: ScoreResult) -> bool:
    """A candidate-owned failure. Pin mismatches and timeouts are not included."""

    if result.status == "crash":
        return True
    return result.status == "invalid" and _failure_kind(result) in CANDIDATE_FAILURES


def patched_broke(stock: ScoreResult, patched: ScoreResult) -> bool:
    """Stock measured a score and the patch failed as a candidate."""

    return stock.status == "ok" and candidate_failed(patched)


def same_preexisting_failure(stock: ScoreResult, patched: ScoreResult) -> bool:
    """Both sides failed the same way before this patch changed anything."""

    if stock.status == "ok" or patched.status != stock.status:
        return False
    detail = stock.detail or ""
    return bool(detail) and detail == (patched.detail or "")


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


def _absent(task_id: str, stock: dict[str, ScoreResult], patched: dict[str, ScoreResult]) -> bool:
    return task_id not in stock or task_id not in patched


def _missing_measurement(
    task_id: str,
    *,
    target_delta: float | None,
    regressions: dict[str, float | None],
) -> Decision:
    """Refuse KEEP when a requested dataset was never measured.

    GOLEM ``GraphVerifier`` rejects a graph on ``False`` or ``ValueError`` and
    lets any other exception escape. A missing stock or patch row used to do
    the same: ``KeyError`` aborted the campaign before the next hypothesis.
    The gap is an unresolved measurement, so this pack cannot KEEP.
    """

    recorded = dict(regressions)
    recorded[task_id] = None
    return Decision(
        keep=False,
        reason=f"missing_measurement {task_id}",
        target_delta=target_delta,
        regression_deltas=recorded,
        infrastructure_error=True,
    )


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
        hib, floor, _sent, mode = _spec(
            task_id, higher_is_better=higher_is_better, min_delta=min_delta, sentinel=sentinel
        )
        if _absent(task_id, stock, patched):
            return _missing_measurement(
                task_id, target_delta=best, regressions=lift_deltas
            )
        if patched_broke(stock[task_id], patched[task_id]):
            lift_deltas[task_id] = None
            continue
        if same_preexisting_failure(stock[task_id], patched[task_id]):
            continue
        a = _numeric(stock[task_id])
        b = _numeric(patched[task_id])
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
    # GOLEM does not treat an invalid fitness as a successful individual.
    # One lift dataset without a score blocks KEEP even if another dataset improved.
    unevaluated = [task_id for task_id in lift_ids if lift_deltas.get(task_id) is None]
    broke = [
        task_id for task_id in unevaluated
        if patched_broke(stock[task_id], patched[task_id])
    ]
    if broke:
        return Decision(
            keep=False,
            reason=f"regression {broke[0]} crashed after patch",
            target_delta=best,
            regression_deltas=regressions,
        )
    if unevaluated:
        return Decision(
            keep=False,
            reason="timeout_or_invalid",
            target_delta=best,
            regression_deltas=regressions,
            infrastructure_error=True,
        )
    for task_id in protect_ids:
        hib, floor, _sent, mode = _spec(
            task_id, higher_is_better=higher_is_better, min_delta=min_delta, sentinel=sentinel
        )
        if _absent(task_id, stock, patched):
            return _missing_measurement(
                task_id, target_delta=best, regressions=regressions
            )
        if patched_broke(stock[task_id], patched[task_id]):
            regressions[task_id] = None
            return Decision(
                keep=False,
                reason=f"regression {task_id} crashed after patch",
                target_delta=best,
                regression_deltas=regressions,
            )
        if same_preexisting_failure(stock[task_id], patched[task_id]):
            continue
        a = _numeric(stock[task_id])
        b = _numeric(patched[task_id])
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

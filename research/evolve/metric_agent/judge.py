"""Judge: hidden holdout exam. No LLM. Gym/cases are inputs here only."""

from __future__ import annotations

from pathlib import Path

from research.evolve.metric_agent.checkout import resolve_fedot_src
from research.evolve.metric_agent.compare import compare_pack
from research.evolve.metric_agent.discover import pytest_snapshot
from research.evolve.metric_agent.eval import run_patched, run_stock
from research.evolve.metric_agent.tasks import hidden_exam, load_task
from research.evolve.metric_agent.types import Decision, Lead, ScoreResult


def measure_stock(
    exam_ids: tuple[str, ...],
    *,
    checkout: Path | None = None,
) -> dict[str, ScoreResult]:
    tree = checkout or resolve_fedot_src()
    return {task_id: run_stock(task_id, checkout=tree) for task_id in exam_ids}


def measure_patched(exam_ids: tuple[str, ...], *, checkout: Path) -> dict[str, ScoreResult]:
    return {task_id: run_patched(task_id, checkout=checkout) for task_id in exam_ids}


def measure_fedot_tests(checkout: Path) -> tuple[str, list[Lead], set[str]]:
    """FEDOT unit tests after a patch. Not an agent hunt signal."""

    return pytest_snapshot(checkout)


def tests_regressed(before: set[str], after: set[str]) -> Decision | None:
    extra = after - before
    if not extra:
        return None
    sample = ", ".join(sorted(extra)[:3])
    return Decision(
        keep=False,
        reason=f"fedot_tests_regressed {sample}",
        target_delta=None,
        regression_deltas={},
    )


def verdict(
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult],
    *,
    lift_ids: tuple[str, ...] | None = None,
    protect_ids: tuple[str, ...] | None = None,
) -> Decision:
    exam_lift, exam_protect = hidden_exam()
    lift_ids = lift_ids or exam_lift
    protect_ids = protect_ids or exam_protect
    spec = load_task(lift_ids[0])
    return compare_pack(
        stock,
        patched,
        lift_ids=lift_ids,
        protect_ids=protect_ids,
        higher_is_better=spec.higher_is_better,
        min_delta=spec.min_delta,
        sentinel=spec.sentinel,
    )

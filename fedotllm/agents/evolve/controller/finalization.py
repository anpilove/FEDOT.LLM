"""One-shot hidden FINAL evaluation after a confirmed DEV keep."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.controller.artifacts import _score_log
from fedotllm.agents.evolve.evaluation.tasks import final_exam
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.storage.scoreboard import append_final
from fedotllm.agents.evolve.types import Decision, PatchCandidate, ScoreResult

ScoreRunner = Callable[..., dict[str, ScoreResult]]
VerdictRunner = Callable[..., Decision]

def record_final_skipped(journal: Path, *, reason: str) -> None:
    append_journal(journal, {"event": "final", "skipped": True, "reason": reason})


def record_final(
    source: Path,
    workspace: Path,
    journal: Path,
    candidate: PatchCandidate,
    last: Decision,
    *,
    run_id: str,
    seeds: tuple[int, ...] = (42, 43, 44),
    lift_ids: tuple[str, ...] | None = None,
    protect_ids: tuple[str, ...] | None = None,
    enabled: bool = True,
    transfer_plan: dict | None = None,
    operation_params: dict | None = None,
    measurement_budget=None,
    measure_stock_fn: ScoreRunner,
    measure_patched_fn: ScoreRunner,
    verdict_fn: VerdictRunner,
) -> Decision | None:
    if not enabled or os.environ.get("EVOLVE_AGENT_SKIP_FINAL") == "1":
        record_final_skipped(journal, reason="disabled")
        return None
    from fedotllm.agents.evolve.controller.transfer import evaluate_transfer, validate_transfer
    error = validate_transfer(transfer_plan)
    if error:
        record_final_skipped(journal, reason=error)
        return Decision(keep=False, reason=error, target_delta=None, stage="final", final_keep=False)
    if lift_ids is None or protect_ids is None:
        default_lift, default_protect = final_exam()
        lift_ids = lift_ids if lift_ids is not None else default_lift
        protect_ids = protect_ids if protect_ids is not None else default_protect
    exam_ids = tuple(dict.fromkeys((*lift_ids, *protect_ids)))
    if not exam_ids:
        record_final_skipped(journal, reason="empty_final")
        return None
    if not seeds:
        record_final_skipped(journal, reason="no_confirmation_seeds")
        return None
    first_seed = seeds[0]
    final_tree = create_experiment_checkout(
        source,
        workspace,
        run_id=run_id,
        candidate_id=f"final-{candidate.candidate_id}",
    )
    try:
        # Check the patch applies before spending any exposure of the hidden split.
        if not apply_patch(final_tree, candidate):
            record_final_skipped(journal, reason="reapply_failed")
            return None
        stock = measure_stock_fn(exam_ids, checkout=source, split="final", seed=first_seed)
        patched = measure_patched_fn(exam_ids, checkout=final_tree, split="final", seed=first_seed)
        decision = verdict_fn(stock, patched, lift_ids=lift_ids, protect_ids=protect_ids)
        transfer_report = None
        if decision.keep:
            transferred, transfer_report = evaluate_transfer(
                source, final_tree, transfer_plan, params=operation_params, split="final",
                budget=measurement_budget,
            )
            decision.keep = transferred
            if not transferred:
                decision.reason = transfer_report["reason"]
    finally:
        discard_experiment_checkout(final_tree, workspace=workspace, source=source)
    failure_reason = decision.reason
    regressed = int(decision.reason.startswith("regression"))
    infrastructure = int(decision.infrastructure_error)
    # FINAL is a one-shot holdout. Seed robustness has already been established
    # on DEV plus SHADOW; repeating FINAL leaks more information from the hidden
    # split and multiplies the most expensive full-suite evaluation.
    decision.keep = bool(decision.keep and not infrastructure)
    decision.reason = "final_confirmed" if decision.keep else "final_confirmation_failed"
    decision.stage = "final"
    decision.final_keep = decision.keep
    row = {
        "event": "final",
        "skipped": False,
        "keep_dev": last.keep,
        "keep_final": decision.keep,
        "reason": decision.reason,
        # The verdict/transfer reason behind a failed FINAL, kept for diagnosis.
        "detail": failure_reason,
        "target_delta": decision.target_delta,
        "regression_deltas": decision.regression_deltas,
        "candidate": candidate.candidate_id,
        "transfer": transfer_report,
        "edits": [asdict(edit) for edit in candidate.edits],
        "file": candidate.file_path,
        "stock": {key: _score_log(value) for key, value in stock.items()},
        "patched": {key: _score_log(value) for key, value in patched.items()},
        "confirmation": {
            "improved_seeds": int(decision.keep),
            "regressed_task_seed_pairs": regressed,
            "infrastructure_failures": infrastructure,
            "seeds": [
                {
                    "seed": first_seed,
                    "keep": decision.keep,
                    "reason": failure_reason,
                    "target_delta": decision.target_delta,
                }
            ],
        },
    }
    append_journal(journal, row)
    append_final(workspace, decision=decision, stock=stock, patched=patched)
    return decision

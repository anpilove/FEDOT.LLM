from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from research.evolve.metric_agent.checkout import (
    make_disposable_checkout,
    resolve_fedot_src,
    revert_checkout,
    snapshot_diff,
)
from research.evolve.metric_agent.eval import run_stock
from research.evolve.metric_agent.fixer import fix_lead
from research.evolve.metric_agent.guard import guard_path
from research.evolve.metric_agent.journal import append_journal
from research.evolve.metric_agent.judge import (
    measure_fedot_tests,
    measure_patched,
    measure_stock,
    tests_regressed,
    verdict,
)
from research.evolve.metric_agent.scoreboard import append_attempt
from research.evolve.metric_agent.scout import scout
from research.evolve.metric_agent.smoke import import_error
from research.evolve.metric_agent.tasks import hidden_exam
from research.evolve.metric_agent.types import Decision, ScoreResult


def run_once(
    *,
    checkout: Path | None = None,
    inference=None,
    workspace: Path | None = None,
    lift_ids: tuple[str, ...] | None = None,
    protect_ids: tuple[str, ...] | None = None,
    max_leads: int | None = None,
) -> Decision:
    """Agent walks the library; tests and holdout run after the patch."""

    exam_lift, exam_protect = hidden_exam()
    lift_ids = lift_ids or exam_lift
    protect_ids = protect_ids or exam_protect
    cap = max_leads if max_leads is not None else int(os.environ.get("METRIC_AGENT_MAX_LEADS", "3"))
    checkout = checkout or make_disposable_checkout()
    workspace = workspace or Path(os.environ.get("METRIC_AGENT_WORK", "/tmp/metric-agent-run"))
    workspace.mkdir(parents=True, exist_ok=True)
    journal = workspace / "journal.jsonl"

    exam_ids = tuple(dict.fromkeys((*lift_ids, *protect_ids)))
    leads = scout(checkout, inference=inference, max_leads=cap)
    append_journal(journal, {"event": "scout", "leads": [asdict(lead) for lead in leads]})
    stock = measure_stock(exam_ids, checkout=checkout)
    if not leads:
        decision = Decision(keep=False, reason="no_lead", target_delta=None, regression_deltas={})
        append_journal(journal, {"event": "decision", **asdict(decision)})
        append_attempt(workspace, lead=None, candidate=None, stock=stock, patched=None, decision=decision)
        return decision

    _, _, baseline_nodes = measure_fedot_tests(checkout)
    last = Decision(keep=False, reason="no_patch", target_delta=None, regression_deltas={})
    dirty = False
    for lead in leads:
        if dirty:
            revert_checkout(checkout)
            dirty = False
        candidate = fix_lead(checkout, lead, inference=inference, workspace=workspace)
        if candidate is None:
            last = Decision(keep=False, reason="no_patch", target_delta=None, regression_deltas={})
            append_attempt(workspace, lead=lead, candidate=None, stock=stock, patched=None, decision=last)
            continue
        dirty = True
        broken = import_error(checkout, candidate.file_path)
        if broken:
            last = Decision(keep=False, reason=f"patch_unimportable {broken}", target_delta=None, regression_deltas={})
            append_journal(
                journal,
                {
                    "event": "decision",
                    "candidate": candidate.candidate_id,
                    "file": candidate.file_path,
                    "lead": asdict(lead),
                    "diff": snapshot_diff(checkout, candidate.file_path),
                    **asdict(last),
                },
            )
            append_attempt(workspace, lead=lead, candidate=candidate, stock=stock, patched=None, decision=last)
            continue
        _text, _after_leads, after_nodes = measure_fedot_tests(checkout)
        blocked = tests_regressed(baseline_nodes, after_nodes)
        if blocked is not None:
            last = blocked
            append_journal(
                journal,
                {
                    "event": "decision",
                    "candidate": candidate.candidate_id,
                    "file": candidate.file_path,
                    "lead": asdict(lead),
                    "fedot_test_failures": sorted(after_nodes)[:20],
                    **asdict(last),
                },
            )
            append_attempt(workspace, lead=lead, candidate=candidate, stock=stock, patched=None, decision=last)
            continue
        patched = measure_patched(exam_ids, checkout=checkout)
        last = verdict(stock, patched, lift_ids=lift_ids, protect_ids=protect_ids)
        append_journal(
            journal,
            {
                "event": "decision",
                "candidate": candidate.candidate_id,
                "file": candidate.file_path,
                "lead": asdict(lead),
                "stock": {key: _score_log(value) for key, value in stock.items()},
                "patched": {key: _score_log(value) for key, value in patched.items()},
                "diff": snapshot_diff(checkout, candidate.file_path),
                **asdict(last),
            },
        )
        append_attempt(
            workspace,
            lead=lead,
            candidate=candidate,
            stock=stock,
            patched=patched,
            decision=last,
        )
        if last.keep:
            return last
    if dirty:
        revert_checkout(checkout)
    return last


def eval_contract(checkout: Path | None = None) -> dict:
    checkout = checkout or resolve_fedot_src()
    ids = ("catboost", "pca->catboost", "fast_ica->lgbm")
    results = {task_id: run_stock(task_id, checkout=checkout) for task_id in ids}
    return {
        "guard_cases": guard_path("data/cases.json"),
        "guard_scorer": guard_path("research/evolve/metric_agent/scorer.py"),
        "scores": {key: _score_log(value) for key, value in results.items()},
    }


def _score_log(result: ScoreResult) -> dict:
    return {
        "task_id": result.task_id,
        "status": result.status,
        "score": result.score,
        "duration_s": result.duration_s,
        "detail": result.detail[:240],
        "traceback_chars": len(result.traceback or ""),
        "cmd": result.cmd,
        "env_hash": result.env_hash,
        "log_tail": result.log_tail[-800:],
    }

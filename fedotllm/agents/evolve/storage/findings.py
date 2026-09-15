"""Append-only research dataset shared by EvolveAgent campaigns.

Detailed prompts, snippets and subprocess output remain in a run workspace.
This dataset is the stable, article-friendly index of what each campaign found,
what was actually verified, and why a patch was kept or rejected.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.storage.journal import append_journal, resolve_run_workspace

SCHEMA_VERSION = 1


def _contract_id(lead: dict[str, Any]) -> str:
    for item in lead.get("evidence") or ():
        text = str(item)
        if text.startswith("observed contract id:"):
            return text.partition(":")[2].strip()
    return ""


def _defect_id(row: dict[str, Any]) -> str:
    """Identify an independently reproduced defect across different patches."""

    reproduction = row.get("reproduction") or {}
    if not (
        reproduction.get("stock") == "failed_as_predicted"
        and reproduction.get("patched") == "resolved"
    ):
        return ""
    lead = row.get("lead") if isinstance(row.get("lead"), dict) else {}
    contract_id = _contract_id(lead)
    if contract_id:
        return f"public_contract:{contract_id}"
    material = reproduction.get("code") or reproduction.get("controller_workloads")
    if not material:
        material = {
            "file_path": lead.get("file_path"),
            "claim": reproduction.get("claim") or lead.get("why"),
            "expected": reproduction.get("expected"),
        }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, default=str)
    return "behavior:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def default_findings_path() -> Path:
    return Path(__file__).resolve().parents[4] / "docs" / "evolve" / "findings.jsonl"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def next_run_number(path: Path) -> int:
    numbers = [
        int(row["run_number"])
        for row in _rows(path)
        if isinstance(row.get("run_number"), int)
    ]
    return max(numbers, default=0) + 1


def has_run(path: Path, run_id: str) -> bool:
    return any(row.get("run_id") == run_id for row in _rows(path))


def _test_gate_passed(row: dict[str, Any]) -> bool:
    explicit = row.get("fedot_test_gate_passed")
    if explicit is not None:
        return bool(explicit)
    status = row.get("fedot_test_status")
    if status == "passed":
        return True
    # Historical journals predate the explicit field. Reaching DEV proves the
    # comparative gate accepted the candidate, even if stock and patched share
    # a known baseline failure.
    return row.get("patched") is not None and not str(
        row.get("reason") or ""
    ).startswith("fedot_tests_regressed")


def begin_run(
    path: Path,
    *,
    run_id: str,
    workspace: Path,
    source: Path,
    source_commit: str,
    source_hash: str,
    campaign_config: dict[str, Any],
    run_number: int | None = None,
) -> int:
    if has_run(path, run_id):
        for row in _rows(path):
            if row.get("run_id") == run_id and isinstance(row.get("run_number"), int):
                return int(row["run_number"])
    number = run_number if run_number is not None else next_run_number(path)
    append_journal(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "run",
            "event": "run_start",
            "run_number": number,
            "run_id": run_id,
            "workspace": str(workspace.resolve()),
            "source": str(source.resolve()),
            "source_commit": source_commit,
            "source_hash": source_hash,
            "campaign_config": campaign_config,
        },
    )
    return number


def classify_finding(row: dict[str, Any]) -> tuple[str, str]:
    """Return outcome and safe application recommendation.

    A passing regression suite does not prove the proposed defect.  A neutral
    correctness fix is eligible for a branch only when a dedicated verifier has
    recorded the same stock failure and its resolution after patching.
    """

    if row.get("infrastructure_error"):
        return "infrastructure_error", "retry_after_infrastructure_fix"
    candidate = row.get("candidate")
    if not candidate:
        return "unverified_no_patch", "do_not_apply"
    test_status = row.get("fedot_test_status")
    test_gate_passed = _test_gate_passed(row)
    if test_status and not test_gate_passed:
        return "rejected_tests", "do_not_apply"
    reason = str(row.get("reason") or "")
    if reason.startswith("regression"):
        return "rejected_dev_regression", "do_not_apply"
    if reason in {"affected_metric_regression", "affected_metric_not_confirmed"}:
        return "rejected_affected_metric", "do_not_apply"
    if reason == "confirmed_fix_affected_metric_final_drop":
        return "affected_metric_final_drop", "do_not_apply"
    if reason == "confirmed_fix_affected_metric_keep":
        return "affected_metric_keep", "queue_for_reviewed_branch"
    if reason == "confirmed_fix_affected_metric_dev_signal":
        return "confirmed_fix_with_dev_signal", "queue_for_reviewed_branch"
    if row.get("maintenance_keep") or reason == "maintenance_keep":
        return "maintenance_keep", "review_maintenance_patch"
    if row.get("metric_signal_keep") or reason == "confirmed_small_metric_keep":
        return "confirmed_small_metric_keep", "review_metric_patch"

    reproduction = row.get("reproduction") or {}
    reproduced = reproduction.get("stock") == "failed_as_predicted"
    resolved = reproduction.get("patched") == "resolved"
    correctness_confirmed = reproduced and resolved and bool(test_gate_passed)

    if row.get("keep"):
        stock = row.get("stock") or {}
        patched = row.get("patched") or {}
        recovered = any(
            isinstance(before, dict)
            and before.get("status") == "crash"
            and isinstance(patched.get(task_id), dict)
            and patched[task_id].get("status") == "ok"
            for task_id, before in stock.items()
        )
        if recovered:
            return "functional_recovery", "queue_for_reviewed_branch"
        return "quality_keep_dev", "confirm_dev_then_final"
    if correctness_confirmed:
        return "correctness_keep", "review_correctness_patch"
    if row.get("target_delta") == 0 or "below per-task threshold" in reason:
        return "metric_neutral_unverified", "do_not_apply"
    return "rejected", "do_not_apply"


def append_finding(
    path: Path,
    *,
    run_number: int,
    run_id: str,
    source_commit: str,
    source_hash: str,
    workspace: Path,
    row: dict[str, Any],
) -> None:
    outcome, recommendation = classify_finding(row)
    existing = _rows(path)
    successful = {
        "correctness_keep",
        "functional_recovery",
        "final_keep",
        "maintenance_keep",
        "affected_metric_keep",
        "confirmed_fix_with_dev_signal",
        "confirmed_small_metric_keep",
    }
    lead = row.get("lead") if isinstance(row.get("lead"), dict) else {}
    patch_hash = str(row.get("patch_hash") or "")
    repeated_patch = bool(
        patch_hash
        and any(
            item.get("record_type") == "finding"
            and item.get("outcome") in successful
            and str(item.get("patch_hash") or "") == patch_hash
            for item in existing
        )
    )
    known_contract = str(lead.get("channel") or "") == "public_contract"
    resumed = bool(row.get("resumed_branch"))
    confirmed = outcome in successful
    defect_id = _defect_id(row)
    repeated_defect = bool(
        defect_id
        and any(
            item.get("record_type") == "finding"
            and str((item.get("novelty") or {}).get("defect_id") or _defect_id(item))
            == defect_id
            and bool((item.get("novelty") or {}).get("confirmed", True))
            for item in existing
        )
    )
    repeated = repeated_patch or repeated_defect
    confirmed_defect = bool(confirmed and defect_id)
    novelty = (
        "not_confirmed"
        if not confirmed
        else "known_contract_check"
        if known_contract and confirmed_defect
        else "repeat"
        if repeated
        else "continuation"
        if resumed
        else "new_unique"
        if confirmed_defect
        else "confirmed_non_defect"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "finding",
        "event": "finding",
        "run_number": run_number,
        "run_id": run_id,
        "workspace": str(workspace.resolve()),
        "source_commit": source_commit,
        "source_hash": source_hash,
        "score_protocol_hash": row.get("score_protocol_hash") or "",
        "evaluation_protocol_hash": row.get("evaluation_protocol_hash") or "",
        "hypothesis_id": row.get("hypothesis_id") or None,
        "revision": row.get("revision"),
        "lead": row.get("lead"),
        "localization": {
            "static_rank": row.get("static_rank"),
            "final_rank": row.get("final_rank"),
            "rank_delta": row.get("rank_delta"),
            "picked_by_llm": row.get("picked_by_llm"),
        },
        "candidate_id": row.get("candidate"),
        "edits": row.get("edits") or [],
        "diff": row.get("diff") or "",
        "patch_hash": row.get("patch_hash") or "",
        "behavior_probe": row.get("behavior_probe") or {},
        "affected_metric": row.get("affected_metric") or {},
        "reproduction": row.get("reproduction")
        or {
            "status": "not_recorded",
            "stock": None,
            "patched": None,
        },
        "tests": {
            "status": row.get("fedot_test_status"),
            "gate_passed": _test_gate_passed(row),
            "failed_nodes": row.get("fedot_test_failures") or [],
            "exit_code": row.get("fedot_test_exit_code"),
            "duration_s": row.get("fedot_test_duration_s"),
            "cmd": row.get("fedot_test_cmd") or "",
            "output_tail": row.get("fedot_test_output_tail") or "",
        },
        "dev": {
            "keep": bool(row.get("keep")),
            "reason": row.get("reason"),
            "target_delta": row.get("target_delta"),
            "regression_deltas": row.get("regression_deltas") or {},
            "stock": row.get("stock"),
            "patched": row.get("patched"),
        },
        "feedback": row.get("feedback") or "",
        "outcome": outcome,
        "application_recommendation": recommendation,
        "novelty": {
            "category": novelty,
            "confirmed": confirmed,
            "confirmed_defect": confirmed_defect,
            "defect_id": defect_id,
            "new_unique": novelty == "new_unique",
            "repeat": repeated,
            "known_contract_check": bool(known_contract and confirmed_defect),
            "continuation": resumed,
        },
    }
    append_journal(path, payload)


def append_configuration_trials(
    path: Path,
    *,
    run_number: int,
    run_id: str,
    source_hash: str,
    score_protocol_hash: str = "",
    evaluation_protocol_hash: str = "",
    workspace: Path,
    trials: list[dict],
) -> None:
    """Persist every deterministic trial so later campaigns never repay it."""

    for trial in trials:
        patch_hash = str(trial.get("patch_hash") or "").strip()
        if not patch_hash:
            continue
        append_journal(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "configuration_trial",
                "event": "configuration_trial",
                "run_number": run_number,
                "run_id": run_id,
                "workspace": str(workspace.resolve()),
                "source_hash": source_hash,
                "score_protocol_hash": score_protocol_hash,
                "evaluation_protocol_hash": evaluation_protocol_hash,
                "patch_hash": patch_hash,
                "candidate_id": trial.get("candidate"),
                "variant": trial.get("variant") or {},
                "stage": trial.get("stage"),
                "reason": trial.get("reason")
                or (trial.get("full_decision") or {}).get("reason")
                or (trial.get("quick_decision") or {}).get("reason"),
                # Keep structured DEV feedback, not only its prose summary.
                # Later campaigns can refine a promising configuration without
                # replaying the same experiment or parsing model-generated text.
                "quick_decision": trial.get("quick_decision") or {},
                "full_decision": trial.get("full_decision") or {},
            },
        )


def append_final_outcome(
    path: Path,
    *,
    run_number: int,
    run_id: str,
    candidate_id: str,
    workspace: Path,
    decision: dict[str, Any],
    patch_hash: str = "",
    source_hash: str = "",
    score_protocol_hash: str = "",
    evaluation_protocol_hash: str = "",
) -> None:
    """Append the terminal FINAL verdict that supersedes a DEV-only finding.

    The candidate finding is intentionally written before FINAL.  Without this
    append-only correction the research dataset keeps reporting
    ``quality_keep_dev / confirm_dev_then_final`` after the campaign has already
    rejected or accepted the candidate on held-out data.
    """

    if any(
        row.get("record_type") == "rejudge"
        and row.get("run_id") == run_id
        and row.get("candidate_id") == candidate_id
        and row.get("final")
        for row in _rows(path)
    ):
        return
    infrastructure = bool(decision.get("infrastructure_error"))
    keep = bool(decision.get("keep")) and not infrastructure
    if infrastructure:
        outcome = "final_infrastructure_error"
        recommendation = "retry_after_infrastructure_fix"
    elif keep:
        outcome = "final_keep"
        recommendation = "review_winner_patch"
    else:
        outcome = "final_drop"
        recommendation = "do_not_apply"
    append_journal(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "rejudge",
            "event": "final_outcome",
            "run_number": run_number,
            "run_id": run_id,
            "workspace": str(workspace.resolve()),
            "candidate_id": candidate_id,
            "patch_hash": patch_hash,
            "source_hash": source_hash,
            "score_protocol_hash": score_protocol_hash,
            "evaluation_protocol_hash": evaluation_protocol_hash,
            "outcome": outcome,
            "application_recommendation": recommendation,
            "final": decision,
        },
    )


def append_semantic_duplicate_outcome(
    path: Path,
    *,
    run_number: int,
    run_id: str,
    candidate_id: str,
    workspace: Path,
    duplicate_of: dict[str, str],
    reason: str,
) -> None:
    """Correct a claimed finding that repeats an earlier defect family."""

    append_journal(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "rejudge",
            "event": "semantic_duplicate_outcome",
            "run_number": run_number,
            "run_id": run_id,
            "workspace": str(workspace.resolve()),
            "candidate_id": candidate_id,
            "outcome": "semantic_duplicate",
            "application_recommendation": "do_not_count_as_new",
            "duplicate_of": duplicate_of,
            "reason": reason,
        },
    )


def append_dev_rejudge(
    path: Path,
    *,
    run_number: int,
    run_id: str,
    candidate_id: str,
    patch_hash: str,
    source_hash: str,
    workspace: Path,
    decision: dict[str, Any],
    reason: str,
    score_protocol_hash: str = "",
    evaluation_protocol_hash: str = "",
) -> None:
    """Append a corrected DEV verdict without rewriting historical evidence.

    A controller/evaluator bug can make an earlier finding invalid.  The old row
    remains auditable, while replay and exact-patch feedback use this later,
    structured measurement as the effective outcome.
    """

    infrastructure = bool(decision.get("infrastructure_error"))
    keep = bool(decision.get("keep")) and not infrastructure
    decision_reason = str(decision.get("reason") or "")
    if infrastructure:
        outcome = "infrastructure_error"
        recommendation = "retry_after_infrastructure_fix"
    elif (
        decision.get("metric_signal_keep")
        or decision_reason == "confirmed_small_metric_keep"
    ):
        outcome = "confirmed_small_metric_keep"
        recommendation = "review_metric_patch"
    elif keep:
        outcome = "quality_keep_dev"
        recommendation = "confirm_dev_then_final"
    elif decision_reason.startswith("regression"):
        outcome = "rejected_dev_regression"
        recommendation = "do_not_apply"
    else:
        outcome = "metric_neutral_unverified"
        recommendation = "do_not_apply"
    append_journal(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "rejudge",
            "event": "dev_rejudge",
            "run_number": run_number,
            "run_id": run_id,
            "workspace": str(workspace.resolve()),
            "candidate_id": candidate_id,
            "patch_hash": patch_hash,
            "source_hash": source_hash,
            "score_protocol_hash": score_protocol_hash,
            "evaluation_protocol_hash": evaluation_protocol_hash,
            "outcome": outcome,
            "application_recommendation": recommendation,
            "supersedes": {"stage": "dev", "reason": reason},
            "dev": {
                "keep": keep,
                "reason": decision_reason,
                "target_delta": decision.get("target_delta"),
                "regression_deltas": decision.get("regression_deltas") or {},
                "infrastructure_error": infrastructure,
                "metric_signal_keep": bool(decision.get("metric_signal_keep")),
            },
        },
    )


def end_run(
    path: Path,
    *,
    run_number: int,
    run_id: str,
    decision: dict[str, Any],
    llm_usage: dict[str, Any],
    immutable_source: bool,
) -> None:
    append_journal(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "run",
            "event": "run_end",
            "run_number": run_number,
            "run_id": run_id,
            "decision": decision,
            "llm_usage": llm_usage,
            "immutable_source": immutable_source,
        },
    )


def import_workspace(workspace: Path, path: Path) -> dict[str, Any]:
    """Import a completed or interrupted historical campaign once."""

    workspace = resolve_run_workspace(workspace)
    summary_path = workspace / "campaign_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    trace_rows = _rows(workspace / "trace.jsonl")
    start = next(
        (row for row in trace_rows if row.get("event") == "campaign_start"), {}
    )
    run_id = str(summary.get("run_id") or start.get("run_id") or workspace.name)
    if has_run(path, run_id):
        return {"imported": False, "reason": "duplicate", "run_id": run_id}
    config = summary.get("campaign_config") or {
        "lift_ids": start.get("lift_ids") or [],
        "protect_ids": start.get("protect_ids") or [],
        **(start.get("limits") or {}),
    }
    run_number = begin_run(
        path,
        run_id=run_id,
        workspace=workspace,
        source=Path(summary.get("source") or "."),
        source_commit=str(summary.get("source_commit") or "unknown"),
        source_hash=str(
            summary.get("source_hash_before") or start.get("source_hash") or "unknown"
        ),
        campaign_config=config,
    )
    decisions = 0
    for row in _rows(workspace / "journal.jsonl"):
        if row.get("event") != "decision" or "revision" not in row:
            continue
        append_finding(
            path,
            run_number=run_number,
            run_id=run_id,
            source_commit=str(summary.get("source_commit") or "unknown"),
            source_hash=str(
                summary.get("source_hash_before")
                or start.get("source_hash")
                or "unknown"
            ),
            workspace=workspace,
            row=row,
        )
        decisions += 1
    configuration_trials = _rows(workspace / "configuration_trials.jsonl")
    append_configuration_trials(
        path,
        run_number=run_number,
        run_id=run_id,
        source_hash=str(
            summary.get("source_hash_before") or start.get("source_hash") or "unknown"
        ),
        evaluation_protocol_hash=str(config.get("evaluation_protocol_hash") or ""),
        score_protocol_hash=str(config.get("score_protocol_hash") or ""),
        workspace=workspace,
        trials=configuration_trials,
    )
    decision = summary.get("decision") or {
        "keep": False,
        "reason": "interrupted_before_campaign_summary",
        "infrastructure_error": True,
    }
    end_run(
        path,
        run_number=run_number,
        run_id=run_id,
        decision=decision,
        llm_usage=summary.get("llm_usage") or {},
        immutable_source=bool(summary.get("immutable_source", False)),
    )
    return {
        "imported": True,
        "run_number": run_number,
        "run_id": run_id,
        "findings": decisions,
        "configuration_trials": len(configuration_trials),
        "completed": summary_path.is_file(),
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    outcomes: dict[str, int] = {}
    for row in rows:
        outcome = row.get("outcome")
        if isinstance(outcome, str):
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    run_numbers = {
        row.get("run_number") for row in rows if isinstance(row.get("run_number"), int)
    }
    latest_rejudge = {
        str(row.get("candidate_id")): row
        for row in rows
        if row.get("record_type") == "rejudge" and row.get("candidate_id")
    }
    effective_rows = [
        row
        for row in rows
        if row.get("record_type") == "finding"
        and str(row.get("candidate_id") or "") not in latest_rejudge
    ] + list(latest_rejudge.values())
    effective_outcomes: dict[str, int] = {}
    for row in effective_rows:
        outcome = row.get("outcome")
        if isinstance(outcome, str):
            effective_outcomes[outcome] = effective_outcomes.get(outcome, 0) + 1
    return {
        "path": str(path.resolve()),
        "runs": len(run_numbers),
        "findings": sum(row.get("record_type") == "finding" for row in rows),
        "rejudges": len(latest_rejudge),
        "outcomes": outcomes,
        "effective_outcomes": effective_outcomes,
    }

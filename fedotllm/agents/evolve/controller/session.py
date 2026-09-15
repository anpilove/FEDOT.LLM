"""Campaign shutdown, immutable-source check and summary artifact."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.storage.findings import end_run
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.storage.checkpoint import save_checkpoint
from fedotllm.agents.evolve.types import Decision


def _llm_attempt_summary(workspace: Path) -> dict:
    """Summarize attempted and successful provider/structured calls."""

    path = workspace / "llm_calls.jsonl"
    counts = {
        "provider_queries_started": 0,
        "provider_queries_succeeded": 0,
        "provider_queries_failed": 0,
        "provider_queries_unfinished": 0,
        "provider_retries": 0,
        "provider_local_rejections": 0,
        "structured_calls_started": 0,
        "structured_calls_succeeded": 0,
        "structured_calls_failed": 0,
        "structured_calls_unfinished": 0,
        "failure_reasons": {},
    }
    if not path.is_file():
        return counts
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return counts
    failures: dict[str, int] = {}
    rows: list[dict] = []
    for raw in lines:
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    has_query_start_events = any(row.get("event") == "llm_query_started" for row in rows)
    has_provider_attempt_events = any(
        row.get("event")
        in {"provider_attempt_started", "provider_attempt_result", "provider_local_rejection"}
        for row in rows
    )
    has_structured_start_events = any(
        row.get("event") == "llm_structured_started" for row in rows
    )
    for row in rows:
        event = row.get("event")
        status = row.get("status")
        if has_provider_attempt_events:
            if event == "provider_attempt_started":
                counts["provider_queries_started"] += 1
                counts["provider_retries"] += int(bool(row.get("is_retry")))
                continue
            if event == "provider_attempt_result":
                key = (
                    "provider_queries_succeeded"
                    if status == "ok"
                    else "provider_queries_failed"
                )
                counts[key] += 1
                if status != "ok":
                    reason = str(row.get("error_type") or "unknown")
                    failures[reason] = failures.get(reason, 0) + 1
                continue
            if event == "provider_local_rejection":
                counts["provider_local_rejections"] += 1
                reason = str(row.get("error_type") or "local_rejection")
                failures[reason] = failures.get(reason, 0) + 1
                continue
        if event == "llm_query_started":
            if not has_provider_attempt_events:
                counts["provider_queries_started"] += 1
            continue
        if event == "llm_structured_started":
            counts["structured_calls_started"] += 1
            continue
        if event == "llm_query":
            if has_provider_attempt_events:
                continue
            if not has_query_start_events:
                counts["provider_queries_started"] += 1
            key = "provider_queries_succeeded" if status == "ok" else "provider_queries_failed"
            counts[key] += 1
        elif event == "llm_structured_result":
            if not has_structured_start_events:
                counts["structured_calls_started"] += 1
            key = "structured_calls_succeeded" if status == "ok" else "structured_calls_failed"
            counts[key] += 1
        else:
            continue
        if status != "ok":
            reason = str(row.get("error_type") or "unknown")
            failures[reason] = failures.get(reason, 0) + 1
    counts["failure_reasons"] = failures
    counts["provider_queries_unfinished"] = max(
        0,
        counts["provider_queries_started"]
        - counts["provider_queries_succeeded"]
        - counts["provider_queries_failed"],
    )
    counts["structured_calls_unfinished"] = max(
        0,
        counts["structured_calls_started"]
        - counts["structured_calls_succeeded"]
        - counts["structured_calls_failed"],
    )
    return counts


def _run_finding_summary(findings_path: Path, run_id: str) -> dict[str, int]:
    counts = {
        "confirmed": 0,
        "new_unique": 0,
        "repeats": 0,
        "known_contract_checks": 0,
        "continuations": 0,
    }
    if not findings_path.is_file():
        return counts
    try:
        lines = findings_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return counts
    for raw in lines:
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if row.get("record_type") != "finding" or row.get("run_id") != run_id:
            continue
        novelty = row.get("novelty") or {}
        counts["confirmed"] += int(bool(novelty.get("confirmed")))
        counts["new_unique"] += int(bool(novelty.get("new_unique")))
        counts["repeats"] += int(bool(novelty.get("repeat")))
        counts["known_contract_checks"] += int(bool(novelty.get("known_contract_check")))
        counts["continuations"] += int(bool(novelty.get("continuation")))
    return counts


def _usage_row(client) -> dict:
    usage = getattr(client, "usage", None)
    return dict(usage) if isinstance(usage, dict) else {}


def finish_campaign(
    decision: Decision,
    *,
    source: Path,
    source_hash: str,
    source_commit: str | None,
    source_fingerprint_fn: Callable[[Path], str],
    scout_inference,
    verifier_inference,
    fixer_inference,
    workspace: Path,
    findings_path: Path,
    run_id: str,
    run_number: int,
    campaign_config: dict,
    previous_trace: str | None,
    previous_llm_audit: str | None,
) -> Decision:
    """Finalize one campaign and restore controller process environment."""

    current_hash = source_fingerprint_fn(source)
    if current_hash != source_hash:
        decision = Decision(
            keep=False,
            reason="immutable_source_changed",
            target_delta=None,
            stage="infrastructure",
            infrastructure_error=True,
        )

    clients = {
        "scout": scout_inference,
        "verifier": verifier_inference,
        "fixer": fixer_inference,
    }
    llm_usage: dict[str, dict] = {}
    seen: dict[int, str] = {}
    for stage, client in clients.items():
        identity = id(client)
        if client is not None and identity in seen:
            llm_usage[stage] = {"shared_with": seen[identity]}
            continue
        if client is not None:
            seen[identity] = stage
        llm_usage[stage] = _usage_row(client)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "source": str(source),
        "source_commit": source_commit,
        "source_hash_before": source_hash,
        "source_hash_after": current_hash,
        "immutable_source": current_hash == source_hash,
        "campaign_config": campaign_config,
        "llm_usage": llm_usage,
        "llm_attempts": _llm_attempt_summary(workspace),
        "finding_counts": _run_finding_summary(findings_path, run_id),
        "decision": asdict(decision),
        "artifacts": {
            "trace": "trace.jsonl",
            "llm_calls": "llm_calls.jsonl",
            "model_contract": "model_contract.json",
            "controller_journal": "journal.jsonl",
            "hypotheses": "hypotheses.jsonl",
            "scoreboard": "scoreboard.jsonl",
            "findings_dataset": str(findings_path.resolve()),
            "final_candidate_patch": (
                "final_candidate.patch"
                if (workspace / "final_candidate.patch").is_file()
                else None
            ),
            "winner_patch": (
                "winner.patch"
                if decision.final_keep is True and (workspace / "winner.patch").is_file()
                else None
            ),
            "correctness_patches": [
                str(path.relative_to(workspace))
                for path in sorted((workspace / "correctness_fixes").glob("*.patch"))
            ],
            "maintenance_patches": [
                str(path.relative_to(workspace))
                for path in sorted((workspace / "maintenance_fixes").glob("*.patch"))
            ],
        },
    }
    (workspace / "campaign_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_checkpoint(
        workspace,
        stage="campaign_complete",
        run_id=run_id,
        decision=asdict(decision),
        immutable_source=current_hash == source_hash,
        campaign_summary="campaign_summary.json",
    )
    append_journal(
        workspace / "trace.jsonl",
        {
            "event": "campaign_end",
            "run_id": run_id,
            "decision": asdict(decision),
            "llm_usage": llm_usage,
            "immutable_source": current_hash == source_hash,
        },
    )
    end_run(
        findings_path,
        run_number=run_number,
        run_id=run_id,
        decision=asdict(decision),
        llm_usage=llm_usage,
        immutable_source=current_hash == source_hash,
    )
    if previous_trace is None:
        os.environ.pop("EVOLVE_AGENT_TRACE", None)
    else:
        os.environ["EVOLVE_AGENT_TRACE"] = previous_trace
    if previous_llm_audit is None:
        os.environ.pop("EVOLVE_AGENT_LLM_AUDIT", None)
    else:
        os.environ["EVOLVE_AGENT_LLM_AUDIT"] = previous_llm_audit
    return decision

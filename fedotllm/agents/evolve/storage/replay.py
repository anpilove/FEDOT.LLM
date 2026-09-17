"""Replay a journal decision: exact cmd, log_tail, diff. Harness-only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


from fedotllm.agents.evolve.types import (
    PatchCandidate,
    PatchEdit,
    MatchSite,
    VerificationResult,
)
from fedotllm.agents.evolve.storage.journal import read_jsonl, resolve_run_workspace
from fedotllm.agents.evolve.storage.hypothesis import behavior_probe_fingerprint


def solved_lift_tasks_from_findings(
    findings_path: Path | None,
    *,
    source_hash: str,
) -> set[str]:
    """Return task identities already improved by a FINAL-confirmed patch.

    The behavior task, rather than a patch hash or source location, is the
    stable identity here. The same defect can be repaired in another method
    with completely different edits. Since every campaign starts from the same
    immutable source, such a task must not earn discovery credit twice.
    """

    if not findings_path or not findings_path.is_file() or not source_hash:
        return set()

    rows: list[dict[str, Any]] = []
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)

    compatible_runs = {
        str(row.get("run_id") or "")
        for row in rows
        if row.get("event") == "run_start"
        and str(row.get("source_hash") or "") == source_hash
    }
    latest_final: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "final_outcome":
            continue
        run_id = str(row.get("run_id") or "")
        if run_id in compatible_runs:
            latest_final[(run_id, str(row.get("candidate_id") or ""))] = row

    solved: set[str] = set()
    for row in latest_final.values():
        final = row.get("final")
        if not isinstance(final, dict) or not final.get("keep"):
            continue
        deltas = final.get("regression_deltas")
        if not isinstance(deltas, dict):
            continue
        for task_id, raw_delta in deltas.items():
            try:
                delta = float(raw_delta)
            except (TypeError, ValueError):
                continue
            if delta <= 0:
                continue
            try:
                from fedotllm.agents.evolve.evaluation.tasks import load_task

                spec = load_task(str(task_id))
                threshold = (
                    spec.min_delta * max(abs(spec.sentinel), 1e-12)
                    if spec.min_delta_mode == "relative"
                    else spec.min_delta
                )
            except (KeyError, ValueError, OSError):
                threshold = 1e-12
            if delta >= threshold:
                solved.add(str(task_id))
    return solved


def load_resume_branch(
    workspace: Path,
    *,
    candidate_id: str,
) -> dict[str, Any] | None:
    """Recover one agent-owned hypothesis branch from an interrupted run."""

    workspace = resolve_run_workspace(workspace)
    rows = []
    journal = workspace / "journal.jsonl"
    if not journal.is_file():
        return None
    for raw in journal.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    decision = next(
        (
            row
            for row in reversed(rows)
            if row.get("event") == "decision" and row.get("candidate") == candidate_id
        ),
        None,
    )
    candidate_payload: dict[str, Any] = {}
    candidate_path = workspace / "candidates" / candidate_id / "candidate.json"
    if candidate_path.is_file():
        try:
            loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                candidate_payload = loaded
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    lead_row = (
        decision.get("lead")
        if decision is not None and isinstance(decision.get("lead"), dict)
        else candidate_payload.get("lead")
    )
    edit_paths = {
        str(edit.get("file_path") or "")
        for edit in candidate_payload.get("edits") or ()
        if isinstance(edit, dict)
    }
    if not isinstance(lead_row, dict):
        matching = next(
            (
                row
                for row in reversed(rows)
                if row.get("event") == "verification"
                and isinstance(row.get("lead"), dict)
                and str(row["lead"].get("file_path") or "") in edit_paths
            ),
            None,
        )
        lead_row = matching.get("lead") if matching is not None else None
    if not isinstance(lead_row, dict):
        return None
    hypothesis_id = str(
        (decision or {}).get("hypothesis_id")
        or candidate_payload.get("hypothesis_id")
        or ""
    )
    verification_row = next(
        (
            row
            for row in reversed(rows)
            if row.get("event") == "verification"
            and hypothesis_id
            and str(row.get("hypothesis_id") or "") == hypothesis_id
        ),
        None,
    )
    if verification_row is None:
        decision_lead = lead_row
        verification_row = next(
            (
                row
                for row in reversed(rows)
                if row.get("event") == "verification"
                and isinstance(row.get("lead"), dict)
                and row["lead"].get("file_path") == decision_lead.get("file_path")
                and int(row["lead"].get("line") or 0)
                == int(decision_lead.get("line") or 0)
            ),
            None,
        )
    if verification_row is None:
        return None
    try:
        lead = MatchSite(
            channel=str(lead_row.get("channel") or "resume"),
            file_path=str(lead_row["file_path"]),
            line=int(lead_row["line"]),
            why=str(lead_row.get("why") or ""),
            evidence=tuple(str(item) for item in lead_row.get("evidence") or ()),
            signals=tuple(str(item) for item in lead_row.get("signals") or ()),
            mechanism=str(lead_row.get("mechanism") or ""),
            proposed_change=str(lead_row.get("proposed_change") or ""),
            expected_metric_effect=str(lead_row.get("expected_metric_effect") or ""),
            hypothesis_kind=str(lead_row.get("hypothesis_kind") or "quality"),
        )
        verification = VerificationResult(
            status=verification_row["status"],
            claim=str(verification_row.get("claim") or ""),
            expected=str(verification_row.get("expected") or ""),
            observed=str(verification_row.get("observed") or ""),
            reproduction_code=str(verification_row.get("reproduction_code") or ""),
            evidence=tuple(
                str(item) for item in verification_row.get("evidence") or ()
            ),
            detail=str(verification_row.get("detail") or ""),
            current_approach=str(verification_row.get("current_approach") or ""),
            proposed_approach=str(verification_row.get("proposed_approach") or ""),
            alternatives_considered=tuple(
                str(item)
                for item in verification_row.get("alternatives_considered") or ()
            ),
            generality=str(verification_row.get("generality") or ""),
            risks=tuple(str(item) for item in verification_row.get("risks") or ()),
            resolved_target=dict(verification_row.get("resolved_target") or {}),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        verification.status == "verified_bug"
        and not verification.reproduction_code
        and not verification.evidence
    ):
        # Legacy journals omitted evidence. Recover the controller marker only
        # when the preserved runtime lead reproduces the exact recorded crash.
        from fedotllm.agents.evolve.agents.verifier import (
            verification_from_observed_crash,
        )

        observed = verification_from_observed_crash(lead)
        if observed is not None and observed.observed == verification.observed:
            verification.evidence = observed.evidence
    candidate = None
    try:
        edits = [
            PatchEdit(
                file_path=str(item["file_path"]),
                old_code=str(item["old_code"]),
                new_code=str(item["new_code"]),
            )
            for item in candidate_payload.get("edits") or ()
            if isinstance(item, dict)
        ]
        if edits:
            candidate = PatchCandidate(
                candidate_id=candidate_id,
                edits=edits,
                rationale=str(candidate_payload.get("rationale") or ""),
                contract=str(candidate_payload.get("contract") or ""),
                behavior_probe=str(candidate_payload.get("behavior_probe") or ""),
            )
    except (KeyError, TypeError, ValueError):
        candidate = None
    diagnostics: list[str] = []
    candidate_dir = workspace / "candidates" / candidate_id
    for name in (
        "probe_preflight_failed.txt",
        "probe_repair_timeout.txt",
        "probe_repair_error.txt",
        "apply_failed.txt",
    ):
        path = candidate_dir / name
        if path.is_file():
            try:
                diagnostics.append(f"{name}:\n{path.read_text(encoding='utf-8')[-4_000:]}")
            except OSError:
                pass
    return {
        "lead": lead,
        "verification": verification,
        "candidate": candidate,
        "candidate_status": str(candidate_payload.get("status") or ""),
        "resume_diagnostic": "\n\n".join(diagnostics),
        "patch_hash": str((decision or {}).get("patch_hash") or ""),
        "source_hash": str((decision or {}).get("source_hash") or ""),
        "candidate_id": candidate_id,
        "hypothesis_id": hypothesis_id,
    }


def _first_word(text: str) -> str:
    words = text.strip().split()
    return words[0] if words else ""


def semantic_site_id(lead: MatchSite | dict[str, Any]) -> str:
    """Stable identity for a site whose meaningful unit is not one line.

    Shared defaults JSON contains many independent operation blocks.  The LLM
    may select a parameter line while the catalog points at the block opening;
    both must deduplicate as the same operation without hiding sibling blocks.
    """

    if isinstance(lead, MatchSite):
        file_path = lead.file_path
        line = lead.line
        why = lead.why
        evidence = lead.evidence
    else:
        file_path = str(lead.get("file_path") or "")
        line = int(lead.get("line") or 0)
        why = str(lead.get("why") or "")
        evidence = tuple(str(item) for item in (lead.get("evidence") or ()))
    normalized = Path(file_path).as_posix()
    for item in evidence:
        if item.startswith("catalog semantic site:"):
            return item.partition(":")[2].strip()
    if normalized.endswith("/default_operation_params.json"):
        operation = ""
        for item in evidence:
            if item.startswith("executed operation:"):
                operation = _first_word(item.partition(":")[2])
                break
        if not operation:
            marker = "default parameters for executed operation "
            if marker in why:
                operation = _first_word(why.partition(marker)[2])
        if operation:
            return f"{normalized}#operation:{operation}"
    for item in evidence:
        if not item.startswith("executed lines in this symbol:"):
            continue
        ranges = item.partition(":")[2].strip()
        match = re.match(r"(\d+)", ranges)
        if match:
            return f"{normalized}#executed-range:{match.group(1)}"
    if why.startswith("executed symbol "):
        symbol = why.removeprefix("executed symbol ").strip()
        if symbol:
            return f"{normalized}#symbol:{symbol}"
    return f"{normalized}:{line}"


def load_replay(
    workspace: Path, *, candidate: str | None = None
) -> dict[str, Any] | None:
    workspace = resolve_run_workspace(workspace)
    path = workspace / "journal.jsonl"
    if not path.is_file():
        return None
    picked: dict[str, Any] | None = None
    for row in read_jsonl(path):
        if row.get("event") != "decision":
            continue
        if candidate and row.get("candidate") != candidate:
            continue
        picked = {
            "candidate": row.get("candidate"),
            "file": row.get("file"),
            "keep": row.get("keep"),
            "reason": row.get("reason"),
            "diff": row.get("diff") or "",
            "stock": _cmds(row.get("stock")),
            "patched": _cmds(row.get("patched")),
        }
    return picked


def tried_sites(workspace: Path) -> set[tuple[str, int]]:
    """Sites already attempted (scoreboard/journal decisions). Not scout-only rows."""

    seen: set[tuple[str, int]] = set()
    for name in ("scoreboard.jsonl", "journal.jsonl"):
        for row in read_jsonl(workspace / name):
            if row.get("event") not in {"attempt", "decision"}:
                continue
            if str(row.get("reason") or "").startswith((
                "verification_inconclusive", "verification_infrastructure_error"
            )):
                continue
            lead = row.get("lead")
            if not isinstance(lead, dict):
                continue
            file_path = lead.get("file_path")
            if not file_path:
                continue
            try:
                seen.add((str(file_path), int(lead.get("line") or 0)))
            except (TypeError, ValueError):
                continue
    return seen


def _recent_completed_findings(
    findings_path: Path | None,
    *,
    source_hash: str = "",
    evaluation_protocol_hash: str = "",
    score_protocol_hash: str = "",
    campaigns: int | None = 1,
) -> list[dict[str, Any]]:
    """Return exact sites explored by the most recent completed campaigns.

    This is a short cooldown, not a durable blacklist.  A campaign must have a
    matching ``run_start`` and a successful immutable ``run_end`` to count, so
    an interrupted process cannot hide locations from the next Scout run.  The
    selected campaign may contain no findings; in that case the cooldown is
    empty and sites from an older campaign become available again.
    Discovery memory can use the numeric workload identity independently of
    acceptance/test changes. Exact patch acceptance still uses its stricter hash.
    """

    if (
        (campaigns is not None and campaigns <= 0)
        or findings_path is None
        or not findings_path.is_file()
    ):
        return []
    rows: list[dict[str, Any]] = []
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)

    starts: dict[str, tuple[int, str]] = {}
    for row in rows:
        if row.get("record_type") != "run" or row.get("event") != "run_start":
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        campaign_config = row.get("campaign_config") or {}
        if score_protocol_hash:
            if campaign_config.get("score_protocol_hash") != score_protocol_hash:
                continue
        elif (
            evaluation_protocol_hash
            and campaign_config.get("evaluation_protocol_hash")
            != evaluation_protocol_hash
        ):
            continue
        run_id = str(row.get("run_id") or "")
        run_number = row.get("run_number")
        if not run_id or not isinstance(run_number, int):
            continue
        starts[run_id] = (run_number, run_id)

    completed = {
        str(row.get("run_id") or "")
        for row in rows
        if row.get("record_type") == "run"
        and row.get("event") == "run_end"
        and row.get("immutable_source") is True
    }
    eligible = sorted(
        (identity for run_id, identity in starts.items() if run_id in completed),
        reverse=True,
    )
    selected_run_ids = {run_id for _, run_id in eligible[:campaigns]}
    if not selected_run_ids:
        return []

    seen: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "finding":
            continue
        if str(row.get("run_id") or "") not in selected_run_ids:
            continue
        # Retrieval/probe/transport failures are not negative evidence about
        # the hypothesis. Keep their audit records, but do not cool down the
        # site or feed them back as completed investigations to Scout.
        if (row.get("reproduction") or {}).get("status") in {
            "inconclusive", "infrastructure_error"
        }:
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        if score_protocol_hash:
            if row.get("score_protocol_hash") != score_protocol_hash:
                continue
        elif (
            evaluation_protocol_hash
            and row.get("evaluation_protocol_hash") != evaluation_protocol_hash
        ):
            continue
        lead = row.get("lead")
        if not isinstance(lead, dict) or not lead.get("file_path"):
            continue
        try:
            int(lead.get("line") or 0)
            seen.append(row)
        except (TypeError, ValueError):
            continue
    return seen


def recent_completed_sites_from_findings(
    findings_path: Path | None, **filters
) -> set[tuple[str, int]]:
    """Exact-site cooldown for compatible completed source-search campaigns."""
    return {
        (str(row["lead"]["file_path"]), int(row["lead"].get("line") or 0))
        for row in _recent_completed_findings(findings_path, **filters)
    }


def recent_completed_semantic_sites_from_findings(
    findings_path: Path | None,
    **filters,
) -> set[str]:
    """Short cooldown for logical sites such as one operation's JSON block."""

    return {
        semantic_site_id(row["lead"])
        for row in _recent_completed_findings(findings_path, **filters)
    }


def confirmed_hypotheses_from_findings(
    findings_path: Path | None,
) -> list[dict[str, Any]]:
    """Version-independent defect memory without blacklisting code locations."""

    if not findings_path or not findings_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    latest_rejudge = {
        str(row.get("candidate_id")): row
        for row in rows
        if row.get("record_type") == "rejudge" and row.get("candidate_id")
    }
    successful = {
        "correctness_keep",
        "confirmed_fix_metric_neutral",
        "functional_recovery",
        "final_keep",
        "maintenance_keep",
        "affected_metric_keep",
        "confirmed_fix_with_dev_signal",
        "confirmed_small_metric_keep",
    }
    hypotheses: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in reversed(rows):
        if row.get("record_type") != "finding":
            continue
        candidate_id = str(row.get("candidate_id") or "")
        effective = latest_rejudge.get(candidate_id, row)
        if effective.get("outcome") not in successful:
            continue
        # Append-only corrections may supply metadata that was absent from the
        # original finding while preserving its immutable historical record.
        lead = effective.get("lead") or row.get("lead")
        if not isinstance(lead, dict) or not lead.get("file_path"):
            continue
        hypothesis = {
            "file_path": str(lead["file_path"]),
            "line": int(lead.get("line") or 0),
            "mechanism": str(lead.get("mechanism") or "")[:800],
            "proposed_change": str(lead.get("proposed_change") or "")[:800],
            "history_kind": "confirmed_defect",
        }
        evidence = lead.get("evidence") or ()
        contract_id = next(
            (
                str(item).partition(":")[2].strip()
                for item in evidence
                if str(item).startswith("observed contract id:")
            ),
            "",
        )
        if contract_id:
            hypothesis["contract_id"] = contract_id
        identity = json.dumps(hypothesis, sort_keys=True)
        if identity not in seen:
            hypotheses.append(hypothesis)
            seen.add(identity)
    return hypotheses[:32]


def recent_completed_hypotheses_from_findings(
    findings_path: Path | None, **filters
) -> list[dict[str, Any]]:
    """Source-only proposal memory; never include scores, verdicts or FINAL feedback."""
    # Sites become eligible again after a short cooldown, but their proposals
    # remain context. Remembering a proposal does not blacklist its operation.
    filters.setdefault("campaigns", None)
    hypotheses = []
    seen = set()
    for row in reversed(_recent_completed_findings(findings_path, **filters)):
        lead = row["lead"]
        hypothesis = {
            "file_path": str(lead["file_path"]),
            "line": int(lead.get("line") or 0),
            "mechanism": str(lead.get("mechanism") or "")[:800],
            "proposed_change": str(lead.get("proposed_change") or "")[:800],
        }
        if not (hypothesis["mechanism"] or hypothesis["proposed_change"]):
            continue
        identity = json.dumps(hypothesis, sort_keys=True)
        if identity not in seen:
            hypotheses.append(hypothesis)
            seen.add(identity)
    return hypotheses[:32]


def _matching_patch_findings(
    findings_path: Path | None,
    *,
    source_hash: str = "",
    evaluation_protocol_hash: str = "",
):
    """Read source/protocol-compatible candidate outcomes."""

    if findings_path is None or not findings_path.is_file():
        return
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("record_type") not in {
            "finding",
            "configuration_trial",
            "rejudge",
        }:
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        if (
            evaluation_protocol_hash
            and row.get("evaluation_protocol_hash") != evaluation_protocol_hash
        ):
            continue
        patch_hash = str(row.get("patch_hash") or "").strip()
        if patch_hash:
            yield patch_hash, row


def _probe_only_rejection(row: dict) -> bool:
    dev = row.get("dev") or {}
    probe = row.get("behavior_probe") or {}
    if not isinstance(dev, dict) or not isinstance(probe, dict):
        return False
    return (
        str(dev.get("reason") or "").startswith("behavior_probe_")
        and dev.get("patched") is None
        and dev.get("target_delta") is None
        and probe.get("status") in {"missing", "invalid", "patched_error"}
    )


def _cheap_screen_not_quality_verdict(row: dict) -> bool:
    """Historical toy δ=0 / probe no_change must stay hour-queue eligible."""

    from fedotllm.agents.evolve.controller.quality_queue import (
        cheap_screen_not_quality_verdict,
    )

    probe = row.get("behavior_probe") if isinstance(row.get("behavior_probe"), dict) else {}
    dev = row.get("dev") if isinstance(row.get("dev"), dict) else {}
    return cheap_screen_not_quality_verdict(
        probe_status=str(probe.get("status") or ""),
        reason=str(dev.get("reason") or row.get("reason") or ""),
        target_delta=dev.get("target_delta") if "target_delta" in dev else row.get("target_delta"),
        stage=str(dev.get("stage") or row.get("stage") or ""),
    )


def tried_patch_hashes_from_findings(
    findings_path: Path | None,
    *,
    source_hash: str = "",
    evaluation_protocol_hash: str = "",
) -> set[str]:
    """Deduplicate source trials, allowing repair of an uninformative probe.

    Cheap 100-tree / bit-identical δ=0 / probe ``no_change`` are not durable
    quality verdicts, so they do not consume hour-queue eligibility.
    """

    return {
        patch_hash
        for patch_hash, row in _matching_patch_findings(
            findings_path,
            source_hash=source_hash,
            evaluation_protocol_hash=evaluation_protocol_hash,
        )
        if not _probe_only_rejection(row) and not _cheap_screen_not_quality_verdict(row)
    }


def rejected_probe_hashes_from_findings(
    findings_path: Path | None,
    *,
    source_hash: str = "",
    evaluation_protocol_hash: str = "",
) -> dict[str, set[str]]:
    """Remember failed (patch, probe) pairs without ruling out the source edit."""

    rejected: dict[str, set[str]] = {}
    for patch_hash, row in _matching_patch_findings(
        findings_path,
        source_hash=source_hash,
        evaluation_protocol_hash=evaluation_protocol_hash,
    ):
        if _probe_only_rejection(row):
            code = row["behavior_probe"].get("code")
            if code is not None:
                rejected.setdefault(patch_hash, set()).add(
                    behavior_probe_fingerprint(code)
                )
    return rejected


def patch_feedback_from_findings(
    findings_path: Path | None,
    *,
    source_hash: str,
    patch_hash: str,
    evaluation_protocol_hash: str = "",
) -> str:
    """Return compact measured feedback for an exact historical patch.

    The model does not receive old free-form hypotheses. It receives only the
    controller-owned verdict and the exact edits it just independently proposed,
    allowing a new revision without paying to rerun the same experiment.
    """

    if not findings_path or not findings_path.is_file() or not patch_hash:
        return ""
    matched: dict[str, Any] | None = None
    matched_edits: list[dict[str, Any]] = []
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("record_type") not in {"finding", "rejudge"}:
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        if (
            evaluation_protocol_hash
            and row.get("evaluation_protocol_hash") != evaluation_protocol_hash
        ):
            continue
        if row.get("patch_hash") == patch_hash:
            # A later diagnostic failure cannot erase an actual source verdict.
            if (
                matched is None
                or not _probe_only_rejection(row)
                or _probe_only_rejection(matched)
            ):
                matched = row
            if row.get("edits"):
                matched_edits = list(row.get("edits") or [])
    if matched is None:
        return ""
    dev = matched.get("dev") or {}
    regressions = {
        key: value
        for key, value in (dev.get("regression_deltas") or {}).items()
        if value not in (None, 0, 0.0)
    }
    edits = matched.get("edits") or matched_edits
    edit_summary = "\n".join(
        f"EDIT {index} {edit.get('file_path')}:\n"
        f"SEARCH:\n{str(edit.get('old_code') or '')[:2_000]}\n"
        f"REPLACE:\n{str(edit.get('new_code') or '')[:2_000]}"
        for index, edit in enumerate(edits[:4], start=1)
        if isinstance(edit, dict)
    )
    if _probe_only_rejection(matched):
        return (
            "This patch was stopped at its causal probe, before metric evaluation.\n"
            f"Historical reason={dev.get('reason')}.\n"
            + json.dumps(matched.get("behavior_probe"), ensure_ascii=False)[:5_000]
            + "\nYou may keep these source edits and supply a materially different "
            "probe measuring their actual runtime effect. Do not repeat the same "
            "patch/probe pair or change source code merely to evade deduplication.\n"
            + edit_summary
        )[:12_000]
    return (
        "This exact normalized patch was already evaluated on the same frozen source.\n"
        f"Historical outcome={matched.get('outcome')}; reason={dev.get('reason')}; "
        f"target_delta={dev.get('target_delta')}; regressions="
        f"{json.dumps(regressions, ensure_ascii=False, sort_keys=True)}.\n"
        + (f"Exact duplicate edits:\n{edit_summary}\n" if edit_summary else "")
        + "Do not repeat or cosmetically rephrase these edits. Use the measured "
        "outcome to produce a materially different causal revision."
    )[:12_000]


def skip_tried(
    leads: list[MatchSite],
    workspace: Path,
) -> list[MatchSite]:
    """Skip exact locations attempted in this workspace only."""

    seen = tried_sites(workspace)
    if not seen:
        return list(leads)
    return [lead for lead in leads if (lead.file_path, lead.line) not in seen]


def _cmds(pack: Any) -> dict[str, dict[str, str]]:
    if not isinstance(pack, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for key, value in pack.items():
        if not isinstance(value, dict):
            continue
        out[str(key)] = {
            "cmd": str(value.get("cmd") or ""),
            "log_tail": str(value.get("log_tail") or ""),
            "status": str(value.get("status") or ""),
        }
    return out

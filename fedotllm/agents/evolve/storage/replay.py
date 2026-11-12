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


def _shared_protect_crash(row: dict[str, Any]) -> bool:
    """A verified patch dropped because stock already crashed the same task."""

    dev_reason = str((row.get("dev") or {}).get("reason") or "")
    return (
        row.get("outcome") == "infrastructure_error"
        and (row.get("tests") or {}).get("status") == "passed"
        and "timeout_or_invalid" in dev_reason
    )


def _stale_binary_key_unproved(row: dict[str, Any]) -> bool:
    """A reject that says a non-binary column is never added, without a probe.

    ``dict.update`` does not delete the key from the earlier transform, so
    ``_correct_binary_ids_features`` still snaps that column. The 06:52
    Europe/Moscow reject stated the opposite and never ran the two transforms.
    """

    reproduction = row.get("reproduction") or {}
    if not isinstance(reproduction, dict) or reproduction.get("status") != "rejected":
        return False
    detail = str(reproduction.get("detail") or "")
    return "more than two unique values" in detail and "never added" in detail


def _union_symbol_unresolved(row: dict[str, Any]) -> bool:
    """The short union name was rejected before any mixed-table probe ran.

    ``ImputationImplementation._categorical_numerical_union`` is the qualified
    symbol. A proposal that names only ``_categorical_numerical_union`` on that
    function's line is the same target. The 07:47 Europe/Moscow run was the
    third recorded inconclusive, which would close the line without a probe.
    """

    reproduction = row.get("reproduction") or {}
    if not isinstance(reproduction, dict) or reproduction.get("status") != "inconclusive":
        return False
    if "symbol and source line disagree" not in str(reproduction.get("observed") or ""):
        return False
    lead = row.get("lead") or {}
    text = str(lead.get("why") or "") + str(lead.get("mechanism") or "")
    return "_categorical_numerical_union" in text


def _all_nan_union_unproved(row: dict[str, Any]) -> bool:
    """A reject that treats a silent narrower array as disproof of the union.

    sklearn 1.3 SimpleImputer drops an all-NaN column. A numerical-only
    transform returns that narrower array without raising. The union with a
    categorical column raises ``ValueError: Shape of passed values is (4, 1),
    indices imply (4, 2)``. The 07:29 Europe/Moscow reject stopped on the
    numerical-only call.
    """

    reproduction = row.get("reproduction") or {}
    if not isinstance(reproduction, dict) or reproduction.get("status") != "rejected":
        return False
    detail = str(reproduction.get("detail") or "")
    return "all-NaN column completed without error" in detail


def _correctness_site(key: tuple[str, int], row: dict[str, Any]) -> MatchSite:
    lead = row["lead"]
    evidence = lead.get("evidence") or ()
    signals = lead.get("signals") or ()
    why = str(lead.get("why") or "")
    observed = str((row.get("reproduction") or {}).get("observed") or "")
    if "probe must import FEDOT" in observed and "import fedot" not in why.lower():
        why = (
            f"{why} The previous probe was rejected before it ran because it did "
            "not import fedot. The next reproduction must import fedot and assert "
            "the claimed behavior on the same fitted object."
        ).strip()
    if (
        "operation_info_by_id" in observed
        and "ImputationImplementation()" not in why
    ):
        why = (
            f"{why} The previous probe built an Operation and crashed with "
            "operation_info_by_id. Construct ImputationImplementation() directly. "
            "Fit once, transform a two-value numeric column, then transform that "
            "same instance when the column has more than two unique values."
        ).strip()
    if (
        "'list' object has no attribute 'size'" in observed
        and "_categorical_numerical_union" in (why + str(lead.get("mechanism") or ""))
        and "must be numpy arrays" not in why
    ):
        why = (
            f"{why} categorical_idx and numerical_idx must be numpy arrays. "
            "A Python list has no size attribute and crashes in fit before the union."
        ).strip()
    if (
        "add_operation" in observed
        and "_categorical_numerical_union" in (why + str(lead.get("mechanism") or ""))
        and "has no add_operation" not in why
    ):
        why = (
            f"{why} PipelineBuilder has no add_operation. Do not build a pipeline. "
            "Construct ImputationImplementation() and call fit and transform on one "
            "table with a numeric all-NaN column and a categorical column."
        ).strip()
    if (
        "symbol and source line disagree" in observed
        and "_categorical_numerical_union" in (why + str(lead.get("mechanism") or ""))
        and "columns=self.non_categorical_ids" not in why
    ):
        why = (
            f"{why} The previous probe named a symbol that does not contain the "
            "source line. Use symbol _categorical_numerical_union and the DataFrame "
            "line that passes columns=self.non_categorical_ids."
        ).strip()
    if _all_nan_union_unproved(row) and "indices imply" not in why:
        why = (
            f"{why} SimpleImputer drops an all-NaN column. A numerical-only "
            "transform can return a narrower array without raising, and that "
            "does not disprove the union. Construct ImputationImplementation(), "
            "fit and transform one table with a numeric all-NaN column and a "
            "categorical column, and assert the ValueError whose message contains "
            "indices imply."
        ).strip()
    if _stale_binary_key_unproved(row) and "does not delete a key" not in why:
        why = (
            f"{why} dict.update does not delete a key from an earlier transform. "
            "fit does not call _find_binary_features. A column with more than two "
            "unique values keeps the old key, and _correct_binary_ids_features still "
            "snaps it. The next reproduction must import fedot, fit once, transform "
            "a two-value column, then transform that same object when the column has "
            "more than two unique values, and assert the second result was not snapped "
            "to the first min and max."
        ).strip()
    return MatchSite(
        channel=str(lead.get("channel") or "llm"),
        file_path=key[0],
        line=key[1],
        why=why,
        evidence=tuple(str(item) for item in evidence),
        signals=tuple(str(item) for item in signals),
        mechanism=str(lead.get("mechanism") or ""),
        proposed_change=str(lead.get("proposed_change") or ""),
        expected_metric_effect=str(lead.get("expected_metric_effect") or ""),
        hypothesis_kind="correctness",
    )


def _supersedes_proven_patch(row: dict[str, Any]) -> bool:
    """A later decisive result replaces a patch that was only dropped on a crash."""

    reproduction = row.get("reproduction") or {}
    if reproduction.get("status") == "rejected":
        return True
    outcome = str(row.get("outcome") or "")
    if outcome in {
        "rejected",
        "correctness_keep",
        "correctness_keep_with_metric_signal",
    }:
        return True
    tests = row.get("tests") or {}
    return isinstance(tests, dict) and tests.get("status") not in {None, "", "passed"}


_MISSED_TARGET_MARKERS = (
    "did not execute the claimed target method",
    "without executing the claimed target method",
    "probe syntax error",
    "crashed before its contract assertion",
)
# Four misses stay open. The 05:19 Moscow attempt still died on an import
# the feedback described as a private entry point, so one more try remains.
# The fifth miss closes the line.
_MAX_OPEN_MISSED_TARGET_ATTEMPTS = 4


def _probe_missed_target(row: dict[str, Any]) -> bool:
    """A probe that never executed the claimed method is not a finished test."""

    reproduction = row.get("reproduction") or {}
    if not isinstance(reproduction, dict):
        return False
    if reproduction.get("status") != "inconclusive":
        return False
    if reproduction.get("target_reached") is True:
        return False
    text = f"{reproduction.get('detail') or ''}\n{reproduction.get('observed') or ''}"
    return any(marker in text for marker in _MISSED_TARGET_MARKERS)


def _correctness_resume_plan(
    findings_path: Path | None,
) -> list[tuple[MatchSite, dict[str, Any] | None]]:
    """Proven shared-crash patches first, then still-open inconclusive probes."""

    if findings_path is None or not findings_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("record_type") == "finding":
            rows.append(row)
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    inconclusive_counts: dict[tuple[str, int], int] = {}
    missed_counts: dict[tuple[str, int], int] = {}
    shared_crash_counts: dict[tuple[str, int], int] = {}
    proven_rows: dict[tuple[str, int], dict[str, Any]] = {}
    superseded: set[tuple[str, int]] = set()
    for row in rows:
        lead = row.get("lead")
        if not isinstance(lead, dict) or not lead.get("file_path"):
            continue
        if lead.get("hypothesis_kind") != "correctness":
            continue
        try:
            key = (str(lead["file_path"]), int(lead.get("line") or 0))
        except (TypeError, ValueError):
            continue
        latest[key] = row
        reproduction = row.get("reproduction") or {}
        if (
            row.get("outcome") == "unverified_no_patch"
            and reproduction.get("status") == "inconclusive"
        ):
            if _probe_missed_target(row):
                missed_counts[key] = missed_counts.get(key, 0) + 1
            else:
                inconclusive_counts[key] = inconclusive_counts.get(key, 0) + 1
        if _shared_protect_crash(row):
            shared_crash_counts[key] = shared_crash_counts.get(key, 0) + 1
            proven_rows[key] = row
            superseded.discard(key)
        elif key in proven_rows and _supersedes_proven_patch(row):
            superseded.add(key)
    proven: list[tuple[MatchSite, dict[str, Any]]] = []
    open_probes: list[tuple[MatchSite, None]] = []
    for key, row in reversed(list(latest.items())):
        proven_row = proven_rows.get(key)
        if (
            proven_row is not None
            and shared_crash_counts.get(key) == 1
            and key not in superseded
        ):
            proven.append((_correctness_site(key, proven_row), proven_row))
            continue
        reproduction = row.get("reproduction") or {}
        if (
            _stale_binary_key_unproved(row)
            or _all_nan_union_unproved(row)
            or _union_symbol_unresolved(row)
        ):
            open_probes.append((_correctness_site(key, row), None))
            continue
        if reproduction.get("status") == "rejected":
            continue
        if _probe_missed_target(row):
            # An older real inconclusive must not resume a line whose latest
            # probe still missed the method after the open-attempt cap.
            if 1 <= missed_counts.get(key, 0) <= _MAX_OPEN_MISSED_TARGET_ATTEMPTS:
                open_probes.append((_correctness_site(key, row), None))
            continue
        if (
            row.get("outcome") == "unverified_no_patch"
            and reproduction.get("status") == "inconclusive"
            and 1 <= inconclusive_counts.get(key, 0) <= 2
        ):
            open_probes.append((_correctness_site(key, row), None))
    return [*proven, *open_probes]


def scored_day_count_resume(
    findings_path: Path | None,
    *,
    source_hash: str = "",
) -> tuple[MatchSite, str] | None:
    """Resume a queued day-count edit whose probe never scored a model.

    Stop once any finding for the line already asks for that score.
    """

    from fedotllm.agents.evolve.controller.probes import mechanism_only_day_count

    if findings_path is None or not findings_path.is_file():
        return None
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("record_type") != "finding":
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        lead = row.get("lead")
        if not isinstance(lead, dict) or not lead.get("file_path"):
            continue
        try:
            key = (str(lead["file_path"]), int(lead.get("line") or 0))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)
    for key, rows in reversed(list(grouped.items())):
        if any(
            "does not show a model improved" in str(row.get("feedback") or "")
            for row in rows
        ):
            continue
        latest = rows[-1]
        reason = str((latest.get("dev") or {}).get("reason") or "")
        if reason != "queued_for_fedot_quality":
            continue
        edits = latest.get("edits") or []
        patch_text = "\n".join(
            str(item.get("new_code") or "")
            for item in edits
            if isinstance(item, dict)
        )
        note = mechanism_only_day_count(latest.get("behavior_probe") or {}, patch_text)
        if not note:
            continue
        lead = latest["lead"]
        site = MatchSite(
            channel=str(lead.get("channel") or "llm"),
            file_path=key[0],
            line=key[1],
            why=str(lead.get("why") or note),
            hypothesis_kind="quality",
        )
        return site, note
    return None


def whole_scale_resume(
    findings_path: Path | None,
    *,
    source_hash: str = "",
) -> tuple[MatchSite, str] | None:
    """Resume a site whose patch divided every converted column.

    The dividing edit can sit on an earlier finding. The latest row is often
    ``no_patch`` with empty edits. Resume once, until some finding already
    records the column-wise instruction.
    """

    from fedotllm.agents.evolve.controller.probes import (
        divides_whole_conversion,
        whole_conversion_scale_revision,
    )

    if findings_path is None or not findings_path.is_file():
        return None
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("record_type") != "finding":
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        lead = row.get("lead")
        if not isinstance(lead, dict) or not lead.get("file_path"):
            continue
        try:
            key = (str(lead["file_path"]), int(lead.get("line") or 0))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)
    for key, rows in reversed(list(grouped.items())):
        if any("only datetime columns" in str(row.get("feedback") or "") for row in rows):
            continue
        latest = rows[-1]
        reason = str((latest.get("dev") or {}).get("reason") or "")
        if reason != "no_patch" and not reason.startswith("fedot_tests_regressed"):
            continue
        dividing = None
        for row in reversed(rows):
            edits = row.get("edits") or []
            patch_text = "\n".join(
                str(item.get("new_code") or "")
                for item in edits
                if isinstance(item, dict)
            )
            if divides_whole_conversion(patch_text):
                dividing = row
                break
        if dividing is None:
            continue
        note = whole_conversion_scale_revision(
            dividing.get("behavior_probe") or {},
            "\n".join(
                str(item.get("new_code") or "")
                for item in (dividing.get("edits") or [])
                if isinstance(item, dict)
            ),
        )
        if not note:
            continue
        lead = latest["lead"] if isinstance(latest.get("lead"), dict) else dividing["lead"]
        site = MatchSite(
            channel=str(lead.get("channel") or "llm"),
            file_path=key[0],
            line=key[1],
            why=str(lead.get("why") or note),
            hypothesis_kind="quality",
        )
        return site, note
    return None


def narrow_datetime_resume(
    findings_path: Path | None,
    *,
    source_hash: str = "",
) -> tuple[MatchSite, str] | None:
    """One more fixer pass for a queued pure-datetime dtype branch.

    The 05:47 Moscow logit probe moved from 0.000 to 0.958 on a single
    datetime64 column. A mixed table never enters that branch. Resume once,
    with feedback that asks for a column-wise conversion.
    """

    from fedotllm.agents.evolve.controller.probes import narrow_datetime_revision

    if findings_path is None or not findings_path.is_file():
        return None
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("record_type") != "finding":
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        lead = row.get("lead")
        if not isinstance(lead, dict) or not lead.get("file_path"):
            continue
        try:
            key = (str(lead["file_path"]), int(lead.get("line") or 0))
        except (TypeError, ValueError):
            continue
        latest[key] = row
    for key, row in reversed(list(latest.items())):
        reason = str((row.get("dev") or {}).get("reason") or "")
        if reason != "queued_for_fedot_quality":
            continue
        if "single datetime64 column" in str(row.get("feedback") or ""):
            continue
        edits = row.get("edits") or []
        patch_text = "\n".join(
            str(item.get("new_code") or "") for item in edits if isinstance(item, dict)
        )
        note = narrow_datetime_revision(row.get("behavior_probe") or {}, patch_text)
        if not note:
            continue
        lead = row["lead"]
        site = MatchSite(
            channel=str(lead.get("channel") or "llm"),
            file_path=key[0],
            line=key[1],
            why=str(lead.get("why") or note),
            hypothesis_kind="quality",
        )
        return site, note
    return None


def unfinished_correctness_leads(
    findings_path: Path | None,
    *,
    limit: int = 2,
) -> list[MatchSite]:
    """Return correctness leads that still deserve a patch attempt.

    A verified patch dropped on one shared protect crash is resumed first,
    even when a later inconclusive probe was recorded for the same line.
    A second drop of the same kind is not resumed. One or two inconclusive
    attempts follow, behind that proven patch. A verifier rejection is
    finished even when the outcome field is still ``unverified_no_patch``.
    A third inconclusive attempt for the same line is not resumed.
    Probes that never executed the claimed method stay open through four
    misses, then close.
    """

    if limit <= 0:
        return []
    return [site for site, _row in _correctness_resume_plan(findings_path)[:limit]]


def _replay_pair(
    row: dict[str, Any],
) -> tuple[VerificationResult, PatchCandidate] | None:
    edits: list[PatchEdit] = []
    for item in row.get("edits") or []:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file_path") or "")
        old_code = str(item.get("old_code") or "")
        new_code = str(item.get("new_code") or "")
        if file_path and old_code and new_code and old_code != new_code:
            edits.append(PatchEdit(file_path, old_code, new_code))
    reproduction = row.get("reproduction") or {}
    code = str(reproduction.get("code") or "")
    if not edits or not code:
        return None
    alternatives = reproduction.get("alternatives_considered") or ()
    risks = reproduction.get("risks") or ()
    verification = VerificationResult(
        "verified_bug",
        claim=str(reproduction.get("claim") or ""),
        expected=str(reproduction.get("expected") or ""),
        observed=str(reproduction.get("observed") or ""),
        reproduction_code=code,
        detail="replayed a verified patch dropped on a shared protect crash",
        current_approach=str(reproduction.get("current_approach") or ""),
        proposed_approach=str(reproduction.get("proposed_approach") or ""),
        alternatives_considered=tuple(str(item) for item in alternatives),
        generality=str(reproduction.get("generality") or ""),
        risks=tuple(str(item) for item in risks),
    )
    candidate = PatchCandidate(
        candidate_id=str(row.get("candidate_id") or "replayed-correctness"),
        edits=edits,
        rationale=(
            "Replay the verified patch. The previous drop was a protect crash "
            "that stock already had."
        ),
    )
    return verification, candidate


def proven_correctness_replay(
    findings_path: Path | None,
) -> dict[tuple[str, int], tuple[VerificationResult, PatchCandidate]]:
    """Reapply a verified patch instead of spending another verifier budget.

    The returned candidate is the exact edit from the finding. The verifier
    result carries the probe that already showed stock failing and the patch
    resolving it, so the campaign can rerun that probe on the replayed edit.
    """

    replayed: dict[tuple[str, int], tuple[VerificationResult, PatchCandidate]] = {}
    for site, row in _correctness_resume_plan(findings_path):
        if row is None:
            continue
        pair = _replay_pair(row)
        if pair is not None:
            replayed[(site.file_path, site.line)] = pair
    return replayed


def closed_correctness_sites(
    findings_path: Path | None,
    *,
    source_hash: str = "",
) -> set[tuple[str, int]]:
    """Sites a new catalog must not pick again on this source.

    A rejected reproduction, a kept correctness patch, and a third inconclusive
    attempt are finished. One or two inconclusive attempts stay available.
    """

    if findings_path is None or not findings_path.is_file():
        return set()
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    inconclusive_counts: dict[tuple[str, int], int] = {}
    missed_counts: dict[tuple[str, int], int] = {}
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("record_type") != "finding":
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        lead = row.get("lead")
        if not isinstance(lead, dict) or lead.get("hypothesis_kind") != "correctness":
            continue
        if not lead.get("file_path"):
            continue
        try:
            key = (str(lead["file_path"]), int(lead.get("line") or 0))
        except (TypeError, ValueError):
            continue
        latest[key] = row
        reproduction = row.get("reproduction") or {}
        if (
            row.get("outcome") == "unverified_no_patch"
            and reproduction.get("status") == "inconclusive"
        ):
            if _probe_missed_target(row):
                missed_counts[key] = missed_counts.get(key, 0) + 1
            else:
                inconclusive_counts[key] = inconclusive_counts.get(key, 0) + 1
    closed: set[tuple[str, int]] = set()
    for key, row in latest.items():
        outcome = str(row.get("outcome") or "")
        reproduction = str((row.get("reproduction") or {}).get("status") or "")
        healthy_without_target = (
            reproduction == "rejected"
            and str((row.get("reproduction") or {}).get("detail") or "").startswith(
                "stock satisfied the proposed healthy contract"
            )
            and (row.get("reproduction") or {}).get("target_reached") is not True
        )
        if healthy_without_target:
            continue
        if (
            _stale_binary_key_unproved(row)
            or _all_nan_union_unproved(row)
            or _union_symbol_unresolved(row)
        ):
            continue
        if (
            _probe_missed_target(row)
            and missed_counts.get(key, 0) <= _MAX_OPEN_MISSED_TARGET_ATTEMPTS
        ):
            continue
        if outcome in {
            "correctness_keep",
            "correctness_keep_with_metric_signal",
            "rejected",
        } or reproduction == "rejected":
            closed.add(key)
        elif missed_counts.get(key, 0) >= _MAX_OPEN_MISSED_TARGET_ATTEMPTS + 1:
            closed.add(key)
        elif (
            outcome == "unverified_no_patch"
            and reproduction == "inconclusive"
            and inconclusive_counts.get(key, 0) >= 3
        ):
            closed.add(key)
    return closed


def closed_hunt_sites(
    findings_path: Path | None,
    *,
    source_hash: str = "",
) -> set[tuple[str, int]]:
    """Sites already judged on this source, including failed quality patches.

    A line whose patch failed the unit tests, or that the fixer refused, is
    not offered to the next catalog. Correctness keeps and exhausted probes
    stay closed as well.
    """

    closed = set(closed_correctness_sites(findings_path, source_hash=source_hash))
    if findings_path is None or not findings_path.is_file():
        return closed
    for raw in findings_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("record_type") != "finding":
            continue
        if source_hash and row.get("source_hash") != source_hash:
            continue
        lead = row.get("lead")
        if not isinstance(lead, dict) or not lead.get("file_path"):
            continue
        try:
            key = (str(lead["file_path"]), int(lead.get("line") or 0))
        except (TypeError, ValueError):
            continue
        tests = (row.get("tests") or {}).get("status")
        outcome = str(row.get("outcome") or "")
        reason = str((row.get("dev") or {}).get("reason") or "")
        if (
            tests == "test_failures"
            or outcome == "rejected_tests"
            or reason.startswith(("fedot_tests_regressed", "duplicate_patch", "cannot_fix"))
            or reason == "cannot_fix"
        ):
            closed.add(key)
    return closed


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
        if (
            not _probe_only_rejection(row)
            and not _cheap_screen_not_quality_verdict(row)
            and not _shared_protect_crash(row)
        )
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

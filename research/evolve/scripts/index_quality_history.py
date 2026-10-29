#!/usr/bin/env python3
"""Build a read-only index of historical quality queues and journals.

The index retains source paths and raw records. Journal rows are attached to a
proposal only when the row names an exact candidate ID in the same run; task
IDs are checked against that proposal's frozen scope. Unresolvable rows remain
explicitly unresolved instead of being paired by date, order, or task alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _journal_task_ids(row: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for value in row.get("task_ids") or []:
        if value:
            found.add(str(value))
    if row.get("task_id"):
        found.add(str(row["task_id"]))
    for job in row.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        task_id = job.get("task_id") or (job.get("spec") or {}).get("source_dataset")
        if task_id:
            found.add(str(task_id))
    return sorted(found)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_quality_history_index(
    workspaces: list[Path], output_path: Path
) -> dict[str, int]:
    """Index runs without changing their source queue, journal, or patch files."""

    output_rows: list[dict[str, Any]] = []
    linked_events = 0
    unresolved_events = 0
    malformed_records = 0
    proposals = 0

    for workspace in sorted({Path(path).resolve() for path in workspaces}, key=str):
        manifest_path = workspace / "run_manifest.json"
        manifest = _load_object(manifest_path) if manifest_path.is_file() else {}
        if manifest_path.is_file() and manifest is None:
            malformed_records += 1
            manifest = {}
        manifest = manifest or {}
        run_id = str(manifest.get("run_id") or workspace.name)
        queue_dir = workspace / "quality_queue"
        proposal_rows: list[dict[str, Any]] = []
        by_candidate: dict[str, list[dict[str, Any]]] = {}

        if queue_dir.is_dir():
            for proposal_path in sorted(queue_dir.glob("*.json")):
                proposal = _load_object(proposal_path)
                if proposal is None:
                    malformed_records += 1
                    output_rows.append(
                        {
                            "record_type": "unresolved_event",
                            "run_id": run_id,
                            "source_path": str(proposal_path),
                            "reason": "malformed_queue_record",
                            "raw_text": proposal_path.read_text(encoding="utf-8", errors="replace"),
                        }
                    )
                    unresolved_events += 1
                    continue
                candidate_id = str(proposal.get("candidate_revision_id") or proposal.get("candidate_id") or "")
                if not candidate_id:
                    output_rows.append(
                        {
                            "record_type": "unresolved_event",
                            "run_id": run_id,
                            "source_path": str(proposal_path),
                            "reason": "missing_candidate_id",
                            "raw_record": proposal,
                        }
                    )
                    unresolved_events += 1
                    continue
                proposal = {
                    "record_type": "proposal",
                    "run_id": run_id,
                    "run_workspace": str(workspace),
                    "registry_identity": str(manifest.get("registry_identity") or proposal.get("registry_identity") or ""),
                    "candidate_id": candidate_id,
                    "proposal_id": str(proposal.get("proposal_id") or candidate_id),
                    "task_ids": sorted({str(task_id) for task_id in proposal.get("task_ids") or []}),
                    "source_path": str(proposal_path),
                    "source_sha256": _sha256(proposal_path),
                    "patch_path": str((queue_dir / str(proposal.get("patch"))).resolve()) if proposal.get("patch") else None,
                    "patch_sha256": proposal.get("patch_sha256"),
                    "queue_record": proposal,
                    "journal_events": [],
                }
                proposal_rows.append(proposal)
                by_candidate.setdefault(candidate_id, []).append(proposal)
                proposals += 1

        journal_path = workspace / "quality_jobs.jsonl"
        if journal_path.is_file():
            for line_number, line in enumerate(journal_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_records += 1
                    output_rows.append(
                        {
                            "record_type": "unresolved_event",
                            "run_id": run_id,
                            "source_path": str(journal_path),
                            "source_line": line_number,
                            "reason": "malformed_journal_json",
                            "raw_text": line,
                        }
                    )
                    unresolved_events += 1
                    continue
                if not isinstance(row, dict):
                    malformed_records += 1
                    output_rows.append(
                        {
                            "record_type": "unresolved_event",
                            "run_id": run_id,
                            "source_path": str(journal_path),
                            "source_line": line_number,
                            "reason": "journal_record_not_object",
                            "raw_record": row,
                        }
                    )
                    unresolved_events += 1
                    continue

                candidate = row.get("candidate")
                candidate_id = row.get("candidate_id")
                if candidate and candidate_id and str(candidate) != str(candidate_id):
                    reason = "conflicting_candidate_ids"
                    candidate_key = None
                else:
                    candidate_key = str(candidate_id or candidate or "") or None
                    reason = "missing_candidate_id" if candidate_key is None else ""
                matches = by_candidate.get(candidate_key or "", [])
                if not reason and len(matches) != 1:
                    reason = "candidate_not_in_run_queue" if not matches else "ambiguous_candidate_in_run_queue"

                task_ids = _journal_task_ids(row)
                if not reason and task_ids:
                    allowed = set(matches[0]["task_ids"])
                    if not set(task_ids).issubset(allowed):
                        reason = "task_not_in_frozen_scope"
                if reason:
                    output_rows.append(
                        {
                            "record_type": "unresolved_event",
                            "run_id": run_id,
                            "candidate_id": candidate_key,
                            "task_ids": task_ids,
                            "source_path": str(journal_path),
                            "source_line": line_number,
                            "reason": reason,
                            "raw_record": row,
                        }
                    )
                    unresolved_events += 1
                    continue

                matches[0]["journal_events"].append(
                    {
                        "source_path": str(journal_path),
                        "source_line": line_number,
                        "task_ids": task_ids,
                        "link_level": "candidate_and_task" if task_ids else "candidate_only",
                        "raw_record": row,
                    }
                )
                linked_events += 1

        output_rows.extend(proposal_rows)

    output_rows.sort(
        key=lambda row: (
            str(row.get("run_id") or ""),
            str(row.get("record_type") or ""),
            str(row.get("candidate_id") or ""),
            int(row.get("source_line") or 0),
            str(row.get("source_path") or ""),
        )
    )
    _append_jsonl(output_path, output_rows)
    return {
        "proposals": proposals,
        "linked_events": linked_events,
        "unresolved_events": unresolved_events,
        "malformed_records": malformed_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, action="append", default=[])
    parser.add_argument(
        "--campaign-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively discover workspace folders containing quality_queue/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Optional JSON summary output")
    args = parser.parse_args()
    workspaces = set(args.workspace)
    for root in args.campaign_root:
        workspaces.update(
            path.parent for path in root.rglob("quality_queue") if path.is_dir()
        )
    if not workspaces:
        parser.error("provide at least one --workspace or --campaign-root")
    summary = build_quality_history_index(list(workspaces), args.output)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

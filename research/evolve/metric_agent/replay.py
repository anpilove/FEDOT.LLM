"""Replay a journal decision: exact cmd, log_tail, diff. Harness-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from research.evolve.metric_agent.types import Lead


def load_replay(workspace: Path, *, candidate: str | None = None) -> dict[str, Any] | None:
    path = workspace / "journal.jsonl"
    if not path.is_file():
        return None
    picked: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
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
        path = workspace / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") not in {"attempt", "decision"}:
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


def skip_tried(leads: list[Lead], workspace: Path) -> list[Lead]:
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

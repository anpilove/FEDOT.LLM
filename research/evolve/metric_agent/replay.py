"""Replay a journal decision: exact cmd, log_tail, diff. Harness-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

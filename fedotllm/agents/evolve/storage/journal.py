from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_run_workspace(workspace: Path) -> Path:
    """Accept either a campaign directory or the root printed by CLI examples."""
    workspace = workspace.resolve()
    marker = workspace / "latest_run.json"
    if not marker.is_file():
        return workspace
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        run_id = str(payload["run_id"])
        remote = Path(payload["workspace"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        # A half-written pointer must not hide the directory the user passed.
        return workspace
    # Prefer a local child so archived campaigns remain readable after moving.
    local = workspace / "runs" / run_id
    target = local if local.is_dir() else remote
    if not target.is_dir():
        raise FileNotFoundError(f"latest campaign directory is missing: {target}")
    return target.resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every well-formed JSON object row; missing file or bad rows yield nothing."""
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


def append_journal(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("schema_version", 1)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via a sibling temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


def write_artifact(dir_path: Path, name: str, text: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    dest = dir_path / name
    dest.write_text(text, encoding="utf-8")
    return dest

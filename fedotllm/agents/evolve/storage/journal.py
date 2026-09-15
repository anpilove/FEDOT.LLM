from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_run_workspace(workspace: Path) -> Path:
    """Accept either a campaign directory or the root printed by CLI examples."""
    workspace = workspace.resolve()
    marker = workspace / "latest_run.json"
    if not marker.is_file():
        return workspace
    payload = json.loads(marker.read_text(encoding="utf-8"))
    # Prefer a local child so archived campaigns remain readable after moving.
    local = workspace / "runs" / str(payload["run_id"])
    target = local if local.is_dir() else Path(payload["workspace"])
    if not target.is_dir():
        raise FileNotFoundError(f"latest campaign directory is missing: {target}")
    return target.resolve()


def append_journal(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("schema_version", 1)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_artifact(dir_path: Path, name: str, text: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    dest = dir_path / name
    dest.write_text(text, encoding="utf-8")
    return dest

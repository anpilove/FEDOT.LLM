from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_journal(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_artifact(dir_path: Path, name: str, text: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    dest = dir_path / name
    dest.write_text(text, encoding="utf-8")
    return dest


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

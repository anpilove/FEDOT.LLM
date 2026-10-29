"""Durable side events, with the JSONL journal as a readable projection."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fedotllm.agents.evolve.storage.journal import (
    JournalCorruptionError,
    read_jsonl,
    write_json_atomic,
)


def save_quality_event(journal: Path, event: dict) -> dict:
    """Persist before appending to JSONL so an append failure cannot lose a score."""
    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    payload.setdefault("event_id", uuid.uuid4().hex)
    payload.setdefault("schema_version", 1)
    key = hashlib.sha256(str(payload.get("candidate", "")).encode()).hexdigest()[:24]
    path = journal.parent / "quality_attempts" / key / f"{payload['event_id']}.json"
    write_json_atomic(path, payload)
    return payload


def read_quality_events(
    journal: Path,
    *,
    candidate_id: str | None = None,
    ignore_incomplete_tail: bool = False,
) -> list[dict]:
    """Merge persisted events and legacy journal rows without hiding corruption."""
    events = read_jsonl(
        journal, strict=True, ignore_incomplete_tail=ignore_incomplete_tail
    )
    seen = {event["event_id"] for event in events if event.get("event_id")}
    directory = journal.parent / "quality_attempts"
    if candidate_id is not None:
        directory /= hashlib.sha256(candidate_id.encode()).hexdigest()[:24]
    for path in sorted(directory.glob("**/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise JournalCorruptionError(
                f"invalid quality event at {path}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict) or not payload.get("event_id"):
            raise JournalCorruptionError(
                f"invalid quality event at {path}: expected event object"
            )
        if payload["event_id"] not in seen:
            events.append(payload)
            seen.add(payload["event_id"])
    # Stable sorting keeps the order of legacy events without timestamps.
    events.sort(key=lambda event: str(event.get("ts") or ""))
    return events

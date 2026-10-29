from __future__ import annotations

import json
import fcntl
import os
import hashlib
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_APPEND_LOCK = threading.Lock()


class JournalCorruptionError(ValueError):
    """A durable journal cannot be trusted; stop instead of skipping its rows."""


def repair_incomplete_jsonl_tail(path: Path) -> dict[str, Any] | None:
    """Repair only an unterminated final JSONL row, preserving discarded bytes.

    Callers must hold the journal's queue lock so no writer can append while the
    tail is inspected. Complete, newline-terminated corruption is never changed.
    """
    if not path.exists():
        return None
    if not os.path.isfile(path) or path.is_symlink():
        raise JournalCorruptionError(f"journal is not a regular file: {path}")
    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return None
    boundary = content.rfind(b"\n") + 1
    for line_number, line in enumerate(content[:boundary].splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise JournalCorruptionError(
                f"invalid journal event at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise JournalCorruptionError(
                f"invalid journal event at {path}:{line_number}: expected object"
            )
    tail = content[boundary:]
    if not tail.strip():
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {"action": "terminated_blank_tail", "offset": boundary, "bytes": 0}
    try:
        row = json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        digest = hashlib.sha256(tail).hexdigest()
        quarantine = path.with_name(f"{path.name}.incomplete-tail.{digest}.bin")
        if quarantine.exists():
            if quarantine.read_bytes() != tail:
                raise JournalCorruptionError(
                    f"journal tail quarantine hash collision: {quarantine}"
                )
        else:
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{quarantine.name}.", suffix=".tmp", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(tail)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temp_name, quarantine)
                except FileExistsError:
                    if quarantine.read_bytes() != tail:
                        raise JournalCorruptionError(
                            f"journal tail quarantine hash collision: {quarantine}"
                        )
            finally:
                Path(temp_name).unlink(missing_ok=True)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        with path.open("r+b") as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "action": "quarantined_incomplete_tail",
            "offset": boundary,
            "bytes": len(tail),
            "sha256": digest,
            "artifact": str(quarantine),
        }
    if not isinstance(row, dict):
        raise JournalCorruptionError(
            f"unterminated journal row is not an object: {path} at byte {boundary}"
        )
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "action": "terminated_complete_tail",
        "offset": boundary,
        "bytes": len(tail),
    }


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


def read_jsonl(
    path: Path, *, strict: bool = False, ignore_incomplete_tail: bool = False
) -> list[dict[str, Any]]:
    """Read JSON object rows; strict mode refuses malformed durable history."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        if (
            ignore_incomplete_tail
            and line_number == len(lines)
            and not line.endswith("\n")
        ):
            continue
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise JournalCorruptionError(
                    f"invalid journal event at {path}:{line_number}: {exc.msg}"
                ) from exc
            continue
        if isinstance(row, dict):
            rows.append(row)
        elif strict:
            raise JournalCorruptionError(
                f"invalid journal event at {path}:{line_number}: expected object"
            )
    return rows


def append_journal(path: Path, event: dict[str, Any]) -> None:
    """Append one durable JSONL event, serialized across task worker threads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("schema_version", 1)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    data = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    with _APPEND_LOCK:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            previous_size = os.fstat(descriptor).st_size
            try:
                offset = 0
                while offset < len(data):
                    count = os.write(descriptor, data[offset:])
                    if count <= 0:
                        raise OSError("journal write made no progress")
                    offset += count
                os.fsync(descriptor)
            except BaseException as original:
                # Undo only this failed append while holding the writer lock.
                # Existing history is preserved; independently saved side events
                # remain available even when this readable projection fails.
                try:
                    os.ftruncate(descriptor, previous_size)
                    os.fsync(descriptor)
                except OSError as cleanup_error:
                    if hasattr(original, "add_note"):
                        original.add_note(
                            f"journal append rollback failed: {cleanup_error}"
                        )
                raise
        finally:
            os.close(descriptor)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via a sibling temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str))
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temp.unlink(missing_ok=True)


def write_artifact(dir_path: Path, name: str, text: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    dest = dir_path / name
    dest.write_text(text, encoding="utf-8")
    return dest

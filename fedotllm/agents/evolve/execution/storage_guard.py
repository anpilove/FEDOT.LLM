"""Read-only capacity checks for controller-owned artifacts; never clean data."""
from __future__ import annotations

import errno
import os
from pathlib import Path

RESERVE_BYTES = 4 * 1024**3
LAUNCH_RESERVE_BYTES = 12 * 1024**3
RESERVE_INODES = 4096


class StorageCapacityError(OSError):
    """Infrastructure failure: leave the queue unmeasured and stop the service."""


def require_storage(path: Path, *, minimum_bytes=RESERVE_BYTES,
                    minimum_inodes=RESERVE_INODES) -> dict:
    path = path.resolve(strict=True)
    stats = os.statvfs(path)
    available = stats.f_bavail * stats.f_frsize
    inodes = stats.f_favail
    if available < minimum_bytes or inodes < minimum_inodes:
        raise StorageCapacityError(errno.ENOSPC,
            f"storage reserve exhausted: available_bytes={available}, "
            f"required_bytes={minimum_bytes}, available_inodes={inodes}, "
            f"required_inodes={minimum_inodes}", str(path))
    return {"path":str(path), "available_bytes":available, "available_inodes":inodes,
            "minimum_bytes":minimum_bytes, "minimum_inodes":minimum_inodes}


def storage_failure_text(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in (
        "no space left on device", "[errno 28]", "disk quota exceeded", "[errno 122]",
        "database or disk is full", "disk i/o error", "storage reserve exhausted",
    ))


def terminal_service_record(path: Path) -> dict | None:
    """Read the final structured record from the bounded tail of a launch log."""
    import json
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 1024 * 1024))
        tail = stream.read().decode("utf-8", errors="replace")
    # A final pretty-printed object starts at column zero; traceback dictionaries
    # are not accepted as terminal lifecycle evidence.
    starts = [0] + [index + 1 for index, char in enumerate(tail) if char == "\n"]
    for start in reversed(starts):
        if not tail[start:].startswith("{"):
            continue
        try:
            value = json.loads(tail[start:])
        except ValueError:
            continue
        if isinstance(value, dict) and "state" in value:
            return value
    return None

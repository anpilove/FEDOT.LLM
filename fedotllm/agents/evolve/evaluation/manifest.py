from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fedotllm.agents.evolve.execution.checkout import source_commit

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "benchmark_manifest.json"


def load_manifest(path: Path | None = None) -> dict:
    target = path or MANIFEST_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def verify_manifest(source: Path, path: Path | None = None) -> list[str]:
    """Return reproducibility violations without changing the source."""

    source = source.resolve()
    manifest = load_manifest(path)
    errors: list[str] = []
    expected_commit = str(manifest.get("fedot_commit") or "")
    actual_commit = source_commit(source)
    if actual_commit != expected_commit:
        errors.append(
            f"FEDOT commit mismatch: expected {expected_commit}, got {actual_commit or '<non-git>'}"
        )
    root = source / str(manifest.get("data_root") or "")
    for rel, expected in sorted((manifest.get("files") or {}).items()):
        target = (root / rel).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            errors.append(f"dataset missing: {rel}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"dataset hash mismatch: {rel}")
    return errors

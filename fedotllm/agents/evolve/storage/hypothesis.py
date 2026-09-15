from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fedotllm.agents.evolve.types import PatchCandidate


@dataclass(frozen=True)
class Hypothesis:
    id: str
    parent_id: str | None
    lead: dict
    claim: str
    expected_effect: str


@dataclass
class ExperimentRecord:
    hypothesis_id: str
    experiment_hash: str
    patch: dict
    tests: dict
    dev_scores: dict
    feedback: str
    revision: int = 1
    schema_version: int = 1


def normalized_patch_hash(
    candidate: PatchCandidate,
    source_hash: str,
    *,
    checkout: Path | None = None,
) -> str:
    """Identify an edit, or its normalized applied source state.

    LLMs often express the same replacement with a wider SEARCH block or an
    equivalent JSON number such as ``1e-4`` instead of ``0.0001``.  Once a
    candidate has been applied, hashing the resulting touched files catches
    those exact semantic duplicates.  The edit-based fallback preserves
    compatibility for callers that deduplicate before creating a checkout.
    """

    if checkout is not None:
        resulting_files: list[tuple[str, str]] = []
        edits_by_file: dict[str, list[str]] = {}
        for edit in candidate.edits:
            edits_by_file.setdefault(edit.file_path.strip(), []).append(edit.new_code)
        for file_path in sorted(edits_by_file):
            target = checkout / file_path
            if not target.is_file():
                break
            text = target.read_text(encoding="utf-8")
            # Some unit/integration callers stub ``fix_lead`` and return an
            # unapplied candidate.  In that case the stock file state says
            # nothing about this edit, so retain the edit-based identity.
            if any(replacement not in text for replacement in edits_by_file[file_path]):
                break
            if target.suffix == ".json":
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    pass
                else:
                    text = json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
            resulting_files.append((file_path, text))
        else:
            payload = json.dumps(
                {"source_hash": source_hash, "resulting_files": resulting_files},
                ensure_ascii=False,
                sort_keys=True,
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    edits = sorted(
        (
            edit.file_path.strip(),
            "\n".join(line.rstrip() for line in edit.old_code.strip().splitlines()),
            "\n".join(line.rstrip() for line in edit.new_code.strip().splitlines()),
        )
        for edit in candidate.edits
    )
    payload = json.dumps(
        {"source_hash": source_hash, "edits": edits},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def as_row(value) -> dict:
    return asdict(value)


def behavior_probe_fingerprint(code: str) -> str:
    """Identify the diagnostic independently of source edits and formatting."""

    code = (code or "").strip()
    try:
        payload = ast.dump(ast.parse(code), include_attributes=False)
    except SyntaxError:
        payload = code
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

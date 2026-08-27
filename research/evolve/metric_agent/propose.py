from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from research.evolve.metric_agent.patch import same_runtime, strip_gutter
from research.evolve.metric_agent.types import PatchCandidate


class PatchProposal(BaseModel):
    file_path: str = Field(description="Path relative to the FEDOT checkout, e.g. fedot/core/foo.py")
    old_code: str
    new_code: str
    rationale: str = ""


class PatchEdit(BaseModel):
    old_code: str
    new_code: str


class MultiPatchProposal(BaseModel):
    file_path: str = Field(description="Path relative to the FEDOT checkout, e.g. fedot/core/foo.py")
    edits: list[PatchEdit] = Field(description="Unique SEARCH/REPLACE pairs; each old_code matches the file once")
    rationale: str = ""


_SYSTEM = """You patch the FEDOT library source. A repo map pointed at a region;
propose one small SEARCH/REPLACE that changes runtime behavior there
(or the real cause next to it): control flow, features, or defaults.
Do not write tests. Do not mention scoring harnesses, datasets, case catalogs, or bug names.
old_code must be copied from the source WITHOUT the "NNN|" line-number prefix
and must match the file uniquely. new_code must differ from old_code.
No comment-only, rename-only, or identical replacements."""

_MULTI = """You may propose up to {n} SEARCH/REPLACE edits in `edits`.
Each old_code must be unique and match the file once. Use several edits only when
fit and transform (or predict) must stay consistent; do not emit overlapping searches."""


def build_prompt(context: str, *, max_edits: int = 1, contract: str = "") -> str:
    parts = [_SYSTEM]
    if max_edits > 1:
        parts.append(_MULTI.format(n=max_edits))
    if contract.strip():
        parts.append("Keep this consistent across the edited methods:\n" + contract.strip())
    parts.append("Source context:\n" + context)
    return "\n\n".join(parts) + "\n"


def propose_patch(
    *,
    inference,
    context: str,
    max_edits: int = 1,
    contract: str = "",
) -> PatchCandidate | None:
    if inference is None or not context.strip():
        return None
    budget = max(1, int(max_edits))
    prompt = build_prompt(context, max_edits=budget, contract=contract)
    model = MultiPatchProposal if budget > 1 else PatchProposal
    try:
        parsed = inference.create(prompt, model)
    except Exception:
        return None
    rel = parsed.file_path.lstrip("/")
    if rel.startswith("fedot/") is False and "fedot/" in rel:
        rel = rel[rel.index("fedot/") :]
    hunks = _hunks_from_parsed(parsed, budget)
    if not hunks:
        return None
    old_codes = [old for old, _ in hunks]
    if len(old_codes) != len(set(old_codes)):
        return None
    old_code, new_code = hunks[0]
    return PatchCandidate(
        candidate_id=uuid.uuid4().hex[:12],
        file_path=rel,
        old_code=old_code,
        new_code=new_code,
        rationale=getattr(parsed, "rationale", "") or "",
        hunks=hunks,
    )


def _hunks_from_parsed(parsed, budget: int) -> list[tuple[str, str]]:
    raw: list[tuple[str, str]] = []
    edits = getattr(parsed, "edits", None) or []
    for item in edits:
        raw.append((item.old_code or "", item.new_code or ""))
    if not raw:
        raw.append((getattr(parsed, "old_code", None) or "", getattr(parsed, "new_code", None) or ""))
    hunks: list[tuple[str, str]] = []
    for old, new in raw[:budget]:
        old_code = strip_gutter(old)
        new_code = strip_gutter(new)
        if not old_code.strip() or same_runtime(old_code, new_code):
            return []
        hunks.append((old_code, new_code))
    return hunks

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


_SYSTEM = """You patch the FEDOT library source. A repo map pointed at a region;
propose one small SEARCH/REPLACE that changes runtime behavior there
(or the real cause next to it): control flow, features, or defaults.
Do not write tests. Do not mention scoring harnesses, datasets, case catalogs, or bug names.
old_code must be copied from the source WITHOUT the "NNN|" line-number prefix
and must match the file uniquely. new_code must differ from old_code.
No comment-only, rename-only, or identical replacements."""


def build_prompt(context: str) -> str:
    return f"{_SYSTEM}\n\nSource context:\n{context}\n"


def propose_patch(*, inference, context: str) -> PatchCandidate | None:
    if inference is None or not context.strip():
        return None
    prompt = build_prompt(context)
    try:
        parsed = inference.create(prompt, PatchProposal)
    except Exception:
        return None
    rel = parsed.file_path.lstrip("/")
    if rel.startswith("fedot/") is False and "fedot/" in rel:
        rel = rel[rel.index("fedot/") :]
    old_code = strip_gutter(parsed.old_code or "")
    new_code = strip_gutter(parsed.new_code or "")
    if not old_code.strip() or same_runtime(old_code, new_code):
        return None
    return PatchCandidate(
        candidate_id=uuid.uuid4().hex[:12],
        file_path=rel,
        old_code=old_code,
        new_code=new_code,
        rationale=parsed.rationale,
    )

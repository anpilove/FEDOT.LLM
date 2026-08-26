from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from research.evolve.metric_agent.types import PatchCandidate


class PatchProposal(BaseModel):
    file_path: str = Field(description="Path relative to the FEDOT checkout, e.g. fedot/core/foo.py")
    old_code: str
    new_code: str
    rationale: str = ""


_SYSTEM = """You patch the FEDOT library source. A repo map pointed at a region;
propose one small SEARCH/REPLACE there (or the real cause next to it).
You only edit FEDOT source. You do not write tests. You do not mention
scoring harnesses, datasets, case catalogs, or bug names.
old_code must match the file uniquely."""


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
    return PatchCandidate(
        candidate_id=uuid.uuid4().hex[:12],
        file_path=rel,
        old_code=parsed.old_code,
        new_code=parsed.new_code,
        rationale=parsed.rationale,
    )

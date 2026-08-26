"""Scoreboard: did hidden-holdout get better after a candidate?"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.evolve.metric_agent.journal import append_journal
from research.evolve.metric_agent.types import Decision, Lead, PatchCandidate, ScoreResult


def scoreboard_path(workspace: Path) -> Path:
    return workspace / "scoreboard.jsonl"


def append_attempt(
    workspace: Path,
    *,
    lead: Lead | None,
    candidate: PatchCandidate | None,
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult] | None,
    decision: Decision,
) -> None:
    row = {
        "event": "attempt",
        "keep": decision.keep,
        "better": bool(decision.keep and (decision.target_delta or 0) > 0),
        "reason": decision.reason,
        "target_delta": decision.target_delta,
        "regression_deltas": decision.regression_deltas,
        "lead": None if lead is None else {
            "channel": lead.channel,
            "file_path": lead.file_path,
            "line": lead.line,
        },
        "candidate_id": None if candidate is None else candidate.candidate_id,
        "file": None if candidate is None else candidate.file_path,
        "stock": {key: _brief(value) for key, value in stock.items()},
        "patched": None if patched is None else {key: _brief(value) for key, value in patched.items()},
    }
    append_journal(scoreboard_path(workspace), row)


def summarize(workspace: Path) -> dict[str, Any]:
    path = scoreboard_path(workspace)
    if not path.is_file():
        return {"attempts": 0, "keeps": 0, "best_delta": None, "getting_better": False}
    attempts = 0
    keeps = 0
    best: float | None = None
    last_keep = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "attempt":
            continue
        attempts += 1
        if row.get("keep"):
            keeps += 1
            last_keep = True
        delta = row.get("target_delta")
        if isinstance(delta, (int, float)) and (best is None or delta > best):
            best = float(delta)
    return {
        "attempts": attempts,
        "keeps": keeps,
        "best_delta": best,
        "getting_better": keeps > 0 and best is not None and best >= 0.01,
        "last_keep": last_keep,
    }


def _brief(result: ScoreResult) -> dict:
    return {
        "status": result.status,
        "score": result.score,
        "duration_s": result.duration_s,
    }

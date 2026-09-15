"""Scoreboard: DEV suite selects; FINAL is the claim. getting_better is DEV, not proof."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.storage.journal import append_journal, resolve_run_workspace
from fedotllm.agents.evolve.types import (
    Decision,
    PatchCandidate,
    PatchSite,
    ScoreResult,
)


def scoreboard_path(workspace: Path) -> Path:
    return workspace / "scoreboard.jsonl"


def append_attempt(
    workspace: Path,
    *,
    lead: PatchSite | None,
    candidate: PatchCandidate | None,
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult] | None,
    decision: Decision,
    localization: dict | None = None,
) -> None:
    row = {
        "event": "attempt",
        "suite": "dev",
        "keep": bool(
            decision.keep
            or decision.correctness_keep
            or decision.maintenance_keep
            or decision.metric_signal_keep
        ),
        "correctness_keep": decision.correctness_keep,
        "maintenance_keep": decision.maintenance_keep,
        "metric_signal_keep": decision.metric_signal_keep,
        "better": bool(
            (decision.keep or decision.metric_signal_keep)
            and (decision.target_delta or 0) > 0
        ),
        "reason": decision.reason,
        "target_delta": decision.target_delta,
        "regression_deltas": decision.regression_deltas,
        "lead": None
        if lead is None
        else {
            "channel": lead.channel,
            "file_path": lead.file_path,
            "line": lead.line,
        },
        "candidate_id": None if candidate is None else candidate.candidate_id,
        "file": None if candidate is None else candidate.file_path,
        "stock": {key: _brief(value) for key, value in stock.items()},
        "patched": None
        if patched is None
        else {key: _brief(value) for key, value in patched.items()},
    }
    if localization:
        row.update(localization)
    append_journal(scoreboard_path(workspace), row)


def append_final(
    workspace: Path,
    *,
    decision: Decision,
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult],
) -> None:
    row = {
        "event": "final",
        "suite": "final",
        "keep": decision.keep,
        "reason": decision.reason,
        "target_delta": decision.target_delta,
        "regression_deltas": decision.regression_deltas,
        "stock": {key: _brief(value) for key, value in stock.items()},
        "patched": {key: _brief(value) for key, value in patched.items()},
    }
    append_journal(scoreboard_path(workspace), row)


def summarize(workspace: Path) -> dict[str, Any]:
    workspace = resolve_run_workspace(workspace)
    path = scoreboard_path(workspace)
    if not path.is_file():
        return {
            "attempts": 0,
            "keeps": 0,
            "correctness_keeps": 0,
            "maintenance_keeps": 0,
            "metric_signal_keeps": 0,
            "best_delta": None,
            "getting_better": False,
            "getting_better_final": False,
            "last_keep": False,
        }
    attempts = 0
    keeps = 0
    metric_keeps = 0
    correctness_keeps = 0
    maintenance_keeps = 0
    metric_signal_keeps = 0
    best: float | None = None
    last_keep = False
    final_keep = False
    final_delta: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "final":
            final_keep = bool(row.get("keep"))
            delta = row.get("target_delta")
            if isinstance(delta, (int, float)):
                final_delta = float(delta)
            continue
        if row.get("event") != "attempt":
            continue
        attempts += 1
        last_keep = bool(row.get("keep"))
        if last_keep:
            keeps += 1
            if (
                not row.get("correctness_keep")
                and not row.get("maintenance_keep")
                and row.get("better", True)
            ):
                metric_keeps += 1
        if row.get("correctness_keep"):
            correctness_keeps += 1
        if row.get("maintenance_keep"):
            maintenance_keeps += 1
        if row.get("metric_signal_keep"):
            metric_signal_keeps += 1
        delta = row.get("target_delta")
        if isinstance(delta, (int, float)) and (best is None or delta > best):
            best = float(delta)
    return {
        "attempts": attempts,
        "keeps": keeps,
        "correctness_keeps": correctness_keeps,
        "maintenance_keeps": maintenance_keeps,
        "metric_signal_keeps": metric_signal_keeps,
        "best_delta": best,
        "getting_better": metric_keeps > 0,
        "getting_better_final": final_keep,
        "final_delta": final_delta,
        "last_keep": last_keep,
    }


def _brief(result: ScoreResult) -> dict:
    return {
        "status": result.status,
        "score": result.score,
        "duration_s": result.duration_s,
        "seed": result.seed,
    }

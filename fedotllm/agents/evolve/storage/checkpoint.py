"""Atomic, append-audited checkpoints for interrupted Evolve campaigns."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.storage.journal import append_journal, write_json_atomic
from fedotllm.agents.evolve.types import MatchSite


def load_checkpoint(workspace: Path) -> dict[str, Any]:
    path = workspace / "checkpoint.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_checkpoint(
    workspace: Path,
    *,
    stage: str,
    run_id: str,
    **updates: Any,
) -> dict[str, Any]:
    """Atomically replace current state and append the same state transition."""

    workspace.mkdir(parents=True, exist_ok=True)
    prior = load_checkpoint(workspace)
    payload = {
        **prior,
        **updates,
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "sequence": int(prior.get("sequence") or 0) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(workspace / "checkpoint.json", payload)
    append_journal(
        workspace / "checkpoint_history.jsonl",
        {
            "event": "checkpoint",
            "run_id": run_id,
            "stage": stage,
            "sequence": payload["sequence"],
            "updates": updates,
        },
    )
    return payload


def checkpoint_leads(workspace: Path) -> list[MatchSite]:
    """Recover Scout selections even when its later catalog walk was interrupted."""

    result: list[MatchSite] = []
    for raw in load_checkpoint(workspace).get("selected_leads") or ():
        if not isinstance(raw, dict):
            continue
        try:
            result.append(
                MatchSite(
                    channel=str(raw.get("channel") or "checkpoint"),
                    file_path=str(raw["file_path"]),
                    line=int(raw["line"]),
                    why=str(raw.get("why") or ""),
                    evidence=tuple(str(item) for item in raw.get("evidence") or ()),
                    signals=tuple(str(item) for item in raw.get("signals") or ()),
                    mechanism=str(raw.get("mechanism") or ""),
                    proposed_change=str(raw.get("proposed_change") or ""),
                    expected_metric_effect=str(raw.get("expected_metric_effect") or ""),
                    hypothesis_kind=str(raw.get("hypothesis_kind") or "quality"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result

"""Stable registry of benchmark defects that cannot count as novel findings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("known_mechanisms.json")


def load_known_mechanisms(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported known-mechanism registry schema")
    rows = payload.get("mechanisms")
    if not isinstance(rows, list) or not rows:
        raise ValueError("known-mechanism registry must contain mechanisms")
    required = {"case_id", "contract_id", "file_path", "mechanism", "proposed_change"}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError("invalid known-mechanism registry row")
        case_id = str(raw["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate known mechanism: {case_id}")
        seen.add(case_id)
        result.append(
            {
                "case_id": case_id,
                "contract_id": str(raw["contract_id"]),
                "file_path": str(raw["file_path"]),
                "line": 0,
                "mechanism": str(raw["mechanism"]),
                "proposed_change": str(raw["proposed_change"]),
                "history_kind": "known_benchmark_defect",
            }
        )
    return result


def combined_known_mechanisms(findings_path: Path | None) -> list[dict[str, Any]]:
    """Combine immutable benchmark cases with append-only confirmed memory."""

    from fedotllm.agents.evolve.storage.replay import (
        confirmed_hypotheses_from_findings,
    )

    rows = [*load_known_mechanisms(), *confirmed_hypotheses_from_findings(findings_path)]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        identity = str(row.get("contract_id") or "").strip()
        if not identity:
            identity = json.dumps(
                {
                    "file_path": row.get("file_path"),
                    "mechanism": row.get("mechanism"),
                    "proposed_change": row.get("proposed_change"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
    return result

"""Phase 2: oracle-location repair. Not imported by scout.

Gives fixer the historical file (DEV split). No holdout. Do not retune scout.
"""

from __future__ import annotations

import ast
import json
import uuid
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    resolve_fedot_src,
)
from fedotllm.agents.evolve.discovery.discover import static_leads
from fedotllm.agents.evolve.agents.fixer import fix_lead
from fedotllm.agents.evolve.storage.journal import append_journal, read_jsonl
from fedotllm.agents.evolve.evaluation.judge import (
    measure_fedot_tests,
    measure_patched,
    measure_stock,
    normalize_test_result,
    tests_regressed,
    verdict,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.commands.recall import split_gold, unique_files
from fedotllm.agents.evolve.execution.smoke import import_error
from fedotllm.agents.evolve.evaluation.tasks import hidden_exam
from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit, MatchSite

_RUNTIME = frozenset({"fit", "transform", "predict", "predict_proba", "predict_for_fit"})


def oracle_lead(checkout: Path, file_path: str) -> MatchSite | None:
    target = checkout / file_path
    if not target.is_file():
        return None
    for lead in static_leads(checkout):
        if lead.file_path == file_path:
            return MatchSite(
                channel="oracle",
                file_path=file_path,
                line=lead.line,
                why=lead.why,
                signals=lead.signals,
            )
    line, name = _first_runtime(target)
    return MatchSite(channel="oracle", file_path=file_path, line=line, why=f"function {name}")


def _first_runtime(path: Path) -> tuple[int, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return 1, "file"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _RUNTIME:
            return int(getattr(node, "lineno", 1) or 1), node.name
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return int(getattr(node, "lineno", 1) or 1), node.name
    return 1, "file"


def measure_repair(
    *,
    inference,
    checkout: Path | None = None,
    workspace: Path | None = None,
    split: str = "dev",
    limit: int = 5,
    files: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    source = (checkout or resolve_fedot_src()).resolve()
    workspace = workspace or Path("/tmp/evolve-agent-oracle")
    workspace.mkdir(parents=True, exist_ok=True)
    run_id = f"repair-{uuid.uuid4().hex[:8]}"
    if files:
        gold = [path for path in files if (source / path).is_file()]
    else:
        gold = [path for path in split_gold()[split] if (source / path).is_file()][: max(1, limit)]
    in_pool = set(unique_files(static_leads(source)))
    rows: list[dict[str, Any]] = []
    for index, rel in enumerate(gold, start=1):
        tree = create_experiment_checkout(
            source, workspace, run_id=run_id, candidate_id=f"case-{index}"
        )
        lead = oracle_lead(tree, rel)
        if lead is None:
            rows.append({"file_path": rel, "status": "missing_lead", "in_pool": rel in in_pool})
            discard_experiment_checkout(tree, workspace=workspace, source=source)
            continue
        candidate = fix_lead(tree, lead, inference=inference, workspace=workspace)
        if candidate is None:
            status = "no_patch"
            err = None
        else:
            err = import_error(tree, candidate.file_path)
            status = "import_fail" if err else "import_ok"
        row = {
            "file_path": rel,
            "line": lead.line,
            "in_pool": rel in in_pool,
            "status": status,
            "import_error": err,
            "candidate_id": None if candidate is None else candidate.candidate_id,
        }
        rows.append(row)
        append_journal(workspace / "repair.jsonl", {"event": "oracle_attempt", **row})
        discard_experiment_checkout(tree, workspace=workspace, source=source)
    n = len(rows)
    patches = sum(1 for row in rows if row["status"] in {"import_ok", "import_fail"})
    imports = sum(1 for row in rows if row["status"] == "import_ok")
    return {
        "split": split,
        "n": n,
        "patch_rate": patches / n if n else 0.0,
        "import_ok_rate": imports / n if n else 0.0,
        "in_pool": sum(1 for row in rows if row.get("in_pool")) / n if n else 0.0,
        "rows": rows,
    }


def _saved_attempts(workspace: Path) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(workspace / "repair.jsonl"):
        cid = row.get("candidate_id")
        if row.get("event") != "oracle_attempt" or not cid or cid in seen:
            continue
        seen.add(cid)
        attempts.append(row)
    return attempts


def load_saved_candidate(workspace: Path, candidate_id: str, file_path: str) -> PatchCandidate | None:
    folder = workspace / "candidates" / candidate_id
    edits_file = folder / "edits.json"
    if edits_file.is_file():
        try:
            payload = json.loads(edits_file.read_text(encoding="utf-8"))
            edits = [PatchEdit(**item) for item in payload]
            return PatchCandidate(
                candidate_id=candidate_id,
                edits=edits,
                rationale=(folder / "rationale.txt").read_text(encoding="utf-8")
                if (folder / "rationale.txt").is_file()
                else "",
                contract=(folder / "contract.txt").read_text(encoding="utf-8")
                if (folder / "contract.txt").is_file()
                else "",
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
    old = folder / "old.py"
    new = folder / "new.py"
    if not old.is_file() or not new.is_file():
        return None
    rationale = folder / "rationale.txt"
    return PatchCandidate(
        candidate_id=candidate_id,
        file_path=file_path,
        old_code=old.read_text(encoding="utf-8"),
        new_code=new.read_text(encoding="utf-8"),
        rationale=rationale.read_text(encoding="utf-8") if rationale.is_file() else "",
    )


def gate_saved_repairs(
    *,
    workspace: Path,
    checkout: Path | None = None,
) -> dict[str, Any]:
    """Re-apply saved oracle patches and run FEDOT unit tests. No LLM, no holdout."""

    source = (checkout or resolve_fedot_src()).resolve()
    attempts = _saved_attempts(workspace)
    before_result = normalize_test_result(measure_fedot_tests(source))
    before = before_result.failed_nodes
    run_id = f"repair-tests-{uuid.uuid4().hex[:8]}"
    rows: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        cid = attempt["candidate_id"]
        rel = attempt["file_path"]
        candidate = load_saved_candidate(workspace, cid, rel)
        if candidate is None:
            rows.append({"candidate_id": cid, "file_path": rel, "status": "missing_artifact"})
            continue
        tree = create_experiment_checkout(
            source, workspace, run_id=run_id, candidate_id=f"case-{index}"
        )
        if not apply_patch(tree, candidate):
            rows.append({"candidate_id": cid, "file_path": rel, "status": "apply_fail"})
            discard_experiment_checkout(tree, workspace=workspace, source=source)
            continue
        after = measure_fedot_tests(tree)
        decision = tests_regressed(before_result, after)
        status = "tests_fail" if decision is not None else "tests_ok"
        row = {
            "candidate_id": cid,
            "file_path": rel,
            "status": status,
            "reason": None if decision is None else decision.reason,
        }
        rows.append(row)
        append_journal(workspace / "repair.jsonl", {"event": "oracle_tests", **row})
        discard_experiment_checkout(tree, workspace=workspace, source=source)
    n = len(rows)
    ok = sum(1 for row in rows if row["status"] == "tests_ok")
    return {
        "n": n,
        "tests_ok_rate": ok / n if n else 0.0,
        "stock_failures": len(before),
        "rows": rows,
    }


def _brief_scores(scores: dict) -> dict[str, dict]:
    return {
        key: {"status": value.status, "score": value.score}
        for key, value in scores.items()
    }


def holdout_saved_repairs(
    *,
    workspace: Path,
    checkout: Path | None = None,
) -> dict[str, Any]:
    """DEV holdout on saved oracle patches. Does not write the e2e scoreboard."""

    source = (checkout or resolve_fedot_src()).resolve()
    attempts = _saved_attempts(workspace)
    lift, protect = hidden_exam()
    exam_ids = tuple(dict.fromkeys((*lift, *protect)))
    stock_scores = measure_stock(exam_ids, checkout=source)
    run_id = f"repair-holdout-{uuid.uuid4().hex[:8]}"
    rows: list[dict[str, Any]] = []
    keeps = 0
    for index, attempt in enumerate(attempts, start=1):
        cid = attempt["candidate_id"]
        rel = attempt["file_path"]
        candidate = load_saved_candidate(workspace, cid, rel)
        if candidate is None:
            rows.append({"candidate_id": cid, "file_path": rel, "status": "apply_fail", "keep": False})
            continue
        tree = create_experiment_checkout(
            source, workspace, run_id=run_id, candidate_id=f"case-{index}"
        )
        if not apply_patch(tree, candidate):
            rows.append({"candidate_id": cid, "file_path": rel, "status": "apply_fail", "keep": False})
            discard_experiment_checkout(tree, workspace=workspace, source=source)
            continue
        patched = measure_patched(exam_ids, checkout=tree)
        decision = verdict(stock_scores, patched)
        if decision.keep:
            keeps += 1
        row = {
            "candidate_id": cid,
            "file_path": rel,
            "status": "keep" if decision.keep else "drop",
            "keep": decision.keep,
            "reason": decision.reason,
            "target_delta": decision.target_delta,
            "regression_deltas": decision.regression_deltas,
            "stock": _brief_scores(stock_scores),
            "patched": _brief_scores(patched),
        }
        rows.append(row)
        append_journal(workspace / "repair.jsonl", {"event": "oracle_holdout", **row})
        discard_experiment_checkout(tree, workspace=workspace, source=source)
    n = len(rows)
    return {
        "n": n,
        "keep_rate": keeps / n if n else 0.0,
        "rows": rows,
    }

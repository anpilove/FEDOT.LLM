"""Serialize campaign decisions and experiment evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.storage.findings import append_finding
from fedotllm.agents.evolve.storage.hypothesis import ExperimentRecord, as_row
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.storage.scoreboard import append_attempt
from fedotllm.agents.evolve.types import (
    Decision,
    PatchCandidate,
    PatchSite,
    ScoreResult,
)

def _record_attempt(
    journal: Path,
    workspace: Path,
    lead: PatchSite,
    candidate: PatchCandidate | None,
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult] | None,
    decision: Decision,
    localization_row: dict,
    revision: int,
    *,
    diff: str = "",
    tests=None,
    hypothesis_id: str = "",
    patch_hash: str = "",
    feedback: str = "",
    reproduction: dict | None = None,
    behavior_probe: dict | None = None,
    affected_metric: dict | None = None,
    findings_path: Path | None = None,
    run_number: int = 0,
    run_id: str = "",
    source_commit_value: str = "",
    source_hash: str = "",
    score_protocol_hash: str = "",
    evaluation_protocol_hash: str = "",
    resumed_branch: bool = False,
) -> None:
    row = {
        "event": "decision",
        "revision": revision,
        "hypothesis_id": hypothesis_id,
        "candidate": None if candidate is None else candidate.candidate_id,
        "file": None if candidate is None else candidate.file_path,
        "edits": [] if candidate is None else [asdict(edit) for edit in candidate.edits],
        "lead": asdict(lead),
        "stock": {key: _score_log(value) for key, value in stock.items()},
        "patched": None if patched is None else {
            key: _score_log(value) for key, value in patched.items()
        },
        "diff": diff,
        "patch_hash": patch_hash,
        "score_protocol_hash": score_protocol_hash,
        "evaluation_protocol_hash": evaluation_protocol_hash,
        "resumed_branch": resumed_branch,
        "feedback": feedback,
        "reproduction": reproduction or {},
        "behavior_probe": behavior_probe or {},
        "affected_metric": affected_metric or {},
        **localization_row,
        **asdict(decision),
    }
    if tests is not None:
        row["fedot_test_status"] = tests.status
        row["fedot_test_exit_code"] = tests.exit_code
        row["fedot_test_duration_s"] = tests.duration_s
        row["fedot_test_cmd"] = tests.cmd
        row["fedot_test_output_tail"] = (tests.output or "")[-6_000:]
        row["fedot_test_failures"] = sorted(tests.failed_nodes)[:20]
        row["fedot_test_gate_passed"] = not decision.reason.startswith(
            "fedot_tests_regressed"
        ) and not decision.infrastructure_error
    append_journal(journal, row)
    if findings_path is not None:
        append_finding(
            findings_path,
            run_number=run_number,
            run_id=run_id,
            source_commit=source_commit_value,
            source_hash=source_hash,
            workspace=workspace,
            row=row,
        )
    append_attempt(
        workspace,
        lead=lead,
        candidate=candidate,
        stock=stock,
        patched=patched,
        decision=decision,
        localization=localization_row,
    )
    if candidate is not None:
        record = ExperimentRecord(
            hypothesis_id=hypothesis_id,
            experiment_hash=patch_hash,
            patch={"candidate_id": candidate.candidate_id, "edits": [asdict(e) for e in candidate.edits]},
            tests={} if tests is None else {
                "status": tests.status,
                "exit_code": tests.exit_code,
                "failed_nodes": sorted(tests.failed_nodes),
                "duration_s": tests.duration_s,
                "cmd": tests.cmd,
                "output_tail": (tests.output or "")[-6_000:],
            },
            dev_scores={} if patched is None else {
                key: _score_log(value) for key, value in patched.items()
            },
            feedback=feedback,
            revision=revision,
        )
        append_journal(workspace / "hypotheses.jsonl", {"event": "experiment", **as_row(record)})


def _score_log(result: ScoreResult) -> dict:
    return {
        "task_id": result.task_id,
        "status": result.status,
        "score": result.score,
        "duration_s": result.duration_s,
        "detail": result.detail[:240],
        "traceback_chars": len(result.traceback or ""),
        "cmd": result.cmd,
        "env_hash": result.env_hash,
        "log_tail": result.log_tail[-800:],
        "seed": result.seed,
    }

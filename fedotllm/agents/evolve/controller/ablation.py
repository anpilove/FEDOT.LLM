"""Remove unnecessary edits after a candidate passes DEV confirmation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.agents.verifier import replay_reproduction
from fedotllm.agents.evolve.evaluation.judge import (
    confirm_candidate_tests,
    normalize_test_result,
)
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.execution.smoke import import_error
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import (
    PatchCandidate,
    TestResult,
    VerificationResult,
)

def ablate_candidate(
    source: Path,
    workspace: Path,
    run_id: str,
    candidate: PatchCandidate,
    exam_ids: tuple[str, ...],
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    journal: Path,
    *,
    confirmation_seeds: tuple[int, ...] = (42, 43, 44),
    baseline_tests: TestResult,
    verification: VerificationResult | None = None,
    compare_behavior_probe_fn: Callable[..., dict],
    behavior_probe_blocks_fn: Callable[[dict], bool],
    measure_fedot_tests_fn: Callable[[Path], TestResult],
    confirm_dev_fn: Callable[..., tuple[bool, dict]],
) -> PatchCandidate:
    edits = list(candidate.edits)
    if len(edits) <= 1:
        return candidate
    index = 0
    while index < len(edits):
        trial_edits = edits[:index] + edits[index + 1 :]
        trial = PatchCandidate(
            candidate_id=f"{candidate.candidate_id}-ab{index}",
            edits=trial_edits,
            rationale=candidate.rationale,
            contract=candidate.contract,
            behavior_probe=candidate.behavior_probe,
            proposed_test_edits=list(candidate.proposed_test_edits),
        )
        tree = create_experiment_checkout(
            source,
            workspace,
            run_id=run_id,
            candidate_id=trial.candidate_id,
        )
        kept = False
        gate: dict[str, object] = {}
        try:
            if apply_patch(tree, trial):
                import_failure = next(
                    (
                        error
                        for rel in dict.fromkeys(edit.file_path for edit in trial.edits)
                        if (error := import_error(tree, rel))
                    ),
                    None,
                )
                gate["import_error"] = import_failure
                reproduction_ok = True
                if verification is not None and verification.reproduction_code:
                    replay = replay_reproduction(tree, verification)
                    gate["reproduction"] = replay
                    reproduction_ok = replay.get("patched") == "resolved"
                behavior_ok = True
                # Match the campaign gate: verified bugs use their reproduction
                # or the frozen evaluator, not an optional quality observation.
                if trial.behavior_probe and (
                    verification is None or verification.status == "quality_hypothesis"
                ):
                    probe = compare_behavior_probe_fn(source, tree, trial.behavior_probe)
                    gate["behavior_probe"] = probe
                    behavior_ok = not behavior_probe_blocks_fn(probe)
                tests_ok = False
                if not import_failure and reproduction_ok and behavior_ok:
                    test_result, blocked, _ = confirm_candidate_tests(
                        baseline_tests,
                        tree,
                        first=normalize_test_result(measure_fedot_tests_fn(tree)),
                        runner=measure_fedot_tests_fn,
                    )
                    gate["tests"] = {
                        "status": test_result.status,
                        "failed_nodes": sorted(test_result.failed_nodes),
                        "blocked": blocked.reason if blocked is not None else None,
                    }
                    tests_ok = blocked is None
                if tests_ok:
                    kept, confirmation = confirm_dev_fn(
                        source,
                        tree,
                        exam_ids,
                        lift_ids,
                        protect_ids,
                        seeds=confirmation_seeds,
                    )
                    gate["confirmation"] = confirmation
            append_journal(
                journal,
                {
                    "event": "ablation",
                    "candidate": candidate.candidate_id,
                    "removed_edit": asdict(edits[index]),
                    "kept_without_edit": kept,
                    "gates": gate,
                },
            )
        finally:
            discard_experiment_checkout(tree, workspace=workspace, source=source)
        if kept:
            edits = trial_edits
        else:
            index += 1
    return PatchCandidate(
        candidate_id=candidate.candidate_id,
        edits=edits,
        rationale=candidate.rationale,
        contract=candidate.contract,
        behavior_probe=candidate.behavior_probe,
        proposed_test_edits=list(candidate.proposed_test_edits),
    )

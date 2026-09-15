"""Render candidate diffs without mutating the immutable source checkout."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.types import PatchCandidate

SnapshotDiff = Callable[..., str]


def candidate_diff(
    experiment: Path,
    source: Path,
    candidate: PatchCandidate,
    *,
    snapshot_diff_fn: SnapshotDiff,
) -> str:
    return "\n\n".join(
        snapshot_diff_fn(experiment, rel, source=source)
        for rel in dict.fromkeys(edit.file_path for edit in candidate.edits)
    )


def render_candidate_patch(
    source: Path,
    workspace: Path,
    run_id: str,
    candidate: PatchCandidate,
    *,
    snapshot_diff_fn: SnapshotDiff,
) -> str:
    tree = create_experiment_checkout(
        source,
        workspace,
        run_id=run_id,
        candidate_id=f"winner-{candidate.candidate_id}",
    )
    try:
        if not apply_patch(tree, candidate):
            return ""
        return candidate_diff(
            tree,
            source,
            candidate,
            snapshot_diff_fn=snapshot_diff_fn,
        )
    finally:
        discard_experiment_checkout(tree, workspace=workspace, source=source)

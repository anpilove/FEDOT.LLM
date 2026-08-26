from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ScoreStatus = Literal["ok", "crash", "timeout", "invalid"]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    kind: str
    nodes: tuple[str, ...]
    left: str | None = None
    right: str | None = None
    join: str | None = None
    tail: tuple[str, ...] = ()
    role: str = ""
    metric: str = "holdout_roc_auc"
    higher_is_better: bool = True
    sentinel: float = 0.5
    min_delta: float = 0.01
    timeout_s: float = 600.0
    must_not_regress: tuple[str, ...] = ()


@dataclass
class ScoreResult:
    task_id: str
    status: ScoreStatus
    score: float
    traceback: str = ""
    detail: str = ""
    duration_s: float = 0.0
    n_train: int = 0
    env_hash: str = ""
    cmd: str = ""
    log_tail: str = ""


@dataclass
class PatchCandidate:
    candidate_id: str
    file_path: str
    old_code: str
    new_code: str
    rationale: str = ""
    hunks: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Lead:
    channel: str
    file_path: str
    line: int
    why: str = ""


@dataclass
class Decision:
    keep: bool
    reason: str
    target_delta: float | None
    regression_deltas: dict[str, float | None] = field(default_factory=dict)

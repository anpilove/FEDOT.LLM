from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

ScoreStatus = Literal["ok", "crash", "timeout", "invalid"]
SnippetStatus = Literal["ok", "timeout", "blocked", "runtime_error", "unavailable"]
TestStatus = Literal[
    "passed",
    "test_failures",
    "incomplete",
    "timeout",
    "collection_error",
    "execution_error",
    "missing_tests",
]


class ToolAction(str, Enum):
    SKIP = "skip"
    READ = "read"
    RUN = "run"
    SEARCH = "search"
    SYMBOL = "symbol"
    CALLERS = "callers"
    DOCS = "docs"
    PICK = "pick"
    PATCH = "patch"
    CANNOT_FIX = "cannot_fix"


class VerificationAction(str, Enum):
    SEARCH = "search"
    SYMBOL = "symbol"
    CALLERS = "callers"
    DOCS = "docs"
    READ = "read"
    RUN = "run"
    FOCUSED_TEST = "focused_test"
    VERIFY_BUG = "verify_bug"
    QUALITY_HYPOTHESIS = "quality_hypothesis"
    REJECT = "reject"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    kind: str
    nodes: tuple[str, ...]
    left: str | None = None
    right: str | None = None
    join: str | None = None
    tail: tuple[str, ...] = ()
    metric: str = "holdout_roc_auc"
    higher_is_better: bool = True
    sentinel: float = 0.5
    min_delta: float = 0.01
    timeout_s: float = 600.0
    dataset: str = "scoring"
    problem: str = "classification"
    target: str = "target"
    drop: tuple[str, ...] = ()
    train_file: str = ""
    test_file: str = ""
    forecast_horizon: int = 0
    history_size: int = 0
    min_delta_mode: str = "absolute"
    index_offset: int = 0


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
    seed: int = 42
    coverage: tuple[dict, ...] = ()
    dataflow: tuple[dict, ...] = ()
    data_evidence: dict = field(default_factory=dict)
    metric_observations: dict = field(default_factory=dict)


@dataclass
class SnippetResult:
    status: SnippetStatus
    code: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    detail: str = ""
    target_reached: bool | None = None

    @property
    def output(self) -> str:
        text = "\n".join(
            part for part in (self.stdout.strip(), self.stderr.strip()) if part
        )
        if text:
            return text
        if self.detail:
            return self.detail
        return "<no output>" if self.status == "ok" else f"<{self.status}>"


@dataclass
class TestResult:
    status: TestStatus
    exit_code: int | None
    failed_nodes: set[str] = field(default_factory=set)
    output: str = ""
    duration_s: float = 0.0
    cmd: str = ""
    leads: list["MatchSite"] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.status in {"passed", "test_failures"}


@dataclass
class VerificationResult:
    status: Literal[
        "verified_bug",
        "quality_hypothesis",
        "rejected",
        "inconclusive",
        "infrastructure_error",
    ]
    claim: str = ""
    expected: str = ""
    observed: str = ""
    reproduction_code: str = ""
    stock_probe: SnippetResult | None = None
    evidence: tuple[str, ...] = ()
    detail: str = ""
    current_approach: str = ""
    proposed_approach: str = ""
    alternatives_considered: tuple[str, ...] = ()
    generality: str = ""
    risks: tuple[str, ...] = ()
    resolved_target: dict = field(default_factory=dict)

    @property
    def proceed(self) -> bool:
        return self.status in {"verified_bug", "quality_hypothesis"}


@dataclass(frozen=True)
class PatchEdit:
    file_path: str
    old_code: str
    new_code: str


@dataclass
class PatchCandidate:
    candidate_id: str
    file_path: str = ""
    old_code: str = ""
    new_code: str = ""
    rationale: str = ""
    hunks: list[tuple[str, str]] = field(default_factory=list)
    edits: list[PatchEdit] = field(default_factory=list)
    contract: str = ""
    behavior_probe: str = ""
    # Suggested contract updates are review artifacts only. They are never
    # applied to the checkout whose tests judge this candidate.
    proposed_test_edits: list[PatchEdit] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.edits:
            pairs = self.hunks or (
                [(self.old_code, self.new_code)] if self.old_code else []
            )
            self.edits = [
                PatchEdit(self.file_path, old, new)
                for old, new in pairs
                if self.file_path and old
            ]
        if self.edits:
            first = self.edits[0]
            self.file_path = self.file_path or first.file_path
            self.old_code = self.old_code or first.old_code
            self.new_code = self.new_code or first.new_code
            if not self.hunks and all(
                edit.file_path == self.file_path for edit in self.edits
            ):
                self.hunks = [(edit.old_code, edit.new_code) for edit in self.edits]


@dataclass(frozen=True)
class MatchSite:
    """Candidate file+line for a patch. Not a found bug.

    The LLM reads source around this site (file or method); that context is not
    a separate type. Scout returns up to ``max_picks`` of these per campaign.
    Historically called ``lead`` throughout the code and journal rows.
    """

    channel: str
    file_path: str
    line: int
    why: str = ""
    evidence: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    mechanism: str = ""
    proposed_change: str = ""
    expected_metric_effect: str = ""
    hypothesis_kind: Literal["correctness", "quality"] = "quality"


@dataclass
class Decision:
    keep: bool
    reason: str
    target_delta: float | None
    regression_deltas: dict[str, float | None] = field(default_factory=dict)
    dev_keep: bool | None = None
    final_keep: bool | None = None
    correctness_keep: bool = False
    maintenance_keep: bool = False
    metric_signal_keep: bool = False
    stage: str = "dev"
    experiment_id: str = ""
    infrastructure_error: bool = False
    metric_goal_keep: bool = False

    def __post_init__(self) -> None:
        if self.dev_keep is None and self.stage == "dev":
            self.dev_keep = self.keep


@dataclass(frozen=True)
class EvolveAgentConfig:
    max_hypotheses: int = 5
    max_revisions: int = 2
    max_edits: int = 4
    max_actions: int = 30
    max_configuration_operations: int = 5
    max_signal_confirmations: int = 3
    site_cooldown_campaigns: int = 1
    dev_seed: int = 42
    confirmation_seeds: tuple[int, ...] = (42, 43, 44)
    metric_only: bool = False
    metric_study_path: str = ""
    # Paired stock/patched benchmark evaluations, shared by a controller run.
    # None preserves an explicitly unbounded research run.
    # Enough for five 3-source screens, three full SHADOW candidates and one
    # three-patch FINAL batch in the current 2-scenario TS profile.
    max_measurement_pairs: int | None = 160
    reserve_final_measurement_pairs: int = 54
    # Optional hard deadline, useful for ephemeral Kaggle/Modal workers.
    max_measurement_seconds: float | None = None
    reserve_final_measurement_seconds: float = 0.0
    transfer_screen_sources: int = 3


@dataclass(frozen=True)
class EvolveRunPolicy:
    """Controller stages enabled for one run.

    Production uses every check. Unit tests can explicitly replace expensive
    stages without making runtime behavior depend on pytest environment state.
    """

    verify_manifest: bool = True
    confirm_and_ablate: bool = True
    confirm_small_signals: bool = True
    evaluate_final: bool = True
    # Hour-long Fedot(best_quality) is queued after a cheap technical filter.
    # Frozen PipelineBuilder scores never veto a quality hypothesis.
    fedot_quality_jobs: bool = True

"""Evidence gate between a plausible Scout lead and source modification."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from fedotllm.agents.evolve.agents.failures import classify_model_failure
from fedotllm.agents.evolve.storage.llm_audit import capture_structured_create
from fedotllm.agents.evolve.discovery.context import context_from_lead, open_runtime
from fedotllm.agents.evolve.discovery.fedot_context import build_fedot_context
from fedotllm.agents.evolve.discovery.targets import resolve_verification_target
from fedotllm.agents.evolve.storage.journal import write_artifact
from fedotllm.agents.evolve.discovery.research_tools import (
    callers_runtime,
    docs_runtime,
    format_snippet_feedback,
    focused_test,
    search_runtime,
    symbol_runtime,
)
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.types import (
    PatchSite,
    VerificationAction,
    VerificationResult,
)

# Cheap models commonly need a correction turn after their first synthetic
# reproduction uses a FEDOT constructor incorrectly.  Ten steps still keep the
# investigation bounded, while allowing the model to inspect the traceback and
# retrieve the actual public API before deciding verify/reject/hypothesis.
MAX_VERIFY_STEPS = 10
MAX_VERIFY_PROBE_CORRECTIONS = 2
QUALITY_CONFIRMATION_STEPS = 2
MAX_QUALITY_CHALLENGE_FAILURES = 2
MAX_VERDICT_CORRECTIONS = 1
MAX_VERIFY_CONTEXT = 24_000
CONTROLLER_OBSERVED_CRASH = "controller_observed_frozen_workload_crash"


class VerificationProposal(BaseModel):
    action: VerificationAction
    claim: str = ""
    expected: str = ""
    observed: str = ""
    query: str = ""
    symbol: str = ""
    file_path: str = ""
    line: int = Field(default=1, ge=1)
    run_code: str = ""
    test_node: str = ""
    reproduction_code: str = ""
    why: str = ""
    current_approach: str = ""
    proposed_approach: str = ""
    alternatives_considered: list[str] = Field(default_factory=list)
    generality: str = ""
    risks: list[str] = Field(default_factory=list)


class QualityChallengeProposal(VerificationProposal):
    """Only actions that can satisfy the post-hypothesis evidence contract."""

    action: Literal[
        VerificationAction.SYMBOL,
        VerificationAction.DOCS,
        VerificationAction.READ,
        VerificationAction.RUN,
        VerificationAction.FOCUSED_TEST,
    ]


class SourceQualityChallengeProposal(VerificationProposal):
    """Fallback after a model-authored runtime probe failed to execute."""

    action: Literal[
        VerificationAction.SYMBOL,
        VerificationAction.DOCS,
        VerificationAction.READ,
        VerificationAction.FOCUSED_TEST,
    ]


class FinalVerificationProposal(VerificationProposal):
    """A real schema boundary: navigation is impossible on a verdict turn."""

    action: Literal[
        VerificationAction.VERIFY_BUG,
        VerificationAction.QUALITY_HYPOTHESIS,
        VerificationAction.REJECT,
    ]


class ContractSupportAudit(BaseModel):
    """Fail-closed audit of the preconditions behind a reproduced assertion."""

    verdict: Literal["supported", "unsupported", "inconclusive"]
    reason: str
    evidence: list[str] = Field(default_factory=list)


def _target_contract_context(
    checkout: Path, file_path: str, line: int
) -> tuple[str, str, str]:
    """Resolve a qualified symbol and its source in the nominated file only.

    Never resolve a bare method name across the repository: constructors and
    methods such as ``fit`` occur in many unrelated classes. Return no context
    when the nominated location cannot be resolved, rather than guessing.
    """

    root = checkout.resolve()
    path = (root / file_path).resolve()
    if not path.is_relative_to(root / "fedot"):
        return "", "", ""
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError):
        return "", "", ""
    ancestry: list[ast.AST] = []

    def visit(node: ast.AST, parents: list[ast.AST]) -> None:
        nonlocal ancestry
        named = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        if named:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            if not start <= line <= node.end_lineno:
                return
            parents = [*parents, node]
            if len(parents) > len(ancestry):
                ancestry = parents
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree, [])
    if not ancestry:
        return "", "", ""
    symbol = ".".join(node.name for node in ancestry)
    target = ancestry[-1]
    start = min([target.lineno, *(d.lineno for d in target.decorator_list)])
    lines = text.splitlines()
    source = f"{file_path}:{start}-{target.end_lineno} ({symbol})\n" + "\n".join(
        f"{i + 1}|{lines[i]}" for i in range(start - 1, target.end_lineno)
    )
    docs = "\n\n".join(
        f"{file_path}:{getattr(node, 'lineno', 1)} "
        f"({getattr(node, 'name', '<module>')}):\n{ast.get_docstring(node)}"
        for node in [tree, *ancestry]
        if ast.get_docstring(node)
    )
    return symbol, source, docs


def _enclosing_symbol(checkout: Path, file_path: str, line: int) -> str:
    return _target_contract_context(checkout, file_path, line)[0]


def _linked_contract_context(checkout: Path, file_path: str, line: int) -> str:
    """Bounded AST-resolved types/bases, not a global same-name symbol search.

    A method's local docstring need not repeat contracts of its return type or
    inherited adapter. Runtime consumers can establish required consistency.
    No tests, evaluator, fixture labels, or external imports are consulted.
    """
    root = checkout.resolve()
    seen, chunks = set(), []

    def read(path):
        path = path.resolve()
        if not path.is_relative_to(root / "fedot"):
            return None
        try:
            content = path.read_text(encoding="utf-8")
            return ast.parse(content), content.splitlines()
        except (OSError, SyntaxError, UnicodeError):
            return None

    initial = read(root / file_path)
    if initial is None:
        return ""
    tree, _ = initial
    owners = [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
              and n.lineno <= line <= n.end_lineno]
    if not owners:
        return ""
    target = min(owners, key=lambda n: n.end_lineno - n.lineno)
    requested_methods = {n.attr for n in ast.walk(target) if isinstance(n, ast.Attribute)
                         and isinstance(n.value, ast.Name) and n.value.id == "self"}

    def resolve(path, name, depth):
        if depth > 3 or len(seen) >= 8 or (str(path), name) in seen:
            return
        loaded = read(path)
        if loaded is None:
            return
        module, lines = loaded
        node = next((n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name), None)
        if node is None:
            for imp in module.body:
                if not isinstance(imp, ast.ImportFrom) or imp.level or not (imp.module or "").startswith("fedot."):
                    continue
                for alias in imp.names:
                    if (alias.asname or alias.name) == name:
                        resolve(root / (imp.module.replace(".", "/") + ".py"), alias.name, depth + 1)
            return
        seen.add((str(path), name))
        identity = f"{path.relative_to(root)}:{node.lineno} ({name}, resolved import/base reference)"
        if node.end_lineno - node.lineno < 85:
            chosen = list(range(node.lineno - 1, node.end_lineno))
        else:
            chosen = [node.lineno - 1]
            for member in node.body:
                if (isinstance(member, (ast.AnnAssign, ast.Assign))
                    or isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant)
                    or isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and member.name in requested_methods | {"__init__"}):
                    chosen.extend(range(member.lineno - 1, member.end_lineno))
        rendered = "\n".join(f"{i + 1}|{lines[i]}" for i in sorted(set(chosen)))
        chunks.append(identity + "\n" + rendered[:4_000] +
                      ("\n[truncated; missing portions are not evidence]" if len(rendered) > 4_000 else ""))
        for base in node.bases:
            if isinstance(base, ast.Name):
                resolve(path, base.id, depth + 1)

    references = []
    if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)) and target.returns:
        references.extend(n.id for n in ast.walk(target.returns) if isinstance(n, ast.Name))
    for owner in owners:
        if isinstance(owner, ast.ClassDef):
            references.extend(b.id for b in owner.bases if isinstance(b, ast.Name))
    references.extend(sorted({n.id for n in ast.walk(target) if isinstance(n, ast.Name)}))
    for name in dict.fromkeys(references):
        resolve(root / file_path, name, 0)
    return "\n\n".join(chunks)[:14_000]


def _audit_contract_support(
    checkout: Path,
    lead: PatchSite,
    proposal: VerificationProposal,
    *,
    inference,
    stock_probe=None,
) -> ContractSupportAudit:
    """Check that a failing synthetic assertion represents supported behavior.

    Executing an assertion proves only what stock currently does. This second,
    fresh model call receives repository-owned contracts and callers so an
    unsupported precondition cannot be promoted into a correctness finding.
    """

    symbol, source, docs = _target_contract_context(
        checkout, lead.file_path, lead.line
    )
    if not source:
        return ContractSupportAudit(
            verdict="inconclusive",
            reason=f"target context unavailable: {lead.file_path}:{lead.line}",
        )
    # Supplemental search is explicitly non-authoritative. Search constructors
    # by their owner, never by the repository-wide '__init__' spelling.
    query = symbol.removesuffix(".__init__")
    related_docs = docs_runtime(checkout, query)
    callers = callers_runtime(checkout, query)
    linked = _linked_contract_context(checkout, lead.file_path, lead.line)
    prompt = f"""You are the final contract-support auditor for a reproduced FEDOT assertion.

The stock assertion already failed. That proves current behavior, but it does not
prove the asserted behavior is required. Return verdict=supported only when the
repository evidence below establishes that every trigger input and lifecycle state
used by the reproduction satisfies the public preconditions established by
repository documentation, resolved types/interfaces, and runtime caller/consumer
constraints. The exact method docstring need not repeat an inherited or return-type
contract. A downstream consumer requiring aligned arrays is evidence of required
consistency, not merely a policy preference. Conversely, the disputed statement
cannot prove its own correctness just because it currently exists. Return
verdict=unsupported when the trigger depends on a deliberately invalid argument,
a label/value excluded by the target docstring, an impossible caller lifecycle,
or an unspecified policy preference with positive repository evidence. Return
verdict=inconclusive when evidence is missing, ambiguous, truncated, or unrelated
to the exact target. Missing evidence is not proof of an unsupported contract. Do not
evaluate the proposed patch and do not infer a contract merely from the assertion.
The refined location must investigate the same public behavior as the original
suspicion below. A valid assertion about an unrelated contract does not support
this lead; report inconclusive when their relationship cannot be established.

Lead: {lead.file_path}:{lead.line}
Original public suspicion: {lead.why}
Claim: {proposal.claim or lead.why}
Expected healthy behavior: {proposal.expected}
Reproduction code:
```python
{proposal.reproduction_code}
```

Controller-observed stock execution (an earlier setup failure does not prove the
claimed invariant was violated; never trust comments saying an API is valid):
{stock_probe.output[-6_000:] if stock_probe is not None else '<not supplied; contract-only audit, not proof of reproduction>'}

Exact target symbol:
{source[:8_000]}

Repository documentation/docstrings:
{docs[:8_000]}

Resolved return/argument types and inherited runtime interfaces:
{linked or '<no resolvable local type/base references>'}

Supplemental documentation search (check identity before using as evidence):
{related_docs[:4_000]}

Possible repository call sites (not resolved references; check target identity):
{callers[:6_000]}
"""
    audited, _ = capture_structured_create(
        inference,
        prompt,
        ContractSupportAudit,
        stage="verifier",
        metadata={
            "lead_file": lead.file_path,
            "lead_line": lead.line,
            "contract_support_audit": True,
        },
    )
    return audited


def verification_from_observed_crash(lead: PatchSite) -> VerificationResult | None:
    """Promote an actual frozen-workload crash without asking LLM to recreate data.

    The evaluator already executed a valid frozen workload and captured exact
    FEDOT frames plus operation dataflow. A synthetic probe with different rows
    cannot disprove that observation. The proposed *cause* remains experimental:
    Fixer must still pass tests and make the same evaluator recover.
    """

    crash = next(
        (
            item.partition(":")[2].strip()
            for item in lead.evidence
            if item.startswith("stock runtime crash:")
        ),
        "",
    )
    frame_chain = next(
        (
            item.partition(":")[2].strip()
            for item in lead.evidence
            if item.startswith("FEDOT frame chain:")
        ),
        "",
    )
    if not crash or not frame_chain or not any(
        item.startswith("workload operation") or item.startswith("runtime data flow")
        for item in lead.evidence
    ):
        return None
    return VerificationResult(
        "verified_bug",
        claim=(
            f"Frozen supported workload crashed in FEDOT at "
            f"{lead.file_path}:{lead.line}: {crash}"
        ),
        expected="the same frozen workload completes after the candidate patch",
        observed=f"{crash}\nFEDOT frame chain:\n{frame_chain}",
        evidence=(CONTROLLER_OBSERVED_CRASH,),
        detail=(
            "Controller-observed frozen workload crash. The crash is factual; "
            "Scout's causal explanation is still a hypothesis and is accepted "
            "only if the patched evaluator recovers without regressions."
        ),
        current_approach=lead.mechanism or lead.why,
        proposed_approach=lead.proposed_change,
        generality=lead.expected_metric_effect,
        risks=("the proposed site may be a downstream symptom rather than the root cause",),
    )


def is_controller_observed_crash(result: VerificationResult) -> bool:
    return CONTROLLER_OBSERVED_CRASH in result.evidence


_SYSTEM = """You are the evidence researcher for a FEDOT source lead.
The lead is only a suspicion or an optimization opportunity. Do not call it a bug
merely because code looks odd, and do not assume a different algorithm is better.

Choose one action per turn:
- search: literal text search in FEDOT Python and frozen repository JSON (`query`)
- symbol: retrieve matching symbol bodies (`query`)
- callers: retrieve call sites (`symbol`)
- docs: retrieve frozen FEDOT docs, docstrings, and operation metadata (`query`)
- read: read a fedot/ function by file_path+line
- run: diagnostic Python against untouched FEDOT (`run_code`)
- focused_test: run one existing test/ node (`test_node`)
- verify_bug: provide a final `reproduction_code`
- quality_hypothesis: no correctness failure is claimed, or one could not be
  reproduced, but a concrete alternative mechanism may improve quality; fill
  current_approach, proposed_approach,
  alternatives_considered, generality, expected, and risks
- reject: evidence does not justify changing source

Every run_code action is standalone; include all imports and setup it needs.

For verify_bug, reproduction_code must construct valid supported FEDOT input and
file_path+line must nominate the exact method actually tested. You may correct
the Scout's approximate location within the same source file using inspected
source. Do not silently switch files or investigate an unrelated public contract.
When supplying symbol, use its exact qualified name at that source location.
The reproduction must
end in a meaningful assertion of healthy behavior. The untouched stock program
must fail that assertion. Never use `assert False`, `raise AssertionError`, mocks,
monkeypatching, deliberately invalid arguments, or source inspection as proof.
A trained model's accuracy, AUC, F1, RMSE, MAE, MSE, precision, recall, or another
quality threshold cannot prove a correctness bug; classify that evidence as a
quality_hypothesis and let frozen DEV judge it. Correctness assertions must test
a documented functional invariant such as no crash, shape, dtype, finite values,
metadata consistency, or deterministic encode/decode behavior. For classification
labels explicitly request `output_mode='labels'`; default prediction may contain
probabilities or scores and must not be compared directly with class labels.
A valid supported public call that unexpectedly raises must be converted into an
asserted healthy contract without hiding setup errors. Build all input outside the
try block, catch only the tested public call, and use this pattern:
`caught = None; try: pipeline.fit(data); except Exception as exc: caught = exc;`
`assert caught is None, f"valid supported call crashed: {caught}"`.
A traceback caused by a broken probe setup is not a reproduced bug.
Prefer reproduction through PipelineBuilder, PipelineNode, Operation, or another
documented public entry point. Instantiating a class from `operation_implementations`
directly is insufficient unless repository metadata/docs show a registered FEDOT
operation resolves to that exact concrete class and the probe uses the same params.

For quality_hypothesis, investigate the current implementation and at least
one realistic alternative. The proposal may replace an algorithm, use information
currently discarded, make a heuristic data-adaptive, improve numerical treatment,
or choose a more suitable representation. It must be general across a task family,
implementable with existing dependencies, and testable by DEV. State why it could
improve the named metric, where it may regress, and how the patch preserves public
contracts. It is not a verified bug and only DEV can confirm superiority.
Diagnostic effect size is not a quality verdict. Do not invent a minimum change
in coefficients, predictions, variance, or another proxy and use it to reject a
grounded mechanism. These quantities depend on input scale and the diagnostic
distribution; a small nonzero change can matter to the downstream metric. An
arbitrary proxy threshold is not a falsified semantic assumption. Distinguish a
mechanism that provably cannot execute or changes nothing from one whose quality
effect is uncertain. For the latter, retain the uncertainty and regression risks
in a quality_hypothesis so the unchanged DEV gates can measure it. This does not
justify an ungrounded alternative, an API violation, or a leakage-producing edit.
DEV is intentionally unavailable at this stage; that is not a reason to reject a
grounded comparative hypothesis. Use docs/symbol/callers to compare repository
siblings, operation defaults, or another existing-dependency mechanism first.
The first complete quality_hypothesis is provisional. After stating it, perform
at least one new adversarial `symbol`, `docs`, `read`, `run`, or `focused_test`
action aimed at disproving its key assumption (units, shape, lifecycle, supported
inputs, or sibling contract). Only then return the final quality_hypothesis. If
the challenge contradicts it, revise or reject it; do not defend the first story.
Runtime evidence may include exact `executed lines` plus the concrete implementation
and parameters used by the frozen workload. A proposed metric mechanism must touch
those executed lines or causally change their inputs/conditions. Treat a synthetic
failure in an unexecuted branch as a possible secondary correctness finding, not as
evidence that the frozen metric will change.
For a conditional quality mechanism, verify that its triggering condition is active
in at least one measured runtime trace. Executing the surrounding method is not
enough when the proposal depends on a particular upstream operation, dtype, shape,
parameter value, or branch. If the supplied dataflow and parameters contradict or
never show that condition, reject the metric hypothesis instead of sending a likely
no-op to DEV. A separate supported-input crash may still use verify_bug when its
functional contract is independently reproduced.
The temporal/train/DEV/FINAL boundary is immutable. A proposal that exposes holdout
features or targets to fitting, directly or through a changed split, is data leakage
and must be rejected even if it would lower the measured error.
Prefer reject over an API preference, cosmetic issue, or untested story.
Do not import evaluator, datasets, benchmark cases, scorer or task catalogs."""


def _probe_is_meaningful(code: str) -> tuple[bool, str]:
    lowered = (code or "").lower()
    if "import fedot" not in lowered and "from fedot" not in lowered:
        return False, "probe must import FEDOT"
    if any(token in lowered for token in ("monkeypatch", "unittest.mock", "mock.patch")):
        return False, "probe cannot mock runtime behavior"
    if re.search(r"\braise\s+AssertionError\b", code or ""):
        return False, "probe cannot raise AssertionError directly"
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return False, f"probe syntax error: {exc}"
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    if not assertions:
        return False, "probe must contain an assertion of healthy behavior"
    quality_tokens = {
        "accuracy",
        "accuracy_score",
        "auc",
        "roc_auc",
        "roc_auc_score",
        "f1",
        "f1_score",
        "precision",
        "precision_score",
        "recall",
        "recall_score",
        "rmse",
        "mae",
        "mse",
        "r2_score",
        "logloss",
        "log_loss",
    }
    for node in assertions:
        if isinstance(node.test, ast.Constant) and not bool(node.test.value):
            return False, "probe cannot use a constant failing assertion"
        names = {
            part.id.lower()
            for part in ast.walk(node.test)
            if isinstance(part, ast.Name)
        }
        names.update(
            part.attr.lower()
            for part in ast.walk(node.test)
            if isinstance(part, ast.Attribute)
        )
        if names & quality_tokens:
            return (
                False,
                "a model-quality threshold cannot verify a correctness bug; "
                "use quality_hypothesis and frozen DEV",
            )
    return True, ""


def _grounded_public_exception(
    proposal: VerificationProposal,
    stock,
) -> tuple[bool, str]:
    """Accept a crash as contract failure only when the public probe reached its target.

    A healthy-behaviour assertion is never executed when ``pipeline.fit`` raises.
    Requiring the terminal exception to be AssertionError made real crash bugs
    impossible to verify.  This gate stays fail-closed for common probe/API
    mistakes and for direct calls into private implementation classes.
    """

    if stock.status != "runtime_error" or "AssertionError" in stock.stderr:
        return False, ""
    code = proposal.reproduction_code or ""
    public_entry = any(
        token in code
        for token in (
            "fedot.core.pipelines.pipeline_builder",
            "fedot.core.pipelines.node",
            "fedot.core.operations.operation",
            "fedot.api.main",
            "PipelineBuilder(",
            "PipelineNode(",
            "Operation(",
            "Fedot(",
        )
    )
    if not public_entry or "operation_implementations" in code:
        return False, "probe did not use a public FEDOT entry point"
    target = (proposal.file_path or "").strip().replace("\\", "/")
    if not target.startswith("fedot/") or not target.endswith(".py"):
        return False, "verify_bug did not identify a FEDOT runtime target"
    stderr = (stock.stderr or "").replace("\\", "/")
    if target not in stderr:
        return False, "exception traceback did not reach the claimed runtime target"
    terminal = next(
        (line.strip() for line in reversed(stderr.splitlines()) if line.strip()),
        "",
    )
    setup_failures = (
        "SyntaxError:",
        "NameError:",
        "ModuleNotFoundError:",
        "ImportError:",
        "AttributeError:",
    )
    if terminal.startswith(setup_failures):
        return False, f"likely probe setup failure: {terminal}"
    if terminal.startswith("TypeError:") and any(
        marker in terminal
        for marker in (
            "unexpected keyword argument",
            "missing 1 required positional argument",
            "missing required positional argument",
            "takes ",
        )
    ):
        return False, f"likely public API misuse: {terminal}"
    return True, terminal


def verify_lead(
    checkout: Path,
    lead: PatchSite,
    *,
    inference,
    workspace: Path | None = None,
    max_model_calls: int | None = None,
    correctness_only: bool = False,
) -> VerificationResult:
    if inference is None:
        return VerificationResult("infrastructure_error", detail="verifier inference unavailable")
    base_context = context_from_lead(lead, checkout, max_chars=8_000)
    card = build_fedot_context(checkout, lead.file_path)
    if card is not None:
        base_context += "\n\nFEDOT architecture card:\n" + card.render(max_chars=3_500)
    tool_history: list[str] = []
    seen_tool_actions: set[tuple[str, str, str, int]] = set()
    seen_read_lines: dict[str, list[int]] = {}
    investigative_actions = 0
    provisional_quality: VerificationProposal | None = None
    quality_challenge_actions = 0
    disproved_bug_claims: list[str] = []
    last_detail = "verification step budget exhausted"
    step = 0
    step_budget = min(
        MAX_VERIFY_STEPS,
        max(1, int(max_model_calls)),
    ) if max_model_calls is not None else MAX_VERIFY_STEPS
    probe_corrections = 0
    failed_quality_challenges = 0
    verdict_corrections = 0
    while step < step_budget:
        step += 1
        quality_challenge_turn = (
            provisional_quality is not None and quality_challenge_actions == 0
        )
        final_verdict_turn = (
            not quality_challenge_turn
            and (
                step == step_budget
                or (
                    provisional_quality is not None
                    and quality_challenge_actions > 0
                )
            )
        )
        prefix = (
            f"{_SYSTEM}\n\nStep {step}/{step_budget}.\n"
            f"Lead context (always preserved):\n{base_context}\n\n"
            "Previous tool results:\n"
        )
        available = max(0, MAX_VERIFY_CONTEXT - len(prefix))
        recent = "\n\n".join(tool_history)[-available:] if available else ""
        prompt = prefix + (recent or "(none)")
        if correctness_only:
            prompt += (
                "\n\nThis is a bounded correctness check. Establish one public "
                "functional contract with an executable reproduction, or reject it. "
                "Do not propose a quality_hypothesis; metric alternatives use a "
                "separate DEV path. At most three model calls are available."
            )
        if quality_challenge_turn:
            prompt += (
                "\n\nMandatory post-hypothesis challenge turn. Return exactly one "
                "NEW symbol/docs/read/run/focused_test action aimed at disproving "
                "the provisional hypothesis's most important semantic assumption. "
                "A verdict is not allowed until this new evidence succeeds."
            )
            if failed_quality_challenges:
                prompt += (
                    " The previous model-authored challenge did not execute or "
                    "returned no evidence. Runtime `run` is now excluded: retrieve "
                    "an exact source, documentation, or frozen-test contract."
                )
        if final_verdict_turn:
            allowed_verdicts = (
                "verify_bug or reject"
                if correctness_only
                else "verify_bug, quality_hypothesis, or reject"
            )
            prompt += (
                "\n\nMandatory final verdict turn. Return " + allowed_verdicts + ". "
                "Navigation and experiments are "
                "excluded by the response schema. Incorporate the completed "
                "adversarial evidence rather than requesting another tool."
            )
        response_model: type[BaseModel]
        if final_verdict_turn:
            response_model = FinalVerificationProposal
        elif quality_challenge_turn and failed_quality_challenges:
            response_model = SourceQualityChallengeProposal
        elif quality_challenge_turn:
            response_model = QualityChallengeProposal
        else:
            response_model = VerificationProposal
        try:
            parsed, _ = capture_structured_create(
                inference,
                prompt,
                response_model,
                stage="verifier",
                metadata={
                    "lead_file": lead.file_path,
                    "lead_line": lead.line,
                    "tool_step": step,
                    "quality_challenge_turn": quality_challenge_turn,
                    "final_verdict_turn": final_verdict_turn,
                },
            )
        except Exception as exc:
            # Keep transport/budget/policy failures distinct from an
            # inconclusive contract investigation. The campaign decides
            # whether the category stops the run or merely skips this lead.
            raise classify_model_failure(exc) from exc
        action = parsed.action
        output = ""
        investigative = action in {
            VerificationAction.SEARCH,
            VerificationAction.SYMBOL,
            VerificationAction.CALLERS,
            VerificationAction.DOCS,
            VerificationAction.READ,
            VerificationAction.RUN,
            VerificationAction.FOCUSED_TEST,
        }
        key = (
            action.value,
            (parsed.file_path or parsed.query or parsed.symbol or parsed.test_node).strip(),
            (parsed.run_code or "").strip(),
            int(parsed.line or 1),
        )
        if final_verdict_turn and investigative:
            last_detail = (
                "Verifier violated the mandatory final-verdict schema with a "
                "navigation action."
            )
            tool_history.append(last_detail)
            if (
                max_model_calls is None
                and verdict_corrections < MAX_VERDICT_CORRECTIONS
            ):
                verdict_corrections += 1
                step_budget += 1
                continue
            break
        if investigative and key in seen_tool_actions:
            output = (
                "Duplicate tool action blocked; its observation is already in history. "
                "Use symbol/search for the missing source or make a final verdict."
            )
            last_detail = output
            tool_history.append(
                f"Step {step} duplicate action={action.value}: {output}"
            )
            continue
        if action is VerificationAction.READ:
            read_path = (parsed.file_path or "").strip()
            read_line = int(parsed.line or 1)
            nearby = next(
                (
                    previous
                    for previous in seen_read_lines.get(read_path, ())
                    if abs(previous - read_line) <= 40
                ),
                None,
            )
            if nearby is not None:
                output = (
                    "Near-duplicate read blocked: this file region was already "
                    f"shown around line {nearby}. Request the enclosing class/function "
                    "with action=symbol, search for the missing name, or make a verdict."
                )
                last_detail = output
                tool_history.append(
                    f"Step {step} near-duplicate read={read_path}:{read_line}: {output}"
                )
                continue
        if investigative:
            seen_tool_actions.add(key)
        if action is VerificationAction.READ:
            seen_read_lines.setdefault((parsed.file_path or "").strip(), []).append(
                int(parsed.line or 1)
            )
        valid_quality_challenge = False
        if action is VerificationAction.SEARCH:
            output = search_runtime(checkout, parsed.query)
            investigative_actions += 1
        elif action is VerificationAction.SYMBOL:
            output = symbol_runtime(checkout, parsed.query or parsed.symbol)
            investigative_actions += 1
            valid_quality_challenge = bool(output.strip()) and not output.lstrip().startswith("<")
        elif action is VerificationAction.CALLERS:
            output = callers_runtime(checkout, parsed.symbol or parsed.query)
            investigative_actions += 1
        elif action is VerificationAction.DOCS:
            output = docs_runtime(checkout, parsed.query)
            investigative_actions += 1
            valid_quality_challenge = bool(output.strip()) and not output.lstrip().startswith("<")
        elif action is VerificationAction.READ:
            output = open_runtime(checkout, parsed.file_path, line=parsed.line)
            investigative_actions += 1
            valid_quality_challenge = bool(output.strip())
        elif action is VerificationAction.RUN:
            result = run_fedot_snippet(checkout, parsed.run_code)
            output = format_snippet_feedback(checkout, result, max_chars=8_000)
            investigative_actions += 1
            # A probe that crashes because it guessed the FEDOT API is new
            # information for the researcher, but it did not challenge the
            # quality mechanism.  Requiring a clean run prevents setup errors
            # from satisfying the independent-evidence gate.
            valid_quality_challenge = result.status == "ok"
        elif action is VerificationAction.FOCUSED_TEST:
            test = focused_test(checkout, parsed.test_node)
            output = json.dumps(
                {
                    "status": test.status,
                    "exit_code": test.exit_code,
                    "output": test.output,
                },
                ensure_ascii=False,
            )
            investigative_actions += 1
            valid_quality_challenge = test.status in {"passed", "test_failures"}
        if (
            valid_quality_challenge
            and provisional_quality is not None
            and action
            in {
                VerificationAction.SYMBOL,
                VerificationAction.DOCS,
                VerificationAction.READ,
                VerificationAction.RUN,
                VerificationAction.FOCUSED_TEST,
            }
        ):
            quality_challenge_actions += 1
        elif quality_challenge_turn and investigative:
            failed_quality_challenges += 1
            if failed_quality_challenges >= MAX_QUALITY_CHALLENGE_FAILURES:
                result = VerificationResult(
                    "inconclusive",
                    claim=provisional_quality.claim if provisional_quality else lead.why,
                    observed=output[:4_000],
                    detail=(
                        "post-hypothesis adversarial evidence failed twice; "
                        "no unsupported verdict was sent to Fixer"
                    ),
                )
                _write_verification(workspace, lead, result)
                return result
            # Preserve one source-grounded challenge turn and one verdict turn,
            # even when the failed runtime probe consumed the original budget.
            remaining_steps = step_budget - step
            if remaining_steps < QUALITY_CONFIRMATION_STEPS:
                step_budget += QUALITY_CONFIRMATION_STEPS - remaining_steps
        if action is VerificationAction.QUALITY_HYPOTHESIS:
            if correctness_only:
                result = VerificationResult(
                    "inconclusive",
                    claim=parsed.claim or lead.why,
                    expected=parsed.expected,
                    observed=parsed.observed,
                    detail=(
                        "bounded correctness verification did not establish a "
                        "public contract violation"
                    ),
                )
                _write_verification(workspace, lead, result)
                return result
            if investigative_actions == 0:
                output = (
                    "A quality hypothesis requires at least one search, symbol, callers, "
                    "docs, read, run, or focused_test action grounded in repository evidence."
                )
                tool_history.append(
                    f"Step {step} attempted quality_hypothesis:\n{output}"
                )
                last_detail = output
                continue
            if (
                not parsed.current_approach.strip()
                or not parsed.proposed_approach.strip()
                or not parsed.expected.strip()
                or not parsed.generality.strip()
                or not parsed.alternatives_considered
            ):
                output = (
                    "A quality hypothesis must name the current mechanism, "
                    "a concrete proposed alternative, alternatives considered, "
                    "expected metric mechanism, and task-family generality."
                )
                tool_history.append(
                    f"Step {step} attempted quality_hypothesis:\n{output}"
                )
                last_detail = output
                continue
            if provisional_quality is None:
                provisional_quality = parsed
                # A cheap model may spend the whole investigation budget before
                # it can state a complete hypothesis.  The evidence contract
                # still requires one *new* adversarial observation followed by
                # a final verdict.  Reserve exactly those missing turns instead
                # of discarding a grounded lead merely because its provisional
                # hypothesis arrived on the originally reserved verdict turn.
                remaining_steps = step_budget - step
                if remaining_steps < QUALITY_CONFIRMATION_STEPS:
                    step_budget += QUALITY_CONFIRMATION_STEPS - remaining_steps
                output = (
                    "Hypothesis recorded as provisional. Before it can reach Fixer, "
                    "use one new symbol/docs/read/run/focused_test action specifically "
                    "to try to disprove its key semantic assumption. Then revise, "
                    "reject, or return the final quality_hypothesis."
                )
                tool_history.append(
                    f"Step {step} provisional quality_hypothesis:\n"
                    f"{parsed.model_dump_json()}\n{output}"
                )
                last_detail = output
                continue
            if quality_challenge_actions == 0:
                output = (
                    "A final quality hypothesis requires a NEW adversarial "
                    "symbol/docs/read/run/focused_test action after the provisional "
                    "hypothesis. Existing pre-hypothesis observations are insufficient."
                )
                tool_history.append(
                    f"Step {step} attempted final quality_hypothesis:\n{output}"
                )
                last_detail = output
                continue
            result = VerificationResult(
                "quality_hypothesis",
                claim=parsed.claim or lead.why,
                expected=parsed.expected,
                observed=parsed.observed,
                evidence=tuple(filter(None, (parsed.why,))),
                current_approach=parsed.current_approach,
                proposed_approach=parsed.proposed_approach,
                alternatives_considered=tuple(parsed.alternatives_considered),
                generality=parsed.generality,
                risks=tuple(parsed.risks),
            )
            _write_verification(workspace, lead, result)
            return result
        elif action is VerificationAction.REJECT:
            result = VerificationResult(
                "rejected",
                claim=parsed.claim or lead.why,
                expected=parsed.expected,
                observed=parsed.observed,
                detail=parsed.why or "verifier rejected the hypothesis",
            )
            _write_verification(workspace, lead, result)
            return result
        elif action is VerificationAction.VERIFY_BUG:
            valid, detail = _probe_is_meaningful(parsed.reproduction_code)
            target_lead, resolved_target = lead, {}
            if valid and correctness_only:
                try:
                    target_lead, resolved_target = resolve_verification_target(
                        checkout, lead, file_path=parsed.file_path,
                        line=parsed.line if "line" in parsed.model_fields_set else None,
                        symbol=parsed.symbol,
                    )
                except (OSError, ValueError, SyntaxError) as exc:
                    valid, detail = False, f"unresolved verification target: {exc}"
            if not valid:
                output = f"Probe rejected: {detail}"
            else:
                symbol = resolved_target.get("symbol", "")
                trace_target = {"file_path": target_lead.file_path, "symbol": symbol} if symbol else None
                stock = run_fedot_snippet(
                    checkout, parsed.reproduction_code,
                    **({"trace_target": trace_target} if trace_target else {}),
                )
                asserted_failure = (
                    stock.status == "runtime_error" and "AssertionError" in stock.stderr
                )
                grounded_exception, exception_detail = _grounded_public_exception(
                    parsed,
                    stock,
                )
                reproduced = asserted_failure or grounded_exception
                if reproduced and trace_target and stock.target_reached is not True:
                    reproduced = False
                    detail = (
                        "probe did not execute the claimed target method; an earlier setup failure "
                        "or assertion about another component is not evidence for this lead. "
                        "Use PipelineBuilder.add_node or PipelineNode with an operation name, "
                        "not an unconfigured base Operation object"
                    )
                if reproduced:
                    evidence = list(filter(None, (parsed.why,)))
                    if grounded_exception:
                        evidence.append(
                            "public FEDOT call reached the claimed runtime and raised "
                            f"before the healthy assertion: {exception_detail}"
                        )
                    if correctness_only:
                        try:
                            support_audit = _audit_contract_support(
                                checkout,
                                target_lead,
                                parsed,
                                inference=inference,
                                stock_probe=stock,
                            )
                        except Exception as exc:
                            failure = classify_model_failure(exc)
                            result = VerificationResult(
                                "infrastructure_error"
                                if failure.infrastructure
                                else "inconclusive",
                                claim=parsed.claim or lead.why,
                                expected=parsed.expected,
                                observed=parsed.observed or stock.output,
                                reproduction_code=parsed.reproduction_code,
                                stock_probe=stock,
                                detail=(
                                    "contract support audit failed closed: "
                                    f"{failure.category}: {failure.detail}"
                                )[:500],
                            )
                            _write_verification(workspace, lead, result)
                            return result
                        if support_audit.verdict != "supported":
                            result = VerificationResult(
                                "rejected" if support_audit.verdict == "unsupported" else "inconclusive",
                                claim=parsed.claim or lead.why,
                                expected=parsed.expected,
                                observed=parsed.observed or stock.output,
                                reproduction_code=parsed.reproduction_code,
                                stock_probe=stock,
                                evidence=tuple(support_audit.evidence),
                                detail=(
                                    ("unsupported public-contract precondition: "
                                     if support_audit.verdict == "unsupported"
                                     else "contract support inconclusive: ")
                                    + support_audit.reason
                                )[:1_000],
                            )
                            _write_verification(workspace, lead, result)
                            return result
                        evidence.append(
                            "contract support audit: " + support_audit.reason
                        )
                        evidence.extend(support_audit.evidence)
                    result = VerificationResult(
                        "verified_bug",
                        claim=parsed.claim or lead.why,
                        expected=parsed.expected,
                        observed=parsed.observed or stock.output,
                        reproduction_code=parsed.reproduction_code,
                        stock_probe=stock,
                        evidence=tuple(evidence),
                        resolved_target=resolved_target,
                    )
                    _write_verification(workspace, lead, result)
                    return result
                if stock.status == "ok":
                    detail = "stock satisfies the asserted healthy contract"
                    disproved_bug_claims.append(parsed.claim or lead.why)
                elif stock.status != "runtime_error":
                    detail = f"probe infrastructure status: {stock.status}"
                elif "AssertionError" not in stock.stderr:
                    detail = "probe crashed before its contract assertion"
                    if (
                        not correctness_only
                        and probe_corrections < MAX_VERIFY_PROBE_CORRECTIONS
                    ):
                        probe_corrections += 1
                        step_budget += 1
                probe_feedback = format_snippet_feedback(
                    checkout,
                    stock,
                    max_chars=4_000,
                )
                output = (
                    f"Bug was not verified: {detail}. Inspect the result and either "
                    "correct the probe, classify a grounded quality_hypothesis, or reject.\n"
                    f"Probe status={stock.status}; output:\n{probe_feedback}"
                )
                if stock.status == "runtime_error" and "AssertionError" not in stock.stderr:
                    output += (
                        "\n\nIf a valid public operation call itself unexpectedly raised, "
                        "do not abandon that correctness finding. Put all input/setup "
                        "before the try block, catch only the tested public call into "
                        "`caught`, then assert `caught is None`. An extra correction "
                        "turn has been reserved for this probe. If setup or arguments "
                        "were invalid, retrieve the exact API and reject or correct it."
                    )
        elif not investigative:
            output = "unsupported verifier action"
        last_detail = output[:4_000]
        tool_history.append(
            f"Step {step} action={action.value} output:\n```\n{last_detail}\n```"
        )
    if disproved_bug_claims and provisional_quality is None:
        result = VerificationResult(
            "rejected",
            claim=lead.why,
            observed=last_detail,
            evidence=tuple(disproved_bug_claims),
            detail=(
                "stock satisfied the proposed healthy contract; no independent "
                "failure or grounded comparative quality hypothesis remained"
            ),
        )
    else:
        result = VerificationResult(
            "inconclusive",
            claim=lead.why,
            observed=last_detail,
            detail="verification step budget exhausted",
        )
    _write_verification(workspace, lead, result)
    return result


def replay_reproduction(checkout: Path, verification: VerificationResult) -> dict:
    research = {
        "claim": verification.claim,
        "expected": verification.expected,
        "observed": verification.observed,
        "current_approach": verification.current_approach,
        "proposed_approach": verification.proposed_approach,
        "alternatives_considered": list(verification.alternatives_considered),
        "generality": verification.generality,
        "risks": list(verification.risks),
    }
    if is_controller_observed_crash(verification) and not verification.reproduction_code:
        return {
            "status": "verified_bug",
            "source": CONTROLLER_OBSERVED_CRASH,
            "stock": "failed_as_predicted",
            "patched": None,
            **research,
        }
    if verification.status != "verified_bug" or not verification.reproduction_code:
        return {
            "status": verification.status,
            "stock": None,
            "patched": None,
            **research,
        }
    patched = run_fedot_snippet(checkout, verification.reproduction_code)
    return {
        "status": "verified_bug",
        "stock": "failed_as_predicted",
        "patched": "resolved" if patched.status == "ok" else "still_failing",
        **research,
        "code": verification.reproduction_code,
        "patched_probe": {
            "status": patched.status,
            "exit_code": patched.exit_code,
            "stdout": patched.stdout,
            "stderr": patched.stderr,
        },
    }


def verification_context(result: VerificationResult) -> str:
    def compact(value: str, limit: int = 2_000) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        # The exception tail contains the violated assertion and concrete field;
        # the beginning supplies the high-level setup without repeating a full stack.
        return text[:500] + "\n...[stock output truncated]...\n" + text[-1_400:]

    return (
        f"Verifier status: {result.status}\n"
        f"Claim: {result.claim}\n"
        f"Expected: {result.expected}\n"
        f"Observed: {compact(result.observed)}\n"
        f"Current approach: {result.current_approach}\n"
        f"Proposed approach: {result.proposed_approach}\n"
        f"Alternatives considered: {', '.join(result.alternatives_considered)}\n"
        f"Generality: {result.generality}\n"
        f"Risks: {', '.join(result.risks)}\n"
        + (
            f"Exact stock-failing contract probe:\n```python\n{result.reproduction_code}\n```\n"
            if result.reproduction_code
            else ""
        )
    )


def _write_verification(
    workspace: Path | None,
    lead: PatchSite,
    result: VerificationResult,
) -> None:
    if workspace is None:
        return
    slug = f"{Path(lead.file_path).stem}-{lead.line}"
    folder = workspace / "verifications" / slug
    payload = {
        "status": result.status,
        "claim": result.claim,
        "expected": result.expected,
        "observed": result.observed,
        "reproduction_code": result.reproduction_code,
        "stock_probe": None if result.stock_probe is None else result.stock_probe.__dict__,
        "evidence": list(result.evidence),
        "detail": result.detail,
        "current_approach": result.current_approach,
        "proposed_approach": result.proposed_approach,
        "alternatives_considered": list(result.alternatives_considered),
        "generality": result.generality,
        "risks": list(result.risks),
        "resolved_target": result.resolved_target,
    }
    write_artifact(folder, "verification.json", json.dumps(payload, ensure_ascii=False, indent=2))
    if result.reproduction_code:
        write_artifact(folder, "reproduction.py", result.reproduction_code)

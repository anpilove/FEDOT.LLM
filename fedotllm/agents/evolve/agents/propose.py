from __future__ import annotations

import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from fedotllm.agents.evolve.storage.llm_audit import capture_structured_create
from fedotllm.agents.evolve.agents.failures import classify_model_failure
from fedotllm.agents.evolve.execution.patch import same_runtime, strip_gutter
from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit, ToolAction

_FIXER_BASE_HEAD_CHARS = 16_000
_FIXER_BASE_TAIL_CHARS = 8_000
_FIXER_TOOL_HISTORY_CHARS = 10_000


class PatchHunk(BaseModel):
    file_path: str = Field(
        default="",
        description="Path relative to checkout. Required when an edit touches another FEDOT file.",
    )
    old_code: str = ""
    new_code: str = ""


class PatchProposal(BaseModel):
    file_path: str = Field(
        description="FEDOT .py or repository .json path relative to the checkout"
    )
    old_code: str = ""
    new_code: str = ""
    edits: list[PatchHunk] = Field(default_factory=list)
    proposed_test_edits: list[PatchHunk] = Field(
        default_factory=list,
        description=(
            "Optional test/** changes documenting an intentional public-contract "
            "change. They are saved for human review and never applied during gates."
        ),
    )
    rationale: str = ""
    contract: str = Field(
        default="",
        description="One short sentence: what must stay consistent after the edit (fit vs transform, column layout).",
    )
    behavior_probe: str = Field(
        default="",
        description=(
            "Standalone deterministic Python run on both stock and patched FEDOT. "
            "It must print one EVOLVE_OBSERVATION=<value> line showing the behavior "
            "that this patch is intended to change."
        ),
    )
    status: ToolAction = Field(
        default=ToolAction.PATCH,
        description="patch, run, read another fedot/ file, or cannot_fix",
    )
    run_code: str = Field(
        default="",
        description="Python to execute in the FEDOT checkout when status=run",
    )
    query: str = Field(
        default="", description="Literal source query for status=search or symbol"
    )
    symbol: str = Field(
        default="", description="Function/method name for status=callers"
    )
    line: int = Field(
        default=1,
        ge=1,
        description="With status=read: 1 = whole file, else that function",
    )


_SYSTEM = """You patch the FEDOT library source. You see one whole file.
Read it and propose one SEARCH/REPLACE that can change model quality
(control flow, features, data contracts, defaults, bugs that affect metrics)
anywhere in that file.

This is not limited to bug repair. If the researcher supplied an improvement
hypothesis, implement its concrete alternative mechanism: for example a more
suitable existing algorithm, a data-adaptive rule, better use of available
training information, a numerically safer calculation, or a better representation.
Do not add dataset-specific constants, new dependencies, or silently replace the
public meaning of an operation. DEV will compare the alternative with stock.
Operation-default improvements belong in FEDOT's existing repository JSON rather
than an operation-specific `if` inside a shared Python strategy. Preserve explicit
user parameters: only change the default used when the user did not provide one.
For a configuration lead, runtime evidence may list the concrete implementation,
effective parameters, and supported parameter names. Change only supported parameters;
prefer a mild, general complexity/regularization alternative with a causal rationale.
Do not change train/test/DEV/FINAL boundaries or make evaluation targets/features
available during fit. A score increase from such leakage is invalid.

You may set status=run and put Python in run_code. It executes in the FEDOT
checkout (`import fedot` works). Use it to check behaviour before patching —
including what InputData looks like after this operation.
Every run_code is standalone: repeat its imports and setup on every run.
FEDOT implementation classes accept ``OperationParameters``, not a raw ``dict``.
For a direct implementation probe, construct parameters with
``OperationParameters(key=value, ...)``. Prefer PipelineBuilder for public pipeline
behavior when direct construction is unnecessary.
You may set status=read with file_path (whole file) or file_path+line (that
function) to pull a helper. Output/source comes back, then you patch, read,
run again, or set cannot_fix.
Use status=search with query to find literal text across FEDOT Python and frozen
repository JSON metadata,
status=symbol with query to retrieve matching symbol bodies, or status=callers
with symbol to inspect call sites. Prefer these over Python open()/grep probes.
Use status=docs with query to retrieve frozen FEDOT documentation, docstrings,
and operation metadata when the role or contract of this code is unclear.

Source edits may touch only `fedot/**`. Never put test changes in `edits`. If the
change intentionally updates a documented public contract, you may describe the
corresponding `test/**` update in `proposed_test_edits`; the controller preserves
it for human review but never applies it to the checkout used by pytest.
Do not import scoring harnesses, datasets, or case catalogs.
Runtime evidence is measured ground truth. Form a causal hypothesis and prefer
the smallest source change that directly tests it. Preserve public behavior and
producer/consumer contracts that the evidence does not show to be broken.
Respect source ownership: context labelled `Inherited runtime behavior` is shared
by sibling operations. For operation-specific behavior, override the method in
the named subclass instead of changing the shared base unless evidence shows all
siblings require the same change.
For a data-flow or metadata bug, inspect the concrete Data/OutputData fields and
the producer constructor that copies them. Do not infer that a field belongs to
`supplementary_data` merely because it is metadata; patch the actual owner and
verify every stale index/name field named by runtime evidence.
old_code must be copied from the source WITHOUT the "NNN|" line-number prefix
and must match the file uniquely. new_code must differ from old_code.
No comment-only, rename-only, or identical replacements.
Every final non-configuration patch must include `behavior_probe`: standalone deterministic Python
that uses a supported FEDOT entry point and prints exactly one compact final line
`EVOLVE_OBSERVATION=<value>`. The controller runs the identical probe on stock and
patched checkouts. For an operation-default repository edit the controller replaces
this with a canonical public repository probe. Do not inspect source text in the probe;
measure runtime behavior. The probe itself must exit successfully on both versions.
When stock raises the defect, catch the exception and encode its type or public message
in EVOLVE_OBSERVATION; do not use an uncaught exception or AssertionError as the signal.
If feedback says `behavior_probe_invalid`, do not guess another internal attribute,
method, return type, or dictionary key. First use status=read/symbol/docs/run to
retrieve the exact API and execute the repaired standalone probe successfully in
the checkout. A source patch accompanied by another unexecuted probe guess is not
a valid revision.
If a correct fix needs coordinated edits in two places, set status to
cannot_fix and leave old_code empty — do not invent a partial patch."""

_SYSTEM_MULTI = """You patch the FEDOT library source. You see source context.
Propose up to {n} SEARCH/REPLACE edits in `edits` that together change model quality.
Every edit has its own `file_path`, `old_code`, and `new_code`.
(control flow, features, data contracts, defaults). Coordinated fit/transform or
fit/predict changes belong in one candidate as separate hunks, not a partial first hunk.
FEDOT implementation classes accept ``OperationParameters``, not a raw ``dict``.
For direct implementation probes use ``OperationParameters(key=value, ...)``;
prefer PipelineBuilder when direct construction is unnecessary.
This includes grounded quality improvements where stock is correct but a concrete
existing-dependency algorithm, adaptive rule, numerical method, or representation
is plausibly stronger. Implement the researcher proposal without dataset-specific
constants or changing the public meaning of unrelated operations.
Operation-default improvements belong in FEDOT's existing repository JSON rather
than an operation-specific `if` inside a shared Python strategy. Preserve explicit
user parameters: only change the default used when the user did not provide one.
For a configuration lead, runtime evidence may list the concrete implementation,
effective parameters, and supported parameter names. Change only supported parameters;
prefer a mild, general complexity/regularization alternative with a causal rationale.
Never change train/test/DEV/FINAL boundaries or expose evaluation rows to fit.
Source `edits` may touch only `fedot/**`. Never put test changes in `edits`. If
the change intentionally updates a documented public contract, put the proposed
`test/**` update in `proposed_test_edits`; it is review-only and cannot make the
candidate pass its own pytest gate. Do not mention scoring harnesses, datasets,
case catalogs, or bug names.
Runtime evidence is measured ground truth. Form a causal hypothesis and prefer
the smallest coordinated source change that directly tests it. Preserve public
behavior and producer/consumer contracts that the evidence does not show to be broken.
Respect source ownership: context labelled `Inherited runtime behavior` is shared
by sibling operations. If DEV feedback reports a sibling regression, move the
behavior into the named subclass rather than editing the shared base class.
For a data-flow or metadata bug, inspect the concrete Data/OutputData fields and
the producer constructor that copies them. Do not infer that a field belongs to
`supplementary_data` merely because it is metadata; patch the actual owner and
verify every stale index/name field named by runtime evidence.
Each old_code must be copied from the source WITHOUT the "NNN|" line-number prefix
and must match the file uniquely. new_code must differ from old_code.
No comment-only, rename-only, or identical replacements.
Every final non-configuration patch must include `behavior_probe`: standalone deterministic Python
that uses a supported FEDOT entry point and prints exactly one compact final line
`EVOLVE_OBSERVATION=<value>`. The controller runs the identical probe on stock and
patched checkouts. For an operation-default repository edit the controller replaces
this with a canonical public repository probe. Do not inspect source text in the probe;
measure runtime behavior. The probe itself must exit successfully on both versions.
When stock raises the defect, catch the exception and encode its type or public message
in EVOLVE_OBSERVATION; do not use an uncaught exception or AssertionError as the signal.
If feedback says `behavior_probe_invalid`, do not guess another internal attribute,
method, return type, or dictionary key. First use status=read/symbol/docs/run to
retrieve the exact API and execute the repaired standalone probe successfully in
the checkout. A source patch accompanied by another unexecuted probe guess is not
a valid revision.
If even {n} hunks cannot make a correct coordinated fix, set status to
cannot_fix and leave edits empty.
You may set status=run and put Python in run_code first; it runs in the checkout.
Every run_code is standalone: repeat its imports and setup on every run.
You may set status=read with a fedot/ `file_path` and optional `line` to inspect
an operation or helper named by runtime evidence before proposing the edits.
Use status=search with query, status=symbol with query, or status=callers with
symbol for repository navigation; do not spend run_code on open()/grep.
Use status=docs with query for FEDOT architectural concepts and operation metadata."""


def build_prompt(context: str, *, max_edits: int = 1) -> str:
    system = _SYSTEM if max_edits <= 1 else _SYSTEM_MULTI.format(n=max(1, max_edits))
    return f"{system}\n\nSource context:\n{context}\n"


def _stable_base_context(context: str) -> str:
    """Keep both the lead source and trailing verifier/DEV feedback visible."""

    limit = _FIXER_BASE_HEAD_CHARS + _FIXER_BASE_TAIL_CHARS
    if len(context) <= limit:
        return context
    return (
        context[:_FIXER_BASE_HEAD_CHARS]
        + "\n\n...[middle of source context omitted; use navigation tools if needed]...\n\n"
        + context[-_FIXER_BASE_TAIL_CHARS:]
    )


def propose_patch(
    *,
    inference,
    context: str,
    errors: list[str] | None = None,
    max_edits: int = 1,
    raw_out: list[str] | None = None,
    checkout: Path | None = None,
    audit_metadata: dict | None = None,
) -> PatchCandidate | None:
    if inference is None or not context.strip():
        return None
    from fedotllm.agents.evolve.execution.run_code import MAX_STEPS, run_fedot_snippet

    base_context = _stable_base_context(context)
    tool_history: list[str] = []
    parsed = None
    last = ""
    last_run_code = ""
    tool_steps = max(1, MAX_STEPS)
    total_steps = tool_steps + 1
    for step in range(1, total_steps + 1):
        synthesis_step = step == total_steps
        recent_tools = "\n\n".join(tool_history)[-_FIXER_TOOL_HISTORY_CHARS:]
        working = (
            f"Step {step}/{total_steps}.\n"
            f"Original lead and experiment context (always preserved):\n{base_context}\n\n"
            f"Previous tool results:\n{recent_tools or '(none)'}"
        )
        if synthesis_step:
            working += (
                "\n\nFINAL SYNTHESIS STEP. Navigation budget is exhausted. "
                "Do not return run/read/search/symbol/callers/docs. Use the evidence "
                "already collected and return either one concrete status=patch "
                "candidate with exact SEARCH/REPLACE and behavior_probe, or "
                "status=cannot_fix if the evidence does not justify a safe patch."
            )
        prompt = build_prompt(working, max_edits=max_edits)
        try:
            parsed, raw = capture_structured_create(
                inference,
                prompt,
                PatchProposal,
                stage="fixer",
                metadata={
                    **(audit_metadata or {}),
                    "tool_step": step,
                    "synthesis_step": synthesis_step,
                },
            )
            if raw_out is not None and raw:
                raw_out.append(raw)
        except Exception as exc:
            failure = classify_model_failure(exc)
            last = str(failure)[:400]
            if errors is not None:
                errors.append(last)
            raise failure from exc
        status = (parsed.status or "patch").strip().lower()
        if synthesis_step and status in {
            "run",
            "read",
            "search",
            "symbol",
            "callers",
            "docs",
        }:
            if errors is not None:
                errors.append(f"final_synthesis_returned_tool:{status}")
            return None
        if status == "run" and checkout is not None and (parsed.run_code or "").strip():
            last_run_code = parsed.run_code.strip()
            run_result = run_fedot_snippet(checkout, parsed.run_code)
            from fedotllm.agents.evolve.discovery.research_tools import (
                format_snippet_feedback,
            )

            output = format_snippet_feedback(checkout, run_result, max_chars=8_000)
            tool_history.append(
                f"Step {step} action=run status={run_result.status}:\n```\n{output[:8_000]}\n```"
            )
            continue
        if (
            status == "read"
            and checkout is not None
            and (parsed.file_path or "").strip()
        ):
            from fedotllm.agents.evolve.discovery.context import open_runtime

            opened = open_runtime(
                checkout, parsed.file_path, line=int(parsed.line or 1)
            )
            if opened:
                tool_history.append(
                    f"Step {step} action=read {parsed.file_path}:\n{opened[:8_000]}"
                )
            else:
                tool_history.append(
                    f"Step {step} action=read: cannot open {parsed.file_path}"
                )
            continue
        if status in {"search", "symbol", "callers", "docs"} and checkout is not None:
            from fedotllm.agents.evolve.discovery.research_tools import (
                callers_runtime,
                docs_runtime,
                search_runtime,
                symbol_runtime,
            )

            if status == "search":
                opened = search_runtime(checkout, parsed.query)
            elif status == "symbol":
                opened = symbol_runtime(checkout, parsed.query or parsed.symbol)
            elif status == "docs":
                opened = docs_runtime(checkout, parsed.query)
            else:
                opened = callers_runtime(checkout, parsed.symbol or parsed.query)
            tool_history.append(f"Step {step} action={status}:\n{opened[:8_000]}")
            continue
        break
    if parsed is None:
        return None
    final_status = (parsed.status or "patch").strip().lower()
    if final_status == "cannot_fix":
        if errors is not None:
            errors.append("cannot_fix")
        return None
    rel = parsed.file_path.lstrip("/")
    if rel.startswith("fedot/") is False and "fedot/" in rel:
        rel = rel[rel.index("fedot/") :]
    edits: list[PatchEdit] = []
    for item in list(parsed.edits or [])[: max(1, max_edits)]:
        old_h = strip_gutter(item.old_code or "")
        new_h = strip_gutter(item.new_code or "")
        if old_h.strip() and not same_runtime(old_h, new_h):
            edit_rel = (item.file_path or rel).lstrip("/")
            if not edit_rel.startswith("fedot/") and "fedot/" in edit_rel:
                edit_rel = edit_rel[edit_rel.index("fedot/") :]
            edits.append(PatchEdit(edit_rel, old_h, new_h))
    if not edits:
        old_code = strip_gutter(parsed.old_code or "")
        new_code = strip_gutter(parsed.new_code or "")
        if old_code.strip() and not same_runtime(old_code, new_code):
            edits = [PatchEdit(rel, old_code, new_code)]
    if not edits:
        if errors is not None:
            errors.append("noop_or_empty_old")
        return None
    if max_edits <= 1:
        edits = edits[:1]
    proposed_test_edits: list[PatchEdit] = []
    for item in list(parsed.proposed_test_edits or [])[: max(1, max_edits)]:
        test_rel = (item.file_path or "").lstrip("/")
        old_h = strip_gutter(item.old_code or "")
        new_h = strip_gutter(item.new_code or "")
        if (
            test_rel.startswith("test/")
            and old_h.strip()
            and not same_runtime(old_h, new_h)
        ):
            proposed_test_edits.append(PatchEdit(test_rel, old_h, new_h))
    return PatchCandidate(
        candidate_id=uuid.uuid4().hex[:12],
        edits=edits,
        rationale=parsed.rationale or "",
        contract=(parsed.contract or "").strip(),
        behavior_probe=(parsed.behavior_probe or last_run_code).strip(),
        proposed_test_edits=proposed_test_edits,
    )

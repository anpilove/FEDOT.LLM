"""Fixer: one LLM patch on a scout lead. Writes only inside the FEDOT checkout."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.discovery.context import (
    _WHOLE_FILE_LINES,
    context_from_lead,
)
from fedotllm.agents.evolve.discovery.fedot_context import build_fedot_context
from fedotllm.agents.evolve.storage.journal import write_artifact
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.agents.probe_builder import repair_behavior_probe
from fedotllm.agents.evolve.agents.failures import (
    AgentModelFailure,
    classify_model_failure,
)
from fedotllm.agents.evolve.agents.propose import propose_patch
from fedotllm.agents.evolve.types import PatchCandidate, PatchSite, SnippetResult


_OBSERVATION_PREFIX = "EVOLVE_OBSERVATION="


def _probe_preflight_diagnostic(result: SnippetResult) -> str:
    """Explain why a model-authored quality probe is not executable on stock."""

    observations = [
        line
        for line in (result.stdout or "").splitlines()
        if line.startswith(_OBSERVATION_PREFIX)
    ]
    if result.status == "ok" and len(observations) == 1:
        return ""
    parts = [
        f"status={result.status}",
        f"exit_code={result.exit_code}",
        f"observation_count={len(observations)}",
    ]
    if result.detail:
        parts.append(f"detail={result.detail[-1_000:]}")
    if result.stdout:
        parts.append(f"stdout_tail={result.stdout[-1_000:]}")
    if result.stderr:
        parts.append(f"stderr_tail={result.stderr[-3_000:]}")
    return "\n".join(parts)


def _frozen_patch_text(candidate: PatchCandidate) -> str:
    return "\n\n".join(
        f"EDIT {index} {edit.file_path}\nSEARCH:\n{edit.old_code}\nREPLACE:\n{edit.new_code}"
        for index, edit in enumerate(candidate.edits, start=1)
    )


def _candidate_payload(
    candidate: PatchCandidate,
    *,
    lead: PatchSite,
    hypothesis_id: str,
    status: str,
) -> dict:
    """Durable branch state written before any fallible follow-up action."""

    return {
        "candidate_id": candidate.candidate_id,
        "hypothesis_id": hypothesis_id,
        "lead": asdict(lead),
        "edits": [
            {
                "file_path": edit.file_path,
                "old_code": edit.old_code,
                "new_code": edit.new_code,
            }
            for edit in candidate.edits
        ],
        "rationale": candidate.rationale,
        "contract": candidate.contract,
        "behavior_probe": candidate.behavior_probe,
        "status": status,
    }


def configuration_behavior_probe(
    lead: PatchSite,
    candidate: PatchCandidate,
) -> str:
    """Canonical public-API probe for an operation-default repository edit."""

    if lead.channel != "configuration" or not any(
        edit.file_path.endswith("default_operation_params.json")
        for edit in candidate.edits
    ):
        return ""
    operation = next(
        (
            item.partition(":")[2].strip()
            for item in lead.evidence
            if item.startswith("executed operation:")
        ),
        "",
    )
    if not operation:
        return ""
    operation_literal = json.dumps(operation)
    return "\n".join(
        (
            "import json",
            "from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository",
            f"params = DefaultOperationParamsRepository().get_default_params_for_operation({operation_literal})",
            'print("EVOLVE_OBSERVATION=" + json.dumps(params, sort_keys=True, default=str))',
        )
    )


def _class_name_at_line(tree: ast.AST, line: int) -> str | None:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if not matches:
        return None
    # Nested classes are possible; the narrowest range is the actual owner.
    owner = min(
        matches, key=lambda node: (node.end_lineno or node.lineno) - node.lineno
    )
    return owner.name


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _ancestor_names(tree: ast.AST, class_name: str) -> set[str]:
    classes = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    ancestors: set[str] = set()
    pending = [class_name]
    while pending:
        current = classes.get(pending.pop())
        if current is None:
            continue
        for base in current.bases:
            name = _base_name(base)
            if name and name not in ancestors:
                ancestors.add(name)
                pending.append(name)
    return ancestors


def _class_tokens(name: str) -> set[str]:
    token = "".join(char for char in name.lower() if char.isalnum())
    out = {token}
    for suffix in ("transformationimplementation", "implementation", "transformation"):
        if token.endswith(suffix):
            out.add(token[: -len(suffix)])
    return {item for item in out if len(item) >= 3}


def _direct_subclasses(tree: ast.AST, base_name: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(_base_name(base) == base_name for base in node.bases)
    }


def _regression_scope_diagnostics(
    checkout: Path,
    lead: PatchSite,
    candidate: PatchCandidate,
    feedback: str,
) -> list[str]:
    """Reject operation-specific fixes that mutate an in-file shared ancestor.

    This is activated only after a measured DEV regression. It does not guess
    whether a base-class change is generally wrong; it uses the experiment as
    evidence that the operation-specific repair leaked into sibling behavior.
    """

    if "outcome=regressed" not in feedback.lower():
        return []
    lead_path = checkout / lead.file_path
    lead_owner = None
    lead_ancestors: set[str] = set()
    if lead_path.is_file():
        try:
            lead_tree = ast.parse(
                lead_path.read_text(encoding="utf-8"), filename=str(lead_path)
            )
            lead_owner = _class_name_at_line(lead_tree, max(1, lead.line))
            if lead_owner is not None:
                lead_ancestors = _ancestor_names(lead_tree, lead_owner)
        except (OSError, SyntaxError, UnicodeError):
            pass

    intent = "".join(
        char
        for char in " ".join(
            (
                lead.file_path,
                lead.why,
                lead.mechanism,
                lead.proposed_change,
                lead.expected_metric_effect,
            )
        ).lower()
        if char.isalnum()
    )
    measured = "".join(char for char in feedback.lower() if char.isalnum())

    diagnostics: list[str] = []
    for edit in candidate.edits:
        edit_path = checkout / edit.file_path
        if not edit_path.is_file():
            continue
        try:
            source = edit_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(edit_path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        offset = source.find(edit.old_code)
        if offset < 0 or source.find(edit.old_code, offset + 1) >= 0:
            # The transactional patch validator reports missing/ambiguous SEARCH.
            continue
        edit_line = source.count("\n", 0, offset) + 1
        edit_owner = _class_name_at_line(tree, edit_line)
        if (
            Path(edit.file_path).as_posix() == Path(lead.file_path).as_posix()
            and edit_owner in lead_ancestors
        ):
            diagnostics.append(
                "measured sibling regression: operation-specific lead "
                f"{lead_owner} cannot edit shared ancestor {edit_owner}; "
                f"override the required behavior inside {lead_owner} instead"
            )
            continue
        if edit_owner is None:
            continue
        siblings = _direct_subclasses(tree, edit_owner)
        if len(siblings) < 2:
            continue
        intended = {
            name
            for name in siblings
            if any(token in intent for token in _class_tokens(name))
        }
        regressed = {
            name
            for name in siblings
            if any(token in measured for token in _class_tokens(name))
        }
        if intended and regressed - intended:
            target = ", ".join(sorted(intended))
            affected = ", ".join(sorted(regressed - intended))
            diagnostics.append(
                "measured sibling regression: edit in shared base "
                f"{edit_owner} targets {target} but changed regressed sibling "
                f"{affected}; override or hook the behavior only in {target}"
            )
    return diagnostics


def fix_lead(
    checkout: Path,
    lead: PatchSite,
    *,
    inference,
    workspace: Path | None = None,
    max_edits: int = 4,
    feedback: str = "",
    verification: str = "",
    validate_behavior_probe: bool = False,
    hypothesis_id: str = "",
    saved_candidate: PatchCandidate | None = None,
) -> PatchCandidate | None:
    # A 80k whole-file prompt made inexpensive models lose the actual lead and
    # verifier result.  The model can pull any omitted helper with read/search.
    ctx = context_from_lead(lead, checkout, max_chars=10_000)
    card = build_fedot_context(checkout, lead.file_path)
    if card is not None:
        ctx += "\n\nFEDOT architecture card:\n" + card.render(max_chars=3_500)
    if verification.strip():
        ctx += f"\n\nIndependent verifier result:\n{verification.strip()}\n"
    if feedback.strip():
        ctx += f"\n\nPrevious experiment feedback (DEV only):\n{feedback.strip()}\n"
    slug = f"{Path(lead.file_path).stem}-{lead.line}"
    if workspace is not None:
        write_artifact(workspace / "context" / slug, "llm_context.txt", ctx)
        target = checkout / lead.file_path
        n_lines = 0
        if target.is_file():
            n_lines = len(
                target.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        mode = "full_file" if n_lines <= _WHOLE_FILE_LINES else "sliced"
        write_artifact(
            workspace / "context" / slug,
            "context_meta.json",
            json.dumps(
                {
                    "context_mode": mode,
                    "context_lines": ctx.count("\n") + 1,
                    "context_chars": len(ctx),
                    "file_lines": n_lines,
                }
            ),
        )
    working_ctx = ctx
    all_errors: list[str] = []
    # Applying is deterministic and transactional.  Give the model a bounded
    # number of compact correction turns when SEARCH blocks are ambiguous or
    # invalid; this is format feedback, not hidden evaluator feedback.
    max_apply_revisions = 3
    for apply_revision in range(1, max_apply_revisions + 1):
        errors: list[str] = []
        candidate = saved_candidate if apply_revision == 1 else None
        if candidate is None:
            try:
                candidate = propose_patch(
                    inference=inference,
                    context=working_ctx,
                    errors=errors,
                    checkout=checkout,
                    max_edits=max_edits,
                    audit_metadata={
                        "lead_file": lead.file_path,
                        "lead_line": lead.line,
                        "apply_revision": apply_revision,
                        "has_dev_feedback": bool(feedback.strip()),
                    },
                )
            except AgentModelFailure as exc:
                if workspace is not None:
                    write_artifact(
                        workspace / "context" / slug,
                        "propose_error.txt",
                        "\n".join([*all_errors, *errors, str(exc)]),
                    )
                    write_artifact(
                        workspace / "context" / slug,
                        "fix_status.txt",
                        exc.category,
                    )
                raise
        all_errors.extend(errors)
        if candidate is None:
            reason = (
                "cannot_fix"
                if any("cannot_fix" in item for item in errors)
                else "no_patch"
            )
            if workspace is not None:
                if all_errors:
                    write_artifact(
                        workspace / "context" / slug,
                        "propose_error.txt",
                        "\n".join(all_errors),
                    )
                write_artifact(workspace / "context" / slug, "fix_status.txt", reason)
            return None
        automatic_probe = configuration_behavior_probe(lead, candidate)
        if automatic_probe:
            candidate.behavior_probe = automatic_probe
        if validate_behavior_probe:
            if not candidate.behavior_probe.strip():
                preflight_result = SnippetResult(
                    "runtime_error",
                    "",
                    detail="behavior probe is missing",
                )
                probe_diagnostic = "status=missing\nobservation_count=0"
            else:
                preflight_result = run_fedot_snippet(checkout, candidate.behavior_probe)
                probe_diagnostic = _probe_preflight_diagnostic(preflight_result)
            if probe_diagnostic:
                all_errors.append(
                    f"behavior_probe_preflight_invalid: {probe_diagnostic}"
                )
                if workspace is not None:
                    rejected = workspace / "candidates" / candidate.candidate_id
                    write_artifact(
                        rejected, "behavior_probe.py", candidate.behavior_probe
                    )
                    write_artifact(
                        rejected, "probe_preflight_failed.txt", probe_diagnostic
                    )
                    # Preserve the already useful source proposal before the
                    # optional probe-only LLM call. A provider timeout must not
                    # erase a patch that can be replayed and validated offline.
                    write_artifact(
                        rejected,
                        "frozen_source_patch.txt",
                        _frozen_patch_text(candidate),
                    )
                    write_artifact(
                        rejected,
                        "candidate.json",
                        json.dumps(
                            _candidate_payload(
                                candidate,
                                lead=lead,
                                hypothesis_id=hypothesis_id,
                                status="awaiting_probe_repair",
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
                try:
                    repaired_probe = repair_behavior_probe(
                        checkout,
                        inference=inference,
                        frozen_patch=_frozen_patch_text(candidate),
                        verification=verification,
                        failed_probe=candidate.behavior_probe,
                        failed_result=preflight_result,
                        audit_metadata={
                            "lead_file": lead.file_path,
                            "lead_line": lead.line,
                            "candidate_id": candidate.candidate_id,
                        },
                    )
                except Exception as exc:
                    failure = (
                        exc
                        if isinstance(exc, AgentModelFailure)
                        else classify_model_failure(exc)
                    )
                    if workspace is not None:
                        write_artifact(
                            rejected,
                            "probe_repair_error.txt",
                            str(failure),
                        )
                        write_artifact(
                            workspace / "context" / slug,
                            "fix_status.txt",
                            failure.category,
                        )
                    raise failure from exc
                if repaired_probe:
                    candidate.behavior_probe = repaired_probe
                    if workspace is not None:
                        write_artifact(
                            rejected, "behavior_probe_repaired.py", repaired_probe
                        )
                else:
                    if workspace is not None:
                        write_artifact(
                            workspace / "context" / slug,
                            "fix_status.txt",
                            "behavior_probe_repair_failed",
                        )
                    return None
        folder = None
        if workspace is not None:
            folder = workspace / "candidates" / candidate.candidate_id
            write_artifact(
                folder,
                "candidate.json",
                json.dumps(
                    _candidate_payload(
                        candidate,
                        lead=lead,
                        hypothesis_id=hypothesis_id,
                        status="ready_to_apply",
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            write_artifact(folder, "rationale.txt", candidate.rationale)
            write_artifact(folder, "contract.txt", candidate.contract)
            write_artifact(folder, "behavior_probe.py", candidate.behavior_probe)
            write_artifact(folder, "old.py", candidate.old_code)
            write_artifact(folder, "new.py", candidate.new_code)
            write_artifact(
                folder,
                "edits.json",
                json.dumps(
                    [
                        {
                            "file_path": edit.file_path,
                            "old_code": edit.old_code,
                            "new_code": edit.new_code,
                        }
                        for edit in candidate.edits
                    ],
                    indent=2,
                ),
            )
            if candidate.proposed_test_edits:
                write_artifact(
                    folder,
                    "proposed_test_edits.json",
                    json.dumps(
                        [
                            {
                                "file_path": edit.file_path,
                                "old_code": edit.old_code,
                                "new_code": edit.new_code,
                            }
                            for edit in candidate.proposed_test_edits
                        ],
                        indent=2,
                    ),
                )
        diagnostics = _regression_scope_diagnostics(checkout, lead, candidate, feedback)
        try:
            applied = not diagnostics and apply_patch(
                checkout, candidate, diagnostics=diagnostics
            )
        except (PermissionError, OSError) as exc:
            if folder is not None:
                write_artifact(
                    folder, "apply_error.txt", f"{type(exc).__name__}: {exc}"
                )
            if workspace is not None:
                write_artifact(
                    workspace / "context" / slug, "fix_status.txt", "apply_error"
                )
            return None
        if applied:
            if workspace is not None:
                write_artifact(
                    folder,
                    "candidate.json",
                    json.dumps(
                        _candidate_payload(
                            candidate,
                            lead=lead,
                            hypothesis_id=hypothesis_id,
                            status="applied_awaiting_decision",
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                write_artifact(
                    workspace / "context" / slug, "fix_status.txt", "applied"
                )
            return candidate
        detail = "; ".join(diagnostics) or "patch validation failed"
        if folder is not None:
            write_artifact(folder, "apply_failed.txt", detail)
        if workspace is not None:
            write_artifact(workspace / "context" / slug, "apply_failed.txt", detail)
        if apply_revision < max_apply_revisions:
            working_ctx = (
                ctx
                + "\n\nPrevious patch was rejected before any file was written.\n"
                + f"Apply diagnostics: {detail}\n"
                + "Return a corrected patch. Every SEARCH block must occur exactly once; "
                + "include neighboring source lines when needed.\n"
            )
            continue
        if workspace is not None:
            write_artifact(
                workspace / "context" / slug, "fix_status.txt", "apply_failed"
            )
        return None
    return None

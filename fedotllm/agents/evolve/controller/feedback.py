"""Compact, DEV-safe feedback passed between campaign revisions."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from fedotllm.agents.evolve.types import (
    Decision,
    PatchCandidate,
    ScoreResult,
    TestResult,
)


def _compact_reproduction_feedback(reproduction: dict) -> str:
    """Keep the patched failure mechanism visible to the next Fixer revision."""

    probe = reproduction.get("patched_probe")
    probe = probe if isinstance(probe, dict) else {}

    def tail(value, limit: int = 1_500) -> str:
        text = str(value or "").strip()
        return text[-limit:]

    payload = {
        "status": reproduction.get("status"),
        "stock": reproduction.get("stock"),
        "patched": reproduction.get("patched"),
        "claim": reproduction.get("claim"),
        "patched_probe": {
            "status": probe.get("status"),
            "exit_code": probe.get("exit_code"),
            "stdout_tail": tail(probe.get("stdout")),
            "stderr_tail": tail(probe.get("stderr")),
        },
    }
    return json.dumps(payload, ensure_ascii=False)[:4_000]


def _runtime_operation_evidence(
    rows: tuple[dict, ...],
    *,
    symbol: str = "",
) -> str:
    """Return causal runtime context without copying every estimator surface.

    A concrete implementation symbol benefits from its effective parameters.
    Shared preprocessing/dispatch functions do not: attaching all parameters of
    every operation made prompts larger while suggesting false causal links.
    """

    symbol_owner = symbol.split(".", 1)[0]
    symbol_token = re.sub(r"[^a-z0-9]", "", symbol_owner.lower())

    def directly_related(row: dict) -> bool:
        implementation = re.sub(
            r"[^a-z0-9]", "", str(row.get("implementation") or "").lower()
        )
        operation = re.sub(
            r"[^a-z0-9]", "", str(row.get("operation") or "").lower()
        )
        return bool(
            len(symbol_token) >= 3
            and (
                symbol_token in implementation
                or implementation in symbol_token
                or operation == symbol_token
            )
        )

    detailed = [row for row in rows if directly_related(row)]
    selected = detailed or list(rows)

    compact: list[str] = []
    seen: set[str] = set()
    for row in selected:
        operation = row.get("operation") or "<operation>"
        implementation = row.get("implementation") or "<implementation>"
        stage = row.get("stage") or "<stage>"
        input_width = (row.get("input") or {}).get("active_width")
        output_width = (row.get("output") or {}).get("active_width")
        error = row.get("error")
        item = f"{operation}/{implementation} {stage} width={input_width}->{output_width}"
        input_shape = (row.get("input") or {}).get("features_shape")
        output_shape = (row.get("output") or {}).get("predict_shape")
        if output_shape is None:
            output_shape = (row.get("output") or {}).get("features_shape")
        if input_shape is not None or output_shape is not None:
            item += f" shape={input_shape}->{output_shape}"
        if detailed:
            params = json.dumps(
                row.get("params") or {}, ensure_ascii=False, sort_keys=True, default=str
            )[:800]
            supported = ",".join(
                str(name) for name in (row.get("supported_parameters") or ())[:16]
            )
            item += f" params={params}"
            if supported:
                item += f" supports=[{supported}]"
        if error:
            item += f" error={error}"
        if item in seen:
            continue
        seen.add(item)
        compact.append(item)
        if len(compact) >= (4 if detailed else 8):
            break
    if not compact:
        return ""
    return "runtime operation instances: " + " | ".join(compact)


def _failure_path(result: ScoreResult) -> tuple[str, ...]:
    """Compare runtime stages without experiment paths or shifted line numbers."""
    return tuple(
        f"fedot/{relative}:{function.strip()}"
        for relative, function in re.findall(
            r'File "[^"\n]*/fedot/([^"\n]+)", line \d+, in ([^\n]+)',
            result.traceback,
        )
    )


def _dev_feedback(
    decision: Decision,
    patched: dict[str, ScoreResult],
    candidate: PatchCandidate | None = None,
    *,
    stock: dict[str, ScoreResult],
) -> str:
    outcome = "improved" if decision.keep else (
        "regressed" if decision.reason.startswith("regression") else "neutral"
    )
    changed_deltas = {
        task_id: value
        for task_id, value in decision.regression_deltas.items()
        if value is not None and abs(value) > 1e-12
    }
    status_changed = {
        task_id
        for task_id, result in patched.items()
        if task_id in stock
        and result.status != stock[task_id].status
    }
    failure_changed = {
        task_id
        for task_id, result in patched.items()
        if task_id in stock
        and result.status == stock[task_id].status == "crash"
        and (result.detail, _failure_path(result))
        != (stock[task_id].detail, _failure_path(stock[task_id]))
    }
    interesting_tasks = set(changed_deltas) | status_changed | failure_changed
    flow_tasks = status_changed | failure_changed | {
        task_id for task_id, value in changed_deltas.items() if value < 0
    }
    statuses = ", ".join(
        f"{task_id}:{patched[task_id].status}"
        for task_id in sorted(interesting_tasks)
    ) or "none"
    details = ", ".join(
        f"{task_id}:{result.detail[:240]}"
        for task_id, result in sorted(patched.items())
        if task_id in interesting_tasks and result.status != "ok" and result.detail
    )
    delta = "unknown" if decision.target_delta is None else f"{decision.target_delta:.8g}"
    feedback = (
        f"outcome={outcome}; DEV_delta={delta}; reason={decision.reason}; "
        f"affected_task_statuses={statuses}"
    )
    if details:
        feedback += f"; task_errors={details}"
    for task_id in sorted(failure_changed):
        feedback += (
            f"\nChanged failure path for {task_id}: "
            f"before={' -> '.join(_failure_path(stock[task_id])[-4:])}; "
            f"after={' -> '.join(_failure_path(patched[task_id])[-4:])}. "
            "This workload still has no valid metric because it crashes; "
            "repair the remaining stage of this same mechanism."
        )
    flow_rows: list[str] = []
    for task_id, result in sorted(patched.items()):
        if task_id not in flow_tasks:
            continue
        for row in result.dataflow[-8:]:
            flow_rows.append(
                f"{task_id} {row.get('operation')}/{row.get('stage')}: "
                f"input={json.dumps(row.get('input') or {}, sort_keys=True)}; "
                f"output={json.dumps(row.get('output') or {}, sort_keys=True)}; "
                f"error={row.get('error') or ''}"
            )
    if flow_rows:
        feedback += (
            "\nAfter-patch runtime data flow for failed/regressed workloads only:\n"
            + "\n".join(flow_rows)
        )
    if candidate is None:
        return feedback
    patch_text = _candidate_patch_text(candidate)
    instruction = (
        "This patch had no measurable effect. Stay within this lead and its "
        "verified causal data flow: do not target unrelated baseline failures, "
        "protect-only workloads, or a different operation. Do not repeat, rephrase, "
        "or cosmetically vary the same edits. Propose a materially different "
        "causal mechanism at this lead, or return cannot_fix."
        if outcome == "neutral"
        else "Use this exact previous patch and measured outcome when deciding whether the branch can be corrected."
    )
    return (
        feedback
        + "\n\nPrevious evaluated patch:\n"
        + patch_text
        + "\n"
        + instruction
    )


def _revision_feedback_context(history: list[str], *, max_chars: int = 8_000) -> str:
    """Keep failed mechanisms visible across a whole hypothesis branch.

    Each experiment is clean, but its causal constraints are cumulative.  The
    previous implementation replaced pytest feedback with the next probe/DEV
    result, so a cheap model repeatedly reintroduced a mutation already proven
    to break FEDOT contracts.  Keep the latest result plus compact, explicit
    evidence from earlier revisions within the Fixer's stable tail budget.
    """

    rows = [item.strip() for item in history if item and item.strip()]
    if not rows or max_chars <= 0:
        return ""
    deduped: list[str] = []
    for row in rows:
        if row not in deduped:
            deduped.append(row)
    if len(deduped) == 1:
        return deduped[0][-max_chars:]

    latest = deduped[-1]
    latest_budget = min(4_800, max_chars)
    if len(latest) > latest_budget:
        head = min(1_600, latest_budget // 2)
        latest = (
            latest[:head]
            + "\n...[latest feedback compacted]...\n"
            + latest[-max(0, latest_budget - head - 40) :]
        )
    remaining = max(0, max_chars - len(latest) - 180)
    previous = deduped[:-1][-3:]
    summaries: list[str] = []
    if previous and remaining:
        per_item = max(500, remaining // len(previous))
        for index, row in enumerate(previous, start=1):
            if len(row) > per_item:
                head = min(700, per_item // 2)
                tail = max(0, per_item - head - 34)
                row = row[:head] + "\n...[prior feedback compacted]...\n" + row[-tail:]
            summaries.append(f"Prior rejected revision {index}:\n{row}")
    prefix = (
        "Persistent branch constraints: do not repeat a patch mechanism that "
        "an earlier import gate, pytest contract, DEV, or SHADOW result "
        "already rejected. A new revision must causally address every retained "
        "failure below. A probe-only failure permits the same source edits with "
        "a better diagnostic; it does not establish that those edits have no effect.\n"
    )
    body = "\n\n".join(summaries + [f"Latest experiment result:\n{latest}"])
    if len(prefix) >= max_chars:
        return prefix[:max_chars]
    # The instruction is the contract for interpreting everything below it.
    # Truncate evidence from its oldest edge, never the contract itself.
    body_budget = max_chars - len(prefix)
    return prefix + body[-body_budget:]


def _test_result_diagnostics(result: TestResult, *, max_chars: int = 5_000) -> str:
    header = json.dumps(
        {
            "status": result.status,
            "exit_code": result.exit_code,
            "duration_s": result.duration_s,
            "cmd": result.cmd,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    output = (result.output or "").strip()
    if output:
        header += "\npytest_output_tail:\n" + output[-max(0, max_chars - len(header) - 24) :]
    return header[-max_chars:]


def _candidate_patch_text(candidate: PatchCandidate) -> str:
    patch_parts: list[str] = []
    remaining = 6_000
    for index, edit in enumerate(candidate.edits, start=1):
        block = (
            f"EDIT {index} {edit.file_path}\n"
            f"SEARCH:\n{edit.old_code}\n"
            f"REPLACE:\n{edit.new_code}\n"
        )
        if len(block) > remaining:
            block = block[: max(0, remaining)] + "\n...[previous patch truncated]"
        patch_parts.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return "\n".join(patch_parts)


def _candidate_confirmation_scope(
    source: Path,
    candidate: PatchCandidate,
    stock: dict[str, ScoreResult],
    operation_hints: dict[str, tuple[str, ...]],
    exam_ids: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Scope repeated splits only for edits inside concrete operation classes.

    Seed-42 DEV has already protected the full suite. Repeating every unaffected
    workload for DEV/SHADOW/FINAL adds no evidence. A scope is safe only when
    every edit belongs to a leaf class and runtime dataflow maps that class to a
    concrete operation. Shared bases, module helpers, ambiguous edits, or
    missing coverage fall back to the complete exam.
    """

    owners: set[str] = set()
    for edit in candidate.edits:
        path = source / edit.file_path
        try:
            text = path.read_text(encoding="utf-8")
            if text.count(edit.old_code) != 1:
                return None
            tree = ast.parse(text, filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            return None
        line = text.count("\n", 0, text.index(edit.old_code)) + 1
        classes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.lineno <= line <= (node.end_lineno or node.lineno)
        ]
        if not classes:
            return None
        owner = min(
            classes,
            key=lambda node: (node.end_lineno or node.lineno) - node.lineno,
        ).name
        has_subclasses = any(
            isinstance(node, ast.ClassDef)
            and any(
                (isinstance(base, ast.Name) and base.id == owner)
                or (isinstance(base, ast.Attribute) and base.attr == owner)
                for base in node.bases
            )
            for node in ast.walk(tree)
        )
        if has_subclasses:
            return None
        owners.add(re.sub(r"[^a-z0-9]", "", owner.lower()))

    operations: set[str] = set()
    direct_tasks: set[str] = set()
    for task_id, result in stock.items():
        for row in result.dataflow:
            implementation = re.sub(
                r"[^a-z0-9]", "", str(row.get("implementation") or "").lower()
            )
            if implementation and any(
                owner == implementation
                or owner in implementation
                or implementation in owner
                for owner in owners
            ):
                direct_tasks.add(task_id)
                operation = str(row.get("operation") or "").strip()
                if operation:
                    operations.add(operation)
    if not direct_tasks or not operations:
        return None
    affected = {
        task_id
        for task_id, hints in operation_hints.items()
        if any(operation in hints for operation in operations)
    }
    affected.update(direct_tasks)
    scope = tuple(task_id for task_id in exam_ids if task_id in affected)
    return scope or None


def _blocking_lift_crash_ids(
    stock: dict[str, ScoreResult],
    lift_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return crash probes only when no healthy lift workload can improve.

    Protect-only baseline crashes must never suppress evaluation of unrelated
    healthy lift tasks. Mixed lift sets also need the ordinary full verdict,
    because a healthy member can establish the required metric improvement.
    """

    available = [stock[task_id] for task_id in lift_ids if task_id in stock]
    if any(result.status == "ok" for result in available):
        return ()
    return tuple(result.task_id for result in available if result.status == "crash")

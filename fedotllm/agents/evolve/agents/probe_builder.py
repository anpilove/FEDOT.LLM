"""Repair a causal behavior probe without allowing source-patch changes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from fedotllm.agents.evolve.discovery.context import open_runtime
from fedotllm.agents.evolve.discovery.research_tools import (
    docs_runtime,
    format_snippet_feedback,
    symbol_runtime,
)
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.storage.llm_audit import capture_structured_create


class ProbeProposal(BaseModel):
    action: Literal["run", "read", "symbol", "docs", "cannot_fix"]
    run_code: str = ""
    file_path: str = ""
    line: int = 1
    query: str = ""
    why: str = ""


_SYSTEM = """You are ProbeBuilder for the FEDOT library. The source patch is frozen:
you cannot change, replace, or discuss its edits. Your only job is to produce a
standalone deterministic Python probe that uses the untouched checkout's supported
FEDOT API and prints exactly one final `EVOLVE_OBSERVATION=<compact value>` line.

Choose one action per turn:
- run: execute a complete proposed probe (`run_code`)
- read: inspect exact FEDOT source (`file_path`, `line`)
- symbol: inspect an exact API symbol (`query`)
- docs: inspect FEDOT docs/docstrings/operation metadata (`query`)
- cannot_fix: only when no supported public probe can measure the frozen mechanism

Never return an unexecuted probe. A successful `run` with exactly one observation is
accepted immediately. Every run_code is standalone and must repeat imports and setup.
FEDOT implementation classes accept `OperationParameters`, not a raw `dict`; use
`OperationParameters(key=value, ...)` for direct construction, or PipelineBuilder.
Do not inspect source text from the probe, import EvolveAgent evaluators, use mocks, or
read benchmark data. Prefer PipelineBuilder/PipelineNode and valid InputData. Treat the
runtime traceback and automatic API recovery below as ground truth."""


def _observation_count(stdout: str) -> int:
    return sum(
        line.startswith("EVOLVE_OBSERVATION=") for line in (stdout or "").splitlines()
    )


def repair_behavior_probe(
    checkout: Path,
    *,
    inference,
    frozen_patch: str,
    verification: str,
    failed_probe: str,
    failed_result,
    max_steps: int = 2,
    audit_metadata: dict | None = None,
) -> str | None:
    """Return an executed stock-valid probe while keeping the source diff frozen.

    ``max_steps`` bounds navigation.  A separate final synthesis call prevents
    a weak model from spending the whole allowance on useful reads and losing
    an otherwise valid source patch before it can execute a probe.
    """

    failure = format_snippet_feedback(checkout, failed_result, max_chars=6_000)
    stable = (
        f"{_SYSTEM}\n\nFrozen source patch (read-only):\n{frozen_patch[:6_000]}\n\n"
        f"Verifier context:\n{verification[:5_000]}\n\n"
        f"Rejected probe:\n```python\n{failed_probe[:5_000]}\n```\n\n"
        f"Observed failure and recovered API:\n{failure}"
    )
    history: list[str] = []
    for step in range(1, max(1, max_steps) + 1):
        prompt = f"{stable}\n\nStep {step}/{max_steps}. Previous tool results:\n" + (
            "\n\n".join(history)[-8_000:] or "(none)"
        )
        parsed, _ = capture_structured_create(
            inference,
            prompt,
            ProbeProposal,
            stage="probe_builder",
            metadata={**(audit_metadata or {}), "tool_step": step},
        )
        if parsed.action == "cannot_fix":
            return None
        if parsed.action == "read":
            output = open_runtime(
                checkout, parsed.file_path, line=max(1, int(parsed.line or 1))
            )
            history.append(f"action=read\n{output[:6_000] or '<not found>'}")
            continue
        if parsed.action == "symbol":
            output = symbol_runtime(checkout, parsed.query)
            history.append(f"action=symbol\n{output[:6_000]}")
            continue
        if parsed.action == "docs":
            output = docs_runtime(checkout, parsed.query)
            history.append(f"action=docs\n{output[:6_000]}")
            continue
        code = (parsed.run_code or "").strip()
        if not code:
            history.append("action=run rejected: run_code is empty")
            continue
        result = run_fedot_snippet(checkout, code)
        if result.status == "ok" and _observation_count(result.stdout) == 1:
            return code
        history.append(
            "action=run failed\n"
            + format_snippet_feedback(checkout, result, max_chars=6_000)
        )
    final_prompt = (
        f"{stable}\n\nFinal synthesis after navigation. Previous tool results:\n"
        + ("\n\n".join(history)[-10_000:] or "(none)")
        + "\n\nReturn action=run with the complete executable probe now, or "
        "action=cannot_fix. No more read, symbol, or docs actions are available."
    )
    parsed, _ = capture_structured_create(
        inference,
        final_prompt,
        ProbeProposal,
        stage="probe_builder",
        metadata={**(audit_metadata or {}), "tool_step": max_steps + 1, "final": True},
    )
    if parsed.action != "run" or not (parsed.run_code or "").strip():
        return None
    code = parsed.run_code.strip()
    result = run_fedot_snippet(checkout, code)
    if result.status == "ok" and _observation_count(result.stdout) == 1:
        return code
    return None

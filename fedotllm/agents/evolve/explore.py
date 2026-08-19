"""Let the agent run code instead of guessing.

Every defect the agent has fixed so far was handed to it: a linter found the spot
and a template built the proof. That ceiling is visible in the numbers — 39 of 40
lint findings in FEDOT are latent, and the project has never once, in six years,
committed a fix for that class.

The one genuinely reachable defect we know of was found by *running* FEDOT and
comparing what was asked for against what happened (a RANSAC parameter set to 0.5
and silently used as 0.96). No static rule can see that. This module gives the
agent the same move: a bounded loop where it writes a snippet, sees the real
output, and forms its next question from the result.

Safety: snippets run in the FEDOT venv with a timeout, in the repository
directory. They are model-written code executing locally — the same trust level as
the tests the agent already writes, and no more.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from fedotllm.llm import AIInference
from fedotllm.log import logger

MAX_STEPS = int(os.environ.get("FEDOTLLM_EXPLORE_STEPS", "6"))
SNIPPET_TIMEOUT_S = int(os.environ.get("FEDOTLLM_EXPLORE_TIMEOUT", "180"))
MAX_OUTPUT_CHARS = 2500

EXPLORE_SYS = (
    "You are investigating the aimclub/FEDOT AutoML library for REAL defects: "
    "behaviour that would surprise or harm a user. You cannot read the whole "
    "repository, but you CAN run Python against it and look at what actually "
    "happens.\n\n"
    "A real defect is something you can demonstrate by execution. For example:\n"
    "  - a parameter is accepted and then silently ignored or overridden;\n"
    "  - an error message exposes another library's internals instead of "
    "explaining the problem in FEDOT's own terms;\n"
    "  - the library writes to stdout instead of logging;\n"
    "  - the same input produces different results with a fixed seed;\n"
    "  - a documented promise does not hold.\n\n"
    "NOT a defect: style, naming, missing type hints, or anything a linter would "
    "flag but no user would ever notice.\n\n"
    "Work in steps. Each step, either run code or report a finding. Your snippets "
    "share state: what you defined earlier is still available, so build on it "
    "instead of starting over.\n\n"
    "To run code:\n"
    "<<<RUN>>>\n<python, prints what you want to see>\n<<<END>>>\n\n"
    "To report when you have DEMONSTRATED something by execution:\n"
    "FINDING: <one sentence on what is wrong>\n"
    "WHY_IT_MATTERS: <what a user experiences>\n"
    "FILE: <repo-relative path where it should be fixed>\n"
    "<<<TEST>>>\n<pytest that FAILS on the current code and passes once fixed>\n<<<END>>>\n\n"
    "Report only what you have actually observed in output you received. If your "
    "experiments show nothing wrong, say FINDING: none — that is a valid answer "
    "and better than inventing one."
)


@dataclass
class Exploration:
    steps: int = 0
    transcript: list[tuple[str, str]] = field(default_factory=list)
    finding: str = ""
    why: str = ""
    file_path: str = ""
    test_code: str = ""

    @property
    def found(self) -> bool:
        return bool(self.finding) and self.finding.strip().lower() != "none"


def echo_last_expression(code: str) -> str:
    """Print the value of a trailing bare expression, the way a notebook would.

    The agent writes REPL-style: it ends a snippet with `params` and expects to
    see the value. In a script that prints nothing, so it read "<no output>",
    assumed the call had failed and retried the same thing — four wasted steps in
    one run. Cheaper to meet the habit than to argue with it in the prompt.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name) \
            and last.value.func.id == "print":
        return code
    lines = code.splitlines()
    start = last.lineno - 1
    expr = "\n".join(lines[start : last.end_lineno])
    indent = expr[: len(expr) - len(expr.lstrip())]
    if indent:  # inside a block — leave it alone
        return code
    return "\n".join(lines[:start] + [f"print({expr.strip()})"] + lines[last.end_lineno :])


def run_snippet(repo: Path, python: str, code: str, history: list[str] | None = None) -> str:
    """Execute one snippet, replaying earlier successful ones first.

    Each snippet runs in a fresh interpreter, but the agent writes as if it were a
    session: it built a model in one step and called it in the next, and lost half
    its budget to `NameError`. Replaying the accepted history restores the
    continuity it assumes.
    """
    script = repo / ".fedotllm_explore.py"
    preamble = "import warnings, logging\nwarnings.filterwarnings('ignore')\n"
    if history:
        preamble += "\n".join(history) + "\n"
    code = echo_last_expression(code)
    try:
        script.write_text(preamble + code, encoding="utf-8")
        proc = subprocess.run(
            [python, str(script)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=SNIPPET_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"<timed out after {SNIPPET_TIMEOUT_S}s>"
    except OSError as exc:
        return f"<could not run: {exc}>"
    finally:
        script.unlink(missing_ok=True)
    out = "\n".join(p for p in (proc.stdout.strip(), proc.stderr.strip()) if p)
    return out[-MAX_OUTPUT_CHARS:] or "<no output>"


def _parse(raw: str) -> tuple[str | None, dict]:
    """Return (snippet_to_run, finding_fields). Exactly one of them is meaningful."""
    m = re.search(r"<<<RUN>>>\s*\n(.*?)\n?<<<END>>>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1), {}

    def field_of(name: str) -> str:
        f = re.search(rf"^{name}:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
        return f.group(1).strip() if f else ""

    test = re.search(r"<<<TEST>>>\s*\n(.*?)(?:\n?<<<END>>>|$)", raw, re.DOTALL | re.IGNORECASE)
    return None, {
        "finding": field_of("FINDING"),
        "why": field_of("WHY_IT_MATTERS"),
        "file_path": field_of("FILE"),
        "test_code": test.group(1) if test else "",
    }


def explore(inference: AIInference, repo: Path, python: str, hint: str = "") -> Exploration:
    """Bounded investigate-by-running loop; returns whatever was demonstrated."""
    result = Exploration()
    accepted: list[str] = []
    messages: list[dict] = [
        {"role": "system", "content": EXPLORE_SYS},
        {
            "role": "user",
            "content": (
                "Investigate FEDOT. `import fedot` works and the repository is the "
                "working directory. Start by checking something you can verify in one "
                "snippet, then follow what the output tells you."
                + (f"\n\nArea to look at first: {hint}" if hint else "")
            ),
        },
    ]

    for step in range(1, MAX_STEPS + 1):
        raw = inference.query(messages) or ""
        snippet, found = _parse(raw)
        if snippet is None:
            result.finding = found.get("finding", "")
            result.why = found.get("why", "")
            result.file_path = found.get("file_path", "")
            result.test_code = found.get("test_code", "")
            result.steps = step
            logger.info("explore: finished at step %s — %s", step, result.finding[:80])
            return result

        output = run_snippet(repo, python, snippet, accepted)
        # Only replay snippets that ran cleanly: a broken line would poison every
        # later step.
        failed = "Traceback" in output or output.startswith("<")
        if not failed:
            accepted.append(snippet)
        result.transcript.append((snippet, output))
        result.steps = step
        logger.info("explore step %s: ran %s chars, got %s chars", step, len(snippet), len(output))
        messages += [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Output:\n```\n{output}\n```\n"
                    + (
                        "This snippet failed, so it was NOT kept in the session — "
                        "anything it defined is gone. Redefine what you need.\n"
                        if failed
                        else ""
                    )
                    + f"Continue. Steps left: {MAX_STEPS - step}."
                ),
            },
        ]

    logger.info("explore: budget of %s steps exhausted without a finding", MAX_STEPS)
    return result

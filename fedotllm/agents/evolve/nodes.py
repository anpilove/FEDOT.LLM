"""Nodes for EvolveAgent (QA + full-repo code evolution).

Default: ensure_repo → run_evolve → END
QA: ensure_repo → collect_context → answer → END
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import AIMessage, convert_to_openai_messages

from fedotllm.agents.evolve.loop import run_evolution_loop
from fedotllm.agents.evolve.state import EvolveAgentState
from fedotllm.llm import AIInference
from fedotllm.log import logger

TARGET_REPO_URL = os.environ.get(
    "FEDOTLLM_REPO_URL", "https://github.com/aimclub/FEDOT.git"
)
REPO_CACHE = Path(
    os.environ.get(
        "FEDOTLLM_REPO_CACHE",
        str(Path(__file__).resolve().parents[3] / ".repo_cache" / "FEDOT"),
    )
)
MAX_FILES = 12
MAX_CHARS_PER_FILE = 4000


def _question(state: EvolveAgentState) -> str:
    msgs = convert_to_openai_messages(state["messages"])
    msgs = [msgs] if isinstance(msgs, dict) else msgs
    for m in reversed(msgs):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return ""


def _wants_evolve(state: EvolveAgentState) -> bool:
    """Default to evolve; force QA only when explicitly requested."""
    mode = (state.get("evolve_mode") or os.environ.get("FEDOTLLM_EVOLVE_MODE", "evolve")).lower()
    if mode == "explore":
        raise ValueError(
            "explore is a research-only workflow, not an EvolveAgent runtime mode"
        )
    if mode in {"qa", "answer"}:
        return False
    q = _question(state).lower()
    qa_markers = (
        "where is",
        "how does",
        "explain",
        "what file",
        "navigate",
        "find where",
    )
    evolve_markers = (
        "evolv",
        "improve",
        "patch",
        "fix the library",
        "fix fedot",
        "repo evolution",
        "update the framework",
        "actualiz",
    )
    if any(m in q for m in evolve_markers):
        return True
    if any(m in q for m in qa_markers) and not any(m in q for m in evolve_markers):
        return False
    return mode != "qa"


def ensure_repo(state: EvolveAgentState) -> EvolveAgentState:
    """Ensure a local checkout of the target repo exists; record its path."""
    local = os.environ.get("FEDOTLLM_REPO_PATH")
    if local and Path(local).is_dir():
        logger.info("EvolveAgent: using local repo at %s", local)
        state["repo_path"] = str(Path(local).resolve())
    else:
        if not REPO_CACHE.exists():
            REPO_CACHE.parent.mkdir(parents=True, exist_ok=True)
            logger.info("EvolveAgent: cloning %s -> %s", TARGET_REPO_URL, REPO_CACHE)
            subprocess.run(
                ["git", "clone", "--depth", "1", TARGET_REPO_URL, str(REPO_CACHE)],
                check=True,
            )
        else:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_CACHE,
                capture_output=True,
                text=True,
                check=True,
            )
            if not status.stdout.strip():
                subprocess.run(
                    ["git", "pull", "--ff-only"],
                    cwd=REPO_CACHE,
                    check=True,
                )
        state["repo_path"] = str(REPO_CACHE.resolve())

    if not state.get("workspace"):
        state["workspace"] = str(Path(state["repo_path"]).parent / "fedotllm-evolve")
    state["evolve_mode"] = "evolve" if _wants_evolve(state) else "qa"
    logger.info("EvolveAgent mode=%s workspace=%s", state["evolve_mode"], state["workspace"])
    return state


def route_after_ensure(state: EvolveAgentState) -> str:
    return "evolve" if state.get("evolve_mode", "evolve") == "evolve" else "qa"


def collect_context(state: EvolveAgentState) -> EvolveAgentState:
    """Build a keyword-relevant slice of the repo's Python source for QA."""
    root = Path(state["repo_path"])
    question = _question(state).lower()
    terms = [t for t in question.replace("/", " ").split() if len(t) > 3]

    py_files = [
        p
        for p in root.rglob("*.py")
        if ".git" not in p.parts and "test" not in p.name.lower()
    ]

    def score(p: Path) -> int:
        hay = (str(p.relative_to(root)) + " " + p.name).lower()
        return sum(hay.count(t) for t in terms)

    ranked = sorted(py_files, key=score, reverse=True)[:MAX_FILES]
    tree = "\n".join(
        sorted(
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if ".git" not in p.parts
        )[:200]
    )
    chunks = [f"# Repository source tree (python files, truncated)\n{tree}\n"]
    for p in ranked:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")[:MAX_CHARS_PER_FILE]
        except OSError:
            continue
        chunks.append(f"\n# FILE: {p.relative_to(root)}\n{text}")
    state["repo_context"] = "\n".join(chunks)
    return state


def answer(state: EvolveAgentState, inference: AIInference) -> EvolveAgentState:
    """Answer the user's question grounded in the collected repo source."""
    prompt = (
        "You are a repository-exploration agent for the aimclub/FEDOT AutoML "
        "framework. Using ONLY the source excerpts below, answer the user's "
        "request about the FEDOT codebase. Be concrete: cite file paths and "
        "symbol names. If asked to propose a change, describe the exact files "
        "and edits.\n\n"
        f"## User request\n{_question(state)}\n\n"
        f"## Repository source (excerpts)\n{state.get('repo_context', '')}"
    )
    response = inference.query(prompt) or "(no response)"
    state["messages"] = state["messages"] + [AIMessage(content=response)]
    return state


def run_evolve(state: EvolveAgentState, inference: AIInference) -> EvolveAgentState:
    """Full-repo scout → propose → validate → audit (background actualization)."""
    repo = Path(state["repo_path"])
    workspace = Path(state["workspace"])
    result = run_evolution_loop(
        inference=inference,
        repo=repo,
        workspace=workspace,
    )
    state["scout_pick"] = result.pick
    state["scout_why"] = result.why
    state["proposal"] = asdict(result.proposal)
    state["evolve_success"] = result.success
    state["evolve_severity"] = result.severity
    state["probe_resolved"] = result.probe_resolved
    state["evolve_abstained"] = result.abstained
    state["evolve_abstain_reason"] = result.abstain_reason
    state["evolve_evidence"] = result.evidence
    state["audit_path"] = result.audit_path
    state["audit_markdown"] = result.audit_markdown
    status = "SUCCESS" if result.success else "INCOMPLETE"
    summary = (
        f"Framework evolution {status}.\n"
        f"Scout pick: `{result.pick}` — {result.why}\n"
        f"Candidate: `{result.proposal.file_path}` — {result.proposal.problem}\n"
        f"Audit: `{result.audit_path}`\n\n"
        f"{result.audit_markdown[:4000]}"
    )
    state["messages"] = state["messages"] + [AIMessage(content=summary)]
    return state

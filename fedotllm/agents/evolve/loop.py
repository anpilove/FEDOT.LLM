"""Repo-level code evolution helpers for aimclub/FEDOT.

Ported from AutoDS-Tools/scripts/fedot_evolve_agent.py — LLM calls go through
AIInference (LiteLLM), not a bespoke OpenRouter urllib client.
"""

from __future__ import annotations

import ast
import datetime as dt
import difflib
import json
import os
import re
import subprocess
from functools import lru_cache
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.fixtures import as_pytest
from fedotllm.llm import AIInference
from fedotllm.log import logger

EXCLUDED_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "docs",
    "examples",
    "jupyter_notebooks",
}
HOTSPOT_PATTERNS = (
    "TODO",
    "FIXME",
    "NotImplemented",
    "workaround",
    "except Exception",
    "except:",
)
MAX_CHARS_PER_FILE = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_CHARS", "12000"))
# Show the picked file symbol by symbol instead of cutting it at a character
# budget. Set to 0 to fall back to truncation and measure the difference.
SYMBOL_VIEW = os.environ.get("FEDOTLLM_EVOLVE_SYMBOL_VIEW", "1") not in {"0", "false", "no"}
# Keep scout prompt small enough for low OpenRouter credit budgets (~10–14k tokens).
MAX_OUTLINE_CHARS = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_OUTLINE", "18000"))
# With lint grounding the scout needs far less raw outline material.
MAX_OUTLINE_CHARS_GROUNDED = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_OUTLINE_GROUNDED", "7000"))
MAX_FIX_TRIES = int(os.environ.get("FEDOTLLM_EVOLVE_FIX_TRIES", "3"))
# Best-of-N: sample independent candidates, keep the first that
# passes every gate. Cheap breadth beats deep tree search for this task.
NUM_CANDIDATES = int(os.environ.get("FEDOTLLM_EVOLVE_CANDIDATES", "3"))
# AutoML acceptance gate: the evolved framework must still fit a model on CPU.
AUTOML_GATE = os.environ.get("FEDOTLLM_EVOLVE_AUTOML_GATE", "1") not in {"0", "false", "no"}
# Runtime probe gate: cases the agent never wrote and never saw, compared before
# and after the patch. Costs one probe run per accepted candidate.
PROBE_GATE = os.environ.get("FEDOTLLM_EVOLVE_PROBE_GATE", "1") not in {"0", "false", "no"}
# Tuning gate: when the defect says a parameter cannot be used, the patch has to
# make the operation tunable, not merely stop the error.
TUNING_GATE = os.environ.get("FEDOTLLM_EVOLVE_TUNING_GATE", "1") not in {"0", "false", "no"}
# Value gate — the right to say "there is nothing here worth fixing".
#
# Every run used to have to produce something, and over 150 runs that something
# was 26 rewordings of an error message. Asking for restraint in the prompt does
# not work: three separate requirements (`FUNCTION`, two hunks, "prefer proven
# defects") were ignored by every model we tried, `gpt-4o` included. So the rule
# is enforced in code instead: a run counts only if it is anchored to evidence
# that existed BEFORE the patch (a template test that already failed, or an
# invariant violation measured on the running library), or if it demonstrably
# removed a runtime defect the agent never saw. Everything else abstains.
VALUE_GATE = os.environ.get("FEDOTLLM_EVOLVE_VALUE_GATE", "1") not in {"0", "false", "no"}
# Lint-derived targets, off by default. They were the first grounding mechanism,
# back when nothing else could tell the agent where to look, and they solved the
# problem of the day: without them the scout picked the same file 30 runs out of
# 30. The runtime scan now does that job strictly better, and the measurement is
# not close — 34 successful patches from runtime evidence, 32 of them class 1,
# against 10 from lint evidence, 6 of which are class 3.
#
# Worse, the class they can prove is the one measured to be dormant: of 40
# `RUF012` findings in FEDOT, 39 are never mutated, and the project has no commit
# about that class in six years. Meanwhile the lint classes that would matter —
# `S113` (a request that hangs forever), `S608` (SQL by concatenation), `B904`
# (a swallowed exception chain), 37 findings in total — cannot be turned into a
# failing test by introspection at all, so they never become targets.
#
# So the branch as it stands proves the useless half and cannot reach the useful
# half. Set FEDOTLLM_EVOLVE_TEMPLATES=1 to bring it back; making it worth having
# needs a triage step (is this finding live or dormant?), not more rules.
TEMPLATES_DEFAULT = "0"
# Triage: let the agent take a lint finding that nothing has proved yet, on the
# condition that it proves it itself. The abstention gate exists to stop invented
# work, but it also forbade the one thing only a model can do here — look at a
# warning and judge whether it is live or dormant. I checked those 40 `RUF012`
# findings by hand to learn that 39 are dormant; that is exactly the judgement
# call worth delegating.
#
# Nothing is taken on trust: the reproduce gate already demands a test that fails
# on the untouched tree, so a claim of "this one is live" has to be demonstrated
# before any patch is considered. Measured risk: on `gpt-4o-mini` this path
# produced 138 tests that did not fail, which is why it was closed. It is worth
# reopening because the model that failed at it is the same one that scored 0/12
# at repair while `gpt-5.4` scores 4/5.
TRIAGE_LINT = os.environ.get("FEDOTLLM_EVOLVE_TRIAGE", "0") not in {"0", "false", "no"}
# While a defect measured at runtime is available, restrict the scout to those
# files. Set to 0 to let lint-template defects compete on equal terms.
INVARIANTS_FIRST = os.environ.get("FEDOTLLM_EVOLVE_INVARIANTS_FIRST", "1") not in {"0", "false", "no"}

CPU_SMOKE_SNIPPET = """
import numpy as np
from sklearn.datasets import load_breast_cancer
from fedot.api.main import Fedot

data = load_breast_cancer()
X, y = data.data[:120], data.target[:120]
model = Fedot(problem="classification", timeout=0.5, preset="fast_train",
              n_jobs=1, logging_level=50)
model.fit(features=X, target=y)
pred = model.predict(X)
assert pred is not None and len(np.atleast_1d(pred).ravel()) == len(y)
print("CPU_SMOKE_OK")
"""
MAX_HOTSPOTS = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_HOTSPOTS", "40"))
# Static-analysis grounding: real, located findings instead of free-form guessing.
MAX_LINT_FINDINGS = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_LINT", "40"))
# Rules that map to genuine correctness/robustness smells (not formatting noise).
# Only rules whose warning can have a consequence at runtime. Measured on this
# repository: ruff reports 572 findings under the broad selection, and 481 of
# them are `RET*`/`SIM*`/`C4*` — an extra variable before a `return`, `dict()`
# instead of `{}`. Those cannot bite by the definition of the rule, and feeding
# them to the agent is paying a model to reformat someone else's library.
#
# The old default was "B,SIM,RET,C4,F", which had the opposite problem too: it
# excluded `RUF` and `S`, so the agent never once saw `RUF012`, `S113` (a request
# that can hang forever) or `S608` (SQL by concatenation) — the classes actually
# worth triaging.
LINT_RULES = os.environ.get(
    "FEDOTLLM_EVOLVE_LINT_RULES",
    "B006,B007,B008,B020,B904,B905,F821,F841,S113,S608,RUF012")
LINT_CMD = (
    os.environ.get("FEDOTLLM_EVOLVE_LINT_CMD", "").split()
    or None
)
MAX_TREE_FILES = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_TREE", "220"))
# Cross-run memory: without it the scout is a "memoryless explorer" and, at low
# temperature, re-picks the same target every run (measured: 30/30 identical).
JOURNAL_ENABLED = os.environ.get("FEDOTLLM_EVOLVE_JOURNAL_OFF", "") not in {"1", "true", "yes"}
JOURNAL_LIMIT = int(os.environ.get("FEDOTLLM_EVOLVE_JOURNAL_LIMIT", "25"))

SCOUT_SYS = (
    "You are a repository-evolution scout for aimclub/FEDOT "
    "(external attached agent performing repo-level code evolution).\n"
    "You receive a COMPLETE inventory of the fedot/ package (all .py paths, "
    "function/class outlines, hotspot index). Your job: pick ONE file that "
    "contains a small, safe, verifiable improvement opportunity.\n\n"
    "Prefer files where the inventory shows a **behavioural defect**, not just an "
    "unclear message. In descending value:\n"
    "  (a) `B008` — a function call in a default argument: the object is built once "
    "at import and shared by every call (a real Python trap);\n"
    "  (b) `B905`/`B007`/`F841` — silent truncation in `zip()`, an unused loop or "
    "local variable that was clearly meant to be used;\n"
    "  (c) unclear errors — KeyError/IndexError/AttributeError or a bare `except` "
    "where a clear ValueError/TypeError belongs;\n"
    "  (d) an explicit TODO/FIXME/workaround from the hotspot index.\n"
    "A repository full of polished error messages is not an evolved repository: do "
    "not keep choosing (c) while (a) or (b) sit unfixed.\n"
    "Use the 'Modules that already have a test module' section: those are the "
    "cheapest to verify, because a regression harness already exists. Pick a "
    "SMALL module — you must be able to prove the fix with one focused test.\n"
    "A file from the proven-defect list beats anything else: the defect is confirmed "
    "and its failing test already exists, so nothing has to be demonstrated from "
    "scratch.\n"
    "Avoid broad refactors and optimizer/core search changes.\n"
    "Do not default to the same file every run — choose on the evidence in the inventory.\n\n"
    "Reply in EXACTLY this format (no JSON):\n"
    "PICK: <repo-relative path under fedot/>\n"
    "WHY: <one sentence>"
)

PROPOSE_SYS = (
    "You are a repository-evolution agent that improves the aimclub/FEDOT AutoML "
    "library (background framework actualization — NOT call-site codegen). "
    "Propose ONE small, safe, self-contained improvement in the provided source file.\n\n"
    "PREFER THE MOST IMPACTFUL FIX YOU CAN PROVE. Rank candidates by severity:\n"
    "  1. **Behavioural defects** — code that is wrong even on valid input: a mutable/"
    "shared object built in a default argument (`def f(x=Task(...))` is created once at "
    "import and leaks between calls), state accidentally shared between instances, a "
    "value assigned but never used where it was clearly meant to be used.\n"
    "  2. **Silent data loss** — e.g. `zip()` over sequences that may differ in length "
    "truncates without warning. Only fix this when you can show the lengths can differ; "
    "adding `strict=True` changes behaviour and needs Python >= 3.10.\n"
    "  3. **Unclear errors** — invalid input raises KeyError/IndexError/AttributeError "
    "instead of a clear ValueError/TypeError with a helpful message.\n"
    "  4. Cosmetic-only changes (unused import, redundant assignment before return) — "
    "acceptable only if nothing above is provable.\n\n"
    "State the severity class (1-4) in RATIONALE. A proven class-3 fix beats an "
    "unprovable class-1 one, but do not settle for class 3 without looking for higher.\n\n"
    "Hard requirements:\n"
    "- Verifiable by a focused pytest test that FAILS on current code and PASSES after the patch.\n"
    "- OLD block MUST be an EXACT, VERBATIM, contiguous snippet copied from the shown file "
    "(full lines, exact indentation), unique in the file. Do NOT paraphrase or truncate.\n"
    "- UNIQUENESS IS STRICT: a single line is often NOT unique — the same parameter line "
    "can appear many times in one file. Before quoting, check how many times your snippet "
    "occurs; if more than once, ADD NEIGHBOURING LINES until the quote occurs exactly once. "
    "Always fill FUNCTION with the enclosing symbol name: it is used to disambiguate a "
    "snippet that still repeats.\n"
    "- If the fix touches TWO places (e.g. a constructed default: the signature must become "
    "`None` AND the body must build the object), emit the OLD/NEW pair TWICE — one pair per "
    "site, each an exact snippet. NEVER merge two sites into one hunk: glued-in body lines "
    "land inside the docstring and do nothing.\n"
    "- The test imports only real existing symbols and uses pytest.raises(...).\n\n"
    "Reply in EXACTLY this delimiter format (do NOT use JSON, so code stays intact):\n"
    "FILE: <repo-relative source path>\n"
    "FUNCTION: <name of the function or class you are editing>\n"
    "TEST_FILE: <repo-relative test path, a new module under test/unit/ is fine>\n"
    "TEST_NAME: <test function name>\n"
    "PROBLEM: <one sentence>\n"
    "RATIONALE: <why safe/correct>\n"
    "<<<OLD>>>\n<exact snippet to replace>\n<<<NEW>>>\n<replacement snippet>\n"
    "(repeat the OLD/NEW pair once per edited site)\n"
    "<<<TEST>>>\n<complete pytest function with all needed imports>\n<<<END>>>"
)


@dataclass
class CommandResult:
    command: str
    exit_code: int
    output: str


@dataclass
class Proposal:
    file_path: str = ""
    problem: str = ""
    rationale: str = ""
    old_code: str = ""
    new_code: str = ""
    test_file: str = ""
    test_name: str = ""
    test_code: str = ""
    # Several (old, new) edits of the same file. `old_code`/`new_code` mirror the
    # first one so older call sites keep working.
    hunks: list[tuple[str, str]] = field(default_factory=list)
    # Enclosing function/class, used to disambiguate a snippet that repeats.
    anchor: str = ""


@dataclass
class EvolveResult:
    proposal: Proposal
    pick: str = ""
    why: str = ""
    success: bool = False
    changed: list[str] = field(default_factory=list)
    reproduce: CommandResult | None = None
    validations: list[CommandResult] = field(default_factory=list)
    attempts: int = 0
    severity: int = 4
    severity_name: str = ""
    # Runtime defects that stopped reproducing after the patch.
    probe_resolved: list[str] = field(default_factory=list)
    audit_path: str = ""
    audit_markdown: str = ""
    # Abstention: the run finished without proposing anything, on purpose.
    abstained: bool = False
    abstain_reason: str = ""
    # What the run was anchored to: "template", "invariant", "probe" or "" (nothing).
    evidence: str = ""


def run_cmd(cmd: list[str], cwd: Path | None = None) -> CommandResult:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    out = "\n".join(p for p in (r.stdout.strip(), r.stderr.strip()) if p)
    return CommandResult(" ".join(cmd), r.returncode, out)


def require_clean_repo(repo: Path) -> None:
    """Refuse to evolve a checkout that contains user work."""
    if not (repo / ".git").exists():
        return
    status = run_cmd(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
    )
    if status.exit_code != 0:
        raise RuntimeError(f"cannot inspect target repository: {status.output}")
    if status.output:
        raise RuntimeError(
            "EvolveAgent requires a clean disposable checkout; refusing to overwrite "
            f"existing changes:\n{status.output}"
        )


def reset_evolution_changes(repo: Path, generated_tests: set[str]) -> None:
    """Restore tracked files and remove only tests created by this run."""
    if (repo / ".git").exists():
        reset = run_cmd(["git", "checkout", "--", "."], cwd=repo)
        if reset.exit_code != 0:
            raise RuntimeError(f"cannot restore target repository: {reset.output}")
    root = repo.resolve()
    for rel in generated_tests:
        path = (repo / rel).resolve()
        if root not in path.parents or not path.is_file():
            continue
        tracked = run_cmd(["git", "ls-files", "--error-unmatch", "--", rel], cwd=repo)
        if tracked.exit_code != 0:
            path.unlink()


def untracked_repo_files(repo: Path) -> set[str]:
    """List untracked files without losing spaces or special characters."""
    if not (repo / ".git").exists():
        return set()
    status = run_cmd(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=repo,
    )
    if status.exit_code != 0:
        raise RuntimeError(f"cannot inspect target repository: {status.output}")
    return {
        entry[3:]
        for entry in status.output.split("\0")
        if entry.startswith("?? ")
    }


def resolve_repo_python(repo: Path) -> str:
    """Python that has fedot + pytest installed (for validation)."""
    env = os.environ.get("FEDOTLLM_REPO_PYTHON")
    if env and Path(env).exists():
        return os.path.abspath(env)
    # Common local demo venv next to AutoDS-Tools
    candidates = [
        Path(__file__).resolve().parents[4]
        / "AutoDS-Tools"
        / "runs"
        / "fedot-evolve-demo"
        / ".venv-fedot"
        / "bin"
        / "python",
        repo.parent / ".venv-fedot" / "bin" / "python",
        repo / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return os.path.abspath(str(c))
    return os.path.abspath(os.environ.get("PYTHON", "python3"))


def iter_repo_python(repo: Path, under: str = "fedot") -> list[Path]:
    root = repo / under
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIR_PARTS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def file_outline(path: Path, rel: str) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return f"{rel} <unreadable>"
    lines = text.splitlines()
    sigs: list[str] = []
    for i, line in enumerate(lines, 1):
        s = line.rstrip()
        if re.match(r"^(async\s+)?def\s+\w+|^class\s+\w+", s):
            sigs.append(f"  L{i}: {s[:120]}")
        if len(sigs) >= 40:
            sigs.append("  ...")
            break
    head = f"{rel} ({len(lines)} lines, {len(text)} bytes)"
    return head + ("\n" + "\n".join(sigs) if sigs else "")


def scan_hotspots(repo: Path, files: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in files:
        rel = str(path.relative_to(repo))
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if any(p.lower() in low for p in HOTSPOT_PATTERNS):
                hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                if len(hits) >= MAX_HOTSPOTS:
                    return hits
    return hits


def scan_lint_findings(repo: Path, limit: int = MAX_LINT_FINDINGS) -> list[str]:
    """Collect real static-analysis findings as grounded evolution candidates.

    Free-form "find something to improve" makes the scout converge on the same
    file every run. A linter instead yields hundreds of concrete, located,
    deterministic candidates across the whole package — and gives an independent
    acceptance signal that does not depend on the model: re-running
    the linter to confirm a fix. Silently returns [] when ruff is unavailable.
    """
    # `--no-fix` and `--isolated` are not optional here. ruff reads configuration
    # from PARENT directories, and the FEDOT checkout sits under a project whose
    # pyproject.toml sets `fix = true`. Without these flags a scan silently
    # rewrote five source files before the agent proposed anything — the "pristine"
    # tree was never pristine.
    args = [
        "check",
        "fedot/",
        f"--select={LINT_RULES}",
        "--output-format=concise",
        "--no-fix",
        "--isolated",
    ]
    # ruff may be on PATH, importable in the venv, or reachable via uvx.
    candidates = [LINT_CMD] if LINT_CMD else [
        ["ruff", *args],
        [sys.executable, "-m", "ruff", *args],
        ["uvx", "ruff", *args],
    ]
    proc = None
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False, timeout=180)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.stdout and ":" in proc.stdout:
            break
        proc = None
    if proc is None:
        logger.info("EvolveAgent: lint scan unavailable (ruff not found) — skipping grounding")
        return []
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ":" in ln]
    if not lines:
        return []
    # Round-robin over rule codes so one noisy rule cannot dominate the list.
    by_rule: dict[str, list[str]] = {}
    for ln in lines:
        m = re.search(r"\s([A-Z]+[0-9]+)\s", ln)
        by_rule.setdefault(m.group(1) if m else "OTHER", []).append(ln)
    spread: list[str] = []
    while len(spread) < limit and any(by_rule.values()):
        for code in list(by_rule):
            if by_rule[code]:
                spread.append(by_rule[code].pop(0))
                if len(spread) >= limit:
                    break
            else:
                del by_rule[code]
    logger.info("EvolveAgent: %s lint findings across %s rules", len(lines), len(set(
        re.search(r"\s([A-Z]+[0-9]+)\s", ln).group(1) for ln in lines if re.search(r"\s([A-Z]+[0-9]+)\s", ln)
    )))
    return spread


# Failures are not pure waste: each one lands in the journal, and the journal is
# what eventually talks the agent out of a bad idea. One module failed three runs
# running — every one of them on the same rejected approach — and then succeeded
# on the fourth in a single attempt, because by then three records said so. A
# threshold of two would have banned it one run before it paid off.
MAX_FAILED_ATTEMPTS = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_FAILS", "4"))


def exhausted_files(journal: list[dict[str, Any]] | None) -> set[str]:
    """Files to stop offering: already fixed, or failed past the point of learning."""
    fixed: set[str] = set()
    failures: dict[str, int] = {}
    for e in journal or []:
        f = e.get("file")
        if not f:
            continue
        if e.get("success"):
            fixed.add(f)
        else:
            failures[f] = failures.get(f, 0) + 1
    return fixed | {f for f, n in failures.items() if n >= MAX_FAILED_ATTEMPTS}


@lru_cache(maxsize=4)
def _all_lint_findings(repo: str) -> tuple[str, ...]:
    """Ruff output for the whole tree, once per process rather than once per call."""
    return tuple(scan_lint_findings(Path(repo), limit=400))


# Rules whose warning can only matter if something actually writes to the
# flagged object. Everything else in ruff's output for this repository is shape,
# not behaviour: of 572 findings, 481 are `RET*`/`SIM*`/`C4*` and cannot have a
# consequence by the definition of the rule.
MUTATION_RULES = ("RUF012",)


def _attribute_at(repo: Path, file_rel: str, line: int) -> tuple[str, str] | None:
    """(class, attribute) named by a mutable-class-attribute warning."""
    path = repo / file_rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not getattr(node, "end_lineno", None):
            continue
        if not (node.lineno <= line <= node.end_lineno):
            continue
        for sub in node.body:
            if getattr(sub, "lineno", None) != line:
                continue
            targets = (sub.targets if isinstance(sub, ast.Assign)
                       else [sub.target] if isinstance(sub, ast.AnnAssign) else [])
            for t in targets:
                if isinstance(t, ast.Name):
                    return node.name, t.id
    return None


def is_dormant(repo: Path, attribute: str) -> bool:
    """True when nothing in the repository ever writes to `attribute`.

    A mutable class attribute is only a defect if it is mutated: shared state
    that everybody reads and nobody changes behaves exactly like a constant.
    Measured on FEDOT: of 40 `RUF012` findings, 39 are dormant and the fortieth
    is a cache, mutated on purpose. Six years of history contain no commit about
    this class — patching them costs a maintainer a review and changes nothing.

    Written by hand first, which is what showed the class was worthless; this is
    the same check, done in seconds instead of an evening.
    """
    writes = (
        re.compile(rf"\bself\.{re.escape(attribute)}\s*(=[^=]|[+\-*/|&]=)"),
        re.compile(rf"\.{re.escape(attribute)}\s*(=[^=]|[+\-*/|&]=)"),
        re.compile(rf"\.{re.escape(attribute)}\.(append|extend|update|add|pop|"
                   r"remove|clear|insert|setdefault|sort)\("),
        re.compile(rf"\.{re.escape(attribute)}\s*\[[^\]]*\]\s*="),
    )
    for path in repo.glob("fedot/**/*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if attribute not in text:
            continue
        for pattern in writes:
            if pattern.search(text):
                return False
    return True


def drop_dormant(repo: Path, findings: list[str]) -> tuple[list[str], list[str]]:
    """Split lint findings into ones worth a look and ones provably inert.

    Deterministic, no model involved: asking an LLM whether an attribute is ever
    assigned is paying for something `ast` answers exactly.
    """
    live, dormant = [], []
    for line in findings:
        try:
            location, rest = line.split(": ", 1)
            file_rel, line_no, _ = location.split(":")[:3]
            rule = rest.split(" ", 1)[0]
        except (ValueError, IndexError):
            live.append(line)
            continue
        if rule not in MUTATION_RULES:
            live.append(line)
            continue
        found = _attribute_at(repo, file_rel, int(line_no))
        if found is None:
            live.append(line)
            continue
        (dormant if is_dormant(repo, found[1]) else live).append(line)
    return live, dormant


def lint_findings_for(repo: Path, file_rel: str, limit: int = 12) -> list[str]:
    """Lint warnings sitting in one file, as leads for triage."""
    here = [ln for ln in _all_lint_findings(str(repo))
            if ln.startswith(file_rel + ":")]
    live, _ = drop_dormant(repo, here)
    return live[:limit]


def triage_section(leads: list[str]) -> str:
    """Task text when the only thing available is an unproven warning.

    The agent is being asked the question a linter cannot answer: does this
    warning describe something that actually goes wrong? Most do not — of 40
    `RUF012` findings in this repository, 39 are dormant, the flagged object is
    only ever read. A patch for a dormant warning costs a maintainer a review and
    changes nothing, which is how projects end up drowning in automated noise.
    """
    listed = "\n".join(f"  - {ln}" for ln in leads)
    return (
        "\n\n## Unproven warnings in this file — decide whether any is real\n"
        f"{listed}\n\n"
        "Nothing here is proven. A linter reports a *shape* in the source; "
        "whether it causes anything is a separate question, and most of the time "
        "the answer is no — in this repository 39 of 40 findings of one rule turn "
        "out to be dormant, the flagged object is never mutated.\n\n"
        "So: pick at most one of these, and only if you can demonstrate it. Your "
        "TEST must FAIL on the current code — not because the shape is present, "
        "but because behaviour is wrong. Reading the source and asserting that a "
        "line looks a certain way proves nothing and will be rejected.\n"
        "If none of them is demonstrably live, say so and change nothing: reply "
        "with PROBLEM: dormant, and leave the OLD/NEW blocks empty."
    )


def proven_defects_section(
    repo: Path, py: str, journal: list[dict[str, Any]] | None = None
) -> str:
    """Defects that already have a failing test, offered as the preferred targets."""
    if os.environ.get("FEDOTLLM_EVOLVE_TEMPLATES", TEMPLATES_DEFAULT) == "0":
        return ""
    try:
        from fedotllm.agents.evolve.templates import module_path, proven_defects_cached

        proven = proven_defects_cached(repo, py)
    except Exception as exc:  # templates must never break the round
        logger.warning("proven-defect list unavailable (%s)", exc)
        return ""

    # Hide what has already been done. Otherwise memory and value pull apart: the
    # journal says "not that file again" and pushes the agent out of the proven
    # list entirely, onto an unproven cosmetic target — which is what happened.
    # There is no need for that: 35 proven defects sit in 23 different modules.
    skip = exhausted_files(journal)
    if skip:
        proven = [
            g
            for g in proven
            if not any(f"from {module_path(f)} import" in g.test_code for f in skip if f)
        ]
    if not proven:
        return ""
    lines = [
        f"## Proven defects — {len(proven)}, each already has a FAILING test",
        "These are the strongest targets in the repository: the defect is real and "
        "the proof exists, so you do not have to invent one. Pick one of these files "
        "unless you have a clearly better reason.",
    ]
    for g in proven[:12]:
        lines.append(f"  - [{g.rule}] {g.target}")
    return "\n".join(lines)


def ready_section(repo: Path, file_rel: str, ready) -> str:
    """Task text when a proven defect exists: exact symbol, ready proof, no test to write."""
    from fedotllm.agents.evolve.templates import symbol_source

    source = symbol_source(repo, file_rel, ready.target)
    located = (
        f"\n\nTHE DEFECT IS IN THIS SYMBOL AND NOWHERE ELSE:\n```python\n{source}\n```\n"
        "Other methods in this file look similar — leave them alone. A patch that "
        "changes them does not make the test pass."
        if source
        else ""
    )
    return (
        "\n\n## The proof already exists — do NOT write a test\n"
        f"Confirmed defect in `{ready.target}` (rule {ready.rule})."
        f"{located}\n"
        f"This test FAILS on the current code:\n```python\n{ready.test_code}```\n"
        "Write ONLY the patch that makes it pass. Copy the test verbatim into the "
        f"TEST block and use TEST_NAME: {ready.test_name}."
    )


def invariant_defects(repo: Path) -> list[dict[str, Any]]:
    """Behavioural defects measured on the running library, with a ready test.

    Produced offline by `invariants.py` (scan, then `--bridge`) and read here
    from its JSON. Kept out of the round itself because the scan fits every
    operation in the repository and costs minutes, while the file it writes is
    valid for as long as the checkout does not move.
    """
    path = Path(os.environ.get("FEDOTLLM_INVARIANTS",
                               "/tmp/fedotllm_invariants.bridge.json"))
    if not path.is_file():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("invariant findings unreadable (%s)", exc)
        return []
    return [i for i in items if (repo / i.get("file", "")).is_file()]


def verified_defects(repo: Path) -> list[dict[str, Any]]:
    """Defects the verifier reproduced through the library's public interface.

    The strongest evidence the pipeline produces, and until now the fixer could
    not see it at all: the evidence sources were `template`, `invariant` and
    `triage`, so a defect with a script that fails on the untouched tree had no
    way of reaching a patch except by hand.

    Written by `examples/verify_leads.py`. Only rows it marked `confirmed` count
    — `internal only` means the code path exists but no caller can walk it, and
    that is not a defect a user will ever meet.
    """
    path = Path(os.environ.get("FEDOTLLM_VERIFIED",
                               "/tmp/fedotllm_verified.json"))
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("verified findings unreadable (%s)", exc)
        return []
    items = []
    for row in rows:
        if row.get("status") != "confirmed" or not row.get("script"):
            continue
        if not (repo / row.get("file", "")).is_file():
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", f"{row['file']}_{row['line']}".lower()).strip("_")
        items.append({**row,
                      "test_name": f"test_verified_{slug}"[:80],
                      "test_code": as_pytest(row["script"], f"test_verified_{slug}"[:80],
                                             row.get("why", ""), row.get("got", ""))})
    return items


def verified_defect_for(repo: Path, file_rel: str) -> dict[str, Any] | None:
    """The verified defect belonging to a file, if there is one."""
    for item in verified_defects(repo):
        if item.get("file") == file_rel:
            return item
    return None


def verified_defects_section(repo: Path) -> str:
    """Above everything else, and the rank is earned by how it was established."""
    items = verified_defects(repo)
    if not items:
        return ""
    lines = [
        f"## Defects reproduced through the public API — {len(items)}, each with a FAILING test",
        "The strongest targets in the repository. For each of these a short script "
        "was run against this checkout and raised, using only `Fedot`, "
        "`FedotBuilder` or `Pipeline` — so a user can reach it. The failing test "
        "already exists; your job is the patch, not the proof.",
    ]
    for i in items[:12]:
        lines.append(f"  - {i['file']}:{i['line']} — {i.get('got', 'raises')} — "
                     f"{(i.get('why') or '').strip()[:160]}")
    return "\n".join(lines)


def verified_section(repo: Path, item: dict[str, Any]) -> str:
    """Task text for a verified defect: here is what breaks, and how it was shown."""
    return (
        "## Your target\n"
        f"`{item['file']}` line {item['line']} — {item.get('why', '').strip()}\n\n"
        f"This was not inferred from reading. The script below was run against "
        f"this checkout and raised `{item.get('got', 'an exception')}`:\n\n"
        f"```python\n{item.get('script', '').strip()}\n```\n\n"
        f"Last line of the failure:\n```\n{(item.get('detail') or '').strip()}\n```\n\n"
        f"Route: {item.get('route', 'public interface')}.\n\n"
        "Patch the library so this script completes. Do not make the script pass "
        "by weakening what it asks for — if the call is legal, it must work or "
        "fail with a message that names what is wrong and what was expected.\n"
    )


def invariant_defects_section(repo: Path) -> str:
    """Ranked above lint and above template defects, and the ranking is earned:
    these are the only findings measured on the library while it runs."""
    items = invariant_defects(repo)
    if not items:
        return ""
    lines = [
        f"## Behavioural defects measured at runtime — {len(items)}, each with a FAILING test",
        "The strongest targets in the repository. Each one was found by running the "
        "library: a hyperparameter was declared through the public API and the fitted "
        "object turned out to hold a different value. No linter can see this. Prefer "
        "these over everything else below.",
    ]
    for i in items[:12]:
        if i.get("kind") == "unusable_parameter":
            lines.append(
                f"  - {i['file']} — `{i['operation']}` declares `{i['param']}` tunable, "
                f"but every value in the declared scope raises "
                f"{i.get('observed', {}).get('error', 'an error')}")
        else:
            lines.append(
                f"  - {i['file']} — `{i['operation']}` accepts {i['param']}={i['value']!r}, "
                f"works with {list((i.get('observed') or {'?': '?'}).values())[0]}")
    return "\n".join(lines)


def invariant_defect_for(repo: Path, py: str, file_rel: str) -> dict[str, Any] | None:
    """The measured invariant violation belonging to a file, if there is one."""
    for item in invariant_defects(repo):
        if item.get("file") == file_rel:
            return item
    return None


def bind_evidence_test(
    proposal: Proposal,
    ready: Any = None,
    invariant: dict[str, Any] | None = None,
    verified: dict[str, Any] | None = None,
) -> Proposal:
    """Keep the pre-patch proof outside the model's control."""
    if verified is not None:
        # The proof is the script the verifier already ran against this
        # checkout, wrapped into a test by code. The model never gets to
        # restate what counts as broken.
        proposal.test_name = verified["test_name"]
        proposal.test_code = verified["test_code"]
    elif ready is not None:
        proposal.test_name = ready.test_name
        proposal.test_code = ready.test_code
    elif invariant is not None:
        proposal.test_name = invariant["test_name"]
        proposal.test_code = invariant["test_code"]
    else:
        return proposal
    proposal.test_file = "test/unit/test_fedotllm_evolve_proof.py"
    return proposal


def invariant_section(repo: Path, item: dict[str, Any]) -> str:
    """Task text for a measured defect: the proof already exists, write the patch."""
    if item.get("kind") == "unusable_parameter":
        body = (
            f"Measured on the running library: `{item['param']}` is declared tunable for "
            f"`{item['operation']}` in `PipelineSearchSpace`, and fitting with "
            f"{item['value']!r} — a value from that same declared scope — raises "
            f"{item.get('observed', {}).get('error', 'an error')} from inside a "
            "third-party library. Every value in the scope fails the same way, so the "
            "parameter cannot be used at all.\n"
            "Two shapes of cause are possible and the patch must establish which: the "
            "operation passes a parameter it should not (for example a `None` it never "
            "filtered out, or a name that collides with one it already sets), or the "
            "declared scope names something the operation does not support."
        )
    else:
        observed = ", ".join(f"{k} = {v}" for k, v in (item.get("observed") or {}).items())
        body = (
            f"Measured on the running library: operation `{item['operation']}` accepts "
            f"`{item['param']} = {item['value']!r}` and then works with "
            f"{observed or 'another value'}.\n"
            "The declared value is silently replaced during fit. Two things follow, and "
            "the second is the expensive one:\n"
            "  1. tuning this parameter is partly meaningless — the value does not survive;\n"
            "  2. `PipelineNode.descriptive_id` is the operations-cache key and embeds the "
            "parameters, so the fitted node is stored under a key nobody looks up, and every "
            "node downstream of it misses the cache too.\n"
            "Keep the adaptive behaviour — the value may still adapt internally; it must "
            "not overwrite what the caller declared."
        )
    return (
        "\n\n## The proof already exists — do NOT write a test\n"
        + body
        + invariant_symbol_section(repo, item)
        + f"\nThis test FAILS on the current code:\n```python\n{item['test_code']}```\n"
        "Write ONLY the patch that makes it pass. Copy the test verbatim into the TEST "
        f"block and use TEST_NAME: {item['test_name']}."
    )


def invariant_symbol_section(repo: Path, item: dict[str, Any]) -> str:
    """Exact source of the class that holds the parameters.

    Without it the agent gets a 350-line file with four near-identical wrappers
    and cannot land a patch: measured, nine attempts, zero applied hunks.
    """
    symbol = item.get("symbol")
    if not symbol or not repo.is_dir():
        return ""
    try:
        from fedotllm.agents.evolve.templates import symbol_source

        source = symbol_source(repo, item["file"], symbol)
    except Exception as exc:
        logger.warning("could not locate %s in %s (%s)", symbol, item["file"], exc)
        return ""
    if not source:
        return ""
    return (
        f"\n\nTHE DEFECT IS IN `{symbol}` AND NOWHERE ELSE:\n"
        f"```python\n{source}\n```\n"
        "Other classes in this file look almost identical — leave them alone. "
        "Quote OLD from the block above, verbatim.\n"
    )


def proven_defect_files(
    repo: Path, py: str, journal: list[dict[str, Any]] | None = None
) -> set[str]:
    """Repo-relative files that still hold an unfixed proven defect."""
    skip_now = exhausted_files(journal)
    # Reproduced through the public interface: the strongest class there is, and
    # it must be reachable whatever else is switched on.
    reproduced = {i["file"] for i in verified_defects(repo) if i["file"] not in skip_now}
    if reproduced:
        return reproduced
    if os.environ.get("FEDOTLLM_EVOLVE_TEMPLATES", TEMPLATES_DEFAULT) == "0":
        # Invariants are a separate source and must survive templates being off,
        # otherwise the two cannot be measured apart.
        return {i["file"] for i in invariant_defects(repo) if i["file"] not in skip_now}
    try:
        from fedotllm.agents.evolve.templates import proven_defects_cached

        skip = exhausted_files(journal)
        files = set()
        for g in proven_defects_cached(repo, py or resolve_repo_python(repo)):
            for line in g.test_code.splitlines():
                if line.startswith("from fedot") and " import " in line:
                    rel = line.split()[1].replace(".", "/") + ".py"
                    if rel not in skip:
                        files.add(rel)
                    break
        measured = {i["file"] for i in invariant_defects(repo) if i["file"] not in skip}
        # Ranking enforced in code, not in the prompt. With the runtime defects
        # merely listed first in the inventory the scout still went for a `B008`
        # default argument, because the system prompt ranks that class first and
        # a request is not a mechanism. While a defect measured on the running
        # library is available, that is the only allowed target.
        # Ranking, not truncation. Returning only `measured` left the scout with
        # 8 files out of 207 and made every other class unreachable, verified
        # defects included. The rank is still enforced here in code — strongest
        # non-empty class wins — and never left to the prompt.
        for tier in (measured, files):
            if tier and INVARIANTS_FIRST:
                return tier
        return files | measured
    except Exception as exc:
        logger.warning("could not list proven-defect files (%s)", exc)
        return set()


def proven_defect_for(repo: Path, py: str, file_rel: str):
    """The proven defect belonging to a file, if there is one."""
    if os.environ.get("FEDOTLLM_EVOLVE_TEMPLATES", TEMPLATES_DEFAULT) == "0":
        return None
    try:
        from fedotllm.agents.evolve.templates import module_path, proven_defects_cached

        wanted = module_path(file_rel)
        for g in proven_defects_cached(repo, py):
            if f"from {wanted} import" in g.test_code:
                return g
    except Exception as exc:
        logger.warning("could not match a proven defect to %s (%s)", file_rel, exc)
    return None


def run_tuning_gate(repo: Path, py: str, operation: str) -> tuple[bool, str]:
    """The operation must actually be tunable after the patch.

    Added after a patch that cleared every other gate and made things worse. The
    finding said "`iterations` is declared tunable for catboost and every value
    in its declared scope raises CatBoostError"; the patch stopped passing
    `iterations` to CatBoost at all. Error gone, test green, runtime probe
    unchanged — and the caller's value silently discarded, which is the very
    defect class this whole scan exists to find.

    No amount of inspecting parameters catches that: FEDOT's own `params` bag
    still reports the declared value, because only the estimator stopped seeing
    it. Asking for the property the defect is about does catch it — tuning still
    yields no metric, so the patch is refused.
    """
    script = Path(__file__).resolve().parent / "invariants.py"
    res = run_cmd([py, str(script), "--tuning", operation], cwd=repo)
    payload = None
    for line in reversed(res.output.strip().splitlines()):
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        logger.warning("tuning gate produced no result for %s", operation)
        return False, "tuning gate unavailable (no result)"
    if payload.get("ok"):
        return True, f"{operation} tunes, metric {payload.get('obtained_metric')}"
    return False, f"{operation} still cannot be tuned: {payload.get('reason', '')}"


def run_probe_gate(repo: Path, py: str) -> tuple[bool, list[str], str]:
    """Compare runtime behaviour before and after the patch.

    The "before" side comes from the cache filled while building the prompt, so
    the pristine tree is probed once per benchmark, not once per patch.
    """
    try:
        from fedotllm.agents.evolve.probe import (
            compare_probes,
            read_pristine_findings,
            run_probe,
        )

        pristine = read_pristine_findings(repo)
        if pristine is None:
            return False, [], "probe gate unavailable (no pristine baseline)"
        after = run_probe(repo, py)
    except Exception as exc:
        logger.warning("probe gate unavailable (%s)", exc)
        return False, [], f"probe gate unavailable: {exc}"
    return compare_probes(pristine, after)


def probe_findings_section(repo: Path) -> str:
    """Runtime findings for the scout prompt; silent when probing is off/unavailable."""
    if os.environ.get("FEDOTLLM_EVOLVE_PROBE", "1") == "0":
        return ""
    try:
        from fedotllm.agents.evolve.probe import probe_section, run_probe_cached

        findings = run_probe_cached(repo, resolve_repo_python(repo))
    except Exception as exc:  # probing must never break the round
        logger.warning("runtime probe unavailable (%s); continuing without it", exc)
        return ""
    if not findings:
        return ""
    return "## Runtime findings — FEDOT was executed, these defects are REAL\n" + probe_section(
        findings
    )


def build_repo_inventory(
    repo: Path, py: str = "", journal: list[dict[str, Any]] | None = None
) -> str:
    files = iter_repo_python(repo, "fedot")
    # Structural signal: which modules already have a test module.
    # Those are the cheapest to evolve, because a regression harness exists.
    tested: dict[str, str] = {}
    for p in files:
        rel = str(p.relative_to(repo))
        found = find_regression_tests(repo, rel)
        if found:
            tested[rel] = found
    # Prefer outlining modules that look easy to patch/validate.
    preferred = [
        p
        for p in files
        if str(p.relative_to(repo)) in tested
        or any(
            x in str(p.relative_to(repo))
            for x in (
                "api/api_utils/params.py",
                "api/api_utils/presets.py",
                "core/caching/",
                "core/repository/",
                "utilities/",
            )
        )
    ]
    outline_order = preferred + [p for p in files if p not in preferred]
    tree_paths = [str(p.relative_to(repo)) for p in files]
    if len(tree_paths) > MAX_TREE_FILES:
        tree = "\n".join(tree_paths[:MAX_TREE_FILES]) + (
            f"\n... ({len(tree_paths) - MAX_TREE_FILES} more paths omitted)"
        )
    else:
        tree = "\n".join(tree_paths)
    # Lint findings already localise the work precisely, so the bulky outline
    # section can shrink when they are present — the prompt ends up SMALLER than
    # the ungrounded one while carrying a better signal.
    lint = scan_lint_findings(repo)
    outlines: list[str] = []
    budget = MAX_OUTLINE_CHARS_GROUNDED if lint else MAX_OUTLINE_CHARS
    for p in outline_order:
        block = file_outline(p, str(p.relative_to(repo)))
        if budget - len(block) < 0:
            outlines.append(
                f"... outline truncated after {len(outlines)} files "
                f"(raise FEDOTLLM_EVOLVE_MAX_OUTLINE to include more)"
            )
            break
        outlines.append(block)
        budget -= len(block) + 2
    hotspots = scan_hotspots(repo, files)
    covered = "\n".join(f"{src}  ->  {tst}" for src, tst in sorted(tested.items())[:80])
    sections = [
        f"# FEDOT package inventory — {len(files)} python files under fedot/",
        "## Complete file list",
        tree,
        "",
        "## Per-file outlines (classes/functions; preferred modules first)",
        "\n\n".join(outlines),
        "",
        f"## Modules that already have a test module — {len(tested)} "
        "(easiest to evolve: a regression harness already exists)",
        covered or "<none>",
        "",
    ]
    # Runtime findings come first: a defect that only appears when the code runs
    # outranks anything a linter can see. Lint-only rounds produced 100% success
    # and 0 real fixes; this section is what makes the target worth fixing.
    # Proven defects come first: a real defect with a ready proof beats any
    # candidate the agent would have to demonstrate itself.
    if verified_text := verified_defects_section(repo):
        sections += [verified_text, ""]
    if invariant_text := invariant_defects_section(repo):
        sections += [invariant_text, ""]
    if proven_text := proven_defects_section(repo, py or resolve_repo_python(repo), journal):
        sections += [proven_text, ""]
    if probe_text := probe_findings_section(repo):
        sections += [probe_text, ""]
    if lint:
        sections += [
            f"## Static-analysis findings — {len(lint)} shown (concrete, located, verifiable)",
            "Each line is `path:line:col: CODE message`. Fixing one of these is a "
            "strong candidate: the location is certain and the fix is checkable by "
            "re-running the linter. Prefer these over guessing.",
            "\n".join(lint),
            "",
        ]
    sections += [
        f"## Hotspots (TODO/FIXME/bare except/…) — {len(hotspots)} hits",
        "\n".join(hotspots) or "<none>",
    ]
    return "\n".join(sections)


# How far to follow the graph. One hop, deliberately: the definitions a symbol
# names directly, not what those in turn name. Two hops on
# `FedotCatBoostImplementation` pulls in the data layer and the repository, and
# the useful part drowns. BitsAI-Fix ships one layer of symbol dependencies in
# production for the same reason.
DEPENDENCY_HOPS = 1
# A definition shorter than this is worth quoting whole; a longer one is
# summarised, because the question is usually "what does it do to my value",
# not "how does it do it".
DEPENDENCY_FULL_CHARS = int(os.environ.get("FEDOTLLM_EVOLVE_DEP_CHARS", "1200"))
MAX_DEPENDENCIES = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_DEPS", "8"))


def _module_file(repo: Path, module: str) -> Path | None:
    """`fedot.core.operations.x` -> the file, if it is inside this repository."""
    if not module.startswith("fedot"):
        return None
    direct = repo / (module.replace(".", "/") + ".py")
    if direct.is_file():
        return direct
    package = repo / module.replace(".", "/") / "__init__.py"
    return package if package.is_file() else None


def _imports_of(tree: ast.AST, repo: Path) -> dict[str, Path]:
    """Imported name -> the FEDOT file that defines it. Third-party is skipped:
    the agent is being asked about FEDOT's behaviour, and pasting sklearn's
    source in would spend the budget on code it must not change anyway."""
    found: dict[str, Path] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            path = _module_file(repo, node.module)
            if path is None:
                continue
            for alias in node.names:
                found[alias.asname or alias.name] = path
    return found


def _names_used(node: ast.AST) -> set[str]:
    """Every identifier the symbol refers to — calls, attributes, bases."""
    used: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            used.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            used.add(sub.attr)
        elif isinstance(sub, ast.ClassDef):
            for base in sub.bases:
                name = getattr(base, "id", None) or getattr(base, "attr", None)
                if name:
                    used.add(name)
    return used


def _find_definition(path: Path, name: str) -> tuple[str, str] | None:
    """(kind, source) of `name` in `path`, or None."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return None
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != name or not getattr(node, "end_lineno", None):
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        return kind, "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


# A method shorter than this is quoted whole even inside an outlined class.
# Measured why: outlining `BoostingStrategy` by the size of the *class* threw
# away methods of 116, 212 and 291 characters — the very code that shows how
# parameters reach the implementation — while saving almost nothing. Elide what
# is actually big, not what happens to sit inside something big.
INLINE_METHOD_CHARS = int(os.environ.get("FEDOTLLM_EVOLVE_INLINE_CHARS", "500"))


def _outline_source(src: str, inline_under: int = INLINE_METHOD_CHARS) -> str:
    """A definition reduced to its shape, keeping short bodies verbatim."""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return src.splitlines()[0] + "\n    ..."
    node = tree.body[0]
    dedented = textwrap.dedent(src).splitlines()
    lines: list[str] = []
    header = f"class {node.name}:" if isinstance(node, ast.ClassDef) else _signature(node) + ":"
    lines.append(header)
    if (doc := _docstring_line(node)):
        lines.append(f'    """{doc}"""')
    for sub in getattr(node, "body", []):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = "\n".join(dedented[sub.lineno - 1 : sub.end_lineno])
            if len(body) <= inline_under:
                lines.append(body)
                continue
            lines.append(f"    {_signature(sub)}:  # body elided ({len(body)} chars)")
            if (doc := _docstring_line(sub)):
                lines.append(f'        """{doc}"""')
        elif isinstance(sub, ast.Assign):
            try:
                lines.append("    " + ast.unparse(sub))
            except Exception:
                pass
    return "\n".join(lines)


def dependency_context(repo: Path, rel: str, focus: str,
                       limit: int = MAX_DEPENDENCIES) -> str:
    """FEDOT's own definitions that the focus symbol depends on.

    The file view answers "what does this class look like"; this answers "what
    happens to my value before it gets here". For the CatBoost defect the
    parameters arrive through `OperationParameters`, which lives in another file
    entirely — without it the agent sees a dictionary appearing from nowhere and
    can only patch the symptom in front of it.

    Only definitions inside `fedot/` are followed, one hop, and long ones are
    summarised rather than pasted.
    """
    path = repo / rel
    if not path.is_file() or not focus:
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return ""

    target = next((n for n in ast.walk(tree)
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == focus), None)
    if target is None:
        return ""

    imports = _imports_of(tree, repo)
    local = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    wanted = [n for n in sorted(_names_used(target))
              if n in imports and n not in local]

    blocks: list[str] = []
    for name in wanted[:limit]:
        found = _find_definition(imports[name], name)
        if found is None:
            continue
        kind, src = found
        where = str(imports[name].relative_to(repo))
        if len(src) <= DEPENDENCY_FULL_CHARS:
            blocks.append(f"### `{name}` — {kind} in `{where}`\n```python\n{src}\n```")
            continue
        # Outline with the methods kept. Collapsing a long class to its first
        # line is nearly useless here: `OperationParameters` is 1 956 characters
        # and the method that matters is `update`, which mutates the bag every
        # defect in this family goes through. A bare `class OperationParameters:`
        # hides exactly the thing worth knowing.
        outline = _outline_source(src)
        blocks.append(
            f"### `{name}` — {kind} in `{where}` ({len(src)} chars, bodies elided)\n"
            f"```python\n{outline}\n```")
    if not blocks:
        return ""
    return (
        f"\n\n## What `{focus}` depends on (FEDOT's own code, one hop)\n"
        "These are the definitions the symbol above refers to. Third-party libraries "
        "are deliberately absent: the defect has to be fixed in FEDOT.\n\n"
        + "\n\n".join(blocks)
    )


MAX_USAGE_SITES = int(os.environ.get("FEDOTLLM_EVOLVE_MAX_USES", "6"))


def _enclosing_definition(tree: ast.AST, line: int):
    """Innermost function or class containing `line`."""
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not getattr(node, "end_lineno", None):
            continue
        if node.lineno <= line <= node.end_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def usage_context(repo: Path, rel: str, focus: str,
                  limit: int = MAX_USAGE_SITES) -> str:
    """Where FEDOT itself uses the focus symbol, with the surrounding code.

    The dependency section says what happens to a value before it reaches the
    symbol; this says what the rest of the library expects from it afterwards.
    Both matter for a fix, and for a whole class of defects the second is where
    the fix actually belongs — the call site decides what gets passed in.

    Names alone are not enough, and that is what was shipped before: a list
    reading "`FedotCatBoostClassificationImplementation` is used by 1 other
    module" tells the agent nothing it can act on. What it needs is the code
    that does the using.
    """
    if not focus:
        return ""
    sites: list[tuple[str, str]] = []
    for path in sorted(repo.glob("fedot/**/*.py")):
        if str(path.relative_to(repo)) == rel:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if focus not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            if name != focus or not hasattr(node, "lineno"):
                continue
            holder = _enclosing_definition(tree, node.lineno)
            if holder is None:
                continue
            src = "\n".join(lines[holder.lineno - 1 : holder.end_lineno])
            where = f"{path.relative_to(repo)}:{holder.lineno} — {holder.name}"
            if len(src) > DEPENDENCY_FULL_CHARS:
                src = _outline_source(src)
            if not any(w == where for w, _ in sites):
                sites.append((where, src))
            break  # one site per file is enough to show the shape of the usage
        if len(sites) >= limit:
            break

    if not sites:
        return ""
    blocks = [f"### `{where}`\n```python\n{src}\n```" for where, src in sites]
    return (
        f"\n\n## Where FEDOT uses `{focus}` ({len(sites)} site(s))\n"
        "A change here has to stay valid for these. Their test modules are part of "
        "acceptance, and for some defects the right fix belongs at the call site "
        "rather than inside the symbol.\n\n"
        + "\n\n".join(blocks)
    )


def _docstring_line(node: ast.AST) -> str:
    """First sentence of a docstring, or empty. Used where a body is elided."""
    doc = ast.get_docstring(node) or ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    return (first[:180] + "…") if len(first) > 180 else first


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        return f"def {node.name}({ast.unparse(node.args)})"
    except Exception:
        return f"def {node.name}(...)"


def source_view(repo: Path, rel: str, focus: str = "") -> str:
    """The file as the agent should see it: whole symbols, never a cut-off tail.

    Truncating at a character budget is the wrong compression. It is silent —
    the model has no way to know the file continues — and it cuts mid-function,
    which is exactly where the code stops meaning anything. Measured on
    `boostings_implementations.py`: 16 966 characters, so a 12 000 budget showed
    71% of the file and hid three of the six implementation classes.

    So the file is pruned by symbol instead, the way RepoRepair prunes for
    repository-level repair (SWE-bench Multimodal 37.1% vs 25.34% for Agentless
    Lite, file localisation 59.8% vs 30.4%): the focus symbol and anything it
    inherits from are given in full, everything else keeps its signature and the
    first line of its docstring. Nothing disappears without a marker.
    """
    path = repo / rel
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text[:MAX_CHARS_PER_FILE]
    lines = text.splitlines()

    def body(node: ast.AST) -> str:
        return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    # Full detail for the focus symbol and for whatever it inherits from inside
    # this file: a fix for a subclass usually belongs to the parent, and a model
    # that cannot see the parent guesses at it.
    keep = {focus} if focus else set()
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    frontier = list(keep)
    while frontier:
        name = frontier.pop()
        node = classes.get(name)
        if node is None:
            continue
        for base in node.bases:
            base_name = getattr(base, "id", None) or getattr(base, "attr", None)
            if base_name in classes and base_name not in keep:
                keep.add(base_name)
                frontier.append(base_name)

    out: list[str] = []
    if (doc := ast.get_docstring(tree)):
        out.append(f'"""{doc.strip().splitlines()[0]}"""')
    out += [body(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]

    elided = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ClassDef):
            if not keep or node.name in keep:
                out.append("\n" + body(node))
                continue
            # Outline: the class is still visible, its bodies are not.
            head = [f"\nclass {node.name}({', '.join(ast.unparse(b) for b in node.bases)}):"]
            if (doc := _docstring_line(node)):
                head.append(f'    """{doc}"""')
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    head.append(f"    {_signature(sub)}:  # body elided")
                    if (doc := _docstring_line(sub)):
                        head.append(f'        """{doc}"""')
                    elided += 1
            out.append("\n".join(head) if len(head) > 1 else head[0] + "\n    ...")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not keep or node.name in keep:
                out.append("\n" + body(node))
            else:
                line = f"\n{_signature(node)}:  # body elided"
                if (doc := _docstring_line(node)):
                    line += f'\n    """{doc}"""'
                out.append(line)
                elided += 1
        else:
            out.append(body(node))

    rendered = "\n".join(out)
    note = (
        f"\n\n# The whole file is represented above: {len(tree.body)} top-level "
        f"definitions, {elided} bodies elided and marked. Nothing is cut off — ask "
        "for a symbol by name if you need a body that is not shown."
    ) if elided else ""
    return rendered + note


def read_source_file(repo: Path, rel: str) -> str:
    path = repo / rel
    if not path.is_file():
        return f"<missing file: {rel}>"
    return path.read_text(encoding="utf-8", errors="ignore")[:MAX_CHARS_PER_FILE]


def parse_pick(raw: str) -> tuple[str, str]:
    s = (raw or "").strip()
    m_pick = re.search(r"^PICK:\s*(.+)$", s, flags=re.MULTILINE | re.IGNORECASE)
    m_why = re.search(r"^WHY:\s*(.+)$", s, flags=re.MULTILINE | re.IGNORECASE)
    pick = (m_pick.group(1).strip().strip("`") if m_pick else "").lstrip("./")
    why = m_why.group(1).strip() if m_why else ""
    if not pick.endswith(".py") or not pick.startswith("fedot/") or ".." in pick.split("/"):
        raise ValueError(f"invalid PICK path: {pick!r}")
    return pick, why


def parse_delimited(raw: str) -> Proposal:
    s = (raw or "").strip()
    if "```" in s:
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)

    def field(name: str) -> str:
        m = re.search(
            rf"^[`*]*{name}[`*]*:\s*(.+)$", s, flags=re.MULTILINE | re.IGNORECASE
        )
        return m.group(1).strip().strip("`").strip() if m else ""

    def block(tag_open: str, tag_close: str) -> str:
        m = re.search(
            re.escape(tag_open) + r"[ \t]*\r?\n(.*?)(?:\r?\n)?" + re.escape(tag_close),
            s,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return m.group(1) if m else ""

    def tail_after(tag: str) -> str:
        """Everything after `tag` — the fallback when the closing marker is absent."""
        m = re.search(re.escape(tag) + r"[ \t]*\r?\n(.*)", s, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).rstrip() if m else ""

    # A fix may need several edits of one file (e.g. a constructed default: the
    # signature becomes None *and* the body must build the object). One hunk per
    # site — merging them makes the model glue unrelated lines together.
    pairs = re.findall(
        r"<<<OLD>>>[ \t]*\r?\n(.*?)(?:\r?\n)?<<<NEW>>>[ \t]*\r?\n(.*?)(?:\r?\n)?"
        # END must terminate a hunk too: without it the marker leaked into the
        # replacement text, the patch stopped parsing as Python and the whole
        # attempt budget was spent re-quoting a snippet that was never at fault.
        r"(?=<<<(?:OLD|TEST|END)>>>)",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return Proposal(
        file_path=field("FILE"),
        test_file=field("TEST_FILE"),
        test_name=field("TEST_NAME"),
        problem=field("PROBLEM"),
        rationale=field("RATIONALE"),
        anchor=field("FUNCTION"),
        old_code=pairs[0][0] if pairs else "",
        new_code=pairs[0][1] if pairs else "",
        hunks=[(o, n) for o, n in pairs],
        # Multi-hunk answers are longer, so the trailing <<<END>>> is the first
        # thing a token limit eats. Losing the marker must not lose the test.
        test_code=block("<<<TEST>>>", "<<<END>>>") or tail_after("<<<TEST>>>"),
    )


def parse_proposal(raw: str) -> Proposal:
    p = parse_delimited(raw)
    required = ("file_path", "test_file", "test_name", "old_code", "new_code", "test_code")
    missing = [k for k in required if not str(getattr(p, k) or "").strip()]
    if missing:
        raise ValueError(f"incomplete LLM proposal, missing: {missing}")
    if (
        not p.test_file.endswith(".py")
        or p.test_file.startswith("/")
        or ".." in p.test_file.split("/")
    ):
        raise ValueError(f"suspicious test_file path: {p.test_file!r}")
    if (
        not p.file_path.endswith(".py")
        or p.file_path.startswith("/")
        or ".." in p.file_path.split("/")
    ):
        raise ValueError(f"suspicious file_path: {p.file_path!r}")
    return p


def proposal_as_delimited(p: Proposal) -> str:
    return (
        f"FILE: {p.file_path}\n"
        f"TEST_FILE: {p.test_file}\n"
        f"TEST_NAME: {p.test_name}\n"
        f"PROBLEM: {p.problem}\n"
        f"RATIONALE: {p.rationale}\n"
        + "".join(
            f"<<<OLD>>>\n{o}\n<<<NEW>>>\n{n}\n"
            for o, n in (p.hunks or [(p.old_code, p.new_code)])
        )
        + f"<<<TEST>>>\n{p.test_code}\n<<<END>>>"
    )


def apply_test(repo: Path, p: Proposal) -> None:
    tf = repo / p.test_file
    if tf.exists() and tf.is_dir():
        raise ValueError(f"test_file resolves to a directory: {p.test_file!r}")
    tf.parent.mkdir(parents=True, exist_ok=True)
    existing = tf.read_text(encoding="utf-8") if tf.exists() else ""
    if p.test_name not in existing:
        sep = "" if existing.endswith("\n") or not existing else "\n"
        tf.write_text(existing + sep + "\n" + p.test_code.rstrip() + "\n", encoding="utf-8")


def _def_name(code: str) -> str | None:
    """Return the name of the first def/class declared in a code block."""
    m = re.search(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", code, re.MULTILINE)
    return m.group(1) if m else None


def apply_source_ast(repo: Path, p: Proposal) -> bool:
    """Fallback: replace a whole function/class node located by name via AST.

    Verbatim `old_code` matching is brittle — the LLM often re-indents or elides
    lines, which burns a retry. When both OLD and NEW declare the same symbol we
    can locate that symbol in the real file and swap its source lines instead.
    """
    sf = repo / p.file_path
    if not sf.is_file():
        return False
    name = _def_name(p.new_code) or _def_name(p.old_code)
    if not name:
        return False
    text = sf.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False

    targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
        and getattr(node, "end_lineno", None)
    ]
    if len(targets) != 1:
        logger.warning("AST fallback: %s matches for symbol %r; skipping", len(targets), name)
        return False

    node = targets[0]
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    # Keep the original indentation of the symbol being replaced.
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    body = textwrap.indent(textwrap.dedent(p.new_code.strip("\n")), indent).rstrip("\n") + "\n"
    patched = "".join(lines[:start]) + body + "".join(lines[node.end_lineno :])
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        logger.warning("AST fallback produced invalid syntax (%s); skipping", exc)
        return False
    sf.write_text(patched, encoding="utf-8")
    logger.info("AST fallback: replaced symbol %r in %s", name, p.file_path)
    return True


def infer_anchor(repo: Path, p: Proposal) -> str:
    """Work out the enclosing symbol when the model did not fill FUNCTION.

    Measured: FUNCTION came back empty in every run, on both models — asking for
    it does not work. The model does, however, name the symbol in PROBLEM
    ("the `task` parameter in `from_csv`"), so take it from there instead.
    """
    if p.anchor:
        return p.anchor
    path = repo / p.file_path
    if not path.is_file():
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return ""
    funcs = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    # Functions before classes: the enclosing class usually contains every
    # occurrence of the snippet and so disambiguates nothing.
    quoted = re.findall(r"`([A-Za-z_]\w*)`", f"{p.problem} {p.rationale}")
    for name in quoted:
        if name in funcs:
            return name
    # The test names the target even when PROBLEM does not — `test_from_numpy_…`
    # and the call inside it both point at `from_numpy`.
    for source in (p.test_name, p.test_code):
        for name in sorted(funcs, key=len, reverse=True):
            if name in (source or ""):
                return name
    for name in quoted:
        if name in classes:
            return name
    for name in re.findall(r"\b([A-Za-z_]\w*)\b", p.problem or ""):
        if name in funcs or name in classes:
            return name
    return ""


def match_ignoring_indent(text: str, old: str) -> tuple[int, int, str] | None:
    """Locate `old` allowing a different amount of leading whitespace.

    Exact matching is brittle for the one thing models get wrong most often. In a
    real run the quoted block existed nowhere in the file: every line was right,
    every line was indented by one space too many. Returns the matched line span
    and the indentation actually used in the file, or None if not unique.
    """
    wanted = [ln.strip() for ln in old.strip("\n").splitlines() if ln.strip()]
    if not wanted:
        return None
    lines = text.splitlines()
    hits: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        if line.strip() != wanted[0]:
            continue
        # Walk forward matching only the non-blank lines: models routinely drop
        # the blank lines inside a docstring when quoting it back.
        k, j = 1, i + 1
        while k < len(wanted) and j < len(lines):
            if not lines[j].strip():
                j += 1
                continue
            if lines[j].strip() != wanted[k]:
                break
            k += 1
            j += 1
        if k != len(wanted):
            continue
        indent = line[: len(line) - len(line.lstrip())]
        hits.append((i, j, indent))
    return hits[0] if len(hits) == 1 else None


def reindent(block: str, old_indent: str, new_indent: str) -> str:
    """Shift `block` from the indentation the model used to the file's own."""
    if old_indent == new_indent:
        return block
    out = []
    for ln in block.splitlines():
        if ln.startswith(old_indent):
            out.append(new_indent + ln[len(old_indent) :])
        else:
            out.append(ln)
    return "\n".join(out)


def replace_within_symbol(text: str, anchor: str, old: str, new: str) -> str | None:
    """Replace `old` inside the function/class named `anchor`; None if not resolvable.

    Used when a snippet repeats across the file: narrowing the search to the one
    symbol the proposal is about usually makes it unique again.
    """
    if not anchor:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    nodes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.name == anchor
        and getattr(n, "end_lineno", None)
    ]
    if len(nodes) != 1:
        return None

    lines = text.splitlines(keepends=True)
    node = nodes[0]
    head, body, tail = (
        "".join(lines[: node.lineno - 1]),
        "".join(lines[node.lineno - 1 : node.end_lineno]),
        "".join(lines[node.end_lineno :]),
    )
    if body.count(old) != 1:
        return None
    return head + body.replace(old, new, 1) + tail


def apply_source(repo: Path, p: Proposal) -> bool:
    sf = repo / p.file_path
    if not sf.is_file():
        return False
    text = sf.read_text(encoding="utf-8")
    hunks = p.hunks or ([(p.old_code, p.new_code)] if p.old_code else [])
    if not hunks:
        return False

    # All or nothing: half of a two-site fix is worse than none — that is how a
    # default became None while the body that should build it never appeared.
    # An elided replacement silently deletes whatever it stands for. The model
    # writes a bare `...` line to mean "unchanged code here"; applying that would
    # wipe the docstring it replaced.
    for _, new in hunks:
        if any(ln.strip() == "..." for ln in new.splitlines()):
            logger.warning("replacement elides code with '...'; refusing to apply")
            return False

    patched = text
    for old, new in hunks:
        # Uniqueness is the real safety property, not length: `    return task`
        # is short but unambiguous, and multi-site fixes quote such lines all the
        # time. Only reject fragments too small to mean anything.
        trivial = len(old.strip()) < 4
        hits = patched.count(old) if old else 0
        if not old or trivial:
            if len(hunks) != 1:
                logger.warning("multi-hunk patch contains an empty or trivial hunk; refusing")
                return False
            logger.warning("hunk %r is empty or trivial; trying AST fallback", old[:40])
            return apply_source_ast(repo, p)
        if hits == 0:
            # Same block, wrong indentation — accept it and re-indent to the file.
            located = match_ignoring_indent(patched, old)
            if located is not None:
                start, end, file_indent = located
                model_indent = old.strip("\n").splitlines()[0]
                model_indent = model_indent[: len(model_indent) - len(model_indent.lstrip())]
                body = reindent(new.strip("\n"), model_indent, file_indent)
                lines = patched.splitlines()
                patched = "\n".join(lines[:start] + body.splitlines() + lines[end:])
                if patched and not patched.endswith("\n"):
                    patched += "\n"
                logger.info("hunk matched after re-indenting by the file's own margin")
                continue
        if hits != 1:
            # A parameter line can repeat verbatim across overloads — in FEDOT's
            # data.py the defective default occurs five times. Asking the model to
            # quote enough context is unreliable, so resolve it ourselves when it
            # named the enclosing symbol.
            anchor = infer_anchor(repo, p)
            scoped = replace_within_symbol(patched, anchor, old, new)
            if scoped is None:
                if len(hunks) != 1:
                    logger.warning(
                        "multi-hunk patch cannot be applied atomically; refusing AST fallback"
                    )
                    return False
                logger.warning(
                    "hunk %r matches %s times and anchor %r did not disambiguate; "
                    "trying AST fallback",
                    old[:40],
                    hits,
                    anchor,
                )
                return apply_source_ast(repo, p)
            logger.info("hunk disambiguated via inferred anchor %r", anchor)
            patched = scoped
            continue
        patched = patched.replace(old, new, 1)

    try:
        ast.parse(patched)
    except SyntaxError as exc:
        logger.warning("multi-hunk patch produced invalid syntax (%s); skipping", exc)
        return False
    sf.write_text(patched, encoding="utf-8")
    return True


def ask_proposal(inference: AIInference, messages: list[dict[str, Any]]) -> Proposal:
    raw = inference.query(messages) or ""
    try:
        return parse_proposal(raw)
    except ValueError as e:
        messages = list(messages)
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Parse error: {e}. Reply again in EXACT delimiter format "
                    "(FILE/TEST_FILE/TEST_NAME/PROBLEM/RATIONALE/<<<OLD>>>/<<<NEW>>>/"
                    "<<<TEST>>>/<<<END>>>), no JSON, no markdown fences."
                ),
            }
        )
        return parse_proposal(inference.query(messages) or "")


# Severity of a change, so "how many fixes" can be read as "how much value".
# Counting patches alone rewards polishing error messages forever.
# Severity comes from the rule code, which is exact — guessing it from the wording
# of a proposal inflated the class (a run was recorded as class 1 because the scout
# had mentioned a default argument, while the accepted patch only reworded an error).
# Counts in parentheses are what FEDOT actually has today.
SEVERITY_BY_RULE = {
    # 1 — wrong even on valid input; state leaks between calls or instances
    "B008": 1,    # function call in a default argument (6)
    "B006": 1,    # mutable default argument
    "B020": 1,
    # 2 — silent data loss, hangs, security
    "B905": 2,    # zip() without strict: truncates without warning (17)
    "S113": 2,    # request without timeout: hangs forever (9)
    "S608": 2,    # SQL built by string concatenation (11)
    "B007": 2, "F841": 2, "B904": 2,
    # 3 — correctness smells with no runtime effect here
    # RUF012 only proves that a class attribute is mutable. It does not prove
    # that code mutates it or that sharing is unintended.
    "RUF012": 3,
    "RUF013": 3,  # implicit Optional (41)
    "F821": 3, "F401": 3, "F403": 3,
}
SEVERITY_NAMES = {1: "behavioural defect", 2: "silent loss / hang / security",
                  3: "correctness smell", 4: "unclear error or cosmetic"}
# Only these count as genuine value: everything else is polish. Reported separately
# from success rate, because an agent that sets its own task can max out success
# rate by always choosing the easiest class (measured: 30/30 success, 0 real fixes).
VALUABLE_SEVERITIES = (1, 2)


def classify_severity(text: str) -> tuple[int, str]:
    """Best-effort severity class for a proposal, from rule codes then wording."""
    codes = re.findall(r"\b([A-Z]{1,3}[0-9]{3})\b", text or "")
    best = min((SEVERITY_BY_RULE[c] for c in codes if c in SEVERITY_BY_RULE), default=None)
    if best:
        return best, SEVERITY_NAMES[best]
    low = (text or "").lower()
    if any(w in low for w in ("default argument", "shared", "mutable")):
        return 1, SEVERITY_NAMES[1]
    if any(w in low for w in ("zip(", "truncat", "never used", "unused")):
        return 2, SEVERITY_NAMES[2]
    return 4, SEVERITY_NAMES[4]


def classify_accepted_severity(
    ready: Any,
    invariant: dict[str, Any] | None,
    probe_resolved: list[str],
    proposal_text: str,
    verified: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """Prefer deterministic evidence over the model's description of its patch."""
    if verified is not None:
        # The verifier currently proves public-API crashes. They are ordinary
        # user-visible defects, not cosmetic work and not silent corruption.
        # Future verifier rows may carry one of the mechanically established
        # critical kinds below; do not infer criticality from model prose.
        critical_kinds = {"declared_not_used", "nondeterministic", "timeout"}
        if verified.get("kind") in critical_kinds:
            return 1, "critical behavioural defect"
        return 2, "ordinary public-API defect"
    if ready is not None:
        severity = SEVERITY_BY_RULE.get(ready.rule, 3)
        return severity, SEVERITY_NAMES[severity]
    if invariant is not None:
        return 1, SEVERITY_NAMES[1]
    if probe_resolved:
        return 2, SEVERITY_NAMES[2]
    return classify_severity(proposal_text)


def journal_path(workspace: Path | None) -> Path:
    """Where the cross-run evolution journal lives (shared by all runs)."""
    env = os.environ.get("FEDOTLLM_EVOLVE_JOURNAL")
    if env:
        return Path(env)
    base = workspace.parent if workspace else Path(".")
    return base / "evolution_journal.jsonl"


def read_journal(path: Path, limit: int = JOURNAL_LIMIT) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-limit:]


def append_journal(path: Path, entry: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("EvolveAgent: cannot write journal (%s)", exc)


def journal_section(entries: list[dict[str, Any]]) -> str:
    """Render past evolutions so the scout does not repeat itself.

    Without this the scout is a "memoryless explorer": with a low temperature it
    re-derives the same argmax every run (we measured 30/30 identical picks).
    Showing what has already been done is the cheap fix used by ACE / SWE-Exp /
    Memory of past attempts, successes and failures alike.
    """
    if not entries:
        return ""
    done = [e for e in entries if e.get("success")]
    tried = [e for e in entries if not e.get("success")]
    lines = [
        "## Already evolved in previous runs — DO NOT repeat these",
        "Picking any of these files again is a wasted run: the improvement is "
        "already made. Choose a DIFFERENT file.",
    ]
    lines += [f"- `{e.get('file','?')}` — {str(e.get('problem',''))[:100]}" for e in done[-JOURNAL_LIMIT:]]
    if tried:
        lines += [
            "",
            "### Previously attempted without success (harder — pick only if you see a clearly better fix)",
        ]
        lines += [f"- `{e.get('file','?')}` — {str(e.get('reason',''))[:80]}" for e in tried[-10:]]
    return "\n".join(lines)


def scout_pick(
    inference: AIInference,
    repo: Path,
    journal: list[dict[str, Any]] | None = None,
    py: str = "",
) -> tuple[str, str]:
    inventory = build_repo_inventory(repo, py, journal)
    memory = journal_section(journal or [])
    if memory:
        inventory = f"{inventory}\n\n{memory}"
        # Novelty rejection: re-ask once if the pick is a repeat.
        banned = {e.get("file") for e in (journal or []) if e.get("success")}
    else:
        banned = set()
    n_files = len(iter_repo_python(repo, "fedot"))
    logger.info("EvolveAgent scout: %s files, inventory %s chars", n_files, len(inventory))
    raw = (
        inference.query(
            [
                {"role": "system", "content": SCOUT_SYS},
                {
                    "role": "user",
                    "content": (
                        f"{inventory}\n\n"
                        "Pick ONE file from the complete list above for a small verifiable fix."
                    ),
                },
            ]
        )
        or ""
    )
    try:
        pick, why = parse_pick(raw)
    except ValueError:
        raw = (
            inference.query(
                [
                    {"role": "system", "content": SCOUT_SYS},
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Invalid PICK. Reply again as:\nPICK: fedot/...\nWHY: ...\n"
                            "The path MUST appear in the inventory file list."
                        ),
                    },
                ]
            )
            or ""
        )
        pick, why = parse_pick(raw)
    # Stay inside the proven-defect list while it still has entries. Asking nicely
    # does not hold: with "prefer proven defects" in the prompt the scout still
    # wandered off to an unproven file and produced a cosmetic change there. A
    # proven defect has a confirmed failing test, an unproven guess has nothing.
    proven_files = proven_defect_files(repo, py, journal)
    if proven_files and pick not in proven_files:
        logger.info("EvolveAgent scout: %s is not a proven defect — re-asking", pick)
        listed = "\n".join(f"  - {f}" for f in sorted(proven_files)[:15])
        raw = (
            inference.query(
                [
                    {"role": "system", "content": SCOUT_SYS},
                    {"role": "user", "content": inventory},
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"`{pick}` has no confirmed defect, so any fix there would "
                            "have to be demonstrated from scratch. Pick a file from this "
                            f"list instead — each already has a failing test:\n{listed}\n"
                            "Reply as:\nPICK: fedot/...\nWHY: ..."
                        ),
                    },
                ]
            )
            or ""
        )
        try:
            pick2, why2 = parse_pick(raw)
            if pick2 in proven_files:
                pick, why = pick2, why2
        except ValueError:
            logger.warning("scout re-ask returned an unusable pick; keeping %s", pick)

    if pick in banned:
        logger.info("EvolveAgent scout: %s already evolved — asking for a different target", pick)
        raw = (
            inference.query(
                [
                    {"role": "system", "content": SCOUT_SYS},
                    {"role": "user", "content": inventory},
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"`{pick}` was already evolved in an earlier run, so that "
                            "improvement exists. Pick a DIFFERENT file that is not in "
                            "the 'Already evolved' list.\n"
                            "PICK: fedot/...\nWHY: ..."
                        ),
                    },
                ]
            )
            or ""
        )
        try:
            new_pick, new_why = parse_pick(raw)
            if new_pick not in banned and (repo / new_pick).is_file():
                pick, why = new_pick, new_why
        except ValueError:
            pass  # keep the original pick rather than fail the run
    if not (repo / pick).is_file():
        raise FileNotFoundError(f"scout picked missing file: {pick}")
    logger.info("EvolveAgent scout pick: %s — %s", pick, why)
    return pick, why


def build_audit(
    repo: Path,
    result: EvolveResult,
    model: str,
) -> str:
    commit = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo).output
    diff = ""
    if result.changed:
        diff = run_cmd(["git", "diff", "--"] + result.changed, cwd=repo).output
        for rel in result.changed:
            path = repo / rel
            tracked = run_cmd(["git", "ls-files", "--error-unmatch", "--", rel], cwd=repo)
            if tracked.exit_code == 0 or not path.is_file():
                continue
            rendered = difflib.unified_diff(
                [],
                path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=rel,
            )
            diff += ("\n" if diff else "") + "".join(rendered)
    p = result.proposal
    lines = [
        "# FEDOT Repo Evolution — FEDOT.LLM EvolveAgent Audit",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Model: `{model}`  ·  Repo: `{repo}`  ·  Base revision: `{commit}`",
        f"Scout pick: `{result.pick}` — {result.why}",
        f"Success: `{result.success}`  ·  LLM attempts used: `{result.attempts}`  ·  "
        f"Severity: `{result.severity}` ({result.severity_name})",
        (f"**Abstained** — {result.abstain_reason}" if result.abstained
         else f"Evidence: `{result.evidence or 'none'}`"),
        "",
        "## Agent loop (full-repo scout via EvolveAgent)",
        "1. Inventory ALL `fedot/**/*.py` (paths + outlines + hotspots); scout PICK one file.",
        "2. Deep-read that file; LLM proposes patch + test via AIInference "
        f"(best-of-{NUM_CANDIDATES} candidates, {MAX_FIX_TRIES} self-fix tries each).",
        "3. Gates: reproduce must FAIL on pristine → apply → py_compile + targeted pytest "
        "→ regression tests of the patched module and its callers → CPU `Fedot.fit` "
        "→ runtime probe → tuning check for unusable-parameter defects. Optional gates "
        "must either pass or be explicitly disabled; unavailable gates fail closed.",
        "4. Write this audit.",
        "",
        "## Found problem",
        f"- File: `{p.file_path}`",
        f"- Problem: {p.problem}",
        f"- Rationale: {p.rationale}",
        "",
        "## Changed files",
    ]
    lines += [f"- `{c}`" for c in result.changed] or ["- (none)"]
    if result.reproduce is not None:
        status = (
            "reproduced (failed as expected — patch is proven)"
            if result.reproduce.exit_code != 0
            else "test passed on pristine — REJECTED by the reproduce gate"
        )
        lines += [
            "",
            "## Reproduction (test on pristine source)",
            f"Exit `{result.reproduce.exit_code}` — {status}",
            "```text",
            result.reproduce.output[-1500:] or "<no output>",
            "```",
        ]
    lines += ["", "## Validation (after patch)"]
    for r in result.validations:
        st = "passed" if r.exit_code == 0 else "FAILED"
        lines += [
            f"### `{r.command}`",
            f"Exit `{r.exit_code}` ({st})",
            "```text",
            r.output[-2500:] or "<no output>",
            "```",
            "",
        ]
    lines += ["## Diff", "```diff", diff or "<no diff>", "```", ""]
    return "\n".join(lines)


def find_regression_tests(repo: Path, source_rel: str, exclude: str = "") -> str | None:
    """Locate an existing test module covering the patched source file.

    Used as the regression gate: the patch must not break tests
    that already passed. Returns a repo-relative path, or None if none found.
    """
    src_parts = {part for part in Path(source_rel).parts[:-1]}
    stem = Path(source_rel).stem

    def usable(p: Path) -> bool:
        return p.is_file() and str(p.relative_to(repo)) != exclude

    # 1) exact name match: fedot/.../params.py -> test/**/test_params.py
    exact = [p for p in repo.glob(f"test/**/test_{stem}.py") if usable(p)]
    if exact:
        return str(exact[0].relative_to(repo))

    # 2) name contains the stem AND lives in a directory echoing the source path
    #    (test/unit/api/test_api_params.py for fedot/api/api_utils/params.py).
    #    Without this check `test_mutation_params.py` would be picked for
    #    `params.py` and the regression gate would fail for unrelated reasons.
    fuzzy = [
        p
        for p in repo.glob(f"test/**/test_*{stem}*.py")
        if usable(p) and src_parts & set(p.relative_to(repo).parts[:-1])
    ]
    if fuzzy:
        fuzzy.sort(key=lambda p: len(p.name))
        return str(fuzzy[0].relative_to(repo))
    return None


def find_symbol_callers(repo: Path, symbol: str, exclude_rel: str = "") -> list[str]:
    """Files under `fedot/` that mention `symbol`, excluding the patched one.

    The agent only ever reads one file, so a change to a shared helper can break
    callers it never saw. `data_strategy_selector`, for instance, is used from two
    modules and owns no test module at all.
    """
    if not symbol or len(symbol) < 3:
        return []
    out = run_cmd(["git", "grep", "-l", "-w", symbol, "--", "fedot/"], cwd=repo)
    if out.exit_code != 0:
        return []
    return [ln.strip() for ln in out.output.splitlines() if ln.strip() and ln.strip() != exclude_rel]


def module_usage_section(repo: Path, rel: str, limit: int = 12) -> str:
    """How widely each symbol of the picked module is used elsewhere.

    Shown before the model chooses what to change: editing a helper with five
    callers is a different risk from editing one nobody imports, and the model
    cannot tell them apart from a single file.
    """
    path = repo / rel
    if not path.is_file():
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return ""
    names = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not n.name.startswith("_")
    ]
    rows = []
    for name in names[:limit]:
        callers = find_symbol_callers(repo, name, rel)
        if callers:
            shown = ", ".join(callers[:3]) + (" …" if len(callers) > 3 else "")
            rows.append(f"  - `{name}` is used by {len(callers)} other module(s): {shown}")
    if not rows:
        return ""
    return (
        "\n\n## Cross-module usage (you see only ONE file — these depend on it)\n"
        + "\n".join(rows)
        + "\nA change to any of these must keep the existing call signature and return "
        "type valid for those callers; their test modules are run as part of acceptance."
    )


def _direct_symbol_test_nodes(repo: Path, symbol: str) -> list[str]:
    """Existing pytest nodes whose function body refers to ``symbol``."""
    found = run_cmd(["git", "grep", "-l", "-w", symbol, "--", "test/"], cwd=repo)
    if found.exit_code != 0:
        return []
    nodes: list[str] = []
    for line in found.output.splitlines():
        rel = line.strip()
        path = repo / rel
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        for item in tree.body:
            candidates = (
                [(item, "")]
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                else [
                    (child, f"{item.name}::")
                    for child in item.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if isinstance(item, ast.ClassDef)
                else []
            )
            for fn, owner in candidates:
                body = ast.get_source_segment(source, fn) or ""
                if fn.name.startswith("test_") and symbol in body:
                    nodes.append(f"{rel}::{owner}{fn.name}")
    return nodes


def find_caller_regression_tests(repo: Path, symbol: str, exclude_rel: str) -> list[str]:
    """Focused existing tests for source and test consumers of ``symbol``."""
    tests: list[str] = []
    symbols = [symbol]
    owner = symbol.split(".", 1)[0]
    if owner and owner != symbol:
        symbols.append(owner)

    # A test may exercise the changed symbol directly without having a
    # source-module counterpart that ``find_regression_tests`` can infer.
    # Collect exact pytest nodes, not whole integration modules: the latter may
    # contain unrelated optional-dependency failures in a minimal environment.
    for name in symbols:
        for node in _direct_symbol_test_nodes(repo, name):
            if node != exclude_rel and node not in tests:
                tests.append(node)
                if len(tests) == 5:
                    return tests

    for name in symbols:
        for caller in find_symbol_callers(repo, name, exclude_rel)[:5]:
            found = find_regression_tests(repo, caller)
            if found and found not in tests:
                tests.append(found)
                if len(tests) == 5:
                    return tests
    return tests


def run_cpu_automl_gate(repo: Path, py: str) -> CommandResult:
    """Run a tiny CPU Fedot.fit against the evolved sources (no LLM involved).

    This is the AutoML-specific acceptance gate: unit tests prove the patch is
    correct in the small, this proves the framework still works end to end.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [py, "-c", CPU_SMOKE_SNIPPET],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    out = "\n".join(p for p in (proc.stdout.strip(), proc.stderr.strip()) if p)
    code = 0 if (proc.returncode == 0 and "CPU_SMOKE_OK" in (proc.stdout or "")) else (proc.returncode or 1)
    return CommandResult("cpu AutoML gate: Fedot.fit on evolved sources", code, out[-2500:])


def _run_evolution_loop(
    inference: AIInference,
    repo: Path,
    workspace: Path,
    venv_python: str | None = None,
) -> EvolveResult:
    """Full scout → propose → validate → audit loop."""
    py = venv_python or resolve_repo_python(repo)
    model = f"{inference.config.provider}/{inference.config.model_name}"
    jpath = journal_path(workspace)
    journal = read_journal(jpath) if JOURNAL_ENABLED else []
    if journal:
        logger.info("EvolveAgent: journal has %s past evolutions", len(journal))
    pick, why = scout_pick(inference, repo, journal, py)
    # Symbol-level view rather than the first N characters: see `source_view`
    # for why truncation is the wrong compression here. The focus symbol comes
    # from whichever evidence selected this file.
    verified = verified_defect_for(repo, pick)
    invariant = invariant_defect_for(repo, py, pick)
    # When the target already has a proven failing test, the agent's job shrinks to
    # writing the patch — which is the one step it was never the bottleneck on.
    ready = proven_defect_for(repo, py, pick)
    # Whichever evidence picked this file also names the symbol to focus on. Both
    # sources are consulted: with only the invariant one, a lint-template target
    # had no focus and the view fell back to showing every body in full — which
    # is the truncation problem again, wearing a different hat.
    focus = ((invariant or {}).get("symbol")
             or (ready.target.split("(")[0].split(".")[-1] if ready else "")
             or "")
    source = source_view(repo, pick, focus) if SYMBOL_VIEW else read_source_file(repo, pick)
    result_evidence = ("verified" if verified else "template" if ready
                       else "invariant" if invariant else "")

    def request_proposal(current_messages: list[dict[str, Any]]) -> Proposal:
        return bind_evidence_test(
            ask_proposal(inference, current_messages),
            ready=ready,
            invariant=invariant,
            verified=verified,
        )

    # Abstention gate. Before spending a single token on a patch: is there
    # evidence that this file holds a defect? If not, the honest answer is
    # "nothing to fix here" — see VALUE_GATE for why this is code and not prompt.
    # A lint finding in the picked file is not evidence, but it is a lead the
    # agent may follow — provided it produces the proof itself and the reproduce
    # gate agrees.
    lint_leads = lint_findings_for(repo, pick) if TRIAGE_LINT else []
    if lint_leads:
        result_evidence = result_evidence or "triage"

    if VALUE_GATE and not result_evidence:
        empty = Proposal(file_path=pick, test_file="", test_name="", problem="",
                         rationale="", old_code="", new_code="", test_code="")
        result = EvolveResult(proposal=empty, pick=pick, why=why)
        result.abstained = True
        result.abstain_reason = (
            f"no pre-existing evidence of a defect in {pick}: neither a template test "
            "that already fails nor a measured invariant violation"
        )
        logger.info("EvolveAgent ABSTAINS: %s", result.abstain_reason)
        workspace.mkdir(parents=True, exist_ok=True)
        audit_md = build_audit(repo, result, model)
        (workspace / "evolution_audit.md").write_text(audit_md, encoding="utf-8")
        result.audit_path = str(workspace / "evolution_audit.md")
        result.audit_markdown = audit_md
        if JOURNAL_ENABLED:
            append_journal(jpath, {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "file": pick, "pick": pick, "problem": "(abstained)",
                "success": False, "attempts": 0, "severity": 4,
                "severity_name": "abstained", "reason": result.abstain_reason,
            })
        return result

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PROPOSE_SYS},
        {
            "role": "user",
            "content": (
                f"Scout selected `{pick}` because: {why}\n\n"
                f"## FILE: {pick}\n{source}"
                f"{dependency_context(repo, pick, focus) if SYMBOL_VIEW else ''}"
                f"{usage_context(repo, pick, focus) if SYMBOL_VIEW else ''}"
                f"{module_usage_section(repo, pick)}"
                + (verified_section(repo, verified) if verified
                   else ready_section(repo, pick, ready) if ready
                   else invariant_section(repo, invariant) if invariant
                   else triage_section(lint_leads) if lint_leads else "")
                + "\n\n"
                "Propose ONE improvement in the delimiter format from the system prompt "
                "(NOT JSON). FILE must stay the scout-selected path unless you must "
                "touch a tiny helper in the same module."
            ),
        },
    ]
    proposal = request_proposal(messages)
    logger.info("EvolveAgent candidate: %s — %s", proposal.file_path, proposal.problem)

    result = EvolveResult(proposal=proposal, pick=pick, why=why, evidence=result_evidence)
    base_messages = list(messages)
    attempts_used = 0
    generated_tests: set[str] = set()

    # Best-of-N: each candidate gets its own self-fix budget; the
    # first one clearing every gate wins. No deep search — breadth is cheaper.
    for candidate in range(1, NUM_CANDIDATES + 1):
        if candidate > 1:
            logger.info("EvolveAgent: candidate %s/%s (fresh sample)", candidate, NUM_CANDIDATES)
            messages = list(base_messages)
            proposal = request_proposal(messages)
            result.proposal = proposal

        for attempt in range(1, MAX_FIX_TRIES + 1):
            attempts_used += 1
            reset_evolution_changes(repo, generated_tests)

            proposal.anchor = infer_anchor(repo, proposal)
            if proposal.test_file and not (repo / proposal.test_file).exists():
                generated_tests.add(proposal.test_file)
            apply_test(repo, proposal)
            reproduce = run_cmd(
                [py, "-m", "pytest", f"{proposal.test_file}::{proposal.test_name}", "-q"],
                cwd=repo,
            )
            result.reproduce = reproduce

            # Gate 1 — the test must FAIL on pristine source, otherwise the patch
            # proves nothing.
            if reproduce.exit_code == 0:
                logger.info(
                    "EvolveAgent attempt %s: test passes on pristine source (no repro) — rejecting",
                    attempt,
                )
                messages.append({"role": "assistant", "content": proposal_as_delimited(proposal)})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your test PASSES on the unpatched source, so it does not "
                            "demonstrate any bug. Write a test that FAILS on the current "
                            "code and only passes after your patch (assert the improved "
                            "behaviour, e.g. pytest.raises(ValueError) where the code "
                            "currently raises KeyError or nothing).\n\n"
                            "Return the corrected reply in the SAME delimiter format."
                        ),
                    }
                )
                proposal = request_proposal(messages)
                result.proposal = proposal
                continue

            ok = apply_source(repo, proposal)
            if not ok:
                logger.info(
                    "EvolveAgent attempt %s: old_code not found/ambiguous, re-quoting",
                    attempt,
                )
                src = (source_view(repo, proposal.file_path, proposal.anchor)
                       if SYMBOL_VIEW else read_source_file(repo, proposal.file_path))
                messages.append(
                    {"role": "assistant", "content": proposal_as_delimited(proposal)}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"old_code was not found as a UNIQUE verbatim match in "
                            f"{proposal.file_path}. Copy a longer exact contiguous snippet "
                            f"(full lines) from the file below.\n\n{src}\n\n"
                            "Return the corrected reply in the SAME delimiter format."
                        ),
                    }
                )
                proposal = request_proposal(messages)
                result.proposal = proposal
                continue

            compile_res = run_cmd(
                [py, "-m", "py_compile", proposal.file_path, proposal.test_file],
                cwd=repo,
            )
            test_res = run_cmd(
                [py, "-m", "pytest", f"{proposal.test_file}::{proposal.test_name}", "-q"],
                cwd=repo,
            )
            validations = [compile_res, test_res]
            result.changed = [proposal.file_path, proposal.test_file]
            result.proposal = proposal

            if compile_res.exit_code == 0 and test_res.exit_code == 0:
                # Gate 2 — existing tests must still pass: those of the patched
                # module AND those of the modules that call the changed symbol.
                # The agent only ever reads one file, so without the second half a
                # broken caller sails straight through acceptance.
                reg_targets = []
                own = find_regression_tests(repo, proposal.file_path, exclude=proposal.test_file)
                if own:
                    reg_targets.append(own)
                for extra in find_caller_regression_tests(repo, proposal.anchor, proposal.file_path):
                    if extra not in reg_targets and extra != proposal.test_file:
                        reg_targets.append(extra)
                reg_rel = " ".join(reg_targets)
                if reg_targets:
                    reg_res = run_cmd([py, "-m", "pytest", *reg_targets, "-q"], cwd=repo)
                    validations.append(reg_res)
                    if reg_res.exit_code != 0:
                        logger.info("EvolveAgent attempt %s: regression gate FAILED (%s)", attempt, reg_rel)
                        result.validations = validations
                        messages.append({"role": "assistant", "content": proposal_as_delimited(proposal)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your patch broke existing tests in {reg_rel}:\n"
                                    f"{reg_res.output[:2500]}\n\n"
                                    "Fix the patch so both your new test AND the existing "
                                    "tests pass. Return the SAME delimiter format."
                                ),
                            }
                        )
                        proposal = request_proposal(messages)
                        result.proposal = proposal
                        continue

                # Gate 3 — the evolved framework must still run AutoML on CPU.
                if AUTOML_GATE:
                    smoke = run_cpu_automl_gate(repo, py)
                    validations.append(smoke)
                    if smoke.exit_code != 0:
                        logger.info("EvolveAgent attempt %s: CPU AutoML gate FAILED", attempt)
                        result.validations = validations
                        messages.append({"role": "assistant", "content": proposal_as_delimited(proposal)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your patch broke the FEDOT runtime — a plain CPU "
                                    f"Fedot.fit now fails:\n{smoke.output[:2000]}\n\n"
                                    "Make the change safe for normal AutoML usage. "
                                    "Return the SAME delimiter format."
                                ),
                            }
                        )
                        proposal = request_proposal(messages)
                        result.proposal = proposal
                        continue

                # Gate 4 — runtime behaviour, measured on cases the agent never
                # authored and never saw. Cheap safety net against a patch that
                # makes a defect vanish by refusing valid input.
                if PROBE_GATE:
                    ok, resolved, summary = run_probe_gate(repo, py)
                    validations.append(
                        CommandResult("runtime probe gate", 0 if ok else 1, summary)
                    )
                    result.probe_resolved = resolved
                    if not ok:
                        logger.info("EvolveAgent attempt %s: probe gate FAILED (%s)", attempt, summary)
                        result.validations = validations
                        messages.append({"role": "assistant", "content": proposal_as_delimited(proposal)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your patch changed FEDOT's runtime behaviour for the "
                                    f"worse: {summary}. Ordinary data must keep working — do "
                                    "not fix a defect by rejecting valid input. Return the "
                                    "SAME delimiter format."
                                ),
                            }
                        )
                        proposal = request_proposal(messages)
                        result.proposal = proposal
                        continue
                    if resolved:
                        logger.info("EvolveAgent: runtime defect(s) resolved: %s", resolved)

                # Gate 5 — for a defect that says "this parameter cannot be
                # used", the operation must end up tunable. Stopping the error
                # by discarding the parameter does not count.
                if TUNING_GATE and invariant and invariant.get("kind") == "unusable_parameter":
                    ok, summary = run_tuning_gate(repo, py, invariant["operation"])
                    validations.append(CommandResult("tuning gate", 0 if ok else 1, summary))
                    if not ok:
                        logger.info("EvolveAgent attempt %s: tuning gate FAILED (%s)", attempt, summary)
                        result.validations = validations
                        messages.append({"role": "assistant", "content": proposal_as_delimited(proposal)})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"The error is gone but the operation still cannot be "
                                f"tuned: {summary}. Do NOT make a parameter usable by "
                                "dropping it — the caller declared it and must get it. "
                                "Fix the disagreement itself (the declared search space "
                                "and the operation's own defaults name the same setting "
                                "twice). Return the SAME delimiter format."
                            ),
                        })
                        proposal = request_proposal(messages)
                        result.proposal = proposal
                        continue

                result.validations = validations
                logger.info(
                    "EvolveAgent GREEN (candidate %s, attempt %s, %s total attempts)",
                    candidate,
                    attempt,
                    attempts_used,
                )
                result.success = True
                break

            result.validations = validations
            logger.info("EvolveAgent attempt %s: validation failed, self-fixing", attempt)
            messages.append(
                {"role": "assistant", "content": proposal_as_delimited(proposal)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"After applying, validation failed.\n"
                        f"py_compile: exit {compile_res.exit_code}\n{compile_res.output[:1500]}\n\n"
                        f"pytest: exit {test_res.exit_code}\n{test_res.output[:2500]}\n\n"
                        "Return the corrected reply in the SAME delimiter format that makes "
                        "pytest pass. Keep OLD as an exact unique snippet from the source file."
                    ),
                }
            )
            proposal = request_proposal(messages)
            result.proposal = proposal

        if result.success:
            break

    result.attempts = attempts_used

    # Second half of the value gate. Passing every gate proves the patch is safe
    # and that its test now passes; it does not prove the test was worth writing.
    # A run counts only if it is anchored to evidence that existed BEFORE the
    # patch, or if it demonstrably removed a runtime defect the agent never saw.
    if VALUE_GATE and result.success:
        anchored = bool(result.evidence) and bool(
            (
                verified
                and result.proposal.test_name == verified["test_name"]
                and result.proposal.test_code == verified["test_code"]
            )
            or (
                ready
                and result.proposal.test_name == ready.test_name
                and result.proposal.test_code == ready.test_code
            )
            or (
                invariant
                and result.proposal.test_name == invariant["test_name"]
                and result.proposal.test_code == invariant["test_code"]
            )
        )
        # A triage run has no ready-made test by construction: the whole point is
        # that the agent had to prove the warning was live. Its proof is the
        # reproduce gate — the test had to fail on the untouched tree before the
        # patch was even considered — so that is what anchors it here. Without
        # this branch the path was open and immediately sealed from the other
        # side: every triage patch would clear six gates and then be discarded.
        if result.evidence == "triage":
            anchored = bool(result.reproduce and result.reproduce.exit_code != 0)

        if not anchored and not result.probe_resolved:
            result.success = False
            result.abstained = True
            result.abstain_reason = (
                "the patch passed every gate but is anchored to a test the agent wrote "
                "for itself, not to a defect proven before the patch"
            )
            logger.info("EvolveAgent ABSTAINS after the fact: %s", result.abstain_reason)

    if not result.success:
        reset_evolution_changes(repo, generated_tests)

    # Classify what was actually fixed, not why the file was chosen. Including the
    # scout's `why` inflated the class: it picked data.py for a default-argument
    # defect and the accepted patch only clarified an error message, yet the run
    # was recorded as class 1.
    result.severity, result.severity_name = classify_accepted_severity(
        ready,
        invariant,
        result.probe_resolved,
        f"{result.proposal.problem} {result.proposal.rationale}",
        verified=verified,
    )
    workspace.mkdir(parents=True, exist_ok=True)
    audit_md = build_audit(repo, result, model)
    audit_path = workspace / "evolution_audit.md"
    audit_path.write_text(audit_md, encoding="utf-8")
    (workspace / "proposal.json").write_text(
        json.dumps(asdict(result.proposal), indent=2), encoding="utf-8"
    )
    result.audit_path = str(audit_path)
    result.audit_markdown = audit_md
    if JOURNAL_ENABLED:
        append_journal(
            jpath,
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "file": result.proposal.file_path or pick,
                "pick": pick,
                "problem": result.proposal.problem,
                "success": result.success,
                "attempts": result.attempts,
                "severity": result.severity,
                "severity_name": result.severity_name,
                "reason": "" if result.success else "validation not passed",
            },
        )
    return result


def run_evolution_loop(
    inference: AIInference,
    repo: Path,
    workspace: Path,
    venv_python: str | None = None,
) -> EvolveResult:
    """Run evolution without risking pre-existing or half-applied user changes."""
    require_clean_repo(repo)
    try:
        return _run_evolution_loop(inference, repo, workspace, venv_python)
    except BaseException:
        reset_evolution_changes(repo, untracked_repo_files(repo))
        raise

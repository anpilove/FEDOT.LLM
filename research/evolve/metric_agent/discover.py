"""Walk FEDOT source for a patch site. No tests, no exam, no gym."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from research.evolve.metric_agent.context import inspect_trace, show_source
from research.evolve.metric_agent.guard import deny_write
from research.evolve.metric_agent.repo_map import _FIT_NAMES, _area, leads_from_map, repo_map
from research.evolve.metric_agent.types import Lead, ScoreResult

EXCLUDED_DIR_PARTS = {".git", "__pycache__", ".pytest_cache", "docs", "examples", "jupyter_notebooks", "caching", "visualisation"}
COSMETIC_PREFIXES = (
    "Q", "E", "W", "D", "ANN", "COM", "I", "TID", "TD", "FIX",
    "ERA", "N", "FA", "UP", "PTH", "EM", "RSE", "ICN", "INP",
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINT_NOISE = re.compile(r"^(F(401|403|404|405|541|811|841)|RUF010)\b")
_FAILED = re.compile(r"^(FAILED|ERROR) (test/\S+)", re.MULTILINE)
_LINT_LIMIT = 40
_FILE_LIMIT = 80
_TEST_LEAD_LIMIT = 8

_LOCATE = """You inspect FEDOT library source. Pick one site that runs when a
pipeline fits, transforms, or predicts: operations, models, preprocessing, data,
defaults. Walk those areas — not one field, not cache/logging/DB, not tests.
Do not mention tests, benchmarks, datasets, or case catalogs. Output one
file_path relative to the FEDOT checkout (fedot/...)."""


class SiteProposal(BaseModel):
    file_path: str = Field(description="Path relative to FEDOT checkout, e.g. fedot/core/foo.py")
    line: int = Field(ge=1)
    why: str = ""


def parse_lint(line: str) -> dict | None:
    try:
        location, rest = line.split(": ", 1)
        file_rel, line_no, column = location.split(":")[:3]
        rule, message = rest.split(" ", 1)
        letters = re.match(r"[A-Z]+", rule)
        cosmetic = bool(letters) and letters.group(0) in COSMETIC_PREFIXES
        return {
            "file": file_rel,
            "line": int(line_no),
            "column": int(column),
            "rule": rule,
            "message": message,
            "cosmetic": cosmetic,
        }
    except (ValueError, IndexError):
        return None


def collect_lint(repo: Path, rules: str = "ALL") -> list[str]:
    unavailable = []
    for command in (["ruff"], [sys.executable, "-m", "ruff"], ["uvx", "ruff"]):
        try:
            result = subprocess.run(
                [
                    *command,
                    "check",
                    "fedot/",
                    f"--select={rules}",
                    "--output-format=concise",
                    "--no-fix",
                    "--isolated",
                    "--exclude=*.ipynb",
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            unavailable.append(f"{command[0]}: {exc}")
            continue
        stdout = _ANSI.sub("", result.stdout or "")
        if stdout.strip():
            return [ln for ln in stdout.splitlines() if ln.strip()]
        if result.returncode == 0:
            return []
        unavailable.append(f"{command[0]} exited {result.returncode}: {(result.stderr or '')[:200]}")
    raise RuntimeError("ruff scan unavailable: " + "; ".join(unavailable))


def lint_leads(checkout: Path, *, limit: int = _LINT_LIMIT) -> list[Lead]:
    rows = [parse_lint(line) for line in collect_lint(checkout)]
    leads: list[Lead] = []
    for row in rows:
        if row is None or row["cosmetic"] or _LINT_NOISE.match(row["rule"]):
            continue
        if deny_write(checkout / row["file"], checkout=checkout):
            continue
        leads.append(
            Lead(channel="lint", file_path=row["file"], line=row["line"], why=f"{row['rule']} {row['message']}")
        )
        if len(leads) >= limit:
            break
    return leads


def core_py_files(checkout: Path, *, limit: int = _FILE_LIMIT) -> list[str]:
    root = checkout / "fedot"
    if not root.is_dir():
        return []
    files: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in EXCLUDED_DIR_PARTS for part in path.parts):
            continue
        rel = path.relative_to(checkout).as_posix()
        if rel.startswith("fedot/api/") or not _fit_path(rel):
            continue
        files.append(rel)
    from research.evolve.metric_agent.repo_map import Symbol, _spread

    fake = [Symbol(rel, Path(rel).stem, "file", 1) for rel in files]
    return [item.file_path for item in _spread(fake, limit)]


def _lead_name(lead: Lead) -> str:
    token = (lead.why or "").strip().split()[-1] if (lead.why or "").strip() else ""
    return token.rsplit(".", 1)[-1] if token else ""


def _runtime_lead(lead: Lead) -> bool:
    return _lead_name(lead) in _FIT_NAMES


def _pkg_rank(path: str) -> int:
    """Prefer operator implementations over pipeline orchestration and helpers."""

    if "/operation_implementations/" in path:
        return 0
    if "/operations/" in path:
        return 1
    if path.startswith("fedot/core/data"):
        return 2
    if path.startswith("fedot/core/repository"):
        return 3
    if path.startswith("fedot/core/pipelines"):
        return 4
    return 5


def _fit_path(path: str) -> bool:
    """Sites that run during pipeline fit/transform/predict — not cache or search infra."""

    return path.startswith(
        (
            "fedot/core/operations/",
            "fedot/core/data/",
            "fedot/core/pipelines/",
            "fedot/core/repository/",
        )
    ) or path in {"fedot/core/data.py", "fedot/core/repository.py"}


def _rank_leads(leads: list[Lead]) -> list[Lead]:
    def key(lead: Lead) -> tuple:
        runtime = 0 if _runtime_lead(lead) else 1
        core = 0 if lead.file_path.startswith("fedot/core/") else 1
        private = 1 if _lead_name(lead).startswith("_") else 0
        return (runtime, core, _pkg_rank(lead.file_path), private, lead.file_path, lead.line)

    return sorted(leads, key=key)


def _spread_leads(leads: list[Lead], *, limit: int | None = None) -> list[Lead]:
    buckets: dict[str, list[Lead]] = {}
    for lead in leads:
        buckets.setdefault(_area(lead.file_path), []).append(lead)
    out: list[Lead] = []
    cap = len(leads) if limit is None else max(1, limit)
    while len(out) < cap:
        progressed = False
        for key in list(buckets):
            group = buckets[key]
            if not group:
                continue
            out.append(group.pop(0))
            progressed = True
            if len(out) >= cap:
                return out
        if not progressed:
            break
    return out


def _order_leads(leads: list[Lead]) -> list[Lead]:
    tiers: dict[tuple[int, int], list[Lead]] = {}
    for lead in leads:
        runtime = 0 if _runtime_lead(lead) else 1
        tiers.setdefault((runtime, _pkg_rank(lead.file_path)), []).append(lead)
    ordered: list[Lead] = []
    for key in sorted(tiers):
        ordered.extend(_spread_leads(tiers[key]))
    return ordered


def format_lint_for_llm(leads: list[Lead]) -> str:
    useful = [lead for lead in leads if lead.channel == "lint"]
    if not useful:
        return "(no non-cosmetic ruff hits)"
    return "\n".join(f"{lead.file_path}:{lead.line} {lead.why}" for lead in useful)


def parse_pytest_output(text: str, checkout: Path) -> list[Lead]:
    """Leads from FEDOT checkout pytest output. Ignores frames outside fedot/."""

    nodes = [match.group(2) for match in _FAILED.finditer(text or "")]
    frames = inspect_trace(text or "", checkout=checkout)
    why = nodes[0] if nodes else "pytest failure"
    leads: list[Lead] = []
    seen: set[tuple[str, int]] = set()
    for frame in reversed(frames):
        if not frame["file"].startswith("fedot/"):
            continue
        key = (frame["file"], int(frame["line"]))
        if key in seen:
            continue
        seen.add(key)
        leads.append(
            Lead(
                channel="fedot_test",
                file_path=frame["file"],
                line=int(frame["line"]),
                why=f"{why} in {frame['func']}",
            )
        )
        if len(leads) >= _TEST_LEAD_LIMIT:
            break
    return leads


def _as_text(blob: str | bytes | None) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", errors="replace")
    return blob


def failed_pytest_nodes(text: str) -> set[str]:
    return {match.group(2) for match in _FAILED.finditer(text or "")}


def pytest_output(checkout: Path, *, timeout_s: float | None = None, maxfail: int = 8) -> str:
    test_root = checkout / "test" / "unit"
    if not test_root.is_dir():
        return ""
    limit = timeout_s if timeout_s is not None else float(os.environ.get("METRIC_AGENT_PYTEST_TIMEOUT", "120"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout.resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "test/unit",
        "-q",
        "--tb=native",
        f"--maxfail={maxfail}",
        "--no-header",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            timeout=limit,
            check=False,
        )
        return _as_text(proc.stdout) + "\n" + _as_text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        return _as_text(exc.stdout) + "\n" + _as_text(exc.stderr)


def pytest_snapshot(checkout: Path, *, timeout_s: float | None = None, maxfail: int = 8) -> tuple[str, list[Lead], set[str]]:
    text = pytest_output(checkout, timeout_s=timeout_s, maxfail=maxfail)
    return text, parse_pytest_output(text, checkout), failed_pytest_nodes(text)


def pytest_leads(checkout: Path, *, timeout_s: float | None = None, maxfail: int = 8) -> list[Lead]:
    return parse_pytest_output(pytest_output(checkout, timeout_s=timeout_s, maxfail=maxfail), checkout)


def _crash_why(result: ScoreResult) -> str:
    """Exception text only. Never the exam task_id."""

    detail = (result.detail or "").strip().splitlines()
    if detail:
        return detail[0][:200]
    for line in reversed((result.traceback or "").splitlines()):
        text = line.strip()
        if text and not text.startswith("File ") and "Traceback" not in text and 'File "' not in text:
            return text[:200]
    return "crash"


def leads_from_scores(stock: dict[str, ScoreResult] | None, checkout: Path) -> list[Lead]:
    """Localize from execution evidence. Ignores dict keys (those are harness ids)."""

    if not stock:
        return []
    leads: list[Lead] = []
    seen: set[tuple[str, int]] = set()
    for result in stock.values():
        if result.status != "crash":
            continue
        why = _crash_why(result)
        frames = inspect_trace(result.traceback or "", checkout=checkout)
        for frame in reversed(frames):
            if not frame["file"].startswith("fedot/"):
                continue
            key = (frame["file"], int(frame["line"]))
            if key in seen:
                continue
            seen.add(key)
            leads.append(
                Lead(
                    channel="trace",
                    file_path=frame["file"],
                    line=int(frame["line"]),
                    why=f"{why} in {frame['func']}",
                )
            )
            if len(leads) >= _TEST_LEAD_LIMIT:
                return leads
    return leads


def discover_leads(
    checkout: Path,
    *,
    inference=None,
    limit: int = _LINT_LIMIT,
) -> list[Lead]:
    pooled = _unique(
        lead
        for lead in leads_from_map(repo_map(checkout, (), limit=_FILE_LIMIT))
        if _fit_path(lead.file_path) and not _lead_name(lead).startswith("_")
    )
    pooled = _order_leads(pooled)
    if inference is not None:
        picked = _llm_pick(inference, checkout, pooled)
        if (
            picked is not None
            and _fit_path(picked.file_path)
            and _runtime_lead(picked)
            and _pkg_rank(picked.file_path) <= 3
            and not _lead_name(picked).startswith("_")
        ):
            pooled = _unique([picked] + pooled)
    return pooled[: max(1, limit)]


def _unique(leads: list[Lead]) -> list[Lead]:
    seen: set[tuple[str, int]] = set()
    out: list[Lead] = []
    for lead in leads:
        key = (lead.file_path, lead.line)
        if key in seen:
            continue
        seen.add(key)
        out.append(lead)
    return out


def _llm_pick(inference, checkout: Path, leads: list[Lead]) -> Lead | None:
    files = core_py_files(checkout)
    mapped = "\n".join(f"{lead.file_path}:{lead.line} {lead.why}" for lead in leads[:30]) or "(empty)"
    prompt = (
        f"{_LOCATE}\n\n"
        f"Repo map:\n{mapped}\n\n"
        f"FEDOT python files (truncated):\n" + "\n".join(files)
    )
    try:
        parsed = inference.create(prompt, SiteProposal)
    except Exception:
        return None
    rel = parsed.file_path.lstrip("/")
    if "fedot/" in rel and not rel.startswith("fedot/"):
        rel = rel[rel.index("fedot/") :]
    target = (checkout / rel).resolve()
    if deny_write(target, checkout=checkout) or not target.is_file():
        return None
    if not _fit_path(rel):
        return None
    if not show_source(target, checkout=checkout, around=parsed.line):
        return None
    return Lead(channel="llm", file_path=rel, line=int(parsed.line), why=parsed.why)

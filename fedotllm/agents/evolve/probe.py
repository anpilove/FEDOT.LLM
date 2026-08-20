"""Runtime probing: find real FEDOT defects by running it, not by linting it.

The lint-driven scout only ever produced cosmetic fixes (measured: 100% success,
all class-4). Static analysis cannot see a defect that only exists when the code
runs. This module feeds the scout targets that came out of a real ``Fedot.fit``.

A finding is raised when FEDOT is given **legal but awkward** input and answers
with somebody else's internal error — e.g. a CatBoost C++ source path reaching an
end user. Control cases (ordinary data) must keep working; they are what stops a
"fix" that simply rejects everything.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from fedotllm.log import logger

PROBE_TIMEOUT_S = int(os.environ.get("FEDOTLLM_PROBE_TIMEOUT", "900"))

# Each case: name, whether it must succeed, and the code building (features, target).
# Kept deliberately small and CPU-cheap — this runs before every evolution round.
CASES: list[tuple[str, bool, str]] = [
    ("control_plain", True, "X = rng.random((120, 4)); y = (X[:, 0] > .5).astype(int)"),
    ("control_noisy", True, "X = rng.random((120, 4)); y = (rng.random(120) > .5).astype(int)"),
    ("constant_column", True, "X = rng.random((120, 4)); X[:, 2] = 0.0; y = (X[:, 0] > .5).astype(int)"),
    ("single_nan", True, "X = rng.random((120, 4)); X[5, 1] = np.nan; y = (X[:, 0] > .5).astype(int)"),
    ("inf_column", True, "X = rng.random((120, 4)); X[:, 3] = np.inf; y = (X[:, 0] > .5).astype(int)"),
    ("float_target", True, "X = rng.random((120, 4)); y = (X[:, 0] > .5).astype(float)"),
    ("single_class_target", False, "X = rng.random((120, 4)); y = np.zeros(120, dtype=int)"),
    ("three_rows", False, "X = rng.random((3, 4)); y = np.array([0, 1, 0])"),
    ("one_feature", True, "X = rng.random((120, 1)); y = (X[:, 0] > .5).astype(int)"),
]

# Substrings that mark an error as leaking a third-party internal detail.
FOREIGN_MARKERS = (
    ".cpp:",
    ".pyx",
    "/site-packages/",
    "catboost/",
    "lightgbm/",
    "xgboost/",
    "sklearn/",
)

RUNNER = '''
import json, sys, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np
from fedot import Fedot

CASES = json.loads(sys.argv[1])
out = []
for name, must_pass, setup in CASES:
    rng = np.random.default_rng(0)
    scope = {"rng": rng, "np": np}
    exec(setup, scope)
    X, y = scope["X"], scope["y"]
    rec = {"case": name, "must_pass": must_pass}
    try:
        m = Fedot(problem="classification", timeout=0.15, seed=1,
                  logging_level=50, with_tuning=False)
        m.fit(features=X, target=y)
        m.predict(features=X)
        rec.update(ok=True)
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__)
        fed = [f for f in tb if "/fedot/" in f.filename]
        rec.update(
            ok=False,
            exc=type(e).__name__,
            message=str(e)[:400],
            # Where FEDOT last had control — that is where validation belongs.
            location=(fed[-1].filename.split("/fedot/")[-1] + ":" + str(fed[-1].lineno))
            if fed else "",
            frames=[f.filename.split("/fedot/")[-1] + ":" + str(f.lineno) for f in fed[-4:]],
        )
    out.append(rec)
print("__PROBE__" + json.dumps(out))
'''


@dataclass
class Finding:
    case: str
    kind: str  # "leaked_foreign_error" | "control_broken" | "unclear_error"
    severity: int
    exc: str
    message: str
    location: str

    def as_prompt_line(self) -> str:
        return f"[{self.case}] {self.location} -> {self.exc}: {self.message[:150]}"


def _classify(rec: dict) -> Finding | None:
    """Turn one probe record into a finding, or None when FEDOT behaved well."""
    if rec.get("ok"):
        return None
    msg = rec.get("message", "")
    loc = rec.get("location", "")
    if rec.get("must_pass"):
        # Ordinary data must never fail — this is the strongest signal there is.
        return Finding(rec["case"], "control_broken", 1, rec["exc"], msg, loc)
    if any(mark in msg for mark in FOREIGN_MARKERS):
        # Legal-but-awkward input is allowed to be refused, but the refusal must
        # come from FEDOT and name the problem — not quote a foreign source file.
        return Finding(rec["case"], "leaked_foreign_error", 2, rec["exc"], msg, loc)
    if rec["exc"] in ("KeyError", "IndexError", "AttributeError", "TypeError"):
        return Finding(rec["case"], "unclear_error", 3, rec["exc"], msg, loc)
    return None


def run_probe(repo: Path, python: str, cases: list | None = None) -> list[Finding]:
    """Run the battery inside the FEDOT venv against `repo`; return findings."""
    selected = cases if cases is not None else CASES
    script = repo / ".fedotllm_probe.py"
    try:
        script.write_text(textwrap.dedent(RUNNER), encoding="utf-8")
        proc = subprocess.run(
            [python, str(script), json.dumps(selected)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"probe timed out after {PROBE_TIMEOUT_S}s")
    finally:
        script.unlink(missing_ok=True)

    marker = "__PROBE__"
    line = next(
        (ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(marker)), None
    )
    if not line:
        raise RuntimeError(
            f"probe produced no result (exit {proc.returncode}): {proc.stderr[-500:]}"
        )
    records = json.loads(line[len(marker) :])
    findings = [f for f in (_classify(r) for r in records) if f is not None]
    findings.sort(key=lambda f: f.severity)
    logger.info("probe: %s cases, %s findings", len(records), len(findings))
    return findings


def _repo_state(repo: Path) -> str:
    """Identity of the working tree: HEAD plus a digest of any local changes.

    A plain dirty/clean flag is not enough — the acceptance gate probes the tree
    both before and after a patch, and two different patches must not share a
    cache entry.
    """
    def git(*args: str) -> str:
        r = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        )
        return r.stdout

    head = git("rev-parse", "HEAD").strip()
    diff = git("diff", "HEAD")
    if not diff.strip():
        return f"{head}:clean"
    return f"{head}:{hashlib.sha1(diff.encode('utf-8')).hexdigest()[:16]}"


def _cache_path() -> Path:
    return Path(os.environ.get("FEDOTLLM_PROBE_CACHE", "/tmp/fedotllm_probe_cache.json"))


def _read_cache() -> dict:
    path = _cache_path()
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return blob if isinstance(blob, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("probe cache unreadable (%s); starting fresh", exc)
        return {}


def run_probe_cached(repo: Path, python: str) -> list[Finding]:
    """`run_probe` memoised on the working-tree state.

    A probe costs ~2 minutes of real fits. Benchmarks reset the repo to pristine
    before every run, so the state — and therefore the result — is identical each
    time; without this the cost would be paid 30 times over for one answer.

    The pristine result is kept in its own slot. A single-slot cache let a probe
    of an already-patched tree evict the baseline, after which the acceptance
    gate had nothing to compare against and silently skipped itself.
    """
    state = _repo_state(repo)
    blob = _read_cache()
    entries = blob.setdefault("entries", {})
    if state in entries:
        logger.info("probe: reusing cached result for %s", state)
        payload = entries[state]
        # A cache hit used to return here without touching the baseline slot,
        # so a baseline recorded for a different checkout stayed in place and
        # the acceptance gate reported "no pristine baseline" on every single
        # run. Measured: seven repair rounds rejected in a row, none of them
        # for anything wrong with the patch.
        if state.endswith(":clean") and (blob.get("baseline") or {}).get("head") != state.split(":")[0]:
            blob["baseline"] = {"head": state.split(":")[0], "findings": payload}
            try:
                _cache_path().write_text(json.dumps(blob), encoding="utf-8")
            except OSError as exc:
                logger.warning("probe: could not adopt baseline (%s)", exc)
            logger.info("probe: baseline adopted for %s", state.split(":")[0])
        return [Finding(**f) for f in payload]

    findings = run_probe(repo, python)
    payload = [asdict(f) for f in findings]
    entries[state] = payload
    if state.endswith(":clean"):
        blob["baseline"] = {"head": state.split(":")[0], "findings": payload}
    # Keep the file small: the baseline plus a handful of recent trees.
    if len(entries) > 8:
        blob["entries"] = dict(list(entries.items())[-8:])
    try:
        _cache_path().write_text(json.dumps(blob), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write probe cache: %s", exc)
    return findings


def read_pristine_findings(repo: Path) -> list[Finding] | None:
    """Cached findings of the *unpatched* tree, or None if not available.

    At gate time the tree is already patched, so re-probing would describe the
    wrong side of the comparison; only the entry recorded for the clean tree
    will do.
    """
    baseline = _read_cache().get("baseline") or {}
    if baseline.get("head") != _repo_state(repo).split(":")[0]:
        return None
    return [Finding(**f) for f in baseline.get("findings", [])]


def compare_probes(before: list[Finding], after: list[Finding]) -> tuple[bool, list[str], str]:
    """Verdict of the runtime gate: (accepted, resolved cases, human summary).

    The agent writes its own unit test, so that test is the easiest thing for it
    to bend. These cases are the opposite: they were fixed before the patch
    existed and the agent never sees them — the closest thing to a hidden test.
    """
    before_cases = {f.case for f in before}
    after_cases = {f.case for f in after}
    resolved = sorted(before_cases - after_cases)
    introduced = sorted(after_cases - before_cases)
    # A "fix" that refuses ordinary data is the cheapest way to make a defect
    # disappear; broken controls therefore veto the patch outright.
    broken_controls = sorted(f.case for f in after if f.kind == "control_broken")

    if broken_controls:
        return False, resolved, f"ordinary data no longer fits: {', '.join(broken_controls)}"
    if introduced:
        return False, resolved, f"new runtime defects: {', '.join(introduced)}"
    if resolved:
        return True, resolved, f"runtime defects resolved: {', '.join(resolved)}"
    return True, [], "no runtime regression (no runtime defect resolved either)"


def probe_section(findings: list[Finding], limit: int = 8) -> str:
    """Prompt section. Empty string when the runtime behaved — never bluff."""
    if not findings:
        return ""
    lines = [
        "FEDOT was actually run on legal but awkward input. Prefer these over "
        "lint findings — a linter cannot see them at all."
    ]
    for f in findings[:limit]:
        what = {
            "control_broken": "ordinary data fails to fit — a plain bug",
            "leaked_foreign_error": "a third-party internal error reaches the user; "
            "FEDOT should validate first and raise a clear error naming the problem",
            "unclear_error": "unclear exception type for invalid input",
        }[f.kind]
        lines.append(f"  - {f.as_prompt_line()}\n      ({what})")
    return "\n".join(lines)


def findings_as_json(findings: list[Finding]) -> str:
    return json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2)

#!/usr/bin/env python3
"""The second pass: does the suspicion survive contact with the interpreter?

The reader (`triage_lint.py --whole-file`) produces suspicions — statements of
the form "this breaks under these conditions". A suspicion is an opinion. This
pass turns each one into a short standalone script and lets the interpreter
decide: the script must fail on the untouched tree, and it must fail with the
error the model predicted in advance.

That last clause is the whole point. A script can exit non-zero because it is
wrong — a typo, a bad import, an attribute that never existed. Counting those as
defects is how a verifier quietly turns into a rubber stamp. So the model must
name the exception type BEFORE the run, and a mismatch is a refutation, not a
success.

    FEDOTLLM_REPO_PATH=/path/to/FEDOT uv run python examples/verify_leads.py \
        --leads ab_whole.json --model deepseek/deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fedotllm.configs.loader import load_config  # noqa: E402
from fedotllm.llm import AIInference  # noqa: E402

# The same fixtures the published issues use, so a script can reach for real
# data without spending ten lines building it.
PREAMBLE = '''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

_rng = np.random.default_rng(42)
_n = 120
_x = _rng.normal(size=(_n, 6))
train_data = InputData(idx=np.arange(_n), features=_x,
                       target=(_x[:, 0] + 0.5 * _x[:, 1] > 0).astype(int).reshape(-1, 1),
                       task=Task(TaskTypesEnum.classification),
                       data_type=DataTypesEnum.table)
_t = np.arange(200)
_series = np.sin(_t / 7.0) * 10 + _t * 0.05 + _rng.normal(size=200) * 0.2
ts_data = InputData(idx=_t, features=_series, target=_series,
                    task=Task(TaskTypesEnum.ts_forecasting,
                              TsForecastingParams(forecast_length=5)),
                    data_type=DataTypesEnum.ts)
'''

SYS = """You verify suspected defects in the FEDOT AutoML library by experiment.

You are given one suspicion about one file, plus that file's source. Write a
short standalone Python script that DEMONSTRATES the defect: it must raise on
the current, unpatched code. If the suspicion is right the script fails; if the
suspicion is wrong the script runs to completion.

Rules that decide whether your answer counts:

* Name the exception type first, on its own line: `EXPECT: IndexError`.
  A run that fails with a different type counts as a refutation, so do not
  guess — if you cannot predict the failure, answer `EXPECT: none`.
* Never `raise` the expected exception yourself and never assert your way to
  it. The library must produce it. A script that manufactures its own failure
  is a fabricated result.
* Call the real public entry point where you can. Reaching into a private
  helper proves less.
* Ten lines is plenty. No pytest, no fixtures, no argument parsing.
* These names already exist and need no imports: `train_data` (tabular
  classification, 120x6) and `ts_data` (time series, 200 points).
* If the suspicion cannot be demonstrated by running code — it is about style,
  or about a branch no caller can reach — answer `EXPECT: none` and explain in
  one line instead of writing a script.

Answer in exactly this shape:

EXPECT: <ExceptionType or none>
```python
<script>
```
"""

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The last "SomeError: message" line of a traceback is the failure that matters.
_EXC = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*Error|[A-Za-z_][A-Za-z0-9_]*Exception|"
                  r"KeyError|IndexError|TypeError|ValueError|AttributeError|"
                  r"AssertionError|StopIteration|KeyboardInterrupt)\b", re.M)
# Failures that mean the script is broken, not the library.
HARNESS_ERRORS = {"SyntaxError", "IndentationError", "ImportError",
                  "ModuleNotFoundError", "NameError", "IndentationError"}


def source_of(repo: Path, file_rel: str) -> str:
    text = (repo / file_rel).read_text(encoding="utf-8", errors="replace")
    return "\n".join(f"{i:5d}| {ln}" for i, ln in enumerate(text.splitlines(), 1))


def parse_answer(raw: str) -> tuple[str, str]:
    expect = "none"
    m = re.search(r"EXPECT:\s*([A-Za-z_][A-Za-z0-9_.]*|none)", raw)
    if m:
        expect = m.group(1)
    block = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.S)
    return expect, (block.group(1) if block else "")


def last_exception(output: str) -> str:
    hits = _EXC.findall(_ANSI.sub("", output))
    return hits[-1] if hits else ""


def run_script(script: str, repo: Path, py: str, workdir: Path, slug: str) -> tuple[int, str]:
    path = workdir / f"lead_{slug}.py"
    path.write_text(PREAMBLE + "\n" + script, encoding="utf-8")
    # cwd is not enough: the script lives in the workdir, so sys.path[0] points
    # there and `import fedot` would silently pick up whatever is installed in
    # site-packages instead of the tree under test.
    env = {**os.environ, "PYTHONPATH": str(repo)}
    try:
        proc = subprocess.run([py, str(path)], cwd=repo, capture_output=True,
                              text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    return proc.returncode, ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()


def fabricated(script: str, expect: str) -> bool:
    """The script raises the expected error itself instead of provoking it."""
    return bool(re.search(rf"^\s*raise\s+{re.escape(expect)}\b", script, re.M))


def verify_lead(inference, repo: Path, lead: dict, py: str, workdir: Path,
                attempts: int = 2) -> dict:
    slug = re.sub(r"[^a-z0-9]+", "_", f"{lead['file']}_{lead['line']}".lower())[-60:]
    prompt = (f"{SYS}\n\nFile: {lead['file']}\nLine: {lead['line']}\n"
              f"Suspicion: {lead['why']}\n\nSource:\n{source_of(repo, lead['file'])}")
    feedback = ""
    for attempt in range(1, attempts + 1):
        raw = inference.query(prompt + feedback) or ""
        expect, script = parse_answer(raw)
        if expect == "none" or not script.strip():
            return {**lead, "status": "not testable", "expect": expect,
                    "detail": raw.strip().splitlines()[0][:200] if raw.strip() else ""}
        if fabricated(script, expect):
            return {**lead, "status": "fabricated", "expect": expect,
                    "script": script, "detail": "script raises the error itself"}
        code, out = run_script(script, repo, py, workdir, slug)
        got = last_exception(out)
        tail = out.strip().splitlines()[-1][:300] if out.strip() else ""
        if code == 0:
            return {**lead, "status": "refuted", "expect": expect, "got": "",
                    "script": script, "detail": "script ran to completion"}
        if got in HARNESS_ERRORS:
            # The script is broken, not the library. Let it try once more.
            feedback = (f"\n\nYour previous script failed with {got}, which is a "
                        f"fault in the script, not in FEDOT:\n{tail}\nRewrite it.")
            continue
        if got != expect:
            return {**lead, "status": "mismatch", "expect": expect, "got": got,
                    "script": script, "detail": tail}
        # The right exception is not enough: it has to come from the file the
        # suspicion is about. Measured — 4 of 11 confirmations died somewhere
        # else entirely, one of them inside the constructor because the script
        # passed `Fedot(task=...)` where the parameter is called `problem`. The
        # type matched and the file had nothing to do with it.
        if lead["file"] not in out:
            return {**lead, "status": "wrong site", "expect": expect, "got": got,
                    "script": script, "detail": tail,
                    "route": f"{got} raised, but not inside {lead['file']}"}
        return {**lead, "status": "confirmed", "expect": expect, "got": got,
                "script": script, "detail": tail}
    return {**lead, "status": "broken script", "expect": expect,
            "script": script, "detail": tail}


# A user of FEDOT reaches the library through these. Everything else is an
# internal helper: making one raise proves the code path exists, not that anyone
# can walk it. This is decided by reading the script's imports, not by asking a
# model — three runs of an LLM reachability judge on the same nine suspicions
# answered 6 yes, then 3 no, then 6 yes.
# Exact module names, never prefixes: with "fedot" in the list every internal
# module in the library matched, and the gate waved through a CLI helper as a
# public entry point.
PUBLIC_ENTRY = frozenset({
    "fedot", "fedot.api", "fedot.api.main", "fedot.api.builder",
    "fedot.core.pipelines.pipeline", "fedot.core.pipelines.pipeline_builder",
    "fedot.core.data.data", "fedot.core.data.multi_modal",
    "fedot.core.repository.tasks", "fedot.core.repository.dataset_types",
    "fedot.core.repository.metrics_repository",
})

PUBLIC_SYS = """The failure you demonstrated went through an internal helper.

That proves the code path exists. It does not prove a user can reach it: the
call may be one FEDOT's own code never makes. Reach the same failure again using
only the public interface — `Fedot`, `FedotBuilder`, `Pipeline` — with no import
from `fedot.api.api_utils`, `fedot.core.operations`, or any other internal
module.

If no public route exists, say so: answer `EXPECT: none`. That is a real answer,
not a failure, and it is the right one when the state genuinely cannot arise in
use.

Let the exception propagate and end the script. Do not wrap the call in
`try`/`except` — a caught exception exits zero and reads as "nothing happened",
which throws away your own result.

Same rules: name the exception first, never raise it yourself, ten lines is
plenty, `train_data` and `ts_data` already exist.

EXPECT: <ExceptionType or none>
```python
<script>
```
"""


def internal_imports(script: str) -> list[str]:
    """Modules the script reaches into that a user has no business importing."""
    bad = []
    for m in re.finditer(r"^\s*(?:from|import)\s+(fedot[\w.]*)", script, re.M):
        mod = m.group(1)
        if mod not in PUBLIC_ENTRY:
            bad.append(mod)
    return sorted(set(bad))


def check_public_route(inference, repo: Path, result: dict, py: str,
                       workdir: Path, slug: str) -> dict:
    """Confirmed through an internal call: demand the same failure in public."""
    internal = internal_imports(result.get("script", ""))
    if not internal:
        return {**result, "route": "public interface"}
    raw = inference.query(
        f"{PUBLIC_SYS}\n\nSuspicion: {result['why']}\n\n"
        f"Script that reproduced it internally (imports {', '.join(internal)}):\n"
        f"```python\n{result['script']}\n```") or ""
    expect, script = parse_answer(raw)
    if expect == "none" or not script.strip() or fabricated(script, expect):
        return {**result, "status": "internal only",
                "route": f"no public route offered; reached via {', '.join(internal)}"}
    still_internal = internal_imports(script)
    if still_internal:
        return {**result, "status": "internal only",
                "route": f"rewrite still imports {', '.join(still_internal)}"}
    code, out = run_script(script, repo, py, workdir, slug + "_public")
    got = last_exception(out)
    if code != 0 and got == result.get("got") and got not in HARNESS_ERRORS:
        tail = out.strip().splitlines()[-1][:300] if out.strip() else ""
        return {**result, "route": "public interface (rewritten)",
                "script": script, "detail": tail}
    return {**result, "status": "internal only",
            "route": f"public rewrite did not reproduce it (got {got or 'no failure'})"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=Path, required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--verdicts", default="extra,live")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6,
                    help="leads verified concurrently; the work is waiting")
    ap.add_argument("--no-reachability", action="store_true",
                    help="skip the public-route gate (for measuring its effect)")
    ap.add_argument("--out", type=Path, default=Path("VERIFIED_LEADS.json"))
    args = ap.parse_args()

    repo = Path(os.environ.get("FEDOTLLM_REPO_PATH", "")).resolve()
    if not repo.is_dir():
        print("FEDOTLLM_REPO_PATH not set", file=sys.stderr)
        return 2
    py = os.environ.get("FEDOTLLM_PYTHON", sys.executable)
    workdir = Path(os.environ.get("FEDOTLLM_WORKDIR", "/tmp")) / "verify_leads"
    workdir.mkdir(parents=True, exist_ok=True)

    wanted = {v.strip() for v in args.verdicts.split(",")}
    rows = json.loads(args.leads.read_text(encoding="utf-8"))
    leads = [r for r in rows if r.get("verdict") in wanted and r.get("why")]
    if args.limit:
        leads = leads[: args.limit]
    print(f"{len(leads)} suspicions to verify against {repo}", flush=True)

    config = load_config(presets="fedotllm:openrouter",
                         overrides=[f"llm.model_name={args.model}"])
    inference = AIInference(config=config.llm)

    # Each lead is one model call plus one subprocess, and both are mostly
    # waiting. Sequentially the 244 suspicions of a full pass take about three
    # hours; a handful in flight brings it under half an hour for the same money.
    def handle(item):
        n, lead = item
        try:
            res = verify_lead(inference, repo, lead, py, workdir)
        except Exception as exc:
            res = {**lead, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
        return n, lead, res

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      for n, lead, res in pool.map(handle, enumerate(leads, 1)):
        done += 1
        if res["status"] == "confirmed" and not args.no_reachability:
            try:
                res = check_public_route(
                    inference, repo, res, py, workdir,
                    re.sub(r"[^a-z0-9]+", "_",
                           f"{res['file']}_{res['line']}".lower())[-60:])
            except Exception as exc:
                res["route"] = f"reachability check failed: {type(exc).__name__}"
        results.append(res)
        mark = {"confirmed": "OK  ", "refuted": "--  ", "mismatch": "?   ",
                "internal only": "int ", "wrong site": "site", "not testable": "n/a "}.get(res["status"], "!!  ")
        print(f"  [{done}/{len(leads)}] {mark}{lead['file']}:{lead['line']} "
              f"{res['status']} (expected {res.get('expect', '?')}, "
              f"got {res.get('got', '-') or '-'})", flush=True)

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out.with_suffix(".usage.json").write_text(
        json.dumps(inference.usage, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + " · ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The cheap first pass: is this lint warning worth anybody's time?

A separate role, and deliberately a separate model. Sorting warnings is
classification, not repair — the expensive model earns its price writing
patches (measured: `gpt-5.4` 4/5, `gpt-4o-mini` 0/12), and pays for nothing
here. At `deepseek-v4-flash` prices the whole of FEDOT's lint output costs
about a dollar, so there is no reason left to pre-filter by hand, which is what
I did before and could not justify: the "lint findings are worthless" verdict
was measured on ONE rule (`RUF012`, 39 of 40 dormant) and quietly applied to all
of them.

The verdict is never taken on trust. A finding called live only earns a run of
the full pipeline, where the reproduce gate demands a test that fails on the
untouched tree. This pass decides what to look at, not what is true.

    uv run python examples/triage_lint.py --model deepseek/deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fedotllm.configs.loader import load_config  # noqa: E402
from fedotllm.llm import AIInference  # noqa: E402

# Rules dropped without asking anyone. Not a judgement call: quoting style and
# line length cannot have a runtime consequence by the definition of the rule,
# and they are 10 778 of FEDOT's 14 062 findings. Paying a model to read them is
# paying it to reformat someone else's library.
COSMETIC_PREFIXES = ("Q", "E", "W", "D", "ANN", "COM", "I", "TID", "TD", "FIX",
                     "ERA", "N", "FA", "UP", "PTH", "EM", "RSE", "ICN", "INP")

TRIAGE_SYS = """You sort static-analysis warnings for the FEDOT AutoML library.

For each warning decide one thing: could it cause something to go wrong at
runtime, or is it inert here?

Inert means the shape the linter objects to has no consequence in this code —
for example a mutable class attribute that nothing ever writes to, an unused
argument that exists to satisfy an interface, a bare `except` around code that
cannot raise. Most warnings in a mature library are inert; say so plainly.

Live means you can name what breaks and under what conditions. "It is bad
practice" is not live. "This request has no timeout and runs in the composer's
main loop, so a hung server hangs the fit" is live.

Judge only what the code shows you. If the snippet is not enough to tell, say
`unclear` rather than guessing.

Reply with one line per warning, nothing else:

<index>: live|inert|unclear | <one clause: what breaks, or why it cannot>
"""


# ruff colours its output whenever it thinks a terminal is watching, and the
# escape codes break every `path:line:col: RULE message` split downstream — the
# symptom is a clean "0 findings" from a repository with 14 062 of them. The same
# trap already caught `templates.py`; stripping belongs next to every ruff call,
# not in one of them.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def collect(repo: Path, rules: str) -> list[str]:
    for cmd in (["ruff"], ["uvx", "ruff"]):
        try:
            proc = subprocess.run(
                [*cmd, "check", "fedot/", f"--select={rules}",
                 "--output-format=concise", "--no-fix", "--isolated"],
                cwd=repo, capture_output=True, text=True, check=False)
        except (OSError, FileNotFoundError):
            continue
        if proc.stdout.strip():
            return [_ANSI.sub("", ln) for ln in proc.stdout.strip().splitlines()]
    return []


def parse(line: str) -> tuple[str, int, str, str] | None:
    try:
        location, rest = line.split(": ", 1)
        file_rel, line_no, _ = location.split(":")[:3]
        rule, message = rest.split(" ", 1)
        return file_rel, int(line_no), rule, message
    except (ValueError, IndexError):
        return None


def is_cosmetic(rule: str) -> bool:
    letters = re.match(r"[A-Z]+", rule)
    return bool(letters) and letters.group(0) in COSMETIC_PREFIXES


def snippet(repo: Path, file_rel: str, line: int, radius: int = 6) -> str:
    try:
        lines = (repo / file_rel).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    lo, hi = max(0, line - radius - 1), min(len(lines), line + radius)
    return "\n".join(f"{n + 1:>5}| {lines[n]}" for n in range(lo, hi))


WHOLE_FILE_SYS = TRIAGE_SYS + """
You are also shown the whole file, not only the lines around each warning. Use
it. A linter reports shapes it has rules for; you can see things it has no rule
for — a guard whose condition can never be true, an exception constructed and
never raised, a parameter accepted and then ignored, a branch commented out.

After the numbered verdicts, you may add extra lines for anything YOU noticed
that the warnings do not cover:

EXTRA <line>: <one clause: what breaks, and under what conditions>

Only what you could demonstrate by running the code. Style, naming and missing
docstrings do not qualify, and neither does anything you are merely unsure
about. Add nothing if there is nothing.
"""


def triage_file(inference, repo: Path, file_rel: str,
                findings: list[tuple[int, str, str]],
                whole_file: bool = False) -> tuple[dict[int, tuple[str, str]],
                                                   list[tuple[int, str]]]:
    """One call per file: the model sees every warning in it, with context.

    With `whole_file` it also sees the entire source and may report defects the
    linter has no rule for. That is the expensive half of the bet — it is worth
    building on only if those extra findings survive execution, so this mode
    exists as a flag first and a default never.
    """
    if whole_file:
        # The source is shown in full below, so a twelve-line snippet per
        # warning would repeat it dozens of times over. One line each instead —
        # which is what makes it affordable to show every warning rather than
        # the pre-filtered few.
        blocks = [f"{i}. {rule} at line {line}"
                  + (" [linter calls this stylistic]" if is_cosmetic(rule) else "")
                  + f" — {message}"
                  for i, (line, rule, message) in enumerate(findings, 1)]
        prompt = (f"File `{file_rel}`, {len(findings)} warning(s).\n\n"
                  + "\n".join(blocks)
                  + f"\n\nReply with exactly {len(findings)} lines.")
    else:
        blocks = []
        for i, (line, rule, message) in enumerate(findings, 1):
            blocks.append(f"### {i}. {rule} at line {line}\n{message}\n"
                          f"```python\n{snippet(repo, file_rel, line)}\n```")
        prompt = (f"File `{file_rel}`, {len(findings)} warning(s).\n\n"
                  + "\n\n".join(blocks)
                  + f"\n\nReply with exactly {len(findings)} lines.")
    if whole_file:
        try:
            text = (repo / file_rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        numbered = "\n".join(f"{n:>5}| {ln}"
                              for n, ln in enumerate(text.splitlines(), 1))
        prompt = (f"File `{file_rel}` in full:\n```python\n{numbered[:60000]}\n```\n\n"
                  + prompt
                  + "\n\nThen add EXTRA lines for anything the warnings miss.")
    raw = inference.query(
        [{"role": "system", "content": WHOLE_FILE_SYS if whole_file else TRIAGE_SYS},
         {"role": "user", "content": prompt}]) or ""
    out: dict[int, tuple[str, str]] = {}
    extra: list[tuple[int, str]] = []
    for ln in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)]\s*(live|inert|unclear)\b\s*\|?\s*(.*)",
                     ln, re.I)
        if m:
            out[int(m.group(1))] = (m.group(2).lower(), m.group(3).strip()[:200])
            continue
        m = re.match(r"\s*EXTRA\s+(\d+)\s*[:.)]\s*(.+)", ln, re.I)
        if m:
            extra.append((int(m.group(1)), m.group(2).strip()[:250]))
    return out, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--rules", default="ALL")
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--whole-file", action="store_true",
                    help="show the model the entire source and let it report "
                         "defects the linter has no rule for")
    ap.add_argument("--drop-cosmetic", action="store_true",
                    help="pre-filter stylistic rules instead of letting "
                         "the reader judge them (for measuring the effect)")
    ap.add_argument("--workers", type=int, default=8,
                    help="files read concurrently; the work is network wait")
    ap.add_argument("--out", type=Path, default=Path("LINT_TRIAGE.md"))
    args = ap.parse_args()

    repo = Path(os.environ.get("FEDOTLLM_REPO_PATH", "")).resolve()
    if not repo.is_dir():
        print("FEDOTLLM_REPO_PATH not set", file=sys.stderr)
        return 2

    raw = collect(repo, args.rules)
    parsed = [p for p in (parse(ln) for ln in raw) if p]
    cosmetic = [p for p in parsed if is_cosmetic(p[2])]
    # Dropping by rule family was my judgement, not the agent's, and it threw
    # away 12 647 of 14 060 findings on the strength of the rule code alone. A
    # rule describes a shape; whether it matters is a question about this code,
    # and that is exactly what the reader is for. So everything is shown, with
    # the stylistic ones marked as such, and the reader decides. In whole-file
    # mode this is nearly free: the source dominates the prompt, the extra
    # warnings are one line each.
    if args.whole_file and not args.drop_cosmetic:
        candidates = parsed
        print(f"ruff: {len(parsed)} findings · none dropped · "
              f"{len(cosmetic)} marked stylistic for the reader to judge", flush=True)
    else:
        candidates = [p for p in parsed if not is_cosmetic(p[2])]
        print(f"ruff: {len(parsed)} findings · {len(cosmetic)} dropped as cosmetic "
              f"· {len(candidates)} to triage", flush=True)

    by_file: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for file_rel, line, rule, message in candidates:
        by_file[file_rel].append((line, rule, message))
    files = sorted(by_file)
    if args.limit_files:
        files = files[: args.limit_files]

    config = load_config(presets="fedotllm:openrouter",
                         overrides=[f"llm.model_name={args.model}"])
    inference = AIInference(config=config.llm)

    # One file per request, and the requests are almost entirely spent waiting on
    # the network — so they go concurrently. Sequentially the 207 files of FEDOT
    # take about five hours; at eight in flight it is closer to half an hour, for
    # the same money.
    def work(item):
        n, file_rel = item
        try:
            return n, file_rel, triage_file(inference, repo, file_rel,
                                            by_file[file_rel],
                                            whole_file=args.whole_file), None
        except Exception as exc:
            return n, file_rel, ({}, []), exc

    rows, verdicts = [], Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      for n, file_rel, result, exc in pool.map(work, enumerate(files, 1)):
        findings = by_file[file_rel]
        answers, extra = result
        done += 1
        if exc is not None:
            print(f"  [{done}/{len(files)}] {file_rel}: {type(exc).__name__}", flush=True)
        for line, why in extra:
            verdicts["extra"] += 1
            rows.append({"file": file_rel, "line": line, "rule": "EXTRA",
                         "message": "found by reading, not by the linter",
                         "verdict": "extra", "why": why})
        for i, (line, rule, message) in enumerate(findings, 1):
            verdict, why = answers.get(i, ("unclear", "no answer from the model"))
            verdicts[verdict] += 1
            rows.append({"file": file_rel, "line": line, "rule": rule,
                         "message": message, "verdict": verdict, "why": why})
        print(f"  [{done}/{len(files)}] {file_rel}: "
              + (f"extra={len(extra)} " if extra else "")
              + ", ".join(f"{k}={v}" for k, v in
                          Counter(answers.get(i, ("unclear", ""))[0]
                                  for i in range(1, len(findings) + 1)).items()),
              flush=True)

    live = [r for r in rows if r["verdict"] == "live"]
    lines = [
        "# Which of the linter's questions about FEDOT are worth asking",
        "",
        f"`ruff --select={args.rules}` reports **{len(parsed)}** findings. "
        f"**{len(cosmetic)}** are dropped without a model — quoting style, line "
        "length, docstring formatting, type annotations. Those cannot bite by the "
        "definition of the rule, and they are the bulk of the output.",
        "",
        f"The remaining **{len(candidates)}** went to `{args.model}`, one call per "
        f"file, {len(files)} files.",
        "",
        "| verdict | count | meaning |",
        "|---|---|---|",
        f"| live | {verdicts['live']} | the model can name what breaks |",
        f"| inert | {verdicts['inert']} | the shape is there, the consequence is not |",
        f"| unclear | {verdicts['unclear']} | the snippet does not settle it |",
        f"| extra | {verdicts['extra']} | noticed by reading, no linter rule covers it |",
        "",
        "A `live` verdict is a lead, not a finding: it earns a full pipeline run, "
        "where the reproduce gate demands a test that fails on the untouched tree. "
        "Nothing here is claimed to be a defect.",
        "",
        "## Live",
        "",
    ]
    for r in sorted(live, key=lambda r: (r["rule"], r["file"])):
        lines.append(f"- `{r['file']}:{r['line']}` **{r['rule']}** — {r['why']}")
    lines += ["", "## By rule", "", "| rule | live | inert | unclear |", "|---|---|---|---|"]
    per_rule: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        per_rule[r["rule"]][r["verdict"]] += 1
    for rule in sorted(per_rule, key=lambda k: -per_rule[k]["live"]):
        c = per_rule[rule]
        lines.append(f"| {rule} | {c['live']} | {c['inert']} | {c['unclear']} |")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    args.out.with_suffix(".usage.json").write_text(
        json.dumps(inference.usage, indent=2),
        encoding="utf-8",
    )
    print(f"\nlive={verdicts['live']} inert={verdicts['inert']} "
          f"unclear={verdicts['unclear']} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

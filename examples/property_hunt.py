#!/usr/bin/env python3
"""The agent invents the property; the machine tries to falsify it.

Everywhere else the agent is handed a defect somebody already decided was a
defect. Here it gets a module and nothing else, and has to say what ought to be
true. That is the postural difference between "can it fix" and "can it find",
and the honest number this script produces is: of the properties it invented,
how many survived falsification.

    uv run python examples/property_hunt.py --operations ransac_lin_reg lagged
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fedotllm.agents.evolve.properties import propose, screen  # noqa: E402
from fedotllm.configs.loader import load_config  # noqa: E402
from fedotllm.llm import AIInference  # noqa: E402


def implementation_source(repo: Path, py: str, operation: str) -> tuple[str, str]:
    """Source of the class that implements `operation`, found by asking FEDOT."""
    # Resolving the implementation class through FEDOT's private machinery is
    # brittle and version-specific; fitting the operation and reading the type of
    # what comes out is not. Reuse the routine the invariant scan already has.
    # The agent sees this source and nothing else -- no findings, no lint, no
    # hint that anything is wrong with it.
    from fedotllm.agents.evolve import invariants

    script = Path(invariants.__file__)
    probe = (
        "import json, sys, warnings; warnings.filterwarnings('ignore')\n"
        f"sys.path.insert(0, {str(script.parent)!r})\n"
        "import invariants as I\n"
        f"op = {operation!r}\n"
        "kind, meta = I._meta(op)\n"
        "if meta is None:\n"
        "    print('SRC' + json.dumps(['', ''])); raise SystemExit\n"
        "import inspect\n"
        "for tk in [t.value for t in meta.task_type]:\n"
        "    for dk in I.DATA_FOR_TASK.get(tk, []):\n"
        "        for chain in I.candidate_chains(op, tk):\n"
        "            try:\n"
        "                p = I.build_and_fit(chain, None, I.make_data(dk))\n"
        "            except Exception:\n"
        "                continue\n"
        "            n = next((x for x in p.nodes if x.name == op), None)\n"
        "            f = getattr(n, 'fitted_operation', None)\n"
        "            if f is None:\n"
        "                continue\n"
        "            try:\n"
        "                src = inspect.getsource(type(f))\n"
        "                path = inspect.getfile(type(f))\n"
        "            except Exception:\n"
        "                continue\n"
        "            print('SRC' + json.dumps([path, src])); raise SystemExit\n"
        "print('SRC' + json.dumps(['', '']))\n"
    )
    proc = subprocess.run([py, "-c", probe], cwd=repo, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("SRC"):
            try:
                path, src = json.loads(line[3:])
                return path, src
            except json.JSONDecodeError:
                break
    return "", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operations", nargs="+", required=True)
    ap.add_argument("--attempts", type=int, default=2,
                    help="properties to invent per operation")
    ap.add_argument("--presets", default="fedotllm:openrouter")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("PROPERTY_HUNT.md"))
    args = ap.parse_args()

    repo = Path(os.environ.get("FEDOTLLM_REPO_PATH", "")).resolve()
    py = os.environ.get("FEDOTLLM_REPO_PYTHON", "")
    if not repo.is_dir() or not py:
        print("FEDOTLLM_REPO_PATH / FEDOTLLM_REPO_PYTHON not set", file=sys.stderr)
        return 2

    overrides = list(args.override)
    if not any(o.startswith("llm.model_name=") for o in overrides):
        overrides.append("llm.model_name=openai/gpt-4o-mini")
    model = next(o.split("=", 1)[1] for o in overrides if o.startswith("llm.model_name="))
    print(f"=== model: {model} · operations: {len(args.operations)} "
          f"· attempts each: {args.attempts} ===", flush=True)

    config = load_config(presets=args.presets, overrides=overrides)
    inference = AIInference(config=config.llm)
    workdir = Path(os.environ.get("FEDOTLLM_PROPERTY_WORKDIR", "/tmp/fedotllm_properties"))

    rows = []
    for operation in args.operations:
        path, source = implementation_source(repo, py, operation)
        if not source:
            print(f"[{operation}] no implementation source — skipped", flush=True)
            continue
        tried: list[str] = []
        for attempt in range(1, args.attempts + 1):
            try:
                prop = propose(inference, repo, operation, source, tried)
            except Exception as exc:
                print(f"[{operation}] attempt {attempt}: unusable reply "
                      f"({type(exc).__name__}: {exc})", flush=True)
                rows.append({"operation": operation, "name": "-", "statement": "",
                             "verdict": "unparsable", "detail": str(exc)[:200],
                             "controls": {}})
                continue
            tried.append(prop.statement or prop.name)
            prop = screen(prop, operation, repo, py, workdir)
            print(f"[{operation}] {prop.name}: {prop.verdict} — {prop.statement}",
                  flush=True)
            rows.append({"operation": operation, "name": prop.name,
                         "statement": prop.statement, "verdict": prop.verdict,
                         "detail": prop.output, "controls": prop.controls,
                         "code": prop.code})

    verdicts = Counter(r["verdict"] for r in rows)
    survived = verdicts.get("candidate", 0) + verdicts.get("shared", 0)
    lines = [
        "# Properties the agent invented, and what survived falsification",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')} · model `{model}`",
        "",
        f"**{survived} of {len(rows)} proposed properties survived.** A property "
        "survives only if it fails on the target operation, twice, and holds on "
        "at least one sibling operation that actually evaluated it — the machine standing in "
        "for the human reviewers that comparable work needs (Anthropic's "
        "property-based testing agent reports 56% of reviewed reports valid, 32% "
        "valid and reportable, filtered by three experts at ~1h per bug).",
        "",
        "| verdict | count | meaning |",
        "|---|---|---|",
        f"| candidate | {verdicts.get('candidate', 0)} | fails on target, holds on every sibling — a defect the agent found itself |",
        f"| shared | {verdicts.get('shared', 0)} | fails on the target and some siblings but holds on others — a defect across a family, not a misreading |",
        f"| holds | {verdicts.get('holds', 0)} | the property is true here — nothing to fix, and saying so is a valid outcome |",
        f"| too_broad | {verdicts.get('too_broad', 0)} | fails on every sibling that answered — the agent's misunderstanding, not a defect |",
        f"| unrunnable | {verdicts.get('unrunnable', 0)} | errored or could not be set up — proves nothing |",
        f"| flaky | {verdicts.get('flaky', 0)} | not reproducible |",
        f"| unchecked | {verdicts.get('unchecked', 0)} | no sibling could answer — unfalsified, so not claimed |",
        f"| unparsable | {verdicts.get('unparsable', 0)} | the reply did not follow the format |",
        "",
        "## Every proposal",
        "",
    ]
    for r in rows:
        lines += [f"### `{r['operation']}` — {r['name']} → **{r['verdict']}**", ""]
        if r.get("statement"):
            lines += [f"> {r['statement']}", ""]
        lines += [f"- target: {r.get('detail', '')}"]
        for k, v in (r.get("controls") or {}).items():
            lines.append(f"- control `{k}`: {v}")
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n=== survived: {survived}/{len(rows)} · " +
          ", ".join(f"{k}: {v}" for k, v in verdicts.most_common()) + " ===")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

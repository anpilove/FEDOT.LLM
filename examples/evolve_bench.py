#!/usr/bin/env python3
"""Measure EvolveAgent reliability: N honest runs, one summary.

Each run starts from a pristine checkout (git reset), goes through the full
scout → propose → gates loop, and is recorded. Prints success-rate, which files
the scout picked, and why runs failed — the honest number to report, and the
tool to use when tuning prompts/gates.

Env: FEDOTLLM_LLM_API_KEY, FEDOTLLM_REPO_PATH, FEDOTLLM_REPO_PYTHON

Example:
  uv run python examples/evolve_bench.py --runs 5 --override llm.model_name=openai/gpt-4o
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

from langchain_core.messages import HumanMessage

from fedotllm.agents.evolve import EvolveAgent
from fedotllm.agents.evolve.loop import VALUABLE_SEVERITIES
from fedotllm.configs.loader import load_config

DEFAULT_MESSAGE = (
    "Please evolve the aimclub/FEDOT library: find one small safe "
    "validation/error-handling improvement, patch it, and validate with pytest."
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a success rate.

    Agent benchmarks are statistically fragile at small n: "5/5" alone is not
    evidence of reliability (its 95% interval starts near 55%). Always report
    the interval alongside the point estimate.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((centre - spread) / d, (centre + spread) / d)


def reset_repo(repo: Path) -> None:
    subprocess.run(["git", "checkout", "--", "."], cwd=repo, capture_output=True, check=False)
    subprocess.run(["git", "clean", "-fdq"], cwd=repo, capture_output=True, check=False)


def failure_reason(state: dict) -> str:
    """Classify why a run did not go green (for the summary histogram)."""
    audit = state.get("audit_markdown", "") or ""
    if "REJECTED by the reproduce gate" in audit:
        return "no-repro (test passed on pristine)"
    if "cpu AutoML gate" in audit and "FAILED" in audit:
        return "broke FEDOT runtime (AutoML gate)"
    for marker, reason in (
        ("py_compile", "patch did not compile"),
        ("pytest", "targeted test still failing"),
    ):
        if f"`{marker}" in audit and "FAILED" in audit:
            return reason
    if not state.get("proposal", {}).get("old_code"):
        return "unusable proposal (parse/format)"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="EvolveAgent reliability benchmark")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--presets", default="fedotllm:openrouter")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--workspace", type=Path, default=Path("fedotllm-output-bench"))
    ap.add_argument("--message", default=DEFAULT_MESSAGE)
    args = ap.parse_args()

    repo = Path(os.environ.get("FEDOTLLM_REPO_PATH", "")).resolve()
    if not repo.is_dir():
        print("FEDOTLLM_REPO_PATH missing/invalid", file=sys.stderr)
        return 2
    if not os.environ.get("FEDOTLLM_LLM_API_KEY"):
        print("FEDOTLLM_LLM_API_KEY not set", file=sys.stderr)
        return 2

    overrides = list(args.override)
    if not any(o.startswith("llm.model_name=") for o in overrides):
        # The mini model localized proven defects but produced no valuable repair
        # in the invariant-first arm. Use the stronger coding model by default;
        # published mini results remain historical and must not be mixed with it.
        overrides.append("llm.model_name=openai/gpt-5.4-mini")
    model = next(o.split("=", 1)[1] for o in overrides if o.startswith("llm.model_name="))
    # Printed up front so an expensive run is obvious before it burns anything.
    print(f"=== model: {model} · runs: {args.runs} ===", flush=True)

    args.workspace.mkdir(parents=True, exist_ok=True)
    config = load_config(presets=args.presets, overrides=overrides)

    rows: list[dict] = []
    for i in range(1, args.runs + 1):
        print(f"\n=== run {i}/{args.runs} ===", flush=True)
        reset_repo(repo)
        run_ws = args.workspace / f"run{i}"
        run_ws.mkdir(parents=True, exist_ok=True)
        agent = EvolveAgent(config=config, workspace=str(run_ws.resolve()))
        graph = agent.create_graph()
        try:
            state = graph.invoke({"messages": [HumanMessage(content=args.message)]})
            success = bool(state.get("evolve_success"))
            row = {
                "run": i,
                "success": success,
                "pick": state.get("scout_pick", ""),
                "file": (state.get("proposal") or {}).get("file_path", ""),
                "problem": (state.get("proposal") or {}).get("problem", "")[:120],
                "severity": state.get("evolve_severity", 4),
                "abstained": bool(state.get("evolve_abstained")),
                "evidence": state.get("evolve_evidence", ""),
                "reason": ("abstained: " + state.get("evolve_abstain_reason", ""))[:160]
                if state.get("evolve_abstained")
                else ("" if success else failure_reason(state)),
            }
        except Exception as exc:  # keep the bench running through provider hiccups
            row = {"run": i, "success": False, "pick": "", "file": "", "severity": 4,
                   "abstained": False, "evidence": "",
                   "problem": "", "reason": f"exception: {type(exc).__name__}: {exc}"[:160]}
        row.update(agent.inference.usage)
        rows.append(row)
        print(
            f"[run {i}] success={row['success']} pick={row['pick']} "
            f"cost=${row['cost_usd']:.4f} reason={row['reason']}",
            flush=True,
        )

    reset_repo(repo)
    ok = sum(r["success"] for r in rows)
    rate = ok / len(rows) if rows else 0.0
    picks = Counter(r["pick"] for r in rows if r["pick"])
    reasons = Counter(r["reason"] for r in rows if not r["success"])
    lo, hi = wilson_ci(ok, len(rows))
    # Diversity: a repo-evolution agent that always edits the same file is not
    # evolving the repository. Report unique targets, not just success rate.
    unique_picks = len(picks)
    unique_files = len({r["file"] for r in rows if r["file"]})
    top_share = (picks.most_common(1)[0][1] / len(rows)) if picks else 0.0
    # Yield of genuinely valuable fixes. Success rate alone is gameable: the agent
    # chooses its own task, so it can score 100% by always picking the easiest
    # class. Measured: 30/30 success with zero real defects repaired.
    abstained = sum(bool(r.get("abstained")) for r in rows)
    # An abstention is not a failure: the run looked, found no evidence and said
    # so. Only runs that actually attempted a patch belong in the success rate.
    attempted = [r for r in rows if not r.get("abstained")]
    ok_attempted = sum(r["success"] for r in attempted)
    a_lo, a_hi = wilson_ci(ok_attempted, len(attempted))
    sev_counts = Counter(r.get("severity", 4) for r in rows if r["success"])
    valuable = sum(sev_counts[s] for s in VALUABLE_SEVERITIES)
    v_lo, v_hi = wilson_ci(valuable, len(rows))
    total_prompt = sum(r["prompt_tokens"] for r in rows)
    total_completion = sum(r["completion_tokens"] for r in rows)
    total_cached = sum(r["cached_tokens"] for r in rows)
    total_cost = sum(r["cost_usd"] for r in rows)

    lines = [
        "# EvolveAgent reliability bench",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: `openrouter/{model}` · Repo: `{repo}`",
        "",
        f"**Valuable fixes: {valuable}/{len(rows)}** (severity 1–2) · 95% CI (Wilson): "
        f"**[{v_lo:.1%}, {v_hi:.1%}]**",
        "",
        f"**Abstentions: {abstained}/{len(rows)} ({abstained / max(len(rows), 1):.0%})** — "
        "runs that found no evidence of a defect and said so instead of producing "
        "something. In evidence-first mode zero abstentions is expected while the "
        "precomputed pool still contains proven defects; it is suspicious only when "
        "the agent is allowed to choose files without external evidence.",
        "",
        "This is the headline number, not the success rate below. The agent chooses "
        "its own task, so success rate is gameable by always picking the easiest "
        "class — and that is what happens: a configuration once scored 30/30 while "
        "repairing nothing but error messages.",
        "",
        "Severity of accepted patches: "
        + " · ".join(f"class {s}: {sev_counts.get(s, 0)}" for s in (1, 2, 3, 4)),
        "",
        f"**Success rate: {ok}/{len(rows)} ({rate:.0%})** · 95% CI (Wilson): "
        f"**[{lo:.1%}, {hi:.1%}]** — machinery works, says nothing about value",
        "",
        f"Of the runs that attempted a patch: **{ok_attempted}/{len(attempted)}** · "
        f"95% CI (Wilson): [{a_lo:.1%}, {a_hi:.1%}]",
        "",
        "**Diversity:** "
        f"unique scout picks: **{unique_picks}** · unique patched files: **{unique_files}** · "
        f"most frequent pick covers **{top_share:.0%}** of runs "
        "(lower is better — an agent that always edits one file is not evolving the repo)",
        "",
        f"**LLM usage:** {sum(r['requests'] for r in rows)} requests · "
        f"{total_prompt:,} input tokens · {total_completion:,} output tokens · "
        f"{total_cached:,} cached tokens · **${total_cost:.4f}**",
        "",
        "## Runs",
        "",
        "| # | success | class | cost | scout pick | problem | failure reason |",
        "|---|---------|-------|------|-----------|---------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {'✅' if r['success'] else '❌'} | {r['severity']} | "
            f"${r['cost_usd']:.4f} | `{r['pick'] or '—'}` | "
            f"{r['problem'] or '—'} | {r['reason'] or '—'} |"
        )
    lines += ["", "## Scout picks", ""]
    lines += [f"- `{k}` × {v}" for k, v in picks.most_common()] or ["- none"]
    lines += ["", "## Failure reasons", ""]
    lines += [f"- {k} × {v}" for k, v in reasons.most_common()] or ["- none"]

    report = args.workspace / "BENCH.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.workspace / "bench.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n=== VALUABLE fixes (severity 1-2): {valuable}/{len(rows)} "
          f"· 95% CI [{v_lo:.1%}, {v_hi:.1%}] ===")
    print("=== severity of accepted patches: "
          + ", ".join(f"class {s}: {sev_counts.get(s, 0)}" for s in (1, 2, 3, 4)) + " ===")
    print(f"=== success rate: {ok}/{len(rows)} ({rate:.0%}) · 95% CI [{lo:.1%}, {hi:.1%}] ===")
    print(f"=== diversity: {unique_picks} unique picks, {unique_files} unique files, "
          f"top pick = {top_share:.0%} of runs ===")
    print(f"=== LLM usage: {total_prompt:,} input + {total_completion:,} output "
          f"({total_cached:,} cached) · ${total_cost:.4f} ===")
    for k, v in reasons.most_common():
        print(f"  fail: {k} × {v}")
    print(f"report: {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

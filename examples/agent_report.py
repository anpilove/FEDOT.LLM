#!/usr/bin/env python3
"""How good is the agent, computed from the archive rather than remembered.

Every number here comes from `bench.json` files and the audits beside them, so
the report cannot drift from what was actually run. Three things it insists on:

* **Wilson intervals, always.** With five or twelve runs per arm the point
  estimate is nearly uninformative on its own; "2/3" and "0/5" have overlapping
  intervals and must not be reported as a ranking.
* **Valuable fixes, not success rate.** The agent picks its own task, so success
  rate is gameable by always picking the easiest class — measured: one arm
  scored 30/30 while repairing nothing but error messages.
* **A column for patches that passed every gate and were wrong anyway.** Two of
  those are known by name in this project, and a report without that column
  would overstate the agent by exactly those two.

    uv run python examples/agent_report.py --out AGENT_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Patches that cleared every gate and still made things worse, found by reading
# the diff. Listed by (workspace, run) so the report can subtract them instead
# of quietly counting them as value.
KNOWN_BAD = {
    ("fedotllm-output-bench-K", 5): (
        "dropped `iterations` from the arguments handed to CatBoost — the error "
        "disappears and the caller's value is silently discarded"),
    ("fedotllm-output-bench-L", 1): (
        "commented out `self.params.update(degree=degree)` in "
        "`PolyfitImplementation` — an out-of-range degree now reaches np.polyfit"),
}

# Patches confirmed correct by reading the diff.
KNOWN_GOOD = {
    ("fedotllm-output-bench-M", 1): "lda: switches to a solver that supports "
                                    "shrinkage instead of discarding it",
    ("fedotllm-output-bench-M", 3): "dask_pca: adapts an estimator copy and "
                                    "leaves the declared params intact",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((centre - spread) / d, (centre + spread) / d)


@dataclass
class Arm:
    workspace: str
    model: str = "?"
    rows: list[dict] = field(default_factory=list)
    reviewed_good: int = 0
    reviewed_bad: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def ok(self) -> int:
        return sum(bool(r.get("success")) for r in self.rows)

    @property
    def valuable(self) -> int:
        return sum(1 for r in self.rows
                   if r.get("success") and r.get("severity", 4) in (1, 2))

    @property
    def cost(self) -> float:
        # The bench writes `cost_usd`; older rows have neither. Reading the
        # wrong key silently reports $0.00 for every arm, which is how this was
        # found.
        return sum(float(r.get("cost_usd") or r.get("cost") or 0) for r in self.rows)

    @property
    def requests(self) -> int:
        return sum(int(r.get("requests") or 0) for r in self.rows)

    @property
    def unique_targets(self) -> int:
        return len({r.get("pick") for r in self.rows if r.get("pick")})


def load_arms(root: Path) -> list[Arm]:
    arms: list[Arm] = []
    for blob in sorted(root.glob("fedotllm-output-*/bench.json")):
        arm = Arm(workspace=blob.parent.name)
        try:
            arm.rows = json.loads(blob.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        report = blob.parent / "BENCH.md"
        if report.is_file():
            m = re.search(r"Model: `openrouter/([^`]+)`", report.read_text(encoding="utf-8"))
            if m:
                arm.model = m.group(1)
        for row in arm.rows:
            key = (arm.workspace, row.get("run"))
            if key in KNOWN_BAD:
                arm.reviewed_bad += 1
                arm.notes.append(f"run {row.get('run')}: {KNOWN_BAD[key]}")
            elif key in KNOWN_GOOD:
                arm.reviewed_good += 1
                arm.notes.append(f"run {row.get('run')}: {KNOWN_GOOD[key]}")
        arms.append(arm)
    # An arm whose every run died on a configuration error measures the harness,
    # not the model, and must not sit in a table about model quality.
    kept = []
    for a in arms:
        broken = sum(1 for r in a.rows
                     if "exception:" in (r.get("reason") or "")
                     and "UnsupportedParams" in (r.get("reason") or ""))
        if a.n and broken == a.n:
            continue
        kept.append(a)
    return [a for a in kept if a.n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("AGENT_REPORT.md"))
    args = ap.parse_args()

    arms = load_arms(args.root)
    if not arms:
        print("no bench.json found", file=sys.stderr)  # noqa: F821
        return 1

    total_runs = sum(a.n for a in arms)
    total_cost = sum(a.cost for a in arms)
    reviewed_bad = sum(a.reviewed_bad for a in arms)
    reviewed_good = sum(a.reviewed_good for a in arms)

    lines = [
        "# How good is the agent, in numbers from the archive",
        "",
        f"{len(arms)} arms · {total_runs} runs · "
        f"${total_cost:.2f} of model spend recorded in the run rows.",
        "",
        "Everything below is computed from `bench.json`, not from notes. Where a "
        "patch was read by hand, that is stated as such and counted separately — "
        "passing every gate is not the same as being right, and this project has "
        "two patches by name that cleared all six gates and made the library worse.",
        "",
        "## Arms",
        "",
        "| arm | model | runs | passed gates | valuable (class 1–2) | 95% CI | unique targets | LLM calls | $ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(arms, key=lambda x: x.workspace):
        lo, hi = wilson(a.valuable, a.n)
        lines.append(
            f"| `{a.workspace.replace('fedotllm-output-', '')}` | `{a.model}` | {a.n} | "
            f"{a.ok} | **{a.valuable}** | [{lo:.0%}, {hi:.0%}] | {a.unique_targets} | "
            f"{a.requests or '—'} | {a.cost:.3f} |")

    lines += [
        "",
        "**Read the intervals, not the fractions.** At five runs an arm scoring "
        "2/5 and one scoring 0/5 have overlapping intervals; this table ranks "
        "nothing on its own.",
        "",
        "## Patches read by hand",
        "",
        f"{reviewed_good} confirmed correct · **{reviewed_bad} passed every gate "
        "and were wrong**.",
        "",
    ]
    for a in sorted(arms, key=lambda x: x.workspace):
        for note in a.notes:
            lines.append(f"- `{a.workspace.replace('fedotllm-output-', '')}` {note}")

    lines += [
        "",
        "The two wrong ones are why the gate list grew to six: a tuning gate (an "
        "operation whose parameter was reported unusable must end up tunable) and "
        "a behaviour fingerprint (the predictions on the untouched checkout must "
        "survive the patch). Both were added after the fact, so arms measured "
        "before them are not comparable with arms measured after.",
        "",
        "## What the agent does without a model",
        "",
        "| stage | needs an LLM? |",
        "|---|---|",
        "| finding the defect (runtime invariant scan, 79 operations) | no |",
        "| proving it (test built from the finding, verified to fail) | no |",
        "| choosing which file to work on | yes, but constrained to files with evidence |",
        "| writing the patch | yes |",
        "| accepting or rejecting the patch (6 gates) | no |",
        "",
        "So the honest claim is narrow: the LLM writes patches and picks among "
        "pre-proven targets. Detection and acceptance are deterministic.",
        "",
        "## Failure reasons across all runs",
        "",
    ]
    reasons: dict[str, int] = {}
    for a in arms:
        for row in a.rows:
            if not row.get("success"):
                reasons[row.get("reason") or "unknown"] = \
                    reasons.get(row.get("reason") or "unknown", 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:12]:
        lines.append(f"- {reason} × {count}")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"arms: {len(arms)} · runs: {total_runs} · cost recorded: ${total_cost:.2f}")
    print(f"hand-reviewed: {reviewed_good} correct, {reviewed_bad} wrong-but-green")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())

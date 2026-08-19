#!/usr/bin/env python3
"""Turn a raw invariant scan into a ranked, honest list of findings.

Ranking follows one question only: *would a maintainer have found this by
running a linter?* Everything that would is worth nothing here. What survives
is behaviour — a value silently replaced, a cache that stops working, an
internal error of a third-party library reaching the user, a hang.

    uv run python examples/invariant_report.py scan.json --repo <fedot checkout>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def mutation_sites(repo: Path) -> dict[str, list[str]]:
    """`self.params.update(...)` calls, file -> lines.

    The scan says which operations actually rewrite a declared value while
    running; this says where in the source it happens. Neither half is useful
    alone: the grep finds 40 call sites and cannot tell which ones ever fire,
    the scan proves which fire and cannot point at a line.
    """
    sites: dict[str, list[str]] = defaultdict(list)
    for path in repo.glob("fedot/**/*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if "self.params.update(" in line:
                rel = str(path.relative_to(repo))
                sites[rel].append(f"{rel}:{n}: {line.strip()}")
    return sites


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", type=Path)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("INVARIANT_FINDINGS.md"))
    args = ap.parse_args()

    results = json.loads(args.scan.read_text(encoding="utf-8"))
    sites = mutation_sites(args.repo)

    scanned = [r for r in results if not r.get("setup")]
    skipped = [r for r in results if r.get("setup")]

    rewritten: dict[str, list[dict]] = defaultdict(list)
    cache_broken: dict[str, list[dict]] = defaultdict(list)
    crashes: list[tuple[str, dict]] = []
    hangs: list[str] = []
    unobservable: dict[str, set[str]] = defaultdict(set)
    nondet: list[tuple[str, dict]] = []

    for r in results:
        op = r["operation"]
        for f in r.get("findings", []):
            kind = f["kind"]
            if kind == "declared_not_used":
                rewritten[op].append(f)
            elif kind == "cache_miss_after_fit":
                cache_broken[op].append(f)
            elif kind == "boundary_crash":
                crashes.append((op, f))
            elif kind == "hang":
                hangs.append(op)
            elif kind == "nondeterministic_replacement":
                nondet.append((op, f))
            elif kind == "not_observable":
                unobservable[op].add(f["param"])

    mm_defects = [(r["operation"], m) for r in results for m in r.get("metamorphic", [])
                  if m.get("severity") == "defect" and m.get("status") != "ok"]

    lines = [
        "# What the runtime scan found in FEDOT",
        "",
        f"Scanned: **{len(scanned)}** operations fitted successfully with their own "
        f"defaults; **{len(skipped)}** could not be set up (missing optional "
        "dependency or no working pipeline) and are excluded rather than counted "
        "as clean.",
        "",
        "Every value tried came from the operation's own declared sampling scope "
        "(`PipelineSearchSpace`), so nothing here is an input the library calls "
        "illegal.",
        "",
        "## 1. A declared hyperparameter is silently replaced during fit",
        "",
        f"**{len(rewritten)} operations.** The caller passes a value through the "
        "public API, the fitted object holds another one. `PipelineNode.parameters` "
        "reports the replacement too, so it is not merely internal.",
        "",
        "| operation | parameter | declared | in force after fit |",
        "|---|---|---|---|",
    ]
    for op in sorted(rewritten):
        seen = set()
        for f in rewritten[op]:
            key = (f["param"], repr(f["value"]))
            if key in seen:
                continue
            seen.add(key)
            observed = "; ".join(f"{v}" for v in (f.get("observed") or {}).values())
            lines.append(f"| `{op}` | `{f['param']}` | {f['value']!r} | {observed} |")

    lines += [
        "",
        "### Why this is not cosmetic",
        "",
        f"**{len(cache_broken)} operations lose the operations cache entirely.** "
        "`PipelineNode.descriptive_id` is the cache key and embeds the node's "
        "parameters *and its parents'*. A node that rewrites its own parameters "
        "during fit is stored under a key nobody will ever look up — and every "
        "node downstream of it misses too.",
        "",
    ]
    for op in sorted(cache_broken):
        f = cache_broken[op][0]
        lines.append(f"- `{op}`: pipeline `{' → '.join(f['chain'])}` — "
                     f"nothing reloaded, missed: {', '.join(f['missed_nodes'])}")

    lines += [
        "",
        "### Where in the source",
        "",
        "`self.params.update(...)` inside fit, found by grep — the scan is what "
        "says which of these actually fire on legal data.",
        "",
    ]
    total_sites = sum(len(v) for v in sites.values())
    lines.append(f"{total_sites} call sites in {len(sites)} files:")
    for rel in sorted(sites):
        lines.append(f"- `{rel}` — {len(sites[rel])}")

    lines += [
        "",
        "## 2. A value from the declared scope crashes or hangs",
        "",
        f"**{len(crashes)} crashes, {len(hangs)} hangs.**",
        "",
    ]
    if crashes:
        lines += ["| operation | parameter | value | error | dies outside FEDOT |",
                  "|---|---|---|---|---|"]
        for op, f in crashes:
            foreign = f.get("foreign")
            where = Path(foreign).name if foreign else "—"
            msg = re.sub(r"\s+", " ", f.get("error", ""))[:110]
            lines.append(f"| `{op}` | `{f['param']}` | {f['value']!r} | "
                         f"{f['error_type']}: {msg} | {where} |")
    else:
        lines.append("None.")
    for op in hangs:
        lines.append(f"- `{op}` — no result within the timeout on legal data")

    lines += [
        "",
        "## 2b. The replacement is not even reproducible",
        "",
        f"**{len(nondet)} cases.** The operation replaced the declared value, and "
        "fitting the same pipeline on the same data a second time produced a "
        "*different* replacement. Checked with the value that triggers the "
        "replacement rather than with the defaults, because the substituting "
        "branch does not run otherwise. Nothing is reseeded between the two fits: "
        "FEDOT seeds the global RNG only from `Fedot(seed=...)`, which defaults "
        "to `None`.",
        "",
    ]
    for op, f in nondet:
        lines.append(f"- `{op}`.`{f['param']}` = {f['value']!r} → "
                     f"{' then '.join(f['observed'])}")
    if not nondet:
        lines.append("None.")

    lines += [
        "",
        "## 3. Metamorphic properties",
        "",
        f"**{len(mm_defects)} violations at defect grade** (`repeat_fit` — the same "
        "call twice must give the same prediction).",
        "",
    ]
    for op, m in mm_defects:
        lines.append(f"- `{op}`: {m['property']} — {m.get('status')} "
                     f"(max delta {m.get('max_delta')})")
    if not mm_defects:
        lines.append("None. The other three properties (column permutation, constant "
                     "column, row order) are recorded as observations only: they fire "
                     "on `rf` because a forest draws `max_features` columns at random "
                     "per split, which is documented sklearn behaviour, not a defect.")

    lines += [
        "",
        "## 4. Not a finding",
        "",
        f"`not_observable` — {sum(len(v) for v in unobservable.values())} parameters "
        f"across {len(unobservable)} operations are accepted but appear nowhere on "
        "the fitted object. Most are renamed on the way through, so this is a lead, "
        "not a defect, and is listed here to keep it out of the counts above.",
        "",
    ]
    for op in sorted(unobservable):
        lines.append(f"- `{op}`: {', '.join(sorted(unobservable[op]))}")

    lines += ["", "## Operations that could not be scanned", ""]
    for r in skipped:
        lines.append(f"- `{r['operation']}` — {r['setup']}")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rewritten-parameter operations: {len(rewritten)}")
    print(f"cache-losing operations:        {len(cache_broken)}")
    print(f"crashes on declared values:     {len(crashes)}")
    print(f"hangs:                          {len(hangs)}")
    print(f"metamorphic defects:            {len(mm_defects)}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

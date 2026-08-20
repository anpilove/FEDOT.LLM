#!/usr/bin/env python3
"""Merge several reader passes into one list of suspicions.

Time is cheap and the reader is cheap; the expensive thing is missing a defect.
So the reader runs more than once over the same file and the passes are unioned
rather than intersected: recall here, precision at the next gate, where the
verdict is an exit code and not an opinion.

Where passes disagree the file is worth noting — one run saw something the
other did not, which is either a real find or a sign the file is hard to read.

    uv run python examples/union_leads.py pass1.json pass2.json --out leads.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Strongest first: a suspicion found by reading beats a linter finding called
# live, which beats anything the passes could not settle.
RANK = {"extra": 3, "live": 2, "unclear": 1, "inert": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("passes", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("LEADS.json"))
    ap.add_argument("--keep", default="extra,live")
    args = ap.parse_args()

    wanted = {v.strip() for v in args.keep.split(",")}
    seen: dict[tuple[str, int], dict] = {}
    found_in: dict[tuple[str, int], set[int]] = defaultdict(set)

    for n, path in enumerate(args.passes):
        rows = json.loads(path.read_text(encoding="utf-8"))
        kept = 0
        for row in rows:
            if row.get("verdict") not in wanted or not row.get("why"):
                continue
            key = (row["file"], int(row["line"]))
            found_in[key].add(n)
            kept += 1
            best = seen.get(key)
            if best is None or RANK.get(row["verdict"], 0) > RANK.get(best["verdict"], 0):
                seen[key] = dict(row)
        print(f"{path.name}: {kept} suspicions worth keeping")

    for key, row in seen.items():
        row["passes"] = sorted(found_in[key])
        row["agreed"] = len(found_in[key]) == len(args.passes)

    merged = sorted(seen.values(), key=lambda r: (r["file"], r["line"]))
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    agreed = sum(1 for r in merged if r["agreed"])
    print(f"\nunion: {len(merged)} suspicions · {agreed} found by every pass · "
          f"{len(merged) - agreed} by only some")
    print(Counter(r["verdict"] for r in merged))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

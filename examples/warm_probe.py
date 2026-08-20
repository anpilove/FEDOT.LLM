#!/usr/bin/env python3
"""Establish the clean-tree probe baseline before any model is paid.

The acceptance gates compare a patched tree against a probe of the clean one,
and that baseline is keyed by the checkout's HEAD. When it belongs to a
different checkout the gate does not fail loudly: each round reports "no
pristine baseline" and rejects the patch for a reason that has nothing to do
with the patch. Seven repair rounds were lost to exactly that before anyone
looked at the reason string closely.

So it is checked once, here, and a missing baseline stops the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fedotllm.agents.evolve.probe import (  # noqa: E402
    read_pristine_findings,
    run_probe_cached,
)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: warm_probe.py <repo> [python]", file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).resolve()
    py = sys.argv[2] if len(sys.argv) > 2 else sys.executable
    run_probe_cached(repo, py)
    if read_pristine_findings(repo) is None:
        print("probe baseline unavailable for this checkout — the gates would "
              "reject every patch blindly", file=sys.stderr)
        return 1
    print("probe baseline ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render the controller/worker reliability report from measured evidence.

The renderer does not run the registry and does not invent a KEEP.
"""

from __future__ import annotations


def render_reliability_report(evidence: dict) -> str:
    required = (
        "doctor",
        "worker_invalid",
        "checkout_contract",
        "return_best_policy",
        "dev_ne_final",
        "budget",
        "recovery",
        "smoke_aa",
    )
    missing = [key for key in required if key not in evidence]
    if missing:
        raise ValueError("reliability report missing evidence: " + ", ".join(missing))
    aa = evidence["smoke_aa"]
    lines = [
        "# EvolveAgent controller reliability",
        "",
        "Scope: controller and worker reliability on `research/evolve-agent`.",
        "No model change, no FEDOT automerge, no quality KEEP.",
        "",
        "## Doctor",
        "",
        str(evidence["doctor"]),
        "",
        "## Worker invalid",
        "",
        str(evidence["worker_invalid"]),
        "",
        "## Checkout contract",
        "",
        str(evidence["checkout_contract"]),
        "",
        "## return_best policy",
        "",
        str(evidence["return_best_policy"]),
        "",
        "## DEV != FINAL, budget, recovery",
        "",
        str(evidence["dev_ne_final"]),
        "",
        str(evidence["budget"]),
        "",
        str(evidence["recovery"]),
        "",
        "## Smoke A/A",
        "",
        f"purpose: {aa.get('purpose')}",
        f"quality_claim: {aa.get('quality_claim')}",
        f"values: {aa.get('values')}",
        f"abs_delta: {aa.get('abs_delta')}",
        f"identical: {aa.get('identical')}",
        "",
        "## Not claimed",
        "",
        "Full OpenML registry KEEP was not run. Identical smoke repeats are",
        "measurement noise only.",
        "",
    ]
    return "\n".join(lines)

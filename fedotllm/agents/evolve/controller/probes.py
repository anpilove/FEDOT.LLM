"""Controller-owned causal probes for stock and patched FEDOT."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.types import SnippetResult
from fedotllm.agents.evolve.storage.hypothesis import behavior_probe_fingerprint

SnippetRunner = Callable[[Path, str], SnippetResult]
_OBSERVATION_RE = re.compile(r"^EVOLVE_OBSERVATION=(.*)$", re.MULTILINE)


def compare_with_prior_probe(source, experiment, code, *, prior_probe, compare_fn):
    """Retry an inconclusive diagnostic once on a prior probe from this branch.

    Rerun both source versions; never reuse an old observation or excuse an
    actual patched-only runtime error. Metric and test gates remain mandatory.
    """
    result = compare_fn(source, experiment, code)
    if result.get("status") not in {"missing", "invalid", "no_change"} or not prior_probe:
        return result
    origin, prior_code = prior_probe
    if not prior_code or behavior_probe_fingerprint(prior_code) == behavior_probe_fingerprint(code):
        return result
    retried = compare_fn(source, experiment, prior_code)
    if retried.get("status") == "changed":
        return {**retried, "code": prior_code, "reused_probe_from_candidate": origin,
                "submitted_probe_result": result}
    return {**result, "prior_probe_attempt": retried, "prior_probe_candidate": origin}

def compare_behavior_probe(
    source: Path,
    experiment: Path,
    code: str,
    *,
    run_snippet_fn: SnippetRunner,
) -> dict:
    """Run one model-supplied causal probe on stock and patched FEDOT.

    This is a cheap semantic diagnostic, not a quality judgment. An unchanged
    toy observation is not a veto: the snippet may be insensitive. A patched
    crash is a technical failure. Invalid probes stay unjudged.
    """

    probe = (code or "").strip()
    if not probe:
        return {
            "status": "missing",
            "code": "",
            "probe_valid": False,
            "hypothesis_result": "inconclusive_invalid_probe",
        }
    stock = run_snippet_fn(source, probe)
    patched = run_snippet_fn(experiment, probe)

    def row(result) -> dict:
        observations = _OBSERVATION_RE.findall(result.stdout or "")
        return {
            "status": result.status,
            "exit_code": result.exit_code,
            "observation": observations[0].strip()[:2_000]
            if len(observations) == 1
            else None,
            "observation_count": len(observations),
            "stdout_tail": (result.stdout or "")[-2_000:],
            "stderr_tail": (result.stderr or "")[-2_000:],
        }

    stock_row = row(stock)
    patched_row = row(patched)
    if stock.status == "runtime_error" and "AssertionError" in (stock.stderr or ""):
        status = "changed" if (
            patched.status == "ok" and patched_row["observation"] is not None
        ) else "invalid"
    elif stock.status != "ok":
        status = "invalid"
    elif patched.status != "ok":
        status = "patched_error"
    elif stock_row["observation"] is None or patched_row["observation"] is None:
        status = "invalid"
    elif stock_row["observation"] == patched_row["observation"]:
        status = "no_change"
    else:
        # Two successful executions with different deterministic observations.
        status = "changed"
    interpretation = {
        "invalid": "inconclusive_invalid_probe",
        "missing": "inconclusive_invalid_probe",
        "patched_error": "patch_regression",
        "no_change": "toy_probe_did_not_see_effect",
        "changed": "mechanism_supported_pending_independent_gates",
    }[status]
    return {
        "status": status,
        "code": probe,
        "probe_valid": status in {"no_change", "changed"},
        "hypothesis_result": interpretation,
        "stock": stock_row,
        "patched": patched_row,
    }


def _behavior_probe_blocks_candidate(result: dict) -> bool:
    """Cheap technical filter only. Toy no_change is not a quality veto.

    Block crashes and probes that never judged the patch. A valid probe whose
    observation did not move means the cheap snippet was insensitive; the
    candidate may still go to hour-long Fedot.
    """

    return result.get("status") not in {"changed", "no_change"}

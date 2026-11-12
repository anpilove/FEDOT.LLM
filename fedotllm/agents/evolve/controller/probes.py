"""Controller-owned causal probes for stock and patched FEDOT."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.types import Decision, SnippetResult
from fedotllm.agents.evolve.storage.hypothesis import behavior_probe_fingerprint

SnippetRunner = Callable[[Path, str], SnippetResult]
_OBSERVATION_RE = re.compile(r"^EVOLVE_OBSERVATION=(.*)$", re.MULTILINE)
_PREFLIGHT_RE = re.compile(r"^EVOLVE_PREFLIGHT (\{.*\})$", re.MULTILINE)


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

def _preflight_import_error(stdout: str) -> str | None:
    matches = _PREFLIGHT_RE.findall(stdout or "")
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    error = payload.get("import_error") if isinstance(payload, dict) else None
    return error if isinstance(error, str) and error else None


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
    shared_import = _preflight_import_error(stock.stdout or "")
    if (
        shared_import
        and shared_import == _preflight_import_error(patched.stdout or "")
        and stock_row["observation"] is None
        and patched_row["observation"] is None
    ):
        status = "infrastructure"
    elif stock.status == "runtime_error" and "AssertionError" in (stock.stderr or ""):
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
        "infrastructure": "environment_import_failed",
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
        "import_error": (shared_import or "") if status == "infrastructure" else "",
        "stock": stock_row,
        "patched": patched_row,
    }


def infrastructure_probe_decision(result: dict, *, experiment_id: str = "") -> Decision | None:
    """Stop the campaign when both checkouts fail before the probe runs."""

    if result.get("status") != "infrastructure":
        return None
    error = str(result.get("import_error") or "").strip()
    reason = "behavior_probe_infrastructure"
    if error:
        reason = f"{reason}: {error}"
    return Decision(
        False,
        reason[:500],
        None,
        experiment_id=experiment_id,
        infrastructure_error=True,
        stage="infrastructure",
    )


def _single_column_stack(code: str) -> bool:
    """True when ``column_stack([...])`` has one argument at the top level.

    A dtype such as ``datetime64[ns]`` contains brackets, so a character class
    cannot see the comma that would mark a second column.
    """

    marker = "column_stack("
    start = code.find(marker)
    if start < 0:
        return False
    index = start + len(marker)
    while index < len(code) and code[index].isspace():
        index += 1
    if index >= len(code) or code[index] != "[":
        return False
    depth = 0
    parens = 0
    saw_comma = False
    for char in code[index:]:
        if char == "(":
            parens += 1
        elif char == ")":
            if parens == 0:
                break
            parens -= 1
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0 and parens == 0:
                return not saw_comma
        elif char == "," and depth == 1 and parens == 0:
            saw_comma = True
    return False


def whole_array_dtype_branch(probe_code: str, patch_text: str) -> bool:
    """A one-column datetime probe cannot see a whole-array dtype branch.

    ``np.column_stack`` of a datetime column and a float column is object
    dtype, so ``issubdtype(..., datetime64)`` is false and the old millisecond
    conversion still runs.
    """

    code = probe_code or ""
    patch = patch_text or ""
    if "datetime64" not in code or "datetime64" not in patch or "issubdtype" not in patch:
        return False
    return _single_column_stack(code)


def narrow_datetime_revision(behavior_probe: dict | None, patch_text: str) -> str:
    """Feedback for a gain that only exists on a pure datetime64 matrix."""

    probe = behavior_probe or {}
    if str(probe.get("status") or "") != "changed":
        return ""
    if not whole_array_dtype_branch(str(probe.get("code") or ""), patch_text):
        return ""
    stock = str((probe.get("stock") or {}).get("observation") or "")
    patched = str((probe.get("patched") or {}).get("observation") or "")
    return (
        f"The probe moved from {stock} to {patched} on a single datetime64 column. "
        "Real tables put a numeric column beside the dates. np.issubdtype on the "
        "whole array is then false, so np_datetime_to_numeric still emits epoch "
        "milliseconds and the gain disappears. Change that helper column by column, "
        "and probe a two-column table of datetime64 values plus a float column."
    )


_WHOLE_CONVERSION_SCALE = re.compile(
    r"np_datetime_to_numeric\([^)]*\)\s*/\s*\d"
)


def divides_whole_conversion(patch_text: str) -> bool:
    """True when the patch divides the helper's entire return value."""

    return _WHOLE_CONVERSION_SCALE.search(patch_text or "") is not None


def whole_conversion_scale_revision(
    behavior_probe: dict | None, patch_text: str
) -> str:
    """Feedback for a scale applied to numeric columns and the target too.

    The 06:07 Moscow revision wrote
    ``np_datetime_to_numeric(...) / 86400000`` for features and target. The
    probe never imported fedot, so both sides printed the same sklearn AUC,
    and three API safety tests failed.
    """

    if not divides_whole_conversion(patch_text):
        return ""
    code = str((behavior_probe or {}).get("code") or "")
    calls_fedot = "import fedot" in code or "from fedot" in code
    detached = ""
    if not calls_fedot:
        detached = (
            " The probe never imported fedot, so its observation does not "
            "measure the patch."
        )
    return (
        "Dividing the whole return of np_datetime_to_numeric scales numeric "
        "columns and the target. Leave those values unchanged. Edit the helper "
        "so only datetime columns become day counts, and probe that helper on "
        "a two-column table of datetime64 values plus a float."
        + detached
    )


def _behavior_probe_blocks_candidate(result: dict) -> bool:
    """Cheap technical filter only. Toy no_change is not a quality veto.

    Block crashes and probes that never judged the patch. A valid probe whose
    observation did not move means the cheap snippet was insensitive; the
    candidate may still go to hour-long Fedot.
    """

    return result.get("status") not in {"changed", "no_change"}

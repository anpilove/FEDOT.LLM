"""Checkout and worker-runtime contract.

An experiment checkout is a copy. Snippets and the worker may write only
there and under the run workspace. The frozen FEDOT source stays untouched.
Smoke A/A repeats one probe twice and records the numeric delta. That delta
is measurement noise, not a quality claim.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fedotllm.agents.evolve.execution.checkout import (
    MARKER,
    create_experiment_checkout,
    discard_experiment_checkout,
)
from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.execution.process import clean_subprocess_env, fedot_python, run_worker

AA_MARKER = "EVOLVE_AA "


def checkout_runtime_contract(
    *,
    source: Path,
    checkout: Path,
    workspace: Path,
    runtime_paths: dict | None = None,
    fedot_file: str | None = None,
) -> dict:
    source = source.resolve()
    checkout = checkout.resolve()
    workspace = workspace.resolve()
    issues: list[str] = []
    if source == checkout or source in checkout.parents or checkout in source.parents:
        issues.append("checkout_not_disjoint")
    if source == workspace or source in workspace.parents or workspace in source.parents:
        issues.append("workspace_not_disjoint")
    if not (checkout / MARKER).is_file():
        issues.append("checkout_marker_missing")
    package = checkout / "fedot" / "__init__.py"
    if not package.is_file():
        issues.append("checkout_package_missing")
    if fedot_file:
        resolved = Path(fedot_file)
        if not resolved.is_absolute():
            issues.append("fedot_file_not_absolute")
        else:
            resolved = resolved.resolve()
            allowed = checkout / "fedot"
            if resolved != allowed and allowed not in resolved.parents:
                issues.append("fedot_file_outside_checkout")
    if runtime_paths is not None:
        for key in ("temp_dir", "golem_data_dir"):
            raw = runtime_paths.get(key)
            if not isinstance(raw, str) or not raw:
                issues.append(f"{key}_missing")
                continue
            path = Path(raw)
            if not path.is_absolute():
                issues.append(f"{key}_not_absolute")
                continue
            resolved = path.resolve()
            if not resolved.is_dir():
                issues.append(f"{key}_not_directory")
            if workspace not in resolved.parents and resolved != workspace:
                issues.append(f"{key}_outside_workspace")
            if source == resolved or source in resolved.parents or resolved in source.parents:
                issues.append(f"{key}_inside_source")
    return {
        "ok": not issues,
        "issues": issues,
        "source": str(source),
        "checkout": str(checkout),
        "workspace": str(workspace),
    }


def _probe_once(source: Path, workspace: Path, label: str) -> dict:
    from fedotllm.agents.evolve.evaluation._fedot_quality_worker import configure_worker_runtime

    checkout = create_experiment_checkout(
        source, workspace, run_id="smoke-aa", candidate_id=label,
    )
    previous_tmpdir = os.environ.get("TMPDIR")
    previous_tempdir = tempfile.tempdir
    try:
        out = workspace / "aa" / f"{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        runtime_paths = configure_worker_runtime(out)
        code = (
            "import json\n"
            "from pathlib import Path\n"
            "import fedot\n"
            "fedot_file = Path(fedot.__file__).resolve()\n"
            "value = fedot_file.read_bytes().__len__()\n"
            f"print({AA_MARKER!r} + json.dumps({{'value': value, 'fedot_file': str(fedot_file)}}))\n"
        )
        proc = run_worker(
            [fedot_python(checkout), "-c", code],
            cwd=checkout,
            env=clean_subprocess_env(checkout, repo_root=repo_root()),
            timeout=30,
        )
        parsed = _parse_aa(proc.stdout)
        contract = checkout_runtime_contract(
            source=source,
            checkout=checkout,
            workspace=workspace,
            runtime_paths=runtime_paths,
            fedot_file=None if parsed is None else parsed.get("fedot_file"),
        )
        return {
            "label": label,
            "returncode": proc.returncode,
            "value": None if parsed is None else parsed.get("value"),
            "fedot_file": None if parsed is None else parsed.get("fedot_file"),
            "contract": contract,
            "detail": "" if proc.returncode == 0 else (proc.stderr or proc.stdout or "")[-300:],
        }
    finally:
        if previous_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = previous_tmpdir
        tempfile.tempdir = previous_tempdir
        discard_experiment_checkout(checkout, workspace=workspace, source=source)


def _parse_aa(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith(AA_MARKER):
            try:
                payload = json.loads(line[len(AA_MARKER):])
            except json.JSONDecodeError:
                return None
            if isinstance(payload, dict) and isinstance(payload.get("value"), int):
                return payload
    return None


def measure_smoke_aa(source: Path, workspace: Path) -> dict:
    """Repeat one checkout probe. ``abs_delta`` is noise, not model quality."""

    repeats = [_probe_once(source, workspace, label) for label in ("a1", "a2")]
    values = [row["value"] if isinstance(row["value"], int) else None for row in repeats]
    comparable = all(isinstance(value, int) for value in values)
    abs_delta = abs(values[0] - values[1]) if comparable else None
    identical = bool(
        comparable
        and abs_delta == 0
        and all(row["returncode"] == 0 and row["contract"]["ok"] for row in repeats)
    )
    return {
        "purpose": "smoke_aa_measurement_noise",
        "quality_claim": False,
        "repeats": repeats,
        "values": values,
        "abs_delta": abs_delta,
        "identical": identical,
    }

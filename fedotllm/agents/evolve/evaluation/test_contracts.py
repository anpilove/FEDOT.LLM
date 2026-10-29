"""Controller-owned corrections to demonstrably invalid frozen test assumptions.

Applied symmetrically to stock and candidate in disposable test copies. Neither
the immutable checkout nor metric execution uses these copies. Changes to the
upstream test disable the correction until it is reviewed again.
"""
from __future__ import annotations

import ast
import hashlib
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout, discard_experiment_checkout,
)

TEST_PATH = "test/unit/pipelines/test_pipeline.py"
TEST_NAME = "test_pipeline_with_custom_params_for_model"
ORIGINAL_AST_HASH = "67dcbc877cddce7874761e1a9c043a967cc8097e1dcc9aca9049e1a4b22ebfcf"
OLD_ASSERTION = "    assert not np.array_equal(custom_params_prediction, default_params_prediction)"
NEW_ASSERTIONS = """    actual_params = pipeline.root_node.fitted_operation.model.get_params()
    assert {key: actual_params[key] for key in custom_params} == custom_params
    # Distinct settings can legitimately produce identical training predictions.
    assert custom_params_prediction.shape == default_params_prediction.shape
    assert custom_params_prediction.shape[0] == len(data.idx)
    assert np.isfinite(custom_params_prediction).all()
    assert np.isfinite(default_params_prediction).all()"""


def repaired_contract_source(source: str) -> str:
    """Replace only the exact audited test; preserve unknown upstream changes."""
    try:
        node = next((node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == TEST_NAME), None)
    except SyntaxError:
        return source
    if node is None:
        return source
    fingerprint = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if fingerprint != ORIGINAL_AST_HASH:
        return source
    lines = source.splitlines(keepends=True)
    start, end = node.lineno - 1, node.end_lineno
    body = "".join(lines[start:end])
    if body.count(OLD_ASSERTION) != 1:
        return source
    return "".join(lines[:start]) + body.replace(OLD_ASSERTION, NEW_ASSERTIONS) + "".join(lines[end:])


def _needs_multimodal_fixture(checkout: Path) -> bool:
    archive = checkout / "examples/data/multimodal.tar.gz"
    directory = checkout / "examples/data/multimodal"
    return archive.is_file() and (not directory.exists() or directory.is_dir() and not any(directory.iterdir()))


def _prepare_multimodal_fixture(copied: Path) -> None:
    """Hydrate the frozen archive only in an owned, disposable test checkout."""
    if not (copied / ".evolve-agent-checkout.json").is_file():
        raise ValueError("test fixtures require an owned experiment")
    destination = (copied / "examples/data/multimodal").resolve()
    parent = destination.parent
    with tarfile.open(copied / "examples/data/multimodal.tar.gz") as archive:
        for member in archive.getmembers():
            target = (parent / member.name).resolve()
            if (not target.is_relative_to(destination) or member.issym() or member.islnk()
                    or not (member.isfile() or member.isdir())):
                raise ValueError("unsafe multimodal test archive member")
        archive.extractall(parent, filter="data")


def run_with_test_repairs(checkout: Path, runner):
    path = checkout / TEST_PATH
    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    repaired = repaired_contract_source(original)
    fixture_needed = _needs_multimodal_fixture(checkout)
    if repaired == original and not fixture_needed:
        return runner(checkout)
    with TemporaryDirectory(prefix="evolve-test-contract-") as temporary:
        workspace = Path(temporary)
        copied = create_experiment_checkout(checkout, workspace, run_id="tests", candidate_id="contract")
        try:
            if repaired != original:
                (copied / TEST_PATH).write_text(repaired, encoding="utf-8")
            if fixture_needed:
                _prepare_multimodal_fixture(copied)
            result = runner(copied)
            if repaired != original:
                result.output += f"\nController test correction: {TEST_PATH}::{TEST_NAME}; explicit estimator parameters checked.\n"
            if fixture_needed:
                result.output += "\nController test setup: frozen multimodal archive extracted in disposable copy; no assertions changed.\n"
            return result
        finally:
            discard_experiment_checkout(copied, workspace=workspace, source=checkout)

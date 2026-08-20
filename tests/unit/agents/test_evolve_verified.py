"""The verifier's output must reach the fixer, and reach it as a proof.

Until this source existed the pipeline stopped one step short: a defect could be
reproduced through the public API and still had no way to a patch. These tests
pin the two properties that make it usable — only reproduced defects count, and
the test that anchors the patch is built by code from the script that already
failed, never by the model.
"""

import json

import pytest

from fedotllm.agents.evolve.fixtures import as_pytest
from fedotllm.agents.evolve.loop import (
    classify_accepted_severity,
    verified_defect_for,
    verified_defects,
    verified_defects_section,
)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    (tmp_path / "fedot" / "api").mkdir(parents=True)
    (tmp_path / "fedot" / "api" / "main.py").write_text("x = 1\n")

    def write(rows):
        path = tmp_path / "verified.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        monkeypatch.setenv("FEDOTLLM_VERIFIED", str(path))
        return tmp_path

    return write


def row(**over):
    base = {"file": "fedot/api/main.py", "line": 360, "status": "confirmed",
            "why": "load() then forecast() raises", "got": "AttributeError",
            "detail": "AttributeError: 'NoneType' object has no attribute 'task'",
            "script": "from fedot.api.main import Fedot\nFedot(problem='classification').forecast()"}
    return {**base, **over}


def test_only_reproduced_defects_count(checkout):
    """`internal only` means no caller can walk the path — not a defect."""
    repo = checkout([row(), row(line=397, status="internal only"),
                     row(line=400, status="refuted")])
    assert [d["line"] for d in verified_defects(repo)] == [360]


def test_a_row_without_its_script_is_not_evidence(checkout):
    """The script is the evidence. Without it there is nothing to anchor to."""
    repo = checkout([row(script="")])
    assert verified_defects(repo) == []


def test_defect_in_a_file_that_is_gone_is_dropped(checkout):
    repo = checkout([row(file="fedot/api/vanished.py")])
    assert verified_defects(repo) == []


def test_proof_is_built_from_the_script_not_by_the_model(checkout):
    repo = checkout([row()])
    item = verified_defect_for(repo, "fedot/api/main.py")
    assert item is not None
    assert item["test_name"].startswith("test_verified_")
    # every line of the script survives into the test body
    for line in item["script"].splitlines():
        assert line in item["test_code"]


def test_the_proof_is_a_test_that_runs(checkout):
    """A proof that does not parse cannot fail on the untouched tree."""
    repo = checkout([row()])
    code = verified_defect_for(repo, "fedot/api/main.py")["test_code"]
    compile(code, "proof.py", "exec")


def test_section_is_empty_without_findings(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDOTLLM_VERIFIED", str(tmp_path / "absent.json"))
    assert verified_defects_section(tmp_path) == ""


def test_section_names_the_route_and_the_failure(checkout):
    repo = checkout([row()])
    text = verified_defects_section(repo)
    assert "fedot/api/main.py:360" in text
    assert "AttributeError" in text


def test_as_pytest_keeps_indentation_legal():
    code = as_pytest("if True:\n    x = 1\n", "test_x", "why")
    compile(code, "proof.py", "exec")


def test_low_level_failure_cannot_be_replaced_by_any_other_exception():
    code = as_pytest("raise KeyError('bad')", "test_x", "why", "KeyError")
    assert "except (ValueError, TypeError) as exc:" in code
    assert "unexpected replacement failure" in code
    compile(code, "proof.py", "exec")


def test_validation_failure_cannot_be_replaced_by_an_unrelated_exception():
    code = as_pytest("raise TypeError('bad')", "test_x", "why", "TypeError")
    assert "unexpected replacement failure" in code
    namespace = {}
    exec(compile(code, "proof.py", "exec"), namespace)
    with pytest.raises(AssertionError, match="unexpected replacement failure"):
        replacement = code.replace("raise TypeError('bad')", "raise RuntimeError('bad')")
        replacement_namespace = {}
        exec(compile(replacement, "proof.py", "exec"), replacement_namespace)
        replacement_namespace["test_x"]()


def test_public_crash_is_classified_as_an_ordinary_defect():
    severity, name = classify_accepted_severity(
        None,
        None,
        [],
        "the model calls this cosmetic",
        verified=row(),
    )
    assert severity == 2
    assert name

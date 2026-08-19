"""What the runtime invariant checker guarantees without an LLM."""

import json
import os
from pathlib import Path

import pytest

from fedotllm.agents.evolve import invariants
from fedotllm.agents.evolve import loop


class TestCandidateValues:
    """Values come from the operation's own declared sampling scope, so a crash
    on one of them is the library rejecting a value it declared legal itself."""

    def test_continuous_scope_yields_both_ends_and_the_middle(self):
        got = invariants.candidate_values(
            {"type": "continuous", "sampling-scope": [0.05, 1.0]})
        assert got == [0.05, 0.525, 1.0]

    def test_discrete_scope_stays_integral(self):
        got = invariants.candidate_values(
            {"type": "discrete", "sampling-scope": [2, 7]})
        assert got == [2, 4, 7]
        assert all(isinstance(v, int) for v in got)

    def test_categorical_scope_yields_every_choice(self):
        got = invariants.candidate_values(
            {"type": "categorical", "sampling-scope": [["linear", "square"]]})
        assert got == ["linear", "square"]

    def test_a_degenerate_scope_yields_nothing_rather_than_guessing(self):
        assert invariants.candidate_values({"type": "continuous", "sampling-scope": []}) == []


class TestComparison:
    def test_floats_compare_by_closeness_not_by_identity(self):
        assert invariants._equal(0.1 + 0.2, 0.3)

    def test_a_rewritten_value_is_not_equal(self):
        assert not invariants._equal(0.9608, 0.5)

    def test_mixed_types_do_not_explode(self):
        assert not invariants._equal("auto", 0.5)


class TestGeneratedTest:
    """The test is built from the finding, so it cannot fail for an unrelated
    reason -- the same reasoning as the lint templates, applied to behaviour."""

    def test_the_generated_test_is_valid_python_and_pins_the_declared_value(self):
        import ast

        built = invariants.build_defect_test(
            "ransac_lin_reg", ["ransac_lin_reg", "ridge"], "regression_outliers",
            "residual_threshold", 0.1)
        ast.parse(built["test_code"])
        assert built["test_name"] == "test_ransac_lin_reg_keeps_declared_residual_threshold"
        assert 'add_node("ransac_lin_reg", params={"residual_threshold": 0.1})' in built["test_code"]
        assert 'add_node("ridge")' in built["test_code"]

    def test_it_asserts_the_cache_consequence_as_well_as_the_value(self):
        built = invariants.build_defect_test(
            "ar", ["ar"], "ts", "lag_1", 101)
        assert "OperationsCache" in built["test_code"]
        assert built["test_name"] + "_cache" in built["test_code"]


class TestBridgeSelection:
    def test_only_declared_not_used_findings_become_defects(self, monkeypatch):
        monkeypatch.setattr(invariants, "implementation_file", lambda *a, **k: "fedot/x.py")
        results = [
            {"operation": "a", "chain": ["a"], "findings": [
                {"kind": "not_observable", "param": "p", "value": 1, "data": "regression"}]},
            {"operation": "b", "chain": ["b"], "findings": [
                {"kind": "boundary_crash", "param": "p", "value": 1, "data": "regression"}]},
        ]
        assert invariants.bridge(results) == []

    def test_a_finding_whose_file_cannot_be_resolved_is_dropped(self, monkeypatch):
        monkeypatch.setattr(invariants, "implementation_file", lambda *a, **k: None)
        results = [{"operation": "a", "chain": ["a"], "findings": [
            {"kind": "declared_not_used", "param": "p", "value": 1, "data": "regression"}]}]
        assert invariants.bridge(results) == []


class TestEvidenceInTheLoop:
    """The agent may only act on a file that already has measured evidence."""

    def _write(self, tmp_path: Path, monkeypatch, items):
        blob = tmp_path / "inv.json"
        blob.write_text(json.dumps(items), encoding="utf-8")
        monkeypatch.setenv("FEDOTLLM_INVARIANTS", str(blob))

    def test_findings_for_files_absent_from_the_checkout_are_ignored(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, [{"file": "fedot/gone.py", "operation": "x",
                                             "param": "p", "value": 1,
                                             "test_name": "t", "test_code": ""}])
        assert loop.invariant_defects(tmp_path) == []

    def test_a_finding_is_matched_to_its_file(self, tmp_path, monkeypatch):
        (tmp_path / "fedot").mkdir()
        (tmp_path / "fedot" / "here.py").write_text("", encoding="utf-8")
        item = {"file": "fedot/here.py", "operation": "x", "param": "p", "value": 1,
                "test_name": "t", "test_code": "assert True", "observed": {"impl": "2"}}
        self._write(tmp_path, monkeypatch, [item])
        assert loop.invariant_defect_for(tmp_path, "python", "fedot/here.py") == item
        assert loop.invariant_defect_for(tmp_path, "python", "fedot/other.py") is None

    def test_a_missing_or_broken_findings_file_is_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEDOTLLM_INVARIANTS", str(tmp_path / "nothing.json"))
        assert loop.invariant_defects(tmp_path) == []
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv("FEDOTLLM_INVARIANTS", str(bad))
        assert loop.invariant_defects(tmp_path) == []

    def test_the_task_text_names_the_cache_consequence(self, tmp_path):
        item = {"file": "f.py", "operation": "ransac_lin_reg", "param": "residual_threshold",
                "value": 0.1, "test_name": "t", "test_code": "x", "observed": {"impl": "3.3"}}
        text = loop.invariant_section(tmp_path, item)
        assert "descriptive_id" in text and "cache" in text
        assert "TEST_NAME: t" in text


class TestUnusableParameters:
    """A crash counts only when every value in the declared scope crashes.

    The looser rule would drag in `lgbmreg objective='poisson'` (crashes because
    the targets happen to be negative) and `stl_arima period=1` — both about the
    data, not about the parameter being unusable.
    """

    def _res(self, values):
        return {"findings": [{"kind": "boundary_crash", "param": "p", "value": v,
                              "data": "regression"} for v in values]}

    def test_a_parameter_that_fails_everywhere_is_reported(self):
        space = {"p": {"type": "discrete", "sampling-scope": [2, 7]}}
        got = invariants.unusable_parameters(self._res([2, 4, 7]), space)
        assert len(got) == 1

    def test_a_parameter_that_fails_at_one_value_is_not(self):
        space = {"p": {"type": "discrete", "sampling-scope": [2, 7]}}
        assert invariants.unusable_parameters(self._res([7]), space) == []

    def test_a_parameter_outside_the_declared_space_is_not_reported(self):
        assert invariants.unusable_parameters(self._res([2, 4, 7]), {}) == []


class TestLocatedSymbol:
    """The file alone was not enough: nine attempts, zero applied hunks."""

    def test_the_task_text_carries_the_exact_class(self, tmp_path, monkeypatch):
        src = tmp_path / "fedot" / "impl.py"
        src.parent.mkdir(parents=True)
        src.write_text(
            "class Other:\n    pass\n\n\nclass Wanted:\n    def fit(self):\n        return 1\n",
            encoding="utf-8")
        text = loop.invariant_symbol_section(
            tmp_path,
            {"file": "fedot/impl.py", "symbol": "Wanted"})
        assert "class Wanted" in text
        assert "class Other" not in text

    def test_no_symbol_means_no_section_rather_than_a_wrong_one(self, tmp_path, monkeypatch):
        assert loop.invariant_symbol_section(
            tmp_path, {"file": "fedot/impl.py", "symbol": ""}
        ) == ""
        assert loop.invariant_symbol_section(
            tmp_path, {"file": "fedot/nope.py", "symbol": "X"}
        ) == ""


class TestFingerprintStripping:
    """A golden value that does not hold on the untouched checkout fails every
    patch, the correct one included, so it is removed rather than shipped.

    Measured: the fingerprint for `ransac_lin_reg` came out as 221.230299 when
    computed inside the scanning process and 219.116627 in a standalone run,
    three times running, because RANSAC draws its subsets from the global numpy
    RNG and the value depends on whatever was fitted before it.
    """

    def _code(self):
        return invariants.build_defect_test(
            "cut", ["cut", "ridge"], "regression", "cut_part", 0.0)

    def test_the_other_two_tests_survive(self):
        name = "test_cut_keeps_declared_cut_part"
        code = ("def " + name + "():\n    pass\n\n\ndef " + name + "_cache():\n    pass\n"
                "\n\ndef " + name + "_behaviour_unchanged():\n    assert 1 == 2\n")
        stripped = invariants.strip_fingerprint(code, name)
        assert "_behaviour_unchanged" not in stripped
        assert "_cache" in stripped
        assert stripped.endswith("\n")

    def test_code_without_a_fingerprint_is_returned_unchanged(self):
        code = "def test_x():\n    pass\n"
        assert invariants.strip_fingerprint(code, "test_x") == code


class TestProposedProperties:
    """The agent invents the property; the machine tries to falsify it.

    Both guards here exist because the first live run produced exactly the
    failure they now catch.
    """

    def test_a_check_that_names_its_target_is_refused(self):
        from fedotllm.agents.evolve import properties

        code = ("def property_holds(operation: str) -> None:\n"
                "    if operation != 'ransac_lin_reg':\n"
                "        raise Skipped('only for ransac')\n")
        why = properties.reject_hardcoded_target(code, "ransac_lin_reg")
        assert why is not None and "controls cannot falsify" in why

    def test_a_generic_check_is_accepted(self):
        from fedotllm.agents.evolve import properties

        code = ("def property_holds(operation: str) -> None:\n"
                "    assert operation is not None\n")
        assert properties.reject_hardcoded_target(code, "ransac_lin_reg") is None

    def test_a_degenerate_reply_is_not_salvaged(self):
        from fedotllm.agents.evolve import properties

        raw = ("NAME: x\nPROPERTY: y\n<<<CHECK>>>\n"
               + "import numpy as np\n" * 40
               + "def property_holds(operation: str) -> None:\n    pass\n<<<END>>>")
        with pytest.raises(ValueError, match="degenerate"):
            properties.parse_property(raw)

    def test_a_check_without_the_required_signature_is_refused(self):
        from fedotllm.agents.evolve import properties

        raw = "NAME: x\nPROPERTY: y\n<<<CHECK>>>\ndef check(op):\n    pass\n<<<END>>>"
        with pytest.raises(ValueError):
            properties.parse_property(raw)

    def test_a_well_formed_reply_parses(self):
        from fedotllm.agents.evolve import properties

        raw = ("NAME: keeps_declared\nPROPERTY: a declared value survives fit\n"
               "<<<CHECK>>>\ndef property_holds(operation: str) -> None:\n"
               "    assert True\n<<<END>>>")
        prop = properties.parse_property(raw)
        assert prop.name == "keeps_declared"
        assert prop.statement == "a declared value survives fit"
        assert "property_holds" in prop.code


class TestFalsificationVerdicts:
    """Which combinations of control outcomes count as evidence.

    The `shared` verdict exists because of a measured false rejection: the agent
    proposed "window_size must stay inside the declared sampling scope after
    fit" for `lagged` — a real defect, found independently by hand — and
    stopping at the first violating control discarded it, because
    `sparse_lagged` breaks it in exactly the same way.
    """

    def _screen_with(self, monkeypatch, target_outcome, control_outcomes):
        from fedotllm.agents.evolve import properties

        calls = {"n": 0}

        def fake_evaluate(prop, operation, repo, py, workdir):
            if operation == "target":
                return target_outcome, ""
            return control_outcomes[operation], ""

        monkeypatch.setattr(properties, "evaluate", fake_evaluate)
        monkeypatch.setattr(properties, "pick_controls",
                            lambda *a, **k: list(control_outcomes))
        prop = properties.ProposedProperty(
            name="p", statement="", code="def property_holds(operation: str) -> None:\n    pass\n")
        return properties.screen(prop, "target", Path("/repo"), "python", Path("/tmp/x"))

    def test_holds_everywhere_else_is_a_candidate(self, monkeypatch):
        got = self._screen_with(monkeypatch, "violated", {"a": "holds", "b": "holds"})
        assert got.verdict == "candidate"

    def test_failing_on_every_answering_control_is_a_misunderstanding(self, monkeypatch):
        got = self._screen_with(monkeypatch, "violated", {"a": "violated", "b": "violated"})
        assert got.verdict == "too_broad"

    def test_failing_on_some_and_holding_on_others_is_a_shared_defect(self, monkeypatch):
        got = self._screen_with(monkeypatch, "violated", {"a": "violated", "b": "holds"})
        assert got.verdict == "shared"

    def test_no_control_can_answer_means_unchecked_not_proven(self, monkeypatch):
        got = self._screen_with(monkeypatch, "violated", {"a": "skipped", "b": "error"})
        assert got.verdict == "unchecked"

    def test_a_property_that_holds_on_the_target_is_not_a_finding(self, monkeypatch):
        got = self._screen_with(monkeypatch, "holds", {"a": "holds"})
        assert got.verdict == "holds"

    def test_a_property_that_does_not_reproduce_is_flaky(self, monkeypatch):
        from fedotllm.agents.evolve import properties

        seen = {"n": 0}

        def fake_evaluate(prop, operation, repo, py, workdir):
            seen["n"] += 1
            return ("violated", "") if seen["n"] == 1 else ("holds", "")

        monkeypatch.setattr(properties, "evaluate", fake_evaluate)
        prop = properties.ProposedProperty(
            name="p", statement="", code="def property_holds(operation: str) -> None:\n    pass\n")
        got = properties.screen(prop, "target", Path("/repo"), "python", Path("/tmp/x"))
        assert got.verdict == "flaky"


class TestInForce:
    """One value, not a mapping — the shape decides what mistakes are possible.

    The earlier helper returned {where: value}. The only property that survived
    screening in a live run reached for `.get(parameter_name)` on it, got None
    every time, compared None to the declared value and reported a defect that
    was not there (measured: pca n_components is 0.7 in both places). An API
    that invites one specific mistake will collect it.
    """

    def test_a_single_resolved_value_is_returned(self, monkeypatch):
        from fedotllm.agents.evolve import invariants, property_harness as properties

        monkeypatch.setattr(invariants, "observed_values",
                            lambda node, name: {"implementation.params": 0.7,
                                                "fitted.operation.get_params": 0.7})
        assert properties.in_force(object(), "n_components") == 0.7

    def test_absence_gives_the_default_rather_than_a_false_mismatch(self, monkeypatch):
        from fedotllm.agents.evolve import invariants, property_harness as properties

        monkeypatch.setattr(invariants, "observed_values", lambda node, name: {})
        assert properties.in_force(object(), "missing") is None
        assert properties.in_force(object(), "missing", 5) == 5

    def test_the_fitted_object_disagreeing_with_itself_is_reported(self, monkeypatch):
        from fedotllm.agents.evolve import invariants, property_harness as properties

        monkeypatch.setattr(invariants, "observed_values",
                            lambda node, name: {"implementation.params": 0.5,
                                                "fitted.operation.residual_threshold": 0.96})
        with pytest.raises(AssertionError, match="different places"):
            properties.in_force(object(), "residual_threshold")


class TestIsLegal:
    """Sample points are not the legal range, and the harness must not blur it.

    Measured: documented as "legal values", the sample list produced two false
    findings in one run — `min_samples=0.4` was reported as illegal because it
    is not one of the three sampled points, while sitting squarely inside the
    declared [0.1, 0.9]. Control screening cannot catch that: a harness that
    misleads, misleads every operation equally.
    """

    def _harness(self, monkeypatch, spec):
        from fedotllm.agents.evolve import property_harness

        monkeypatch.setattr(property_harness, "_space", lambda op: spec)
        return property_harness

    def test_a_value_inside_the_interval_is_legal_even_if_never_sampled(self, monkeypatch):
        h = self._harness(monkeypatch, {"min_samples": {"type": "continuous",
                                                        "sampling-scope": [0.1, 0.9]}})
        assert h.is_legal("op", "min_samples", 0.4) is True
        assert 0.4 not in h.declared_parameters("op")["min_samples"]

    def test_a_value_outside_the_interval_is_not(self, monkeypatch):
        h = self._harness(monkeypatch, {"p": {"type": "continuous",
                                              "sampling-scope": [0.1, 0.9]}})
        assert h.is_legal("op", "p", 1.5) is False

    def test_categorical_membership(self, monkeypatch):
        h = self._harness(monkeypatch, {"loss": {"type": "categorical",
                                                 "sampling-scope": [["ls", "lad"]]}})
        assert h.is_legal("op", "loss", "ls") is True
        assert h.is_legal("op", "loss", "huber") is False

    def test_an_undeclared_parameter_is_not_legal(self, monkeypatch):
        h = self._harness(monkeypatch, {})
        assert h.is_legal("op", "whatever", 1) is False


class TestSourceView:
    """Whole symbols, never a cut-off tail.

    Truncating at a character budget hid three of six implementation classes in
    `boostings_implementations.py` (16 966 characters against a 12 000 budget),
    and did it silently — the model had no way to know the file continued.
    """

    def _write(self, tmp_path):
        src = tmp_path / "fedot" / "m.py"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(
            '"""Module doc."""\n'
            "import os\n\n\n"
            "class Base:\n"
            '    """Base doc."""\n'
            "    def helper(self, x):\n"
            "        return x + 1\n\n\n"
            "class Target(Base):\n"
            '    """Target doc."""\n'
            "    def fit(self, data):\n"
            "        return self.helper(data)\n\n\n"
            "class Other:\n"
            '    """Other doc, not relevant."""\n'
            "    def far_away(self, y):\n"
            "        return y * 2\n",
            encoding="utf-8")
        return tmp_path

    def test_the_focus_symbol_is_shown_in_full(self, tmp_path):
        from fedotllm.agents.evolve.loop import source_view

        view = source_view(self._write(tmp_path), "fedot/m.py", "Target")
        assert "return self.helper(data)" in view

    def test_its_base_class_is_shown_in_full_too(self, tmp_path):
        from fedotllm.agents.evolve.loop import source_view

        # A fix for a subclass usually belongs to the parent; a model that
        # cannot see the parent guesses at it.
        view = source_view(self._write(tmp_path), "fedot/m.py", "Target")
        assert "return x + 1" in view

    def test_unrelated_bodies_are_elided_but_announced(self, tmp_path):
        from fedotllm.agents.evolve.loop import source_view

        view = source_view(self._write(tmp_path), "fedot/m.py", "Target")
        assert "return y * 2" not in view
        assert "class Other" in view and "far_away" in view
        assert "body elided" in view
        assert "Other doc, not relevant" in view, "the docstring must survive elision"

    def test_nothing_is_dropped_without_a_marker(self, tmp_path):
        from fedotllm.agents.evolve.loop import source_view

        view = source_view(self._write(tmp_path), "fedot/m.py", "Target")
        assert "Nothing is cut off" in view

    def test_a_file_that_does_not_parse_falls_back_instead_of_failing(self, tmp_path):
        from fedotllm.agents.evolve.loop import source_view

        bad = tmp_path / "fedot" / "broken.py"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("def oops(:\n", encoding="utf-8")
        assert source_view(tmp_path, "fedot/broken.py", "oops") == "def oops(:\n"


class TestDependencyContext:
    """What the focus symbol depends on, one hop, FEDOT only.

    The file view answers "what does this class look like". This answers "what
    happened to my value before it got here" — and for the whole family of
    parameter defects that answer lives in `OperationParameters`, in a different
    file. Without it the agent sees a dictionary appearing from nowhere and can
    only patch the symptom in front of it.
    """

    def _repo(self, tmp_path):
        (tmp_path / "fedot" / "core").mkdir(parents=True, exist_ok=True)
        (tmp_path / "fedot" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "fedot" / "core" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "fedot" / "core" / "params.py").write_text(
            "class Params:\n"
            '    """Holds parameters."""\n'
            "    def update(self, **kw):\n"
            "        self._d.update(kw)\n",
            encoding="utf-8")
        (tmp_path / "fedot" / "impl.py").write_text(
            "from sklearn.linear_model import Ridge\n"
            "from fedot.core.params import Params\n\n\n"
            "class Impl:\n"
            "    def fit(self, data):\n"
            "        self.params = Params()\n"
            "        return Ridge()\n",
            encoding="utf-8")
        return tmp_path

    def test_a_fedot_dependency_is_pulled_in(self, tmp_path):
        from fedotllm.agents.evolve.loop import dependency_context

        ctx = dependency_context(self._repo(tmp_path), "fedot/impl.py", "Impl")
        assert "`Params`" in ctx
        assert "self._d.update(kw)" in ctx, "a short definition is quoted whole"

    def test_third_party_dependencies_are_left_out(self, tmp_path):
        from fedotllm.agents.evolve.loop import dependency_context

        # The defect has to be fixed in FEDOT, so sklearn's source would spend
        # the budget on code the agent must not touch.
        ctx = dependency_context(self._repo(tmp_path), "fedot/impl.py", "Impl")
        assert "Ridge" not in ctx

    def test_an_unknown_focus_yields_nothing_rather_than_guessing(self, tmp_path):
        from fedotllm.agents.evolve.loop import dependency_context

        assert dependency_context(self._repo(tmp_path), "fedot/impl.py", "Nope") == ""
        assert dependency_context(self._repo(tmp_path), "fedot/impl.py", "") == ""

    def test_a_long_definition_keeps_its_method_names(self, tmp_path):
        from fedotllm.agents.evolve.loop import _outline_source

        long_class = ("class Big:\n"
                      '    """Doc."""\n'
                      "    def update(self, **kw):\n"
                      "        " + "x = 1\n        " * 60 + "\n")
        outline = _outline_source(long_class)
        assert "def update" in outline and "body elided" in outline


class TestUsageContext:
    """Where FEDOT itself uses the symbol — with the code, not just a count.

    The previous version listed "`X` is used by 1 other module(s)", which tells
    the agent nothing it can act on. For part of this defect family the fix
    belongs at the call site: that is where the parameters are assembled.
    """

    def _repo(self, tmp_path):
        (tmp_path / "fedot" / "core").mkdir(parents=True, exist_ok=True)
        (tmp_path / "fedot" / "impl.py").write_text(
            "class Impl:\n"
            "    def fit(self, data):\n"
            "        return 1\n",
            encoding="utf-8")
        (tmp_path / "fedot" / "core" / "strategy.py").write_text(
            "from fedot.impl import Impl\n\n\n"
            "class Strategy:\n"
            "    def build(self, params):\n"
            "        return Impl(**params)\n",
            encoding="utf-8")
        (tmp_path / "fedot" / "core" / "unrelated.py").write_text(
            "def elsewhere():\n    return 0\n", encoding="utf-8")
        return tmp_path

    def test_the_call_site_is_shown_with_its_code(self, tmp_path):
        from fedotllm.agents.evolve.loop import usage_context

        ctx = usage_context(self._repo(tmp_path), "fedot/impl.py", "Impl")
        assert "return Impl(**params)" in ctx
        assert "strategy.py" in ctx

    def test_the_defining_file_is_not_reported_as_a_user_of_itself(self, tmp_path):
        from fedotllm.agents.evolve.loop import usage_context

        ctx = usage_context(self._repo(tmp_path), "fedot/impl.py", "Impl")
        assert "fedot/impl.py:" not in ctx

    def test_files_that_never_mention_it_are_absent(self, tmp_path):
        from fedotllm.agents.evolve.loop import usage_context

        ctx = usage_context(self._repo(tmp_path), "fedot/impl.py", "Impl")
        assert "unrelated" not in ctx

    def test_an_unused_symbol_produces_nothing(self, tmp_path):
        from fedotllm.agents.evolve.loop import usage_context

        assert usage_context(self._repo(tmp_path), "fedot/impl.py", "Nope") == ""
        assert usage_context(self._repo(tmp_path), "fedot/impl.py", "") == ""

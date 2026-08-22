import json
import os
import threading
import types

import pytest
from langchain_core.messages import HumanMessage

from fedotllm.agents.evolve import agent as agent_module
from fedotllm.agents.evolve import benchmark as benchmark_module
from fedotllm.agents.evolve.agent import EvolveAgent
from fedotllm.agents.evolve.benchmark import (
    load_split,
    run_discovery_eval,
    run_semantic_static_eval,
    score_artifacts,
    threshold_failures,
)
from fedotllm.agents.evolve.leads import (
    defect_class,
    fixer_attempt_queue,
    fixer_candidates,
    is_run_success,
    is_style_noise,
    union_reader_passes,
    verifier_candidates,
    worth_fixing,
)
from fedotllm.agents.evolve.lint_scan import group_by_file, parse_lint
from fedotllm.agents.evolve.results import ReaderResult
from fedotllm.agents.evolve.loop import (
    CommandResult,
    EvolveResult,
    Proposal,
    classify_accepted_severity,
    queued_reader_defect,
    static_semantic_defects,
    static_semantic_section,
    value_gate_anchored,
)
from fedotllm.agents.evolve.pipeline_nodes import (
    fixer_stage,
    reader_stage,
    scan_stage,
    verifier_stage,
)
from fedotllm.agents.evolve.reader import (
    LEFTOVER_NUDGE,
    NONE_NUDGE,
    READER_SYSTEM,
    parse_reader_answer,
    read_one_file,
    run_peek,
)
from fedotllm.agents.evolve.semantic_scan import scan_repository as scan_semantics
from fedotllm.agents.evolve.verifier import (
    VERIFIER_SYSTEM,
    behaviour_matches_semantic_lead,
    check_public_route,
    fabricated,
    parse_fixable,
    parse_plausible,
    review_plausible,
    static_semantic_confirmation,
    traceback_mentions,
    verify_lead,
)
from fedotllm.configs.schema import (
    AppConfig,
    AutoMLConfig,
    EvolveConfig,
    LLMConfig,
    TemplatesConfig,
)


def config():
    return AppConfig(
        llm=LLMConfig(provider="test", model_name="fixer", api_key="test"),
        automl=AutoMLConfig(
            templates=TemplatesConfig(code="", train="", evaluate="", predict="")
        ),
    )


def test_reader_keeps_the_configured_completion_budget():
    cfg = config()
    cfg.llm.completion_params = {"max_tokens": 6144}
    agent = EvolveAgent(config=cfg, workspace=None)
    for stage in (EvolveAgent.READER, EvolveAgent.VERIFIER, EvolveAgent.FIXER):
        assert agent.inference[stage].completion_params["max_tokens"] == 6144


def test_stage_models_come_from_evolve_config():
    cfg = config()
    cfg.evolve = EvolveConfig(
        reader_model="cheap", verifier_model="proof", fixer_model="patch"
    )
    agent = EvolveAgent(config=cfg, workspace=None)
    assert agent.inference[EvolveAgent.READER].config.model_name == "cheap"
    assert agent.inference[EvolveAgent.VERIFIER].config.model_name == "proof"
    assert agent.inference[EvolveAgent.FIXER].config.model_name == "patch"


def test_unset_stage_models_use_llm_model_name():
    cfg = config()
    cfg.evolve = EvolveConfig(reader_model="cheap")
    agent = EvolveAgent(config=cfg, workspace=None)
    assert agent.inference[EvolveAgent.READER].config.model_name == "cheap"
    assert agent.inference[EvolveAgent.VERIFIER].config.model_name == "fixer"
    assert agent.inference[EvolveAgent.FIXER].config.model_name == "fixer"


def test_lint_seed_keeps_cosmetic_signals_as_metadata():
    row = parse_lint("fedot/a.py:3:5: E501 Line too long")
    assert row == {
        "file": "fedot/a.py",
        "line": 3,
        "column": 5,
        "rule": "E501",
        "message": "Line too long",
        "cosmetic": True,
    }


def test_reader_file_group_can_include_unlinted_source_files(tmp_path):
    (tmp_path / "fedot" / "a.py").parent.mkdir(parents=True)
    (tmp_path / "fedot" / "a.py").write_text("value = 1\n")
    (tmp_path / "fedot" / "b.py").write_text("value = 2\n")
    grouped = group_by_file(
        [{"file": "fedot/a.py", "line": 1, "rule": "B001", "message": "x"}],
        repo=tmp_path,
        include_all=True,
    )
    assert set(grouped) == {"fedot/a.py", "fedot/b.py"}
    assert grouped["fedot/b.py"] == []


def test_reader_file_group_can_be_limited_for_known_bug_eval(tmp_path):
    (tmp_path / "fedot" / "a.py").parent.mkdir(parents=True)
    (tmp_path / "fedot" / "a.py").write_text("value = 1\n")
    (tmp_path / "fedot" / "b.py").write_text("value = 2\n")
    grouped = group_by_file(
        [{"file": "fedot/a.py", "line": 1, "rule": "B001", "message": "x"}],
        repo=tmp_path,
        include_all=True,
        only_files={"fedot/b.py"},
    )
    assert grouped == {"fedot/b.py": []}


def test_semantic_scan_detects_high_value_runtime_shapes(tmp_path):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from fedot.missing.module import Tool\n\n"
        "def load(config, features, task):\n"
        "    parsed = eval(config['task'])\n"
        "    params = {'force ': True}\n"
        "    if task == TaskTypesEnum.ts_forecasting:\n"
        "        pass\n"
        "    del features['idx']\n"
        "    return parsed, params\n\n"
        "def split(shuffle, random_seed):\n"
        "    return KFold(n_splits=3, shuffle=shuffle, random_state=random_seed)\n\n"
        "class Op:\n"
        "    def __init__(self):\n"
        "        self.params = {}\n"
        "        self.balance_ratio = 0\n"
        "        self.random_seed = 0\n"
        "    def fix(self):\n"
        "        if not self.balance_ratio:\n"
        "            self.params.update(balance_ratio=1)\n"
        "        self.random_seed = self.random_seed or 42\n"
        "        self.params.update(degree=self.default_order)\n\n"
        "def fit(data, categorical_ids):\n"
        "    categorical_columns = data.iloc[:, categorical_ids].astype(str)\n"
        "    for column_id in categorical_ids:\n"
        "        return categorical_columns.iloc[:, column_id]\n\n"
        "def from_csv_time_series(task, is_predict):\n"
        "    if isinstance(task, str):\n"
        "        task = Task(TaskTypesEnum(task))\n"
        "    if is_predict:\n"
        "        return task.task_params.forecast_length\n\n"
        "class Selector:\n"
        "    def __init__(self):\n"
        "        self.params = {'n_features_to_select': 2}\n"
        "    def build(self):\n"
        "        rfe_params = {k: self.params.get(k) for k in ['n_features_to_select', 'step']}\n"
        "        return RFE(**rfe_params)\n\n"
        "class Model:\n"
        "    def __init__(self):\n"
        "        self.params = {'log': None}\n"
        "    def predict(self, input_data):\n"
        "        return run(logger=self.params['log'])\n\n"
        "def transform(features):\n"
        "    features.iloc[:, [0]]\n"
        "    return features[self.bool_ids]\n\n"
        "def train(train_data):\n"
        "    x_train = train_data.features\n"
        "    transformed_x_train, flag = check_input_array(x_train)\n"
        "    transformed_x_train = np.expand_dims(x_train, -1)\n"
        "    return transformed_x_train\n\n"
        "def h2o(data):\n"
        "    return data.target.reshape(1, -1)\n\n"
        "def custom(input_data):\n"
        "    input_data.target = input_data.target[:, 0]\n\n"
        "def surrogate(data):\n"
        "    data.target = prediction.predict\n\n"
        "def template(node):\n"
        "    params = node.parameters\n"
        "    params['dtype'] = params['dtype'].__name__\n\n"
        "def quantiles(pipeline, data):\n"
        "    return pipeline.fit(data)\n\n"
        "def nested(params):\n"
        "    out = []\n"
        "    for sample in params:\n"
        "        copied = deepcopy(sample)\n"
        "        for item in sample:\n"
        "            out.append(copied)\n"
        "    return out\n",
        encoding="utf-8",
    )

    rows = scan_semantics(tmp_path, mode="automl")
    rules = {row["rule"] for row in rows}

    assert {
        "SEM_IMPORT",
        "SEM_EVAL",
        "SEM_TRAILING_KEY",
        "SEM_ENUM_COMPARE",
        "SEM_INPUT_MUTATION",
        "SEM_KFOLD_RANDOM_STATE",
        "SEM_FALSY_DEFAULT",
        "SEM_UPDATE_KEY_MISMATCH",
        "SEM_SUBSET_ILOC",
        "SEM_TASK_PARAMS",
        "SEM_PARAM_GET_KWARGS",
        "SEM_PARAMS_SUBSCRIPT",
        "SEM_DF_POSITIONAL_SUBSCRIPT",
        "SEM_TRANSFORM_DISCARDED",
        "SEM_TARGET_ROW_RESHAPE",
        "SEM_APPEND_INNER_LOOP",
        "SEM_INPUT_TARGET_MUTATION",
        "SEM_PARAM_ALIAS_MUTATION",
        "SEM_INPUT_PIPELINE_FIT",
    } <= rules


def test_semantic_scan_automl_keeps_wrapper_kwargs_rule(tmp_path):
    source = tmp_path / "fedot" / "ops.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class Selector:\n"
        "    def __init__(self):\n"
        "        self.params = {'n_features_to_select': 2}\n"
        "    def build(self):\n"
        "        copied = {k: self.params.get(k) for k in ['n_features_to_select', 'step']}\n"
        "        return Model(**copied)\n",
        encoding="utf-8",
    )

    assert scan_semantics(tmp_path)[0]["rule"] == "SEM_PARAM_GET_KWARGS"
    assert scan_semantics(tmp_path, mode="fedot-calibration")[0]["rule"] == (
        "SEM_PARAM_GET_KWARGS"
    )
    assert not scan_semantics(tmp_path, mode="generic")


def test_semantic_scan_automl_includes_parameter_guard_invariants(tmp_path):
    source = tmp_path / "fedot" / "ops.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from random import random\n\n"
        "class RuntimeParamGuards:\n"
        "    def __init__(self):\n"
        "        self.params = {}\n"
        "        self.window_size = 1000\n"
        "        self.cut_part = 0\n"
        "        self.solver = None\n\n"
        "    def fix_window(self, max_window_size):\n"
        "        if self.window_size > max_window_size:\n"
        "            self.window_size = int(random() * max_window_size)\n\n"
        "    def fix_cut(self):\n"
        "        if not self.cut_part:\n"
        "            self.params.update(cut_part=0.5)\n\n"
        "    def fix_solver_default(self):\n"
        "        if self.solver is not None and self.solver == 'svd':\n"
        "            raise NotImplementedError()\n",
        encoding="utf-8",
    )

    automl_rules = {row["rule"] for row in scan_semantics(tmp_path, mode="automl")}
    generic_rules = {row["rule"] for row in scan_semantics(tmp_path, mode="generic")}

    assert {
        "SEM_RANDOM_PARAM_REWRITE",
        "SEM_DEFAULT_REWRITE",
    } <= automl_rules
    assert {
        "SEM_RANDOM_PARAM_REWRITE",
        "SEM_DEFAULT_REWRITE",
    }.isdisjoint(generic_rules)


def test_reader_stage_merges_semantic_leads(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("FEDOTLLM_EVOLVE_INCLUDE_ALL_FILES", "0")

    class Inference:
        usage = {}

        def query(self, _):
            return "NONE"

    state = {
        "repo_path": str(tmp_path),
        "workspace": str(tmp_path / "work"),
        "messages": [HumanMessage(content="run")],
        "lint_findings": [
            {"file": "fedot/a.py", "line": 1, "rule": "B001", "message": "x"}
        ],
        "semantic_findings": [
            {
                "file": "fedot/a.py",
                "line": 1,
                "verdict": "extra",
                "rule": "SEM_TEST",
                "why": "deterministic runtime defect",
            }
        ],
    }

    out = reader_stage(state, Inference())

    assert out["leads"] == [
        {
            "file": "fedot/a.py",
            "line": 1,
            "verdict": "extra",
            "rule": "SEM_TEST",
            "why": "deterministic runtime defect",
            "passes": [1],
            "agreed": False,
        }
    ]
    assert json.loads((tmp_path / "work" / "pipeline" / "leads.json").read_text()) == out["leads"]


def test_known_bug_benchmark_scores_pipeline_artifacts(tmp_path):
    dataset = {
        "F001": {"id": "F001", "file": "fedot/a.py", "line": 10},
        "F002": {"id": "F002", "file": "fedot/b.py", "line": 20},
        "F003": {"id": "F003", "file": "fedot/a.py", "line": 30},
    }
    split = {"train": ["F001", "F002"]}
    artifacts = tmp_path / "pipeline"
    artifacts.mkdir()
    (artifacts / "leads.json").write_text(
        json.dumps(
            [
                {"file": "fedot/a.py", "line": 11, "verdict": "extra", "why": "x"},
                {"file": "fedot/c.py", "line": 1, "verdict": "extra", "why": "noise"},
            ]
        ),
        encoding="utf-8",
    )
    (artifacts / "verified.json").write_text(
        json.dumps(
            [
                {"file": "fedot/a.py", "line": 10, "status": "confirmed"},
                {"file": "fedot/a.py", "line": 30, "status": "confirmed"},
                {"file": "fedot/c.py", "line": 1, "status": "confirmed"},
            ]
        ),
        encoding="utf-8",
    )

    score = score_artifacts(dataset, split, artifacts, subset="train", line_slack=1)

    assert score["reader_hits"] == ["F001"]
    assert score["confirmed_hits"] == ["F001"]
    assert score["queued_hits"] == ["F001"]
    assert score["reader_recall"] == 0.5
    assert score["confirmed_precision_on_subset_files"] == pytest.approx(1 / 3)
    assert score["confirmed_precision_known_dataset"] == pytest.approx(2 / 3)


def test_known_bug_benchmark_loads_all_split_groups(tmp_path):
    dataset = {
        "F001": {"id": "F001", "file": "fedot/a.py", "line": 1},
        "F002": {"id": "F002", "file": "fedot/b.py", "line": 2},
        "F003": {"id": "F003", "file": "fedot/c.py", "line": 3},
        "F004": {"id": "F004", "file": "fedot/d.py", "line": 4},
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["F001"],
                "test": ["F002"],
                "excluded_from_this_split": ["F003"],
            }
        ),
        encoding="utf-8",
    )

    split = load_split(split_path, dataset)

    assert split["split"] == ["F001", "F002"]
    assert split["excluded"] == ["F003"]
    assert split["unassigned"] == ["F004"]
    assert split["all"] == ["F001", "F002", "F003", "F004"]


def test_semantic_static_eval_writes_artifacts_and_scores(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "fedot" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = eval('1')\n", encoding="utf-8")
    dataset = {"F001": {"id": "F001", "file": "fedot/a.py", "line": 1}}
    split = {"train": ["F001"]}

    summary = run_semantic_static_eval(
        repo=repo,
        dataset=dataset,
        split=split,
        output=tmp_path / "eval",
        subsets=["train"],
        line_slack=0,
    )

    assert summary["train"]["leads"] == 1
    assert summary["train"]["verified"] == 1
    assert summary["train"]["queue"] == 1
    assert summary["train"]["score"]["confirmed_hits"] == ["F001"]
    assert (tmp_path / "eval" / "train" / "semantic.json").is_file()
    assert (tmp_path / "eval" / "train" / "verified.json").is_file()


def test_discovery_eval_runs_reader_verifier_without_static_confirmation(
    tmp_path, monkeypatch
):
    dataset = {"F001": {"id": "F001", "file": "fedot/a.py", "line": 7}}
    split = {"train": ["F001"]}
    seen: dict[str, str] = {}

    class Config:
        session_id = "session"
        llm = LLMConfig(provider="test", model_name="model", api_key="key")

    class Inference:
        def __init__(self, config, session_id=None):
            self.config = config
            self.session_id = session_id
            self.usage = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "cost_usd": 0.0,
            }

    def fake_scan(state):
        seen["only_files"] = os.environ["FEDOTLLM_EVOLVE_ONLY_FILES"]
        seen["semantic_mode"] = os.environ["FEDOTLLM_EVOLVE_SEMANTIC_MODE"]
        seen["static_verify"] = os.environ["FEDOTLLM_EVOLVE_STATIC_SEMANTIC_VERIFY"]
        out = tmp_path / "eval" / "train" / "pipeline"
        out.mkdir(parents=True)
        return {**state, "lint_findings": [], "semantic_findings": []}

    def fake_reader(state, inference):
        return {
            **state,
            "leads": [{"file": "fedot/a.py", "line": 7, "verdict": "extra"}],
        }

    def fake_verifier(state, inference):
        verified = [{"file": "fedot/a.py", "line": 7, "status": "confirmed"}]
        out = tmp_path / "eval" / "train" / "pipeline"
        (out / "leads.json").write_text(json.dumps(state["leads"]), encoding="utf-8")
        (out / "verified.json").write_text(json.dumps(verified), encoding="utf-8")
        return {**state, "verified": verified, "fixer_queue": verified}

    monkeypatch.setattr(benchmark_module, "load_config", lambda **_: Config())
    monkeypatch.setattr(benchmark_module, "AIInference", Inference)
    monkeypatch.setattr(benchmark_module, "scan_stage", fake_scan)
    monkeypatch.setattr(benchmark_module, "reader_stage", fake_reader)
    monkeypatch.setattr(benchmark_module, "verifier_stage", fake_verifier)

    summary = run_discovery_eval(
        repo=tmp_path / "repo",
        dataset=dataset,
        split=split,
        output=tmp_path / "eval",
        subsets=["train"],
        presets="fedotllm:openrouter",
        semantic_mode="automl",
        line_slack=0,
    )

    assert seen == {
        "only_files": "fedot/a.py",
        "semantic_mode": "automl",
        "static_verify": "0",
    }
    assert summary["train"]["score"]["confirmed_hits"] == ["F001"]
    assert summary["train"]["queue"] == 1


def test_known_bug_benchmark_threshold_failures_name_metric():
    summary = {
        "train": {
            "score": {
                "confirmed_recall": 0.9,
                "queued_recall": 1.0,
                "confirmed_precision_known_dataset": 0.75,
            }
        }
    }

    failures = threshold_failures(
        summary,
        min_confirmed_recall=1.0,
        min_queued_recall=1.0,
        min_known_precision=0.8,
    )

    assert failures == [
        "train.confirmed_recall=0.9000 below 1.0000",
        "train.confirmed_precision_known_dataset=0.7500 below 0.8000",
    ]


def test_reader_passes_are_unioned_by_location_and_strongest_verdict():
    merged = union_reader_passes(
        [
            [
                {
                    "file": "fedot/a.py",
                    "line": 3,
                    "verdict": "live",
                    "why": "fails",
                }
            ],
            [
                {
                    "file": "fedot/a.py",
                    "line": 3,
                    "verdict": "extra",
                    "why": "stronger",
                }
            ],
        ]
    )
    assert merged == [
        {
            "file": "fedot/a.py",
            "line": 3,
            "verdict": "extra",
            "why": "stronger",
            "passes": [0, 1],
            "agreed": True,
        }
    ]


def test_reader_omits_cosmetic_warnings_and_treats_missing_answers_as_inert(
    tmp_path,
):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")

    class Inference:
        def query(self, messages):
            prompt = messages[-1]["content"]
            assert "D100" not in prompt
            assert "B001" in prompt
            return "2: live | context manager can suppress the exception"

    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [
            {"line": 1, "rule": "D100", "message": "docstring", "cosmetic": True},
            {"line": 1, "rule": "B001", "message": "bare except", "cosmetic": False},
        ],
    )
    assert rows[0]["verdict"] == "inert"
    assert rows[1]["verdict"] == "live"


def test_reader_does_not_attach_context_until_requested(tmp_path):
    package = tmp_path / "fedot"
    package.mkdir()
    (package / "cache.py").write_text("def key(node):\n    return node.params\n")
    (package / "a.py").write_text("from fedot.cache import key\nvalue = key(None)\n")
    prompts = []

    class Inference:
        def query(self, messages):
            prompts.append(messages[-1]["content"])
            return "NONE"

    read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "docstring", "cosmetic": True}],
    )
    assert len(prompts) == 1
    assert "Requested `key`" not in prompts[0]
    assert "Lint hints" in prompts[0]


def test_reader_fetches_requested_symbol_then_gives_final_verdict(tmp_path):
    package = tmp_path / "fedot"
    package.mkdir()
    (package / "cache.py").write_text(
        "def unused():\n    return 0\n\n\ndef key(node):\n    return node.params\n"
    )
    (package / "a.py").write_text("from fedot.cache import key\nvalue = key(None)\n")
    prompts = []

    class Inference:
        def query(self, messages):
            prompt = messages[-1]["content"]
            prompts.append(prompt)
            if len(prompts) == 1:
                return "NEED_CONTEXT: fedot/cache.py:key"
            return "NONE"

    read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "docstring", "cosmetic": True}],
    )
    assert "Requested `key` from `fedot/cache.py`" in prompts[1]
    assert "return node.params" in prompts[1]
    assert "def unused" not in prompts[1]
    assert "File `" not in prompts[1]


def test_reader_fetches_context_from_dotted_module_path(tmp_path):
    package = tmp_path / "fedot" / "core" / "data"
    package.mkdir(parents=True)
    (tmp_path / "fedot" / "__init__.py").write_text("")
    (tmp_path / "fedot" / "core" / "__init__.py").write_text("")
    (tmp_path / "fedot" / "core" / "data" / "__init__.py").write_text("")
    (package / "data.py").write_text(
        "class InputData:\n"
        "    def __init__(self):\n"
        "        self.idx = None\n"
    )
    (tmp_path / "fedot" / "a.py").write_text("value = 1\n")
    prompts = []

    class Inference:
        def query(self, messages):
            prompts.append(messages[-1]["content"])
            if len(prompts) == 1:
                return "NEED_CONTEXT: fedot.core.data.data:InputData"
            return "NONE"

    read_one_file(Inference(), tmp_path, "fedot/a.py", [])

    assert "Requested `InputData` from `fedot/core/data/data.py`" in prompts[1]
    assert "class InputData" in prompts[1]


def test_reader_peeks_with_a_print_only_when_it_asks(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("def window():\n    return 1\n")
    prompts = []

    class Inference:
        def query(self, messages):
            prompts.append(messages[-1]["content"])
            if len(prompts) == 1:
                return "NEED_RUN: print(window())"
            return "EXTRA 1: window is constant"

    monkeypatch.setattr(
        "fedotllm.agents.evolve.reader.run_peek",
        lambda *args, **kwargs: (0, "1"),
    )
    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "B001", "message": "x", "cosmetic": False}],
        python="python",
    )
    assert "File `" not in prompts[1]
    assert "Peek (exit 0)" in prompts[1]
    assert rows[0]["verdict"] == "extra"


def test_reader_gives_a_small_file_but_refuses_a_large_one(tmp_path):
    package = tmp_path / "fedot"
    package.mkdir()
    (package / "tiny.py").write_text("FLAG = 1\n")
    (package / "huge.py").write_text("x = 1\n" * 90)
    (package / "a.py").write_text("from fedot.tiny import FLAG\n")
    prompts = []

    class Inference:
        def query(self, messages):
            prompts.append(messages[-1]["content"])
            if len(prompts) == 1:
                return (
                    "NEED_CONTEXT: fedot/tiny.py\n"
                    "NEED_CONTEXT: fedot/huge.py"
                )
            return "NONE"

    read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "docstring", "cosmetic": True}],
    )
    assert "Small file `fedot/tiny.py`" in prompts[1]
    assert "FLAG = 1" in prompts[1]
    assert "is not a small file" in prompts[1]
    assert "x = 1" not in prompts[1]


def test_run_peek_refuses_a_proof_script(tmp_path):
    code, output = run_peek("x = 1\n" * 12, tmp_path, "python", tmp_path, "x")
    assert code == 2
    assert "print" in output


def test_reader_parses_markdown_and_source_line_numbers():
    findings = [{"line": 23, "rule": "S307", "message": "eval", "cosmetic": False}]
    rows = parse_reader_answer(
        "**23: live** — eval of the task config",
        "fedot/remote/pipeline_run_config.py",
        findings,
    )
    assert any(row["line"] == 23 and row["verdict"] == "live" for row in rows)


def test_reader_parses_extra_on_a_line_with_no_lint():
    rows = parse_reader_answer(
        "EXTRA 112: Task compared to enum is always False",
        "fedot/api/api_utils/api_data.py",
        [{"line": 73, "rule": "BLE001", "message": "bare except", "cosmetic": False}],
    )
    extras = [row for row in rows if row["verdict"] == "extra"]
    assert extras[0]["line"] == 112
    assert "always False" in extras[0]["why"]


def test_reader_parses_extra_line_word():
    rows = parse_reader_answer(
        "EXTRA line 9: name used after failed import",
        "fedot/a.py",
        [],
    )
    assert rows[0]["line"] == 9
    assert rows[0]["verdict"] == "extra"


def test_reader_prefers_a_cited_source_line_over_a_lint_index():
    rows = parse_reader_answer(
        "EXTRA 6: exception constructed but never raised on line 98",
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert extras[0]["line"] == 98


def test_reader_numbered_extra_uses_cited_line_when_they_differ():
    rows = parse_reader_answer(
        "6: extra | constructed error never raised on line 98",
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert extras[0]["line"] == 98


def test_reader_uses_the_defect_citation_not_the_first_line_mention():
    rows = parse_reader_answer(
        "EXTRA 6: lint at line 62 is unused and harmless. "
        "The missing raise on line 98 is a genuine logic bug",
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert extras[0]["line"] == 98


def test_reader_relocates_extra_from_import_to_quoted_usage():
    source = (
        "from pkg.jobs import Job, JobEnum\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        if self.job == JobEnum.ready:\n"
        "            skip()\n"
    )
    rows = parse_reader_answer(
        "EXTRA 1: `self.job == JobEnum.ready` compares a Job object to an enum, always False",
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
        source,
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert extras[0]["line"] == 4


def test_reader_parses_extra_line_ranges():
    rows = parse_reader_answer(
        "EXTRA lines 23-24: eval of the config string",
        "fedot/remote/pipeline_run_config.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert {row["line"] for row in extras} == {23, 24}


def test_reader_parses_comma_separated_extra_lines():
    rows = parse_reader_answer(
        "EXTRA 90, 96: missing data_type in both early returns",
        "fedot/a.py",
        [],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert {row["line"] for row in extras} == {90, 96}


def test_reader_nudge_after_none_on_silent_shapes(tmp_path):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("value = eval(config['task'])\n")
    replies = iter(["NONE", "EXTRA 1: eval of a config string"])

    class Inference:
        def query(self, _):
            return next(replies)

    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert extras and extras[0]["line"] == 1


def test_reader_nudge_after_none_on_enum_comparison(tmp_path):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("if self.job == JobEnum.ready:\n    skip_nan()\n")
    replies = iter(["NONE", "EXTRA 1: comparison cannot be true"])

    class Inference:
        def query(self, _):
            return next(replies)

    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert extras and extras[0]["line"] == 1


def test_reader_nudge_when_silent_shape_is_not_the_named_extra(tmp_path):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text(
        "def load(config=None):\n"
        "    if config is None:\n"
        "        return\n"
        "    value = eval(config['task'])\n"
    )
    replies = iter(
        [
            "EXTRA 2: empty config leaves the object with no attributes",
            "EXTRA 2: empty config leaves the object with no attributes\n"
            "EXTRA 4: eval of a config string",
        ]
    )

    class Inference:
        def query(self, _):
            return next(replies)

    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extra_lines = {row["line"] for row in rows if row.get("verdict") == "extra"}
    assert 4 in extra_lines


def test_reader_skips_nudge_after_none_without_silent_shape(tmp_path):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("params.update(wrong=1)\n")
    calls = []

    class Inference:
        def query(self, _):
            calls.append(1)
            return "NONE"

    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    assert calls == [1]
    assert not [row for row in rows if row.get("verdict") == "extra"]


def test_reader_leftover_nudge_only_if_another_silent_shape_is_uncovered(
    tmp_path,
):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text(
        "value = eval(config['task'])\n"
        "params = {'force ': True}\n"
    )
    replies = iter(
        [
            "EXTRA 1: eval of a config string",
            "EXTRA 1: eval of a config string\n"
            "EXTRA 2: trailing space in key",
        ]
    )

    class Inference:
        def query(self, _):
            return next(replies)

    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extra_lines = {row["line"] for row in rows if row.get("verdict") == "extra"}
    assert extra_lines == {1, 2}


def test_unfenced_verifier_script_is_still_parsed():
    from fedotllm.agents.evolve.verifier import parse_script

    raw = (
        "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
        "from fedot.a import value\n"
        "print('DEFECT', value)\n"
        "raise AssertionError('DEFECT: always false')\n"
    )
    script = parse_script(raw)
    assert "from fedot.a import value" in script
    assert "raise AssertionError" in script


def test_semantic_behaviour_proof_must_match_the_rule():
    assert not behaviour_matches_semantic_lead(
        {"rule": "SEM_PARAM_GET_KWARGS"},
        "DEFECT: trailing space in key 'force_row_wise '",
    )
    assert behaviour_matches_semantic_lead(
        {"rule": "SEM_PARAM_GET_KWARGS"},
        "DEFECT: kwargs pass None and override sklearn defaults",
    )


def test_static_semantic_confirmation_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FEDOTLLM_EVOLVE_STATIC_SEMANTIC_VERIFY", raising=False)
    result = static_semantic_confirmation(
        {
            "file": "fedot/a.py",
            "line": 3,
            "rule": "SEM_EVAL",
            "why": "eval executes config text",
            "semantic": True,
            "verdict": "extra",
        }
    )

    assert result is None


def test_static_semantic_confirmation_uses_no_synthetic_script(monkeypatch):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_STATIC_SEMANTIC_VERIFY", "1")
    result = static_semantic_confirmation(
        {
            "file": "fedot/a.py",
            "line": 3,
            "rule": "SEM_EVAL",
            "why": "eval executes config text",
            "semantic": True,
            "verdict": "extra",
        }
    )

    assert result is not None
    assert result["status"] == "confirmed"
    assert result["kind"] == "static"
    assert result["script"] == ""
    assert result["route"] == "deterministic semantic scan"


def test_verifier_stage_confirms_static_semantic_leads_without_llm_when_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_STATIC_SEMANTIC_VERIFY", "1")

    class NoLLM:
        usage = {}

        def query(self, _):
            raise AssertionError("static semantic verifier must not call LLM")

    state = {
        "repo_path": str(tmp_path),
        "workspace": str(tmp_path / "work"),
        "messages": [],
        "leads": [
            {
                "file": "fedot/a.py",
                "line": 3,
                "rule": "SEM_EVAL",
                "why": "eval executes config text",
                "semantic": True,
                "verdict": "extra",
            }
        ],
    }

    out = verifier_stage(state, NoLLM())

    assert out["verified"][0]["status"] == "confirmed"
    assert out["verified"][0]["script"] == ""
    assert out["fixer_queue"][0]["defect_class"] == "ordinary"
    assert json.loads((tmp_path / "work" / "pipeline" / "verified.json").read_text())[0][
        "route"
    ] == "deterministic semantic scan"


def test_static_semantic_evidence_is_not_reader_queue(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("eval('1')\n", encoding="utf-8")
    evidence = tmp_path / "verified.json"
    evidence.write_text(
        json.dumps(
            [
                {
                    "file": "fedot/a.py",
                    "line": 1,
                    "rule": "SEM_EVAL",
                    "why": "eval executes config text",
                    "status": "confirmed",
                    "kind": "static",
                    "script": "",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FEDOTLLM_VERIFIED", str(evidence))

    rows = static_semantic_defects(tmp_path)

    assert queued_reader_defect(tmp_path) is None
    assert rows[0]["rule"] == "SEM_EVAL"
    assert "deterministic FEDOT semantic scan" in static_semantic_section(rows[0])
    assert "no pre-written proof script" in static_semantic_section(rows[0])


def test_behaviour_proof_may_import_helpers_next_to_the_suspected_file(tmp_path):
    result = {
        "file": "fedot/api/api_utils/api_data.py",
        "line": 112,
        "status": "confirmed",
        "kind": "behaviour",
        "script": (
            "from fedot.api.api_utils.params import ApiParams\n"
            "from fedot.api.api_utils.api_data import ApiDataProcessor\n"
            "print('DEFECT', ApiDataProcessor, ApiParams)\n"
            "raise AssertionError('DEFECT: always false')\n"
        ),
    }

    class Boom:
        def query(self, _):
            raise AssertionError("must not rewrite onto Fedot.fit")

    out = check_public_route(Boom(), tmp_path, result, "python", tmp_path / "work")
    assert out["status"] == "confirmed"
    assert out["route"] == "suspected module"


def test_reader_prompt_is_not_a_holdout_checklist():
    banned = (
        "tasktypesenum",
        "catboost",
        "num_trees",
        "shrinkage",
        "ripser",
        "gph",
        "iloc",
        "window_size",
        "labelencoder",
    )
    text = READER_SYSTEM.lower() + "\n" + NONE_NUDGE.lower() + "\n" + LEFTOVER_NUDGE.lower()
    assert not any(token in text for token in banned)


def test_verifier_prompt_is_not_a_holdout_checklist():
    banned = (
        "tasktypesenum",
        "catboost",
        "num_trees",
        "shrinkage",
        "ripser",
        "gph",
        "iloc",
        "window_size",
        "labelencoder",
    )
    assert not any(token in VERIFIER_SYSTEM.lower() for token in banned)


def test_semantic_scan_has_no_dataset_fingerprints():
    from pathlib import Path

    from fedotllm.agents.evolve import semantic_scan, verifier

    banned = (
        "catboost",
        "catboostreg",
        "force_row_wise",
        "border_count",
        "max_leaves",
        "grow_policy",
        "sampling-scope",
        "shrinkage",
        "custom_model",
        "cut_part",
        "validation_blocks",
        "labelencoder",
        "num_trees",
        "interaction_only",
    )
    text = (
        Path(semantic_scan.__file__).read_text(encoding="utf-8").lower()
        + "\n"
        + str(verifier._SEMANTIC_OUTPUT_TERMS).lower()
    )
    assert not any(token in text for token in banned)


def test_reader_does_not_treat_none_plus_prose_as_abstention():
    rows = parse_reader_answer(
        "NONE\n\nThe use of eval on line 23 is a security risk",
        "fedot/remote/pipeline_run_config.py",
        [{"line": 1, "rule": "D100", "message": "doc", "cosmetic": True}],
    )
    extras = [row for row in rows if row.get("verdict") == "extra"]
    assert extras and extras[0]["line"] == 23


def test_reader_parses_line_prefix_and_ignores_none_substring():
    rows = parse_reader_answer(
        "None of the lint items matter.\nLine 112: comparison is always false",
        "fedot/a.py",
        [{"line": 73, "rule": "BLE001", "message": "x", "cosmetic": False}],
    )
    extras = [row for row in rows if row["verdict"] == "extra"]
    assert extras[0]["line"] == 112


def test_reader_retries_then_treats_unparseable_as_inert(tmp_path):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    calls = []

    class Inference:
        def query(self, messages):
            calls.append(1)
            if len(calls) == 1:
                return "I looked at the file and it seems fine overall."
            return "NONE"

    rows = read_one_file(
        Inference(),
        tmp_path,
        "fedot/a.py",
        [{"line": 1, "rule": "B001", "message": "x", "cosmetic": False}],
    )
    assert len(calls) == 2
    assert rows[0]["verdict"] == "inert"


def test_classifier_only_routes_confirmed_non_cosmetic_defects():
    rows = [
        {"status": "confirmed", "file": "a", "line": 1},
        {"status": "confirmed", "kind": "timeout", "file": "b", "line": 2},
        {"status": "refuted", "file": "c", "line": 3},
        {
            "status": "not testable",
            "file": "d",
            "line": 4,
            "why": "eval of config task",
            "plausible": True,
            "fixable": True,
        },
        {
            "status": "not testable",
            "file": "e",
            "line": 5,
            "why": "print traceback is ugly",
            "plausible": True,
            "fixable": True,
        },
        {
            "status": "not testable",
            "file": "g",
            "line": 6,
            "why": "comparison is always false so the branch never runs",
            "plausible": True,
            "fixable": False,
        },
    ]
    assert defect_class(rows[0]) == "ordinary"
    assert defect_class(rows[1]) == "critical"
    assert defect_class(rows[2]) == "excluded"
    assert defect_class(rows[3]) == "unproven"
    assert defect_class(rows[4]) == "excluded"
    assert defect_class(rows[5]) == "excluded"
    assert is_style_noise(rows[4]["why"])
    assert worth_fixing(rows[3])
    assert not worth_fixing(rows[5])
    assert defect_class({"status": "confirmed", "kind": "behaviour", "file": "f"}) == (
        "ordinary"
    )
    assert (
        defect_class(
            {
                "status": "confirmed",
                "kind": "behaviour",
                "file": "h",
                "why": "unused loop variable values is never read",
            }
        )
        == "excluded"
    )
    assert [row["file"] for row in fixer_candidates(rows)] == ["a", "b", "d"]
    queued = fixer_attempt_queue(
        fixer_candidates(rows), max_unproven=0, max_total=0
    )
    assert [row["file"] for row in queued] == ["a", "b"]
    assert [row["file"] for row in fixer_attempt_queue(
        fixer_candidates(rows), max_unproven=-1
    )] == ["a", "b", "d"]
    assert [row["file"] for row in fixer_attempt_queue(queued, max_total=2)] == [
        "a",
        "b",
    ]
    public_queue = [
        {
            "file": "internal.py",
            "status": "confirmed",
            "defect_class": "ordinary",
            "route": "suspected module",
        },
        {
            "file": "public_a.py",
            "status": "confirmed",
            "defect_class": "ordinary",
            "route": "public interface",
        },
        {
            "file": "public_b.py",
            "status": "confirmed",
            "defect_class": "ordinary",
            "route": "public interface (rewritten)",
        },
    ]
    assert [row["file"] for row in fixer_attempt_queue(public_queue)] == [
        "public_a.py",
        "public_b.py",
    ]
    confirmed = EvolveResult(
        proposal=Proposal(), success=True, evidence="verified"
    )
    cosmetic = EvolveResult(
        proposal=Proposal(), success=True, evidence="reader"
    )
    assert is_run_success({"status": "confirmed", "defect_class": "ordinary"}, confirmed)
    assert not is_run_success(
        {"status": "not testable", "defect_class": "unproven"}, cosmetic
    )


def test_verifier_candidates_drop_speculative_reader_noise():
    rows = [
        {"file": "fedot/a.py", "line": 1, "why": "private import may break in future versions"},
        {"file": "fedot/b.py", "line": 2, "why": "eval of config string"},
        {"file": "fedot/c.py", "line": 3, "why": "self.model is None if instantiated directly"},
        {"file": "fedot/d.py", "line": 4, "why": "no defect, the check is actually correct"},
    ]
    assert verifier_candidates(rows) == [rows[1]]


def test_verifier_candidates_prioritize_high_value_runtime_shapes():
    rows = [
        {"file": "fedot/z.py", "line": 2, "why": "plain crash"},
        {"file": "fedot/a.py", "line": 3, "why": "wrong key silently ignored"},
    ]
    assert verifier_candidates(rows) == [rows[1], rows[0]]


def test_queued_reader_defect_pins_unproven_lead(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    target = repo / "fedot" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n")
    evidence = tmp_path / "lead.json"
    evidence.write_text(
        json.dumps(
            [
                {
                    "status": "not testable",
                    "file": "fedot/a.py",
                    "line": 23,
                    "why": "eval of config task",
                }
            ]
        )
    )
    monkeypatch.setenv("FEDOTLLM_VERIFIED", str(evidence))
    row = queued_reader_defect(repo)
    assert row["file"] == "fedot/a.py"
    assert row["status"] == "not testable"


def test_review_plausible_reads_yes_no_without_a_script(tmp_path):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("value = eval(config)\n")

    class Inference:
        def query(self, _):
            return "PLAUSIBLE: yes\nFIX: yes\nREASON: eval runs the config string"

    result = review_plausible(
        Inference(),
        tmp_path,
        {
            "status": "not testable",
            "file": "fedot/a.py",
            "line": 1,
            "why": "eval of config",
        },
    )
    assert result["plausible"] is True
    assert result["fixable"] is True
    assert parse_plausible("PLAUSIBLE: no\nREASON: style") is False
    assert parse_fixable("PLAUSIBLE: yes\nFIX: no\nREASON: style") is False
    skipped = review_plausible(
        Inference(),
        tmp_path,
        {"status": "refuted", "file": "fedot/a.py", "line": 1, "why": "x"},
    )
    assert "plausible" not in skipped


def test_verifier_requires_predicted_exception_and_right_traceback_site(
    tmp_path, monkeypatch
):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("x = 1\n")

    class Inference:
        def query(self, _):
            return (
                "EXPECT: ValueError\n"
                "AFTER: success\n"
                "MESSAGE_TERMS: none\n"
                "```python\nfrom fedot.a import x\nprint(x.bad)\n```"
            )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: (
            1,
            'Traceback\n  File "fedot/a.py", line 1\nValueError: bad',
        ),
    )
    result = verify_lead(
        Inference(),
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "bad value"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert result["got"] == "ValueError"


def test_behaviour_setup_error_is_retried_as_broken_script(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("value = 0\n")
    replies = iter(
        [
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import value\nBroken(1, 2)\n```",
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import value\nprint('DEFECT', value)\n"
            "raise AssertionError('DEFECT: ignored setting')\n```",
        ]
    )
    inference = type("Inference", (), {"query": lambda self, prompt: next(replies)})()
    executions = iter(
        [
            (1, "TypeError: Broken.__init__() takes 1 positional argument but 2 were given"),
            (1, "DEFECT 0\nAssertionError: DEFECT: ignored setting"),
        ]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: next(executions),
    )
    result = verify_lead(
        inference,
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "setting ignored"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert result["kind"] == "behaviour"


def test_verifier_confirms_a_silent_behaviour_defect(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("def window():\n    return 1\n")

    class Inference:
        def query(self, _):
            return (
                "KIND: behaviour\n"
                "EXPECT: none\n"
                "AFTER: success\n"
                "MESSAGE_TERMS: none\n"
                "```python\n"
                "from fedot.a import window\n"
                "print('DEFECT', window())\n"
                "raise AssertionError('DEFECT: unreproducible window')\n"
                "```"
            )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: (
            1,
            "DEFECT 1\nAssertionError: DEFECT: unreproducible window",
        ),
    )
    result = verify_lead(
        Inference(),
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "random window"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert result["kind"] == "behaviour"


def test_broken_init_typeerror_asks_to_match_constructor(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("class Box:\n    def __init__(self, **kw):\n        pass\n")
    prompts: list[str] = []
    replies = iter(
        [
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import Box\nBox({'x': 1})\n```",
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import Box\nprint('DEFECT')\n"
            "raise AssertionError('DEFECT: x')\n```",
        ]
    )

    class Inference:
        def query(self, prompt):
            prompts.append(prompt)
            return next(replies)

    executions = iter(
        [
            (1, "TypeError: Box.__init__() takes 1 positional argument but 2 were given"),
            (1, "DEFECT\nAssertionError: DEFECT: x"),
        ]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: next(executions),
    )
    result = verify_lead(
        Inference(),
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "wrong field"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert any("takes 1 positional argument but 2 were given" in prompt for prompt in prompts)


def test_import_error_retry_dumps_the_interpreter_output(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("value = eval(config)\n")
    prompts: list[str] = []
    replies = iter(
        [
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.tasks import TsTaskParams\n```",
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import value\nprint('DEFECT', value)\n"
            "raise AssertionError('DEFECT: eval')\n```",
        ]
    )

    class Inference:
        def query(self, prompt):
            prompts.append(prompt)
            return next(replies)

    executions = iter(
        [
            (
                1,
                "ImportError: cannot import name 'TsTaskParams' from 'fedot.tasks'",
            ),
            (1, "DEFECT eval\nAssertionError: DEFECT: eval"),
        ]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: next(executions),
    )
    result = verify_lead(
        Inference(),
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "eval of config"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert any("cannot import name 'TsTaskParams'" in prompt for prompt in prompts)


def test_wrong_site_retry_dumps_the_interpreter_output(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("def boom():\n    raise TypeError('in suspected')\n")
    prompts: list[str] = []
    replies = iter(
        [
            "KIND: crash\nEXPECT: TypeError\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.data import Input\nInput()\n```",
            "KIND: crash\nEXPECT: TypeError\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import boom\nboom()\n```",
        ]
    )

    class Inference:
        def query(self, prompt):
            prompts.append(prompt)
            return next(replies)

    executions = iter(
        [
            (
                1,
                'Traceback\n  File "fedot/data.py", line 1\n'
                "TypeError: Input.__init__() missing 1 required positional "
                "argument: 'data_type'",
            ),
            (
                1,
                'Traceback\n  File "fedot/a.py", line 2\n'
                "TypeError: in suspected",
            ),
        ]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: next(executions),
    )
    result = verify_lead(
        Inference(),
        tmp_path,
        {"file": "fedot/a.py", "line": 2, "why": "crashes on default"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert any("missing 1 required positional" in prompt for prompt in prompts)


def test_crash_script_that_completes_retries_as_behaviour(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("flag = False\n")
    replies = iter(
        [
            "KIND: crash\nEXPECT: ValueError\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot import a\n```",
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import flag\nprint('DEFECT', flag)\n"
            "raise AssertionError('DEFECT: always false')\n```",
        ]
    )
    inference = type("Inference", (), {"query": lambda self, prompt: next(replies)})()
    executions = iter(
        [
            (0, "ok"),
            (1, "DEFECT False\nAssertionError: DEFECT: always false"),
        ]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: next(executions),
    )
    result = verify_lead(
        inference,
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "comparison cannot be true"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert result["kind"] == "behaviour"


def test_expect_none_without_kind_is_treated_as_behaviour(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("value = False\n")

    class Inference:
        def query(self, _):
            return (
                "EXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
                "```python\n"
                "from fedot.a import value\n"
                "print('DEFECT', value)\n"
                "raise AssertionError('DEFECT: always false')\n"
                "```"
            )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: (1, "DEFECT False\nAssertionError: DEFECT: always false"),
    )
    result = verify_lead(
        Inference(),
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "comparison cannot be true"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"
    assert result["kind"] == "behaviour"


def test_importing_the_suspected_file_is_a_valid_behaviour_route(tmp_path):
    result = {
        "file": "fedot/api/api_utils/api_data.py",
        "line": 112,
        "status": "confirmed",
        "kind": "behaviour",
        "script": (
            "from fedot.api.api_utils.api_data import ApiDataProcessor\n"
            "print('DEFECT')\n"
            "raise AssertionError('DEFECT: always false')\n"
        ),
    }

    class Boom:
        def query(self, _):
            raise AssertionError("must not rewrite onto Fedot.fit")

    out = check_public_route(Boom(), tmp_path, result, "python", tmp_path / "work")
    assert out["status"] == "confirmed"
    assert out["route"] == "suspected module"


def test_verifier_retries_a_script_that_does_not_reproduce(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir()
    source.write_text("raise ValueError('bad')\n")
    replies = iter(
        [
            "EXPECT: ValueError\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nprint('missed target')\n```",
            "EXPECT: ValueError\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot import a\n```",
        ]
    )
    inference = type("Inference", (), {"query": lambda self, prompt: next(replies)})()
    executions = iter(
        [
            (0, ""),
            (1, 'Traceback\n  File "/repo/fedot/a.py", line 1\nValueError: bad'),
        ]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: next(executions),
    )
    result = verify_lead(
        inference,
        tmp_path,
        {"file": "fedot/a.py", "line": 1, "why": "raises"},
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "confirmed"


def test_public_route_rewrite_gets_a_second_attempt(tmp_path, monkeypatch):
    replies = iter(
        [
            "EXPECT: none\nAFTER: success\nMESSAGE_TERMS: none",
            "EXPECT: ValueError\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.core.data.data import InputData\n"
            "InputData.from_csv('missing')\n```",
        ]
    )
    inference = type("Inference", (), {"query": lambda self, prompt: next(replies)})()
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: (
            1,
            '  File "/repo/fedot/internal.py", line 1\nValueError: missing',
        ),
    )
    result = check_public_route(
        inference,
        tmp_path,
        {
            "file": "fedot/internal.py",
            "line": 1,
            "why": "failure",
            "status": "confirmed",
            "got": "ValueError",
            "script": "from fedot.internal.helper import run\nrun()",
        },
        "python",
        tmp_path / "work",
    )
    assert result["route"] == "public interface (rewritten)"


def test_public_rewrite_stays_anchored_to_the_same_semantic_lead(
    tmp_path, monkeypatch
):
    prompts: list[str] = []
    replies = iter(
        [
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import value\nprint('DEFECT: trailing space key')\n"
            "raise AssertionError('DEFECT: wrong key')\n```",
            "KIND: behaviour\nEXPECT: none\nAFTER: success\nMESSAGE_TERMS: none\n"
            "```python\nfrom fedot.a import value\nprint('DEFECT: kwargs pass None')\n"
            "raise AssertionError('DEFECT: kwargs None')\n```",
        ]
    )

    class Inference:
        def query(self, prompt):
            prompts.append(prompt)
            return next(replies)

    executions = iter(
        [
            (1, "DEFECT: trailing space key\nAssertionError: DEFECT: wrong key"),
            (1, "DEFECT: kwargs pass None\nAssertionError: DEFECT: kwargs None"),
        ]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: next(executions),
    )

    result = check_public_route(
        Inference(),
        tmp_path,
        {
            "file": "fedot/a.py",
            "line": 7,
            "why": "kwargs pass None into estimator defaults",
            "rule": "SEM_PARAM_GET_KWARGS",
            "status": "confirmed",
            "kind": "behaviour",
            "got": "AssertionError",
            "script": "from fedot.b import helper\nhelper()",
        },
        "python",
        tmp_path / "work",
    )

    assert result["route"] == "public interface (rewritten)"
    assert "File: fedot/a.py" in prompts[0]
    assert "Line: 7" in prompts[0]
    assert "fedot/a.py:7" in prompts[1]
    assert "do not demonstrate a different defect" in prompts[1]


def test_public_rewrite_does_not_confirm_a_crash_in_the_wrong_file(
    tmp_path, monkeypatch
):
    inference = type(
        "Inference",
        (),
        {
            "query": lambda self, prompt: (
                "EXPECT: TypeError\nAFTER: success\nMESSAGE_TERMS: none\n"
                "```python\n"
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n"
                "PipelineBuilder().add_node('cut', parameters={})\n"
                "```"
            )
        },
    )()
    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.run_script",
        lambda *args, **kwargs: (
            1,
            "TypeError: OptGraphBuilder.add_node() got an unexpected "
            "keyword argument 'parameters'",
        ),
    )
    result = check_public_route(
        inference,
        tmp_path,
        {
            "file": "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/ts_transformations.py",
            "line": 667,
            "why": "cut_part ignored",
            "status": "confirmed",
            "got": "TypeError",
            "script": "from fedot.core.operations.evaluation."
            "operation_implementations.data_operations "
            "import ts_transformations",
        },
        "python",
        tmp_path / "work",
    )
    assert result["status"] == "internal only"
    assert result["route"] != "public interface (rewritten)"


def test_verifier_rejects_manufactured_assertion_and_message_only_site():
    assert fabricated("assert False, 'manufactured'", "AssertionError")
    assert not traceback_mentions(
        "ValueError: failed in fedot/a.py",
        "fedot/a.py",
    )
    assert traceback_mentions(
        '  File "/repo/fedot/a.py", line 3, in run\nValueError: bad',
        "fedot/a.py",
    )


def test_scan_stage_can_consume_external_verified_evidence(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    verified = tmp_path / "verified.json"
    verified.write_text(
        json.dumps(
            [
                {
                    "status": "confirmed",
                    "file": "fedot/a.py",
                    "line": 1,
                    "why": "public failure",
                }
            ]
        )
    )
    monkeypatch.setenv("FEDOTLLM_VERIFIED", str(verified))
    result = scan_stage(
        {
            "repo_path": str(repo),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
        }
    )
    assert result["pipeline_stages"] == ["scan"]
    assert result["pipeline_external_verified"] is True
    assert result["fixer_queue"][0]["defect_class"] == "ordinary"


def test_scan_stage_ignores_langgraph_runnable_config(tmp_path):
    repo = tmp_path / "repo"
    (repo / "fedot").mkdir(parents=True)
    (repo / "fedot" / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = scan_stage(
        {
            "repo_path": str(repo),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
        },
        config={"tags": ["langgraph"]},
    )
    assert result["pipeline_stages"] == ["scan"]
    assert result["pipeline_external_verified"] is False


def test_reader_fails_closed_when_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_reader_pass",
        lambda *args, **kwargs: [
            {
                "file": "fedot/a.py",
                "line": 1,
                "verdict": "unclear",
                "why": "reader error: provider unavailable",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="failed for every file"):
        reader_stage(
            {
                "repo_path": str(tmp_path),
                "workspace": str(tmp_path / "workspace"),
                "messages": [],
                "lint_findings": [
                    {
                        "file": "fedot/a.py",
                        "line": 1,
                        "rule": "X1",
                        "message": "x",
                    }
                ],
            },
            type("Inference", (), {"usage": {}})(),
        )


def test_reader_passes_run_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_READER_PASSES", "2")
    barrier = threading.Barrier(2)

    def run_pass(*args, **kwargs):
        barrier.wait(timeout=2)
        return [
            {
                "file": "fedot/a.py",
                "line": 1,
                "verdict": "live",
                "why": "runtime failure",
            }
        ]

    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_reader_pass",
        run_pass,
    )
    result = reader_stage(
        {
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
            "lint_findings": [
                {
                    "file": "fedot/a.py",
                    "line": 1,
                    "rule": "B001",
                    "message": "x",
                }
            ],
        },
        type("Inference", (), {"usage": {}})(),
    )
    assert len(result["reader_passes"]) == 2
    assert result["leads"][0]["agreed"] is True


def test_reader_defaults_to_a_single_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("FEDOTLLM_EVOLVE_READER_PASSES", raising=False)

    def run_pass(*args, **kwargs):
        return [
            {
                "file": "fedot/a.py",
                "line": 1,
                "verdict": "live",
                "why": "runtime failure",
            }
        ]

    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_reader_pass",
        run_pass,
    )
    result = reader_stage(
        {
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
            "lint_findings": [
                {
                    "file": "fedot/a.py",
                    "line": 1,
                    "rule": "B001",
                    "message": "x",
                }
            ],
        },
        type("Inference", (), {"usage": {}})(),
    )
    assert len(result["reader_passes"]) == 1


def test_fixer_tries_the_next_verified_candidate_after_rejection(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_PROBE_GATE", "0")
    outcomes = iter([True, True])
    picks = []

    def run_loop(*, repo, workspace, inference, **kwargs):
        success = next(outcomes)
        picks.append(workspace.name)
        return EvolveResult(
            proposal=Proposal(file_path=f"fedot/{workspace.name}.py"),
            pick=f"fedot/{workspace.name}.py",
            success=success,
        )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.require_clean_repo",
        lambda repo: None,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_evolution_loop",
        run_loop,
    )
    result = fixer_stage(
        {
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
            "fixer_queue": [
                {"file": "fedot/first.py", "line": 1},
                {"file": "fedot/second.py", "line": 2},
            ],
        },
        type("Inference", (), {"usage": {}})(),
    )
    assert picks == ["fedot-first-py-1", "fedot-second-py-2"]
    assert result["evolve_success"] is True
    assert result["selected_defect"]["file"] == "fedot/second.py"


def test_fixer_skips_internal_when_public_proofs_exist(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_PROBE_GATE", "0")
    picks = []

    def run_loop(*, repo, workspace, inference, **kwargs):
        picks.append(workspace.name)
        return EvolveResult(
            proposal=Proposal(file_path=f"fedot/{workspace.name}.py"),
            pick=f"fedot/{workspace.name}.py",
            success=True,
        )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.require_clean_repo",
        lambda repo: None,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_evolution_loop",
        run_loop,
    )
    fixer_stage(
        {
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
            "fixer_queue": [
                {
                    "file": "fedot/internal.py",
                    "line": 1,
                    "status": "confirmed",
                    "defect_class": "ordinary",
                    "route": "suspected module",
                },
                {
                    "file": "fedot/public.py",
                    "line": 2,
                    "status": "confirmed",
                    "defect_class": "ordinary",
                    "route": "public interface",
                },
            ],
        },
        type("Inference", (), {"usage": {}})(),
    )
    assert picks == ["fedot-public-py-2"]


def test_fixer_skips_unproven_queue_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_PROBE_GATE", "0")
    called = []

    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_evolution_loop",
        lambda **kwargs: called.append(kwargs) or EvolveResult(success=True),
    )
    result = fixer_stage(
        {
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
            "fixer_queue": [
                {
                    "file": "fedot/cosmetic.py",
                    "line": 1,
                    "status": "not testable",
                    "defect_class": "unproven",
                    "why": "empty args IndexError",
                }
            ],
        },
        type("Inference", (), {"usage": {}})(),
    )
    assert called == []
    assert result["evolve_success"] is False
    assert "no confirmed defect" in result["evolve_abstain_reason"]


def test_fixer_does_not_count_unproven_gate_pass_as_success(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_PROBE_GATE", "0")
    monkeypatch.setenv("FEDOTLLM_EVOLVE_MAX_UNPROVEN_FIXES", "-1")
    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.require_clean_repo",
        lambda repo: None,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_evolution_loop",
        lambda **kwargs: EvolveResult(
            proposal=Proposal(file_path="fedot/cosmetic.py"),
            pick="fedot/cosmetic.py",
            success=True,
            evidence="reader",
        ),
    )
    result = fixer_stage(
        {
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
            "fixer_queue": [
                {
                    "file": "fedot/cosmetic.py",
                    "line": 36,
                    "status": "not testable",
                    "defect_class": "unproven",
                }
            ],
        },
        type("Inference", (), {"usage": {}})(),
    )
    assert result["evolve_success"] is False


def test_reader_suspicion_is_not_a_value_gate_anchor():
    result = EvolveResult(
        proposal=Proposal(test_name="t", test_code="assert False"),
        success=True,
        evidence="reader",
        reproduce=CommandResult("pytest", 1, "IndexError"),
    )
    assert value_gate_anchored(result) is False
    triage = EvolveResult(
        proposal=Proposal(test_name="t", test_code="c"),
        success=True,
        evidence="triage",
        reproduce=CommandResult("pytest", 1, "failed"),
    )
    assert value_gate_anchored(triage) is True
    semantic = EvolveResult(
        proposal=Proposal(test_name="t", test_code="c"),
        success=True,
        evidence="semantic",
        reproduce=CommandResult("pytest", 1, "failed"),
    )
    assert value_gate_anchored(semantic) is True


def test_semantic_evidence_classifies_severity_from_rule():
    severity, name = classify_accepted_severity(
        None,
        None,
        [],
        "the model calls this cosmetic",
        semantic={"rule": "SEM_INPUT_MUTATION"},
    )

    assert severity == 1
    assert name == "behavioural defect"


def test_default_evolve_graph_runs_four_runtime_stages(tmp_path, monkeypatch):
    def ensure(state):
        return {
            **state,
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "evolve_mode": "pipeline",
        }

    def stage(name):
        def run(state, inference=None, **kwargs):
            return {
                **state,
                "pipeline_stages": [*state.get("pipeline_stages", []), name],
                "evolve_success": name == "fixer",
            }

        return run

    monkeypatch.setattr(agent_module, "ensure_repo", ensure)
    monkeypatch.setattr(agent_module, "scan_stage", stage("scan"))
    monkeypatch.setattr(agent_module, "reader_stage", stage("reader"))
    monkeypatch.setattr(agent_module, "verifier_stage", stage("verifier"))
    monkeypatch.setattr(agent_module, "fixer_stage", stage("fixer"))
    graph = EvolveAgent(config(), workspace=str(tmp_path / "workspace")).create_graph()

    result = graph.invoke(
        {"messages": [HumanMessage(content="Evolve FEDOT")]}
    )

    assert result["pipeline_stages"] == ["scan", "reader", "verifier", "fixer"]
    assert result["evolve_success"] is True


def test_empty_reader_passes_are_not_a_stage_failure():
    result = ReaderResult.from_passes(
        [[], []],
        scanned_files=True,
        leads=[],
    )
    assert result.failed is False
    assert result.empty is True
    assert result.reason == "reader found no defects"


def test_reader_stage_treats_none_as_empty_not_error(tmp_path, monkeypatch):
    source = tmp_path / "fedot" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "fedotllm.agents.evolve.pipeline_nodes.run_reader_pass",
        lambda *args, **kwargs: [],
    )
    result = reader_stage(
        {
            "repo_path": str(tmp_path),
            "workspace": str(tmp_path / "workspace"),
            "messages": [],
            "lint_findings": [
                {"file": "fedot/a.py", "line": 1, "rule": "B001", "message": "x"}
            ],
            "evolve_config": EvolveConfig(include_all_files=True).model_dump(),
        },
        type("Inference", (), {"usage": {}})(),
    )
    assert result["reader_empty"] is True
    assert result["leads"] == []
    assert result["reader_reason"] == "reader found no defects"


def test_reader_parses_schema_json_extras():
    from fedotllm.agents.evolve.reader import parse_reader_answer

    rows = parse_reader_answer(
        '{"none": false, "extras": [{"line": 4, "why": "mutates caller input"}]}',
        "fedot/a.py",
        [],
        "a\nb\nc\ndata.pop(0)\n",
    )
    assert rows[0]["line"] == 4
    assert rows[0]["verdict"] == "extra"


def test_evolve_config_reads_env_only_when_set(monkeypatch):
    monkeypatch.delenv("FEDOTLLM_EVOLVE_READER_PASSES", raising=False)
    base = EvolveConfig(reader_passes=3)
    assert EvolveConfig.from_env(base).reader_passes == 3
    monkeypatch.setenv("FEDOTLLM_EVOLVE_READER_PASSES", "2")
    assert EvolveConfig.from_env(base).reader_passes == 2


def test_legacy_probe_env_maps_onto_probe_gate(monkeypatch):
    monkeypatch.setenv("FEDOTLLM_EVOLVE_PROBE", "0")
    monkeypatch.delenv("FEDOTLLM_EVOLVE_PROBE_GATE", raising=False)
    assert EvolveConfig.from_env().probe_gate is False


def test_probe_findings_honor_probe_gate(monkeypatch, tmp_path):
    from fedotllm.agents.evolve.gates import probe_findings_section
    from fedotllm.agents.evolve.loop import apply_loop_flags

    called = []
    monkeypatch.setattr(
        "fedotllm.agents.evolve.probe.run_probe_cached",
        lambda *args, **kwargs: called.append(1) or [],
    )
    try:
        apply_loop_flags(EvolveConfig(probe_gate=False))
        assert probe_findings_section(tmp_path) == ""
        assert called == []
    finally:
        apply_loop_flags(EvolveConfig())


def test_templates_flag_from_config_not_only_env(tmp_path):
    from fedotllm.agents.evolve.evidence import (
        apply_evidence_flags,
        proven_defects_section,
    )
    from fedotllm.agents.evolve.templates import GeneratedTest

    ready = GeneratedTest(
        rule="B008",
        test_name="test_x",
        test_code="from fedot.a import x\n",
        target="fedot/a.py:1 B008",
    )

    def fake_cached(repo, py):
        return [ready]

    import fedotllm.agents.evolve.templates as templates_mod

    original = getattr(templates_mod, "proven_defects_cached", None)
    templates_mod.proven_defects_cached = fake_cached
    try:
        apply_evidence_flags(EvolveConfig(templates=False))
        assert proven_defects_section(tmp_path, "python") == ""
        apply_evidence_flags(EvolveConfig(templates=True))
        text = proven_defects_section(tmp_path, "python")
        assert "Proven defects" in text
    finally:
        apply_evidence_flags(EvolveConfig())
        if original is None:
            del templates_mod.proven_defects_cached
        else:
            templates_mod.proven_defects_cached = original


def test_max_failed_attempts_follows_config():
    from fedotllm.agents.evolve.evidence import exhausted_files
    from fedotllm.agents.evolve.loop import apply_loop_flags

    journal = [{"file": "fedot/a.py", "success": False} for _ in range(2)]
    try:
        apply_loop_flags(EvolveConfig(max_failed_attempts=4))
        assert "fedot/a.py" not in exhausted_files(journal)
        apply_loop_flags(EvolveConfig(max_failed_attempts=2))
        assert "fedot/a.py" in exhausted_files(journal)
    finally:
        apply_loop_flags(EvolveConfig())


def test_run_script_uses_configured_verify_timeout(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.verifier import run_script, verify_leads

    seen = {}

    def fake_run(*args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("fedotllm.agents.evolve.verifier.subprocess.run", fake_run)
    run_script("print(1)", tmp_path, "python", tmp_path, "x", timeout=9)
    assert seen["timeout"] == 9

    def fake_verify_lead(*args, **kwargs):
        seen["lead_timeout"] = kwargs.get("timeout")
        return {"file": "fedot/a.py", "line": 1, "status": "not testable"}

    monkeypatch.setattr(
        "fedotllm.agents.evolve.verifier.verify_lead", fake_verify_lead
    )
    verify_leads(
        type("Inference", (), {"usage": {}})(),
        tmp_path,
        [{"file": "fedot/a.py", "line": 1, "why": "x"}],
        "python",
        tmp_path,
        config=EvolveConfig(verify_timeout=11, verify_attempts=1, public_attempts=1),
    )
    assert seen["lead_timeout"] == 11



def test_reader_model_list_defaults_to_one_model():
    from fedotllm.configs.schema import EvolveConfig

    cfg = EvolveConfig()
    assert cfg.reader_model_list("fallback/model") == ["fallback/model"]


def test_reader_model_list_prefers_the_explicit_single_model():
    from fedotllm.configs.schema import EvolveConfig

    cfg = EvolveConfig(reader_model="cheap/reader")
    assert cfg.reader_model_list("fallback/model") == ["cheap/reader"]


def test_reader_model_list_splits_several_models():
    """Each pass gets its own model — that is the point of the field."""
    from fedotllm.configs.schema import EvolveConfig

    cfg = EvolveConfig(reader_models=" cheap/one , free/two ")
    assert cfg.reader_model_list("fallback/model") == ["cheap/one", "free/two"]


def test_reader_model_list_ignores_an_empty_field():
    from fedotllm.configs.schema import EvolveConfig

    cfg = EvolveConfig(reader_models="  ,  ", reader_model="cheap/one")
    assert cfg.reader_model_list("fallback/model") == ["cheap/one"]

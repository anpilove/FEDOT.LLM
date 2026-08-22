from functools import partial

from langgraph.graph import END, START, StateGraph

from fedotllm.agents.base import Agent
from fedotllm.agents.evolve.pipeline_nodes import (
    fixer_stage,
    reader_stage,
    scan_stage,
    verifier_stage,
)
from fedotllm.agents.evolve.repo_setup import ensure_repo
from fedotllm.agents.evolve.state import EvolveAgentState
from fedotllm.configs.schema import AppConfig, EvolveConfig
from fedotllm.llm import AIInference


class EvolveAgent(Agent):
    """Find, verify, and repair a runtime defect in aimclub/FEDOT."""

    ENSURE_REPO = "ensure_repo"
    SCAN = "scan"
    READER = "reader"
    VERIFIER = "verifier"
    FIXER = "fixer"

    def __init__(self, config: AppConfig, workspace: str | None = None):
        self.evolve = EvolveConfig.from_env(config.evolve)
        self.workspace = workspace
        fallback = config.llm.model_name
        stage_model = {
            self.READER: self.evolve.reader_model or fallback,
            self.VERIFIER: self.evolve.verifier_model or fallback,
            self.FIXER: self.evolve.fixer_model or fallback,
        }
        self.inference = {
            stage: AIInference(
                config.llm.model_copy(update={"model_name": model}),
                config.session_id,
            )
            for stage, model in stage_model.items()
        }
        # One inference per reader pass. Different models disagree far more than
        # one model run twice — measured on ten FEDOT files, two models shared
        # 3 of 16 suspicions — so a second model is a second pass, not a
        # replacement, and a free one costs only wall time.
        self.reader_inferences = [
            AIInference(
                config.llm.model_copy(update={"model_name": model}),
                config.session_id,
            )
            for model in self.evolve.reader_model_list(fallback)
        ]

    def create_graph(self):
        workflow = StateGraph(EvolveAgentState)

        def _ensure(state: EvolveAgentState) -> EvolveAgentState:
            extra = {"evolve_config": self.evolve.model_dump()}
            if self.workspace and not state.get("workspace"):
                extra["workspace"] = str(self.workspace)
            state = {**state, **extra}
            return ensure_repo(state)

        llm_nodes = {
            self.READER: reader_stage,
            self.VERIFIER: verifier_stage,
            self.FIXER: fixer_stage,
        }
        workflow.add_node(self.ENSURE_REPO, _ensure)
        workflow.add_node(self.SCAN, partial(scan_stage, evolve_cfg=self.evolve))
        for stage, fn in llm_nodes.items():
            extra = ({"inferences": self.reader_inferences}
                     if stage == self.READER else {})
            workflow.add_node(
                stage,
                partial(fn, inference=self.inference[stage],
                        evolve_cfg=self.evolve, **extra),
            )

        workflow.add_edge(START, self.ENSURE_REPO)
        workflow.add_edge(self.ENSURE_REPO, self.SCAN)
        workflow.add_edge(self.SCAN, self.READER)
        workflow.add_edge(self.READER, self.VERIFIER)
        workflow.add_edge(self.VERIFIER, self.FIXER)
        workflow.add_edge(self.FIXER, END)
        return workflow.compile().with_config(run_name=EvolveAgent)

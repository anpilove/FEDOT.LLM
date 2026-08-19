from functools import partial

from langgraph.graph import END, START, StateGraph

from fedotllm.agents.base import Agent
from fedotllm.agents.evolve.nodes import (
    answer,
    collect_context,
    ensure_repo,
    route_after_ensure,
    run_evolve,
)
from fedotllm.agents.evolve.state import EvolveAgentState
from fedotllm.configs.schema import AppConfig
from fedotllm.llm import AIInference

ENSURE_REPO = "ensure_repo"
COLLECT_CONTEXT = "collect_context"
ANSWER = "answer"
EVOLVE = "run_evolve"


class EvolveAgent(Agent):
    """Background framework-evolution agent for aimclub/FEDOT.

    Default: ensure_repo → run_evolve → END
    QA mode (FEDOTLLM_EVOLVE_MODE=qa): ensure_repo → collect_context → answer → END
    """

    def __init__(self, config: AppConfig, workspace: str | None = None):
        self.inference = AIInference(config.llm, config.session_id)
        self.workspace = workspace

    def create_graph(self):
        workflow = StateGraph(EvolveAgentState)

        def _ensure(state: EvolveAgentState) -> EvolveAgentState:
            if self.workspace and not state.get("workspace"):
                state = {**state, "workspace": str(self.workspace)}
            return ensure_repo(state)

        workflow.add_node(ENSURE_REPO, _ensure)
        workflow.add_node(COLLECT_CONTEXT, collect_context)
        workflow.add_node(ANSWER, partial(answer, inference=self.inference))
        workflow.add_node(EVOLVE, partial(run_evolve, inference=self.inference))

        workflow.add_edge(START, ENSURE_REPO)
        workflow.add_conditional_edges(
            ENSURE_REPO,
            route_after_ensure,
            {"evolve": EVOLVE, "qa": COLLECT_CONTEXT},
        )
        workflow.add_edge(COLLECT_CONTEXT, ANSWER)
        workflow.add_edge(ANSWER, END)
        workflow.add_edge(EVOLVE, END)
        return workflow.compile().with_config(run_name=EvolveAgent)

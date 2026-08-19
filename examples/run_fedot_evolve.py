#!/usr/bin/env python3
"""Run full-repo FEDOT evolution via FEDOT.LLM EvolveAgent + OpenRouter preset.

Env:
  FEDOTLLM_LLM_API_KEY   OpenRouter (or compatible) key
  FEDOTLLM_REPO_PATH     local aimclub/FEDOT checkout (recommended)
  FEDOTLLM_REPO_PYTHON   python with fedot+pytest (recommended)
  FEDOTLLM_EVOLVE_MODE   evolve (default) | qa

Example:
  export FEDOTLLM_LLM_API_KEY=sk-or-v1-...
  export FEDOTLLM_REPO_PATH=../AutoDS-Tools/runs/fedot-evolve-demo/fedot
  export FEDOTLLM_REPO_PYTHON=../AutoDS-Tools/runs/fedot-evolve-demo/.venv-fedot/bin/python
  uv run python examples/run_fedot_evolve.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.messages import HumanMessage

from fedotllm.agents.evolve import EvolveAgent
from fedotllm.configs.loader import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="FEDOT.LLM EvolveAgent demo")
    ap.add_argument(
        "--presets",
        default="fedotllm:openrouter",
        help="Config preset (default: fedotllm:openrouter)",
    )
    ap.add_argument(
        "--workspace",
        type=Path,
        default=Path("fedotllm-output-evolve"),
        help="Where to write evolution_audit.md",
    )
    ap.add_argument(
        "--message",
        default=(
            "Please evolve the aimclub/FEDOT library: find one small safe "
            "validation/error-handling improvement, patch it, and validate with pytest."
        ),
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="OmegaConf override, e.g. llm.model_name=openai/gpt-4o (repeatable)",
    )
    args = ap.parse_args()

    if not os.environ.get("FEDOTLLM_LLM_API_KEY"):
        print("FEDOTLLM_LLM_API_KEY is not set", file=sys.stderr)
        return 2

    config = load_config(presets=args.presets, overrides=args.override or None)
    args.workspace.mkdir(parents=True, exist_ok=True)
    graph = EvolveAgent(
        config=config, workspace=str(args.workspace.resolve())
    ).create_graph()
    result = graph.invoke({"messages": [HumanMessage(content=args.message)]})
    success = bool(result.get("evolve_success"))
    audit = result.get("audit_path", "")
    print(f"[evolve] {'SUCCESS' if success else 'INCOMPLETE'} — audit: {audit}")
    if result.get("messages"):
        content = getattr(result["messages"][-1], "content", "") or ""
        print(content[:1500])
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

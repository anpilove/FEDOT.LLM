# FEDOT.LLM

Личный форк [aimclub/FEDOT.LLM](https://github.com/aimclub/FEDOT.LLM).

## EvolveAgent (thesis prod)

Поиск патчей FEDOT + hour quality evaluation. **Документация и architecture map:**

**[fedotllm/agents/evolve/README.md](fedotllm/agents/evolve/README.md)**

```bash
python -m fedotllm.agents.evolve doctor --fedot /path/to/FEDOT
python -m fedotllm.agents.evolve run --fedot /path/to/FEDOT --workspace /tmp/evolve-run
```

Research campaign artifacts: **[research/evolve/README.md](research/evolve/README.md)**

## Upstream (не thesis entry)

- `fedotllm/main.py` — FedotAI (Supervisor / Researcher / AutoML)
- `fedotllm/agents/automl/` — LangGraph AutoML codegen
- Streamlit / Docker — как в upstream

## Установка

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv sync
# FEDOTLLM_LLM_API_KEY в .env (не коммитить)
```

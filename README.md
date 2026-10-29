# FEDOT.LLM

Личный форк [aimclub/FEDOT.LLM](https://github.com/aimclub/FEDOT.LLM).

## EvolveAgent (thesis prod)

Поиск патчей FEDOT + hour quality evaluation. **Документация и architecture map:**

**[fedotllm/agents/evolve/README.md](fedotllm/agents/evolve/README.md)**

Цели, ограничения и схема решения: [docs/ml_system_design_doc.md](docs/ml_system_design_doc.md).

Перед коммитом включите pre-commit. Он запускает Ruff `0.11.13`: `ruff-check` и `ruff-format`.

```bash
uvx --from pre-commit==4.2.0 pre-commit install
uvx --from pre-commit==4.2.0 pre-commit run --all-files
```

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

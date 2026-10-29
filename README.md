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

## Проверки перед коммитом

После установки зависимостей включите Git-хуки в своей рабочей копии:

```bash
uv run pre-commit install
```

При `git commit` pre-commit запускает Ruff для добавленных в коммит Python-файлов:
`ruff-check` проверяет ошибки кода, `ruff-format` исправляет форматирование.
Если форматер изменил файл, добавьте исправления через `git add` и повторите коммит.
Версия Ruff зафиксирована одинаково в зависимостях и конфигурации хуков.

Проверить весь отслеживаемый код вручную:

```bash
uv run pre-commit run --all-files
```

Эта команда может изменить форматирование файлов. Для проверки без изменений:

```bash
uv run ruff check .
uv run ruff format --check .
```

В существующем коде остаются замечания линтера и форматирования; подключение
хуков само по себе их не исправляет. В GitHub Actions запускается та же
конфигурация pre-commit для всех отслеживаемых файлов.

## Дизайн системы

[Дизайн-документ EvolveAgent](docs/ml_system_design_doc.md): цели, требования,
ограничения, архитектура и протокол проверки улучшений FEDOT.

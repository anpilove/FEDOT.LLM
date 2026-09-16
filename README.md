# FEDOT.LLM — EvolveAgent

Личный форк [aimclub/FEDOT.LLM](https://github.com/aimclub/FEDOT.LLM): агент ищет правки исходников [FEDOT](https://github.com/aimclub/FEDOT) и ставит их в очередь на часовой `Fedot(best_quality)`. Продуктовый LLM-AutoML API (Supervisor / Streamlit) здесь не точка входа.

Модель фиксирована: `z-ai/glm-5.3-flash` на Scout, Verifier и Fixer. Fallback и апгрейд класса модели запрещены — тот же preset до и после изменения агента.

## Два контура

```text
hunt (LLM)                         controller / server
─────────                          ───────────────────
FEDOT source → lead → patch        stock Fedot(1h) → cache
cheap probe / toy screen           quality-job читает очередь
     │                             stock из кэша vs patch
     ▼                             KEEP только по часовым δ
enqueue, hunt качество не судит
```

**Hunt** (`python -m fedotllm.agents.evolve run`) читает checkout FEDOT, предлагает патч, гоняет дешёвый технический экран и кладёт валидный кандидат в `<workspace>/quality_queue/<id>.json` + `.patch`. Статус hunt: `queued_for_fedot_quality`. Часовой Fedot hunt не запускает.

**Controller / `quality-job`** — единственный судья качества. Сначала (или заранее) считает stock `Fedot(preset=best_quality, timeout=3600s, with_tuning)` на полных OpenML-задачах и пишет кэш `EVOLVE_QUALITY_STOCK_CACHE`. Потом сравнивает патч с этим кэшем. Composing, который не стартовал (`search_ran=false`), — инфраструктурная ошибка, не DROP по метрике.

### Дешёвый экран не veto

Toy / frozen PipelineBuilder / behavior-probe `no_change` (δ=0) **не отклоняет** кандидата. Это только приоритет очереди:

| probe / toy                         | priority | эффект          |
|-------------------------------------|----------|-----------------|
| `changed` или toy `early_gain`      | `high`   | раньше в drain  |
| `no_change`, toy δ=0                | `normal` | всё равно в очередь |

Блокирует только технический брак: патч не применился, patched crash, невалидный probe. См. `queue_priority` и `compare_behavior_probe`.

### Реестр качества

`fedotllm/agents/evolve/evaluation/quality_registry.json`: 24 полные classification-задачи, official OpenML fold 0 (`repeat=0`), плюс 3 полные public TS-серии FEDOT (`fedot-ts-beer`, `fedot-ts-australia`, `fedot-ts-salaries`). OpenML-правило зафиксировано до скоров: первые 20 task_id из OpenML-CC18 (study 99) плюс уже зарегистрированные AMLB/CC18 вне этого префикса (`10101`, `3917`, `9952`, `146818`). Не toy CSV и не весь 72-набор.

Стартовый набор — весь реестр. Default `quality-drain` гоняет 24 OpenML на табличные патчи и TS-задачи на TS-патчи; неподходящие task_id пропускаются и не становятся verdict. KEEP: хотя бы одна применимая задача с δ ≥ `min_delta` и ни одной регрессии сильнее порога.

## Research vs prod

| Prod (точка входа) | Research (не вход) |
|---|---|
| `python -m fedotllm.agents.evolve …` | `research/`, `docs/evolve/` notes, `evolve-artifacts/` |
| `fedotllm/agents/evolve/` | ночные брифы, campaign dumps, live_run |
| `tests/unit/agents/test_evolve_*.py` | `tests/research/` |

`research/` не импортируется как пакет агента. Скрипты кампании и клон FEDOT внутри research не коммитятся.

## Запуск

Нужны checkout FEDOT и ключ `FEDOTLLM_LLM_API_KEY` (OpenRouter). Секреты только в `.env`, не в репозитории.

```bash
# среда и контракт модели / FEDOT
.venv/bin/python -m fedotllm.agents.evolve doctor --fedot /path/to/FEDOT

# hunt: патчи в очередь, без часового Fedot
.venv/bin/python -m fedotllm.agents.evolve run \
  --fedot /path/to/FEDOT \
  --workspace /tmp/evolve-agent-run \
  --presets fedotllm:openrouter

# сервер: один раз прогреть stock-кэш (~1 ч на задачу, параллельно по cpu-quota)
export EVOLVE_QUALITY_STOCK_CACHE=/path/to/stock_cache
.venv/bin/python -m fedotllm.agents.evolve quality-job \
  --stock-only \
  --fedot /path/to/FEDOT \
  --workspace /path/to/quality_run \
  --n-jobs 10 --cpu-quota 32

# сервер: сравнить конкретный checkout патча с кэшем
.venv/bin/python -m fedotllm.agents.evolve quality-job \
  --fedot /path/to/FEDOT \
  --patch-checkout /path/to/patched-fedot \
  --workspace /path/to/quality_run
```

`run` пишет `<workspace>/runs/<timestamp>-<id>/` и `latest_run.json`. Очередь: `<workspace>/quality_queue/`.

## Перед релизом

Без LLM и без часового Fedot:

```bash
.venv/bin/python -m pytest -q tests/unit/agents/test_evolve_fedot_quality.py \
  tests/unit/agents/test_evolve_model_contract.py \
  tests/unit/agents/test_evolve_review_regressions.py

.venv/bin/python -m fedotllm.agents.evolve doctor --fedot /path/to/FEDOT
```

Проверить вручную:

1. `fedotllm/configs/openrouter.yaml` и `model_contract.py`: везде `z-ai/glm-5.3-flash`, `fallback_models: ""`.
2. Hunt после валидного патча оставляет `quality_queue/*.json` со `status=queued`, а не вызывает `quality-job` inline.
3. Повторный `quality-job` на том же registry/n_jobs читает stock из `EVOLVE_QUALITY_STOCK_CACHE`, не пересчитывает час.
4. `queue_priority(no_change, toy_metric_moved=False) == "normal"` — кандидат не отброшен.

Полный pytest продукта: `.venv/bin/python -m pytest -q`.

## CLI (остальное)

| команда | зачем |
|---|---|
| `doctor` | checkout, импорт, smoke evaluator |
| `run` | hunt + очередь |
| `quality-job` | stock-кэш или stock vs patch |
| `quality-drain` | stock всего реестра, затем очередь патчей |
| `continue` | продолжить ветку из старого workspace |
| `leads` | обход repo map, без тестов |
| `findings` / `scoreboard` / `replay` | журнал, не вход охоты |
| `benchmark` | компонентные регрессии; LLM только с `--allow-llm` |

## Установка (продукт)

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv sync
# ключи: FEDOTLLM_LLM_API_KEY в .env (файл не коммитить)
```

Docker / Streamlit / `FedotAI` — как в upstream. Этот README про evolve-контур.

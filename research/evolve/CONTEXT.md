# Evolve: главная документация (контекст)

Живой канон для новых чатов. Research only. Не точка входа prod.
Дата среза: 2026-08-26 (тик research: silent-лиды). Автор: Кирилл.

Prod `fedotllm/` не смешивать. Коммит/push только по явной просьбе.
Шпаргалку патчей (`_local_fedot_patches/`, `library.py`) в промпт агента не класть.

**LLM не видит gym и не думает про gym.** Harness (Python) вызывает скорер сам. В промпт не попадают: `gym/`, `fedot_quality.py`, `cases.json` целиком, поле `bug`, `replacements/`, канон «leftover-idx / isolated catboost ~0.85». Модель видит только checkout FEDOT + то, что вернули typed tools: traceback процесса, куски исходников FEDOT, скаляр score/status если tool это отдал. Нет tool `read_gym`. `guard.py` режет запись и чтение путей evaluator/data.

Старые срезы (не канон): `docs/ENVOLE_AGENT.md` (2026-08-25), `research/evolve/DEVELOPMENT_REPORT.md` (агент A), `research/evolve/CURSOR_IMPROVE_LOOP.md` (lint/gold-set), `research/evolve/archive/2026-08-20/`.
**Hunt:** агент не ищет одну подстроку и не стартует с тестов/экзамена. Walk `fedot/core` целиком. Pytest и holdout — harness после патча.

Практический план: [`research/evolve/AGENT_IMPROVEMENT_PLAN.md`](AGENT_IMPROVEMENT_PLAN.md).
Как собирать: **workflow** (фиксированный Python-цикл), не OpenHands и не AUC в verifier A. Код — `research/evolve/metric_agent/`. Каталог графов уже есть: `data/cases.json` (`pca->catboost`, `catboost`, `fast_ica->lgbm`). Скорер — перенос из `_local_fedot_patches/gym/` без `library.py`. LLM только `propose_patch`.

---

## Цель

Внешний агент (в FEDOT.LLM) патчит **исходники библиотеки FEDOT**.
Пользователь по-прежнему вызывает stock `Fedot.fit()` — в runtime FEDOT нет LLM.

Патч оставляют только если **замороженная метрика задачи** на holdout лучше stock.

```python
def keep(patched, stock, others_patched, others_stock, *,
         higher_is_better, min_delta, sentinel):
    p = sentinel if patched is None else patched
    s = sentinel if stock is None else stock
    delta = (p - s) if higher_is_better else (s - p)
    if delta < min_delta:
        return False
    for op, os_ in zip(others_patched, others_stock):
        od = (op - os_) if higher_is_better else (os_ - op)
        if od < -min_delta:
            return False
    return True
```

Задача = `(данные, граф/workload, metric, higher_is_better, sentinel, min_delta)`.
Меняется только evaluator `h()`, не архитектура агента.

Первый экзамен (не вся цель): official scoring 20000/4000, ROC-AUC,
crash = 0.5, `min_delta=0.01`, граф `pca→catboost`, must-not-regress `fast_ica→lgbm`.
Для RMSE sentinel и порог другие — не копировать 0.5/0.01.

Успех: Δметрики на holdout. Не: число багов, lint, pytest fail→pass, LACE-пайплайн.

---

## Не цель

| | Почему нет |
|---|---|
| **A. Текущий EvolveAgent** `scan → reader → verifier → fixer` | Оракул = reproduce-тест. Молчаливые quality-баги (knn без scale, bernb threshold) отваливаются. |
| **C. LACE / AutoMLGen / AIDE / AutoGluon Assistant** | Эволюционируют *пайплайн задачи* (`train.py`, sklearn class), не исходники FEDOT. |
| FormulaCode-задачи на sklearn/pandas | Метрика = **время ASV**, не качество модели. |
| Болтать AUC на текущий verifier | «Сначала напиши падающий тест» фильтрует не те баги. |

Баги и crash-графы — частный случай цели (метрика с 0.5 до ~0.85), не вся цель.

---

## Как должен работать агент (B)

Объект = diff в checkout FEDOT. Fitness = `h(task)` на holdout. Evaluator заморожен и **вне** дерева, которое патчит LLM (`gym/` / `cases.json` / скорер — read-only).

```
stock-run графа → худший crash/chance
→ fixer видит граф + traceback + исходник кадра
→ SEARCH/REPLACE
→ cascade: apply → pca→catboost → fast_ica→lgbm → полный cases.json только финалистам
→ keep / revert
```

Оставить от A: disposable FEDOT checkout, `apply_source`, journal rejected diffs.
Выкинуть с critical path: tree-wide lint/reader, «тест обязан падать на pristine».

Pytest = регрессия после keep, не доказательство.

Каскад как у AlphaEvolve: дешёвый фильтр, потом дорогой `Fedot.fit`.
Timeout живого графа ≠ sentinel 0.5 (иначе эволюция любит зависать).

---

## Tool architecture и deep research

Подробный разбор похожих tools, ranking и staged plan теперь живёт только в [`AGENT_IMPROVEMENT_PLAN.md`](AGENT_IMPROVEMENT_PLAN.md).

Короткий канон:

- Лучший путь для B: custom Python workflow + typed domain tools + frozen evaluator + replay journal.
- LLM делает только `propose_patch`; `run_stock`, `run_patched`, `compare`, `guard`, `revert`, `journal` — детерминированный код.
- Не клонировать OpenHands/SWE-agent/OpenEvolve целиком: они полезны как источники архитектурных идей, но не закрывают `Fedot.fit()` metric gate.
- Сначала single-candidate runner, потом context builder, paired table и только затем population/islands.

---

## Первый полигон: leftover-idx

После optional one-hot устаревают `categorical_idx` / `numerical_idx` / `encoded_idx`.
Downstream (pca, kernel_pca, fast_ica, rfe, poly, impute, …) сужает таблицу → IndexError или чужие столбцы.

Канон scoring 20k/4k: isolated catboost ~0.85; `pca→catboost` crash (0.5). Must-not-regress: `fast_ica→lgbm`.

Семья одна — **фиксы не сливать**. Не копировать PCA keep-cats на FastICA. Не копировать dense-cat между LGBM/CatBoost/XGB/RF.

Это экзамен агента **без шпаргалки**, не продукт целиком.

Silent-quality (knn без scale, bernb threshold) — **второй тип лида**, не мешать в первый crash-тик. Traceback нет: лид = два замера `h()` на одном holdout.

```python
# crash-лид: exception → кадр стека
# silent-лид: тот же task, два графа; оракул = разница метрик, не тест
if auc("knn") + min_delta < auc("lgbm"):
    lead = ("knn_path", auc_knn, auc_lgbm)  # чинить knn/preproc, не lgbm
```

Соседи (не копировать домен): **EKKA** — silent error = расхождение с reference (HF vs vLLM), не crash; **TransFuzz** — silent bugs в PyTorch/TF, 96% не ловит CPU↔GPU diff, нужен свой оракул. У нас reference — сильный узел (lgbm) или sklearn-эквивалент оператора, не pytest.

---

## Где код (не смешивать)

| | Что | Статус |
|---|---|---|
| `fedotllm/agents/evolve/` | EvolveAgent A (`ensure_repo → scan → reader → verifier → fixer`) | Не менять под B, пока явно не попросят. Last commit агента `170008d`. Грязный незакоммиченный рефактор уже был. |
| Draft PR `cursor/evolve-agent-metrics-ec2f` | был leftover-idx + gym | Сброшен на `main`. Уникальных файлов нет. Можно закрыть. |
| `_local_fedot_patches/` (не в git) | снимок PR: `from_pr/research/evolve/library.py`, `gym/`, `replacements/`, HANDOFF | Не коммитить. Не в промпт агента. `gym/` сейчас считает **только stock**. |
| `data/cases.json` | каталог графов, `holdout_roc_auc` | Research. |

FEDOT 0.7.5 stock. CSV scoring не в git (`/tmp/fedot-official-scoring` или `FEDOT_SCORING_CACHE`).

---

## Запреты

- PCA keep-cats → FastICA; merge poly/imputation leftover-фиксов; dense-cat копипаст LGBM/CatBoost/XGB/RF
- QDA/LDA как канон (mini-split артефакт)
- Шпаргалка `replacements/` / `library.py` в промпт
- Research-скрипты как prod entrypoint; вторые таблицы/v2 без просьбы
- Эволюционировать sklearn-Pipeline вместо FEDOT
- Дать LLM редактировать evaluator / `cases.json` / `roc_auc`
- Коммит `_local_fedot_patches`

---

## Следующий шаг (когда скажут кодить)

Research-пакет, не прод: замороженный evaluator из `gym/` + fixer на один граф + cascade + journal. Без scan/reader как backbone. Первый экзамен: `pca→catboost` без cheat sheet.
)

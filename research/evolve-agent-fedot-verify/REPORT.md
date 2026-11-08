# Отчёт: независимая проверка EvolveAgent × FEDOT

Ветка: `cursor/evolve-agent-fedot-verify-01e8`  
База: `origin/research/evolve-agent-setup` @ `467957bd10a150559f8a79a204a100fd90fac6b5`  
Обновлено: 2026-11-08 15:39 Europe/Moscow  
Ничего не слито автоматически.

## Goal progress (plan `/goal`)

Спека: `docs/superpowers/specs/2026-11-08-evolve-agent-8h-improve-goal.md`

Закрыто в коде (с тестами):
1. Doctor/preflight pin API (`fedot_pin_preflight.py` + `return_best_policy.py`)
2. Worker: `KeyError`/missing isolated optimizer → `status=invalid`, `failure_kind=fedot_pin_incompatible`, `metric_valid=false`
3. Microbenchmark: явный hint про missing `dask`
4. Политика `refuse` (default) / `research_worktree` (offline only)

Ещё не exit идеала: полный registry A/A на Cloud Environment пользователя; upstream FEDOT `return_best` в pin.

## 0. Окружение (обязательное)

Пользователь готовит Cloud Environment. **Его и нужно использовать** для полного registry A/A / OpenML offline / quality-прогонов.

Состояние этого run:
- Environment id: `498342df-a151-11f1-b532-320a589b8025`
- Агент загружен **без** build (`build: null`)
- Последний SUCCEEDED build — системный recurring snapshot без кастомного Evolve/FEDOT/OpenML install
- В репо нет `environment.json`

В этой сессии сделан временный bootstrap (uv/Python 3.11, dask для import pin FEDOT) только чтобы не простаивать 8-часовой план. Целевой стек — env пользователя.

## 1. Исходная точка

| Артефакт | Значение | Статус |
|---|---|---|
| Рабочая ветка | `cursor/evolve-agent-fedot-verify-01e8` | создана, запушена |
| Task 6 `949f285` | нет на remote | блокер «чистой точки после Task 5/6/7» |
| FEDOT pin | `8990f28b0664c99f7eb2fddcee3931151fe16e79` | доступен |
| Manifest CSV | 13/13 SHA-256 на git-checkout | ok |
| PyPI `fedot==0.7.5` | без `real_cases` data | gap |
| OpenML offline | нет кэша на 32 задачи | gap |
| `FEDOTLLM_REPO_PATH` | `/tmp/fedot-verify/FEDOT` (+ symlink `.repo_cache/FEDOT`) | выставлен для тестов |

## 2. Цикл агента

| Набор | Результат |
|---|---|
| transfer/budget/isolation/recovery/review | **198 passed** |
| весь `tests/unit/agents/ -k evolve` | **1060 passed**, 3 fail без FEDOT path |
| после `FEDOTLLM_REPO_PATH` + `dask` | metric_study resume **passed**; microbenchmark stock contracts **passed**; `does_not_write_fedot` **passed** |
| focused DEV/FINAL/transfer/parallel/recover | **6 passed** |

Ключевые выводы (файл → сценарий → тест):

| Вывод | Файл | Тест |
|---|---|---|
| DEV ≠ FINAL acceptance | `test_evolve_review_regressions.py` | `test_dev_success_is_not_final_acceptance…` |
| DEV/FINAL/SHADOW дизъюнктны | `test_evolve_mvp.py` + `scorer.py` | `test_dev_final_indices_are_disjoint_and_repeatable` |
| FINAL без transfer не accept | `test_evolve_transfer.py` | `test_ordinary_final_without_transfer…` |
| FINAL reserve только FINAL | `test_evolve_transfer.py` | `test_final_reserve_is_available_only_to_final_stage` |
| Параллелизм ≤ 7 | `test_evolve_quality_cancellation.py` | `test_dataset_parallelism_never_exceeds_seven…` |
| Interrupt → recovery без метрики | `test_evolve_quality_reliability.py` | `test_unfinished_spawned_event_recovers…` |
| FINAL не в LLM memory | `storage/replay.py` | source-only hypotheses |

Microbenchmark blocker без env: pin FEDOT требует `dask` (`ModuleNotFoundError`) при import checkout. После установки dask + pin `scikit-learn<1.7` — green.

## 3. Benchmark-манифесты

Аудит: `evidence/manifest-audit.json`

- 40 задач: 24 classification / 8 regression / 8 ts_forecasting
- sources: 32 openml_task + 3 public_ts + 5 local_ts
- starter ↔ datasets: полное совпадение; selection ↔ openml_task: полное совпадение
- manifest hashes: 13 ok / 0 bad / 0 missing (на git FEDOT)
- registry не содержит FINAL-метрик; prompt-scan `keep_final`/`final_score` → 0 hits
- budgets: side 5400 / search 4500 / reserve 1500 / watchdog 10800; policy `return_best`; cache `disabled-sqlite-v1`

## 4. Улучшение качества в AutoML-поиске FEDOT

### Гипотеза
Worker EvolveAgent передаёт `timeout_policy='return_best'`, `final_fit_reserve_seconds`, `max_pipeline_evaluation_time`. Stock FEDOT @ pin отвечает `KeyError`. Без API search может съесть весь budget → нет scorable incumbent → нет честного A/A.

### Прототип (research only)
1. Budget arithmetic: `prototype/return_best_budget.py` (+ 4 unit)
2. Патч на worktree `/tmp/fedot-verify/FEDOT-return-best`:
   - diff: `patches/0001-return-best-timeout-policy.fedot.diff`
   - принимает worker kwargs
   - `ApiTime` вычитает `final_fit_reserve` из composing/tuning окна при `return_best`
   - `Fedot.fit_budget_outcome` атрибут-заглушка
3. Интеграционные тесты: `prototype/test_patched_fedot_return_best.py` → **3 passed**
4. Ограниченный A/A smoke (synthetic classification, timeout 0.15 min): scores `[1.0, 1.0]`, equal — см. `evidence/aa-smoke.json`

**Вывода об улучшении качества AutoML на registry нет.** A/A smoke проверяет только принимаемость API и измеримость при коротком бюджете. Полный registry A/A ждёт Cloud Environment + OpenML offline cache.

### Недостающие ассеты для полного эксперимента
1. Готовый Cloud Environment пользователя
2. OpenML offline cache (32 задачи)
3. Upstream/mergeable FEDOT `return_best` (+ optional `IsolatedEvoGraphOptimizer`)
4. Remote tip после локальных Task 5/6/7 (`949f285` отсутствует)

## 5. Команды

```bash
export FEDOTLLM_REPO_PATH=/tmp/fedot-verify/FEDOT
uv sync --python 3.11 --group dev
uv pip install --python .venv/bin/python 'dask<2026' 'dask-ml<2026' 'scikit-learn>=1.0,<1.7'
.venv/bin/python -m pytest tests/unit/agents/ -k evolve -q
.venv/bin/python -m pytest research/evolve-agent-fedot-verify/prototype -v
```

## 6. Изменённые файлы (только research)

- `research/evolve-agent-fedot-verify/REPORT.md`
- `research/evolve-agent-fedot-verify/prototype/*`
- `research/evolve-agent-fedot-verify/patches/*`
- `research/evolve-agent-fedot-verify/evidence/*`

Production EvolveAgent / контроллер / `main` не менялись. Merge не выполнялся.

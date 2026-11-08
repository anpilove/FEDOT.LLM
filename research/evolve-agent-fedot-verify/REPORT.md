# Отчёт: независимая проверка EvolveAgent × FEDOT

Ветка: `cursor/evolve-agent-fedot-verify-01e8`  
База: `origin/research/evolve-agent-setup` @ `467957bd10a150559f8a79a204a100fd90fac6b5`  
Время фиксации: 2026-11-08 15:29 Europe/Moscow  
Ничего не слито автоматически.

## 0. Окружение (обязательное)

Пользователь готовит Cloud Environment для этой проверки. **Его и нужно использовать** для A/A, FEDOT checkout, offline-ассетов и quality-прогонов — не собирать параллельный ad-hoc стек вместо него.

Состояние на момент этого run:
- Environment id: `498342df-a151-11f1-b532-320a589b8025` ([dashboard](https://cursor.com/dashboard/cloud-agents/environments/e/498342df-a151-11f1-b532-320a589b8025))
- Этот агент **не** загрузился из build (`build: null` в `environment-info`)
- Последний SUCCEEDED build `bld-20261108-14754188-…` — системный recurring snapshot без кастомного `install` под EvolveAgent/FEDOT/OpenML
- В репо нет `environment.json`

Правило для следующих шагов: дождаться готового env (или нового агента, загруженного из него); не объявлять quality-результат на самосборном venv.

## 1. Исходная точка (0,5 ч)

| Артефакт | Значение | Статус |
|---|---|---|
| Рабочая ветка | `cursor/evolve-agent-fedot-verify-01e8` | создана от setup, запушена |
| Локальный Task 6 `949f285` | отсутствует на remote | **блокер чистоты относительно локальных Task 5/6/7** |
| FEDOT pin (manifest) | `8990f28b0664c99f7eb2fddcee3931151fe16e79` | доступен на GitHub |
| FEDOT git checkout | `/tmp/fedot-verify/FEDOT` @ тот же SHA | 13/13 SHA-256 ассетов совпали |
| PyPI `fedot==0.7.5` | site-packages | **без** `examples/real_cases/data` |
| Python | 3.11.18 через `uv` | временный bootstrap этой сессии; целевой — env пользователя |
| OpenML offline cache | только `tests/fixtures/openml/task_31.xml` | **не подготовлен** для 32 OpenML задач |
| Teleport / ключи / платные API | не использовались | ок |

Вывод фазы 1: репозиторий и pin FEDOT доступны; полный quality-эксперимент на registry **невозможен** без готового Cloud Environment, OpenML offline-кэша и FEDOT API `timeout_policy=return_best`. Отчёт продолжен по плану.

## 2. Цикл агента (2 ч)

Наборы: `198 passed` на transfer/budget/isolation/recovery/review; плюс 6 фокусных сценариев.

| Вывод | Файл | Сценарий | Подтверждение |
|---|---|---|---|
| DEV ≠ FINAL acceptance | `tests/unit/agents/test_evolve_review_regressions.py` | `test_dev_success_is_not_final_acceptance_and_final_uses_ablated_edits` | PASSED |
| DEV/FINAL/SHADOW индексы дизъюнктны | `tests/unit/agents/test_evolve_mvp.py` | `test_dev_final_indices_are_disjoint_and_repeatable` | PASSED; код split: `fedotllm/agents/evolve/evaluation/scorer.py` `_single_csv_split` |
| FINAL без transfer не принимается | `tests/unit/agents/test_evolve_transfer.py` | `test_ordinary_final_without_transfer_never_evaluates_or_accepts` | PASSED |
| FINAL reserve только для FINAL | `tests/unit/agents/test_evolve_transfer.py` | `test_final_reserve_is_available_only_to_final_stage` | PASSED |
| Бюджет hunt/run конечен | `tests/unit/agents/test_evolve_run_budget.py`, `test_evolve_supervisor.py` | per-run budget / exhausted hunt | в пакете 198 |
| Параллелизм ≤ 7 пар | `tests/unit/agents/test_evolve_quality_cancellation.py` | `test_dataset_parallelism_never_exceeds_seven_even_with_large_quota` | PASSED |
| Восстановление после interrupt | `tests/unit/agents/test_evolve_quality_reliability.py` | `test_unfinished_spawned_event_recovers_as_interrupted_attempt` | PASSED; метрика не выдумывается |
| FINAL метрики не в LLM-памяти | `fedotllm/agents/evolve/storage/replay.py` | `recent_completed_hypotheses_from_findings` | source-only, «never include scores/FINAL» |
| Tuning isolation + recover | `tests/unit/agents/test_evolve_tuning_isolation.py` | budget/recover suite | в пакете 198 |

Команды:

```bash
uv sync --python 3.11 --group dev
.venv/bin/python -m pytest \
  tests/unit/agents/test_evolve_transfer.py \
  tests/unit/agents/test_evolve_run_budget.py \
  tests/unit/agents/test_evolve_quality_cancellation.py \
  tests/unit/agents/test_evolve_tuning_isolation.py \
  tests/unit/agents/test_evolve_quality_reliability.py \
  tests/unit/agents/test_evolve_review_regressions.py -q
# → 198 passed
```

## 3. Benchmark-манифесты (1,5 ч)

### `benchmark_manifest.json`

- `fedot_commit`: `8990f28b…`
- `data_root`: `examples/real_cases/data` (**внутри checkout FEDOT**, не FEDOT.LLM)
- split: seed `42`; single `[0.6,0.2,0.2]`; separate_test `[0.5,0.5]`; confirmation seeds `[42,43,44]`
- 13 файлов: **все хеши совпали** на git-checkout FEDOT; в PyPI-пакете — **0/13**

### `quality_registry.json`

- identity: `fedot-best_quality-side5400-fit4500s-…-isolated-tuning`
- бюджеты: side `5400`, search `4500`, reserve `1500`, watchdog `10800`
- policy: `fedot_timeout_policy=return_best`, cache `disabled-sqlite-v1`
- задачи: **24 classification + 8 regression + 8 ts_forecasting = 40**
- FINAL-метрик в манифесте нет; закрытые FINAL-значения не читались и модели не передавались

### Offline-ассеты

| Класс | Источник | Offline готовность |
|---|---|---|
| classification / regression OpenML (32) | OpenML tasks | нет подготовленного кэша (`EVOLVE_QUALITY_DATA_OFFLINE=1` упадёт) |
| local TS (5) | FEDOT `real_cases` | да, в git-checkout |
| public TS (3: beer/australia/salaries) | FEDOT public loaders | требуют runtime FEDOT assets / сеть при первом кэше |

Доп. тесты манифеста/OpenML isolation: `122 passed` (`test_evolve_fedot_quality.py`, openml/dataset isolation).

## 4. Одно улучшение качества в поиске AutoML (3 ч)

### Механизм

Quality worker (`fedotllm/agents/evolve/evaluation/_fedot_quality_worker.py`) передаёт в `Fedot`:

- `timeout_policy="return_best"`
- `final_fit_reserve_seconds`
- `max_pipeline_evaluation_time`
- опционально `IsolatedEvoGraphOptimizer`

Stock FEDOT @ pin / PyPI `0.7.5`:

```text
KeyError: "Invalid key parameters {'final_fit_reserve_seconds', 'timeout_policy'}"
ModuleNotFoundError: fedot.core.optimisers.isolated_gp_optimizer
```

Контракт описан в `docs/superpowers/specs/2026-09-30-fedot-budget-outcomes-design.md`, но **не реализован** в зафиксированном FEDOT.

### Гипотеза (проверяемая)

Если добавить в FEDOT политику `return_best` с явным `final_fit_reserve` и лимитом на кандидата, то при том же `side_timeout` worker чаще завершает сторону с fitted incumbent и валидной метрикой вместо `KeyError` / budget timeout без модели. Это необходимое условие честного A/A и последующих quality-сравнений, а не само доказательство роста ROC-AUC/RMSE/MAE.

### Прототип

- код: `research/evolve-agent-fedot-verify/prototype/return_best_budget.py`
- тесты: `research/evolve-agent-fedot-verify/prototype/test_return_best_budget.py` → **4 passed**
- sketch diff: `research/evolve-agent-fedot-verify/patches/0001-return-best-timeout-policy.sketch.diff`

### A/A / ограниченная проверка качества

**Не запускалась.** Недостающие ассеты/условия:

1. Готовый Cloud Environment пользователя (этот run без `build`)
2. OpenML offline cache для 32 задач registry
3. Реализованный в FEDOT API `timeout_policy=return_best` (+ isolated optimizer, если dispatcher `isolated-candidate-process-v1`)
4. Checkout FEDOT с `examples/real_cases/data` (не PyPI wheel) как `FEDOTLLM_REPO_PATH`
5. Локальный commit `949f285` / пост-Task5/7 tip — отсутствует на remote

**Вывода об улучшении качества нет.**

### Тест-план после появления ассетов

1. Применить минимальный FEDOT patch `return_best` на pin `8990f28b`, прогнать unit-тесты FEDOT + prototype.
2. Подготовить offline OpenML cache для starter subset (`openml-31`, `openml-10101`, …) и 2 local TS.
3. A/A stock vs stock на одном classification + одном TS с укороченным бюджетом; ожидание: обе стороны `metric_valid`, одинаковый identity.
4. Только затем — сравнение stock vs patched search budget на DEV; FINAL не открывать агенту.

## 5. Ограничения и изменённые файлы

Изменения только в research-дереве этой ветки:

- `research/evolve-agent-fedot-verify/REPORT.md`
- `research/evolve-agent-fedot-verify/prototype/*`
- `research/evolve-agent-fedot-verify/patches/*`
- `research/evolve-agent-fedot-verify/evidence/*`

Не менялись: контроллер EvolveAgent, production evaluation, `main`, локальная ветка пользователя `research/evolve-agent-setup` beyond fork point.

Хеши версий: `research/evolve-agent-fedot-verify/evidence/versions.txt`.

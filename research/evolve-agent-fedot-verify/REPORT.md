# Отчёт кампании: EvolveAgent → pilot ideal

Ветка: `cursor/evolve-agent-fedot-verify-01e8`  
База: `origin/research/evolve-agent-setup`  
Обновлено: 2026-11-08 16:45 Europe/Moscow  
Спека цели: `docs/superpowers/specs/2026-11-08-evolve-agent-8h-improve-goal.md`  
Версии: `evidence/versions.txt`  
Ничего не слито автоматически. FINAL-метрики модели не передавались.

## Журнал incompleteness

| Причина | Статус | Где |
|---|---|---|
| Stock FEDOT pin без `return_best` / isolated optimizer | **отказано doctor/worker** | `evidence/doctor-stock-pin.json` |
| OpenML offline cache на 32 задачи | отсутствует | блокирует registry A/A |
| Cloud Environment build | `null` в этом run | ждать env пользователя |
| Task 6 `949f285` | нет на remote | база = setup tip |
| Unrelated: real tuner isolation [regression] | fail на PyPI FEDOT env (pre-existing) | вне goal subset; зафиксировано в `evidence/all-evolve-unit.txt` |
| Quality KEEP на registry | не запускался | нет claim |

## Требования цели (evidence)

### 1. Явный исход стороны
- Worker: `KeyError` на `return_best` → `status=invalid`, `failure_kind=fedot_pin_incompatible`, `metric_valid=false`
- Missing `IsolatedEvoGraphOptimizer` → тот же контракт
- Код: `evaluation/_fedot_quality_worker.py`, `evaluation/fedot_pin_preflight.py`
- Тесты: `test_evolve_quality_worker_diagnostics.py` (pin invalid mapping)

### 2. Doctor/preflight early fail
- Pin API (`timeout_policy`/`final_fit_reserve`/`max_pipeline_evaluation_time`), isolated optimizer, dask import site
- **FEDOT path** (`FEDOTLLM_REPO_PATH` / cache)
- **Checksum assets** (`benchmark_manifest` real_cases)
- **Runtime dask**
- Все early gates собираются вместе (pin fail не скрывает assets)
- Код: `commands/doctor.py`, `fedot_pin_preflight.py`, `return_best_policy.py`
- Тесты: `test_evolve_fedot_pin_preflight.py`, `test_evolve_return_best_policy.py`
- Live: `evidence/doctor-stock-pin.json` (path/assets/dask ok, pin refuse)

### 3. DEV≠FINAL / budget / parallel / recovery
- Регрессия: `evidence/goal-regression.txt` → **59 passed**
- Включает disjoint DEV/FINAL, transfer, budget, parallelism≤7, interrupt recovery

### 4. Измеримый smoke A/A + путь к registry
- Synthetic: `evidence/aa-smoke.json`
- Local TS temperature: `evidence/aa-local-ts.json` (равный finite MAE)
- Документированный registry path: `REGISTRY_AA_PATH.md` (**без KEEP claim**)

### 5. Воспроизводимый отчёт
- Этот файл + `evidence/versions.txt` + incompleteness table + команды ниже

## Команды

```bash
export FEDOTLLM_REPO_PATH=/path/to/FEDOT@8990f28b
uv sync --python 3.11 --group dev
uv pip install --python .venv/bin/python 'dask<2026' 'scikit-learn>=1.0,<1.7'
.venv/bin/python -m fedotllm.agents.evolve doctor --source "$FEDOTLLM_REPO_PATH" --workspace /tmp/evolve-doctor
.venv/bin/python -m pytest tests/unit/agents/test_evolve_fedot_pin_preflight.py \
  tests/unit/agents/test_evolve_return_best_policy.py \
  tests/unit/agents/test_evolve_quality_worker_diagnostics.py -q
```

## Изменённые prod-пути EvolveAgent

- `fedotllm/agents/evolve/evaluation/fedot_pin_preflight.py`
- `fedotllm/agents/evolve/evaluation/return_best_policy.py`
- `fedotllm/agents/evolve/evaluation/_fedot_quality_worker.py`
- `fedotllm/agents/evolve/commands/doctor.py`
- `fedotllm/agents/evolve/benchmark/micro.py`
- соответствующие unit-тесты

Research-only: `research/evolve-agent-fedot-verify/**` (не prod entrypoint).

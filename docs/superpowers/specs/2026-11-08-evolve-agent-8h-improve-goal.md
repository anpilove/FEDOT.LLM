# Цель: 8 часов улучшения EvolveAgent (цикл к идеалу)

Дата: 2026-11-08 (Europe/Moscow)  
Ветка работы: `cursor/evolve-agent-fedot-verify-01e8` (отдельно от локальной setup)  
Опора: `fedotllm/agents/evolve/docs/ml_system_design_doc.md`  
Окружение: целевое Cloud Environment пользователя; ad-hoc bootstrap — только fallback

## Идеал (критерий остановки цикла)

EvolveAgent считается «достаточно близким к идеалу пилота», когда одновременно:

1. **Исход любой стороны явен:** нет необработанного `KeyError`/crash worker на stock FEDOT pin; timeout/invalid/infrastructure разведены.
2. **Doctor/preflight** отказывает рано, если pin FEDOT не поддерживает `timeout_policy=return_best`, нет `dask`, нет `FEDOTLLM_REPO_PATH`/ассетов.
3. **DEV≠FINAL**, FINAL не утекает в LLM, transfer/budget/parallelism≤7/recovery — зелёные на unit-наборе.
4. **Измеримый smoke A/A** на ≥1 local TS или synthetic + задокументированный путь к registry A/A (без claim KEEP без данных).
5. **Отчёт кампании** воспроизводим: версии, журнал, причина incompleteness.

Полный registry `KEEP` — цель пилота, но **не обязательный exit** этих 8 часов, если нет готового env/OpenML cache.

## Не цель этих 8 часов

- Смена LLM / fallback моделей  
- Автоmerge в FEDOT  
- Research-скрипты как prod entrypoint  
- Переписывание Scout/Fixer промптов без измеримого preflight (отложено)

## Распределение 8 часов (технические блоки, не календарь)

| Блок | Содержание | Exit критерий |
|---|---|---|
| **1.5 ч — Preflight hardening** | `doctor` + worker: детект отсутствия `return_best` / `IsolatedEvoGraphOptimizer` / `dask`; структурированный `invalid` вместо traceback | unit-тесты doctor/worker |
| **2 ч — Worker fail-closed** | Любой несовместимый pin → `metric_valid=false`, `failure_kind` стабильный; журнал без выдуманной метрики | тесты quality worker diagnostics |
| **1.5 ч — Checkout/runtime contract** | Явные требования pin+deps в controller install/smoke; microbenchmark не молчит про missing dask | microbench + smoke tests |
| **2 ч — Return_best bridge** | Минимальный совместимый слой: либо отказ с инструкцией, либо подключаемый research-patch path **без** merge в prod FEDOT; тесты на оба режима | prototype + worker tests |
| **1 ч — Отчёт и регрессия** | Прогон evolve unit subset; обновление `research/evolve-agent-fedot-verify/REPORT.md`; PR | suite green / зафиксированные gaps |

## Определение «лучше» на каждом витке

Виток засчитывается только если:

- есть failing→passing тест (или новый тест на уже зелёное инвариантное поведение), и  
- diff минимальный, в зоне `fedotllm/agents/evolve/` (+ research evidence), и  
- не открывает FINAL метрики модели.

## Порядок витков (фиксированный)

1. Doctor: pin API + deps  
2. Worker: KeyError → structured invalid  
3. Smoke/micro: dask/path messaging  
4. Return_best compatibility policy (explicit)  
5. Report + residual backlog к идеалу

## Риски

- Cloud Environment всё ещё `build: null` → preflight должен это диагностировать, не маскировать.  
- PyPI `fedot==0.7.5` ≠ git pin `8990f28b` → не смешивать в одном identity.  
- Нельзя объявлять quality win без registry/offline ассетов.

## Approval gate

Старт реализации — после явного «ок / начинай» по этой цели.  
Отклонения scope — только новой записью в этом файле.

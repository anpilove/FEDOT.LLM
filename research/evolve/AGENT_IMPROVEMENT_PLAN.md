# План улучшения metric-improvement агента FEDOT

Дата: 2026-08-26. Research only. Цель: спроектировать следующий агент B, который проходит по исходникам FEDOT, предлагает patch и оставляет его только если frozen holdout metric лучше stock.

## Короткий вывод

Полного готового инструмента “patch AutoML library source → improve model metric” в найденных источниках нет. Ближайшие аналоги делятся на три группы:

- coding agents для issue/test repair: SWE-agent, OpenHands, aider, debug-gym;
- code optimization benchmarks: FormulaCode, SWE-Perf;
- evolutionary code optimizers: AlphaEvolve, OpenEvolve, CodeEvolve.

Для нас это значит: не надо начинать с большого фреймворка эволюции или общего SWE-agent clone. Нужен свой маленький harness вокруг FEDOT:

```python
def accept_patch(target_stock, target_patched, other_pairs, *,
                 higher_is_better, min_delta, sentinel):
    def num(r):
        if r.status == "crash":
            return sentinel
        if r.status in {"timeout", "invalid"}:
            return None
        return r.score

    s, p = num(target_stock), num(target_patched)
    if s is None or p is None:
        return False
    delta = (p - s) if higher_is_better else (s - p)
    if delta < min_delta:
        return False
    for stock_r, patched_r in other_pairs:  # тот же task_id, stock vs patched
        a, b = num(stock_r), num(patched_r)
        if a is None or b is None:
            return False
        d = (b - a) if higher_is_better else (a - b)
        if d < -min_delta:
            return False
    return True
```

Keep — целевая метрика выросла на `min_delta`, контрольные task_id не просели. Timeout ≠ 0.5.

## Research scope

Вопрос: какие существующие tools/architectures ближе всего к агенту, который улучшает метрику FEDOT через patch исходников, и какой loop/tool architecture строить.

Что сравнивалось:

- coding-agent harness: SWE-agent, mini-SWE-agent, OpenHands, OpenAI Agents SDK, LangGraph, aider;
- code-search / repair systems: AutoCodeRover, Agentless, Moatless/SWE-Search, debug-gym;
- metric/evolution systems: AlphaEvolve, OpenEvolve, CodeEvolve;
- benchmark/evaluator designs: FormulaCode, SWE-Perf;
- negative controls: LACE, AIDE, AutoMLGen, multi-role frameworks.

Критерии fit:

| Criterion | Почему важно для FEDOT B |
|---|---|
| Patches repository source | Наша единица действия — diff в FEDOT, не notebook/pipeline. |
| Supports continuous metric | Нужен `ROC-AUC/RMSE/etc`, не только pass/fail. |
| Separates evaluator from writable code | Иначе agent может “улучшить” scorer, а не FEDOT. |
| Has structured context/search | FEDOT большой; весь repo в prompt не помещается и шумит. |
| Allows deterministic gates | `keep/drop` должен быть кодом, не мнением LLM. |
| Handles multiple workloads/regressions | One-task lift не доказывает улучшение библиотеки. |
| Replayable/auditable | Нужны exact commands, diffs, logs, scores. |
| Can scale later | После v0 нужны parallel candidates / population, но не раньше. |

## Evidence matrix

| Tool/system | Primary evidence | What it really proves | Fit to our goal | Decision |
|---|---|---|---|---|
| SWE-agent | Docs say it lets LMs use tools to fix real GitHub repositories and is configurable/research-oriented; paper frames this as agent-computer interface design. | Good ACI and repository-edit loop matter. | Medium: issue/test repair, not metric-improvement. | Copy ACI lessons, not full loop. |
| mini-SWE-agent | Project positions itself as a very small SWE-bench agent. | Simple loop can be enough; big framework is not required for v0. | Medium: minimality good, bash-first bad. | Copy simplicity, reject shell-only control. |
| OpenHands SDK | Docs split agent, tools, workspace, events, security policy, local/remote modes. | Clean component boundaries and sandbox abstraction. | Medium-high as reference architecture. | Use as design reference; do not adopt as dependency first. |
| OpenAI Agents SDK | Official docs support local runtime tools, function tools, guardrails/tripwires, tracing. | Typed tools + trace + local execution are mature patterns. | High if we later need SDK runtime. | Useful after custom v0; evaluator still custom. |
| LangGraph | Docs emphasize mixing deterministic steps with LLM-driven steps in one graph. | Good fit for long-running stateful workflows. | Medium now, higher later. | Do not start v0 with it; use if loop needs resume/parallel. |
| aider repo map | Docs describe concise repo map with symbols/call signatures and relevance ranking. | Context compression for large repos works. | High for context builder. | Implement aider-style FEDOT repo map. |
| AutoCodeRover | Paper/repo emphasize AST/class/method code search and patch localization. | Structure-aware search beats treating repo as files. | High for localization, low for metric gate. | Implement AST search tools around FEDOT operations. |
| Agentless | Repo describes localization → repair → validation. | Simple staged pipeline can outperform wandering agents. | High for v0 loop shape. | Use deterministic phases; one LLM step. |
| Moatless/SWE-Search | Project argues good tools insert right context and handle responses; SWE-Search explores search states. | Tool design and controlled state transitions matter. | Medium-high. | Use controlled states; leave MCTS for later. |
| debug-gym | Repo/blog expose pdb/bash/code viewers/edit/breakpoint tools for interactive debugging. | Traceback-led exploration is useful for crash cases. | High for crash tasks, low for silent-quality. | Add `inspect_trace`; optional pdb tool later. |
| AlphaEvolve | DeepMind/Google Cloud describe seed program + deterministic evaluator returning scalar metrics. | Best conceptual match for metric-based code improvement. | High conceptually, but closed/single-program oriented. | Copy evaluator-first + cascade; not EVOLVE-BLOCK dependency. |
| OpenEvolve | README describes prompt sampler, LLM ensemble, evaluator pool, program DB, MAP-Elites/islands. | Population search helps after evaluator is reliable. | Low for v0, high for v4. | Defer population. |
| CodeEvolve | Paper uses island GA, weighted LLM ensemble, meta-prompting and execution feedback. | Evolution machinery can improve exploration. | Later-stage only. | Use after paired table exists. |
| FormulaCode | Paper designs repo-level optimization with many workloads, correctness constraints, continuous metrics. | Benchmark design closest to “optimize codebase with metric”. | High as evaluator-design reference, metric differs. | Copy multi-workload discipline and score ledger. |
| SWE-Perf | Project uses real performance PRs, target functions, executable environment, verified speedup. | Optimization benchmark must validate real improvements statistically. | Medium: runtime not ML quality. | Copy baseline-vs-patched benchmark contract. |
| LACE/AIDE/AutoMLGen | They optimize user task pipeline/code. | LLM can improve ML task score, but object is wrong. | Low. | Negative control only. |

## Решение: best architecture под нашу цель

Лучший вариант для B: **custom Python workflow + typed domain tools + frozen evaluator + replay journal**.

Не лучший вариант: “взять OpenHands/SWE-agent/OpenEvolve целиком”. Они решают соседние задачи, но в них главный loop заточен под GitHub issue, tests или generic code evolution. У нас другой центр тяжести: `Fedot.fit()` на frozen tasks и deterministic `keep/drop`.

| Место в архитектуре | Лучший выбор | Почему |
|---|---|---|
| Main loop | обычный Python `for` / state machine | `compare()` и `guard()` нельзя отдавать LLM. Для v0 LangGraph не нужен. |
| LLM interface | один fixer-call с structured output | Агент должен предложить один patch, а не вести бесконечный чат с bash. **Не знает, что существует gym.** |
| Tools | свои typed functions | FEDOT scoring, traceback, repo map и apply patch доменные; generic shell слишком широкий. |
| Repo context | AST/repo-map + traceback frames | Aider/AutoCodeRover показывают, что structured context лучше “прочитай всё”. |
| Patch format | SEARCH/REPLACE сначала, unified diff fallback | Уже есть `apply_source`; проще валидировать и журналировать. |
| Evaluator | отдельный read-only process/package | Защита от reward hacking: агент не должен менять scorer/data. |
| Sandbox | disposable FEDOT checkout + optional Docker позже | Сначала локально быстрее; Docker нужен для reproducibility stage. |
| Observability | JSONL journal + per-candidate artifact dir | Нужны replay, rejected diffs, stdout/stderr, cost, exact scores. |
| Scaling | после v0: parallel candidates, затем population/islands | OpenEvolve/CodeEvolve полезны только после стабильного evaluator. |

## Match: goal -> loop -> tools

Наша цель не “find bug”. Цель: найти в FEDOT место, где изменение исходника улучшает metric на frozen workloads.

| Goal requirement | Loop step | Tool(s) | Gate |
|---|---|---|---|
| Увидеть baseline обычного FEDOT | stock scoring | `list_tasks`, `run_stock` | `ScoreResult` с `status`, `score`, `traceback`, `duration` |
| Локализовать место для правки | context build | `inspect_trace`, `repo_map`, `show_source` | context содержит target source и callers, но не oracle patches |
| Предложить одну правку | LLM patch proposal | `propose_patch` | structured candidate: target, rationale, patch text |
| Применить правку только в FEDOT | patch apply | `guard_path`, `apply_patch` | changed files внутри disposable checkout |
| Проверить целевую метрику | patched scoring | `run_patched(target)` | crash/timeout/invalid различимы |
| Проверить соседей | regression scoring | `run_patched(regression_task)` | no regression ниже `-min_delta` |
| Принять/отклонить без LLM | deterministic decision | `compare` | `delta >= min_delta` and no regression |
| Оставить воспроизводимый след | audit/replay | `write_journal`, `snapshot_diff` | replay command + env hash + diff + logs |

Цикл должен выглядеть так:

```python
def goal_loop(task_id, regression_ids, checkout, llm):
    task = list_tasks()[task_id]
    stock = run_stock(task.id)

    context = build_context(
        trace=inspect_trace(task.id) if stock.status == "crash" else None,
        repo=repo_map(task.search_query),
        source=show_source(task.seed_symbol),
    )

    candidate = propose_patch(llm, task=task, context=context)
    guard_path(candidate.patch, writable_root=checkout.path)
    apply_result = apply_patch(candidate, checkout)
    if not apply_result.ok:
        return drop(candidate, "patch_apply_failed")

    target = run_patched(task.id, checkout)
    regressions = [run_patched(t, checkout) for t in regression_ids]
    decision = compare(stock, target, regressions)
    write_journal({
        "task": task.id,
        "candidate": candidate.id,
        "stock": stock,
        "target": target,
        "regressions": regressions,
        "decision": decision,
        "diff": snapshot_diff(checkout),
    })
    if not decision.keep:
        revert_patch(checkout)
    return decision
```

Идея простая: LLM делает только `propose_patch`; всё остальное — проверяемый код.

## Tool/toolkit ranking

Оценка под нашу цель, не “лучший агент вообще”.

| Tool/system | Fit | Роль в нашем проекте | Почему не брать целиком |
|---|---:|---|---|
| Custom Python harness | 10/10 | Основной путь v0-v2 | Только он точно фиксирует `Fedot.fit` metric gate и запрет на scorer edits. |
| Aider-style repo map | 9/10 | Context builder | Aider сам по себе interactive pair programmer, не evaluator harness. |
| AutoCodeRover-style AST search | 9/10 | Локализация symbols/callers | Заточен под GitHub issues и tests, не ML metric. |
| OpenAI Agents SDK function tools + tracing | 8/10 | Если нужен typed tool runtime и trace UI | SDK не заменяет evaluator и path guards. |
| LangGraph | 7/10 | Позже, если loop станет stateful/long-running | Для v0 создаст лишнюю абстракцию; A уже разросся именно так. |
| debug-gym | 7/10 | Crash-localization идеи, возможно pdb tool | Не нужен для silent metric leads; судья там tests. |
| SWE-agent / mini-swe-agent | 6/10 | ACI lessons, minimal loop style | Shell-first и issue-first; guardrails придётся строить самим. |
| OpenHands | 6/10 | Sandbox/workspace architecture reference | Слишком широкий framework; цель не generic coding agent. |
| FormulaCode / SWE-Perf | 6/10 | Benchmark-design reference | Метрика runtime, не model quality; взять только workload discipline. |
| OpenEvolve / CodeEvolve | 5/10 now, 8/10 later | Population/islands после v2 | Без стабильного evaluator будет искать reward hacks. |
| LACE / AIDE / AutoMLGen | 3/10 | Negative control | Оптимизируют user pipeline, не FEDOT source. |
| CrewAI / AutoGen / MetaGPT style role agents | 2/10 | Не нужно | Много ролей не решают metric gate и усложняют replay. |

## Как собирать (решение)

Не клонировать OpenHands / SWE-agent / OpenEvolve. Не дописывать AUC в EvolveAgent A.

По Anthropic (*Building Effective Agents*): это **workflow** (фиксированный путь в коде), не открытый agent (LLM сам выбирает bash). Evaluator-optimizer, но «evaluator» — `Fedot.fit` + holdout, не LLM-as-judge. Mini-SWE-agent учит линейный цикл и `subprocess.run` без stateful shell; **bash-only брать нельзя**: LLM тогда допишет `gym/` / скорер. Берём линейный цикл + typed tools (урок ACI из полного SWE-agent: bounded output, абсолютные пути). LangGraph в A уже есть (`fedotllm/agents/evolve/agent.py`) и вокруг него вырос `scan→reader→verifier`. Для B v0 — обычный `for` в Python, чтобы `compare()` нельзя было обойти.

```python
def run_once(task_id, regress_ids, llm):
    stock = run_stock(task_id)
    ctx = inspect_trace(task_id) if stock.status == "crash" else repo_map(task_id)
    cand = llm.propose_patch(ctx)
    apply_patch(cand)
    patched = run_patched(task_id)
    others = [(run_stock(r), run_patched(r)) for r in regress_ids]
    decision = compare(stock, patched, others)
    if not decision.keep:
        revert_patch(cand)
    write_journal(...)
    return decision
```

Каскад: syntax/apply fail → стоп; target graph; `fast_ica->lgbm`; полный `cases.json` только keep-финалистам. Timeout ≠ sentinel 0.5.

### Куда класть код

Research-пакет, не prod: `research/evolve/metric_agent/` (имя можно сменить, дерево такое).

| Файл | Зачем |
|---|---|
| `types.py` | `TaskSpec`, `ScoreResult`, `Decision` |
| `tasks.py` | registry: `pca->catboost`, `catboost`, `fast_ica->lgbm` из gym `cases.json` |
| `eval.py` | `run_stock` / `run_patched`: PYTHONPATH = checkout, вызывает gym scorer |
| `checkout.py` | copy/isolate FEDOT; идея как `ensure_repo`, не импорт prod-графа A |
| `patch.py` | copy `apply_source` SEARCH/REPLACE (+ AST fallback); без `apply_test` |
| `context.py` | traceback frames + bounded source; без tree-wide reader |
| `guard.py` | deny-list путей: `gym/`, `cases.json`, `research/`, `fedotllm/` |
| `journal.py` | jsonl, идея `audit.append_journal` |
| `loop.py` | цикл выше; CLI `python -m research.evolve.metric_agent` |
| `tools.py` | схемы tools из таблицы ниже; LLM видит только их |

`fedotllm/` не трогать, пока этап 5.

Срез репо 2026-08-26 (что реально лежит, не «как задумано»):

| Артефакт | Где сейчас | Этап 0 |
|---|---|---|
| Каталог графов | **`data/cases.json`** уже в дереве; те же id в `_local_fedot_patches/gym/cases.json` | `tasks.py` читает `data/cases.json` (id как в файле: `pca->catboost`) |
| Скорер | только `_local_fedot_patches/gym/fedot_quality.py` + `util.py` + `cases.py` | скопировать в `research/evolve/` (scoring only). Сейчас `cases.py` импортирует `research.evolve.fedot_quality`, которого **нет** в tracked tree — gym **не runnable** |
| Monkeypatches | `_local_fedot_patches/from_pr/.../library.py`, `replacements/` | не копировать, не в промпт |

`eval.py` **не** ставит `PYTHONPATH` в текущем процессе: `fedot` уже импортирован — патч не подхватится. Stock и patched — отдельные `subprocess` с `PYTHONPATH=<checkout>` и timeout; hang → status `timeout`, не 0.5. CSV scoring не в git (`FEDOT_SCORING_CACHE` / `/tmp/fedot-official-scoring`).

Дыра для `inspect_trace`: gym сейчас пишет только `fit_or_predict_failed={type}: {exc}` без `traceback.format_exc()` (`fedot_quality.py`, ~строка 384). Без полного стека context builder не увидит кадр pca. Этап 0: в `ScoreResult` поле `traceback`, в скорере `traceback.format_exc()`, затем `inspect_trace` режет frames внутри checkout (не site-packages).

`patch.py` — копия `apply_source`, не `from fedotllm.agents.evolve.patching import ...`: иначе research B зависит от prod A.

### Что взять из репо, что выкинуть

| Взять | Файл в репо | Не брать |
|---|---|---|
| disposable checkout | `fedotllm/agents/evolve/repo_setup.py` → `ensure_repo` | `scan`, `reader`, `verifier`, lint seed, `agent.py` StateGraph |
| SEARCH/REPLACE + AST fallback | `fedotllm/agents/evolve/patching.py` → `apply_source` / `apply_source_ast` | `apply_test`, `Proposal.test_*` |
| jsonl journal | `fedotllm/agents/evolve/audit.py` → `append_journal` | journal как «этот файл уже чинили» для scout |
| ids графов | `data/cases.json`: `pca->catboost` (fail), `catboost` (reference), `fast_ica->lgbm` (must_not_regress) | `library.py`, `replacements/` |
| crash→status | `_local_fedot_patches/gym/cases.py` → `score_case` (`fit_or_predict_failed=` → crash; auc&lt;0.55 → chance) | chance как keep-сигнал на method-лидах (knn) — это этап silent, не этап 1 |
| LLM client | `fedotllm/llm.py` → `AIInference` | EvolveAgent `loop.py` как backbone |

Экзамен этапа 1: task_id **`pca->catboost`**, без шпаргалки. Канон: **`catboost`** ~0.85, crash=0.5, must-not-regress **`fast_ica->lgbm`**.

## Почему это не “deep clone” существующего tool

| Вариант | Почему кажется привлекательным | Почему проигрывает нашей цели |
|---|---|---|
| Взять OpenHands как runtime | Уже есть sandbox, tools, server, events. | Основная сложность не runtime, а FEDOT evaluator contract. Придётся всё равно писать scorer, guards, decision, journal. |
| Взять SWE-agent/mini-swe-agent | Уже умеет issue → patch. | Issue/test oracle не равен metric oracle; shell-first повышает риск поменять scorer. |
| Взять OpenEvolve | Уже оптимизирует code по evaluator. | Хорош для single program / algorithm search. FEDOT repo patch + multiple tasks + path guard сначала надо сделать самому. |
| Взять LangGraph | Уже есть graph orchestration. | У нас v0 линейный и должен быть труднообходимым: `compare()` deterministic. Graph полезен позже для resume/parallel. |
| Использовать текущий EvolveAgent A | Уже в `fedotllm/agents/evolve/`. | Его core invariant “reproduce failing test first” отсекает silent metric improvements. |

## Loop design по фазам

### Phase 0: evaluator-first

Цель: доказать, что scorer честный и воспроизводимый, до первого LLM patch.

```python
def phase0_eval_contract(tasks):
    results = {task.id: run_stock(task.id) for task in tasks}
    assert results["catboost"].status == "ok"
    assert results["pca->catboost"].status == "crash"
    assert results["fast_ica->lgbm"].status == "ok"
    assert guard_path("research/evolve/cases.json") == "deny"
    return results
```

Что делает формула: она проверяет, что базовые графы дают ожидаемые статусы, а guard запрещает редактировать evaluator.

### Phase 1: one candidate, one target

Цель: получить первый replayable decision, не “починить всё”.

```python
def phase1_one_candidate(task_id="pca->catboost"):
    stock = run_stock(task_id)
    ctx = context_from_traceback(stock.traceback)
    candidate = llm_patch(ctx)
    apply_patch(candidate)
    target = run_patched(task_id)
    guard = run_patched("fast_ica->lgbm")
    return compare(stock, target, [(run_stock("fast_ica->lgbm"), guard)])
```

Что делает формула: она проверяет одну правку на целевом графе и одном обязательном графе-регрессии.

### Phase 2: context quality

Цель: agent должен получать правильный контекст, а не весь FEDOT.

```python
def context_budget(traceback, query, max_tokens=8000):
    frames = traceback_symbols(traceback)
    ranked = repo_map(query, symbols=frames)
    return trim_to_budget(frames + ranked.callers + ranked.field_usages, max_tokens)
```

Что делает формула: она собирает source context из traceback, callers и usage полей индексов, затем режет его до бюджета.

### Phase 3: paired table

Цель: перейти от “один удачный patch” к evidence artifact.

```python
def paired_row(candidate, task_id):
    stock = run_stock(task_id)
    patched = run_patched(task_id)
    return {
        "candidate": candidate.id,
        "task": task_id,
        "stock": stock.score,
        "patched": patched.score,
        "delta": patched.score - stock.score,
        "status_pair": (stock.status, patched.status),
    }
```

Что делает формула: она сохраняет сравнение stock/patched для одной задачи так, чтобы результат можно было перепроверить.

### Phase 4: search expansion

Цель: после честного evaluator добавить exploration.

```python
def phase4_search(task_ids, families):
    candidates = []
    for family in families:
        for task_id in task_ids:
            candidates.append(run_once(task_id, family=family))
    return pareto_rank(candidates, keys=["target_delta", "regression_delta", "cost"])
```

Что делает формула: она пробует несколько семейств операций и ранжирует кандидатов не одной средней цифрой, а по нескольким критериям.

## Целевая архитектура

```
read-only gym/cases/scorer
        |
        v
task registry -> stock scorer -> context builder -> LLM patch proposal
                                                \-> repo map / traceback tools
        |
        v
patch applier in disposable FEDOT checkout
        |
        v
patched scorer -> regression scorer -> deterministic keep/drop -> journal
```

Главное разделение:

| Компонент | Можно редактировать агенту? | Ответственность |
|---|---:|---|
| `gym/`, `cases.json`, scorer, scoring cache | нет | frozen evaluator и защита от reward hacking |
| disposable FEDOT checkout | да | место, куда применяется patch |
| `fedotllm/` prod | нет на первом этапе | не смешивать research и продукт |
| journal | append-only | replay, rejected diffs, cost, source evidence |
| task registry | нет в run-time агента | список frozen workloads и thresholds |

## Минимальные tools

| Tool | Тип | Возврат | Правило |
|---|---|---|---|
| `list_tasks()` | read-only | task id, graph, dataset, metric, sentinel, threshold | Без данных train/test, только metadata. |
| `run_stock(task_id)` | evaluator | score/status/traceback/timing | Пишет только artifact в output dir. |
| `inspect_trace(task_id)` | read-only | traceback frames + bounded source snippets | Только для crash/exception. |
| `repo_map(query)` | read-only | ranked files/symbols/callers | Бюджетировать output, не весь repo. |
| `show_source(target)` | read-only | bounded source slice | Требовать точный symbol/path. |
| `propose_patch(target, rationale)` | journal | candidate id + structured rationale | Не пишет FEDOT. |
| `apply_patch(candidate_id)` | write FEDOT checkout | changed files + apply status | Запрет на файлы вне checkout. |
| `run_patched(task_id)` | evaluator | score/status/traceback/timing | Evaluator read-only. |
| `compare(candidate_id)` | deterministic gate | delta, regression deltas, keep/drop reason | LLM не участвует в decision. |
| `revert_patch(candidate_id)` | write FEDOT checkout | clean/dirty status | Обязателен после drop. |
| `write_journal(event)` | append-only | event path/hash | Все decisions воспроизводимы. |

## Срез кода 2026-08-26 (v0 лежит в дереве)

Пакет: `research/evolve/metric_agent/`. CLI: `python -m research.evolve.metric_agent eval-contract|run`.

Этап 0 **пройден** на `.repo_cache/FEDOT` (stock 0.7.5): `catboost` ok AUC≈0.847; `pca->catboost` crash 0.5 + traceback в `data.py:get_not_encoded_data`; `fast_ica->lgbm` ok ≈0.792; `guard_path("data/cases.json")` = deny. Юниты: `tests/research/test_metric_agent.py`.

Этап 1 **написан, LLM-прогон не сделан**. `run` без `FEDOTLLM_LLM_API_KEY` сразу `no_patch`. Checkout копируется без `.git` — revert = recopy, не `git checkout`.

### Что ещё дырявое (не делать этап 4/5)

| Дыра | Зачем | Куда |
|---|---|---|
| Нет реального `run` с LLM | нет keep/drop journal, нет доказательства что prompt не течёт gym | `run --task pca->catboost` |
| Нет `repo_map` / callers / field usage | silent-графы (knn, bernb) без traceback; crash-контекст сейчас только кадры стека | этап 2: `repo_map.py` |
| Journal без stdout/stderr скорера и без diff | acceptance этапа 1: replay | `eval.py` артефакт + `snapshot_diff` |
| `ScoreResult.env_hash` пустой | не отличить «тот же FEDOT» | hash `fedot/__init__.py` + pip |
| Нет paired table / replay CLI | этап 3 | `results.py`, не CSV scoring |
| `tools.py` нет | план хотел схемы tools; сейчас loop зовёт модули напрямую — ок для workflow, схемы нужны если появится LLM-tool loop | не блокер v0 |
| Контекст режет 40 строк вокруг кадра | в prompt попадает docstring `fit` раньше `get_not_encoded_data` | сузить radius / брать только deepest N кадров с телом |

Не трогать: OpenEvolve, LangGraph, prod `fedotllm/`, `library.py`.

## Дорожная карта

### Этап 0. Зафиксировать benchmark contract — done 2026-08-26

Deliverable:

- `TaskSpec`: `task_id`, dataset loader, graph/workload, metric, `higher_is_better`, sentinel, `min_delta`, timeout, must-not-regress tasks.
- `ScoreResult`: status `ok/crash/timeout/invalid`, score, traceback, duration, environment hash.
- `Decision`: keep/drop, target delta, regression deltas, reason.

Acceptance (id строго как в `data/cases.json`):

- `run_stock("pca->catboost")` → status `crash`, score = sentinel 0.5.
- `run_stock("catboost")` → status `ok`, AUC около 0.85.
- `run_stock("fast_ica->lgbm")` → status `ok`.
- `apply_patch` на путь внутри gym/cases/scorer падает (guard).
- Timeout и crash в `ScoreResult.status` различимы.

Implementation backlog:

1. Создать `research/evolve/metric_agent/types.py`.
2. Перенести scorer из `_local_fedot_patches/gym/` в research-only runnable место без `library.py`.
3. Сделать `tasks.py`, который читает `data/cases.json` и отдаёт `TaskSpec`.
4. Сделать `eval.py`, который запускает scorer subprocess-ом с `PYTHONPATH` на FEDOT checkout.
5. Добавить `tests`/smoke script только на guard/status parsing, не на качество патчей.

### Этап 1. Single-candidate runner для `pca→catboost`

Почему первым: есть crash, traceback и ясный sentinel; это лучший smoke test для архитектуры.

Flow:

1. Прогнать stock.
2. Построить context из traceback frames и source snippets.
3. Попросить LLM один patch, без доступа к `_local_fedot_patches/`.
4. Apply в disposable FEDOT checkout.
5. Прогнать target graph.
6. Прогнать `fast_ica→lgbm`.
7. Keep/drop deterministic.
8. Записать journal.

Acceptance:

- rejected patch оставляет clean checkout;
- accepted patch имеет `(stock_score, patched_score, delta, regression_delta)`;
- journal содержит diff и stdout/stderr;
- ни один tool не пишет в evaluator/cases/data.

Implementation backlog:

1. `checkout.py`: disposable copy/worktree FEDOT 0.7.5.
2. `guard.py`: deny paths outside checkout and inside evaluator.
3. `patch.py`: адаптировать `apply_source` из A.
4. `loop.py`: один проход `pca->catboost`.
5. `journal.py`: append-only JSONL + per-candidate dir.

### Этап 2. Context builder вместо tree-wide reader

Заменить “прочитай весь FEDOT” на ranked context.

Sources:

- traceback frames;
- AST symbols around operation implementation;
- callers of changed method;
- supplementary data fields: `categorical_idx`, `numerical_idx`, `encoded_idx`, `non_int_idx`;
- operation-family map: pca/kernel_pca/fast_ica/poly/imputation/catboost/xgboost/rf.

Acceptance:

- prompt на candidate умещается в фиксированный token budget;
- context includes target method and direct callers;
- no oracle replacement files appear in prompt.

Implementation backlog:

1. `context.py`: parse traceback into file/function/line frames.
2. `repo_map.py`: AST scan Python files в FEDOT checkout.
3. `symbols.py`: `search_class`, `search_method`, `search_callers`, `search_field_usage`.
4. Prompt contract: model returns only structured patch proposal.

### Этап 3. Paired stock/patched table

Сделать не “агент победил”, а measurement artifact.

Columns:

- candidate id;
- FEDOT commit/version;
- task id;
- graph/workload;
- stock status/score;
- patched status/score;
- delta;
- regression task deltas;
- changed files;
- keep/drop;
- reason.

Acceptance:

- для каждого candidate можно replay exact command;
- crash, timeout и invalid различаются;
- timeout не равен score sentinel автоматически;
- таблица не коммитит CSV/data.

Implementation backlog:

1. `results.py`: normalize `ScoreResult` to rows.
2. `compare.py`: deterministic decision + regression deltas.
3. `replay.py`: exact command from journal row.
4. `export.py`: markdown/csv table in `/tmp` or research artifact, never scoring CSV.

### Этап 4. Расширить поиск только после честного runner

После этапов 0-3 можно добавить:

- несколько operation families;
- parallel candidates;
- memory of rejected diffs;
- lightweight population/island search;
- Pareto ranking по задачам вместо раннего среднего.

Не раньше: OpenEvolve/CodeEvolve-style population без надёжного evaluator будет ускорять reward hacking.

Implementation backlog:

1. Add `CandidateMemory`: rejected diffs + reasons.
2. Add parallel runs over independent operation families.
3. Add Pareto rank: target delta, worst regression, runtime, patch size, cost.
4. Add optional island/population loop only after journal replay is stable.

### Этап 5. Перенос в production-контур

Только когда есть минимум несколько accepted candidates с reproducible paired evidence.

Что переносить:

- typed tools;
- evaluator API;
- journal schema;
- guardrails;
- replay CLI.

Что не переносить:

- oracle patches;
- old `library.py`;
- research-only notebooks/logs;
- scoring CSV.

Implementation backlog:

1. Freeze public CLI/API only after v0-v3 evidence.
2. Move generic pieces from `research/evolve/metric_agent/` into `fedotllm/agents/evolve_metric/` or a new package.
3. Keep evaluator task packs as external/read-only artifacts.
4. Add CI checks for path guard and replay schema, not for private scoring data.

## Что специально не делать

- Не превращать текущий EvolveAgent A в B простым добавлением AUC в verifier: reproduce-first фильтр выкинет silent metric improvements.
- Не брать LACE/AIDE/AutoMLGen как backbone: они улучшают pipeline задачи, а не FEDOT source.
- Не клонировать mini-SWE-agent bash-only: линейный цикл — да, свободный shell по FEDOT+gym — нет.
- Не использовать pytest как основной oracle: pytest только regression check после metric keep.
- Не давать LLM полный bash без typed tools: слишком трудно защищать evaluator.
- Не объединять несколько patch families в один diff.
- Не считать one-task lift финальной победой.
- **LLM не подсматривает gym.** Нет чтения `gym/`, `fedot_quality.py`, `cases.json`, поля `bug`, `replacements/`, `_local_fedot_patches/`. Harness один вызывает `run_stock`/`run_patched`. В промпт — FEDOT source + traceback/repo_map из tools. Канон leftover-idx / «catboost ~0.85» — для нас, не для модели.

## Source ledger

| Source | Type | Used for | URL |
|---|---|---|---|
| SWE-agent documentation | official docs | ACI, repo-editing tool loop, research-oriented configuration | https://swe-agent.com/latest/ |
| SWE-agent paper | arXiv paper | agent-computer interface claim | https://arxiv.org/abs/2405.15793 |
| mini-SWE-agent | official GitHub repo | minimal loop baseline / simplicity argument | https://github.com/SWE-agent/mini-swe-agent |
| OpenHands SDK architecture | official docs | agent/tools/workspace/events/security separation | https://docs.openhands.dev/sdk/arch/overview |
| OpenHands observability/metrics/security | official docs | tracing, cost/metrics, action confirmation patterns | https://docs.openhands.dev/sdk/guides/observability |
| aider repository map | official docs | repo map and relevance-ranked code context | https://aider.chat/docs/repomap.html |
| AutoCodeRover paper | arXiv paper | AST/class/method code search, localization before patching | https://arxiv.org/abs/2404.05427 |
| AutoCodeRover repo | official GitHub repo | implementation framing for analysis/debugging + patch | https://github.com/AutoCodeRoverSG/auto-code-rover |
| Agentless | official GitHub repo | localization → repair → validation staged workflow | https://github.com/openautocoder/agentless |
| Moatless Tools | official GitHub repo | controlled code-editing tools and context insertion | https://github.com/aorwall/moatless-tools |
| SWE-Search | arXiv paper | MCTS / state-search idea for later phases | https://arxiv.org/abs/2410.20285 |
| LangGraph overview | official docs | deterministic + LLM steps in one graph | https://docs.langchain.com/oss/python/langgraph/overview |
| Anthropic Building Effective Agents | first-party engineering article | keep workflow simple; tool docs/testing | https://www.anthropic.com/engineering/building-effective-agents |
| OpenAI Agents SDK tools | official docs | function tools, local runtime tools, ApplyPatchTool | https://openai.github.io/openai-agents-python/tools/ |
| OpenAI Agents SDK guardrails | official docs | tripwires and tool guardrails | https://openai.github.io/openai-agents-python/guardrails/ |
| OpenAI Agents SDK tracing | official docs | replay/observability shape | https://openai.github.io/openai-agents-python/tracing/ |
| AlphaEvolve announcement | DeepMind blog | automated evaluators + evolutionary framework | https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ |
| AlphaEvolve Cloud blog | Google Cloud blog | seed program + deterministic evaluator contract | https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone |
| AlphaEvolve harness architecture | Google Cloud docs | validation → verification → evaluation gate shape | https://docs.cloud.google.com/gemini/enterprise/docs/alphaevolve/developer-guide/logical-architecture |
| OpenEvolve | official GitHub repo | evaluator pool, program DB, MAP-Elites, islands | https://github.com/algorithmicsuperintelligence/openevolve |
| CodeEvolve paper | arXiv paper | island GA, weighted LLM ensemble, execution feedback | https://arxiv.org/abs/2510.14150 |
| FormulaCode docs | official docs | benchmark harness with Docker, correctness, per-workload speedup | https://formulacode.org/docs/ |
| FormulaCode paper | arXiv paper | multi-workload continuous metric benchmark design | https://arxiv.org/abs/2603.16011 |
| SWE-Perf project page | project page | real PR optimization benchmark and validated speedup | https://swe-perf.github.io/ |
| debug-gym repo | official GitHub repo | text observations, debugger/code tools | https://github.com/microsoft/debug-gym |
| debug-gym Microsoft blog | first-party research blog | pdb/breakpoints/navigation for agent debugging | https://www.microsoft.com/en-us/research/blog/debug-gym-an-environment-for-ai-coding-tools-to-learn-how-to-debug-code-like-programmers/ |

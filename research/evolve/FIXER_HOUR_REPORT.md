# Fixer hour report

Сессия: cloud, freeze `713217f` (`research/evolve/FIXER_SESSION.json`).  
LLM: **KEY_MISSING** (`FEDOTLLM_LLM_API_KEY` нет). Патчи модели не фейкались.  
FEDOT stock: `.repo_cache/FEDOT` tag `v0.7.5`.

## Changes — files, flags, tests

| Что | Где | e2e |
|---|---|---|
| Freeze HEAD | `research/evolve/FIXER_SESSION.json` | — |
| `context_from_lead(..., mode=)` | `context.py`: `auto` (как было), `slice`, `whole`, `dep` | default `auto`, clip 80 строк не тронут |
| `propose_patch(..., max_edits=1, contract="")` | `propose.py`; при `max_edits>1` схема `edits[]` → `PatchCandidate.hunks` | `loop.py` / `fix_lead` без изменений, budget=1 |
| Oracle harness | `oracle_bench.py` (не импортируется scout/discover/fixer) | CLI `python -m research.evolve.metric_agent oracle-fixer` |
| Артефакты прогона | `/tmp/metric-agent-oracle-fixer/` и `research/evolve/runs/fixer-oracle/` | — |
| Тесты | multi-hunk `apply_patch`; mode slice/whole/dep; `max_edits` hunks; AST semantic checkers; scout-stack без `pca->catboost` / leftover / `oracle_bench` | `45 passed` |

Промпт `_SYSTEM` **не** менялся: в этой сессии не было модельных патчей, нельзя утверждать «shallow». `contract` только если oracle передаёт `prompt_arm=hypothesis`. Multi-edit в основной цикл не включался.

## Oracle results table

Локации (AST, FEDOT 0.7.5): PCA `PCAImplementation` L112; KNN `FedotKnnClassImplementation.fit` L53; Imputation `ImputationImplementation.fit` L274.

| case | exact symbol | context arm | samples | edit budget | semantic success | tests | DEV delta |
|---|---|---|---|---|---|---|---|
| pca | `.../sklearn_transformations.py:112 method PCAImplementation` | auto | 1 | 1 | KEY_MISSING | — | — |
| pca | то же | auto | 3 | 1 | KEY_MISSING | — | — |
| pca | то же | auto | 1 | 3 | KEY_MISSING | — | — |
| pca | то же | slice | 1 | 1 | KEY_MISSING | — | — |
| pca | то же | dep | 1 | 1 | KEY_MISSING | — | — |
| pca | то же | whole | 1 | 1 | KEY_MISSING | — | — |
| knn | `.../knn.py:53 method FedotKnnClassImplementation.fit` | auto | 1 | 1 | KEY_MISSING | — | — |
| knn | то же | auto | 3 | 1 | KEY_MISSING | — | — |
| knn | то же | dep | 1 | 1 | KEY_MISSING | — | — |
| imputation | `.../sklearn_transformations.py:274 method ImputationImplementation.fit` | auto | 1 | 1 | KEY_MISSING | — | — |
| imputation | то же | auto | 1 | 3 | KEY_MISSING | — | — |

Контекст (без LLM; `runs/fixer-oracle/context_probe.json`):

| case | arm | chars | виден баг-сайт |
|---|---|---|---|
| pca | auto | 4323 | нет: только `__init__` + field map; **нет** `self.pca.fit(` / `self.pca.transform(` |
| pca | slice | 906 | нет, ещё уже |
| pca | dep | 7495 | да: parent `ComponentAnalysisImplementation.fit`+`transform` |
| pca | whole | 20254 | да, плюс чужой imputation в том же файле |
| knn | auto | 3909 | нет parent `predict`; есть `fit`+`predict_proba` |
| knn | dep | 5139 | да: parent `KNeighborsImplementation.predict` |
| imputation | auto | 9072 | да: `fit`+`transform` уже в auto |

Stock AST gold-like: pca/knn/imputation = False (чекер не принимает текущий код).

1 vs 3 / unique hashes / seed 42: **не измерено** (KEY_MISSING). Не утверждаем независимость.

## Key conclusion

**context**

На оракул-символе PCA auto **не кладёт в промпт методы, которые надо менять** (они в parent в том же файле). KNN auto не кладёт parent `predict`, где должен жить scaler на infer. Этого достаточно, чтобы объяснить overnight 0/10 KEEP на oracle locations, не трогая judge: модель не видит SEARCH-текст fit/transform.

Imputation — не этот bottleneck (fit+transform уже в auto); там скорее hypothesis / два hunk’а. На overnight KEEP это не главный кейс.

Не выбрано: sampling, single-edit, prompt, model capability — нет LLM-патчей в этой сессии.

## Recommendation

1. С ключом: `python -m research.evolve.metric_agent oracle-fixer --matrix` (PCA/KNN, auto vs dep, 1 vs 3, `max_edits` 1 vs 3). **Не** включать `dep` / multi-edit в e2e, пока не будет `semantic_repair_success` на PCA (worker `ok`).
2. Если dep всё ещё даёт shallow — короткий `contract` («fit и transform/predict одно и то же правило на признаки») без gold-патча в промпте; иначе остановиться.

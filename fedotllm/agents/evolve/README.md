# EvolveAgent

**This is an agent** in FEDOT.LLM. It is not the AutoML helper.

AutoML calls FEDOT to *train on a dataset*.  
EvolveAgent calls FEDOT to *check FEDOT itself*, then patches the library source.

---

## What it does

1. Scans [aimclub/FEDOT](https://github.com/aimclub/FEDOT) for a suspected defect (no ticket).
2. Confirms it with a verifier script run on a clean checkout
   (`Fedot.fit` / public API where possible). AutoML semantic rules are hints,
   not proof, unless an explicit calibration mode is enabled.
3. Writes **one** patch.
4. Keeps the patch only if independent gates still pass. Otherwise it reverts and abstains.

Lint is a hint. A crash that the model invented, or that only happens on an
internal helper, does not count. A semantic lead is not a free pass: the
verifier or the generated test must fail on pristine source before the patch is
accepted.

```
ensure_repo → scan → reader → verifier → fixer
                 one pass   cheap      strong
```

---

## Models

| Who | Model | Job |
|:----|:------|:----|
| **Reader** | `deepseek/deepseek-v4-flash` + `stealth/ox-alpha` | Reads every file, marks real suspicions — one pass per model |
| **Verifier** | `deepseek/deepseek-v4-flash` | Writes a proof script (crash or silent behaviour) |
| **Fixer** | `openai/gpt-5.4` | Patches one confirmed defect |

| Change this with | |
|:-----------------|:--|
| Reader, one model | `FEDOTLLM_EVOLVE_READER_MODEL` |
| Reader, one pass per model | `FEDOTLLM_EVOLVE_READER_MODELS` (comma-separated) |
| Verifier | `FEDOTLLM_EVOLVE_VERIFIER_MODEL` |
| Fixer | `--override llm.model_name=openai/gpt-5.4` |
| Semantic leads | `FEDOTLLM_EVOLVE_SEMANTIC_MODE=automl` (default), `generic`, or `off` |

Without OpenRouter the reader uses the same model as the fixer.

---

## Cost (measured, OpenRouter)

One **full** pass over FEDOT (`origin/master`, no file cap):

| | |
|:--|--:|
| Wall time | **~40 minutes** |
| Total | **~$4** |
| Reader | ~$3.50 (hundreds of cheap calls) |
| Verifier | ~$0.25 |
| Fixer | ~$0.15–0.20 |

Almost all money is the reader walking the tree. The fixer is the expensive model, but it runs a handful of times.

**Two reader models, not one.** Measured on ten FEDOT files, `deepseek-v4-flash`
and `stealth/ox-alpha` reported 12 and 7 suspicions and shared only **3** of the
16 between them — the same lesson as two passes of one model agreeing on 36 of
244. So the second model is a second pass, not a replacement. `ox-alpha` is free
and about seven times slower (643s against 84s for those ten files); the passes
run concurrently, so it is paid for in wall time rather than money.

Being a reasoning model, it also needs room to think: at the old 120s request
timeout it lost half the reader's files to `LLMRequestTimeout` while the model
was still working. The preset now allows 600s, which bounds a hung provider
without cutting off a slow answer.

A smoke run (`LIMIT_FILES=20`, `LIMIT_LEADS=20`) is a few minutes and cents, not dollars.

---

## Run

Need a **clean disposable** FEDOT checkout. The agent writes files into it.

```bash
export FEDOTLLM_LLM_API_KEY=...
export FEDOTLLM_REPO_PATH=/path/to/FEDOT
export FEDOTLLM_REPO_PYTHON=/path/to/fedot-venv/bin/python

uv run python -m fedotllm.agents.evolve \
  --presets fedotllm:openrouter \
  --override llm.model_name=openai/gpt-5.4 \
  --workspace /tmp/fedotllm-evolve
```

```bash
# smaller / cheaper
export FEDOTLLM_EVOLVE_LIMIT_FILES=20
export FEDOTLLM_EVOLVE_LIMIT_LEADS=20
export FEDOTLLM_EVOLVE_MAX_FIXER_CANDIDATES=3
```

For full FEDOT runs, leave `FEDOTLLM_EVOLVE_SEMANTIC_MODE=automl` enabled. It
adds general AutoML/library-wrapper bug shapes to the reader output, but does
not mark them confirmed. The verifier still has to reproduce the defect, or the
fixer has to write a test that fails on pristine source.

| Exit | |
|:----:|:--|
| **0** | Patch accepted |
| **1** | No accepted patch (incomplete or abstain) |
| **2** | `FEDOTLLM_LLM_API_KEY` missing |

Artifacts: `workspace/pipeline/` (`lint.json`, `leads.json`, `verified.json`, `fixer/.../evolution_audit.md`).

The dataset gate below works without an LLM key. A live run still needs a
valid key; a bad key fails on the first reader/verifier/fixer call.

---

## FEDOT Dataset Calibration Gate

The known-bug dataset is a **hidden label set** for discovery scoring. It must
not be copied into reader/verifier prompts or into AST fingerprints. Live runs
keep `FEDOTLLM_EVOLVE_STATIC_SEMANTIC_VERIFY` off: a semantic lead is only a
hint until the verifier or a generated test fails on pristine source.

```bash
uv run python -m fedotllm.agents.evolve.benchmark \
  --dataset research/evolve/fedot_bug_dataset/dataset.json \
  --split research/evolve/fedot_bug_dataset/split.json \
  --repo /tmp/fedot-evolve-clean-main \
  --eval-output /tmp/fedotllm-benchmark-cli-threshold-gate \
  --scan-eval \
  --subsets train,test,all
```

`--semantic-static-eval` still exists as a no-LLM sanity check of generic
shapes. Do not treat 33/33 static recall as agent quality. Score live
`--discovery-eval` (reader + verifier) instead.

`--subset split`, `--subset excluded`, `--subset unassigned`, and `--subset all`
are also supported for artifact scoring.

---

## Gates (fixer)

The model’s own test is not enough. After a patch, in order:

1. Proof **fails** on the old tree, **passes** after the patch  
2. `OLD` snippet matches once, file still parses  
3. Targeted pytest + caller tests  
4. CPU `Fedot.fit` still works  
5. Hidden probe cases do not get worse  
6. If the bug was “this parameter cannot be used” — tuning must work; dropping the parameter does not count  

Failure → checkout reset.

```bash
uv run python -m pytest tests/unit/agents tests/unit/llm/test_inference.py -q
```

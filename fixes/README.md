# Fixes written by hand and verified

Two patches for the "declared value is overwritten during fit" class, kept here
because they are the reference shape the agent has to learn: the operation may
keep adapting, it may not overwrite what the caller declared.

`verified_fixes.patch` — `poly.py` and `sklearn_filters.py`.

Both follow the same shape: the corrected value moves to a private attribute,
`self.params` is left holding what the caller passed. That matters because
`PipelineNode.descriptive_id` embeds the parameters and is the operations-cache
key, so rewriting them costs the whole pipeline its cache.

What was measured, on commit `310061e`:

| | tests generated from the finding | FEDOT `test/unit` |
|---|---|---|
| untouched checkout | value ✗, cache ✗, behaviour ✓ | 4 failed, 642 passed |
| the agent's patch for `poly.py` | value ✓, cache ✓, **behaviour ✗** | not run |
| these patches | value ✓, cache ✓, behaviour ✓ | 4 failed, 642 passed |

The four failures are identical in both columns and are `topological_features`
tests that need an optional dependency this environment does not have; they fail
on the untouched checkout too.

Apply with:

    git apply fixes/verified_fixes.patch

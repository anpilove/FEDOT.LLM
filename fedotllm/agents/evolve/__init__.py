"""EvolveAgent: hunt FEDOT source for patches, judge them with hour-long Fedot.

Hunt (Scout/Verifier/Fixer) only enqueues technically valid candidates;
KEEP/DROP is decided by ``quality-drain`` on the preregistered task registry.
The LLM never reads this package's scorer, holdout splits or benchmark cases.
"""

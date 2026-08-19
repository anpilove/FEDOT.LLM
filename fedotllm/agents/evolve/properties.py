"""Properties the agent invents, screened by machine instead of by reviewers.

Everything else in this package hands the agent a defect that a human already
decided was a defect: a lint rule, or one of the invariants in
:mod:`fedotllm.agents.evolve.invariants`.  That answers "can it fix things" and
dodges "can it find things".

The state of the art does not dodge it.  Anthropic's property-based testing
agent (2026) derives properties from type annotations, docstrings and comments
with no human-supplied property class, and LogicHunter (arXiv 2607.06195) has
the oracle itself consult documentation, source and runtime state.  Both work.
Both also report the catch: of 50 manually reviewed reports from the Anthropic
run, **56%** were valid bugs and **32%** were valid *and* worth sending to a
maintainer.  Two out of three freely invented properties encode the agent's own
misunderstanding.  They filtered with a scoring rubric plus three experts at
roughly an hour per bug.

We cannot afford three experts, and we do not need them, because FEDOT gives us
something that ecosystem-wide studies do not have: **79 operations that are
supposed to obey the same contract**.  So the agent writes its property as a
function of an operation name, and the machine tries to falsify it:

    unrunnable   the test errors out (import, syntax, missing fixture)
                 -> proves nothing, discarded
    holds        passes on the target -> no defect here, and saying so is a
                 legitimate outcome, not a failure
    too_broad    fails on the control operations as well -> the property is
                 wrong about FEDOT, not about the target.  This is the machine
                 standing in for the human reviewer, and it is the only reason
                 this module can be trusted without one.
    flaky        fails only sometimes -> unusable as evidence either way
    candidate    fails on the target, holds on every control, reproducibly

Only ``candidate`` reaches the patching loop.  A property that survives is a
defect the agent both found and proved, with no human having named the class.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from fedotllm.agents.evolve.property_harness import (  # noqa: F401  (re-exported)
    Skipped,
    declared_parameters,
    fit_operation,
    in_force,
    is_legal,
    make_data,
    observed_everywhere,
)
from fedotllm.log import logger

# How many sibling operations a property is tried on before it is believed.
# Three is enough to catch "this is just how FEDOT works" and cheap enough to
# run inside a round; the whole point is falsification, not coverage.
NUM_CONTROLS = int(os.environ.get("FEDOTLLM_PROPERTY_CONTROLS", "3"))
PROPERTY_TIMEOUT = int(os.environ.get("FEDOTLLM_PROPERTY_TIMEOUT", "300"))

PROPERTY_SYS = """You are looking for behavioural defects in the FEDOT AutoML library.

Nobody has told you what is wrong. You decide what SHOULD be true of a correct
implementation, then write a check that fails when it is not.

A property is a statement that must hold for ANY correct implementation, e.g.:
  - a hyperparameter the caller declared is the one in force after fit;
  - fitting twice on the same data gives the same result;
  - a value the library itself declares tunable does not raise;
  - a transform that filters nothing returns the rows it was given;
  - what the docstring promises is what the code does.

Do NOT restate a lint rule or a style preference. Do NOT propose something you
already know FEDOT violates on purpose.

Your check runs on the operation below AND on sibling operations. If it fails
everywhere, it is rejected as YOUR misunderstanding rather than a defect. State
something that ought to hold for every operation and that you suspect THIS one
gets wrong.

You do NOT need to know FEDOT's API. These are already defined for you; use them
and import nothing from fedot yourself:

    fit_operation(operation, params=None)  -> (pipeline, node, data)
        Builds a working pipeline around `operation`, fits it, and hands back
        the fitted pipeline, that operation's node, and the data used.
        Raises Skipped if the operation cannot be set up.
        `node.parameters` is what the node reports; `node.fitted_operation` is
        the fitted object (its `.params.to_dict()`, and `.operation` / `.model`
        for the wrapped estimator).
    in_force(node, name, default=None) -> the value of `name` actually in force
        on the fitted object — ONE value, already resolved wherever the wrapper
        keeps it. Returns `default` when the parameter appears nowhere. Use this
        instead of guessing attribute names like `.model` or `.solver`.
    make_data(kind)     -> InputData; kind is "classification", "regression",
                           "regression_outliers" or "ts"
    declared_parameters(operation) -> {name: [values worth trying]} — sample
        points (both ends of the declared interval plus the middle, or every
        categorical choice), ready to pass to fit_operation. These are NOT the
        whole legal range, so never test membership with `value in ...`.
    is_legal(operation, name, value) -> bool — does the library's own declared
        scope allow this value? Use this for "is the value in force allowed",
        never `in declared_parameters(...)`.
    np                  -> numpy
    Skipped             -> raise Skipped("why") when the check cannot be evaluated

Note: `node.parameters` holds only what was explicitly set plus the operation's
own defaults, so use `.get(name)` and treat a missing key as "not applicable"
(raise Skipped), never as a violation.

Reply in EXACTLY this format, no JSON, no markdown fences:

NAME: short_snake_case_name
PROPERTY: one sentence — what must hold, and why any correct implementation obeys it
<<<CHECK>>>
def property_holds(operation: str) -> None:
    pipeline, node, data = fit_operation(operation)
    ...
    assert <condition>, "what went wrong"
<<<END>>>

Enforced by machine, your reply is discarded otherwise:
  - it MUST define `property_holds(operation: str) -> None`;
  - the operation NAME must never appear as a literal in your code — the check
    has to work for whichever operation is passed in, or the controls cannot
    falsify it;
  - raise AssertionError when the property is violated, return when it holds;
  - raise Skipped when the check does not apply, rather than asserting;
  - no file access, no network.
"""


@dataclass
class ProposedProperty:
    name: str
    statement: str
    code: str
    target: str = ""
    verdict: str = ""
    controls: dict[str, str] = field(default_factory=dict)
    output: str = ""


RUNNER = '''
import json
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")


class Skipped(Exception):
    pass


import builtins

builtins.Skipped = Skipped

sys.path.insert(0, {package_dir!r})
import property_harness as _H

# Exactly what the prompt advertises, nothing else.
_ns = {{
    "Skipped": Skipped,
    "np": _H.np,
    "make_data": _H.make_data,
    "fit_operation": _H.fit_operation,
    "in_force": _H.in_force,
    "observed_everywhere": _H.observed_everywhere,
    "declared_parameters": _H.declared_parameters,
    "is_legal": _H.is_legal,
}}
try:
    exec(compile(open({code_path!r}).read(), "property", "exec"), _ns)
except BaseException as exc:
    print("RESULT" + json.dumps({{"outcome": "unrunnable",
                                 "detail": type(exc).__name__ + ": " + str(exc)[:400]}}))
    sys.exit(0)

fn = _ns.get("property_holds")
if not callable(fn):
    print("RESULT" + json.dumps({{"outcome": "unrunnable",
                                 "detail": "property_holds is not defined"}}))
    sys.exit(0)

try:
    fn({operation!r})
except AssertionError as exc:
    print("RESULT" + json.dumps({{"outcome": "violated",
                                 "detail": str(exc)[:400] or "AssertionError"}}))
except Skipped as exc:
    print("RESULT" + json.dumps({{"outcome": "skipped", "detail": str(exc)[:400]}}))
except BaseException as exc:
    # An unexpected exception is NOT a violation. It usually means the property
    # itself is broken -- and counting it as evidence is exactly how an agent
    # ends up "proving" a defect that does not exist.
    print("RESULT" + json.dumps({{"outcome": "error",
                                 "detail": type(exc).__name__ + ": " + str(exc)[:400],
                                 "trace": traceback.format_exc()[-1200:]}}))
else:
    print("RESULT" + json.dumps({{"outcome": "holds", "detail": ""}}))
'''


def reject_hardcoded_target(code: str, target: str) -> str | None:
    """Refuse a property that only applies to the operation under test.

    The prompt says "must not hardcode it". The first live reply said:

        def property_holds(operation: str) -> None:
            if operation != 'ransac_lin_reg':
                raise Skipped("This check is only for the ransac_lin_reg operation.")

    which quietly disables every control and turns falsification into a
    rubber stamp. Asking is not a mechanism -- this is the fourth time in this
    project a stated requirement was ignored by the model -- so the target's
    name is simply not allowed to appear in the check.
    """
    if re.search(rf"""['"]{re.escape(target)}['"]""", code):
        return (f"the check mentions {target!r} literally; it must work for any "
                "operation passed in, otherwise the controls cannot falsify it")
    return None


def parse_property(raw: str) -> ProposedProperty:
    """Read the agent's reply. Delimiters, not JSON: code breaks JSON escaping."""
    name = ""
    statement = ""
    for line in raw.splitlines():
        if line.upper().startswith("NAME:") and not name:
            name = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PROPERTY:") and not statement:
            statement = line.split(":", 1)[1].strip()

    # Degeneration guard. Measured on the first live run: asked for a property
    # about `ransac_lin_reg` -- from a 746-character source, so context length is
    # not the cause -- gpt-4o-mini looped on a single import line and produced
    # 36 225 characters until it hit the token ceiling. Salvaging a CHECK block
    # from a reply like that means running code the model never meant to write.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        most_common = max(set(lines), key=lines.count)
        if lines.count(most_common) > 5:
            raise ValueError(
                f"degenerate reply: {lines.count(most_common)} repetitions of "
                f"{most_common[:60]!r}")

    match = re.search(r"<<<CHECK>>>(.*?)(?:<<<END>>>|$)", raw, re.S)
    if not match:
        raise ValueError("no <<<CHECK>>> block")
    code = textwrap.dedent(match.group(1)).strip("\n")
    # Models wrap code in fences even when told not to.
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip(), flags=re.M)
    if "def property_holds" not in code:
        raise ValueError("the CHECK block does not define property_holds")
    if not re.search(r"def property_holds\s*\(\s*operation", code):
        raise ValueError("property_holds must take the operation name as its argument")
    if not name:
        name = "unnamed_property"
    return ProposedProperty(name=name, statement=statement, code=code)


def evaluate(prop: ProposedProperty, operation: str, repo: Path, py: str,
             workdir: Path) -> tuple[str, str]:
    """Run one property against one operation in a fresh process.

    A separate process per operation because a property that hangs or segfaults
    is a result we want to record, not a lost round.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    code_path = workdir / f"property_{prop.name}.py"
    code_path.write_text(prop.code, encoding="utf-8")
    runner_path = workdir / f"run_{prop.name}_{operation}.py"
    runner_path.write_text(
        RUNNER.format(code_path=str(code_path), operation=operation,
                      package_dir=str(Path(__file__).resolve().parent)),
        encoding="utf-8")
    try:
        proc = subprocess.run([py, str(runner_path)], cwd=repo, capture_output=True,
                              text=True, timeout=PROPERTY_TIMEOUT)
    except subprocess.TimeoutExpired:
        return "timeout", f"no result in {PROPERTY_TIMEOUT}s"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT"):
            try:
                payload = json.loads(line[len("RESULT"):])
            except json.JSONDecodeError:
                continue
            return payload.get("outcome", "error"), payload.get("detail", "")
    return "unrunnable", (proc.stderr or proc.stdout)[-600:]


def pick_controls(repo: Path, py: str, target: str, limit: int = NUM_CONTROLS) -> list[str]:
    """Sibling operations to falsify the property against, closest relatives first.

    Siblings, not arbitrary operations: a property about a classifier says
    nothing when tried on a time-series transform, so a failure there would
    prove nothing either.

    Ordering by shared tags is not cosmetic. Measured: with three arbitrary
    siblings, a property about `residual_threshold` had all three controls skip
    (they do not take that parameter), and a genuine defect came out as
    "unchecked". Sorting by tag overlap puts `ransac_non_lin_reg` first, which
    is the one control that can actually answer the question. The list returned
    here is a ranked pool -- the caller keeps trying until enough of them really
    evaluate.
    """
    code = (
        "import warnings, json; warnings.filterwarnings('ignore');"
        "from fedot.core.repository.operation_types_repository import "
        "OperationTypesRepository as R;"
        "out=[];\n"
        "for k in ('model','data_operation'):\n"
        "    for o in R(k).operations:\n"
        "        out.append([o.id, [t.value for t in o.task_type], list(o.tags)])\n"
        "print('OPS'+json.dumps(out))"
    )
    proc = subprocess.run([py, "-c", code], cwd=repo, capture_output=True, text=True)
    ops = []
    for line in proc.stdout.splitlines():
        if line.startswith("OPS"):
            try:
                ops = json.loads(line[3:])
            except json.JSONDecodeError:
                ops = []
            break
    if not ops:
        logger.warning("could not list operations for controls")
        return []
    tasks = next((t for i, t, _ in ops if i == target), [])
    own_tags = set(next((tags for i, _, tags in ops if i == target), []))
    siblings = [
        (len(own_tags & set(tags)), i)
        for i, t, tags in ops
        if i != target and set(t) & set(tasks) and "non-default" not in tags
    ]
    siblings.sort(key=lambda pair: -pair[0])
    return [i for _, i in siblings[: max(limit * 5, 15)]]


def screen(prop: ProposedProperty, target: str, repo: Path, py: str,
           workdir: Path) -> ProposedProperty:
    """Try to falsify the property. Only `candidate` is evidence of a defect."""
    prop.target = target
    if (why := reject_hardcoded_target(prop.code, target)) is not None:
        prop.verdict = "hardcoded"
        prop.output = why
        return prop
    outcome, detail = evaluate(prop, target, repo, py, workdir)
    prop.output = f"{target}: {outcome} — {detail}"

    if outcome in ("unrunnable", "error", "timeout"):
        prop.verdict = "unrunnable"
        return prop
    if outcome == "skipped":
        prop.verdict = "unrunnable"
        return prop
    if outcome == "holds":
        # The honest empty-handed outcome, and the one the agent was never able
        # to reach before: it looked, and there was nothing here.
        prop.verdict = "holds"
        return prop

    # Violated on the target. Before believing it, try to break it: run it twice
    # more on the target, then on the siblings.
    again, _ = evaluate(prop, target, repo, py, workdir)
    if again != "violated":
        prop.verdict = "flaky"
        prop.output += f" | second run: {again}"
        return prop

    # Work down the ranked pool until enough controls have actually answered.
    # A control that cannot be set up, or that does not take the parameter in
    # question, says nothing and must not be counted as agreement.
    #
    # Every answering control is consulted, not just the first that disagrees.
    # Measured why: for `lagged`, the agent proposed "window_size must stay
    # inside the declared sampling scope after fit" -- a genuine defect, found
    # independently by hand the same night -- and stopping at the first
    # violating control threw it away, because `sparse_lagged` breaks it too.
    # A property that fails EVERYWHERE is the agent misreading the library; one
    # that fails on some operations and holds on others discriminates, and a
    # whole family sharing a defect is a finding, not a false alarm.
    holds_on, violated_on = [], []
    for control in pick_controls(repo, py, target):
        if len(holds_on) + len(violated_on) >= NUM_CONTROLS:
            break
        c_outcome, c_detail = evaluate(prop, control, repo, py, workdir)
        prop.controls[control] = f"{c_outcome} — {c_detail}"
        if c_outcome == "violated":
            violated_on.append(control)
        elif c_outcome == "holds":
            holds_on.append(control)

    if not holds_on and not violated_on:
        # Nothing to compare against. Better to admit that than to pass off an
        # unfalsified property as a proven defect.
        prop.verdict = "unchecked"
        return prop
    if not holds_on:
        prop.verdict = "too_broad"
        return prop
    prop.verdict = "shared" if violated_on else "candidate"
    return prop


def propose(inference, repo: Path, target: str, source: str,
            tried: list[str] | None = None) -> ProposedProperty:
    """Ask the agent for one property about `target`."""
    already = ""
    if tried:
        already = ("\n\nProperties already tried on this operation, do not repeat "
                   "them:\n" + "\n".join(f"  - {t}" for t in tried))
    messages = [
        {"role": "system", "content": PROPERTY_SYS},
        {"role": "user", "content": (
            f"Operation: `{target}`\n\n"
            f"Its implementation:\n```python\n{source[:12000]}\n```"
            f"{already}\n\n"
            "State ONE property in the required format."
        )},
    ]
    raw = inference.query(messages) or ""
    try:
        return parse_property(raw)
    except ValueError as exc:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": (
            f"Parse error: {exc}. Reply again in the EXACT format: NAME:, PROPERTY:, "
            "then <<<CHECK>>> ... <<<END>>> defining "
            "`def property_holds(operation: str) -> None`. No JSON, no fences."
        )})
        return parse_property(inference.query(messages) or "")

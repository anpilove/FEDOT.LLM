"""Data every proof script may assume, in one place.

The verifier writes standalone scripts and the fixer needs those same scripts as
pytest proofs, so both sides have to agree on what `train_data` and `ts_data`
mean. Keeping the text here rather than in one of the two scripts is what makes
a verified defect transferable: the proof the verifier ran and the test the
fixer must not weaken are built from the same source.
"""

from __future__ import annotations

PREAMBLE = '''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

_rng = np.random.default_rng(42)
_n = 120
_x = _rng.normal(size=(_n, 6))
train_data = InputData(idx=np.arange(_n), features=_x,
                       target=(_x[:, 0] + 0.5 * _x[:, 1] > 0).astype(int).reshape(-1, 1),
                       task=Task(TaskTypesEnum.classification),
                       data_type=DataTypesEnum.table)
_t = np.arange(200)
_series = np.sin(_t / 7.0) * 10 + _t * 0.05 + _rng.normal(size=200) * 0.2
ts_data = InputData(idx=_t, features=_series, target=_series,
                    task=Task(TaskTypesEnum.ts_forecasting,
                              TsForecastingParams(forecast_length=5)),
                    data_type=DataTypesEnum.ts)
'''


def as_pytest(script: str, test_name: str, why: str, got: str = "") -> str:
    """Wrap a verifier script into a test that fails while the defect stands.

    The obvious wrapping — run the script, let any exception be the failure — is
    wrong, and a real round showed it. The verified defect in `params.py` is
    "`Fedot(problem='nonsense')` dies with a bare `KeyError`". A correct patch
    does not stop it raising; it makes it raise `ValueError` naming the accepted
    problems. Under the obvious wrapping that patch still failed the test, and a
    good fix would have been thrown away.

    So the proof pins the defect without allowing a patch to replace it with an
    unrelated crash. Low-level AttributeError/IndexError/KeyError failures may
    become a clear ValueError/TypeError; every other unexpected exception fails.
    """
    body = "\n".join("        " + line if line.strip() else ""
                      for line in script.strip().splitlines())
    if not got or not got.isidentifier():
        # Nothing was observed to disappear, so the only thing to assert is that
        # the script now runs.
        plain = "\n".join("    " + line if line.strip() else ""
                           for line in script.strip().splitlines())
        return (PREAMBLE + "\n\n"
                + f"def {test_name}():\n"
                + f'    """{why[:200]}"""\n'
                + plain + "\n")
    low_level = got in {"AttributeError", "IndexError", "KeyError"}
    replacement = (
        "    except (ValueError, TypeError) as exc:\n"
        "        assert str(exc).strip(), \"replacement validation error must explain the problem\"\n"
        if low_level
        else ""
    )
    unexpected = (
        "    except Exception as exc:\n"
        "        raise AssertionError(\n"
        "            f\"unexpected replacement failure: {type(exc).__name__}: {exc}\"\n"
        "        ) from exc\n"
    )
    return (PREAMBLE + "\n\n"
            + f"def {test_name}():\n"
            + f'    """{why[:200]}"""\n'
            + "    try:\n"
            + body + "\n"
            + f"    except {got}:\n"
            + f'        raise AssertionError(\n'
            + f'            "still raises {got} — the defect this test was built from"\n'
            + f'        ) from None\n'
            + replacement
            + unexpected)

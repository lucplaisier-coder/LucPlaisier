"""
inputs.py: turn raw columns into the nine inputs (rules P1-P9; frozen 24 Nov 2026, see src/features.py).

Plain-language summary
----------------------
Each rule below is a FIXED formula applied to one loan at a time. Nothing
is learned from the data here: no averages, no medians, no category lists.
(Those "learned" steps happen later, inside the model pipelines, and only on
the training years. That is how we avoid leakage.)

Missing values stay missing (NaN). We never guess them here.
If a value is impossible (e.g. FICO 900, negative income, unknown sub-grade)
the pipeline STOPS so a person can look, instead of hiding the problem.

The rules come from decision_records/preparation_rules_v1.json
(G2-PREP-2026-11-24-v1). The rule number is shown next to each step.
"""

import numpy as np
import pandas as pd

from . import config
from .sample import strict_number


def _stop_if(mask: pd.Series, message: str, values: pd.Series) -> None:
    """Helper: if any row breaks a rule, stop and show a few examples."""
    if mask.any():
        examples = values[mask].unique()[:5].tolist()
        raise ValueError(f"{message}: {int(mask.sum())} rows, e.g. {examples}")


def build_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new table: id + the nine proposed inputs, same row order."""
    out = pd.DataFrame({"id": df["id"].values}, index=df.index)

    # P1 FICO midpoint = (low + high) / 2. Original FICO only, never last_fico.
    lo = strict_number(df["fico_range_low"], "fico_range_low")
    hi = strict_number(df["fico_range_high"], "fico_range_high")
    both = lo.notna() & hi.notna()
    _stop_if(both & ((lo < 300) | (hi > 850) | (lo > hi)), "P1 FICO out of 300-850 or low>high", lo)
    out["fico_mid"] = (lo + hi) / 2.0          # NaN if either end is missing

    # P2 DTI in percentage points (18.5 means 18.5%). Zero is a real value.
    dti = strict_number(df["dti"], "dti")
    _stop_if(dti < 0, "P2 negative DTI", dti)
    out["dti"] = dti

    # P3 natural log of annual income. Income must be > 0.
    inc = strict_number(df["annual_inc"], "annual_inc")
    _stop_if(inc <= 0, "P3 income zero or negative", inc)
    out["log_annual_inc"] = np.log(inc)        # NaN stays NaN

    # P4 employment length: fixed map 0..10; empty or "n/a" -> missing.
    emp = df["emp_length"].astype("string").str.strip()
    emp_missing = emp.isna() | (emp == "") | (emp.str.lower() == "n/a")
    emp_unknown = (~emp_missing) & (~emp.isin(config.EMPLOYMENT_MAP))
    _stop_if(emp_unknown, "P4 unknown employment label", emp)
    out["emp_length_ord"] = emp.where(~emp_missing).map(config.EMPLOYMENT_MAP).astype("float64")

    # P5 home ownership: trim + UPPERCASE, keep every label (NONE is a label).
    home = df["home_ownership"].astype("string").str.strip().str.upper()
    out["home_ownership"] = home.where(home != "")

    # P6 purpose: trim + lowercase, keep every label.
    purpose = df["purpose"].astype("string").str.strip().str.lower()
    out["purpose"] = purpose.where(purpose != "")

    # P7 listed loan amount in USD (not funded_amnt, not installment).
    amt = strict_number(df["loan_amnt"], "loan_amnt")
    _stop_if(amt <= 0, "P7 loan amount zero or negative", amt)
    out["loan_amnt"] = amt

    # P8 sub-grade A1=1 ... G5=35. Must be present and valid.
    sg = df["sub_grade"].astype("string").str.strip().str.upper()
    valid_sg = sg.str.fullmatch(r"[A-G][1-5]").fillna(False)
    _stop_if(~valid_sg, "P8 missing or invalid sub_grade", sg.fillna("<missing>"))
    out["sub_grade_ord"] = (sg.str[0].map(lambda c: "ABCDEFG".index(c)) * 5
                            + sg.str[1].astype(int)).astype("float64")

    # P9 interest rate in % AFTER the approved 148-case fix (see rates.py).
    rate = df["int_rate"]
    _stop_if(rate.isna() | (rate <= 0), "P9 missing or non-positive rate", rate.fillna(-1))
    out["int_rate"] = rate.astype("float64")

    # Consistency check: the broad grade letter must match the sub-grade.
    grade = df["grade"].astype("string").str.strip().str.upper()
    _stop_if(grade != sg.str[0], "grade does not match sub_grade", grade.fillna("<missing>"))

    cols = ["id"] + config.PROPOSED_NUMERIC_INPUTS + config.PROPOSED_NOMINAL_INPUTS
    return out[cols]

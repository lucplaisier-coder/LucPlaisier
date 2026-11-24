"""
selection.py: pick 80% of loans within each grade (control document F.7).

Plain-language summary
----------------------
Every approach (except fund-all) does the same thing: inside each credit
grade (A, B, C, ...) it ranks the loans from "least risky" to "most risky"
and funds the least risky 80%. Only the RANKING differs:

  * Platform rule: Lending Club's own sub-grade (A1 best ... G5 worst); within
    a sub-grade, the lower interest rate first; remaining ties broken by a
    random draw with a fixed seed (42), so the result is repeatable.
  * Logit / HGB: the model's predicted probability of default (PD), lowest
    first; exact ties broken by the same seeded random draw.

Selecting WITHIN each grade matters: it stops a model from "winning" just by
avoiding risky grades. It must find better loans than its neighbours in the
same grade.

How many loans are kept in a grade of n loans: round(0.8 * n), with halves
rounded up (floor(0.8 * n + 0.5)).
"""

import math

import numpy as np
import pandas as pd

from . import config

Q = 0.80                 # selection rate, fixed in F.7
MIN_GRADE_SIZE = 200     # F.7: merge F and G if a grade is smaller in the test year


def strata_for(grades: pd.Series, merge_fg: bool) -> pd.Series:
    """Grade letter used as the selection group. F and G become 'FG' if merged."""
    g = grades.astype(str)
    return g.replace({"F": "FG", "G": "FG"}) if merge_fg else g


def decide_fg_merge(inputs: pd.DataFrame, test_year: int = 2014) -> tuple[bool, pd.Series]:
    """
    F.7 rule: if grade F or G has fewer than 200 loans in the TEST year, merge
    them. This uses only loan COUNTS by grade (an input), not outcomes.
    """
    counts = inputs.loc[inputs["vintage"] == test_year, "grade"].value_counts().sort_index()
    merge = bool((counts.reindex(["F", "G"]).fillna(0) < MIN_GRADE_SIZE).any())
    return merge, counts


def tie_break_key(ids: pd.Series, seed: int = config.SEED) -> pd.Series:
    """
    A repeatable random number per loan. Loans are first put in id order so
    the draw does not depend on how the file happens to be sorted.
    """
    order = ids.astype("int64").sort_values().index
    rng = np.random.default_rng(seed)
    key = pd.Series(np.nan, index=ids.index)
    key.loc[order] = rng.random(len(order))
    return key


def select_within_strata(df: pd.DataFrame, rank_cols: list[str], strata: pd.Series,
                         q: float = Q) -> pd.Series:
    """
    Return True/False per loan: True = funded.
    rank_cols are sorted ascending (smallest = least risky = funded first).
    The last rank column should be the seeded tie-break key.
    """
    selected = pd.Series(False, index=df.index)
    for _, idx in df.groupby(strata).groups.items():
        part = df.loc[idx].sort_values(rank_cols, kind="mergesort")
        n_keep = math.floor(q * len(part) + 0.5)
        selected.loc[part.index[:n_keep]] = True
    return selected


def platform_rule(df: pd.DataFrame, strata: pd.Series, key: pd.Series, q: float = Q) -> pd.Series:
    """F.7 B: sub-grade, then interest rate, then seeded random (riskiest skipped first)."""
    tmp = pd.DataFrame({"sg": df["sub_grade_ord"], "rate": df["int_rate"], "key": key})
    return select_within_strata(tmp, ["sg", "rate", "key"], strata, q)


def model_rule(pd_score: pd.Series, strata: pd.Series, key: pd.Series, q: float = Q) -> pd.Series:
    """F.7 C/D: lowest predicted default probability first, then seeded random."""
    tmp = pd.DataFrame({"pd": pd_score, "key": key})
    return select_within_strata(tmp, ["pd", "key"], strata, q)

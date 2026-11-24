"""
sample.py: choose which loans are in the study, and define the target.

Plain-language summary
----------------------
Control document F.3 / F.4, applied in this order:
  1. issue year (vintage) 2010-2015
  2. 36-month loans only
  3. policy_code = 1
  4. individual (not joint) applications
  5. drop statuses starting "Does not meet the credit policy"
     -> this is the CANDIDATE POOL (also the scope of the approved rate fix)
  6. keep only loans with a final status (Fully Paid = 0, Charged Off = 1)
     -> this is the RESEARCH SAMPLE

We count how many loans each step removes, per year (the "funnel").
For 2014 and 2015 we only count statuses as final / not final, which the
F.4 resolution rule needs. We never split them into paid vs charged off.
"""

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Strict text-to-number conversion
# ---------------------------------------------------------------------------
def strict_number(series: pd.Series, name: str) -> pd.Series:
    """
    Convert text to numbers. An EMPTY cell becomes NaN (missing).
    Any non-empty text that is not a valid number STOPS the pipeline
    (preparation rules: "Do not hide parse errors as NaN").
    """
    text = series.astype("string").str.strip()
    empty = text.isna() | (text == "")
    numbers = pd.to_numeric(text.where(~empty), errors="coerce")
    bad = (~empty) & numbers.isna()
    if bad.any():
        examples = text[bad].unique()[:5].tolist()
        raise ValueError(f"{name}: {int(bad.sum())} non-numeric values, e.g. {examples}")
    if np.isinf(numbers).any():
        raise ValueError(f"{name}: infinite values found")
    return numbers.astype("float64")


def parse_vintage(issue_d: pd.Series) -> pd.Series:
    """'Dec-2015' -> 2015. Rows without a valid date (e.g. the 2 summary
    lines at the bottom of the raw file) get <NA> and fall out at step 1."""
    dates = pd.to_datetime(issue_d, format="%b-%Y", errors="coerce")
    return dates.dt.year.astype("Int64")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def build_candidate_pool(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply filters 1-5. Returns (candidate pool, funnel table per year)."""
    df = raw.copy()
    df["vintage"] = parse_vintage(df["issue_d"])

    # Filter 1: years 2010-2015
    df = df[df["vintage"].isin(config.YEARS)].copy()
    df["vintage"] = df["vintage"].astype(int)
    funnel = {"1_issued_2010_2015": df.groupby("vintage").size()}

    # Filter 2: 36-month term (the raw text has a leading space: " 36 months")
    df = df[df["term"].str.strip() == config.TERM_KEEP]
    funnel["2_term_36m"] = df.groupby("vintage").size()

    # Filter 3: policy_code = 1
    policy = strict_number(df["policy_code"], "policy_code")
    df = df[policy == config.POLICY_CODE_KEEP]
    funnel["3_policy_code_1"] = df.groupby("vintage").size()

    # Filter 4: individual applications
    df = df[df["application_type"].str.strip() == config.APPLICATION_TYPE_KEEP]
    funnel["4_individual"] = df.groupby("vintage").size()

    # Filter 5: drop "Does not meet the credit policy..." statuses
    status = df["loan_status"].str.strip()
    df = df[~status.str.startswith(config.EXCLUDED_STATUS_PREFIX)].copy()
    df["loan_status"] = df["loan_status"].str.strip()
    funnel["5_candidate_pool"] = df.groupby("vintage").size()

    # Every loan must appear once only.
    if df["id"].duplicated().any():
        raise ValueError("Duplicate loan ids in the candidate pool")

    # Any status we did not plan for -> stop and look.
    known = set(config.TARGET_MAP) | set(config.UNRESOLVED_STATUSES)
    unknown = set(df["loan_status"].unique()) - known
    if unknown:
        raise ValueError(f"Unexpected loan_status values: {sorted(unknown)}")

    funnel_table = pd.DataFrame(funnel).fillna(0).astype(int)
    funnel_table.index.name = "vintage"
    return df, funnel_table


def resolution_table(pool: pd.DataFrame) -> pd.DataFrame:
    """
    F.4 resolution rule: share of candidate loans with a FINAL status.
    Only final vs not-final is counted. Not-final loans are broken down by
    their status name (Current, Late, Default...), which is needed for the
    F.3 'Default' rule. Paid vs charged off is NOT counted here.
    """
    final = pool["loan_status"].isin(config.TARGET_MAP)
    t = pd.DataFrame({
        "candidate_loans": pool.groupby("vintage").size(),
        "resolved": final.groupby(pool["vintage"]).sum(),
    })
    t["unresolved"] = t["candidate_loans"] - t["resolved"]
    for s in config.UNRESOLVED_STATUSES:
        t[f"unresolved_{s}"] = (pool["loan_status"] == s).groupby(pool["vintage"]).sum()
    t["resolved_share"] = t["resolved"] / t["candidate_loans"]
    t["default_status_share"] = t["unresolved_Default"] / t["candidate_loans"]
    t["passes_99pct_rule"] = t["resolved_share"] >= config.RESOLUTION_THRESHOLD
    t["default_sensitivity_needed"] = (
        t["default_status_share"] > config.DEFAULT_STATUS_SENSITIVITY_SHARE
    )
    t.index.name = "vintage"
    return t


def keep_resolved(pool: pd.DataFrame) -> pd.DataFrame:
    """Filter 6: keep final statuses only, and attach the target 0/1."""
    df = pool[pool["loan_status"].isin(config.TARGET_MAP)].copy()
    df["target"] = df["loan_status"].map(config.TARGET_MAP).astype("int8")
    df["split"] = df["vintage"].map(config.SPLIT_BY_YEAR)

    counts = df.groupby("vintage").size().to_dict()
    if counts != config.EXPECTED_RESOLVED_COUNTS:
        raise ValueError(
            "Research-sample sizes differ from the independent Gate 1 audit:\n"
            f"  expected {config.EXPECTED_RESOLVED_COUNTS}\n  found    {counts}"
        )
    return df

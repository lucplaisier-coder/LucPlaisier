"""
step5_build_clean_dataset.py: ROADMAP STEP 5, build the clean dataset.

Run it from the repository's top folder:

    python -m src.step5_build_clean_dataset              # raw file already in data/raw/
    python -m src.step5_build_clean_dataset --source kaggle
    python -m src.step5_build_clean_dataset --source /path/to/accepted_2007_to_2018Q4.csv.gz

Plain-language summary of what happens
--------------------------------------
 1. Find the raw file and check its fingerprint (stop if it is a different file).
 2. Read only the columns we need, as text.
 3. Apply the sample filters (F.3), counting what each one removes per year.
 4. Apply the approved 148-loan interest-rate fix (G2-RATE).
 5. Check how many loans per year have a final status (F.4 resolution rule).
 6. Keep final-status loans and set the target (Charged Off = 1, Fully Paid = 0).
 7. Build the nine inputs with fixed formulas (P1-P9; proposed when step 5 ran, frozen
    later on 24 Nov 2026 in src/features.py). Nothing is learned.
 8. Write four loan-level files to data/clean/ (NOT uploaded to GitHub):
      step5_inputs.parquet                    inputs for all years 2010-2015
      step5_audit.parquet                     lineage: stored rate, raw labels,
                                              verification_status (audit only)
      step5_outcomes_2010_2013.parquet        target + cash fields, train/validation
      SEALED_step5_outcomes_2014_2015.parquet target + cash fields, test/robustness.
                                              Written once, fingerprinted, never
                                              summarised before the step 8 freeze.
 9. Write small summary tables to outputs/step5/ (these ARE committed).
10. Check the raw file's fingerprint again (proof it was not changed).

What this script deliberately does NOT do
-----------------------------------------
* It does not freeze the feature list (Lucas did that afterwards, at Gate 2).
* It does not fit anything: no medians, scaling, category lists or models.
* It does not compute any outcome statistic for 2014 or 2015.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, inputs, rates, raw_data, sample

DEV_YEARS = [y for y in config.YEARS if y not in config.SEALED_YEARS]


# ---------------------------------------------------------------------------
# The "seal": any outcome summary must go through this guard.
# ---------------------------------------------------------------------------
def dev_years_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only 2010-2013 rows, and prove no sealed year slipped through."""
    out = df[df["vintage"].isin(DEV_YEARS)]
    assert not out["vintage"].isin(config.SEALED_YEARS).any(), "SEAL BROKEN"
    return out


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Summary tables (inputs only, or outcomes for 2010-2013 only)
# ---------------------------------------------------------------------------
def input_reports(feat: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Missing shares, distributions, out-of-training-range counts, categories."""
    num = config.PROPOSED_NUMERIC_INPUTS
    by = feat.groupby("vintage")

    missing = by[num + config.PROPOSED_NOMINAL_INPUTS].apply(lambda g: g.isna().mean())

    rows = []
    for y, g in by:
        for c in num:
            s = g[c].dropna()
            rows.append({"vintage": y, "input": c, "n_nonmissing": len(s),
                         "mean": s.mean(), "min": s.min(), "p01": s.quantile(0.01),
                         "median": s.median(), "p99": s.quantile(0.99), "max": s.max()})
    dist = pd.DataFrame(rows)

    # Range seen in the TRAINING years (2010-2012). Descriptive only: used to
    # count how many later loans fall outside it (approved change-log row,
    # "input-only out-of-range count per year"). Nothing is capped.
    train = feat[feat["split"] == "train"]
    lo, hi = train[num].min(), train[num].max()
    rows = []
    for y, g in by:
        for c in num:
            rows.append({"vintage": y, "input": c,
                         "train_min": lo[c], "train_max": hi[c],
                         "below_train_min": int((g[c] < lo[c]).sum()),
                         "above_train_max": int((g[c] > hi[c]).sum()),
                         "n": len(g)})
    oor = pd.DataFrame(rows)

    rows = []
    train_labels = {c: set(train[c].dropna()) for c in config.PROPOSED_NOMINAL_INPUTS}
    for c in config.PROPOSED_NOMINAL_INPUTS:
        counts = feat.groupby(["vintage", c], dropna=False).size().rename("n").reset_index()
        counts = counts.rename(columns={c: "label"})
        counts["input"] = c
        counts["share_of_year"] = counts["n"] / counts.groupby("vintage")["n"].transform("sum")
        counts["seen_in_train_2010_2012"] = counts["label"].isin(train_labels[c])
        rows.append(counts)
    cats = pd.concat(rows)[["input", "vintage", "label", "n", "share_of_year",
                            "seen_in_train_2010_2012"]]
    return {"input_missing_share_by_vintage": missing,
            "input_distribution_by_vintage": dist,
            "input_out_of_training_range_by_vintage": oor,
            "category_counts_by_vintage": cats}


def cash_identity_check(out_dev: pd.DataFrame) -> pd.DataFrame:
    """
    Control document B8 (2010-2013 only):
      principal + interest + late fees + recoveries should equal total_pymnt.
    We report the size of the gap per year.
    """
    gap = (out_dev["total_rec_prncp"] + out_dev["total_rec_int"]
           + out_dev["total_rec_late_fee"] + out_dev["recoveries"]
           - out_dev["total_pymnt"])
    g = pd.DataFrame({"vintage": out_dev["vintage"], "abs_gap": gap.abs()})
    return g.groupby("vintage").agg(
        loans=("abs_gap", "size"),
        max_abs_gap_usd=("abs_gap", "max"),
        loans_gap_over_1usd=("abs_gap", lambda s: int((s > 1).sum())),
    )


# ---------------------------------------------------------------------------
def main(source: str | None) -> None:
    t0 = time.time()
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    config.STEP5_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Raw file + fingerprint ------------------------------------------------
    path = raw_data.fetch_raw_file(source)
    log(f"Checking fingerprint of {path.name} ...")
    raw_hash_before = raw_data.verify_raw_fingerprint(path)
    log("Fingerprint OK")

    # 2. Load -------------------------------------------------------------------
    log("Reading raw columns (takes a few minutes) ...")
    raw = raw_data.load_raw_columns(path)
    raw_rows = len(raw)
    log(f"Read {raw_rows:,} rows x {raw.shape[1]} columns")

    # 3. Filters 1-5 ------------------------------------------------------------
    pool, funnel = sample.build_candidate_pool(raw)
    del raw
    log(f"Candidate pool: {len(pool):,} loans")

    # 4. Approved rate fix ------------------------------------------------------
    pool, rate_audit, rate_info, still_flagged = rates.apply_approved_rate_fix(pool)
    log(f"Rate fix: {rate_info['cases_found_and_corrected']} loans corrected; "
        f"{rate_info['screen_after_union']} still flagged after fix")

    # 5. Resolution rule --------------------------------------------------------
    resolution = sample.resolution_table(pool)
    if not resolution.loc[2014, "resolved_share"] >= config.TEST_FALLBACK_THRESHOLD:
        raise ValueError("2014 below 98% resolved: F.4 fallback design applies. Stop.")

    # 6. Research sample + target ----------------------------------------------
    df = sample.keep_resolved(pool)
    funnel["6_resolved_research_sample"] = df.groupby("vintage").size()
    log(f"Research sample: {len(df):,} loans (matches the Gate 1 audit counts)")

    # 7. Inputs -----------------------------------------------------------------
    feat = inputs.build_inputs(df)
    feat.insert(1, "vintage", df["vintage"].values)
    feat.insert(2, "split", df["split"].values)
    feat.insert(3, "grade", df["grade"].str.strip().values)        # selection stratum
    feat.insert(4, "sub_grade", df["sub_grade"].str.strip().values)  # platform rule
    feat = feat.reset_index(drop=True)
    log("Nine proposed inputs built")

    # 8. Write loan-level files -------------------------------------------------
    audit = pd.DataFrame({
        "id": df["id"].values,
        "vintage": df["vintage"].values,
        "int_rate_stored": df["int_rate_stored"].values,
        "int_rate_working": df["int_rate"].values,
        "rate_overridden": df["rate_overridden"].values,
        "verification_status_raw": df["verification_status"].values,  # audit only
        "emp_length_raw": df["emp_length"].values,
        "home_ownership_raw": df["home_ownership"].values,
        "purpose_raw": df["purpose"].values,
        "annual_inc_raw": df["annual_inc"].values,
    })
    outcomes = pd.DataFrame({"id": df["id"].values, "vintage": df["vintage"].values,
                             "split": df["split"].values, "target": df["target"].values})
    for c in config.OUTCOME_COLUMNS:
        outcomes[c] = sample.strict_number(df[c], c).values

    files = {
        "step5_inputs.parquet": feat,
        "step5_audit.parquet": audit,
        "step5_outcomes_2010_2013.parquet": dev_years_only(outcomes),
        "SEALED_step5_outcomes_2014_2015.parquet":
            outcomes[outcomes["vintage"].isin(config.SEALED_YEARS)],
    }
    fingerprints = {}
    for name, frame in files.items():
        p = config.CLEAN_DIR / name
        frame.to_parquet(p, index=False)
        fingerprints[name] = {"rows": len(frame), "columns": list(frame.columns),
                              "sha256": file_sha256(p)}
    sealed_rows = len(files["SEALED_step5_outcomes_2014_2015.parquet"])
    del outcomes, files
    log("Loan-level files written to data/clean/")

    # 9. Summary tables ---------------------------------------------------------
    out_dev = pd.read_parquet(config.CLEAN_DIR / "step5_outcomes_2010_2013.parquet")
    out_dev = dev_years_only(out_dev)

    sample_table = pd.DataFrame({
        "split": pd.Series(config.SPLIT_BY_YEAR),
        "research_sample_loans": feat.groupby("vintage").size(),
        "resolved_share": resolution["resolved_share"],
        "mean_int_rate_working_pct": feat.groupby("vintage")["int_rate"].mean(),
        "mean_int_rate_stored_pct": audit.groupby("vintage")["int_rate_stored"].mean(),
        "rate_fixes_applied": audit.groupby("vintage")["rate_overridden"].sum(),
        "charge_off_rate": out_dev.groupby("vintage")["target"].mean(),
    })
    sample_table.index.name = "vintage"
    sample_table["charge_off_rate"] = sample_table["charge_off_rate"].astype(object)
    sample_table.loc[config.SEALED_YEARS, "charge_off_rate"] = "SEALED until step 8"

    tables = {
        "funnel_by_vintage": funnel,
        "resolution_by_vintage": resolution,
        "sample_by_vintage": sample_table,
        "rate_fix_audit_148": rate_audit,
        "rate_screens_still_flagged_after_fix": still_flagged,
        "cash_identity_check_2010_2013": cash_identity_check(out_dev),
        **input_reports(feat),
    }
    for name, t in tables.items():
        keep_index = name not in {"rate_fix_audit_148", "rate_screens_still_flagged_after_fix",
                                  "input_distribution_by_vintage",
                                  "input_out_of_training_range_by_vintage",
                                  "category_counts_by_vintage"}
        t.to_csv(config.STEP5_OUT / f"{name}.csv", index=keep_index, float_format="%.6g")

    # 10. Raw fingerprint again ------------------------------------------------
    raw_hash_after = raw_data.verify_raw_fingerprint(path)

    import pyarrow
    import sklearn
    summary = {
        "step": "Roadmap step 5: clean dataset",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__,
                     "pyarrow": pyarrow.__version__, "scikit-learn": sklearn.__version__},
        "raw_file": config.RAW_FILENAME,
        "raw_sha256_before": raw_hash_before,
        "raw_sha256_after": raw_hash_after,
        "raw_unchanged": raw_hash_before == raw_hash_after == config.RAW_SHA256,
        "raw_rows_read": raw_rows,
        "candidate_pool_loans": len(pool),
        "research_sample_loans": len(feat),
        "research_sample_by_split": feat["split"].value_counts().to_dict(),
        "rate_fix": rate_info,
        "decisions_applied": [
            "G2-RATE-2026-11-24-v1 (148 listing rates)",
            "G2-VERIFY-EXCLUDE-2026-11-24-v1 (verification_status not an input; kept in audit file)",
            "G2-PREP-2026-11-24-v1 (fixed per-loan rules P1-P9 only; nothing fitted)",
        ],
        "proposed_numeric_inputs": config.PROPOSED_NUMERIC_INPUTS,
        "proposed_nominal_inputs": config.PROPOSED_NOMINAL_INPUTS,
        "features_frozen": config.FEATURES_FROZEN,
        "anything_fitted": False,
        "sealed_outcomes": {
            "years": config.SEALED_YEARS,
            "file": "data/clean/SEALED_step5_outcomes_2014_2015.parquet",
            "rows": sealed_rows,
            "statistics_computed": "none (row count and file fingerprint only)",
        },
        "clean_files": fingerprints,
    }
    (config.STEP5_OUT / "step5_run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"Done in {summary['runtime_seconds']} s. Summaries in outputs/step5/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Roadmap step 5: build the clean dataset")
    ap.add_argument("--source", default=None,
                    help="'kaggle' or a path to the raw .csv.gz (default: data/raw/)")
    main(ap.parse_args().source)

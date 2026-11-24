"""
step11_robustness.py: ROADMAP STEP 11, robustness (control document F.12, freeze record F.5 / G4).

    python -m src.step11_robustness --dry-run   # rehearsal: 2014 (already open) as stand-in for 2015
    python -m src.step11_robustness --unseal    # THE robustness run: unseal 2015 once

Plain-language summary
----------------------
Two robustness checks, both fixed in the signed freeze record:

 1. A LATER YEAR (2015). The same final models (fitted once on 2010-2013, the
    frozen settings) score the 2015 loans; each approach funds 80% within each
    grade; the same 1,000-draw paired bootstrap (seed 42) and the same four
    category rules are applied (G4). 2015 is "consistent" with 2014 if exactly
    the same categories hold. 2015 can never change the 2014 category.
 2. GRADE GROUPS A-B, C and D-G (D, E, F+G), for 2014 and 2015: net cash gain
    per dollar and charge-off rate per approach, using the loans each
    approach already funded (point estimates, as F.12 specifies).

The --dry-run runs the exact same code with 2014 in the place of 2015. Because
2014 was already opened in step 9, this touches nothing sealed, and its money
results must reproduce step 9 exactly (a built-in check that the code is right).
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from . import config, load_clean, metrics
from .step6_baselines import log
from .step9_test_2014 import APPROACHES, SHORT, load_freeze, run

GROUPS = {"A-B": ["A", "B"], "C": ["C"], "D-G": ["D", "E", "F", "G"]}


def grade_groups(df: pd.DataFrame, scores: pd.DataFrame, year: int) -> pd.DataFrame:
    """Net cash gain and charge-off rate per approach within each grade group."""
    d = df.merge(scores, on="id", validate="one_to_one")
    rows = []
    for g, letters in GROUPS.items():
        part = d[d["grade"].isin(letters)]
        for a in APPROACHES:
            chosen = part[part[f"selected_{SHORT[a]}"]]
            fo = metrics.financial_outcomes(chosen, 0.01)
            rows.append({"year": year, "grade_group": g, "approach": a, "loans_funded": fo["loans"],
                         "loans_in_group": len(part),
                         "net_cash_gain_per_dollar": fo["net_cash_gain_per_dollar"],
                         "charge_off_rate": fo["charge_off_rate"]})
    return pd.DataFrame(rows)


def main(mode: str):
    rec = load_freeze()
    inputs = load_clean.load_inputs()
    dev = inputs.merge(load_clean.load_dev_outcomes()[["id", "target"] + config.OUTCOME_COLUMNS],
                       on="id", how="inner", validate="one_to_one")
    fit_df = dev[dev["vintage"].isin(rec["final_fit_years"])]
    cats_2014 = pd.read_csv(config.REPO_ROOT / "outputs" / "step9" / "success_categories_2014.csv",
                            index_col=0)["applies"].astype(bool).to_dict()

    def year_frame(year):
        out = load_clean.load_sealed_outcomes(year, rec["date_local"])
        return inputs[inputs["vintage"] == year].merge(
            out[["id", "target"] + config.OUTCOME_COLUMNS], on="id", how="inner", validate="one_to_one")

    t2014 = year_frame(2014)   # already opened in step 9

    if mode == "dry-run":
        out_dir = config.REPO_ROOT / "outputs" / "step11_dryrun_2014"
        summary, scores = run(fit_df, t2014, 2014, out_dir, rec, "robustness dry run")
        # Built-in check: must reproduce the frozen step 9 run exactly.
        a = pd.read_csv(out_dir / "financial_2014_q80.csv")
        b = pd.read_csv(config.REPO_ROOT / "outputs" / "step9" / "financial_2014_q80.csv")
        pd.testing.assert_frame_equal(a, b)
        saved = pd.read_parquet(config.CLEAN_DIR / "step9_scores_selection_2014.parquet")
        pd.testing.assert_frame_equal(scores.reset_index(drop=True), saved.reset_index(drop=True))
        grade_groups(t2014, scores, 2014).to_csv(out_dir / "grade_groups_2014.csv", index=False,
                                                 float_format="%.6f")
        summary["reproduces_step9_exactly"] = True
        summary["consistent_with_2014_if_this_were_2015"] = summary["categories"] == cats_2014
        (out_dir / "dryrun_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log("Dry run reproduces step 9 exactly. 2015 untouched.")
        return

    # ---- THE ONE-TIME 2015 RUN ---------------------------------------------
    out_dir = config.REPO_ROOT / "outputs" / "step11"
    opened_log = out_dir / "robustness_2015_opened_log.json"
    if opened_log.exists():
        raise PermissionError("2015 has already been opened once; rerun only as a logged bug fix.")
    out_dir.mkdir(parents=True, exist_ok=True)
    opened_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t2015 = year_frame(2015)
    assert len(t2015) == 282787
    summary, scores15 = run(fit_df, t2015, 2015, out_dir, rec, "robustness")
    scores15.to_parquet(config.CLEAN_DIR / "step11_scores_selection_2015.parquet", index=False)
    scores14 = pd.read_parquet(config.CLEAN_DIR / "step9_scores_selection_2014.parquet")
    pd.concat([grade_groups(t2014, scores14, 2014), grade_groups(t2015, scores15, 2015)]).to_csv(
        out_dir / "grade_groups_2014_2015.csv", index=False, float_format="%.6f")
    consistent = summary["categories"] == cats_2014
    summary.update({
        "categories_2014": cats_2014,
        "consistent_with_2014": consistent,
        "rule": "G4: consistent = exactly the same F.11 categories hold in 2015 as in 2014",
        "opened_utc": opened_at,
        "freeze_record_sha256": rec["record_sha256"],
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                      cwd=config.REPO_ROOT).stdout.strip(),
    })
    opened_log.write_text(json.dumps(summary, indent=2, default=str))
    log(f"2015 robustness done. Categories: {summary['categories']}. Consistent with 2014: {consistent}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--unseal", action="store_true")
    a = ap.parse_args()
    main("dry-run" if a.dry_run else "unseal")

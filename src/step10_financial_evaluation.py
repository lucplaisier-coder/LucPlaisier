"""
step10_financial_evaluation.py: ROADMAP STEP 10, tables and figures from the 2014 run.

    python -m src.step10_financial_evaluation

Plain-language summary
----------------------
Nothing is re-fitted or re-chosen here. This step only READS the frozen
2014 run (step 9: the saved predicted probabilities and funded/not-funded
flags) and turns it into the outputs listed in the control document (F.13)
and the freeze record:

  Output 1  sample and vintage table (2010-2014; 2015 still sealed)
  Output 2  predictive table: AUC, Brier, ECE (2013 = "used for tuning", 2014 = test)
  Output 3  reliability figure 2014 (copied from step 9)
  Output 4  net cash gain against the selection rate q = 50%..100% (G1, display only)
  Output 6  main financial table at 80% with bootstrap intervals
  Annex     sub-grade mix of funded loans; fee-0% sensitivity; input whitelist/blacklist

Grade groups and 2015 are step 11; odds ratios and permutation importance are step 12.
"""

import json
import shutil

import matplotlib
matplotlib.use("Agg")
from . import plotstyle  # noqa: F401  (same fonts on every computer)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, features, load_clean, metrics, selection
from .step6_baselines import log

OUT = config.REPO_ROOT / "outputs" / "step10"
S9 = config.REPO_ROOT / "outputs" / "step9"
S7 = config.REPO_ROOT / "outputs" / "step7"
APPROACHES = ["A. Fund-all", "B. Platform rule", "C. Logit", "D. HGB"]
SHORT = {"A. Fund-all": "fundall", "B. Platform rule": "platform", "C. Logit": "logit", "D. HGB": "hgb"}
Q_GRID = [round(0.50 + 0.05 * i, 2) for i in range(11)]   # G1: 50%, 55%, ..., 100%


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = json.loads((config.DECISION_DIR / "methodology_freeze_v1.json").read_text())
    run_log = json.loads((S9 / "test_opened_log.json").read_text())
    assert run_log["freeze_record_sha256"] == rec["record_sha256"]

    # ---- Load the frozen 2014 run -----------------------------------------
    inputs = load_clean.load_inputs()
    t14 = inputs[inputs["vintage"] == 2014].merge(
        load_clean.load_sealed_outcomes(2014, rec["date_local"])[["id", "target"] + config.OUTCOME_COLUMNS],
        on="id", validate="one_to_one")
    scores = pd.read_parquet(config.CLEAN_DIR / "step9_scores_selection_2014.parquet")
    df = t14.merge(scores, on="id", validate="one_to_one")
    df = df.sort_values("id", key=lambda s: s.astype("int64")).reset_index(drop=True)

    # Consistency check: re-applying the frozen rule to the saved scores must
    # give exactly the saved 80% selections (proves nothing drifted).
    strata = selection.strata_for(df["grade"], True)
    key = selection.tie_break_key(df["id"])
    rule = {"B. Platform rule": selection.platform_rule(df, strata, key),
            "C. Logit": selection.model_rule(df["pd_logit"], strata, key),
            "D. HGB": selection.model_rule(df["pd_hgb"], strata, key)}
    for a, s in rule.items():
        assert (s.to_numpy() == df[f"selected_{SHORT[a]}"].to_numpy()).all(), f"{a} drifted"
    log("Saved 2014 selections reproduce exactly")

    # ---- Output 1: sample and vintage table --------------------------------
    dev = load_clean.load_dev_outcomes()
    s5 = pd.read_csv(config.STEP5_OUT / "sample_by_vintage.csv", index_col="vintage")
    t1 = s5[["split", "research_sample_loans", "resolved_share", "mean_int_rate_working_pct"]].copy()
    co = dev.groupby("vintage")["target"].mean()
    co.loc[2014] = df["target"].mean()
    t1["charge_off_rate"] = co.astype(object)
    t1.loc[2015, "charge_off_rate"] = "SEALED until step 11"
    t1.to_csv(OUT / "output1_sample_by_vintage.csv", float_format="%.5f")

    # ---- Output 2: predictive table ----------------------------------------
    p13 = pd.read_csv(S7 / "predictive_validation_2013.csv").iloc[:2]
    p13["model"] = ["Logit", "HGB"]
    p13["year"] = "2013 validation (dev fit 2010-2012, used for tuning)"
    p14 = pd.read_csv(S9 / "predictive_2014.csv")
    p14["year"] = "2014 test (final fit 2010-2013)"
    pd.concat([p13[["year", "model", "auc", "brier", "ece"]], p14[["year", "model", "auc", "brier", "ece"]]]) \
        .to_csv(OUT / "output2_predictive.csv", index=False, float_format="%.5f")

    # ---- Output 3: reliability figure --------------------------------------
    shutil.copyfile(S9 / "reliability_2014.png", OUT / "output3_reliability_2014.png")

    # ---- Output 4: net cash gain vs selection rate (display only) ----------
    rows = []
    for q in Q_GRID:
        sels = {"A. Fund-all": pd.Series(True, index=df.index),
                "B. Platform rule": selection.platform_rule(df, strata, key, q),
                "C. Logit": selection.model_rule(df["pd_logit"], strata, key, q),
                "D. HGB": selection.model_rule(df["pd_hgb"], strata, key, q)}
        for a in APPROACHES:
            fo = metrics.financial_outcomes(df[sels[a]], 0.01)
            rows.append({"q": q, "approach": a, "loans": fo["loans"],
                         "net_cash_gain_per_dollar": fo["net_cash_gain_per_dollar"],
                         "charge_off_rate": fo["charge_off_rate"]})
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "output4_selection_rate_curve_2014.csv", index=False, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    styles = {"A. Fund-all": dict(color="grey", ls="--"), "B. Platform rule": dict(color="#8c564b"),
              "C. Logit": dict(color="#1f77b4"), "D. HGB": dict(color="#ff7f0e")}
    for a in APPROACHES:
        c = curve[curve.approach == a]
        ax.plot(c["q"] * 100, c["net_cash_gain_per_dollar"] * 100, marker="o", ms=3, label=a, **styles[a])
    ax.axvline(80, color="black", lw=0.8, ls=":")
    ax.text(80.5, ax.get_ylim()[0], " frozen q = 80%", fontsize=7, va="bottom")
    ax.set_xlabel("Share of loans funded within each grade (%)")
    ax.set_ylabel("Net cash gain per $ invested (cents)")
    ax.set_title("2014 test: net cash gain by selection rate (display only)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "output4_selection_rate_curve_2014.png")
    plt.close(fig)

    # ---- Output 6: main financial table ------------------------------------
    fin = pd.read_csv(S9 / "financial_2014_q80.csv")
    boot = pd.read_csv(S9 / "bootstrap_contrasts_2014.csv")
    fin.to_csv(OUT / "output6a_financial_2014_q80.csv", index=False, float_format="%.6f")
    main_c = boot[boot.metric.isin(["net_cash_gain_per_dollar", "charge_off_rate",
                                    "principal_loss_rate", "auc"])].copy()
    main_c["significant_at_95"] = (main_c.ci95_low > 0) | (main_c.ci95_high < 0)
    main_c.to_csv(OUT / "output6b_contrasts_2014_q80.csv", index=False, float_format="%.6f")

    # ---- Annex -------------------------------------------------------------
    mix = {a: df.loc[df[f"selected_{SHORT[a]}"], "sub_grade"].value_counts(normalize=True)
           for a in APPROACHES}
    pd.DataFrame(mix).sort_index().fillna(0).to_csv(OUT / "annex_subgrade_mix_2014.csv", float_format="%.5f")
    boot[boot.metric == "net_cash_gain_per_dollar_sf0"].to_csv(
        OUT / "annex_fee0_sensitivity_2014.csv", index=False, float_format="%.6f")
    pd.DataFrame({"input": features.ALL_INPUTS,
                  "type": ["numeric"] * len(features.NUMERIC_INPUTS) + ["category"] * len(features.NOMINAL_INPUTS),
                  "status": "whitelist (frozen Gate 2)"}).to_csv(OUT / "annex_inputs_whitelist.csv", index=False)
    log(f"Step 10 outputs written to {OUT}")


if __name__ == "__main__":
    main()

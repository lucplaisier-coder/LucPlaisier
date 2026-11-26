"""
step12_interpretability.py: ROADMAP STEP 12, what drives the models (F.12, freeze G2/G3).

    python -m src.step12_interpretability

Plain-language summary
----------------------
Nothing new is chosen. Both final models are refitted with the frozen settings
on 2010-2013 (exactly as in step 9; the code checks the 2014 predictions are
identical to the saved ones) and then:

 1. LOGIT ODDS RATIOS (G3). For the five borrower numbers (FICO, DTI, log income,
    employment length, loan amount): how the odds of charge-off change when the
    input rises by one standard deviation (measured in 2010-2013), HOLDING THE
    PLATFORM'S PRICE (sub-grade and interest rate) FIXED. Odds ratio > 1 means
    riskier; < 1 means safer. All other coefficients are listed in an annex
    without interpretation.
 2. HGB PERMUTATION IMPORTANCE on 2014 (G2). Shuffle one input across the 2014
    loans (breaking its link with the outcome), re-score, and measure how much
    the AUC drops. 10 repeats, seed 42. A big drop = the model relies on it.
    Sub-grade and interest rate are also shuffled TOGETHER as one "platform
    pricing" block, because they carry overlapping information and would each
    look less important on their own.
"""

import json

import matplotlib
matplotlib.use("Agg")
from . import plotstyle  # noqa: F401  (same fonts on every computer)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, features, load_clean, models
from .step6_baselines import fit_and_score, log
from .step9_test_2014 import load_freeze

OUT = config.REPO_ROOT / "outputs" / "step12"
BORROWER_NUMBERS = ["fico_mid", "dti", "log_annual_inc", "emp_length_ord", "loan_amnt"]
LABELS = {"fico_mid": "FICO (midpoint)", "dti": "Debt-to-income", "log_annual_inc": "Log annual income",
          "emp_length_ord": "Employment length", "loan_amnt": "Loan amount", "sub_grade_ord": "Sub-grade",
          "int_rate": "Interest rate", "home_ownership": "Home ownership", "purpose": "Loan purpose",
          "PLATFORM PRICING BLOCK": "Platform pricing block (sub-grade + rate)"}
REPEATS, SEED = 10, 42


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = load_freeze()
    inputs = load_clean.load_inputs()
    dev = inputs.merge(load_clean.load_dev_outcomes()[["id", "target"]], on="id", validate="one_to_one")
    fit_df = dev[dev["vintage"].isin(rec["final_fit_years"])]
    t14 = inputs[inputs["vintage"] == 2014].merge(
        load_clean.load_sealed_outcomes(2014, rec["date_local"])[["id", "target"]], on="id", validate="one_to_one")
    t14 = t14.sort_values("id", key=lambda s: s.astype("int64")).reset_index(drop=True)
    X_fit, y_fit = models.model_frame(fit_df), fit_df["target"].to_numpy()
    X14, y14 = models.model_frame(t14), t14["target"].to_numpy()

    lg, hg = rec["final_models"]["logit"], rec["final_models"]["hgb"]
    logit = models.logit_pipeline(lg["C"])
    p_logit, _ = fit_and_score(logit, X_fit, y_fit, X14)
    hgb = models.hgb_pipeline(hg["max_depth"], hg["max_iter"])
    p_hgb, _ = fit_and_score(hgb, X_fit, y_fit, X14)

    # Same models as the frozen step 9 run?
    saved = pd.read_parquet(config.CLEAN_DIR / "step9_scores_selection_2014.parquet")
    assert (saved["id"].to_numpy() == t14["id"].to_numpy()).all()
    assert np.allclose(saved["pd_logit"], p_logit, rtol=0, atol=1e-12)
    assert np.allclose(saved["pd_hgb"], p_hgb, rtol=0, atol=1e-12)
    log("Final models reproduce the frozen step 9 predictions exactly")

    # ---- 1. Logit odds ratios ----------------------------------------------
    names = logit.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(logit.named_steps["logit"].coef_[0], index=names)
    scaler = logit.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    sd = pd.Series(scaler.scale_, index=features.NUMERIC_INPUTS)
    rows = []
    for c in BORROWER_NUMBERS:
        b = coef[f"num__{c}"]
        rows.append({"input": c, "label": LABELS[c], "sd_2010_2013": sd[c], "coefficient_per_sd": b,
                     "odds_ratio_per_sd": np.exp(b),
                     "reading": f"+1 SD -> odds of charge-off x{np.exp(b):.3f}, holding sub-grade and rate fixed"})
    orr = pd.DataFrame(rows)
    orr.to_csv(OUT / "logit_odds_ratios_per_sd.csv", index=False, float_format="%.5f")
    annex = coef.rename("coefficient").to_frame()
    annex.loc["intercept", "coefficient"] = logit.named_steps["logit"].intercept_[0]
    annex["interpreted"] = [n in {f"num__{c}" for c in BORROWER_NUMBERS} for n in annex.index]
    annex.to_csv(OUT / "annex_logit_all_coefficients_final_fit.csv", float_format="%.5f")

    # ---- 2. HGB permutation importance on 2014 -----------------------------
    base_auc = roc_auc_score(y14, p_hgb)
    groups = {c: [c] for c in features.ALL_INPUTS}
    groups["PLATFORM PRICING BLOCK"] = features.PLATFORM_PRICING_BLOCK
    rng = np.random.default_rng(SEED)
    rows = []
    for name, cols in groups.items():
        for r in range(REPEATS):
            perm = rng.permutation(len(X14))
            Xp = X14.copy()
            for c in cols:                       # block: the SAME shuffle for both columns
                Xp[c] = X14[c].to_numpy()[perm]
            auc = roc_auc_score(y14, hgb.predict_proba(Xp)[:, 1])
            rows.append({"input": name, "repeat": r, "auc_drop": base_auc - auc})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "hgb_permutation_importance_2014_raw.csv", index=False, float_format="%.6f")
    imp = raw.groupby("input")["auc_drop"].agg(["mean", "std"]).rename(
        columns={"mean": "mean_auc_drop", "std": "sd_over_repeats"}).sort_values("mean_auc_drop", ascending=False)
    imp["label"] = [LABELS[i] for i in imp.index]
    imp["base_auc_2014"] = base_auc
    imp.to_csv(OUT / "hgb_permutation_importance_2014.csv", float_format="%.6f")

    # Output 5 figure
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    order = imp.sort_values("mean_auc_drop")
    colors = ["#8c564b" if i == "PLATFORM PRICING BLOCK" else "#ff7f0e" for i in order.index]
    ax.barh(order["label"], order["mean_auc_drop"], xerr=order["sd_over_repeats"], color=colors, capsize=2)
    ax.set_xlabel("Drop in 2014 AUC when shuffled (mean of 10 repeats)")
    ax.set_title(f"HGB permutation importance, 2014 test\n(base AUC {base_auc:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "output5_permutation_importance_2014.png")
    plt.close(fig)

    (OUT / "step12_run_summary.json").write_text(json.dumps({
        "final_models_reproduce_step9": True, "base_auc_2014_hgb": base_auc,
        "permutation_repeats": REPEATS, "seed": SEED, "metric": "drop in AUC (G2)",
        "odds_ratios_for": BORROWER_NUMBERS, "freeze_record_sha256": rec["record_sha256"]}, indent=2))
    log("Step 12 outputs written")


if __name__ == "__main__":
    main()

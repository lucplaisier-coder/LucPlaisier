"""
step6_baselines.py: ROADMAP STEP 6, logistic regression + platform benchmark.

Run from the repository's top folder (after step 5):

    python -m src.step6_baselines

Plain-language summary of what happens
--------------------------------------
 1. Open the step-5 files through load_clean (fingerprints checked). Only
    2010-2013 outcomes are opened; 2014-2015 stay sealed.
 2. Selection groups: decide whether grades F and G are merged (F.7 rule,
    uses only loan counts by grade in 2014, never outcomes).
 3. Leakage alarm (F.6): fit a one-input logit for each of the nine inputs on
    2010-2012 and score 2013. Any single input with AUC above 0.80 stops the run.
 4. Logistic regression: fit on 2010-2012 for C = 0.01, 0.1 and 1; score 2013;
    choose C by AUC (ties within 0.002 go to the smaller C). Full-model AUC
    above 0.85 stops the run.
 5. On 2013 (validation, "used for tuning"), compare three approaches at 80%
    within each grade: fund-all, platform rule, logit. Report charge-off rate,
    principal loss rate and net cash gain per dollar (service fee 1% and 0%).
 6. Write tables and a reliability figure to outputs/step6/.

What this script deliberately does NOT do
-----------------------------------------
* No HistGB (that is step 7).
* No refit on 2010-2013 and nothing on 2014/2015 (that is step 9, after the
  step 8 freeze).
* No bootstrap / confidence intervals (only on the test, step 10).
"""

import json
import time
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
from . import plotstyle  # noqa: F401  (same fonts on every computer)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from . import config, features, load_clean, metrics, models, selection

OUT = config.REPO_ROOT / "outputs" / "step6"
TRAIN_YEARS, VALID_YEAR = [2010, 2011, 2012], 2013
SINGLE_INPUT_ALARM, FULL_MODEL_ALARM = 0.80, 0.85


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_and_score(pipe, X_tr, y_tr, X_va):
    """Fit on training rows; return validation probabilities and any warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X_tr, y_tr)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return pipe.predict_proba(X_va)[:, 1], converged


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    inputs = load_clean.load_inputs()
    outcomes = load_clean.load_dev_outcomes()          # 2010-2013 only
    assert set(outcomes["vintage"]) <= {2010, 2011, 2012, 2013}
    dev = inputs.merge(outcomes[["id", "target"] + config.OUTCOME_COLUMNS], on="id",
                       how="inner", validate="one_to_one")
    train = dev[dev["vintage"].isin(TRAIN_YEARS)].copy()
    valid = dev[dev["vintage"] == VALID_YEAR].copy()
    log(f"Train 2010-2012: {len(train):,} loans | Validation 2013: {len(valid):,} loans")

    X_tr, y_tr = models.model_frame(train), train["target"].to_numpy()
    X_va, y_va = models.model_frame(valid), valid["target"].to_numpy()
    models.check_training_columns(X_tr)

    # 2. Selection groups ---------------------------------------------------
    merge_fg, test_grade_counts = selection.decide_fg_merge(inputs)
    strata_va = selection.strata_for(valid["grade"], merge_fg)
    key_va = selection.tie_break_key(valid["id"])
    log(f"Grades F and G merged: {merge_fg} (2014 counts F={test_grade_counts.get('F', 0)}, "
        f"G={test_grade_counts.get('G', 0)})")

    # 3. Leakage alarm: one input at a time --------------------------------
    rows = []
    for c in features.ALL_INPUTS:
        num = [c] if c in features.NUMERIC_INPUTS else []
        nom = [c] if c in features.NOMINAL_INPUTS else []
        pipe = models.logit_pipeline(1.0, numeric=num, nominal=nom)
        p, _ = fit_and_score(pipe, models.model_frame(train, num, nom), y_tr,
                             models.model_frame(valid, num, nom))
        auc = roc_auc_score(y_va, p)
        rows.append({"input": c, "validation_auc_2013": auc,
                     "alarm_threshold": SINGLE_INPUT_ALARM, "alarm": auc > SINGLE_INPUT_ALARM})
    single = pd.DataFrame(rows).sort_values("validation_auc_2013", ascending=False)
    single.to_csv(OUT / "leakage_alarm_single_input_auc_2013.csv", index=False, float_format="%.4f")
    if single["alarm"].any():
        raise RuntimeError("LEAKAGE ALARM: a single input has validation AUC > 0.80. Stop and audit.")
    log(f"Leakage alarm (single inputs): highest AUC {single['validation_auc_2013'].max():.3f}, no alarm")

    # 4. Logit: tune C on 2013 ---------------------------------------------
    rows, probs, pipes = [], {}, {}
    for C in models.C_GRID:
        pipe = models.logit_pipeline(C)
        p, conv = fit_and_score(pipe, X_tr, y_tr, X_va)
        s = metrics.predictive_scores(y_va, p)
        rows.append({"C": C, **{f"validation_{k}": v for k, v in s.items()}, "converged": conv})
        probs[C], pipes[C] = p, pipe
    tuning = pd.DataFrame(rows)
    C_best = models.choose_c(dict(zip(tuning["C"], tuning["validation_auc"])))
    tuning["chosen"] = tuning["C"] == C_best
    tuning.to_csv(OUT / "logit_tuning_C_2013.csv", index=False, float_format="%.5f")
    if tuning["validation_auc"].max() > FULL_MODEL_ALARM:
        raise RuntimeError("LEAKAGE ALARM: full-model validation AUC > 0.85. Stop and audit.")
    p_va = probs[C_best]
    log(f"Logit: chosen C = {C_best} (validation AUC {tuning.loc[tuning.chosen, 'validation_auc'].item():.4f})")

    # Coefficients of the chosen development fit (for transparency only;
    # the reported odds ratios come from the final 2010-2013 fit in step 12).
    pipe = pipes[C_best]
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pd.DataFrame({"model_column": names, "coefficient": pipe.named_steps["logit"].coef_[0]})
    coef.loc[len(coef)] = ["intercept", pipe.named_steps["logit"].intercept_[0]]
    coef.to_csv(OUT / "logit_coefficients_dev_fit_2010_2012.csv", index=False, float_format="%.5f")

    # 5. Predictive table + reliability ------------------------------------
    pred = [{"approach": "Logit (C=%g)" % C_best, **metrics.predictive_scores(y_va, p_va)}]
    # Platform: sub-grade is a rank, not a probability -> AUC only (descriptive).
    pred.append({"approach": "Platform sub-grade (rank only)",
                 "auc": roc_auc_score(y_va, valid["sub_grade_ord"]), "brier": np.nan, "ece": np.nan})
    pd.DataFrame(pred).to_csv(OUT / "predictive_validation_2013.csv", index=False, float_format="%.5f")

    ece, bins = metrics.ece_and_bins(y_va, p_va)
    bins.to_csv(OUT / "reliability_bins_logit_2013.csv", float_format="%.5f")
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    lim = max(bins["mean_predicted"].max(), bins["observed_default_rate"].max()) * 1.1
    ax.plot([0, lim], [0, lim], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.plot(bins["mean_predicted"], bins["observed_default_rate"], marker="o", label="logit")
    ax.set_xlabel("Average predicted default probability")
    ax.set_ylabel("Observed charge-off rate")
    ax.set_title(f"Reliability, validation 2013 (ECE = {ece:.3f})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "reliability_logit_2013.png")
    plt.close(fig)

    # 6. Selection at 80% within grade + money ------------------------------
    sel = {
        "A. Fund-all": pd.Series(True, index=valid.index),
        "B. Platform rule": selection.platform_rule(valid, strata_va, key_va),
        "C. Logit": selection.model_rule(pd.Series(p_va, index=valid.index), strata_va, key_va),
    }
    rows, mix, by_grade = [], [], []
    for name, s in sel.items():
        chosen = valid[s]
        r = {"approach": name, **metrics.financial_outcomes(chosen, 0.01)}
        r["net_cash_gain_per_dollar_sf0"] = metrics.financial_outcomes(chosen, 0.0)["net_cash_gain_per_dollar"]
        r["mean_int_rate_pct"] = chosen["int_rate"].mean()
        rows.append(r)
        m = chosen["sub_grade"].value_counts(normalize=True).rename(name)
        mix.append(m)
        for g, part in chosen.groupby(strata_va[s]):
            by_grade.append({"approach": name, "grade_stratum": g,
                             "selected": len(part), "available": int((strata_va == g).sum()),
                             **{k: v for k, v in metrics.financial_outcomes(part).items()
                                if k in ("charge_off_rate", "net_cash_gain_per_dollar")}})
    fin = pd.DataFrame(rows)
    fin.to_csv(OUT / "financial_validation_2013_q80.csv", index=False, float_format="%.5f")
    pd.concat(mix, axis=1).sort_index().fillna(0).to_csv(
        OUT / "subgrade_mix_selected_2013.csv", float_format="%.5f")
    pd.DataFrame(by_grade).to_csv(OUT / "by_grade_validation_2013_q80.csv", index=False,
                                  float_format="%.5f")
    test_grade_counts.rename("loans_2014_inputs_only").to_csv(OUT / "grade_counts_2014_for_FG_rule.csv")

    # Run summary ---------------------------------------------------------
    import sklearn
    summary = {
        "step": "Roadmap step 6: baseline logistic + platform benchmark",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "scikit_learn": sklearn.__version__,
        "features_frozen_on": features.FROZEN_ON,
        "inputs": features.ALL_INPUTS,
        "train_years": TRAIN_YEARS, "validation_year": VALID_YEAR,
        "train_loans": len(train), "validation_loans": len(valid),
        "fg_merged": merge_fg,
        "selection_rate": selection.Q,
        "seed": config.SEED,
        "logit_C_grid": models.C_GRID, "logit_C_chosen": C_best,
        "all_fits_converged": bool(tuning["converged"].all()),
        "leakage_alarm_triggered": False,
        "sealed_2014_2015_outcomes_opened": False,
        "hgb_started": False,
    }
    (OUT / "step6_run_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Done in {summary['runtime_seconds']} s. Results in outputs/step6/")


if __name__ == "__main__":
    main()

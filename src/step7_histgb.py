"""
step7_histgb.py: ROADMAP STEP 7, add histogram gradient boosting (HGB).

Run from the repository's top folder (after step 5):

    python -m src.step7_histgb

Plain-language summary of what happens
--------------------------------------
 1. Open the step-5 files (fingerprints checked). Only 2010-2013 outcomes;
    2014-2015 stay sealed.
 2. Fit HGB on 2010-2012 for the four fixed settings in the plan
    (depth 3 or 5, 200 or 500 trees; learning rate 0.05, at least 200 loans
    per leaf, seed 42, early stopping OFF). Score each on 2013.
 3. Choose the setting with the best 2013 AUC; within 0.002 of the best, the
    simpler one wins (smaller depth, then fewer trees).
 4. Leakage alarm: a full-model AUC above 0.85 stops the run.
 5. Gate 4 check (control document F.15): does the HGB pipeline work
    correctly? (no alarm, grid completed, sensible numbers). This is NOT a
    test of whether HGB adds value; that is answered once, on 2014.
 6. On 2013 (validation, "used for tuning"), compare all four approaches at
    80% within each grade: fund-all, platform rule, logit (C fixed at the
    signed-off 0.01) and HGB.

Not done here: refit on 2010-2013, anything on 2014/2015, bootstrap.
"""

import json
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
from . import plotstyle  # noqa: F401  (same fonts on every computer)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, features, load_clean, metrics, models, selection
from .step6_baselines import FULL_MODEL_ALARM, TRAIN_YEARS, VALID_YEAR, fit_and_score, log

OUT = config.REPO_ROOT / "outputs" / "step7"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    inputs = load_clean.load_inputs()
    outcomes = load_clean.load_dev_outcomes()
    assert set(outcomes["vintage"]) <= {2010, 2011, 2012, 2013}
    dev = inputs.merge(outcomes[["id", "target"] + config.OUTCOME_COLUMNS], on="id",
                       how="inner", validate="one_to_one")
    train = dev[dev["vintage"].isin(TRAIN_YEARS)].copy()
    valid = dev[dev["vintage"] == VALID_YEAR].copy()
    X_tr, y_tr = models.model_frame(train), train["target"].to_numpy()
    X_va, y_va = models.model_frame(valid), valid["target"].to_numpy()
    models.check_training_columns(X_tr)
    models.check_category_limit(X_tr)
    log(f"Train 2010-2012: {len(train):,} | Validation 2013: {len(valid):,}")

    # 2-3. HGB grid ---------------------------------------------------------
    rows, probs = [], {}
    for depth, iters in models.HGB_GRID:
        pipe = models.hgb_pipeline(depth, iters)
        p, _ = fit_and_score(pipe, X_tr, y_tr, X_va)
        hgb = pipe.named_steps["hgb"]
        assert hgb.early_stopping is False and hgb.n_iter_ == iters, "early stopping must be off"
        rows.append({"max_depth": depth, "max_iter": iters, "trees_built": hgb.n_iter_,
                     **{f"validation_{k}": v for k, v in metrics.predictive_scores(y_va, p).items()}})
        probs[(depth, iters)] = p
        log(f"HGB depth={depth}, trees={iters}: AUC {rows[-1]['validation_auc']:.4f}")
    tuning = pd.DataFrame(rows)
    best = models.choose_hgb({(r.max_depth, r.max_iter): r.validation_auc for r in tuning.itertuples()})
    tuning["chosen"] = [(d, i) == best for d, i in zip(tuning["max_depth"], tuning["max_iter"])]
    tuning.to_csv(OUT / "hgb_tuning_2013.csv", index=False, float_format="%.5f")
    p_hgb = probs[best]

    # 4. Leakage alarm ------------------------------------------------------
    if tuning["validation_auc"].max() > FULL_MODEL_ALARM:
        raise RuntimeError("LEAKAGE ALARM: HGB validation AUC > 0.85. Stop and audit.")

    # Logit with the signed-off C (step 6) -----------------------------------
    signoff = json.loads((config.DECISION_DIR / "step6_signoff.json").read_text())
    C = signoff["logit_C_chosen_on_2013"]
    p_logit, conv = fit_and_score(models.logit_pipeline(C), X_tr, y_tr, X_va)

    # 5. Predictive comparison ---------------------------------------------
    pred = pd.DataFrame([
        {"approach": f"Logit (C={C:g})", **metrics.predictive_scores(y_va, p_logit)},
        {"approach": f"HGB (depth={best[0]}, trees={best[1]})", **metrics.predictive_scores(y_va, p_hgb)},
        {"approach": "Platform sub-grade (rank only)",
         "auc": roc_auc_score(y_va, valid["sub_grade_ord"]), "brier": np.nan, "ece": np.nan},
    ])
    pred.to_csv(OUT / "predictive_validation_2013.csv", index=False, float_format="%.5f")

    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=150)
    lim = 0.0
    for name, p in [("logit", p_logit), ("HGB", p_hgb)]:
        ece, bins = metrics.ece_and_bins(y_va, p)
        bins.assign(model=name).to_csv(OUT / f"reliability_bins_{name.lower()}_2013.csv",
                                       float_format="%.5f")
        ax.plot(bins["mean_predicted"], bins["observed_default_rate"], marker="o",
                label=f"{name} (ECE {ece:.3f})")
        lim = max(lim, bins["mean_predicted"].max(), bins["observed_default_rate"].max())
    ax.plot([0, lim * 1.1], [0, lim * 1.1], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.set_xlabel("Average predicted default probability")
    ax.set_ylabel("Observed charge-off rate")
    ax.set_title("Reliability, validation 2013", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "reliability_2013.png")
    plt.close(fig)

    # 6. Selection at 80% within grade -------------------------------------
    merge_fg, _ = selection.decide_fg_merge(inputs)
    strata = selection.strata_for(valid["grade"], merge_fg)
    key = selection.tie_break_key(valid["id"])
    sel = {
        "A. Fund-all": pd.Series(True, index=valid.index),
        "B. Platform rule": selection.platform_rule(valid, strata, key),
        "C. Logit": selection.model_rule(pd.Series(p_logit, index=valid.index), strata, key),
        "D. HGB": selection.model_rule(pd.Series(p_hgb, index=valid.index), strata, key),
    }
    rows, mix, by_grade = [], [], []
    for name, s in sel.items():
        chosen = valid[s]
        r = {"approach": name, **metrics.financial_outcomes(chosen, 0.01)}
        r["net_cash_gain_per_dollar_sf0"] = metrics.financial_outcomes(chosen, 0.0)["net_cash_gain_per_dollar"]
        r["mean_int_rate_pct"] = chosen["int_rate"].mean()
        rows.append(r)
        mix.append(chosen["sub_grade"].value_counts(normalize=True).rename(name))
        for g, part in chosen.groupby(strata[s]):
            fo = metrics.financial_outcomes(part)
            by_grade.append({"approach": name, "grade_stratum": g, "selected": len(part),
                             "charge_off_rate": fo["charge_off_rate"],
                             "net_cash_gain_per_dollar": fo["net_cash_gain_per_dollar"]})
    pd.DataFrame(rows).to_csv(OUT / "financial_validation_2013_q80.csv", index=False, float_format="%.5f")
    pd.concat(mix, axis=1).sort_index().fillna(0).to_csv(OUT / "subgrade_mix_selected_2013.csv",
                                                         float_format="%.5f")
    pd.DataFrame(by_grade).to_csv(OUT / "by_grade_validation_2013_q80.csv", index=False,
                                  float_format="%.5f")

    # How different are the two models' choices?
    both = (sel["C. Logit"] & sel["D. HGB"]).sum()
    overlap = {"selected_each": int(sel["D. HGB"].sum()), "selected_by_both": int(both),
               "share_same_loans": both / sel["D. HGB"].sum(),
               "rank_correlation_pd": pd.Series(p_logit).corr(pd.Series(p_hgb), method="spearman")}
    pd.Series(overlap).to_csv(OUT / "logit_vs_hgb_overlap_2013.csv", header=["value"])

    # Gate 4 (revised): is the HGB pipeline working correctly? ---------------
    gate4 = {
        "no_leakage_alarm": bool(tuning["validation_auc"].max() <= FULL_MODEL_ALARM),
        "grid_completed": len(tuning) == len(models.HGB_GRID),
        "early_stopping_off_all_fits": bool((tuning["trees_built"] == tuning["max_iter"]).all()),
        "probabilities_between_0_and_1": bool((p_hgb > 0).all() and (p_hgb < 1).all()),
    }

    import sklearn
    summary = {
        "step": "Roadmap step 7: add HistGB",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "scikit_learn": sklearn.__version__,
        "inputs": features.ALL_INPUTS,
        "hgb_fixed_settings": models.HGB_FIXED,
        "hgb_grid": models.HGB_GRID,
        "hgb_chosen": {"max_depth": best[0], "max_iter": best[1]},
        "logit_C_used": C, "logit_converged": conv,
        "gate4_checks": gate4, "gate4_automatic_checks_passed": all(gate4.values()),
        # "Sensible numbers" is a judgement for Lucas; for reference the plan
        # (B12) expected an application-time AUC of about 0.65-0.72.
        "chosen_hgb_validation_auc": float(tuning.loc[tuning["chosen"], "validation_auc"].iloc[0]),
        "plan_B12_expected_auc_range": [0.65, 0.72],
        "fg_merged": merge_fg, "selection_rate": selection.Q, "seed": config.SEED,
        "sealed_2014_2015_outcomes_opened": False,
    }
    (OUT / "step7_run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"Chosen HGB: depth={best[0]}, trees={best[1]}. Gate 4 automatic checks passed: {summary['gate4_automatic_checks_passed']}. "
        f"Done in {summary['runtime_seconds']} s.")


if __name__ == "__main__":
    main()

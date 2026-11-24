"""
step9_test_2014.py: ROADMAP STEP 9, the one-time 2014 test.

    python -m src.step9_test_2014 --dry-run     # rehearsal: fit 2010-2012, "test" on 2013
    python -m src.step9_test_2014 --unseal      # THE test: fit 2010-2013, test on 2014 (once)

Everything here follows the signed methodology freeze record
(docs/methodology_freeze_v1.md, decision_records/methodology_freeze_v1.json).
The script checks that record's fingerprint before doing anything, and takes
every model setting from it.

Plain-language summary of what happens (--unseal)
-------------------------------------------------
 1. Check the freeze record (fingerprint + status FROZEN).
 2. Retrain both models ONCE on all 2010-2013 loans with the frozen settings.
    All preparation (medians, scaling, category lists) is relearned on
    2010-2013 only.
 3. Unseal ONLY the 2014 outcomes (2015 stays sealed) and score 2014.
 4. Each approach funds 80% of loans within each grade (F+G merged).
 5. Report AUC / Brier / ECE, the money outcomes, and the paired bootstrap
    (1,000 draws, seed 42) with 95% intervals for the differences.
 6. Apply the four pre-registered success categories (F.11).
 7. Write a "test opened" log so the single run is on record.

--dry-run does the same on 2013 with models trained on 2010-2012, without
touching the sealed file. It exists only to prove the code works before the
real run; its outputs go to outputs/step9_dryrun_2013/.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, load_clean, metrics, models, selection
from .step6_baselines import fit_and_score, log

FREEZE_JSON = config.DECISION_DIR / "methodology_freeze_v1.json"
FREEZE_MD = config.REPO_ROOT / "docs" / "methodology_freeze_v1.md"
APPROACHES = ["A. Fund-all", "B. Platform rule", "C. Logit", "D. HGB"]
SHORT = {"A. Fund-all": "fundall", "B. Platform rule": "platform", "C. Logit": "logit", "D. HGB": "hgb"}


# ---------------------------------------------------------------------------
def load_freeze() -> dict:
    """Read the signed record and prove it is the exact text Lucas froze."""
    rec = json.loads(FREEZE_JSON.read_text())
    if rec["status"] != "FROZEN":
        raise PermissionError("Methodology not frozen")
    if hashlib.sha256(FREEZE_MD.read_bytes()).hexdigest() != rec["record_sha256"]:
        raise ValueError("Freeze record text changed after signing")
    # The code's constants must equal the frozen values.
    h = rec["final_models"]["hgb"]
    assert models.HGB_FIXED == {"learning_rate": h["learning_rate"], "min_samples_leaf": h["min_samples_leaf"],
                                "random_state": h["random_state"], "early_stopping": h["early_stopping"]}
    assert selection.Q == rec["selection_rate"] and config.SEED == rec["bootstrap"]["seed"]
    return rec


# ---------------------------------------------------------------------------
# Money outcomes with bootstrap weights (vectorised F.8 formulas)
# ---------------------------------------------------------------------------
def money_components(out: pd.DataFrame, service_fee: float) -> dict[str, np.ndarray]:
    P = out["funded_amnt"].to_numpy()
    PR, IN = out["total_rec_prncp"].to_numpy(), out["total_rec_int"].to_numpy()
    LF, RC = out["total_rec_late_fee"].to_numpy(), out["recoveries"].to_numpy()
    CF, y = out["collection_recovery_fee"].to_numpy(), out["target"].to_numpy()
    cash = PR + IN + LF + RC - CF - service_fee * (PR + IN + LF)
    loss = np.where(y == 1, np.clip(P - PR - (RC - CF), 0, None), 0.0)
    return {"P": P, "cash": cash, "loss": loss, "y": y.astype(float)}


def weighted_money(comp, sel: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
    """w can be one weight vector (n,) or a matrix of bootstrap weights (B, n)."""
    ws = w * sel
    inv = ws @ comp["P"]
    return {"net_cash_gain_per_dollar": (ws @ comp["cash"]) / inv - 1,
            "principal_loss_rate": (ws @ comp["loss"]) / inv,
            "charge_off_rate": (ws @ comp["y"]) / ws.sum(axis=-1)}


# ---------------------------------------------------------------------------
def run(fit_df, test_df, test_year: int, out_dir, rec: dict, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    test_df = test_df.sort_values("id", key=lambda s: s.astype("int64")).reset_index(drop=True)

    # 1. Final models ------------------------------------------------------
    X_fit, y_fit = models.model_frame(fit_df), fit_df["target"].to_numpy()
    X_te, y_te = models.model_frame(test_df), test_df["target"].to_numpy()
    models.check_training_columns(X_fit)
    models.check_category_limit(X_fit)
    lg, hg = rec["final_models"]["logit"], rec["final_models"]["hgb"]
    p_logit, conv = fit_and_score(models.logit_pipeline(lg["C"]), X_fit, y_fit, X_te)
    hgb_pipe = models.hgb_pipeline(hg["max_depth"], hg["max_iter"])
    p_hgb, _ = fit_and_score(hgb_pipe, X_fit, y_fit, X_te)
    hgb = hgb_pipe.named_steps["hgb"]
    assert hgb.n_iter_ == hg["max_iter"] and hgb.max_leaf_nodes == hg["max_leaf_nodes"] \
        and hgb.l2_regularization == hg["l2_regularization"] and hgb.max_bins == hg["max_bins"]
    log(f"[{label}] models fitted on {len(fit_df):,} loans; scoring {len(test_df):,} loans of {test_year}")

    # Save loan-level scores for later steps (git-ignored data folder).
    scores = pd.DataFrame({"id": test_df["id"], "pd_logit": p_logit, "pd_hgb": p_hgb})

    # 2. Selection at 80% within grade --------------------------------------
    merge_fg = rec["strata"] == ["A", "B", "C", "D", "E", "FG"]
    strata = selection.strata_for(test_df["grade"], merge_fg)
    key = selection.tie_break_key(test_df["id"])
    sel = {
        "A. Fund-all": pd.Series(True, index=test_df.index),
        "B. Platform rule": selection.platform_rule(test_df, strata, key),
        "C. Logit": selection.model_rule(pd.Series(p_logit, index=test_df.index), strata, key),
        "D. HGB": selection.model_rule(pd.Series(p_hgb, index=test_df.index), strata, key),
    }
    for a in APPROACHES:
        scores[f"selected_{SHORT[a]}"] = sel[a].to_numpy()

    # 3. Predictive metrics -------------------------------------------------
    pred = pd.DataFrame([
        {"model": "Logit", **metrics.predictive_scores(y_te, p_logit)},
        {"model": "HGB", **metrics.predictive_scores(y_te, p_hgb)},
    ])
    pred.to_csv(out_dir / f"predictive_{test_year}.csv", index=False, float_format="%.5f")
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=150)
    lim = 0.0
    for name, p in [("Logit", p_logit), ("HGB", p_hgb)]:
        ece, bins = metrics.ece_and_bins(y_te, p)
        bins.assign(model=name).to_csv(out_dir / f"reliability_bins_{name.lower()}_{test_year}.csv",
                                       float_format="%.5f")
        ax.plot(bins["mean_predicted"], bins["observed_default_rate"], marker="o",
                label=f"{name} (ECE {ece:.3f})")
        lim = max(lim, bins["mean_predicted"].max(), bins["observed_default_rate"].max())
    ax.plot([0, lim * 1.1], [0, lim * 1.1], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.set_xlabel("Average predicted default probability")
    ax.set_ylabel("Observed charge-off rate")
    ax.set_title(f"Reliability, {label} {test_year}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"reliability_{test_year}.png")
    plt.close(fig)

    # 4. Money outcomes (point estimates) -----------------------------------
    comp1, comp0 = money_components(test_df, 0.01), money_components(test_df, 0.0)
    ones = np.ones(len(test_df))
    rows = []
    for a in APPROACHES:
        s = sel[a].to_numpy()
        m = weighted_money(comp1, s, ones)
        rows.append({"approach": a, "loans": int(s.sum()),
                     "invested_usd": float(comp1["P"][s].sum()),
                     "charge_off_rate": float(m["charge_off_rate"]),
                     "principal_loss_rate": float(m["principal_loss_rate"]),
                     "net_cash_gain_per_dollar": float(m["net_cash_gain_per_dollar"]),
                     "net_cash_gain_per_dollar_sf0":
                         float(weighted_money(comp0, s, ones)["net_cash_gain_per_dollar"]),
                     "mean_int_rate_pct": float(test_df.loc[s, "int_rate"].mean())})
    money = pd.DataFrame(rows)
    money.to_csv(out_dir / f"financial_{test_year}_q80.csv", index=False, float_format="%.6f")

    # 5. Paired bootstrap ---------------------------------------------------
    B, n = rec["bootstrap"]["draws"], len(test_df)
    rng = np.random.default_rng(rec["bootstrap"]["seed"])
    W = np.empty((B, n))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    boot = {a: weighted_money(comp1, sel[a].to_numpy(), W) for a in APPROACHES}
    boot_sf0 = {a: weighted_money(comp0, sel[a].to_numpy(), W)["net_cash_gain_per_dollar"]
                for a in APPROACHES}
    auc_l = np.array([roc_auc_score(y_te, p_logit, sample_weight=W[b]) for b in range(B)])
    auc_h = np.array([roc_auc_score(y_te, p_hgb, sample_weight=W[b]) for b in range(B)])

    point = {r["approach"]: r for r in rows}
    contrasts = [("D. HGB", "C. Logit", "primary"), ("D. HGB", "B. Platform rule", "primary"),
                 ("C. Logit", "B. Platform rule", "secondary"), ("D. HGB", "A. Fund-all", "secondary"),
                 ("C. Logit", "A. Fund-all", "secondary"), ("B. Platform rule", "A. Fund-all", "secondary")]
    rows = []
    for a, b, role in contrasts:
        for metric in ["net_cash_gain_per_dollar", "charge_off_rate", "principal_loss_rate"]:
            d = boot[a][metric] - boot[b][metric]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"contrast": f"{a} minus {b}", "role": role if metric == "net_cash_gain_per_dollar"
                         else "secondary", "metric": metric,
                         "difference": point[a][metric] - point[b][metric], "ci95_low": lo, "ci95_high": hi})
        d = boot_sf0[a] - boot_sf0[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"contrast": f"{a} minus {b}", "role": "sensitivity (fee 0%)",
                     "metric": "net_cash_gain_per_dollar_sf0",
                     "difference": point[a]["net_cash_gain_per_dollar_sf0"] - point[b]["net_cash_gain_per_dollar_sf0"],
                     "ci95_low": lo, "ci95_high": hi})
    d = auc_h - auc_l
    lo, hi = np.percentile(d, [2.5, 97.5])
    auc_diff = pred.loc[1, "auc"] - pred.loc[0, "auc"]
    rows.append({"contrast": "D. HGB minus C. Logit", "role": "secondary", "metric": "auc",
                 "difference": auc_diff, "ci95_low": lo, "ci95_high": hi})
    inf = pd.DataFrame(rows)
    inf.to_csv(out_dir / f"bootstrap_contrasts_{test_year}.csv", index=False, float_format="%.6f")

    # 6. Pre-registered categories (F.11) ----------------------------------
    def ci(a, b, metric="net_cash_gain_per_dollar"):
        r = inf[(inf.contrast == f"{a} minus {b}") & (inf.metric == metric)].iloc[0]
        return r.ci95_low, r.ci95_high
    hl, hp, lp = ci("D. HGB", "C. Logit"), ci("D. HGB", "B. Platform rule"), ci("C. Logit", "B. Platform rule")
    auc_ci = ci("D. HGB", "C. Logit", "auc")
    above = lambda c: c[0] > 0
    includes0 = lambda c: c[0] <= 0 <= c[1]
    cats = {
        "AI adds economic value": above(hl) and above(hp),
        "AI adds predictive but not economic value": above(auc_ci) and (includes0(hl) or includes0(hp)),
        "Simpler model preferable": (not above(hl)) and includes0(auc_ci),
        "Platform pricing hard to beat": (not above(lp)) and (not above(hp)),
    }
    cats = {k: bool(v) for k, v in cats.items()}
    pd.Series(cats, name="applies").to_csv(out_dir / f"success_categories_{test_year}.csv")

    summary = {"label": label, "test_year": test_year, "fit_years": sorted(fit_df["vintage"].unique().tolist()),
               "fit_loans": len(fit_df), "test_loans": len(test_df), "logit_converged": conv,
               "fg_merged": merge_fg, "bootstrap_draws": B, "categories": cats,
               "runtime_seconds": round(time.time() - t0, 1)}
    return summary, scores


# ---------------------------------------------------------------------------
def main(mode: str):
    rec = load_freeze()
    inputs = load_clean.load_inputs()
    dev = inputs.merge(load_clean.load_dev_outcomes()[["id", "target"] + config.OUTCOME_COLUMNS],
                       on="id", how="inner", validate="one_to_one")

    if mode == "dry-run":
        fit_df = dev[dev["vintage"].isin([2010, 2011, 2012])]
        test_df = dev[dev["vintage"] == 2013]
        out_dir = config.REPO_ROOT / "outputs" / "step9_dryrun_2013"
        summary, _ = run(fit_df, test_df, 2013, out_dir, rec, "dry run")
        (out_dir / "dryrun_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log(f"Dry run done: {summary['categories']}")
        return

    # ---- THE ONE-TIME TEST ------------------------------------------------
    out_dir = config.REPO_ROOT / "outputs" / "step9"
    opened_log = out_dir / "test_opened_log.json"
    if opened_log.exists():
        raise PermissionError("2014 has already been opened once. A rerun is allowed only as a "
                              "logged bug fix (delete this log in the same commit that records the fix).")
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_df = dev[dev["vintage"].isin(rec["final_fit_years"])]
    opened_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sealed = load_clean.load_sealed_outcomes(2014, methodology_frozen_on=rec["date_local"])
    test_df = inputs[inputs["vintage"] == 2014].merge(
        sealed[["id", "target"] + config.OUTCOME_COLUMNS], on="id", how="inner", validate="one_to_one")
    assert len(test_df) == 162570
    summary, scores = run(fit_df, test_df, 2014, out_dir, rec, "test")
    scores.to_parquet(config.CLEAN_DIR / "step9_scores_selection_2014.parquet", index=False)

    import sklearn
    summary.update({
        "test_opened_utc": opened_at,
        "freeze_record_sha256": rec["record_sha256"],
        "code_commit": __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                    text=True, cwd=config.REPO_ROOT).stdout.strip(),
        "scikit_learn": sklearn.__version__,
        "year_2015_opened": False,
    })
    opened_log.write_text(json.dumps(summary, indent=2, default=str))
    log(f"2014 test done. Categories: {summary['categories']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--unseal", action="store_true")
    a = ap.parse_args()
    main("dry-run" if a.dry_run else "unseal")

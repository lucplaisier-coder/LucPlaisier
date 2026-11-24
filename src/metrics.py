"""
metrics.py: how we score predictions and selected portfolios.

Plain-language summary
----------------------
Prediction quality (F.9):
  * AUC   - if you pick one loan that defaulted and one that did not, how often
            does the model give the defaulted one the higher risk? 0.5 = coin
            flip, 1.0 = perfect.
  * Brier - average squared gap between the predicted probability and what
            happened (0 or 1). Lower is better.
  * ECE   - calibration: sort loans into 10 equal-size groups by predicted risk;
            in each group compare the average prediction with the actual default
            share; average the gaps (weighted by group size). Lower is better.

Money (F.8), for a set S of funded loans:
  * charge-off rate      = share of S that charged off
  * principal loss rate  = principal NOT recovered on charged-off loans / money invested
  * net cash gain per $  = (all cash received - collection fees - service fee) / invested - 1
P = funded_amnt, PR = principal received, IN = interest, LF = late fees,
RC = recoveries, CF = collection fee. Service fee SF = 1% of PR+IN+LF (an
assumption; we also show SF = 0). Not annualised, not an IRR.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

N_BINS = 10


def ece_and_bins(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple[float, pd.DataFrame]:
    """Expected calibration error with 10 equal-frequency bins (F.9)."""
    order = np.argsort(p, kind="mergesort")
    bins = np.empty(len(p), dtype=int)
    bins[order] = np.arange(len(p)) * n_bins // len(p)
    t = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_default_rate=("y", "mean"))
    ece = float((t["n"] * (t["mean_predicted"] - t["observed_default_rate"]).abs()).sum() / t["n"].sum())
    return ece, t


def predictive_scores(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    ece, _ = ece_and_bins(y, p)
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "ece": ece}


def financial_outcomes(out: pd.DataFrame, service_fee: float = 0.01) -> dict:
    """F.8 formulas on a set of loans (rows of the outcomes table)."""
    P, PR, IN = out["funded_amnt"], out["total_rec_prncp"], out["total_rec_int"]
    LF, RC, CF = out["total_rec_late_fee"], out["recoveries"], out["collection_recovery_fee"]
    co = out["target"] == 1
    invested = P.sum()
    loss = (P - PR - (RC - CF)).clip(lower=0)[co].sum()
    sf = service_fee * (PR + IN + LF)
    gain = (PR + IN + LF + RC - CF - sf).sum() / invested - 1
    return {
        "loans": len(out),
        "invested_usd": invested,
        "charge_off_rate": co.mean(),
        "principal_loss_rate": loss / invested,
        "net_cash_gain_per_dollar": gain,
    }

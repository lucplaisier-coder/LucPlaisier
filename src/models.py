"""
models.py: the logistic regression pipeline (control document F.6 / F.7 C).

Plain-language summary
----------------------
A scikit-learn "Pipeline" bundles the data preparation and the model, so that
everything learned from data (medians, means, standard deviations, the list of
categories, the model coefficients) is learned ONLY from the rows passed to
.fit(). We fit on 2010-2012 and only *apply* the result to 2013. That is how
we avoid leakage. The layout follows preparation rules G2-PREP-2026-11-24-v1
and the approved change-log rows:

  numbers (7)      -> fill gaps with the training median -> standardise (z-score)
  missing flags    -> a separate 0/1 column for every number that had gaps in
                      training (in practice: employment length). NOT scaled.
                      A gap that first appears later gets the median but no
                      new column (error_on_new=False).
  categories (2)   -> gaps become "__MISSING__" -> one 0/1 column per category
                      seen in training. A category never seen in training
                      (e.g. home ownership "ANY") becomes all zeros.
  model            -> logistic regression with an L2 penalty (strength 1/C),
                      max_iter = 1000.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import features

C_GRID = [0.01, 0.1, 1.0]      # F.7: chosen on 2013 by AUC
TIE_TOLERANCE = 0.002          # F.7: AUC within 0.002 -> the simpler (smaller C) wins
MISSING_TOKEN = "__MISSING__"


def model_frame(df: pd.DataFrame, numeric=None, nominal=None) -> pd.DataFrame:
    """The input columns only, in fixed order, with plain numpy types."""
    numeric = features.NUMERIC_INPUTS if numeric is None else numeric
    nominal = features.NOMINAL_INPUTS if nominal is None else nominal
    X = pd.DataFrame(index=df.index)
    for c in numeric:
        X[c] = df[c].astype("float64")
    for c in nominal:
        X[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return X


def check_training_columns(X_train: pd.DataFrame) -> None:
    """Preparation rule: an input that is completely empty in training -> stop."""
    empty = [c for c in X_train.columns if X_train[c].isna().all()]
    if empty:
        raise ValueError(f"Inputs completely empty in the training window: {empty}")


def logit_pipeline(C: float, numeric=None, nominal=None) -> Pipeline:
    numeric = features.NUMERIC_INPUTS if numeric is None else numeric
    nominal = features.NOMINAL_INPUTS if nominal is None else nominal
    parts = []
    if numeric:
        parts.append(("num", Pipeline([
            ("median", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
        parts.append(("missing_flag", MissingIndicator(features="missing-only",
                                                       error_on_new=False), numeric))
    if nominal:
        parts.append(("cat", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
        ]), nominal))
    prep = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)
    model = LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=1000)
    return Pipeline([("prep", prep), ("logit", model)])


def choose_c(auc_by_c: dict[float, float]) -> float:
    """F.7 tuning rule: best AUC; within 0.002 of the best, take the smallest C."""
    best = max(auc_by_c.values())
    return min(c for c, a in auc_by_c.items() if a >= best - TIE_TOLERANCE)


# ---------------------------------------------------------------------------
# Histogram gradient boosting (control document F.7 D, step 7)
# ---------------------------------------------------------------------------
# Plain language: gradient boosting builds many small decision trees one after
# the other; each new tree corrects the mistakes of the trees before it.
# "Histogram" means numbers are first grouped into at most 255 bins, which
# makes it fast. Settings fixed in the plan:
#   learning_rate = 0.05, min_samples_leaf = 200, random_state = 42,
#   max_depth in {3, 5} and max_iter in {200, 500}, chosen on 2013 by AUC,
#   early_stopping = False (approved change-log row: the library would
#   otherwise switch it on by itself above 10,000 rows).
# All other settings are the scikit-learn 1.9.1 defaults (e.g. no extra
# L2 penalty, at most 31 leaves per tree, 255 bins).
#
# Preparation (preparation rules, hgb_plan):
#   numbers     -> passed through unchanged; gaps stay NaN (the trees learn
#                  which side missing values go to). No filling, no scaling.
#   categories  -> each label gets an integer code learned from TRAINING rows
#                  only; a label never seen in training, or a gap, becomes NaN.
#                  The model is told explicitly that these two columns are
#                  categories (unordered), not numbers.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

HGB_FIXED = {"learning_rate": 0.05, "min_samples_leaf": 200, "random_state": 42,
             "early_stopping": False}
HGB_GRID = [(depth, iters) for depth in (3, 5) for iters in (200, 500)]
HGB_MAX_BINS = 255


def hgb_pipeline(max_depth: int, max_iter: int) -> Pipeline:
    numeric, nominal = features.NUMERIC_INPUTS, features.NOMINAL_INPUTS
    prep = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), nominal),
    ], remainder="drop", verbose_feature_names_out=False)
    categorical_mask = [False] * len(numeric) + [True] * len(nominal)
    model = HistGradientBoostingClassifier(max_depth=max_depth, max_iter=max_iter,
                                           categorical_features=categorical_mask,
                                           max_bins=HGB_MAX_BINS, **HGB_FIXED)
    return Pipeline([("prep", prep), ("hgb", model)])


def check_category_limit(X_train: pd.DataFrame) -> None:
    """Preparation rule: stop rather than merge levels if a category list is too long."""
    for c in features.NOMINAL_INPUTS:
        n = X_train[c].dropna().nunique()
        if n > HGB_MAX_BINS:
            raise ValueError(f"{c} has {n} categories, above the HGB limit of {HGB_MAX_BINS}")


def choose_hgb(auc_by_setting: dict[tuple[int, int], float]) -> tuple[int, int]:
    """F.7 tuning rule: best AUC; within 0.002 of the best, take the simpler
    setting (smaller depth first, then fewer iterations)."""
    best = max(auc_by_setting.values())
    near = [s for s, a in auc_by_setting.items() if a >= best - TIE_TOLERANCE]
    return min(near)   # tuples sort by depth, then by iterations

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

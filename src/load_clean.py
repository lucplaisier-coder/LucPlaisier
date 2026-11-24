"""
load_clean.py: how later steps should open the step-5 files.

Plain-language summary
----------------------
Later steps (modelling, evaluation) must NOT read data/clean/ directly.
They call these functions, which
  * check each file's fingerprint against outputs/step5/step5_run_summary.json
    (so nobody can quietly edit the clean data), and
  * refuse to open the SEALED 2014-2015 outcomes unless the caller states,
    in code, that the step 8 methodology freeze has happened.
"""

import hashlib
import json

import pandas as pd

from . import config


def _checked_read(name: str) -> pd.DataFrame:
    summary = json.loads((config.STEP5_OUT / "step5_run_summary.json").read_text())
    expected = summary["clean_files"][name]["sha256"]
    path = config.CLEAN_DIR / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{name} changed since step 5 was run. Rerun step 5.")
    return pd.read_parquet(path)


def load_inputs() -> pd.DataFrame:
    """Nine proposed inputs + id, vintage, split, grade, sub_grade. All years."""
    return _checked_read("step5_inputs.parquet")


def load_audit() -> pd.DataFrame:
    """Lineage columns (stored rate, raw labels, verification_status). Never model inputs."""
    return _checked_read("step5_audit.parquet")


def load_dev_outcomes() -> pd.DataFrame:
    """Target and cash fields for 2010-2013 (train + validation)."""
    return _checked_read("step5_outcomes_2010_2013.parquet")


def load_sealed_outcomes(methodology_frozen_on: str | None = None) -> pd.DataFrame:
    """
    Target and cash fields for 2014-2015. Pass the date of the step 8 freeze,
    e.g. load_sealed_outcomes(methodology_frozen_on="2026-11-26").
    Without it, this function refuses.
    """
    if not methodology_frozen_on:
        raise PermissionError(
            "2014-2015 outcomes are sealed until the step 8 methodology freeze."
        )
    return _checked_read("SEALED_step5_outcomes_2014_2015.parquet")

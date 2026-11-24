"""
config.py: every fixed setting for the project, in one place.

Plain-language summary
----------------------
Nothing in this file is "learned" from the data. These are rules that were
written down BEFORE we looked at results (project control document, Part 2,
plus the approved change-log rows of 24 Nov 2026). Keeping them in one file
means a marker can check every rule without hunting through the code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Folders
# ---------------------------------------------------------------------------
# REPO_ROOT is the top folder of this repository (one level above src/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# The raw Lending Club file goes here. This folder is in .gitignore, so the
# raw data is never uploaded to GitHub (control document F.2).
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_FILENAME = "accepted_2007_to_2018Q4.csv.gz"
RAW_PATH = RAW_DIR / RAW_FILENAME

# The SHA-256 "fingerprint" of the exact raw file used in the project.
# If even one byte of the file differs, the fingerprint changes and the
# pipeline stops. This proves everyone runs on the same file.
RAW_SHA256 = "55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d"

# Cleaned, loan-level data is written here (also never uploaded to GitHub).
CLEAN_DIR = REPO_ROOT / "data" / "clean"

# Small summary tables (counts, shares, no loan-level rows) go here.
# These ARE committed, so a marker can see them without rerunning.
STEP5_OUT = REPO_ROOT / "outputs" / "step5"

# The step-4 decision records approved by Lucas (copied unchanged, with
# their SHA-256 manifest, from step4_closure/decision_records/).
DECISION_DIR = REPO_ROOT / "decision_records"

# ---------------------------------------------------------------------------
# 2. Which loans are in the study (control document F.3)
# ---------------------------------------------------------------------------
TERM_KEEP = "36 months"           # only 3-year loans
YEARS = [2010, 2011, 2012, 2013, 2014, 2015]
POLICY_CODE_KEEP = 1              # loans that met Lending Club's public credit policy
APPLICATION_TYPE_KEEP = "Individual"
EXCLUDED_STATUS_PREFIX = "Does not meet the credit policy"

# Target: what we want to predict. Only these two "final" statuses count.
TARGET_MAP = {"Fully Paid": 0, "Charged Off": 1}

# Statuses that are not final yet. These loans are left out and counted.
UNRESOLVED_STATUSES = [
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
]

# ---------------------------------------------------------------------------
# 3. Time design (control document F.4)
# ---------------------------------------------------------------------------
SPLIT_BY_YEAR = {
    2010: "train",
    2011: "train",
    2012: "train",
    2013: "validation",
    2014: "test",
    2015: "robustness",
}

# The outcomes of these years stay SEALED until the step 8 methodology
# freeze. The code writes them to a separate file and records its
# fingerprint, but never computes any statistic on them.
SEALED_YEARS = [2014, 2015]

# Resolution rule: a year is usable if at least 99% of its loans have a final
# status. (2014 has a fallback threshold of 98%, see F.4.)
RESOLUTION_THRESHOLD = 0.99
TEST_FALLBACK_THRESHOLD = 0.98

# "Default" status rule: if it is more than 0.5% of a year, report a
# sensitivity treating it as charged off.
DEFAULT_STATUS_SENSITIVITY_SHARE = 0.005

# Cross-check: the number of resolved loans per year found by the earlier,
# independent Gate 1 audit. If this run finds different numbers, something
# changed (file, filter or code) and the pipeline stops. These are sample
# SIZES only, not outcome statistics.
EXPECTED_RESOLVED_COUNTS = {
    2010: 8466,
    2011: 14101,
    2012: 43470,
    2013: 100422,
    2014: 162570,
    2015: 282787,
}

# ---------------------------------------------------------------------------
# 4. The nine proposed inputs (F.5 + approved change log). NOT FROZEN.
# ---------------------------------------------------------------------------
# Lucas decides the feature freeze (Gate 2 / roadmap). Until then this is the
# *proposed* list from preparation rules G2-PREP-2026-11-24-v1.
# verification_status was removed on 24 Nov (decision G2-VERIFY-EXCLUDE).
PROPOSED_NUMERIC_INPUTS = [
    "fico_mid",        # P1 average of the two FICO bounds at origination
    "dti",             # P2 debt-to-income, in percentage points
    "log_annual_inc",  # P3 natural log of annual income
    "emp_length_ord",  # P4 employment length 0..10 (NaN = not answered)
    "loan_amnt",       # P7 requested/listed loan amount in USD
    "sub_grade_ord",   # P8 platform sub-grade A1=1 .. G5=35
    "int_rate",        # P9 interest rate in % (with approved 148-case fix)
]
PROPOSED_NOMINAL_INPUTS = [
    "home_ownership",  # P5 RENT / OWN / MORTGAGE / OTHER / NONE / ANY
    "purpose",         # P6 stated purpose of the loan
]
FEATURES_FROZEN = False   # Only Lucas can change this, at the freeze step.

# Employment length labels (preparation rules, employment_mapping).
EMPLOYMENT_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

# Columns read from the raw file. Everything else is never loaded.
RAW_COLUMNS = [
    # identity and filters
    "id", "term", "issue_d", "policy_code", "application_type", "loan_status",
    # platform pricing / strata
    "grade", "sub_grade", "int_rate",
    # borrower inputs
    "fico_range_low", "fico_range_high", "dti", "annual_inc", "emp_length",
    "home_ownership", "purpose", "loan_amnt",
    # kept for audit only (NOT an input, decision G2-VERIFY-EXCLUDE)
    "verification_status",
    # cash outcome fields (F.8). Used only AFTER a loan has finished.
    "funded_amnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "total_pymnt",
]

OUTCOME_COLUMNS = [
    "funded_amnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "total_pymnt",
]

SEED = 42  # used later for the platform-rule tie-break and the bootstrap

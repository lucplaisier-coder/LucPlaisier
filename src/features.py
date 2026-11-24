"""
features.py: the FROZEN list of model inputs (control document F.5, Gate 2).

Plain-language summary
----------------------
Lucas froze these nine inputs on 24 Nov 2026, before any model was fitted
(record: decision_records/gate2_feature_freeze.json). From now on every model
uses exactly this list. Changing it would need a new, logged decision, and
after the step 8 freeze only bug fixes are allowed.

Not inputs, on purpose:
  grade               -> selection stratum only (we pick 80% within each grade)
  verification_status -> removed 24 Nov (timing precaution, G2-VERIFY-EXCLUDE)
  term, installment   -> constant / determined by amount, rate and term
"""

FROZEN_ON = "2026-11-24"
FREEZE_RECORD = "decision_records/gate2_feature_freeze.json"

NUMERIC_INPUTS = [
    "fico_mid",        # FICO midpoint at origination
    "dti",             # debt-to-income, percentage points
    "log_annual_inc",  # natural log of annual income
    "emp_length_ord",  # employment length 0..10, NaN = not answered
    "loan_amnt",       # listed loan amount, USD
    "sub_grade_ord",   # platform sub-grade A1=1 .. G5=35
    "int_rate",        # interest rate %, with the approved 148-loan fix
]

NOMINAL_INPUTS = [
    "home_ownership",
    "purpose",
]

ALL_INPUTS = NUMERIC_INPUTS + NOMINAL_INPUTS

# The two platform-pricing inputs (used together as one block in step 12).
PLATFORM_PRICING_BLOCK = ["sub_grade_ord", "int_rate"]

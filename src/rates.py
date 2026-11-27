"""
rates.py: apply the approved interest-rate fix for 148 loans (G2-RATE-2026-11-24-v1).

Plain-language summary
----------------------
For 148 loans the raw file stores a rate that differs from the listing rate in
Lending Club's SEC posting supplements (146 of them exactly 6.00%). The stored
value is strongly linked to the outcome, so using it risks leaking information
that arose after issue; a 6% cap is consistent with a known servicing mechanism,
but the cause and timing are not proven loan by loan. Lucas approved replacing
them with the sourced listing rate.

What this module does:
  1. Checks that the approved decision files are byte-for-byte unchanged
     (their SHA-256 manifest).
  2. Re-runs the two automatic "warning screens" from the decision on the
     stored rates:
       a. rate exactly 6.00%
       b. "sparse off-grid": within one issue month x sub-grade with 20+
          loans, a rate held by at most 2 loans that is more than 3
          percentage points away from the most common rate(s).
     Every flagged loan MUST already be in the approved ledger. A new flag
     means "stop and review the evidence", never an automatic fix.
  3. Runs the approved overlay code (decision_records/approved_rate_overlay.py,
     used unchanged) to swap in the listing rate for exactly those 148 loans.
  4. Re-runs the screens on the corrected rates (expected: nothing left).
  5. Keeps the original stored rate next to the corrected one in a separate
     AUDIT table. The audit columns are never model inputs.
The third route in the decision (comparison with a saved web archive) cannot
be re-run here because that archive is not part of the repository; its 9
cases are all in the ledger.
"""

import hashlib
import importlib.util

import numpy as np
import pandas as pd

from . import config


def verify_decision_manifest() -> None:
    """Stop if any approved decision file was edited after approval."""
    manifest = (config.DECISION_DIR / "MANIFEST.sha256").read_text().splitlines()
    for line in manifest:
        if not line.strip():
            continue
        expected, name = line.split()
        actual = hashlib.sha256((config.DECISION_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Approved decision file changed: {name}")


def _load_overlay_module():
    """Import decision_records/approved_rate_overlay.py exactly as approved."""
    path = config.DECISION_DIR / "approved_rate_overlay.py"
    spec = importlib.util.spec_from_file_location("approved_rate_overlay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def warning_screens(pool: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    """
    Return one row per flagged loan with the reason(s).
    Works on the candidate pool (the decision's scope). Uses only issue
    month, sub-grade and rate, so no outcome information is involved.
    """
    rate = pool[rate_col].round(2)
    flags = pd.DataFrame({"id": pool["id"].values, "stored_6": (rate == 6.00).values})

    # --- sparse off-grid screen -------------------------------------------
    cell = pool[["issue_d", "sub_grade"]].astype(str).agg("|".join, axis=1)
    work = pd.DataFrame({"cell": cell.values, "rate": rate.values, "id": pool["id"].values})
    sparse_ids = set()
    for _, g in work.groupby("cell"):
        if len(g) < 20:
            continue
        counts = g["rate"].value_counts()
        non6 = counts[counts.index != 6.00]
        if non6.empty:
            continue
        modal = non6[non6 == non6.max()].index.to_numpy()  # ties kept
        for r, n in counts.items():
            if n <= 2 and np.min(np.abs(modal - r)) > 3.0 + 1e-9:
                sparse_ids.update(g.loc[g["rate"] == r, "id"])
    flags["sparse_off_grid"] = flags["id"].isin(sparse_ids)
    flagged = flags[flags["stored_6"] | flags["sparse_off_grid"]].copy()
    return flagged.reset_index(drop=True)


def apply_approved_rate_fix(pool: pd.DataFrame):
    """
    Returns
      pool  : same rows, with int_rate (numeric %) replaced for the 148 loans
              and int_rate_stored keeping the raw value
      audit : one row per corrected loan (id, stored, listing rate, source)
      info  : counts for the report
      after : any loans still flagged after the fix (expected: none)
    """
    verify_decision_manifest()
    overlay = _load_overlay_module()
    ledger = overlay.load_approved_ledger(config.DECISION_DIR)  # checks hash, 148 rows

    from .sample import strict_number
    pool = pool.copy()
    pool["int_rate_stored"] = strict_number(pool["int_rate"], "int_rate")
    if pool["int_rate_stored"].isna().any():
        raise ValueError("Missing interest rate in candidate pool: stop for review (P9)")

    # Step 2: screens on stored rates -> every flag must be in the ledger
    before = warning_screens(pool, "int_rate_stored")
    new_flags = sorted(set(before["id"]) - set(ledger))
    if new_flags:
        raise ValueError(f"New rate flags not in the approved ledger: {new_flags[:10]}")

    # Step 3: the approved overlay, unchanged
    frame = pool[["id", "int_rate_stored"]].rename(columns={"int_rate_stored": "int_rate"})
    corrected, audit_rows = overlay.apply_listing_rates(frame, ledger)
    overlay.verify_coverage([a["id"] for a in audit_rows], ledger)
    pool["int_rate"] = corrected["int_rate"].astype("float64").values
    pool["rate_overridden"] = pool["id"].isin(ledger)

    # Step 4: screens again on corrected rates
    after = warning_screens(pool, "int_rate")

    audit = pd.DataFrame(audit_rows)
    audit = audit.merge(pool[["id", "vintage", "issue_d", "sub_grade"]], on="id", how="left")
    info = {
        "ledger_cases": len(ledger),
        "cases_found_and_corrected": len(audit_rows),
        "screen_before_stored_6": int(before["stored_6"].sum()),
        "screen_before_sparse_off_grid": int(before["sparse_off_grid"].sum()),
        "screen_before_union": len(before),
        "screen_before_flags_outside_ledger": 0,
        "screen_after_union": len(after),
        "corrected_by_vintage": audit["vintage"].value_counts().sort_index().to_dict(),
    }
    return pool, audit, info, after

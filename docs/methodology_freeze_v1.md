# Methodology freeze record v1 (roadmap step 8): FROZEN

**Status: FROZEN by Lucas on 24 Nov 2026 (2026-11-24T21:05:35Z UTC), exactly as drafted in commit `f556c98`,
including the proposed values G1-G4.** Record: `decision_records/methodology_freeze_v1.json`. At the
moment of freezing, the 2014 test and 2015 robustness outcomes were still sealed and unopened.

Prepared: 24 Nov 2026. Basis: control document v1.0 Part 2 with its change log
(`docs/project_control_document.md`) and the sign-offs in `decision_records/`.

---

## What "freeze" means, in plain words

When you say "freeze", everything in this record is locked **before anyone sees a 2014 or 2015
outcome**. Then the test runs **once**. After that:

* **No setting in this record may change.** That covers models, inputs, the selection rule, metrics,
  the bootstrap and the success categories. Not even if the results look disappointing.
* **The test is not rerun to "try again".** The only allowed change is a **bug fix**: the code did not do
  what this record says. Each fix is logged with its date and reason and disclosed in the manuscript annex
  (F.15).
* **Whatever the result, it is reported.** That includes a null result such as "HGB adds no value"
  (F.11 categories, set before the test).

This is the single most important protection against the rubric's main worry: choosing methods after
seeing the answer.

---

## A. Data and sample (locked)

| Item | Frozen value |
|---|---|
| Raw file | `accepted_2007_to_2018Q4.csv.gz`, SHA-256 `55c16f75…bd62a2d` |
| Sample | 36-month, 2010-2015, policy_code 1, Individual, credit-policy statuses excluded; final statuses only (step 5) |
| Target | Charged Off = 1, Fully Paid = 0 |
| Rate fix | 148 loans, listing rate (G2-RATE-2026-11-24-v1) |
| Clean files | as fingerprinted in `outputs/step5/step5_run_summary.json` |
| Years | train 2010-2012, validation 2013, **test 2014 (162,570 loans)**, **robustness 2015 (282,787 loans, 99.95% resolved, passes the 99% rule)** |

## B. Inputs (locked at Gate 2)

Nine inputs, `src/features.py`: FICO midpoint, debt-to-income, log annual income, employment length,
home ownership, purpose, loan amount, sub-grade (1-35), interest rate. Not inputs: grade (stratum only),
verification status, term, installment. Preparation rules G2-PREP-2026-11-24-v1.

## C. Final models (locked)

Both models are **refitted once on 2010-2013** (166,459 loans) with the settings below. All preparation
is relearned on 2010-2013 only: medians, means, standard deviations, the category lists and the
missing-flag columns. 2014 and 2015 are then only *scored*. Nothing is learned from them.

| Model | Frozen settings |
|---|---|
| **Logit** | L2 penalty, **C = 0.01** (chosen on 2013, step 6), lbfgs, max_iter 1000. Median fill + standardise the 7 numbers; separate unscaled 0/1 missing flag for each number with gaps in 2010-2013; one-hot categories, unseen = all zeros |
| **HGB** | **max_depth 3, max_iter 200** (chosen on 2013, step 7), learning_rate 0.05, min_samples_leaf 200, random_state 42, **early_stopping False**, other settings at scikit-learn 1.9.1 defaults (31 leaves, no L2, 255 bins). Native missing values; categories coded from 2010-2013, unseen = missing |
| Software | Python 3.11+, pandas 2.3.3, numpy 2.3.5, pyarrow 22.0.0, scikit-learn 1.9.1, matplotlib 3.10.8 |

## D. Selection rule (locked)

| Item | Frozen value |
|---|---|
| Rate | **q = 80% within each grade** |
| Strata | A, B, C, D, E and **F+G merged** (G has 179 loans in 2014, under the 200 rule) |
| Loans kept per stratum | floor(0.8 × n + 0.5) |
| A. Fund-all | every loan |
| B. Platform rule | sub-grade (riskiest skipped first), then interest rate (higher skipped first), then seeded random |
| C. Logit / D. HGB | predicted default probability (highest skipped first), then seeded random |
| Tie-break draw | seed 42, one random number per loan in loan-id order, drawn separately within each year scored |

## E. Outcomes and metrics (locked)

* **Money (F.8):** charge-off rate; principal loss rate; **net cash gain per dollar** with service
  fee 1% of principal + interest + late fees (assumption), plus a sensitivity with fee 0%. Not
  annualised, never called IRR. Mean interest rate and sub-grade mix are descriptive only.
* **Prediction (F.9):** AUC, Brier score, ECE with 10 equal-frequency bins, and a reliability plot. On
  2014 and 2015 from the final models. On 2013 from the development models fitted on 2010-2012, labelled
  "used for tuning".

## F. Test procedure and inference (locked)

1. Open the sealed file **once**, with this record's freeze date. Score 2014, then 2015.
2. **Primary metric:** net cash gain per dollar at q = 80%.
   **Primary contrasts:** HGB − Logit and HGB − Platform. Everything else is secondary.
3. **Paired bootstrap:** resample the 2014 test loans with replacement **1,000 times, seed 42**. Each
   approach's funded/not-funded flags stay fixed. Report 95% percentile intervals for the differences in
   net cash gain, charge-off rate and principal loss, and for the HGB − Logit AUC difference. Stated
   limitation: resampling single loans ignores shocks shared by loans issued in the same month.
4. **Success categories (F.11), applied to 2014 only.** Report every category that applies:
   * *AI adds economic value*: both primary 95% intervals are above 0.
   * *AI adds predictive but not economic value*: the HGB − Logit AUC interval is above 0, and at least one
     primary interval includes 0.
   * *Simpler model preferable*: the HGB − Logit net-cash interval includes 0 or is below it, **and** the
     AUC interval includes 0.
   * *Platform pricing hard to beat*: neither the logit nor HGB has a net-cash interval above 0 against
     the platform rule.
5. **Robustness (F.12).** 2015 is reported as consistent or inconsistent with the 2014 category, and can
   never redefine it. Grade groups A-B, C and D-G (D, E, F+G) get net cash gain and charge-off rate per
   approach.
6. **Interpretability (F.12).** Logit odds ratios per one standard deviation for the borrower numbers,
   read as "holding platform pricing fixed". HGB permutation importance on 2014 with 10 repeats, for each
   input and for the sub-grade + interest-rate block permuted together.
7. **Six core outputs plus the annex**, as F.13. Selection-rate figure for q from 50% to 100%, display only.

## G. Four details the plan leaves open: values confirmed by the freeze

These are not decided anywhere yet. I propose the simplest standard choice for each, set now so that
nothing is chosen after seeing 2014.

| # | Open detail | Proposed frozen value | Why |
|---|---|---|---|
| G1 | Points on the selection-rate figure (F.13 #4) | q = 50%, 55%, …, 100% (11 points) | Even steps; display only, never used to pick a rate |
| G2 | What permutation importance measures | Drop in 2014 **AUC**, 10 repeats, seed 42 | Same ranking metric used for tuning |
| G3 | Odds ratios for home ownership and purpose | Not interpreted. Odds ratios only for the 5 borrower numbers (FICO, DTI, log income, employment length, loan amount); category coefficients shown in the annex without interpretation | With L2 and no reference category, single category coefficients have no clean "versus" meaning |
| G4 | How 2015 is judged "consistent" | Run the same 1,000-draw paired bootstrap (seed 42) on 2015 and apply the same four category rules; consistent = the same categories hold | Uses the 2014 rules unchanged instead of a new rule |

## H. Disclosures carried into the manuscript

* An earlier external review (before 24 Nov) displayed a few loan statuses. This is disclosed in the
  step-4 decision records. No 2014/2015 outcome statistics were computed in steps 5 to 8.
* 2014-2015 input drift (DTI above the training maximum, rising missing employment length, rates below
  the training minimum in 2015) is disclosed without caps (change log, step 5 report).
* Validation 2013 results (steps 6-7) are labelled "used for tuning".

## I. Signature

| | |
|---|---|
| Frozen by | Lucas |
| Date | 24 Nov 2026, 2026-11-24T21:05:35Z UTC |
| Code version | draft commit `f556c989645670957091967e90ffa69e37652a62`; the freeze commit holds this signed record |
| Authorization text | "freeze" |

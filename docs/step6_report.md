# Step 6 report: logistic regression + platform benchmark (for Lucas's sign-off)

**Status: signed off by Lucas on 24 Nov 2026, including both details in section 5** (`decision_records/step6_signoff.json`).

Date: 24 Nov 2026. Code: `src/step6_baselines.py` (runs in under 10 seconds after step 5).
All numbers come from `outputs/step6/`. **These are 2013 validation results, "used for tuning".
They are not the test result and have no confidence intervals yet.** 2014 and 2015 stay sealed.

## In one paragraph

The nine inputs are frozen (Gate 2). The logistic regression was trained on 2010 to 2012 only
(66,037 loans) and checked on 2013 (100,422 loans). No leakage alarm went off: the strongest single
input reaches an AUC of 0.65, far below the 0.80 alarm. The plan's tuning rule picked **C = 0.01**
(validation AUC **0.665**). On 2013, when each approach funds 80% of the loans within each grade, the
**logit earns 10.14 cents per dollar invested, the platform rule 9.72 cents and fund-all 9.85 cents**.

## 1. Gate 2: the inputs are frozen

`src/features.py` now holds the nine inputs, and `decision_records/gate2_feature_freeze.json` records your
"yes". I also added one row to the change log in the control document, in both the repository and the
project files copy.

## 2. Leakage alarm (F.6)

For each input on its own, I trained a one-input logit on 2010 to 2012 and scored 2013:

| Input | AUC 2013 |
|---|---:|
| interest rate | 0.649 |
| sub-grade | 0.649 |
| FICO | 0.593 |
| log income | 0.593 |
| debt-to-income | 0.564 |
| home ownership | 0.552 |
| purpose | 0.534 |
| loan amount | 0.527 |
| employment length | 0.526 |

Nothing is close to 0.80, and the full model (0.665) is far below the 0.85 alarm. That is the normal
range for credit data (control document B12 expected 0.65 to 0.72), so there is no sign that a future
fact slipped in.

## 3. The logistic regression (F.6, F.7 C)

* **Preparation:** everything is learned from 2010 to 2012 only. Numbers are filled with the training median
  and standardised. There is a separate 0/1 flag for missing employment length, which is not
  standardised. Categories get one 0/1 column each, and a category never seen in training becomes
  all zeros.
* **Tuning:**

| C | AUC 2013 | Brier | ECE |
|---:|---:|---:|---:|
| **0.01 (chosen)** | **0.6652** | 0.1046 | 0.0173 |
| 0.1 | 0.6644 | 0.1048 | 0.0175 |
| 1 | 0.6642 | 0.1048 | 0.0176 |

  The three are almost identical. C = 0.01 is both the best and the simplest. All fits converged.
* **Platform sub-grade on its own** ranks 2013 loans with an AUC of 0.649. So the logit adds about
  0.016 AUC on validation. Whether that difference is real is decided only on the test, with a bootstrap.
* **Calibration** (`reliability_logit_2013.png`): the logit predicts slightly too much risk in 2013.
  It predicts about 14% on average, against 12.3% actual. This fits the plan's note that calibration
  shifts with each year's base default rate. Selection uses only the ranking, so this does not affect
  which loans are picked.

## 4. Money at 80% within each grade, validation 2013 (F.8)

| Approach | Loans | Charge-off rate | Principal loss rate | Net cash gain per $ (fee 1%) | (fee 0%) | Mean rate |
|---|---:|---:|---:|---:|---:|---:|
| A. Fund-all | 100,422 | 12.33% | 6.00% | 9.85 c | 10.95 c | 13.37% |
| B. Platform rule | 80,337 | 11.82% | 5.70% | 9.72 c | 10.83 c | 13.00% |
| C. Logit | 80,337 | 11.14% | 5.52% | **10.14 c** | 11.25 c | 13.20% |

In plain words:
* **The platform rule avoids some charge-offs, but it earns less than funding everything.** Inside a grade
  it skips the highest-rate loans, and on average their extra interest paid for their extra defaults.
* **The logit avoids more charge-offs than the platform rule and still keeps more of the interest.** It
  ends about 0.4 cents per dollar ahead of the platform rule and about 0.3 cents ahead of fund-all.
* Per grade (`by_grade_validation_2013_q80.csv`), the logit beats the platform rule on charge-off rate in
  grades A to E. In the small merged F+G group (498 loans) it does not.

## 5. Two small implementation details you should know (please confirm)

1. **Grades F and G are merged into one selection group.** The plan's rule says to merge them if either has
   fewer than 200 loans in the test year. G has 179 loans in 2014. That check uses only loan counts, not
   outcomes.
2. **How many loans are kept in a grade:** 80% of the count, rounded to the nearest whole loan (halves
   round up). The plan says "80%" but not how to round, so I fixed this before any test.
   The same seeded random draw (seed 42) breaks exact ties for the logit as well as for the platform rule.

## 6. What was not done

No HistGB (that is step 7). No refit on 2010 to 2013, no 2014 or 2015 numbers, and no bootstrap: those
come after the step 8 freeze. The coefficient table (`logit_coefficients_dev_fit_2010_2012.csv`) is only
for transparency. The reported odds ratios will come from the final fit in step 12.

## What I need from you

**Sign off step 6**, including the two details in section 5, or tell me what to change. Then step 7 (add
HistGB) can start. Its settings are already fixed in the plan, with early stopping off.

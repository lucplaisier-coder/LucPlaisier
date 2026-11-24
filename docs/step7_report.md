# Step 7 report: adding HistGB (for Lucas's sign-off)

**Status: signed off by Lucas on 24 Nov 2026, including the defaults in section 5** (`decision_records/step7_signoff.json`).

Date: 24 Nov 2026. Code: `src/step7_histgb.py` (runs in about 15 seconds after step 5).
All numbers come from `outputs/step7/`. **These are 2013 validation results, "used for tuning".
They are not the test result and have no confidence intervals.** 2014 and 2015 stay sealed.

## In one paragraph

HistGB is built exactly as fixed in your plan, with early stopping off, and it works correctly, so
Gate 4 passes. Your tuning rule picked the **simplest setting: trees of depth 3, 200 trees**. On 2013 it
performs **almost exactly like the logit**: AUC 0.665 for both, and 10.15 cents versus 10.14 cents
net cash gain per dollar. The two models pick 95% of the same loans. Whether HGB adds anything is
decided only once, on the 2014 test.

## 1. How HGB is set up (F.6, F.7 D)

* **Fixed settings:** learning rate 0.05, at least 200 loans per leaf, seed 42, **early stopping off**.
  The code also checks that every fit really built all its trees, which proves early stopping did not
  switch itself on.
* **Learned only from 2010 to 2012:** numbers go in unchanged, and blanks stay blank because the trees
  handle them themselves. Home ownership and purpose are told to the model as categories. Their list of
  labels comes from the training years only, so a label never seen there (like "ANY") is treated as blank.
* **Everything not named in the plan uses the scikit-learn 1.9.1 defaults.** That means at most 31
  leaves per tree, no extra penalty and 255 bins. Please confirm that is what you intended (see section 5).

## 2. Tuning on 2013 (your rule: best AUC; within 0.002, the simpler one wins)

| Depth | Trees | AUC 2013 | Brier | ECE |
|---:|---:|---:|---:|---:|
| **3** | **200** | **0.6649** | **0.1043** | **0.0136** |
| 3 | 500 | 0.6636 | 0.1044 | 0.0125 |
| 5 | 200 | 0.6632 | 0.1044 | 0.0126 |
| 5 | 500 | 0.6586 | 0.1048 | 0.0133 |

The smallest model is also the best one. Bigger and longer models do slightly *worse* on 2013, which is
a typical sign of fitting noise in the training years. So the choice did not need the tie rule.

## 3. Gate 4 (revised): does HGB work correctly?

| Check | Result |
|---|---|
| Leakage alarm (AUC above 0.85) | No. The highest is 0.665 |
| Grid completed (all 4 settings) | Yes |
| Early stopping off in every fit | Yes |
| Predicted probabilities valid (between 0 and 1) | Yes |
| Numbers sensible | AUC 0.665 is inside the 0.65 to 0.72 range the plan expected (B12) |

As your plan requires, this gate does **not** ask whether HGB is better. It only asks whether HGB works.

## 4. All four approaches on 2013, 80% within each grade

| Approach | Charge-off rate | Principal loss rate | Net cash gain per $ (fee 1%) | (fee 0%) | AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. Fund-all | 12.33% | 6.00% | 9.85 c | 10.95 c | | | |
| B. Platform rule | 11.82% | 5.70% | 9.72 c | 10.83 c | 0.649 (rank) | | |
| C. Logit (C = 0.01) | 11.14% | 5.52% | 10.14 c | 11.25 c | 0.6652 | 0.1046 | 0.0173 |
| D. HGB (depth 3, 200 trees) | 11.11% | 5.46% | **10.15 c** | 11.26 c | 0.6649 | 0.1043 | 0.0136 |

In plain words:
* **HGB and the logit are practically tied on 2013.** HGB is a hair better on charge-offs, losses,
  Brier score and calibration (`reliability_2013.png`). The logit is a hair better on AUC. Every gap is
  tiny.
* **They agree on most loans.** 76,065 of the 80,337 loans each one funds are the same (95%), and their
  risk rankings have a correlation of 0.97.
* **Both beat the platform rule by about 0.4 cents per dollar on 2013.** The test will show whether
  that holds on new data and is more than noise.
* The logit numbers are identical to step 6 (same code, same C), which is a good consistency check.

This is an honest early signal for the manuscript. On these nine inputs, the extra flexibility of
boosting finds little that the logit misses. The pre-registered categories (F.11) are applied only to 2014.

## 5. One thing to confirm

The plan fixes learning rate, leaf size, depth, number of trees, seed and early stopping. It does not
mention the other HGB settings, so I left them at the library defaults (31 leaves per tree, no extra L2
penalty, 255 bins) rather than choosing anything myself. **Please confirm the defaults are fine.** If
you would rather fix them differently, tell me before the step 8 freeze.

## What I need from you

**Sign off step 7**, including the defaults in section 5. Step 8 is the methodology freeze. After that,
the 2014 test is opened once (step 9).

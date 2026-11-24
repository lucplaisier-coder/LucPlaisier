# Step 13 report: final figures and tables

Date: 24 Nov 2026. Code: `src/step13_final_outputs.py`. Outputs: `outputs/final/`. All tables in one place: `docs/final_tables.md`.

## What this step did

It collected the results of steps 5 to 12 into the tables and figures the manuscript will use. **Nothing was
re-run, re-fitted or re-chosen.** The script only reads saved result files and lays them out. No sealed data
is touched.

## The manuscript set

| Item | What it shows | Comes from |
|---|---|---|
| Table 1 | Loans, charge-off rate and average rate for each year 2010-2015 | steps 5, 10, 11 |
| Table 2 | AUC, Brier and calibration error for both models in 2013, 2014 and 2015 | steps 7, 9, 11 |
| Table 3 | **Main result**: the four approaches at 80% in 2014, with the six bootstrap comparisons | steps 9, 10 |
| Figure 1 | Calibration in 2014 and 2015 side by side (new layout of saved numbers) | steps 9, 11 |
| Figure 2 | Money per dollar across selection rates 50-100% in 2014 | step 10 |
| Figure 3 | What the HGB relies on (permutation importance) | step 12 |

## Annex

A1 2015 robustness; A2 grade groups; A3 fee 0% sensitivity; A4 sub-grade mix of the funded loans;
A5 odds ratios; A6 all logit coefficients; A7 the four pre-registered categories in 2014 and 2015;
A8 the input whitelist and blacklist.

## Two small corrections to earlier reports (no result changes)

* Step 11 report, grade groups, 2015 C fund-all: the saved value is 5.21 c, not 5.22 c (rounding typo).
* Step 11 report, 2014 D-G HGB: the saved value is 7.995 c; the annex shows 7.99 c, the report showed 8.00 c.
  The annex values are taken straight from the saved file.

## What I need from you

**Signed off by Lucas on 24 Nov 2026 ("approved").** Next on the roadmap: step 14 (manuscript).

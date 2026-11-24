# Step 10 report: financial evaluation of the 2014 test

Date: 24 Nov 2026. Code: `src/step10_financial_evaluation.py`. Outputs: `outputs/step10/`.
**Nothing was re-fitted or re-chosen.** This step only reads the frozen 2014 run from step 9. As a
check, it re-applies the frozen selection rule to the saved predictions and gets exactly the same funded
loans. **2015 is still sealed.**

## The tables and figures (control document F.13)

| # | Output | File |
|---|---|---|
| 1 | Sample and vintage table: loans, resolved share, mean rate and charge-off rate per year (2015 still sealed) | `output1_sample_by_vintage.csv` |
| 2 | Predictive table: AUC, Brier, ECE for 2013 (labelled "used for tuning") and 2014 | `output2_predictive.csv` |
| 3 | Reliability figure, 2014 | `output3_reliability_2014.png` |
| 4 | Net cash gain against selection rate, 50% to 100% (display only) | `output4_selection_rate_curve_2014.png` / `.csv` |
| 6 | Main financial table at 80%, with bootstrap intervals | `output6a_financial_2014_q80.csv`, `output6b_contrasts_2014_q80.csv` |
| Annex | Sub-grade mix of funded loans; 0% fee sensitivity; frozen input list | `annex_*.csv` |

Output 5 (permutation importance) is step 12. The grade-group annex and 2015 are step 11.

## What the new outputs show

**Charge-off rate by year (output 1):** 9.9% (2010), 10.6% (2011), 13.6% (2012), 12.3% (2013), **13.7% (2014)**.
The test year is the riskiest year in the sample, which is a fair stress test.

**Selection-rate figure (output 4).** It is for display only, and the result is still judged at the frozen 80%.

| Share funded | Fund-all | Platform | Logit | HGB |
|---:|---:|---:|---:|---:|
| 50% | 7.74 c | 7.43 c | 8.35 c | **8.40 c** |
| 70% | 7.74 c | 7.66 c | 8.05 c | **8.19 c** |
| **80% (frozen)** | 7.74 c | 7.68 c | 7.95 c | **8.01 c** |
| 90% | 7.74 c | 7.69 c | 7.86 c | **7.87 c** |
| 100% | 7.74 c | 7.74 c | 7.74 c | 7.74 c |

In plain words:
* **HGB is at or above the logit at every selection rate.** The gap is largest when you fund 70% or less.
  So the 80% result was not a lucky pick of the cut-off.
* **Both models gain more the more selective you are.** Skipping more of the riskiest loans helps.
* **The platform rule loses money the more selective it is.** Within a grade it skips the highest-rate
  loans. In 2014 the extra interest on those loans more than paid for their extra defaults. This is a
  clean finding for your discussion: inside a grade, Lending Club's finer pricing does not identify
  worse investments.
* Everything meets at 100%, where every approach funds every loan, so the curve checks itself.

**Sub-grade mix (annex).** The platform rule funds few or no loans from the riskiest sub-grade of each grade:
A5 is 2.3% of its funded loans against 6.2% for fund-all, B5 is 0.4%, and C5 to G5 are zero. It simply cuts
the top of each grade. The models keep loans from every sub-grade and pick individual loans
instead. This is the simplest evidence that within-grade selection works as intended.

**Main table (output 6).** These are the same numbers as step 9: HGB 8.01 c, logit 7.95 c, platform
7.68 c and fund-all 7.74 c. HGB − Logit is +0.054 c [+0.009, +0.100] and HGB − Platform is +0.332 c
[+0.259, +0.404]. All six pairwise comparisons and their intervals are in `output6b`.

## What I need from you

**Sign off step 10.** Next is step 11 (robustness). That unseals 2015 once, applies the same frozen
models and rules, and adds the grade groups A-B, C and D-G.

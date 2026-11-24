# Step 9 report: the 2014 test result

Run: 24 Nov 2026, 21:10 UTC (22:10 Amsterdam), **once**, exactly per methodology freeze v1
(record SHA-256 `e653b82d…`). The test code was pushed *before* unsealing, in commit `6491974`, and the
run log (`outputs/step9/test_opened_log.json`) records that commit. **2015 is still sealed.**

## The answer in one paragraph

On the untouched 2014 loans (162,570), **HGB earns the most per dollar invested: 8.01 cents**, against
7.95 for the logit, 7.68 for the platform rule and 7.74 for funding everything. Both pre-registered
primary comparisons have 95% intervals entirely above zero, so the pre-registered category is
**"AI adds economic value"**, and it is the only category that applies. **But the gain over the logit is
very small:** 0.05 cents per dollar, with an interval from 0.01 to 0.10 cents. The gain over Lending
Club's own ranking is bigger: 0.33 cents, with an interval from 0.26 to 0.40 cents.

## 1. Money at 80% within each grade (F.8)

| Approach | Loans funded | Charge-off rate | Principal loss rate | **Net cash gain per $** (fee 1%) | (fee 0%) |
|---|---:|---:|---:|---:|---:|
| A. Fund-all | 162,570 | 13.73% | 6.69% | 7.74 c | 8.82 c |
| B. Platform rule | 130,056 | 13.16% | 6.38% | 7.68 c | 8.75 c |
| C. Logit | 130,056 | 12.62% | 6.24% | 7.95 c | 9.04 c |
| D. HGB | 130,056 | **12.58%** | **6.11%** | **8.01 c** | **9.09 c** |

## 2. Primary comparisons, paired bootstrap, 1,000 draws (F.10)

| Comparison | Difference in net cash gain per $ | 95% interval |
|---|---:|---|
| **HGB minus Logit** | **+0.054 c** | +0.009 c to +0.100 c |
| **HGB minus Platform** | **+0.332 c** | +0.259 c to +0.404 c |

Secondary results, all in `bootstrap_contrasts_2014.csv`:
* The logit also beats the platform rule: +0.28 c, interval +0.21 to +0.35.
* HGB has lower principal losses than the logit (−0.13 points, interval −0.17 to −0.09). Its charge-off
  rate is not clearly different (−0.04 points, interval −0.11 to +0.03).
* **The platform rule earns less than funding everything** (−0.07 c, interval −0.13 to −0.01). The same
  happened on 2013.
* With a 0% service fee, every conclusion stays the same.

## 3. Prediction quality on 2014 (F.9)

| Model | AUC | Brier | ECE |
|---|---:|---:|---:|
| Logit | 0.670 | 0.1137 | 0.012 |
| HGB | 0.672 | 0.1135 | 0.014 |

HGB − Logit AUC = +0.0020, interval +0.0010 to +0.0029. **2014 is riskier than the training years**:
13.7% charged off, against 12.4% in 2010 to 2013. So both models predict a bit too little risk for
most loans (`reliability_2014.png`). Selection uses only the ranking, so this does not change which
loans are funded.

## 4. Pre-registered categories (F.11), 2014

| Category | Applies? | Why |
|---|---|---|
| AI adds economic value | **Yes** | Both primary intervals are above 0 |
| AI adds predictive but not economic value | No | Its condition needs a primary interval that includes 0 |
| Simpler model preferable | No | The HGB − Logit interval is above 0 |
| Platform pricing hard to beat | No | Both models beat the platform rule |

## 5. How to read this honestly (for the manuscript)

* **Statistically clear, economically small versus the logit.** 0.05 cents per dollar is about **$5 per
  $10,000 invested** over the loans' life. Versus the platform ranking it is about **$33 per $10,000**.
  Both models add real value over Lending Club's own sub-grade ranking. Boosting adds only a little on top
  of a well-built logit.
* **The interval for HGB − Logit starts close to zero (+0.009 c).** The plan already states that resampling
  single loans ignores shocks shared by loans issued in the same month. That limitation matters most here,
  because a wider interval could include zero. Say this plainly in the discussion.
* **This is the only test run.** Nothing was changed after seeing it. 2015 (step 11) can only show whether
  this result is *consistent*. It cannot change the 2014 category.

## Files

`outputs/step9/`: `financial_2014_q80.csv`, `bootstrap_contrasts_2014.csv`, `predictive_2014.csv`,
`success_categories_2014.csv`, `reliability_2014.png`, `test_opened_log.json`.
The loan-level scores and selection flags are saved for steps 10-12 in
`data/clean/step9_scores_selection_2014.parquet`. That file is not in git.

# Step 11 report: robustness (2015 and grade groups)

Run: 24 Nov 2026, 21:24 UTC (22:24 Amsterdam), **once**, exactly as committed *before* unsealing
(commit `4cf52af`, freeze record SHA `e653b82d…`). Log: `outputs/step11/robustness_2015_opened_log.json`.
Every year is now open. Nothing in the method was changed.

## The answer in one paragraph

**2015 is consistent with 2014.** The same single category holds: **"AI adds economic value"**. On the
282,787 loans from 2015, HGB again earns the most per dollar: 5.48 cents, against 5.44 for the logit,
5.34 for the platform rule and 5.30 for fund-all. Both primary intervals are above zero again. All the
gains are smaller than in 2014. **The grade groups show where it works:** both models beat the platform
rule in grades A to C in both years. But in the riskiest group (D-G) in 2015, the platform rule did best.

## 1. 2015 at 80% within each grade

| Approach | Charge-off rate | Principal loss rate | Net cash gain per $ (fee 1%) | (fee 0%) |
|---|---:|---:|---:|---:|
| A. Fund-all | 14.88% | 7.56% | 5.30 c | 6.35 c |
| B. Platform rule | 14.21% | 7.16% | 5.34 c | 6.40 c |
| C. Logit | 13.89% | 7.20% | 5.44 c | 6.49 c |
| D. HGB | 13.90% | **7.09%** | **5.48 c** | **6.54 c** |

| Comparison (paired bootstrap, 1,000 draws) | 2015 | 95% interval | (2014, for reference) |
|---|---:|---|---:|
| **HGB minus Logit** | +0.044 c | +0.011 to +0.076 | +0.054 c |
| **HGB minus Platform** | +0.139 c | +0.081 to +0.201 | +0.332 c |
| Logit minus Platform | +0.096 c | +0.034 to +0.156 | +0.278 c |
| Platform minus Fund-all | +0.044 c | −0.009 to +0.093 | −0.067 c |
| HGB minus Logit, AUC | +0.0010 | +0.0002 to +0.0018 | +0.0020 |

* **Consistent (G4 rule):** exactly the same categories hold in 2015 as in 2014.
* **HGB over the logit is remarkably stable:** +0.05 c in 2014 and +0.04 c in 2015, both small and both
  above zero.
* **The edge over the platform rule shrinks** from 0.33 c to 0.14 c. In 2015 the platform rule no longer
  loses money against fund-all; the difference is not clearly different from zero.
* **2015 was a harder year:** 14.9% charged off, against 13.7% in 2014. The net cash gain falls for every
  approach. Calibration is worse (ECE 0.031 to 0.035, `reliability_2015.png`): both models trained on
  2010 to 2013 predict too little risk. Selection uses only the ranking, so it still works. This fits the
  drift disclosed in step 5 (higher DTI, more missing employment, lower rates).

## 2. Grade groups (net cash gain per $, 80% within each grade)

| Year | Group | Fund-all | Platform | Logit | HGB |
|---|---|---:|---:|---:|---:|
| 2014 | A-B | 7.64 c | 7.53 c | **7.80 c** | 7.78 c |
| 2014 | C | 8.21 c | 8.20 c | 8.50 c | **8.54 c** |
| 2014 | D-G | 7.38 c | 7.38 c | 7.64 c | **8.00 c** |
| 2015 | A-B | 6.00 c | 5.96 c | **6.10 c** | 6.09 c |
| 2015 | C | 5.22 c | 5.26 c | 5.49 c | **5.54 c** |
| 2015 | D-G | 2.60 c | **2.97 c** | 2.76 c | 2.82 c |

In plain words:
* **Grades A to C (most of the money):** both models beat the platform rule and fund-all in both years.
  The logit and HGB are nearly tied in A-B. HGB is slightly ahead in C.
* **Grades D to G (the riskiest):** this is where HGB's advantage came from in 2014. In 2015 the platform
  rule did best there. That is an honest weakness: in the riskiest loans, the advantage did not hold in
  the later year. As F.12 specifies, these group numbers are point estimates without intervals, so small
  gaps should not be over-read.
* Charge-off rates (`grade_groups_2014_2015.csv`): both models have fewer charge-offs than the platform
  rule in every group and both years.

## 3. What this means for the manuscript

* **Main claim, robust across both years:** machine-learning selection within grade beats Lending Club's
  own sub-grade ranking, and HGB adds a small but consistent extra over the logit (about $4 to $5 per
  $10,000 invested).
* **Limits to state:** the gains are small and shrink in 2015. The riskiest grades in 2015 are an
  exception. The iid bootstrap ignores shocks shared by loans issued in the same month. Calibration
  drifts as default rates rise.

## Files

`outputs/step11/`: `financial_2015_q80.csv`, `bootstrap_contrasts_2015.csv`, `predictive_2015.csv`,
`success_categories_2015.csv`, `reliability_2015.png`, `grade_groups_2014_2015.csv`, and the run log.

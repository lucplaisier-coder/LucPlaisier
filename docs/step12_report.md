# Step 12 report: what drives the models (interpretability)

Date: 24 Nov 2026. Code: `src/step12_interpretability.py`. Outputs: `outputs/step12/`.
Everything here was fixed in the freeze (F.12, G2, G3). The final models are refitted exactly as in step 9,
and the code checks that their 2014 predictions are identical to the frozen run.

## In one paragraph

**The platform's own price is by far the strongest signal**, and both models lean on it. The borrower
information that adds most *on top of* that price is **income**, then **loan size**, **debt-to-income**
and **FICO**. Employment length, home ownership and loan purpose add very little. This fits the main
result: the models beat Lending Club's ranking mainly by using a few borrower facts that the sub-grade
does not fully price in.

## 1. Logit odds ratios (final fit 2010-2013; per one standard deviation, holding sub-grade and rate fixed)

| Borrower input | 1 SD equals | Odds ratio | Plain reading |
|---|---|---:|---|
| Log annual income | about ×1.7 income | **0.76** | Among loans with the same platform price, higher income means about 24% lower odds of charge-off |
| FICO (midpoint) | 32 points | **0.85** | A higher score means about 15% lower odds |
| Loan amount | $7,241 | **1.12** | A bigger loan at the same price means about 12% higher odds |
| Debt-to-income | 7.6 points | **1.11** | More existing debt means about 11% higher odds |
| Employment length | 3.5 years | 1.01 | Practically no effect |

* "Holding the platform's price fixed" matters. These effects come **in addition to** what sub-grade and
  rate already say. FICO, for example, is largely priced in already, so its extra effect is modest.
* As frozen (G3), home ownership, purpose, sub-grade and rate are not interpreted. Their coefficients are
  listed in `annex_logit_all_coefficients_final_fit.csv`.

## 2. HGB permutation importance on 2014 (drop in AUC when an input is shuffled; 10 repeats, seed 42)

Base AUC in 2014: 0.672.

| Input | AUC drop |
|---|---:|
| **Platform pricing block (sub-grade + rate together)** | **0.095** |
| Interest rate (alone) | 0.062 |
| Log annual income | 0.024 |
| Loan amount | 0.008 |
| Debt-to-income | 0.006 |
| Sub-grade (alone) | 0.005 |
| FICO | 0.004 |
| Loan purpose | 0.002 |
| Home ownership | 0.002 |
| Employment length | 0.002 |

The figure is in `output5_permutation_importance_2014.png`; this is output 5 of the plan. In plain words:
* **Shuffling the platform's price costs the model almost 0.10 AUC.** The AUC falls from 0.67 to about 0.58,
  so the borrower inputs alone carry much less signal.
* **Why the block matters:** shuffled alone, sub-grade looks unimportant (0.005), because the interest rate
  carries nearly the same information. Shuffled together, the true reliance on platform pricing shows. That
  is exactly why the plan pre-registered the block (control document C3).
* **Income is the most useful borrower input in both models:** it has the largest logit odds ratio per SD and the largest HGB drop among borrower inputs. That is a clean, defendable finding.

## What this means for the manuscript

The models do **not** replace Lending Club's pricing. They **refine** it, using income, loan size, debt load
and FICO to separate better and worse loans *within* the same price. That is consistent with the small,
steady gains in steps 9 to 11.

## What I need from you

**Sign off step 12.** That completes the analysis. Next on your roadmap: step 13 (final figures and tables),
step 14 (manuscript), step 15 (reproducibility audit), step 16 (defence preparation) and step 17 (submission).

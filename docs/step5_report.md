# Step 5 report: the clean dataset (for Lucas's sign-off)

**Status: signed off by Lucas on 24 Nov 2026** (`decision_records/step5_signoff.json`).

Date: 24 Nov 2026. Code: `src/step5_build_clean_dataset.py`. All numbers below come from
`outputs/step5/` and can be regenerated in about one minute.

## In one paragraph

The clean dataset is built. Starting from the 2,260,701 rows of the raw file, the study keeps
**611,816 finished 36-month loans from 2010 to 2015**, exactly the same counts as the independent
Gate 1 audit. The approved fix for the 148 wrong interest rates is applied and checked. All nine
proposed inputs are built with fixed rules; nothing is learned from the data yet, the feature list is
not frozen and no model has been started. **The 2014 and 2015 outcomes are sealed**: they are saved in
a separate file, but nothing was counted or averaged on them.

## 1. Which loans are in (control document F.3, F.4)

| Year | Role | Issued (all terms) | 36-month | Individual, meets credit policy | Finished (in study) |
|---|---|---:|---:|---:|---:|
| 2010 | train | 12,537 | 9,156 | 8,466 | 8,466 |
| 2011 | train | 21,721 | 14,101 | 14,101 | 14,101 |
| 2012 | train | 53,367 | 43,470 | 43,470 | 43,470 |
| 2013 | validation | 134,814 | 100,422 | 100,422 | 100,422 |
| 2014 | test | 235,629 | 162,570 | 162,570 | 162,570 |
| 2015 | robustness | 421,095 | 283,173 | 282,934 | 282,787 |

In plain words:
* 690 loans from 2010 are dropped because they were marked "Does not meet the credit policy".
* 239 joint applications from 2015 are dropped.
* 147 loans from 2015 were still running in Dec 2018 (72 current, 9 grace period, 66 late), so they
  have no final result and are left out. That is 0.05%, so **2015 passes the 99% rule (99.95%)** and
  every other year is 100% finished. The "Default" status never occurs, so no extra sensitivity is needed.
* Training has 66,037 loans, validation 100,422, test 162,570, robustness 282,787.

## 2. The interest-rate fix (decision G2-RATE-2026-11-24-v1)

* All **148** approved loans were found and given their SEC listing rate (3 in 2010, 3 in 2011,
  8 in 2012, 18 in 2013, 35 in 2014, 81 in 2015). The original stored rate is kept next to it, in a
  separate audit file that is never a model input.
* The two automatic warning screens were run again on the whole candidate pool. They flag 146 loans at
  exactly 6.00% and 122 "odd" rates for their month and sub-grade, the same counts as in step 4, and
  **every flagged loan is already in the approved list**. After the fix **nothing is flagged**.
* The fix moves the average rate by less than 0.004 percentage points in any year, so it matters for the few
  loans involved, not for the averages.

## 3. The nine proposed inputs (preparation rules G2-PREP-2026-11-24-v1)

FICO midpoint, debt-to-income, log income, employment length, home ownership, purpose, loan amount,
sub-grade (1 to 35) and interest rate. `verification_status` is **not** an input (decision
G2-VERIFY-EXCLUDE); its raw value is kept in the audit file only.

What the checks found:
* **No impossible values.** No FICO outside 300 to 850, no zero or negative income, no negative DTI,
  no unknown employment label, no missing or invalid sub-grade or rate. Grade always matches sub-grade.
* **Only one input has blanks: employment length**, rising from 3.0% (2010) to 5.9% (2014) and 6.7%
  (2015). They stay blank here; the models handle them later as agreed (median plus a 0/1 flag for the
  logit, native handling for boosting).
* **Drift to disclose (approved change-log row, no caps):**
  * DTI: in the training years it never goes above 34.99. In 2014, 2,253 loans are above that, and in
    2015, 9,701 (the maximum becomes 39.99). This looks like a policy change by Lending Club.
  * Interest rate: 2015 has 9,651 loans priced below the lowest training rate (5.42%), down to 5.32%,
    and a few hundred above the highest training rate in 2013 to 2015.
  * Income: at most 11 loans per year outside the training range.
  * Home ownership: one loan in 2014 and one in 2015 say "ANY", a label never seen in training. The
    agreed "unseen category" rule handles it later.

Full tables: `input_missing_share_by_vintage.csv`, `input_distribution_by_vintage.csv`,
`input_out_of_training_range_by_vintage.csv`, `category_counts_by_vintage.csv`.

## 4. Outcomes (train and validation years only)

* Charge-off rate: 9.9% (2010), 10.6% (2011), 13.6% (2012), 12.3% (2013). For 2014 and 2015 the table
  says "SEALED until step 8".
* Cash check (control document B8): principal + interest + late fees + recoveries equals total payments
  to within 1 cent for every loan in 2010 to 2013. So the money fields add up and the F.8 formulas
  can use them.

## 5. How the seal works

* The 2014 and 2015 target and cash fields go into `data/clean/SEALED_step5_outcomes_2014_2015.parquet`
  (445,357 rows). Its fingerprint is in `step5_run_summary.json`, so any later change would be visible.
* Every outcome summary in the code goes through a guard that only lets 2010 to 2013 through.
* Later steps open data through `src/load_clean.py`, which refuses to open the sealed file unless the
  step 8 freeze date is given.

## 6. Reproducibility

* The raw file's fingerprint is checked before and after the run (unchanged).
* Running the script twice gives byte-identical clean files.
* Pinned versions: pandas 2.3.3, numpy 2.3.5, pyarrow 22.0.0, scikit-learn 1.9.1.
* Runtime about 1 minute on a 4-core machine.

## What I need from you

1. **Sign off step 5**, or tell me what to change.
2. Nothing in this step freezes the features or starts modelling. The next roadmap step is yours to open.

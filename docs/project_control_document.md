# Project Control Document: Idea C (v1.0, 23 Nov 2026)

This is the single shared specification for Lucas and both AI advisers. Part 1 reviews Lucas's 24-section draft and answers questions A–E. Part 2 is the **locked minimum specification** (question F). After Step 1, change anything in Part 2 only through the change log in §F.15.

**Labels**
- **[G]** Capstone Guidelines.
- **[R]** Rubric.
- **[Course]** Lecture slides (PDF page numbers) or notebooks.
- **[DR]** My design review ([idea_C_design_review.md](idea_C_design_review.md)).
- **[Audit]** A fact Phase 1 must confirm on the actual file.

---

# PART 1: REVIEW OF THE DRAFT

## A. Parts that are fully consistent with the course and rubric

| Draft section | Consistent with |
|---|---|
| §1 Type A, investor decision among listed loans | G §1 Type A: "quantify economic value vs standard benchmarks". R1: "precisely defined, economically meaningful". |
| §2 Incremental value against simpler approaches; null result accepted | G §2: "incremental value relative to simpler models or naive strategies". L2 p44 (linear models as benchmark). L1 p53 ("use only if they beat simpler models"). L5 p4 rule 2. |
| §3 Defendability as a hard constraint | G §7 and R §6: failure to show understanding of any part can cut the grade, up to a fail. |
| §4 Fund-all, platform rule, L2 logit, HistGB | L2 (logit as the regulator-friendly benchmark), L3 p26 workflow (logit before the ensemble), L3 p36 ("HistGB is the default for this course"). Fund-all and the platform rule are the naive and domain benchmarks R1 asks for. |
| §7 Leakage rule, preprocessing fitted on train only | R2: "leakage explicitly addressed". L5 p6 ("every input must be known at… t"). This avoids the scaling-before-split and contamination-from-labels errors in the s4 notebook. |
| §8 Out-of-time vintage split, test used once | L3 p25 (dated rows keep their time order; the test is scored once). L2 p38–40. The NYC exemplar uses yearly blocks and freezes its choice before the test. |
| §9 Within-grade selection at 80% | [DR] condition C1. This is the step that makes the comparison test incremental information rather than "safe vs risky". |
| §10 Three observable financial outcomes, not annualised | R3: "financial metrics". L3 p17–18 (AUC differences must be translated into money). [DR] condition C4. |
| §11 AUC, Brier score, reliability plot, ECE | L3 p17 (AUC), p18 and p24 (ECE and calibration). L2 annex p51–56. |
| §13 Paired bootstrap | Plays the same role as the DM tests in L5: is the difference more than noise? It is simple and easy to explain. |
| §14 Odds ratios and held-out permutation importance | L2 and LogitR101 (odds ratios). L3 p21 ("permutation importance on held-out data is the measure to report"). |
| §15 Later vintage and grade groups | R3: "robustness across time periods or market conditions". The NYC exemplar audits by segment. |
| §16 Success defined before the test | L5 p23 and p25 (selection bias, backtest overfitting). R3: "critical, honest assessment". |
| §17 Reproducibility kit | R4: "runs start to finish… data sources and dependencies stated". The NYC exemplar uses a checksum, a versions file and a runtime log. |
| §18 Six core outputs | Fits the 2,000–3,000 word limit [G]. |
| §19 Manuscript sections | Exactly the structure in G §2. |
| §20 Two-minute narrative | G §4 (up to 2 minutes plus one question, strictly enforced). |
| §23 What not to do | Matches every weakness listed in the audit (§B "weaknesses not to repeat"). |

**One item is missing from the draft: a GenAI disclosure.** G §6 and L1 p17 require stating which AI tools were used and for what. Both advisers have to be named. It is added in Part 2 §F.14.

## B. Parts that are still assumptions rather than locked facts

Phase 1 must confirm each of these on the real file. Each one has a decision rule in Part 2, written now, so that discovering the answer cannot become a design choice.

| # | Assumption | Where it is used | Decision rule if false |
|---|---|---|---|
| B1 | The Kaggle `wordsforthewise/lending-club` file `accepted_2007_to_2018Q4.csv.gz` is available (about 2.26M rows × 151 columns, status snapshot about Dec 2018) | Everything | See D1 |
| B2 | Its licence permits research use | Repository, manuscript | See D2 |
| B3 | All required columns exist: grade, sub_grade, int_rate, funded_amnt, the total_rec_* fields, recoveries, collection_recovery_fee, fico_range_*, loan_status, issue_d | Target, features, outcome | A missing payment field stops the project at Gate 1 |
| B4 | The data dictionary has a browseNotes sheet listing the fields investors saw at listing | Whitelist | If it is absent, use LoanStats definitions and flag any field whose timing is ambiguous as blacklisted |
| B5 | At least 99% of 36-month loans issued 2010–2014 have a terminal status | Split | §F.4 rule |
| B6 | The structure of the "Default" status, and how often it occurs | Target | §F.3 rule |
| B7 | Loans marked "Does not meet the credit policy" appear only in early vintages | Sample | Excluded and counted (§F.3) |
| B8 | Identity check: total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries ≈ total_pymnt | Outcome | Gate 2 fails until the gap is explained |
| B9 | The investor service fee is about 1% of borrower payments received | Outcome | It is an assumption by definition. It stays labelled, with a sensitivity at 0% |
| B10 | Late fees were passed through to note investors | Outcome | If unclear, exclude late fees from cash received (they are small) and say so |
| B11 | Sample sizes are roughly 60–70k (train), 90–100k (validation), 150–160k (test) | Power | Report the actual counts. There is no rule here, because the sizes do not change the design. |
| B12 | Application-time AUC is about 0.65–0.72 | Leakage alarm | An AUC above 0.85 triggers an audit (§F.6) |
| B13 | Lending Club's own terms for the original public statistics, which are unknown | Licence | See D2 |

## C. Hidden methodological problems the draft misses

These are ordered by importance. All of them can be fixed without adding a model or a new method.

1. **Gate 4 is a selection-bias trap.** "HistGradientBoosting adds enough methodological value to justify inclusion" invites dropping HGB if its validation results look weak. HGB is *the object of the research question*. Removing it after looking at results is exactly the kind of researcher discretion the draft forbids elsewhere. **Fix:** Gate 4 should check only that HGB is *working correctly*: no leakage alarm, sensible validation numbers, and a grid search that completed. Whether it adds value is answered once, on the test.

2. **The platform rule needs a tie-break, which the draft leaves unspecified.** Within a sub-grade at a given date, Lending Club set essentially one interest rate. So ranking "by sub-grade, then rate" leaves large ties at the 80% cut-off, and the result would depend on the arbitrary row order. **Fix:** rank by sub-grade (the riskiest sub-grade is skipped first), then int_rate (higher rate skipped first), then a seeded random draw. Record the seed.

3. **Collinear platform features distort both interpretation tools.**
   - `grade` is an exact combination of the `sub_grade` dummies. `int_rate` is close to a function of sub-grade and date.
   - (a) The logit's odds ratios on these variables are not individually interpretable.
   - (b) Permutation importance splits the shared credit among correlated features, so each one can look unimportant.
   - **Fix:** drop `grade` as a *feature*. It remains the stratum for selection, and sub_grade nests it. Interpret odds ratios only for borrower variables, read as "holding platform pricing fixed". Permute `sub_grade` and `int_rate` together as one "platform pricing" block, in addition to the single-feature importances. This answers one clear question: how much does the model rely on the platform's own price?

4. **`term` is constant**, because the sample is 36-month loans only, so it carries no information. Remove it from the feature list. Also exclude `installment`, which is determined by amount, rate and term.

5. **Primary versus secondary comparisons are not declared.** There are four approaches and three financial metrics, plus AUC and Brier, so there is a real risk of finding *something* significant by chance. **Fix:** declare one primary metric (net cash gain per dollar at 80%) and two primary contrasts (HGB vs logit, and HGB vs platform rule). Everything else is secondary and described as such.

6. **The success categories need numbers.** §16 lists the categories but not the thresholds. Thresholds are given in §F.11.

7. **The procedure for refitting on train+validation is unspecified.** Early stopping chosen on 2013 cannot be reused when 2013 is part of the training data. **Fix:** use fixed `max_iter` values in the grid (no early stopping). Refit on 2010–2013 with exactly the selected hyperparameters.

8. **The missing-data rule is unspecified.** R2 explicitly asks for it. HGB handles NaN natively. The logit needs median imputation plus a missing-indicator column, fitted on training data only. Lock that now (§F.6).

9. **The bootstrap ignores common shocks.** Loans issued in the same months share macro conditions, so an iid loan bootstrap understates uncertainty for the *levels* of each strategy's outcomes. For *paired differences* between strategies on the same test loans, most of the common shock cancels. **Fix:** report paired differences as the inference, keep the iid paired bootstrap, and state this limitation in one sentence. Do not add a block bootstrap.

10. **The robustness-across-time fallback is missing.** If 2015 fails the resolution rule, robustness across time rests on one vintage. **Fix:** pre-commit that in that case, the 2013 validation results are reported as a second out-of-time period. They must be labelled as "used for tuning", so they are not independent.

11. **Calibration across vintages largely reflects shifts in the base default rate.** This is a finding, not a flaw, and selection is rank-based, so it does not depend on calibration. However, the manuscript must not present poor calibration as poor ranking. Show the default rate by vintage in the sample table.

12. **Investors could not always get the loans they chose.** Popular loans could be fully funded by other investors first, and Lending Club sold some loans as whole loans. Mention this as an implementation limitation. No modelling is needed.

13. **Two sample restrictions to lock:** individual applications only (`application_type`), and `policy_code` = 1. Otherwise mixed populations enter silently.

14. **`loan_amnt` vs `funded_amnt`.** `loan_amnt` (requested, known at listing) may be a feature. `funded_amnt` is used only as the investment base P in the outcome formulas.

## D. Data and licensing issues that could kill the project

| # | Issue | Likelihood [my judgement] | Kills the project? | Response |
|---|---|---|---|---|
| D1 | The Kaggle file has been removed or cannot be downloaded | Low | Yes, for this route | Stop at Gate 1. Freddie Mac is a *different* project (credit-loss forecasting), so return to me before switching. |
| D2 | The licence prohibits research use | Low | Yes, if it explicitly forbids research | If it says CC0 or another open licence, you are clear. If it is silent or unclear, research use without redistribution is standard, and the manuscript states the source and terms. Screenshot whatever the page shows. |
| D3 | Payment or recovery fields are missing | Very low | Yes | Stop at Gate 1 |
| D4 | The 2014 vintage is not resolved (below 98%) | Low | No | Shift the test to 2013 and validation to 2012, train on 2010–2011. This uses the same rule, decided now (§F.4). |
| D5 | Kaggle requires extra account verification for the API | Low–Medium | No | One manual step, documented in the README |
| D6 | The file disappears after submission, so a grader cannot rerun | Low | No | Record the checksum and file version. Keep a private copy that is not redistributed. |

None of these is likely to kill the project. D1–D3 are checked in the first hour of Phase 1.

## E. Parts that are still more complicated than they need to be

| Item | Recommendation |
|---|---|
| The optional bureau variables "if the core set proves inadequate" | **Remove the option.** "Inadequate" is undefined, so it is an unspecified discretionary decision. Lock the core feature set now (§F.5). The platform sub-grade already summarises the bureau information. |
| HGB early stopping on an internal split | Replace it with a fixed 4-cell grid (C7). |
| Monotonic constraints (mentioned in my design review) | **Drop them.** They are not needed to answer the question. |
| Reliability plot *and* ECE | Keep both. ECE is one number (L3 p18), and the plot shows why. Lock 10 equal-frequency bins. |
| Selection-rate figure (50–100%) | Keep it, as a display only. It is declared now, so it cannot be used to choose a rate afterwards. |
| Subgrade distribution of selected loans | Keep it as one annex table. It is the simplest evidence that within-grade selection works as intended. |
| 12 phases and 8 gates | Fine for planning, and nothing in them adds analysis. |
| Principal loss and charge-off rate alongside net cash | Keep all three. They answer three plain questions: how often loans default, how much principal is lost, and the overall cash result. |

There is nothing else to cut. The draft is already close to the minimum the rubric requires.

---

# PART 2 (F): LOCKED MINIMUM SPECIFICATION v1.0

## F.1 Identity
- **Type:** A, Industry Focus.
- **Title:** *Out-of-Time Credit Risk Prediction and Loan Selection: Does Gradient Boosting Add Value Beyond Platform Pricing and Logistic Regression?*
- **Research question:** "Among already-approved, listed 36-month Lending Club loans, does histogram gradient boosting, when used to select loans within each credit grade, produce lower realised charge-off rates, lower principal losses and higher undiscounted lifetime net cash gains than Lending Club's sub-grade ranking and a regularised logistic regression, on a later, untouched vintage? Fund-all is the reference."
- **Decision-maker:** a note investor choosing which listed loans to fund.
- **Not studied:** approve/decline decisions for applicants, and real-time trading.
- **Study type:** out-of-time validation (not a real-time backtest), because training outcomes resolve years after issue.

## F.2 Data
- **Source:** Kaggle `wordsforthewise/lending-club`, `accepted_2007_to_2018Q4.csv.gz`. Record its SHA-256, the download date, the row and column counts, and a screenshot of the licence.
- **Data dictionary:** `LCDataDictionary.xlsx` (LoanStats and browseNotes sheets).
- **Raw data is never committed to the repository.** The README gives Kaggle API steps.

## F.3 Sample and target
- **Filters:**
  - `term` = 36 months;
  - `issue_d` year in 2010–2015;
  - `policy_code` = 1;
  - `application_type` = Individual;
  - exclude every status prefixed "Does not meet the credit policy".
- **Target:** `Fully Paid` = 0, `Charged Off` = 1.
- **"Default" status, decided now:**
  - exclude it from the main sample and count it as unresolved;
  - if it exceeds 0.5% of any vintage used, report a one-line sensitivity treating it as 1.
- **Unresolved statuses** (Current, In Grace Period, Late 16–30, Late 31–120, Default) are excluded. Their share is reported for each vintage.

## F.4 Time design
- **Vintage** = calendar year of `issue_d`.
- **Splits:** train 2010–2012, validate 2013, final test 2014, robustness 2015.
- **Resolution rule, decided now:** a vintage is usable if at least 99% of its filtered loans have a terminal status.
  - If 2014 is below 98%: test 2013, validate 2012, train 2010–2011.
  - If 2015 is below 99%: it is not used, and the 2013 validation results are shown as a second period, labelled "used for tuning".
- **Final fit:** refit on 2010–2013 with the frozen hyperparameters, then score 2014 once and 2015 once.

## F.5 Features (core set, locked)
- **Whitelist** (each entry must appear in browseNotes, otherwise it moves to the blacklist):
  - FICO = mean(`fico_range_low`, `fico_range_high`)
  - `dti`
  - log(`annual_inc`)
  - `emp_length` (ordinal 0–10, NaN kept)
  - `home_ownership`
  - `purpose`
  - `loan_amnt`
  - `verification_status`
  - `sub_grade` (ordinal 1–35)
  - `int_rate`
- `grade` is the selection stratum, **not** a feature. `term` and `installment` are excluded.
- **Blacklist, outcome or post-origination:** `loan_status` (target only); `total_pymnt*`; `total_rec_*`; `recoveries`; `collection_recovery_fee`; `last_pymnt_*`; `next_pymnt_d`; `out_prncp*`; `last_credit_pull_d`; `last_fico_range_*`; `pymnt_plan`; `hardship_*`; `debt_settlement_*`; `settlement_*`; `funded_amnt_inv`.
- **Blacklist, other:** `id`, `member_id`, `url`, `desc`, `title`, `emp_title`, `zip_code`, `addr_state`, `policy_code`, plus every column not on the whitelist.
- **Freeze:** the whitelist is saved to `features.py` and frozen at Gate 2.

## F.6 Preprocessing and leakage controls
- One sklearn `Pipeline` per model, fitted on the training vintages only.
- **Logit:** median imputation plus a missing-indicator column, standard scaling, and one-hot encoding of categoricals (unseen categories ignored).
- **HGB:** native NaN handling and categorical support.
- **Leakage alarm:** stop and audit if any single feature gives a validation AUC above 0.80, or a full model gives one above 0.85.

## F.7 Approaches (locked)

All four select 80% of the loans within each grade, except fund-all.

| Approach | Ranking score | Settings |
|---|---|---|
| A. Fund-all | none (100%) | — |
| B. Platform rule | sub-grade (riskiest skipped first), then int_rate (higher skipped first), then a seeded random draw | seed 42 |
| C. Logit | predicted probability of default (PD) | L2 penalty, C chosen on 2013 from {0.01, 0.1, 1}, `max_iter`=1000 |
| D. HGB | PD | `learning_rate`=0.05, `min_samples_leaf`=200, `max_depth` from {3, 5} and `max_iter` from {200, 500}, both chosen on 2013. No early stopping. `random_state`=42 |

- **Selection rate:** q = 80% within each grade, fixed now. Grades F and G are merged into one stratum if a grade has fewer than 200 loans in the test vintage.
- **Tuning criterion on 2013:** AUC. Ties within 0.002 go to the simpler setting (smaller C or depth, then fewer iterations).

## F.8 Financial outcomes (locked formulas, test vintage)

For a selected set S:
- **Charge-off rate** = share of loans in S with target 1.
- **Principal loss rate** L_S = Σ max(0, P − PR − (RC − CF)) / Σ P, where only charged-off loans contribute.
- **Net cash gain per dollar** G_S = Σ (PR + IN + LF + RC − CF − SF) / Σ P − 1.
- **Definitions:**
  - P = `funded_amnt`; PR = `total_rec_prncp`; IN = `total_rec_int`; LF = `total_rec_late_fee`; RC = `recoveries`; CF = `collection_recovery_fee`. All are **observed**.
  - SF = 0.01 × (PR + IN + LF). This is an **assumption**; a sensitivity is reported with SF = 0.
- **Descriptive only:** the mean `int_rate` of the selected loans, and their distribution across sub-grades.
- **Never called:** IRR, annualised return or NAR.
- **Unavailable, stated as limitations:** payment timing, reinvestment, cash drag, taxes, secondary-market prices, platform risk and funding competition.

## F.9 Predictive metrics (locked)
- AUC and Brier score on 2013 and 2014 (and 2015 if usable).
- A reliability plot and ECE using 10 equal-frequency bins.
- No other predictive metrics.

## F.10 Inference (locked)
- **Primary metric:** G_S at q = 80%.
- **Primary contrasts:** HGB − Logit, and HGB − Platform.
- **Secondary:** every other contrast, the charge-off and loss rates, and AUC differences.
- **Method:** paired bootstrap. Resample test loans with replacement 1,000 times (seed 42), keeping each approach's selection flags fixed, and report 95% percentile CIs of the differences.
- **Stated limitation:** iid resampling ignores common shocks across issue months.

## F.11 Success categories (pre-registered; these apply to the 2014 test only)

| Category | Criterion |
|---|---|
| **AI adds economic value** | The 95% CIs for HGB − Logit and for HGB − Platform on G_S are both above 0 |
| **AI adds predictive but not economic value** | The 95% CI of the AUC difference HGB − Logit is above 0, but at least one of the two primary G_S CIs includes 0 |
| **Simpler model preferable** | The CI for HGB − Logit on G_S includes 0 or is negative, and the AUC CI includes 0. The logit is preferred on grounds of simplicity. |
| **Platform pricing hard to beat** | Neither the logit nor HGB has a G_S CI above 0 against the platform rule |

- 2015 (or 2013, if 2015 is unusable) and the grade groups are reported as **consistent or inconsistent** with the 2014 category. They never redefine it.
- The categories are not mutually exclusive. Report every one that applies.

## F.12 Interpretability and robustness (locked)
- **Logit:** odds ratios per one standard deviation for the borrower variables, interpreted "holding platform pricing fixed".
- **HGB:** held-out permutation importance on 2014, with 10 repeats. The sub-grade and int_rate pair is also permuted together as one "platform pricing" block.
- **Robustness:** (1) the later vintage, according to the §F.4 rule; (2) grade groups A–B, C and D–G, with G_S and the charge-off rate per group.
- No other analyses.

## F.13 Outputs (locked)
1. Sample and vintage table: counts, resolved share, default rate, mean rate.
2. Predictive table: AUC, Brier, ECE.
3. Reliability figure.
4. G_S against q (50–100%) for the four approaches.
5. Permutation-importance figure.
6. Main financial table at 80%, with bootstrap CIs.

**Annex:** sub-grade mix of the selected loans; grade-group results; the SF = 0 sensitivity; the feature whitelist and blacklist.

## F.14 Reproducibility and disclosure
- **Repository contents:** README (Kaggle API steps and the licence note), `requirements.txt` with pinned versions, the expected checksum, `features.py`, seeds, and one entry point that regenerates every table and figure.
- **Runtime:** logged; Colab is acceptable.
- **GenAI disclosure (G §6):** names both AI advisers, plus any coding assistant, and the purpose of each.
- **Manuscript and defence:** follow draft §19–20.

## F.15 Gates (Gate 4 revised) and change control
- **Gates:** keep the draft's Gates 1–3 and 5–8. Gate 4 is replaced with: "HGB pipeline runs correctly on validation (no leakage alarm, grid completed, sensible numbers)". It does not ask whether HGB adds value.
- **Change log:** any change to Part 2 after Gate 2 is recorded here with its date, the reason, and confirmation that the test vintage has not been seen. After Gate 5 (specification frozen), only bug fixes are allowed. Each fix must be disclosed in the manuscript annex.

| Date | Change | Reason | Test seen? |
|---|---|---|---|
| 2026-11-23 | v1.0 created | — | No |
| 2026-11-24 | F.5 int_rate: for 148 fixed loan IDs (146 stored at 6.00%, plus 1119682 and 56924541), use the SEC posting-supplement listing rate in the working copy. Raw file unchanged; stored rate kept alongside. Record G2-RATE-2026-11-24-v1; ledger `approved_listing_rate_ledger_v1.csv` (SHA-256 0de39103…6dfe). | Stored value is a post-issue servicing rate (SCRA), not the listing rate, and is outcome-linked. | No 2014 outcomes used; an earlier external review displayed a few statuses (disclosed) |
| 2026-11-24 | F.5 whitelist: verification_status removed, with no derived flag or proxy; raw field and sample kept. Record G2-VERIFY-EXCLUDE-2026-11-24-v1. | The final verification category has no completion timestamp, and checks could finish after investor commitment, so its value at decision time is unverified. A timing precaution. | No 2014 outcomes used; an earlier external review displayed a few statuses (disclosed) |
| 2026-11-24 | F.6 preparation: adopt preparation rules G2-PREP-2026-11-24-v1 for the nine inputs. Logit employment-missing flag added by a separate branch, `MissingIndicator(error_on_new=False)`, left unscaled. | Fixes one pipeline layout so two people build the same thing. | No |
| 2026-11-24 | F.7 HGB: set `early_stopping=False` explicitly. Clarification, not a change. | The library turns it on by default above 10,000 rows, against F.7. | No |
| 2026-11-24 | F.14 disclosure: report the 2014–15 DTI range shift and rising blank employment; no caps. Add an input-only out-of-range count per year. | Non-stationarity must be addressed (rubric, Data and Methodology). | No (inputs only) |
| 2026-11-24 | F.15: this table is the official log. Separate decision records stay as supporting detail (step4_closure/decision_records/). | One place for a marker to check. | No |
| 2026-11-24 | F.5 Gate 2: the nine inputs are frozen in `features.py` (FICO midpoint, dti, log income, employment length, home ownership, purpose, loan amount, sub-grade, int_rate). Record G2-FEATURE-FREEZE-2026-11-24. | Gate 2 freeze, approved by Lucas before any model was fitted. | No |

---

**Status:** ready for Phase 1 (data access and audit) once Lucas accepts this version. Phase 1 must run on Lucas's machine or Colab, because my environment cannot reach Kaggle.

# Does gradient boosting add value beyond platform pricing? (Lending Club, 36-month loans)

Capstone project, AI for Finance (Type A, Industry Focus). Author: Lucas.

**Research question.** Among already-approved, listed 36-month Lending Club loans, does histogram
gradient boosting, used to select loans within each credit grade, give lower charge-off rates,
lower principal losses and higher lifetime net cash gains than Lending Club's own sub-grade ranking
and a regularised logistic regression, on a later, untouched vintage? Funding every loan is the reference.

The full locked specification, with its change log, is in
[`docs/project_control_document.md`](docs/project_control_document.md).

## For the marker

| What | Where |
|---|---|
| Manuscript | Handed in as a separate document; this repository holds only the code and the results it produces |
| Every final table and figure | [`docs/final_tables.md`](docs/final_tables.md) and `outputs/final/` |
| Rerun everything and check it | `python -m src.reproduce_all` (about 8 minutes; prints `REPRODUCED`), or the Colab notebook below |
| Locked method and change log | [`docs/methodology_freeze_v1.md`](docs/methodology_freeze_v1.md), [`docs/project_control_document.md`](docs/project_control_document.md) |
| One-time test logs (2014, 2015) | `outputs/step9/test_opened_log.json`, `outputs/step11/robustness_2015_opened_log.json` |
| Author's decisions and sign-offs | `decision_records/` |

## Status

| Roadmap step | Status |
|---|---|
| 1-4 Data audit, timing checks, rate fix, preparation rules | Closed 24 Nov 2026 (decision records in `decision_records/`) |
| 5 Clean dataset | Signed off 24 Nov 2026 ([report](docs/step5_report.md), `decision_records/step5_signoff.json`) |
| Gate 2 feature freeze | Nine inputs frozen 24 Nov 2026 (`src/features.py`, `decision_records/gate2_feature_freeze.json`) |
| 6 Baseline logistic + platform benchmark | Signed off 24 Nov 2026 ([report](docs/step6_report.md), `decision_records/step6_signoff.json`) |
| 7 Add HistGB | Signed off 24 Nov 2026 ([report](docs/step7_report.md), `decision_records/step7_signoff.json`) |
| 8 Methodology freeze | FROZEN 24 Nov 2026 ([record](docs/methodology_freeze_v1.md), `decision_records/methodology_freeze_v1.json`) |
| 9 Untouched 2014 test | Run once and signed off 24 Nov 2026 ([report](docs/step9_report.md), `decision_records/step9_signoff.json`): "AI adds economic value" |
| 10 Financial evaluation | Signed off 24 Nov 2026 ([report](docs/step10_report.md), `decision_records/step10_signoff.json`) |
| 11 Robustness | 2015 run once and signed off 24 Nov 2026 ([report](docs/step11_report.md), `decision_records/step11_signoff.json`): consistent with 2014 |
| 12 Interpretability | Signed off 24 Nov 2026 ([report](docs/step12_report.md), `decision_records/step12_signoff.json`) |
| 13 Final figures and tables | Signed off 24 Nov 2026 (`decision_records/step13_signoff.json`, [report](docs/step13_report.md), [all tables](docs/final_tables.md)) |
| 14 Manuscript | Signed off 25 Nov 2026 (`decision_records/step14_signoff.json`); handed in as a separate document, kept outside this repository |
| 15 Reproducibility audit | Signed off 26 Nov 2026 (`decision_records/step15_signoff.json`, [report](docs/step15_report.md)): fresh clone, clean install, full rerun from the raw file, every table and figure identical |
| 16 Defence preparation | Signed off 26 Nov 2026 (slides and Q&A kept outside this repository) |
| 17 Submission | Final version of this repository; see "For the marker" above |

## Run everything

**Colab (one click):** open [`notebooks/run_all_colab.ipynb`](notebooks/run_all_colab.ipynb) in Google Colab
and choose *Runtime > Run all*. It clones the code, installs the pinned packages, gets the raw file
(Google Drive or Kaggle API) and runs every step.

**Your own computer** (tested with Python 3.12 and 3.13):

```bash
pip install -r requirements.txt
# put accepted_2007_to_2018Q4.csv.gz in data/raw/   (or use --source, see below)
python -m src.reproduce_all     # reruns every step (about 7 minutes) and checks the results
```

`reproduce_all` runs the steps below in order and ends by asking git whether any committed table or
figure changed. A faithful rerun prints `REPRODUCED`. You can also run the steps one by one:

```bash
python -m src.step5_build_clean_dataset
python -m src.step6_baselines
python -m src.step7_histgb
python -m src.step9_test_2014 --dry-run      # rehearsal on 2013
python -m src.step9_test_2014 --reproduce    # reruns the frozen 2014 test, checks = committed results
python -m src.step10_financial_evaluation
python -m src.step11_robustness --dry-run    # rehearsal with 2014 as stand-in
python -m src.step11_robustness --reproduce  # reruns the frozen 2015 run, checks = committed results
python -m src.step12_interpretability
python -m src.step13_final_outputs          # collects saved results; no new analysis
```

**The one-time tests.** `--unseal` was the single, logged opening of 2014 and of 2015
(`outputs/step9/test_opened_log.json`, `outputs/step11/robustness_2015_opened_log.json`). It refuses
to run again. `--reproduce` runs the same frozen code, writes to `outputs/reproduce/` (git-ignored) and
stops with an error if any result table differs from the committed one.

## Getting the data

The raw data is **not** in this repository (size and licence). Source: Kaggle dataset
[`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
file `accepted_2007_to_2018Q4.csv.gz` (2,260,701 loan rows, 151 columns).

* **Kaggle API:** create an API key (kaggle.com > Settings > API), set `KAGGLE_USERNAME` and
  `KAGGLE_KEY` (or place `kaggle.json` in `~/.kaggle/`), then run
  `python -m src.step5_build_clean_dataset --source kaggle`.
* **Manual:** download the file from the page above and pass its path with `--source <path>`,
  or copy it into `data/raw/`.

The code checks the file's SHA-256 fingerprint before and after every run and stops if it differs:

```
55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d  accepted_2007_to_2018Q4.csv.gz
```


The raw file is opened read-only and never changed.

## Repository layout

```
src/config.py                     every fixed rule (years, filters, splits, seeds) in one place
src/raw_data.py                   get the raw file, check its fingerprint, read the needed columns
src/sample.py                     sample filters (F.3), resolution rule (F.4), target
src/rates.py                      approved 148-loan listing-rate fix (G2-RATE-2026-11-24-v1)
src/inputs.py                     fixed per-loan input rules P1-P9 (G2-PREP-2026-11-24-v1)
src/step5_build_clean_dataset.py  step 5 entry point
src/load_clean.py                 how later steps open the clean files (fingerprint check + seal)
src/features.py                   the FROZEN list of nine inputs (Gate 2)
src/models.py                     logit and HistGB pipelines (preparation fitted on training rows only)
src/selection.py                  80%-within-grade selection; platform rule
src/metrics.py                    AUC, Brier, ECE and the F.8 money formulas
src/step6_baselines.py            step 6 entry point
src/step7_histgb.py               step 7 entry point
src/step9_test_2014.py            step 9: the one-time 2014 test (per the freeze record)
src/step10_financial_evaluation.py  step 10: tables and figures from the frozen 2014 run
src/step11_robustness.py          step 11: 2015 and grade groups (per the freeze record)
src/step12_interpretability.py    step 12: logit odds ratios and HGB permutation importance
src/step13_final_outputs.py       step 13: final tables and figures (outputs/final/)
src/reproduce_all.py              step 15: rerun everything and check every committed result
decision_records/                 approved step-4 decisions, unchanged, with SHA-256 manifest
outputs/step5/ ... outputs/final/  summary tables and figures from each step (no loan-level data)
notebooks/run_all_colab.ipynb     one-click rerun
docs/                             control document and step reports
data/                             raw and clean loan-level data (git-ignored, created by the code)
```

## Rules the code enforces

* **No look at the test years.** 2014 (test) and 2015 (robustness) outcomes are written to a separate
  `SEALED_...` file whose fingerprint is recorded. No statistic is computed on them, and
  `load_clean.load_sealed_outcomes()` refuses to open them until the step 8 methodology freeze is declared.
* **No leakage from preparation.** Step 5 only applies fixed per-loan formulas. Anything learned from
  data (medians, scaling, category lists) happens later inside the model pipelines, fitted on the
  training years only.
* **Stop, don't guess.** Unexpected values (unknown status, invalid sub-grade, FICO outside 300-850,
  zero income, new rate flags, changed sample sizes) stop the run with a message.

## Pinned software

See `requirements.txt`: pandas 2.3.3, numpy 2.3.5, pyarrow 22.0.0, scikit-learn 1.9.1, kaggle 2.2.5, matplotlib 3.10.8.
Each run records the versions it used in `outputs/step5/step5_run_summary.json`.

## GenAI disclosure

Two generative AI tools were used. ChatGPT (OpenAI) was used as an adviser in the design phase, to
discuss and review the research design. Claude (Anthropic) was used as an adviser for specification
review, as a coding assistant that wrote and ran this code under the author's direction, and to draft
the manuscript (handed in separately) from the signed-off results. Every methodological decision was made and approved by the
author; the decision records are in `decision_records/` and the control document's change log.

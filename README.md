# Does gradient boosting add value beyond platform pricing? (Lending Club, 36-month loans)

Capstone project, AI for Finance (Type A, Industry Focus). Author: Lucas.

**Research question.** Among already-approved, listed 36-month Lending Club loans, does histogram
gradient boosting, used to select loans within each credit grade, give lower charge-off rates,
lower principal losses and higher lifetime net cash gains than Lending Club's own sub-grade ranking
and a regularised logistic regression, on a later vintage not used for fitting or tuning? Funding every loan
is the reference. This is a retrospective experiment across issue-year cohorts, using outcomes observed
later (to about March 2019); it is not a strategy that could have been trained and run live in 2014,
because many 2013 loans had not yet finished when 2014 loans were issued.

The full locked specification, with its change log, is in
[`docs/project_control_document.md`](docs/project_control_document.md).

## For the marker

| What | Where |
|---|---|
| Manuscript | Handed in as a separate document; this repository holds only the code and the results it produces |
| Every final table and figure | [`docs/final_tables.md`](docs/final_tables.md) and `outputs/final/` |
| Rerun everything and check it | `python -m src.reproduce_all --expect-commit <SHA>` (about 8 minutes; prints `REPRODUCED` only after a complete, successful run), or the Colab notebook below |
| Code version to check | **PENDING**: the submission commit and tag are fixed when the final changes are merged (see "Version pin" below) |
| What was seen before the freeze | [`decision_records/clarification_2026-11-27_test_years_and_reproduction.md`](decision_records/clarification_2026-11-27_test_years_and_reproduction.md) |
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
| 9 2014 test (first model evaluation of 2014) | Run once and signed off 24 Nov 2026 ([report](docs/step9_report.md), `decision_records/step9_signoff.json`): "AI adds economic value" |
| 10 Financial evaluation | Signed off 24 Nov 2026 ([report](docs/step10_report.md), `decision_records/step10_signoff.json`) |
| 11 Robustness | 2015 run once and signed off 24 Nov 2026 ([report](docs/step11_report.md), `decision_records/step11_signoff.json`): consistent with 2014 |
| 12 Interpretability | Signed off 24 Nov 2026 ([report](docs/step12_report.md), `decision_records/step12_signoff.json`) |
| 13 Final figures and tables | Signed off 24 Nov 2026 (`decision_records/step13_signoff.json`, [report](docs/step13_report.md), [all tables](docs/final_tables.md)) |
| 14 Manuscript | Signed off 25 Nov 2026 (`decision_records/step14_signoff.json`); handed in as a separate document, kept outside this repository |
| 15 Reproducibility audit | Signed off 26 Nov 2026 (`decision_records/step15_signoff.json`, [report](docs/step15_report.md)): fresh clone, clean install, full rerun from the raw file. Checker hardened 27 Nov 2026 (exact bytes, full inventory, failed stages stop the run); the hardened check has not yet been run on the real data |
| 16 Defence preparation | Signed off 26 Nov 2026 (slides and Q&A kept outside this repository) |
| 17 Submission | Final version of this repository; see "For the marker" above |

## Run everything

**Colab (one click):** open [`notebooks/run_all_colab.ipynb`](notebooks/run_all_colab.ipynb) in Google Colab,
set `CODE_COMMIT` in the first code cell, and choose *Runtime > Run all*. It clones the code, checks out
exactly that commit (and refuses an existing checkout at another commit or with modified files), installs
the pinned packages into the Python that runs the steps, gets the raw file (Google Drive or Kaggle API)
and runs every stage. Every command is checked: the notebook stops at the first failure.

**Your own computer** (tested with Python 3.12 and 3.13):

```bash
pip install -r requirements.txt
# put accepted_2007_to_2018Q4.csv.gz in data/raw/   (or use --source, see below)
python -m src.reproduce_all --expect-commit <full commit SHA>   # every stage (about 8 minutes), then the check
```

`reproduce_all` runs the stages below in order, records each one in `outputs/reproduce/run_record.json`
(with the Python and package versions actually used) and stops at the first failure. It prints
`REPRODUCED` only if **all** of these hold:

* every stage of this run finished successfully, in order;
* the code is the commit given with `--expect-commit` (or, without it, the commit printed) and no tracked
  code file is modified;
* every file in the expected-artifact inventory (`reproduction/expected_artifacts.json`, 104 committed
  result files) is present: files the stages rewrite must have been rewritten by this run and be
  byte-identical to the committed version; the 2014 and 2015 one-time tables must be reproduced byte for
  byte in `outputs/reproduce/`; one-time logs must be unchanged; the five run-summary files must match
  except for their time, duration and machine fields.

There is no numerical tolerance. Unchanged committed files alone cannot pass. (The steps' own
`--reproduce` comparison uses pandas' default tolerance and is only an early warning; the verdict is
`reproduce_all`'s.) The stages, which the notebook runs one by one with `--stage <name>`:

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
python -m src.present_figures               # redraws Figure 3 from outputs/final/figure3_data.csv only
```

**Version pin.** A file cannot contain its own commit identifier, so the pin lives outside the analysis
code: once the final changes are merged, the submission commit is tagged and its full SHA is written into
the notebook's `CODE_COMMIT` and the manuscript (a later, launcher-only commit may do this; the notebook
then checks out and verifies the pinned analysis commit). Until then the pin is **pending** and a run
reports only which commit it checked.

**Tests of the checker** (synthetic fixtures only; no data, no model, a few seconds):
`python -m unittest discover -s tests -v`. Passing them says the checker works, not that the project
was reproduced.

**The one-time tests.** `--unseal` was the single, logged opening of 2014 and of 2015
(`outputs/step9/test_opened_log.json`, `outputs/step11/robustness_2015_opened_log.json`). It refuses
to run again. `--reproduce` runs the same frozen code and writes to `outputs/reproduce/` (git-ignored);
`reproduce_all` then compares those tables byte for byte with the committed one-time results.

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
src/reproduce_all.py              step 15: rerun every stage, then verify the full artifact inventory
src/present_figures.py            presentation only: Figure 3 redrawn from its saved data
reproduction/expected_artifacts.json  the 104 committed result files and their SHA-256 values
tests/                            synthetic tests of the checker (no data, no model)
decision_records/                 approved step-4 decisions, unchanged, with SHA-256 manifest
outputs/step5/ ... outputs/final/  summary tables and figures from each step (no loan-level data)
outputs/presentation/             presentation-only redraws (Figure 3)
notebooks/run_all_colab.ipynb     one-click rerun
docs/                             control document and step reports
data/                             raw and clean loan-level data (git-ignored, created by the code)
```

## Rules the code enforces

* **No model sees the test years before the freeze.** In this code, 2014 (test) and 2015 (robustness)
  outcomes are written to a separate `SEALED_...` file whose fingerprint is recorded, no statistic is
  computed on them, and `load_clean.load_sealed_outcomes()` refuses to open them until the step 8
  methodology freeze is declared. Earlier data audits outside this code (23 Nov 2026, before any model)
  did compute descriptive 2014-2015 outcome summaries; see the 27 Nov clarification record.
* **No leakage from preparation.** Step 5 only applies fixed per-loan formulas. Anything learned from
  data (medians, scaling, category lists) happens later inside the model pipelines, fitted on the
  training years only.
* **Stop, don't guess.** Unexpected values (unknown status, invalid sub-grade, FICO outside 300-850,
  zero income, new rate flags, changed sample sizes) stop the run with a message.

## Pinned software

See `requirements.txt`: pandas 2.3.3, numpy 2.3.5, pyarrow 22.0.0, scikit-learn 1.9.1, kaggle 2.2.5, matplotlib 3.10.8.
Only these direct dependencies are pinned; their own dependencies are whatever pip resolves.

What is actually recorded about past runs (nothing more is known):
* the original runs of 24 Nov 2026: Python 3.13.18 on Linux x86-64 with pandas 2.3.3, numpy 2.3.5,
  pyarrow 22.0.0 and scikit-learn 1.9.1 (`outputs/step5/step5_run_summary.json`; scikit-learn also in the
  one-time logs);
* the step 15 reruns: Python 3.12 and 3.13 on a 4-core cloud machine (`docs/step15_report.md`,
  `outputs/step15/`); memory and full package lists were not recorded;
* Lucas's Colab runs of 25-26 Nov (Colab's Python 3.12; no package list kept).

From 27 Nov every `reproduce_all` run writes the Python version, platform and installed versions of the
pinned packages to `outputs/reproduce/run_record.json`.

**Memory.** The paired bootstrap keeps a 1,000 x n weight matrix in 64-bit floats. For 2015
(n = 282,787) that is 2.1 GiB, and the weighted copy made inside each calculation is another 2.1 GiB, so
step 11 needs roughly 4.2 GiB for these arrays alone (about 2.5 GiB for 2014). Plan for at least 8 GB of
RAM; a standard Colab session (about 12 GB) is enough. Run time is about 8 minutes on a 4-core machine.

## GenAI disclosure

Two generative AI tools were used. ChatGPT (OpenAI) was used as an adviser in the design phase, to
discuss and review the research design. Claude (Anthropic) was used as an adviser for specification
review, as a coding assistant that wrote and ran this code under the author's direction, and to draft
the manuscript (handed in separately) from the signed-off results. Every methodological decision was made and approved by the
author; the decision records are in `decision_records/` and the control document's change log.

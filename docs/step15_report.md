# Step 15 report: reproducibility audit

Date: 25 Nov 2026. New code: `src/reproduce_all.py` and a `--reproduce` mode in steps 9 and 11.
Nothing in the method or the results changed (change-log row of 25 Nov in the control document).

## The answer in one paragraph

**A stranger can rerun the whole project from the raw file and get exactly our results.** I made a
fresh copy of the repository and a clean install of the pinned package versions. Then I ran everything
from the raw Kaggle file with one command, `python -m src.reproduce_all`. It took 7.2 minutes, and
**every committed table and figure came out byte-identical**, including the 2014 test and the 2015
robustness results. The same check passed on Python 3.12, the version Colab uses, and on Python 3.13.

## 1. What was tested

| Check | Result |
|---|---|
| Fresh clone of the repository, new empty Python environment | done |
| `pip install -r requirements.txt` installs exactly the pinned versions | yes: pandas 2.3.3, numpy 2.3.5, pyarrow 22.0.0, scikit-learn 1.9.1, matplotlib 3.10.8, kaggle 2.2.5 |
| Raw file fingerprint checked before and after | unchanged |
| Clean data files (step 5) | byte-identical to the originals, including the sealed 2014-2015 file |
| Steps 6, 7, 10, 12, 13: every table and figure | byte-identical |
| 2014 test (step 9), reproduced | all 6 result tables identical |
| 2015 robustness (step 11), reproduced | all 7 result tables identical |
| Final manuscript tables (`docs/final_tables.md`) | identical |
| Python 3.13 / Python 3.12 | REPRODUCED in both (logs: `outputs/step15/`) |
| Total run time (4-core cloud machine) | 7.2 minutes (3.13), 8.0 minutes (3.12) |

The only files that differ are the five run-summary files that record the date, run time and machine of
each run. They are meant to change. The rerun's reliability plots for 2014 and 2015 differ only in their
title ("reproduction" instead of "test"), and they go to a separate folder.

## 2. What I fixed

1. **The one-time tests could not be rerun by a marker.** That was correct for us: `--unseal` refuses
   to open 2014 or 2015 a second time. But a marker must be able to check them. The new `--reproduce`
   mode runs **the same frozen code**. It writes to a separate, git-ignored folder and stops with an
   error if any result differs from the committed one. The one-time logs stay untouched.
2. **One command for everything** (control document F.14): `python -m src.reproduce_all`. It runs all
   steps in order and finishes by asking git whether any committed result changed. It prints
   `REPRODUCED` or lists what changed.
3. **The Colab notebook now covers every step**, not just steps 5 to 7. Its last cell runs the same
   check.
4. **README:** new run instructions, the tested Python versions, the full GenAI disclosure (ChatGPT and
   Claude), and a stale status row removed.
5. **Manuscript:** two sentences now name the single command. These are the opening line and Section
   3.7. The main text is about 2,875 words. The PDF and Word files are rebuilt.

## 3. What I could not test here, and what you need to do

* **Downloading the raw file from Kaggle.** This environment cannot reach Kaggle, so the runs used the
  identical file (same SHA-256 fingerprint). The download code is unchanged since Gate 1.
* **Running the notebook inside Colab itself.** I tested the same commands on Colab's Python version,
  3.12, but not in a Colab session. **Please do one "Run all" in Colab before you submit.** It takes
  about 10 to 20 minutes.
* **Open decision: the repository is private.** A marker cannot open it as it is. Before submission,
  either make it public, or add the marker as a collaborator. The raw data is not in the repository, so
  making it public shares only code and summary tables.

## What I need from you

**Sign off step 15**, and decide on the repository access (public, or add the marker). Next on your
roadmap: step 16 (defence preparation).

## 4. Colab run by Lucas (25 Nov) and the fix

Lucas ran the notebook in Colab. **Every table and number reproduced**: the 2014 and 2015 checks
passed, and no CSV file differed. Only 9 figures (PNG) differed byte for byte. The cause was the font,
not the data. The machine that drew the original figures has a system-wide matplotlib settings file
that selects the "Inter" font, and Colab uses matplotlib's own DejaVu Sans. Same numbers, different
pixels.

The fix is `src/plotstyle.py`. It resets matplotlib to its built-in defaults, so every computer draws
with DejaVu Sans, which ships inside matplotlib. The 9 figures were redrawn and committed. I checked a
fresh copy with that settings file removed, as on Colab: `REPRODUCED`, byte-identical. The two
reliability figures written by the one-time 2014 and 2015 runs (`outputs/step9/`, `outputs/step11/`)
stay as logged. The manuscript uses the redrawn Figure 1. No number changed.

**Colab rerun by Lucas, 26 Nov 2026:** after the fix, the notebook's final check printed
`REPRODUCED: every committed table and figure is identical`.

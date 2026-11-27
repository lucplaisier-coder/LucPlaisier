# Clarification of earlier records (added 27 Nov 2026)

This note is **new**. It was written on 27 November 2026, after an external review of the
submission snapshot (PR 14, commit `e14ed82`). It does not change, replace or backdate any earlier
record, approval or result. The earlier records below stay byte-for-byte as they were signed; this note
only explains how to read some of their wording. It records no new decision by the author.

## 1. "Sealed", "unopened", "untouched": what was and was not seen before the freeze

Records that use this wording (all unchanged):

* `docs/methodology_freeze_v1.md`, status line: "the 2014 test and 2015 robustness outcomes were
  still sealed and unopened";
* `decision_records/methodology_freeze_v1.json`: `"sealed_outcomes_opened_before_freeze": false`;
* `decision_records/gate2_feature_freeze.json`: `"test_or_robustness_outcomes_seen": false`;
* step reports 5-11 ("2014 and 2015 stay sealed") and the README.

How to read them. These statements are accurate about **the project code**: the clean 2014-2015
outcome file written by step 5 was fingerprinted and `load_clean.load_sealed_outcomes()` refused to open
it before the freeze record existed. They are too broad if read as "nobody had seen any 2014-2015
outcome". Three different events must be kept apart:

1. **Descriptive outcome auditing (23 Nov 2026, before any model existed).** The Gate 1 and Step 1
   data audits, run outside this repository, computed outcome summaries for every year 2010-2015,
   including 2014-2015 charge-off counts, recovery shares and the fund-all net cash gain (for example
   `step1_independent_review/my_fundall_descriptive.csv` in the project files: 7.74 cents for 2014,
   5.30 cents for 2015). No model score or selection existed then. The decision records state that
   inputs and settings were chosen on other grounds and all tuning used 2013; the inspected code and
   records show no tuning on 2014-2015 outcomes, but this note cannot certify the absence of every
   unrecorded look.
2. **The first recorded formal model evaluation (24 Nov 2026, after the freeze).** 2014 was scored
   by `src/step9_test_2014.py --unseal` at 2026-11-24T21:10:08Z (code commit `6491974`, log
   `outputs/step9/test_opened_log.json`) and 2015 by `src/step11_robustness.py --unseal` (code commit
   `4cf52af`, log `outputs/step11/robustness_2015_opened_log.json`). These are the first recorded
   model results for those years. The logs are written after a successful run, so they record the
   successful opening, not every attempt.
3. **Later deterministic reproduction runs (25-26 Nov 2026 and any later run).** `--reproduce` and
   `src.step12_interpretability` deliberately fit and score again with the frozen code and settings to
   check the results. They are verification events, not new tests, and they changed nothing. "Scored
   once" in the manuscript therefore means "evaluated once as the test", not "never scored again".

## 2. "Byte-identical" in the step 15 report

`docs/step15_report.md` (unchanged; signed off 26 Nov 2026) says every committed table came out
byte-identical, including the 2014 and 2015 results. For the files rewritten in place (steps 5-7, 10,
12, 13) this was established by git's byte comparison. For the six 2014 and seven 2015 one-time tables
the check used was `compare_with_committed` in steps 9 and 11, which calls
`pandas.testing.assert_frame_equal` with its default tolerance (relative 1e-5) and only compares the
tables that the rerun happened to write. That shows the values agree within that tolerance; it does not
by itself prove byte identity or that every table was produced. From 27 Nov 2026,
`src/reproduce_all.py` checks every one of these tables byte for byte against the committed file, and
requires a complete, successful current run (see the README). No new reproduction run has been done
with the new check yet.

## 3. What was not changed

Raw data, sample rules, time splits, the nine inputs, preparation rules, the approved rate overlay,
model settings, tuning, seeds, the selection rule, the financial formulas, the bootstrap and every
reported number are unchanged. No model was fitted or scored to write this note.

## 4. The old notebook check

Until 27 Nov 2026 the notebook ran each step with `!python`, which does not stop on an error, and its
last cell (`reproduce_all --check-only`) only asked git whether committed files had changed. It could
therefore print `REPRODUCED` even if a stage had failed and left the committed files untouched. The
two command-line reruns in `outputs/step15/` used `subprocess.run(..., check=True)` and their logs show
every stage finishing, so they are not affected. For the Colab runs the notebook kept no per-stage
record: the 25 Nov run did rewrite figures (git reported 9 changed PNGs), which shows its stages
executed, but the 26 Nov `REPRODUCED` message by itself does not prove that every cell succeeded. This
casts no doubt on any reported number; it limits what that message can certify. The new notebook stops at the first
failed command and ends with `reproduce_all --verify`.

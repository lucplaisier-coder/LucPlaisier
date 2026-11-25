"""
reproduce_all.py: ROADMAP STEP 15, one command that reruns the whole project.

    python -m src.reproduce_all              # rerun everything, then check
    python -m src.reproduce_all --check-only # only the check (e.g. after the notebook)

Plain-language summary
----------------------
Runs every step in order, from the raw file to the final tables, on the
frozen settings. Nothing is chosen. The one-time 2014 and 2015 runs are NOT
repeated as new tests: their reproduction mode reruns the same frozen code and
checks the results equal the committed ones (the one-time logs stay as they are).

At the end it asks git whether any committed result file changed. If the
rerun is faithful, every table and figure is byte-identical to the committed
version; the only allowed differences are run times and timestamps in the
run-summary files listed in VOLATILE.
"""

import subprocess
import sys
import time

from . import config

STEPS = [
    ["src.step5_build_clean_dataset"],
    ["src.step6_baselines"],
    ["src.step7_histgb"],
    ["src.step9_test_2014", "--dry-run"],
    ["src.step9_test_2014", "--reproduce"],
    ["src.step10_financial_evaluation"],
    ["src.step11_robustness", "--dry-run"],
    ["src.step11_robustness", "--reproduce"],
    ["src.step12_interpretability"],
    ["src.step13_final_outputs"],
]

# Files that record when/how long a run took, so they differ on every run by design.
VOLATILE = {
    "outputs/step5/step5_run_summary.json",
    "outputs/step6/step6_run_summary.json",
    "outputs/step7/step7_run_summary.json",
    "outputs/step9_dryrun_2013/dryrun_summary.json",
    "outputs/step11_dryrun_2014/dryrun_summary.json",
}


def check_committed_outputs():
    """Ask git whether any committed result file differs from what the rerun wrote."""
    changed = subprocess.run(["git", "status", "--porcelain", "--", "outputs", "docs/final_tables.md"],
                             capture_output=True, text=True, cwd=config.REPO_ROOT, check=True).stdout.split("\n")
    changed = [line[3:] for line in changed if line.strip()]
    changed = [c for c in changed if c not in VOLATILE and not c.startswith("outputs/reproduce")]
    if changed:
        tables = [c for c in changed if not c.endswith(".png")]
        print("NOT REPRODUCED. These committed result files changed:\n  " + "\n  ".join(changed))
        if not tables:
            print("Only figures (.png) differ; every table and number is identical. This points to a "
                  "drawing difference (fonts or matplotlib build), not a different result.")
        sys.exit(1)
    print("REPRODUCED: every committed table and figure is identical "
          "(only run times/timestamps in run-summary files may differ).")


def main(check_only: bool = False):
    t0 = time.time()
    if not check_only:
        for step in STEPS:
            print(f"\n===== python -m {' '.join(step)} =====", flush=True)
            t = time.time()
            subprocess.run([sys.executable, "-m", *step], check=True, cwd=config.REPO_ROOT)
            print(f"----- done in {time.time() - t:.0f} s", flush=True)
        print(f"\nTotal run time: {(time.time() - t0) / 60:.1f} minutes")
    check_committed_outputs()


if __name__ == "__main__":
    main(check_only="--check-only" in sys.argv)

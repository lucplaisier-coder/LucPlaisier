"""
reproduce_all.py: ROADMAP STEP 15, rerun the whole project and verify it.

    python -m src.reproduce_all                          # rerun every stage, then verify
    python -m src.reproduce_all --expect-commit <SHA>    # same, and refuse any other code revision
    python -m src.reproduce_all --stage <name>           # run ONE stage (used by the Colab notebook)
    python -m src.reproduce_all --verify                 # verify a run that the --stage calls completed
    python -m src.reproduce_all --list-stages

Plain-language summary
----------------------
Runs every stage in order, from the raw file to the final tables, on the frozen
settings. Nothing is chosen. The one-time 2014 and 2015 runs are NOT repeated as
new tests: their --reproduce mode reruns the same frozen code and writes to the
git-ignored folder outputs/reproduce/.

"REPRODUCED" is printed only when ALL of these hold (27 Nov 2026 hardening):

 1. Every stage of THIS run finished with exit code 0, in order, and the run
    record (outputs/reproduce/run_record.json) says so. A stage that fails stops
    the run; a later --stage or --verify call refuses to continue.
 2. The code revision did not change during the run, the tracked code files are
    unmodified, and, if --expect-commit was given, HEAD is exactly that commit.
 3. Every file in the expected-artifact inventory (reproduction/expected_artifacts.json)
    is accounted for, and the inventory covers every committed result file:
      * regenerated  - rewritten by this run AND byte-identical (SHA-256) to the committed file;
      * reproduce_copy - the one-time 2014/2015 tables: the copy rewritten by this run
                       in outputs/reproduce/stepN/ is byte-identical to the committed table;
      * historical   - one-time logs and figures that no stage rewrites: must be unchanged;
      * volatile     - run-summary JSON files: rewritten by this run and identical except
                       for the listed keys that record when/where/how long a run took.
    There is no numerical tolerance anywhere: every comparison is on exact bytes, or on
    exact JSON values for the non-volatile keys.

Unchanged committed files alone can therefore never produce "REPRODUCED".

This module only orchestrates and compares. It fits nothing itself; the stages it
launches are the frozen analysis scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# The stages, in order. name -> python -m arguments
# ---------------------------------------------------------------------------
STAGES = [
    ("step5", ["src.step5_build_clean_dataset"]),
    ("step6", ["src.step6_baselines"]),
    ("step7", ["src.step7_histgb"]),
    ("step9_dryrun", ["src.step9_test_2014", "--dry-run"]),
    ("step9_reproduce", ["src.step9_test_2014", "--reproduce"]),
    ("step10", ["src.step10_financial_evaluation"]),
    ("step11_dryrun", ["src.step11_robustness", "--dry-run"]),
    ("step11_reproduce", ["src.step11_robustness", "--reproduce"]),
    ("step12", ["src.step12_interpretability"]),
    ("step13", ["src.step13_final_outputs"]),
    ("presentation", ["src.present_figures"]),   # redraws Figure 3 from saved CSV only
]

INVENTORY_PATH = "reproduction/expected_artifacts.json"
RUN_DIR = "outputs/reproduce"
RUN_RECORD = "outputs/reproduce/run_record.json"
# Result files the inventory must cover (everything committed under these paths).
RESULT_PATHS = ["outputs", "docs/final_tables.md"]

# Files that no stage rewrites. They stay exactly as committed.
HISTORICAL_PREFIXES = ("outputs/step9/", "outputs/step11/", "outputs/step15/")
# The one-time result tables: the reproduction writes a copy here instead.
REPRODUCE_COPY_DIRS = {"outputs/step9/": "outputs/reproduce/step9/",
                       "outputs/step11/": "outputs/reproduce/step11/"}
# Run summaries: these keys record time, duration, Python build or machine and differ by design.
# All other keys (sample sizes, clean-data SHA-256, package versions, settings) must match exactly.
VOLATILE_KEYS = {
    "outputs/step5/step5_run_summary.json": ["run_utc", "runtime_seconds", "python", "platform"],
    "outputs/step6/step6_run_summary.json": ["run_utc", "runtime_seconds"],
    "outputs/step7/step7_run_summary.json": ["run_utc", "runtime_seconds"],
    "outputs/step9_dryrun_2013/dryrun_summary.json": ["runtime_seconds"],
    "outputs/step11_dryrun_2014/dryrun_summary.json": ["runtime_seconds"],
}
# Packages whose installed versions are written into the run record.
RECORDED_PACKAGES = ["pandas", "numpy", "pyarrow", "scikit-learn", "matplotlib", "kaggle"]


class ReproductionError(RuntimeError):
    """Raised whenever a check fails. The message says what to do."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def head_commit(root: Path) -> str:
    return git(root, "rev-parse", "HEAD").strip()


def modified_code_files(root: Path) -> list[str]:
    """Tracked files outside the result paths that differ from HEAD (the code must be the commit)."""
    out = git(root, "status", "--porcelain", "--untracked-files=no", "--", ".",
              *[f":(exclude){p}" for p in RESULT_PATHS])
    return [line[3:] for line in out.splitlines() if line.strip()]


def tracked_result_files(root: Path) -> list[str]:
    return sorted(git(root, "ls-files", "--", *RESULT_PATHS).split())


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_record() -> dict:
    from importlib import metadata
    pkgs = {}
    for p in RECORDED_PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    return {"python_executable": sys.executable, "python": platform.python_version(),
            "platform": platform.platform(), "packages": pkgs}


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def classify(path: str) -> dict:
    if path in VOLATILE_KEYS:
        return {"class": "volatile", "ignored_keys": VOLATILE_KEYS[path]}
    for src_dir, copy_dir in REPRODUCE_COPY_DIRS.items():
        if path.startswith(src_dir) and path.endswith(".csv"):
            return {"class": "reproduce_copy", "copy": copy_dir + path[len(src_dir):]}
    if path.startswith(HISTORICAL_PREFIXES):
        return {"class": "historical"}
    return {"class": "regenerated"}


def build_inventory(root: Path) -> dict:
    """Inventory of every committed result file, from the files exactly as committed at HEAD."""
    dirty = git(root, "status", "--porcelain", "--", *RESULT_PATHS).strip()
    if dirty:
        raise ReproductionError("Result files differ from HEAD; build the inventory from a clean checkout.")
    files = {}
    for p in tracked_result_files(root):
        entry = classify(p)
        entry["sha256"] = sha256(root / p)
        if entry["class"] == "volatile":
            entry["expected_json"] = json.loads((root / p).read_text())
        files[p] = entry
    return {"description": "Expected artifacts of a full reproduction (src/reproduce_all.py). "
                           "SHA-256 values are those of the committed result files.",
            "stages": [name for name, _ in STAGES], "files": files}


def load_inventory(root: Path) -> dict:
    path = root / INVENTORY_PATH
    if not path.exists():
        raise ReproductionError(f"Expected-artifact inventory missing: {INVENTORY_PATH}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------
def read_record(root: Path) -> dict | None:
    p = root / RUN_RECORD
    return json.loads(p.read_text()) if p.exists() else None


def write_record(root: Path, rec: dict) -> None:
    p = root / RUN_RECORD
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2))


def check_revision(root: Path, expect_commit: str | None, rec: dict | None = None) -> str:
    head = head_commit(root)
    if expect_commit and head != expect_commit:
        raise ReproductionError(f"Wrong code revision: HEAD is {head}, expected {expect_commit}.")
    if rec is not None and head != rec["commit"]:
        raise ReproductionError(f"Code revision changed during the run: started at {rec['commit']}, now {head}.")
    changed = modified_code_files(root)
    if changed:
        raise ReproductionError("Tracked code files differ from the commit:\n  " + "\n  ".join(changed))
    return head


def begin_run(root: Path, stages: list, expect_commit: str | None) -> dict:
    head = check_revision(root, expect_commit)
    run_dir = root / RUN_DIR
    # outputs/reproduce/ is git-ignored scratch: clear it so no file from an earlier run can count.
    if run_dir.exists():
        for p in sorted(run_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() or p.is_symlink() else p.rmdir()
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / ".run_start"
    marker.write_text(now_utc())
    rec = {"run_id": uuid.uuid4().hex, "commit": head, "expected_commit": expect_commit,
           "started_utc": now_utc(), "start_mtime": marker.stat().st_mtime,
           "environment": environment_record(), "planned_stages": [n for n, _ in stages],
           "stages": [], "status": "running"}
    write_record(root, rec)
    return rec


def run_stage(root: Path, stages: list, name: str, expect_commit: str | None) -> dict:
    names = [n for n, _ in stages]
    if name not in names:
        raise ReproductionError(f"Unknown stage {name!r}. Stages: {', '.join(names)}")
    i = names.index(name)
    if i == 0:
        rec = begin_run(root, stages, expect_commit)
    else:
        rec = read_record(root)
        if rec is None:
            raise ReproductionError(f"No run in progress. Start with stage {names[0]!r}.")
        if rec["status"] != "running":
            raise ReproductionError(f"This run is {rec['status']}; start again with stage {names[0]!r}.")
        done = [s["name"] for s in rec["stages"] if s["returncode"] == 0]
        if done != names[:i]:
            raise ReproductionError(f"Stage {name!r} needs {names[:i]} done first, in order; done: {done}.")
        check_revision(root, expect_commit or rec["expected_commit"], rec)
    args = dict(stages)[name]
    print(f"\n===== [{name}] python -m {' '.join(args)} =====", flush=True)
    t0 = time.time()
    started = now_utc()
    rc = subprocess.run([sys.executable, "-m", *args], cwd=root).returncode
    rec["stages"].append({"name": name, "command": ["python", "-m", *args], "returncode": rc,
                          "started_utc": started, "seconds": round(time.time() - t0, 1)})
    if rc != 0:
        rec["status"] = f"failed at {name}"
        write_record(root, rec)
        raise ReproductionError(f"Stage {name!r} failed (exit code {rc}). The run is stopped; "
                                f"nothing after it ran. Fix the cause and start again from {names[0]!r}.")
    if i == len(names) - 1:
        rec["status"] = "stages complete"
    write_record(root, rec)
    print(f"----- {name} done in {time.time() - t0:.0f} s", flush=True)
    return rec


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(root: Path, stages: list, expect_commit: str | None = None) -> list[str]:
    """Return a list of problems (empty = reproduced). Never trusts unchanged files alone."""
    problems: list[str] = []
    rec = read_record(root)
    if rec is None:
        return ["No run record: no reproduction run was executed in this checkout."]
    names = [n for n, _ in stages]
    ok = [s["name"] for s in rec["stages"] if s["returncode"] == 0]
    if rec["status"] != "stages complete" or ok != names:
        problems.append(f"The run is not complete and successful (status: {rec['status']}; "
                        f"succeeded: {ok}; required: {names}).")
    try:
        check_revision(root, expect_commit or rec.get("expected_commit"), rec)
    except ReproductionError as e:
        problems.append(str(e))
    try:
        inv = load_inventory(root)
    except ReproductionError as e:
        return problems + [str(e)]
    start = rec["start_mtime"]
    files = inv["files"]

    tracked = set(tracked_result_files(root))
    for p in sorted(tracked - set(files)):
        problems.append(f"Committed result file not in the inventory: {p}")
    for p in sorted(set(files) - tracked):
        problems.append(f"Inventory file is not a committed result file: {p}")

    for p, e in sorted(files.items()):
        f = root / p
        if not f.exists():
            problems.append(f"Missing artifact: {p}")
            continue
        cls = e["class"]
        if cls == "historical":
            if sha256(f) != e["sha256"]:
                problems.append(f"Historical file changed: {p}")
        elif cls == "regenerated":
            if f.stat().st_mtime < start:
                problems.append(f"Not rewritten by this run: {p}")
            elif sha256(f) != e["sha256"]:
                problems.append(f"Differs from the committed version: {p}")
        elif cls == "reproduce_copy":
            if sha256(f) != e["sha256"]:
                problems.append(f"Historical one-time table changed: {p}")
            c = root / e["copy"]
            if not c.exists():
                problems.append(f"Missing reproduced table: {e['copy']}")
            elif c.stat().st_mtime < start:
                problems.append(f"Reproduced table not written by this run: {e['copy']}")
            elif sha256(c) != e["sha256"]:
                problems.append(f"Reproduced table differs from the committed one-time result: {e['copy']} vs {p}")
        elif cls == "volatile":
            if f.stat().st_mtime < start:
                problems.append(f"Not rewritten by this run: {p}")
            else:
                new, old = json.loads(f.read_text()), dict(e["expected_json"])
                for k in e["ignored_keys"]:
                    new.pop(k, None)
                    old.pop(k, None)
                if new != old:
                    diff = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
                    problems.append(f"Differs from the committed version (keys {diff}): {p}")
        else:
            problems.append(f"Unknown inventory class {cls!r} for {p}")
    return problems


def report(root: Path, problems: list[str]) -> int:
    rec = read_record(root) or {}
    if problems:
        print("NOT REPRODUCED.\n  " + "\n  ".join(problems))
        return 1
    print(f"REPRODUCED at commit {rec['commit']}: every stage of this run succeeded, and every one of the "
          f"expected artifacts is present and byte-identical to the committed version "
          f"(run summaries: identical apart from their time/machine fields).")
    return 0


# ---------------------------------------------------------------------------
def main(argv=None, root: Path = REPO_ROOT, stages: list = STAGES) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stage", help="run one stage (stages must be run in order)")
    g.add_argument("--verify", action="store_true", help="verify a completed run")
    g.add_argument("--check-only", action="store_true", help="old name for --verify")
    g.add_argument("--list-stages", action="store_true")
    g.add_argument("--write-inventory", action="store_true",
                   help="maintenance: rebuild the inventory from the committed result files")
    ap.add_argument("--expect-commit", default=None, help="full commit SHA the code must be at")
    a = ap.parse_args(argv)
    try:
        if a.list_stages:
            for n, args in stages:
                print(f"{n:18s} python -m {' '.join(args)}")
            return 0
        if a.write_inventory:
            inv = build_inventory(root)
            (root / INVENTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
            (root / INVENTORY_PATH).write_text(json.dumps(inv, indent=1, sort_keys=True) + "\n")
            print(f"Wrote {INVENTORY_PATH} with {len(inv['files'])} files.")
            return 0
        if a.stage:
            run_stage(root, stages, a.stage, a.expect_commit)
            return 0
        if a.verify or a.check_only:
            return report(root, verify(root, stages, a.expect_commit))
        t0 = time.time()
        for name, _ in stages:
            run_stage(root, stages, name, a.expect_commit)
        print(f"\nTotal run time: {(time.time() - t0) / 60:.1f} minutes")
        return report(root, verify(root, stages, a.expect_commit))
    except ReproductionError as e:
        print(f"NOT REPRODUCED. {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

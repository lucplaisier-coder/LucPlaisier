"""
Lightweight tests of the reproduction checker (src/reproduce_all.py) and the
presentation-only Figure 3 redraw (src/present_figures.py).

    python -m unittest discover -s tests -v

Everything here runs on SYNTHETIC fixtures in a temporary git repository: tiny
fake "stages" that write a few small files. No analysis stage is launched, no raw
or clean data is read, and no model is fitted, scored or ranked. Passing these
tests says the checker behaves correctly; it is NOT a reproduction of the project.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import reproduce_all as ra  # noqa: E402

# Fake stages: each writes deterministic files, like the real stages do.
FAKE_A = """
from pathlib import Path
d = Path("outputs/stepA"); d.mkdir(parents=True, exist_ok=True)
(d / "table.csv").write_text("x,y\\n1,0.123456\\n")
import json, time
(d / "run_summary.json").write_text(json.dumps({"rows": 1, "runtime_seconds": time.time()}))
"""
FAKE_B = """
from pathlib import Path
import os
if os.environ.get("FAKE_FAIL") == "B":
    raise SystemExit(3)
d = Path("outputs/stepB"); d.mkdir(parents=True, exist_ok=True)
(d / "figure.txt").write_text("drawn\\n")
r = Path("outputs/reproduce/step9"); r.mkdir(parents=True, exist_ok=True)
value = os.environ.get("FAKE_VALUE", "0.500000")
(r / "result.csv").write_text(f"metric,value\\nm,{value}\\n")
if os.environ.get("FAKE_SKIP") != "B":
    (Path("docs") / "final_tables.md").write_text("final\\n")
"""
STAGES = [("a", ["fakes.stage_a"]), ("b", ["fakes.stage_b"])]


def sh(root, *args):
    return subprocess.run(list(args), cwd=root, check=True, capture_output=True, text=True).stdout


class Fixture:
    """A temporary git repository shaped like the project, with committed results."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp()
        r = self.root = Path(self.tmp)
        (r / "fakes").mkdir()
        (r / "fakes" / "__init__.py").write_text("")
        (r / "fakes" / "stage_a.py").write_text(textwrap.dedent(FAKE_A))
        (r / "fakes" / "stage_b.py").write_text(textwrap.dedent(FAKE_B))
        (r / ".gitignore").write_text("outputs/reproduce/\n__pycache__/\n")
        (r / "docs").mkdir()
        # committed results = what a correct run produces
        self.run_fake("fakes.stage_a")
        self.run_fake("fakes.stage_b")
        (r / "outputs" / "step9").mkdir(parents=True)
        shutil.copy(r / "outputs/reproduce/step9/result.csv", r / "outputs/step9/result.csv")
        (r / "outputs/step9/test_opened_log.json").write_text('{"opened": "once"}')
        shutil.rmtree(r / "outputs" / "reproduce")
        sh(r, "git", "init", "-q")
        sh(r, "git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
        sh(r, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "results")
        # the inventory, built from the committed files, then committed
        self.patch()
        inv = ra.build_inventory(r)
        (r / "reproduction").mkdir()
        (r / ra.INVENTORY_PATH).write_text(json.dumps(inv))
        sh(r, "git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
        sh(r, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "inventory")
        self.head = sh(r, "git", "rev-parse", "HEAD").strip()
        self.inv = inv
        time.sleep(0.05)

    def run_fake(self, module):
        subprocess.run([sys.executable, "-m", module], cwd=self.root, check=True)

    def patch(self):
        ra.VOLATILE_KEYS = {"outputs/stepA/run_summary.json": ["runtime_seconds"]}

    def main(self, *argv):
        self.patch()
        return ra.main(list(argv), root=self.root, stages=STAGES)

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class ReproduceAllTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(ra.VOLATILE_KEYS)
        self.f = Fixture()
        for k in ("FAKE_FAIL", "FAKE_VALUE", "FAKE_SKIP"):
            os.environ.pop(k, None)

    def tearDown(self):
        ra.VOLATILE_KEYS = self.saved
        self.f.close()
        for k in ("FAKE_FAIL", "FAKE_VALUE", "FAKE_SKIP"):
            os.environ.pop(k, None)

    def test_inventory_classes(self):
        cls = {p: e["class"] for p, e in self.f.inv["files"].items()}
        self.assertEqual(cls["outputs/stepA/table.csv"], "regenerated")
        self.assertEqual(cls["outputs/stepA/run_summary.json"], "volatile")
        self.assertEqual(cls["outputs/step9/result.csv"], "reproduce_copy")
        self.assertEqual(cls["outputs/step9/test_opened_log.json"], "historical")
        self.assertEqual(cls["docs/final_tables.md"], "regenerated")

    def test_full_run_passes(self):
        self.assertEqual(self.f.main("--expect-commit", self.f.head), 0)

    def test_unchanged_committed_outputs_without_a_run_fail(self):
        # The old --check-only printed REPRODUCED here. Now there is no run record: fail.
        self.assertEqual(self.f.main("--verify"), 1)
        self.assertEqual(self.f.main("--check-only"), 1)

    def test_failed_stage_stops_the_run(self):
        os.environ["FAKE_FAIL"] = "B"
        self.assertEqual(self.f.main(), 1)
        rec = ra.read_record(self.f.root)
        self.assertEqual(rec["status"], "failed at b")
        self.assertEqual(self.f.main("--verify"), 1)

    def test_stages_must_run_in_order_and_after_a_failure_restart(self):
        self.assertEqual(self.f.main("--stage", "b"), 1)          # no run started
        os.environ["FAKE_FAIL"] = "B"
        self.assertEqual(self.f.main("--stage", "a"), 0)
        self.assertEqual(self.f.main("--stage", "b"), 1)
        os.environ.pop("FAKE_FAIL")
        self.assertEqual(self.f.main("--stage", "b"), 1)          # run is marked failed
        self.assertEqual(self.f.main("--verify"), 1)
        self.assertEqual(self.f.main("--stage", "a"), 0)          # fresh run
        self.assertEqual(self.f.main("--stage", "b"), 0)
        self.assertEqual(self.f.main("--verify"), 0)

    def test_wrong_version_rejected(self):
        self.assertEqual(self.f.main("--expect-commit", "0" * 40), 1)
        self.assertIsNone(ra.read_record(self.f.root))            # nothing ran

    def test_modified_code_rejected(self):
        with open(self.f.root / "fakes" / "stage_a.py", "a") as fh:
            fh.write("\n# edited\n")
        self.assertEqual(self.f.main(), 1)

    def test_commit_change_during_run_rejected(self):
        self.assertEqual(self.f.main("--stage", "a"), 0)
        sh(self.f.root, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
           "--allow-empty", "-m", "moved")
        self.assertEqual(self.f.main("--stage", "b"), 1)

    def test_missing_reproduced_table_detected(self):
        self.assertEqual(self.f.main("--stage", "a"), 0)
        self.assertEqual(self.f.main("--stage", "b"), 0)
        (self.f.root / "outputs/reproduce/step9/result.csv").unlink()
        problems = ra.verify(self.f.root, STAGES)
        self.assertTrue(any("Missing reproduced table" in p for p in problems), problems)

    def test_missing_committed_artifact_detected(self):
        self.assertEqual(self.f.main("--stage", "a"), 0)
        self.assertEqual(self.f.main("--stage", "b"), 0)
        (self.f.root / "outputs/stepB/figure.txt").unlink()
        problems = ra.verify(self.f.root, STAGES)
        self.assertTrue(any("Missing artifact: outputs/stepB/figure.txt" in p for p in problems), problems)

    def test_file_not_rewritten_by_this_run_detected(self):
        os.environ["FAKE_SKIP"] = "B"          # stage b "forgets" to rewrite docs/final_tables.md
        self.assertEqual(self.f.main(), 1)
        problems = ra.verify(self.f.root, STAGES)
        self.assertTrue(any("Not rewritten by this run: docs/final_tables.md" in p for p in problems), problems)

    def test_tiny_numeric_difference_is_not_tolerated(self):
        # 0.500001 vs 0.500000: pandas' default assert_frame_equal (rtol 1e-5) would accept this.
        import pandas as pd
        a = pd.DataFrame({"value": [0.500000]})
        b = pd.DataFrame({"value": [0.500001]})
        pd.testing.assert_frame_equal(a, b)          # the old stage-level check passes...
        os.environ["FAKE_VALUE"] = "0.500001"
        self.assertEqual(self.f.main(), 1)           # ...the byte-exact verification does not
        problems = ra.verify(self.f.root, STAGES)
        self.assertTrue(any("differs from the committed one-time result" in p for p in problems), problems)

    def test_volatile_file_only_ignores_listed_keys(self):
        self.assertEqual(self.f.main(), 0)            # runtime_seconds differs: allowed
        p = self.f.root / "outputs/stepA/run_summary.json"
        d = json.loads(p.read_text())
        d["rows"] = 2
        p.write_text(json.dumps(d))
        problems = ra.verify(self.f.root, STAGES)
        self.assertTrue(any("keys ['rows']" in p for p in problems), problems)

    def test_historical_file_change_detected(self):
        self.assertEqual(self.f.main(), 0)
        (self.f.root / "outputs/step9/test_opened_log.json").write_text('{"opened": "twice"}')
        problems = ra.verify(self.f.root, STAGES)
        self.assertTrue(any("Historical file changed" in p for p in problems), problems)

    def test_stale_files_from_an_earlier_run_do_not_count(self):
        self.assertEqual(self.f.main(), 0)
        os.environ["FAKE_FAIL"] = "B"
        self.assertEqual(self.f.main(), 1)
        self.assertFalse((self.f.root / "outputs/reproduce/step9/result.csv").exists())

    def test_uninventoried_committed_result_detected(self):
        self.assertEqual(self.f.main(), 0)
        (self.f.root / "outputs/stepA/extra.csv").write_text("new\n")
        sh(self.f.root, "git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
        sh(self.f.root, "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "extra")
        self.assertEqual(self.f.main(), 1)


class RealInventoryTests(unittest.TestCase):
    """Static checks of the committed inventory against the committed files (reads hashes only)."""

    def test_inventory_matches_committed_files(self):
        inv = ra.load_inventory(REPO)
        tracked = set(ra.tracked_result_files(REPO))
        self.assertEqual(set(inv["files"]), tracked)
        self.assertEqual(inv["stages"], [n for n, _ in ra.STAGES])
        for p, e in inv["files"].items():
            blob = subprocess.run(["git", "show", f"HEAD:{p}"], cwd=REPO, capture_output=True, check=True).stdout
            import hashlib
            self.assertEqual(hashlib.sha256(blob).hexdigest(), e["sha256"], p)

    def test_every_one_time_table_has_a_reproduced_copy(self):
        inv = ra.load_inventory(REPO)
        copies = {p for p, e in inv["files"].items() if e["class"] == "reproduce_copy"}
        self.assertEqual(len([p for p in copies if p.startswith("outputs/step9/")]), 6)
        self.assertEqual(len([p for p in copies if p.startswith("outputs/step11/")]), 7)


class Figure3RedrawTests(unittest.TestCase):
    """The presentation redraw plots exactly the saved CSV values, order and error bars."""

    def test_bars_equal_saved_data(self):
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src import present_figures
        captured = {}
        real_savefig = matplotlib.figure.Figure.savefig

        def fake_savefig(fig, *a, **k):
            captured["fig"] = fig
        matplotlib.figure.Figure.savefig = fake_savefig
        real_close = plt.close
        plt.close = lambda *a, **k: None
        try:
            present_figures.figure3()
        finally:
            matplotlib.figure.Figure.savefig = real_savefig
            plt.close = real_close
        ax = captured["fig"].axes[0]
        data = pd.read_csv(REPO / "outputs/final/figure3_data.csv").sort_values("mean_auc_drop")
        widths = [b.get_width() for b in ax.patches]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(widths, data["mean_auc_drop"].tolist())
        self.assertEqual(labels, data["label"].tolist())
        from matplotlib.container import ErrorbarContainer
        eb = next(c for c in ax.containers if isinstance(c, ErrorbarContainer))
        err = eb.lines[2][0].get_segments()
        half_widths = [(s[1][0] - s[0][0]) / 2 for s in err]
        for h, sd in zip(half_widths, data["sd_over_repeats"]):
            self.assertAlmostEqual(h, sd, places=12)
        plt.close(captured["fig"])


if __name__ == "__main__":
    unittest.main()

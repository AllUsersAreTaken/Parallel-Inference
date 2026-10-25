"""Regression tests for harness/analyze_mixed_within.py.

Stdlib only. Run:  python harness\\test_mixed_within.py -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_mixed_within as A  # noqa: E402
import analyze_modality_vote as M  # noqa: E402

HEADER = "id\tstream\tmode\tfinish\tgenerated\tttft_ms\tdone_ms\ttext"


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def stream_rows(pid, cot_a, cot_b, code):
    rows = []
    for s in range(24):
        if s < 8:
            mode, text = "cot", cot_a
        elif s < 16:
            mode, text = "cot", cot_b
        else:
            mode, text = "code", code
        rows.append(f"{pid}\t{s}\t{mode}\tanswer\t16\t1.0\t{100.0 + s}\t{esc(text)}")
    return rows


class TestFirstKAgree(unittest.TestCase):
    def test_reaches_k_in_done_order(self):
        entries = [("7", 30.0, 2), ("5", 10.0, 0), ("5", 20.0, 1),
                   ("5", 25.0, 4), ("7", 40.0, 3)]
        ans, reached, done = A.first_k_agree(entries, k=3)
        self.assertTrue(reached)
        self.assertEqual(M.canon(ans), "5")
        self.assertEqual(done, 25.0)

    def test_falls_back_when_k_not_reached(self):
        entries = [("5", 10.0, 0), ("7", 20.0, 1), ("7", 30.0, 2)]
        ans, reached, done = A.first_k_agree(entries, k=3)
        self.assertFalse(reached)
        self.assertEqual(M.canon(ans), "7")
        self.assertEqual(done, 30.0)

    def test_fallback_tie_break_uses_stream_index_not_finish_order(self):
        # Stream indices 0,1 say 5 and 2,3 say 7, but 7 finishes first. The
        # fallback tie-break must use the original stream index (5), not the
        # done_ms rank (which would pick 7).
        entries = [("5", 30.0, 0), ("5", 40.0, 1), ("7", 10.0, 2), ("7", 20.0, 3)]
        ans, reached, done = A.first_k_agree(entries, k=3)
        self.assertFalse(reached)
        self.assertEqual(M.canon(ans), "5")
        self.assertEqual(done, 40.0)


class TestBuildArms(unittest.TestCase):
    def test_second_half_swap(self):
        answers = {}
        for s in range(24):
            if s < 8:
                answers[s] = "5"
            elif s < 16:
                answers[s] = "7"
            else:
                answers[s] = "5"
        votes = A.build_arm_votes(answers)
        self.assertEqual(M.canon(votes["C8"]["answer"]), "5")
        self.assertFalse(votes["C8"]["tied"])
        self.assertTrue(votes["C16"]["tied"])
        self.assertEqual({M.canon(x) for x in votes["C16"]["top"]}, {"5", "7"})
        self.assertEqual(M.canon(votes["M16"]["answer"]), "5")
        self.assertFalse(votes["M16"]["tied"])


class TestAnalyzeRun(unittest.TestCase):
    def test_synthetic_end_to_end(self):
        rows = [HEADER]
        for pid in (1, 2):
            rows += stream_rows(
                pid,
                "The answer is: 5",
                "The answer is: 7",
                "```python\nprint(5)\n```",
            )
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False,
                                         encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(rows) + "\n")
            path = fh.name
        try:
            r = A.analyze_run("t", path, "", {1: "5", 2: "5"}, 8.0)
        finally:
            os.unlink(path)
        self.assertEqual(r["arm_metrics"]["C8"]["correct"], 2)
        self.assertEqual(r["arm_metrics"]["C8"]["tied"], 0)
        self.assertEqual(r["arm_metrics"]["C16"]["tied"], 2)
        self.assertEqual(r["arm_metrics"]["M16"]["point"], 100.0)
        self.assertEqual(r["arm_metrics"]["M16"]["tied"], 0)
        self.assertEqual(r["stream_stats"]["code_16_23"]["correct"], 16)
        self.assertEqual(r["oracle"]["union"][1], True)
        # C16 is tied 8-8 and the deterministic tie-break lands on gold, so its
        # point accuracy matches M16; the tie count is what differs.
        self.assertEqual(r["tests"]["M16_vs_C16"]["delta"], 0.0)
        self.assertEqual(r["arm_metrics"]["C16"]["tied"], 2)
        self.assertEqual(r["arm_metrics"]["M16"]["tied"], 0)


if __name__ == "__main__":
    unittest.main()

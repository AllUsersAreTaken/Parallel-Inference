"""Regression tests for harness/analyze_format_within.py.

Stdlib only. Run:  python harness\\test_format_within.py -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_format_within as A  # noqa: E402
import analyze_modality_vote as M  # noqa: E402

HEADER = "id\tstream\tmode\tfinish\tgenerated\tttft_ms\tdone_ms\ttext"


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def stream_rows(pid, wrapper_first, wrap_text, plain_text):
    """Build 16 streams; --swap-parity puts the wrapped half first on odd ids."""
    rows = []
    for s in range(16):
        first_half = s < 8
        wrapped = (first_half and wrapper_first) or (not first_half and not wrapper_first)
        mode = "cotw" if wrapped else "cot"
        text = wrap_text if wrapped else plain_text
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


class TestWrongAnswerAgreement(unittest.TestCase):
    def test_only_wrong_answers_count(self):
        by_id = {
            1: {
                0: {"mode": "cot", "text": "The answer is: 5", "generated": 1, "done_ms": 1.0},
                1: {"mode": "cot", "text": "The answer is: 7", "generated": 1, "done_ms": 2.0},
                2: {"mode": "cot", "text": "The answer is: 7", "generated": 1, "done_ms": 3.0},
            }
        }
        a = A.wrong_answer_agreement(by_id, [1], "cot", {1: "5"})
        self.assertEqual(a["pair_both"], 1)
        self.assertEqual(a["pair_agree"], 1)
        self.assertEqual(a["pair_frac"], 100.0)
        self.assertEqual(a["problems_with_two_wrong"], 1)


class TestParityLayout(unittest.TestCase):
    def test_alternating_layout(self):
        by_id = {
            1: {s: {"mode": "cotw" if s < 8 else "cot"} for s in range(16)},
            2: {s: {"mode": "cot" if s < 8 else "cotw"} for s in range(16)},
        }
        lay = A.parity_layout(by_id, [1, 2])
        self.assertTrue(lay["alternating"])
        self.assertEqual(lay["match"], 2)

    def test_single_layout_is_not_alternating(self):
        by_id = {
            1: {s: {"mode": "cot" if s < 8 else "cotw"} for s in range(16)},
            2: {s: {"mode": "cot" if s < 8 else "cotw"} for s in range(16)},
        }
        self.assertFalse(A.parity_layout(by_id, [1, 2])["alternating"])


class TestAnalyzeRun(unittest.TestCase):
    def test_synthetic_end_to_end(self):
        rows = [HEADER]
        # pid 1 (odd): wrapped half first, wrapper says 5 (gold), plain says 7.
        rows += stream_rows(1, True, "The answer is: 5", "The answer is: 7")
        # pid 2 (even): plain half first, plain says 7 (wrong), wrapper says 5.
        rows += stream_rows(2, False, "The answer is: 5", "The answer is: 7")
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False,
                                         encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(rows) + "\n")
            path = fh.name
        try:
            r = A.analyze_run("t", path, "", {1: "5", 2: "5"})
        finally:
            os.unlink(path)

        self.assertTrue(r["layouts"]["alternating"])
        self.assertEqual(r["met"]["cot"]["point"], 0.0)
        self.assertEqual(r["met"]["cotw"]["point"], 100.0)
        self.assertEqual(r["oracle"]["cotw"][1], True)
        self.assertEqual(r["support"]["cot"][1], 0)
        self.assertEqual(r["gold_absent_streams"]["cot"], 2)
        self.assertEqual(r["gold_absent_top"]["cot"], 2)
        self.assertEqual(r["fks"]["cotw"]["reached"], 2)
        self.assertEqual(r["fks"]["cotw"]["acc_reached"], 100.0)
        self.assertEqual(r["test"]["delta"], 100.0)
        self.assertEqual(r["test"]["b"], 2)
        self.assertEqual(r["test"]["c"], 0)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["leak"], (0, 16))

    def test_rejects_mixed_half(self):
        rows = [HEADER]
        rows += stream_rows(2, True, "The answer is: 5", "The answer is: 7")
        bad = []
        for line in rows[1:]:
            parts = line.split("\t")
            parts[2] = "cot"  # both halves the same mode
            bad.append("\t".join(parts))
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False,
                                         encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join([HEADER] + bad) + "\n")
            path = fh.name
        try:
            with self.assertRaises(SystemExit):
                A.analyze_run("t", path, "", {2: "5"})
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

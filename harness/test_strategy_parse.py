"""Regression tests for harness/analyze_strategy_vote.py (stdlib only).

Run:  python harness\\test_strategy_parse.py -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_strategy_vote as S  # noqa: E402

HEADER = S.HEADER


def row(i, stream, mode, done_ms, text, finish="answer", gen=10):
    return {"id": i, "stream": stream, "mode": mode, "finish": finish, "generated": gen,
            "ttft_ms": 0.0, "done_ms": done_ms, "text": text}


def write_tsv(path, rows):
    lines = ["\t".join(HEADER)]
    for r in rows:
        text = M_escape(r["text"])
        lines.append("\t".join(str(x) for x in
                     [r["id"], r["stream"], r["mode"], r["finish"], r["generated"],
                      r["ttft_ms"], r["done_ms"], text]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def M_escape(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


class TestFirstKAgree(unittest.TestCase):
    def _by_id(self, answers, dones):
        rows = [row(1, j, "cot", dones[j], f"The answer is: {a}") for j, a in enumerate(answers)]
        return {1: {j: rows[j] for j in range(len(rows))}}

    def test_third_agreeing_stream_in_time_order(self):
        # sorted by done: s3(1)=8, s4(2)=8, s5(3)=8 -> answer 8 reaches count 3 at done=3.
        by_id = self._by_id([7, 7, 7, 8, 8, 8, 8, 8], [5, 9, 7, 1, 2, 3, 4, 6])
        ans, reached, done = S.first_k_agree(by_id, 1, k=3)
        self.assertEqual(ans, "8")
        self.assertTrue(reached)
        self.assertEqual(done, 3)

    def test_fallback_when_never_reaches_k(self):
        by_id = self._by_id([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])
        ans, reached, done = S.first_k_agree(by_id, 1, k=3)
        self.assertFalse(reached)
        self.assertEqual(done, 8)
        self.assertIn(ans, {"1", "2", "3", "4", "5", "6", "7", "8"})

    def test_fallback_tie_break_uses_stream_index_not_finish_order(self):
        # Stream order puts 5 at indices 0,1 and 7 at 2,3, but finish order is
        # reversed. The fallback tie-break must prefer 5 (lowest stream-index
        # sum), not the first answer to finish (7).
        rows = [row(1, 0, "cot", 30, "The answer is: 5"),
                row(1, 1, "cot", 40, "The answer is: 5"),
                row(1, 2, "cot", 10, "The answer is: 7"),
                row(1, 3, "cot", 20, "The answer is: 7")]
        by_id = {1: {r["stream"]: r for r in rows}}
        ans, reached, done = S.first_k_agree(by_id, 1, k=3)
        self.assertFalse(reached)
        self.assertEqual(ans, "5")
        self.assertEqual(done, 40)

    def test_abstentions_excluded_from_agreement(self):
        rows = [row(1, 0, "cot", 1, "no number here at all"),
                row(1, 1, "cot", 2, "The answer is: 5"),
                row(1, 2, "cot", 3, "The answer is: 5"),
                row(1, 3, "cot", 4, "The answer is: 5")]
        by_id = {1: {j: rows[j] for j in range(4)}}
        ans, reached, done = S.first_k_agree(by_id, 1, k=3)
        self.assertTrue(reached)
        self.assertEqual(ans, "5")
        self.assertEqual(done, 4)


class TestAnalyzePair(unittest.TestCase):
    def _make(self, tmp):
        hom = []
        div = []
        modes = S.STRATEGIES
        per_problem_hom = {1: ["10"] * 8, 2: ["19"] * 8, 3: ["30"] * 4 + ["31"] * 4,
                           4: ["41"] * 8}
        per_problem_div = {1: ["10"] * 8,
                           2: ["20"] * 8,
                           3: ["30"] * 6 + ["31"] * 2,
                           4: ["40"] * 8}
        for pid in (1, 2, 3, 4):
            for j in range(8):
                hom.append(row(pid, j, "cot", 100 + j,
                               f"Reasoning\nThe answer is: {per_problem_hom[pid][j]}"))
                div.append(row(pid, j, modes[j], 200 + j,
                               f"Reasoning\nThe answer is: {per_problem_div[pid][j]}"))
        hp = os.path.join(tmp, "hom.tsv")
        dp = os.path.join(tmp, "div.tsv")
        write_tsv(hp, hom)
        write_tsv(dp, div)
        return hp, dp

    def test_metrics(self):
        gold_raw = {1: "10", 2: "20", 3: "30", 4: "40"}
        with tempfile.TemporaryDirectory() as tmp:
            hp, dp = self._make(tmp)
            r = S.analyze_pair("t", hp, dp, gold_raw)
        # HOM8: id1 right, id2 wrong, id3 tied 4/4 (point picks via tie-break),
        # id4 wrong -> point = 2; upper adds the tied id3 -> 2 as well.
        self.assertEqual(r["hom_met"]["correct"], 2)
        self.assertEqual(r["hom_met"]["tied"], 1)
        self.assertAlmostEqual(r["hom_met"]["upper"], 50.0)
        # DIV8: all four right -> 100.
        self.assertEqual(r["div_met"]["correct"], 4)
        self.assertEqual(r["div_met"]["point"], 100.0)
        self.assertEqual(sum(r["div_oracle"].values()), 4)
        self.assertEqual(sum(r["hom_oracle"].values()), 2)
        # distinct answers: hom id3 has 2; div id3 has 2.
        self.assertEqual(r["hom_distinct"][3], 2)
        self.assertEqual(r["div_distinct"][3], 2)
        # gold support
        self.assertEqual(r["hom_support"][2], 0)
        self.assertEqual(r["div_support"][2], 8)
        # delta and verdict: +50 pts, tieCI lower > 0 -> PASS
        self.assertAlmostEqual(r["mcnemar"]["delta"], 50.0)
        self.assertGreater(r["mcnemar"]["lo_tie"], 0)
        self.assertEqual(r["verdict"], "PASS")
        # per-mode accuracy: first 6 modes answer 30 on id3; units/check answer 31.
        for m in S.STRATEGIES[:6]:
            self.assertEqual(r["div_mode_acc"][m]["correct"], 4)
        for m in ("units", "check"):
            self.assertEqual(r["div_mode_acc"][m]["correct"], 3)

    def test_csv_written(self):
        gold_raw = {1: "10", 2: "20", 3: "30", 4: "40"}
        with tempfile.TemporaryDirectory() as tmp:
            hp, dp = self._make(tmp)
            r = S.analyze_pair("t", hp, dp, gold_raw)
            out = os.path.join(tmp, "out.csv")
            S.write_csv(r, out)
            with open(out, encoding="utf-8") as fh:
                lines = fh.read().strip().splitlines()
        self.assertEqual(len(lines), 5)  # header + 4 problems
        self.assertEqual(lines[0].split(",")[0], "id")
        self.assertEqual(len(lines[0].split(",")), 26 + 16)


class TestAnalyzeMulti(unittest.TestCase):
    def _make(self, tmp):
        gold_raw = {1: "10", 2: "20", 3: "30", 4: "40"}
        per_problem = {
            "ZS8": {1: ["99"] * 8, 2: ["20"] * 8, 3: ["30"] * 8, 4: ["40"] * 8},
            "FS8": {1: ["10"] * 8, 2: ["20"] * 8, 3: ["30"] * 8, 4: ["40"] * 8},
            "FD8": {1: ["10"] * 4 + ["99"] * 4, 2: ["20"] * 6 + ["21"] * 2,
                    3: ["30"] * 5 + ["31"] * 3, 4: ["40"] * 4 + ["99"] * 4},
        }
        specs = []
        for label in ("ZS8", "FS8", "FD8"):
            rows = []
            for pid in (1, 2, 3, 4):
                for j in range(8):
                    rows.append(row(pid, j, "cot", 100 + j,
                                    f"Reasoning\nThe answer is: {per_problem[label][pid][j]}"))
            path = os.path.join(tmp, f"{label}.tsv")
            write_tsv(path, rows)
            specs.append((label, path))
        return specs, gold_raw

    def test_multi_metrics_and_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs, gold_raw = self._make(tmp)
            r = S.analyze_multi("t", specs, gold_raw)
        self.assertEqual(len(r["pairs"]), 3)
        by = {a["label"]: a for a in r["arms"]}
        self.assertEqual(by["FS8"]["met"]["point"], 100.0)
        self.assertEqual(by["ZS8"]["met"]["point"], 75.0)
        self.assertEqual(by["FS8"]["met"]["tied"], 0)
        self.assertEqual(by["FD8"]["met"]["tied"], 2)
        self.assertEqual(sum(by["FD8"]["oracle"].values()), 4)
        # manipulation check: distinct answers lower within-arm agreement.
        self.assertGreater(r["manipulation"]["delta"], 50.0)
        self.assertLess(by["FD8"]["agreement"]["pair_frac"], by["FS8"]["agreement"]["pair_frac"])
        # first-K reached-only accuracy.
        fk = by["FD8"]["fks"]
        self.assertEqual(fk["reached"], 4)
        self.assertEqual(fk["correct_reached"], 4)
        self.assertAlmostEqual(fk["acc_reached"], 100.0)
        # FD8 == FS8 in point accuracy -> DEAD.
        self.assertEqual(r["verdict"], "DEAD")
        # direction lookup flips the reverse comparison.
        dfs = S.lookup_pair(r["pairs"], "FD8", "FS8")
        self.assertAlmostEqual(dfs["delta"], 0.0)

    def test_multi_csv_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs, gold_raw = self._make(tmp)
            r = S.analyze_multi("t", specs, gold_raw)
            out = os.path.join(tmp, "m.csv")
            S.write_multi_csv(r, out)
            with open(out, encoding="utf-8") as fh:
                lines = fh.read().strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(len(lines[0].split(",")), 2 + 3 * 10)

    def test_multi_rejects_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs, gold_raw = self._make(tmp)
            bad = os.path.join(tmp, "bad.tsv")
            write_tsv(bad, [row(9, 0, "cot", 1, "The answer is: 1")])
            with self.assertRaises(SystemExit):
                S.analyze_multi("t", [specs[0], ("X", bad)], gold_raw)


class TestFirstKReachedOnly(unittest.TestCase):
    def test_unreached_problem_excluded(self):
        # id1 reaches k=3 with 8; id2 never reaches k.
        rows = [row(1, 0, "cot", 1, "The answer is: 8"),
                row(1, 1, "cot", 2, "The answer is: 8"),
                row(1, 2, "cot", 3, "The answer is: 8"),
                row(2, 0, "cot", 1, "The answer is: 1"),
                row(2, 1, "cot", 2, "The answer is: 2"),
                row(2, 2, "cot", 3, "The answer is: 3")]
        by_id = {1: {r["stream"]: r for r in rows if r["id"] == 1},
                 2: {r["stream"]: r for r in rows if r["id"] == 2}}
        gold = {1: "8", 2: "3"}
        s = S.first_k_stats(by_id, [1, 2], gold)
        self.assertEqual(s["reached"], 1)
        self.assertEqual(s["correct_reached"], 1)
        self.assertAlmostEqual(s["acc_reached"], 100.0)
        self.assertEqual(s["correct"], 1)


if __name__ == "__main__":
    unittest.main()

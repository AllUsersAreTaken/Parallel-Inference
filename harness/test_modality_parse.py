"""Regression tests for harness/analyze_modality_vote.py parsing + tie stats.

Stdlib only. Run:  python harness\\test_modality_parse.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_modality_vote as M  # noqa: E402


class TestBoxedExtraction(unittest.TestCase):
    def test_simple_numeric(self):
        self.assertEqual(M.extract_cot("so \\boxed{55} done"), ("55", "boxed_numeric"))

    def test_nested_frac(self):
        self.assertEqual(M.extract_cot("we get \\boxed{\\frac{104}{4}}"), ("26", "boxed_frac"))

    def test_ratio_form(self):
        self.assertEqual(M.extract_cot("\\boxed{1/2}"), ("0.5", "boxed_frac"))

    def test_decimal(self):
        self.assertEqual(M.extract_cot("\\boxed{0.5}"), ("0.5", "boxed_numeric"))

    def test_money_commas(self):
        self.assertEqual(M.extract_cot("\\boxed{$1,200}"), ("1200", "boxed_numeric"))

    def test_arithmetic_expression(self):
        self.assertEqual(M.extract_cot("\\boxed{3*4+2}"), ("14", "boxed_expr"))

    def test_text_falls_back_to_last_number(self):
        v, path = M.extract_cot("he rode \\boxed{160 \\text{ miles}}")
        self.assertEqual(v, "160")
        self.assertEqual(path, "last_number")

    def test_symbolic_rejected(self):
        self.assertEqual(M.classify_boxed("20x"), (None, None))
        self.assertEqual(M.classify_boxed("L - 8"), (None, None))
        self.assertEqual(M.classify_boxed("n + 1"), (None, None))

    def test_multiple_commas_rejected(self):
        self.assertEqual(M.classify_boxed("80, 230"), (None, None))
        self.assertEqual(M.classify_boxed("3,4"), (None, None))
        self.assertEqual(M.extract_cot("\\boxed{80, 230}"), ("230", "last_number"))

    def test_incomplete_box_not_accepted(self):
        self.assertIsNone(M.find_balanced_boxed("\\boxed{55"))
        self.assertIsNone(M.find_balanced_boxed("\\boxed{\\frac{104}{4}"))
        self.assertEqual(M.extract_cot("\\boxed{55"), ("55", "last_number"))


class TestAnswerLineFallback(unittest.TestCase):
    def test_answer_line(self):
        self.assertEqual(M.extract_cot("reasoning here\nThe answer is: 42"), ("42", "answer_line"))

    def test_box_beats_answer_line(self):
        self.assertEqual(
            M.extract_cot("The answer is: 7\n\\boxed{12}"), ("12", "boxed_numeric")
        )

    def test_think_block_then_box(self):
        self.assertEqual(M.extract_cot("</think" + ">thinking...\n\\boxed{9}"), ("9", "boxed_numeric"))


class TestCodeCompliance(unittest.TestCase):
    def test_single_print_single_token(self):
        self.assertTrue(M.code_is_compliant("print(42)", "42\n"))

    def test_single_print_multi_token_constant(self):
        self.assertTrue(M.code_is_compliant("print(7)", "1\n7\n"))

    def test_multi_arg_print(self):
        self.assertFalse(M.code_is_compliant("print(1, 2)", "1 2\n"))

    def test_name_argument_with_extra_output(self):
        self.assertFalse(M.code_is_compliant("x = 1\nprint(x)", "0\n1\n"))

    def test_empty_print(self):
        self.assertFalse(M.code_is_compliant("print()", ""))

    def test_single_token_from_name(self):
        self.assertTrue(M.code_is_compliant("x = 5\nprint(x)", "5\n"))


class TestSafeArith(unittest.TestCase):
    def test_rejects_names(self):
        self.assertIsNone(M.safe_arith("x + 1"))
        self.assertIsNone(M.safe_arith("len([1])"))

    def test_division_by_zero(self):
        self.assertIsNone(M.safe_arith("1 / 0"))

    def test_precedence(self):
        self.assertEqual(M.safe_arith("2 + 3 * 4"), 14)


class TestTieMetrics(unittest.TestCase):
    def test_random_tie_expectation(self):
        ids = [1]
        gold = {1: "3"}
        arms = {1: {"answer": "3", "tied": True, "top": ["2", "3", "5"], "n_votes": 3}}
        m = M.tie_metrics(arms, ids, gold)
        self.assertEqual(m["correct"], 1)
        self.assertAlmostEqual(m["point"], 100.0)
        self.assertAlmostEqual(m["rand"], 100.0 / 3.0)
        self.assertAlmostEqual(m["upper"], 100.0)

    def test_untied_miss(self):
        ids = [1]
        gold = {1: "4"}
        arms = {1: {"answer": "3", "tied": False, "top": ["3"], "n_votes": 1}}
        m = M.tie_metrics(arms, ids, gold)
        self.assertEqual(m["correct"], 0)
        self.assertAlmostEqual(m["rand"], 0.0)
        self.assertAlmostEqual(m["upper"], 0.0)


class TestTieAwareBootstrap(unittest.TestCase):
    def test_identical_arms_zero_width(self):
        ids = [1, 2, 3, 4]
        gold = {i: "1" for i in ids}
        arms = {i: {"answer": "1", "tied": False, "top": ["1"], "n_votes": 1} for i in ids}
        lo, hi = M.tie_aware_paired_bootstrap(arms, arms, ids, gold, reps=200)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_always_vs_never(self):
        ids = [1, 2, 3, 4]
        gold = {i: "1" for i in ids}
        good = {i: {"answer": "1", "tied": False, "top": ["1"], "n_votes": 1} for i in ids}
        bad = {i: {"answer": "2", "tied": False, "top": ["2"], "n_votes": 1} for i in ids}
        lo, hi = M.tie_aware_paired_bootstrap(good, bad, ids, gold, reps=200)
        self.assertEqual((lo, hi), (100.0, 100.0))

    def test_deterministic_and_ordered(self):
        ids = [1, 2]
        gold = {1: "5", 2: "5"}
        a = {1: {"answer": "6", "tied": True, "top": ["5", "6"], "n_votes": 2},
             2: {"answer": "6", "tied": True, "top": ["5", "6"], "n_votes": 2}}
        b = {1: {"answer": "6", "tied": False, "top": ["6"], "n_votes": 1},
             2: {"answer": "6", "tied": False, "top": ["6"], "n_votes": 1}}
        r1 = M.tie_aware_paired_bootstrap(a, b, ids, gold, reps=500)
        r2 = M.tie_aware_paired_bootstrap(a, b, ids, gold, reps=500)
        self.assertEqual(r1, r2)
        self.assertLessEqual(r1[0], r1[1])


if __name__ == "__main__":
    unittest.main()

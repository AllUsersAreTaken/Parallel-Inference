"""Generate the linter-rollback benchmark cases.

Original set: 10 Python functions x 5 variants x 2 prompt types = 100 cases.
Expanded set: 20 functions (the original 10 plus 10 library-tempting,
builtin-solvable tasks) = 200 cases. Pass --expanded to write the 200-case
files; without it the original 100-case files are reproduced byte-for-byte.

Temptation prompts allow the standard library; clean prompts forbid imports.
Test expectations are produced by reference implementations, so the JSON is
deterministic. `--strength strong` makes the temptation prompt explicitly
encourage the standard library (same functions, same tests, same ids). It writes
data/linter_cases_strong_100.tsv (or _strong_200.tsv with --expanded).

Outputs (original):
  data/linter_cases_100.tsv        id, type, template, prompt, func, hint
  data/linter_cases_strong_100.tsv same with the strong temptation tail
  data/linter_tests.json           {id: {"func": ..., "cases": [[args, expected], ...]}}

Outputs (--expanded):
  data/linter_cases_200.tsv
  data/linter_cases_strong_200.tsv
  data/linter_tests_200.json

Rows are interleaved (temptation, clean) per variant so `--limit 6` yields
3 temptation + 3 clean cases.
No third-party dependencies; runs on Windows Python.
"""
import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV = os.path.join(ROOT, "data", "linter_cases_100.tsv")
TSV_STRONG = os.path.join(ROOT, "data", "linter_cases_strong_100.tsv")
TESTS = os.path.join(ROOT, "data", "linter_tests.json")
TSV_200 = os.path.join(ROOT, "data", "linter_cases_200.tsv")
TSV_STRONG_200 = os.path.join(ROOT, "data", "linter_cases_strong_200.tsv")
TESTS_200 = os.path.join(ROOT, "data", "linter_tests_200.json")

STRONG_TEMPTATION = (
    "You are encouraged to use the Python standard library where it simplifies "
    "the solution, e.g. math.gcd, math.isqrt, math.factorial."
)

# Extra, function-specific library bait for the expanded templates. The
# original 10 keep STRONG_TEMPTATION untouched so their prompts stay frozen.
LIB_TEMPTATION = {
    "chunk": "itertools.islice",
    "pairwise": "itertools.pairwise",
    "word_freq": "collections.Counter",
    "group_by_len": "collections.defaultdict",
    "dedupe": "dict.fromkeys",
    "clamp": "math or min/max helpers",
    "round_half_up": "decimal.Decimal or math.floor",
    "parse_ints": "re.findall",
    "csv_row": "csv.reader",
    "running_max": "itertools.accumulate",
}

ORIGINAL_FUNCS = [
    "isqrt", "gcd", "lcm", "factorial", "fib", "is_prime", "mean", "flatten",
    "unique", "reverse_words",
]


def _isqrt(n):
    return math.isqrt(n)


def _gcd(a, b):
    return math.gcd(a, b)


def _lcm(a, b):
    return math.lcm(a, b)


def _factorial(n):
    return math.factorial(n)


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _mean(xs):
    return sum(xs) / len(xs)


def _flatten(xss):
    return [x for xs in xss for x in xs]


def _unique(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def _reverse_words(s):
    return " ".join(reversed(s.split()))


def _chunk(xs, n):
    return [list(xs[i:i + n]) for i in range(0, len(xs), n)]


def _pairwise(xs):
    return [[xs[i], xs[i + 1]] for i in range(len(xs) - 1)]


def _word_freq(s):
    out = {}
    for w in s.split():
        out[w] = out.get(w, 0) + 1
    return out


def _group_by_len(words):
    out = {}
    for w in words:
        out.setdefault(len(w), []).append(w)
    return out


def _dedupe(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def _clamp(x, lo, hi):
    return max(lo, min(x, hi))


def _round_half_up(x):
    return int(math.floor(x + 0.5))


def _parse_ints(s):
    out = []
    i = 0
    while i < len(s):
        neg = False
        if s[i] == "-" and i + 1 < len(s) and s[i + 1].isdigit():
            neg = True
            i += 1
        if i < len(s) and s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            v = int(s[i:j])
            out.append(-v if neg else v)
            i = j
        else:
            i += 1
    return out


def _csv_row(s):
    out = []
    field = []
    quoted = False
    i = 0
    while i < len(s):
        c = s[i]
        if quoted:
            if c == '"':
                if i + 1 < len(s) and s[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                quoted = False
                i += 1
                continue
            field.append(c)
            i += 1
            continue
        if c == '"':
            quoted = True
            i += 1
            continue
        if c == ',':
            out.append("".join(field))
            field = []
            i += 1
            continue
        field.append(c)
        i += 1
    out.append("".join(field))
    return out


def _running_max(xs):
    out = []
    best = None
    for x in xs:
        best = x if best is None else max(best, x)
        out.append(best)
    return out


REF = {
    "isqrt": _isqrt,
    "gcd": _gcd,
    "lcm": _lcm,
    "factorial": _factorial,
    "fib": _fib,
    "is_prime": _is_prime,
    "mean": _mean,
    "flatten": _flatten,
    "unique": _unique,
    "reverse_words": _reverse_words,
    "chunk": _chunk,
    "pairwise": _pairwise,
    "word_freq": _word_freq,
    "group_by_len": _group_by_len,
    "dedupe": _dedupe,
    "clamp": _clamp,
    "round_half_up": _round_half_up,
    "parse_ints": _parse_ints,
    "csv_row": _csv_row,
    "running_max": _running_max,
}

HINTS = {
    "isqrt": "isqrt(0) must return 0.",
    "gcd": "gcd(0, 0) must return 0.",
    "lcm": "lcm(0, 0) must return 0.",
    "factorial": "factorial(0) must return 1.",
    "fib": "fib(0) must return 0.",
    "is_prime": "is_prime(1) must return False.",
    "mean": "mean must return a float; use sum/len.",
    "flatten": "flatten([]) must return [].",
    "unique": "preserve first-occurrence order.",
    "reverse_words": "split on whitespace and reverse word order.",
    "chunk": "chunk([], 3) must return [] and n is at least 1.",
    "pairwise": "pairwise([]) and pairwise([7]) must return [].",
    "word_freq": "split on whitespace; an empty string returns {}.",
    "group_by_len": "keep words in input order within each length; empty "
                    "input returns {}.",
    "dedupe": "preserve first-occurrence order; dedupe([]) returns [].",
    "clamp": "lo <= hi; return x unchanged when lo <= x <= hi.",
    "round_half_up": "inputs are non-negative; round_half_up(2.5) is 3.",
    "parse_ints": "a '-' immediately before digits is part of the number; "
                  "ignore other characters.",
    "csv_row": "quoted fields may contain commas; a doubled quote inside "
               "quotes is one literal quote.",
    "running_max": "running_max([]) returns []; compare each prefix maximum.",
}

DESC = {
    "isqrt": "returns the integer square root of a non-negative integer n "
             "(the largest integer x such that x*x <= n) without using floating point",
    "gcd": "returns the greatest common divisor of two non-negative integers a and b "
           "(gcd(0, 0) is 0)",
    "lcm": "returns the least common multiple of two non-negative integers a and b "
           "(lcm(0, n) and lcm(n, 0) are 0)",
    "factorial": "returns n factorial for a non-negative integer n (factorial(0) is 1)",
    "fib": "returns the n-th Fibonacci number for a non-negative integer n, "
           "with fib(0) = 0 and fib(1) = 1",
    "is_prime": "returns True if a non-negative integer n is prime and False otherwise "
                "(n < 2 is not prime)",
    "mean": "returns the arithmetic mean of a non-empty list of numbers as a float",
    "flatten": "returns a single list containing the elements of a list of lists, "
               "in order, one level deep",
    "unique": "returns a new list with duplicates removed while preserving "
              "first-occurrence order",
    "reverse_words": "returns a string with the whitespace-separated words in reverse "
                     "order joined by a single space",
    "chunk": "splits a list into consecutive lists of length n, with the last "
             "chunk shorter when needed",
    "pairwise": "returns a list of consecutive two-element lists from a list, "
                "in order",
    "word_freq": "returns a dict mapping each whitespace-separated word to the "
                 "number of times it occurs",
    "group_by_len": "returns a dict mapping each word length to the list of "
                    "words of that length, in input order",
    "dedupe": "returns a new list with duplicates removed while preserving "
              "first-occurrence order",
    "clamp": "returns x limited to the inclusive range [lo, hi]",
    "round_half_up": "returns the nearest integer to a non-negative number, "
                     "rounding exact halves up",
    "parse_ints": "returns the list of integers appearing in a string, in "
                  "order; a leading minus sign belongs to the number",
    "csv_row": "parses one CSV row string into a list of fields, honoring "
               "double-quoted fields and doubled quotes",
    "running_max": "returns a list where each element is the maximum of the "
                   "input prefix ending at that index",
}

# Five variants per function: different examples and test inputs.
VARIANTS = {
    "isqrt": [
        dict(examples=[16, 10], tests=[0, 1, 15, 17, 99]),
        dict(examples=[9, 24], tests=[2, 3, 4, 25, 100]),
        dict(examples=[25, 8], tests=[5, 8, 26, 120, 144]),
        dict(examples=[100, 50], tests=[6, 7, 35, 63, 10000]),
        dict(examples=[1, 0], tests=[11, 12, 48, 80, 101]),
    ],
    "gcd": [
        dict(examples=[(12, 18), (7, 13)],
             tests=[(0, 5), (5, 0), (0, 0), (48, 18), (100, 75)]),
        dict(examples=[(48, 18), (17, 17)],
             tests=[(270, 192), (36, 60), (8, 12), (14, 21), (9, 28)]),
        dict(examples=[(100, 75), (30, 45)],
             tests=[(16, 24), (21, 35), (11, 22), (13, 39), (81, 27)]),
        dict(examples=[(270, 192), (36, 60)],
             tests=[(6, 9), (10, 15), (24, 36), (7, 14), (45, 60)]),
        dict(examples=[(8, 12), (14, 21)],
             tests=[(2, 4), (3, 5), (20, 30), (12, 16), (99, 121)]),
    ],
    "lcm": [
        dict(examples=[(4, 6), (3, 5)],
             tests=[(2, 3), (5, 7), (8, 12), (6, 9), (10, 15)]),
        dict(examples=[(8, 12), (6, 9)],
             tests=[(0, 5), (5, 0), (0, 0), (12, 18), (21, 14)]),
        dict(examples=[(10, 15), (7, 3)],
             tests=[(4, 10), (9, 12), (16, 24), (11, 13), (25, 30)]),
        dict(examples=[(12, 18), (21, 14)],
             tests=[(2, 5), (3, 8), (14, 21), (18, 27), (20, 25)]),
        dict(examples=[(4, 10), (9, 12)],
             tests=[(6, 10), (15, 20), (7, 9), (22, 33), (100, 75)]),
    ],
    "factorial": [
        dict(examples=[5, 0], tests=[1, 2, 3, 4, 6]),
        dict(examples=[3, 7], tests=[0, 5, 8, 9, 10]),
        dict(examples=[4, 6], tests=[1, 2, 7, 8, 9]),
        dict(examples=[2, 9], tests=[0, 3, 4, 5, 10]),
        dict(examples=[1, 10], tests=[2, 6, 7, 8, 9]),
    ],
    "fib": [
        dict(examples=[10, 5], tests=[0, 1, 2, 3, 6]),
        dict(examples=[0, 1], tests=[4, 5, 7, 8, 9]),
        dict(examples=[7, 12], tests=[2, 3, 10, 11, 13]),
        dict(examples=[3, 15], tests=[1, 4, 6, 9, 14]),
        dict(examples=[8, 14], tests=[0, 5, 12, 13, 15]),
    ],
    "is_prime": [
        dict(examples=[7, 10], tests=[0, 1, 2, 4, 9]),
        dict(examples=[2, 15], tests=[3, 5, 11, 17, 21]),
        dict(examples=[17, 21], tests=[13, 19, 23, 25, 29]),
        dict(examples=[23, 25], tests=[27, 31, 37, 49, 53]),
        dict(examples=[29, 31], tests=[41, 43, 47, 97, 101]),
    ],
    "mean": [
        dict(examples=[[1, 2, 3], [4, 5]],
             tests=[[2, 4], [10, 20, 30], [5], [1, 2], [0, 0, 0]]),
        dict(examples=[[10, 20, 30], [5]],
             tests=[[1, 2, 3, 4], [7, 8], [-1, 1], [2.5, 2.5], [100, 200]]),
        dict(examples=[[7, 8], [-1, 1]],
             tests=[[0, 5], [3, 6, 9], [1.5, 2.5], [10, 10, 10], [2, 3, 4, 5]]),
        dict(examples=[[2.5, 2.5], [100, 200]],
             tests=[[4, 5, 6], [9], [0], [1, 100], [20, 30, 40]]),
        dict(examples=[[3, 6, 9], [1.5, 2.5]],
             tests=[[2, 2], [6, 7], [1, 2, 3, 4, 5], [50], [7, 9, 11, 13]]),
    ],
    "flatten": [
        dict(examples=[[[1, 2], [3]], [[1, 2]]],
             tests=[[[], [1]], [[1], [2], [3]], [[1, 2, 3]], [[], [1], []],
                    [[4, 5], [6, 7]]]),
        dict(examples=[[[1], [2], [3]], [[1, 2, 3]]],
             tests=[[[1], [], [2, 3]], [[0]], [[1, 2], [], [3]], [[7], [8, 9]]]),
        dict(examples=[[[], [1], []], [[4, 5], [6, 7]]],
             tests=[[[9, 10]], [[], [], []], [[1, 1], [2, 2]], [[5], [6], [7]]]),
        dict(examples=[[[9, 10]]],
             tests=[[[1, 2], [3], [4, 5]], [[]], [[11], [12, 13], [14]],
                    [[0, 0], [0]]]),
        dict(examples=[[[1, 1], [2, 2]], [[5], [6], [7]]],
             tests=[[[1, 2], [], [3, 4]], [[15]], [[8], [9], [10, 11]],
                    [[2], [3], [4], [5]]]),
    ],
    "unique": [
        dict(examples=[[1, 2, 1, 3], [3, 3, 3]],
             tests=[[], [1, 1, 1], [1, 2, 3], [1, 2, 1, 3, 2], [5, 4, 5, 4]]),
        dict(examples=[[1, 2, 3], [1, 2, 1, 3, 2]],
             tests=[[0, 0], [1, 2], [7, 8, 7], [9, 9, 8, 9], [2, 1, 2, 1, 3]]),
        dict(examples=[[5, 4, 5, 4], [0, 0]],
             tests=[[1, 2, 3, 1], [4], [3, 1, 3, 2], [10, 20, 10], [6, 5, 6, 4]]),
        dict(examples=[[7, 8, 7], [9, 9, 8, 9]],
             tests=[[1, 1, 2, 2, 3, 3], [2, 3, 2], [0, 1, 0], [100, 100],
                    [8, 7, 8]]),
        dict(examples=[[2, 1, 2, 1, 3], [1, 2, 3, 1]],
             tests=[[4, 5, 4, 6], [1], [9, 8, 9, 7], [3, 2, 1, 3, 2],
                    [11, 12, 11]]),
    ],
    "reverse_words": [
        dict(examples=["hello world", "a b c"],
             tests=["the sky is blue", "one", "", "a  b", "hello world"]),
        dict(examples=["the sky is blue", "one"],
             tests=["", "a  b", "hello world", "python is fun", "x y z"]),
        dict(examples=["python is fun", "x y z"],
             tests=["one two", "a b c d", "  leading", "trailing  ", "same"]),
        dict(examples=["one two", "a b c d"],
             tests=["hello", "hello world", "a b", "1 2 3", "foo bar baz"]),
        dict(examples=["  leading", "trailing  "],
             tests=["same", "the sky is blue", "python is fun", "one", "a b c"]),
    ],
    "chunk": [
        dict(examples=[([1, 2, 3, 4, 5], 2), ([1, 2, 3], 5)],
             tests=[([], 2), ([1], 1), ([1, 2, 3, 4], 2), ([1, 2, 3, 4, 5], 3),
                    ([9, 8, 7], 4)]),
        dict(examples=[([1, 2, 3], 1), ([4, 5, 6, 7], 3)],
             tests=[([], 1), ([1, 2], 2), ([1, 2, 3], 2), ([1, 2, 3, 4, 5, 6], 4),
                    ([5], 5)]),
        dict(examples=[([10, 20, 30, 40], 3), ([1, 2], 2)],
             tests=[([], 5), ([1, 2, 3], 5), ([1, 2, 3, 4, 5, 6, 7], 3),
                    ([8, 9], 1), ([0, 0, 0], 2)]),
        dict(examples=[([1, 2, 3, 4, 5, 6], 4), ([7, 8, 9], 2)],
             tests=[([], 3), ([1, 2, 3, 4], 4), ([1, 2, 3, 4, 5], 4),
                    ([1, 2, 3, 4, 5, 6, 7, 8], 5), ([2, 4, 6], 3)]),
        dict(examples=[([5, 6, 7, 8], 2), ([1, 2, 3, 4, 5], 4)],
             tests=[([], 1), ([1], 2), ([1, 2, 3, 4, 5, 6], 2),
                    ([1, 2, 3, 4, 5, 6, 7], 6), ([9, 9, 9, 9], 2)]),
    ],
    "pairwise": [
        dict(examples=[[1, 2, 3, 4], [5, 6]],
             tests=[[], [1], [1, 2], [1, 2, 3], [4, 5, 6, 7]]),
        dict(examples=[[10, 20, 30], []],
             tests=[[], [7], [7, 8], [1, 2, 3, 4], [9, 8, 7, 6, 5]]),
        dict(examples=[[1, 2], [3, 4, 5, 6]],
             tests=[[], [0], [0, 1], [2, 3, 4], [5, 6, 7, 8, 9]]),
        dict(examples=[[2, 4, 6, 8], [1, 3, 5]],
             tests=[[], [1], [1, 2], [10, 20, 30, 40], [7, 7, 7]]),
        dict(examples=[[9, 8, 7], [6, 5, 4, 3]],
             tests=[[], [42], [1, 2], [3, 4, 5], [1, 2, 3, 4, 5, 6]]),
    ],
    "word_freq": [
        dict(examples=["a b a", "hello"],
             tests=["", "a", "a b a", "one two two three", "x x x"]),
        dict(examples=["red blue red", ""],
             tests=["", "red", "red blue red", "a a b b c",
                    "hello world hello"]),
        dict(examples=["to be or not to be", "x"],
             tests=["", "to be or not to be", "a b c d",
                    "one one two three three three", "same same"]),
        dict(examples=["one two three", "a a"],
             tests=["", "one two three", "a a a", "cats dogs cats",
                    "go go go stop"]),
        dict(examples=["alpha beta alpha gamma", "z"],
             tests=["", "alpha beta alpha gamma", "a b c a b c",
                    "hello hello world", "q"]),
    ],
    "group_by_len": [
        dict(examples=[["a", "bb", "c"], ["dd", "eee"]],
             tests=[[], ["a"], ["a", "bb", "c"], ["aa", "b", "cc", "d"],
                    ["one", "two", "three"]]),
        dict(examples=[["hi", "there"], ["a", "bc", "def"]],
             tests=[[], ["a", "b"], ["hi", "there", "ok"],
                    ["x", "yy", "zzz", "ww"], ["aa", "bb", "c"]]),
        dict(examples=[["red", "blue", "green"], ["a"]],
             tests=[[], ["red", "blue", "green"],
                    ["a", "bb", "ccc", "dddd"], ["one", "two", "six"],
                    ["ab", "cd", "ef"]]),
        dict(examples=[["ant", "bee", "cat"], ["aa", "bb"]],
             tests=[[], ["ant", "bee", "cat"], ["aa", "bb", "cc", "dd"],
                    ["x", "yy", "zzz"], ["hello", "world", "hi"]]),
        dict(examples=[["a", "bb", "ccc", "dddd"], ["aa", "bb"]],
             tests=[[], ["a", "b", "c"], ["aa", "bb", "cc", "dd"],
                    ["one", "two", "three", "four"],
                    ["x", "yy", "zzz", "wwww"]]),
    ],
    "dedupe": [
        dict(examples=[[1, 2, 1, 3], [3, 3, 3]],
             tests=[[], [1], [1, 1, 1], [1, 2, 1, 3, 2], [5, 4, 5, 4]]),
        dict(examples=[[1, 2, 3], [1, 2, 1, 3, 2]],
             tests=[[], [0, 0], [1, 2], [7, 8, 7], [9, 9, 8, 9]]),
        dict(examples=[[5, 4, 5, 4], [0, 0]],
             tests=[[], [1, 2, 3, 1], [4], [3, 1, 3, 2], [10, 20, 10]]),
        dict(examples=[[7, 8, 7], [9, 9, 8, 9]],
             tests=[[], [1, 1, 2, 2, 3, 3], [2, 3, 2], [0, 1, 0],
                    [100, 100]]),
        dict(examples=[[2, 1, 2, 1, 3], [1, 2, 3, 1]],
             tests=[[], [4, 5, 4, 6], [1], [9, 8, 9, 7], [3, 2, 1, 3, 2]]),
    ],
    "clamp": [
        dict(examples=[(5, 0, 10), (15, 0, 10)],
             tests=[(0, 0, 10), (10, 0, 10), (5, 0, 10), (-3, -5, 5),
                    (15, 0, 10)]),
        dict(examples=[(7, 1, 9), (-2, 0, 5)],
             tests=[(1, 1, 9), (9, 1, 9), (0, 0, 5), (5, 1, 5), (12, 1, 9)]),
        dict(examples=[(50, 10, 100), (150, 10, 100)],
             tests=[(10, 10, 100), (100, 10, 100), (50, 10, 100),
                    (9, 10, 100), (150, 10, 100)]),
        dict(examples=[(-5, -10, 0), (5, -10, 0)],
             tests=[(-10, -10, 0), (0, -10, 0), (-5, -10, 0),
                    (-11, -10, 0), (1, -10, 0)]),
        dict(examples=[(3, 2, 4), (1, 2, 4)],
             tests=[(2, 2, 4), (4, 2, 4), (3, 2, 4), (1, 2, 4), (5, 2, 4)]),
    ],
    "round_half_up": [
        dict(examples=[2.5, 3.4],
             tests=[0.0, 0.4, 0.5, 1.5, 2.5]),
        dict(examples=[1.5, 0.5],
             tests=[0.0, 0.49, 0.5, 1.49, 1.5]),
        dict(examples=[10.5, 9.4],
             tests=[2.0, 2.49, 2.5, 2.51, 3.0]),
        dict(examples=[4.5, 4.4],
             tests=[1.0, 1.25, 1.5, 1.75, 2.0]),
        dict(examples=[0.5, 5.5],
             tests=[3.0, 3.49, 3.5, 3.99, 4.0]),
    ],
    "parse_ints": [
        dict(examples=["a1 b22 -3", "no digits"],
             tests=["", "a1 b22 -3", "no digits", "7", "-42"]),
        dict(examples=["x-7 y8", "10,20,30"],
             tests=["", "x-7y8", "10,20,30", "v1.2", "abc"]),
        dict(examples=["p3q45", "-1-2"],
             tests=["", "p3q45", "-1-2", "tmp99", "a 0 b"]),
        dict(examples=["n 100 m -50", ""],
             tests=["", "n 100 m -50", "a1b2c3", "-0", "z"]),
        dict(examples=["42", "a-9b"],
             tests=["", "42", "a-9b", "1 22 333", "x-y"]),
    ],
    "csv_row": [
        dict(examples=["a,b,c", '"a,b",c'],
             tests=["a,b,", "a,b,c", '"a,b",c', "a,,c", '"a""b",c']),
        dict(examples=['"x,y",z', "1,2,3"],
             tests=["a", "1,2,3", '"x,y",z', "a,b", '""']),
        dict(examples=["a,b,c,d", '"hello, world",x'],
             tests=["a,b", "a,b,c,d", '"hello, world",x', "one,,three",
                    '"q""q",r']),
        dict(examples=['"a""b",c', "x,y"],
             tests=["x", "x,y", '"a""b",c', ",", '"",z']),
        dict(examples=["red,green,blue", '"a,b",c,d'],
             tests=["a,b,c", "red,green,blue", '"a,b",c,d', "a,b,",
                    '"x,y,z",w']),
    ],
    "running_max": [
        dict(examples=[[1, 3, 2, 5], [4, 4, 2]],
             tests=[[], [1], [1, 3, 2, 5], [5, 4, 3, 2], [2, 1, 4, 3]]),
        dict(examples=[[10, 1, 9, 12], [-1, -3, -2]],
             tests=[[], [7], [10, 1, 9, 12], [1, 2, 3, 4], [4, 3, 2, 1]]),
        dict(examples=[[2, 2, 2], [0, -1, 5]],
             tests=[[], [0], [2, 2, 2], [0, -1, 5], [-5, -4, -3]]),
        dict(examples=[[3, 1, 4, 1, 5], [9, 8, 7]],
             tests=[[], [1, 1], [3, 1, 4, 1, 5], [9, 8, 7, 6],
                    [2, 4, 2, 4]]),
        dict(examples=[[1, 2, 3], [5, 3, 8, 2]],
             tests=[[], [6, 6], [1, 2, 3], [5, 3, 8, 2], [10, 0, 10, 0]]),
    ],
}


def call_sig(func, args):
    return "%s(%s)" % (func, ", ".join(repr(a) for a in args))


def examples_text(func, examples):
    parts = []
    for a in examples:
        args = list(a) if isinstance(a, tuple) else [a]
        parts.append("%s -> %r" % (call_sig(func, args), REF[func](*args)))
    return "; ".join(parts)


def strong_tail(func):
    hint = LIB_TEMPTATION.get(func)
    if hint:
        return ("You are encouraged to use the Python standard library where "
                "it simplifies the solution, e.g. %s." % hint)
    return STRONG_TEMPTATION


def build(strength="normal", expanded=False):
    rows = []
    tests = {}
    strong = strength == "strong"
    funcs = list(DESC) if expanded else ORIGINAL_FUNCS
    for func in funcs:
        for vi, var in enumerate(VARIANTS[func], start=1):
            ex = examples_text(func, var["examples"])
            base = ("Write a Python function named `%s` that %s. Examples: %s. "
                    % (func, DESC[func], ex))
            tail = "Output only the function in one ```python block."
            tempt_tail = strong_tail(func) if strong else \
                "You may use the Python standard library."
            prompts = {
                "temptation": base + tempt_tail + " " + tail,
                "clean": base + "Do not use any import statements. Use only Python "
                                "builtins. " + tail,
            }
            cases = []
            for a in var["tests"]:
                args = list(a) if isinstance(a, tuple) else [a]
                cases.append([args, REF[func](*args)])
            assert 4 <= len(cases) <= 6
            for kind in ("temptation", "clean"):
                cid = "%s_%s_%d" % ("t" if kind == "temptation" else "k", func, vi)
                rows.append((cid, kind, func, prompts[kind], func, HINTS[func]))
                tests[cid] = {"func": func, "cases": cases}
    want = 200 if expanded else 100
    assert len(rows) == want, len(rows)
    assert len({r[0] for r in rows}) == want
    return rows, tests


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", choices=("normal", "strong"), default="normal")
    ap.add_argument("--expanded", action="store_true",
                    help="write the 200-case files (20 templates)")
    args = ap.parse_args()
    if args.expanded:
        out_tsv = TSV_STRONG_200 if args.strength == "strong" else TSV_200
        out_tests = TESTS_200
    else:
        out_tsv = TSV_STRONG if args.strength == "strong" else TSV
        out_tests = TESTS
    rows, tests = build(args.strength, args.expanded)
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("id\ttype\ttemplate\tprompt\tfunc\thint\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    with open(out_tests, "w", encoding="utf-8", newline="\n") as f:
        json.dump(tests, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote", len(rows), "cases ->", os.path.relpath(out_tsv, ROOT))
    print("wrote", len(tests), "test specs ->", os.path.relpath(out_tests, ROOT))

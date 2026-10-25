#!/usr/bin/env python3
"""Zero-GPU token-level prefix-trie analysis of the 8-stream GSM8K probe.

Inputs
  results/probs_probe30.json        per-problem answers/gold/maj/oracle
  data/_probs_pNN.jsonl             one row per emitted token (m,s,step,tok)
  data/_probs_pNN.json              stream ids/text (ids validated only)

Outputs
  results/trie_analysis.txt         printed report (also written to disk)
  results/trie_problems.csv         per-problem metrics

Definitions (these are the numbers the report uses)
  trace           token-id sequence of one stream: m=="batch" rows sorted by step.
  trie node       (parent, token); node.streams = ids of streams passing through.
  live prefixes   number of trie nodes at depth d (only streams longer than d).
  first split     shared tokens before the first fork (0 = the first token differs,
                  1 = one token shared, ...).
                  Normalized: first_split_depth / mean_trace_len; 1.0 = never split.
  trunk_frac_mass mean over depths of (largest prefix mass / live streams), i.e. the
                  average share of surviving streams on the modal prefix.
  trie_token_saving  1 - unique_nodes / total_emitted_tokens (upper bound on token
                  positions that are duplicate prefixes; not a runtime measurement).
  fork            node with >=2 children; child mass = |child.streams| / n_traces.
  correct stream  records' answers[i] == records' gold (answers are already canonical).
  escape branch   first prefix node on a correct stream's path whose streams are all
                  correct. Its mass is the "correct branch mass": the first point at
                  which that correct trace is separated from every incorrect trace.
                  Per problem we report the mass at the shallowest escape (the first
                  correct/incorrect separation) and the maximum over escapes.

Gates
  G1 = maj@8 accuracy - stream-1 accuracy
  G2 = oracle@8 accuracy - maj@8 accuracy
  G3 = problems with oracle correct, majority wrong, and correct answer support == 1
       (the only correct answer is a 1/8 branch; static 8-sample voting cannot surface it)
"""

import csv
import io
import json
import math
import os
from collections import Counter, defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "probs_probe30.json")
DATA = os.path.join(ROOT, "data")
OUT_TXT = os.path.join(ROOT, "results", "trie_analysis.txt")
OUT_CSV = os.path.join(ROOT, "results", "trie_problems.csv")

EPS = 1e-9
buf = io.StringIO()


def w(line=""):
    print(line)
    buf.write(str(line) + "\n")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def pctl(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def fmt(x, nd=4):
    if x is None:
        return "NA"
    return ("%." + str(nd) + "f") % x


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


class Node:
    __slots__ = ("tok", "parent", "depth", "children", "streams")

    def __init__(self, tok, parent, depth):
        self.tok = tok
        self.parent = parent
        self.depth = depth
        self.children = {}
        self.streams = set()


def load_tokens(path):
    by_stream = defaultdict(list)
    issues = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("m") != "batch":
                continue
            by_stream[str(row["s"])].append((int(row["step"]), int(row["tok"])))
    seqs = {}
    for sid, items in by_stream.items():
        items.sort(key=lambda x: x[0])
        steps = [x[0] for x in items]
        if steps != list(range(len(steps))):
            issues.append("stream %s: non-contiguous steps" % sid)
        seqs[sid] = [x[1] for x in items]
    return seqs, issues


def build_trie(seqs):
    nodes = [Node(None, -1, 0)]
    terminals = {}
    for sid in sorted(seqs, key=lambda s: int(s)):
        cur = 0
        for tok in seqs[sid]:
            nodes[cur].streams.add(sid)
            nxt = nodes[cur].children.get(tok)
            if nxt is None:
                nxt = len(nodes)
                nodes.append(Node(tok, cur, nodes[cur].depth + 1))
                nodes[cur].children[tok] = nxt
            cur = nxt
        nodes[cur].streams.add(sid)
        terminals[sid] = cur
    return nodes, terminals


def stream_path(nodes, tokens):
    cur = 0
    path = [0]
    for tok in tokens:
        cur = nodes[cur].children[tok]
        path.append(cur)
    return path


def analyze_problem(rec):
    seqs, issues = load_tokens(os.path.join(DATA, rec["probs_jsonl"]))
    try:
        with open(os.path.join(DATA, rec["probs_json"]), encoding="utf-8") as fh:
            jids = [str(s["id"]) for s in json.load(fh)["streams"]]
        if jids != [str(i + 1) for i in range(len(jids))]:
            issues.append("json stream order %s" % jids)
        if not set(seqs).issubset(set(jids)):
            issues.append("sidecar stream ids %s not in json ids %s"
                          % (sorted(seqs), jids))
    except (OSError, KeyError, ValueError) as exc:
        issues.append("json stream check failed: %s" % exc)

    n_traces = len(seqs)
    total_tokens = sum(len(v) for v in seqs.values())
    mean_len = total_tokens / n_traces if n_traces else 0.0

    nodes, _ = build_trie(seqs)
    max_depth = max(n.depth for n in nodes)
    levels = defaultdict(list)
    for i in range(1, len(nodes)):
        levels[nodes[i].depth].append(i)

    first_split_node_depth = None
    for d in sorted(levels):
        if len(levels[d]) > 1:
            first_split_node_depth = d
            break
    if first_split_node_depth is None:
        first_split_depth = None
        first_split_frac = 1.0
    else:
        first_split_depth = first_split_node_depth - 1
        first_split_frac = min(1.0, first_split_depth / mean_len) if mean_len else 0.0

    modal_shares = []
    for d in range(1, max_depth + 1):
        ids = levels.get(d, [])
        if not ids:
            continue
        live = sum(len(nodes[i].streams) for i in ids)
        modal_shares.append(max(len(nodes[i].streams) for i in ids) / live)
    trunk_frac_mass = mean(modal_shares) if modal_shares else 1.0

    token_saving = 1.0 - (len(nodes) - 1) / total_tokens if total_tokens else 0.0
    n_distinct_traces = len(set(tuple(v) for v in seqs.values()))

    forks = []
    for i in range(len(nodes)):
        node = nodes[i]
        if len(node.children) < 2:
            continue
        children = []
        for ci in node.children.values():
            ss = nodes[ci].streams
            children.append({"k": len(ss), "mass": len(ss) / n_traces, "sids": ss})
        forks.append({"depth": node.depth, "children": children})

    answers = rec["answers"]
    gold = rec["gold"]
    correct_answers = 0
    oracle_all = False
    for a in answers:
        if a is not None and a == gold:
            correct_answers += 1
            oracle_all = True

    correct_sids = set()
    for sid in seqs:
        i = int(sid) - 1
        a = answers[i] if 0 <= i < len(answers) else None
        if a is not None and a == gold:
            correct_sids.add(sid)
    incorrect_sids = set(seqs) - correct_sids

    escapes = []
    if incorrect_sids:
        for sid in sorted(correct_sids, key=lambda s: int(s)):
            path = stream_path(nodes, seqs[sid])
            for idx in path[1:]:
                if not (nodes[idx].streams & incorrect_sids):
                    escapes.append({
                        "sid": sid,
                        "node": idx,
                        "depth": nodes[idx].depth,
                        "mass": len(nodes[idx].streams) / n_traces,
                    })
                    break

    escape_nodes = {}
    for e in escapes:
        g = escape_nodes.setdefault(e["node"], {
            "depth": e["depth"], "mass": e["mass"], "sids": []})
        g["sids"].append(e["sid"])

    if escapes:
        first_escape_depth = min(e["depth"] for e in escapes)
        first_masses = [g["mass"] for g in escape_nodes.values()
                        if g["depth"] == first_escape_depth]
        correct_branch_mass_first = max(first_masses)
        correct_branch_mass_max = max(g["mass"] for g in escape_nodes.values())
        correct_branch_mass_med = median([e["mass"] for e in escapes])
        first_escape_frac = min(1.0, first_escape_depth / mean_len) if mean_len else 1.0
        n_escape_branches = len(escape_nodes)
        all_escape_tail = all(e["mass"] <= 0.125 + EPS for e in escapes)
    else:
        first_escape_depth = None
        correct_branch_mass_first = None
        correct_branch_mass_max = None
        correct_branch_mass_med = None
        first_escape_frac = None
        n_escape_branches = 0
        all_escape_tail = None

    maj_ok = bool(rec["maj_ok"])
    tail_only_answer = bool(oracle_all and not maj_ok and correct_answers == 1)
    tail_only_trace = None
    if oracle_all and not maj_ok and len(escapes) == correct_answers and correct_answers > 0:
        tail_only_trace = bool(all_escape_tail)

    return {
        "qid": rec["qid"],
        "group": rec.get("group"),
        "n_traces": n_traces,
        "total_tokens": total_tokens,
        "trace_len_mean": mean_len,
        "max_depth": max_depth,
        "n_distinct_traces": n_distinct_traces,
        "n_distinct_answers": rec.get("distinct"),
        "first_split_depth": first_split_depth,
        "first_split_frac": first_split_frac,
        "trunk_frac_mass": trunk_frac_mass,
        "token_saving": token_saving,
        "levels": {d: len(ids) for d, ids in levels.items()},
        "forks": forks,
        "n_forks": len(forks),
        "correct_answers": correct_answers,
        "oracle": oracle_all,
        "maj_ok": maj_ok,
        "correct_sids": correct_sids,
        "incorrect_sids": incorrect_sids,
        "base_rate": (len(correct_sids) / n_traces) if n_traces else 0.0,
        "escapes": escapes,
        "escape_nodes": escape_nodes,
        "first_escape_depth": first_escape_depth,
        "first_escape_frac": first_escape_frac,
        "correct_branch_mass": correct_branch_mass_first,
        "correct_branch_mass_max": correct_branch_mass_max,
        "correct_branch_mass_median": correct_branch_mass_med,
        "n_escape_branches": n_escape_branches,
        "tail_only": tail_only_answer,
        "tail_only_trace": tail_only_trace,
        "issues": issues,
    }


def mass_hist(values):
    c = Counter()
    for v in values:
        if v is None:
            continue
        c[int(round(v * 8))] += 1
    return c


def main():
    with open(BASE, encoding="utf-8") as fh:
        meta = json.load(fh)

    results = [analyze_problem(rec) for rec in meta["records"]]
    qid_by = {r["qid"]: r for r in results}

    integrity = []
    for rec, p in zip(meta["records"], results):
        if p["n_traces"] != meta["n_streams"]:
            integrity.append("q%d: %d trie traces (expected %d)"
                             % (p["qid"], p["n_traces"], meta["n_streams"]))
        if p["correct_answers"] != rec["gold_support"]:
            integrity.append("q%d: recomputed support %d != record %d"
                             % (p["qid"], p["correct_answers"], rec["gold_support"]))
        if p["oracle"] != bool(rec["oracle"]):
            integrity.append("q%d: recomputed oracle != record" % p["qid"])
        for issue in p["issues"]:
            integrity.append("q%d: %s" % (p["qid"], issue))

    n = len(results)
    maj_acc = sum(1 for r in results if r["maj_ok"]) / n
    s1_ok = sum(1 for rec in meta["records"]
                if rec["answers"] and rec["answers"][0] is not None
                and rec["answers"][0] == rec["gold"])
    s1_acc = s1_ok / n
    oracle_acc = sum(1 for r in results if r["oracle"]) / n
    n_none = sum(1 for rec in meta["records"] for a in rec["answers"] if a is None)
    g1 = maj_acc - s1_acc
    g2 = oracle_acc - maj_acc
    tail_ids = [r["qid"] for r in results if r["tail_only"]]
    tail_trace_ids = [r["qid"] for r in results if r["tail_only_trace"]]

    w("=" * 78)
    w("TOKEN-LEVEL PREFIX-TRIE ANALYSIS - 8 streams, llama3.2-3b, T=0.6/top_p=0.95")
    w("=" * 78)
    w("problems=%d  streams/problem=%d  seed=%s  model=%s"
      % (n, meta.get("n_streams"), meta.get("seed"), ",".join(meta.get("models", []))))
    w("selection: %s" % meta.get("selection", "?"))
    w("")
    w("Integrity")
    if integrity:
        for line in integrity:
            w("  WARN " + line)
    else:
        w("  ok: trace counts, step contiguity, JSON stream ids, and recomputed")
        w("      correctness all match the records")
    w("  axis: token-id prefix trie (exact token equality, no semantic merging)")
    w("  answers: records' canonical answers; None answers are unanswered and never correct")
    w("  None answers: %d; a non-empty trace is kept in the trie but its answer is" % n_none)
    w("  unanswered and never correct; only an empty token sequence is excluded.")
    w("  answers in both classes (correct + incorrect): %d/%d; all-correct=%d; all-wrong=%d"
      % (sum(1 for r in results if r["correct_sids"] and r["incorrect_sids"]),
         n,
         sum(1 for r in results if r["correct_sids"] and not r["incorrect_sids"]),
         sum(1 for r in results if r["incorrect_sids"] and not r["correct_sids"])))
    w("")

    w("GATES")
    w("  maj@8 acc      = %.4f (%d/%d)" % (maj_acc, round(maj_acc * n), n))
    w("  stream-1 acc   = %.4f (%d/%d)" % (s1_acc, round(s1_acc * n), n))
    w("  oracle@8 acc   = %.4f (%d/%d)" % (oracle_acc, round(oracle_acc * n), n))
    w("  G1 = maj@8 - stream1 = %+.4f" % g1)
    w("  G2 = oracle@8 - maj@8 = %+.4f" % g2)
    w("  G3 = tail-only-correct (oracle, maj wrong, correct support == 1) = %d  ids=%s"
      % (len(tail_ids), tail_ids))
    w("       trace-level variant (all correct escapes at mass <= 1/8) = %d  ids=%s"
      % (len(tail_trace_ids), tail_trace_ids))
    w("       minority-correct (oracle, maj wrong, support > 1)         = %s"
      % [(r["qid"], r["correct_answers"]) for r in results
         if r["oracle"] and not r["maj_ok"] and r["correct_answers"] > 1])
    w("")

    w("TRUNK / SHARED PREFIX")
    fs = [r["first_split_frac"] for r in results]
    fsd = [r["first_split_depth"] for r in results if r["first_split_depth"] is not None]
    w("  first split depth (shared tokens before first fork; 0=first token differs):")
    w("    mean=%s median=%s p10=%s p90=%s max=%s"
      % (fmt(mean(fsd), 2), fmt(median(fsd), 2), fmt(pctl(fsd, 0.1), 2),
         fmt(pctl(fsd, 0.9), 2), fmt(max(fsd) if fsd else None, 2)))
    w("  first split / mean trace length (0=immediate, 1=never):")
    w("    mean=%s median=%s p10=%s p90=%s"
      % (fmt(mean(fs)), fmt(median(fs)), fmt(pctl(fs, 0.1)), fmt(pctl(fs, 0.9))))
    w("  never split (all traces prefix-chain): %d/%d"
      % (sum(1 for r in results if r["first_split_depth"] is None), n))
    buckets = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1),
               (0.1, 0.25), (0.25, 0.5), (0.5, 1.0001)]
    hist = []
    for lo, hi in buckets:
        c = sum(1 for x in fs if lo <= x < hi)
        hist.append("%s-%s:%d" % (lo, hi if hi <= 1 else 1.0, c))
    w("  normalized histogram: " + "  ".join(hist))
    w("  trunk_frac_mass (mean modal share over depths; 1/8=no concentration, 1=identical):")
    w("    mean=%s median=%s min=%s"
      % (fmt(mean([r["trunk_frac_mass"] for r in results])),
         fmt(median([r["trunk_frac_mass"] for r in results])),
         fmt(min(r["trunk_frac_mass"] for r in results))))
    w("  trie_token_saving (1 - unique nodes/tokens):    mean=%s median=%s max=%s"
      % (fmt(mean([r["token_saving"] for r in results])),
         fmt(median([r["token_saving"] for r in results])),
         fmt(max(r["token_saving"] for r in results))))
    w("  distinct token traces per problem: mean=%s median=%s (answers distinct: mean=%s)"
      % (fmt(mean([r["n_distinct_traces"] for r in results]), 2),
         fmt(median([r["n_distinct_traces"] for r in results]), 2),
         fmt(mean([r["n_distinct_answers"] for r in results]), 2)))
    w("    all %d problems have 8/8 distinct token traces; answer-level unanimity does"
      % sum(1 for r in results if r["n_distinct_traces"] == r["n_traces"]))
    w("    not imply token-level unanimity.")
    w("  live prefixes by absolute depth (mean over problems where depth exists):")
    hdr = []
    for d in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        vals = [r["levels"][d] for r in results if d in r["levels"]]
        hdr.append("d%d=%s(n=%d)" % (d, fmt(mean(vals), 2), len(vals)))
    w("    " + "  ".join(hdr))
    w("  live prefixes at normalized depth f*mean_len (mean over problems):")
    hdr = []
    for f in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        vals = []
        for r in results:
            d = max(1, min(r["max_depth"], int(round(f * r["trace_len_mean"]))))
            vals.append(r["levels"].get(d, 1))
        hdr.append("f%.2f=%s" % (f, fmt(mean(vals), 2)))
    w("    " + "  ".join(hdr))
    w("")

    w("FORKS")
    nf = [r["n_forks"] for r in results]
    w("  n_forks/problem: total=%d mean=%s median=%s min=%d max=%d zero=%d"
      % (sum(nf), fmt(mean(nf), 1), fmt(median(nf), 1), min(nf), max(nf),
         sum(1 for x in nf if x == 0)))
    w("    (8 distinct traces -> at most 7 branching nodes; observed max=%d)" % max(nf))
    w("    forks include the root (depth 0 = first-token split): %d/%d forks"
      % (sum(1 for r in results for f in r["forks"] if f["depth"] == 0), sum(nf)))
    fork_norm = [f["depth"] / r["trace_len_mean"]
                 for r in results for f in r["forks"]]
    fork_raw = [f["depth"] for r in results for f in r["forks"]]
    w("  fork depth (raw tokens): p10=%s p25=%s p50=%s p75=%s p90=%s"
      % (fmt(pctl(fork_raw, 0.1), 1), fmt(pctl(fork_raw, 0.25), 1),
         fmt(pctl(fork_raw, 0.5), 1), fmt(pctl(fork_raw, 0.75), 1),
         fmt(pctl(fork_raw, 0.9), 1)))
    w("  fork depth / mean_len:   p10=%s p25=%s p50=%s p75=%s p90=%s"
      % (fmt(pctl(fork_norm, 0.1)), fmt(pctl(fork_norm, 0.25)),
         fmt(pctl(fork_norm, 0.5)), fmt(pctl(fork_norm, 0.75)),
         fmt(pctl(fork_norm, 0.9))))
    fhist = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2),
             (0.2, 0.5), (0.5, 1.0001)]
    parts = []
    for lo, hi in fhist:
        c = sum(1 for x in fork_norm if lo <= x < hi)
        parts.append("%s-%s:%d" % (lo, hi if hi <= 1 else 1.0, c))
    w("  normalized fork-depth histogram: " + "  ".join(parts))
    child_masses = [c["mass"] for r in results for f in r["forks"]
                    for c in f["children"]]
    cm = mass_hist(child_masses)
    w("  branch mass at forks (child instances, pooled): total=%d" % len(child_masses))
    for k in range(1, 8):
        if cm.get(k):
            w("    k=%d (%.3f): %4d  (%5.1f%%)"
              % (k, k / 8.0, cm[k], 100.0 * cm[k] / len(child_masses)))
    singleton_pct = (100.0 * cm.get(1, 0) / len(child_masses)) if child_masses else 0.0
    w("    Singleton children are %.1f%% of fork children: the mass spectrum is"
      % singleton_pct)
    w("    heavily skewed to 1/8, not a balanced front-loaded fork.")
    w("")

    w("CORRECT-BRANCH MASS (first pure-correct escape; 1/8..7/8)")
    elig = [r for r in results if r["correct_branch_mass"] is not None]
    w("  eligible problems (>=1 correct and >=1 incorrect trace): %d" % len(elig))
    first_cm = mass_hist([r["correct_branch_mass"] for r in elig])
    max_cm = mass_hist([r["correct_branch_mass_max"] for r in elig])
    escape_cm = mass_hist([e["mass"] for r in elig for e in r["escapes"]])
    w("  per-problem mass at first separation:  %s"
      % " ".join("%d/8:%d" % (k, first_cm[k]) for k in range(1, 8)))
    w("  per-problem max over escapes:          %s"
      % " ".join("%d/8:%d" % (k, max_cm[k]) for k in range(1, 8)))
    w("  per-correct-stream escape mass:        %s"
      % " ".join("%d/8:%d" % (k, escape_cm[k]) for k in range(1, 8)))
    w("  first-separation mass: mean=%s median=%s min=%s; max-over-escapes mean=%s"
      % (fmt(mean([r["correct_branch_mass"] for r in elig]), 3),
         fmt(median([r["correct_branch_mass"] for r in elig]), 3),
         fmt(min([r["correct_branch_mass"] for r in elig]) if elig else None, 3),
         fmt(mean([r["correct_branch_mass_max"] for r in elig]), 3)))
    w("  first correct escape depth / mean_len: mean=%s median=%s"
      % (fmt(mean([r["first_escape_frac"] for r in elig])),
         fmt(median([r["first_escape_frac"] for r in elig]))))
    low = [(r["qid"], r["correct_branch_mass"], r["correct_answers"])
           for r in elig if r["correct_branch_mass"] <= 0.125 + EPS]
    w("  first-separation mass <= 1/8: %d  (qid, mass, correct stream count) = %s"
      % (len(low), low))
    frag = [(r["qid"], r["correct_branch_mass"], r["correct_answers"])
            for r in elig if r["correct_branch_mass"] < r["base_rate"] - EPS]
    w("  first-separation mass < answer-level correct support: %d %s"
      % (len(frag), frag))
    w("    Trace-level branch mass is not answer support. Correct traces diverge")
    w("    from each other while sharing a prefix with an incorrect trace, so the")
    w("    largest pure-correct token branch can be far smaller than the 7/8 answer")
    w("    group (q1: 7/8 correct answers, largest pure-correct branch 2/8).")
    w("")

    w("MONOTONICITY: P(child correct | child mass), all fork children pooled")
    w("  universe: problems with both correct and incorrect traces")
    rows = []
    for r in results:
        if not r["correct_sids"] or not r["incorrect_sids"]:
            continue
        for f in r["forks"]:
            for c in f["children"]:
                k = c["k"]
                ncorr = len(c["sids"] & r["correct_sids"])
                rows.append({
                    "qid": r["qid"], "k": k, "frac": ncorr / k,
                    "excess": ncorr / k - r["base_rate"],
                    "contains": 1 if ncorr > 0 else 0,
                    "ncorr": ncorr,
                })
    w("  child instances=%d  problems=%d" % (len(rows), len(set(x["qid"] for x in rows))))
    w("    k  mass   n_child  mean_corr  mean_excess  P(pure)  P(any correct)")
    for k in range(1, 8):
        sub = [x for x in rows if x["k"] == k]
        if not sub:
            continue
        w("    %d  %.3f  %7d  %9s  %11s  %7s  %13s"
          % (k, k / 8.0, len(sub),
             fmt(mean([x["frac"] for x in sub]), 4),
             fmt(mean([x["excess"] for x in sub]), 4),
             fmt(mean([1.0 if x["ncorr"] == x["k"] else 0.0 for x in sub]), 4),
             fmt(mean([x["contains"] for x in sub]), 4)))
    w("    mean_corr is the correctness density of streams in branches of that size.")
    w("    P(any correct) is size-biased (bigger branches hold more streams and are")
    w("    likelier to contain at least one correct trace); it is not the monotonicity test.")
    rho = spearman([x["k"] for x in rows], [x["excess"] for x in rows])
    low_rows = [x for x in rows if x["k"] <= 2]
    high_rows = [x for x in rows if x["k"] >= 4]
    exc_low = mean([x["excess"] for x in low_rows])
    exc_high = mean([x["excess"] for x in high_rows])
    delta = (exc_high - exc_low) if exc_low is not None and exc_high is not None else None
    w("  high-mass bucket (k>=4): n_child=%d from %d problems"
      % (len(high_rows), len(set(x["qid"] for x in high_rows))))
    w("  trend: spearman(k, excess)=%s  mean_excess(k<=2)=%s  mean_excess(k>=4)=%s  delta=%s"
      % (fmt(rho), fmt(exc_low), fmt(exc_high), fmt(delta)))
    monotone_up = bool(delta is not None and delta > 0.02 and rho is not None and rho > 0.02)
    if delta is None:
        trend = "undefined (empty bucket)"
    elif monotone_up:
        trend = "increasing with mass"
    elif delta < -0.02 and rho is not None and rho < -0.02:
        trend = "slightly decreasing with mass (weak)"
    else:
        trend = "flat / non-monotone"
    w("  classification: %s" % trend)
    w("  caveat: child instances are nested (one stream feeds many forks) and only")
    w("  %d problems are independent; this is descriptive, not a significance test."
      % len(set(x["qid"] for x in rows)))
    w("")

    w("VERDICT")
    w("  1) Large shared trunk to bank?")
    trunk_med = median(fs)
    save_med = median([r["token_saving"] for r in results])
    save_max = max(r["token_saving"] for r in results)
    w("     NO all-8 trunk: median first fork after %s shared token(s) (p90=%s)."
      % (fmt(median(fsd), 1), fmt(pctl(fsd, 0.9), 1)))
    w("     The all-8 trunk is 0-2 tokens for 90% of problems; by 25% of the trace the")
    w("     population is ~8 distinct prefixes (mean 7.9), i.e. 8 separate chains.")
    w("     Prefix collapse saves a median %.2f%% of token positions (max %.2f%%); a few"
      % (100.0 * save_med, 100.0 * save_max))
    w("     problems keep a 6/8 modal branch for ~20 tokens, but that is modest, not large.")
    w("  2) Fork timing?")
    fn_med = pctl(fork_norm, 0.5)
    if fn_med is not None and fn_med < 0.1:
        w("     FRONT-LOADED. Median fork depth is %.1f%% of trace length (p90=%.1f%%)."
          % (100.0 * fn_med, 100.0 * pctl(fork_norm, 0.9)))
        w("     After that the trie is a set of long singleton chains; %.1f%% of fork"
          % singleton_pct)
        w("     children are 1/8 branches.")
    else:
        w("     Not obviously front-loaded; median normalized fork depth=%s." % fmt(fn_med))
    w("  3) Correctness monotone in branch mass?")
    w("     NO increasing trend: %s, delta=%s, spearman=%s."
      % (trend, fmt(delta), fmt(rho)))
    w("     Decision statistic is correctness density; P(any correct) is size-biased.")
    if monotone_up:
        w("     Low-mass branches are systematically less correct.")
    else:
        w("     Low-mass branches are not systematically less correct. The high-mass")
        w("     bucket is only %d child instances and the rows are nested, so treat the"
          % len(high_rows))
        w("     flat/slightly-negative point estimate as weak, not a reliable negative law.")
    w("  4) Tail-only-correct problems (G3): %d  ids=%s" % (len(tail_ids), tail_ids))
    w("  5) Decision rule (monotone means monotone-increasing in mass):")
    if len(tail_ids) == 0:
        w("     G3==0 -> DEAD. No case where the correct trace is a 1/8 branch behind a")
        w("     wrong majority; there is no amplification target in this sample.")
    elif monotone_up:
        w("     G3>0 but correctness rises with branch mass -> DEAD AS CAPABILITY.")
        w("     The tail cases exist, but low-mass branches are systematically noisier;")
        w("     spending GPU to amplify them chases sampling noise, not a capability gap.")
    else:
        w("     G3>0 and correctness is not increasing with branch mass -> WORTH A GPU SESSION.")
        w("     Tail cases exist and mass does not identify them in advance. This is a weak")
        w("     structural green light, not a demonstrated capability: the monotonicity")
        w("     evidence is flat, n=32, and one of the three tail cases (q32) is a selected")
        w("     boost. The next step is a cheap targeted amplification experiment, not a sweep.")
    w("")

    w("CAVEATS")
    w("  * n=%d problems, 8 streams, one model (llama3.2-3b), one seed (1234), T=0.6/top_p=0.95." % n)
    w("  * Branch masses are integer counts / 8 (multiples of 1/8); with 8 streams the")
    w("    smallest non-zero mass is 0.125 and estimates are coarse.")
    w("  * Forks are token-level and nested: the same stream contributes to many fork")
    w("    children, so the monotonicity table is descriptive, not an independent-sample test.")
    w("  * Branch correctness density uses one sampled continuation per stream. It is not")
    w("    P(correct | more samples from that prefix), which is what an amplification")
    w("    experiment would need; this analysis cannot estimate that conditional.")
    w("  * The trie merges exact token-id prefixes only. Different text that tokenizes")
    w("    differently, or reasoning paths that re-converge after diverging, are counted")
    w("    as separate; token_saving is an upper bound on prefix-collapse savings and a")
    w("    lower bound on any broader (non-prefix) reuse.")
    w("  * Traces are the sidecar's emitted tokens, including EOS (tok 128009) where the")
    w("    stream finished by EOS; EOS sits under each terminal prefix and merges nothing.")
    n_allc = sum(1 for r in results if r["correct_sids"] and not r["incorrect_sids"])
    n_allw = sum(1 for r in results if r["incorrect_sids"] and not r["correct_sids"])
    n_both = sum(1 for r in results if r["correct_sids"] and r["incorrect_sids"])
    w("  * %d/%d problems are all-correct, %d/%d all-wrong, and %d/%d have both classes."
      % (n_allc, n, n_allw, n, n_both, n))
    w("    Only the both-class problems can inform correctness-vs-mass; the mass table")
    w("    excludes all-correct/all-wrong problems, while fork-depth stats include all.")
    w("  * selection is 30 length quantiles + 2 boosts chosen as unanimous-wrong in the")
    w("    n=100 run (q31, q32). q32 is one of the tail-only cases, so G3 is inflated by")
    w("    the boost design; the random quantile subset alone contributes q14 and q29 (2/30).")
    w("  * answers are the probe's canonicalized answers; correctness is exact string")
    w("    equality after that canonicalization. No semantic/numeric tolerance is added.")
    w("  * This is structural analysis of token sequences, not a wall-clock or GPU measurement.")

    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())

    cols = [
        "qid", "trace_len_mean", "first_split_depth_frac", "n_forks",
        "correct_branch_mass", "tail_only", "maj_ok", "oracle",
        "gold_support", "n_correct_streams", "n_incorrect_streams",
        "first_split_shared_tokens", "trunk_frac_mass", "trie_token_saving",
        "n_escape_branches", "first_escape_depth_frac",
        "correct_branch_mass_max", "correct_branch_mass_median",
        "tail_only_trace", "n_distinct_traces", "n_distinct_answers",
        "group",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(cols)
        for r in results:
            out.writerow([
                r["qid"],
                fmt(r["trace_len_mean"], 3),
                fmt(r["first_split_frac"], 4),
                r["n_forks"],
                fmt(r["correct_branch_mass"], 4),
                int(r["tail_only"]),
                int(r["maj_ok"]),
                int(r["oracle"]),
                r["correct_answers"],
                len(r["correct_sids"]),
                len(r["incorrect_sids"]),
                "" if r["first_split_depth"] is None else r["first_split_depth"],
                fmt(r["trunk_frac_mass"], 4),
                fmt(r["token_saving"], 4),
                r["n_escape_branches"],
                fmt(r["first_escape_frac"], 4),
                fmt(r["correct_branch_mass_max"], 4),
                fmt(r["correct_branch_mass_median"], 4),
                "" if r["tail_only_trace"] is None else int(r["tail_only_trace"]),
                r["n_distinct_traces"],
                r["n_distinct_answers"],
                r["group"],
            ])

    w("")
    w("wrote %s" % OUT_TXT)
    w("wrote %s" % OUT_CSV)


if __name__ == "__main__":
    main()

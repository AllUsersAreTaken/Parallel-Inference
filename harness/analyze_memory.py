"""Test 2 analysis: recall by arm/type, nested stage funnel, latency."""
import csv
import statistics
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "results/mem_full.csv"
rows = list(csv.DictReader(open(P)))
ARMS = {0: "B0 no-mem", 1: "B1 raw-BM25", 2: "B2 summarize", 3: "B3 full", 4: "B4 select"}
QT = ["direct", "paraphrased", "multi", "topic_jump"]


def rate(rs, key):
    v = [int(r[key]) for r in rs if r.get(key) not in ("", "-1", None)]
    return 100 * sum(v) / len(v) if v else float("nan")


print("recall (%) by arm x question type")
print(f"{'arm':>13} " + " ".join(f"{q:>12}" for q in QT) + f" {'all':>7} {'ins_tok':>8}")
for a in sorted({int(r['arm']) for r in rows}):
    rs = [r for r in rows if int(r['arm']) == a]
    cells = []
    for q in QT:
        sub = [r for r in rs if r['qtype'] == q]
        cells.append(f"{rate(sub,'recall'):>12.0f}" if sub else f"{'-':>12}")
    ins = statistics.median(int(r['inserted_tok']) for r in rs)
    print(f"{ARMS.get(a,a):>13} " + " ".join(cells) + f" {rate(rs,'recall'):>7.0f} {ins:>8.0f}")

print("\nnested stage funnel (conditional)")
print(f"{'arm':>13} {'P(cap)':>8} {'P(ret|cap)':>11} {'P(brief|ret)':>13} {'P(corr|brief)':>14}")
for a in (2, 4):
    rs = [r for r in rows if int(r['arm']) == a]
    if not rs:
        continue
    c = rate(rs, 'captured')
    s1 = [r for r in rs if r['captured'] == '1']
    r1 = rate(s1, 'retrieved')
    s2 = [r for r in s1 if r['retrieved'] == '1']
    b1 = rate(s2, 'brief_has')
    s3 = [r for r in s2 if r['brief_has'] == '1']
    u1 = rate(s3, 'use')
    print(f"{ARMS[a]:>13} {c:>8.0f} {r1:>11.0f} {b1:>13.0f} {u1:>14.0f}")

print("\nlatency by arm (median)")
print(f"{'arm':>13} {'ttft_ms':>9} {'total_ms':>9}")
for a in sorted({int(r['arm']) for r in rows}):
    rs = [r for r in rows if int(r['arm']) == a]
    print(f"{ARMS.get(a,a):>13} {statistics.median(float(r['ttft_ms']) for r in rs):>9.0f} "
          f"{statistics.median(float(r['total_ms']) for r in rs):>9.0f}")

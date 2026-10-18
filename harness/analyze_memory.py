"""Test 2 analysis: recall by arm and question type, stage diagnostics, latency."""
import csv
import statistics
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "results/mem_full.csv"
rows = list(csv.DictReader(open(P)))
ARMS = {0: "B0 no-mem", 1: "B1 raw-BM25", 2: "B2 memory", 3: "B3 full"}


def rate(rs, key):
    v = [int(r[key]) for r in rs if r[key] not in ("", "-1")]
    return 100 * sum(v) / len(v) if v else float("nan")


print("recall (%) by arm x question type")
qtypes = ["direct", "paraphrased", "multi", "topic_jump"]
print(f"{'arm':>12} " + " ".join(f"{q:>12}" for q in qtypes) + f" {'all':>7}")
for a in range(4):
    rs = [r for r in rows if int(r["arm"]) == a]
    cells = []
    for q in qtypes:
        sub = [r for r in rs if r["qtype"] == q]
        cells.append(f"{rate(sub,'recall'):>12.0f}" if sub else f"{'-':>12}")
    print(f"{ARMS[a]:>12} " + " ".join(cells) + f" {rate(rs,'recall'):>7.0f}")

print("\nB2 stage diagnostics (capture / retrieval / use), overall and by type")
rs2 = [r for r in rows if int(r["arm"]) == 2]
print(f"  capture   {rate(rs2,'capture'):.0f}%   retrieval {rate(rs2,'retrieval'):.0f}%   use {rate(rs2,'use'):.0f}%")
for q in qtypes:
    sub = [r for r in rs2 if r["qtype"] == q]
    if sub:
        print(f"  {q:>12}: capture {rate(sub,'capture'):.0f}%  retrieval {rate(sub,'retrieval'):.0f}%  use {rate(sub,'use'):.0f}%")

print("\nlatency / size by arm (median)")
print(f"{'arm':>12} {'ttft_ms':>9} {'total_ms':>9} {'reply_tok':>10} {'inserted_tok':>13}")
for a in range(4):
    rs = [r for r in rows if int(r["arm"]) == a]
    print(f"{ARMS[a]:>12} {statistics.median(float(r['ttft_ms']) for r in rs):>9.0f} "
          f"{statistics.median(float(r['total_ms']) for r in rs):>9.0f} "
          f"{statistics.median(int(r['reply_tok']) for r in rs):>10.0f} "
          f"{statistics.median(int(r['inserted_tok']) for r in rs):>13.0f}")

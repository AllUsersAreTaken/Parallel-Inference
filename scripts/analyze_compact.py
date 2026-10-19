"""Context-compaction demo analysis."""
import csv
import statistics

rows = list(csv.DictReader(open("S:/Agent/QSExplore/RecycleLM/results/compact_10.csv")))


def get(c):
    return [r for r in rows if r["config"] == c]


def med(rs, k):
    v = [float(r[k]) for r in rs if r[k] not in ("-1", "")]
    return statistics.median(v) if v else float("nan")


ser, par, fu = get("SERIAL"), get("PARALLEL"), get("FOLLOWUP")
print(f"n={len(ser)}  prefix_tokens median={med(ser,'prefix_tok'):.0f}")
print(f"{'metric':>14} {'SERIAL':>9} {'PARALLEL':>9} {'ratio':>7}")
for k, n in [("ans_ttft_ms", "ans TTFT"), ("ans_total_ms", "ans total"),
             ("mem_total_ms", "mem total"), ("both_ready_ms", "both ready"),
             ("mem_tok", "mem tokens")]:
    s, p = med(ser, k), med(par, k)
    print(f"{n:>14} {s:>9.0f} {p:>9.0f} {p/s:>6.2f}x")
print()
nf = len(fu)
print(f"follow-up (n={nf}):")
print(f"  full-context correct   {100*sum(int(r['full_follow_ok']) for r in fu)/nf:.0f}%")
print(f"  compacted correct      {100*sum(int(r['compact_follow_ok']) for r in fu)/nf:.0f}%")
print(f"  compacted asserted stale: {sum(int(r['compact_stale']) for r in fu)}")
fp, cp = med(fu, "full_prefix_tok"), med(fu, "compact_prefix_tok")
print(f"  follow-up prefix tokens: full {fp:.0f}  compact {cp:.0f}  reduction {100*(1-cp/fp):.0f}%")

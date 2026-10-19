"""Ensemble vote vs per-stream average (the low-variance N=1 estimate)."""
import csv

for tag in ["1b", "05b", "smol"]:
    rows = list(csv.DictReader(open(f"S:/Agent/QSExplore/RecycleLM/results/e2_{tag}_sa.csv")))
    ids = {r["id"] for r in rows}
    print(f"--- {tag} (n={len(ids)}) ---")
    print("   N   vote%   per-stream%   delta")
    for N in (1, 2, 4, 8, 16):
        rs = [r for r in rows if int(r["N"]) == N]
        vote = 100 * sum(int(r["correct"]) for r in rs) / len(rs)
        ps = 100 * sum(float(r["stream_acc"]) for r in rs) / len(rs)
        print(f"  {N:>2}  {vote:>6.1f}  {ps:>11.1f}  {vote-ps:>+7.1f}")

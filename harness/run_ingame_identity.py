"""Angle 32a - Slot 0 text-identity invariant (RecycleLM, follow-up to 31).

Reviewer debt M1 + the angle-31 seed-1237 artifact (0.881x latency ratio was
a 69-vs-58 generated-token text divergence at the same seed): does running
Slot 1 concurrently alter Slot 0's output text (batched-GEMM numerics), and
if so is the divergence lexical or semantic?

(a) FREE rescore of all existing angle-27..31 raws (results/ingame_*_raw/,
    5 solo + 5 batch each): Slot 0 text identity solo-vs-batch per seed.
    Both arms were mangled by the same deterministic locale-encoding path,
    so equality comparisons are valid; pairs touching U+FFFD are flagged.
(b) Fresh dedicated probe: angle-31 compaction prompts (solo = Slot 0 only,
    data/ingame_mem_solo.tsv; batch = k=2, data/ingame_bleed_batch.tsv;
    s0 prompt byte-verified identical at run time), 10 NEW seeds 2000-2009,
    temp 0.2, max_tokens 160. Per seed: s0 identical? generated-token counts
    solo vs batch; divergent pairs get difflib.SequenceMatcher ratio +
    first-divergence word index; full texts kept in the JSON.

Writes results/ingame_identity.json and results/ingame_identity.txt.
Fresh per-rep CLI JSON in results/ingame_identity_raw/.
Run: python harness\\run_ingame_identity.py [--fresh-reps N]
     [--historical-only | --fresh-only] [--rescore] [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingame_common import (  # noqa: E402
    MAX_TOKENS, MODEL, ROOT, SEEDS, S0_MARKER, SYSTEM, TEMP, TOP_P,
    cli_provenance, get_stream, load_raw, run_cli, s0_identity,
)

SOLO_TSV = "data/ingame_mem_solo.tsv"          # s0 prompt only (angle-27/31)
BATCH_TSV = "data/ingame_bleed_batch.tsv"       # s0 + s1 (angle-31 prompts)
RAW_DIR = ROOT / "results" / "ingame_identity_raw"
OUT_JSON = ROOT / "results" / "ingame_identity.json"
OUT_TXT = ROOT / "results" / "ingame_identity.txt"
FRESH_SEEDS = list(range(2000, 2010))

# (angle, raw dir, solo file stem) for the FREE historical rescore.
HISTORICAL = [
    (27, "results/ingame_memory_raw", "solo"),
    (28, "results/ingame_memory_nudge_raw", "solo"),
    (29, "results/ingame_letta_compact_raw", "solo0"),
    (30, "results/ingame_mention_probe_raw", "solo"),
    (31, "results/ingame_bleed_gate_raw", "solo0"),
]


def tsv_s0_prompt(path):
    """Return the s0 prompt string from a harness TSV (unescape \\n)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("s0\t"):
            return line.split("\t", 1)[1].replace("\\n", "\n").replace("\\t", "\t")
    raise KeyError(f"no s0 row in {path}")


def pair_record(seed, solo, batch):
    s0_solo = get_stream(solo, S0_MARKER)
    s0_batch = get_stream(batch, S0_MARKER)
    ident = s0_identity(s0_solo["text"], s0_batch["text"],
                        s0_solo["generated"], s0_batch["generated"])
    return {
        "seed": seed,
        **ident,
        "s0_solo_text": s0_solo["text"],
        "s0_batch_text": s0_batch["text"],
        "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
    }


def rescore_historical():
    angles = []
    for angle, raw_dir, stem in HISTORICAL:
        pairs = []
        for seed in SEEDS:
            raw = ROOT / raw_dir
            solo = load_raw(raw / f"rep{seed}_{stem}.json")
            batch = load_raw(raw / f"rep{seed}_batch.json")
            rec = pair_record(seed, solo, batch)
            rec["angle"] = angle
            pairs.append(rec)
        ident = sum(p["s0_identical"] for p in pairs)
        angles.append({
            "angle": angle,
            "raw_dir": raw_dir,
            "n": len(pairs),
            "identical": ident,
            "identity_rate": ident / len(pairs),
            "gen_deltas": [p["s0_gen_delta"] for p in pairs],
            "mojibake_flagged": sum(p["s0_mojibake_in_either"] for p in pairs),
            "pairs": pairs,
        })
    n = sum(a["n"] for a in angles)
    ident = sum(a["identical"] for a in angles)
    return {"angles": angles,
            "pooled": {"n": n, "identical": ident, "identity_rate": ident / n}}


def run_fresh(seeds, rescore):
    pairs = []
    for seed in seeds:
        if rescore:
            solo = load_raw(RAW_DIR / f"rep{seed}_solo.json")
            batch = load_raw(RAW_DIR / f"rep{seed}_batch.json")
        else:
            print(f"[run] seed={seed} solo (serial, Slot 0)", flush=True)
            solo = run_cli("serial", SOLO_TSV, seed, RAW_DIR / f"rep{seed}_solo.json")
            print(f"[run] seed={seed} batch k=2 (angle-31 compaction)", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rec = pair_record(seed, solo, batch)
        pairs.append(rec)
        print(f"[rep {seed}] identical={rec['s0_identical']} "
              f"gens={rec['s0_solo_generated']}->{rec['s0_batch_generated']} "
              f"sim={rec['s0_similarity']:.3f} "
              f"first_div_word={rec['s0_first_divergence_word']}", flush=True)
    ident = sum(p["s0_identical"] for p in pairs)
    return {"seeds": seeds, "n": len(pairs), "identical": ident,
            "identity_rate": ident / len(pairs), "pairs": pairs}


def divergent_pool(*pair_lists):
    out = []
    for pairs in pair_lists:
        out.extend(p for p in pairs if not p["s0_identical"])
    out.sort(key=lambda p: p["s0_similarity"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-reps", type=int, default=10)
    ap.add_argument("--historical-only", action="store_true")
    ap.add_argument("--fresh-only", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="re-analyze saved fresh raws without re-running the GPU")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = FRESH_SEEDS[: args.fresh_reps]
    if args.dry_run:
        print("plan: historical angles 27-31 (free rescore); fresh seeds",
              seeds, "solo", SOLO_TSV, "batch", BATCH_TSV)
        return 0

    s0_ok = tsv_s0_prompt(ROOT / SOLO_TSV) == tsv_s0_prompt(ROOT / BATCH_TSV)
    print(f"[check] s0 prompt byte-identical solo TSV vs batch TSV: {s0_ok}",
          flush=True)
    if not s0_ok:
        raise SystemExit("s0 prompt mismatch between solo and batch TSVs")

    historical = None if args.fresh_only else rescore_historical()
    fresh = None
    if not args.historical_only:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        fresh = run_fresh(seeds, args.rescore)

    pair_lists = []
    if historical:
        pair_lists.extend(a["pairs"] for a in historical["angles"])
    if fresh:
        pair_lists.append(fresh["pairs"])
    pooled_n = sum(len(p) for p in pair_lists)
    pooled_ident = sum(sum(x["s0_identical"] for x in p) for p in pair_lists)
    divergent = divergent_pool(*pair_lists)

    summary = {
        "experiment": ("Angle 32a Slot 0 text-identity invariant (batched-GEMM "
                       "numerics, solo vs batch k=2, same seed)"),
        "parent": "results/ingame_bleed_gate.json (angle 31)",
        "model": MODEL, "system": SYSTEM,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "metadata": cli_provenance(),
        "data_files": {"solo": SOLO_TSV, "batch": BATCH_TSV,
                       "note": "angle-31 prompts; s0 byte-verified identical"},
        "s0_prompt_identical_verified": s0_ok,
        "historical": historical,
        "fresh": fresh,
        "pooled": {"n": pooled_n, "identical": pooled_ident,
                   "identity_rate": pooled_ident / pooled_n if pooled_n else None},
        "divergent_pair_count": len(divergent),
        "divergent_pairs": divergent,
        "mojibake_caveat": (
            "historical raws were written by a locale-decoded run_cli; both "
            "arms were mangled identically so equality is valid, but pairs "
            "flagged s0_mojibake_in_either have lossy U+FFFD in text"
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = []
    L.append("Angle 32a - Slot 0 text-identity invariant: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} "
             f"system={SYSTEM!r}")
    L.append(f"cli_sha256={summary['metadata']['cli_sha256']}  "
             f"built={summary['metadata']['cli_built_mtime']}  "
             f"tree={summary['metadata']['git_tree']}")
    L.append(f"solo={SOLO_TSV}  batch={BATCH_TSV}  "
             f"s0_prompt_byte_identical={s0_ok}")
    L.append("")
    if historical:
        L.append("HISTORICAL rescore (angles 27-31, seeds 1234-1238, free):")
        L.append(f"{'angle':>5} {'dir':<34} {'ident':>5} {'rate':>6} "
                 f"{'gen_deltas(batch-solo)':<24} {'mojibake':>8}")
        for a in historical["angles"]:
            L.append(f"{a['angle']:>5} {a['raw_dir']:<34} "
                     f"{a['identical']:>2}/{a['n']:<2} "
                     f"{a['identity_rate']:>6.0%} "
                     f"{str(a['gen_deltas']):<24} {a['mojibake_flagged']:>8}")
        p = historical["pooled"]
        L.append(f"{'pool':>5} {'':<34} {p['identical']:>2}/{p['n']:<2} "
                 f"{p['identity_rate']:>6.0%}")
        L.append("")
    if fresh:
        L.append("FRESH probe (angle-31 prompts, seeds 2000-2009):")
        L.append(f"{'seed':>6} {'ident':>6} {'gens solo->batch':>17} "
                 f"{'sim':>6} {'div_word':>8}")
        for r in fresh["pairs"]:
            L.append(f"{r['seed']:>6} {str(r['s0_identical']):>6} "
                     f"{r['s0_solo_generated']:>7}->{r['s0_batch_generated']:<9} "
                     f"{r['s0_similarity']:>6.3f} "
                     f"{str(r['s0_first_divergence_word']):>8}")
        L.append(f"fresh identity rate: {fresh['identical']}/{fresh['n']} = "
                 f"{fresh['identity_rate']:.0%}")
        L.append("")
    L.append(f"POOLED identity rate (historical + fresh): "
             f"{summary['pooled']['identical']}/{summary['pooled']['n']} = "
             f"{summary['pooled']['identity_rate']:.0%}")
    L.append(f"divergent pairs: {len(divergent)}")
    L.append("")
    L.append("WORST divergences (lowest SequenceMatcher ratio; full texts):")
    for r in divergent[:3]:
        origin = (f"angle {r['angle']}" if "angle" in r else "fresh probe")
        L.append(f"--- {origin} seed {r['seed']} sim={r['s0_similarity']:.3f} "
                 f"first_div_word={r['s0_first_divergence_word']} "
                 f"gens={r['s0_solo_generated']}->{r['s0_batch_generated']}")
        L.append(f"  solo : {r['s0_solo_text']!r}")
        L.append(f"  batch: {r['s0_batch_text']!r}")
    text = "\n".join(L) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

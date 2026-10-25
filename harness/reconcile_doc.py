"""Deterministic evaluation for the first Reading-Compiler experiment.

Parses the per-view record streams produced by `app/prefix_cli.exe --views`,
scores provenance by verbatim substring lookup (no LLM judge), checks the
pre-registered key facts, and writes a reconciliation graph.

Usage:
    python harness/reconcile_doc.py \
        --run results/_rc_views.json \
        --doc data/_rc_doc.txt \
        --key data/_rc_key.json \
        --out results/_rc_graph.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

SCHEMAS: dict[str, dict[str, Any]] = {
    "v1_claims": {
        "prefix": "C|",
        "fields": ["claim", "quote"],
        "quote": 1,
        "kind": "claim",
    },
    "v2_entities": {
        "prefix": "E|",
        "fields": ["name", "alias", "type"],
        "quote": None,
        "kind": "entity",
    },
    "v3_numbers": {
        "prefix": "N|",
        "fields": ["value", "unit", "quote"],
        "quote": 2,
        "kind": "number",
    },
    "v4_obligations": {
        "prefix": "O|",
        "fields": ["actor", "action", "condition"],
        "quote": None,
        "kind": "obligation",
    },
    "v5_conflicts": {
        "prefix": "X|",
        "fields": ["fact_a", "fact_b", "quote"],
        "quote": 2,
        "kind": "conflict",
    },
    "v6_timeline": {
        "prefix": "T|",
        "fields": ["date", "event"],
        "quote": None,
        "kind": "event",
    },
    "v7_gaps": {
        "prefix": "Q|",
        "fields": ["question"],
        "quote": None,
        "kind": "gap",
    },
}

PROSE_VIEWS = {"v8_abstract"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def ascii_field_ok(s: str) -> bool:
    """Match the grammar's byte class: printable ASCII, pipe excluded."""
    return all(0x20 <= ord(c) <= 0x7E and c != "|" for c in s)


def parse_view(view_id: str, text: str) -> dict[str, Any]:
    """Split a view stream into complete records plus a trailing partial line."""
    if view_id in PROSE_VIEWS:
        return {"records": [], "rejected": 0, "partial": "", "conforming": 0}

    schema = SCHEMAS[view_id]
    lines = text.split("\n")
    partial = ""
    if text and not text.endswith("\n"):
        partial = lines.pop()

    records: list[dict[str, Any]] = []
    rejected = 0
    for line in lines:
        if not line:
            continue
        if not line.startswith(schema["prefix"]):
            rejected += 1
            continue
        body = line[len(schema["prefix"]):]
        parts = body.split("|")
        if len(parts) != len(schema["fields"]) or not all(ascii_field_ok(p) for p in parts):
            rejected += 1
            continue
        rec = {"view": view_id, "raw": line}
        rec.update(dict(zip(schema["fields"], parts)))
        records.append(rec)

    conforming = len(records)
    return {
        "records": records,
        "rejected": rejected,
        "partial": partial,
        "conforming": conforming,
    }


def find_span(quote: str, doc: str, doc_bytes: bytes) -> dict[str, int] | None:
    q = quote.strip()
    if not q:
        return None
    pos = doc_bytes.find(q.encode("utf-8"))
    if pos < 0:
        return None
    return {"byte_start": pos, "byte_end": pos + len(q.encode("utf-8"))}


def split_aliases(field: str) -> list[str]:
    raw = field.strip()
    if not raw or raw in {"-", "->", "none", "None"}:
        return []
    parts = re.split(r"\s*->\s*", raw)
    return [p for p in (x.strip() for x in parts) if p and p not in {"-", "None", "none"}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "results" / "_rc_views.json"))
    ap.add_argument("--doc", default=str(ROOT / "data" / "_rc_doc.txt"))
    ap.add_argument("--key", default=str(ROOT / "data" / "_rc_key.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "_rc_graph.json"))
    ap.add_argument("--min-quote", type=int, default=20,
                    help="diagnostic threshold for a 'substantive' quote")
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    doc = Path(args.doc).read_text(encoding="utf-8")
    doc_bytes = Path(args.doc).read_bytes()
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))

    streams = {s["id"]: s for s in run.get("streams", [])}
    doc_norm = norm(doc)
    views: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []

    for view_id, stream in streams.items():
        parsed = parse_view(view_id, stream.get("text", ""))
        records = parsed["records"]
        schema = SCHEMAS.get(view_id)
        quoted = 0
        verified = 0
        verified_norm = 0
        for rec in records:
            rec["verified"] = None
            rec["verified_normalized"] = None
            rec["span"] = None
            if schema and schema["quote"] is not None:
                q = rec[schema["fields"][schema["quote"]]]
                rec["quote"] = q.strip()
                quoted += 1
                span = find_span(q, doc, doc_bytes)
                if span:
                    verified += 1
                    rec["verified"] = True
                    rec["span"] = span
                else:
                    rec["verified"] = False
                if rec["quote"] and norm(rec["quote"]) in doc_norm:
                    verified_norm += 1
                    rec["verified_normalized"] = True
                else:
                    rec["verified_normalized"] = False
            all_records.append(rec)

        grammar_ok = parsed["rejected"] == 0
        views[view_id] = {
            "generated": stream.get("generated"),
            "finish": stream.get("finish"),
            "suffix_tokens": stream.get("suffix_tokens"),
            "records": len(records),
            "rejected": parsed["rejected"],
            "grammar_ok": grammar_ok,
            "quoted_records": quoted,
            "quotes_verified": verified,
            "quotes_verified_normalized": verified_norm,
            "truncated": bool(parsed["partial"]) and view_id not in PROSE_VIEWS,
            "partial_tail": parsed["partial"][-80:],
            "is_prose": view_id in PROSE_VIEWS,
        }

    quote_total = sum(v["quoted_records"] for v in views.values())
    quote_ok = sum(v["quotes_verified"] for v in views.values())
    quote_ok_norm = sum(v["quotes_verified_normalized"] for v in views.values())
    substantive = [r for r in all_records
                   if r.get("quote") and len(r["quote"]) >= args.min_quote]
    substantive_ok = sum(1 for r in substantive if r.get("verified"))

    # Key-fact recall: the quote substring must appear inside a parsed record or,
    # for the prose view, inside its free text.
    fact_rows = []
    for fact in key["facts"]:
        q = fact["quote"]
        in_doc = q in doc
        caught_by = []
        for view_id, stream in streams.items():
            text = stream.get("text", "")
            if view_id in PROSE_VIEWS:
                if q in text:
                    caught_by.append(view_id)
                continue
            for rec in all_records:
                if rec["view"] != view_id:
                    continue
                if q in rec["raw"]:
                    caught_by.append(view_id)
                    break
        fact_rows.append({
            "id": fact["id"],
            "fact": fact["fact"],
            "quote": q,
            "in_doc": in_doc,
            "caught_by": caught_by,
            "recalled": bool(caught_by),
        })
    recall = sum(1 for f in fact_rows if f["recalled"])

    # Reconcile pass: V2 entities -> nodes, V1/V4/V6 -> edges.
    entities: dict[str, dict[str, Any]] = {}
    for rec in all_records:
        if rec["view"] != "v2_entities":
            continue
        canon = rec.get("name", "").strip()
        if not canon:
            continue
        nid = norm(canon)
        node = entities.setdefault(nid, {
            "id": nid,
            "label": canon,
            "type": rec.get("type", "").strip(),
            "aliases": [],
            "provenance": [],
        })
        if rec.get("type"):
            node["type"] = rec["type"].strip()
        for alias in split_aliases(rec.get("alias", "")):
            if norm(alias) != nid and alias not in node["aliases"]:
                node["aliases"].append(alias)
        span = find_span(canon, doc, doc_bytes)
        if span and span not in node["provenance"]:
            node["provenance"].append(span)

    edges: list[dict[str, Any]] = []
    for rec in all_records:
        span = rec.get("span")
        if rec["view"] == "v1_claims" and rec.get("claim"):
            edges.append({
                "kind": "claim",
                "from": "document",
                "predicate": "asserts",
                "to": rec["claim"].strip(),
                "quote": rec.get("quote", ""),
                "verified": rec.get("verified"),
                "span": span,
            })
        elif rec["view"] == "v4_obligations" and rec.get("actor"):
            edges.append({
                "kind": "obligation",
                "from": rec["actor"].strip(),
                "predicate": rec.get("action", "").strip() or "must",
                "to": rec.get("condition", "").strip(),
                "verified": None,
                "span": span,
            })
        elif rec["view"] == "v6_timeline" and rec.get("date"):
            edges.append({
                "kind": "event",
                "from": rec["date"].strip(),
                "predicate": "event",
                "to": rec.get("event", "").strip(),
                "verified": None,
                "span": span,
            })

    # Make every edge endpoint a node so the graph is self-contained.
    for e in edges:
        for side in ("from", "to"):
            label = e[side]
            if not label or label == "document":
                continue
            nid = norm(label)
            if nid not in entities:
                entities[nid] = {
                    "id": nid,
                    "label": label,
                    "type": "mention",
                    "aliases": [],
                    "provenance": [],
                }

    views_with_3 = sum(1 for v in views.values() if v["is_prose"] is False and v["records"] >= 3)
    truncated_views = [k for k, v in views.items() if v["truncated"]]
    grammar_bad = [k for k, v in views.items() if not v["is_prose"] and not v["grammar_ok"]]
    overall_quote_pct = 100.0 * quote_ok / quote_total if quote_total else 0.0
    overall_quote_pct_norm = 100.0 * quote_ok_norm / quote_total if quote_total else 0.0

    print(f"{'view':<16}{'gen':>5}{'finish':>13}{'rec':>5}{'rej':>5}{'gram':>6}{'quotes':>9}{'norm':>7}{'trunc':>7}")
    for view_id, v in views.items():
        quotes = "-" if v["is_prose"] or not v["quoted_records"] else f"{v['quotes_verified']}/{v['quoted_records']}"
        norm_s = "-" if v["is_prose"] or not v["quoted_records"] else str(v["quotes_verified_normalized"])
        gram = "n/a" if v["is_prose"] else ("ok" if v["grammar_ok"] else "BAD")
        trunc = "yes" if v["truncated"] else "no"
        print(f"{view_id:<16}{v['generated']:>5}{str(v['finish']):>13}{v['records']:>5}"
              f"{v['rejected']:>5}{gram:>6}{quotes:>9}{norm_s:>7}{trunc:>7}")

    print()
    print(f"QUOTE VERIFICATION: {quote_ok}/{quote_total} ({overall_quote_pct:.1f}%) "
          f"[whitespace-normalized {quote_ok_norm}/{quote_total}, {overall_quote_pct_norm:.1f}%]")
    print(f"SUBSTANTIVE QUOTES (>= {args.min_quote} chars): {substantive_ok}/{len(substantive)}")
    print(f"KEY-FACT RECALL:    {recall}/{len(fact_rows)}")
    for f in fact_rows:
        mark = "ok " if f["recalled"] else "MISS"
        print(f"  [{mark}] {f['id']:<4} doc={'yes' if f['in_doc'] else 'NO ':<3} {f['fact']}"
              f"  <- {','.join(f['caught_by']) or '-'}")
    print(f"VIEWS >=3 RECORDS:  {views_with_3}/{len(views)}")
    print(f"TRUNCATED VIEWS:    {len(truncated_views)} {truncated_views}")
    print(f"GRAMMAR FAILURES:   {grammar_bad or 'none'}")
    print(f"WALL: {run.get('wall_ms', 0):.0f} ms  PREFILL: {run.get('prefill_ms', 0):.0f} ms  "
          f"PREFIX: {run.get('shared_prefix_tokens', 0)} tok")

    success = (
        not grammar_bad
        and views_with_3 >= 4
        and overall_quote_pct >= 80.0
        and recall >= 8
    )
    dead = (
        len(truncated_views) > 2
        or (quote_total > 0 and overall_quote_pct < 50.0)
        or recall < 5
    )
    verdict = "SUCCESS" if success else ("DEAD END" if dead else "MIXED")
    print(f"VERDICT: {verdict}")

    graph = {
        "doc": str(Path(args.doc)),
        "doc_bytes": len(doc_bytes),
        "run": {
            "wall_ms": run.get("wall_ms"),
            "prefill_ms": run.get("prefill_ms"),
            "shared_prefix_tokens": run.get("shared_prefix_tokens"),
            "mode": run.get("mode"),
            "n": run.get("n"),
        },
        "views": views,
        "totals": {
            "records": sum(v["records"] for v in views.values()),
            "quoted_records": quote_total,
            "quotes_verified": quote_ok,
            "quote_verification_pct": round(overall_quote_pct, 2),
            "quotes_verified_normalized": quote_ok_norm,
            "quote_verification_pct_normalized": round(overall_quote_pct_norm, 2),
            "substantive_quotes": len(substantive),
            "substantive_quotes_verified": substantive_ok,
            "substantive_min_chars": args.min_quote,
            "key_fact_recall": recall,
            "key_facts": len(fact_rows),
            "views_with_3_records": views_with_3,
            "truncated_views": truncated_views,
            "grammar_failures": grammar_bad,
            "verdict": verdict,
        },
        "key_facts": fact_rows,
        "nodes": list(entities.values()),
        "edges": edges,
        "records": all_records,
    }
    out = Path(args.out)
    out.write_text(json.dumps(graph, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"graph -> {out}")


if __name__ == "__main__":
    main()

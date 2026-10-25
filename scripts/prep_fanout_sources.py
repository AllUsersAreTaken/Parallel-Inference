"""Build data/fanout_sources_20.tsv from the two frozen English inputs.

First 10 conversation rows from data/compact_convs.tsv and first 10 questions
from data/gsm8k_dev200.tsv. The inputs are read-only. A sidecar JSON records
the exact source lines, fields, sizes, and SHA-256 hashes.
"""

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPACT = ROOT / "data" / "compact_convs.tsv"
GSM = ROOT / "data" / "gsm8k_dev200.tsv"
OUT = ROOT / "data" / "fanout_sources_20.tsv"
PROV = ROOT / "data" / "fanout_sources_20.provenance.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return [line.rstrip("\r\n") for line in f]


def pick_conversation(fields):
    for i, value in enumerate(fields):
        if value.startswith("user:"):
            return i, value
    if len(fields) > 1:
        return 1, fields[1]
    return 0, fields[0]


def main():
    compact_lines = read_lines(COMPACT)
    gsm_lines = read_lines(GSM)

    sources = []
    items = []

    for lineno, line in enumerate(compact_lines[:10], start=1):
        fields = line.split("\t")
        field_index, text = pick_conversation(fields)
        if not text:
            sys.exit(f"empty conversation at {COMPACT.name}:{lineno}")
        sources.append(text)
        items.append({
            "source": COMPACT.name,
            "line": lineno,
            "field_1based": field_index + 1,
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })

    for lineno, line in enumerate(gsm_lines[:10], start=1):
        fields = line.split("\t")
        text = fields[0]
        if not text:
            sys.exit(f"empty question at {GSM.name}:{lineno}")
        sources.append(text)
        items.append({
            "source": GSM.name,
            "line": lineno,
            "field_1based": 1,
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })

    if len(sources) != 20:
        sys.exit(f"expected 20 sources, got {len(sources)}")

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for text in sources:
            if "\n" in text or "\r" in text or "\t" in text:
                sys.exit("source text contains a line or field separator")
            f.write(text + "\n")

    provenance = {
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha256(OUT),
        "inputs": {
            str(COMPACT.relative_to(ROOT)).replace("\\", "/"): {
                "sha256": sha256(COMPACT),
                "lines_used": 10,
            },
            str(GSM.relative_to(ROOT)).replace("\\", "/"): {
                "sha256": sha256(GSM),
                "lines_used": 10,
            },
        },
        "selection": (
            "compact_convs.tsv: first 10 lines, conversation field selected by "
            "the field beginning with 'user:' (field 2 in the current file). "
            "gsm8k_dev200.tsv: first 10 lines, question field 1."
        ),
        "languages": {
            "fra": "French", "spa": "Spanish", "deu": "German", "ita": "Italian",
            "por": "Portuguese", "rus": "Russian", "zho": "Chinese", "jpn": "Japanese",
        },
        "items": items,
    }
    with PROV.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {OUT} ({len(sources)} sources)")
    print(f"wrote {PROV}")


if __name__ == "__main__":
    main()

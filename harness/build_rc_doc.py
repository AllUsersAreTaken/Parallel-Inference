"""Build data/_rc_doc.txt for the Reading-Compiler experiment.

Concatenates real repository prose in a fixed order, then trims from the end at
a line boundary until the body fits a byte budget. The model token count is
reported separately by a 1-token calibration run of app/prefix_cli.exe.

Usage:
    python harness/build_rc_doc.py [--bytes N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CAPACITY_BLOCK = """## Context capacity (parallel/README.md)

Covers: explicit capacity exhaustion (no silent truncation), relative token cap
vs. truncation, branch isolation, and cancel-one-branch. Set `PARALLEL_DEBUG=1`
to trace each tick.

parallel::RuntimeConfig cfg;
cfg.model_path = "model.gguf";
cfg.max_context_tokens = 8192;
"""

SOURCES = [
    ("AGENTS.md", None),
    ("capacity", CAPACITY_BLOCK),
    ("PREREGISTRATION.md", None),
    ("app/PLAYGROUND.md", None),
    ("ASYNC_SELFCORRECT_DEMO.md", None),
]


def build(bytes_budget: int) -> tuple[str, list[dict]]:
    parts: list[dict] = []
    for name, fixed in SOURCES:
        if fixed is not None:
            text, path = fixed, name
        else:
            p = ROOT / name
            if not p.exists():
                continue
            text, path = p.read_text(encoding="utf-8"), name
        parts.append({"source": path, "chars": len(text)})

    body = "\n\n".join(
        fixed if fixed is not None else (ROOT / name).read_text(encoding="utf-8")
        for name, fixed in SOURCES
        if fixed is not None or (ROOT / name).exists()
    )
    if len(body.encode("utf-8")) > bytes_budget:
        cut = body.encode("utf-8")[:bytes_budget].decode("utf-8", errors="ignore")
        nl = cut.rfind("\n")
        if nl > 0:
            cut = cut[: nl + 1]
        body = cut.rstrip() + "\n"
    if not body.endswith("\n"):
        body += "\n"
    return body + "\n", parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=22000,
                    help="UTF-8 byte budget for the concatenated body")
    ap.add_argument("--out", default=str(ROOT / "data" / "_rc_doc.txt"))
    args = ap.parse_args()

    body, parts = build(args.bytes)
    out = Path(args.out)
    out.write_text(body, encoding="utf-8", newline="\n")
    print(json.dumps({
        "out": str(out),
        "bytes": len(body.encode("utf-8")),
        "sources": parts,
    }, indent=2))


if __name__ == "__main__":
    main()

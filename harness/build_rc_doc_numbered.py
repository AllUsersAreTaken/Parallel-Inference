"""Build data/_rc_doc_numbered.txt for the question-driven Reading Compiler run.

Turns data/_rc_doc.txt into plain numbered lines:

  * markdown headers, bold markers, bullets and inline code are stripped;
  * hard-wrapped paragraphs are unwrapped into one line (blank-line separated);
  * code fences are dropped but their lines stay separate;
  * every surviving line is prefixed with ``[N] `` (1-based).

The numbered text is used verbatim as the shared prefix of app/prefix_cli.exe.
Token count is measured separately by a 1-token calibration run; use
``--max-lines`` to trim from the end until the prefix fits the budget.

Usage:
    python harness/build_rc_doc_numbered.py [--max-lines N]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ASCII_MAP = {
    "\u2014": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
    "\u2192": "->",
}

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
BULLET_RE = re.compile(r"^[-*+]\s+(.*)$")
NUMBER_RE = re.compile(r"^\d+[.)]\s+(.*)$")
RULE_RE = re.compile(r"^([-*_])\1{2,}$")


def _ascii(text: str) -> str:
    for src, dst in ASCII_MAP.items():
        text = text.replace(src, dst)
    return text


def _inline(text: str) -> str:
    text = _ascii(text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(text: str) -> list[str]:
    lines: list[str] = []
    para: list[str] = []
    in_code = False

    def flush() -> None:
        nonlocal para
        if para:
            joined = _inline(" ".join(para))
            if joined:
                lines.append(joined)
            para = []

    for raw in text.splitlines():
        stripped = _ascii(raw).strip()
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            if stripped:
                lines.append(_inline(stripped))
            continue
        if not stripped:
            flush()
            continue
        if RULE_RE.fullmatch(stripped):
            flush()
            continue

        m = HEADER_RE.match(stripped)
        if m:
            flush()
            title = _inline(m.group(2))
            if title:
                lines.append(title)
            continue

        m = BULLET_RE.match(stripped)
        if m:
            flush()
            para = [m.group(1)]
            continue

        m = NUMBER_RE.match(stripped)
        if m:
            flush()
            para = [m.group(1)]
            continue

        if stripped.startswith(">"):
            stripped = stripped.lstrip(">").strip()
        if stripped.startswith("|"):
            flush()
            row = _inline(stripped.strip("|").replace("|", " "))
            if row:
                lines.append(row)
            continue

        para.append(stripped)

    flush()
    return lines


def numbered(lines: list[str], max_lines: int | None = None) -> tuple[str, list[str]]:
    kept = lines if max_lines is None else lines[:max_lines]
    body = "\n".join(f"[{i}] {line}" for i, line in enumerate(kept, 1))
    return body + "\n", kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "_rc_doc.txt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "_rc_doc_numbered.txt"))
    ap.add_argument("--max-lines", type=int, default=0,
                    help="keep only the first N numbered lines (0 = all)")
    args = ap.parse_args()

    src = Path(args.src)
    lines = preprocess(src.read_text(encoding="utf-8"))
    body, kept = numbered(lines, args.max_lines or None)

    out = Path(args.out)
    out.write_text(body, encoding="utf-8", newline="\n")
    print(json.dumps({
        "out": str(out),
        "source_lines": len(lines),
        "kept_lines": len(kept),
        "chars": len(body),
        "bytes": len(body.encode("utf-8")),
        "max_line_chars": max((len(x) for x in kept), default=0),
    }, indent=2))


if __name__ == "__main__":
    main()

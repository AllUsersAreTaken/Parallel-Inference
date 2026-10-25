"""Build a bug-repair eval from HumanEval: inject a bug into each canonical
solution, keep only mutants that actually fail the official tests."""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "_humaneval_canon.jsonl")
OUT = os.path.join(ROOT, "data", "_repair_eval.jsonl")

API = ("https://datasets-server.huggingface.co/rows?dataset=openai%2Fopenai_humaneval"
       "&config=openai_humaneval&split=test&offset=0&length=100")


def fetch():
    if os.path.exists(CACHE):
        return [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    d = json.load(urllib.request.urlopen(API, timeout=60))
    rows = [r["row"] for r in d["rows"]]
    with open(CACHE, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def mutants(code):
    yield code.replace("==", "!=", 1)
    yield code.replace("!=", "==", 1)
    yield re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1), code, count=1)
    yield code.replace(" < ", " <= ", 1)
    yield code.replace(" > ", " >= ", 1)
    yield code.replace(" and ", " or ", 1)
    yield code.replace(" + ", " - ", 1)
    yield code.replace(" - ", " + ", 1)
    yield code.replace("return True", "return False", 1)
    yield code.replace("return False", "return True", 1)


def passes(sol, row):
    prog = row["prompt"] + sol + "\n" + row["test"] + "\n\ncheck(%s)\n" % row["entry_point"]
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(prog)
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    rows = fetch()
    kept = 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for row in rows:
            canon = row["canonical_solution"]
            if not passes(canon, row):
                continue
            for m in mutants(canon):
                if m == canon:
                    continue
                if passes(m, row):
                    continue
                fh.write(json.dumps({
                    "task_id": row["task_id"], "prompt": row["prompt"],
                    "buggy": row["prompt"] + m, "canonical": canon,
                    "test": row["test"], "entry_point": row["entry_point"],
                }) + "\n")
                kept += 1
                break
    print("problems with a verified failing mutant: %d / %d -> %s" % (kept, len(rows), OUT))


if __name__ == "__main__":
    main()

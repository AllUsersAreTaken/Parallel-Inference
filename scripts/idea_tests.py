#!/usr/bin/env python3
"""Smoke-run 15 multi-stream use cases on the existing CLI (no architecture changes).

Each idea is one shared source plus N instruction variants, submitted as one
batched run. The script writes the TSV, runs parallel_cli.exe, and reports
whether each stream produced the expected shape.
"""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "app", "parallel_cli.exe")
MODEL = os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
DATA = os.path.join(ROOT, "app", "_ideas_in.tsv")
OUT = os.path.join(ROOT, "app", "_ideas_out.json")
RESULT = os.path.join(ROOT, "results", "idea_tests.json")

ARTICLE = ("The city council approved a new housing plan on Tuesday. The plan will add "
           "2,000 homes over five years, including 600 affordable units. Construction is "
           "expected to begin next spring. Critics say the plan does not do enough for renters.")
MUG = ("A $49 smart mug keeps coffee at 55 C for four hours, charges on a coaster, and has "
       "a 10-hour battery.")
SCHEMA = ("users(id, name, email, created_at); orders(id, user_id, total, status, created_at)")
CODE = ("def get_user(conn, name):\n    return conn.execute(f\"SELECT * FROM users "
        "WHERE name='{name}'\")")
LOGS = ("At 14:02 API 5xx rose from 0.1% to 12%; p99 latency 8 s; DB CPU 95%; connection "
        "pool exhausted. Deploy happened at 13:55.")
BOT = ("A customer-support chatbot can read orders, issue refunds up to $50, and send email.")
POLICY = ("The city proposes a 15-minute-city plan: cars restricted in the center, more bike "
          "lanes, and 20-minute transit targets.")
TICKET = ("My order arrived damaged. The box was crushed and the mug is cracked. I want a refund.")
LAUNCH = ("We are launching the Ember mug: $49, 55 C for four hours, coaster charging, "
          "10-hour battery, available in five colors.")
LOGIN = ("A login screen with email+password, SSO, remember-me, password reset, and a "
         "sign-up link.")

LANGS14 = ["Spanish", "French", "German", "Portuguese", "Italian", "Dutch", "Polish",
           "Japanese", "Korean", "Chinese", "Arabic", "Hindi", "Russian", "Turkish"]
LANGS15 = LANGS14 + ["Vietnamese"]

IDEAS = [
    dict(name="2-language release", n=2, source=LAUNCH,
         prompts=[f"Translate this announcement into {l}. Output only the translation.\n\n{LAUNCH}"
                  for l in ["Spanish", "French"]],
         check=lambda t, i: len(t) > 40),
    dict(name="3-format summary", n=3, source=ARTICLE,
         prompts=[f"Summarize the article as exactly one sentence.\n\n{ARTICLE}",
                  f"Summarize the article as exactly three bullet points.\n\n{ARTICLE}",
                  f"Write an executive brief of the article in 2-3 sentences.\n\n{ARTICLE}"],
         check=lambda t, i: ("-" in t) if i == 1 else (t.count(".") <= 4)),
    dict(name="4 reading levels", n=4, source=ARTICLE,
         prompts=[f"Explain the article at grade-3 reading level.\n\n{ARTICLE}",
                  f"Explain the article at grade-8 reading level.\n\n{ARTICLE}",
                  f"Explain the article at high-school reading level.\n\n{ARTICLE}",
                  f"Explain the article for an expert policy audience.\n\n{ARTICLE}"],
         check=lambda t, i: len(t) > 40),
    dict(name="5 audiences", n=5, source=LOGS,
         prompts=[f"Explain the incident to a CEO in 3 sentences.\n\n{LOGS}",
                  f"Explain the incident to an on-call engineer.\n\n{LOGS}",
                  f"Explain the incident to a customer.\n\n{LOGS}",
                  f"Explain the incident to a regulator.\n\n{LOGS}",
                  f"Explain the incident to a journalist.\n\n{LOGS}"],
         check=lambda t, i: len(t) > 40),
    dict(name="6 SQL dialects", n=6, source=SCHEMA,
         prompts=[f"Write a {d} query for the top 10 users by total order value in 2024. "
                  f"Schema: {SCHEMA}" for d in
                  ["PostgreSQL", "MySQL", "SQLite", "BigQuery", "Snowflake", "DuckDB"]],
         check=lambda t, i: "select" in t.lower()),
    dict(name="7 code-review lenses", n=7, source=CODE,
         prompts=[f"Review this code for {l} issues. Be specific.\n\n{CODE}" for l in
                  ["security", "performance", "readability", "tests", "API design",
                   "privacy", "reliability"]],
         check=lambda t, i: len(t) > 40),
    dict(name="8 marketing channels", n=8, source=MUG,
         prompts=[f"Write {c} copy for this product.\n\n{MUG}" for c in
                  ["an email", "a landing page", "an X post", "a LinkedIn post",
                   "an SMS", "a push notification", "a Google ad", "a press release"]],
         check=lambda t, i: len(t) > 20),
    dict(name="9 schema formats", n=9, source=SCHEMA,
         prompts=[f"Convert this schema to {f}.\n\n{SCHEMA}" for f in
                  ["JSON", "YAML", "TOML", "SQL DDL", "CSV", "a Markdown table",
                   "XML", "a GraphQL schema", "TypeScript interfaces"]],
         check=lambda t, i: len(t) > 20),
    dict(name="10 incident hypotheses", n=10, source=LOGS,
         prompts=[f"Give the most likely {h} hypothesis for this incident and one check.\n\n{LOGS}"
                  for h in ["network", "disk", "memory", "auth", "config", "dependency",
                            "capacity", "DNS", "TLS", "deploy"]],
         check=lambda t, i: len(t) > 30),
    dict(name="11 red-team angles", n=11, source=BOT,
         prompts=[f"Red-team this system for {a}. Give one concrete attack and a mitigation.\n\n{BOT}"
                  for a in ["prompt injection", "data exfiltration", "resource exhaustion",
                            "privilege escalation", "supply chain", "model inversion",
                            "denial of service", "social engineering", "tool abuse",
                            "PII leakage", "audit evasion"]],
         check=lambda t, i: len(t) > 30),
    dict(name="12 test types", n=12, source="A URL shortener: POST /shorten, GET /{code} redirects.",
         prompts=[f"Write one {t} test for this URL shortener. Include the assertion.\n\n"
                  "A URL shortener: POST /shorten, GET /{code} redirects." for t in
                  ["happy-path", "edge-case", "property-based", "integration", "security",
                   "performance", "accessibility", "regression", "fuzz", "concurrency",
                   "idempotency", "rollback"]],
         check=lambda t, i: ("assert" in t.lower()) or ("def test" in t.lower())),
    dict(name="13 policy perspectives", n=13, source=POLICY,
         prompts=[f"Analyze this policy from a {p} perspective in 2 sentences.\n\n{POLICY}" for p in
                  ["economist", "ethicist", "engineer", "lawyer", "citizen", "regulator",
                   "historian", "security", "environment", "business", "accessibility",
                   "public health", "media"]],
         check=lambda t, i: len(t) > 30),
    dict(name="14 support languages", n=14, source=TICKET,
         prompts=[f"Reply to this customer in {l}. Be concise and apologetic.\n\n{TICKET}"
                  for l in LANGS14],
         check=lambda t, i: len(t) > 30),
    dict(name="15 launch languages", n=15, source=LAUNCH,
         prompts=[f"Translate this launch announcement into {l}. Output only the translation.\n\n{LAUNCH}"
                  for l in LANGS15],
         check=lambda t, i: len(t) > 40),
    dict(name="15 design review", n=15, source=LOGIN,
         prompts=[f"Review this login screen from a {p} perspective. Give one issue and one fix.\n\n{LOGIN}"
                  for p in ["security", "privacy", "accessibility", "UX", "performance",
                            "mobile", "i18n", "legal", "support", "data", "reliability",
                            "fraud", "marketing", "QA", "executive"]],
         check=lambda t, i: len(t) > 30),
]


def main():
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    results = []
    for k, idea in enumerate(IDEAS, 1):
        with open(DATA, "w", encoding="utf-8", newline="\n") as f:
            f.write("id\tprompt\n")
            for i, p in enumerate(idea["prompts"]):
                f.write(f"{i+1}\t{p.replace(chr(9),' ').replace(chr(10),' ')}\n")
        args = [CLI, "--model", MODEL, "--data", DATA, "--out", OUT,
                "--mode", "batch", "--max-tokens", "96", "--temperature", "0.3",
                "--top-p", "0.9", "--seed", "1234"]
        subprocess.run(args, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=600)
        d = json.load(open(OUT, encoding="utf-8"))
        streams = d.get("streams", [])
        checks = []
        for i, s in enumerate(streams):
            t = s.get("text", "")
            ok = bool(t.strip()) and bool(idea["check"](t, i))
            checks.append({"id": s.get("id"), "ok": ok, "text": t,
                           "tok_s": s.get("tok_s"), "completion_ms": s.get("completion_ms")})
        results.append({"name": idea["name"], "n": idea["n"],
                        "wall_ms": d.get("wall_ms"), "streams": checks,
                        "ok": sum(c["ok"] for c in checks), "total": len(checks)})
        print(f"{k:>2}. {idea['name']:<24} N={idea['n']:<2} "
              f"ok={sum(c['ok'] for c in checks)}/{len(checks)} "
              f"wall={d.get('wall_ms', 0)/1000:.2f}s")
        for c in checks:
            if not c["ok"]:
                print(f"    stream {c['id']}: {c['text'][:120]!r}")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(results, open(RESULT, "w", encoding="utf-8"), indent=1)
    print("wrote", RESULT)


if __name__ == "__main__":
    main()

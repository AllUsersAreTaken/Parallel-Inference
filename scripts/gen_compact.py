"""Generate conversations for the two-stream context-compaction demo.

Each conversation has a few turns with a planted fact, a later correction, a
preference, and an open decision, then a latest user message and a follow-up
question whose answer depends on the corrected fact. Ground truth is the value
the follow-up must contain (and the original value, which must NOT be asserted).

Emits TSV: id <TAB> history(text with | separators) <TAB> latest <TAB> followup <TAB> gold <TAB> stale
"""
import random
import sys

VERSION = "compact-v1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/compact_convs.tsv"
rng = random.Random(SEED)

NAMES = ["Maya", "Jonas", "Priya", "Tomas", "Lena"]
CITIES = ["Lisbon", "Porto", "Tallinn", "Riga", "Bergen"]
D1 = ["March 14", "April 2", "May 9", "June 21", "July 30"]
D2 = ["March 21", "April 16", "May 23", "June 28", "August 6"]
N1 = [40, 120, 250, 640, 1500]
N2 = [55, 90, 310, 720, 1180]
PREFS = ["prefers async written updates", "prefers calls over email", "prefers morning meetings",
         "prefers a weekly cadence", "prefers no Friday meetings"]

rows = []
for k in range(1, N + 1):
    who = rng.choice(NAMES)
    city1, city2 = rng.sample(CITIES, 2)
    d1, d2 = rng.choice(D1), rng.choice(D2)
    n1, n2 = rng.choice(N1), rng.choice(N2)
    pref = rng.choice(PREFS)
    turns = [
        f"user: Let's plan the {who} rollout. Initial target city is {city1}, kickoff on {d1}, budget {n1}k.",
        f"assistant: Noted - {who} rollout in {city1}, kickoff {d1}, budget {n1}k.",
        f"user: Also, {who} {pref}.",
        f"assistant: Understood, {who} {pref}.",
        f"user: Change of plan: move the city to {city2} and the kickoff to {d2}. The budget goes to {n2}k.",
        f"assistant: Updated - city {city2}, kickoff {d2}, budget {n2}k.",
        f"user: One open question: do we still need a local partner in {city1}?",
        f"assistant: That is still undecided.",
    ]
    filler = []
    for _ in range(rng.randint(10, 14)):
        topic = rng.choice(["the vendor shortlist", "the analytics stack", "the onboarding flow", "the pricing tiers", "the support rota", "the design review", "the data migration"])
        filler.append(f"user: Can you expand on {topic}? Give a few sentences.")
        filler.append(f"assistant: Sure. On {topic}, the practical view is that the details matter more than the headline, and the trade-offs depend on timing. Start with the general mechanism, then adjust for local conditions, and check the exceptions before committing. The main risk is applying a rule of thumb outside its range.")
    turns = turns[:8] + filler + turns[8:]
    latest = f"What is the current kickoff date and budget for the {who} rollout?"
    followup = f"Remind me: which city and kickoff date are we using for the {who} rollout now?"
    gold = f"{city2}"      # follow-up must contain the corrected city
    stale = f"{city1}"     # asserting the old city is a critical loss
    rows.append((k, " | ".join(turns), latest, followup, gold, stale))

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"wrote {len(rows)} conversations (version {VERSION}, seed {SEED})")

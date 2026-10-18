"""Scripted conversation generator for the memory test (Test 2).

Planted-fact ground truth per stage. Spec:
  - 8-15 user turns
  - 2-3 facts planted in the first 3 turns (name, number, preference, date)
  - filler turns push history past the 2k window so early turns are evicted
  - 3 question types: direct, paraphrased, multi-fact; plus topic-jump questions
  - assistant replies are generated live by the runner (text = null here)

Version and seed are recorded so results are reproducible.

Usage: python gen_conversations.py [n] [seed] > data/conversations.jsonl
"""
import json
import random
import sys

VERSION = "conv-v1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
rng = random.Random(SEED)

NAMES = ["Aurora", "Nimbus", "Vertex", "Halcyon", "Cobalt", "Meridian",
         "Larkspur", "Onyx", "Quillon", "Sable", "Tessera", "Umbra"]
NUMBERS = [47000, 128, 3600, 19, 812, 275, 9400, 64, 1530, 88]
PREFER = ["prefers tea over coffee", "prefers window seats", "prefers async standups",
          "prefers written summaries over calls", "prefers morning flights",
          "prefers dark mode", "prefers a weekly cadence"]
DATES = ["March 14", "November 2", "July 9", "February 21", "September 30", "June 5"]

FILLER_TOPICS = [
    "the weather in Reykjavik", "last weekend's football results",
    "a sourdough recipe", "the history of the bicycle", "tide tables for the harbor",
    "a documentary about coral reefs", "the rules of padel", "a novel about a lighthouse",
    "trail conditions on the ridge", "a podcast on urban planning", "the price of copper",
    "how espresso machines work", "a museum exhibit on cartography", "kayak maintenance",
]

FILLER_SENT = [
    "It is worth noting that {t} has a long and fairly tangled history.",
    "People often disagree about {t}, and the disagreement tends to be more about framing than facts.",
    "If you look at {t} closely, the details matter more than the headline.",
    "There is a common misconception about {t} that keeps resurfacing.",
    "A useful way to think about {t} is to separate the short-run from the long-run effects.",
    "Most discussions of {t} eventually turn to cost, which is not always the right lens.",
    "The practical advice on {t} is usually simpler than the theory behind it.",
    "It helps to keep a few concrete examples of {t} in mind rather than reasoning in the abstract.",
]


def filler_turn():
    t = rng.choice(FILLER_TOPICS)
    k = rng.randint(4, 7)
    return " ".join(rng.choice(FILLER_SENT).format(t=t) for _ in range(k))


def plant(t, value):
    if t == "name":
        return f"Quick context: our new initiative is called {value}. Let's use that name from now on."
    if t == "number":
        return f"By the way, the budget we agreed on for this is {value} dollars."
    if t == "preference":
        return f"For the record, the client {value}. Please keep that in mind."
    if t == "date":
        return f"Heads up, the launch date is set for {value}."
    return f"Note: {value}."


def question(qtype, facts):
    if qtype == "direct":
        f = facts[0]
        return {
            "type": "direct",
            "fact_idx": 0,
            "value": f["value"],
            "text": {
                "name": "What is the initiative called?",
                "number": "What is the budget?",
                "preference": "What does the client prefer?",
                "date": "When is the launch date?",
            }[f["type"]],
        }
    if qtype == "paraphrased":
        f = facts[1] if len(facts) > 1 else facts[0]
        return {
            "type": "paraphrased",
            "fact_idx": 1 if len(facts) > 1 else 0,
            "value": f["value"],
            "text": {
                "name": "Remind me what we settled on for the thing we've been calling the project?",
                "number": "What figure did we land on for the spend?",
                "preference": "What were the client's stated leanings again?",
                "date": "What day did we pin down for the rollout?",
            }[f["type"]],
        }
    if qtype == "multi":
        a, b = facts[0], facts[-1]
        return {
            "type": "multi",
            "fact_idx": [0, len(facts) - 1],
            "value": [a["value"], b["value"]],
            "text": f"What were the {a['type']} and the {b['type']} again?",
        }
    # topic jump: asks about something never planted
    return {
        "type": "topic_jump",
        "fact_idx": -1,
        "value": rng.choice(["an unplanted detail", "something never mentioned"]),
        "text": rng.choice([
            "What was the name of the supplier we discussed?",
            "What was the figure for the second quarter?",
            "What did the client say about the timeline?",
        ]),
    }


convs = []
for cid in range(N):
    nfacts = rng.randint(2, 3)
    types = rng.sample(["name", "number", "preference", "date"], nfacts)
    facts = []
    for t in types:
        v = {"name": rng.choice(NAMES), "number": rng.choice(NUMBERS),
             "preference": rng.choice(PREFER), "date": rng.choice(DATES)}[t]
        facts.append({"type": t, "value": str(v)})

    turns = []          # user turns only; runner fills assistant replies
    planted = []        # (turn_index, fact_idx)
    for fi, f in enumerate(facts):
        turns.append(plant(f["type"], f["value"]))
        planted.append((len(turns) - 1, fi))

    # filler until history is comfortably past 2k tokens (assume ~1.3 tok/char)
    n_filler = rng.randint(5, 8)
    for _ in range(n_filler):
        turns.append(filler_turn())

    qs = []
    for qt in ["direct", "paraphrased", "multi"]:
        qs.append(question(qt, facts))
    if rng.random() < 0.4:
        qs.append(question("topic_jump", facts))

    convs.append({
        "id": cid,
        "version": VERSION,
        "seed": SEED,
        "facts": facts,
        "planted": planted,
        "turns": turns,
        "questions": qs,
    })

OUT = sys.argv[3] if len(sys.argv) > 3 else None
if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")
else:
    for c in convs:
        print(json.dumps(c))

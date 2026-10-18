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

FILLER_REQ = [
    "Why do tides vary through the month?",
    "Can you explain how an espresso machine works?",
    "What is a good hydration ratio for sourdough?",
    "How does padel scoring work?",
    "What causes coral bleaching?",
    "Give me a short summary of the history of the bicycle.",
    "How should I maintain a sea kayak?",
    "What is interesting about urban planning in the Netherlands?",
    "How are tides predicted for a harbor?",
    "What are the basics of trail running on a ridge?",
    "Why does the price of copper move so much?",
    "Recommend a documentary about lighthouses.",
    "How do I get better at chess endgames?",
    "What is a simple recipe for flatbread?",
    "How do noise-cancelling headphones work?",
    "What should I look for in a used road bike?",
]


def filler_turn():
    return rng.choice(FILLER_REQ)


# Canned assistant replies for filler turns. These make the history grow past the
# 2k window without a live generation and without teaching a summarize rut. The
# push stream only cares about the planted-fact turns, which are real exchanges.
CANNED = [
    "Good question. The short version is that it depends on several factors that interact. First, the underlying mechanism sets the broad shape of the answer. Second, local conditions shift the details, which is why simple rules of thumb often fail. Third, timing matters: the same inputs can give different outcomes depending on when they occur. In practice, people usually start with the general mechanism, then adjust for the specifics they can observe. If you want, I can go deeper on any one of these.",
    "There are a few ways to look at this. The most common explanation focuses on the main driver and treats everything else as a correction. A second view emphasizes the boundary conditions, where the simple story breaks down. A third is purely practical and ignores the theory, relying instead on what has worked before. None of these is wrong; they answer slightly different questions. For most everyday purposes the practical view is enough, but the mechanism view helps when something unusual happens.",
    "Briefly: the topic has a long history and the modern consensus is narrower than the popular accounts suggest. Early work established the basic pattern, later work refined the exceptions, and recent work has mostly been about measurement. The practical takeaway is that the headline is usually right in the middle of the range and misleading at the extremes. If you are deciding something, look at the middle of the distribution rather than the memorable anecdotes.",
    "It helps to separate two things: what is happening and why it matters. What is happening is fairly well understood and can be described in a few sentences. Why it matters depends on your goals, and reasonable people differ there. A useful habit is to state the mechanism first, then the consequence, then the uncertainty. That ordering keeps the discussion honest and makes it obvious where the disagreement actually lies.",
    "The honest answer is that it varies. In the common case the simple model works well, and you can reason about it with one or two variables. In the tails it does not, and you need the fuller picture. Most confusion comes from applying the simple model outside its range. If you tell me the specific situation, I can say which regime you are in and what to watch for.",
]


def filler_turn_with_reply():
    return rng.choice(FILLER_REQ), rng.choice(CANNED)


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
    canned = []         # canned assistant reply per turn ("" = generate live)
    planted = []        # (turn_index, fact_idx)
    for fi, f in enumerate(facts):
        turns.append(plant(f["type"], f["value"]))
        canned.append("")
        planted.append((len(turns) - 1, fi))

    # filler until history is comfortably past 2k tokens (canned replies are long)
    n_filler = rng.randint(30, 38)
    for _ in range(n_filler):
        u, a = filler_turn_with_reply()
        turns.append(u)
        canned.append(a)

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
        "canned": canned,
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

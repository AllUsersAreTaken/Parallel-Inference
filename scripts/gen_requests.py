"""Generate messy-request resolution scenarios for the edge-assistant demo.

Each scenario: local message context + a request, plus ground truth.
Two families: ambiguous (two Alexes, a corrected time, a missing field) and
straightforward (control, so a system that always asks for clarification does
not look good).

Emits:
  prompts.tsv   id <TAB> full prompt text (context + request + schema)
  truth.jsonl   id -> ground truth
"""
import json
import random
import sys

VERSION = "resolve-v1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
PROMPT_OUT = sys.argv[3] if len(sys.argv) > 3 else "data/resolve_prompts.tsv"
TRUTH_OUT = sys.argv[4] if len(sys.argv) > 4 else "data/resolve_truth.jsonl"
rng = random.Random(SEED)

SCHEMA = (
    "Fill in this JSON object exactly, using only facts from the context above. "
    "Output only the JSON, nothing else. If a detail is missing, put it in "
    "missing_information and set needs_clarification true.\n"
    '{"intended_event": "", "resolved_date_time": "", "requested_action": "", '
    '"supporting_message_ids": [], "missing_information": [], "needs_clarification": false}'
)


def ctx(msgs):
    return "Context:\n" + "\n".join(f"[{i}] {s}" for i, s in msgs)


scenarios = []
for k in range(N):
    ambiguous = k % 2 == 0
    other = rng.choice(["Alex Chen", "Alex Kim", "Alex Novak"])
    day1, day2 = rng.choice([("Tuesday", "Thursday"), ("Monday", "Wednesday"), ("Friday", "Tuesday")])
    t1, t2 = rng.choice([("3pm", "4pm"), ("10am", "2pm"), ("1pm", "11am")])
    day3 = rng.choice(["Friday", "Monday", "Wednesday"])
    loc = rng.choice(["Room 2", "the annex", "the main office", "the north lab"])
    if ambiguous:
        msgs = [
            (1, f"Alex Rivera's meeting is {day1} at {t1}, in {loc}."),
            (2, f"Correction: move Alex Rivera's meeting to {day2} at {t2}. Location unchanged."),
            (3, f"{other} confirmed the launch budget earlier today."),
            (4, f"Remember the adapter for the projector in {loc}."),
        ]
        request = "Remind me to bring the adapter to Alex's meeting."
        truth = {
            "intended_event": "Alex Rivera meeting",
            "resolved_date_time": f"{day2} {t2}",
            "requested_action": "bring the adapter",
            "supporting_message_ids": [1, 2, 4],
            "missing_information": ["reminder time"],
            "needs_clarification": True,
        }
    else:
        msgs = [
            (1, f"Alex Rivera's meeting is {day2} at {t2}, in {loc}."),
            (2, f"Please bring the adapter for the projector."),
        ]
        request = f"Remind me to bring the adapter to Alex Rivera's meeting on {day2} at {t2}."
        truth = {
            "intended_event": "Alex Rivera meeting",
            "resolved_date_time": f"{day2} {t2}",
            "requested_action": "bring the adapter",
            "supporting_message_ids": [1, 2],
            "missing_information": [],
            "needs_clarification": False,
        }
    prompt = SCHEMA + "\n\n" + ctx(msgs) + f"\n\nRequest: {request}\n\nJSON:"
    scenarios.append({"id": k + 1, "prompt": prompt, "truth": truth})

with open(PROMPT_OUT, "w", encoding="utf-8") as f:
    for s in scenarios:
        f.write(f"{s['id']}\t{s['prompt'].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
with open(TRUTH_OUT, "w", encoding="utf-8") as f:
    for s in scenarios:
        f.write(json.dumps({"id": s["id"], "truth": s["truth"]}) + "\n")
print(f"wrote {len(scenarios)} scenarios (version {VERSION}, seed {SEED})")

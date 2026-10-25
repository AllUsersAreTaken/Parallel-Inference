#!/usr/bin/env python3
"""Concurrent async self-correction benchmark for `async_llm` (Mini-SGLang).

Arms (per case):
  control      actor only (no thinker)
  checkpointed actor -> thinker -> continuation (serial)
  concurrent   actor in 12-token chunks; thinker runs as a background asyncio
               task; its block is integrated into the actor's cache view at the
               next chunk boundary. The engine batches both streams' decode
               requests into the same forward pass.

Cases: deterministic synthetic biographies with two dated events. Half have the
events in an impossible order (contradiction), half are consistent (clean).

Metrics: explicit correction rate, contradiction removed, invented years,
thinker false positives on clean controls, latency, actor/thinker overlap, and
joint decode ticks (engine proof of batching).

Run under WSL2 with the async_llm venv:
  bash scripts/run_async_concurrent.sh --smoke
  bash scripts/run_async_concurrent.sh
"""

import argparse
import asyncio
import csv
import math
import os
import random
import re
import statistics
import time

import torch

from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext

ASSISTANT = "<|im_start|>assistant\n"
NOTHINK = ASSISTANT + "<think>\n\n</think>\n\n"

ACTOR_TOKENS = 96
TRIGGER = 6
CHUNK = 12
TAIL_TOKENS = 24
THINKER_MAX = 48
CHECK_ACTOR = 64
CHECK_CONT = 90


def fix_prefix(think_text, leading):
    """Continuation frame chosen from the thinker's verdict.

    The thinker's *explanation* is in its cache block; this only picks the
    instruction that follows it. A leading `NO` gets the correction frame;
    otherwise the actor continues normally. This keeps clean controls from
    being primed to invent a correction.
    """
    no = bool(re.search(r"\bno\b", think_text, re.I))
    if no:
        body = (
            "<|im_start|>user\nThe fact-checker's report is above. It found an "
            "inconsistency. State the correction explicitly and keep the facts as "
            "stated; do not invent or change any dates.<|im_end|>\n" + NOTHINK
            + "Correction: "
        )
    else:
        body = (
            "<|im_start|>user\nThe fact-checker's report is above. It found no "
            "inconsistency. Continue the biography.<|im_end|>\n" + NOTHINK
        )
    return ("<|im_end|>\n" + body) if leading else body

STRICT = ("impossible", "cannot", "inconsisten", "contradict", "not possible")
CORRECTION_TERMS = STRICT + ("not the", "rather than", "instead", "should be",
                             "corrected", "error", "audit failed")
THINK_TERMS = STRICT + ("before", "after")
NO_RE = re.compile(r"\bno\b", re.I)
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20[0-2]\d)\b")

# template: rel=gt means impossible when event1 year > event2 year.
TEMPLATES = [
    dict(key="born_graduation", e1="the person was born", e2="graduated from high school",
         f1="{name} was born in {y1}.", f2="{name} graduated from high school in {y2}.",
         y1=(1870, 1970), cd=(2, 10), kd=(18, 25), rel="gt"),
    dict(key="founded_retired", e1="the person founded the company", e2="retired from the company",
         f1="{name} founded the company in {y1}.", f2="{name} retired from the company in {y2}.",
         y1=(1930, 1985), cd=(2, 15), kd=(10, 35), rel="gt"),
    dict(key="prize_death", e1="the person won the national prize", e2="died",
         f1="{name} won the national prize in {y1}.", f2="{name} died in {y2}.",
         y1=(1940, 2005), cd=(1, 8), kd=(1, 15), rel="gt"),
    dict(key="book_death", e1="the person published the book", e2="died",
         f1="{name} published the book in {y1}.", f2="{name} died in {y2}.",
         y1=(1940, 1990), cd=(1, 8), kd=(1, 15), rel="gt"),
    dict(key="married_birth", e1="the person married", e2="was born",
         f1="{name} married in {y1}.", f2="{name} was born in {y2}.",
         y1=(1930, 1975), cd=(18, 35), kd=(18, 35), rel="lt"),
    dict(key="school_graduation", e1="the person started school", e2="graduated",
         f1="{name} started school in {y1}.", f2="{name} graduated in {y2}.",
         y1=(1930, 1980), cd=(2, 12), kd=(4, 12), rel="gt"),
    dict(key="factory_sold", e1="the person built the factory", e2="sold it",
         f1="{name} built the factory in {y1}.", f2="{name} sold it in {y2}.",
         y1=(1900, 1970), cd=(2, 15), kd=(5, 30), rel="gt"),
    dict(key="joined_left", e1="the person joined the company", e2="left the company",
         f1="{name} joined the company in {y1}.", f2="{name} left the company in {y2}.",
         y1=(1940, 1985), cd=(2, 15), kd=(5, 30), rel="gt"),
    dict(key="wrote_death", e1="the person wrote the book", e2="died",
         f1="{name} wrote the book in {y1}.", f2="{name} died in {y2}.",
         y1=(1930, 1980), cd=(1, 10), kd=(1, 20), rel="gt"),
    dict(key="award_birth", e1="the person received the award", e2="was born",
         f1="{name} received the award in {y1}.", f2="{name} was born in {y2}.",
         y1=(1940, 1985), cd=(1, 25), kd=(1, 25), rel="lt"),
]

FIRST = [
    "Aaron", "Bella", "Carlos", "Diana", "Elias", "Fiona", "Gabriel", "Hana",
    "Ibrahim", "Julia", "Kai", "Lena", "Mateo", "Nadia", "Omar", "Priya",
    "Quentin", "Rosa", "Samir", "Tara", "Umar", "Vera", "Wei", "Xenia",
    "Yusuf", "Zara", "Adrian", "Beatriz", "Cyrus", "Delia", "Emil", "Farah",
    "Gustav", "Helena", "Ivan", "Jasmine", "Karim", "Laura", "Mikhail", "Nora",
    "Oscar", "Paloma", "Rafael", "Sofia", "Tomas", "Uma", "Victor", "Wendy",
    "Xavier", "Yara",
]
LAST = [
    "Achterberg", "Bianchi", "Chen", "Dubois", "Eriksen", "Fischer", "Garcia",
    "Haddad", "Ivanov", "Jensen", "Kowalski", "Lopez", "Moreau", "Nakamura",
    "Okafor", "Petrov", "Quinn", "Rossi", "Silva", "Tanaka",
]
THIRD = [
    "The rest of the record describes a long and varied career.",
    "Contemporary newspapers covered the later years in some detail.",
    "The remaining documents are preserved in the local archive.",
    "Friends described the later period as quiet and productive.",
    "The later part of the story is less well documented.",
]

_TICK_STATS = None


def set_tick_stats(st):
    global _TICK_STATS
    _TICK_STATS = st


def install_tick_counter(api):
    orig = api.async_engine.tick

    def tick():
        st = _TICK_STATS
        if st is not None:
            st["ticks"] += 1
            try:
                tags = {getattr(r.context, "_bench_tag", None)
                        for r in api.async_engine._decode_queue}
                if "actor" in tags and "thinker" in tags:
                    st["joint"] += 1
            except Exception:
                pass
        return orig()

    api.async_engine.tick = tick


def build_cases(seed=1234, n_contra=50, n_clean=50):
    rng = random.Random(seed)
    combos = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(combos)
    names = combos[:n_contra + n_clean]
    cases = []
    idx = 0
    for kind, n in (("contradiction", n_contra), ("clean", n_clean)):
        for i in range(n):
            tpl = TEMPLATES[i % len(TEMPLATES)]
            name = names[idx]
            idx += 1
            for _ in range(200):
                y1 = rng.randint(*tpl["y1"])
                d = rng.randint(*(tpl["cd"] if kind == "contradiction" else tpl["kd"]))
                if tpl["rel"] == "gt":
                    y2 = y1 - d if kind == "contradiction" else y1 + d
                else:
                    y2 = y1 + d if kind == "contradiction" else y1 - d
                if 1000 <= y2 <= 2025 and y1 != y2:
                    break
            source = (
                tpl["f1"].format(name=name, y1=y1, y2=y2) + " "
                + tpl["f2"].format(name=name, y1=y1, y2=y2) + " "
                + rng.choice(THIRD)
            )
            prompt = source + (
                "\n\nWrite a short biography (four to six sentences). You must mention "
                "both dates in the source."
            )
            question = (
                f"The biography above says {tpl['e1']} in {y1} and {tpl['e2']} in {y2}. "
                "Is that possible?"
            )
            cases.append(dict(
                case_id=f"{'c' if kind == 'contradiction' else 'k'}{i + 1:03d}",
                type=kind, template=tpl["key"], source=source, prompt=prompt,
                question=question, y1=str(y1), y2=str(y2),
                _e1=tpl["e1"], _e2=tpl["e2"],
            ))
    return cases


def strip_think(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def decode(api, toks):
    return strip_think(api.tokenizer.decode(toks)) if toks else ""


def intervals(start, times):
    out = []
    prev = start
    for t in times:
        if t > prev:
            out.append((prev, t))
        prev = t
    return out


def merge(iv):
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlap_ms(a_start, a_times, b_start, b_times):
    A = merge(intervals(a_start, a_times))
    B = merge(intervals(b_start, b_times))
    total = 0.0
    i = j = 0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if hi > lo:
            total += hi - lo
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return total * 1000.0


async def setup_prompt(api, case):
    pb = await api.create_block()
    msgs = [
        {"role": "system", "content": "You are a careful biographer. Use only the facts in the source."},
        {"role": "user", "content": case["prompt"]},
    ]
    try:
        text = api.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = api.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    ids = api.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > 1:
        await api.forward(ids[:-1], write_to=pb)
    return pb, ids[-1]


async def drain(api, ctx, n, toks, times, forbid_eos=False):
    kw = {"forbid_ids": [int(api.tokenizer.eos_token_id)]} if forbid_eos else {}
    async for tok in api.async_generate(ctx, max_steps=n, **kw):
        tok = int(tok)
        if tok == api.tokenizer.eos_token_id:
            return True
        toks.append(tok)
        times.append(time.perf_counter())
    return False


async def run_thinker(api, pb, ab, case):
    start = time.perf_counter()
    tb = None
    toks = []
    times = []
    try:
        tb = await api.create_block()
        prompt = (
            "<|im_start|>user\n" + case["question"] + " Answer YES or NO, then explain "
            "in one sentence.<|im_end|>\n" + NOTHINK + "Answer:"
        )
        ids = api.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > 1:
            await api.forward(ids[:-1], cache_view=[pb, ab, tb])
        ctx = AsyncContext(cache_view=[pb, ab, tb])
        ctx._bench_tag = "thinker"
        ctx.next_input_id = ids[-1] if ids else None
        if ctx.next_input_id is not None:
            async for tok in api.async_generate(ctx, max_steps=THINKER_MAX):
                tok = int(tok)
                if tok == api.tokenizer.eos_token_id:
                    break
                toks.append(tok)
                times.append(time.perf_counter())
        return {"block": tb, "tokens": toks, "times": times, "start": start,
                "end": time.perf_counter(), "error": None}
    except Exception as e:
        return {"block": tb, "tokens": toks, "times": times, "start": start,
                "end": time.perf_counter(), "error": repr(e)}


async def free_blocks(api, blocks):
    await asyncio.sleep(0)
    seen = set()
    for b in blocks:
        if b is None or id(b) in seen:
            continue
        seen.add(id(b))
        try:
            await api.free_block(b)
        except Exception as e:
            print("free_block failed:", e, flush=True)


KEYS = {
    "born_graduation": ("born", "graduat"),
    "founded_retired": ("found", "retir"),
    "prize_death": ("prize", "died"),
    "book_death": ("publish", "died"),
    "married_birth": ("marri", "born"),
    "school_graduation": ("school", "graduat"),
    "factory_sold": ("built", "sold"),
    "joined_left": ("join", "left"),
    "wrote_death": ("wrote", "died"),
    "award_birth": ("award", "born"),
}


def assignment_swap(text, case):
    """Did the continuation move event1's year onto event2 (or vice versa)?"""
    if not text:
        return 0
    k1, k2 = KEYS.get(case["template"], ("", ""))
    if not k1:
        return 0
    for clause in re.split(r"[;,\n]|\.\s+", text):
        cl = clause.lower()
        if (case["y1"] in clause and k2 in cl) or (case["y2"] in clause and k1 in cl):
            return 1
    return 0


def metrics(case, final_text, thinker_text, is_clean, post_text=""):
    y1, y2 = case["y1"], case["y2"]
    both = (y1 in final_text) and (y2 in final_text)
    aborted = "[AUDIT FAILED]" in final_text
    eval_text = post_text if post_text else final_text
    exp = True if aborted else any(t in eval_text.lower() for t in CORRECTION_TERMS)
    swap = 0 if (is_clean or aborted) else assignment_swap(post_text, case)
    faithful = "" if is_clean else int(exp and not swap)
    inv = sorted({y for y in YEAR_RE.findall(final_text) if y not in (y1, y2)})
    return {
        "actor_both_dates": int(both),
        "contradiction_removed": faithful,
        "faithful_correction": faithful,
        "assignment_swap": swap,
        "explicit_correction": int(exp),
        "invented_year": int(bool(inv)),
        "invented_years": "|".join(inv),
    }


def thinker_metrics(thinker_text):
    low = thinker_text.lower()
    return {
        "thinker_no": int(bool(NO_RE.search(thinker_text))),
        "thinker_flagged": int(bool(NO_RE.search(thinker_text))
                               or any(t in low for t in THINK_TERMS)),
    }


async def run_control(api, case):
    t0 = time.perf_counter()
    pb, first = await setup_prompt(api, case)
    ab = await api.create_block()
    blocks = [pb, ab]
    ctx = AsyncContext(cache_view=[pb, ab])
    ctx._bench_tag = "actor"
    ctx.next_input_id = first
    toks, times = [], []
    t_actor0 = time.perf_counter()
    eos = False
    while len(toks) < ACTOR_TOKENS and not eos:
        eos = await drain(api, ctx, min(CHUNK, ACTOR_TOKENS - len(toks)), toks, times,
                          forbid_eos=True)
    t_actor1 = time.perf_counter()
    final_text = decode(api, toks)
    rec = dict(case_id=case["case_id"], type=case["type"], template=case["template"],
               arm="control", actor_text=final_text, thinker_text="", final_text=final_text,
               actor_ms=round((t_actor1 - t_actor0) * 1000, 1), thinker_ms=0.0,
               total_ms=round((time.perf_counter() - t0) * 1000, 1), overlap_ms=0.0,
               integrated=0, tail=0, joint_decode_ticks=0, decode_ticks=0,
               _timeline=None)
    rec.update(metrics(case, final_text, "", case["type"] == "clean"))
    rec.update({"thinker_no": 0, "thinker_flagged": 0})
    await free_blocks(api, blocks)
    return rec


async def run_checkpointed(api, case):
    t0 = time.perf_counter()
    pb, first = await setup_prompt(api, case)
    ab = await api.create_block()
    blocks = [pb, ab]
    ctx = AsyncContext(cache_view=[pb, ab])
    ctx._bench_tag = "actor"
    ctx.next_input_id = first
    atoks, atimes = [], []
    t_actor0 = time.perf_counter()
    eos = False
    while len(atoks) < CHECK_ACTOR and not eos:
        eos = await drain(api, ctx, min(CHUNK, CHECK_ACTOR - len(atoks)), atoks, atimes,
                          forbid_eos=True)
    t_actor1 = time.perf_counter()

    know = await run_thinker(api, pb, ab, case)
    blocks.append(know["block"])
    cb = await api.create_block()
    blocks.append(cb)
    ids = api.tokenizer.encode(fix_prefix(decode(api, know["tokens"]), leading=False),
                               add_special_tokens=False)
    if len(ids) > 1:
        await api.forward(ids[:-1], cache_view=[pb, ab, know["block"], cb])
    cctx = AsyncContext(cache_view=[pb, ab, know["block"], cb])
    cctx._bench_tag = "actor"
    cctx.next_input_id = ids[-1] if ids else None
    ctoks, ctimes = [], []
    if cctx.next_input_id is not None:
        eos2 = False
        while len(ctoks) < CHECK_CONT and not eos2:
            eos2 = await drain(api, cctx, min(CHUNK, CHECK_CONT - len(ctoks)), ctoks, ctimes)

    actor_text = decode(api, atoks)
    think_text = decode(api, know["tokens"])
    post_text = decode(api, ctoks)
    final_text = actor_text + "\n" + post_text
    rec = dict(case_id=case["case_id"], type=case["type"], template=case["template"],
               arm="checkpointed", actor_text=actor_text, thinker_text=think_text,
               final_text=final_text,
               actor_ms=round((t_actor1 - t_actor0) * 1000, 1),
               thinker_ms=round((know["end"] - know["start"]) * 1000, 1),
               total_ms=round((time.perf_counter() - t0) * 1000, 1), overlap_ms=0.0,
               integrated=1, tail=0, joint_decode_ticks=0, decode_ticks=0,
               _timeline=None)
    rec.update(metrics(case, final_text, think_text, case["type"] == "clean",
                       post_text=post_text))
    rec.update(thinker_metrics(think_text))
    await free_blocks(api, blocks)
    return rec


async def run_concurrent(api, case, flag_only=False):
    t0 = time.perf_counter()
    pb, first = await setup_prompt(api, case)
    ab = await api.create_block()
    blocks = [pb, ab]
    ctx = AsyncContext(cache_view=[pb, ab])
    ctx._bench_tag = "actor"
    ctx.next_input_id = first
    atoks, atimes = [], []
    t_actor0 = time.perf_counter()
    eos = await drain(api, ctx, TRIGGER, atoks, atimes, forbid_eos=True)

    task = None
    if not eos:
        task = asyncio.create_task(run_thinker(api, pb, ab, case))

    know = None
    integrated = False
    integ_at = 0
    tail = False
    aborted = False
    st = {"ticks": 0, "joint": 0}
    set_tick_stats(st)

    async def integrate():
        nonlocal ctx, integrated, integ_at, know, aborted
        know = task.result() if task is not None and task.done() else await task
        tb = know.get("block")
        if tb is None or know.get("error") or not know.get("tokens"):
            return False
        think_text_local = decode(api, know["tokens"])
        fatal = (bool(re.search(r"\bno\b", think_text_local, re.I))
                 or any(t in think_text_local.lower() for t in STRICT))
        if flag_only and fatal:
            integrated = True
            integ_at = len(atoks)
            aborted = True
            return True
        pending = ctx.next_input_id
        if pending is not None:
            await api.forward([int(pending)], cache_view=[pb, ab])
        nb = await api.create_block()
        blocks.append(nb)
        ids = api.tokenizer.encode(fix_prefix(decode(api, know["tokens"]), leading=True),
                                   add_special_tokens=False)
        if len(ids) > 1:
            await api.forward(ids[:-1], cache_view=[pb, ab, tb, nb])
        ctx = AsyncContext(cache_view=[pb, ab, tb, nb])
        ctx._bench_tag = "actor"
        ctx.next_input_id = ids[-1] if ids else pending
        integrated = True
        integ_at = len(atoks)
        return True

    while not eos and len(atoks) < ACTOR_TOKENS:
        if task is not None and task.done() and not integrated:
            await integrate()
            if aborted:
                break
        eos = await drain(api, ctx, min(CHUNK, ACTOR_TOKENS - len(atoks)), atoks, atimes,
                          forbid_eos=True)
    t_actor1 = time.perf_counter()

    if not integrated and task is not None and not eos and not aborted:
        # The actor hit its budget before the thinker finished. Give the audit a
        # serial tail (reported separately as tail=1).
        if await integrate():
            tail = True
            eos2 = False
            while len(atoks) < ACTOR_TOKENS + TAIL_TOKENS and not eos2:
                eos2 = await drain(
                    api, ctx, min(CHUNK, ACTOR_TOKENS + TAIL_TOKENS - len(atoks)),
                    atoks, atimes, forbid_eos=True,
                )
    if task is not None and know is None:
        know = await task
    set_tick_stats(None)

    think_text = decode(api, know["tokens"]) if know else ""
    post_text = ""
    if aborted:
        actor_text = decode(api, atoks)
        reason = think_text.strip().replace("\n", " ")
        final_text = (actor_text + "\n[AUDIT FAILED] " + reason
                      + " The original facts are retained; no dates were changed.")
    elif integrated:
        pre = decode(api, atoks[:integ_at])
        post_text = decode(api, atoks[integ_at:])
        actor_text = pre
        sep = "[CORRECTION]" if bool(re.search(r"\bno\b", think_text, re.I)) else "[CHECK]"
        final_text = (pre + "\n" + sep + " " + post_text) if post_text else pre
    else:
        actor_text = decode(api, atoks)
        final_text = actor_text
    rec = dict(case_id=case["case_id"], type=case["type"], template=case["template"],
               arm=("flag_only" if flag_only else "concurrent"),
               actor_text=actor_text, thinker_text=think_text,
               final_text=final_text,
               actor_ms=round((t_actor1 - t_actor0) * 1000, 1),
               thinker_ms=round((know["end"] - know["start"]) * 1000, 1) if know else 0.0,
               total_ms=round((time.perf_counter() - t0) * 1000, 1),
               overlap_ms=round(overlap_ms(t_actor0, atimes, know["start"], know["times"]), 1)
               if know else 0.0,
               integrated=int(integrated), tail=int(tail), aborted=int(aborted),
               joint_decode_ticks=st["joint"], decode_ticks=st["ticks"],
               _timeline={
                   "actor_start": t_actor0 - t0,
                   "actor_times": [x - t0 for x in atimes],
                   "think_start": (know["start"] - t0) if know else None,
                   "think_times": [(x - t0) for x in know["times"]] if know else [],
                   "integrated_at": integ_at,
                   "integrated": integrated,
                   "tail": tail,
               })
    rec.update(metrics(case, final_text, think_text, case["type"] == "clean",
                       post_text=post_text))
    rec.update(thinker_metrics(think_text))
    await free_blocks(api, blocks)
    return rec


CSV_FIELDS = [
    "case_id", "type", "template", "arm", "actor_text", "thinker_text", "final_text",
    "thinker_no", "thinker_flagged", "actor_both_dates", "contradiction_removed",
    "faithful_correction", "assignment_swap",
    "explicit_correction", "invented_year", "invented_years",
    "actor_ms", "thinker_ms", "total_ms", "overlap_ms", "integrated", "tail", "aborted",
    "joint_decode_ticks", "decode_ticks",
]


def pctl(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return s[k]


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(records, cases):
    lines = []
    lines.append("Async concurrent self-correction benchmark")
    lines.append("model/GPU: Qwen3.5-2B bf16 / RTX 4060 8GB in WSL2")
    lines.append("")
    for arm in ("control", "checkpointed", "concurrent", "flag_only"):
        rs = [r for r in records if r["arm"] == arm]
        if not rs:
            continue
        contra = [r for r in rs if r["type"] == "contradiction"]
        clean = [r for r in rs if r["type"] == "clean"]

        def rate(rows, key):
            return 100.0 * sum(1 for r in rows if bool(r[key])) / len(rows) if rows else 0.0

        exp = rate(contra, "explicit_correction")
        faith = rate(contra, "faithful_correction")
        swap = rate(contra, "assignment_swap")
        inv = rate(contra, "invented_year")
        tfp = rate(clean, "thinker_no")
        tflag = rate(clean, "thinker_flagged")
        afp = rate(clean, "explicit_correction")
        cinv = rate(clean, "invented_year")
        abt = rate(contra, "aborted")
        cabt = rate(clean, "aborted")
        lo_e, hi_e = wilson(sum(1 for r in contra if r["explicit_correction"]), len(contra))
        lo_f, hi_f = wilson(sum(1 for r in clean if r["thinker_no"]), len(clean))
        tot = [r["total_ms"] for r in rs]
        lines.append(f"=== {arm} (n={len(rs)}) ===")
        lines.append(f"  contradiction cases: {len(contra)} | clean controls: {len(clean)}")
        lines.append(f"  explicit correction: {exp:.1f}%  95% CI [{100*lo_e:.1f}, {100*hi_e:.1f}]")
        lines.append(f"  faithful correction (flag, no date swap): {faith:.1f}%")
        lines.append(f"  date swap: {swap:.1f}%")
        lines.append(f"  invented a new year (contradictions): {inv:.1f}%")
        lines.append(f"  thinker false positive (verdict NO on clean): {tfp:.1f}%  "
                     f"CI [{100*lo_f:.1f}, {100*hi_f:.1f}]")
        lines.append(f"  thinker flagged (any NO/before/after): {tflag:.1f}%")
        lines.append(f"  actor false correction on clean: {afp:.1f}%")
        lines.append(f"  actor invented a new year on clean: {cinv:.1f}%")
        lines.append(f"  aborted (flag-only) on contradictions: {abt:.1f}%; on clean: {cabt:.1f}%")
        lines.append(f"  total ms: median {statistics.median(tot):.0f}, p95 {pctl(tot,95):.0f}")
        if arm in ("concurrent", "flag_only"):
            ov = [r["overlap_ms"] for r in rs]
            mid = sum(1 for r in rs if r["integrated"] and not r["tail"]) / len(rs)
            tl = sum(1 for r in rs if r["tail"]) / len(rs)
            jt = [r["joint_decode_ticks"] for r in rs]
            dt = [r["decode_ticks"] for r in rs]
            lines.append(f"  overlap ms: median {statistics.median(ov):.0f}, p95 {pctl(ov,95):.0f}")
            lines.append(f"  mid-stream integration: {100*mid:.1f}%; serial tail: {100*tl:.1f}%")
            lines.append(f"  joint decode ticks: mean {statistics.mean(jt):.1f} of {statistics.mean(dt):.1f} decode ticks")
        lines.append("")

    # examples
    conc = [r for r in records if r["arm"] == "concurrent"]
    ex = [r for r in conc if r["type"] == "contradiction" and r["explicit_correction"]][:2]
    if not ex:
        ex = [r for r in conc if r["type"] == "contradiction"][:2]
    for r in ex:
        c = next(c for c in cases if c["case_id"] == r["case_id"])
        lines.append(f"--- example {r['case_id']} ({r['template']}, contradiction) ---")
        lines.append("SOURCE: " + c["source"])
        lines.append("ACTOR:  " + r["actor_text"].replace("\n", " "))
        lines.append("THINK:  " + r["thinker_text"].replace("\n", " "))
        lines.append("FINAL:  " + r["final_text"].replace("\n", " "))
        lines.append("")
    clean_ex = [r for r in conc if r["type"] == "clean"][:1]
    for r in clean_ex:
        c = next(c for c in cases if c["case_id"] == r["case_id"])
        lines.append(f"--- example {r['case_id']} ({r['template']}, clean control) ---")
        lines.append("SOURCE: " + c["source"])
        lines.append("ACTOR:  " + r["actor_text"].replace("\n", " "))
        lines.append("THINK:  " + r["thinker_text"].replace("\n", " "))
        lines.append("FINAL:  " + r["final_text"].replace("\n", " "))
        lines.append("")

    # timeline example
    tl = next((r for r in conc if r.get("_timeline") and r["joint_decode_ticks"] > 0), None)
    if tl:
        t = tl["_timeline"]
        lines.append("--- concurrency timeline (relative ms) ---")
        lines.append(f"case {tl['case_id']}: actor_start={t['actor_start']*1000:.0f}")
        if t["think_start"] is not None:
            lines.append(f"thinker_start={t['think_start']*1000:.0f}, "
                         f"thinker_tokens={len(t['think_times'])}")
        lines.append(f"actor_tokens={len(t['actor_times'])}, integrated={t['integrated']} "
                     f"at token {t['integrated_at']}, tail={t['tail']}")
        lines.append(f"overlap_ms={tl['overlap_ms']}, joint_decode_ticks={tl['joint_decode_ticks']}")
        lines.append("")
    return "\n".join(lines)


async def async_main(args):
    cases = build_cases(args.seed, args.n_contra, args.n_clean)
    if args.smoke:
        cases = ([c for c in cases if c["type"] == "contradiction"][:2]
                 + [c for c in cases if c["type"] == "clean"][:2])
    if args.dry_run:
        for c in cases[:6]:
            print(c["case_id"], c["type"], c["template"], c["y1"], c["y2"])
            print("  " + c["source"])
            print("  Q: " + c["question"])
        print("cases:", len(cases))
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    kw = dict(dtype=torch.bfloat16)
    if args.pages:
        kw["num_page_override"] = args.pages
    api = AsyncLLM(args.model, **kw)
    install_tick_counter(api)

    # warmup (discard)
    print("warmup ...", flush=True)
    for arm in ("control", "checkpointed", "concurrent", "flag_only"):
        if arm in arms:
            try:
                if arm == "control":
                    await run_control(api, cases[0])
                elif arm == "checkpointed":
                    await run_checkpointed(api, cases[0])
                elif arm == "flag_only":
                    await run_concurrent(api, cases[0], flag_only=True)
                else:
                    await run_concurrent(api, cases[0])
            except Exception as e:
                print("warmup", arm, "failed:", repr(e), flush=True)
    print("warmup done", flush=True)

    f = open(args.out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    f.flush()

    records = []
    for i, case in enumerate(cases):
        for arm in arms:
            if arm == "control":
                r = await run_control(api, case)
            elif arm == "checkpointed":
                r = await run_checkpointed(api, case)
            elif arm == "flag_only":
                r = await run_concurrent(api, case, flag_only=True)
            else:
                r = await run_concurrent(api, case)
            records.append(r)
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
            f.flush()
            print(f"{i+1}/{len(cases)} {case['case_id']} {case['type']} {arm} "
                  f"exp={r['explicit_correction']} inv={r['invented_year']} "
                  f"t={r['total_ms']:.0f}ms ov={r['overlap_ms']:.0f} "
                  f"joint={r['joint_decode_ticks']}", flush=True)
    f.close()

    text = summarize(records, cases)
    with open(args.out_summary, "w", encoding="utf-8") as sf:
        sf.write(text + "\n")
    print("\n" + text, flush=True)
    await api.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DEMO_MODEL", "/home/max/models/Qwen3.5-2B"))
    ap.add_argument("--pages", type=int, default=int(os.environ.get("DEMO_PAGES", "65536")))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-contra", type=int, default=50)
    ap.add_argument("--n-clean", type=int, default=50)
    ap.add_argument("--arms", default="control,checkpointed,concurrent")
    ap.add_argument("--out-csv", default="results/async_concurrent_bench.csv")
    ap.add_argument("--out-summary", default="results/async_concurrent_summary.txt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

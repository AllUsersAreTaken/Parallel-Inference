"""AsyncLLM "hallucination catcher" demo: a biographer stream and a fact-checker
stream share one cache, and the biographer's *continuation* reads the checker's
critique through a cache view.

This runs on the `async_llm` framework (github.com/dvmazur/async_llm, built on
Mini-SGLang). It is Linux + NVIDIA only; on this machine it runs under WSL2
Ubuntu 26.04 against the RTX 4060. See ASYNC_SELFCORRECT_DEMO.md for the exact
setup (uv venv, CUDA 13.0 nvcc from pip, CUDA_HOME, and the glibc/rsqrt header
shim). Run with:

    bash scripts/run_async_demo.sh

Environment:
    DEMO_MODEL  path to a Qwen3.5 HF checkpoint (default ~/models/Qwen3.5-2B)
    DEMO_PAGES  KV pages for the engine (default 4096; bypasses the allocator's
                memory_ratio formula, which is too conservative on an 8 GB card)
"""

import asyncio
import os
import time

import torch

from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext

MODEL = os.environ.get("DEMO_MODEL", "/home/max/models/Qwen3.5-2B")
PAGES = int(os.environ.get("DEMO_PAGES", "4096"))
ASSISTANT = "<|im_start|>assistant\n"
NOTHINK = ASSISTANT + "<think>\n\n</think>\n\n"

CASES = [
    {
        "name": "birth_vs_graduation",
        "source": "He was born in 1882 in a small coastal town. He graduated high school in 1880. He later became a telegraph engineer.",
        "d1": "1882", "d2": "1880",
        "question": "The biography above says he was born in 1882 and graduated high school in 1880. Is that possible?",
    },
    {
        "name": "founded_vs_retired",
        "source": "She founded the company in 1990. She retired from the company in 1985. She later advised young founders.",
        "d1": "1990", "d2": "1985",
        "question": "The biography above says she founded the company in 1990 and retired in 1985. Is that possible?",
    },
    {
        "name": "award_vs_death",
        "source": "He won the national prize in 2001. He died in 1999. His work influenced a generation of engineers.",
        "d1": "2001", "d2": "1999",
        "question": "The biography above says he won the prize in 2001 and died in 1999. Is that possible?",
    },
]


def build_chat(api, source):
    user = (source + "\n\nWrite the first two sentences of the biography. You must "
            "mention both dates in the source.")
    msgs = [{"role": "system", "content": "You are a careful biographer. Use only the facts in the source."},
            {"role": "user", "content": user}]
    return api.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def strip_think(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


async def gen_from_last(api, first_token_id, view, max_steps):
    ctx = AsyncContext(cache_view=view)
    out = []
    async for tok in api.async_generate(ctx, first_token_id=first_token_id, max_steps=max_steps):
        if tok == api.tokenizer.eos_token_id:
            break
        out.append(tok)
    return strip_think(api.tokenizer.decode(out))


async def gen_prefix(api, prefix, view, max_steps):
    ids = api.tokenizer.encode(prefix, add_special_tokens=False)
    if len(ids) > 1:
        await api.forward(ids[:-1], cache_view=view)
    return await gen_from_last(api, ids[-1], view, max_steps)


async def run_case(api, case, use_thinker):
    pb = await api.create_block()
    ab = await api.create_block()
    tb = await api.create_block()
    cb = await api.create_block()
    pids = api.tokenizer.encode(build_chat(api, case["source"]), add_special_tokens=False)
    await api.forward(pids[:-1], write_to=pb)
    actor = await gen_from_last(api, pids[-1], [pb, ab], 64)

    thinker = ""
    if use_thinker:
        thinker = await gen_prefix(
            api,
            "<|im_start|>user\n" + case["question"] + " Answer YES or NO, then explain "
            "in one sentence.<|im_end|>\n" + NOTHINK + "Answer:",
            [pb, ab, tb],
            100,
        )
        prefix = (
            "<|im_start|>user\nThe fact-checker's report is above. The source may contain "
            "an impossible sequence of events. Do not invent or change any dates. If the "
            "source is internally inconsistent, say so explicitly and keep the facts as "
            "stated; otherwise continue the biography.<|im_end|>\n" + NOTHINK + "Correction: "
        )
        view = [pb, ab, tb, cb]
    else:
        prefix = ("<|im_start|>user\nContinue the biography in two sentences.<|im_end|>\n"
                  + NOTHINK)
        view = [pb, ab, cb]
    cont = await gen_prefix(api, prefix, view, 90)

    both = (case["d1"] in cont) and (case["d2"] in cont)
    explicit = any(k in cont.lower() for k in
                   ["impossible", "cannot", "before", "inconsisten", "corrected",
                    "contradict", "not possible", "after"])
    return {"actor": actor, "thinker": thinker, "cont": cont,
            "both_dates": both, "explicit": explicit}


async def main():
    kw = dict(dtype=torch.bfloat16)
    if PAGES:
        kw["num_page_override"] = PAGES
    api = AsyncLLM(MODEL, **kw)
    await run_case(api, CASES[0], False)  # warmup

    results = []
    for case in CASES:
        t = time.time()
        ctrl = await run_case(api, case, False)
        th = await run_case(api, case, True)
        results.append((case["name"], ctrl, th, time.time() - t))
        print("\n=====", case["name"], "=====")
        print("ACTOR:   ", repr(ctrl["actor"]))
        print("CONTROL: ", repr(ctrl["cont"]), "both_dates=", ctrl["both_dates"])
        print("THINKER: ", repr(th["thinker"]))
        print("FIXED:   ", repr(th["cont"]), "both_dates=", th["both_dates"],
              "explicit=", th["explicit"])

    print("\nSUMMARY")
    for name, ctrl, th, sec in results:
        print(f"{name}: control_both_dates={ctrl['both_dates']} "
              f"fixed_both_dates={th['both_dates']} fixed_explicit={th['explicit']} "
              f"sec={sec:.1f}")
    await api.close()


asyncio.run(main())

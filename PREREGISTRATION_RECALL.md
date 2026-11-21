# Pre-registration: recall program (kill-test ladder)

Declared 2026-11-16, before any run in this line. No data exists for it yet.

This document is not a claim. It encodes a **moderately confident judgment** as
a sequence of cheap tests that can falsify it, each with a stop rule written
before the run. If a test fails, the line stops at that test and the failure is
the result.

Scope is unchanged from `AGENTS.md`: one RTX 4060 (8 GB, WDDM), local GGUF
models only, no extrapolation to other hardware. Every number below is either
(a) cited external literature, **not reproduced here**, or (b) a
repo-measured result named by file. The two are never mixed.

---

## 0. Corrections to the draft judgment (recorded before the ladder)

The draft rested partly on two premises that the repo does not support as
stated. Both are corrected here; the ladder below is written against the
corrected premises.

### C1 - §88 does not establish that control beats per-item rules *item by item*

The draft cites "§88: state-aware control beats per-item rules" and adds only
that the stream is synthetic. The repo's own follow-up is stronger than that
caveat. `results/littlebit/prior_art_annealing.md` §88b reclassifies §88's
criterion-4 pass:

- overall D3 margin, no labels anywhere: **+0.281 +/- 0.120** over a ceiling
  that *equals* the best fixed arm (0.547). This survives and is the real
  result.
- post-switch margin: **+0.163 +/- 0.166**, CI lower bound **-0.003**. Reported
  as **"NOT established"**, not as a narrow pass. No transition prior in the
  declared sweep raises the lower bound above +0.046.

So the defensible claim is: *state tracking beats any item-only policy in
aggregate when state is present and inferable, on a semi-synthetic stream whose
cue `s_t` was hand-chosen.* It is **not** "and we know it works in the window
where it should matter most". The recall program must not lean on §88 for
per-item or post-change attribution.

### C2 - The LARQL / 270M / 8-18-to-5-18 evidence is not in this repo

The draft calls it "the repo's own evidence". It is not, and the file record is
unambiguous:

- `LARQL` / `larql`: **0 matches** repo-wide.
- `270M`: **0 matches**. No 270M model is present in `models/` (smallest is
  SmolLM2-360M).
- `8/18` appears once, in `results/ifeval_smoke_summary.txt`, as an IFEval
  smoke pass count (baseline 8/18, final 8/18) - unrelated to fact retention.
- No paraphrase-retention artifact exists in `results/`.

Treat that evidence as **external and unverified here** until a run in this repo
produces it. It may still be a good reason for low prior confidence; it is not
a measured result of this project.

### C3 - The repo *does* already have a hand-wired-memory negative

This supports the draft rather than correcting it, and it should be cited
instead of the missing patch evidence: Test 2 (`B0`-`B4`) found B2 (staged
summary brief) **below** B1 (raw top-k), with capture at **65%** as the ceiling
and all brief arms collapsing to 10-12% under 150 template-matched distractors
(`README.md`, `LIMITATIONS.md` "Memory (Test 2)", `results/mem_full.csv`,
`harness/analyze_memory.py`). That is the repo's actual evidence that a
hand-wired memory stage without a trained reader is brittle, and that the
bottleneck sat upstream of the reader (capture), not in the summary policy.

---

## 1. Premise under test (external, cited, not reproduced)

- Scaling the model barely moves long-tail recall: PopQA's 4,000 least popular
  entities go 16% (GPT-J 6B) to 19% (GPT-3).
- Retrieval lets a much smaller model beat an unassisted larger one on tail
  knowledge (2.7B + retrieval vs GPT-3 closed-book).
- Fixed retrieval *hurts* popular entities (GPT-3 -10%), and adaptive retrieval
  (retrieve only for unpopular entities) beats fixed retrieval by up to 5.3%
  at ~half the calls.

None of these are measured on this machine. They justify spending half a day on
K1, nothing more.

---

## 2. Frozen setup for the whole ladder

- **Core model: `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M`** (already in `models/`,
  used for the +11.8-point parallel result). Phi / TWIL are not fetched; adding
  one is an amendment plus a full K1 rerun, not a config flag.
- **Data: PopQA-style frequency-binned factual QA**, 3 pinned bins by entity
  popularity (head / body / **tail**), **>= 100 held-out items in the tail bin**,
  plus separate slices for **confusable entities**, **negation**, and
  **abstention** (unanswerable). Frozen question list hashed before any run;
  question selection never depends on outcomes.
- **Splits:** tuning on a dev split only; every reported number on the held-out
  remainder.
- **Statistics:** per-item outcomes, paired bootstrap (10k) or McNemar, 95% CI;
  3 seeds (1234 / 5678 / 9012). Difference in points = paired, not
  difference-of-means.
- **Oracle passage** = the gold supporting passage for that item, verbatim.
  **Paraphrase passage** = same fact, meaning-preserving rewrite, no shared
  long n-grams.
- Every arm logs model file + hash, seed, tokens inserted, TTFT, p50 wall clock,
  peak VRAM, and the GPU clock (thermal drift is unmanaged on this card).

---

## 3. Kill tests, in order

### K1 - Is there a gap at all? (~half a day)

Arms: closed-book, oracle-passage. Same prompt scaffolding, same answer
extraction, greedy + 3 seeds.

**Continue only if both hold:**

1. **tail-bin oracle gain >= +10 points**, paired 95% CI lower bound > 0.
2. **context-use check:** paraphrase-passage accuracy retains **>= 80%** of the
   exact-passage gain. (Guards against "gain" that is surface-string matching.)

**STOP if:** tail-bin gain < +10 pts, or its CI includes zero, or paraphrase
retention < 50%.

**Also record, not gate:** head-bin oracle gain. If head gain >= tail gain, the
"long tail" framing is wrong for this core - stop the framing, keep the eval
suite, and do not build K2.

Consequence of STOP: no memory work, no reader work. The eval suite (section 4)
ships as the result.

### K2 - Set the bar: plain hybrid RAG + adaptive controller

Build the bar *before* anything else. Hybrid = BM25/lexical + dense retrieval,
top-k, one fixed reader prompt, no memory store, plus the frequency-threshold
controller (retrieve only when the entity is unpopular). All knobs chosen on the
dev split.

**Continue to K3 only if:** the bar still leaves **>= 10 points** of the K1 tail
oracle gap unclaimed **and** at least **20%** of tail items wrong.

**STOP if:** the bar closes **>= 80%** of the K1 tail gap. Then the premise is
already satisfied by off-the-shelf parts, and a solo memory design has no room;
pivot to the controller/eval suite and say so.

Cost envelope to record with the bar (the number K4 must beat): median inserted
tokens, median wall clock, and accuracy per bin.

### K3 - Decompose the bar's failures (no new architecture)

Take every wrong tail item from K2 and classify it, using the oracle arm as the
separator:

- **retrieval miss** - gold passage not in the retrieved set;
- **context neglect** - gold passage present, answer still closed-book-wrong;
- **reader error** - passage present and attended, answer miscomposed;
- **abstention error** - answered an unanswerable, or refused an answerable.

**Gate (single rule, declared now):**

- **context neglect + reader error >= 50%** of tail errors -> **STOP the memory
  branch.** The fix is a reader, and the memory design does not matter.
- **retrieval miss >= 50%** -> an associative memory is the relevant lever;
  proceed to K4.
- **neither >= 50%** -> report the split, **build nothing**.

### K4 - Only then: factored memory, only against the measured category

Build against whichever category K3 measured, and nothing else. No reader
changes ride along; if the reader changes, K3 must be rerun.

**Pass:** tail-bin accuracy beats the K2 bar by **>= 5 points** with a paired
95% CI excluding zero, **on the frozen held-out set**, at **<= 1.25x** the bar's
median inserted tokens and **<= 1.10x** median wall clock on the 4060.

**STOP / publish as a bounded negative if:** the margin is < 5 points, its CI
includes zero, or either cost bound is exceeded. A negative here is a
publishable result because K1-K3 localize *why*.

### Controller sub-gate (applies wherever the controller is claimed)

Any claim that adaptive control beats fixed retrieval needs **overall** bin
accuracy only. **Post-change / first-N-after-switch claims need >= 6 seeds**
(pre-registered here because §88b showed 3 seeds leave that window
underpowered at ~80 scored items per seed). Do not report a post-change win on
3 seeds.

---

## 4. The part that pays either way: the evaluation suite

Ships regardless of K1-K4 outcomes. Frequency-binned recall + confusable
entities + negation + abstention, on a frozen core, with per-bin reporting and
no aggregate that hides the tail. Frozen before the first run; additions are
dated amendments. This is the deliverable if the architecture fails.

---

## 5. Discipline

- Gates in this file are frozen. Any change is a new dated section plus a rerun,
  as in `PREREGISTRATION.md`.
- Report the failure category, not just pass/fail.
- External literature is labeled as cited, never as measured here.
- No metric is reported from a run that did not happen.

---

## AMENDMENT 1 - 2026-11-16 (before any recall run; no data read)

Three implementation constraints found while building the harness. All three
are declared before the first measurement; none is informed by an outcome.

**A1.1 K1 sampling.** The separate greedy row is dropped. K1 runs at
`temperature 0.2` with seeds 1234 / 5678 / 9012 (three seeds, per the shared
statistics rule). Answer extraction is a deterministic alias match, so a greedy
row would duplicate a seed rather than add information. Head and body bins are
**descriptive**; only the tail bin gates.

**A1.2 The dataset is locally authored, not PopQA.** PopQA is not available
offline and its Wikipedia passages are not in this workspace, so K1-K4 use a
built set of 190 items (40 head / 40 body / 110 tail), each with one short gold
passage and one meaning-preserving paraphrase. Bins are assigned by author
judgment, **not** by measured pageviews. This is the main construct threat and
every number is scoped to it: the tail bin means "obscure to a general reader",
not "rank 4000+ by pageviews".

**A1.3 K2 retrieval is BM25 only, and the controller is an upper bound.** No
embedding model is present locally, so the bar is BM25 top-3 over the item
corpus (dense arm deferred). A deployable adaptive controller needs a
query-time popularity proxy that does not exist here, so the controller is run
as an **oracle-gated upper bound** (the bin label gates retrieval) and is
labeled as such. It is never reported as a controller result.

**A1.4 Slices deferred.** Confusable-entity, negation and abstention slices run
with the suite (section 4), after the ladder; K1-K4 measure the three frequency
bins only.

---

## AMENDMENT 2 - 2026-11-16 (before the K1 grid; smoke only)

The smoke row (6 items held out of the scored set) showed the pinned reasoning
core needs a different budget, not a different core.

**A2.1 K1 is scored on the answer segment, not the scratchpad.** The pinned
`DeepSeek-R1-Distill-Qwen-1.5B` emits `thinking ... </think>` and only then the
answer. At a 32-token cap every arm was truncated inside the scratchpad, which
measures formatting, not recall. K1 therefore runs at `--max-tokens 256` and
scores the text after the last `</think>` (falling back to the whole response
when there is no closing tag). Smoke: 4/6 finished under 256 tokens, 2/6 hit the
cap, and the paraphrase arm contradicted its own passage once (said El Salvador
for Angel Falls/Venezuela), which is a real reader failure rather than a format
artifact.

**A2.2 Truncation is reported, not hidden.** Every record carries `finish`. The
gate uses all items; accuracy restricted to items that finished under both arms
is reported beside it as `*_eos`. If the two disagree about the gate, both are
reported and the disagreement is a stated limitation.

**A2.3 Secondary core row.** `Qwen2.5-3B-Instruct` at a 32-token cap runs as a
labeled **secondary** row (no scratchpad). It is not the gated result. The gate
is applied to the pinned core only; where the two cores disagree the report says
so rather than selecting the one that passes.

---

## OUTCOME - 2026-11-16 (post-run, append-only; gates above unchanged)

K1 PASSED, K2 STOPPED the ladder, K3 explains why. Full record, tables and
limitations: `results/RESULTS_recall.md`. Summary of the frozen gates:

- **K1 (primary, `DeepSeek-R1-Distill-Qwen-1.5B`, 256 tok, 3 seeds).** Tail
  closed 0.088 -> oracle 0.711, gain **+0.623** CI [+0.561, +0.682], paraphrase
  retention 0.89. Head gain 0.350 < tail 0.623. `continue = true`.
  Finished-only tail subset (n=39): 0.141 -> 0.974, so the gap is not a cap
  artifact.
- **K2 (BM25 top-3 over 464 docs).** Tail 0.743 against oracle 0.711 and closed
  0.088: the bar closes **105%** of the K1 gap, `hit@3 = 1.00`.
  `closes_ge_80pct = true` -> **STOP, K4 not built.**
- **K3 (post-stop diagnostic).** 24/152 tail items wrong, all 24 capped. No
  retrieval miss, no context neglect, no reader error once the budget is enough:
  re-read at 600 tokens, 23/24 become correct. The residual was the token budget.

Two findings that bound the above, both in `results/RESULTS_recall.md`:

1. **The authored tail is not the PopQA tail.** On `Qwen2.5-3B-Instruct` at 32
   tokens the tail bin is **0.857 closed-book** (head 1.000). The K1 gap on the
   1.5B is therefore largely a capability gap, not a long-tail regime.
2. **Retrieval was saturated.** `hit@3 = 1.00` on a 464-document corpus of short
   passages with unique entity strings, so K2's stop is a statement about this
   corpus, not about RAG in general.

Amendments 1 and 2 were declared before their runs. The 256-token numbers are
kept as measured; the 600-token re-read is labeled a non-random diagnostic and
is not used to update K2's accuracy. No gate was changed after a number was
read. Nothing here is extrapolated beyond this machine.

---

## AMENDMENT 3 - 2026-11-16 (zero-recall regime; before generation and before any run)

The K1-K3 pass could not test the memory question, because the core already knew
most of the answers: the authored tail was 0.857 closed-book on a 3B
(`results/RESULTS_recall.md` section 2). This amendment replaces the construct
instead of the model.

**A3.1 Zero-recall by construction.** Core fixed at `Qwen2.5-3B-Instruct`
(32-token cap, no scratchpad). Facts are **freshly generated fictional facts**
(invented entity stems, 4-digit ids, attributes and values), seeded and
deterministic (`seed 20261116`). No model has them in parameters, so the
closed-book arm is a floor and the oracle arm is the ceiling; the only variable
is whether the fact is in the corpus. Generated once, before any run; the file
sha256 and counts are recorded and the set is frozen.

**A3.2 Subsets (40 items each, 240 total).**
- `single_hop` - one stated fact, direct question.
- `paraphrase` - same facts, query surface reworded (no long n-gram shared with
  the statement).
- `confusable` - entity ids in dense numeric clusters (same stem, ids differing
  by 1-2 digits), so lexical retrieval must separate siblings.
- `two_hop` - the answer needs two corpus facts chained (assignment fact ->
  attribute fact); the question never names the target entity.
- `absent` - the entity is in the corpus with other attributes; the asked
  attribute is never stated. Correct behaviour is abstention, scored as an
  abstention phrase with no digit in the answer segment.
- `update` - two docs state the same attribute, the second revises it. Gold is
  the revised value; the oracle arm sees only the revised doc, so precedence
  resolution is the RAG arm's own burden.

**A3.3 Arms.** `closed` (question only), `oracle` (gold doc; both docs for
`two_hop`; the entity's own docs for `absent`), `rag` (BM25 top-3 over the base
corpus), `rag_large` (BM25 top-3 over base + 1200 template-matched distractor
docs, same fictional vocabulary). Three seeds (1234/5678/9012).

**A3.4 STOP RULE, written before the run.** Margin M = 5 points, paired over
items, per subset.
- **STOP** (a second bounded negative; do not build memory) if, on **every**
  subset, `oracle - rag <= M` AND the paired 95% CI upper bound of that
  difference is `<= 10` points.
- **PROCEED** (a located gap; aim K4 at it) if **any** subset shows
  `oracle - rag >= 15` points with the paired CI **lower** bound `> 0`.
- **Otherwise** report the split and build nothing.

`rag_large` is descriptive (corpus-size stress). A gap that appears only there
is reported as a located candidate that needs its own pre-registration at that
scale; it does not by itself trigger K4.

**A3.5 What this does not test.** Fictional facts remove surface ambiguity that
real long-tail entities have (decoy siblings that genuinely exist, aliases,
noisy Wikipedia text). A clean result here is evidence about the *architecture*
question (can a reader with retrieval match a reader with the right passage),
not about PopQA. No Phi/TWIL weights are fetched: earlier claims about them were
not independently verified, so the experiment is not built on them.

---

## AMENDMENT 4 - 2026-11-16 (three-core comparison in the zero-recall regime; before any core runs)

The Amendment-3 regime (fictional facts, zero parametric recall) held the core
fixed at Qwen2.5-3B. This amendment asks the question directly: does a
reasoning-tuned core with weak recall close the same gap with RAG, and at what
cost?

**A4.1 Cores.** Same fictional set, same corpus, same arms, same scoring.
- `Qwen2.5-3B-Instruct` - control. Already measured by Amendment 3.
- `Phi-4-mini-reasoning` (unsloth Q4_K_M, 3.8B) - reasoning-tuned Phi.
- `TwIL-8B` (`webAI-Official/TwIL-8B`, Q4_K_M **balanced**, 8.79B, Granite 4.2
  base, non-commercial). Weights downloaded from HF; vendor benchmarks are
  vendor claims and are not used here as evidence.

**A4.2 Budgets and grids, fixed now.** Reasoning cores need room to finish, so:
- Qwen: 32 tok, batch 10, ctx 8192, seeds 1234/5678/9012 (unchanged).
- Phi: 256 tok, batch 8, ctx 8192, seeds 1234/5678/9012.
- TwIL: 256 tok, batch 4, ctx 4096, **seed 1234 only** - the 8.79B model at
  8 GB is the cost bound, and one seed fits the time budget. TwIL's rows are
  therefore **point estimates without a CI** and are labeled underpowered; they
  are never used to declare STOP on their own.

**A4.3 Cost, recorded per arm.** Mean generated tokens per answer, mean
completion time, and TTFT for every subset x arm, because the user's objection
is economic: a reasoning core that needs thousands of tokens per answer is not a
practical substitute even if it is accurate.

**A4.4 Gate.** Amendment 3's STOP rule applies per core, on the cores that have
3 seeds. In addition:
- **Context-use precheck.** If a core's oracle accuracy is below 0.50 on a
  subset, that core cannot use retrieved context on that subset and its RAG row
  is reported as `context-unusable`, **not** as a memory failure.
- **Practical-system criterion.** A new core replaces Qwen only if its RAG
  accuracy is within 5 points of Qwen's on **every** subset **and** its mean
  tokens per answer is `<=` Qwen's on that subset. Failing either is reported as
  a fail with the measured numbers.

**A4.5 Limits.** TwIL is non-commercial (license noted, no redistribution
beyond this machine), single GPU, 8 GB, no power telemetry. Vendor benchmark
claims (MATH-500 98.2, GPQA 70.2, tokens-per-problem figures) are not
independently verified and are not treated as evidence anywhere in this file;
only the runs below count.

## Amendment 5 — chat templates and instruction format (2026-11-16, before scoring)

Registered before any of the three cores produced a scored record.

**A5.1 Finding that forced it.** The batch engine fed raw prompt text and
`Runtime::format_chat` always passed `add_assistant=false`, so no assistant turn
was opened. Qwen2.5-Instruct tolerates a raw prompt; Phi-4-mini and TwIL (Granite
role tokens) do not — probes returned dict fragments and chat-around-the-passage
text. A `--chat` flag now renders system+user through the model's own GGUF
template and opens the assistant turn (`Runtime::format_chat_prompt`, add_assistant
= true). Default stays off, so every earlier run remains reproducible.

**A5.2 Instruction variant.** `--strict` adds "Reply with the value only" to the
user instruction plus a value-only system message. It is applied to every arm of
a core equally, so within-core closed/oracle/rag contrasts are unaffected.

**A5.3 Comparability.** The Qwen control is re-run under the identical format
(`results/recall_fictional_qwen_chatstrict.json`, 256 tokens, 3 seeds). The
earlier 32-token control stays as the original record; cross-core claims use the
chat/strict runs only.

**A5.4 Echo diagnostic.** `matches` is substring-based, so a core that quotes the
passage scores correct without answering. Every record now carries
`echoes_context` (normalised answer >12 chars and contained in the oracle
passage) and `ans_chars`. A core with a high oracle echo rate has its oracle
number reported as quotation, not context use, and A4.4's context-use precheck is
read together with it.

**A5.5 Phi degeneration.** Observed before scoring: Phi-4-mini-reasoning Q4_K_M
emits `.</strong>` / `</p>` loops at temp 0.0, 0.2 and 0.6 on passage prompts,
and generates exam-style multiple-choice items under a raw completion prompt.
Recorded as a property of this quant/format pairing on this machine, not as a
memory result.

## OUTCOME — three cores, measured (2026-11-16)

Record: `results/RESULTS_recall_cores.md`; runs `results/recall_fictional_qwen_chatstrict.json`,
`results/recall_fictional_twil.json`, `results/recall_fictional_phi.json`.

- **Recall gap closes on single-fact subsets for every core.** Closed 0.000 ->
  RAG 1.000 (Qwen single/para/confusable, TwIL all four, update 0.975/1.000),
  with Phi lower (0.500-0.650) and costly. Facts outside the weights plus three
  passages do the work; no memory architecture was needed for those subsets.
- **Two-hop RAG is 0.000 for all three cores** while oracle is 1.000/1.000/0.750.
  Core-independent, retriever-side, consistent with the median-rank-49
  diagnostic. This is the one real failure and it is not fixed by swapping cores.
- **Abstention fails for all three**, worst for TwIL (oracle 0.075) and Phi
  (0.150), and only marginal for Qwen (oracle 0.575). A4.4 context-use precheck:
  TwIL and Phi are `context-unusable` on `absent`; their RAG rows there are not
  memory failures.
- **Cost:** RAG answers cost 3.5-4.0 tokens / 52-64 ms (Qwen), 5.0-5.3 / 170-178 ms
  (TwIL), 187-222 / 2.8-3.6 s (Phi). On failing subsets TwIL rises to 160 tokens /
  4.4 s (two_hop) and 131 / 3.4 s (absent).
- **A4.4 practical-system criterion: fails for both new cores.** TwIL: -9.2 points
  on absent, 1.4-1.5x tokens on working subsets, ~50x on failing ones. Phi:
  -27 to -50 points on five subsets, ~55x tokens. Qwen2.5-3B stays the core.
- **Phi-4-mini-reasoning Q4_K_M is not usable as a reader here** (markup loops at
  three temperatures, invalid UTF-8 output, exam-question drift under raw
  completion). Its table is an upper bound, not a capability measurement.
- **A3 STOP rule was not declared by any core**: two_hop gap stays +1.000
  (Qwen, TwIL) and +0.750 (Phi), far outside the 5-point margin.
- **What would be tested next, not tested here:** rewriting two-hop into two
  chained retrievals, and an explicit "not stated" abstention instruction. Both
  are retriever/prompt changes, not core changes.

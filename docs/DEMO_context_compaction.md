# Demo: context compaction by a shared-prefix fork

## Headline
On an RTX 4060, a shared-prefix answer-and-memory fork reduced time until both
outputs were ready by 25%, with an 11% increase in answer latency. The generated
memory reduced subsequent context by 78% and preserved the corrected facts in all
ten tested conversations.

**Scope:** Qwen2.5-3B Q4, ~1.3k-token histories, ten conversations, one
compaction pass. These are prototype settings, not optimal values.

## What the engine does
Prefill the shared prefix once (common instructions + history + latest user
message), `seq_cp`-fork it, then run two branches in one batched decode:

```
Shared prefix (common instructions + history + latest user message)
                         │  fork cached prefix
          ┌──────────────┴──────────────┐
          │                             │
  seq 0: "answer the latest      seq 1: "produce a compact memory
   user message"                  of this conversation"
          │                             │
      Response                     Memory block (bounded)
```

Both task instructions are appended after the shared history; same model and
quantization on both branches. The memory branch has a narrow job: five sections
(goal; decisions and latest corrections; constraints and preferences; exact
facts/numbers; open questions/next steps), preserve negations and changes of
mind, no invented facts, no JSON. The transcript is retained; the memory is a
working context, not a replacement.

## Measured (10 conversations)

| Claim | Evidence |
|---|---|
| Answering remains responsive | TTFT 265 -> 288 ms; completion 514 -> 569 ms |
| Both outputs are available sooner together | 3041 -> 2269 ms (about 25% less elapsed) |
| The resulting working context is much smaller | 1374 -> 302 prefix tokens (78% reduction) |
| The tested correction survived compaction | 10/10 follow-ups correct, no stale-city assertions |

Concurrent execution reduced time until both outputs were ready by 25%, while
increasing foreground completion latency by 11%.

Note the two token counts: the parallel memory branch produced **188** tokens
against **220** in the serial run, so the 25% is an observed end-to-end workflow
improvement, not a clean measurement of batching speedup alone. Identical
generations are not required for a useful demo.

## What is *not* claimed
- The memory job is **not** mostly hidden under the answer: the answer finishes
  at ~0.57 s while the memory finishes at ~2.27 s, so most memory work runs
  after the answer. The 25% is overlap, not a 2.3 s task disappearing under a
  0.5 s one.
- **Next-turn latency payback is not quantified here.** The 78% is a token-count
  reduction, not a latency measurement; the full-context alternative can reuse a
  KV cache, and no follow-up TTFT/completion was recorded. Say: "the next turn
  used 78% fewer context tokens with no loss on the ten tested follow-up
  questions."

## Why it is worth showing
The appeal is not "we invented conversation summaries." It is that the engine
produces a second useful artifact from the same cached conversation without the
full cost of a separate sequential workflow. The output is visible immediately:
an answer, a compact memory, and a subsequent response that remembers the
correction. No claim of a smarter model, lossless compression, or proven phone
performance is needed.

## Artifacts
`scripts/gen_compact.py`, `harness/engine/compact.cpp` (+ `Engine::decode_requests`),
`scripts/analyze_compact.py`, `results/compact_10.csv`.

## Follow-ups (not blockers)
Longer contexts; recursive updates; idle-time scheduling; next-turn TTFT/completion
measured with and without KV reuse; edge-device measurements.

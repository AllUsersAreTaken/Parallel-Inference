# PREREGISTRATION — reranker pivot (decision-head distillation of Qwen3-Reranker-0.6B)

Frozen 2026-11-20, **before the Step 2 distillation run**. Dated amendments go
below; this section is not edited.

Companion to `PREREGISTRATION.md` and `PREREGISTRATION_QUBO_INIT.md`. It covers
one question and nothing else: *at 0.55 bits per weight, does a decision model
keep its ranking quality — and does the `qubo_act` sign search still buy
anything once the decision head is distilled?*

Why this target: the generative line is settled (A13, `itq` loses to `qubo_act`
by 9.0 % PPL on the full test split), and a reranker's whole output is one
projection of the final hidden state onto two vocabulary rows, so the artifact
that matters is a ranker, not a generator. `Qwen/Qwen3-Reranker-0.6B` is the
open-weight stand-in for a hosted 1-bit reranker; **no claim is made about that
product's deployed pipeline** (see Non-claims).

## Question and arms

Model `Qwen/Qwen3-Reranker-0.6B` (28 layers, hidden 1024, inter 3072, 16/8
heads, vocab 151,669, `tie_word_embeddings=True`). Target **0.55 bpp**,
`kv_rank_mult=1`, primary+residual paths, ranks from
`littlebit.convert.convert_model` and therefore identical across arms.

| arm | binary factors chosen by |
|---|---|
| `itq` | Joint-ITQ, T=50 — the LittleBit-2 baseline |
| `qubo_act` | Joint-ITQ init, then QUBO sign search on ‖(W − Ŵ)X‖_F, X from **SciFact train-split reranker prompts** |

Signs only; SVD magnitudes, ranks, scales and the two-path split are held fixed,
exactly as in the generative ablation. Rounds = 2, outer = 2, restarts = 4,
solver = 1-opt descent.

## Data: SciFact, and why the number is pool-limited

`BeIR/scifact`: 5,183 abstracts, `scifact-qrels` **train (809 queries) / test
(300 queries)** — a real split, so no query appears on both sides.

- **Evaluation**: first 100 test queries by ascending id (`rr_split.json`), each
  with the BM25 top-20 candidates **plus** the qrels gold documents. Reported
  metric: nDCG@10 (binary gain, qrels score > 0), plus Recall@10/20, MRR and
  pair AUC over (relevant, non-relevant) candidates.
- **Training**: the 809 train queries; positives from train qrels (up to 2 per
  query) and hard negatives from BM25 ranks 1–8 that are not judged relevant
  (6 per query). Negative-only pairs are unjudged, not verified irrelevant.

The pool is deliberately small so 100 queries are affordable on an 8 GB card,
and it **inflates** nDCG relative to the published full-corpus number (easy
negatives are removed before scoring). Every figure below is therefore only
comparable to other figures from this harness, never to a paper's SciFact row.

Scoring recipe, fixed by the model card: system prefix + `<Instruct>: …\n<Query>:
…\n<Document>: …` + `<|im_end|>\n<|im_start|>assistant\nthink\n\n</think>\n\n`,
left padding, score = `logits[-1][yes] − logits[-1][no]`. Verified against the
card before freezing: +7.412 / −10.914 (card: +7.625 / −11.375), identical at
batch 1 and batch 4 and unchanged by left padding.

## Step 1 — no-QAT feasibility gate (harness check, not a result)

`qubo_act` and `itq` must both (a) build at 0.5495 bpw over the 196 target
linears, and (b) not collapse:

- **S1**: pair AUC ≥ 0.70 on the gate pool (24 test queries × BM25 top-12 + gold).
- **S2**: nDCG@10 ≥ 50 % of the fp32 teacher's on the same pool.

Both are floors, not predictions. A13 showed an init advantage fully decayed to
1.07–1.09× by step 3,400, so **the Step 1 margin is not evidence of a trained
advantage** and is reported as a diagnostic only.

## Frozen distillation budget (not re-negotiable after data exists)

Identical for both arms, launched only after this file exists:

- **1,000 optimizer steps**, `batch_size=5`, `max_len=384` (prompt bodies are
  ~250–330 tokens; longer ones are truncated) → **1.92M tokens**.
- Adam 8-bit moments, peak lr 8e-5, 2 % warmup, cosine decay, clip 1.0, seed 0,
  data order seed 0, gradient checkpointing on, TF32 for training matmuls only.
- Embeddings and `lm_head` frozen and shared with the teacher (the embedding
  table is 296 MiB of the artifact; quantizing it is a separate question).
- Loss: `KL(teacher‖student)` on the two-way `[no, yes]` logits (temperature 1)
  **+ 1.0 ×** normalised final-hidden MSE over all positions,
  `‖h_s − h_t‖² / ‖h_t‖²`.
- Evaluated on a fixed 30-query subset every 200 steps and on all 100 test
  queries at the end; same pools, same order, for both arms and the teacher.

## Gates

- **R1 — usability.** The better arm's final nDCG@10 must be **≥ 90 % of the
  fp32 teacher's**, on the same 100-query pool. Below that, 0.55 bwz is not an
  interesting decision model at this budget and the finding is recorded as such.
- **R2 — relative edge.** `qubo_act` must beat `itq` by **≥ +0.01 absolute
  nDCG@10**, and must not lose on Recall@10 or MRR. With one seed per arm there
  is **no measured noise floor** for this metric; the gate is set at 1 point
  precisely because the floor is unknown, and a pass inside ±0.01 is reported as
  "indistinguishable", not as a win.
- **R3 — does the init gap survive?** Recorded, not gated: the nDCG@10 gap at
  step 0, 200, 400, 600, 800 and 1,000. A13's pattern (gap decays to ~1.07×) is
  the prior; if it happens here too, that is the reported result.

Recorded but not gated: decision-sign agreement with the teacher, Pearson
correlation of margins, `w_in` / `unw` per arm, build and training wall clock,
peak VRAM, and the `1/bpw` prefill MAC ratio recomputed at the ranks that
actually get built.

## Non-claims

- Pool-limited nDCG@10; **not** comparable to published SciFact numbers.
- We measure our sign search in **our** format and bit budget. NanoQuant's
  deployed mid-rank/3-scale pipeline is not reproduced, and the hosted 1-bit
  reranker has no open checkpoint, so "beats <product>" is not available from
  this experiment. The claim this can support is: *"X bpw decision model retains
  Y nDCG@10 at Z prefill cost"*.
- No kernel work is done or promised until R1 and R2 land. Prefill headroom at
  0.55 bpw is bounded by the `1/bpw` identity (≈1.995× on linears, 1.48×
  end-to-end at 2K context, 1.06× at 32K) — a 3–4× linear speedup needs
  0.25–0.35 bpw, which is the next experiment if this one passes.

## Cost and hardware

RTX 4060 8 GB only; no cloud instance is used for this pivot. If the runs
overrun, the budget above is honoured and the run is reported as slow, not
shortened.

---

## Amendment A1 — 2026-11-20, before either Step 2 arm was run

Four changes forced by measured memory behaviour on the 8 GB card. None of them
changes the question, the arms, the metric or the number of optimizer steps.

1. **Batch 5 → 4** (1.92M → 1.54M tokens over the same 1,000 steps). The frozen
   text assumed batch 5 would fit; measured peak for the student plus
   optimizer was already 6.2 GiB at batch 4, and WDDM adds desktop VRAM on top.
   Both arms get the same 1.54M tokens, so the head-to-head is still matched.
2. **KD targets from a bf16 teacher; reported reference numbers from fp32.**
   fp32 teacher (2.4 GB) + student did not fit: the process spilled into shared
   memory and ran ~30× slow. `teacher_chunk()` now computes per-step targets in
   bf16 (~0.4 % relative error on the hidden state, logits cast to fp32 for the
   KL), while every teacher number quoted in the results is the fp32 teacher on
   the same pool. The loss is unchanged; only the target precision is.
3. **Teacher targets are computed per 64-pair chunk** with the teacher parked on
   the host between chunks, instead of a teacher forward inside every step. The
   objective is identical (targets are deterministic), and it keeps one model on
   the card at a time.
4. **Evaluation scores the unpacked dense equivalent** rather than the eager bit
   path. Verified bit-exact to rounding before use: max |Δmargin| = 1.1e-4 on
   `itq`, 8.3e-6 on `qubo_act` over the probe pairs in `rr_gate.json`. This is an
   accuracy measurement only; no speed claim is made or implied.

Also recorded: the pool is 2 positives + 6 BM25 hard negatives per train query
(≈5.5k pairs, 1,000 × 4 = 4,000 pair-draws ≈ 0.7 epochs), and SciFact margins
are all negative on abstracts, so the readout is the ranking, never the sign of
the margin.

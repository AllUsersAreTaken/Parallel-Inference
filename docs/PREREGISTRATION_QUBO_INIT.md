# PREREGISTRATION — QUBO initialization ablation (LittleBit-2, Qwen3-0.6B)

Frozen 2026-11-17, **before the first arm was run**. Any change is a dated
amendment appended below; the tables above it are not edited.

Companion to `PREREGISTRATION.md` (the project's general gates). This one covers
a single question and nothing else: *at a fixed bits-per-weight budget, does
choosing the binary factors by discrete search beat Joint-ITQ's continuous
Procrustes rotation, and does any advantage survive distillation?*

## Question and arms

Model `Qwen/Qwen3-0.6B`, target **0.55 bpp**, `kv_rank_mult=1`, primary+residual
paths, ranks assigned by `littlebit.convert.convert_model` and therefore
**identical across arms** (recorded per arm in the output JSON).

| arm | binary factors chosen by |
|---|---|
| `itq` | Joint-ITQ, T=50 (LittleBit-2 baseline) |
| `qubo_free` | Joint-ITQ init, then QUBO sign search on ‖W − Ŵ‖_F |
| `qubo_act` | Joint-ITQ init, then QUBO sign search on ‖(W − Ŵ)X‖_F, X = WikiText-2 **train** activations |
| `svid` | Dual-SVID (LittleBit-1) — reference only, no gate |

The QUBO arms keep Dual-SVID's SVD magnitudes and change only signs, so scales
and rank are held fixed and the comparison isolates the sign assignment. Each
arm builds its own residual path from its own primary path's leftover. Rounds =
2 U/V alternations, solver = exact 1-opt descent; both fixed before running.

Calibration and scoring data are disjoint **by token**: Σ is built from the first
16k tokens of the WikiText-2 train split; layer errors are scored on the next 8k
tokens of train; perplexity is scored on the test split. Calibration is never
scored.

## Gates

Scored in order; a failed gate stops the chain (no tuning, no re-running on a
friendlier layer).

- **G1 — does discrete search beat the rotation on its own objective?**
  `qubo_act` mean `w_in` (activation-weighted layer error, the objective that
  arm optimises) must be **≥2% relatively lower** than `itq`'s.
  Prior: weight-space descent from ITQ+refit gained 4.5% on Llama-2 q_proj
  (`prior_art_annealing.md` §7), so 2% is a low bar that a real effect clears.
- **G2 — is the objective the lever, or is the search?**
  `qubo_act` must beat `qubo_free` by **≥1 percentage point of relative `w_in`
  reduction**. If it does not, activation weighting is not doing anything here
  and the honest reading is "sign search, objective-agnostic".
- **G3 — does the better basin show up with no training at all?**
  Best QUBO arm's **init-only WikiText-2 PPL must be ≥2% lower** than `itq`'s.
  This is the user-facing claim: a better discrete basin should be visible
  immediately, in a forward pass, not only in a weight-space proxy.
- **G4 — basin washout.**
  After a **matched** distillation budget (below), the arm that won G3 must
  either (a) finish **≥1% lower PPL** than `itq`, or (b) reach `itq`'s final PPL
  in **≤75% of the steps**. Either passes. If it loses on both, the finding is
  recorded as "the initialization advantage is erased by SmoothSign" — which is
  itself the answer to the basin question and is reported as such, not as a null.

Additional recorded (not gated) quantities: unweighted Frobenius error, flips as
a fraction of signs, wall clock per stage, `qubo_free`/`qubo_act` vs `svid`.

## Frozen distillation budget (not re-negotiable after data exists)

Identical for every arm: **600 optimizer steps**, `seq_len=1024`,
`batch_size=2`, `grad_accum=1` → **1.23M tokens**; Adam, peak lr 8e-5, 2% warmup,
cosine decay; loss = the reference `kd_loss` with `lam=10`, `mse_norm=True`,
`mse_skip_first=1` (the Qwen3 sink-aware recipe from the repo README); WikiText-2
train only, `wiki_repeat=1`, fixed seed and therefore the same data order;
gradient checkpointing on; embeddings and `lm_head` frozen and shared with the
teacher. PPL evaluated every 100 steps on the first 40 test windows of
WikiText-2, full-test never.

Labelled honestly: 1.23M tokens is **~1/870th** of the 1B-token reference run
(30.64 PPL). It is enough to see whether an initialization difference survives
contact with gradients — the basin question — and **is not enough to produce a
deployable model**. No result from this budget may be described as the
method's accuracy.

Statistics: one seed per arm. A one-step difference of <1% PPL is treated as
noise and stated as such; no confidence interval is claimed over seeds.

## Non-goals

Not tested here: any GPU other than this RTX 4060, any model other than
Qwen3-0.6B, bit-packed inference, SA/tabu beyond the single calibrated-schedule
arm already closed in `prior_art_annealing.md` §7.7.4, or QAT to convergence.

---

## Amendment log

(none yet)

## Amendment A1 (2026-11-17, written before any arm was run)

The solver is **best-of-4 kicked 1-opt descent**, not a single descent. Reason: a
single 1-opt descent is not globally optimal on a dense random QUBO. Measured at
r = 12, seed 0: one descent reached energy -32.00 against a brute-force minimum
of -38.29 over all 4096 assignments. Calling a single descent "the QUBO optimum"
would therefore overstate the baseline's headroom in the other direction, so the
arm is allowed four restarts (each a random flip of 10% of bits away from the
best solution so far, then monotone descent) and keeps the best local optimum
per row. Verified in `harness/littlebit/test_qubo.py` check 6: 4 restarts reach
-6642.83 against -6414.22 for one (-3.6%).

Nothing else changed: rounds = 2, outer = 2, scales frozen, arms and gates as
frozen above. The restarts apply identically to `qubo_free` and `qubo_act`.

## Amendment A2 (2026-11-17, written before any arm was run)

Activation capture stores only the second moment Sigma = X^T X / n, normalised so
mean(diag Sigma) = 1, rather than the activations themselves. This is not an
approximation: every objective here is the quadratic form tr(D Sigma D^T), and
keeping d_in x 16k activations for 196 modules would need ~12 GB on an 8 GB card.
`harness/littlebit/test_qubo.py` checks 2-4 verify that the QUBO energy
difference equals the direct objective difference computed from Sigma itself
(max error 4.8e-7), and that Sigma = I reproduces the Frobenius branch.

One consequence is recorded because it changes what is claimed: columns of V are
solved as a Jacobi step (all columns optimal given the others' previous values),
not a Gauss-Seidel sweep, so `update_cols` keeps the result only when the path
objective actually falls. Reverted column steps are counted in the output
(`cols_reverted`).

## Amendment A3 (2026-11-17, written before any distillation number existed)

The frozen distillation budget is reduced from 600 steps to **300 steps**, and the
optimizer is **8-bit Adam in fp32** (no autocast), for both arms.

Reason, measured on this card before the run:
`harness/littlebit/time_step.py` reports a converted-model training step at
**24.91 s/step in fp32 and 21.20 s/step under bf16 autocast, at a peak of
9.60 GiB** on the 8 GB 4060. 9.60 GiB exceeds the card, so WDDM pages and the
step cost is ~5x what an in-VRAM step should be. The reference implementation
ships `--optim adam8bit` for exactly this ("~4x less optimizer memory"), which
removes ~1.4 GiB of Adam state. FP32 is kept rather than bf16 because autocast
measured *slower* (21.20 s/step vs 24.91 is within noise, but bf16 autocast is
35x slower for pure inference here: 20.8 s/window vs 0.60 s/window).

Consequences that are accepted and reported rather than hidden:
- 300 steps at seq 1024 x batch 2 = **614,400 tokens** per arm, not 1.23M.
- G4's clause (b), "reaches itq's final PPL in <=75% of the steps", is only
  resolvable if an arm crosses below the other's final PPL within the budget.
  Otherwise the outcome is reported as a PPL-trajectory comparison, and the
  step-count clause is recorded as not resolved at this budget.
- Everything else is unchanged and matched: same data order, seed, schedule
  (lr 8e-5, 2% warmup, cosine), loss (lam 10, mse_norm, skip-first 1), frozen
  and shared embeddings. PPL every 50 steps on the same 40 test windows.

## Amendment A4 (2026-11-17, written before any distillation number existed)

A3's reduction of the budget to 300 steps is **withdrawn**; the frozen **600
steps** stand, and only A3's optimizer/precision change is kept.

Reason: A3 was written from a measurement taken *before* the memory fix existed.
With `adam8bit` plus shared frozen embeddings the same step was re-measured with
`harness/littlebit/time_step.py` at **1.55 s/step (1,322 tok/s), peak 5.25 GiB**
-- 16x faster than the 24.91 s/step paging regime, and comfortably inside the
card. 600 steps therefore costs ~15.5 min per arm, so there is no reason to give
up the pre-registered token budget (1.23M tokens per arm).

Final frozen configuration, identical for every arm: 600 steps, seq_len 1024,
batch 2, grad_accum 1 (1,228,800 tokens), Adam8bit, fp32 (no autocast), lr 8e-5
with 2% warmup and cosine decay, lam 10, mse_norm, mse_skip_first 1, WikiText-2
train only (wiki_repeat 1, seed 0 so the data order is identical), gradient
checkpointing, embeddings and lm_head frozen and shared with the teacher, PPL
every 100 steps on the first 40 WikiText-2 test windows.

## Amendment A5 (2026-11-17, written before any distillation number existed)

The 1.23M-token budget is unchanged; it is delivered as **batch 1 x grad_accum 2**
instead of batch 2 x grad_accum 1.

Reason, measured on this card during the first attempt at the run: with batch 2
the process sat at 7,880 of 8,188 MiB at 100% utilisation and had not completed 20
of 600 steps in ~12 minutes, i.e. WDDM was paging. The probe in A4 that reported
1.55 s/step measured only the body forward/backward and skipped the 151k-vocab
head and the teacher KL, so it under-measured the real step.

Splitting the same tokens into two micro-batches halves the activation footprint
of both the teacher forward and the student forward/backward, and is also what the
reference repo's own 8 GB Qwen3 recipe does (`--batch-size 1 --grad-accum 2`).
Tokens per optimizer step, data order, learning-rate schedule, loss and
evaluation are all unchanged, and identical across arms.

## Amendment A6 (2026-11-17, written before any distillation number existed)

Final budget: **300 optimizer steps**, PPL every 50 steps, TF32 enabled for the
training matmuls only. A4's 600 steps is withdrawn; A5's batch 1 x grad_accum 2
stands.

Reasons, each measured on this card before the run:
- With A5's memory layout the step no longer pages (6.1 GiB, 99% utilisation) but
  still cost >15 s/step, because fp32 GEMMs without TF32 leave the tensor cores
  idle. 600 steps would have been 2.5+ hours per arm.
- TF32 is enabled for the training forward/backward only. Every reported PPL is
  computed with TF32 disabled (`_ppl` in ablation.py) so that the curve stays
  comparable to the fp32 init-stage table.
- 300 steps at seq 1024 x batch 1 x grad_accum 2 = **614,400 tokens** per arm.
  That is the same step count as the reference repo's own 300-step ablation row
  for this model (Qwen3-0.6B @ 0.55 bpp, sink-aware loss, 300 steps -> PPL 820),
  which gives an external anchor for the absolute numbers.

Because the total budget is 300 steps, G4's clause (b) can only be resolved if an
arm crosses below the other's final PPL and by how many steps; if it cannot, the
outcome is reported as a trajectory comparison and the step clause is recorded as
not resolved at this budget. Clause (a), "finishes lower", is fully resolvable.

`Sigma` and the converted Joint-ITQ base are now cached under
`results/littlebit/_cache/` keyed by model/bit-rate/kv-mult/calibration size, so
repeated arms and re-runs do not repay the 16k-token capture and the SVD/ITQ pass.
The cache holds only derived matrices, and `--no-cache` forces a rebuild.

## Amendment A7 (2026-11-17, written before the pilot ran)

**G4 could not be executed at the pre-registered budget on this machine, and is
reported as unresolved.** What was run instead is a bounded washout *pilot*:
seq_len 256, batch 1, grad_accum 1, **120 optimizer steps** (30,720 tokens), PPL
every 40 steps, identical for both arms.

Why: with A6's configuration the training loop completed **fewer than 20 of 300
steps in 13 minutes (>39 s/step)** at 7,909 of 8,188 MiB, so the 300-step budget
would have cost ~3.3 h per arm and the 600-step frozen budget ~6.6 h per arm.
The cost is not memory paging alone; the converted forward in eager mode (no
Triton, so no `torch.compile` on this Windows install) issues hundreds of small
kernels per layer across three passes per micro-batch, which dominates at this
scale on WDDM.

What the pilot can and cannot answer:
- CAN: whether SmoothSign's first ~120 updates immediately pull the two
  initialisations together (the basin-washout question in its coarsest form).
- CANNOT: G4 clause (b) at the frozen step count, or any statement about
  convergence to the paper's PPL. 30,720 tokens is ~5% of one frozen arm.

The frozen 300-step protocol stands as the intended experiment; it needs either a
larger card or a Triton/`torch.compile` capable install, not a different gate.

## Amendment A8 (2026-11-17, written before the G4 runs started)

**G4 is now run to a real post-distillation budget for all three arms** (`itq`,
`qubo_free`, `qubo_act`), superseding A7's pilot:

| | value |
|---|---|
| seq_len x batch x accum | 256 x 1 x 1 |
| optimizer steps | **1,200** |
| tokens per arm | **307,200** |
| checkpoint + PPL eval | every 100 steps (12 points per arm) |
| PPL windows | 40 (fp32, TF32 off) |
| zero-shot | full PiQA validation (1,838) + full ARC-Challenge test (1,172), loglikelihood, item-level bootstrap 95% CI, 2,000 resamples, at init and at final |
| optimizer / precision | Adam8bit / fp32 training (TF32 on for training only) |

Budget justification, measured, not preferred: on this card the converted model
runs at **1.66-2.03 s/step** at seq 256 (measured over the 20-step resume test,
the same configuration), i.e. ~150-160 tokens/s. 1,200 steps is therefore
~35 min of training per arm rather than the ~3.3 h that 300 steps cost at seq
1024. 307,200 tokens is the largest budget that fits three arms plus six
full-set zero-shot evaluations into one working session on this GPU; seq 1024
was measured at >39 s/step and is not viable here.

Restart safety (required, since the runner has been interrupted twice): each arm
writes `results/littlebit/runs/<arm>/train.pt` every 100 steps containing the
trainable parameters, the 8-bit Adam state, the step, the data cursor and the
CPU/CUDA RNG states, and resumes from it by default. Validated by an A/B test
before the runs: a 20-step uninterrupted run versus a run stopped at step 10 and
resumed reproduced **every loss and every PPL exactly** (worst relative PPL
difference 0.00e+00). The first A/B attempt failed and the failure was fixed, not
worked around: checkpoints originally stored only the LittleBit latents, so the
trainable LayerNorm weights were silently reset on resume; they are now saved.

The QUBO column subproblem is *not* re-solved during distillation and the init
objective is unchanged, so the near-total rejection of batched column steps under
the activation Hessian (1,567/1,568 at 0.55 bpp) stands. A probe at the final
checkpoint reports whether that is still the case after distillation.

## Amendment A9 (2026-11-17, written during the G4 campaign, after `itq` reached step 100)

Three changes were made while the A8 runs were in flight. None of them changes the
objective, the data order, the schedule, the loss, the evaluation, or anything the
gate is computed from; all three are recorded because they change what the runs
*report* or how fast they run.

1. **`torch.cuda.empty_cache()` before the training loop.** The init-stage
   evaluation (40 PPL windows plus two full-set zero-shot passes) left the CUDA
   pool fragmented, and the first 100 steps ran at 3.44 s/step at 7,841 MiB
   (paging) against 1.66-2.03 s/step measured before the run. After the call the
   same arm ran at **0.87-0.88 s/step at 6.0 GiB** for the remaining 1,100 steps.
   Throughput only.
2. **Sign-flip diagnostic.** Every checkpoint now records the fraction of binary
   signs that differ from the arm's own initialization (bit-packed, 220,774,400
   signs tracked, reference rebuilt from `runs/<arm>/init.pt`). Read-only, O(bits)
   once per 100 steps. The `itq` arm's first 100 steps ran without it; because the
   quantity is cumulative *versus init*, nothing is lost - the step-200 point
   already includes any flips made in those 100 steps.
3. **Gauss-Seidel column solver** (`qubo.update_cols_gs`), added in response to the
   1,567/1,568 column rejections. It is **post-hoc**: no number in this campaign
   was produced with it, the init-stage numbers stand as recorded, and the column
   probe at each final checkpoint confirms the *old* batched step was still being
   rejected 8/8 after distillation. Re-running the init stage with GS at model
   scale is the next experiment and is not claimed here.

The resume path was exercised for real twice during this campaign (a runner
restart, and a deliberate stop to add item 2): each time the affected arm
continued from its checkpoint. A8's exactness claim is unchanged, and the
diagnostic's packing helpers were validated before use (0.0 flips at init, 1.0
after negating every V sign, 31/32 after restoring one bit).

## Amendment A10 (2026-11-18, after G4 was scored; a new experiment, not a G4 arm)

**Gauss-Seidel columns, run at model scale.** A4/A8 recorded that the batched V step
was rejected 1,567/1,568 times under the calibration Sigma. `qubo.update_cols_gs`
replaces it with a sweep that re-bases the residual after each block (rank-`block`
update), so later columns see earlier ones; `refine_layer` gained `gs` and `gs_final`.

New arm **`qubo_act_gs`**: identical to `qubo_act` except that the alternating rounds
stay on Jacobi and one V-only GS sweep runs at the end of each layer (`block=16`).
It is **not** a G4 arm - G4 stands on `qubo_act` as run - and it was taken to the init
stage only.

Result: `w_in` **0.47159** vs 0.49025 (**-3.8%**), `unw` 0.87568 vs 0.86170 (worse),
init PPL **17,623** vs **9,246** (**1.9x worse**), build 1,920 s vs 320 s. The column
half was therefore not headroom for perplexity; it improves the surrogate and degrades
the metric G3 is written on. Recorded as a negative result rather than tuned away.

Block size came from `--probe-layers` measurement (3 layers, layer 0): block 8 gave
-3.7/-5.2/-5.7% on q/k/v_proj at 31.9 min projected, block 16 -3.4/-3.6/-5.2% at
20.4 min, block 32 +-0% at 13.9 min.

Two defects were found and fixed while doing this, both of which had falsified
something before being fixed:

1. **Unguarded coarse blocks were harmful.** With `block=32` and no guard, GS made
   `w_in` 2.6x *worse* than Jacobi (0.69728 vs 0.26821 on `q_proj`) because a coarse
   in-block Jacobi step overshoots and was accepted unconditionally. `block > 1` now
   compares each block's own contribution to the objective (`<F Sigma, F> = <S, F>`)
   and reverts the block if it did not fall. `block=1` is exact by construction and is
   not guarded. `test_qubo.py` check 7 covers blocks 1/4/8/32/48.
2. **A completed 1,920 s build was discarded** because `build_arm`'s meta aggregation
   indexed `h["rows"]` on the new cols-only `final_sweep` entries (KeyError). Fixed
   by `arm_meta()`, and the build now writes `runs/<arm>/init.partial.pt` every 25
   layers and resumes from it, so a crash costs at most 25 layers.

No previously reported number changed: the four init arms, all three G4 arms and every
gate verdict above are computed from the artifacts recorded before A10.

## Amendment A11 (2026-11-18, after A10; an extension, not a G4 change)

**`qubo_act` pushed past the G4 budget.** Warm-started from its 1,200-step weights and
run to **4,000 steps / 1,024,000 tokens** at seq 256 (`results/littlebit/
distill_qubo_act_long.json`): PPL **112.975**, `w_in` 0.54953, `unw` 0.86891, column
probe 8/8 reverted.

Read this as a new phase, not a continuation: the cosine horizon is `args.steps`, so
extending the run re-warms the LR and the first 100 steps regress (206.9 -> 237.9)
before recovering past the G4 value. The curve is **not** comparable point-for-point
with the G4 curve.

It **plateaus around 113 and does not reach double digits**: 116.0/116.4/116.8 at
steps 3,400-3,600, then 114.1/113.3/113.2 at 3,700-3,900. Full-set zero-shot at the
plateau: PiQA .5169 [.4946, .5403] (chance inside the interval), ARC-C **.1954
[.1732, .2184]**, i.e. significantly *below* the 0.25 chance line. At PPL 113 the
tasks still do not separate from chance, so the reasoning-regime question is
unanswered rather than answered negatively; reaching 35-60 PPL needs hours of GPU
time on this card, not a short finishing phase.

A seq-1024 finishing pass was started and produced **no result** (the runner was
restarted before its first optimizer step); it is not reported as a number.

Provenance defect, recorded because it affects reproducibility: the long phase was
first launched with a `run_dir` that accepted `--run-name` but ignored it, so it
resumed the real `qubo_act` checkpoint and advanced it to step 1,400 before being
stopped. No reported number changed - the G4 JSON was restored from a backup taken
before that run could finish writing - but `runs/qubo_act/train.pt` no longer holds
the 1,200-step state (it was moved to `runs/qubo_act_long/train.pt` at step 1,400).
Rebuilding the 1,200-step checkpoint from `runs/qubo_act/init.pt` costs ~18 min.
`run_dir` now honours `--run-name`, and long runs must pass an explicit `--out` so
they cannot overwrite a scored arm's JSON.

## Amendment A12 (2026-11-18, post-hoc: second model family)

**Status: an extension test written up after it ran. It is not a preregistered gate,
and no G4 verdict depends on it.** It is recorded because it narrows the claim.

Question: is the G4 advantage specific to Qwen3-0.6B's weights? Protocol is G4's,
unchanged: 1,200 steps, seq 256, batch 1, **307,200 tokens per arm**, same seeded data
order, same `kd_loss(lam=10, mse_norm=True, mse_skip=1)`, PPL on 40 WikiText-2 test
windows with TF32 off. Only the model changes:

| | value |
|---|---|
| model | Qwen2.5-0.5B (qwen2, 24 layers, hidden 896, 494M params) |
| teacher PPL | 12.2288 |
| bpw | 0.5511 |

Llama-3.2-1B was rejected as the second family **on memory, not preference**:
`distill.py` keeps an fp32 teacher and an fp32 student resident, ~4.9 GB each, which
cannot fit an 8 GB card. Qwen2.5-0.5B peaks at 5.3 GiB.

Result (307,200 tokens/arm, `distill_qwen25_{itq,qubo_act}.json`):

| | itq | qubo_act | ratio |
|---|---|---|---|
| init PPL | 117,530.9 | 18,029.7 | 6.5x better |
| final PPL | 188.3 | **160.0** | **0.849** (-15.1%) |
| step reaching itq's final | - | **800** of 1,200 | 1.5x fewer tokens |
| final `w_in` | 0.73051 | **0.61073** | better |
| final `unw` | **0.79465** | 0.87415 | worse |

**Verdict: the direction replicates, the magnitude does not.** `qubo_act` beats `itq`
on a second family, with the same error signature as on Qwen3 (lower activation-weighted
error `w_in`, higher unweighted `unw`), and it is 6.5x better *at init*. But the payoff
collapses from 2.41x to 1.18x, and sample efficiency from 5.9x fewer tokens to 1.5x.
The init advantage also starts much smaller here (6.5x vs Qwen3's 6,293x), so the two
facts are consistent: this family's Joint-ITQ init is simply not as catastrophic, and
distillation closes most of whatever gap is left.

Consequences for the G4 write-up: the 5.9x sample-efficiency figure must be quoted as
**Qwen3-0.6B-specific**. The cross-family claim that survives is weaker - "QUBO init
beats Joint-ITQ at equal tokens on both families tested, by 1.18x-2.41x final PPL" -
plus the observation that the gain is largest exactly when the ITQ init is worst.
The final-third ratio is stable (0.834-0.849, 1.9% spread), so the margin is not still
closing at this budget, but both arms remain far from the 12.23 teacher and the curves
have not converged.

## Amendment A13 (2026-11-18, written before any A13 step was taken)

**A11's plateau is a hardware statement, not a method statement, and this amendment
buys the hardware time to test that.** A11 reached 112.975 PPL at 1,024,000 tokens on
the 8 GB card and could not go further; A13 runs the same arm at a budget that card
cannot afford.

### Hardware, and what it may be called

| | |
|---|---|
| GPU | **RTX A6000**, 49,140 MiB, driver 610.43.02, sm_86, 300 W |
| host | 6 vCPU, 47 GB RAM, Ubuntu 22.04.5 |
| software | Python 3.12.13, torch 2.12.1+cu130, transformers 5.23.0, datasets 5.2.0 |

The A6000 is **not** the L40 named in A4. Numbers from it are logged as
`gpu=a6000_cloud`, `source=a6000_cloud`, and **no number produced here may be labelled
`l40_sim_4060`**; likewise nothing here may be compared with a 4060 number as if the
cards were interchangeable. The only claims A13 supports are internal ones: this arm
versus that arm, on this card, at this token count.

### Exactly two things change against A11

| | A11 (4060) | A13 (A6000) |
|---|---|---|
| `seq_len` | 256 | **2048** |
| training mix | WikiText-2 train, `wiki_repeat=1` | **C4 100M tokens + 4x WikiText-2 train** |
| tokens of training data | ~2.4M | **110,067,712** (53,744 seqs x 2048) |

The mix is the reference repo's Section 4.1 recipe and is built by
`littlebit.data.train_chunks(tokenizer, seq_len, c4_tokens, wiki_repeat)`, which existed
and was simply not wired into `distill.py`; it is now driven by `--c4-mtokens` /
`--wiki-repeat`, and the built mix is recorded in the result JSON (`data.train_seqs`,
`data.train_tokens`, `data.epochs`). Document order is the repo's fixed seed-0
permutation, unchanged across arms.

### Everything else is held fixed, and the initialization is verified, not assumed

The loss, the schedule and the starting point are A8's:

- **Loss unchanged and already sink-aware**: `kd_loss(student, teacher, ids, lam=10,
  mse_norm=True, mse_skip_first=1)` - the Qwen3 sink-aware recipe from the reference
  README. There is no "sink-aware vs standard" contrast to run here; both arms of every
  run since A8 have used the sink-aware form, and A13 records it in the JSON
  (`loss.attention_sink_aware`).
- **Identical initialization, verified against the 4060 artifact.** `runs/qubo_act/
  init.pt` was copied to the A6000 unmodified (SHA-verified by size/reload). Loading it
  there reproduces the recorded init statistics: **PPL 9246.1966** vs 9246.1953
  recorded, `w_in` **0.49025**, `unw` **0.86170**. The 7th-digit PPL difference is the
  torch 2.11 -> 2.12 difference, not a different init. So A13 measures the effect of
  seq_len and data mix alone, not a fresh QUBO build on a different GPU.
- Rebuilt on the A6000 because they are hardware-local: Sigma (196 modules, 16,384
  calibration / 8,192 held-out tokens, identical slices) and the Joint-ITQ base
  (`bpw 0.5495`, 333.5 s). Same seeds, same arguments.

### One change that affects throughput only, recorded because A6 claimed otherwise

A6 states that TF32 is enabled for the training matmuls. In `distill.py` it never was:
`_ppl()` saves and restores `torch.backends.cuda.matmul.allow_tf32`, so with no explicit
enable the flag stayed at torch's default `False` for the whole of G4, A11 and A12. A13
adds an explicit `--tf32` flag (default off, so no existing artifact changes meaning) and
runs with it on. Measured, batch 1, seq 2048, same mix:

| | s/step |
|---|---|
| no TF32 (12-step smoke, incl. 2 PPL evals + 2 x 1.39 GB ckpt writes) | 11.5 |
| `--tf32` (40 steps, incl. 1 eval + 1 write) | **3.82** |

That is an upper bound on the TF32 gain, not a clean A/B: the two runs differ in
eval/checkpoint overhead. **Every reported PPL still goes through `_ppl()` with TF32
disabled**, so the metric remains on the fp32 criterion and is comparable to the init-stage
tables and to the A8/A11 points.

### Budget, and the early stop

`--batch-size 4` at seq 2048 = **8,192 tokens/step**, chosen on measurement, not
preference: batch 1 is CPU-bound at 11 % GPU utilisation and 536 tok/s, batch 4 is
GPU-bound at 99 % utilisation and **1,370 tok/s** (5.98 s/step, peak 12.6 GiB of 49 GiB).

- **Cap: 6,000 steps = 49.2M tokens**, ~10 h at the measured rate. The cosine horizon is
  these 6,000 steps; a further phase would be a new amendment, not a continuation (A11's
  reasoning).
- **Early stop: PPL < 35.0.** Checkpoints and PPL every 200 steps (1.64M tokens), so the
  target is detected within ~20 min of being crossed. On detection the run stops
training and immediately does the full protocol: full-test-split PPL (all 146 windows at
seq 2048, not the 40-window curve) and full PiQA validation + ARC-Challenge test with
item-level bootstrap 95 % CIs.
- The `itq` arm is run afterwards at **the token count `qubo_act` actually consumed**, so
  the comparison stays matched at whatever budget the early stop lands on.

### Decision rules, fixed now

1. "Double digits" means **full-test-split PPL < 100** at 146 windows. A 40-window curve
   point is a progress indicator, never the headline number.
2. If the early stop fires, the reported figure is the stop point's full-set PPL, and the
   token count at which it fired is the sample-efficiency result.
3. If the cap is reached above 35, that is the reported result, and it is reported as a
   plateau - not extrapolated towards 30.64.
4. The A12 cross-family claim is not revisited here: A13 is one family, two arms.

### What A13 still cannot show

It is not the reference's 1B-token, 30.64-PPL run. At 49.2M tokens it is ~4.5 % of that
budget, one seed per arm, and if it lands in the 35-60 band the correct description is
"reaches the deployable band", not "matches the benchmark".

## Amendment A14 (2026-11-18, post-hoc: third family, first non-Qwen architecture)

**Status: an extension run after A12 and written up after it ran. Not a preregistered gate;
no G4 verdict depends on it.** A12's second family was Qwen2.5-0.5B, so "cross-family" still
meant one lineage. A14 runs the same race on an architecture that shares none of Qwen's
block layout.

Protocol is A12's, unchanged: 1,200 steps, seq 256, batch 1, **307,200 tokens per arm**, same
seeded WikiText-2-only data order (`wiki_repeat=1`), same `kd_loss(lam=10, mse_norm=True, mse_skip_first=1)`, PPL on 40 windows with TF32 off
and the final on the full split.

| | value |
|---|---|
| model | `HuggingFaceTB/SmolLM2-360M` |
| architecture | `model_type=llama`, 32 layers, hidden 960, tied embeddings |
| size | 362M params |
| teacher PPL | 10.9873 |
| bpw | 0.5493 |

Result (`distill_smollm2_{itq,qubo_act}.json`):

| | itq | qubo_act | ratio |
|---|---|---|---|
| init PPL (40 windows) | 6,000,848.4 | 2,401,882.5 | 0.400 |
| final PPL (148 windows) | 407.367 | **267.160** | **0.656 (-34.4 %)** |
| step reaching itq's final | - | **400** of 1,200 | 3.0x fewer tokens |
| final `w_in` | 0.66964 | **0.53637** | better |
| final `unw` | **0.78466** | 0.85787 | worse |

Final-third ratio 0.633 / 0.636 / 0.639 (0.95 % spread). The error signature is the one both
Qwen models showed: lower activation-weighted error, higher unweighted error.

**This amendment retracts A12's mechanism claim.** A12 proposed that the payoff "scales with
how catastrophic the ITQ init was". A third family falsifies that as a general rule:

| family | init gap (itq/act) | final ratio | tokens to match |
|---|---|---|---|
| Qwen3-0.6B (A8) | 6,293x | 0.415 | 5.9x fewer |
| Qwen2.5-0.5B (A12) | 6.5x | 0.849 | 1.5x fewer |
| SmolLM2-360M (A14) | **2.50x** | **0.656** | 3.0x fewer |

The two orderings disagree: SmolLM2 has the *smallest* init gap and a larger final gain than
Qwen2.5, which has a 6.5x gap. "The worse the ITQ init, the bigger the win" therefore
describes only the Qwen3 extreme, and Qwen2.5 - not SmolLM2 - is the outlier. What survives
is the ordinal claim: **QUBO sign search beats Joint-ITQ at equal tokens on all three families
and both architectures tested, by 1.18x-2.41x final PPL and 1.5x-5.9x fewer tokens to
match**, with no supported explanation yet for where in that range a new model lands.

Provenance defect, recorded: `ablation.py --stage init` defaults `--out` to
`results/littlebit/qubo_init.json`, so `run_m2.sh` and `run_m3.sh` each overwrote it - after
A14 it holds SmolLM2, not Qwen3. No reported number depends on it (every arm's init PPL is
also in its own `distill_*.json` and in `RESULTS_qubo_init.md`), the SmolLM2 copy is preserved
as `qubo_init_smollm2.json`, and `run_m3.sh` now passes an explicit `--out`.

## Amendment A15 (2026-11-18, written while A13 `qubo_act` was at step ~2200)

**Status: changes the protocol for runs started after this date. It does not touch A13.**
A13's own early stop stays at "single eval under 35.0", because `chain_a13.sh` launches the
`itq` arm as a fresh `python3 distill.py` at the moment `qubo_act` exits: editing the stop
rule now would apply it to the control arm only, which is exactly the inter-arm asymmetry
the rest of this document goes to some trouble to avoid. Recorded because three defects in
A13's evaluation protocol became visible once its curve stopped being monotone.

### Defect 1: the curve is a deterministic subset, not a sample

`distill.py:335` builds `eval_ch = eval_chunks(tok, args.eval_len)` **once**, before the
training loop, and every periodic eval passes that same tensor with `--eval-batches 40`.
`perplexity()` (`vendor/little-bit/littlebit/evaluate.py:10`) consumes `chunks[0:40]` in
order and stops at `max_batches`. So every curve point is scored on **the first 40 of the
146 WikiText-2 test windows**, identical at every checkpoint, in fp32 with TF32 off.

Consequence, and it matters for how the curve may be read: consecutive curve points differ
by **zero sampling error**. A 1 % wobble (A13: 81.5 -> 82.0 at step 1400, 66.4 -> 67.3 at
step 1800) is a real regression of the quantized state on the same windows, not noise, and
no re-measurement can separate the two. Equally, a single curve point cannot be averaged
away: it is one deterministic function of the weights.

### Defect 2: selection and reporting happen on the same subset

"Best so far" (A13: 64.2 at step 2000) is a minimum taken over 40 of 146 windows, so it
carries optimistic selection bias, and the 35.0 stop is a decision made on those same 40
windows. Neither is wrong for *tracking* - it is cheap and, per Defect 1, repeatable - but
neither may be reported as a result.

Prescription, for A15-era runs:

1. **Two-eval stop.** Training stops after **two consecutive curve evals** below the target,
   not one. A single eval decides 1,640K-token chunks of spend on a 40-window subset.
2. **Curve selects, full set reports.** The 40-window curve may choose checkpoints
   (exploratory only). Every headline PPL is scored on all 146 windows - already the case
   via `--final-eval-batches 0`.
3. **Held-out scoring for any best-checkpoint claim.** A checkpoint chosen as "best" is
   additionally scored on the **106 windows the curve never saw** (`chunks[40:]`) and is
   labelled exploratory. The endpoint checkpoint needs no such correction: its 146-window
   number already contains windows 0-39, so the headline endpoint stays unbiased.
4. **Same held-out set for every arm**, so a two-arm comparison is never affected by which
   windows an arm happened to be selected on.

Point 3 is not implemented yet: `perplexity()` has no offset argument, so held-out scoring
needs a small `chunks[40:]` wrapper. Recorded here as the follow-up, not as done.

### Defect 3: the flip-fraction model was cited as a forecast, and it should not be

Regressing A13's log PPL on flip fraction over steps 200-2000 gives U: slope -0.5715,
R^2 0.903, i.e. PPL 35 at flipU 4.45 % (3.80 % at step 2000); V: -0.7228, R^2 0.871, 35 at
flipV 3.29 %. Refitted to the last six points only, the U map demands 6.06 %, not 4.45 %:
the exchange rate between flips and PPL is itself decaying.

Those two thresholds give very different answers, and only the pessimistic one survives
arithmetic. U rises 0.24-0.26 % per 200 steps and decelerating (decay factors 0.969, 0.935,
0.897, 0.923):

| threshold | rate / decay | crossing | verdict |
|---|---|---|---|
| 4.45 % (all 10 points) | any | ~3 intervals, step ~2600 | **rejected** - needs 64.2 -> 35 in 600 steps after 800 steps bought only 81.5 -> 64.2, i.e. twice the observed rate |
| 6.06 % (last 6 points) | 0.24-0.26, constant | 9-10 intervals, step 3800-4000 | optimistic |
| 6.06 % (last 6 points) | 0.24, decay 0.947 | 13 intervals, step ~4600 | central |
| 6.06 % (last 6 points) | 0.24, decay 0.930 | 15 intervals, step ~5000 | pessimistic |
| 6.06 % with faster decay | plateau below 6.06 % | never | live possibility |

The first-pass reading of that table, at step 2000, was "step 3800-5000 if the exchange rate
holds, never if it keeps falling". It is quoted here only to show what the numbers said then;
the out-of-sample check below supersedes it. R^2 0.90 in that table is close to guaranteed:
PPL and flip fraction are both monotone in step, drawn from ten autocorrelated points with
two free parameters, so it mostly says "both rise with time". The 3.80 % -> 6.06 %
extrapolation was also a 60 % reach beyond the observed flip range.

Out-of-sample checks, added as the next three evals landed. The all-10 map predicts, against
the observed curve:

| step | flipU | map predicts | observed | error |
|---|---|---|---|---|
| 2200 | 4.02 % | 44.8 | 59.7 | +33 % |
| 2400 | 4.21 % | 40.2 | 58.5 | +46 % |
| 2600 | 4.38 % | 36.5 | **55.6** | +53 % |

The 4.45 % crossing the map places at PPL 35 arrives in the data at step 2600 with PPL
55.6. The branch is not merely implausible, it is **refuted at the value it predicted**: the
map is a correlation between two series that both rise with step, not a mechanism, and its
exchange rate is falling faster than the flip rate is (
all-13 refit demands flipU 4.91 %, last-6 7.05 %, last-4 6.36 %, while the flip rate is now
decaying 15-23 % per interval and plateaus near 5.2-5.6 %). Under the refitted maps the
crossing does not exist at all, which is why the flip model is retired as a forecasting tool
and only the direct PPL pace is used below.

### The forecast, on the direct PPL pace only

From step 2600 at PPL 55.6, ln(35/55.6) = -0.463 and the last three intervals delivered
-0.0480 per interval:

| deceleration of the log-rate | intervals to 35 | crossing |
|---|---|---|
| none | 9.7 | step ~4530 (2.5 h) |
| 5 % per interval | ~13 | step ~5200, cap edge |
| 10 % per interval | ~33 | never inside 6,000 |

**P(stop at 35.0 fires) = 30 % (range 25-35 %), recorded before the outcome.** That is a
small raise on the 25-30 % stated at step 2000, and the reason is only that the direct pace
puts the crossing inside the cap rather than at its edge - not the flip map, which is retired.
The floored curve fits (asymptote 43-54) still argue the other way, and 55.6 at step 2600 is
above every one of those asymptotes. If the stop does not fire, the result is the step-6000
endpoint reported as a plateau, per A13 decision rule 3.

**Update at step 2,800 (2026-11-18, still before the outcome): the 30 % above is
superseded.** Step 2,800 reads PPL 57.5 after 55.6 - the third micro-regression on identical
windows, after 81.5 -> 82.0 and 66.4 -> 67.3 - with flipU 4.53 % and flipV 3.44 %. Both
flip thresholds the map proposed (U 4.45 %, V 3.29 %) are now crossed with PPL at 56-58, and
the map over-predicts by 72 % there (33.5 predicted, 57.5 observed). On the last four evals
the log-rate is 0.973 per interval, which places 35 at step ~6,400 - just outside the cap;
on the last three it is step ~10,700. **P(stop fires) now 15-20 %.** The floored fits do not
narrow this: an all-14-points power law gives a 22.7 floor and tail-only fits 43-54, so the
asymptote is unidentified by this data and is not being used as a prediction either way.

**Second update at step 3,200 (2026-11-18, still before the outcome): the 15-20 % above was
written on a single regressed eval and was too pessimistic. P(stop fires) 20-30 %.** Steps
2,800-3,200 read 57.5 -> 56.2 -> 52.4 with flipU 4.53 -> 4.66 -> 4.77 %, so the step-2,800
rise was noise of the same order as the two before it, and revising on one interval was a
mistake I am not repeating: the interval-to-interval variance now swamps the pace estimate.
Recorded honestly rather than scrubbed - the 30 %, the 15-20 % and this line all stay.

The pace itself is genuinely ambiguous, which is why the band is wide: 35 lands at step
~4,900 on the last two intervals, ~6,100 on the last four, ~7,300 on the last three
(the third still contains the 57.5 wobble). Projected step-6,000 PPL is 27-40 depending on
which window is used, i.e. the cap straddles the threshold. Worth stating plainly that a
free refit of the flip map now reports a shallower slope and only 20 % over-prediction at
U 4.77 - that improvement is an artefact of refitting on the points it is being scored
against. Held at the original all-10 slope, the out-of-sample miss is 29.2 predicted against
52.4 observed, 80 % over, and it has grown at every eval since step 2,000.

**Third update at step 4,400 (2026-11-19, still before the outcome): the stop will almost
certainly not fire. P(stop fires) < 10 %, superseding the 20-30 % band above.** Steps
3,800-4,400 read 50.2 -> 49.7 -> 47.8 -> 49.1, with flipU 5.03 -> 5.09 -> 5.13 -> 5.17 % and
increments 0.06 -> 0.04 -> 0.04. Fitting the decay of the flip increments over the last eight
intervals gives r = 0.836 for U and 0.831 for V, i.e. plateaus of 5.41 % and 4.22 %: U has
about 0.25 points of flippable mass left, so the sign lattice is nearly exhausted while PPL
still moves by +-1.3 between evals. On the last eight intervals 35 would land at step ~7,800;
the log-linear fit from step 2,200 gives ~7,400 and a step-6,000 endpoint of 40.4. Every
window now puts 35 beyond the cap, the two- and four-interval windows by a factor of five or
more. The endpoint is therefore the reported result, per decision rule 3, and the +1.3 wobble
at step 4,400 is a fourth instance of the pattern noted at 2,800 - a large drop followed two
ev iterations later by a small rise - which we are recording as observed behaviour, not
explained. Held at its original slope, the flip map predicts 23.2 at U 5.17 against 49.1
observed: 112 % over.

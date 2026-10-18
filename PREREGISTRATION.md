# Pre-registration

Written before any Project A / Project B data collection. Gates are frozen; do
not edit after runs begin. Any change gets a new dated section and a re-run.

Pinned build: llama.cpp `547208dca250ceb3e4c8615598d043f2b602a527` (b11718).
Builds: `build/cuda-graphs` (CUDA graphs ON), `build/cuda-nographs` (OFF).
GPU: RTX 4060 desktop 8 GB, driver 617.42, WDDM, 115 W. No NVML power telemetry.

## Shared rules

- **Splits.** 200-problem dev split used only for tuning (N, vote rule, caps,
  brief length). All reported numbers are on the held-out remainder.
- **Statistics.** Paired bootstrap (10k resamples) or McNemar on per-problem
  outcomes; 95% CIs. 3 sampling seeds for anything stochastic (default
  1234/5678/9012).
- **Logging.** Every CSV row carries: commit, config (graphs/nographs),
  kv_unified, model file, seed, gpu clock. Non-unified rows flagged
  `kv_unified=0` and excluded from headline plots.
- **Hygiene.** Close GPU apps; warmup pass discarded; report p50 and p99.

## Project A - parallel reasoning on a latency frontier

### E0 cost curve (no gates; characterization)

Step time for N = 1, 2, 4, 8, 16 streams sharing a prompt, with unique
(unshared) tails at depths 0, 512, 1024, 2048 tokens. Unified KV.

**Stop gate.** If E0 shows N=8 is more than **2.0x** the N=1 step time at a
realistic tail depth (>=1024), the latency claim fails and Project A stops.

### E2 gates

- **Pass:** N=8 on the 1.5B costs at most **1.6x** the N=1 wall-clock latency
  AND gains at least **3 points** accuracy (paired 95% CI excludes zero).
- **Flagship (small ensemble beats larger single model at matched wall time):**
  may fail on MATH-500. Report honestly; a 1.5B ensemble may still win on GSM8K.
- E3 diversity is exploratory (no pass/fail), reported as vote accuracy and pass@N.

### Metrics (A)
accuracy, wall-clock s/problem (median and mean), total tokens, aggregate tok/s,
peak VRAM, step-time-vs-step curve, straggler ratio (max trace / median trace).

## Project B - staged memory brief

Reader and summarizer: same 3B Q4. Memory store built offline (short bullets per
session). Retrieval on CPU with a small embedding model. 30% of probes are topic
jumps (turn N off-topic) to measure hit rate and fallback.

Arms: B0 no memory; B1 raw top-k at turn N+1 (budgets 500 / 1000 tok);
B2 staged brief (retrieve at turn N, piggybacked <=200-token summary, appended
before turn N+1); B3 staged raw (retrieve at turn N, paste raw); B4 full-context
upper bound (3B at 16k).

### Gates (B)

- **Pass:** B2 matches B1's accuracy within **1 point** with at least **50%
  fewer inserted tokens** at equal TTFT, AND does at least as well as B3.
- If B2 does not beat B3, summarization adds no value beyond staging.

### Metrics (B)
systems: TTFT at turn N+1, A's added latency during turn N, fraction of briefs
finished before A's reply ends (reply lengths 20..500), hit rate, inserted tokens.
quality: token F1 vs gold, plus a manual check of 50 samples (local judge is weak).

## Order
1. Shared infra + these gates.
2. E0, then E1 harness, E2 GSM8K, E2 MATH-500, E3.
3. Project B infra, then B0..B4.
4. Cost-curve write-up (kernel regimes, unified-KV sharing, unshared-tail curve)
   as the common thread; A and B as the two demonstrations. Prior-art search on
   each result before committing to a framing.

---

## AMENDMENT 1 - 2026-10-18 (after E0 cost data, before any accuracy data)

E0 (cost only, no accuracy information) showed N=8 sits on the slow side of the
MMVQ->MMQ kernel switch (ne11=9); N=16 is cheaper per step than N=8 at short
tails. The original gate was written for N=8.

- The **primary gate stays exactly as written** (N=8 <= 1.6x wall clock). N=8
  peak step time is 1.70-1.89x; recorded as-is.
- **N=12 and N=16 are added as AMENDED arms** (cost-motivated, pre-accuracy) and
  will be reported labeled "amended".
- Two exploratory, cost-only additions:
  - **Quantized KV arm**: re-run E0 with q8_0 (and q4_0) KV cache, since the tail
    term is KV bandwidth. Any accuracy effect measured on the 200-problem dev
    split; V-cache quantization requires flash attention (fa=1).
  - **Larger-model E0**: 7B Q4 (N<=8, tail<=1024), since the tail penalty is
    relative to weight size. Also record the model's actual KV bytes/token
    (architecture-dependent KV-head count).
- **Stopping design for E1/E2**: first-K-agree is the PRIMARY stopping rule
  (stop when K finished streams agree), replacing wait-for-all. Wall clock is
  measured per problem against N=1's median wall clock on the same problems,
  with the same temperature and length cap. Log straggler ratio (max/median
  trace length) and finish-time quantiles.

No accuracy data informed this amendment.

---

## AMENDMENT 2 - 2026-10-18 (after dev-split simulation, before the held-out run)

### Frozen configuration
**N=16, K=3, first-K-agree, temperature 0.6, cap 1024 tokens, unified KV,
flash attention on.** Chosen on the 200-problem dev split from
`results/dev200_s16.tsv` + the E0 cost model (`harness/simulate.py`):

- dev N=1 baseline: 64.0% accuracy, median 486 tokens.
- N=16 K=3: 85.5% accuracy at a predicted 1.41x wall clock (+21.5 points).
- Median stopping length = 0.88x the N=1 median length.

### Added baselines (to be reported on the held-out split)
- **N=1 with loop detection and restart**: if the single trace repeats a suffix
  or hits the cap without a completed answer, discard and resample (up to R
  restarts). This is the fair strong baseline, since the ensemble's advantage
  partly comes from avoiding degenerate traces.
- **N=2 with retry**: two samples, retry the second if the first is degenerate.
- **short-m@k-style variants**: m of N samples at a matched token budget.

### Retirement handling and wall-clock measurement
- A stream retires when it emits a completed answer (`\boxed{...}` after the
  think block) or hits the cap.
- The live engine calls `llama_memory_seq_rm(seq)` on retirement. E0 retirement
  test (results/retire_rm1.csv vs retire_rm0.csv): seq_rm vs no-rm changes the
  step time by ~1%; the dominant effect is the MMVQ(<=8)/MMQ(>=9) kernel regime
  (retiring to 8 active is *slower* per step than 16 active). The cost model
  already encodes this via the per-N E0 table.
- Wall clock = sum over steps of `cost(active_streams(t), tail=t)` from the E0
  table; N=1 baseline = its own single-trace wall clock on the same problems.

### Retirement risk note
The simulation assumes a retired stream stops costing. Measured: leftover KV
cells cost ~1% (negligible), so the assumption holds to within the cost model's
accuracy (~3%).

---

## AMENDMENT 3 - 2026-10-18 (cost model completed; composition prediction frozen)

### Full E0 table (N=1..16, tail 0/512/1024/2032)
`results/e0_full.csv`. The kernel cliff is exactly at N=8 -> N=9 (ne11=9):
at tail 1024, N=8 = 10.56 ms vs N=9 = 8.96 ms, i.e. **-1.60 ms for +1 token**.
Within the MMQ regime (N>=9) step time grows gently (8.96 -> 10.76 ms).

### Scheduling rules (measured)
- **Never idle at 8.** Padding 8 real streams to 9 with filler tokens makes the
  step *cheaper* by ~1.2-1.3 ms at every tail depth (results/pad8_9.csv vs
  e0_full N=8). Treat the 8-slot as a hole to fill with useful work
  (consolidation, speculation); filling it is negative-cost.
- **Always seq_rm on retirement** (cheap). Leftover cells cost ~1% at retire@512
  and a few percent at retire@1024; the residual does not grow enough to matter.

### Composition prediction (written before the run)
For a composition of processes on one decode step:
```
predicted_step_ms = E0_lookup(active_tokens, tail_depth)
                  + sum(probe_tokens) * marginal_ms_per_token(active_tokens)
```
where `E0_lookup` is bilinear interpolation on `results/e0_full.csv` and
`marginal_ms_per_token` is the local slope of that table.

**Acceptance band: live p50 within 5% of predicted p50**, per composition
configuration. Residuals are reported by active count; if they cluster at the
N=8/9 boundary, the cost model's treatment of the cliff is what needs work.

### Composition test configurations (in order)
1. foreground alone
2. foreground + 4 traces
3. foreground + 16 traces
4. each of the above + a probe every 64 steps
5. + the consolidation job (last)

### Correctness invariants (engine)
- foreground tokens byte-identical with and without the other processes
- batch-index logits routing test passes
- every step logs active count, kernel regime (MMVQ/MMQ), step time, GPU clock

---

## AMENDMENT 4 - 2026-10-18 (L40 for traces/accuracy; 4060 for wall clock)

To avoid the 4060 grind, traces and accuracy are generated on a Thunder Compute
L40 and wall clock is simulated from the 4060 cost table. Frozen gates unchanged.

### Hardware split
- **Accuracy / traces / stopping rules: L40** (same model file, same Q4_K_M
  quantization, same sampler temp 0.6, same stopping rule, 3 seeds).
  L40: 48 GB, driver 610.43.02, Ubuntu 22.04, CUDA 13.x, Ada (sm_89). Bandwidth
  ~864 GB/s (to be confirmed by the L40 E0 curve).
- **Wall clock: simulated** by replaying L40 traces through `harness/simulate.py`
  with the **4060** E0 table (`results/e0_full.csv`). The 4060 is the target
  hardware; L40 timing does not transfer (different bandwidth and
  compute-to-bandwidth ratio), so it is never reported as the latency result.

### Traces transfer, timing does not
Traces from a different GPU differ in the last floating-point digit but are
statistically equivalent (see `results/identity_break.txt`: KL ~1e-3). All arms
(N=1,4,9,12,16), K=2..5, 3 seeds, and the Amendment-2 baselines run on the L40.

### Live validation (4060)
A single live run of the **primary arms (N=1 and N=16, K=3)** on a fixed
**150-problem subset** of the held-out split, to confirm the simulator.
- **Tolerance: live vs simulated wall clock within 5% at p50**, and the sign of
  the residual consistent with the Amendment-3 composition residuals.
- If it agrees, all other arms are reported as **simulated** and labeled so.
- The registered claim is confirmed by the live primary arms (one seed).

### Labeling
Every row from the L40 carries `gpu=L40`, `driver=610.43.02`, `os=ubuntu22.04`
in the logging schema. Any L40-derived number is labeled `source=l40_sim_4060`.
Simulated wall-clock rows are labeled `wallclock=simulated(4060)`.

### E0 on the L40
Exploratory second cost curve (cliff size and unshared-tail penalty scale with
bandwidth), reported as characterization, not as the latency result. Optional:
NVML energy on Linux (not available on the 4060).





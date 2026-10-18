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


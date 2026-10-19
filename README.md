# RecycleLM

Measuring where spare GPU compute on a consumer card (RTX 4060, 8 GB) can be
reused: batching secondary work (consolidation, parallel reasoning, memory
briefs) into a primary decode step, and characterizing the hardware limits.

Everything here is measured on one machine; results are hardware-specific and
are not extrapolated to other GPUs.

## Findings so far (short)

- **Marginal decode cost is low but not free.** On a 3B Q4_K model, an extra
  batched token costs ~0.1-0.35 ms against a ~10 ms step. The 5% knee is
  ~1-2 tokens; the useful framing is marginal cost, not "free tokens".
- **Kernel regimes.** Batches up to `ne11<=8` use the vector mat-vec kernels
  (per-column cost); larger batches switch to the tensor-core MMQ path with a
  fixed entry cost. N=8 is the worst point; N=16 is cheaper per step than N=8.
- **Unified KV is required for multi-sequence work.** Non-unified KV shows a ~2x
  step-time cliff when a batch spans two sequences with a multi-token chunk.
  Common prefixes share KV reads; distinct prefixes cost ~1.85x.
- **Quantized KV does not help here** (dequant overhead cancels the bandwidth
  saving; quantizing V is actively worse).
- **Early-exit probes (offline).** k-consecutive-agreeing probes cut ~28% of
  reasoning tokens at ~1 pt accuracy loss, beating fixed-budget and
  logprob-confidence baselines.
- **Parallel reasoning (live, 400 held-out).** N=16 with first-K-agree (K=3):
  **+11.8 pts** over a single sample, 95% CI [+7.8, +15.8], at **1.37x** median
  wall clock (batched decode with shared KV). Paired McNemar confirms it
  (N=1 vs N=16, p<0.001).
- **Accuracy vs N (200 held-out, paired).** Plateaus at N=8: N=8 beats N=4 by
  +5.5 pts (CI [+1.5, +10.0], p=0.019), and N=16 is within noise of N=8
  (-2.0 pts, CI [-5.5, +1.5], p=0.42) while costing ~1.8x the tokens
  (6702 vs 3662). So N=16 buys no measurable accuracy over N=8.
- **Edge (ratio-matched estimate, not a phone measurement).** Desktop gives
  1.37-1.64x; a phone CPU engine ~2.3-5x, an NPU engine ~3.7x. Recommend
  **N=4 as the phone default** (bandwidth-bound below the compute crossover, so
  extra streams add little latency) and **N=8 as an optional high-accuracy
  mode**. Accuracy saturates around N=8 within noise; token cost is what a phone
  pays.

Known limitations (energy telemetry, the ggml N=8 threshold, untested
phone-sized-model accuracy, an N=1 cap caveat) are in `LIMITATIONS.md`.

- **Voting helps, but less as models weaken; ensembling buys points, not
  competence.** Sub-1B sweeps (60 held-out each), ensemble vote vs the
  low-variance per-stream average:

  | model | N=1 | N=8 vote (per-stream) | N=16 vote (per-stream) |
  |---|---|---|---|
  | Llama-3.2-1B | 50.0 | 58.3 (36.0) | 68.3 (25.1) |
  | Qwen2.5-0.5B | 41.7 | 58.3 (34.0) | 55.0 (25.1) |
  | SmolLM2-360M | 13.3 | 21.7 (14.6) | 20.0 (10.1) |

  The gain over the per-stream average shrinks as the model weakens (+43 pts on
  the 1B, +10 on the 360M): weaker models make *correlated* errors, so votes
  cluster on the wrong value. A 1B reaches ~68% with N=16; a 360M reaches ~22%,
  which is not usable. Ensemble diversity (different prompts per stream) is the
  natural next lever for tiny models.

See `PREREGISTRATION.md` for the frozen gates and the dated amendment.

## Layout

```
harness/    C++ measurement tools + Python analysis
  phase1b.cpp            mixed decode+chunk knee harness (NVML energy, clocks)
  earlyexit.cpp          offline trace + forced-answer probe tool
  parallel_cost.cpp      E0 cost curve: N streams, unshared tails, KV types
  trace_gen.cpp          N sampled traces/problem (parallel, per-stream sampler)
  int4_bench.py          int4 (tinygemm) vs fp16 matmul at small M
  analyze_earlyexit.py   stop-rule Pareto + baselines
  analyze_e0.py          E0 summary + gate verdicts
  simulate.py            replay simulator over fixed traces + cost model
scripts/    build + data prep
  build_llama.bat        build llama.cpp CUDA (graphs on/off)
  build_target.bat       build one llama.cpp target (e.g. test-backend-ops)
  build_tool.bat         build one harness .cpp
  prep_gsm8k.py          GSM8K jsonl -> TSV
  run_phase1a.ps1        pure-batching sweep with a context-budget guard
data/       small inputs (GSM8K subset)
results/    summaries, pareto CSVs, E0 cost curves
PREREGISTRATION.md   gates, frozen before runs
```

## Build

Requires VS 2022 Build Tools (MSVC + bundled CMake/Ninja), CUDA 13.x.

1. Fetch llama.cpp at the pinned commit:
   ```
   git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
   git -C vendor/llama.cpp checkout 547208dca250ceb3e4c8615598d043f2b602a527
   ```
2. Build two CUDA configs:
   ```
   scripts\build_llama.bat S:\...\build\cuda-graphs   ON
   scripts\build_llama.bat S:\...\build\cuda-nographs OFF
   ```
3. Build a harness tool:
   ```
   scripts\build_tool.bat build\cuda-graphs harness\phase1b.cpp harness\phase1b.exe
   ```

Models (fetched separately, not in this repo):
- `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (bartowski)
- `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` (bartowski)
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`

## Environment

RTX 4060 desktop 8 GB, driver 617.42, WDDM, 115 W. No NVML power telemetry on
this card (`nvidia-smi` reports N/A), so energy work needs a wall meter.

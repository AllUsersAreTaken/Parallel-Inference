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
- **Parallel reasoning (simulated).** N=16 with first-K-agree (K=3) gives
  +21.5 points over a single sample at a predicted 1.41x wall clock.

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

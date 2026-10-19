# Known limitations

Open items and caveats that bound the claims. None of these are fatal; they
define where the results stop.

## Measurement

- **Energy is wall-meter only.** The RTX 4060 reports `power.draw = N/A` via
  NVML and `nvidia-smi dmon` (`-`), so per-run joules were never measured. The
  harness reports `method=unavailable` and `nan`. Phase-3 energy claims need a
  smart plug.
- **Clock locking is unavailable** on this card (`nvidia-smi -lgc` -> "does not
  have permission"; GeForce/WDDM, non-admin). Runs are therefore subject to
  thermal drift; every row logs the SM clock and throttled runs are flagged.
- **The N=8 step-time dip is a regime, not physics.** Both the 1.5B and 7B CPU
  curves dip at N=8 (ratio 3.47x / 3.44x, throughput below N=4). That is a
  batch/threading threshold in the ggml CPU matmul path. The exact threshold was
  not located in `ggml-cpu`; do not read N=8 as a meaningful operating point
  until it is.

## Claims

- **Accuracy transfer to phone-sized models is untested.** The +11.8 points
  (Test 1) came from DeepSeek-R1-Distill-Qwen-1.5B. Ensemble gains vary with
  model size; a 0.5-1B phone model would need its own run.
- **The edge estimate is a ratio-matched estimate, not a phone measurement.**
  Anchors are the desktop 4060 and the 9800X3D CPU. Phone figures (56 t/s
  prefill, 690 t/s NPU prefill, ~8 t/s decode) are from secondary sources
  (arXiv mobile study; Qualcomm AI Hub marketing) and are approximate.
- **The desktop CPU is not compute-limited at N=16.** Measured: 822 GFLOP/s
  decode against a 14.5 TFLOP/s prefill ceiling (5.6%). The phone CPU, 17x lower
  compute, would be ~97% of its ceiling at N=16. The N=16 ratio range on phones
  (~2.3-5x CPU engine, ~3.7x NPU engine) is therefore an estimate.
- **A1r (loop-detect + restart) is a partial baseline.** It retries on a
  repeated suffix or a missing answer; it does not yet test loops that emit a
  wrong answer. Report it as "retry-on-loop does not close the gap", not "no
  single-trace method can".
- **The N=1 baseline may be capped, so the ensemble gain may be overstated.**
  In the reported Test-1 and sweep runs, `max_trace` was compared against the
  absolute sequence position, so the effective generation cap was
  `1024 - prompt_len` (roughly 500 tokens for these prompts), not 1024. About
  52% of N=1 traces ended at >= 500 tokens and scored 61% vs 83% for the
  shorter ones. There is no spike at a single cap value (prompt lengths vary),
  but long N=1 traces were truncated. The bug is fixed in `e2.cpp` (cap is now
  relative); the reported numbers predate the fix. A cheap follow-up is to
  rerun N=1 only, at a higher relative cap.

## Memory (Test 2)

- **The distractor set is template-matched and harsh.** B1, B2 and B4 all
  collapse to 10-12% under 150 distractor bullets/raw turns that share the
  planted-fact templates. Real conversation is likely gentler. The scale test is
  symmetric (B1's index is stressed too) but synthetic.
- **Capture is the ceiling (65%).** Bullet quality is limited by the 3B writer;
  select-and-paste fixes the brief stage (P(brief|ret)=100%) but not capture.
- **No dense retrieval / reranking.** The standard fix for retrieval at scale
  (small CPU embedding model + rerank) was not implemented.

## Scope

- Single GPU (RTX 4060 8 GB, WDDM), single CPU (9800X3D), one OS (Windows).
- GSM8K only for the parallel-reasoning and early-exit results; MATH-500 not run.
- Test 3 (retrieval brief) held until B2 clears its gate.

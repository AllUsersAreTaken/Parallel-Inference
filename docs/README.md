# Research record

The code, harnesses and benchmarks live in `harness/`, `app/`, `parallel/`,
`scripts/` and `results/`. This folder holds the written record that would
otherwise be a paper: what was predicted, what was measured, and where the
predictions were wrong.

Read in this order if you want to judge the work rather than run it.

## Gates, frozen before data collection

| file | covers |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | The main gates: parallel reasoning (Project A), staged memory brief (Project B). Six dated amendments. |
| [`PREREGISTRATION_QUBO_INIT.md`](PREREGISTRATION_QUBO_INIT.md) | Does discrete sign search beat Joint-ITQ at a fixed bits-per-weight budget? Fifteen amendments, including a retracted mechanism claim. |
| [`PREREGISTRATION_RERANKER.md`](PREREGISTRATION_RERANKER.md) | Does the sign-search advantage survive decision-head distillation? |
| [`PREREGISTRATION_READOUT.md`](PREREGISTRATION_READOUT.md) | Readout vs rate as the binding constraint on a compressed state. |
| [`PREREGISTRATION_RECALL.md`](PREREGISTRATION_RECALL.md) | Associative recall at a fixed state size. |
| [`PREREGISTRATION_FANOUT.md`](PREREGISTRATION_FANOUT.md) | Fan-out / parallelism sweeps. |
| [`PREREGISTRATION_LINTER.md`](PREREGISTRATION_LINTER.md) | Structured-output linter experiments. |

Amendments are appended and dated, never edited into the original claim. Several
results are negative, and three retract earlier results of mine:

`PREREGISTRATION_QUBO_INIT.md` and `PREREGISTRATION_RERANKER.md` are implemented
in a separate repository, [`littlebit-qubo`](https://github.com/AllUsersAreTaken/littlebit-qubo),
so that the code being scored and the document that fixes the scoring are not
edited in the same commits.

- `PREREGISTRATION_QUBO_INIT.md` **A10** — a change improved the surrogate
  objective by 3.8% and made the metric the claim was written on 1.9x worse.
- Same file, **A14** — a third model family falsified the mechanism claim made in
  A12; the amendment says so and narrows the surviving claim.
- Same file, **A13/A15** — a forecasting model was refuted out-of-sample at the
  value it predicted and retired as a tool, with the refutation left in place.

## Narrative and theory

| file | covers |
|---|---|
| [`DEMO_context_compaction.md`](DEMO_context_compaction.md) | Shared-prefix answer+memory fork: 25% sooner both-ready, +11% answer latency, 78% smaller follow-up context. |
| [`ASYNC_SELFCORRECT_DEMO.md`](ASYNC_SELFCORRECT_DEMO.md) | Asynchronous self-correction demo and what it does not show. |
| [`THEORY_inference_state.md`](THEORY_inference_state.md) | Why the KV cache is the *minimal* state for the function attention computes, what a predictive state has to be instead, and the measurement that narrows the claim. |

## Where the claims stop

[`../LIMITATIONS.md`](../LIMITATIONS.md) at the repository root. It is not a
formality: it names the unmeasured energy, the unlocked clocks, the kernel-regime
artifact at N=8, and the cap bug that predates the reported accuracy numbers.

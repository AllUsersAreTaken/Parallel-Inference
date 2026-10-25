# RecycleLM Playground

A small GUI for the parallel-stream runtime: **one model, 1–10 concurrent
streams**, each with its own prompt. The backend batches the decodes on the
GPU and reports per-stream timings.

## Run

```
app\run_playground.bat
```

or

```
python app\parallel_playground.py
```

The backend is `app\parallel_cli.exe`, built against the existing CUDA build:

```
scripts\build_tool.bat build\cuda-graphs app\parallel_cli.cpp app\parallel_cli.exe ^
  parallel/include build\parallel\parallel.lib
```

## Controls

- **Streams** 1–10. Each stream gets a prompt box and an output box.
- **Max tokens**, **Temp**, **Seed**. Streams use `seed + i`, so they diverge.
- **System** prompt, shared by all streams.
- **Model** defaults to `models\Llama-3.2-3B-Instruct-Q4_K_M.gguf`.
- **Copy #1 to all** duplicates the first prompt.
- **Run** starts one backend run with all N branches; **Stop** kills it.

## What the numbers mean

Per stream (server/runtime reported):

- **tok/s** = generated tokens / generation time (`completion_ms`).
- **total** = the stream's completion time (generation; the GPU is shared).
- **tokens**, **ttft**, **finish**.

Footer:

- **wall** = the whole run from submit to all branches done.
- **aggregate** = total generated tokens / wall.
- **per-stream avg** = mean of the per-stream tok/s.
- **solo sum** = sum of the N=1 baseline completion times (same prompts/seeds).
- **speedup** = solo sum / batched wall (the true batching speedup).
- **median latency** = median per-stream (batched / solo completion − 1).
- **overlap** = sum of the batched completion times / wall; not a speedup.
- **Scaling N=1..10** = replicate stream 1's prompt at each N and show
  per-stream latency, TTFT, aggregate tok/s, and both ratios versus N=1.
- **joint** = ticks with ≥2 branches active / total ticks.

## Notes

- The model is loaded **once per run** and shared by all N branches. This is
  continuous/batched decoding, **not** weight streaming.
- The first run of a session includes the model load (~1–2 s); it is outside
  the runtime's reported wall.
- `app\_gui_input.tsv`, `app\_gui_out.json`, `app\_gui_run.log` are scratch
  files from the last run.
- The 3B Q4 model fits the 8 GB card at 10 streams; longer prompts or contexts
  reduce the safe stream count.

## Edge memory engine (Memory tab + `memory` subcommand)

The frozen k=3 chain from research rows 33–35, packaged two ways that share the
same state file (`app\memory_state.json`):

- **GUI** — the **Memory** tab: type a message and hit **Send**, or click
  **Demo turn 1/2** for the frozen vehicle dialogue. The tab shows the
  assistant reply, last-turn facts, the live graph-edge table (open vs
  superseded), indexed queries, and a color-coded event log
  (green `ADD`, red `CLOSE`, amber `CONFLICT`, blue repair/query lines).
  The **Memory graph** canvas draws the graph as it grows: `User` at the
  center, one node per remembered object on a ring around it, edges labeled
  with the relation. Open edges are solid green; superseded versions remain
  as dim dashed "ghost" edges, so you can watch values evolve (e.g. the
  Honda edge going quiet when the Tesla edge lights up). The graph redraws
  after every turn and on resize. The meta line under the reply shows the
  batching story: `k=3 batched · wall X ms · slot completions a/b/c ms` —
  three streams, one decode. **Reset memory** clears the state.
- **CLI** — same engine, for scripting and logs:

```bat
app\run_playground.bat memory --demo               # frozen 2-turn vehicle demo
app\run_playground.bat memory --turn "your message" # run one turn
app\run_playground.bat memory --show               # print current graph state
app\run_playground.bat memory --reset              # clear app\memory_state.json
```

The memory run's model/params are the frozen recipe (Llama-3.2-3B, temp 0.2,
seed 1234, max 256) and are intentionally not editable in the GUI, so results
stay comparable to the recorded experiments.

Each turn runs one batched decode with three parallel streams:

- **Slot 0** — assistant reply (same prompt/latency as solo: ~1.1x).
- **Slot 1** — scribe: emits `{"facts": [...], "relations": [[subject,predicate,object],...]}`.
  Subject rule (amended 2026-10-21): `"User"` **or** a concrete entity
  (pi4, car) — verbs/actions/pronouns are never subjects, so entity-to-entity
  triples like `[pi4, RUNS, Plex]` work without reopening the verb-in-subject
  failure. The patch is applied at prompt load in `memory_engine.py`; the
  recorded evidence TSV stays byte-identical. **Delta scoping** (same date):
  the prompt now instructs the scribe to extract only from the latest user
  message, not re-extract the whole transcript — this was the main latency
  lever (T2 wall 1225→638 ms, Slot 1 100→16 tok on a two-turn test). The
  delta sentence lives only in the proximal tail before `Output:`; head
  placement was measured and does not bind.
- **Slot 2** — reverse-HyDE: emits three search queries over the conversation
  (runs within the shared 256-token budget; typically ~43 tokens).

Host-side deterministic work (no model call):

- `robust_parse` repairs Slot 1's one recurring defect (premature EOS missing a
  trailing `}`).
- `recover_parse` is a Slot 2 safety net: if the primary JSON parse fails (the
  frozen brace-repair included), it grabs the first complete `{...}` object
  and ignores any stray trailing character (a recurring EOS artifact where the
  model emits a lone `"` after the closing `}`). Used only on the failure
  path, so no gate-scoring behavior changes; logged visibly when it fires.
- `clean_entity` normalizes every subject/object before it touches the store:
  strips leading articles (`a pi4` == `pi4`, so re-extraction with an article
  dedupes instead of forking a second node) and predicate-bleed phrases
  (`USE FOR homelab/server` → `homelab/server`; bare junk like `USE FOR` is
  dropped with a visible note).
- **Misaligned-triple guard** (2026-10-21): a triple whose predicate slot is
  not uppercase (`["Owns","Cheddar","a plush toy"]` — verb shifted into the
  subject slot) is dropped and reported as a conflict instead of being keyed
  into the store as a garbage edge. All valid historical triples satisfy the
  uppercase-predicate rule (schema rule 2); this closes a host-side failure
  mode that previously inserted verb-subject keys like `owns|cheddar|…`.
- Relations are applied to a bi-temporal edge store with **predicate
  cardinality routing**. Functional predicates (`SINGLE_VALUED_PREDICATES`:
  drives, lives_in, employed_by, current_location, marital_status,
  located_in) are keyed `(subject, predicate)` — a new value closes the open
  version (CLOSE) and appends the new one (ADD). All other predicates are
  **set-valued** (has, owns, likes, knows, bought, …): keyed
  `(subject, predicate, object)`, so members coexist and a later
  `[HAS, cookies]` never closes `[HAS, butter]`. Unknown predicates default
  to set-valued (a missed close is recoverable; a false close silently drops
  a fact). Within one turn, a functional key keeps the first value and
  reports later different values as conflicts; set-valued duplicates of the
  exact same object are no-ops. Superseded versions are retained, not
  overwritten, so the graph keeps its history. State persists in
  `app\memory_state.json` across turns; older state files are upgraded in
  place on load.
- Within one turn, the first triple per `(subject, predicate)` wins; later
  conflicting values are dropped and printed as `graph CONFLICT` (the model
  sometimes echoes the old entity alongside the new one).

Expected `--demo` outcome: after turn 2, the old vehicle edge is closed and a
new vehicle edge is open (typically `user|drives -> Tesla`; the exact predicate
verb can vary run-to-run between `drives`/`owns`/`has` — model variance on
predicate choice, not a host defect; the graph stays semantically coherent).
Timing is one ~0.6–1.1 s batched run per turn.

Scratch files: `app\_mem_tmp.tsv` / `app\_mem_tmp.json` (last run), plus the
state file above.

### Known limitation (stale echoes on functional predicates)

If one turn's Slot 1 output gives several values for the *same functional*
relation (e.g. `DRIVES` Tesla then the stale Honda), the host keeps the first
and prints the rest as `graph CONFLICT`. Neither first-nor-last ordering is
consistently correct: the model sometimes lists the current value first
(first-wins correct), sometimes the stale one first (first-wins keeps the
stale value). Set-valued predicates are unaffected — multiple distinct
members coexist by design. The ambiguity is in the model's relation
ordering, not the host; it is surfaced rather than hidden, and closed
versions remain in the history for audit.

# parallel-inference

A small C++17 library + CLI that runs several generations of one GGUF model
through a **single decode step**: one model, one context, one batch. A shared
prefix is prefilled once, then copied to each branch; the engine builds one
`llama_batch`, decodes once, and routes logits back by batch index. This is the
reuse mechanism measured in the RecycleLM study (batching secondary work into a
primary decode step).

Two workflows ship with it:

- **compact** — answer the latest turn *and* produce a compact memory of the
  conversation, concurrently (foreground answer + auxiliary memory branch).
- **reasoning** — N independent reasoning streams over the same problem, with an
  optional agreement stop rule (stop once K streams emit the same final number).

## Build

Requires llama.cpp built as shared libraries (CUDA or CPU) and a GGUF model.

```bat
parallel\build.bat <pkg-dir> <build-dir> <llama.cpp-dir>
```

Example (this machine, RTX 4060 / CUDA):

```bat
parallel\build.bat S:\Agent\QSExplore\RecycleLM\parallel ^
                   S:\Agent\QSExplore\RecycleLM\build\parallel ^
                   S:\Agent\QSExplore\RecycleLM\vendor\llama.cpp
```

Outputs land in `<build-dir>`: `parallel_cli.exe`, `test_runtime.exe`, the two
examples, and the llama.cpp `*.dll` under `bin\` (put `bin\` on `PATH`).

## CLI

Conversation / problem files are plain text, one turn per line
(`user: ...` / `assistant: ...`). See `examples/conversation.txt` and
`examples/problem.txt`.

```bat
:: answer + background memory
parallel_cli compact --model models\Qwen2.5-3B-Instruct-Q4_K_M.gguf ^
                     --conversation parallel\examples\conversation.txt ^
                     --report run.json

:: N reasoning streams, stop when 3 agree
parallel_cli reasoning --model models\Qwen2.5-3B-Instruct-Q4_K_M.gguf ^
                       --conversation parallel\examples\problem.txt ^
                       --streams 8 --agree 3 --report run_reason.json
```

## Verified output (RTX 4060 8 GB, Qwen2.5-3B-Instruct Q4_K_M)

`compact` on `examples/conversation.txt` (a city/kickoff/budget correction):

```
=== answer ===
The latest city settled on is Porto.
[answer done]

=== memory (complete) ===
**DECISIONS AND LATEST CORRECTIONS:** The initial target city of Lisbon was
changed to Porto, with the kickoff date moved to March 21. The budget was
adjusted to 55k.
**EXACT FACTS / NUMBERS:** The current target city is Porto, the kickoff date
is March 21, and the budget is 55k.

[prefix 149 tok, prefill 56 ms, all done 1237 ms]
```

`reasoning` on a GSM8K problem (48 + 24 = 72), `--streams 8 --agree 3`:

```
=== candidates ===
  s1: 72 (agreement, 144 tok)
  s4: 72 (agreement, 145 tok)
  s7: 72 (agreement, 136 tok)
  s0..s6: (cancelled)
[prefix 45 tok, all done 3070 ms]
```

Three streams agreed on `72`; the remaining streams were cancelled by the
agreement rule.

## Tests

```bat
set PARALLEL_TEST_MODEL=models\Qwen2.5-3B-Instruct-Q4_K_M.gguf
test_runtime.exe
```

Covers: explicit capacity exhaustion (no silent truncation), relative token cap
vs. truncation, branch isolation, and cancel-one-branch. Set `PARALLEL_DEBUG=1`
to trace each tick.

## API sketch

```cpp
#include "parallel/runtime.hpp"
#include "parallel/workflows.hpp"

parallel::RuntimeConfig cfg;
cfg.model_path = "model.gguf";
cfg.max_context_tokens = 8192;
parallel::Runtime rt(cfg);

// custom run: shared prefix + independent branches
parallel::RunRequest req;
req.shared_prefix = rt.format_chat({ { "user", "..." } });
parallel::Branch a; a.id = "a"; a.suffix = "Task: ..."; a.max_new_tokens = 64;
req.branches = { a };
parallel::Run run = rt.submit(req);
while (auto e = run.next()) { /* Event::BranchToken / BranchDone / RunDone */ }
parallel::RunResult r = run.result();
```

## Scope

Hardware-specific results. Everything here was measured on one RTX 4060 (8 GB).
See `../LIMITATIONS.md`. `src/backends/llama_backend.hpp` is the extracted
engine from `../harness/engine/core.h` (same logic).

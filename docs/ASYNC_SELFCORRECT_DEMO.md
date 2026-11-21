# Async self-correction demo: the hallucination catcher

A biographer stream writes a biography. A fact-checker stream reads the
biographer's cache block, writes a critique into its own block, and the
biographer's **next forward pass sees that critique through a shared cache
view**. The result is a mid-stream correction without restarting the request.

This is the `async_llm` architecture (`github.com/dvmazur/async_llm`, built on
Mini-SGLang), run on the RTX 4060 under WSL2. It is not the C++ RecycleLM
engine; it is the Python framework the project is now evaluating for
asynchronous, cache-sharing agents.

## What was observed

Model: `Qwen/Qwen3.5-2B` (bf16), RTX 4060 8 GB, WSL2 Ubuntu 26.04.
Greedy-ish default sampling, 64 actor tokens, 100 thinker tokens, 90
continuation tokens.

### Architecture check (needle)

The thinker writes a sentence containing a secret word. The actor, whose cache
view includes the thinker's block, is asked for the secret word:

```
ACTOR:   He was born in 1882 ... graduated high school in 1880.
THINKER: The secret word is BANANA.
CONT:    BANANA
```

This is the load-bearing result: the actor's next forward pass reads tokens
written by the other stream through the shared view. No prompt copy, no
re-prefill of the thinker's text.

### The demo (3 planted contradictions)

Each case has the same structure: the actor writes the contradiction, the
thinker reads it and answers, then the actor continues. The control is the
same actor with no thinker block in its view.

| Case | Control continuation | With thinker |
|---|---|---|
| born 1882 / graduated 1880 | continues with career, ignores the contradiction | **“1880 is impossible because it is before 1882. The source contains an impossible sequence of events.”** |
| founded 1990 / retired 1985 | “her career spanned from 1985 to 1990” (repeats the impossible order) | **“1985 is the retirement date, and 1990 is the founding date, which is impossible. The source contains an impossible sequence of events.”** |
| prize 2001 / died 1999 | continues with legacy, ignores the contradiction | **“The source contains an impossible sequence of events… which is chronologically inconsistent.”** |

Full transcripts are in `results/async_correct_demo.txt`.

The correction instruction explicitly says “do not invent or change any dates;
if the source is internally inconsistent, say so and keep the facts as
stated.” Without that instruction the 2B “repairs” by swapping dates, which is
itself a hallucination. With it, all three cases flag the inconsistency and
keep the source dates.

### The 0.8B boundary

`Qwen/Qwen3.5-0.8B` detects the contradiction (“The date of his graduation from
high school is impossible”) but then **repeats the flawed sentence** in the
continuation. The cache view works; the model cannot act on it. The 2B can.
This is the same selection/competence boundary measured in the ensemble work,
now in a streaming setting.

## How it works

```
prompt_block  (source + task)
     │
     ├── actor_block     ← biographer writes 64 tokens
     │        │
     │        └── thinker_block   ← fact-checker reads actor_block, writes critique
     │                 │
     └── continuation_block       ← biographer's view = [prompt, actor, thinker, cont]
                                      so the next forward pass sees the critique
```

The framework's `CacheBlock` holds tokens and KV; a `CacheView` is an ordered
list of blocks a forward pass attends to. The engine batches concurrent
coroutines into the same forward pass and handles the position/RoPE layout.

This demo is **checkpointed**: the actor runs to a sentence boundary, the
thinker runs, then the actor continues. It is not a mid-token interrupt. The
property that matters is that the continuation's forward pass reads the
thinker's tokens from shared memory.

## Setup (WSL2, no root)

Native Windows is not supported by `async_llm` (Linux + NVIDIA only). The
working setup on this machine:

```bash
# 1. Framework
git clone https://github.com/dvmazur/async_llm ~/async_llm
cd ~/async_llm
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .

# 2. CUDA 13.0 toolchain from pip (no sudo)
uv pip install nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 \
               nvidia-cuda-crt==13.0.88 ninja
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
ln -sfn lib "$CUDA_HOME/lib64"
ln -sfn libcudart.so.13 "$CUDA_HOME/lib64/libcudart.so"
mkdir -p "$CUDA_HOME/lib64/stubs"
ln -sfn /usr/lib/wsl/lib/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so"

# 3. Ubuntu 26.04 glibc shim: add noexcept to CUDA's rsqrt/rsqrtf declarations
H="$CUDA_HOME/include/crt/math_functions.h"
sed -i 's/rsqrt(double x);/rsqrt(double x) noexcept;/' "$H"
sed -i 's/rsqrtf(float x);/rsqrtf(float x) noexcept;/' "$H"

# 4. Model
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-2B", local_dir="/home/max/models/Qwen3.5-2B")
PY
```

Run the demo from the repo:

```bash
cd /mnt/s/Agent/QSExplore/RecycleLM
DEMO_MODEL=/home/max/models/Qwen3.5-2B DEMO_PAGES=4096 \
  bash scripts/run_async_demo.sh
```

`DEMO_PAGES` bypasses the engine's `memory_ratio` allocator, which reserves KV
as `ratio * initial_free - model_memory`; on an 8 GB card that formula is too
conservative for a 4.3 GB model. 4096 pages is enough for this demo.

## Honest limits

- Three hand-built cases, one model, one GPU. No statistical claim.
- The 2B's binary verdict is unreliable: in the birth case it says “YES” while
  explaining that he graduated before he was born. The **explanation** is what
  the actor uses; a stricter thinker prompt or a larger model would clean this
  up.
- The thinker runs serially at the checkpoint in this demo. The framework can
  batch it with the actor's next steps; that concurrency is the next experiment.
- The 0.8B cannot repair, only detect. Do not present this as an 0.8B
  capability.
- Native Windows still uses the C++ RecycleLM engine; this Python path is
  WSL2-only.
- The `rsqrt` header edit and the `libcuda.so` symlink are local WSL
  workarounds, not upstream fixes; they will need revisiting on a framework
  update.

## Quantified concurrent benchmark (50 contradictions + 50 clean controls)

Model: Qwen3.5-2B bf16, RTX 4060 8 GB, WSL2. Engine: Mini-SGLang, 4096-token
KV page budget (or 65536 pages in the bench), greedy-ish default sampling.
Arms: `control` (actor only), `checkpointed` (actor 64 tokens -> thinker ->
continuation 90), `concurrent` (actor chunks; thinker runs as an asyncio task;
critique integrated at the next 12-token boundary). The actor runs its full
96-token budget (EOS forbidden); the continuation frame is chosen from the
thinker's verdict (NO -> "Correction:", YES -> "Continue").

### Results (n=100 per arm, 50 contradiction + 50 clean)

| arm | explicit correction | faithful (flag, no date swap) | date swap | thinker detection (contradictions) | thinker FP (clean) | median latency |
|---|---:|---:|---:|---:|---:|---:|
| control | 20% (10/50) | 20% (10/50) | 0% | -- | -- | 3859 ms |
| checkpointed | 68% (34/50) | 16% (8/50) | 68% (34/50) | 92% (46/50) | 0/50 | 6689 ms |
| **concurrent** | **68% (34/50)** | **44% (22/50)** | **36% (18/50)** | **100% (50/50)** | **0/50** | **3966 ms** |

- **The audit works:** explicit correction goes from 20% (actor alone) to 68%
  with a thinker in the loop (95% CI [54.2, 79.2]).
- **Overlap is nearly free:** the concurrent arm is **1.69x faster** than the
  serial checkpointed arm (3966 vs 6689 ms median) and only **+2.8%** over the
  no-audit control (3859 ms). Mid-stream integration: **100%**; serial tail:
  **0%**; actor/thinker token overlap: **1746 ms median**.
- **The engine really batches them:** **43.4 joint decode ticks of 93** (47%)
  had both an actor and a thinker decode request in the same forward pass.
- **Concurrent is also more faithful:** 44% vs 16% faithful, and 36% vs 68%
  date swaps. The serial continuation (90 tokens) more often invents a
  consistent-but-swapped timeline; the concurrent continuation (30 tokens)
  more often flags the inconsistency without rewriting the assignment.
- **No thinker false positives** on the 50 clean controls (0/50 both arms).
  The actor invented a new year on clean controls in 6% (concurrent) / 8%
  (checkpointed) / 2% (control) of cases, a residual cost of continuing.
- **No invented new years on contradictions** (0% all arms); the residual
  failure is *date reassignment*, not new-year fabrication.

### What this means

The fan-out claim now has a measured shape: a second stream can audit a
generating stream and change its continuation **without stopping it**, at
roughly the cost of the primary stream. The remaining quality problem is not
the architecture; it is that a 2B model asked to repair a contradiction often
"fixes" it by swapping the dates. A flag-only policy (state the inconsistency,
do not rewrite) or a stronger checker is the next lever.

### Reproduce

```bash
# from WSL2, with the async_llm venv installed (see above)
DEMO_MODEL=/home/max/models/Qwen3.5-2B \
  bash /mnt/s/Agent/QSExplore/RecycleLM/scripts/run_async_concurrent.sh --smoke
DEMO_MODEL=/home/max/models/Qwen3.5-2B \
  bash /mnt/s/Agent/QSExplore/RecycleLM/scripts/run_async_concurrent.sh
```

Artifacts: `results/async_concurrent_bench.csv` (300 rows),
`results/async_concurrent_summary.txt`, `results/async_concurrent_smoke.*`.

### Limits

- Synthetic templates (10 temporal patterns x 5 variants), one 2B model, one
  GPU, one seed. The absolute rates are model-specific; the latency and
  overlap results are hardware-specific.
- The faithful/date-swap checks are clause-based heuristics. They catch the
  obvious reassignment ("founded in 1962" for a 1968 founding) but can miss
  paraphrases; the raw transcript in the CSV is the evidence.
- The thinker's verdict was reliable on clean controls but the 2B still
  swaps dates in a third of the corrected contradictions. Do not present this
  as production-grade fact-checking.

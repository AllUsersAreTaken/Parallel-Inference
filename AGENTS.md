# AGENTS.md — RecycleLM

Shared context for all agents working in this repository. Read this before
making changes.

## What this project is

RecycleLM measures where spare GPU compute on a consumer card (RTX 4060, 8 GB)
can be reused: batching secondary work (consolidation, parallel reasoning,
memory briefs) into a primary decode step, and characterizing the hardware
limits.

## Hardware and scope (do not violate)

- One machine: RTX 4060 desktop, 8 GB, driver 617.42, WDDM, 115 W.
- CUDA 13.x plus VS 2022 Build Tools (MSVC + bundled CMake/Ninja).
- No NVML power telemetry on this card (`nvidia-smi` reports N/A); energy work
  needs a wall meter.
- **Everything is measured on this one machine. Results are hardware-specific
  and must never be extrapolated to other GPUs.**

## Layout

- `harness/` — C++ measurement tools + Python analysis (`phase1b.cpp`,
  `earlyexit.cpp`, `parallel_cost.cpp`, `trace_gen.cpp`, `int4_bench.py`,
  `analyze_*.py`, `simulate.py`).
- `scripts/` — build + data prep (`build_llama.bat`, `build_target.bat`,
  `build_tool.bat`, `prep_gsm8k.py`, `run_phase1a.ps1`, ...).
- `data/` — small inputs (GSM8K subset).
- `results/` — summaries, pareto CSVs, E0 cost curves, research log.
- `docs/` — the written research record: frozen gates, dated amendments, demos,
  theory memo. Start at `docs/README.md`. This is the "paper", kept out of the
  repository root on purpose.
- `models/` — fetched separately, not in this repo.

## Build and test

- Build llama.cpp CUDA: `scripts\build_llama.bat <out-dir> ON|OFF`
  (two configs: `build\cuda-graphs`, `build\cuda-nographs`).
- Build one harness tool:
  `scripts\build_tool.bat build\cuda-graphs harness\phase1b.cpp harness\phase1b.exe`
- Analysis scripts live in `harness/analyze_*.py`.
- "Tests" here are long CUDA builds and benchmark sweeps, not a fast unit-test
  suite. **Prefer the cheapest, smallest-scope experiment that can falsify the
  hypothesis first.** Flag anything long-running before launching it.

## Rules

- `docs/PREREGISTRATION*.md` hold frozen gates. Do not change them silently;
  any needed change is a dated amendment appended below the frozen section.
- Never hardcode or commit secrets or tokens.
- Respect `LICENSE` and `NOTICE`.
- Only real, observed runs count as evidence. Never fabricate metrics.

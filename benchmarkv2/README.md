# Benchmark v2

This folder contains the benchmark v2 runner and outputs. It does not overwrite
the original `full_benchmark_results.csv`.

## Overview

Benchmark v2 tests whether NPC responses improve when knowledge is embodied in
one or more game agents rather than given omnisciently. The main comparisons are
single NPC vs NPC party, perfect vs embodied knowledge for the single-NPC
baseline, deterministic vs LLM vs local SLM responses, 15x15 vs 40x40 worlds,
and party conditions with no competitor, one competitor, or all competitors.

The benchmark uses paired seeds across conditions, randomized separated
corner-style starts for NPCs and the player, and stable reserved object anchors
so target/distractor placement stays comparable for the same seed. NPC sight
range is fixed at 1, player sight range is fixed at 2, SLM region grounding is
off, memory decay/selective attention are off, and party knowledge mode is
independent with local deterministic NPC-to-NPC exchanges enabled. Exploration ticks are
calibrated by grid size and party size to produce a mix of target-seen and
target-unseen trials.

Each result row reports response correctness from the speaker's knowledge and
from the party's combined knowledge, target-availability timing, direct vs
knowledge coverage, tool calls, groundedness, relevance, committal behavior,
and response latency. LLM judging is intentionally off for v2; the original
benchmark can still be used for judge-only style metrics such as naturalness.

Dry-run the condition matrix:

```bash
python -m benchmarkv2.run_benchmark_v2 --list-conditions
```

Run the default matrix. By default this covers 15x15 and 40x40 worlds:

```bash
python -m benchmarkv2.run_benchmark_v2
```

Useful lower-cost runs:

```bash
python -m benchmarkv2.run_benchmark_v2 --num-trials 5 --response-modes deterministic
python -m benchmarkv2.run_benchmark_v2 --num-trials 10 --response-modes deterministic,slm
```

Each run writes to a timestamped directory under `benchmarkv2/results/`:

- `benchmark_v2_results.csv`
- `manifest.json`

Game logs still go to `logs/runs/`. Each manifest records both the raw
`config.py` defaults and the effective benchmark settings used by the runner.

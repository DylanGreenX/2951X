"""
Full 12-condition benchmark runner.

Runs CORE_CONDITIONS (6: 2 knowledge × 3 response) + EXTENDED_CONDITIONS
(6: 3 modalities × 2 response) on the current config, writes a single CSV.

Each condition runs 50 trials. Judge (gemini-2.5-pro) is on by default per
config.NPC_USE_LLM_JUDGE so every trial is dual-logged with regex + judge
metrics.

SLM conditions use the F13c endstate configuration documented in
docs/slm_evolution.md.
"""
from __future__ import annotations
import time
import pandas as pd

from experiment import (
    ExperimentRunner,
    CORE_CONDITIONS,
    EXTENDED_CONDITIONS,
)


NUM_TRIALS = 50
OUTPUT_CSV = "full_benchmark_results.csv"


def main() -> None:
    runner = ExperimentRunner()  # reuses one SLMClient across SLM conditions
    conditions = CORE_CONDITIONS + EXTENDED_CONDITIONS

    print(f"Full benchmark: {len(conditions)} conditions × {NUM_TRIALS} trials "
          f"= {len(conditions) * NUM_TRIALS} trials total")
    for c in conditions:
        print(f"  - {c.name}")
    print()

    all_results: list[dict] = []
    start = time.time()

    for i, condition in enumerate(conditions, start=1):
        cond_start = time.time()
        print(f"[{i}/{len(conditions)}] {condition.name}")
        results = runner.run_condition(condition, num_trials=NUM_TRIALS)
        all_results.extend(results)
        elapsed = time.time() - cond_start
        total_elapsed = time.time() - start
        print(f"    done in {elapsed:.0f}s (cumulative {total_elapsed:.0f}s)\n")

        # Incremental save — protects long-running benchmark from losing data
        # to a late crash in some later condition.
        pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)

    total = time.time() - start
    print(f"=== FULL BENCHMARK COMPLETE ===")
    print(f"Total trials: {len(df)}")
    print(f"Wall-clock: {total/60:.1f} min")
    print(f"Written to: {OUTPUT_CSV}")

    # Per-condition summary
    print("\nPer-condition outcome distribution:")
    summary = df.groupby("condition")["outcome_bucket"].value_counts().unstack(fill_value=0)
    print(summary.to_string())


if __name__ == "__main__":
    main()

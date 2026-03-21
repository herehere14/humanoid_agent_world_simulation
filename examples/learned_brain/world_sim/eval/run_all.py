#!/usr/bin/env python3
"""Run the full evaluation pipeline.

Usage:
    python -m learned_brain.world_sim.eval.run_all [--skip-scale-llm]

Steps:
    1. Scale benchmark (1000 agents, performance + variance) — $0 without LLM, ~$1 with
    2. Heart vs LLM comparison (8 agents, 10 moments) — ~$1
    3. Judge comparison (GPT-4o blind judging) — ~$6

Total estimated cost: ~$8
"""

from __future__ import annotations

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Run full heart vs LLM evaluation")
    parser.add_argument("--skip-scale-llm", action="store_true",
                        help="Skip LLM spot-check in scale benchmark (saves ~$1)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judging step (saves ~$6)")
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Scale benchmark
    print("\n" + "=" * 70)
    print("  STEP 1: Scale Benchmark (1000 agents)")
    print("=" * 70)
    from .scale_benchmark import run_scale_benchmark
    run_scale_benchmark(n_agents=1000, skip_llm=args.skip_scale_llm)

    # Step 2: Heart vs LLM comparison
    print("\n" + "=" * 70)
    print("  STEP 2: Heart vs LLM Comparison (8 agents × 10 moments)")
    print("=" * 70)
    from .eval_heart_vs_llm import run_evaluation
    run_evaluation(output_path="eval_samples.json")

    # Step 3: Judge comparison
    if not args.skip_judge:
        print("\n" + "=" * 70)
        print("  STEP 3: Blind Judging (GPT-4o)")
        print("=" * 70)
        from .judge_comparison import run_judging
        run_judging(samples_path="eval_samples.json", output_path="judge_results.json")

    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print(f"  ALL STEPS COMPLETE in {elapsed:.1f}s")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

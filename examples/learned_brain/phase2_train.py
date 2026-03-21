#!/usr/bin/env python3
"""Phase 2 Training — end-to-end pipeline for reward model + prompt policy.

Stage 1: Collect reward training data (50 rollouts, ~$10-15 API)
Stage 2: Train reward model (~30 sec CPU)
Stage 3: Train prompt policy via REINFORCE (~$5 API)

Usage:
    cd examples
    OPENAI_API_KEY=... python -m learned_brain.phase2_train
    OPENAI_API_KEY=... python -m learned_brain.phase2_train --stage 1  # just collect data
    OPENAI_API_KEY=... python -m learned_brain.phase2_train --stage 2  # just train reward model
    OPENAI_API_KEY=... python -m learned_brain.phase2_train --stage 3  # just train policy
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def stage1_collect(rollouts: int = 50, judge_model: str = "gpt-4o-mini"):
    """Stage 1: Collect reward model training data."""
    print("=" * 70)
    print("  STAGE 1: Collecting reward model training data")
    print("=" * 70)
    t0 = time.time()

    from learned_brain.reward_model import collect_reward_data
    collect_reward_data(
        n_rollouts=rollouts,
        n_candidates=4,
        judge_model=judge_model,
    )

    print(f"\nStage 1 complete in {time.time() - t0:.0f}s")


def stage2_train_reward():
    """Stage 2: Train the reward model."""
    print("\n" + "=" * 70)
    print("  STAGE 2: Training reward model")
    print("=" * 70)
    t0 = time.time()

    from learned_brain.reward_model import train_reward_model
    train_reward_model(epochs=150, lr=1e-3)

    print(f"\nStage 2 complete in {time.time() - t0:.0f}s")


def stage3_train_policy(episodes: int = 500):
    """Stage 3: Train the prompt policy via REINFORCE."""
    print("\n" + "=" * 70)
    print("  STAGE 3: Training prompt policy (REINFORCE)")
    print("=" * 70)
    t0 = time.time()

    from learned_brain.prompt_policy import train_prompt_policy
    train_prompt_policy(n_episodes=episodes)

    print(f"\nStage 3 complete in {time.time() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Training Pipeline")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage (1/2/3) or 0 for all")
    parser.add_argument("--rollouts", type=int, default=50,
                        help="Number of rollouts for data collection")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                        help="Judge model for reward data collection")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of REINFORCE episodes")
    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print(f"  PHASE 2 TRAINING PIPELINE")
    print(f"  Stage: {'all' if args.stage == 0 else args.stage}")
    print(f"{'#' * 70}")

    if args.stage == 0 or args.stage == 1:
        stage1_collect(args.rollouts, args.judge_model)

    if args.stage == 0 or args.stage == 2:
        stage2_train_reward()

    if args.stage == 0 or args.stage == 3:
        stage3_train_policy(args.episodes)

    print(f"\n{'#' * 70}")
    print("  ALL STAGES COMPLETE")
    print(f"  Run evaluation: python -m learned_brain.eval_phase2")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agent Improvement Mode demonstration: same task, system improves.

This script shows the standard adaptive prompt forest behavior where
the system optimizes routing and branch weights over repeated episodes.

Run:
    python -m examples.agent_improvement_demo
"""

from __future__ import annotations

from prompt_forest.modes.orchestrator import ModeOrchestrator


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main() -> None:
    print_section("Agent Improvement Mode: Adaptive Performance Over Episodes")

    orch = ModeOrchestrator(mode="agent_improvement")

    tasks = [
        "Write a risk assessment for deploying a new microservice to production.",
        "Summarize the key findings from this security audit report.",
        "Create a recovery plan for a database failover scenario.",
        "Analyze the trade-offs between consistency and availability in this system.",
        "Draft a post-mortem for the recent outage caused by a configuration change.",
    ]

    print("  Running 5 tasks through Agent Improvement Mode...\n")

    for i, task_text in enumerate(tasks):
        result = orch.run_task(task_text, task_type="general")
        print(f"  Episode {i+1}: \"{task_text[:60]}...\"")

        if "routing" in result:
            routing = result["routing"]
            if isinstance(routing, dict):
                activated = routing.get("activated_branches", [])
                print(f"    Activated: {activated[:4]}")
        print()

    # Show final branch weights
    state = orch.get_state()
    print("  Final branch weights:")
    for branch_info in state.get("branches", []):
        print(f"    {branch_info['name']:30s} w={branch_info['weight']:.4f}  "
              f"avg_r={branch_info['avg_reward']:.4f}")

    print_section("Agent Improvement Mode Complete")
    print("  The system adapts branch weights based on evaluation feedback.")
    print("  Higher-performing branches gain weight; poor ones are demoted.")


if __name__ == "__main__":
    main()

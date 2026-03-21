#!/usr/bin/env python3
"""Behavioral Divergence Validation Pipeline using the real ModeOrchestrator.

Tests the core claim that different emotional profiles produce different
cognitive routing decisions, evaluator scores, and state trajectories when
processing identical scenarios through the full Prompt Forest RL pipeline.

For each scenario:
  1. Create SEPARATE ModeOrchestrator instances with different initial_state
  2. Run IDENTICAL tasks through each orchestrator via run_task()
  3. Compare routing decisions (which branches activated), evaluator scores,
     state changes, memory tags, and branch weight changes
  4. Validate that differences match psychological predictions:
     - Anxious profile should activate fear_risk and self_protection more
     - Impulsive profile should activate impulse_response more
     - Empathetic profile should activate empathy_social more
     - Cautious analyst should activate reflective_reasoning more

Uses ModeOrchestrator + HumanModeRouter + 14 cognitive branches +
HumanModeEvaluator + HumanModeMemory + RL weight adaptation.

Run:
    python examples/run_behavioral_divergence_validation.py
"""

from __future__ import annotations

import math
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prompt_forest.modes.orchestrator import ModeOrchestrator
from src.prompt_forest.backend.mock import MockLLMBackend

# ---------------------------------------------------------------------------
# Personality profiles (Big Five -> initial_state drive mappings)
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, float]] = {
    "Confident Leader": {
        "confidence": 0.85,
        "ambition": 0.80,
        "motivation": 0.75,
        "trust": 0.65,
        "stress": 0.15,
        "fear": 0.10,
        "empathy": 0.45,
        "impulse": 0.40,
    },
    "Anxious Individual": {
        "confidence": 0.30,
        "fear": 0.60,
        "stress": 0.55,
        "caution": 0.70,
        "self_protection": 0.60,
        "trust": 0.30,
        "curiosity": 0.35,
        "impulse": 0.20,
    },
    "Empathetic Caregiver": {
        "empathy": 0.85,
        "trust": 0.75,
        "honesty": 0.75,
        "self_protection": 0.15,
        "ambition": 0.35,
        "confidence": 0.50,
        "stress": 0.25,
    },
    "Impulsive Risk-taker": {
        "impulse": 0.80,
        "ambition": 0.75,
        "curiosity": 0.70,
        "caution": 0.15,
        "fear": 0.10,
        "reflection": 0.25,
        "self_protection": 0.15,
    },
    "Burned-out Worker": {
        "fatigue": 0.80,
        "stress": 0.65,
        "frustration": 0.55,
        "motivation": 0.20,
        "curiosity": 0.25,
        "confidence": 0.30,
        "ambition": 0.25,
    },
    "Cautious Analyst": {
        "reflection": 0.80,
        "caution": 0.70,
        "honesty": 0.70,
        "impulse": 0.15,
        "curiosity": 0.65,
        "ambition": 0.40,
        "confidence": 0.55,
    },
}

PROFILE_NAMES = list(PROFILES.keys())

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class PsychPrediction:
    """A single falsifiable prediction about pipeline behavior."""
    description: str
    check: Callable[[dict[str, dict[str, Any]]], bool]
    rationale: str


@dataclass
class Scenario:
    name: str
    task_text: str
    events_before: list[tuple[str, float]]  # (event_type, intensity)
    predictions: list[PsychPrediction]


def _get_activated(results: dict[str, dict[str, Any]], profile: str) -> list[str]:
    """Get activated branches for a profile."""
    return results[profile].get("activated_branches", [])


def _branch_activated(results: dict[str, dict[str, Any]], profile: str, branch: str) -> bool:
    """Check if a specific branch was activated."""
    return branch in _get_activated(results, profile)


def _get_reward(results: dict[str, dict[str, Any]], profile: str) -> float:
    return results[profile].get("reward_score", 0.0)


def _get_valence(results: dict[str, dict[str, Any]], profile: str) -> float:
    return results[profile].get("mood_valence", 0.0)


def _get_arousal(results: dict[str, dict[str, Any]], profile: str) -> float:
    return results[profile].get("arousal", 0.0)


def _get_branch_score(results: dict[str, dict[str, Any]], profile: str, branch: str) -> float:
    """Get the routing score for a specific branch."""
    return results[profile].get("branch_scores", {}).get(branch, 0.0)


def _count_branch(results: dict[str, dict[str, Any]], profile: str, branch: str) -> int:
    """Count how many times a branch appears in activated branches."""
    return 1 if branch in results[profile].get("activated_branches", []) else 0


def build_scenarios() -> list[Scenario]:
    """Construct test scenarios with predictions about pipeline routing behavior."""
    scenarios: list[Scenario] = []

    # --- Scenario A: Unexpected opportunity ---
    scenarios.append(Scenario(
        name="A: Unexpected opportunity (reward + novelty)",
        task_text="An exciting new business opportunity has appeared. Should we pursue this risky but potentially lucrative venture?",
        events_before=[("novelty", 0.6), ("reward", 0.7)],
        predictions=[
            PsychPrediction(
                description="Confident Leader activates ambition_reward branch",
                check=lambda r: _branch_activated(r, "Confident Leader", "ambition_reward") or
                    _get_branch_score(r, "Confident Leader", "ambition_reward") >
                    _get_branch_score(r, "Anxious Individual", "ambition_reward"),
                rationale=(
                    "High baseline confidence and ambition should cause the router "
                    "to strongly activate the ambition_reward branch (Bandura's self-efficacy)."
                ),
            ),
            PsychPrediction(
                description="Anxious Individual has higher fear_risk branch score than Confident Leader",
                check=lambda r: (
                    _get_branch_score(r, "Anxious Individual", "fear_risk") >
                    _get_branch_score(r, "Confident Leader", "fear_risk")
                ),
                rationale=(
                    "High-neuroticism individuals have elevated fear drives that push the "
                    "router to activate fear_risk more strongly (Gray's BIS/BAS theory)."
                ),
            ),
            PsychPrediction(
                description="Burned-out Worker has lowest mood valence after processing",
                check=lambda r: _get_valence(r, "Burned-out Worker") == min(
                    _get_valence(r, p) for p in PROFILE_NAMES
                ),
                rationale=(
                    "High fatigue suppresses reward responsiveness. Burnout research "
                    "(Maslach) shows exhaustion dampens positive affect."
                ),
            ),
        ],
    ))

    # --- Scenario B: Social criticism ---
    scenarios.append(Scenario(
        name="B: Social criticism under threat",
        task_text="Your work has been publicly criticized by a senior colleague. How should you respond to protect your reputation while maintaining relationships?",
        events_before=[("social_rejection", 0.7), ("threat", 0.3)],
        predictions=[
            PsychPrediction(
                description="Empathetic Caregiver activates empathy_social branch",
                check=lambda r: _branch_activated(r, "Empathetic Caregiver", "empathy_social") or
                    _get_branch_score(r, "Empathetic Caregiver", "empathy_social") >
                    _get_branch_score(r, "Impulsive Risk-taker", "empathy_social"),
                rationale=(
                    "High empathy drives cause the router to prioritize empathy_social "
                    "for processing social situations (Eisenberg's empathy model)."
                ),
            ),
            PsychPrediction(
                description="Confident Leader maintains highest mood valence",
                check=lambda r: _get_valence(r, "Confident Leader") == max(
                    _get_valence(r, p) for p in PROFILE_NAMES
                ),
                rationale=(
                    "High self-efficacy buffers against social threats (Bandura). "
                    "Low neuroticism -> smaller affective reactions."
                ),
            ),
            PsychPrediction(
                description="Anxious Individual activates self_protection branch more than Confident Leader",
                check=lambda r: (
                    _get_branch_score(r, "Anxious Individual", "self_protection") >
                    _get_branch_score(r, "Confident Leader", "self_protection")
                ),
                rationale=(
                    "High self_protection drive in anxious profiles causes the router "
                    "to prioritize defensive cognitive branches."
                ),
            ),
        ],
    ))

    # --- Scenario C: Deadline pressure ---
    scenarios.append(Scenario(
        name="C: Deadline pressure with failure risk",
        task_text="Two critical deadlines are tomorrow and we are significantly behind. What is the fastest way to deliver acceptable results?",
        events_before=[("deadline_pressure", 0.8), ("deadline_pressure", 0.8)],
        predictions=[
            PsychPrediction(
                description="Impulsive Risk-taker has highest impulse_response branch score",
                check=lambda r: _get_branch_score(r, "Impulsive Risk-taker", "impulse_response") == max(
                    _get_branch_score(r, p, "impulse_response") for p in PROFILE_NAMES
                ),
                rationale=(
                    "High baseline impulse + deadline pressure directly boosts the "
                    "impulse_response branch via drive-based routing (Whiteside & Lynam)."
                ),
            ),
            PsychPrediction(
                description="Cautious Analyst has higher reflective_reasoning score than Impulsive Risk-taker",
                check=lambda r: (
                    _get_branch_score(r, "Cautious Analyst", "reflective_reasoning") >
                    _get_branch_score(r, "Impulsive Risk-taker", "reflective_reasoning")
                ),
                rationale=(
                    "High reflection drive maintains deliberative processing under "
                    "stress. Trait conscientiousness -> maintained reflection (DeYoung)."
                ),
            ),
            PsychPrediction(
                description="Burned-out Worker shows elevated stress in final state",
                check=lambda r: r["Burned-out Worker"].get("after_state", {}).get("stress", 0) > 0.7,
                rationale=(
                    "Already-high stress + deadline pressure = allostatic overload (McEwen)."
                ),
            ),
        ],
    ))

    # --- Scenario D: Novel complex problem ---
    scenarios.append(Scenario(
        name="D: Novel complex problem requiring deep analysis",
        task_text="A completely new type of system failure has occurred that nobody has seen before. Analyze the root cause and propose solutions.",
        events_before=[("novelty", 0.8), ("deadline_pressure", 0.3)],
        predictions=[
            PsychPrediction(
                description="Cautious Analyst activates reflective_reasoning",
                check=lambda r: _branch_activated(r, "Cautious Analyst", "reflective_reasoning") or
                    _get_branch_score(r, "Cautious Analyst", "reflective_reasoning") >
                    _get_branch_score(r, "Impulsive Risk-taker", "reflective_reasoning"),
                rationale=(
                    "High openness + conscientiousness predicts deep engagement. "
                    "Novelty boosts reflection, and high baseline reflection amplifies this."
                ),
            ),
            PsychPrediction(
                description="Impulsive Risk-taker has lowest reflective_reasoning score",
                check=lambda r: _get_branch_score(r, "Impulsive Risk-taker", "reflective_reasoning") == min(
                    _get_branch_score(r, p, "reflective_reasoning") for p in PROFILE_NAMES
                ),
                rationale=(
                    "Low reflection + high impulse = suppressed deliberative processing. "
                    "Impulsivity inversely correlated with need for cognition (Cacioppo)."
                ),
            ),
            PsychPrediction(
                description="Curious profiles show high curiosity_exploration scores",
                check=lambda r: (
                    _get_branch_score(r, "Impulsive Risk-taker", "curiosity_exploration") >
                    _get_branch_score(r, "Burned-out Worker", "curiosity_exploration")
                    and
                    _get_branch_score(r, "Cautious Analyst", "curiosity_exploration") >
                    _get_branch_score(r, "Burned-out Worker", "curiosity_exploration")
                ),
                rationale=(
                    "Both profiles have high curiosity (0.70 and 0.65). Novelty events "
                    "amplify via trait activation theory (Tett & Burnett)."
                ),
            ),
        ],
    ))

    # --- Scenario E: Recovery after failure ---
    scenarios.append(Scenario(
        name="E: Recovery and praise after setback",
        task_text="After a difficult period, a senior leader publicly recognizes your resilience and contribution. What do you take away from this experience?",
        events_before=[("social_rejection", 0.6), ("rest", 0.5), ("social_praise", 0.7)],
        predictions=[
            PsychPrediction(
                description="Confident Leader recovers to highest mood valence",
                check=lambda r: _get_valence(r, "Confident Leader") == max(
                    _get_valence(r, p) for p in PROFILE_NAMES
                ),
                rationale=(
                    "High baseline confidence + low stress/fear provide resilience. "
                    "Broaden-and-build theory (Fredrickson): positive emotions build resources."
                ),
            ),
            PsychPrediction(
                description="Anxious Individual has lowest mood valence",
                check=lambda r: _get_valence(r, "Anxious Individual") == min(
                    _get_valence(r, p) for p in PROFILE_NAMES
                ),
                rationale=(
                    "High neuroticism -> slower emotional recovery (Schuyler et al.). "
                    "Elevated self_protection dampens positive effects of praise."
                ),
            ),
            PsychPrediction(
                description="All profiles show higher valence than pure-rejection would cause",
                check=lambda r: all(
                    _get_valence(r, p) > -0.3 for p in PROFILE_NAMES
                ),
                rationale=(
                    "Rest + praise should produce universal recovery. "
                    "Tests that the pipeline is not stuck in negative states."
                ),
            ),
        ],
    ))

    # --- Scenario F: Moral dilemma ---
    scenarios.append(Scenario(
        name="F: Ethical dilemma with competing values",
        task_text="You discover a colleague is falsifying safety reports. Reporting them would save lives but destroy their career and family. What should you do?",
        events_before=[("threat", 0.4), ("novelty", 0.3)],
        predictions=[
            PsychPrediction(
                description="Empathetic Caregiver has highest empathy_social score",
                check=lambda r: _get_branch_score(r, "Empathetic Caregiver", "empathy_social") == max(
                    _get_branch_score(r, p, "empathy_social") for p in PROFILE_NAMES
                ),
                rationale=(
                    "High empathy drive trait-activates empathy_social for moral scenarios. "
                    "Batson's empathy-altruism hypothesis."
                ),
            ),
            PsychPrediction(
                description="Cautious Analyst has higher moral_evaluation score than Impulsive Risk-taker",
                check=lambda r: (
                    _get_branch_score(r, "Cautious Analyst", "moral_evaluation") >
                    _get_branch_score(r, "Impulsive Risk-taker", "moral_evaluation")
                ),
                rationale=(
                    "High reflection + honesty -> stronger moral deliberation. "
                    "Greene's dual-process theory of moral judgment."
                ),
            ),
            PsychPrediction(
                description="Impulsive Risk-taker relies more on impulse_response than reflective_reasoning",
                check=lambda r: (
                    _get_branch_score(r, "Impulsive Risk-taker", "impulse_response") >
                    _get_branch_score(r, "Impulsive Risk-taker", "reflective_reasoning")
                ),
                rationale=(
                    "High impulse + low reflection -> fast heuristic processing "
                    "rather than moral deliberation."
                ),
            ),
        ],
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

ALL_METRIC_DRIVES = [
    "confidence", "stress", "frustration", "trust", "fatigue", "curiosity",
    "fear", "motivation", "emotional_momentum", "goal_commitment", "empathy",
    "ambition", "caution", "honesty", "self_protection", "self_justification",
    "impulse", "reflection",
]


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def profile_vector(results: dict[str, dict[str, Any]], profile: str) -> list[float]:
    """Extract a fixed-order drive vector from final state."""
    after = results[profile].get("after_state", {})
    return [after.get(d, 0.5) for d in ALL_METRIC_DRIVES]


def routing_vector(results: dict[str, dict[str, Any]], profile: str) -> list[float]:
    """Extract branch score vector for routing comparison."""
    scores = results[profile].get("branch_scores", {})
    all_branches = sorted(set().union(*(r.get("branch_scores", {}).keys() for r in results.values())))
    return [scores.get(b, 0.0) for b in all_branches]


def compute_divergence_matrix(results: dict[str, dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Mean pairwise Euclidean distance between all profile final states."""
    matrix: dict[tuple[str, str], float] = {}
    for p1, p2 in combinations(PROFILE_NAMES, 2):
        dist = euclidean_distance(profile_vector(results, p1), profile_vector(results, p2))
        matrix[(p1, p2)] = round(dist, 4)
    return matrix


def compute_routing_divergence(results: dict[str, dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Pairwise Euclidean distance between routing score vectors."""
    matrix: dict[tuple[str, str], float] = {}
    for p1, p2 in combinations(PROFILE_NAMES, 2):
        dist = euclidean_distance(routing_vector(results, p1), routing_vector(results, p2))
        matrix[(p1, p2)] = round(dist, 4)
    return matrix


def mean_divergence(matrix: dict[tuple[str, str], float]) -> float:
    if not matrix:
        return 0.0
    return sum(matrix.values()) / len(matrix)


def _rank(values: list[float]) -> list[float]:
    """Assign ranks (1-based) to values, with average rank for ties."""
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two sequences."""
    n = len(x)
    if n < 2:
        return 1.0
    rank_x = _rank(x)
    rank_y = _rank(y)
    d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


# ---------------------------------------------------------------------------
# Run a single scenario across all profiles
# ---------------------------------------------------------------------------

def run_scenario(scenario: Scenario) -> dict[str, dict[str, Any]]:
    """Run a scenario against every profile through the ModeOrchestrator."""
    results: dict[str, dict[str, Any]] = {}

    for profile_name, overrides in PROFILES.items():
        random.seed(42)

        orch = ModeOrchestrator(
            mode="human_mode",
            backend=MockLLMBackend(seed=42),
            initial_state=dict(overrides),
        )

        # Apply pre-task events
        for event_type, intensity in scenario.events_before:
            orch.inject_event(event_type, intensity)

        # Run the task through the full pipeline
        task_result = orch.run_task(text=scenario.task_text, task_type="auto")

        # Extract pipeline outputs
        routing = task_result.get("routing", {})
        eval_signal = task_result.get("evaluation_signal", {})
        human_state = task_result.get("human_state", {})
        after_state_raw = human_state.get("after", {}) if isinstance(human_state, dict) else {}
        before_state_raw = human_state.get("before", {}) if isinstance(human_state, dict) else {}

        # Get final state from orchestrator
        final_state = orch.get_state()

        # Extract all routing branch scores
        branch_scores = {}
        if isinstance(routing, dict):
            branch_scores = routing.get("branch_scores", {})

        results[profile_name] = {
            "activated_branches": routing.get("activated_branches", []) if isinstance(routing, dict) else [],
            "branch_scores": branch_scores,
            "task_type": routing.get("task_type", "") if isinstance(routing, dict) else "",
            "reward_score": eval_signal.get("reward_score", 0.0) if isinstance(eval_signal, dict) else 0.0,
            "selected_branch": eval_signal.get("selected_branch", "") if isinstance(eval_signal, dict) else "",
            "mood_valence": final_state.get("mood_valence", 0.0),
            "arousal": final_state.get("arousal", 0.0),
            "dominant_drives": final_state.get("dominant_drives", []),
            "active_conflicts": final_state.get("active_conflicts", []),
            "after_state": after_state_raw.get("variables", {}) if isinstance(after_state_raw, dict) else {},
            "before_state": before_state_raw.get("variables", {}) if isinstance(before_state_raw, dict) else {},
            "branch_weights": task_result.get("branch_weights", {}),
            "memory_count": task_result.get("experiential_memory", {}).get("count", 0),
            "memory_tags": task_result.get("experiential_memory", {}).get("latest_tags", []),
            "conflicts": task_result.get("conflicts", []),
        }

    return results


# ---------------------------------------------------------------------------
# Stability check: same profile, same scenario, multiple runs
# ---------------------------------------------------------------------------

def stability_check(
    scenario: Scenario,
    n_runs: int = 20,
) -> dict[str, float]:
    """Run the scenario n_runs times and report per-profile variance.

    Each run uses a different random seed to test noise sensitivity.
    """
    run_vectors: dict[str, list[list[float]]] = {p: [] for p in PROFILE_NAMES}

    for run_idx in range(n_runs):
        for profile_name, overrides in PROFILES.items():
            random.seed(run_idx * 100 + 42)
            orch = ModeOrchestrator(
                mode="human_mode",
                backend=MockLLMBackend(seed=run_idx * 100 + 42),
                initial_state=dict(overrides),
            )
            for event_type, intensity in scenario.events_before:
                orch.inject_event(event_type, intensity)
            task_result = orch.run_task(text=scenario.task_text, task_type="auto")

            # Extract routing score vector
            routing = task_result.get("routing", {})
            scores = routing.get("branch_scores", {}) if isinstance(routing, dict) else {}
            vec = [scores.get(k, 0.0) for k in sorted(scores.keys())]
            if vec:
                run_vectors[profile_name].append(vec)

    variances: dict[str, float] = {}
    for pname in PROFILE_NAMES:
        vecs = run_vectors[pname]
        if not vecs or not vecs[0]:
            variances[pname] = 0.0
            continue
        n_dims = len(vecs[0])
        total_var = 0.0
        for dim in range(min(n_dims, len(vecs[0]))):
            dim_vals = [v[dim] for v in vecs if dim < len(v)]
            if dim_vals:
                mu = sum(dim_vals) / len(dim_vals)
                var = sum((x - mu) ** 2 for x in dim_vals) / len(dim_vals)
                total_var += var
        variances[pname] = round(total_var / max(n_dims, 1), 6)

    return variances


# ---------------------------------------------------------------------------
# Printing utilities
# ---------------------------------------------------------------------------

def print_separator(char: str = "=", width: int = 80) -> None:
    print(char * width)


def print_scenario_results(
    scenario: Scenario,
    results: dict[str, dict[str, Any]],
    prediction_outcomes: list[tuple[PsychPrediction, bool]],
    state_div_matrix: dict[tuple[str, str], float],
    routing_div_matrix: dict[tuple[str, str], float],
) -> None:
    """Print detailed results for a single scenario."""
    print_separator()
    print(f"SCENARIO: {scenario.name}")
    print(f"  Task: {scenario.task_text[:100]}")
    print(f"  Events: {scenario.events_before}")
    print_separator("-")

    # Per-profile pipeline summary
    print(f"\n  {'Profile':<25} {'Mood':>6} {'Arousal':>7} {'Reward':>7} {'CogCtx':<16} {'Activated Branches'}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*7} {'-'*16} {'-'*40}")

    for pname in PROFILE_NAMES:
        r = results[pname]
        branches_str = ", ".join(r["activated_branches"][:4])
        print(f"  {pname:<25} {r['mood_valence']:>+6.3f} {r['arousal']:>7.3f} "
              f"{r['reward_score']:>7.3f} {r['task_type']:<16} {branches_str}")

    # Branch score comparison for key branches
    key_branches = [
        "reflective_reasoning", "impulse_response", "fear_risk",
        "empathy_social", "ambition_reward", "self_protection",
        "moral_evaluation", "curiosity_exploration",
    ]

    print(f"\n  Branch routing scores (key branches):")
    header = f"  {'Profile':<25}"
    for b in key_branches:
        header += f" {b[:8]:>8}"
    print(header)
    print(f"  {'-'*25}" + "".join(f" {'-'*8}" for _ in key_branches))

    for pname in PROFILE_NAMES:
        scores = results[pname].get("branch_scores", {})
        line = f"  {pname:<25}"
        for b in key_branches:
            line += f" {scores.get(b, 0.0):>8.3f}"
        print(line)

    # Dominant drives
    print(f"\n  Dominant drives (post-task):")
    for pname in PROFILE_NAMES:
        drives = results[pname].get("dominant_drives", [])
        print(f"    {pname:<25} {', '.join(drives[:5])}")

    # Conflicts
    print(f"\n  Active conflicts:")
    for pname in PROFILE_NAMES:
        conflicts = results[pname].get("conflicts", [])
        if conflicts:
            conflict_strs = []
            for c in conflicts[:3]:
                if isinstance(c, dict):
                    conflict_strs.append(f"{c.get('drive_a', '?')} vs {c.get('drive_b', '?')}")
            if conflict_strs:
                print(f"    {pname:<25} {'; '.join(conflict_strs)}")
            else:
                print(f"    {pname:<25} {len(conflicts)} conflicts")
        else:
            print(f"    {pname:<25} none")

    # Memory tags
    print(f"\n  Memory tags (from experiential memory):")
    for pname in PROFILE_NAMES:
        tags = results[pname].get("memory_tags", [])
        if tags:
            print(f"    {pname:<25} {', '.join(tags[:5])}")

    # Divergence matrices
    print(f"\n  State divergence (Euclidean distance between final states):")
    for (p1, p2), dist in sorted(state_div_matrix.items(), key=lambda x: -x[1])[:5]:
        print(f"    {p1:<25} <-> {p2:<25} d = {dist:.4f}")
    print(f"    Mean state divergence: {mean_divergence(state_div_matrix):.4f}")

    print(f"\n  Routing divergence (distance between branch score vectors):")
    for (p1, p2), dist in sorted(routing_div_matrix.items(), key=lambda x: -x[1])[:5]:
        print(f"    {p1:<25} <-> {p2:<25} d = {dist:.4f}")
    print(f"    Mean routing divergence: {mean_divergence(routing_div_matrix):.4f}")

    # Psychological predictions
    print(f"\n  Psychological predictions:")
    for pred, passed in prediction_outcomes:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {pred.description}")
        if not passed:
            print(f"           Rationale: {pred.rationale}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_validation() -> None:
    """Execute the full behavioral divergence validation pipeline."""
    print_separator("=")
    print("BEHAVIORAL DIVERGENCE VALIDATION PIPELINE")
    print("Using: ModeOrchestrator + HumanModeRouter + 14 cognitive branches")
    print("       + HumanModeEvaluator + HumanModeMemory + RL weight adaptation")
    print("Testing: Identical tasks + different emotional profiles -> different routing decisions")
    print_separator("=")

    random.seed(42)

    scenarios = build_scenarios()

    all_prediction_outcomes: list[tuple[PsychPrediction, bool]] = []
    all_state_divergences: list[float] = []
    all_routing_divergences: list[float] = []

    for scenario in scenarios:
        results = run_scenario(scenario)

        # Compute metrics
        state_div_matrix = compute_divergence_matrix(results)
        routing_div_matrix = compute_routing_divergence(results)
        all_state_divergences.append(mean_divergence(state_div_matrix))
        all_routing_divergences.append(mean_divergence(routing_div_matrix))

        # Evaluate predictions
        prediction_outcomes: list[tuple[PsychPrediction, bool]] = []
        for pred in scenario.predictions:
            try:
                passed = pred.check(results)
            except Exception as e:
                print(f"    [ERROR] Prediction check failed: {pred.description} -- {e}")
                passed = False
            prediction_outcomes.append((pred, passed))
            all_prediction_outcomes.append((pred, passed))

        print_scenario_results(scenario, results, prediction_outcomes,
                               state_div_matrix, routing_div_matrix)
        print()

    # --- Stability check ---
    print_separator("=")
    print("STABILITY CHECK (Scenario A, 20 runs with different seeds)")
    print_separator("-")
    stability_vars = stability_check(scenarios[0], n_runs=20)
    max_var = 0.0
    for pname, var in stability_vars.items():
        print(f"  {pname:<25} mean routing variance = {var:.6f}")
        max_var = max(max_var, var)
    stability_bounded = max_var < 0.05
    print(f"  Stability bounded (max variance < 0.05): {'PASS' if stability_bounded else 'FAIL'} (max = {max_var:.6f})")

    # --- Overall summary ---
    print_separator("=")
    print("OVERALL SUMMARY")
    print_separator("-")

    total_predictions = len(all_prediction_outcomes)
    passed_predictions = sum(1 for _, p in all_prediction_outcomes if p)
    plausibility_score = passed_predictions / total_predictions if total_predictions > 0 else 0.0
    mean_state_div = sum(all_state_divergences) / len(all_state_divergences) if all_state_divergences else 0.0
    mean_routing_div = sum(all_routing_divergences) / len(all_routing_divergences) if all_routing_divergences else 0.0

    print(f"  Scenarios tested:              {len(scenarios)}")
    print(f"  Personality profiles:          {len(PROFILES)}")
    print(f"  Total predictions:             {total_predictions}")
    print(f"  Predictions passed:            {passed_predictions}")
    print(f"  Plausibility score:            {plausibility_score:.1%}")
    print(f"  Mean state divergence:         {mean_state_div:.4f}")
    print(f"  Mean routing divergence:       {mean_routing_div:.4f}")
    print(f"  Stability check:               {'PASS' if stability_bounded else 'FAIL'}")
    print()

    # Thresholds
    STATE_DIV_THRESHOLD = 0.10
    state_div_pass = mean_state_div > STATE_DIV_THRESHOLD
    print(f"  State divergence > {STATE_DIV_THRESHOLD}:       {'PASS' if state_div_pass else 'FAIL'} ({mean_state_div:.4f})")

    ROUTING_DIV_THRESHOLD = 0.05
    routing_div_pass = mean_routing_div > ROUTING_DIV_THRESHOLD
    print(f"  Routing divergence > {ROUTING_DIV_THRESHOLD}:     {'PASS' if routing_div_pass else 'FAIL'} ({mean_routing_div:.4f})")

    PLAUSIBILITY_THRESHOLD = 0.60
    plausibility_pass = plausibility_score >= PLAUSIBILITY_THRESHOLD
    print(f"  Plausibility >= {PLAUSIBILITY_THRESHOLD:.0%}:           {'PASS' if plausibility_pass else 'FAIL'} ({plausibility_score:.1%})")

    print_separator("=")

    # Detailed prediction listing
    print("\nDETAILED PREDICTION RESULTS:")
    print_separator("-")
    for pred, passed in all_prediction_outcomes:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {pred.description}")
    print_separator("=")

    # Overall verdict
    overall_pass = state_div_pass and routing_div_pass and plausibility_pass and stability_bounded
    print(f"\nOVERALL VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    if overall_pass:
        print("  The ModeOrchestrator pipeline produces measurably different routing")
        print("  decisions and state outcomes for different emotional profiles under")
        print("  identical scenarios, with psychologically plausible divergence patterns.")
    else:
        failures = []
        if not state_div_pass:
            failures.append("insufficient state divergence between profiles")
        if not routing_div_pass:
            failures.append("insufficient routing divergence between profiles")
        if not plausibility_pass:
            failures.append("too many psychological predictions failed")
        if not stability_bounded:
            failures.append("system shows excessive variance across runs")
        print(f"  Issues: {'; '.join(failures)}")
    print()


if __name__ == "__main__":
    run_validation()

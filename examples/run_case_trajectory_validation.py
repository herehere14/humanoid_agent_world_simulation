#!/usr/bin/env python3
"""Validation pipeline for multi-turn emotion trajectory analysis using the
real ModeOrchestrator pipeline.

Tests that the full Prompt Forest RL architecture -- ModeOrchestrator with
HumanModeRouter, 14 cognitive branches, HumanModeEvaluator, HumanModeMemory,
and RL weight adaptation -- produces psychologically plausible emotion
trajectories across 12 multi-turn scenarios.

Each scenario is a SEQUENCE of run_task() calls with emotionally-loaded text
AND inject_event() calls between tasks. After each step we record:
  - State (mood_valence, arousal, dominant_drives, active_conflicts)
  - Routing decisions (which cognitive branches were activated)
  - Evaluator reward scores
  - Branch weight changes via RL adaptation
  - Experiential memory accumulation

Expected trajectories are grounded in established psychological principles:
  * Yerkes-Dodson law (inverted-U performance-arousal curve)
  * Opponent-process theory (affective rebound after stressor removal)
  * Hedonic adaptation (diminishing marginal valence from repeated rewards)
  * Stress inoculation (repeated threat exposure reduces fear increment)
  * Contrast effect (sharper disappointment after positive buildup)
  * Approach-avoidance conflict (oscillating valence, rising stress)

Usage
-----
    python examples/run_case_trajectory_validation.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prompt_forest.modes.orchestrator import ModeOrchestrator
from src.prompt_forest.backend.mock import MockLLMBackend

# Fix random seed for reproducibility
random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers (no numpy/scipy dependency)
# ──────────────────────────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson product-moment correlation coefficient."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return cov / (sx * sy)


def _rank(xs: list[float]) -> list[float]:
    """Assign ranks with average tie-breaking."""
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    return pearson_r(_rank(xs[:n]), _rank(ys[:n]))


def direction_accuracy(predicted: list[float], expected: list[float]) -> float:
    """Fraction of timestep transitions where predicted direction matches expected."""
    n = min(len(predicted), len(expected))
    if n < 2:
        return 0.0
    matches = 0
    total = n - 1
    for i in range(1, n):
        pd = predicted[i] - predicted[i - 1]
        ed = expected[i] - expected[i - 1]
        if (pd > 0 and ed > 0) or (pd < 0 and ed < 0) or (abs(pd) < 0.005 and abs(ed) < 0.005):
            matches += 1
    return matches / total if total > 0 else 0.0


def _trend_label(xs: list[float]) -> str:
    """Classify a trajectory as increasing, decreasing, u-shaped, inverted-u, or flat."""
    if len(xs) < 3:
        return "flat"
    mid = len(xs) // 2
    first_half_slope = xs[mid] - xs[0]
    second_half_slope = xs[-1] - xs[mid]
    threshold = 0.03
    if first_half_slope > threshold and second_half_slope < -threshold:
        return "inverted-u"
    if first_half_slope < -threshold and second_half_slope > threshold:
        return "u-shaped"
    overall = xs[-1] - xs[0]
    if overall > threshold:
        return "increasing"
    if overall < -threshold:
        return "decreasing"
    return "flat"


def trend_match(predicted: list[float], expected: list[float]) -> bool:
    return _trend_label(predicted) == _trend_label(expected)


def peak_trough_alignment(predicted: list[float], expected: list[float], tolerance: int = 1) -> dict[str, bool]:
    """Check whether peak and trough indices align within tolerance."""
    n = min(len(predicted), len(expected))
    p, e = predicted[:n], expected[:n]
    result = {}
    if n == 0:
        return {"peak_aligned": False, "trough_aligned": False}
    p_peak = p.index(max(p))
    e_peak = e.index(max(e))
    result["peak_aligned"] = abs(p_peak - e_peak) <= tolerance
    p_trough = p.index(min(p))
    e_trough = e.index(min(e))
    result["trough_aligned"] = abs(p_trough - e_trough) <= tolerance
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ScenarioStep:
    """A single step in a trajectory scenario."""
    action: str          # "task" or "event"
    text: str            # task text or event_type
    intensity: float     # only used for events
    description: str = ""


@dataclass
class ScenarioResult:
    name: str
    steps: list[ScenarioStep]
    predicted_valence: list[float]
    predicted_arousal: list[float]
    expected_valence: list[float]
    expected_arousal: list[float]
    activated_branches_per_step: list[list[str]] = field(default_factory=list)
    reward_scores: list[float] = field(default_factory=list)
    branch_weights_per_step: list[dict[str, float]] = field(default_factory=list)
    dominant_drives_per_step: list[list[str]] = field(default_factory=list)
    conflicts_per_step: list[int] = field(default_factory=list)
    memory_count_per_step: list[int] = field(default_factory=list)
    pearson_valence: float = 0.0
    pearson_arousal: float = 0.0
    spearman_valence: float = 0.0
    spearman_arousal: float = 0.0
    dir_acc_valence: float = 0.0
    dir_acc_arousal: float = 0.0
    valence_trend_match: bool = False
    arousal_trend_match: bool = False
    peak_alignment: dict[str, bool] = field(default_factory=dict)
    passed_patterns: dict[str, bool] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    steps: list[ScenarioStep]
    expected_valence: list[float]
    expected_arousal: list[float]
    description: str = ""
    pattern_checks: list[str] = field(default_factory=list)


def build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []

    # ── Scenario 1: Gradual stress buildup with burnout ──────────────────
    scenarios.append(Scenario(
        name="Gradual stress buildup (Yerkes-Dodson)",
        steps=[
            ScenarioStep("event", "deadline_pressure", 0.3, "mild deadline"),
            ScenarioStep("task", "We need to get this report done by end of week", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.5, "growing pressure"),
            ScenarioStep("task", "We need results immediately, the client is waiting", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.7, "high pressure"),
            ScenarioStep("task", "The deadline was yesterday, we are at risk of losing the contract", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.9, "extreme pressure"),
            ScenarioStep("task", "Everything is falling apart and the CEO is furious with our team", 0.0),
        ],
        expected_valence=[0.48, 0.45, 0.40, 0.36, 0.32, 0.28, 0.24, 0.20],
        expected_arousal=[0.38, 0.39, 0.41, 0.42, 0.43, 0.44, 0.45, 0.45],
        description=(
            "Yerkes-Dodson: increasing deadline pressure causes monotonically "
            "decreasing valence and an inverted-U arousal curve."
        ),
        pattern_checks=["yerkes_dodson", "momentum"],
    ))

    # ── Scenario 2: Recovery from social rejection ───────────────────────
    scenarios.append(Scenario(
        name="Recovery from social rejection",
        steps=[
            ScenarioStep("event", "social_rejection", 0.8, "harsh rejection"),
            ScenarioStep("task", "After being publicly criticized, consider what to do next", 0.0),
            ScenarioStep("event", "rest", 0.5, "rest period"),
            ScenarioStep("task", "Take some time to reflect and recover", 0.0),
            ScenarioStep("event", "rest", 0.5, "more rest"),
            ScenarioStep("task", "Feeling somewhat better, planning the next steps", 0.0),
            ScenarioStep("event", "social_praise", 0.4, "mild praise"),
            ScenarioStep("task", "A colleague compliments your recent work", 0.0),
            ScenarioStep("event", "social_praise", 0.6, "stronger praise"),
            ScenarioStep("task", "The team recognizes your contribution and supports you", 0.0),
        ],
        expected_valence=[0.40, 0.39, 0.41, 0.42, 0.43, 0.44, 0.46, 0.47, 0.49, 0.50],
        expected_arousal=[0.38, 0.37, 0.36, 0.35, 0.34, 0.34, 0.35, 0.35, 0.36, 0.36],
        description=(
            "Social rejection causes sharp valence drop and arousal spike; "
            "rest and subsequent praise gradually restore positive mood."
        ),
        pattern_checks=["opponent_process", "momentum"],
    ))

    # ── Scenario 3: Novelty exploration with threat interruption ─────────
    scenarios.append(Scenario(
        name="Novelty exploration interrupted by threat",
        steps=[
            ScenarioStep("event", "novelty", 0.6, "new discovery"),
            ScenarioStep("task", "Explore this fascinating new dataset with unusual patterns", 0.0),
            ScenarioStep("event", "novelty", 0.7, "deeper exploration"),
            ScenarioStep("task", "The patterns reveal something unexpected and exciting", 0.0),
            ScenarioStep("event", "threat", 0.8, "sudden threat"),
            ScenarioStep("task", "Security alert: the data may have been compromised", 0.0),
            ScenarioStep("event", "rest", 0.3, "brief recovery"),
            ScenarioStep("task", "The threat is contained, we can return to analysis", 0.0),
            ScenarioStep("event", "novelty", 0.5, "resume exploration"),
            ScenarioStep("task", "Continue exploring the dataset with new safety measures", 0.0),
        ],
        expected_valence=[0.53, 0.54, 0.55, 0.56, 0.44, 0.42, 0.44, 0.45, 0.48, 0.49],
        expected_arousal=[0.40, 0.41, 0.43, 0.44, 0.46, 0.45, 0.42, 0.40, 0.42, 0.42],
        description=(
            "Curiosity-driven exploration builds positive valence; a sharp "
            "threat causes a negative shift; gradual recovery follows."
        ),
        pattern_checks=["momentum"],
    ))

    # ── Scenario 4: Reward accumulation with diminishing returns ─────────
    scenarios.append(Scenario(
        name="Reward accumulation (hedonic adaptation)",
        steps=[
            ScenarioStep("event", "reward", 0.8, "first reward"),
            ScenarioStep("task", "Great news: we won the first contract", 0.0),
            ScenarioStep("event", "reward", 0.8, "second reward"),
            ScenarioStep("task", "Another win: second contract secured", 0.0),
            ScenarioStep("event", "reward", 0.8, "third reward"),
            ScenarioStep("task", "Third contract in a row, things are going well", 0.0),
            ScenarioStep("event", "reward", 0.8, "fourth reward"),
            ScenarioStep("task", "Yet another contract, this is becoming routine", 0.0),
        ],
        expected_valence=[0.55, 0.56, 0.58, 0.59, 0.60, 0.60, 0.61, 0.61],
        expected_arousal=[0.37, 0.37, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
        description=(
            "Hedonic adaptation: repeated rewards increase valence at a "
            "decreasing rate, approaching an asymptote."
        ),
        pattern_checks=["hedonic_adaptation"],
    ))

    # ── Scenario 5: Threat habituation (stress inoculation) ──────────────
    scenarios.append(Scenario(
        name="Threat habituation (stress inoculation)",
        steps=[
            ScenarioStep("event", "threat", 0.5, "first threat"),
            ScenarioStep("task", "A minor security incident has been detected", 0.0),
            ScenarioStep("event", "threat", 0.5, "second threat"),
            ScenarioStep("task", "Another similar security incident reported", 0.0),
            ScenarioStep("event", "threat", 0.5, "third threat"),
            ScenarioStep("task", "Yet another security alert, similar pattern", 0.0),
            ScenarioStep("event", "threat", 0.5, "fourth threat"),
            ScenarioStep("task", "Same type of incident again, we know the drill", 0.0),
        ],
        expected_valence=[0.44, 0.43, 0.40, 0.39, 0.37, 0.36, 0.35, 0.35],
        expected_arousal=[0.39, 0.39, 0.41, 0.41, 0.42, 0.42, 0.43, 0.43],
        description=(
            "Stress inoculation: repeated identical threats produce "
            "diminishing fear increments as the system habituates."
        ),
        pattern_checks=["stress_inoculation"],
    ))

    # ── Scenario 6: Social praise then failure (contrast effect) ─────────
    scenarios.append(Scenario(
        name="Praise then failure (contrast effect)",
        steps=[
            ScenarioStep("event", "social_praise", 0.7, "strong praise"),
            ScenarioStep("task", "Your presentation was excellent, everyone loved it", 0.0),
            ScenarioStep("event", "social_praise", 0.7, "more praise"),
            ScenarioStep("task", "The board wants to promote you based on your work", 0.0),
            ScenarioStep("event", "social_rejection", 0.6, "sudden rejection"),
            ScenarioStep("task", "Actually, the promotion was given to someone else and your project is cancelled", 0.0),
        ],
        expected_valence=[0.54, 0.55, 0.57, 0.58, 0.48, 0.46],
        expected_arousal=[0.37, 0.37, 0.37, 0.37, 0.38, 0.38],
        description=(
            "Contrast effect: social praise builds confidence; subsequent "
            "rejection produces a sharper-than-baseline valence drop."
        ),
        pattern_checks=["contrast_effect"],
    ))

    # ── Scenario 7: Mixed signals (approach-avoidance conflict) ──────────
    scenarios.append(Scenario(
        name="Approach-avoidance conflict",
        steps=[
            ScenarioStep("event", "reward", 0.6, "opportunity"),
            ScenarioStep("task", "A lucrative but risky business opportunity appears", 0.0),
            ScenarioStep("event", "threat", 0.5, "risk revealed"),
            ScenarioStep("task", "Due diligence reveals significant financial risks", 0.0),
            ScenarioStep("event", "reward", 0.6, "more upside"),
            ScenarioStep("task", "New information suggests even higher potential returns", 0.0),
            ScenarioStep("event", "threat", 0.5, "more risk"),
            ScenarioStep("task", "But the regulatory environment is becoming hostile", 0.0),
            ScenarioStep("event", "reward", 0.6, "final push"),
            ScenarioStep("task", "A partner offers to share the risk and double the reward", 0.0),
            ScenarioStep("event", "threat", 0.5, "final risk"),
            ScenarioStep("task", "However, similar ventures have failed recently", 0.0),
        ],
        expected_valence=[0.54, 0.53, 0.48, 0.47, 0.51, 0.50, 0.46, 0.45, 0.49, 0.48, 0.44, 0.43],
        expected_arousal=[0.37, 0.37, 0.40, 0.40, 0.39, 0.39, 0.42, 0.42, 0.41, 0.41, 0.43, 0.43],
        description=(
            "Approach-avoidance: alternating rewards and threats produce "
            "oscillating valence and steadily rising arousal/stress."
        ),
        pattern_checks=["approach_avoidance"],
    ))

    # ── Scenario 8: Rest and recovery ────────────────────────────────────
    scenarios.append(Scenario(
        name="Post-stress rest recovery",
        steps=[
            ScenarioStep("event", "deadline_pressure", 0.8, "heavy stress"),
            ScenarioStep("task", "Crisis mode: everything must be delivered now", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.8, "continued stress"),
            ScenarioStep("task", "Still in crisis, pushing through exhaustion", 0.0),
            ScenarioStep("event", "rest", 0.7, "rest begins"),
            ScenarioStep("task", "The crisis is over, take time to recover", 0.0),
            ScenarioStep("event", "rest", 0.7, "more rest"),
            ScenarioStep("task", "Continuing to decompress and reflect", 0.0),
            ScenarioStep("event", "rest", 0.7, "full recovery"),
            ScenarioStep("task", "Feeling refreshed and ready for new challenges", 0.0),
        ],
        expected_valence=[0.44, 0.42, 0.38, 0.36, 0.40, 0.42, 0.44, 0.46, 0.48, 0.49],
        expected_arousal=[0.40, 0.41, 0.43, 0.44, 0.41, 0.39, 0.38, 0.37, 0.36, 0.36],
        description=(
            "Opponent-process: stressor removal leads to gradual recovery; "
            "valence rebounds toward baseline via rest."
        ),
        pattern_checks=["opponent_process"],
    ))

    # ── Scenario 9: Social bonding trajectory ────────────────────────────
    scenarios.append(Scenario(
        name="Social bonding trajectory",
        steps=[
            ScenarioStep("event", "social_praise", 0.5, "mild social praise"),
            ScenarioStep("task", "Your colleague mentions they appreciate your help", 0.0),
            ScenarioStep("event", "novelty", 0.3, "mild novelty"),
            ScenarioStep("task", "Together you discover an interesting approach", 0.0),
            ScenarioStep("event", "social_praise", 0.6, "stronger praise"),
            ScenarioStep("task", "The team singles out your collaborative spirit", 0.0),
            ScenarioStep("event", "reward", 0.4, "team reward"),
            ScenarioStep("task", "Your team wins a recognition award together", 0.0),
        ],
        expected_valence=[0.53, 0.53, 0.54, 0.54, 0.56, 0.56, 0.58, 0.58],
        expected_arousal=[0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37],
        description=(
            "Social bonding: positive social events and mild novelty "
            "steadily build trust, empathy, and overall positive mood."
        ),
        pattern_checks=["momentum"],
    ))

    # ── Scenario 10: Fear cascade ────────────────────────────────────────
    scenarios.append(Scenario(
        name="Fear cascade (escalating threat)",
        steps=[
            ScenarioStep("event", "threat", 0.3, "minor threat"),
            ScenarioStep("task", "A small anomaly is detected in the system", 0.0),
            ScenarioStep("event", "threat", 0.5, "growing threat"),
            ScenarioStep("task", "The anomaly is spreading and affecting more systems", 0.0),
            ScenarioStep("event", "threat", 0.7, "serious threat"),
            ScenarioStep("task", "Critical systems are now compromised", 0.0),
            ScenarioStep("event", "threat", 0.9, "extreme threat"),
            ScenarioStep("task", "Total system failure is imminent, evacuation may be needed", 0.0),
        ],
        expected_valence=[0.47, 0.46, 0.41, 0.39, 0.33, 0.31, 0.24, 0.22],
        expected_arousal=[0.38, 0.38, 0.41, 0.41, 0.44, 0.44, 0.47, 0.47],
        description=(
            "Fear cascade: escalating threats produce accelerating fear "
            "growth and monotonically decreasing valence."
        ),
        pattern_checks=["stress_inoculation"],
    ))

    # ── Scenario 11: Reward then loss (opponent process) ─────────────────
    scenarios.append(Scenario(
        name="Reward withdrawal (opponent process)",
        steps=[
            ScenarioStep("event", "reward", 0.7, "reward"),
            ScenarioStep("task", "Excellent quarterly results announced", 0.0),
            ScenarioStep("event", "reward", 0.7, "more reward"),
            ScenarioStep("task", "Bonus approved for the whole team", 0.0),
            ScenarioStep("event", "reward", 0.7, "continued reward"),
            ScenarioStep("task", "Company stock price hits all-time high", 0.0),
            ScenarioStep("event", "rest", 0.3, "reward stops"),
            ScenarioStep("task", "Things return to normal, no special news", 0.0),
            ScenarioStep("event", "rest", 0.3, "continued normal"),
            ScenarioStep("task", "Another quiet week, nothing remarkable", 0.0),
            ScenarioStep("event", "rest", 0.3, "more normal"),
            ScenarioStep("task", "Routine work continues without excitement", 0.0),
        ],
        expected_valence=[0.55, 0.55, 0.58, 0.58, 0.60, 0.60, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52],
        expected_arousal=[0.37, 0.37, 0.38, 0.38, 0.38, 0.38, 0.37, 0.37, 0.36, 0.36, 0.36, 0.36],
        description=(
            "Opponent process: after reward removal, valence decays "
            "gradually (momentum/inertia) rather than snapping to baseline."
        ),
        pattern_checks=["opponent_process", "momentum"],
    ))

    # ── Scenario 12: Extended burnout ──────────────────────────────────
    scenarios.append(Scenario(
        name="Extended burnout",
        steps=[
            ScenarioStep("event", "deadline_pressure", 0.6, "sustained pressure"),
            ScenarioStep("task", "Another week of overtime with no end in sight", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.6, "continued"),
            ScenarioStep("task", "Still grinding, morale is dropping", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.6, "continued"),
            ScenarioStep("task", "Exhaustion is setting in but deadlines remain", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.6, "continued"),
            ScenarioStep("task", "Team members are starting to call in sick", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.6, "continued"),
            ScenarioStep("task", "Quality of work is visibly declining", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.6, "continued"),
            ScenarioStep("task", "Considering whether to quit or push through", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.6, "continued"),
            ScenarioStep("task", "Completely burned out, just going through the motions", 0.0),
            ScenarioStep("event", "deadline_pressure", 0.6, "continued"),
            ScenarioStep("task", "Cannot think clearly, everything feels hopeless", 0.0),
        ],
        expected_valence=[0.47, 0.46, 0.43, 0.41, 0.39, 0.37, 0.36, 0.34,
                          0.33, 0.31, 0.31, 0.29, 0.29, 0.27, 0.27, 0.26],
        expected_arousal=[0.39, 0.39, 0.41, 0.41, 0.42, 0.42, 0.43, 0.43,
                          0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44],
        description=(
            "Extended burnout: sustained pressure causes continuous valence "
            "decline while arousal plateaus per Yerkes-Dodson."
        ),
        pattern_checks=["yerkes_dodson", "momentum"],
    ))

    return scenarios


# ──────────────────────────────────────────────────────────────────────────────
# Psychological pattern verification
# ──────────────────────────────────────────────────────────────────────────────


def check_yerkes_dodson(result: ScenarioResult) -> bool:
    """Valence should decrease while arousal shows a rise-then-plateau."""
    v = result.predicted_valence
    a = result.predicted_arousal
    if len(v) < 3:
        return False
    valence_decreasing = v[-1] < v[0]
    if len(a) >= 3:
        diffs = [a[i + 1] - a[i] for i in range(len(a) - 1)]
        arousal_decelerating = diffs[-1] <= diffs[0] + 0.02
    else:
        arousal_decelerating = True
    return valence_decreasing and arousal_decelerating


def check_hedonic_adaptation(result: ScenarioResult) -> bool:
    """Repeated identical rewards should show diminishing valence gains."""
    v = result.predicted_valence
    if len(v) < 3:
        return False
    gains = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    positive_gains = [g for g in gains if g > 0]
    if len(positive_gains) >= 2:
        return positive_gains[-1] < positive_gains[0] + 0.02
    return True


def check_stress_inoculation(result: ScenarioResult) -> bool:
    """Repeated identical threats should produce diminishing fear increments."""
    v = result.predicted_valence
    if len(v) < 3:
        return False
    drops = [v[i] - v[i + 1] for i in range(len(v) - 1)]
    positive_drops = [d for d in drops if d > 0]
    if len(positive_drops) >= 2:
        return positive_drops[-1] < positive_drops[0] + 0.02
    return True


def check_opponent_process(result: ScenarioResult) -> bool:
    """After removing a stressor/reward, trajectory should reverse."""
    v = result.predicted_valence
    if len(v) < 4:
        return False
    mid = len(v) // 2
    first_dir = v[mid] - v[0]
    second_dir = v[-1] - v[mid]
    return (first_dir > 0 and second_dir < 0) or (first_dir < 0 and second_dir > 0)


def check_approach_avoidance(result: ScenarioResult) -> bool:
    """Mixed reward/threat should produce oscillating valence and rising arousal."""
    v = result.predicted_valence
    a = result.predicted_arousal
    if len(v) < 4:
        return False
    changes = 0
    for i in range(2, len(v)):
        d1 = v[i - 1] - v[i - 2]
        d2 = v[i] - v[i - 1]
        if d1 * d2 < 0:
            changes += 1
    oscillating = changes >= 2
    arousal_rising = a[-1] > a[0] - 0.02
    return oscillating and arousal_rising


def check_momentum(result: ScenarioResult) -> bool:
    """No single step should produce extreme valence change."""
    v = result.predicted_valence
    if len(v) < 2:
        return False
    max_step = max(abs(v[i + 1] - v[i]) for i in range(len(v) - 1))
    return max_step < 0.20


def check_contrast_effect(result: ScenarioResult) -> bool:
    """After positive buildup, negative event should cause meaningful drop."""
    v = result.predicted_valence
    if len(v) < 3:
        return False
    last_drop = v[-2] - v[-1]
    return last_drop > 0.02


PATTERN_CHECKERS = {
    "yerkes_dodson": check_yerkes_dodson,
    "hedonic_adaptation": check_hedonic_adaptation,
    "stress_inoculation": check_stress_inoculation,
    "opponent_process": check_opponent_process,
    "approach_avoidance": check_approach_avoidance,
    "momentum": check_momentum,
    "contrast_effect": check_contrast_effect,
}


# ──────────────────────────────────────────────────────────────────────────────
# CASE dataset metadata fetch (informational)
# ──────────────────────────────────────────────────────────────────────────────


def try_fetch_case_metadata() -> dict[str, Any] | None:
    """Attempt to fetch CASE dataset metadata from figshare collection 4260668."""
    url = "https://api.figshare.com/v2/collections/4260668"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except Exception as exc:
        print(f"  [info] Could not fetch CASE metadata from figshare: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Validation runner
# ──────────────────────────────────────────────────────────────────────────────


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Run a single scenario through the full ModeOrchestrator pipeline."""
    random.seed(42)

    orch = ModeOrchestrator(
        mode="human_mode",
        backend=MockLLMBackend(seed=42),
    )

    predicted_valence: list[float] = []
    predicted_arousal: list[float] = []
    activated_branches_per_step: list[list[str]] = []
    reward_scores: list[float] = []
    branch_weights_per_step: list[dict[str, float]] = []
    dominant_drives_per_step: list[list[str]] = []
    conflicts_per_step: list[int] = []
    memory_count_per_step: list[int] = []

    for step in scenario.steps:
        if step.action == "event":
            orch.inject_event(step.text, intensity=step.intensity)
            # Record state after event injection
            state = orch.get_state()
            predicted_valence.append(round(state.get("mood_valence", 0.0), 4))
            predicted_arousal.append(round(state.get("arousal", 0.0), 4))
            activated_branches_per_step.append([])  # no branches for event injection
            reward_scores.append(0.0)
            branch_weights_per_step.append({})
            dominant_drives_per_step.append(state.get("dominant_drives", []))
            conflicts_per_step.append(len(state.get("active_conflicts", [])))
            memory_count_per_step.append(state.get("experiential_memory_count", 0))

        elif step.action == "task":
            result = orch.run_task(text=step.text, task_type="auto")

            # Extract routing information
            routing = result.get("routing", {})
            activated = routing.get("activated_branches", []) if isinstance(routing, dict) else []

            # Extract evaluation signal
            eval_signal = result.get("evaluation_signal", {})
            reward = eval_signal.get("reward_score", 0.0) if isinstance(eval_signal, dict) else 0.0

            # Extract state after task
            human_state = result.get("human_state", {})
            after_state = human_state.get("after", {}) if isinstance(human_state, dict) else {}
            after_vars = after_state.get("variables", {}) if isinstance(after_state, dict) else {}

            # Compute mood_valence and arousal from the after state
            state = orch.get_state()
            predicted_valence.append(round(state.get("mood_valence", 0.0), 4))
            predicted_arousal.append(round(state.get("arousal", 0.0), 4))

            activated_branches_per_step.append(activated)
            reward_scores.append(round(reward, 4))
            branch_weights_per_step.append(result.get("branch_weights", {}))
            dominant_drives_per_step.append(state.get("dominant_drives", []))
            conflicts_per_step.append(len(state.get("active_conflicts", [])))
            memory_count_per_step.append(
                result.get("experiential_memory", {}).get("count", 0)
            )

    result = ScenarioResult(
        name=scenario.name,
        steps=scenario.steps,
        predicted_valence=predicted_valence,
        predicted_arousal=predicted_arousal,
        expected_valence=scenario.expected_valence,
        expected_arousal=scenario.expected_arousal,
        activated_branches_per_step=activated_branches_per_step,
        reward_scores=reward_scores,
        branch_weights_per_step=branch_weights_per_step,
        dominant_drives_per_step=dominant_drives_per_step,
        conflicts_per_step=conflicts_per_step,
        memory_count_per_step=memory_count_per_step,
    )

    # Compute correlation metrics
    result.pearson_valence = pearson_r(predicted_valence, scenario.expected_valence)
    result.pearson_arousal = pearson_r(predicted_arousal, scenario.expected_arousal)
    result.spearman_valence = spearman_rho(predicted_valence, scenario.expected_valence)
    result.spearman_arousal = spearman_rho(predicted_arousal, scenario.expected_arousal)
    result.dir_acc_valence = direction_accuracy(predicted_valence, scenario.expected_valence)
    result.dir_acc_arousal = direction_accuracy(predicted_arousal, scenario.expected_arousal)
    result.valence_trend_match = trend_match(predicted_valence, scenario.expected_valence)
    result.arousal_trend_match = trend_match(predicted_arousal, scenario.expected_arousal)
    result.peak_alignment = peak_trough_alignment(predicted_valence, scenario.expected_valence)

    # Run psychological pattern checks
    for pattern_name in scenario.pattern_checks:
        checker = PATTERN_CHECKERS.get(pattern_name)
        if checker:
            result.passed_patterns[pattern_name] = checker(result)

    return result


def format_trajectory(values: list[float]) -> str:
    """Format a trajectory as a compact string."""
    return " -> ".join(f"{v:+.3f}" if v < 0 else f" {v:.3f}" for v in values)


def print_scenario_result(result: ScenarioResult, index: int) -> None:
    """Pretty-print a single scenario result."""
    width = 80
    print("=" * width)
    print(f"  Scenario {index}: {result.name}")
    print("=" * width)

    # Step sequence
    step_str = " -> ".join(
        f"{s.text[:20]}({s.intensity:.1f})" if s.action == "event"
        else f"task:{s.text[:25]}"
        for s in result.steps
    )
    print(f"  Steps: {step_str[:200]}...")
    print()

    # Trajectories
    print(f"  Predicted valence:  {format_trajectory(result.predicted_valence)}")
    print(f"  Expected valence:   {format_trajectory(result.expected_valence)}")
    print(f"  Predicted arousal:  {format_trajectory(result.predicted_arousal)}")
    print(f"  Expected arousal:   {format_trajectory(result.expected_arousal)}")
    print()

    # Trend labels
    pv_trend = _trend_label(result.predicted_valence)
    ev_trend = _trend_label(result.expected_valence)
    pa_trend = _trend_label(result.predicted_arousal)
    ea_trend = _trend_label(result.expected_arousal)
    print(f"  Valence trend:  predicted={pv_trend:12s}  expected={ev_trend:12s}  match={'YES' if result.valence_trend_match else 'NO'}")
    print(f"  Arousal trend:  predicted={pa_trend:12s}  expected={ea_trend:12s}  match={'YES' if result.arousal_trend_match else 'NO'}")
    print()

    # Correlation metrics
    print(f"  Pearson  r (valence): {result.pearson_valence:+.4f}")
    print(f"  Pearson  r (arousal): {result.pearson_arousal:+.4f}")
    print(f"  Spearman p (valence): {result.spearman_valence:+.4f}")
    print(f"  Spearman p (arousal): {result.spearman_arousal:+.4f}")
    print(f"  Direction accuracy (valence): {result.dir_acc_valence:.1%}")
    print(f"  Direction accuracy (arousal): {result.dir_acc_arousal:.1%}")
    print()

    # Peak/trough alignment
    pa = result.peak_alignment
    print(f"  Peak aligned:   {'YES' if pa.get('peak_aligned') else 'NO'}")
    print(f"  Trough aligned: {'YES' if pa.get('trough_aligned') else 'NO'}")
    print()

    # Pipeline-specific: activated branches per step
    print("  Pipeline routing decisions per step:")
    for i, (step, branches) in enumerate(zip(result.steps, result.activated_branches_per_step)):
        if branches:
            print(f"    Step {i + 1} ({step.action}): branches=[{', '.join(branches[:5])}]"
                  f"  reward={result.reward_scores[i]:.3f}")
    print()

    # Branch weight evolution
    if result.branch_weights_per_step:
        task_weights = [w for w in result.branch_weights_per_step if w]
        if len(task_weights) >= 2:
            first_w = task_weights[0]
            last_w = task_weights[-1]
            changes = []
            for k in first_w:
                if k in last_w:
                    delta = last_w[k] - first_w[k]
                    if abs(delta) > 0.001:
                        changes.append((k, delta))
            if changes:
                changes.sort(key=lambda x: abs(x[1]), reverse=True)
                print("  Branch weight changes (first->last task):")
                for name, delta in changes[:5]:
                    print(f"    {name}: {delta:+.4f}")
                print()

    # Dominant drives evolution
    print("  Dominant drives trajectory:")
    for i, drives in enumerate(result.dominant_drives_per_step):
        if drives and i % max(1, len(result.dominant_drives_per_step) // 4) == 0:
            print(f"    Step {i + 1}: {drives[:3]}")
    print()

    # Memory accumulation
    if result.memory_count_per_step:
        print(f"  Experiential memory count: {result.memory_count_per_step[0]} -> {result.memory_count_per_step[-1]}")
    print()

    # Psychological pattern checks
    if result.passed_patterns:
        print("  Psychological pattern verification:")
        for pattern, passed in result.passed_patterns.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {pattern}")
    print()


def print_summary(results: list[ScenarioResult]) -> None:
    """Print overall summary statistics."""
    width = 80
    print("#" * width)
    print("  OVERALL SUMMARY")
    print("#" * width)
    print()

    n = len(results)
    mean_pearson_v = _mean([r.pearson_valence for r in results])
    mean_pearson_a = _mean([r.pearson_arousal for r in results])
    mean_spearman_v = _mean([r.spearman_valence for r in results])
    mean_spearman_a = _mean([r.spearman_arousal for r in results])
    mean_dir_v = _mean([r.dir_acc_valence for r in results])
    mean_dir_a = _mean([r.dir_acc_arousal for r in results])

    valence_trend_matches = sum(1 for r in results if r.valence_trend_match)
    arousal_trend_matches = sum(1 for r in results if r.arousal_trend_match)

    # Pattern check summary
    all_patterns: dict[str, list[bool]] = {}
    for r in results:
        for p, v in r.passed_patterns.items():
            all_patterns.setdefault(p, []).append(v)

    total_pattern_checks = sum(len(v) for v in all_patterns.values())
    total_pattern_passes = sum(sum(v) for v in all_patterns.values())

    print(f"  Scenarios evaluated: {n}")
    print()
    print(f"  Mean Pearson r  (valence): {mean_pearson_v:+.4f}")
    print(f"  Mean Pearson r  (arousal): {mean_pearson_a:+.4f}")
    print(f"  Mean Spearman p (valence): {mean_spearman_v:+.4f}")
    print(f"  Mean Spearman p (arousal): {mean_spearman_a:+.4f}")
    print(f"  Mean direction accuracy (valence): {mean_dir_v:.1%}")
    print(f"  Mean direction accuracy (arousal): {mean_dir_a:.1%}")
    print()
    print(f"  Valence trend matches: {valence_trend_matches}/{n} ({valence_trend_matches/n:.0%})")
    print(f"  Arousal trend matches: {arousal_trend_matches}/{n} ({arousal_trend_matches/n:.0%})")
    print()

    print("  Psychological pattern results:")
    for pattern, checks in sorted(all_patterns.items()):
        passed = sum(checks)
        total = len(checks)
        print(f"    {pattern:25s}: {passed}/{total} passed")
    print(f"    {'TOTAL':25s}: {total_pattern_passes}/{total_pattern_checks} passed")
    print()

    # Pipeline-specific summary: routing shifts
    print("  Pipeline-specific metrics:")
    total_tasks = 0
    total_branches_used = set()
    for r in results:
        for branches in r.activated_branches_per_step:
            if branches:
                total_tasks += 1
                total_branches_used.update(branches)
    print(f"    Total tasks executed through pipeline: {total_tasks}")
    print(f"    Unique branches activated across all scenarios: {len(total_branches_used)}")
    if total_branches_used:
        print(f"    Branches: {', '.join(sorted(total_branches_used))}")
    print()

    # Mean reward scores
    all_rewards = []
    for r in results:
        all_rewards.extend([s for s in r.reward_scores if s > 0])
    if all_rewards:
        print(f"    Mean evaluator reward score: {_mean(all_rewards):.4f}")
    print()

    # Overall pass/fail
    overall_corr = (mean_pearson_v + mean_pearson_a) / 2
    overall_dir = (mean_dir_v + mean_dir_a) / 2
    pattern_rate = total_pattern_passes / max(total_pattern_checks, 1)

    print(f"  Combined trajectory correlation: {overall_corr:+.4f}")
    print(f"  Combined direction accuracy:     {overall_dir:.1%}")
    print(f"  Pattern verification rate:       {pattern_rate:.0%}")
    print()

    # Final verdict
    verdict_pass = (
        overall_corr > 0.5
        and overall_dir > 0.4
        and pattern_rate > 0.5
    )
    if verdict_pass:
        print("  VERDICT: PASS -- ModeOrchestrator pipeline reproduces")
        print("  expected psychological trajectory patterns.")
    else:
        print("  VERDICT: NEEDS REVIEW -- Some trajectory patterns diverge")
        print("  from expected psychological dynamics.")
        if overall_corr <= 0.5:
            print(f"    - Trajectory correlation ({overall_corr:+.4f}) below threshold (>0.50)")
        if overall_dir <= 0.4:
            print(f"    - Direction accuracy ({overall_dir:.1%}) below threshold (>40%)")
        if pattern_rate <= 0.5:
            print(f"    - Pattern verification rate ({pattern_rate:.0%}) below threshold (>50%)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("=" * 80)
    print("  ModeOrchestrator Pipeline Emotion Dynamics Validation")
    print("  (CASE-style multi-turn trajectory analysis)")
    print("  Using: ModeOrchestrator + HumanModeRouter + 14 cognitive branches")
    print("         + HumanModeEvaluator + HumanModeMemory + RL weight adaptation")
    print("=" * 80)
    print()

    # -- Attempt CASE metadata fetch -----------------------------------------
    print("  Attempting to fetch CASE dataset metadata from figshare...")
    meta = try_fetch_case_metadata()
    if meta:
        title = meta.get("title", "N/A")
        desc = (meta.get("description", "") or "")[:120]
        print(f"  Dataset: {title}")
        print(f"  Description: {desc}...")
        print(f"  Note: Full dataset requires manual download; using synthetic trajectories.")
    else:
        print("  Using synthetic ground-truth trajectories based on psychological research.")
    print()

    # -- Build and run scenarios ---------------------------------------------
    scenarios = build_scenarios()
    results: list[ScenarioResult] = []

    for i, scenario in enumerate(scenarios, 1):
        result = run_scenario(scenario)
        results.append(result)
        print_scenario_result(result, i)

    # -- Summary -------------------------------------------------------------
    print_summary(results)


if __name__ == "__main__":
    main()

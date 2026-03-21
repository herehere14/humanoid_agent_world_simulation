#!/usr/bin/env python3
"""Blind judging of Heart vs Pure-LLM outputs.

GPT-4o reads event history + both outputs (randomized A/B order),
scores on 4 dimensions, picks a winner.

Usage:
    python -m learned_brain.world_sim.eval.judge_comparison [--samples eval_samples.json]
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from openai import OpenAI

JUDGE_PROMPT = """You are evaluating two AI systems that simulate a character's behavior in a small-town drama. You will see the character's personality, what happened to them, and two responses (A and B) at a specific moment.

CHARACTER: {name}
Background: {background}
Temperament: {temperament}

TIMELINE OF EVENTS THIS CHARACTER EXPERIENCED:
{event_history}

CURRENT MOMENT: {time_str} at {location}
People nearby: {nearby}

--- RESPONSE A ---
Action: {action_a}
Speech/Behavior: "{speech_a}"

--- RESPONSE B ---
Action: {action_b}
Speech/Behavior: "{speech_b}"

Score each response on these criteria (1-5 scale):

1. CONSISTENCY: Does the response logically follow from the character's history and personality? Does it account for recent events?

2. EMOTIONAL REALISM: Does the emotional tone match what a real person with this background would feel at this moment? Not over-dramatic, not under-reactive.

3. BEHAVIORAL COHERENCE: Is the chosen action appropriate for someone in this emotional state?

4. TEMPORAL AWARENESS: Does the response reflect appropriate emotional persistence or decay? (e.g., still upset days later vs. immediately over it)

Respond in this exact JSON format:
{{
  "consistency_a": <1-5>,
  "consistency_b": <1-5>,
  "realism_a": <1-5>,
  "realism_b": <1-5>,
  "coherence_a": <1-5>,
  "coherence_b": <1-5>,
  "temporal_a": <1-5>,
  "temporal_b": <1-5>,
  "preferred": "A" or "B" or "TIE",
  "reasoning": "<2-3 sentences explaining your preference>"
}}"""


def judge_sample(client: OpenAI, sample: dict, is_heart_a: bool) -> dict:
    """Judge one sample with randomized A/B assignment."""
    if is_heart_a:
        action_a, speech_a = sample["heart_action"], sample["heart_speech"]
        action_b, speech_b = sample["llm_action"], sample["llm_speech"]
    else:
        action_a, speech_a = sample["llm_action"], sample["llm_speech"]
        action_b, speech_b = sample["heart_action"], sample["heart_speech"]

    event_history = "\n".join(sample["event_history"][-12:]) if sample["event_history"] else "Nothing notable yet."
    nearby = ", ".join(sample["nearby_agents"][:8]) if sample["nearby_agents"] else "No one"

    prompt = JUDGE_PROMPT.format(
        name=sample["agent_name"],
        background=sample["agent_background"],
        temperament=sample["agent_temperament"],
        event_history=event_history,
        time_str=sample["time_str"],
        location=sample["location"],
        nearby=nearby,
        action_a=action_a,
        speech_a=speech_a,
        action_b=action_b,
        speech_b=speech_b,
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
        temperature=0.0,
    )

    try:
        judgment = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        judgment = {
            "consistency_a": 3, "consistency_b": 3,
            "realism_a": 3, "realism_b": 3,
            "coherence_a": 3, "coherence_b": 3,
            "temporal_a": 3, "temporal_b": 3,
            "preferred": "TIE", "reasoning": "Failed to parse judge response."
        }

    # Unscramble: map A/B back to heart/llm
    if is_heart_a:
        result = {
            "heart_consistency": judgment.get("consistency_a", 3),
            "llm_consistency": judgment.get("consistency_b", 3),
            "heart_realism": judgment.get("realism_a", 3),
            "llm_realism": judgment.get("realism_b", 3),
            "heart_coherence": judgment.get("coherence_a", 3),
            "llm_coherence": judgment.get("coherence_b", 3),
            "heart_temporal": judgment.get("temporal_a", 3),
            "llm_temporal": judgment.get("temporal_b", 3),
            "preferred_raw": judgment.get("preferred", "TIE"),
        }
        pref = judgment.get("preferred", "TIE")
        result["winner"] = "heart" if pref == "A" else "llm" if pref == "B" else "tie"
    else:
        result = {
            "heart_consistency": judgment.get("consistency_b", 3),
            "llm_consistency": judgment.get("consistency_a", 3),
            "heart_realism": judgment.get("realism_b", 3),
            "llm_realism": judgment.get("realism_a", 3),
            "heart_coherence": judgment.get("coherence_b", 3),
            "llm_coherence": judgment.get("coherence_a", 3),
            "heart_temporal": judgment.get("temporal_b", 3),
            "llm_temporal": judgment.get("temporal_a", 3),
            "preferred_raw": judgment.get("preferred", "TIE"),
        }
        pref = judgment.get("preferred", "TIE")
        result["winner"] = "heart" if pref == "B" else "llm" if pref == "A" else "tie"

    result["reasoning"] = judgment.get("reasoning", "")
    result["agent_id"] = sample["agent_id"]
    result["tick"] = sample["tick"]
    result["time_str"] = sample["time_str"]
    return result


def print_results(results: list[dict]):
    """Print aggregate comparison results."""
    n = len(results)
    heart_wins = sum(1 for r in results if r["winner"] == "heart")
    llm_wins = sum(1 for r in results if r["winner"] == "llm")
    ties = sum(1 for r in results if r["winner"] == "tie")

    print(f"\n{'═' * 70}")
    print(f"  HEART vs PURE-LLM COMPARISON RESULTS ({n} samples)")
    print(f"{'═' * 70}")

    print(f"\n  Overall: Heart {heart_wins} | LLM {llm_wins} | Tie {ties}")
    print(f"  Heart win rate: {heart_wins / n * 100:.1f}%")

    # Average scores per dimension
    dims = ["consistency", "realism", "coherence", "temporal"]
    print(f"\n  {'Dimension':<14s} {'Heart':>7s} {'LLM':>7s} {'Delta':>7s}")
    print(f"  {'─' * 14} {'─' * 7} {'─' * 7} {'─' * 7}")
    for dim in dims:
        h_avg = sum(r[f"heart_{dim}"] for r in results) / n
        l_avg = sum(r[f"llm_{dim}"] for r in results) / n
        delta = h_avg - l_avg
        marker = "←" if delta > 0.2 else "→" if delta < -0.2 else " "
        print(f"  {dim:<14s} {h_avg:>7.2f} {l_avg:>7.2f} {delta:>+7.2f} {marker}")

    # Win rate by phase
    baseline_ticks = {72}
    crisis_ticks = {82, 105, 106, 113, 128}
    recovery_ticks = {158, 187, 230, 286}

    phases = [
        ("Baseline", baseline_ticks),
        ("Crisis", crisis_ticks),
        ("Recovery", recovery_ticks),
    ]
    print(f"\n  Win rate by phase:")
    for phase_name, ticks in phases:
        phase_results = [r for r in results if r["tick"] in ticks]
        if not phase_results:
            continue
        h = sum(1 for r in phase_results if r["winner"] == "heart")
        l = sum(1 for r in phase_results if r["winner"] == "llm")
        t = sum(1 for r in phase_results if r["winner"] == "tie")
        pn = len(phase_results)
        print(f"    {phase_name:<12s}: Heart {h}/{pn} ({h / pn * 100:.0f}%) | LLM {l}/{pn} ({l / pn * 100:.0f}%) | Tie {t}/{pn}")

    # Win rate by agent type
    laid_off = {"marcus", "rosa", "jake", "greg"}
    not_laid_off = {"diana", "sarah", "tom"}
    management = {"richard"}

    groups = [
        ("Laid off", laid_off),
        ("Not laid off", not_laid_off),
        ("Management", management),
    ]
    print(f"\n  Win rate by agent type:")
    for group_name, agent_ids in groups:
        group_results = [r for r in results if r["agent_id"] in agent_ids]
        if not group_results:
            continue
        h = sum(1 for r in group_results if r["winner"] == "heart")
        l = sum(1 for r in group_results if r["winner"] == "llm")
        t = sum(1 for r in group_results if r["winner"] == "tie")
        gn = len(group_results)
        print(f"    {group_name:<14s}: Heart {h}/{gn} ({h / gn * 100:.0f}%) | LLM {l}/{gn} ({l / gn * 100:.0f}%) | Tie {t}/{gn}")

    # Show reasoning for a few interesting judgments
    print(f"\n  Sample judge reasoning:")
    for r in results[:5]:
        w = r["winner"].upper()
        print(f"    [{r['time_str']}] {r['agent_id']}: {w} — {r['reasoning'][:80]}")

    print(f"{'═' * 70}")


def run_judging(samples_path: str = "eval_samples.json", output_path: str = "judge_results.json"):
    """Run blind judging on all samples."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    with open(samples_path) as f:
        samples = json.load(f)

    print(f"Judging {len(samples)} samples...")
    random.seed(42)  # reproducible A/B assignment

    results = []
    for i, sample in enumerate(samples):
        is_heart_a = random.random() > 0.5
        print(f"  [{i + 1}/{len(samples)}] {sample['time_str']} / {sample['agent_name']}...", end=" ", flush=True)
        result = judge_sample(client, sample, is_heart_a)
        results.append(result)
        print(f"{result['winner']}")

    # Save raw results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print_results(results)
    print(f"\n  Raw results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="eval_samples.json")
    parser.add_argument("--output", default="judge_results.json")
    args = parser.parse_args()
    run_judging(args.samples, args.output)

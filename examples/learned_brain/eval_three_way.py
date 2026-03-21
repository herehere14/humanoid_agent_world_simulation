#!/usr/bin/env python3
"""Three-way comparison: Plain LLM vs Hardcoded Brain vs Learned Brain.

Same scenarios, same personality, same judge.
Tests whether the learned brain (trained on real conversations)
produces better emotional context than the hardcoded brain.

Usage:
    cd examples
    python -m learned_brain.eval_three_way
    JUDGE_MODEL=gpt-4o python -m learned_brain.eval_three_way
    N_TRIALS=3 python -m learned_brain.eval_three_way
"""

from __future__ import annotations

import json
import os
import sys
import random
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brain_adaptive_prototype import PersonalityProfile
from brain_vs_plain_llm import PlainLLMBaseline, BlindJudge
from brain_enhanced_llm import BrainEnhancedLLM
from brain_rl_evaluation import SCENARIOS
from learned_brain.brain_learned_llm import LearnedBrainLLM


def judge_pair(
    judge: BlindJudge,
    scenario: str,
    personality: PersonalityProfile,
    event_desc: str,
    says: str,
    response_a: str,
    response_b: str,
    conv_summary: str,
    turn: int,
) -> dict:
    """Judge a pair of responses. Returns scores for both."""
    return judge.judge(
        scenario, personality, event_desc, says,
        response_a, response_b, conv_summary, turn,
    )


def run_three_way(personality: PersonalityProfile, scenarios: list[dict]):
    """Run head-to-head-to-head comparison."""

    print(f"\n{'=' * 100}")
    print(f"  THREE-WAY: Plain LLM vs Hardcoded Brain vs Learned Brain")
    print(f"  Personality: {personality.temperament[:70]}")
    print(f"  {len(scenarios)} scenarios, blind LLM judge")
    print(f"{'=' * 100}")

    # Initialize all three systems
    plain = PlainLLMBaseline(personality, scenarios[0]["scenario"])
    hardcoded = BrainEnhancedLLM(personality, scenarios[0]["scenario"])
    learned = LearnedBrainLLM(personality, scenarios[0]["scenario"])

    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    judge = BlindJudge(model=judge_model)
    print(f"  Judge model: {judge_model}")

    # Score tracking
    all_plain = []
    all_hardcoded = []
    all_learned = []
    all_results = []

    for si, scenario_data in enumerate(scenarios):
        sname = scenario_data["name"]
        scenario_text = scenario_data["scenario"]
        turns = scenario_data["turns"]

        plain.reset_for_scenario(scenario_text)
        hardcoded.reset_for_scenario(scenario_text)
        learned.reset_for_scenario(scenario_text)

        print(f"\n  {'─' * 96}")
        print(f"  SCENARIO {si+1}: {sname}")
        print(f"  {'─' * 96}")

        conversation_lines = []

        for ti, turn in enumerate(turns):
            event = turn["event"]
            says = turn["says"]
            event_desc = event.get("description", event.get("type", "unknown"))

            # Generate all three responses
            plain_speech = plain.respond(says, event_desc)
            hc_result = hardcoded.respond(event, says, turn.get("pre_events"))
            hc_speech = hc_result["speech"]
            lr_result = learned.respond(event, says, turn.get("pre_events"))
            lr_speech = lr_result["speech"]

            conv_summary = "\n".join(conversation_lines[-8:]) if conversation_lines else "(start)"

            # Judge: learned vs plain
            j_lp = judge_pair(
                judge, scenario_text, personality, event_desc, says,
                lr_speech, plain_speech, conv_summary, ti + 1,
            )
            # Judge: learned vs hardcoded
            j_lh = judge_pair(
                judge, scenario_text, personality, event_desc, says,
                lr_speech, hc_speech, conv_summary, ti + 1,
            )
            # Judge: hardcoded vs plain
            j_hp = judge_pair(
                judge, scenario_text, personality, event_desc, says,
                hc_speech, plain_speech, conv_summary, ti + 1,
            )

            # Extract scores (brain_scores = first arg, plain_scores = second arg)
            lr_vs_plain = j_lp["brain_scores"].get("total", 15)
            plain_vs_lr = j_lp["plain_scores"].get("total", 15)

            lr_vs_hc = j_lh["brain_scores"].get("total", 15)
            hc_vs_lr = j_lh["plain_scores"].get("total", 15)

            hc_vs_plain = j_hp["brain_scores"].get("total", 15)
            plain_vs_hc = j_hp["plain_scores"].get("total", 15)

            # Average scores across matchups for each system
            # Learned: scored in 2 matchups (vs plain, vs hardcoded)
            learned_avg = (lr_vs_plain + lr_vs_hc) / 2
            # Hardcoded: scored in 2 matchups (vs learned, vs plain)
            hardcoded_avg = (hc_vs_lr + hc_vs_plain) / 2
            # Plain: scored in 2 matchups (vs learned, vs hardcoded)
            plain_avg = (plain_vs_lr + plain_vs_hc) / 2

            all_learned.append(learned_avg)
            all_hardcoded.append(hardcoded_avg)
            all_plain.append(plain_avg)

            all_results.append({
                "scenario": sname, "turn": ti + 1,
                "plain_speech": plain_speech,
                "hardcoded_speech": hc_speech,
                "learned_speech": lr_speech,
                "learned_avg": learned_avg,
                "hardcoded_avg": hardcoded_avg,
                "plain_avg": plain_avg,
                "learned_emotions": lr_result.get("state", {}).get("top_emotions", []),
            })

            # Determine winner
            scores = {"LEARNED": learned_avg, "HARDCODED": hardcoded_avg, "PLAIN": plain_avg}
            winner = max(scores, key=scores.get)
            marks = {"LEARNED": "★", "HARDCODED": "◆", "PLAIN": "○"}

            print(f"\n    Turn {ti+1}: {event['type']}")
            print(f"    Plain:     \"{plain_speech[:80]}{'...' if len(plain_speech) > 80 else ''}\"")
            print(f"    Hardcoded: \"{hc_speech[:80]}{'...' if len(hc_speech) > 80 else ''}\"")
            print(f"    Learned:   \"{lr_speech[:80]}{'...' if len(lr_speech) > 80 else ''}\"")
            print(f"    Score: Plain={plain_avg:.0f}  Hardcoded={hardcoded_avg:.0f}  Learned={learned_avg:.0f}  {marks[winner]} {winner}")

            # Learned brain's emotional read
            if lr_result.get("state", {}).get("top_emotions"):
                top_e = lr_result["state"]["top_emotions"][:3]
                emo_str = ", ".join(f"{e}({p:.0%})" for e, p in top_e)
                print(f"    Learned brain sees: {emo_str}")

            conversation_lines.append(f"  Them: \"{says[:60]}...\"")
            conversation_lines.append(f"  {personality.name}: \"{lr_speech[:60]}...\"")

    # --- Final Summary ---
    print(f"\n\n{'═' * 100}")
    print(f"  FINAL RESULTS")
    print(f"{'═' * 100}")

    n = len(all_plain)
    l_mean = mean(all_learned)
    h_mean = mean(all_hardcoded)
    p_mean = mean(all_plain)

    print(f"\n  Overall ({n} turns, avg score /30):")
    print(f"    {'System':<20s} {'Score':>6s} {'vs Plain':>10s} {'vs Hardcoded':>14s}")
    print(f"    {'─'*20} {'─'*6} {'─'*10} {'─'*14}")
    print(f"    {'Learned Brain':<20s} {l_mean:>5.1f} {l_mean - p_mean:>+9.1f} {l_mean - h_mean:>+13.1f}")
    print(f"    {'Hardcoded Brain':<20s} {h_mean:>5.1f} {h_mean - p_mean:>+9.1f} {'---':>14s}")
    print(f"    {'Plain LLM':<20s} {p_mean:>5.1f} {'---':>10s} {'---':>14s}")

    # Win counts
    l_wins = sum(1 for l, h, p in zip(all_learned, all_hardcoded, all_plain) if l >= h and l >= p)
    h_wins = sum(1 for l, h, p in zip(all_learned, all_hardcoded, all_plain) if h > l and h >= p)
    p_wins = sum(1 for l, h, p in zip(all_learned, all_hardcoded, all_plain) if p > l and p > h)
    print(f"\n  Wins: Learned={l_wins}  Hardcoded={h_wins}  Plain={p_wins}")

    # Improvement percentage
    if p_mean > 0:
        improvement = ((l_mean - p_mean) / p_mean) * 100
        print(f"\n  Learned Brain improvement over Plain: {improvement:+.1f}%")
        print(f"  Target: +20%  {'✓ ACHIEVED' if improvement >= 20 else '✗ not yet'}")

    # Per-scenario
    print(f"\n  Per-scenario:")
    idx = 0
    for sd in scenarios:
        nt = len(sd["turns"])
        ls = all_learned[idx:idx + nt]
        hs = all_hardcoded[idx:idx + nt]
        ps = all_plain[idx:idx + nt]
        print(f"    {sd['name']:<32s} L={mean(ls):.1f}  H={mean(hs):.1f}  P={mean(ps):.1f}")
        idx += nt

    print(f"\n  Judge calls: {judge._calls}")
    print(f"{'═' * 100}")

    return l_mean, h_mean, p_mean


def main():
    fiery = PersonalityProfile(
        name="Alex",
        background="32 years old, 8 years experience, underpaid for 2 years. Tired of being undervalued.",
        temperament="Hot-tempered, direct, takes disrespect personally. Speaks from the gut. Quick to escalate.",
        emotional_tendencies={
            "anger": "quick to flare, expressed openly",
            "patience": "runs out fast",
            "impulse": "high, speaks before thinking",
        },
    )

    n_trials = int(os.environ.get("N_TRIALS", "1"))
    if n_trials > 1:
        all_l, all_h, all_p = [], [], []
        for t in range(n_trials):
            print(f"\n\n{'#' * 100}")
            print(f"  TRIAL {t+1}/{n_trials}")
            print(f"{'#' * 100}")
            l, h, p = run_three_way(fiery, SCENARIOS)
            all_l.append(l)
            all_h.append(h)
            all_p.append(p)

        from statistics import stdev
        print(f"\n\n{'█' * 100}")
        print(f"  AGGREGATE ({n_trials} trials)")
        print(f"{'█' * 100}")
        print(f"  Learned:   {mean(all_l):.1f}/30  (std: {stdev(all_l):.1f})")
        print(f"  Hardcoded: {mean(all_h):.1f}/30  (std: {stdev(all_h):.1f})")
        print(f"  Plain:     {mean(all_p):.1f}/30  (std: {stdev(all_p):.1f})")
        improvement = ((mean(all_l) - mean(all_p)) / mean(all_p)) * 100
        print(f"  Learned vs Plain: {improvement:+.1f}%  {'✓ TARGET' if improvement >= 20 else '✗ not yet'}")
        print(f"{'█' * 100}")
    else:
        run_three_way(fiery, SCENARIOS)


if __name__ == "__main__":
    main()

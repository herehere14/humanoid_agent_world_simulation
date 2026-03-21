#!/usr/bin/env python3
"""Phase 2 Evaluation — Phase2Pipeline vs Plain LLM.

Usage:
    cd examples
    OPENAI_API_KEY=... python -m learned_brain.eval_phase2
    OPENAI_API_KEY=... JUDGE_MODEL=gpt-4o python -m learned_brain.eval_phase2
    OPENAI_API_KEY=... N_TRIALS=3 python -m learned_brain.eval_phase2
"""

from __future__ import annotations

import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brain_adaptive_prototype import PersonalityProfile
from brain_vs_plain_llm import PlainLLMBaseline, BlindJudge
from brain_rl_evaluation import SCENARIOS
from learned_brain.phase2_pipeline import Phase2Pipeline


def run_phase2_eval(personality: PersonalityProfile, scenarios: list[dict]):
    """Run Phase2Pipeline vs PlainLLM, blind judged."""

    print(f"\n{'=' * 100}")
    print(f"  PHASE 2 EVALUATION: Phase2 Pipeline vs Plain LLM")
    print(f"  Personality: {personality.temperament[:70]}")
    print(f"  {len(scenarios)} scenarios, blind LLM judge")
    print(f"{'=' * 100}")

    # Initialize systems
    phase2 = Phase2Pipeline(personality, scenarios[0]["scenario"])
    plain = PlainLLMBaseline(personality, scenarios[0]["scenario"])

    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    judge = BlindJudge(model=judge_model)
    print(f"  Judge model: {judge_model}")
    print(f"  Phase2 config: best-of-{phase2.n_candidates}, policy={'ON' if phase2.use_policy else 'OFF'}, arc={'ON' if phase2.use_arc_planner else 'OFF'}")

    all_phase2 = []
    all_plain = []
    all_results = []

    for si, scenario_data in enumerate(scenarios):
        sname = scenario_data["name"]
        scenario_text = scenario_data["scenario"]
        turns = scenario_data["turns"]

        phase2.reset_for_scenario(scenario_text)
        plain.reset_for_scenario(scenario_text)

        print(f"\n  {'─' * 96}")
        print(f"  SCENARIO {si+1}: {sname}")
        if phase2.arc_plan:
            strategies = [t.get("strategy_bias", "?") for t in phase2.arc_plan]
            print(f"  Arc plan: {' → '.join(strategies)}")
        print(f"  {'─' * 96}")

        conversation_lines = []

        for ti, turn in enumerate(turns):
            event = turn["event"]
            says = turn["says"]
            event_desc = event.get("description", event.get("type", "unknown"))

            # Phase2 system
            p2_result = phase2.respond(event, says, turn.get("pre_events"))
            p2_speech = p2_result["speech"]

            # Plain LLM
            plain_speech = plain.respond(says, event_desc)

            # Blind judge
            conv_summary = "\n".join(conversation_lines[-8:]) if conversation_lines else "(start)"
            judgment = judge.judge(
                scenario_text, personality, event_desc, says,
                p2_speech, plain_speech, conv_summary, ti + 1,
            )

            p2s = judgment["brain_scores"]
            ps = judgment["plain_scores"]
            p2_total = p2s.get("total", sum(p2s.get(k, 0) for k in ["emotional_accuracy", "naturalness", "consistency"]))
            p_total = ps.get("total", sum(ps.get(k, 0) for k in ["emotional_accuracy", "naturalness", "consistency"]))

            all_phase2.append(p2_total)
            all_plain.append(p_total)
            all_results.append({
                "scenario": sname, "turn": ti + 1,
                "strategy": p2_result.get("branch", "?"),
                "phase2_speech": p2_speech,
                "plain_speech": plain_speech,
                "phase2_scores": p2s, "plain_scores": ps,
            })

            winner = "PHASE2" if p2_total > p_total else "PLAIN" if p_total > p2_total else "TIE"
            mark = {"PHASE2": "★", "PLAIN": "○", "TIE": "="}[winner]

            print(f"\n    Turn {ti+1}: {event['type']} | Strategy: {p2_result.get('branch', '?')}")
            print(f"    Phase2: \"{p2_speech[:90]}{'...' if len(p2_speech) > 90 else ''}\"")
            print(f"    Plain:  \"{plain_speech[:90]}{'...' if len(plain_speech) > 90 else ''}\"")
            print(f"    Score: Phase2={p2_total}/30  Plain={p_total}/30  {mark} {winner}")

            # Show emotions
            if p2_result.get("state", {}).get("top_emotions"):
                top_e = p2_result["state"]["top_emotions"][:3]
                emo_str = ", ".join(f"{e}({p:.0%})" for e, p in top_e)
                print(f"    Brain: {emo_str}")

            conversation_lines.append(f"  Them: \"{says[:60]}...\"")
            conversation_lines.append(f"  {personality.name}: \"{p2_speech[:60]}...\"")

    # --- Final Results ---
    print(f"\n\n{'═' * 100}")
    print(f"  FINAL RESULTS")
    print(f"{'═' * 100}")

    n = len(all_phase2)
    p2_mean = mean(all_phase2)
    p_mean = mean(all_plain)

    print(f"\n  Overall ({n} turns):")
    print(f"    Phase2 Pipeline: {p2_mean:.1f}/30  ({p2_mean/30:.0%})")
    print(f"    Plain LLM:       {p_mean:.1f}/30  ({p_mean/30:.0%})")
    diff = p2_mean - p_mean
    print(f"    Difference:      {diff:+.1f} {'(Phase2 wins!)' if diff > 0 else '(Plain wins)' if diff < 0 else '(tie)'}")

    # Improvement percentage
    if p_mean > 0:
        improvement = ((p2_mean - p_mean) / p_mean) * 100
        print(f"    Improvement:     {improvement:+.1f}%  {'✓ TARGET HIT' if improvement >= 20 else ''}")

    # Win counts
    p2_wins = sum(1 for a, b in zip(all_phase2, all_plain) if a > b)
    p_wins = sum(1 for a, b in zip(all_phase2, all_plain) if b > a)
    ties = sum(1 for a, b in zip(all_phase2, all_plain) if a == b)
    print(f"    Phase2 wins: {p2_wins}  |  Plain wins: {p_wins}  |  Ties: {ties}")

    # Per-criterion
    p2_ea = mean(r["phase2_scores"].get("emotional_accuracy", 0) for r in all_results)
    p_ea = mean(r["plain_scores"].get("emotional_accuracy", 0) for r in all_results)
    p2_nat = mean(r["phase2_scores"].get("naturalness", 0) for r in all_results)
    p_nat = mean(r["plain_scores"].get("naturalness", 0) for r in all_results)
    p2_con = mean(r["phase2_scores"].get("consistency", 0) for r in all_results)
    p_con = mean(r["plain_scores"].get("consistency", 0) for r in all_results)

    print(f"\n  {'Criterion':<25s} {'Phase2':>7s} {'Plain':>6s} {'Diff':>7s}")
    print(f"  {'─'*25} {'─'*7} {'─'*6} {'─'*7}")
    print(f"  {'Emotional accuracy':<25s} {p2_ea:>6.1f} {p_ea:>5.1f} {p2_ea - p_ea:>+6.1f}")
    print(f"  {'Naturalness':<25s} {p2_nat:>6.1f} {p_nat:>5.1f} {p2_nat - p_nat:>+6.1f}")
    print(f"  {'Consistency':<25s} {p2_con:>6.1f} {p_con:>5.1f} {p2_con - p_con:>+6.1f}")

    # Per-scenario
    print(f"\n  Per-scenario:")
    idx = 0
    for sd in scenarios:
        nt = len(sd["turns"])
        p2s = all_phase2[idx:idx + nt]
        ps = all_plain[idx:idx + nt]
        p2w = sum(1 for a, b in zip(p2s, ps) if a > b)
        pw = sum(1 for a, b in zip(p2s, ps) if b > a)
        print(f"    {sd['name']:<32s} P2={mean(p2s):.1f}  Plain={mean(ps):.1f}  (P2:{p2w} P:{pw})")
        idx += nt

    print(f"\n  Judge calls: {judge._calls}")
    print(f"{'═' * 100}")

    return p2_mean, p_mean


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
        all_p2, all_p = [], []
        for t in range(n_trials):
            print(f"\n\n{'#' * 100}")
            print(f"  TRIAL {t+1}/{n_trials}")
            print(f"{'#' * 100}")
            p2, p = run_phase2_eval(fiery, SCENARIOS)
            all_p2.append(p2)
            all_p.append(p)

        from statistics import stdev
        print(f"\n\n{'█' * 100}")
        print(f"  AGGREGATE ({n_trials} trials)")
        print(f"{'█' * 100}")
        print(f"  Phase2: {mean(all_p2):.1f}/30  (std: {stdev(all_p2):.1f})")
        print(f"  Plain:  {mean(all_p):.1f}/30  (std: {stdev(all_p):.1f})")
        improvement = ((mean(all_p2) - mean(all_p)) / mean(all_p)) * 100
        print(f"  Improvement: {improvement:+.1f}%  {'✓ TARGET' if improvement >= 20 else '✗ not yet'}")
        print(f"{'█' * 100}")
    else:
        run_phase2_eval(fiery, SCENARIOS)


if __name__ == "__main__":
    main()

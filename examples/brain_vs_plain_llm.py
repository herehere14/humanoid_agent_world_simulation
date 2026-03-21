#!/usr/bin/env python3
"""Head-to-head: Adaptive Brain+PF system vs Plain LLM.

Same scenario, same personality description, same turns.
  A) Our system: Brain → LLM deltas → State Machine → Stable Branches → Speech
  B) Plain LLM: One good system prompt + conversation history → Speech

An LLM judge (gpt-4o) scores BOTH outputs blindly on:
  1. Emotional accuracy (does the speech match what the person SHOULD feel?)
  2. Naturalness (does it sound like a real human?)
  3. Consistency (does emotional tone carry across turns?)

The judge doesn't know which is system A or B — outputs are shuffled.
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openai


def _token_limit_kwargs(model: str, limit: int) -> dict:
    """Return the right max tokens kwarg for the model."""
    if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}

from brain_adaptive_prototype import (
    AdaptiveBrainEngine,
    PersonalityProfile,
    build_emotional_context,
)
from brain_rl_evaluation import (
    RLPipeline,
    _create_stable_branches,
    SCENARIOS,
)


# ---------------------------------------------------------------------------
# Plain LLM baseline (single call, good prompt, no brain)
# ---------------------------------------------------------------------------

class PlainLLMBaseline:
    """Just a good system prompt + conversation history. No brain, no forest."""

    def __init__(self, personality: PersonalityProfile, scenario: str):
        self.personality = personality
        self.scenario = scenario
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []
        self.total_tokens = 0

    def reset_for_scenario(self, scenario: str):
        self.scenario = scenario
        self.conversation_history = []

    def _system_prompt(self) -> str:
        return f"""You are {self.personality.name}, a real human being in the following situation:

{self.scenario}

About you:
{self.personality.background}
Temperament: {self.personality.temperament}

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- Respond naturally. Keep responses 1-4 sentences.
- Your emotional reactions should be realistic and consistent.
- Let your emotions BUILD across the conversation — don't reset each turn.
- If something makes you angry, STAY angry until something changes.
- If you're frustrated, show it. If you're happy, show it.
- React to what's said AND to the accumulated history of the conversation.
- Your personality is: {self.personality.temperament}
- Never say "I understand your perspective" or other AI-speak."""

    def respond(self, other_says: str, event_description: str) -> str:
        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_says})

        # Plain LLM always uses base model (not fine-tuned)
        plain_model = os.environ.get("PLAIN_MODEL", "gpt-4o-mini")
        try:
            response = self._client.chat.completions.create(
                model=plain_model,
                messages=messages,
                temperature=0.9,
                **_token_limit_kwargs(plain_model, 150),
            )
            speech = response.choices[0].message.content.strip().strip('"')
            # Strip name prefix
            if speech.lower().startswith(self.personality.name.lower() + ":"):
                speech = speech[len(self.personality.name) + 1:].strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0
        except Exception as e:
            speech = f"[Error: {e}]"

        self.conversation_history.append({"role": "user", "content": other_says})
        self.conversation_history.append({"role": "assistant", "content": speech})
        return speech


# ---------------------------------------------------------------------------
# Blind LLM Judge (doesn't know which is A or B)
# ---------------------------------------------------------------------------

_BLIND_JUDGE_PROMPT = """You are an expert judge evaluating two roleplay responses. You do NOT know
which system produced which response. Judge ONLY on quality.

Given the scenario, the character's personality, what just happened (event), and two responses
(labeled Response 1 and Response 2), score EACH response on three criteria:

1. EMOTIONAL_ACCURACY (0-10): Does the response reflect how this specific person would actually
   feel given the event and conversation history? Consider their temperament and the accumulated
   emotional weight of the conversation.

2. NATURALNESS (0-10): Does it sound like a real human? Not too polished, not too AI-like.
   Real humans use fragments, repeat themselves, trail off, use filler words.

3. CONSISTENCY (0-10): Does the emotional tone make sense given what came before? If the person
   was angry last turn, did that carry forward appropriately?

Output JSON:
{
  "response_1": {"emotional_accuracy": int, "naturalness": int, "consistency": int, "total": int, "note": "one sentence"},
  "response_2": {"emotional_accuracy": int, "naturalness": int, "consistency": int, "total": int, "note": "one sentence"}
}

Be STRICT. Most responses should score 4-7. Only exceptional responses get 8+. Robotic or
AI-sounding responses should score 3 or below on naturalness."""


class BlindJudge:
    """Judges two responses without knowing which system produced which."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._calls = 0

    def judge(
        self,
        scenario: str,
        personality: PersonalityProfile,
        event_desc: str,
        other_says: str,
        response_a: str,
        response_b: str,
        conversation_summary: str,
        turn_num: int,
    ) -> dict:
        """Judge two responses blindly. Randomly assigns to position 1 or 2.

        Returns dict with 'brain_scores', 'plain_scores', 'brain_position'.
        """
        # Randomly assign positions to prevent position bias
        brain_is_first = random.random() < 0.5
        if brain_is_first:
            r1, r2 = response_a, response_b
        else:
            r1, r2 = response_b, response_a

        user_prompt = f"""SCENARIO: {scenario}

CHARACTER: {personality.name}
Background: {personality.background}
Temperament: {personality.temperament}

CONVERSATION SO FAR (turn {turn_num}):
{conversation_summary}

EVENT THAT JUST HAPPENED: {event_desc}
Other person said: "{other_says}"

RESPONSE 1: "{r1}"

RESPONSE 2: "{r2}"

Score each response."""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _BLIND_JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                **_token_limit_kwargs(self._model, 300),
            )
            self._calls += 1
            data = json.loads(response.choices[0].message.content.strip())

            r1_scores = data.get("response_1", {})
            r2_scores = data.get("response_2", {})

            # Map back to brain/plain
            if brain_is_first:
                brain_scores = r1_scores
                plain_scores = r2_scores
            else:
                brain_scores = r2_scores
                plain_scores = r1_scores

            return {
                "brain_scores": brain_scores,
                "plain_scores": plain_scores,
                "brain_position": 1 if brain_is_first else 2,
            }

        except Exception as e:
            return {
                "brain_scores": {"emotional_accuracy": 5, "naturalness": 5, "consistency": 5, "total": 15},
                "plain_scores": {"emotional_accuracy": 5, "naturalness": 5, "consistency": 5, "total": 15},
                "brain_position": 1,
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(personality: PersonalityProfile, scenarios: list[dict]):
    """Run head-to-head comparison across scenarios."""

    print(f"\n{'=' * 90}")
    print(f"  HEAD-TO-HEAD: Brain+PF System vs Plain LLM")
    print(f"  Personality: {personality.temperament[:70]}")
    print(f"  {len(scenarios)} scenarios, blind LLM judge")
    print(f"{'=' * 90}")

    # Systems
    brain_pipeline = RLPipeline(
        personality=personality,
        scenario=scenarios[0]["scenario"],
        use_learning=False,  # fair comparison — no RL advantage
        epsilon=0.0,
    )
    plain_llm = PlainLLMBaseline(personality, scenarios[0]["scenario"])
    judge = BlindJudge()

    all_brain_totals = []
    all_plain_totals = []
    all_results = []

    for si, scenario_data in enumerate(scenarios):
        sname = scenario_data["name"]
        scenario_text = scenario_data["scenario"]
        turns = scenario_data["turns"]

        brain_pipeline.reset_for_scenario(scenario_text)
        plain_llm.reset_for_scenario(scenario_text)

        print(f"\n  {'─' * 86}")
        print(f"  SCENARIO {si+1}: {sname}")
        print(f"  {'─' * 86}")

        conversation_lines = []

        for ti, turn in enumerate(turns):
            event = turn["event"]
            says = turn["says"]
            event_desc = event.get("description", event.get("type", "unknown"))

            # Brain system
            brain_result = brain_pipeline.process_turn(
                event, says, turn.get("pre_events"),
            )
            brain_speech = brain_result["speech"]

            # Plain LLM
            plain_speech = plain_llm.respond(says, event_desc)

            # Build conversation summary for judge
            conv_summary = "\n".join(conversation_lines[-8:]) if conversation_lines else "(conversation start)"

            # Blind judge
            judgment = judge.judge(
                scenario_text, personality, event_desc, says,
                brain_speech, plain_speech, conv_summary, ti + 1,
            )

            bs = judgment["brain_scores"]
            ps = judgment["plain_scores"]
            brain_total = bs.get("total", sum(bs.get(k, 0) for k in ["emotional_accuracy", "naturalness", "consistency"]))
            plain_total = ps.get("total", sum(ps.get(k, 0) for k in ["emotional_accuracy", "naturalness", "consistency"]))

            all_brain_totals.append(brain_total)
            all_plain_totals.append(plain_total)
            all_results.append({
                "scenario": sname, "turn": ti + 1,
                "regime": brain_result["regime"],
                "brain_speech": brain_speech,
                "plain_speech": plain_speech,
                "brain_scores": bs, "plain_scores": ps,
            })

            winner = "BRAIN" if brain_total > plain_total else "PLAIN" if plain_total > brain_total else "TIE"
            winner_mark = {"BRAIN": "◀", "PLAIN": "▶", "TIE": "="}[winner]

            print(f"\n    Turn {ti+1}: {event['type']} | Regime: {brain_result['regime']}")
            print(f"    Brain [{brain_result['branch']:<20s}]: \"{brain_speech[:80]}{'...' if len(brain_speech) > 80 else ''}\"")
            print(f"    Plain:                     \"{plain_speech[:80]}{'...' if len(plain_speech) > 80 else ''}\"")
            print(f"    Score: Brain={brain_total}/30  Plain={plain_total}/30  {winner_mark} {winner}")
            if bs.get("note"):
                print(f"      Brain note: {bs['note']}")
            if ps.get("note"):
                print(f"      Plain note: {ps['note']}")

            # Update conversation context
            conversation_lines.append(f"  Them: \"{says[:60]}...\"")
            conversation_lines.append(f"  {personality.name} (brain): \"{brain_speech[:60]}...\"")
            conversation_lines.append(f"  {personality.name} (plain): \"{plain_speech[:60]}...\"")

    # --- Final Summary ---
    print(f"\n\n{'═' * 90}")
    print(f"  FINAL RESULTS")
    print(f"{'═' * 90}")

    n = len(all_brain_totals)
    brain_mean = mean(all_brain_totals)
    plain_mean = mean(all_plain_totals)

    print(f"\n  Overall ({n} turns):")
    print(f"    Brain system: {brain_mean:.1f}/30 avg")
    print(f"    Plain LLM:    {plain_mean:.1f}/30 avg")
    diff = brain_mean - plain_mean
    print(f"    Difference:   {diff:+.1f} {'(brain wins)' if diff > 0 else '(plain wins)' if diff < 0 else '(tie)'}")

    # Win/loss/tie counts
    brain_wins = sum(1 for b, p in zip(all_brain_totals, all_plain_totals) if b > p)
    plain_wins = sum(1 for b, p in zip(all_brain_totals, all_plain_totals) if p > b)
    ties = sum(1 for b, p in zip(all_brain_totals, all_plain_totals) if b == p)
    print(f"    Brain wins: {brain_wins}  |  Plain wins: {plain_wins}  |  Ties: {ties}")

    # Per-criterion breakdown
    brain_ea = mean(r["brain_scores"].get("emotional_accuracy", 0) for r in all_results)
    plain_ea = mean(r["plain_scores"].get("emotional_accuracy", 0) for r in all_results)
    brain_nat = mean(r["brain_scores"].get("naturalness", 0) for r in all_results)
    plain_nat = mean(r["plain_scores"].get("naturalness", 0) for r in all_results)
    brain_con = mean(r["brain_scores"].get("consistency", 0) for r in all_results)
    plain_con = mean(r["plain_scores"].get("consistency", 0) for r in all_results)

    print(f"\n  Per-criterion (0-10 scale):")
    print(f"    {'Criterion':<25s} {'Brain':>6s} {'Plain':>6s} {'Diff':>7s}")
    print(f"    {'─'*25} {'─'*6} {'─'*6} {'─'*7}")
    print(f"    {'Emotional accuracy':<25s} {brain_ea:>5.1f} {plain_ea:>5.1f} {brain_ea - plain_ea:>+6.1f}")
    print(f"    {'Naturalness':<25s} {brain_nat:>5.1f} {plain_nat:>5.1f} {brain_nat - plain_nat:>+6.1f}")
    print(f"    {'Consistency':<25s} {brain_con:>5.1f} {plain_con:>5.1f} {brain_con - plain_con:>+6.1f}")

    # Per-scenario
    print(f"\n  Per-scenario:")
    idx = 0
    for scenario_data in scenarios:
        n_turns = len(scenario_data["turns"])
        bs = all_brain_totals[idx:idx + n_turns]
        ps = all_plain_totals[idx:idx + n_turns]
        b_wins = sum(1 for b, p in zip(bs, ps) if b > p)
        p_wins = sum(1 for b, p in zip(bs, ps) if p > b)
        print(f"    {scenario_data['name']:<32s} Brain={mean(bs):.1f}  Plain={mean(ps):.1f}  "
              f"(B:{b_wins} P:{p_wins})")
        idx += n_turns

    print(f"\n  Judge calls: {judge._calls}")
    print(f"\n{'═' * 90}")


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

    run_comparison(fiery, SCENARIOS)


if __name__ == "__main__":
    main()

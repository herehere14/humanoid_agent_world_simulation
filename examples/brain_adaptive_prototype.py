#!/usr/bin/env python3
"""Adaptive Brain + Prompt Forest + LLM — No Hardcoded Emotions.

Key changes from brain_forest_prototype.py:
  1. Brain's _event_to_deltas() replaced by LLM call that interprets events
     given personality + context + current state → 16 emotional deltas
  2. Prompt branches generated dynamically from emotional state instead of
     9 fixed templates
  3. Brain state machine (momentum, decay, cross-interactions, regime detection)
     stays as deterministic math
  4. Prompt Forest framework (router, evaluator, optimizer) stays

Architecture:
  [Event] → [LLM #1: interpret event → deltas] → [Brain state machine] → [Regime]
                                                          ↓
                                              [LLM #2: generate branch template]
                                                          ↓
                                              [Prompt Forest Router]
                                                          ↓
                                              [LLM #3: generate speech]
                                                          ↓
                                              [Evaluator → Optimizer]
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openai

from src.prompt_forest.branches.base import PromptBranch
from src.prompt_forest.types import BranchState, BranchStatus, TaskInput
from src.prompt_forest.core.executor import PromptExecutor
from src.prompt_forest.backend.openai_chat import OpenAIChatBackend

from brain_llm_prototype import (
    BrainEngine,
    BrainState,
    _clamp,
    _BASELINE_DICT,
    _OPPOSING_PAIRS,
)


# ---------------------------------------------------------------------------
# Personality Profile
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProfile:
    """Defines how a character reacts emotionally and behaves."""
    name: str
    background: str
    temperament: str
    emotional_tendencies: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM-based Delta Generator (replaces hardcoded _event_to_deltas)
# ---------------------------------------------------------------------------

_EMOTIONAL_VARS = [
    "confidence", "stress", "frustration", "trust", "anger", "hope",
    "curiosity", "fear", "motivation", "patience", "impulse", "empathy",
    "caution", "fatigue", "reflection",
]

_DELTA_SYSTEM_PROMPT = """You are an emotional dynamics engine. Given a character's personality,
current emotional state, and an event that just happened, output the emotional CHANGES (deltas)
that this specific person would experience.

Rules:
- Output a JSON object with exactly these 16 keys: confidence, stress, frustration, trust, anger,
  hope, curiosity, fear, motivation, patience, impulse, empathy, caution, fatigue, reflection, reasoning
- Each delta is a float in [-0.4, 0.4]. Positive = emotion increases, negative = decreases.
- Most deltas should be small (-0.15 to 0.15). Only extreme events push beyond that.
- Consider the CHARACTER'S PERSONALITY — a patient person has smaller anger deltas, an anxious
  person has larger fear deltas, a confident person shrugs off insults.
- Consider the CURRENT STATE — if someone is already very angry (0.8), an additional insult
  might not increase anger much more, but could decrease patience sharply.
- The "reasoning" field should be 1-2 sentences explaining your logic.
- Be psychologically realistic. Real humans don't change all 15 emotions on every event.
  Most events affect 3-6 variables. Set the rest to 0.0."""


def _generate_deltas_via_llm(
    client: openai.OpenAI,
    event: dict,
    current_state: dict[str, float],
    personality: PersonalityProfile,
    scenario: str,
    regime: str,
) -> tuple[dict[str, float], str]:
    """Call LLM to interpret event → emotional deltas.

    Returns (deltas_dict, reasoning_string).
    """
    state_summary = ", ".join(f"{k}={v:.0%}" for k, v in current_state.items()
                              if k != "emotional_momentum")

    tendencies_text = ""
    if personality.emotional_tendencies:
        tendencies_text = "\nEmotional tendencies: " + ", ".join(
            f"{k}: {v}" for k, v in personality.emotional_tendencies.items()
        )

    user_prompt = f"""CHARACTER: {personality.name}
Background: {personality.background}
Temperament: {personality.temperament}{tendencies_text}

CURRENT EMOTIONAL STATE: {state_summary}
Current regime: {regime}

SCENARIO: {scenario}

EVENT THAT JUST HAPPENED:
  Type: {event.get('type', 'unknown')}
  Intensity: {event.get('intensity', 0.5)}
  Description: {event.get('description', event.get('type', 'unknown event'))}

Output the emotional deltas as JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _DELTA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        deltas = {}
        for var in _EMOTIONAL_VARS:
            val = float(data.get(var, 0.0))
            deltas[var] = max(-0.4, min(0.4, val))

        reasoning = data.get("reasoning", "")
        return deltas, reasoning

    except Exception as e:
        # Fallback: return zero deltas rather than crash
        return {var: 0.0 for var in _EMOTIONAL_VARS}, f"LLM delta call failed: {e}"


# ---------------------------------------------------------------------------
# Adaptive Brain Engine
# ---------------------------------------------------------------------------

class AdaptiveBrainEngine(BrainEngine):
    """Brain engine that uses LLM to interpret events instead of hardcoded deltas.

    Inherits all state machine math: momentum, decay, cross-interactions,
    regime detection, conflict detection.
    """

    def __init__(
        self,
        personality: PersonalityProfile,
        scenario: str,
        momentum: float = 0.30,
        decay_rate: float = 0.02,
    ):
        super().__init__(momentum=momentum, decay_rate=decay_rate)
        self.personality = personality
        self.scenario = scenario
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._delta_calls: list[dict] = []

    def _event_to_deltas(self, event: dict) -> dict[str, float]:
        """Override: use LLM instead of hardcoded lookup table."""
        current_state = self.state.as_dict()
        regime = self._detect_regime()

        started = time.perf_counter()
        deltas, reasoning = _generate_deltas_via_llm(
            self._client, event, current_state,
            self.personality, self.scenario, regime,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self._delta_calls.append({
            "turn": self.turn,
            "event": event,
            "deltas": deltas,
            "reasoning": reasoning,
            "latency_ms": round(elapsed_ms, 1),
        })

        return deltas

    def process_event(self, event: dict) -> dict:
        """Process event — same as parent but adds delta reasoning to result."""
        result = super().process_event(event)
        # Attach the LLM's reasoning for this event
        if self._delta_calls:
            last = self._delta_calls[-1]
            result["delta_reasoning"] = last["reasoning"]
            result["delta_latency_ms"] = last["latency_ms"]
        return result

    def delta_call_summary(self) -> dict:
        """Summary of delta generation LLM calls."""
        if not self._delta_calls:
            return {"calls": 0, "total_ms": 0, "avg_ms": 0}
        total_ms = sum(c["latency_ms"] for c in self._delta_calls)
        return {
            "calls": len(self._delta_calls),
            "total_ms": round(total_ms, 1),
            "avg_ms": round(total_ms / len(self._delta_calls), 1),
        }


# ---------------------------------------------------------------------------
# Dynamic Branch Generator (replaces 9 fixed templates)
# ---------------------------------------------------------------------------

_BRANCH_GEN_SYSTEM_PROMPT = """You are a prompt engineer for emotional roleplay. Given a character's
personality, emotional state, and regime, generate a behavioral prompt template.

The template MUST contain exactly these two placeholders: {context} and {task}
The template should be a roleplay instruction that tells an LLM how to speak as this character
given their current emotional state.

Include:
1. A brief instruction framing the roleplay
2. 3-5 SPECIFIC behavioral rules for how the character speaks RIGHT NOW given their emotions
3. 2-3 example sentences showing the exact tone and style
4. A sentence length constraint (e.g. "1-3 sentences max")
5. End with: "Now respond to this as the character. Stay in character:\n\n{task}"

The template should start with:
"You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:\n\n{context}\n\n"

Output ONLY the template text. No explanation, no markdown formatting."""


class DynamicBranchGenerator:
    """Generates prompt branches dynamically from emotional state."""

    def __init__(self, personality: PersonalityProfile, scenario: str):
        self.personality = personality
        self.scenario = scenario
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._cache: dict[str, PromptBranch] = {}
        self._gen_calls = 0
        self._gen_time_ms = 0.0

    def _cache_key(self, regime: str, state: dict[str, float]) -> str:
        """Bucket emotional values to 0.1 to avoid excessive regeneration."""
        top_vars = ["anger", "trust", "stress", "frustration", "patience", "hope",
                     "confidence", "fatigue"]
        bucketed = {v: round(state.get(v, 0.5) * 10) / 10 for v in top_vars}
        parts = [regime] + [f"{k}{v:.1f}" for k, v in sorted(bucketed.items())]
        return "|".join(parts)

    def generate(self, brain_result: dict) -> list[tuple[str, PromptBranch, float]]:
        """Generate ranked branch candidates for current state.

        Returns list of (name, branch, score) tuples.
        """
        regime = brain_result["regime"]
        state = brain_result["state"]
        key = self._cache_key(regime, state)

        if key in self._cache:
            primary = self._cache[key]
        else:
            primary = self._generate_branch(regime, state, brain_result)
            self._cache[key] = primary

        # Primary candidate gets score 1.0
        candidates = [(regime, primary, 1.0)]

        # Add best cached branch from a different regime as fallback
        for cached_key, cached_branch in self._cache.items():
            cached_regime = cached_key.split("|")[0]
            if cached_regime != regime:
                candidates.append((cached_regime, cached_branch, 0.3))
                break

        # If no fallback from cache, generate a neutral fallback
        if len(candidates) < 2:
            fallback = self._fallback_branch()
            candidates.append(("fallback_neutral", fallback, 0.2))

        return candidates[:3]

    def _generate_branch(self, regime: str, state: dict[str, float],
                         brain_result: dict) -> PromptBranch:
        """Call LLM to generate a branch template for this emotional state."""
        state_summary = ", ".join(f"{k}={v:.0%}" for k, v in state.items()
                                   if k != "emotional_momentum")
        conflicts = brain_result.get("conflicts", [])
        conflict_text = ""
        if conflicts:
            conflict_text = "\nInternal conflicts: " + ", ".join(
                f"{c['drives'][0]} vs {c['drives'][1]}" for c in conflicts
            )

        tendencies_text = ""
        if self.personality.emotional_tendencies:
            tendencies_text = "\nEmotional tendencies: " + ", ".join(
                f"{k}: {v}" for k, v in self.personality.emotional_tendencies.items()
            )

        user_prompt = f"""CHARACTER: {self.personality.name}
Temperament: {self.personality.temperament}{tendencies_text}

CURRENT EMOTIONAL STATE: {state_summary}
Regime: {regime}{conflict_text}

Generate the behavioral prompt template for this character in this emotional state."""

        try:
            started = time.perf_counter()
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _BRANCH_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=400,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._gen_calls += 1
            self._gen_time_ms += elapsed_ms

            template = response.choices[0].message.content.strip()

            # Validate template has required placeholders
            if "{context}" not in template or "{task}" not in template:
                template = self._inject_placeholders(template)

            return PromptBranch(BranchState(
                name=regime,
                purpose=f"Dynamic branch for {regime} state",
                prompt_template=template,
                weight=1.0,
                status=BranchStatus.ACTIVE,
            ))

        except Exception:
            return self._fallback_branch()

    def _inject_placeholders(self, template: str) -> str:
        """Ensure template has {context} and {task} placeholders."""
        if "{context}" not in template:
            template = ("You are roleplaying as a real person. Here is the situation "
                       "and your current emotional state:\n\n{context}\n\n" + template)
        if "{task}" not in template:
            template += "\n\nNow respond to this as the character. Stay in character:\n\n{task}"
        return template

    def _fallback_branch(self) -> PromptBranch:
        """Emergency fallback branch if LLM generation fails."""
        return PromptBranch(BranchState(
            name="fallback_neutral",
            purpose="Fallback branch for when generation fails",
            prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

Respond naturally as this character given their emotional state. Stay in character. 1-3 sentences max.

{task}""",
            weight=0.5,
            status=BranchStatus.ACTIVE,
        ))

    def gen_call_summary(self) -> dict:
        return {
            "calls": self._gen_calls,
            "cache_size": len(self._cache),
            "total_ms": round(self._gen_time_ms, 1),
            "avg_ms": round(self._gen_time_ms / max(1, self._gen_calls), 1),
        }


# ---------------------------------------------------------------------------
# Emotional Context Builder
# ---------------------------------------------------------------------------

def build_emotional_context(
    brain_result: dict,
    scenario: str,
    personality: PersonalityProfile,
) -> str:
    """Build the {context} string for branch templates."""
    state = brain_result["state"]
    regime = brain_result["regime"]
    conflicts = brain_result.get("conflicts", [])

    conflict_lines = ""
    if conflicts:
        conflict_lines = "\nInternal conflicts:\n" + "\n".join(
            f"  - {c['drives'][0]} ({c['values'][0]:.0%}) vs {c['drives'][1]} ({c['values'][1]:.0%})"
            for c in conflicts
        )

    return f"""SCENARIO: {scenario}

CHARACTER: {personality.name}
{personality.background}
Temperament: {personality.temperament}

CURRENT INTERNAL STATE:
  Regime: {regime}
  Confidence: {state['confidence']:.0%} | Stress: {state['stress']:.0%}
  Trust: {state['trust']:.0%} | Patience: {state['patience']:.0%}
  Anger: {state['anger']:.0%} | Hope: {state['hope']:.0%}
  Frustration: {state['frustration']:.0%} | Fatigue: {state['fatigue']:.0%}
  Motivation: {state['motivation']:.0%} | Emotional momentum: {state['emotional_momentum']:.0%}
{conflict_lines}

IMPORTANT: You are {personality.name}, a real person. Never break character. Never say "I'm an AI"."""


# ---------------------------------------------------------------------------
# Evaluator (reused from brain_forest_prototype, enhanced)
# ---------------------------------------------------------------------------

class EmotionalCoherenceJudge:
    """Scores how well the LLM output matches the intended emotional state."""

    def score(self, output: str, brain_result: dict, branch_name: str) -> float:
        state = brain_result["state"]
        regime = brain_result["regime"]
        words = output.lower().split()
        n_words = len(words)

        score = 0.5

        # Length appropriateness
        if regime in ("defeated_resigned", "angry_reactive"):
            if n_words <= 25:
                score += 0.15
            elif n_words > 50:
                score -= 0.15
        elif regime in ("confident_assertive", "optimistic_engaged"):
            if 15 <= n_words <= 60:
                score += 0.10

        # Emotional markers
        polite_words = {"appreciate", "understand", "thank", "thanks", "please", "sorry"}
        aggressive_words = {"no", "enough", "can't", "won't", "unacceptable", "demand",
                           "ridiculous", "seriously", "done", "fix", "now", "immediately"}
        defeated_words = {"fine", "whatever", "sure", "okay", "just", "guess"}

        polite_count = sum(1 for w in words if w.strip(".,!?\"'") in polite_words)
        aggressive_count = sum(1 for w in words if w.strip(".,!?\"'") in aggressive_words)
        defeated_count = sum(1 for w in words if w.strip(".,!?\"'") in defeated_words)

        caps_words = sum(1 for w in output.split() if w.isupper() and len(w) > 1)

        if state.get("anger", 0) > 0.4:
            if polite_count >= 3:
                score -= 0.15
            if aggressive_count >= 1:
                score += 0.15
            if caps_words >= 1:
                score += 0.10

        if state.get("trust", 1.0) < 0.3:
            if polite_count >= 3:
                score -= 0.10

        if regime == "defeated_resigned":
            if defeated_count >= 1:
                score += 0.15
            if "!" in output:
                score -= 0.10

        if regime in ("optimistic_engaged", "warm_collaborative"):
            if polite_count >= 1:
                score += 0.10

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Branch Weight Optimizer (reused)
# ---------------------------------------------------------------------------

class BranchWeightOptimizer:
    """Adapts branch weights based on coherence scores."""

    def __init__(self, lr: float = 0.15, min_weight: float = 0.3, max_weight: float = 3.0):
        self.lr = lr
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._regime_history: dict[str, list[tuple[str, float]]] = {}

    def update(self, selected_branch: str, score: float, regime: str,
               branch: PromptBranch) -> dict[str, float]:
        if regime not in self._regime_history:
            self._regime_history[regime] = []
        self._regime_history[regime].append((selected_branch, score))

        history = self._regime_history[regime]
        baseline = mean(s for _, s in history[-20:]) if len(history) >= 3 else 0.5

        advantage = score - baseline
        old_weight = branch.state.weight
        new_weight = old_weight + self.lr * advantage
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        branch.state.weight = new_weight
        branch.apply_reward(score)

        return {selected_branch: new_weight}


# ---------------------------------------------------------------------------
# Full Adaptive Pipeline
# ---------------------------------------------------------------------------

class AdaptiveBrainForestPipeline:
    """Complete Adaptive Brain → Dynamic Prompt Forest → LLM pipeline."""

    def __init__(
        self,
        scenario: str,
        personality: PersonalityProfile,
        momentum: float = 0.30,
        decay_rate: float = 0.02,
    ):
        self.scenario = scenario
        self.personality = personality

        # Adaptive brain (LLM-driven deltas)
        self.brain = AdaptiveBrainEngine(
            personality=personality,
            scenario=scenario,
            momentum=momentum,
            decay_rate=decay_rate,
        )

        # Dynamic branch generator
        self.branch_gen = DynamicBranchGenerator(personality, scenario)

        # Executor (uses real Prompt Forest executor + OpenAI backend)
        self.backend = OpenAIChatBackend(
            model="gpt-4o-mini",
            temperature=0.9,
            max_output_tokens=150,
            system_prompt="You are a character in a roleplay scenario. Follow the prompt exactly.",
        )
        self.executor = PromptExecutor(self.backend)

        # Evaluator + Optimizer
        self.judge = EmotionalCoherenceJudge()
        self.optimizer = BranchWeightOptimizer()

        # Conversation history
        self.conversation_history: list[dict] = []

    def process_turn(
        self,
        event: dict,
        other_person_says: str,
        pre_events: list[dict] | None = None,
    ) -> dict:
        """Process one turn through the full adaptive pipeline."""
        # 1. Pre-events
        for pre in (pre_events or []):
            self.brain.process_event(pre)

        # 2. Brain processes main event (LLM-driven deltas)
        brain_result = self.brain.process_event(event)

        # 3. Build emotional context
        context = build_emotional_context(brain_result, self.scenario, self.personality)

        # 4. Dynamic branch generator produces candidates
        candidates = self.branch_gen.generate(brain_result)

        # 5. Execute top candidate
        history_text = ""
        for msg in self.conversation_history[-6:]:
            role = "Them" if msg["role"] == "user" else self.personality.name
            history_text += f"\n{role}: \"{msg['content']}\""
        task_text = f"{history_text}\nThem: \"{other_person_says}\""

        task = TaskInput(
            task_id=f"turn_{brain_result['turn']}",
            text=task_text.strip(),
            task_type="roleplay",
            metadata={"brain_regime": brain_result["regime"]},
        )

        selected_name, selected_branch, selected_score = candidates[0]
        branch_output = self.executor.run_branch(
            selected_branch, task,
            task_type="roleplay",
            context=context,
        )
        speech = branch_output.output.strip().strip('"')
        # Strip character name prefix
        if speech.lower().startswith(self.personality.name.lower() + ":"):
            speech = speech[len(self.personality.name) + 1:].strip().strip('"')

        # 6. Judge coherence
        score = self.judge.score(speech, brain_result, selected_name)

        # 7. If score too low and we have fallback, try it
        if score < 0.4 and len(candidates) > 1:
            alt_name, alt_branch, _ = candidates[1]
            alt_output = self.executor.run_branch(
                alt_branch, task,
                task_type="roleplay",
                context=context,
            )
            alt_speech = alt_output.output.strip().strip('"')
            if alt_speech.lower().startswith(self.personality.name.lower() + ":"):
                alt_speech = alt_speech[len(self.personality.name) + 1:].strip().strip('"')
            alt_score = self.judge.score(alt_speech, brain_result, alt_name)
            if alt_score > score:
                selected_name = alt_name
                selected_branch = alt_branch
                speech = alt_speech
                score = alt_score

        # 8. Optimizer
        weight_updates = self.optimizer.update(selected_name, score,
                                                brain_result["regime"], selected_branch)

        # 9. Update conversation history
        self.conversation_history.append({"role": "user", "content": other_person_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        return {
            "turn": brain_result["turn"],
            "event": event,
            "regime": brain_result["regime"],
            "state": brain_result["state"],
            "conflicts": brain_result.get("conflicts", []),
            "delta_reasoning": brain_result.get("delta_reasoning", ""),
            "selected_branch": selected_name,
            "speech": speech,
            "coherence_score": score,
            "weight_updates": weight_updates,
            "candidates": [(n, s) for n, _, s in candidates],
        }


# ---------------------------------------------------------------------------
# Demo: Same negotiation, TWO different personalities
# ---------------------------------------------------------------------------

NEGOTIATION_SCENARIO = (
    "You are in a salary negotiation for a senior software engineer position at a tech company. "
    "You currently earn $140k and believe you deserve $180k based on your experience and market rates. "
    "The hiring manager is sitting across from you. This job would be a significant career step."
)

TURNS = [
    {
        "event": {"type": "opportunity", "intensity": 0.7,
                  "description": "The negotiation opens. You're excited about the opportunity."},
        "says": "Thanks for coming in, Alex. We're excited about your candidacy. Let's talk compensation. What are you looking for?",
        "narration": "The negotiation opens.",
    },
    {
        "event": {"type": "positive_outcome", "intensity": 0.4,
                  "description": "The manager seems open to discussing your number. Small win."},
        "says": "I appreciate you being upfront. $180k is above our initial range, but let me see what we can do. Your skills are definitely what we need.",
        "narration": "Manager seems receptive.",
    },
    {
        "event": {"type": "negative_outcome", "intensity": 0.6,
                  "description": "They came back with $155k — $25k below your ask. A significant lowball."},
        "says": "So I spoke with our comp team, and honestly the best we can do right now is $155k. I know that's below what you mentioned.",
        "narration": "Significant lowball.",
    },
    {
        "event": {"type": "pressure", "intensity": 0.8,
                  "description": "End of day deadline and mention of other candidates. Classic pressure tactic."},
        "says": "Look, I need an answer by end of day. We have other strong candidates and the team wants to close this position this week.",
        "narration": "Pressure tactics.",
    },
    {
        "event": {"type": "insult", "intensity": 0.7,
                  "description": "The manager implies you're not qualified enough for the role — that you'd be 'stretching'. Dismissive of your experience."},
        "says": "Between us, $155k is generous given that your current role is a level below what we're hiring for. You'd be stretching into this position.",
        "narration": "Experience dismissed.",
    },
    {
        "event": {"type": "negative_outcome", "intensity": 0.6,
                  "description": "They move to $160k but frame it as a hard ceiling. Still $20k below target."},
        "says": "I understand your frustration, but the budget is the budget. I can maybe push to $160k, but that's truly the ceiling.",
        "narration": "Small concession, still far off.",
    },
    {
        "event": {"type": "surprise", "intensity": 0.6,
                  "description": "VP wants to meet you directly. An unexpected door opens — someone with real authority."},
        "says": "Actually — I just got a message. Our VP wants to meet you. She might have flexibility that I don't. Would you be open to a quick chat with her?",
        "narration": "Unexpected VP meeting.",
    },
    {
        "event": {"type": "praise", "intensity": 0.7,
                  "description": "The manager says your interview was one of the best in 10 years. Genuine recognition of your value."},
        "says": "Alex, I've been in this role for 10 years, and your technical interview was one of the best I've ever seen. I genuinely want you on this team.",
        "narration": "Genuine recognition.",
    },
]


def run_personality(personality: PersonalityProfile, label: str):
    """Run the negotiation scenario with one personality."""
    pipeline = AdaptiveBrainForestPipeline(
        scenario=NEGOTIATION_SCENARIO,
        personality=personality,
        momentum=0.30,
        decay_rate=0.02,
    )

    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"  Temperament: {personality.temperament}")
    print(f"{'=' * 90}")

    all_results = []

    for i, turn in enumerate(TURNS):
        result = pipeline.process_turn(
            event=turn["event"],
            other_person_says=turn["says"],
            pre_events=turn.get("pre_events"),
        )
        all_results.append(result)

        state = result["state"]
        print(f"\n{'─' * 90}")
        print(f"  TURN {i+1}: {turn['narration']}")
        print(f"{'─' * 90}")
        print(f"  Event: {turn['event']['type']} | Regime: {result['regime']}")

        # Emotional bars (compact)
        key_vars = ["confidence", "stress", "frustration", "trust",
                     "anger", "patience", "hope", "fatigue"]
        for var in key_vars:
            val = state[var]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"    {var:<14s} {bar} {val:.0%}")

        if result["conflicts"]:
            for c in result["conflicts"]:
                print(f"    ⚡ {c['drives'][0]} vs {c['drives'][1]}")

        print(f"\n  LLM reasoning: \"{result['delta_reasoning']}\"")
        print(f"\n  Manager: \"{turn['says'][:80]}{'...' if len(turn['says']) > 80 else ''}\"")
        print(f"\n  {personality.name} [{result['selected_branch']}]: \"{result['speech']}\"")
        print(f"  Coherence: {result['coherence_score']:.2f}")

    # Journey summary
    print(f"\n{'─' * 90}")
    print(f"  EMOTIONAL JOURNEY — {label}")
    print(f"{'─' * 90}")
    print(f"  {'Turn':<5s} {'anger':>6s} {'trust':>6s} {'stress':>6s} {'frust':>6s} "
          f"{'patie':>6s} {'hope':>6s}  {'regime'}")
    print(f"  {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*20}")
    for r in all_results:
        s = r["state"]
        print(f"  {r['turn']:<5d} {s['anger']:>5.0%} {s['trust']:>5.0%} {s['stress']:>5.0%} "
              f"{s['frustration']:>5.0%} {s['patience']:>5.0%} {s['hope']:>5.0%}  "
              f"{r['regime']}")

    # LLM usage
    delta_stats = pipeline.brain.delta_call_summary()
    gen_stats = pipeline.branch_gen.gen_call_summary()
    speech_stats = pipeline.backend.usage_summary()
    print(f"\n  LLM Calls: delta={delta_stats['calls']} ({delta_stats['avg_ms']:.0f}ms avg) | "
          f"branch_gen={gen_stats['calls']} (cache={gen_stats['cache_size']}) | "
          f"speech={speech_stats['call_count']} ({speech_stats['total_tokens']} tokens)")

    return all_results


def main():
    print("=" * 90)
    print("  ADAPTIVE BRAIN + PROMPT FOREST + LLM")
    print("  No hardcoded emotions — LLM interprets events, generates branches dynamically")
    print("  Same scenario, TWO different personalities → different emotional arcs")
    print("=" * 90)

    # Personality A: Calm, analytical
    calm = PersonalityProfile(
        name="Alex",
        background=(
            "32 years old, 8 years experience, underpaid at current job for 2 years. "
            "This negotiation matters but you've prepared thoroughly."
        ),
        temperament="Patient, analytical, keeps emotions in check. Prefers data-driven arguments. "
                     "Rarely raises voice. Processes frustration internally before responding.",
        emotional_tendencies={
            "anger": "slow to build, expressed coldly rather than explosively",
            "patience": "naturally high, erodes slowly",
            "confidence": "steady, grounded in preparation",
        },
    )

    # Personality B: Fiery, direct
    fiery = PersonalityProfile(
        name="Alex",
        background=(
            "32 years old, 8 years experience, underpaid at current job for 2 years. "
            "This negotiation matters and you're tired of being undervalued."
        ),
        temperament="Hot-tempered, direct, takes disrespect personally. Speaks from the gut. "
                     "Wears emotions on sleeve. Quick to escalate when feeling dismissed.",
        emotional_tendencies={
            "anger": "quick to flare, expressed openly and forcefully",
            "patience": "runs out fast, especially when patronized",
            "impulse": "high, often speaks before thinking",
        },
    )

    results_calm = run_personality(calm, "PERSONALITY A — Calm Negotiator")
    results_fiery = run_personality(fiery, "PERSONALITY B — Fiery Negotiator")

    # Side-by-side comparison
    print(f"\n\n{'═' * 90}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'═' * 90}")
    print(f"\n  {'Turn':<5s} {'Calm regime':<25s} {'Fiery regime':<25s} "
          f"{'Calm anger':>10s} {'Fiery anger':>11s}")
    print(f"  {'─'*5} {'─'*25} {'─'*25} {'─'*10} {'─'*11}")
    for rc, rf in zip(results_calm, results_fiery):
        print(f"  {rc['turn']:<5d} {rc['regime']:<25s} {rf['regime']:<25s} "
              f"{rc['state']['anger']:>9.0%} {rf['state']['anger']:>10.0%}")

    print(f"\n{'═' * 90}")


if __name__ == "__main__":
    main()

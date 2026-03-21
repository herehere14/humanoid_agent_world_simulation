#!/usr/bin/env python3
"""Brain-Enhanced LLM using the LEARNED brain (GRU model) instead of hardcoded rules.

Key improvements over the hardcoded brain:
  1. Emotional state comes from a model trained on 25k real conversations
  2. Event type is used to generate BEHAVIORAL directives, not just feelings
  3. Personality-aware behavioral mapping: same emotion → different behavior for different people

The emotional narrative is split into two parts:
  - WHAT YOU FEEL: from the learned brain (data-driven)
  - HOW YOU ACT: behavioral directives derived from emotion + personality + event
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openai
from brain_adaptive_prototype import PersonalityProfile
from learned_brain.learned_brain_engine import LearnedBrainEngine


# Behavioral directives based on emotional state + personality type
_NEGATIVE_BEHAVIORS_VOLATILE = {
    "mild": [
        "Your responses have an edge — you're not hiding your irritation",
        "You'd speak with more force than usual",
    ],
    "moderate": [
        "Short, sharp responses — no filter right now",
        "You wouldn't bother with pleasantries or diplomatic softening",
        "Your responses would have attitude",
    ],
    "strong": [
        "You're speaking from the gut — raw, unfiltered, maybe even reckless",
        "You might interrupt, raise your voice, or make demands",
        "Niceties are gone. You're in attack mode",
    ],
    "sustained": [
        "You've been angry/frustrated for a while and it's become cold fury",
        "Don't repeat yourself — escalate. Demand action, threaten consequences",
        "Every word they say makes it worse. You're one wrong word from ending this",
    ],
}

_NEGATIVE_BEHAVIORS_CALM = {
    "mild": [
        "Controlled but there's steel underneath",
        "You'd be precise in your words — choosing them to cut",
    ],
    "moderate": [
        "You're clipped and cold — the anger shows in brevity, not volume",
        "You'd question promises and ask for specifics",
    ],
    "strong": [
        "Cold precision — you've moved past emotion into ruthless pragmatism",
        "Your words are surgical. No wasted syllables",
    ],
    "sustained": [
        "Sustained anger has made you ice-cold. This is contempt, not heat",
        "You assume they're lying or stalling. Challenge every claim",
    ],
}

_POSITIVE_BEHAVIORS = {
    "mild": [
        "You'd respond naturally, with a bit more warmth than usual",
    ],
    "moderate": [
        "You'd speak with conviction — you know your position is strong",
        "Let the confidence show but don't oversell",
    ],
    "strong": [
        "You're riding high — swagger, energy, boldness",
        "You own this and everyone should recognize it",
    ],
    "sustained": [
        "Sustained confidence — you've been in control and it shows in everything",
        "You're not just confident, you're commanding",
    ],
}

_ANXIOUS_BEHAVIORS = {
    "mild": [
        "Slightly on edge — you want answers, not reassurance",
    ],
    "moderate": [
        "You're anxious and it's making you demanding — you need certainty NOW",
        "You might jump to worst-case scenarios",
    ],
    "strong": [
        "Anxiety is driving you — you're reactive, impatient, maybe aggressive as a defense",
        "You'd push back hard against vague or uncertain answers",
    ],
    "sustained": [
        "You've been anxious for too long — it's turning into frustration or desperation",
        "You might lash out, not from anger but from fear",
    ],
}

# Negative emotion keywords
_NEGATIVE_EMOTIONS = {"angry", "annoyed", "furious", "disgusted", "frustrated",
                       "disappointed", "devastated", "afraid", "terrified", "ashamed",
                       "embarrassed", "guilty", "jealous", "lonely", "sad"}
_POSITIVE_EMOTIONS = {"confident", "proud", "hopeful", "joyful", "excited", "grateful",
                       "content", "impressed", "trusting", "faithful", "caring", "prepared"}
_ANXIOUS_EMOTIONS = {"anxious", "apprehensive", "anticipating", "afraid", "terrified"}


def _get_behavioral_directives(
    top_emotions: list[tuple[str, float]],
    personality: PersonalityProfile,
    turn: int,
    sustained_count: int = 0,
) -> list[str]:
    """Generate behavioral directives from emotion state + personality."""
    if not top_emotions:
        return ["Respond naturally in your normal style"]

    primary_emotion, primary_prob = top_emotions[0]

    # Classify emotional valence
    neg_weight = sum(p for e, p in top_emotions[:3] if e in _NEGATIVE_EMOTIONS)
    pos_weight = sum(p for e, p in top_emotions[:3] if e in _POSITIVE_EMOTIONS)
    anx_weight = sum(p for e, p in top_emotions[:3] if e in _ANXIOUS_EMOTIONS)

    is_volatile = any(w in personality.temperament.lower()
                      for w in ("hot-tempered", "impulsive", "direct", "quick to escalate"))

    # Determine intensity level
    dominant_weight = max(neg_weight, pos_weight, anx_weight)
    if sustained_count >= 3:
        level = "sustained"
    elif dominant_weight > 0.5:
        level = "strong"
    elif dominant_weight > 0.25:
        level = "moderate"
    else:
        level = "mild"

    # Select behavior set
    if neg_weight >= pos_weight and neg_weight >= anx_weight:
        behaviors = (_NEGATIVE_BEHAVIORS_VOLATILE if is_volatile
                     else _NEGATIVE_BEHAVIORS_CALM).get(level, [])
    elif anx_weight >= pos_weight:
        behaviors = _ANXIOUS_BEHAVIORS.get(level, [])
    else:
        behaviors = _POSITIVE_BEHAVIORS.get(level, [])

    # Late-conversation escalation
    if turn >= 4 and neg_weight > 0.4:
        behaviors = list(behaviors) + [
            "This has gone on too long. A real person would escalate — ultimatums, threats to leave, mention authorities"
        ]

    return behaviors[:3]  # Max 3 directives


class LearnedBrainLLM:
    """Single LLM call augmented with LEARNED brain state + behavioral directives."""

    def __init__(
        self,
        personality: PersonalityProfile,
        scenario: str,
        checkpoint_path: str | None = None,
    ):
        self.personality = personality
        self.scenario = scenario
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []
        self.total_tokens = 0

        # Learned brain engine
        self.brain = LearnedBrainEngine(checkpoint_path=checkpoint_path)
        self.brain.reset(scenario)

    def reset_for_scenario(self, scenario: str):
        """Reset for new scenario, keeping personality."""
        self.scenario = scenario
        self.conversation_history = []
        self.brain.reset(scenario)

    def _build_system_prompt(self, emotional_narrative: str, behavioral_hints: str) -> str:
        """Build system prompt: personality + emotional state + behavioral directives."""
        tendencies = ""
        if self.personality.emotional_tendencies:
            tendencies = "\nEmotional tendencies: " + ", ".join(
                f"{k} — {v}" for k, v in self.personality.emotional_tendencies.items()
            )

        return f"""You are {self.personality.name}, a real human being in this situation:

{self.scenario}

About you:
{self.personality.background}
Your temperament: {self.personality.temperament}{tendencies}

EMOTIONAL CONTEXT (how your feelings have been building across this conversation):
{emotional_narrative}

HOW THIS AFFECTS YOUR BEHAVIOR RIGHT NOW:
{behavioral_hints}

Rules:
- You ARE {self.personality.name}. Never break character.
- Your personality and temperament are PRIMARY. The emotional context above enriches your performance — use it to calibrate your intensity and tone.
- The behavioral hints tell you HOW your emotions translate to speech patterns.
- 1-3 sentences. Real humans keep it short.
- Sound HUMAN: fragments, contractions, "look", "honestly", "come on". Imperfect grammar.
- Never say "I understand your perspective", "I appreciate your transparency", "I hear you".
- Let your {self.personality.temperament.split(',')[0].lower()} nature show through in EVERY response."""

    def respond(
        self,
        event: dict,
        other_person_says: str,
        pre_events: list[dict] | None = None,
    ) -> dict:
        """Process one turn: learned brain tracks state, LLM generates speech."""
        # 1. Feed the other person's utterance through the brain
        self.brain.process_utterance(other_person_says)

        # 2. Get emotional narrative from learned model
        emotional_narrative = self.brain.get_emotional_narrative()

        # 3. Generate behavioral directives from emotion + personality
        sustained = self.brain._get_sustained_emotions()
        max_sustained = max((n for _, n in sustained), default=0)
        directives = _get_behavioral_directives(
            self.brain.state.top_emotions,
            self.personality,
            self.brain.state.turn,
            max_sustained,
        )
        behavioral_hints = "\n".join(f"- {d}" for d in directives)

        # 4. Build prompt
        system_prompt = self._build_system_prompt(emotional_narrative, behavioral_hints)

        # 5. Single LLM call
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_person_says})

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.85,
                max_tokens=150,
            )
            speech = response.choices[0].message.content.strip().strip('"')
            if speech.lower().startswith(self.personality.name.lower() + ":"):
                speech = speech[len(self.personality.name) + 1:].strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0
        except Exception as e:
            speech = f"[Error: {e}]"

        # 6. Feed our OWN response through the brain too
        self.brain.process_utterance(speech)

        # 7. Update conversation history (clean)
        self.conversation_history.append({"role": "user", "content": other_person_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        return {
            "speech": speech,
            "regime": "learned",
            "branch": f"top:{self.brain.state.top_emotions[0][0] if self.brain.state.top_emotions else 'neutral'}",
            "state": {
                "top_emotions": self.brain.state.top_emotions,
                "latent_norm": float(self.brain.state.latent.sum() ** 2) ** 0.5
                if self.brain.state.latent is not None else 0,
            },
            "emotional_narrative": emotional_narrative,
            "behavioral_hints": behavioral_hints,
            "turn": self.brain.state.turn,
        }

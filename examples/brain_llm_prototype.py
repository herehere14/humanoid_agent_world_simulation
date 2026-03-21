#!/usr/bin/env python3
"""Brain + LLM Prototype — Cognitive state engine driving human-like speech.

Architecture:
  [World Events] → [BRAIN] → [Structured Emotional State] → [LLM] → [Human Speech]

The brain tracks internal emotional/cognitive state over time.
The LLM receives that state and generates speech that SOUNDS like
a person feeling those emotions. The brain decides WHAT to feel,
the LLM decides HOW to express it.

Demo: A job negotiation scenario where emotional state evolves
across multiple turns based on what happens.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

import openai


# ---------------------------------------------------------------------------
# Brain State Engine
# ---------------------------------------------------------------------------

@dataclass
class BrainState:
    """Internal emotional/cognitive state — all values in [0, 1]."""

    # Core emotions
    confidence: float = 0.55
    stress: float = 0.20
    frustration: float = 0.10
    trust: float = 0.50
    anger: float = 0.05
    hope: float = 0.50

    # Drives
    curiosity: float = 0.60
    fear: float = 0.15
    motivation: float = 0.60
    patience: float = 0.70
    impulse: float = 0.30
    empathy: float = 0.50
    caution: float = 0.35

    # Cognitive
    fatigue: float = 0.15
    reflection: float = 0.50
    emotional_momentum: float = 0.0  # how fast emotions are shifting

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.__dict__.items()}

    def dominant_emotions(self, top_n: int = 4) -> list[tuple[str, float]]:
        """Return the strongest active emotions/drives."""
        items = [(k, v) for k, v in self.__dict__.items()
                 if k != "emotional_momentum"]
        items.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        return [(k, v) for k, v in items[:top_n]]


_BASELINES = BrainState()
_BASELINE_DICT = _BASELINES.as_dict()

# Opposing drives — when both high, creates internal conflict
_OPPOSING_PAIRS = [
    ("curiosity", "fear"),
    ("impulse", "patience"),
    ("empathy", "anger"),
    ("trust", "caution"),
    ("hope", "frustration"),
    ("confidence", "stress"),
]


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Brain Engine — updates state from events
# ---------------------------------------------------------------------------

class BrainEngine:
    """Tracks emotional state over time, responding to world events.

    The engine handles:
    - Event-driven state updates (outcomes change emotions)
    - Emotional momentum (changes are gradual, not instant)
    - Homeostatic decay (emotions drift back toward baseline)
    - Drive conflicts (opposing drives create internal tension)
    - Personality adaptation (baselines shift per individual)
    """

    def __init__(self, momentum: float = 0.6, decay_rate: float = 0.04):
        self.state = BrainState()
        self.prev_state = BrainState()
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.turn = 0
        self.history: list[dict] = []
        self._rng = random.Random(42)

        # Per-individual baseline adaptation
        self._running_sums: dict[str, float] = {k: v for k, v in _BASELINE_DICT.items()}
        self._running_counts: dict[str, int] = {k: 1 for k in _BASELINE_DICT}
        self._adapted_baselines: dict[str, float] = dict(_BASELINE_DICT)

    def process_event(self, event: dict) -> dict:
        """Process a world event and update emotional state.

        Parameters
        ----------
        event : dict
            Must contain 'type' and optional parameters.
            Types: positive_outcome, negative_outcome, threat,
                   insult, praise, betrayal, opportunity,
                   pressure, relief, boredom, surprise

        Returns
        -------
        dict with regime, conflicts, dominant emotions, and full state.
        """
        self.turn += 1
        self.prev_state = BrainState(**self.state.__dict__)

        # Compute deltas from event
        deltas = self._event_to_deltas(event)

        # Apply deltas with momentum (gradual change)
        for var, delta in deltas.items():
            old_val = getattr(self.state, var)
            target = _clamp(old_val + delta)
            new_val = old_val * self.momentum + target * (1.0 - self.momentum)
            # Add small noise for bounded rationality
            noise = self._rng.gauss(0, 0.03)
            setattr(self.state, var, _clamp(new_val + noise))

        # Cross-variable interactions
        self._apply_interactions()

        # Homeostatic decay toward adapted baselines
        self._apply_decay()

        # Track momentum
        velocity = sum(
            abs(getattr(self.state, k) - getattr(self.prev_state, k))
            for k in _BASELINE_DICT
        )
        self.state.emotional_momentum = _clamp(velocity * 3.0)

        # Adapt baselines to individual
        if self.turn > 10:
            for k in _BASELINE_DICT:
                self._running_sums[k] += getattr(self.state, k)
                self._running_counts[k] += 1
                self._adapted_baselines[k] = (
                    self._running_sums[k] / self._running_counts[k]
                )

        # Detect regime and conflicts
        regime = self._detect_regime()
        conflicts = self._detect_conflicts()
        dominant = self.state.dominant_emotions(5)

        result = {
            "turn": self.turn,
            "event": event,
            "regime": regime,
            "conflicts": conflicts,
            "dominant": dominant,
            "state": self.state.as_dict(),
        }
        self.history.append(result)
        return result

    def _event_to_deltas(self, event: dict) -> dict[str, float]:
        """Map world events to emotional deltas."""
        etype = event.get("type", "neutral")
        intensity = event.get("intensity", 0.5)  # 0-1 scale
        deltas: dict[str, float] = {}

        if etype == "positive_outcome":
            deltas = {
                "confidence": 0.15 * intensity,
                "motivation": 0.12 * intensity,
                "hope": 0.15 * intensity,
                "stress": -0.10 * intensity,
                "frustration": -0.12 * intensity,
                "trust": 0.05 * intensity,
                "fear": -0.05 * intensity,
            }

        elif etype == "negative_outcome":
            # Frustration amplifies stress (emotional cascade)
            stress_amp = 1.0 + self.state.frustration * 0.5
            deltas = {
                "confidence": -0.15 * intensity,
                "stress": 0.15 * intensity * stress_amp,
                "frustration": 0.18 * intensity,
                "motivation": -0.10 * intensity,
                "hope": -0.12 * intensity,
                "fatigue": 0.05 * intensity,
                "fear": 0.08 * intensity,
                "patience": -0.10 * intensity,
            }

        elif etype == "threat":
            deltas = {
                "fear": 0.25 * intensity,
                "stress": 0.20 * intensity,
                "caution": 0.15 * intensity,
                "curiosity": -0.10 * intensity,
                "confidence": -0.10 * intensity,
                "anger": 0.05 * intensity,
            }

        elif etype == "insult":
            deltas = {
                "anger": 0.25 * intensity,
                "trust": -0.20 * intensity,
                "frustration": 0.15 * intensity,
                "empathy": -0.10 * intensity,
                "stress": 0.10 * intensity,
                "patience": -0.15 * intensity,
                "confidence": -0.05 * intensity,
            }

        elif etype == "praise":
            deltas = {
                "confidence": 0.20 * intensity,
                "trust": 0.15 * intensity,
                "motivation": 0.15 * intensity,
                "stress": -0.10 * intensity,
                "hope": 0.10 * intensity,
                "empathy": 0.05 * intensity,
            }

        elif etype == "betrayal":
            deltas = {
                "trust": -0.35 * intensity,
                "anger": 0.30 * intensity,
                "stress": 0.20 * intensity,
                "caution": 0.25 * intensity,
                "hope": -0.20 * intensity,
                "empathy": -0.15 * intensity,
                "fear": 0.10 * intensity,
            }

        elif etype == "opportunity":
            deltas = {
                "curiosity": 0.20 * intensity,
                "hope": 0.15 * intensity,
                "motivation": 0.15 * intensity,
                "confidence": 0.10 * intensity,
                "fear": 0.05 * intensity,  # opportunity also has risk
            }

        elif etype == "pressure":
            deltas = {
                "stress": 0.20 * intensity,
                "impulse": 0.15 * intensity,
                "fatigue": 0.10 * intensity,
                "patience": -0.12 * intensity,
                "reflection": -0.10 * intensity,
            }

        elif etype == "relief":
            deltas = {
                "stress": -0.25 * intensity,
                "frustration": -0.15 * intensity,
                "fatigue": -0.10 * intensity,
                "hope": 0.10 * intensity,
                "patience": 0.10 * intensity,
            }

        elif etype == "boredom":
            deltas = {
                "curiosity": -0.15 * intensity,
                "motivation": -0.15 * intensity,
                "fatigue": 0.15 * intensity,
                "impulse": 0.10 * intensity,
                "patience": -0.10 * intensity,
            }

        elif etype == "surprise":
            deltas = {
                "curiosity": 0.20 * intensity,
                "fear": 0.10 * intensity,
                "reflection": 0.15 * intensity,
                "impulse": 0.10 * intensity,
            }

        return deltas

    def _apply_interactions(self):
        """Cross-variable emotional interactions."""
        s = self.state
        # High stress suppresses reflection, increases impulse
        if s.stress > 0.6:
            s.reflection = _clamp(s.reflection - 0.03)
            s.impulse = _clamp(s.impulse + 0.02)
        # High fatigue reduces motivation and curiosity
        if s.fatigue > 0.7:
            s.motivation = _clamp(s.motivation - 0.03)
            s.curiosity = _clamp(s.curiosity - 0.02)
        # High anger reduces patience and empathy
        if s.anger > 0.5:
            s.patience = _clamp(s.patience - 0.03)
            s.empathy = _clamp(s.empathy - 0.02)
        # High confidence boosts motivation
        if s.confidence > 0.7:
            s.motivation = _clamp(s.motivation + 0.02)
        # High frustration + low patience → anger builds
        if s.frustration > 0.5 and s.patience < 0.3:
            s.anger = _clamp(s.anger + 0.04)

    def _apply_decay(self):
        """Decay emotions toward homeostatic baselines."""
        for var, baseline in self._adapted_baselines.items():
            if var == "emotional_momentum":
                continue
            current = getattr(self.state, var)
            decayed = current + (baseline - current) * self.decay_rate
            setattr(self.state, var, _clamp(decayed))

    def _detect_regime(self) -> str:
        """Detect current cognitive-emotional regime.

        Thresholds are deliberately lower than you'd expect —
        emotions in [0,1] rarely exceed 0.5 due to momentum + decay,
        so regime detection must be sensitive to moderate levels.
        """
        s = self.state
        # Order matters — most intense regimes checked first
        if s.anger > 0.40 and s.patience < 0.45:
            return "angry_reactive"
        if s.frustration > 0.35 and s.impulse > 0.35:
            return "frustrated_impulsive"
        if s.frustration > 0.30 and s.hope < 0.35:
            return "defeated_resigned"
        if s.stress > 0.50 and s.fatigue > 0.30:
            return "exhausted_stressed"
        if s.trust < 0.25 and s.caution > 0.30:
            return "guarded_defensive"
        if s.trust < 0.30 and s.anger > 0.20:
            return "guarded_defensive"
        # Stressed-guarded: high stress + dropping trust even if not fully distrustful
        if s.stress > 0.35 and s.trust < 0.45 and s.anger > 0.10:
            return "guarded_defensive"
        # Frustrated under pressure: stress rising + frustration growing + patience eroding
        if s.stress > 0.35 and s.frustration > 0.18 and s.patience < 0.55:
            return "frustrated_impulsive"
        if s.confidence > 0.60 and s.motivation > 0.55:
            return "confident_assertive"
        if s.curiosity > 0.55 and s.hope > 0.50:
            return "optimistic_engaged"
        if s.trust > 0.55 and s.empathy > 0.45:
            return "warm_collaborative"
        return "baseline_neutral"

    def _detect_conflicts(self) -> list[dict]:
        """Detect active drive conflicts (opposing drives both high)."""
        conflicts = []
        for a, b in _OPPOSING_PAIRS:
            va = getattr(self.state, a)
            vb = getattr(self.state, b)
            # Both elevated and close to each other → conflict
            if va > 0.4 and vb > 0.4 and abs(va - vb) < 0.25:
                intensity = (va + vb) / 2.0
                conflicts.append({
                    "drives": (a, b),
                    "values": (round(va, 2), round(vb, 2)),
                    "intensity": round(intensity, 2),
                })
        conflicts.sort(key=lambda c: c["intensity"], reverse=True)
        return conflicts[:3]


# ---------------------------------------------------------------------------
# LLM Bridge — translates brain state into natural language context
# ---------------------------------------------------------------------------

def brain_state_to_prompt(
    brain_result: dict,
    scenario_context: str,
    character_name: str = "Alex",
    character_background: str = "",
) -> str:
    """Convert brain state into an LLM system prompt.

    This is the key bridge: structured emotional state → natural language
    instructions that the LLM uses to generate human-like speech.
    """
    state = brain_result["state"]
    regime = brain_result["regime"]
    conflicts = brain_result["conflicts"]
    dominant = brain_result["dominant"]

    # Build emotional portrait
    emotion_lines = []
    for name, val in dominant:
        if val > 0.65:
            emotion_lines.append(f"- {name}: HIGH ({val:.0%}) — strongly felt")
        elif val < 0.25:
            emotion_lines.append(f"- {name}: LOW ({val:.0%}) — notably absent")
        elif abs(val - 0.5) > 0.1:
            direction = "elevated" if val > 0.5 else "diminished"
            emotion_lines.append(f"- {name}: {direction} ({val:.0%})")

    emotion_portrait = "\n".join(emotion_lines) if emotion_lines else "- emotionally neutral"

    # Build conflict description
    conflict_lines = []
    for c in conflicts:
        a, b = c["drives"]
        va, vb = c["values"]
        conflict_lines.append(
            f"- Internal tension between {a} ({va:.0%}) and {b} ({vb:.0%})"
        )
    conflict_desc = "\n".join(conflict_lines) if conflict_lines else "No major internal conflicts"

    # Regime to behavioral instruction
    regime_instructions = {
        "angry_reactive": "Speak with barely contained anger. Short sentences. May raise voice or use sharp language. Patience is gone.",
        "frustrated_impulsive": "Express growing irritation. May say things impulsively that they'd normally filter. Getting close to snapping.",
        "guarded_defensive": "Be careful with words. Deflect personal questions. Keep responses measured and protective. Trust is low.",
        "confident_assertive": "Speak with conviction and clarity. Make direct points. Show initiative and strength.",
        "exhausted_stressed": "Show weariness. Shorter responses. May sigh or trail off. Having trouble engaging fully.",
        "optimistic_engaged": "Show genuine interest and positive energy. Ask questions. Be open and forward-looking.",
        "warm_collaborative": "Be warm, open, and cooperative. Show genuine care for the other person. Build rapport.",
        "defeated_resigned": "Speak with a sense of giving up. Flat affect. May agree just to end the conversation.",
        "baseline_neutral": "Respond naturally with moderate engagement. Neither particularly emotional nor flat.",
    }

    regime_instruction = regime_instructions.get(regime, regime_instructions["baseline_neutral"])

    # Key state values for nuanced expression
    key_states = {
        "confidence": state["confidence"],
        "stress": state["stress"],
        "trust": state["trust"],
        "patience": state["patience"],
        "anger": state["anger"],
        "hope": state["hope"],
    }

    prompt = f"""You are {character_name}, a real human being in the following situation:

{scenario_context}

{character_background}

--- YOUR CURRENT INTERNAL STATE ---

Emotional regime: {regime}
{regime_instruction}

Dominant feelings:
{emotion_portrait}

Internal conflicts:
{conflict_desc}

Key emotional readings:
  Confidence: {key_states['confidence']:.0%} | Stress: {key_states['stress']:.0%}
  Trust: {key_states['trust']:.0%} | Patience: {key_states['patience']:.0%}
  Anger: {key_states['anger']:.0%} | Hope: {key_states['hope']:.0%}

Emotional momentum: {state['emotional_momentum']:.0%} (how fast your feelings are shifting)

--- INSTRUCTIONS ---

Respond AS {character_name} in this moment. Your speech should naturally reflect your internal state:
- If stressed and frustrated, it should come through in word choice, sentence length, and tone
- If confident, be more articulate and decisive
- If trust is low, be guarded about sharing information
- If patience is running out, show it — interrupt, cut short, or express irritation
- If there are internal conflicts, show ambivalence — start to say one thing then qualify it
- Do NOT name your emotions explicitly (don't say "I'm feeling stressed") — SHOW them through behavior
- Keep responses natural — 1-4 sentences typical for conversation
- You are a real person, not an AI. Never break character."""

    return prompt


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------

def generate_speech(
    system_prompt: str,
    conversation_history: list[dict],
    user_message: str,
) -> str:
    """Call LLM to generate human-like speech driven by brain state."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=200,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Demo Scenario: Job Salary Negotiation
# ---------------------------------------------------------------------------

def run_negotiation_demo():
    """Multi-turn negotiation where brain state evolves and drives speech."""

    brain = BrainEngine(momentum=0.35, decay_rate=0.03)

    scenario = (
        "You are in a salary negotiation for a senior software engineer position at a tech company. "
        "You currently earn $140k and believe you deserve $180k based on your experience and market rates. "
        "The hiring manager is sitting across from you. This job would be a significant career step."
    )
    background = (
        "Background: You're 32 years old, 8 years of experience. You've been underpaid at your "
        "current job for 2 years and this negotiation matters a lot to you. You're generally "
        "professional but have a temper when you feel disrespected."
    )

    # Scripted events and manager responses that drive the scenario
    turns = [
        {
            "event": {"type": "opportunity", "intensity": 0.7},
            "manager_says": "Thanks for coming in, Alex. We're excited about your candidacy. Let's talk compensation. What are you looking for?",
            "narration": "The negotiation opens. You feel the opportunity ahead.",
        },
        {
            "event": {"type": "positive_outcome", "intensity": 0.4},
            "manager_says": "I appreciate you being upfront. $180k is above our initial range, but let me see what we can do. Your skills are definitely what we need.",
            "narration": "The manager seems receptive. A small win.",
        },
        {
            "event": {"type": "negative_outcome", "intensity": 0.6},
            "manager_says": "So I spoke with our comp team, and honestly the best we can do right now is $155k. I know that's below what you mentioned.",
            "narration": "A significant lowball. Below expectations.",
        },
        {
            "event": {"type": "pressure", "intensity": 0.7},
            "manager_says": "Look, I need an answer by end of day. We have other strong candidates and the team wants to close this position this week.",
            "narration": "Pressure tactics. Deadline imposed.",
        },
        {
            "event": {"type": "insult", "intensity": 0.5},
            "manager_says": "Between us, $155k is generous given that your current role is a level below what we're hiring for. You'd be stretching into this position.",
            "narration": "Your experience is being dismissed. Feels disrespectful.",
        },
        {
            "event": {"type": "negative_outcome", "intensity": 0.4},
            "manager_says": "I understand your frustration, but the budget is the budget. I can maybe push to $160k, but that's truly the ceiling.",
            "narration": "A small concession, but still far from your target.",
        },
        {
            "event": {"type": "surprise", "intensity": 0.6},
            "manager_says": "Actually — I just got a message. Our VP wants to meet you. She might have flexibility that I don't. Would you be open to a quick chat with her?",
            "narration": "An unexpected turn. New possibility opens up.",
        },
        {
            "event": {"type": "praise", "intensity": 0.7},
            "manager_says": "Alex, I've been in this role for 10 years, and your technical interview was one of the best I've ever seen. I genuinely want you on this team.",
            "narration": "Genuine recognition. Your value is acknowledged.",
        },
    ]

    print("=" * 78)
    print("  BRAIN + LLM PROTOTYPE — Salary Negotiation")
    print("  Brain controls emotions → LLM generates human speech")
    print("=" * 78)

    conversation_history: list[dict] = []

    for i, turn in enumerate(turns):
        # Brain processes the event
        result = brain.process_event(turn["event"])

        # Print brain state
        print(f"\n{'─' * 78}")
        print(f"  TURN {i + 1}: {turn['narration']}")
        print(f"{'─' * 78}")
        print(f"  Event: {turn['event']['type']} (intensity={turn['event']['intensity']})")
        print(f"  Regime: {result['regime']}")

        # Show key emotions with visual bars
        state = result["state"]
        key_vars = ["confidence", "stress", "frustration", "trust",
                     "anger", "patience", "hope", "motivation"]
        print(f"\n  Emotional State:")
        for var in key_vars:
            val = state[var]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            changed = val - getattr(brain.prev_state, var)
            arrow = "↑" if changed > 0.02 else ("↓" if changed < -0.02 else " ")
            print(f"    {var:<14s} {bar} {val:.0%} {arrow}")

        if result["conflicts"]:
            print(f"\n  Internal Conflicts:")
            for c in result["conflicts"]:
                a, b = c["drives"]
                print(f"    ⚡ {a} vs {b} (intensity={c['intensity']})")

        # Generate speech via LLM
        system_prompt = brain_state_to_prompt(
            result, scenario, "Alex", background,
        )
        manager_msg = turn["manager_says"]
        print(f"\n  Manager: \"{manager_msg}\"")

        speech = generate_speech(system_prompt, conversation_history, manager_msg)
        print(f"\n  Alex:    \"{speech}\"")

        # Track conversation for context
        conversation_history.append({"role": "user", "content": manager_msg})
        conversation_history.append({"role": "assistant", "content": speech})

    # Final state summary
    print(f"\n{'═' * 78}")
    print(f"  EMOTIONAL JOURNEY")
    print(f"{'═' * 78}")
    key_vars = ["confidence", "stress", "frustration", "trust", "anger", "patience", "hope"]
    print(f"\n  {'Turn':<6s}", end="")
    for v in key_vars:
        print(f"  {v[:6]:>6s}", end="")
    print(f"  {'regime':<24s}")
    print(f"  {'─' * 6}", end="")
    for _ in key_vars:
        print(f"  {'─' * 6}", end="")
    print(f"  {'─' * 24}")
    for entry in brain.history:
        t = entry["turn"]
        s = entry["state"]
        r = entry["regime"]
        print(f"  {t:<6d}", end="")
        for v in key_vars:
            print(f"  {s[v]:>5.0%}", end="")
        print(f"  {r}")

    print(f"\n{'═' * 78}")


# ---------------------------------------------------------------------------
# Demo Scenario 2: Customer Getting Progressively Angry
# ---------------------------------------------------------------------------

def run_customer_service_demo():
    """A customer calling support, getting increasingly frustrated."""

    brain = BrainEngine(momentum=0.30, decay_rate=0.02)

    scenario = (
        "You are calling customer support for your internet provider. Your internet has been "
        "down for 3 days and you work from home. You've already called twice before and each time "
        "they said it would be fixed 'within 24 hours'. You're calling for the third time now."
    )
    background = (
        "Background: You're a freelance designer with a deadline tomorrow. You've lost $500 in "
        "productivity already. You started calm but you're running out of patience. You have a "
        "direct communication style."
    )

    turns = [
        {
            "event": {"type": "frustration_carryover", "intensity": 0.0},
            "agent_says": "Thank you for calling TechNet support! My name is Jordan. How can I help you today?",
            "narration": "Third call. Already carrying frustration from previous failures.",
            "pre_events": [
                {"type": "negative_outcome", "intensity": 0.5},  # memory of past failures
                {"type": "negative_outcome", "intensity": 0.5},
            ],
        },
        {
            "event": {"type": "betrayal", "intensity": 0.6},
            "agent_says": "I see, let me pull up your account... Okay I can see the previous tickets. It looks like a technician was scheduled but... hmm, it seems they marked it as resolved?",
            "narration": "They marked it resolved without fixing it. Feels like betrayal.",
        },
        {
            "event": {"type": "insult", "intensity": 0.7},
            "agent_says": "Have you tried restarting your router? Sometimes these issues resolve themselves if you just power cycle the equipment.",
            "narration": "The classic script response. Not being listened to AT ALL.",
        },
        {
            "event": {"type": "negative_outcome", "intensity": 0.8},
            "agent_says": "I understand your frustration. Unfortunately, the earliest I can schedule a technician is... next Thursday. Would that work?",
            "narration": "A WEEK more? With a deadline tomorrow? Absolutely unacceptable.",
        },
        {
            "event": {"type": "insult", "intensity": 0.6},
            "agent_says": "Sir/Ma'am, I want to help but I need you to stay calm so we can work through this together. There's a process we need to follow.",
            "narration": "Being told to calm down. Condescending. Makes it worse.",
        },
        {
            "event": {"type": "opportunity", "intensity": 0.5},
            "agent_says": "Actually, let me check one thing... I might be able to escalate this to our emergency repair team. They handle outages affecting work-from-home customers. Can you hold for just two minutes?",
            "narration": "A glimmer of hope. Someone is actually trying.",
        },
        {
            "event": {"type": "positive_outcome", "intensity": 0.7},
            "agent_says": "Great news — I got approval. We're sending a technician tomorrow morning between 8-10 AM, and I'm also crediting your account for the 3 days of downtime. Does that work?",
            "narration": "Finally! Resolution. The anger starts to subside.",
        },
    ]

    print("\n\n")
    print("=" * 78)
    print("  BRAIN + LLM PROTOTYPE — Customer Service Call")
    print("  Brain controls emotions → LLM generates human speech")
    print("=" * 78)

    conversation_history: list[dict] = []

    for i, turn in enumerate(turns):
        # Process any pre-events (carrying emotional baggage)
        for pre in turn.get("pre_events", []):
            brain.process_event(pre)

        # Brain processes the main event
        result = brain.process_event(turn["event"])

        print(f"\n{'─' * 78}")
        print(f"  TURN {i + 1}: {turn['narration']}")
        print(f"{'─' * 78}")
        print(f"  Event: {turn['event']['type']} (intensity={turn['event']['intensity']})")
        print(f"  Regime: {result['regime']}")

        state = result["state"]
        key_vars = ["confidence", "stress", "frustration", "trust",
                     "anger", "patience", "hope", "fatigue"]
        print(f"\n  Emotional State:")
        for var in key_vars:
            val = state[var]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"    {var:<14s} {bar} {val:.0%}")

        if result["conflicts"]:
            print(f"\n  Internal Conflicts:")
            for c in result["conflicts"]:
                a, b = c["drives"]
                print(f"    ⚡ {a} vs {b} (intensity={c['intensity']})")

        system_prompt = brain_state_to_prompt(
            result, scenario, "Alex", background,
        )
        agent_msg = turn["agent_says"]
        print(f"\n  Support: \"{agent_msg}\"")

        speech = generate_speech(system_prompt, conversation_history, agent_msg)
        print(f"\n  Alex:    \"{speech}\"")

        conversation_history.append({"role": "user", "content": agent_msg})
        conversation_history.append({"role": "assistant", "content": speech})

    # Emotional journey
    print(f"\n{'═' * 78}")
    print(f"  EMOTIONAL JOURNEY")
    print(f"{'═' * 78}")
    key_vars = ["confid", "stress", "frustr", "trust", "anger", "patien", "hope"]
    full_vars = ["confidence", "stress", "frustration", "trust", "anger", "patience", "hope"]
    print(f"\n  {'Turn':<6s}", end="")
    for v in key_vars:
        print(f"  {v:>6s}", end="")
    print(f"  {'regime':<24s}")
    for entry in brain.history:
        t = entry["turn"]
        s = entry["state"]
        r = entry["regime"]
        print(f"  {t:<6d}", end="")
        for v in full_vars:
            print(f"  {s[v]:>5.0%}", end="")
        print(f"  {r}")

    print(f"\n{'═' * 78}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_negotiation_demo()
    run_customer_service_demo()

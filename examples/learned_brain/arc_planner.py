#!/usr/bin/env python3
"""Arc Planner — plans emotional trajectory at conversation start.

One LLM call at the beginning of a conversation to plan a 6-turn arc.
Used as a soft bias on the prompt policy's strategy selection.
"""

from __future__ import annotations

import json
import os

import openai

from .prompt_policy import STRATEGY_NAMES

_ARC_PROMPT = """You are an expert at human emotional dynamics. Given a scenario and a character's personality, plan how their emotional state should evolve over a 6-turn conversation.

SCENARIO: {scenario}

CHARACTER:
Name: {name}
Background: {background}
Temperament: {temperament}

AVAILABLE BEHAVIORAL STRATEGIES (one per turn):
1. raw_explosive — fragments, demands, threats
2. cold_controlled — clipped, minimal, icy precision
3. sarcastic_bitter — eye-rolls, rhetorical questions
4. anxious_scattered — broken rhythm, trailing off
5. defeated_minimal — ultra-short, flat, giving up
6. confident_direct — bold, clear, owning the room
7. warm_engaged — enthusiastic, collaborative
8. cautious_measured — hedging, careful

Output a JSON array of 6 objects, one per turn:
[
  {{"turn": 1, "valence": float (-1 to 1), "intensity": float (0 to 1), "strategy": "strategy_name", "note": "brief reason"}},
  ...
]

RULES:
- Emotions should BUILD realistically — no sudden jumps without cause
- Match the character's temperament (hot-tempered people escalate faster)
- The arc should feel like a real human's emotional journey
- Intensity generally increases in conflict scenarios, decreases in positive ones
- Consider how THIS specific person would react given their background"""


class ArcPlanner:
    """Plans a conversation emotional arc with one LLM call."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def plan_arc(self, scenario: str, personality) -> list[dict]:
        """Generate a 6-turn emotional arc plan.

        Args:
            scenario: scenario description text
            personality: PersonalityProfile with name, background, temperament

        Returns:
            List of 6 dicts with: turn, valence, intensity, strategy, note
        """
        prompt = _ARC_PROMPT.format(
            scenario=scenario,
            name=personality.name,
            background=personality.background,
            temperament=personality.temperament,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            # Handle both {"arc": [...]} and direct [...] formats
            if isinstance(data, dict):
                arc = data.get("arc", data.get("turns", data.get("plan", [])))
                if not arc:
                    # Try to find any list value
                    for v in data.values():
                        if isinstance(v, list):
                            arc = v
                            break
            elif isinstance(data, list):
                arc = data
            else:
                arc = []

            # Validate and clean
            cleaned = []
            for i, entry in enumerate(arc[:6]):
                strategy = entry.get("strategy", "cautious_measured")
                if strategy not in STRATEGY_NAMES:
                    strategy = "cautious_measured"
                cleaned.append({
                    "turn": i + 1,
                    "valence": float(entry.get("valence", 0.0)),
                    "intensity": float(entry.get("intensity", 0.5)),
                    "strategy_bias": strategy,
                    "note": entry.get("note", ""),
                })

            # Pad to 6 if needed
            while len(cleaned) < 6:
                cleaned.append({
                    "turn": len(cleaned) + 1,
                    "valence": 0.0,
                    "intensity": 0.5,
                    "strategy_bias": "cautious_measured",
                    "note": "default",
                })

            return cleaned

        except Exception as e:
            # Fallback: neutral arc
            return [
                {"turn": i + 1, "valence": 0.0, "intensity": 0.5,
                 "strategy_bias": "cautious_measured", "note": f"fallback ({e})"}
                for i in range(6)
            ]

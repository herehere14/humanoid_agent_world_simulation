#!/usr/bin/env python3
"""Ablation test: Brain vs No-Brain.

Runs the same customer service scenario twice:
1. WITH brain — emotional state evolves and drives speech
2. WITHOUT brain — LLM gets no emotional context, just the scenario

Shows both side by side so you can see whether the brain is
actually changing the LLM's output.
"""

from __future__ import annotations

import os
import openai

from brain_llm_prototype import BrainEngine, brain_state_to_prompt

SCENARIO = (
    "You are calling customer support for your internet provider. Your internet has been "
    "down for 3 days and you work from home. You've already called twice before and each time "
    "they said it would be fixed 'within 24 hours'. You're calling for the third time now."
)
BACKGROUND = (
    "Background: You're a freelance designer with a deadline tomorrow. You've lost $500 in "
    "productivity already. You started calm but you're running out of patience. You have a "
    "direct communication style."
)

NO_BRAIN_PROMPT = f"""You are Alex, a real human being in the following situation:

{SCENARIO}

{BACKGROUND}

Respond AS Alex naturally. Keep responses 1-4 sentences.
You are a real person, not an AI. Never break character."""

TURNS = [
    {
        "event": {"type": "negative_outcome", "intensity": 0.5},
        "pre_events": [
            {"type": "negative_outcome", "intensity": 0.5},
            {"type": "negative_outcome", "intensity": 0.5},
        ],
        "says": "Thank you for calling TechNet support! My name is Jordan. How can I help you today?",
    },
    {
        "event": {"type": "betrayal", "intensity": 0.6},
        "says": "I see, let me pull up your account... Okay I can see the previous tickets. It looks like a technician was scheduled but... hmm, it seems they marked it as resolved?",
    },
    {
        "event": {"type": "insult", "intensity": 0.7},
        "says": "Have you tried restarting your router? Sometimes these issues resolve themselves if you just power cycle the equipment.",
    },
    {
        "event": {"type": "negative_outcome", "intensity": 0.8},
        "says": "I understand your frustration. Unfortunately, the earliest I can schedule a technician is... next Thursday. Would that work?",
    },
    {
        "event": {"type": "insult", "intensity": 0.6},
        "says": "Sir/Ma'am, I want to help but I need you to stay calm so we can work through this together. There's a process we need to follow.",
    },
    {
        "event": {"type": "positive_outcome", "intensity": 0.7},
        "says": "Great news — I got approval. We're sending a technician tomorrow morning between 8-10 AM, and I'm also crediting your account for the 3 days of downtime. Does that work?",
    },
]


def call_llm(system_prompt: str, history: list[dict], user_msg: str) -> str:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=200, temperature=0.9,
    )
    return r.choices[0].message.content.strip()


def main():
    print("=" * 90)
    print("  ABLATION TEST: Does the brain actually change the LLM's speech?")
    print("  Same scenario, same LLM — WITH brain state vs WITHOUT brain state")
    print("=" * 90)

    # --- WITH BRAIN ---
    brain = BrainEngine(momentum=0.30, decay_rate=0.02)
    brain_history: list[dict] = []
    brain_responses: list[str] = []

    # --- WITHOUT BRAIN ---
    nobrain_history: list[dict] = []
    nobrain_responses: list[str] = []

    for i, turn in enumerate(TURNS):
        # Process pre-events
        for pre in turn.get("pre_events", []):
            brain.process_event(pre)

        # Brain processes event
        result = brain.process_event(turn["event"])
        state = result["state"]

        # Generate WITH brain
        brain_prompt = brain_state_to_prompt(result, SCENARIO, "Alex", BACKGROUND)
        brain_resp = call_llm(brain_prompt, brain_history, turn["says"])
        brain_history.append({"role": "user", "content": turn["says"]})
        brain_history.append({"role": "assistant", "content": brain_resp})
        brain_responses.append(brain_resp)

        # Generate WITHOUT brain (same scenario, no emotional context)
        nobrain_resp = call_llm(NO_BRAIN_PROMPT, nobrain_history, turn["says"])
        nobrain_history.append({"role": "user", "content": turn["says"]})
        nobrain_history.append({"role": "assistant", "content": nobrain_resp})
        nobrain_responses.append(nobrain_resp)

        # Display
        print(f"\n{'─' * 90}")
        print(f"  TURN {i+1}")
        print(f"  Brain: regime={result['regime']}  "
              f"anger={state['anger']:.0%}  trust={state['trust']:.0%}  "
              f"stress={state['stress']:.0%}  patience={state['patience']:.0%}  "
              f"frustration={state['frustration']:.0%}")
        print(f"{'─' * 90}")
        print(f"  Support: \"{turn['says']}\"")
        print(f"\n  WITH BRAIN:    \"{brain_resp}\"")
        print(f"\n  WITHOUT BRAIN: \"{nobrain_resp}\"")

    print(f"\n\n{'═' * 90}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'═' * 90}")

    # Simple metrics
    for i, (br, nbr) in enumerate(zip(brain_responses, nobrain_responses)):
        b_len = len(br.split())
        nb_len = len(nbr.split())
        # Count "aggressive" indicators
        aggressive_words = {"unacceptable", "ridiculous", "absurd", "enough", "seriously",
                           "done", "can't", "won't", "demand", "need", "now", "immediately",
                           "fine", "whatever", "honestly"}
        b_agg = sum(1 for w in br.lower().split() if w.strip(".,!?\"'") in aggressive_words)
        nb_agg = sum(1 for w in nbr.lower().split() if w.strip(".,!?\"'") in aggressive_words)

        print(f"  Turn {i+1}: Brain words={b_len:>3d} (aggressive={b_agg})  "
              f"| NoBrain words={nb_len:>3d} (aggressive={nb_agg})")

    print(f"\n{'═' * 90}")


if __name__ == "__main__":
    main()

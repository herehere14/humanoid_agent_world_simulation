#!/usr/bin/env python3
"""Blind character-identification test for world-sim agents.

Goal:
  - Generate 20 blind samples (5 characters x 4 moments) using the current
    Heart -> action -> LLM speech pipeline.
  - Hide direct names in the generated text.
  - Ask a judge model to map each sample back to one of the candidate
    characters from action + dialogue alone.

This tests whether the agents are individually identifiable, or whether their
speech/action patterns collapse into generic "good enough" prose.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from ..action_table import Action, get_action_description
from ..scenarios import AGENTS, build_small_town
from ..world import World

# Five deliberately distinct candidates. If the system is working, these should
# be easier than average to tell apart.
CANDIDATE_AGENT_IDS = ["rosa", "chen", "jake", "diana", "richard"]

# Four shared moments across the arc = 20 total samples.
OBSERVATION_TICKS = [
    72,   # Day 4, 00:00-ish baseline / routine state
    82,   # Day 4, 10:00 layoff rumors
    106,  # Day 5, 10:00 individual terminations
    187,  # Day 8, 19:00 confrontation at the bar
]

# One extra known proper noun from scenario backgrounds.
EXTRA_REDACTIONS = ["Lily"]


@dataclass
class BlindSample:
    tick: int
    time_str: str
    true_agent_id: str
    true_agent_name: str
    action: str
    speech: str


@dataclass
class Judgment:
    tick: int
    time_str: str
    true_agent_id: str
    predicted_agent_id: str
    correct: bool
    confidence: float
    reasoning: str
    action: str
    speech: str


def _build_event_history(world: World, agent_id: str, up_to_tick: int) -> list[str]:
    agent = world.agents[agent_id]
    history = []
    for entry in agent.memory:
        if entry.tick <= up_to_tick:
            day = entry.tick // 24 + 1
            hour = entry.tick % 24
            val_marker = "+" if entry.valence_at_time > 0.5 else "-" if entry.valence_at_time < 0.4 else " "
            history.append(f"[Day {day}, {hour:02d}:00] {val_marker} {entry.description}")
    return history


def _get_nearby_names(world: World, agent_id: str) -> list[str]:
    agent = world.agents[agent_id]
    return [
        other.personality.name
        for other in world.agents.values()
        if other.agent_id != agent_id and other.location == agent.location
    ]


def _all_name_tokens() -> list[str]:
    tokens: set[str] = set(EXTRA_REDACTIONS)
    for personality in AGENTS:
        tokens.add(personality.name)
        for token in personality.name.split():
            if len(token) >= 3:
                tokens.add(token)
    return sorted(tokens, key=len, reverse=True)


NAME_TOKENS = _all_name_tokens()


def _redact_names(text: str) -> str:
    redacted = text
    for token in NAME_TOKENS:
        redacted = re.sub(rf"\b{re.escape(token)}\b", "[NAME]", redacted, flags=re.IGNORECASE)
    return redacted


def _call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int,
) -> str:
    # GPT-5 mini occasionally returns an empty text payload on short prompts.
    # Retry once with a larger output budget before giving up.
    budgets = [max_output_tokens, max(max_output_tokens, 220)]
    for budget in budgets:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=budget,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        text = (resp.output_text or "").strip()
        if text:
            return text
    return ""


def _parse_json_object(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None

    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def generate_heart_sample(client: OpenAI, world: World, agent_id: str, model: str) -> BlindSample:
    agent = world.agents[agent_id]
    s = agent.heart
    nearby = _get_nearby_names(world, agent_id)
    memories = _build_event_history(world, agent_id, world.tick_count)

    prompt = f"""You are {agent.personality.name}, {agent.personality.background}
Temperament: {agent.personality.temperament}

Current emotional state:
- Internal feeling: {s.internal_emotion} (arousal: {s.arousal:.2f}, valence: {s.valence:.2f})
- What you show outwardly: {s.surface_emotion} (divergence: {s.divergence:.2f})
- Tension: {s.tension:.2f}, Energy: {s.energy:.2f}, Vulnerability: {s.vulnerability:.2f}
- Impulse control: {s.impulse_control:.2f}
- Active emotional wounds: {len(s.wounds)}

You are currently: {get_action_description(Action[agent.last_action], agent)}
Location: {agent.location}
Time: {world.time_str}

Recent events:
{chr(10).join(memories[-8:]) if memories else "Nothing notable."}

People nearby: {", ".join(nearby) if nearby else "No one"}

Given your emotional state and what you are doing, write 1-2 sentences of what you say or do right now.
Stay in character. If your surface emotion differs from your internal emotion, reflect that tension.
Do not narrate. Write only dialogue or brief action."""

    raw = _call_model(client, model, prompt, max_output_tokens=180)
    speech = _redact_names(raw)
    return BlindSample(
        tick=world.tick_count,
        time_str=world.time_str,
        true_agent_id=agent_id,
        true_agent_name=agent.personality.name,
        action=agent.last_action,
        speech=speech,
    )


def _candidate_card(agent_id: str, world: World) -> str:
    agent = world.agents[agent_id]
    p = agent.personality
    return f"- {agent_id}: {p.name}. Background: {p.background} Temperament: {p.temperament}"


def judge_sample(client: OpenAI, model: str, world: World, sample: BlindSample) -> Judgment:
    candidate_cards = "\n".join(_candidate_card(agent_id, world) for agent_id in CANDIDATE_AGENT_IDS)
    prompt = f"""You are running a blind identity test for simulated characters.

Below is the candidate cast. Each sample comes from exactly one of these characters.

{candidate_cards}

Blind sample:
- Action: {sample.action}
- Speech or behavior: "{sample.speech}"

Pick the most likely character.
Base your choice on coping style, emotional signature, social stance, wording, and what kind of person would behave this way.

Return only valid JSON in this format:
{{
  "predicted_agent_id": "<one of: {", ".join(CANDIDATE_AGENT_IDS)}>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<1-2 sentences>"
}}"""

    raw = _call_model(client, model, prompt, max_output_tokens=420)
    payload = _parse_json_object(raw)
    if payload is None:
        payload = {
            "predicted_agent_id": "",
            "confidence": 0.0,
            "reasoning": f"Unparseable judge output: {raw[:180]}",
        }

    predicted = str(payload.get("predicted_agent_id", "")).strip().lower()
    if predicted not in CANDIDATE_AGENT_IDS:
        predicted = "invalid"

    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return Judgment(
        tick=sample.tick,
        time_str=sample.time_str,
        true_agent_id=sample.true_agent_id,
        predicted_agent_id=predicted,
        correct=predicted == sample.true_agent_id,
        confidence=confidence,
        reasoning=str(payload.get("reasoning", "")),
        action=sample.action,
        speech=sample.speech,
    )


def run_test(
    *,
    generation_model: str,
    judge_model: str,
    output_path: str,
) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    world = build_small_town()
    world.initialize()

    max_tick = max(OBSERVATION_TICKS)
    samples: list[BlindSample] = []

    print(f"Generating blind samples with {generation_model}...")
    for tick in range(1, max_tick + 1):
        world.tick()
        if tick not in OBSERVATION_TICKS:
            continue

        print(f"  Tick {tick} / {world.time_str}")
        for agent_id in CANDIDATE_AGENT_IDS:
            sample = generate_heart_sample(client, world, agent_id, generation_model)
            samples.append(sample)
            print(f"    {agent_id:<8s} -> {sample.action:<12s} {sample.speech[:70]}")

    print(f"\nJudging {len(samples)} blind samples with {judge_model}...")
    judgments: list[Judgment] = []
    for idx, sample in enumerate(samples, start=1):
        judgment = judge_sample(client, judge_model, world, sample)
        judgments.append(judgment)
        status = "OK" if judgment.correct else "MISS"
        print(
            f"  [{idx:02d}/{len(samples)}] {sample.time_str} / {sample.true_agent_id:<8s} "
            f"-> {judgment.predicted_agent_id:<8s} {status}"
        )

    accuracy = sum(1 for j in judgments if j.correct) / len(judgments)
    by_agent = {}
    confusion: dict[str, dict[str, int]] = {}
    for agent_id in CANDIDATE_AGENT_IDS:
        agent_results = [j for j in judgments if j.true_agent_id == agent_id]
        by_agent[agent_id] = {
            "accuracy": round(sum(1 for j in agent_results if j.correct) / len(agent_results), 3),
            "n": len(agent_results),
        }
        confusion[agent_id] = {}
        for pred_id in CANDIDATE_AGENT_IDS + ["invalid"]:
            count = sum(1 for j in agent_results if j.predicted_agent_id == pred_id)
            if count:
                confusion[agent_id][pred_id] = count

    summary = {
        "generation_model": generation_model,
        "judge_model": judge_model,
        "candidate_agent_ids": CANDIDATE_AGENT_IDS,
        "observation_ticks": OBSERVATION_TICKS,
        "sample_count": len(samples),
        "chance_accuracy": round(1 / len(CANDIDATE_AGENT_IDS), 3),
        "accuracy": round(accuracy, 3),
        "by_agent": by_agent,
        "confusion": confusion,
        "samples": [asdict(s) for s in samples],
        "judgments": [asdict(j) for j in judgments],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n{'=' * 76}")
    print("BLIND CHARACTER IDENTITY TEST")
    print(f"{'=' * 76}")
    print(f"Generation model: {summary['generation_model']}")
    print(f"Judge model:      {summary['judge_model']}")
    print(f"Candidates:       {', '.join(summary['candidate_agent_ids'])}")
    print(f"Samples:          {summary['sample_count']}")
    print(f"Chance accuracy:  {summary['chance_accuracy'] * 100:.1f}%")
    print(f"Measured accuracy:{summary['accuracy'] * 100:.1f}%")

    print("\nPer-agent accuracy:")
    for agent_id, stats in summary["by_agent"].items():
        print(f"  {agent_id:<8s} {stats['accuracy'] * 100:>5.1f}% ({stats['n']} samples)")

    print("\nConfusion:")
    for agent_id, row in summary["confusion"].items():
        pairs = ", ".join(f"{pred}:{count}" for pred, count in sorted(row.items()))
        print(f"  {agent_id:<8s} {pairs}")
    print(f"{'=' * 76}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument(
        "--output",
        default="artifacts/character_identity_blind_test.json",
    )
    args = parser.parse_args()

    summary = run_test(
        generation_model=args.generation_model,
        judge_model=args.judge_model,
        output_path=args.output,
    )
    print_summary(summary)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

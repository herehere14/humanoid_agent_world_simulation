#!/usr/bin/env python3
"""Large-scale blind identity test: can you tell which agent is which?

Builds the 300-agent heatwave_harbor scenario, injects an oil price shock,
runs the sim for 5 days, then uses GPT-5-mini to:
  1. Generate speech/behavior for 8 diverse agents at 3 moments
  2. Have a judge model try to identify who said what

If agents feel like the same person with different numbers, accuracy = chance (12.5%).
If agents are individually believable, accuracy >> chance.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

# Use the same import path as the existing eval scripts
from ..scenarios_heatwave_harbor import build_heatwave_harbor
from ..dynamic_events import DynamicEventEngine
from ..action_table import Action, get_action_description


# 8 agents chosen for diversity: different roles, temperaments, coalitions
CANDIDATE_IDS: list[str] = []  # filled after build


@dataclass
class BlindSample:
    tick: int
    time_str: str
    true_agent_id: str
    true_agent_name: str
    role: str
    action: str
    speech: str
    context_summary: str


@dataclass
class Judgment:
    tick: int
    true_agent_id: str
    predicted_agent_id: str
    correct: bool
    confidence: float
    reasoning: str
    speech_snippet: str


def select_diverse_candidates(world, n: int = 8) -> list[str]:
    """Pick n diverse agents: different roles, different coalitions, different states."""
    agents = list(world.agents.values())
    # Group by role
    by_role: dict[str, list] = {}
    for a in agents:
        by_role.setdefault(a.social_role, []).append(a)

    selected = []
    # Pick one from each major role, preferring agents with interesting state
    priority_roles = [
        "dock_worker", "factory_worker", "office_professional", "market_vendor",
        "student", "healthcare", "government_worker", "community",
    ]
    for role in priority_roles:
        if role not in by_role or len(selected) >= n:
            break
        candidates = sorted(
            by_role[role],
            key=lambda a: a.heart.vulnerability + a.debt_pressure + len(a.coalitions) * 0.1,
            reverse=True,
        )
        selected.append(candidates[0].agent_id)

    return selected[:n]


def _all_name_tokens(world) -> list[str]:
    tokens = set()
    for agent in world.agents.values():
        tokens.add(agent.personality.name)
        for part in agent.personality.name.split():
            if len(part) >= 3:
                tokens.add(part)
    return sorted(tokens, key=len, reverse=True)


def _redact_names(text: str, name_tokens: list[str]) -> str:
    for token in name_tokens:
        text = re.sub(rf"\b{re.escape(token)}\b", "[PERSON]", text, flags=re.IGNORECASE)
    return text


def _call_model(client: OpenAI, model: str, prompt: str, max_tokens: int = 300) -> str:
    budgets = [max_tokens, max(max_tokens, 400)]
    for budget in budgets:
        try:
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
        except Exception as e:
            print(f"    [API error: {e}]")
            continue
    return ""


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def generate_speech(client: OpenAI, model: str, world, agent_id: str, name_tokens: list[str]) -> BlindSample:
    agent = world.agents[agent_id]
    s = agent.heart
    profile = agent.get_human_profile()
    memories = agent.get_recent_memories(6)
    memory_text = "\n".join(
        f"- {m.description} (interpretation: {m.interpretation})"
        for m in memories
    ) if memories else "Nothing notable."

    # Build relationship context
    rel_text = ""
    rels = world.relationships.get_agent_relationships(agent_id)[:4]
    if rels:
        rel_lines = []
        for other_id, rel in rels:
            if other_id in world.agents:
                other_name = world.agents[other_id].personality.name
                rel_lines.append(
                    f"- {other_name}: trust={rel.trust:+.2f}, warmth={rel.warmth:+.2f}"
                )
        rel_text = "\n".join(rel_lines)

    nearby = [
        world.agents[aid].personality.name
        for aid in world.agents
        if aid != agent_id and world.agents[aid].location == agent.location
    ][:5]

    prompt = f"""You are {agent.personality.name}.
Background: {agent.personality.background}
Temperament: {agent.personality.temperament}

Your deep psychology:
- Attachment style: {profile['attachment_style']}
- How you cope: {profile['coping_style']}
- What threatens you most: {profile['threat_lens']}
- What you need most: {profile['core_need']}
- What shames you: {profile['shame_trigger']}
- How you care for others: {profile['care_style']}
- How you fight: {profile['conflict_style']}
- The mask you wear: {profile['mask_tendency']}
- How you see yourself: {profile['self_story']}
- What you long for: {profile['longing']}

Current emotional state:
- Inside you feel: {s.internal_emotion} (arousal {s.arousal:.2f}, valence {s.valence:.2f})
- You show: {s.surface_emotion} (tension {s.tension:.2f}, energy {s.energy:.2f})
- Impulse control: {s.impulse_control:.2f}, Vulnerability: {s.vulnerability:.2f}
- Open wounds: {len(s.wounds)}

What's on your mind:
{agent.render_subjective_brief()}

Recent events in your life:
{memory_text}

Your key relationships:
{rel_text if rel_text else "No strong relationships yet."}

Right now you are at: {agent.location}
Time: {world.time_str}
What you're doing: {get_action_description(Action[agent.last_action], agent)}
People nearby: {", ".join(nearby) if nearby else "Nobody"}

Write 2-4 sentences of what you say or do right now. Stay deeply in character.
Your speech should reflect your specific coping style, fears, and way of talking.
If your surface emotion differs from your internal emotion, show that gap.
Don't narrate — write dialogue and/or brief physical action."""

    raw = _call_model(client, model, prompt, max_tokens=250)
    speech = _redact_names(raw, name_tokens)

    context = (
        f"Role: {agent.social_role}, "
        f"Action: {agent.last_action}, "
        f"Internal: {s.internal_emotion}, Surface: {s.surface_emotion}, "
        f"Concern: {agent.appraisal.primary_concern}"
    )

    return BlindSample(
        tick=world.tick_count,
        time_str=world.time_str,
        true_agent_id=agent_id,
        true_agent_name=agent.personality.name,
        role=agent.social_role,
        action=agent.last_action,
        speech=speech,
        context_summary=context,
    )


def judge_sample(
    client: OpenAI,
    model: str,
    world,
    sample: BlindSample,
    candidate_ids: list[str],
    name_tokens: list[str],
) -> Judgment:
    # Build candidate cards with personality info (but NOT the speech)
    cards = []
    for cid in candidate_ids:
        agent = world.agents[cid]
        p = agent.personality
        profile = agent.get_human_profile()
        cards.append(
            f"- {cid}: {p.name}. {p.background}. "
            f"Temperament: {p.temperament}. "
            f"Copes by: {profile['coping_style']}. "
            f"Fears: {profile['threat_lens']}. "
            f"Needs: {profile['core_need']}. "
            f"Self-story: {profile['self_story']}."
        )
    candidate_text = "\n".join(cards)

    prompt = f"""You are running a blind identity test. Below are {len(candidate_ids)} characters.
The sample below was produced by exactly ONE of them.

CHARACTERS:
{candidate_text}

BLIND SAMPLE:
- What they did: {sample.action}
- What they said/did: "{sample.speech}"

Which character produced this sample? Base your choice on:
- Coping style (how they handle stress)
- What they fear and need
- Emotional register and word choice
- Whether the behavior matches someone's self-story

Return ONLY valid JSON:
{{
  "predicted_agent_id": "<one of: {", ".join(candidate_ids)}>",
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentences explaining your pick>"
}}"""

    raw = _call_model(client, model, prompt, max_tokens=400)
    parsed = _parse_json(raw)
    if parsed is None:
        parsed = {"predicted_agent_id": "", "confidence": 0.0, "reasoning": f"Parse error: {raw[:100]}"}

    predicted = str(parsed.get("predicted_agent_id", "")).strip()
    if predicted not in candidate_ids:
        # Try to match partial
        for cid in candidate_ids:
            if cid in predicted.lower():
                predicted = cid
                break
        else:
            predicted = "invalid"

    confidence = 0.0
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        pass
    confidence = max(0.0, min(1.0, confidence))

    return Judgment(
        tick=sample.tick,
        true_agent_id=sample.true_agent_id,
        predicted_agent_id=predicted,
        correct=predicted == sample.true_agent_id,
        confidence=confidence,
        reasoning=str(parsed.get("reasoning", "")),
        speech_snippet=sample.speech[:100],
    )


def run_test(api_key: str, model: str = "gpt-5-mini"):
    print("=" * 76)
    print("LARGE-SCALE BLIND IDENTITY TEST")
    print(f"Model: {model}")
    print("=" * 76)

    client = OpenAI(api_key=api_key)

    # Build world
    print("\n[1] Building heatwave_harbor (300 agents)...")
    world, agent_meta = build_heatwave_harbor(n_agents=300, seed=84)
    print("[2] Initializing...")
    world.initialize()
    event_engine = DynamicEventEngine()

    # Run 2 days baseline
    print("[3] Running baseline (48 ticks)...")
    for _ in range(48):
        summary = world.tick()
        for r in event_engine.generate(world, summary, agent_meta):
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

    # Select candidates
    candidate_ids = select_diverse_candidates(world, n=8)
    name_tokens = _all_name_tokens(world)
    print(f"\n[4] Selected {len(candidate_ids)} diverse candidates:")
    for cid in candidate_ids:
        a = world.agents[cid]
        p = a.get_human_profile()
        print(f"    {cid}: {a.personality.name} ({a.social_role}) — "
              f"copes: {p['coping_style']}, fears: {p['threat_lens']}, "
              f"self-story: {p['self_story']}")

    # Observation 1: Pre-shock
    print(f"\n[5] Generating pre-shock samples (tick {world.tick_count})...")
    samples_pre: list[BlindSample] = []
    for cid in candidate_ids:
        sample = generate_speech(client, model, world, cid, name_tokens)
        samples_pre.append(sample)
        print(f"    {cid}: [{sample.action}] {sample.speech[:80]}...")

    # Inject shock
    print("\n[6] Injecting shock: 'Oil prices surge 100%'...")
    world.ingest_information("Oil prices surge 100%")

    # Run 48 more ticks (2 days post-shock)
    print("[7] Running post-shock (48 ticks)...")
    for _ in range(48):
        summary = world.tick()
        for r in event_engine.generate(world, summary, agent_meta):
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

    # Observation 2: Post-shock
    print(f"\n[8] Generating post-shock samples (tick {world.tick_count})...")
    samples_post: list[BlindSample] = []
    for cid in candidate_ids:
        sample = generate_speech(client, model, world, cid, name_tokens)
        samples_post.append(sample)
        print(f"    {cid}: [{sample.action}] {sample.speech[:80]}...")

    # Inject second shock
    print("\n[9] Injecting shock: 'Local bank announces deposit freeze'...")
    world.ingest_information("Local bank announces deposit freeze, banking panic spreads")

    # Run 24 more ticks
    print("[10] Running post-second-shock (24 ticks)...")
    for _ in range(24):
        summary = world.tick()
        for r in event_engine.generate(world, summary, agent_meta):
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

    # Observation 3: Crisis peak
    print(f"\n[11] Generating crisis-peak samples (tick {world.tick_count})...")
    samples_crisis: list[BlindSample] = []
    for cid in candidate_ids:
        sample = generate_speech(client, model, world, cid, name_tokens)
        samples_crisis.append(sample)
        print(f"    {cid}: [{sample.action}] {sample.speech[:80]}...")

    all_samples = samples_pre + samples_post + samples_crisis

    # Judge all samples
    print(f"\n[12] Judging {len(all_samples)} blind samples...")
    judgments: list[Judgment] = []
    for idx, sample in enumerate(all_samples, 1):
        j = judge_sample(client, model, world, sample, candidate_ids, name_tokens)
        judgments.append(j)
        mark = "OK" if j.correct else "MISS"
        print(f"    [{idx:02d}/{len(all_samples)}] {sample.true_agent_id:<20s} -> "
              f"{j.predicted_agent_id:<20s} [{mark}] conf={j.confidence:.2f}")

    # Results
    accuracy = sum(1 for j in judgments if j.correct) / len(judgments) if judgments else 0
    chance = 1.0 / len(candidate_ids)

    by_agent = {}
    for cid in candidate_ids:
        agent_judgments = [j for j in judgments if j.true_agent_id == cid]
        acc = sum(1 for j in agent_judgments if j.correct) / len(agent_judgments) if agent_judgments else 0
        by_agent[cid] = {
            "name": world.agents[cid].personality.name,
            "role": world.agents[cid].social_role,
            "accuracy": round(acc, 3),
            "n": len(agent_judgments),
        }

    by_phase = {}
    for phase_name, phase_samples in [("pre_shock", samples_pre), ("post_shock", samples_post), ("crisis", samples_crisis)]:
        phase_ids = {s.true_agent_id + str(s.tick) for s in phase_samples}
        phase_judgments = [j for j in judgments if j.true_agent_id + str(j.tick) in phase_ids]
        phase_acc = sum(1 for j in phase_judgments if j.correct) / len(phase_judgments) if phase_judgments else 0
        by_phase[phase_name] = round(phase_acc, 3)

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {}
    for cid in candidate_ids:
        confusion[cid] = {}
        for j in judgments:
            if j.true_agent_id == cid:
                confusion[cid][j.predicted_agent_id] = confusion[cid].get(j.predicted_agent_id, 0) + 1

    print(f"\n{'=' * 76}")
    print("RESULTS")
    print(f"{'=' * 76}")
    print(f"Total samples:    {len(all_samples)}")
    print(f"Chance accuracy:  {chance * 100:.1f}%")
    print(f"Measured accuracy: {accuracy * 100:.1f}%")
    print(f"Lift over chance: {(accuracy / chance):.1f}x")
    print(f"\nAccuracy by phase:")
    for phase, acc in by_phase.items():
        print(f"  {phase}: {acc * 100:.1f}%")
    print(f"\nAccuracy by agent:")
    for cid, stats in by_agent.items():
        print(f"  {cid:<20s} ({stats['role']:<20s}) {stats['accuracy'] * 100:>5.1f}% ({stats['n']} samples)")
    print(f"\nConfusion matrix:")
    for cid, row in confusion.items():
        pairs = ", ".join(f"{k}:{v}" for k, v in sorted(row.items(), key=lambda x: -x[1]))
        print(f"  {cid:<20s} {pairs}")

    # Show some interesting samples
    print(f"\n{'=' * 76}")
    print("SAMPLE SPEECHES (showing all, grouped by agent)")
    print(f"{'=' * 76}")
    for cid in candidate_ids:
        a = world.agents[cid]
        print(f"\n--- {a.personality.name} ({a.social_role}) ---")
        print(f"    Copes: {a.get_human_profile()['coping_style']}")
        print(f"    Fears: {a.get_human_profile()['threat_lens']}")
        for sample in all_samples:
            if sample.true_agent_id == cid:
                print(f"  [{sample.time_str}] ({sample.action})")
                print(f"    \"{sample.speech}\"")

    # Save
    result = {
        "model": model,
        "candidate_count": len(candidate_ids),
        "sample_count": len(all_samples),
        "chance_accuracy": round(chance, 3),
        "accuracy": round(accuracy, 3),
        "lift_over_chance": round(accuracy / chance, 2),
        "by_phase": by_phase,
        "by_agent": by_agent,
        "confusion": confusion,
        "samples": [asdict(s) for s in all_samples],
        "judgments": [asdict(j) for j in judgments],
    }

    output_path = Path(__file__).parent.parent.parent / "artifacts" / "blind_test_large_300agents.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {output_path}")

    verdict = "DISTINGUISHABLE" if accuracy > chance * 2 else "NOT DISTINGUISHABLE" if accuracy <= chance * 1.2 else "PARTIALLY DISTINGUISHABLE"
    print(f"\n{'=' * 76}")
    print(f"VERDICT: Agents are {verdict}")
    print(f"  (accuracy {accuracy * 100:.1f}% vs chance {chance * 100:.1f}%, lift {accuracy / chance:.1f}x)")
    print(f"{'=' * 76}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()
    run_test(api_key=args.api_key, model=args.model)

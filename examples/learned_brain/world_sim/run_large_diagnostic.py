#!/usr/bin/env python3
"""Run a 50-day diagnostic simulation for the 300-agent large town.

Outputs:
  - Markdown report with high-signal findings, sampled communications, and gaps
  - JSON artifact with daily summaries and final state for all agents

The goal is not to narrate every single utterance. It is to:
  1. Simulate the whole town for a long horizon
  2. Capture what every agent is privately oriented around
  3. Sample representative communications from the most dramatic interactions
  4. Show where the current system still flattens or stalls
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from openai import OpenAI

from .dynamic_events import DISTRICT_MAP, DynamicEventEngine, compute_district_stats
from .relationship import RelationshipVector
from .scenarios_heatwave_harbor import build_heatwave_harbor
from .scenarios_large import build_large_town
from .world import World


SCENARIO_BUILDERS = {
    "large_town": build_large_town,
    "heatwave_harbor": build_heatwave_harbor,
}

SCENARIO_LABELS = {
    "large_town": "Crossroads Chemical Crisis",
    "heatwave_harbor": "Harbor Heatwave and Buyout Crisis",
}


@dataclass
class InteractionSnapshot:
    score: float
    tick: int
    time_str: str
    location: str
    district: str
    interaction_type: str
    agent_a_id: str
    agent_b_id: str
    agent_a_name: str
    agent_b_name: str
    agent_a_role: str
    agent_b_role: str
    agent_a_brief: str
    agent_b_brief: str
    agent_a_action: str
    agent_b_action: str
    agent_a_memories: list[str]
    agent_b_memories: list[str]
    relationship_summary: str


def _call_model(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 260) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )
    return (resp.output_text or "").strip()


def _recent_memories(agent, n: int = 5) -> list[str]:
    lines = []
    for m in agent.get_recent_memories(n):
        day = m.tick // 24 + 1
        hour = m.tick % 24
        lines.append(f"[Day {day}, {hour:02d}:00] {m.description}")
    return lines


def _relationship_summary(world: World, aid_a: str, aid_b: str) -> str:
    rel = world.relationships.get(aid_a, aid_b)
    if not rel:
        return "No prior relationship."
    res_ab = world.relationships.get_resentment(aid_a, aid_b)
    res_ba = world.relationships.get_resentment(aid_b, aid_a)
    parts = [
        f"trust={rel.trust:+.2f}",
        f"warmth={rel.warmth:+.2f}",
        f"familiarity={rel.familiarity}",
        f"issue={rel.last_issue}",
        f"alliance={rel.alliance_strength:+.2f}",
        f"rivalry={rel.rivalry:.2f}",
    ]
    if rel.support_events:
        parts.append(f"support={rel.support_events}")
    if rel.conflict_events:
        parts.append(f"conflict={rel.conflict_events}")
    if rel.practical_help_events:
        parts.append(f"practical_help={rel.practical_help_events}")
    if world.relationships.get_grievance(aid_a, aid_b) > 0.05:
        parts.append(f"{aid_a}→{aid_b} grievance={world.relationships.get_grievance(aid_a, aid_b):.2f}")
    if world.relationships.get_grievance(aid_b, aid_a) > 0.05:
        parts.append(f"{aid_b}→{aid_a} grievance={world.relationships.get_grievance(aid_b, aid_a):.2f}")
    if world.relationships.get_debt(aid_a, aid_b) > 0.05:
        parts.append(f"{aid_a} owes {aid_b}={world.relationships.get_debt(aid_a, aid_b):.2f}")
    if world.relationships.get_debt(aid_b, aid_a) > 0.05:
        parts.append(f"{aid_b} owes {aid_a}={world.relationships.get_debt(aid_b, aid_a):.2f}")
    if res_ab > 0.05:
        parts.append(f"{aid_a}→{aid_b} resentment={res_ab:.2f}")
    if res_ba > 0.05:
        parts.append(f"{aid_b}→{aid_a} resentment={res_ba:.2f}")
    return ", ".join(parts)


def _interaction_score(world: World, ix: dict, agent_meta: dict[str, dict]) -> float:
    a = world.agents[ix["agent_a"]]
    b = world.agents[ix["agent_b"]]
    score = (
        a.heart.vulnerability + b.heart.vulnerability +
        (1.0 - a.heart.valence) + (1.0 - b.heart.valence) +
        a.heart.arousal + b.heart.arousal
    )
    if agent_meta[ix["agent_a"]]["role"] != agent_meta[ix["agent_b"]]["role"]:
        score += 1.0
    if ix["type"] == "conflict":
        score += 0.8
    if ix["type"] == "support":
        score += 0.35
    return score


def _capture_interaction_snapshot(world: World, ix: dict, agent_meta: dict[str, dict]) -> InteractionSnapshot:
    a = world.agents[ix["agent_a"]]
    b = world.agents[ix["agent_b"]]
    district = DISTRICT_MAP.get(a.location, "Unknown")
    return InteractionSnapshot(
        score=_interaction_score(world, ix, agent_meta),
        tick=world.tick_count,
        time_str=world.time_str,
        location=a.location,
        district=district,
        interaction_type=ix["type"],
        agent_a_id=a.agent_id,
        agent_b_id=b.agent_id,
        agent_a_name=a.personality.name,
        agent_b_name=b.personality.name,
        agent_a_role=agent_meta[a.agent_id]["role"],
        agent_b_role=agent_meta[b.agent_id]["role"],
        agent_a_brief=a.render_subjective_brief(),
        agent_b_brief=b.render_subjective_brief(),
        agent_a_action=a.last_action,
        agent_b_action=b.last_action,
        agent_a_memories=_recent_memories(a, 4),
        agent_b_memories=_recent_memories(b, 4),
        relationship_summary=_relationship_summary(world, a.agent_id, b.agent_id),
    )


def _dialogue_prompt(snapshot: InteractionSnapshot) -> str:
    return f"""Two people in a city-wide crisis are interacting.

TIME: {snapshot.time_str}
PLACE: {snapshot.location} ({snapshot.district})
Interaction type: {snapshot.interaction_type}

PERSON A:
Name: {snapshot.agent_a_name}
Role: {snapshot.agent_a_role}
Action: {snapshot.agent_a_action}
Private state:
{snapshot.agent_a_brief}
Recent memories:
{chr(10).join(snapshot.agent_a_memories) if snapshot.agent_a_memories else "No notable recent memories."}

PERSON B:
Name: {snapshot.agent_b_name}
Role: {snapshot.agent_b_role}
Action: {snapshot.agent_b_action}
Private state:
{snapshot.agent_b_brief}
Recent memories:
{chr(10).join(snapshot.agent_b_memories) if snapshot.agent_b_memories else "No notable recent memories."}

Relationship:
{snapshot.relationship_summary}

Write 4-6 short lines of grounded dialogue followed by one short line of subtext.
Format:
{snapshot.agent_a_name}: "..."
{snapshot.agent_b_name}: "..."
[Subtext: ...]"""


def _agent_export(world: World, agent_id: str, agent_meta: dict[str, dict]) -> dict:
    agent = world.agents[agent_id]
    rels = world.relationships.get_agent_relationships(agent_id)[:5]
    return {
        "agent_id": agent.agent_id,
        "name": agent.personality.name,
        "role": agent_meta.get(agent_id, {}).get("role", "unknown"),
        "location": agent.location,
        "action": agent.last_action,
        "llm_salience": round(agent.llm_salience, 3),
        "llm_salience_level": agent.llm_salience_level,
        "llm_active": agent.llm_active,
        "llm_candidate_rank": agent.llm_candidate_rank,
        "llm_salience_reasons": list(agent.llm_salience_reasons),
        "llm_salience_factors": {
            key: round(value, 3) for key, value in agent.llm_salience_factors.items()
        },
        "llm_packet_preview": agent.llm_packet_preview,
        "heart": {
            "arousal": round(agent.heart.arousal, 3),
            "valence": round(agent.heart.valence, 3),
            "tension": round(agent.heart.tension, 3),
            "impulse_control": round(agent.heart.impulse_control, 3),
            "energy": round(agent.heart.energy, 3),
            "vulnerability": round(agent.heart.vulnerability, 3),
            "internal_emotion": agent.heart.internal_emotion,
            "surface_emotion": agent.heart.surface_emotion,
            "wounds": len(agent.heart.wounds),
        },
        "subjective_brief": agent.render_subjective_brief(),
        "future_branches": agent.get_future_branches(),
        "recent_memories": _recent_memories(agent, 6),
        "coalitions": list(agent.coalitions),
        "identity_tags": list(agent.identity_tags),
        "private_burden": agent.private_burden,
        "debt_pressure": round(agent.debt_pressure, 3),
        "secret_pressure": round(agent.secret_pressure, 3),
        "ambition": round(agent.ambition, 3),
        "top_relationships": [
            {
                "other_id": other_id,
                "other_name": world.agents[other_id].personality.name if other_id in world.agents else other_id,
                "trust": round(rel.trust, 3),
                "warmth": round(rel.warmth, 3),
                "resentment_toward": round(world.relationships.get_resentment(agent_id, other_id), 3),
                "resentment_from": round(world.relationships.get_resentment(other_id, agent_id), 3),
                "grievance_toward": round(world.relationships.get_grievance(agent_id, other_id), 3),
                "grievance_from": round(world.relationships.get_grievance(other_id, agent_id), 3),
                "debt_toward": round(world.relationships.get_debt(agent_id, other_id), 3),
                "debt_from": round(world.relationships.get_debt(other_id, agent_id), 3),
                "familiarity": rel.familiarity,
                "support_events": rel.support_events,
                "conflict_events": rel.conflict_events,
                "practical_help_events": rel.practical_help_events,
                "alliance_strength": round(rel.alliance_strength, 3),
                "rivalry": round(rel.rivalry, 3),
                "betrayal_events": rel.betrayal_events,
                "last_issue": rel.last_issue,
            }
            for other_id, rel in rels
        ],
    }


def _pair_metrics(rel: RelationshipVector, world: World, a: str, b: str) -> dict:
    return {
        "pair": [a, b],
        "names": [
            world.agents[a].personality.name if a in world.agents else a,
            world.agents[b].personality.name if b in world.agents else b,
        ],
        "trust": round(rel.trust, 3),
        "warmth": round(rel.warmth, 3),
        "familiarity": rel.familiarity,
        "resentment_ab": round(rel.resentment_ab, 3),
        "resentment_ba": round(rel.resentment_ba, 3),
        "grievance_ab": round(rel.grievance_ab, 3),
        "grievance_ba": round(rel.grievance_ba, 3),
        "debt_ab": round(rel.debt_ab, 3),
        "debt_ba": round(rel.debt_ba, 3),
        "support_events": rel.support_events,
        "conflict_events": rel.conflict_events,
        "practical_help_events": rel.practical_help_events,
        "alliance_strength": round(rel.alliance_strength, 3),
        "rivalry": round(rel.rivalry, 3),
        "betrayal_events": rel.betrayal_events,
        "last_issue": rel.last_issue,
    }


def run_diagnostic(
    *,
    days: int,
    output_md: str,
    output_json: str,
    llm_samples: int,
    llm_model: str,
    scenario: str,
    external_information: list[str] | None = None,
) -> dict:
    if scenario not in SCENARIO_BUILDERS:
        raise ValueError(f"Unknown scenario: {scenario}")
    world, agent_meta = SCENARIO_BUILDERS[scenario](n_agents=300, seed=42)
    baseline_pairs = {
        pair: RelationshipVector(
            trust=rel.trust,
            warmth=rel.warmth,
            resentment_ab=rel.resentment_ab,
            resentment_ba=rel.resentment_ba,
            familiarity=rel.familiarity,
            last_interaction=rel.last_interaction,
            support_events=rel.support_events,
            conflict_events=rel.conflict_events,
            practical_help_events=rel.practical_help_events,
            alliance_strength=rel.alliance_strength,
            last_issue=rel.last_issue,
            grievance_ab=rel.grievance_ab,
            grievance_ba=rel.grievance_ba,
            debt_ab=rel.debt_ab,
            debt_ba=rel.debt_ba,
            rivalry=rel.rivalry,
            betrayal_events=rel.betrayal_events,
        )
        for pair, rel in world.relationships._pairs.items()  # noqa: SLF001
    }
    base_pair_count = world.relationships.pair_count

    world.initialize()
    event_engine = DynamicEventEngine()
    world.event_engine = event_engine  # expose for reporting helpers
    injected_signals = []
    for info in external_information or []:
        injected_signals.append(world.ingest_information(info))

    total_ticks = days * 24
    start = time.time()
    daily_summaries: list[dict] = []
    checkpoint_days = {1, 3, 5, 7, 10, 20, 30, 40, 50}
    checkpoints: dict[str, dict[str, dict]] = {}
    cumulative_action_counts: Counter[str] = Counter()
    cumulative_interaction_types: Counter[str] = Counter()
    total_interactions = 0
    ripple_events_added = 0
    fired_events = 0
    generated_dynamic_counts: Counter[str] = Counter()
    fired_dynamic_counts: Counter[str] = Counter()
    top_snapshots: list[tuple[float, int, int, InteractionSnapshot]] = []
    snapshot_counter = 0

    for tick in range(1, total_ticks + 1):
        summary = world.tick()
        fired_events += len(summary.get("events", []))
        for event in summary.get("events", []):
            kind = event.get("kind", "scheduled")
            if kind != "scheduled":
                fired_dynamic_counts[kind] += 1

        ripples = event_engine.generate(world, summary, agent_meta)
        for ripple in ripples:
            ripple.description = f"[{ripple.kind}] {ripple.description}"
            world.schedule_event(ripple)
            generated_dynamic_counts[ripple.kind] += 1
        ripple_events_added += len(ripples)

        for action_data in summary.get("actions", {}).values():
            cumulative_action_counts[action_data["action"]] += 1

        interactions = summary.get("interactions", [])
        total_interactions += len(interactions)
        for ix in interactions:
            cumulative_interaction_types[ix["type"]] += 1
            snapshot = _capture_interaction_snapshot(world, ix, agent_meta)
            snapshot_counter += 1
            heapq.heappush(top_snapshots, (snapshot.score, snapshot.tick, snapshot_counter, snapshot))
            if len(top_snapshots) > max(llm_samples * 4, 80):
                heapq.heappop(top_snapshots)

        if world.hour_of_day == 23:
            day = world.day
            district_stats = compute_district_stats(world)
            top_distressed = sorted(
                world.agents.values(),
                key=lambda a: (a.heart.vulnerability, len(a.heart.wounds), 1.0 - a.heart.valence),
                reverse=True,
            )[:8]
            day_summary = {
                "day": day,
                "time": world.time_str,
                "events_fired_today": len([e for e in world.tick_log[-24:] if e.get("events")]),
                "interaction_count_today": sum(len(t.get("interactions", [])) for t in world.tick_log[-24:]),
                "avg_valence": round(mean(a.heart.valence for a in world.agents.values()), 3),
                "avg_vulnerability": round(mean(a.heart.vulnerability for a in world.agents.values()), 3),
                "district_stats": district_stats,
                "top_distressed": [
                    {
                        "agent_id": a.agent_id,
                        "name": a.personality.name,
                        "role": agent_meta[a.agent_id]["role"],
                        "location": a.location,
                        "action": a.last_action,
                        "brief": a.render_subjective_brief(),
                    }
                    for a in top_distressed
                ],
            }
            daily_summaries.append(day_summary)

            if day in checkpoint_days:
                checkpoints[f"day_{day}"] = {
                    aid: {
                        "action": agent.last_action,
                        "location": agent.location,
                        "brief": agent.render_subjective_brief(),
                        "internal_emotion": agent.heart.internal_emotion,
                        "vulnerability": round(agent.heart.vulnerability, 3),
                        "future_branches": agent.get_future_branches(),
                    }
                    for aid, agent in world.agents.items()
                }

    elapsed = time.time() - start

    # Final exports for all agents
    final_agents = {
        aid: _agent_export(world, aid, agent_meta)
        for aid in sorted(world.agents.keys())
    }

    # Relationship summaries
    final_pairs = world.relationships._pairs  # noqa: SLF001
    trust_sorted = sorted(final_pairs.items(), key=lambda item: item[1].trust, reverse=True)[:20]
    resentment_sorted = sorted(
        final_pairs.items(),
        key=lambda item: max(item[1].resentment_ab, item[1].resentment_ba),
        reverse=True,
    )[:20]
    familiarity_sorted = sorted(final_pairs.items(), key=lambda item: item[1].familiarity, reverse=True)[:20]

    changed_pairs = []
    for pair, rel in final_pairs.items():
        base = baseline_pairs.get(pair)
        delta_trust = rel.trust - (base.trust if base else 0.0)
        delta_warmth = rel.warmth - (base.warmth if base else 0.0)
        delta_familiarity = rel.familiarity - (base.familiarity if base else 0)
        delta_resentment = max(rel.resentment_ab, rel.resentment_ba) - max(
            base.resentment_ab if base else 0.0,
            base.resentment_ba if base else 0.0,
        )
        delta_grievance = max(rel.grievance_ab, rel.grievance_ba) - max(
            base.grievance_ab if base else 0.0,
            base.grievance_ba if base else 0.0,
        )
        delta_debt = max(rel.debt_ab, rel.debt_ba) - max(
            base.debt_ab if base else 0.0,
            base.debt_ba if base else 0.0,
        )
        delta_rivalry = rel.rivalry - (base.rivalry if base else 0.0)
        change_score = (
            abs(delta_trust) +
            abs(delta_warmth) +
            delta_familiarity * 0.02 +
            abs(delta_resentment) * 2 +
            abs(delta_grievance) * 2.2 +
            abs(delta_debt) * 1.4 +
            abs(delta_rivalry) * 1.5
        )
        changed_pairs.append(
            (
                change_score, pair, rel, delta_trust, delta_warmth, delta_familiarity,
                delta_resentment, delta_grievance, delta_debt, delta_rivalry,
            )
        )
    changed_pairs.sort(key=lambda item: item[0], reverse=True)

    # Top interaction samples
    top_snapshots_sorted = [item[3] for item in sorted(top_snapshots, key=lambda item: (item[0], item[1], item[2]), reverse=True)]
    selected_snapshots = top_snapshots_sorted[:llm_samples]

    sampled_dialogues = []
    api_key = os.environ.get("OPENAI_API_KEY")
    if llm_samples > 0 and api_key:
        client = OpenAI(api_key=api_key)
        for snapshot in selected_snapshots:
            dialogue = _call_model(client, llm_model, _dialogue_prompt(snapshot), max_output_tokens=280)
            sampled_dialogues.append(
                {
                    "time": snapshot.time_str,
                    "location": snapshot.location,
                    "district": snapshot.district,
                    "interaction_type": snapshot.interaction_type,
                    "agents": [snapshot.agent_a_name, snapshot.agent_b_name],
                    "roles": [snapshot.agent_a_role, snapshot.agent_b_role],
                    "score": round(snapshot.score, 3),
                    "dialogue": dialogue,
                }
            )

    final_concerns = Counter(agent.appraisal.primary_concern for agent in world.agents.values())
    final_action_styles = Counter(agent.motives.action_style for agent in world.agents.values())
    final_actions = Counter(agent.last_action for agent in world.agents.values())
    final_emotions = Counter(agent.heart.internal_emotion for agent in world.agents.values())
    coalition_summary = {}
    for group_name, profile in world.group_profiles.items():
        members = [agent for agent in world.agents.values() if group_name in agent.coalitions]
        if not members:
            continue
        coalition_summary[group_name] = {
            "label": profile.get("label", group_name),
            "members": len(members),
            "avg_loyalty": round(mean(agent.appraisal.loyalty_pressure for agent in members), 3),
            "avg_injustice": round(mean(agent.appraisal.injustice for agent in members), 3),
            "avg_economic_pressure": round(mean(agent.appraisal.economic_pressure for agent in members), 3),
            "avg_secrecy_pressure": round(mean(agent.appraisal.secrecy_pressure for agent in members), 3),
            "home_location": profile.get("home_location", "unknown"),
            "issue": profile.get("issue", "general strain"),
            "rivals": profile.get("rivals", []),
        }

    # Where-we-lack diagnostics
    post_day10_days = [d for d in daily_summaries if d["day"] > 10]
    avg_post10_events = mean(d["events_fired_today"] for d in post_day10_days) if post_day10_days else 0.0
    routine_share = (
        (cumulative_action_counts["WORK"] + cumulative_action_counts["REST"] + cumulative_action_counts["IDLE"])
        / max(1, sum(cumulative_action_counts.values()))
    )
    top3_concern_share = sum(count for _, count in final_concerns.most_common(3)) / max(1, len(world.agents))
    llm_dialogue_coverage = len(sampled_dialogues) / max(1, total_interactions)

    lacks = []
    if avg_post10_events < 2.0:
        lacks.append(
            "After the scripted crisis arc, the town still does not generate enough fresh macro situations on its own. The new event engine extends the tail, but the society still needs stronger institutional and faction mechanics."
        )
    if routine_share > 0.75:
        lacks.append(
            "Most behavior is still routine (`WORK`/`REST`/`IDLE`). Emotional individuality is stronger now, but action diversity is still too low for a world this large."
        )
    if top3_concern_share > 0.6:
        lacks.append(
            "Too many agents collapse into a small set of concerns. The new subjective layer helps, but archetypes still bunch together under the same crisis."
        )
    if cumulative_interaction_types.get("conflict", 0) < max(50, cumulative_interaction_types.get("support", 0) * 0.12):
        lacks.append(
            "Conflict loops are stronger than before, but they still resolve too politely. Coalitions harden, yet too little of that pressure converts into durable interpersonal hostility."
        )
    if llm_dialogue_coverage < 0.02:
        lacks.append(
            "Communication is still mostly unrendered. The sim forms relationships numerically, but only a tiny fraction of interactions become actual dialogue."
        )
    if sum(delta_grievance + delta_debt for _, _, _, _, _, _, _, delta_grievance, delta_debt, _ in changed_pairs[:20]) < 8.0:
        lacks.append(
            "Relationships now track grievances, debts, rivalry, and betrayal, but they still need stronger promise-keeping and named shared-history objects to feel fully lived in."
        )
    group_event_count = sum(
        world.event_engine.generated_counts[kind]
        for kind in ("coalition_caucus", "boycott_call", "whistleblower_leak", "debt_crunch", "accountability_hearing")
        if kind in world.event_engine.generated_counts
    ) if getattr(world, "event_engine", None) else 0
    if group_event_count < 30:
        lacks.append(
            "Group dynamics are present, but they still need stronger leadership turnover, defection, and faction-level bargaining to feel fully societal."
        )

    report = {
        "meta": {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
            "external_information": list(external_information or []),
            "injected_signals": injected_signals,
            "days": days,
            "ticks": total_ticks,
            "runtime_seconds": round(elapsed, 2),
            "agent_count": len(world.agents),
            "initial_relationship_pairs": base_pair_count,
            "final_relationship_pairs": world.relationships.pair_count,
            "events_fired": fired_events,
            "ripple_events_added": ripple_events_added,
            "generated_dynamic_counts": dict(generated_dynamic_counts),
            "fired_dynamic_counts": dict(fired_dynamic_counts),
            "total_interactions": total_interactions,
        },
        "aggregate": {
            "cumulative_action_counts": dict(cumulative_action_counts),
            "cumulative_interaction_types": dict(cumulative_interaction_types),
            "final_concerns": dict(final_concerns),
            "final_action_styles": dict(final_action_styles),
            "final_actions": dict(final_actions),
            "final_emotions": dict(final_emotions),
        },
        "coalitions": coalition_summary,
        "daily_summaries": daily_summaries,
        "checkpoints": checkpoints,
        "top_relationships": {
            "highest_trust": [_pair_metrics(rel, world, a, b) for (a, b), rel in trust_sorted],
            "highest_resentment": [_pair_metrics(rel, world, a, b) for (a, b), rel in resentment_sorted],
            "most_familiar": [_pair_metrics(rel, world, a, b) for (a, b), rel in familiarity_sorted],
            "largest_changes": [
                {
                    **_pair_metrics(rel, world, pair[0], pair[1]),
                    "delta_trust": round(delta_trust, 3),
                    "delta_warmth": round(delta_warmth, 3),
                    "delta_familiarity": delta_familiarity,
                    "delta_peak_resentment": round(delta_resentment, 3),
                    "delta_peak_grievance": round(delta_grievance, 3),
                    "delta_peak_debt": round(delta_debt, 3),
                    "delta_rivalry": round(delta_rivalry, 3),
                }
                for _, pair, rel, delta_trust, delta_warmth, delta_familiarity, delta_resentment, delta_grievance, delta_debt, delta_rivalry in changed_pairs[:20]
            ],
        },
        "sampled_dialogues": sampled_dialogues,
        "representative_dynamic_events": event_engine.generated_history[:30],
        "final_agents": final_agents,
        "where_we_lack": lacks,
    }

    # Markdown report
    lines: list[str] = []
    lines.append(f"# 300-Agent {days}-Day Diagnostic\n")
    lines.append(
        f"Scenario: {SCENARIO_LABELS.get(scenario, scenario)}"
    )
    lines.append(
        f"Simulated {len(world.agents)} agents for {days} days ({total_ticks} ticks) in a single town with 8 districts."
    )
    lines.append("")
    lines.append("## Headline Findings")
    lines.append(f"- Runtime: {elapsed:.2f}s")
    lines.append(f"- Final relationship pairs: {world.relationships.pair_count} (from {base_pair_count} seeded pairs)")
    lines.append(f"- Fired events: {fired_events}")
    lines.append(f"- Ripple events added during sim: {ripple_events_added}")
    lines.append(f"- Generated dynamic event mix: {dict(generated_dynamic_counts)}")
    lines.append(f"- Fired dynamic event mix: {dict(fired_dynamic_counts)}")
    lines.append(f"- Total resolved pair interactions: {total_interactions}")
    lines.append(f"- Final dominant concerns: {dict(final_concerns.most_common(8))}")
    lines.append(f"- Final dominant action styles: {dict(final_action_styles.most_common(8))}")
    lines.append("")
    lines.append("## Daily Arc")
    for day_summary in daily_summaries[:10]:
        lines.append(
            f"- Day {day_summary['day']}: events={day_summary['events_fired_today']}, "
            f"interactions={day_summary['interaction_count_today']}, "
            f"avg_valence={day_summary['avg_valence']:.2f}, avg_vulnerability={day_summary['avg_vulnerability']:.2f}"
        )
    if daily_summaries:
        lines.append("- ...")
        for day_summary in daily_summaries[-5:]:
            lines.append(
                f"- Day {day_summary['day']}: events={day_summary['events_fired_today']}, "
                f"interactions={day_summary['interaction_count_today']}, "
                f"avg_valence={day_summary['avg_valence']:.2f}, avg_vulnerability={day_summary['avg_vulnerability']:.2f}"
            )

    lines.append("")
    lines.append("## Sample Communications")
    if sampled_dialogues:
        for sample in sampled_dialogues[: min(12, len(sampled_dialogues))]:
            lines.append(
                f"### {sample['time']} — {sample['location']} ({sample['district']})"
            )
            lines.append(
                f"*{sample['interaction_type']} | {sample['agents'][0]} ({sample['roles'][0]}) × "
                f"{sample['agents'][1]} ({sample['roles'][1]}) | score={sample['score']:.2f}*"
            )
            lines.append(sample["dialogue"])
            lines.append("")
    else:
        lines.append("- No LLM dialogue samples generated.")

    lines.append("## Representative Dynamic Events")
    if report["representative_dynamic_events"]:
        for event in report["representative_dynamic_events"][:10]:
            day = event["tick"] // 24 + 1
            hour = event["tick"] % 24
            lines.append(
                f"- Day {day}, {hour:02d}:00 [{event['kind']}] {event['location']}: {event['description']}"
            )
    else:
        lines.append("- No dynamic events recorded.")

    lines.append("## Relationship Formation")
    lines.append("### Highest Trust")
    for pair in report["top_relationships"]["highest_trust"][:10]:
        lines.append(
            f"- {pair['names'][0]} × {pair['names'][1]}: trust={pair['trust']:+.2f}, "
            f"warmth={pair['warmth']:+.2f}, familiarity={pair['familiarity']}"
        )
    lines.append("### Highest Resentment")
    for pair in report["top_relationships"]["highest_resentment"][:10]:
        lines.append(
            f"- {pair['names'][0]} × {pair['names'][1]}: resentment_ab={pair['resentment_ab']:.2f}, "
            f"resentment_ba={pair['resentment_ba']:.2f}, trust={pair['trust']:+.2f}"
        )
    lines.append("### Biggest Changes")
    for pair in report["top_relationships"]["largest_changes"][:10]:
        lines.append(
            f"- {pair['names'][0]} × {pair['names'][1]}: "
            f"Δtrust={pair['delta_trust']:+.2f}, Δwarmth={pair['delta_warmth']:+.2f}, "
            f"Δfamiliarity={pair['delta_familiarity']}, Δresentment={pair['delta_peak_resentment']:+.2f}, "
            f"Δgrievance={pair['delta_peak_grievance']:+.2f}, Δdebt={pair['delta_peak_debt']:+.2f}"
        )

    lines.append("## Coalitions")
    for group_name, summary in sorted(coalition_summary.items(), key=lambda item: item[1]["members"], reverse=True):
        lines.append(
            f"- {summary['label']}: members={summary['members']}, issue={summary['issue']}, "
            f"avg_loyalty={summary['avg_loyalty']:.2f}, avg_injustice={summary['avg_injustice']:.2f}, "
            f"avg_economic={summary['avg_economic_pressure']:.2f}, avg_secrecy={summary['avg_secrecy_pressure']:.2f}"
        )

    lines.append("")
    lines.append("## Representative Final Minds")
    representative_ids = []
    roles_seen = set()
    for aid, agent in sorted(world.agents.items(), key=lambda item: item[1].heart.vulnerability, reverse=True):
        role = agent_meta[aid]["role"]
        if role not in roles_seen:
            roles_seen.add(role)
            representative_ids.append(aid)
        if len(representative_ids) >= 12:
            break

    for aid in representative_ids:
        agent_data = final_agents[aid]
        lines.append(f"### {agent_data['name']} [{agent_data['role']}]")
        lines.append(f"- Action: {agent_data['action']} at {agent_data['location']}")
        lines.append(f"- Thought: {agent_data['subjective_brief']}")
        if agent_data["top_relationships"]:
            top_rel = agent_data["top_relationships"][0]
            lines.append(
                f"- Strongest tie: {top_rel['other_name']} "
                f"(trust={top_rel['trust']:+.2f}, warmth={top_rel['warmth']:+.2f}, "
                f"resentment={top_rel['resentment_toward']:.2f})"
            )
        lines.append(f"- Futures: {agent_data['future_branches'][0]['summary']}")

    lines.append("")
    lines.append("## Where We Lack")
    for item in lacks:
        lines.append(f"- {item}")

    # Add macro metrics to report
    macro_summary = world.get_macro_summary()
    shock_impact = world.get_shock_impact_report()
    info_spread = world.get_info_spread_report()

    report["macro_summary"] = macro_summary
    report["shock_impact"] = shock_impact
    report["info_spread"] = info_spread

    # Add macro section to markdown
    lines.append("")
    lines.append("## Macro Outcomes")
    if "current" in macro_summary:
        current = macro_summary["current"]
        deltas = macro_summary.get("deltas", {})
        lines.append(f"- Consumer Confidence: {current.get('consumer_confidence', 0):.3f} (delta: {deltas.get('consumer_confidence', 0):+.4f})")
        lines.append(f"- Social Cohesion: {current.get('social_cohesion', 0):.3f} (delta: {deltas.get('social_cohesion', 0):+.4f})")
        lines.append(f"- Institutional Trust: {current.get('institutional_trust', 0):.3f} (delta: {deltas.get('institutional_trust', 0):+.4f})")
        lines.append(f"- Civil Unrest Potential: {current.get('civil_unrest_potential', 0):.3f} (delta: {deltas.get('civil_unrest_potential', 0):+.4f})")
        lines.append(f"- Market Pressure: {current.get('market_pressure', 0):.3f} (delta: {deltas.get('market_pressure', 0):+.4f})")
        lines.append(f"- Population Mood: {current.get('population_mood', 0):+.3f} (delta: {deltas.get('population_mood', 0):+.4f})")
        awareness = current.get("information_awareness", {})
        if awareness:
            lines.append("- Information Awareness:")
            for label, pct in awareness.items():
                lines.append(f"  - {label}: {pct * 100:.1f}% of population")

    if "impact" in shock_impact:
        lines.append("")
        lines.append("### Shock Impact Analysis")
        lines.append(f"- Shock onset: {shock_impact.get('shock_onset_time', 'unknown')}")
        for metric, vals in shock_impact["impact"].items():
            lines.append(f"- {metric}: {vals['pre']:.3f} → {vals['post']:.3f} ({vals['pct_change']:+.1f}%)")

    output_md_path = Path(output_md)
    output_json_path = Path(output_json)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("\n".join(lines))
    output_json_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic run for the 300-agent world sim")
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument("--scenario", choices=sorted(SCENARIO_BUILDERS), default="large_town")
    parser.add_argument(
        "--output-md",
        default="artifacts/large_town_50d_report.md",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/large_town_50d_report.json",
    )
    parser.add_argument("--llm-samples", type=int, default=20)
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--inject", action="append", default=[], help="External information shock to inject before the run")
    args = parser.parse_args()

    report = run_diagnostic(
        days=args.days,
        output_md=args.output_md,
        output_json=args.output_json,
        llm_samples=args.llm_samples,
        llm_model=args.llm_model,
        scenario=args.scenario,
        external_information=args.inject,
    )

    print(f"\n{'=' * 78}")
    print(f"300-AGENT {args.days}-DAY DIAGNOSTIC COMPLETE")
    print(f"{'=' * 78}")
    print(f"Runtime: {report['meta']['runtime_seconds']}s")
    print(f"Scenario: {report['meta']['scenario_label']}")
    print(f"Events fired: {report['meta']['events_fired']}")
    print(f"Ripple events added: {report['meta']['ripple_events_added']}")
    print(f"Interactions: {report['meta']['total_interactions']}")
    print(f"Final relationship pairs: {report['meta']['final_relationship_pairs']}")
    print(f"Markdown report: {args.output_md}")
    print(f"JSON report:     {args.output_json}")


if __name__ == "__main__":
    main()

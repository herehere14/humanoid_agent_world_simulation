"""Structured decision packets for hybrid LLM agents.

Phase 2 does not hand control to the LLM yet. It builds the exact packet and
decision contract for the `llm_active` agents so the next phase can plug an
LLM chooser in without changing the world model again.
"""

from __future__ import annotations

from dataclasses import dataclass

from .action_table import Action, TickContext
from .relationship import RelationshipStore
from .world_agent import WorldAgent


LLM_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "chosen_action",
        "intent",
        "tactic",
        "tone",
        "surface_move",
        "private_thought",
        "speech",
    ],
    "properties": {
        "chosen_action": {"type": "string"},
        "target_agent_id": {"type": ["string", "null"]},
        "intent": {"type": "string"},
        "tactic": {"type": "string"},
        "tone": {"type": "string"},
        "surface_move": {"type": "string"},
        "private_thought": {"type": "string"},
        "speech": {"type": "string"},
        "self_update": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "story_beat": {"type": "string"},
                "trust_delta_hint": {"type": "number"},
                "grievance_delta_hint": {"type": "number"},
                "debt_delta_hint": {"type": "number"},
            },
        },
    },
}


@dataclass
class ActionOption:
    action: Action
    score: float
    reason: str
    likely_target_agent_id: str | None = None

    def as_dict(self, world_agents: dict[str, WorldAgent]) -> dict:
        target_name = None
        if self.likely_target_agent_id and self.likely_target_agent_id in world_agents:
            target_name = world_agents[self.likely_target_agent_id].personality.name
        return {
            "action": self.action.name,
            "score": round(self.score, 3),
            "reason": self.reason,
            "likely_target_agent_id": self.likely_target_agent_id,
            "likely_target_name": target_name,
        }


def _top_relationship_targets(agent: WorldAgent, relationships: RelationshipStore) -> list[str]:
    rels = relationships.get_agent_relationships(agent.agent_id)[:8]
    support_target = agent.appraisal.support_target if agent.appraisal.support_target in {other_id for other_id, _ in rels} else None
    blame_target = agent.appraisal.blame_target if agent.appraisal.blame_target in {other_id for other_id, _ in rels} else None
    top_warm = None
    top_grievance = None
    best_warm = -999.0
    best_grievance = -999.0
    for other_id, rel in rels:
        if rel.warmth > best_warm:
            best_warm = rel.warmth
            top_warm = other_id
        grievance = max(
            relationships.get_grievance(agent.agent_id, other_id),
            relationships.get_resentment(agent.agent_id, other_id),
        )
        if grievance > best_grievance:
            best_grievance = grievance
            top_grievance = other_id
    ordered = []
    for target in (support_target, blame_target, top_warm, top_grievance):
        if target and target not in ordered:
            ordered.append(target)
    return ordered[:3]


def _score_action_options(agent: WorldAgent, ctx: TickContext, relationships: RelationshipStore) -> list[ActionOption]:
    motives = agent.motives
    appraisal = agent.appraisal
    heart = agent.heart
    profile = agent.get_human_profile()
    nearby = list(ctx.nearby_agent_ids)
    target_candidates = _top_relationship_targets(agent, relationships)

    def target_for(kind: str) -> str | None:
        if kind == "support":
            return target_candidates[0] if target_candidates else (nearby[0] if nearby else None)
        if kind == "conflict":
            for target in target_candidates:
                if target in nearby:
                    return target
            return nearby[0] if nearby else None
        return nearby[0] if nearby else None

    options: list[ActionOption] = []

    options.append(ActionOption(
        action=Action.WORK,
        score=max(0.05, motives.regain_control * 0.7 + motives.protect_status * 0.35),
        reason="Lean into structure, routine, and competence to keep the situation bounded.",
    ))
    options.append(ActionOption(
        action=Action.WITHDRAW,
        score=max(0.05, motives.seek_safety * 0.7 + motives.hide_weakness * 0.45),
        reason="Reduce stimulation and regroup away from witnesses.",
    ))
    options.append(ActionOption(
        action=Action.VENT,
        score=max(0.05, motives.seek_support * 0.65 + motives.discharge_pressure * 0.4),
        reason="Tell someone part of the truth to lower the internal pressure.",
        likely_target_agent_id=target_for("support"),
    ))
    options.append(ActionOption(
        action=Action.SEEK_COMFORT,
        score=max(0.05, motives.seek_support * 0.75 + motives.repair_bonds * 0.45),
        reason="Approach one trusted person and ask for steadiness instead of carrying it alone.",
        likely_target_agent_id=target_for("support"),
    ))
    options.append(ActionOption(
        action=Action.HELP_OTHERS,
        score=max(0.05, motives.protect_others * 0.8 + (0.08 if "care" in profile["coping_style"] or "practical" in profile["care_style"] else 0.0)),
        reason="Stay in motion by stabilizing somebody else first.",
        likely_target_agent_id=target_for("support"),
    ))
    options.append(ActionOption(
        action=Action.CONFRONT,
        score=max(0.05, motives.discharge_pressure * 0.65 + appraisal.injustice * 0.55 + motives.protect_status * 0.25),
        reason="Push directly against the person or side you think is causing the pressure.",
        likely_target_agent_id=target_for("conflict"),
    ))
    options.append(ActionOption(
        action=Action.SOCIALIZE,
        score=max(0.05, motives.repair_bonds * 0.5 + appraisal.loyalty_pressure * 0.35 + motives.seek_support * 0.2),
        reason="Stay in the room and work the social field rather than committing to one person.",
        likely_target_agent_id=target_for("support"),
    ))
    options.append(ActionOption(
        action=Action.FLEE,
        score=max(0.05, motives.seek_safety * 0.8 + heart.arousal * 0.3 + max(0.0, 0.4 - heart.impulse_control) * 0.3),
        reason="Leave before the situation gets even harder to control.",
    ))

    if ctx.hour_of_day >= 22 or ctx.hour_of_day < 6:
        options.append(ActionOption(
            action=Action.REST,
            score=max(0.05, heart.energy * 0.35 + motives.seek_safety * 0.25),
            reason="Back off and recover because the hour itself favors disengagement.",
        ))

    if "humor" in profile["coping_style"] or "joke" in profile["mask_tendency"]:
        for option in options:
            if option.action in {Action.VENT, Action.SOCIALIZE}:
                option.score += 0.06
    if "control" in profile["core_need"]:
        for option in options:
            if option.action in {Action.WORK, Action.CONFRONT}:
                option.score += 0.05
    if "belonging" in profile["core_need"]:
        for option in options:
            if option.action in {Action.SEEK_COMFORT, Action.SOCIALIZE}:
                option.score += 0.05
    if "safety" in profile["core_need"]:
        for option in options:
            if option.action in {Action.WITHDRAW, Action.FLEE, Action.SEEK_COMFORT}:
                option.score += 0.05

    if not nearby:
        for option in options:
            if option.action in {Action.VENT, Action.SEEK_COMFORT, Action.HELP_OTHERS, Action.CONFRONT, Action.SOCIALIZE}:
                option.score *= 0.55
                option.likely_target_agent_id = None

    options.sort(key=lambda item: item.score, reverse=True)
    deduped: list[ActionOption] = []
    seen = set()
    for option in options:
        if option.action in seen:
            continue
        deduped.append(option)
        seen.add(option.action)
        if len(deduped) >= 5:
            break
    return deduped


def build_agent_decision_packet(
    agent: WorldAgent,
    *,
    world_time: str,
    ctx: TickContext,
    relationships: RelationshipStore,
    recommended_action: Action,
    world_agents: dict[str, WorldAgent],
    live_events: list[dict],
) -> dict:
    """Build the packet a future LLM chooser will consume."""
    options = _score_action_options(agent, ctx, relationships)
    nearby_people = []
    for other_id in ctx.nearby_agent_ids[:6]:
        other = world_agents[other_id]
        rel = relationships.get(agent.agent_id, other_id)
        nearby_people.append(
            {
                "agent_id": other_id,
                "name": other.personality.name,
                "role": other.social_role,
                "action": other.last_action,
                "surface_emotion": other.heart.surface_emotion,
                "internal_emotion": other.heart.internal_emotion,
                "primary_concern": other.appraisal.primary_concern,
                "trust": round(rel.trust, 3) if rel else 0.0,
                "warmth": round(rel.warmth, 3) if rel else 0.0,
                "grievance_toward": round(relationships.get_grievance(agent.agent_id, other_id), 3),
                "debt_toward": round(relationships.get_debt(agent.agent_id, other_id), 3),
            }
        )

    relationship_hotspots = []
    for other_id, rel in relationships.get_agent_relationships(agent.agent_id)[:5]:
        relationship_hotspots.append(
            {
                "agent_id": other_id,
                "name": world_agents[other_id].personality.name if other_id in world_agents else other_id,
                "trust": round(rel.trust, 3),
                "warmth": round(rel.warmth, 3),
                "alliance_strength": round(rel.alliance_strength, 3),
                "rivalry": round(rel.rivalry, 3),
                "grievance_toward": round(relationships.get_grievance(agent.agent_id, other_id), 3),
                "resentment_toward": round(relationships.get_resentment(agent.agent_id, other_id), 3),
                "debt_toward": round(relationships.get_debt(agent.agent_id, other_id), 3),
                "last_issue": rel.last_issue,
            }
        )

    recent_memories = [
        {
            "tick": memory.tick,
            "description": memory.description,
            "interpretation": memory.interpretation,
            "story_beat": memory.story_beat,
            "other_agent_id": memory.other_agent_id,
        }
        for memory in agent.get_recent_memories(6)
    ]

    return {
        "schema_version": "v1",
        "agent_id": agent.agent_id,
        "name": agent.personality.name,
        "role": agent.social_role,
        "time": world_time,
        "location": agent.location,
        "recommended_action": recommended_action.name,
        "decision_contract": {
            "instructions": [
                "Choose exactly one action from allowed_actions.",
                "Do not invent new world facts, relationships, or events.",
                "Use target_agent_id only if it appears in nearby_people or relationship_hotspots.",
                "Keep the choice grounded in the private state and ongoing story.",
            ],
            "schema": LLM_DECISION_SCHEMA,
        },
        "private_state": {
            "heart": {
                "arousal": round(agent.heart.arousal, 3),
                "valence": round(agent.heart.valence, 3),
                "tension": round(agent.heart.tension, 3),
                "impulse_control": round(agent.heart.impulse_control, 3),
                "energy": round(agent.heart.energy, 3),
                "vulnerability": round(agent.heart.vulnerability, 3),
                "surface_emotion": agent.heart.surface_emotion,
                "internal_emotion": agent.heart.internal_emotion,
                "divergence": round(agent.heart.divergence, 3),
            },
            "appraisal": {
                "primary_concern": agent.appraisal.primary_concern,
                "interpretation": agent.appraisal.interpretation,
                "ongoing_story": agent.appraisal.ongoing_story,
                "blame_target": agent.appraisal.blame_target,
                "support_target": agent.appraisal.support_target,
                "economic_pressure": round(agent.appraisal.economic_pressure, 3),
                "loyalty_pressure": round(agent.appraisal.loyalty_pressure, 3),
                "secrecy_pressure": round(agent.appraisal.secrecy_pressure, 3),
                "opportunity_pressure": round(agent.appraisal.opportunity_pressure, 3),
            },
            "motives": {
                "priority": agent.motives.priority,
                "mask_style": agent.motives.mask_style,
                "action_style": agent.motives.action_style,
                "inner_voice": agent.motives.inner_voice,
            },
            "human_profile": agent.get_human_profile(),
            "subjective_brief": agent.render_subjective_brief(),
        },
        "recent_memories": recent_memories,
        "relationship_hotspots": relationship_hotspots,
        "nearby_people": nearby_people,
        "live_events": live_events,
        "allowed_actions": [option.as_dict(world_agents) for option in options],
    }

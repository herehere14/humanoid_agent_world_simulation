"""Deterministic salience scoring for hybrid LLM promotion.

Phase 1 of the hybrid architecture is not calling an LLM yet. It identifies
which agents are the best candidates for LLM cognition on a given tick and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .relationship import RelationshipStore
from .world_agent import WorldAgent


EVENT_WEIGHTS = {
    "conflict_flashpoint": 0.18,
    "accountability_hearing": 0.17,
    "whistleblower_leak": 0.16,
    "boycott_call": 0.14,
    "debt_crunch": 0.13,
    "hospital_surge": 0.12,
    "organizing_meeting": 0.11,
    "rumor_wave": 0.1,
    "coalition_caucus": 0.1,
    "mutual_aid_hub": 0.09,
    "neighborhood_meeting": 0.08,
    "waterfront_watch": 0.08,
    "macro_cost_shock": 0.14,
    "external_macro_signal": 0.12,
}


DEFAULT_STORY = "trying to stay steady without giving away too much"


@dataclass
class SalienceContext:
    tick: int
    hour_of_day: int
    nearby_agents: tuple[WorldAgent, ...] = ()
    event_count: int = 0
    event_kinds: tuple[str, ...] = ()
    is_event_target: bool = False


@dataclass
class AgentSalience:
    agent_id: str
    score: float
    level: str
    active: bool = False
    rank: int | None = None
    factors: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def as_summary(self, agent: WorldAgent) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": agent.personality.name,
            "score": round(self.score, 3),
            "level": self.level,
            "active": self.active,
            "rank": self.rank,
            "reasons": list(self.reasons),
            "factors": {k: round(v, 3) for k, v in self.factors.items()},
            "location": agent.location,
            "primary_concern": agent.appraisal.primary_concern,
            "priority_motive": agent.motives.priority,
            "action_style": agent.motives.action_style,
        }


def _level_for_score(score: float) -> str:
    if score >= 0.78:
        return "critical"
    if score >= 0.58:
        return "high"
    if score >= 0.36:
        return "medium"
    return "low"


def _top_relationship_pressure(agent: WorldAgent, relationships: RelationshipStore) -> tuple[float, float, float, float]:
    max_grievance = 0.0
    max_resentment = 0.0
    max_debt = 0.0
    max_alliance = 0.0
    for other_id, rel in relationships.get_agent_relationships(agent.agent_id)[:8]:
        max_grievance = max(
            max_grievance,
            relationships.get_grievance(agent.agent_id, other_id),
            relationships.get_grievance(other_id, agent.agent_id),
        )
        max_resentment = max(
            max_resentment,
            relationships.get_resentment(agent.agent_id, other_id),
            relationships.get_resentment(other_id, agent.agent_id),
        )
        max_debt = max(
            max_debt,
            relationships.get_debt(agent.agent_id, other_id),
            relationships.get_debt(other_id, agent.agent_id),
        )
        max_alliance = max(max_alliance, rel.alliance_strength)
    return max_grievance, max_resentment, max_debt, max_alliance


def score_agent_salience(
    agent: WorldAgent,
    ctx: SalienceContext,
    relationships: RelationshipStore,
) -> AgentSalience:
    """Score how valuable an LLM call would be for this agent right now."""
    heart = agent.heart
    appraisal = agent.appraisal
    motives = agent.motives

    max_grievance, max_resentment, max_debt, max_alliance = _top_relationship_pressure(agent, relationships)
    nearby_rivals = sum(1 for other in ctx.nearby_agents if agent.rival_overlap(other))
    nearby_allies = sum(1 for other in ctx.nearby_agents if agent.shared_coalitions(other))
    nearby_distress = max((other.heart.vulnerability for other in ctx.nearby_agents), default=0.0)

    factors = {
        "emotion": min(
            0.26,
            heart.vulnerability * 0.12 +
            heart.tension * 0.08 +
            heart.divergence * 0.08 +
            max(0.0, 0.45 - heart.valence) * 0.05,
        ),
        "event": min(
            0.28,
            sum(EVENT_WEIGHTS.get(kind, 0.05) for kind in ctx.event_kinds) +
            ctx.event_count * 0.015 +
            (0.14 if ctx.is_event_target else 0.0),
        ),
        "relationship": min(
            0.24,
            max_grievance * 0.09 +
            max_resentment * 0.07 +
            max_debt * 0.06 +
            max_alliance * 0.04 +
            nearby_rivals * 0.03 +
            nearby_allies * 0.015,
        ),
        "stakes": min(
            0.22,
            appraisal.secrecy_pressure * 0.08 +
            appraisal.economic_pressure * 0.08 +
            appraisal.loyalty_pressure * 0.06 +
            appraisal.opportunity_pressure * 0.05 +
            (0.05 if agent.private_burden else 0.0),
        ),
        "scene": min(
            0.16,
            len(ctx.nearby_agents) * 0.02 +
            nearby_distress * 0.06 +
            (0.05 if motives.priority in {"release pressure", "hold the bloc", "collect leverage", "take control"} else 0.0),
        ),
        "story": 0.0,
    }

    if appraisal.ongoing_story and appraisal.ongoing_story != DEFAULT_STORY:
        factors["story"] = 0.06
        if any(
            keyword in appraisal.ongoing_story
            for keyword in (
                "security is one bad week from collapse",
                "loyalty breaks the moment pressure rises",
                "the story can be stolen if I do not frame it",
            )
        ):
            factors["story"] += 0.03

    score = min(1.0, sum(factors.values()))
    level = _level_for_score(score)

    reasons: list[tuple[str, float]] = []
    if ctx.is_event_target:
        reasons.append(("targeted live event", 0.22))
    if factors["emotion"] >= 0.14:
        reasons.append(("high internal volatility", factors["emotion"]))
    if factors["relationship"] >= 0.12:
        reasons.append(("relationship pressure is active", factors["relationship"]))
    if appraisal.secrecy_pressure > 0.45 and agent.private_burden:
        reasons.append(("private burden is near the surface", appraisal.secrecy_pressure))
    if appraisal.economic_pressure > 0.45 or max_debt > 0.45:
        reasons.append(("economic pressure is shaping choices", max(appraisal.economic_pressure, max_debt)))
    if appraisal.loyalty_pressure > 0.45:
        reasons.append(("coalition or loyalty stakes are high", appraisal.loyalty_pressure))
    if ctx.event_count > 0 and ctx.event_kinds:
        reasons.append((f"inside a live scene: {ctx.event_kinds[0]}", factors["event"]))
    if factors["scene"] >= 0.09:
        reasons.append(("social hotspot around the agent", factors["scene"]))
    if factors["story"] > 0:
        reasons.append(("ongoing private story is still unresolved", factors["story"]))

    reasons.sort(key=lambda item: item[1], reverse=True)
    return AgentSalience(
        agent_id=agent.agent_id,
        score=score,
        level=level,
        factors=factors,
        reasons=[label for label, _ in reasons[:4]],
    )


def promote_llm_candidates(
    scored_agents: list[AgentSalience],
    *,
    agent_count: int,
    min_count: int = 4,
    max_count: int = 24,
    fraction: float = 0.06,
    activation_floor: float = 0.34,
) -> list[AgentSalience]:
    """Pick which agents should be active LLM candidates this tick."""
    target_count = min(max_count, max(min_count, int(round(agent_count * fraction))))
    ordered = sorted(scored_agents, key=lambda item: item.score, reverse=True)

    active_indices = [
        idx for idx, item in enumerate(ordered)
        if item.score >= activation_floor
    ][:target_count]

    if len(active_indices) < min_count:
        active_indices = list(range(min(min_count, len(ordered))))

    for rank, idx in enumerate(active_indices, start=1):
        ordered[idx].active = True
        ordered[idx].rank = rank

    return ordered

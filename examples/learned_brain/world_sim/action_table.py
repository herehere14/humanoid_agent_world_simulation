"""Deterministic action selection from heart state.

Pure function: (HeartState, personality, context) → Action.
Same inputs always produce the same output. No randomness.

Priority-ordered rules — first match wins.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass

from .world_agent import WorldAgent, HeartState, Personality
from .relationship import RelationshipStore


class Action(Enum):
    """What the agent does this tick."""

    # High-priority emotional actions
    COLLAPSE = auto()       # Breaking point — agent shuts down, can't function
    LASH_OUT = auto()       # Impulse control gone + high arousal → explosive behavior
    CONFRONT = auto()       # Targeted confrontation with specific person
    FLEE = auto()           # Overwhelmed — leaves current location

    # Moderate emotional actions
    WITHDRAW = auto()       # Retreats to isolation (home, alone)
    SEEK_COMFORT = auto()   # Goes to closest positive relationship
    RUMINATE = auto()       # Stays in place but mentally spiraling
    VENT = auto()           # Seeks someone to talk to (not confrontation)

    # Positive emotional actions
    SOCIALIZE = auto()      # Actively engages with nearby people
    CELEBRATE = auto()      # Shares good news, high energy + positive
    HELP_OTHERS = auto()    # High energy + positive + sees someone struggling

    # Neutral / routine
    WORK = auto()           # Follow schedule, do assigned task
    REST = auto()           # Sleep or passive recovery
    IDLE = auto()           # Default — no strong driver

    @property
    def triggers_speech(self) -> bool:
        """Does this action require an LLM call for dialogue?"""
        return self in (
            Action.CONFRONT, Action.SEEK_COMFORT, Action.VENT,
            Action.SOCIALIZE, Action.CELEBRATE, Action.HELP_OTHERS,
            Action.LASH_OUT,
        )

    @property
    def is_social(self) -> bool:
        return self in (
            Action.CONFRONT, Action.SEEK_COMFORT, Action.VENT,
            Action.SOCIALIZE, Action.CELEBRATE, Action.HELP_OTHERS,
            Action.LASH_OUT,
        )

    @property
    def changes_location(self) -> bool:
        return self in (Action.FLEE, Action.WITHDRAW, Action.SEEK_COMFORT)


@dataclass
class TickContext:
    """Context available for action selection."""
    tick: int
    hour_of_day: int           # 0-23
    scheduled_location: str    # where schedule says to be
    current_location: str
    nearby_agent_ids: list[str]
    nearby_agents: dict[str, WorldAgent]  # id → WorldAgent for checking their state
    relationships: RelationshipStore
    agent_id: str

    @property
    def is_work_hours(self) -> bool:
        return 9 <= self.hour_of_day < 17

    @property
    def at_work(self) -> bool:
        return self.current_location == self.scheduled_location and self.is_work_hours


def select_action(agent: WorldAgent, ctx: TickContext) -> Action:
    """Deterministic action selection from heart state.

    Priority-ordered rules — first match wins.
    Same state always produces the same action.

    Design principle: routine (WORK/IDLE) is the default.
    Emotional actions only fire when heart state is clearly non-neutral.
    """
    s = agent.heart

    # --- Sleep hours (22-6): always rest unless extreme state ---
    if ctx.hour_of_day >= 22 or ctx.hour_of_day < 6:
        if s.impulse_control < 0.15 and s.arousal > 0.6:
            return Action.RUMINATE  # can't sleep, spiraling
        return Action.REST

    # --- Extreme states (override everything, even work) ---

    # 1. Total collapse: no energy + high vulnerability
    if s.energy < 0.1 and s.vulnerability > 0.7:
        return Action.COLLAPSE

    # 2. Impulse control gone + high arousal → explosive
    if s.impulse_control < 0.2 and s.arousal > 0.5 and s.valence < 0.3:
        if ctx.nearby_agent_ids:
            for nid in ctx.nearby_agent_ids:
                if ctx.relationships.get_resentment(agent.agent_id, nid) > 0.3:
                    return Action.CONFRONT
            return Action.LASH_OUT
        return Action.FLEE

    # 3. Overwhelmed: high arousal + high tension + low energy
    if s.arousal > 0.6 and s.tension > 0.5 and s.energy < 0.3:
        return Action.FLEE

    # 4. Depleted: low energy → withdraw
    if s.energy < 0.2:
        return Action.WITHDRAW

    # 5. Deeply negative + vulnerable → seek comfort or withdraw
    if s.valence < 0.25 and s.vulnerability > 0.4:
        for nid in ctx.nearby_agent_ids:
            rel = ctx.relationships.get(agent.agent_id, nid)
            if rel and rel.warmth > 0.4 and rel.trust > 0.3:
                return Action.SEEK_COMFORT
        return Action.WITHDRAW

    # --- Work hours at work: default to WORK unless clearly distressed ---
    if ctx.at_work:
        # Only override work for sustained negative states
        if (s.valence < 0.3 and s.tension > 0.4 and
                len(s.valence_history) >= 3 and all(v < 0.35 for v in s.valence_history[-3:])):
            return Action.RUMINATE

        if s.valence < 0.35 and s.arousal > 0.4 and s.impulse_control < 0.4:
            for nid in ctx.nearby_agent_ids:
                rel = ctx.relationships.get(agent.agent_id, nid)
                if rel and rel.warmth > 0.2:
                    return Action.VENT
        return Action.WORK

    # --- Outside work hours / not at work location ---

    # 6. Sustained negative → ruminate
    if (s.valence < 0.35 and s.tension > 0.3 and
            len(s.valence_history) >= 3 and all(v < 0.4 for v in s.valence_history[-3:])):
        return Action.RUMINATE

    # 7. Needs to vent: negative + has someone to talk to
    if s.valence < 0.4 and s.arousal > 0.3 and s.impulse_control < 0.4:
        for nid in ctx.nearby_agent_ids:
            rel = ctx.relationships.get(agent.agent_id, nid)
            if rel and rel.warmth > 0.2:
                return Action.VENT

    # 8. High positive + high energy → celebrate (rare)
    if s.valence > 0.8 and s.energy > 0.6 and s.arousal > 0.5:
        if ctx.nearby_agent_ids:
            return Action.CELEBRATE

    # 9. Help others: only when a nearby agent is actually struggling
    if s.valence > 0.6 and s.energy > 0.5:
        for nid in ctx.nearby_agent_ids:
            other = ctx.nearby_agents.get(nid)
            if other and other.heart.valence < 0.3 and other.heart.vulnerability > 0.3:
                rel = ctx.relationships.get(agent.agent_id, nid)
                if rel and rel.warmth > 0.2:
                    return Action.HELP_OTHERS

    # 10. Socialize: evening/weekend + positive + people nearby
    if (s.valence > 0.55 and s.arousal > 0.3 and ctx.nearby_agent_ids
            and ctx.hour_of_day >= 17):
        return Action.SOCIALIZE

    # --- Fallback: routine ---
    if ctx.is_work_hours:
        return Action.WORK
    return Action.IDLE


def get_action_description(action: Action, agent: WorldAgent) -> str:
    """Human-readable description of what the agent is doing."""
    s = agent.heart
    name = agent.personality.name

    descriptions = {
        Action.COLLAPSE: f"{name} has shut down — sitting motionless, unable to function",
        Action.LASH_OUT: f"{name} snaps at whoever is nearby — no filter left",
        Action.CONFRONT: f"{name} confronts someone they're angry at",
        Action.FLEE: f"{name} abruptly leaves — needs to get away",
        Action.WITHDRAW: f"{name} withdraws to be alone",
        Action.SEEK_COMFORT: f"{name} reaches out to someone they trust",
        Action.RUMINATE: f"{name} is lost in thought, mentally spiraling",
        Action.VENT: f"{name} needs to talk — unloading on whoever will listen",
        Action.SOCIALIZE: f"{name} is chatting with people around them",
        Action.CELEBRATE: f"{name} is in high spirits, sharing good energy",
        Action.HELP_OTHERS: f"{name} notices someone struggling and reaches out",
        Action.WORK: f"{name} is working",
        Action.REST: f"{name} is resting",
        Action.IDLE: f"{name} is idle",
    }
    return descriptions.get(action, f"{name} is doing something")

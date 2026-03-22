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
    event_count: int = 0
    event_kinds: tuple[str, ...] = ()
    is_event_target: bool = False

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
    motives = agent.motives
    appraisal = agent.appraisal
    nearby_resentment = max(
        (ctx.relationships.get_resentment(agent.agent_id, nid) for nid in ctx.nearby_agent_ids),
        default=0.0,
    )
    nearby_grievance = max(
        (ctx.relationships.get_grievance(agent.agent_id, nid) for nid in ctx.nearby_agent_ids),
        default=0.0,
    )
    nearby_debt = max(
        (ctx.relationships.get_debt(agent.agent_id, nid) for nid in ctx.nearby_agent_ids),
        default=0.0,
    )
    has_rival_present = any(
        agent.rival_overlap(other) for other in ctx.nearby_agents.values()
    )
    has_ally_present = any(
        agent.shared_coalitions(other) for other in ctx.nearby_agents.values()
    )

    # --- Sleep hours (22-6): always rest unless extreme state ---
    if ctx.hour_of_day >= 22 or ctx.hour_of_day < 6:
        if (
            s.impulse_control < 0.15 and s.arousal > 0.6 and motives.hide_weakness >= motives.seek_support
        ) or appraisal.secrecy_pressure > 0.6:
            return Action.RUMINATE  # can't sleep, spiraling
        return Action.REST

    # --- Extreme states (override everything, even work) ---

    # 1. Total collapse: no energy + high vulnerability
    if s.energy < 0.1 and s.vulnerability > 0.7:
        return Action.COLLAPSE

    # 2. Impulse control gone + high arousal → explosive
    if s.impulse_control < 0.2 and s.arousal > 0.5 and s.valence < 0.3:
        if motives.seek_safety > motives.discharge_pressure and motives.seek_safety > motives.protect_status:
            return Action.FLEE
        if ctx.nearby_agent_ids:
            for nid in ctx.nearby_agent_ids:
                if (
                    ctx.relationships.get_resentment(agent.agent_id, nid) > 0.24
                    and (
                        motives.protect_status > 0.35 or
                        appraisal.injustice > 0.3 or
                        ctx.relationships.get_grievance(agent.agent_id, nid) > 0.2
                    )
                ):
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
        if has_rival_present and nearby_grievance > 0.3 and motives.discharge_pressure >= motives.seek_support:
            return Action.CONFRONT
        for nid in ctx.nearby_agent_ids:
            rel = ctx.relationships.get(agent.agent_id, nid)
            if (
                rel and rel.warmth > 0.4 and rel.trust > 0.3
                and motives.seek_support >= motives.hide_weakness
            ):
                return Action.SEEK_COMFORT
        return Action.WITHDRAW

    # Live events should create social behavior rather than passive attendance.
    if ctx.event_count > 0 and ctx.nearby_agent_ids:
        event_kinds = set(ctx.event_kinds)
        if event_kinds & {"conflict_flashpoint", "accountability_hearing", "whistleblower_leak", "boycott_call"}:
            if (
                appraisal.injustice > 0.24 or
                motives.discharge_pressure > 0.34 or
                nearby_grievance > 0.22 or
                ctx.is_event_target or
                has_rival_present
            ):
                for nid in ctx.nearby_agent_ids:
                    if (
                        ctx.relationships.get_resentment(agent.agent_id, nid) > 0.18 or
                        ctx.relationships.get_grievance(agent.agent_id, nid) > 0.18
                    ):
                        return Action.CONFRONT
                if motives.seek_support >= motives.hide_weakness and has_ally_present:
                    return Action.VENT
        if event_kinds & {"mutual_aid_hub", "neighborhood_meeting", "hospital_surge", "waterfront_watch", "debt_crunch"}:
            if motives.protect_others > 0.32:
                return Action.HELP_OTHERS
            if nearby_debt > 0.35 and appraisal.economic_pressure > 0.45 and has_rival_present:
                return Action.CONFRONT
            if motives.seek_support > 0.32:
                return Action.VENT
        if event_kinds & {"organizing_meeting", "rumor_wave", "slow_burn_followup", "coalition_caucus"}:
            if appraisal.loyalty_pressure > 0.42 and has_ally_present:
                return Action.SOCIALIZE
            if appraisal.injustice > 0.2 or motives.seek_support > 0.28 or appraisal.opportunity_pressure > 0.35:
                if s.valence < 0.5 or motives.seek_support >= motives.hide_weakness:
                    return Action.VENT
                return Action.SOCIALIZE

    # --- Work hours at work: default to WORK unless clearly distressed ---
    if ctx.at_work:
        # Only override work for sustained negative states
        if (s.valence < 0.3 and s.tension > 0.4 and
                len(s.valence_history) >= 3 and all(v < 0.35 for v in s.valence_history[-3:])
                and motives.hide_weakness >= motives.seek_support):
            return Action.RUMINATE

        if motives.protect_others > 0.55:
            for nid in ctx.nearby_agent_ids:
                other = ctx.nearby_agents.get(nid)
                if other and other.heart.vulnerability > 0.45:
                    return Action.HELP_OTHERS

        if (
            has_rival_present and
            (nearby_grievance > 0.22 or appraisal.injustice > 0.35) and
            motives.discharge_pressure >= motives.seek_support
        ):
            return Action.CONFRONT

        if s.valence < 0.35 and s.arousal > 0.4 and s.impulse_control < 0.4:
            for nid in ctx.nearby_agent_ids:
                rel = ctx.relationships.get(agent.agent_id, nid)
                if rel and rel.warmth > 0.2 and motives.seek_support > motives.regain_control:
                    return Action.VENT

        if motives.regain_control > 0.55 or motives.protect_status > 0.5:
            return Action.WORK
        return Action.WORK

    # --- Outside work hours / not at work location ---

    # 6. Sustained negative → ruminate
    if (s.valence < 0.35 and s.tension > 0.3 and
            len(s.valence_history) >= 3 and all(v < 0.4 for v in s.valence_history[-3:])
            and motives.hide_weakness >= motives.seek_support):
        return Action.RUMINATE

    # 7. Needs to vent: negative + has someone to talk to
    if s.valence < 0.4 and s.arousal > 0.3 and s.impulse_control < 0.4:
        if has_rival_present and nearby_grievance > 0.24 and motives.discharge_pressure > motives.seek_support:
            return Action.CONFRONT
        for nid in ctx.nearby_agent_ids:
            rel = ctx.relationships.get(agent.agent_id, nid)
            if rel and rel.warmth > 0.2 and motives.seek_support >= motives.hide_weakness:
                return Action.VENT

    # 8. High positive + high energy → celebrate (rare)
    if s.valence > 0.8 and s.energy > 0.6 and s.arousal > 0.5:
        if ctx.nearby_agent_ids:
            return Action.CELEBRATE

    # 9. Help others: only when a nearby agent is actually struggling
    if s.valence > 0.6 and s.energy > 0.5 or motives.protect_others > 0.5:
        for nid in ctx.nearby_agent_ids:
            other = ctx.nearby_agents.get(nid)
            if other and other.heart.valence < 0.3 and other.heart.vulnerability > 0.3:
                rel = ctx.relationships.get(agent.agent_id, nid)
                if rel and rel.warmth > 0.2:
                    return Action.HELP_OTHERS

    # 10. Socialize: evening/weekend + positive + people nearby
    if (s.valence > 0.55 and s.arousal > 0.3 and ctx.nearby_agent_ids
            and ctx.hour_of_day >= 17):
        if has_ally_present and appraisal.loyalty_pressure > 0.32:
            return Action.SOCIALIZE
        if motives.action_style in {"joking deflection", "protective caretaking", "plainspoken honesty"}:
            return Action.SOCIALIZE
        return Action.SOCIALIZE

    # --- Fallback: routine ---
    if ctx.is_work_hours:
        return Action.WORK
    return Action.IDLE


def get_action_description(action: Action, agent: WorldAgent) -> str:
    """Human-readable description of what the agent is doing."""
    s = agent.heart
    name = agent.personality.name
    style = agent.motives.action_style

    descriptions = {
        Action.COLLAPSE: f"{name} has shut down — sitting motionless, unable to function",
        Action.LASH_OUT: f"{name} snaps at whoever is nearby with {style} — no filter left",
        Action.CONFRONT: f"{name} confronts someone they're angry at through {style}",
        Action.FLEE: f"{name} abruptly leaves — needs to get away",
        Action.WITHDRAW: f"{name} withdraws to be alone and regroup",
        Action.SEEK_COMFORT: f"{name} reaches out to someone they trust with {style}",
        Action.RUMINATE: f"{name} is lost in thought, mentally spiraling behind {agent.motives.mask_style}",
        Action.VENT: f"{name} needs to talk — unloading through {style}",
        Action.SOCIALIZE: f"{name} is chatting with people around them through {style}",
        Action.CELEBRATE: f"{name} is in high spirits, sharing good energy",
        Action.HELP_OTHERS: f"{name} notices someone struggling and reaches out with {style}",
        Action.WORK: f"{name} is working in a {style} mode",
        Action.REST: f"{name} is resting",
        Action.IDLE: f"{name} is idle",
    }
    return descriptions.get(action, f"{name} is doing something")

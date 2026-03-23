"""Individual agent shock appraisal — each agent interprets shocks for themselves.

This replaces the hardcoded role_impacts lookup table. Instead of:
    "dock_worker → debt_pressure +0.22"

Each agent individually computes their personal impact based on:
    - Their financial exposure (debt_pressure, identity_tags, private_burden)
    - Their psychological profile (threat_lens, core_need, coping_style)
    - Their social position (coalitions, relationships, social_role)
    - Their current emotional state (vulnerability, existing wounds)
    - The NATURE of the shock (economic vs health vs existential vs moral)

The result: macro outcomes EMERGE from 300 individual appraisals,
not from a pre-written lookup table.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world_agent import WorldAgent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Shock classification — what KIND of threat is this?
# ---------------------------------------------------------------------------

@dataclass
class ShockProfile:
    """What kind of threat does this shock represent?

    Each shock is a blend of threat dimensions. An oil price surge is
    primarily economic with some supply-chain and livelihood components.
    A pandemic is primarily health with economic secondary effects.
    """
    economic_threat: float = 0.0      # threatens income, savings, purchasing power
    employment_threat: float = 0.0    # threatens jobs directly
    health_threat: float = 0.0        # threatens physical safety / health
    existential_threat: float = 0.0   # threatens survival / existence itself
    moral_threat: float = 0.0         # threatens ethical identity / moral standing
    supply_threat: float = 0.0        # threatens access to goods / services
    institutional_threat: float = 0.0 # threatens trust in institutions
    social_threat: float = 0.0        # threatens social fabric / community
    exposure_threat: float = 0.0      # threatens secrets / reputation

    # Which locations are epicenters of this shock?
    epicenter_locations: list[str] = field(default_factory=list)

    # Raw severity multiplier (extracted from text, e.g. "100%" → 2.0)
    severity_multiplier: float = 1.0


def classify_shock(kind: str, source_text: str, severity: float) -> ShockProfile:
    """Classify a shock into threat dimensions.

    This is the ONLY place where shock-type-specific logic lives.
    It says WHAT the shock threatens, but NOT how individual agents react.
    """
    lowered = source_text.lower()

    # Extract percentage for scaling
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lowered)
    pct_mult = 1.0 + float(pct_match.group(1)) / 100.0 if pct_match else 1.0
    pct_mult = max(1.0, min(3.0, pct_mult))

    profiles = {
        "oil_price_surge": ShockProfile(
            economic_threat=0.7 * pct_mult,
            supply_threat=0.6 * pct_mult,
            employment_threat=0.3 * pct_mult,
            social_threat=0.2,
            institutional_threat=0.2,
            epicenter_locations=["docks", "warehouse", "main_market", "factory_floor"],
            severity_multiplier=pct_mult,
        ),
        "mass_layoffs": ShockProfile(
            employment_threat=0.9 * pct_mult,
            economic_threat=0.8 * pct_mult,
            social_threat=0.4,
            institutional_threat=0.3,
            moral_threat=0.2,
            epicenter_locations=["office_tower", "factory_floor", "workers_canteen"],
            severity_multiplier=pct_mult,
        ),
        "brand_scandal": ShockProfile(
            moral_threat=0.8,
            exposure_threat=0.5,
            economic_threat=0.3,
            employment_threat=0.3,
            institutional_threat=0.4,
            social_threat=0.3,
            epicenter_locations=["office_tower", "main_market", "student_union"],
            severity_multiplier=1.0,
        ),
        "banking_panic": ShockProfile(
            economic_threat=0.9,
            institutional_threat=0.8,
            supply_threat=0.5,
            employment_threat=0.5,
            social_threat=0.4,
            existential_threat=0.3,
            epicenter_locations=["office_tower", "main_market", "community_center"],
            severity_multiplier=1.0,
        ),
        "health_crisis": ShockProfile(
            health_threat=0.9,
            existential_threat=0.5,
            social_threat=0.4,
            economic_threat=0.3,
            supply_threat=0.3,
            institutional_threat=0.3,
            epicenter_locations=["hospital", "north_school", "community_center"],
            severity_multiplier=1.0,
        ),
        "military_crisis": ShockProfile(
            existential_threat=0.95,
            health_threat=0.4,
            social_threat=0.6,
            institutional_threat=0.4,
            economic_threat=0.4,
            supply_threat=0.5,
            epicenter_locations=["office_tower", "community_center", "student_union", "central_park"],
            severity_multiplier=1.0,
        ),
    }

    profile = profiles.get(kind, ShockProfile(
        economic_threat=0.3,
        social_threat=0.3,
        institutional_threat=0.2,
        severity_multiplier=1.0,
    ))

    # Scale by overall severity
    scale = severity / 2.0  # normalize severity (typical range 1.5-3.0) to ~0.75-1.5
    profile.economic_threat *= scale
    profile.employment_threat *= scale
    profile.health_threat *= scale
    profile.existential_threat *= scale
    profile.moral_threat *= scale
    profile.supply_threat *= scale
    profile.institutional_threat *= scale
    profile.social_threat *= scale
    profile.exposure_threat *= scale

    return profile


# ---------------------------------------------------------------------------
# Individual agent appraisal — the core of emergent behavior
# ---------------------------------------------------------------------------

@dataclass
class AgentShockReaction:
    """One agent's computed personal reaction to a shock."""
    agent_id: str
    debt_pressure_delta: float = 0.0
    dread_pressure_delta: float = 0.0
    secret_pressure_delta: float = 0.0
    tension_delta: float = 0.0
    valence_delta: float = 0.0
    personal_severity: float = 0.0  # 0-1, how hard this hits this specific person

    # Why this agent reacted this way (for inspection/debugging)
    factors: dict[str, float] = field(default_factory=dict)
    interpretation: str = ""


def appraise_shock(agent: "WorldAgent", shock: ShockProfile) -> AgentShockReaction:
    """Each agent individually interprets a shock based on who they are.

    This is the function that makes macro outcomes emergent.
    Two dock workers with different threat lenses, financial situations,
    and coping styles will react differently to the same shock.
    """
    profile = agent.get_human_profile()
    reaction = AgentShockReaction(agent_id=agent.agent_id)

    # ── 1. Financial exposure ──
    # How exposed is this person's livelihood to economic disruption?
    financial_exposure = 0.0

    # Role-based base exposure (not the reaction — just how exposed they are)
    role_exposure = {
        "dock_worker": 0.7,      # physical labor, exposed to shipping costs
        "factory_worker": 0.65,  # dependent on industrial demand
        "market_vendor": 0.7,    # directly sells goods, thin margins
        "bartender": 0.55,       # service sector, discretionary spending
        "office_worker": 0.4,    # somewhat insulated
        "office_professional": 0.35,
        "manager": 0.3,          # more job security
        "teacher": 0.25,         # public sector, stable
        "healthcare": 0.2,       # essential, recession-proof
        "government_worker": 0.15,
        "student": 0.35,         # dependent on others, future uncertain
        "community": 0.5,        # mixed, often informal economy
        "retiree": 0.45,         # fixed income, vulnerable to inflation
    }.get(agent.social_role, 0.4)
    financial_exposure += role_exposure

    # Personal financial vulnerability from identity tags and burden
    tags_text = " ".join(agent.identity_tags).lower()
    burden_text = agent.private_burden.lower()
    all_text = f"{tags_text} {burden_text} {agent.personality.background.lower()}"

    if any(w in all_text for w in ("rent stressed", "bills", "loan", "debt", "overdue", "arrears")):
        financial_exposure += 0.25
    if any(w in all_text for w in ("provider", "family crew", "kids", "toddler", "tuition")):
        financial_exposure += 0.15
    if any(w in all_text for w in ("savings", "pension", "retirement fund", "nest egg")):
        financial_exposure += 0.1  # has savings → exposed to financial shocks
    if any(w in all_text for w in ("cash flow watcher", "survival math", "thin margins")):
        financial_exposure += 0.2

    # Existing debt pressure amplifies new economic shocks
    financial_exposure += agent.debt_pressure * 0.3

    financial_exposure = _clamp(financial_exposure, 0.0, 1.0)
    reaction.factors["financial_exposure"] = round(financial_exposure, 3)

    # ── 2. Threat lens alignment ──
    # Does this shock hit what this person fears most?
    threat_lens = profile.get("threat_lens", "chaos")
    lens_multiplier = 1.0

    lens_alignment = {
        "scarcity": shock.economic_threat * 0.6 + shock.supply_threat * 0.5 + shock.employment_threat * 0.4,
        "betrayal": shock.moral_threat * 0.6 + shock.institutional_threat * 0.5 + shock.exposure_threat * 0.3,
        "humiliation": shock.exposure_threat * 0.5 + shock.moral_threat * 0.4 + shock.social_threat * 0.3,
        "chaos": shock.existential_threat * 0.6 + shock.social_threat * 0.4 + shock.health_threat * 0.3,
        "abandonment": shock.social_threat * 0.6 + shock.existential_threat * 0.4 + shock.health_threat * 0.3,
        "exposure": shock.exposure_threat * 0.7 + shock.moral_threat * 0.4 + shock.institutional_threat * 0.2,
    }

    lens_score = lens_alignment.get(threat_lens, 0.3)
    lens_multiplier = 1.0 + lens_score  # 1.0 (no alignment) to ~2.0 (strong alignment)
    reaction.factors["threat_lens_alignment"] = round(lens_score, 3)
    reaction.factors["threat_lens"] = threat_lens

    # ── 3. Core need vulnerability ──
    # If the shock threatens what you need most, it hits deeper
    core_need = profile.get("core_need", "safety")
    need_vulnerability = 0.0

    need_mapping = {
        "safety": shock.economic_threat * 0.5 + shock.existential_threat * 0.5 + shock.health_threat * 0.3,
        "dignity": shock.moral_threat * 0.4 + shock.exposure_threat * 0.4 + shock.employment_threat * 0.3,
        "belonging": shock.social_threat * 0.6 + shock.existential_threat * 0.3,
        "control": shock.institutional_threat * 0.4 + shock.existential_threat * 0.3 + shock.economic_threat * 0.3,
        "usefulness": shock.health_threat * 0.4 + shock.social_threat * 0.3 + shock.employment_threat * 0.3,
        "justice": shock.moral_threat * 0.5 + shock.institutional_threat * 0.4,
        "truth": shock.exposure_threat * 0.4 + shock.moral_threat * 0.3 + shock.institutional_threat * 0.3,
        "autonomy": shock.institutional_threat * 0.4 + shock.employment_threat * 0.3 + shock.economic_threat * 0.3,
    }

    need_vulnerability = need_mapping.get(core_need, 0.3)
    reaction.factors["core_need_vulnerability"] = round(need_vulnerability, 3)

    # ── 4. Location exposure ──
    # Are you at or near the epicenter?
    location_exposure = 0.0
    if agent.location in shock.epicenter_locations:
        location_exposure = 0.3
    # Check if scheduled locations overlap with epicenters
    scheduled_epicenter = any(
        loc in shock.epicenter_locations
        for loc in agent.schedule.values()
    )
    if scheduled_epicenter:
        location_exposure = max(location_exposure, 0.15)
    reaction.factors["location_exposure"] = round(location_exposure, 3)

    # ── 5. Emotional vulnerability ──
    # Current emotional state amplifies or dampens the reaction
    emotional_vulnerability = (
        agent.heart.vulnerability * 0.4 +
        (1.0 - agent.heart.impulse_control) * 0.2 +
        len(agent.heart.wounds) * 0.08 +
        agent.heart.tension * 0.15 +
        max(0.0, 0.5 - agent.heart.valence) * 0.2
    )
    emotional_vulnerability = _clamp(emotional_vulnerability)
    reaction.factors["emotional_vulnerability"] = round(emotional_vulnerability, 3)

    # ── 6. Coping style modifier ──
    # Some coping styles make you more reactive, others more resilient
    coping = profile.get("coping_style", "")
    coping_modifier = 1.0  # neutral

    resilient_coping = ("intellectualize", "control the room", "perform competence", "disappear into work")
    reactive_coping = ("confront head-on", "deflect with humor", "reach for connection", "seek witnesses")
    vulnerable_coping = ("caretake first",)  # caretakers absorb others' stress

    if coping in resilient_coping:
        coping_modifier = 0.85  # slightly dampened reaction
    elif coping in reactive_coping:
        coping_modifier = 1.1  # slightly amplified
    elif coping in vulnerable_coping:
        coping_modifier = 1.15  # absorb more
    reaction.factors["coping_modifier"] = round(coping_modifier, 3)

    # ── 7. Self-story modifier ──
    # Your narrative about yourself affects how you process threats
    self_story = profile.get("self_story", "survivor")
    story_modifier = 1.0

    if self_story == "survivor":
        story_modifier = 0.9  # been through worse, slightly resilient
    elif self_story == "provider":
        # Providers feel economic shocks more intensely (they have people depending on them)
        story_modifier = 1.0 + shock.economic_threat * 0.15
    elif self_story == "guardian":
        # Guardians feel community/health threats more
        story_modifier = 1.0 + (shock.social_threat + shock.health_threat) * 0.1
    elif self_story == "witness":
        story_modifier = 1.0 + shock.moral_threat * 0.1  # moral shocks resonate
    elif self_story == "climber":
        story_modifier = 1.0 + shock.employment_threat * 0.15  # career threats hit hard
    elif self_story == "fixer":
        story_modifier = 0.95  # takes action, slightly resilient
    elif self_story == "loyalist":
        story_modifier = 1.0 + shock.institutional_threat * 0.1
    reaction.factors["self_story"] = self_story
    reaction.factors["story_modifier"] = round(story_modifier, 3)

    # ── 8. Compute personal severity ──
    # Combine all factors into a single severity score
    personal_severity = (
        financial_exposure * shock.economic_threat * 0.25 +
        financial_exposure * shock.employment_threat * 0.2 +
        lens_score * 0.2 +
        need_vulnerability * 0.15 +
        location_exposure * 0.1 +
        emotional_vulnerability * 0.1 +
        shock.existential_threat * 0.15 +  # existential threats hit everyone
        shock.health_threat * 0.1  # health threats hit everyone
    )
    personal_severity *= lens_multiplier * coping_modifier * story_modifier
    personal_severity = _clamp(personal_severity, 0.0, 1.0)
    reaction.personal_severity = personal_severity

    # ── 9. Translate personal severity into state changes ──

    # Economic shocks → debt_pressure (proportional to financial exposure)
    if shock.economic_threat > 0.1 or shock.employment_threat > 0.1:
        econ_impact = (shock.economic_threat + shock.employment_threat) * 0.5
        reaction.debt_pressure_delta = _clamp(
            econ_impact * financial_exposure * 0.35 * shock.severity_multiplier
        )

    # Health/existential/moral shocks → dread_pressure
    dread_sources = (
        shock.health_threat * 0.4 +
        shock.existential_threat * 0.5 +
        shock.moral_threat * 0.3
    )
    if dread_sources > 0.1:
        reaction.dread_pressure_delta = _clamp(
            dread_sources * personal_severity * 0.5
        )

    # Exposure/moral shocks → secret_pressure (for people with things to hide)
    if shock.exposure_threat > 0.1 or shock.moral_threat > 0.1:
        exposure_vuln = 0.0
        if agent.private_burden:
            exposure_vuln += 0.3
        if agent.secret_pressure > 0.1:
            exposure_vuln += 0.2
        if any(w in all_text for w in ("hide", "deleted", "screenshots", "quiet", "waiver")):
            exposure_vuln += 0.2
        reaction.secret_pressure_delta = _clamp(
            (shock.exposure_threat + shock.moral_threat) * 0.5 * exposure_vuln * 0.4
        )

    # Tension: proportional to personal severity
    reaction.tension_delta = _clamp(personal_severity * 0.2)

    # Valence: proportional to personal severity (negative)
    reaction.valence_delta = -_clamp(personal_severity * 0.12)

    # ── 10. Build interpretation ──
    # What does this person TELL THEMSELVES about the shock?
    reaction.interpretation = _build_interpretation(agent, profile, shock, personal_severity)

    return reaction


def _build_interpretation(agent: "WorldAgent", profile: dict, shock: ShockProfile, severity: float) -> str:
    """Generate a first-person interpretation based on the agent's psychology.

    This is what the agent's inner voice says — NOT hardcoded by role,
    but derived from their threat lens, core need, and self-story.
    """
    threat_lens = profile.get("threat_lens", "chaos")
    core_need = profile.get("core_need", "safety")
    self_story = profile.get("self_story", "survivor")

    # High economic threat
    if shock.economic_threat > 0.5 and severity > 0.3:
        if threat_lens == "scarcity":
            return "This is what I always knew would come. The floor is moving and my people need me to hold it."
        if self_story == "provider":
            return "I keep counting the bills and the numbers stopped working. Someone is going to ask me how we get through this."
        if core_need == "safety":
            return "Everything I built to feel safe just got thinner. I can feel the margin closing."

    # High health threat
    if shock.health_threat > 0.5 and severity > 0.3:
        if self_story == "guardian":
            return "The people I protect are in the path of this and I cannot shield them all."
        if threat_lens == "chaos":
            return "The systems I trusted to hold are cracking. Nobody has a plan that fits what is actually happening."
        if core_need == "usefulness":
            return "I should be helping but I do not know if what I do is enough or if I am making it worse."

    # High existential threat
    if shock.existential_threat > 0.5:
        if threat_lens == "abandonment":
            return "The people who could leave will leave. The rest of us are here because here is all we have."
        if self_story == "survivor":
            return "I have been through worse. But the room feels different this time and I am not sure why."
        return "Everything that felt solid just became conditional. I am paying attention to who stays."

    # High moral threat
    if shock.moral_threat > 0.5:
        if core_need == "justice" or self_story == "witness":
            return "Someone has to name this for what it is before the story gets rewritten."
        if core_need == "dignity":
            return "Being associated with this diminishes me. I need to decide what side of this I stand on."
        if threat_lens == "betrayal":
            return "They knew. The people in charge always knew and they kept going."

    # Default
    if severity > 0.4:
        return "Something shifted and I am still figuring out how close it is to me personally."
    return "I am watching and waiting to see what this means for my situation."


# ---------------------------------------------------------------------------
# Batch appraisal — apply to all agents in the world
# ---------------------------------------------------------------------------

def appraise_shock_for_all(
    agents: dict[str, "WorldAgent"],
    shock: ShockProfile,
) -> dict[str, AgentShockReaction]:
    """Compute individual reactions for every agent. No hardcoded role lookup."""
    return {
        agent_id: appraise_shock(agent, shock)
        for agent_id, agent in agents.items()
    }


def apply_reactions(
    agents: dict[str, "WorldAgent"],
    reactions: dict[str, AgentShockReaction],
    tick: int,
    shock_label: str,
    shock_kind: str,
    source_text: str,
) -> dict:
    """Apply computed individual reactions to agent state."""
    impacted = 0
    total_severity = 0.0
    awareness_set: set[str] = set()

    for agent_id, reaction in reactions.items():
        if reaction.personal_severity < 0.01:
            continue
        agent = agents[agent_id]
        impacted += 1
        total_severity += reaction.personal_severity

        agent.debt_pressure = _clamp(agent.debt_pressure + reaction.debt_pressure_delta)
        agent.dread_pressure = _clamp(agent.dread_pressure + reaction.dread_pressure_delta)
        agent.secret_pressure = _clamp(agent.secret_pressure + reaction.secret_pressure_delta)
        agent.heart.tension = _clamp(agent.heart.tension + reaction.tension_delta)
        agent.heart.valence = _clamp(agent.heart.valence + reaction.valence_delta)

        # Memory with personal interpretation
        agent.add_memory(
            tick,
            f"[external:{shock_kind}] {shock_label}: {source_text}",
        )
        # Override the generic interpretation with the personal one
        if agent.memory and reaction.interpretation:
            agent.memory[-1].interpretation = reaction.interpretation

        awareness_set.add(agent_id)

    return {
        "impacted_agents": impacted,
        "total_agents": len(agents),
        "avg_severity": round(total_severity / max(1, impacted), 4),
        "awareness_set": awareness_set,
    }

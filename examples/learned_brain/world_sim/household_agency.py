"""Household Agency — EVERY agent makes LLM-driven personal finance decisions.

This replaces the executive-only LLM agency. Now every agent is a household
that decides:
  - How to adjust spending (cut groceries, skip eating out, cancel subscriptions)
  - Whether to tap savings or take on credit
  - Whether to look for a second job / side hustle
  - Whether to ask family/friends for help
  - Whether to defer rent/bills or negotiate with landlord
  - Whether to organize with neighbors or protest

These individual household decisions ARE the economy. When 60% of households
cut spending, that IS the recession. When households tap savings, that IS
the depletion. The macro outcomes emerge from millions of kitchen-table decisions.

Cost management:
  - Not every agent gets an LLM call every tick (too expensive)
  - Agents are sampled: ~20-30 per tick from different segments
  - Their decisions are applied to similar agents nearby (representative sampling)
  - High-stress agents get priority for LLM calls
  - Executives still get LLM calls for corporate decisions via llm_agency.py

Unemployment circuit breakers (from real-world research):
  1. Government auto-stabilizers: unemployment benefits kick in automatically
  2. Essential sectors keep running (healthcare, utilities, government, food)
  3. Companies prefer hour cuts over layoffs (cheaper than rehiring later)
  4. Savings buffers absorb months of reduced income
  5. Credit access extends the runway
  6. Family/community support networks catch people before total destitution
  7. Side hustles / gig economy provides partial replacement income
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World
    from .world_agent import WorldAgent
    from .ripple_engine import OrganizationalFabric, RippleEvent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Household decision options — what real people actually decide
# ---------------------------------------------------------------------------

HOUSEHOLD_DECISIONS = [
    "CUT_DISCRETIONARY",     # stop eating out, cancel subscriptions, buy generic
    "CUT_ESSENTIALS",        # reduce groceries, skip meals, defer medical
    "TAP_SAVINGS",           # withdraw from savings to cover bills
    "TAKE_ON_CREDIT",        # use credit cards, borrow from bank
    "SEEK_SECOND_JOB",       # side hustle, gig work, extra shifts
    "ASK_FAMILY_HELP",       # borrow from relatives, move in with parents
    "DEFER_BILLS",           # skip rent, negotiate with landlord, late on utilities
    "SPEND_NORMALLY",        # maintain current lifestyle (if affordable)
    "INCREASE_SAVINGS",      # save more if feeling anxious about future
    "ORGANIZE_COMMUNITY",    # mutual aid, community support, group action
]

# Essential sectors that don't fully shut down even in severe crises
ESSENTIAL_ROLES = {"healthcare_worker", "government_worker", "teacher"}

# Sectors that are most exposed to demand shocks
EXPOSED_ROLES = {"retail_worker", "gig_worker", "small_business_owner"}


# ---------------------------------------------------------------------------
# Unemployment circuit breakers — prevent unrealistic collapse
# ---------------------------------------------------------------------------

def apply_circuit_breakers(world: "World", tick: int):
    """Apply real-world mechanisms that prevent total economic collapse.

    These fire automatically (not via LLM) because they represent
    STRUCTURAL features of modern economies, not individual decisions.
    """
    agents = world.agents
    n = len(agents)
    unemployed = sum(1 for a in agents.values() if not a.employed)
    unemp_rate = unemployed / n

    # --- 1. Automatic unemployment benefits ---
    # Real: laid-off workers get ~40-60% of income via UI (RESEARCH: UI is 8x more
    # effective per dollar than tax cuts per NBER). $300-600/week federal + state.
    # This prevents debt from spiraling to infinity after job loss.
    for agent in agents.values():
        if not agent.employed and agent.debt_pressure > 0.1:
            # UI benefit: replaces ~45% of income, reducing debt buildup significantly
            ui_benefit = 0.025  # stronger — real UI prevents destitution
            agent.debt_pressure = max(0.05, agent.debt_pressure - ui_benefit)
            # UI also provides income floor
            agent.income_level = max(agent.income_level, 0.25)  # ~45% replacement

    # --- 2. Essential sector protection ---
    # RESEARCH: ~60-65% of economy is recession-proof. Government alone = 22M workers.
    # Healthcare, education, utilities, government keep running through any crisis.
    # In COVID, 55M workers were classified "essential" (36% of workforce).
    for agent in agents.values():
        if agent.social_role in ESSENTIAL_ROLES:
            if not agent.employed:
                agent.employed = True  # essential workers get rehired quickly
                agent.income_level = max(agent.income_level, 0.4)
        # Non-essential but recession-resistant: tech workers with remote capability
        if agent.social_role == "tech_worker" and not agent.employed:
            if agent.savings_buffer > 0.1:  # can survive while job hunting
                agent.income_level = max(agent.income_level, 0.15)  # freelance/gig income

    # --- 3. Government emergency hiring + fiscal stimulus ---
    # RESEARCH: $1.5T (2008) / $5T (COVID) in fiscal response.
    # ARRA created ~3.5M jobs. PPP supported ~7.5M jobs.
    # Social Security alone pumps ~$2T/year regardless of conditions.
    if unemp_rate > 0.06:
        unemployed_agents = [a for a in agents.values() if not a.employed]
        # Government absorbs ~3% of unemployed per cycle (PPP/ARRA effect)
        n_absorb = max(1, int(len(unemployed_agents) * 0.03))
        for agent in random.sample(unemployed_agents, min(n_absorb, len(unemployed_agents))):
            agent.employed = True
            agent.income_level = 0.3  # emergency/gig work
            agent.debt_pressure = max(0.0, agent.debt_pressure - 0.02)

    # Stimulus checks: when unemployment > 10%, broad debt relief
    # RESEARCH: $844B in stimulus checks across 3 rounds
    if unemp_rate > 0.10:
        for agent in agents.values():
            agent.debt_pressure = max(0.0, agent.debt_pressure - 0.008)  # stimulus effect

    # --- 4. Company retention incentives (prefer cuts over layoffs) ---
    # Real: companies keep workers at reduced hours because rehiring is expensive
    # Effect: income_level drops but employed stays True
    for agent in agents.values():
        if agent.employed and agent.income_level < 0.15:
            # Floor on income for employed workers (minimum wage equivalent)
            agent.income_level = 0.15

    # --- 5. Savings floor (people don't go to literally zero) ---
    # Real: people always have some emergency reserve, even if tiny
    for agent in agents.values():
        if agent.savings_buffer < 0.01:
            agent.savings_buffer = 0.01  # always have pocket change


# ---------------------------------------------------------------------------
# Household decision prompt
# ---------------------------------------------------------------------------

def _build_household_prompt(agent: "WorldAgent") -> str:
    profile = agent.get_human_profile()
    employed_str = "employed" if agent.employed else "UNEMPLOYED"

    return f"""You are {agent.personality.name}, {agent.personality.background}.
Personality: {agent.personality.temperament}
You cope by: {profile.get('coping_style', '?')}. You fear: {profile.get('threat_lens', '?')}.
What you need most: {profile.get('core_need', '?')}. Your self-story: {profile.get('self_story', '?')}.

YOUR HOUSEHOLD FINANCES:
- Employment: {employed_str}
- Income level: {agent.income_level:.0%} of normal
- Savings: {agent.savings_buffer:.0%} remaining
- Credit available: {agent.credit_access:.0%}
- Debt pressure: {agent.debt_pressure:.0%} (how stressed your finances are)
- Dread level: {agent.dread_pressure:.0%}

RECENT EVENTS:
{chr(10).join('- ' + m.description for m in agent.get_recent_memories(3)) or 'Nothing notable.'}

You need to make a REAL decision about your household finances this week.
Think about your personality — are you the type to cut spending first,
or tap savings, or ask for help, or hustle harder?

YOUR OPTIONS:
- CUT_DISCRETIONARY: stop eating out, cancel subscriptions, buy cheap brands
- CUT_ESSENTIALS: reduce groceries, skip medical, defer repairs
- TAP_SAVINGS: dip into savings to maintain lifestyle
- TAKE_ON_CREDIT: use credit cards or borrow to cover shortfall
- SEEK_SECOND_JOB: look for side work, gig jobs, extra shifts
- ASK_FAMILY_HELP: borrow from relatives or accept help
- DEFER_BILLS: skip rent or utility payments, negotiate delays
- SPEND_NORMALLY: keep spending as usual (if you can afford it)
- INCREASE_SAVINGS: save more because you're worried about the future
- ORGANIZE_COMMUNITY: help neighbors, join mutual aid, collective action

Return ONLY valid JSON:
{{
  "decision": "<one option from above>",
  "intensity": <0.1 to 1.0>,
  "what_you_cut_or_do": "<specific thing: e.g. 'cancelled Netflix and stopped buying coffee'>",
  "what_you_tell_family": "<what you say at the dinner table>",
  "what_you_worry_about": "<your private fear>"
}}"""


# ---------------------------------------------------------------------------
# Parse and apply household decisions
# ---------------------------------------------------------------------------

@dataclass
class HouseholdDecision:
    agent_id: str
    agent_name: str
    decision: str = "SPEND_NORMALLY"
    intensity: float = 0.5
    what_cut: str = ""
    family_talk: str = ""
    worry: str = ""
    valid: bool = True


def _parse_household_decision(raw: str, agent_id: str, agent_name: str) -> HouseholdDecision:
    d = HouseholdDecision(agent_id=agent_id, agent_name=agent_name)
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        d.valid = False
        return d
    json_str = text[start:end + 1] if end > start else text[start:]
    if not json_str.endswith("}"):
        json_str = json_str.rstrip().rstrip(",") + '"}'
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            repaired = json_str
            if repaired.count('"') % 2: repaired += '"'
            while repaired.count("{") > repaired.count("}"): repaired += "}"
            data = json.loads(repaired)
        except json.JSONDecodeError:
            d.valid = False
            return d

    decision = str(data.get("decision", "SPEND_NORMALLY")).strip().upper()
    if decision not in [d for d in HOUSEHOLD_DECISIONS]:
        decision = "SPEND_NORMALLY"
    d.decision = decision
    d.intensity = max(0.1, min(1.0, float(data.get("intensity", 0.5) or 0.5)))
    d.what_cut = str(data.get("what_you_cut_or_do", ""))[:200]
    d.family_talk = str(data.get("what_you_tell_family", ""))[:200]
    d.worry = str(data.get("what_you_worry_about", ""))[:200]
    return d


def _apply_household_decision(agent: "WorldAgent", decision: HouseholdDecision, world: "World"):
    """Apply a household decision to the agent's state."""
    mag = decision.intensity
    tick = world.tick_count

    if decision.decision == "CUT_DISCRETIONARY":
        # Reduces spending → vendors lose revenue (via reduced spending pressure)
        # But protects savings
        agent.debt_pressure = max(0.0, agent.debt_pressure - 0.02 * mag)
        agent.heart.valence = max(0.1, agent.heart.valence - 0.01 * mag)  # slight mood hit

    elif decision.decision == "CUT_ESSENTIALS":
        # Bigger savings but more suffering
        agent.debt_pressure = max(0.0, agent.debt_pressure - 0.04 * mag)
        agent.heart.valence = max(0.1, agent.heart.valence - 0.03 * mag)
        agent.dread_pressure = _clamp(agent.dread_pressure + 0.02 * mag)

    elif decision.decision == "TAP_SAVINGS":
        # Use savings to cover bills
        if agent.savings_buffer > 0.05:
            drawn = min(0.08 * mag, agent.savings_buffer * 0.15)
            agent.savings_buffer = max(0.01, agent.savings_buffer - drawn)
            agent.debt_pressure = max(0.0, agent.debt_pressure - drawn * 0.6)

    elif decision.decision == "TAKE_ON_CREDIT":
        if agent.credit_access > 0.05:
            borrowed = min(0.06 * mag, agent.credit_access * 0.1)
            agent.credit_access = max(0.0, agent.credit_access - borrowed)
            agent.debt_pressure = max(0.0, agent.debt_pressure - borrowed * 0.4)
            # But future debt increases (credit has to be repaid)
            agent.debt_pressure = _clamp(agent.debt_pressure + borrowed * 0.1)

    elif decision.decision == "SEEK_SECOND_JOB":
        # Partial income recovery but exhaustion
        agent.income_level = min(1.0, agent.income_level + 0.08 * mag)
        agent.heart.energy = max(0.1, agent.heart.energy - 0.03 * mag)
        if not agent.employed:
            agent.employed = True
            agent.income_level = max(agent.income_level, 0.2)

    elif decision.decision == "ASK_FAMILY_HELP":
        # Small debt relief, emotional cost
        agent.debt_pressure = max(0.0, agent.debt_pressure - 0.03 * mag)
        agent.heart.tension = _clamp(agent.heart.tension + 0.02 * mag)  # shame/pride hit

    elif decision.decision == "DEFER_BILLS":
        # Temporary relief but builds future pressure
        agent.debt_pressure = max(0.0, agent.debt_pressure - 0.05 * mag)
        # Deferred bills come back later (stored as future debt)
        agent.debt_pressure = _clamp(agent.debt_pressure + 0.02 * mag)

    elif decision.decision == "SPEND_NORMALLY":
        # No change — but if can't afford it, debt rises
        if agent.income_level < 0.4 and agent.debt_pressure > 0.3:
            agent.debt_pressure = _clamp(agent.debt_pressure + 0.01)

    elif decision.decision == "INCREASE_SAVINGS":
        # Tighten belt to save more
        agent.savings_buffer = min(1.0, agent.savings_buffer + 0.02 * mag)
        agent.debt_pressure = _clamp(agent.debt_pressure + 0.01 * mag)  # slight cost

    elif decision.decision == "ORGANIZE_COMMUNITY":
        # Build social capital
        nearby = [a for a in world.agents.values()
                  if a.location == agent.location and a.agent_id != agent.agent_id][:4]
        for other in nearby:
            rel = world.relationships.get_or_create(agent.agent_id, other.agent_id)
            rel.warmth = min(1.0, rel.warmth + 0.03)
            rel.alliance_strength = min(1.0, rel.alliance_strength + 0.02)

    # Memory
    agent.add_memory(tick, f"[household:{decision.decision}] {decision.what_cut or decision.decision}")
    if agent.memory and decision.worry:
        agent.memory[-1].interpretation = decision.worry


# ---------------------------------------------------------------------------
# Household Agency Engine
# ---------------------------------------------------------------------------

class HouseholdAgencyEngine:
    """Runs LLM calls for sampled household agents each tick.

    Samples ~20 agents per tick across all segments.
    Priority: agents under most financial stress get called first.
    Their decisions also influence similar nearby agents (representative effect).
    """

    def __init__(self, api_key: str, model: str = "gpt-5-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.total_calls = 0
        self.total_decisions = 0
        self.decision_log: list[dict] = []
        self._last_call_tick: dict[str, int] = {}

    def tick(self, world: "World", max_calls: int = 15) -> list[HouseholdDecision]:
        """Sample household agents and get their decisions."""
        tick = world.tick_count

        # Don't call every tick — every 4 ticks (4 hours)
        if tick % 4 != 0:
            return []

        # Apply circuit breakers every tick
        apply_circuit_breakers(world, tick)

        # Select agents: prioritize stressed ones, ensure diversity
        candidates = []
        for aid, agent in world.agents.items():
            if self._last_call_tick.get(aid, 0) + 48 > tick:  # 48-tick cooldown (2 days)
                continue
            # Skip executives (they're handled by llm_agency.py)
            if agent.social_role in ("manager", "office_professional", "government_worker"):
                continue
            stress = agent.debt_pressure + agent.dread_pressure + (0.3 if not agent.employed else 0)
            candidates.append((stress, aid))

        candidates.sort(reverse=True)

        # Take top stressed + random sample for diversity
        n_stressed = min(max_calls * 2 // 3, len(candidates))
        n_random = min(max_calls - n_stressed, len(candidates) - n_stressed)
        selected_ids = [aid for _, aid in candidates[:n_stressed]]
        if n_random > 0 and len(candidates) > n_stressed:
            remaining = [aid for _, aid in candidates[n_stressed:]]
            selected_ids.extend(random.sample(remaining, min(n_random, len(remaining))))

        selected_ids = selected_ids[:max_calls]
        decisions: list[HouseholdDecision] = []

        for aid in selected_ids:
            agent = world.agents[aid]
            prompt = _build_household_prompt(agent)

            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=300,
                    reasoning={"effort": "low"},
                )
                raw = (resp.output_text or "").strip()
            except Exception:
                continue

            self.total_calls += 1
            decision = _parse_household_decision(raw, aid, agent.personality.name)
            if not decision.valid:
                continue

            self.total_decisions += 1
            self._last_call_tick[aid] = tick

            # Apply decision
            _apply_household_decision(agent, decision, world)

            # Representative effect: similar nearby agents make similar choices
            # (not via LLM, but via behavioral mirroring)
            similar = [
                a for a in world.agents.values()
                if a.social_role == agent.social_role
                and a.location == agent.location
                and a.agent_id != aid
                and abs(a.debt_pressure - agent.debt_pressure) < 0.15
            ][:3]
            for sim_agent in similar:
                # Apply 50% of the decision effect
                _apply_household_decision(sim_agent, HouseholdDecision(
                    agent_id=sim_agent.agent_id,
                    agent_name=sim_agent.personality.name,
                    decision=decision.decision,
                    intensity=decision.intensity * 0.5,
                ), world)

            decisions.append(decision)
            self.decision_log.append({
                "tick": tick,
                "agent": decision.agent_name,
                "role": agent.social_role,
                "decision": decision.decision,
                "intensity": round(decision.intensity, 2),
                "employed": agent.employed,
                "debt": round(agent.debt_pressure, 2),
                "savings": round(agent.savings_buffer, 2),
                "what": decision.what_cut[:80],
                "family": decision.family_talk[:80],
                "worry": decision.worry[:80],
            })

        return decisions

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_decisions": self.total_decisions,
            "recent": self.decision_log[-10:],
        }

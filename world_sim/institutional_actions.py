"""Institutional decision system — government and corporate agents make policy/business decisions.

Government agents (with different roles/priorities) collectively deliberate and produce
policy actions: lockdowns, stimulus packages, emergency regulations, public messaging.
Corporate agents (managers, executives) make business decisions: layoff waves, hiring
freezes, investment cuts, price controls.

These decisions emerge from individual agent psychology and cascade into the economy.

Architecture:
  - Government agents "vote" based on their individual priorities/pressures
  - Corporate agents decide based on their sector's economic stress
  - Institutional decisions create ScheduledEvents that affect many agents
  - Decisions happen at institutional decision intervals (every 48-72 ticks / 2-3 days)
  - Each decision has delayed second-order effects
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World, ScheduledEvent
    from .world_agent import WorldAgent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Institutional action types
# ---------------------------------------------------------------------------

@dataclass
class InstitutionalDecision:
    """One institutional decision taken by a group of agents."""
    decision_type: str
    magnitude: float  # 0-1
    support_ratio: float  # what fraction of decision-makers supported it
    description: str
    effects: dict = field(default_factory=dict)  # what it does to the world


@dataclass
class InstitutionalResolution:
    """Summary of institutional decisions this tick."""
    government_decisions: list[InstitutionalDecision] = field(default_factory=list)
    corporate_decisions: list[InstitutionalDecision] = field(default_factory=list)
    total_agents_affected: int = 0

    def as_dict(self) -> dict:
        return {
            "government_decisions": [
                {"type": d.decision_type, "magnitude": round(d.magnitude, 3),
                 "support": round(d.support_ratio, 2), "description": d.description}
                for d in self.government_decisions
            ],
            "corporate_decisions": [
                {"type": d.decision_type, "magnitude": round(d.magnitude, 3),
                 "support": round(d.support_ratio, 2), "description": d.description}
                for d in self.corporate_decisions
            ],
            "total_agents_affected": self.total_agents_affected,
        }


# ---------------------------------------------------------------------------
# Government deliberation
# ---------------------------------------------------------------------------

def _government_deliberation(world: "World") -> list[InstitutionalDecision]:
    """Government agents collectively decide policy based on their individual priorities.

    Each gov agent has their own threat_lens, core_need, and pressure state.
    They "vote" on policy options. Majority determines action.
    """
    gov_agents = [a for a in world.agents.values() if a.social_role == "government_worker"]
    if len(gov_agents) < 3:
        return []

    # What are the conditions?
    from .macro_aggregator import SECTOR_MAP
    from statistics import mean
    all_agents = list(world.agents.values())
    avg_debt = mean(a.debt_pressure for a in all_agents)
    avg_dread = mean(a.dread_pressure for a in all_agents)
    avg_tension = mean(a.heart.tension for a in all_agents)
    unemployment_proxy = sum(1 for a in all_agents if a.debt_pressure > 0.5) / len(all_agents)
    health_crisis = avg_dread > 0.15
    economic_crisis = avg_debt > 0.2

    decisions = []

    # --- Stimulus package vote ---
    if economic_crisis:
        votes_for = 0
        for gov in gov_agents:
            profile = gov.get_human_profile()
            # Agents with safety/belonging needs favor stimulus
            # Agents with control/dignity needs prefer austerity
            favors = (
                (profile["core_need"] in ("safety", "belonging", "usefulness")) or
                (profile["threat_lens"] == "scarcity") or
                (gov.debt_pressure > 0.3) or
                (gov.appraisal.economic_pressure > 0.4)
            )
            if favors:
                votes_for += 1
        support = votes_for / len(gov_agents)
        if support > 0.4:  # some support needed
            magnitude = min(1.0, avg_debt * 1.5 * support)
            decisions.append(InstitutionalDecision(
                decision_type="STIMULUS_PACKAGE",
                magnitude=magnitude,
                support_ratio=support,
                description=f"Government passes stimulus ({support * 100:.0f}% support): emergency relief for affected workers and businesses",
                effects={"debt_relief": magnitude * 0.08, "confidence_boost": magnitude * 0.03},
            ))

    # --- Emergency health measures vote ---
    if health_crisis:
        votes_for = 0
        for gov in gov_agents:
            profile = gov.get_human_profile()
            favors = (
                (profile["core_need"] in ("safety", "usefulness", "control")) or
                (profile["threat_lens"] in ("chaos", "abandonment")) or
                (gov.dread_pressure > 0.15)
            )
            if favors:
                votes_for += 1
        support = votes_for / len(gov_agents)
        if support > 0.5:
            magnitude = min(1.0, avg_dread * 2.0 * support)
            decisions.append(InstitutionalDecision(
                decision_type="HEALTH_EMERGENCY_MEASURES",
                magnitude=magnitude,
                support_ratio=support,
                description=f"Government declares health emergency ({support * 100:.0f}% support): closures, quarantine, hospital surge funding",
                effects={"dread_relief": magnitude * 0.04, "economic_cost": magnitude * 0.06},
            ))

    # --- Public messaging / reassurance ---
    if avg_tension > 0.15 or avg_dread > 0.1:
        # Government always tries to communicate; effectiveness varies
        trust_level = mean(gov.heart.valence for gov in gov_agents)
        magnitude = min(1.0, trust_level * 0.8)
        decisions.append(InstitutionalDecision(
            decision_type="PUBLIC_REASSURANCE",
            magnitude=magnitude,
            support_ratio=1.0,
            description="Government issues public statement urging calm and outlining response plans",
            effects={"tension_relief": magnitude * 0.02, "trust_boost": magnitude * 0.01},
        ))

    return decisions


# ---------------------------------------------------------------------------
# Corporate decisions
# ---------------------------------------------------------------------------

def _corporate_decisions(world: "World") -> list[InstitutionalDecision]:
    """Managers/executives make business decisions based on their sector's economic stress."""
    managers = [a for a in world.agents.values() if a.social_role in ("manager", "office_professional")]
    if len(managers) < 2:
        return []

    decisions = []
    from statistics import mean

    avg_manager_debt = mean(a.debt_pressure for a in managers)
    avg_manager_tension = mean(a.heart.tension for a in managers)
    avg_pessimism = mean(getattr(a, "expectation_pessimism", 0.0) for a in managers)

    # --- Layoff wave ---
    # Managers collectively decide layoffs when economic pressure is sustained
    if avg_manager_debt > 0.25 and avg_pessimism > 0.1:
        votes_for = 0
        for mgr in managers:
            profile = mgr.get_human_profile()
            favors = (
                mgr.debt_pressure > 0.2 or
                getattr(mgr, "expectation_pessimism", 0) > 0.15 or
                (profile["core_need"] == "control" and mgr.heart.tension > 0.2)
            )
            if favors:
                votes_for += 1
        support = votes_for / len(managers)
        if support > 0.5:
            magnitude = min(1.0, (avg_manager_debt + avg_pessimism) * support)
            decisions.append(InstitutionalDecision(
                decision_type="CORPORATE_LAYOFF_WAVE",
                magnitude=magnitude,
                support_ratio=support,
                description=f"Companies announce layoffs ({support * 100:.0f}% of managers support): cost-cutting across sectors",
                effects={"jobs_lost_pct": magnitude * 0.05, "debt_spike": magnitude * 0.1},
            ))

    # --- Hiring freeze ---
    if avg_pessimism > 0.08:
        magnitude = min(1.0, avg_pessimism * 2.0)
        decisions.append(InstitutionalDecision(
            decision_type="HIRING_FREEZE",
            magnitude=magnitude,
            support_ratio=0.8,
            description="Companies freeze hiring as outlook darkens",
            effects={"pessimism_sustain": magnitude * 0.02},
        ))

    # --- Investment cut ---
    if avg_manager_debt > 0.3:
        magnitude = min(1.0, avg_manager_debt * 1.2)
        decisions.append(InstitutionalDecision(
            decision_type="INVESTMENT_CUT",
            magnitude=magnitude,
            support_ratio=0.7,
            description="Companies cut capital investment and delay expansion plans",
            effects={"market_pressure": magnitude * 0.03},
        ))

    return decisions


# ---------------------------------------------------------------------------
# Apply institutional decisions to the world
# ---------------------------------------------------------------------------

def _apply_government_decision(world: "World", decision: InstitutionalDecision) -> int:
    """Apply a government decision to all relevant agents."""
    affected = 0
    effects = decision.effects

    if decision.decision_type == "STIMULUS_PACKAGE":
        # Reduce debt pressure for most vulnerable agents
        relief = effects.get("debt_relief", 0.05)
        conf_boost = effects.get("confidence_boost", 0.02)
        for agent in world.agents.values():
            if agent.debt_pressure > 0.15:
                agent.debt_pressure = _clamp(agent.debt_pressure - relief * (1 + agent.debt_pressure))
                agent.heart.tension = max(0.0, agent.heart.tension - conf_boost)
                affected += 1
                agent.add_memory(world.tick_count, "[gov] Stimulus package: emergency relief funds distributed")

    elif decision.decision_type == "HEALTH_EMERGENCY_MEASURES":
        # Reduce dread but increase economic cost
        dread_relief = effects.get("dread_relief", 0.03)
        econ_cost = effects.get("economic_cost", 0.04)
        for agent in world.agents.values():
            if agent.dread_pressure > 0.1:
                agent.dread_pressure = _clamp(agent.dread_pressure - dread_relief)
                affected += 1
            # Economic cost of lockdowns/closures
            if agent.social_role in ("market_vendor", "bartender", "factory_worker", "dock_worker"):
                agent.debt_pressure = _clamp(agent.debt_pressure + econ_cost)
                agent.add_memory(world.tick_count, "[gov] Emergency health measures: business restrictions imposed")

    elif decision.decision_type == "PUBLIC_REASSURANCE":
        tension_relief = effects.get("tension_relief", 0.015)
        trust_boost = effects.get("trust_boost", 0.01)
        for agent in world.agents.values():
            agent.heart.tension = max(0.0, agent.heart.tension - tension_relief)
            affected += 1

    return affected


def _apply_corporate_decision(world: "World", decision: InstitutionalDecision) -> int:
    """Apply a corporate decision to workers."""
    affected = 0
    effects = decision.effects

    if decision.decision_type == "CORPORATE_LAYOFF_WAVE":
        # Hit workers across industrial/service sectors
        workers = [
            a for a in world.agents.values()
            if a.social_role in ("factory_worker", "dock_worker", "office_worker", "bartender")
        ]
        jobs_lost_pct = effects.get("jobs_lost_pct", 0.03)
        debt_spike = effects.get("debt_spike", 0.08)
        import random
        rng = random.Random(world.tick_count)
        # Some workers get laid off, rest get survivor anxiety
        n_laid_off = max(1, int(len(workers) * jobs_lost_pct))
        laid_off = rng.sample(workers, min(n_laid_off, len(workers)))

        for worker in laid_off:
            worker.debt_pressure = _clamp(worker.debt_pressure + 0.25)
            worker.dread_pressure = _clamp(worker.dread_pressure + 0.12)
            worker.heart.valence = max(0.1, worker.heart.valence - 0.12)
            worker.heart.wounds.append((0.06, 0.995))
            worker.add_memory(world.tick_count, "[corporate] Laid off in company-wide cuts — income gone")
            affected += 1

        # Survivor anxiety for remaining workers
        survivors = [w for w in workers if w not in laid_off]
        for worker in survivors:
            worker.dread_pressure = _clamp(worker.dread_pressure + 0.04)
            worker.heart.tension = min(1.0, worker.heart.tension + 0.03)
            affected += 1

        # Community impact: families of laid-off workers
        for agent in world.agents.values():
            if agent.social_role in ("community", "retiree") and agent.debt_pressure > 0.1:
                agent.debt_pressure = _clamp(agent.debt_pressure + debt_spike * 0.3)

    elif decision.decision_type == "HIRING_FREEZE":
        # Students and job seekers feel it most
        pessimism_sustain = effects.get("pessimism_sustain", 0.015)
        for agent in world.agents.values():
            if agent.social_role in ("student", "community"):
                agent.expectation_pessimism = _clamp(
                    getattr(agent, "expectation_pessimism", 0) + pessimism_sustain
                )
                affected += 1

    elif decision.decision_type == "INVESTMENT_CUT":
        mp = effects.get("market_pressure", 0.02)
        for agent in world.agents.values():
            if agent.social_role in ("office_professional", "manager"):
                agent.expectation_pessimism = _clamp(
                    getattr(agent, "expectation_pessimism", 0) + mp
                )
                affected += 1

    return affected


# ---------------------------------------------------------------------------
# Main entry point — called from world tick loop
# ---------------------------------------------------------------------------

def resolve_institutional_decisions(world: "World", tick: int) -> InstitutionalResolution:
    """Government and corporate agents make institutional decisions.

    Called every 48 ticks (2 days) — institutional decisions happen on
    a slower timescale than individual actions.
    """
    # Only deliberate every 48 ticks (2 sim-days) during work hours
    if tick % 48 != 0 or world.hour_of_day < 9 or world.hour_of_day > 17:
        return InstitutionalResolution()

    resolution = InstitutionalResolution()

    # Government deliberation
    gov_decisions = _government_deliberation(world)
    for decision in gov_decisions:
        affected = _apply_government_decision(world, decision)
        resolution.total_agents_affected += affected
    resolution.government_decisions = gov_decisions

    # Corporate decisions
    corp_decisions = _corporate_decisions(world)
    for decision in corp_decisions:
        affected = _apply_corporate_decision(world, decision)
        resolution.total_agents_affected += affected
    resolution.corporate_decisions = corp_decisions

    return resolution

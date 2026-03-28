"""Ripple Engine — concrete cause-and-effect chains between named individuals.

This makes the world ALIVE. When oil prices surge:
  - Captain Daria at the docks calculates new fuel costs
  - She tells her crew: overtime is gone, effective immediately
  - Dock worker Taro loses $400/month income
  - Taro tells his wife they can't afford the after-school program
  - He reduces spending at Hans's market stall
  - Hans notices sales are down 30% this week
  - Hans raises bread prices by 15%
  - Community elder Kira can't afford her usual groceries
  - She calls an emergency mutual aid meeting
  - Government worker Caleb hears about the meeting and drafts a memo
  - The memo reaches the city council
  - Council votes on emergency food assistance

Each step is a concrete event between named agents, driven by their
individual personality, relationships, and economic position.

Architecture:
  - OrganizationalLinks: who employs whom, who supplies whom, who rents from whom
  - RippleChain: a traceable chain of cause → effect → cause → effect
  - RippleEngine: detects when an agent's state change should trigger
    a concrete consequence for a specific other agent
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World
    from .world_agent import WorldAgent
    from .relationship import RelationshipStore


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Organizational links — the structural fabric of the world
# ---------------------------------------------------------------------------

@dataclass
class OrgLink:
    """A structural relationship between two agents."""
    from_id: str
    to_id: str
    link_type: str  # "employs", "supplies", "rents_from", "teaches", "treats", "manages"
    strength: float = 1.0  # how dependent the relationship is


class OrganizationalFabric:
    """Maps who employs/supplies/rents from whom.

    Built from agent roles and co-location patterns.
    These structural links determine HOW economic pressure flows.
    """

    def __init__(self):
        self.links: list[OrgLink] = []
        self._by_from: dict[str, list[OrgLink]] = defaultdict(list)
        self._by_to: dict[str, list[OrgLink]] = defaultdict(list)
        self._by_type: dict[str, list[OrgLink]] = defaultdict(list)

    def add(self, link: OrgLink):
        self.links.append(link)
        self._by_from[link.from_id].append(link)
        self._by_to[link.to_id].append(link)
        self._by_type[link.link_type].append(link)

    def get_dependents(self, agent_id: str) -> list[OrgLink]:
        """Who depends on this agent (employees, customers, tenants)."""
        return self._by_from[agent_id]

    def get_dependencies(self, agent_id: str) -> list[OrgLink]:
        """Who this agent depends on (employer, supplier, landlord)."""
        return self._by_to[agent_id]

    def links_of_type(self, link_type: str) -> list[OrgLink]:
        return self._by_type[link_type]


def build_organizational_fabric(world: "World") -> OrganizationalFabric:
    """Infer organizational links from agent roles, locations, and coalitions.

    Managers employ workers at the same work location.
    Market vendors supply community members who shop there.
    Healthcare workers treat community members.
    Teachers teach students.
    """
    fabric = OrganizationalFabric()
    agents = world.agents
    rng = random.Random(42)

    # Group agents by work location
    by_work_loc: dict[str, list[str]] = defaultdict(list)
    for aid, agent in agents.items():
        for hour, loc in agent.schedule.items():
            if 9 <= hour <= 16:
                by_work_loc[loc].append(aid)
                break

    # Managers → workers at same location
    for loc, agent_ids in by_work_loc.items():
        managers_here = [aid for aid in agent_ids if agents[aid].social_role in ("manager", "office_professional")]
        workers_here = [aid for aid in agent_ids if agents[aid].social_role in ("factory_worker", "dock_worker", "office_worker", "bartender")]
        for mgr_id in managers_here:
            assigned = rng.sample(workers_here, min(8, len(workers_here)))
            for wid in assigned:
                fabric.add(OrgLink(mgr_id, wid, "employs", strength=0.8))

    # Market vendors → community customers
    vendors = [aid for aid, a in agents.items() if a.social_role == "market_vendor"]
    consumers = [aid for aid, a in agents.items() if a.social_role in ("community", "factory_worker", "dock_worker", "retiree", "teacher", "student")]
    for vid in vendors:
        n_customers = min(15, len(consumers))
        customers = rng.sample(consumers, n_customers)
        for cid in customers:
            fabric.add(OrgLink(vid, cid, "supplies", strength=0.5))

    # Healthcare → community patients
    healthcare = [aid for aid, a in agents.items() if a.social_role == "healthcare"]
    for hid in healthcare:
        patients = rng.sample(consumers, min(12, len(consumers)))
        for pid in patients:
            fabric.add(OrgLink(hid, pid, "treats", strength=0.4))

    # Teachers → students
    teachers = [aid for aid, a in agents.items() if a.social_role == "teacher"]
    students = [aid for aid, a in agents.items() if a.social_role == "student"]
    for tid in teachers:
        assigned = rng.sample(students, min(10, len(students)))
        for sid in assigned:
            fabric.add(OrgLink(tid, sid, "teaches", strength=0.5))

    # Government → everyone (policy reach)
    gov = [aid for aid, a in agents.items() if a.social_role == "government_worker"]
    all_ids = list(agents.keys())
    for gid in gov:
        # Each gov worker has a "constituency" they're responsible for
        constituency = rng.sample(all_ids, min(30, len(all_ids)))
        for cid in constituency:
            if cid != gid:
                fabric.add(OrgLink(gid, cid, "governs", strength=0.3))

    return fabric


# ---------------------------------------------------------------------------
# Ripple chain — traceable cause-and-effect
# ---------------------------------------------------------------------------

@dataclass
class RippleEvent:
    """One link in a cause-and-effect chain."""
    tick: int
    actor_id: str
    actor_name: str
    target_id: str
    target_name: str
    action: str  # what the actor did
    consequence: str  # what happened to the target
    mechanism: str  # via what structural link
    debt_delta: float = 0.0
    dread_delta: float = 0.0
    tension_delta: float = 0.0
    valence_delta: float = 0.0
    parent_event_idx: int | None = None  # index of the event that caused this one


@dataclass
class RippleResolution:
    """All ripple events this tick."""
    events: list[RippleEvent] = field(default_factory=list)
    chains_started: int = 0
    total_affected: int = 0

    def as_dict(self) -> dict:
        return {
            "events": [
                {
                    "actor": e.actor_name,
                    "target": e.target_name,
                    "action": e.action,
                    "consequence": e.consequence,
                    "mechanism": e.mechanism,
                }
                for e in self.events[:10]  # cap for performance
            ],
            "chains_started": self.chains_started,
            "total_affected": self.total_affected,
        }


# ---------------------------------------------------------------------------
# Ripple Engine — detects and propagates concrete consequences
# ---------------------------------------------------------------------------

class RippleEngine:
    """Detects when an agent's state change should ripple to specific others.

    Called each tick. Looks for agents whose economic pressure crossed
    a threshold since last tick and generates concrete consequences
    for the specific people connected to them via organizational links.
    """

    def __init__(self, fabric: OrganizationalFabric, seed: int = 42):
        self.fabric = fabric
        self.rng = random.Random(seed)
        self._prev_debt: dict[str, float] = {}
        self._prev_dread: dict[str, float] = {}
        self.event_log: list[RippleEvent] = []
        self._cooldowns: dict[str, int] = {}  # agent_id → tick when they last rippled

    def tick(self, world: "World") -> RippleResolution:
        """Detect state changes and propagate ripples."""
        resolution = RippleResolution()
        tick = world.tick_count
        affected_ids: set[str] = set()

        for agent_id, agent in world.agents.items():
            # Cooldown: each agent can only trigger one ripple per 6 ticks (6 hours)
            if self._cooldowns.get(agent_id, 0) > tick:
                continue

            prev_debt = self._prev_debt.get(agent_id, agent.debt_pressure)
            prev_dread = self._prev_dread.get(agent_id, agent.dread_pressure)

            debt_spike = agent.debt_pressure - prev_debt
            dread_spike = agent.dread_pressure - prev_dread

            # Store current for next tick
            self._prev_debt[agent_id] = agent.debt_pressure
            self._prev_dread[agent_id] = agent.dread_pressure

            # Only ripple if significant state change happened
            if debt_spike < 0.05 and dread_spike < 0.05:
                continue

            self._cooldowns[agent_id] = tick + 6  # 6-tick cooldown (6 hours)

            # What kind of ripple does this agent create?
            role = agent.social_role
            profile = agent.get_human_profile()
            dependents = self.fabric.get_dependents(agent_id)

            if not dependents:
                continue

            resolution.chains_started += 1

            # --- Manager/employer under pressure → cuts worker income ---
            if role in ("manager", "office_professional") and debt_spike > 0.06:
                employees = [l for l in dependents if l.link_type == "employs"]
                if employees:
                    targets = self.rng.sample(employees, min(3, len(employees)))
                    for link in targets:
                        if link.to_id not in world.agents:
                            continue
                        target = world.agents[link.to_id]
                        income_cut = 0.0
                        # Reduce income (not additive debt) — cuts don't stack infinitely
                        if target.income_level > 0.15:
                            income_cut = min(0.1, target.income_level * 0.15 * link.strength)
                            target.income_level = max(0.1, target.income_level - income_cut)
                            debt_bump = income_cut * 0.25
                            target.debt_pressure = _clamp(target.debt_pressure + debt_bump)
                            target.heart.tension = _clamp(target.heart.tension + debt_bump * 0.3)
                            affected_ids.add(link.to_id)

                        if income_cut > 0:
                            event = RippleEvent(
                                tick=tick,
                                actor_id=agent_id,
                                actor_name=agent.personality.name,
                                target_id=link.to_id,
                                target_name=target.personality.name,
                                action=f"cut hours/overtime due to cost pressure (dp={agent.debt_pressure:.2f})",
                                consequence=f"lost income, debt_pressure +{income_cut:.3f}",
                                mechanism="employs",
                                debt_delta=income_cut,
                            )
                            resolution.events.append(event)
                            self.event_log.append(event)

                            target.add_memory(tick,
                                f"{agent.personality.name} cut my hours — income is shrinking")

            # --- Vendor under pressure → raises prices for customers ---
            if role == "market_vendor" and debt_spike > 0.05:
                customers = [l for l in dependents if l.link_type == "supplies"]
                if customers:
                    price_hike = debt_spike * 0.15  # reduced from 0.3 — prices don't double overnight
                    targets = self.rng.sample(customers, min(6, len(customers)))
                    for link in targets:
                        if link.to_id not in world.agents:
                            continue
                        target = world.agents[link.to_id]
                        cost_increase = price_hike * link.strength * 0.3  # reduced from 0.5
                        target.debt_pressure = _clamp(target.debt_pressure + cost_increase)
                        affected_ids.add(link.to_id)

                        event = RippleEvent(
                            tick=tick,
                            actor_id=agent_id,
                            actor_name=agent.personality.name,
                            target_id=link.to_id,
                            target_name=target.personality.name,
                            action=f"raised prices to cover costs (dp={agent.debt_pressure:.2f})",
                            consequence=f"paying more for essentials, dp +{cost_increase:.3f}",
                            mechanism="supplies",
                            debt_delta=cost_increase,
                        )
                        resolution.events.append(event)
                        self.event_log.append(event)

                        target.add_memory(tick,
                            f"Prices went up at {agent.personality.name}'s stall — budget is tighter")

            # --- Worker under pressure → reduces spending at vendors ---
            if role in ("factory_worker", "dock_worker", "community", "retiree") and debt_spike > 0.06:
                # Find vendors this agent buys from
                dependencies = self.fabric.get_dependencies(agent_id)
                suppliers = [l for l in dependencies if l.link_type == "supplies"]
                # Reverse: this agent is the customer, so we find vendors who supply them
                vendor_links = [l for l in self.fabric.links if l.link_type == "supplies" and l.to_id == agent_id]
                for link in vendor_links[:3]:
                    if link.from_id not in world.agents:
                        continue
                    vendor = world.agents[link.from_id]
                    revenue_loss = debt_spike * 0.15 * link.strength
                    vendor.debt_pressure = _clamp(vendor.debt_pressure + revenue_loss)
                    affected_ids.add(link.from_id)

                    event = RippleEvent(
                        tick=tick,
                        actor_id=agent_id,
                        actor_name=agent.personality.name,
                        target_id=link.from_id,
                        target_name=vendor.personality.name,
                        action=f"cut spending due to financial stress (dp={agent.debt_pressure:.2f})",
                        consequence=f"lost a customer's spending, dp +{revenue_loss:.3f}",
                        mechanism="customer_reduces",
                        debt_delta=revenue_loss,
                    )
                    resolution.events.append(event)
                    self.event_log.append(event)

            # --- Government worker senses crisis → initiates response ---
            if role == "government_worker" and (debt_spike > 0.04 or dread_spike > 0.06):
                governed = [l for l in dependents if l.link_type == "governs"]
                if governed and self.rng.random() < 0.2:  # not every tick
                    # Pick distressed constituents to help
                    distressed = [
                        l for l in governed
                        if l.to_id in world.agents and world.agents[l.to_id].debt_pressure > 0.3
                    ]
                    helped = distressed[:min(5, len(distressed))]
                    for link in helped:
                        target = world.agents[link.to_id]
                        relief = 0.02
                        target.debt_pressure = _clamp(target.debt_pressure - relief)
                        target.heart.tension = max(0.0, target.heart.tension - 0.01)
                        affected_ids.add(link.to_id)

                        event = RippleEvent(
                            tick=tick,
                            actor_id=agent_id,
                            actor_name=agent.personality.name,
                            target_id=link.to_id,
                            target_name=target.personality.name,
                            action=f"initiated relief measure for constituent",
                            consequence=f"received small assistance, dp -{relief:.3f}",
                            mechanism="governs",
                            debt_delta=-relief,
                        )
                        resolution.events.append(event)
                        self.event_log.append(event)

            # --- Healthcare worker overwhelmed → reduced care quality ---
            if role == "healthcare" and dread_spike > 0.05:
                patients = [l for l in dependents if l.link_type == "treats"]
                if patients:
                    targets = self.rng.sample(patients, min(4, len(patients)))
                    for link in targets:
                        if link.to_id not in world.agents:
                            continue
                        target = world.agents[link.to_id]
                        anxiety = dread_spike * 0.2
                        target.dread_pressure = _clamp(target.dread_pressure + anxiety)
                        affected_ids.add(link.to_id)

                        event = RippleEvent(
                            tick=tick,
                            actor_id=agent_id,
                            actor_name=agent.personality.name,
                            target_id=link.to_id,
                            target_name=target.personality.name,
                            action=f"overwhelmed, care quality declining",
                            consequence=f"patient feels neglected, dread +{anxiety:.3f}",
                            mechanism="treats",
                            dread_delta=anxiety,
                        )
                        resolution.events.append(event)
                        self.event_log.append(event)

            # --- Teacher under stress → students affected ---
            if role == "teacher" and (debt_spike > 0.04 or dread_spike > 0.04):
                students = [l for l in dependents if l.link_type == "teaches"]
                if students:
                    targets = self.rng.sample(students, min(5, len(students)))
                    for link in targets:
                        if link.to_id not in world.agents:
                            continue
                        target = world.agents[link.to_id]
                        anxiety = max(debt_spike, dread_spike) * 0.15
                        target.heart.tension = _clamp(target.heart.tension + anxiety)
                        affected_ids.add(link.to_id)

                        event = RippleEvent(
                            tick=tick,
                            actor_id=agent_id,
                            actor_name=agent.personality.name,
                            target_id=link.to_id,
                            target_name=target.personality.name,
                            action=f"stressed and distracted in class",
                            consequence=f"student picks up on teacher's anxiety",
                            mechanism="teaches",
                            tension_delta=anxiety,
                        )
                        resolution.events.append(event)
                        self.event_log.append(event)

        resolution.total_affected = len(affected_ids)
        return resolution

    def get_recent_chains(self, n: int = 20) -> list[dict]:
        """Get the most recent ripple events for display."""
        recent = self.event_log[-n:]
        return [
            {
                "tick": e.tick,
                "actor": e.actor_name,
                "target": e.target_name,
                "action": e.action,
                "consequence": e.consequence,
                "mechanism": e.mechanism,
            }
            for e in recent
        ]

    def get_chain_summary(self) -> dict:
        """Summary of all ripple chains in the simulation."""
        by_mechanism: dict[str, int] = defaultdict(int)
        by_actor_role: dict[str, int] = defaultdict(int)
        for e in self.event_log:
            by_mechanism[e.mechanism] += 1
        return {
            "total_ripple_events": len(self.event_log),
            "by_mechanism": dict(by_mechanism),
        }

"""Economic action system — agent decisions that create cascading effects.

This is the missing piece: agents can now take ECONOMIC actions that directly
change other agents' state. When a manager cuts shifts, workers lose income.
When a vendor raises prices, customers feel cost pressure. When a consumer
reduces spending, vendors lose revenue.

These actions create the cascading feedback loops that sustain and amplify
macro shocks over weeks/months instead of decaying in days.

Architecture:
  - Economic actions are selected deterministically based on agent state,
    role, and economic pressure (same as emotional actions)
  - Each action has a TRANSMIT function that modifies other agents
  - The world tick loop calls resolve_economic_actions() after regular actions
  - LLM agents can also choose economic actions from their allowed list

Economic action cascade:
  Oil shock → dock worker debt_pressure rises
  → dock worker REDUCES_SPENDING → market vendor revenue drops
  → market vendor RAISES_PRICES → community debt_pressure rises
  → community REDUCES_SPENDING → more vendors feel pressure
  → manager sees falling revenue → CUT_SHIFTS → workers lose income
  → workers DEFAULT_ON_DEBT → creditors absorb losses
  → feedback loop sustains for weeks
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relationship import RelationshipStore
    from .world_agent import WorldAgent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Economic action definitions
# ---------------------------------------------------------------------------

@dataclass
class EconomicAction:
    """One economic action taken by one agent this tick."""
    actor_id: str
    action_type: str  # RAISE_PRICES, CUT_SHIFTS, REDUCE_SPENDING, etc.
    magnitude: float  # 0-1, how severe
    reason: str = ""


# Who can take which economic actions (by social_role)
ROLE_ECONOMIC_ACTIONS = {
    "market_vendor": ("RAISE_PRICES", "REDUCE_INVENTORY", "CLOSE_STALL"),
    "factory_worker": ("REDUCE_SPENDING", "DEFAULT_ON_DEBT", "HOARD_SUPPLIES"),
    "dock_worker": ("REDUCE_SPENDING", "DEFAULT_ON_DEBT", "HOARD_SUPPLIES"),
    "office_professional": ("REDUCE_SPENDING", "CUT_SHIFTS", "DEMAND_PAYMENT"),
    "office_worker": ("REDUCE_SPENDING", "DEFAULT_ON_DEBT"),
    "manager": ("CUT_SHIFTS", "RAISE_PRICES", "DEMAND_PAYMENT", "LAYOFF"),
    "bartender": ("RAISE_PRICES", "REDUCE_SPENDING"),
    "teacher": ("REDUCE_SPENDING",),
    "healthcare": ("REDUCE_SPENDING", "DEMAND_OVERTIME"),
    "government_worker": ("REDUCE_SPENDING", "IMPOSE_RESTRICTIONS"),
    "community": ("REDUCE_SPENDING", "HOARD_SUPPLIES", "DEFAULT_ON_DEBT"),
    "student": ("REDUCE_SPENDING", "DEFAULT_ON_DEBT"),
    "retiree": ("REDUCE_SPENDING", "HOARD_SUPPLIES"),
}

# Which locations are economic hubs (economic actions here affect more people)
ECONOMIC_HUB_LOCATIONS = {
    "main_market", "food_court", "artisan_alley",  # retail
    "factory_floor", "warehouse",  # industrial
    "office_tower", "trading_floor",  # white collar
    "docks", "fish_market",  # waterfront
}


def select_economic_action(agent: "WorldAgent", nearby_agents: dict[str, "WorldAgent"]) -> EconomicAction | None:
    """Deterministic economic action selection based on agent state.

    Returns None if no economic action is warranted (most ticks).
    Economic actions only fire when economic pressure exceeds thresholds.
    """
    role = agent.social_role
    allowed = ROLE_ECONOMIC_ACTIONS.get(role, ())
    if not allowed:
        return None

    dp = agent.debt_pressure
    dread = agent.dread_pressure
    tension = agent.heart.tension
    valence = agent.heart.valence
    energy = agent.heart.energy

    # Economic pressure composite
    econ_stress = dp * 0.5 + tension * 0.25 + (1.0 - valence) * 0.15 + dread * 0.1

    # Forward-looking pessimism amplifies economic actions
    pessimism = getattr(agent, "expectation_pessimism", 0.0)
    econ_stress += pessimism * 0.15

    # Economic actions only fire when pressure is ELEVATED ABOVE the agent's
    # personal baseline. Baseline is an EMA that adapts slowly, so scenario
    # pre-existing stress becomes "normal" and only shock-induced spikes trigger.
    baseline = getattr(agent, "_econ_baseline_pressure", None)
    if baseline is None:
        agent._econ_baseline_pressure = econ_stress
        return None

    # Update baseline with very slow EMA (adapts over ~500 ticks / 20 days)
    # This means a shock creates elevated pressure for weeks before it becomes "normal"
    agent._econ_baseline_pressure = baseline * 0.998 + econ_stress * 0.002

    pressure_above_baseline = econ_stress - agent._econ_baseline_pressure
    if pressure_above_baseline < 0.15:
        return None

    # Stochastic gating: not everyone acts every tick.
    # Probability of acting scales with how elevated pressure is.
    import random
    act_probability = min(0.15, pressure_above_baseline * 0.5)  # max 15% chance per tick
    if random.random() > act_probability:
        return None

    # --- Manager/authority roles: cut shifts or demand payment ---
    if role in ("manager", "office_professional") and "CUT_SHIFTS" in allowed:
        if pressure_above_baseline > 0.2 and dp > 0.25:
            magnitude = min(1.0, pressure_above_baseline * 1.5)
            return EconomicAction(
                actor_id=agent.agent_id,
                action_type="CUT_SHIFTS",
                magnitude=magnitude,
                reason=f"Economic pressure at {econ_stress:.2f}, cutting costs",
            )

    if role == "manager" and "LAYOFF" in allowed:
        if econ_stress > 0.65 and dp > 0.4:
            magnitude = min(1.0, (econ_stress - 0.55) * 2.0)
            return EconomicAction(
                actor_id=agent.agent_id,
                action_type="LAYOFF",
                magnitude=magnitude,
                reason=f"Severe economic pressure {econ_stress:.2f}, forced layoffs",
            )

    # --- Vendor roles: raise prices ---
    if role in ("market_vendor", "bartender", "manager") and "RAISE_PRICES" in allowed:
        if pressure_above_baseline > 0.15 and dp > 0.25:
            magnitude = min(1.0, dp * 0.8)
            return EconomicAction(
                actor_id=agent.agent_id,
                action_type="RAISE_PRICES",
                magnitude=magnitude,
                reason=f"Costs up, passing to customers (debt_pressure={dp:.2f})",
            )

    # --- Consumer roles: reduce spending ---
    if "REDUCE_SPENDING" in allowed:
        if pressure_above_baseline > 0.15:
            magnitude = min(1.0, econ_stress * 0.7)
            return EconomicAction(
                actor_id=agent.agent_id,
                action_type="REDUCE_SPENDING",
                magnitude=magnitude,
                reason=f"Tightening belt (econ_stress={econ_stress:.2f})",
            )

    # --- Desperate roles: hoard supplies ---
    if "HOARD_SUPPLIES" in allowed:
        if dread > 0.3 and pressure_above_baseline > 0.25:
            magnitude = min(1.0, dread * 0.6)
            return EconomicAction(
                actor_id=agent.agent_id,
                action_type="HOARD_SUPPLIES",
                magnitude=magnitude,
                reason=f"Panic buying (dread={dread:.2f})",
            )

    # --- Default on debt ---
    if "DEFAULT_ON_DEBT" in allowed:
        if dp > 0.6 and energy < 0.3 and pressure_above_baseline > 0.3:
            magnitude = min(1.0, dp * 0.5)
            return EconomicAction(
                actor_id=agent.agent_id,
                action_type="DEFAULT_ON_DEBT",
                magnitude=magnitude,
                reason=f"Cannot keep up with obligations (dp={dp:.2f})",
            )

    return None


# ---------------------------------------------------------------------------
# Economic action resolution — transmit effects to other agents
# ---------------------------------------------------------------------------

@dataclass
class EconomicResolution:
    """Summary of all economic actions resolved this tick."""
    actions_taken: int = 0
    agents_affected: int = 0
    price_increases: int = 0
    shift_cuts: int = 0
    spending_reductions: int = 0
    layoffs: int = 0
    defaults: int = 0
    hoarding: int = 0
    details: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "actions_taken": self.actions_taken,
            "agents_affected": self.agents_affected,
            "price_increases": self.price_increases,
            "shift_cuts": self.shift_cuts,
            "spending_reductions": self.spending_reductions,
            "layoffs": self.layoffs,
            "defaults": self.defaults,
            "hoarding": self.hoarding,
        }


def resolve_economic_actions(
    agents: dict[str, "WorldAgent"],
    by_location: dict[str, list[str]],
    relationships: "RelationshipStore",
    tick: int,
) -> EconomicResolution:
    """Select and resolve economic actions for all agents.

    Called once per tick in the world loop. Creates cascading effects.
    """
    resolution = EconomicResolution()
    actions: list[EconomicAction] = []

    # Select economic actions
    for agent_id, agent in agents.items():
        nearby = {
            aid: agents[aid]
            for aid in by_location.get(agent.location, [])
            if aid != agent_id
        }
        action = select_economic_action(agent, nearby)
        if action:
            actions.append(action)

    if not actions:
        return resolution

    # Resolve each action — transmit effects to other agents
    affected_set: set[str] = set()

    for action in actions:
        actor = agents[action.actor_id]
        actor_location = actor.location
        colocated = [
            aid for aid in by_location.get(actor_location, [])
            if aid != action.actor_id
        ]

        if action.action_type == "RAISE_PRICES":
            resolution.price_increases += 1
            # Affects consumers at the same location
            for cid in colocated:
                consumer = agents[cid]
                # Impact scales with magnitude and consumer's existing vulnerability
                impact = action.magnitude * 0.04 * (1.0 + consumer.debt_pressure * 0.5)
                consumer.debt_pressure = _clamp(consumer.debt_pressure + impact)
                consumer.heart.tension = _clamp(consumer.heart.tension + impact * 0.3)
                affected_set.add(cid)
                consumer.add_memory(tick, f"Prices went up at {actor_location} — costs are climbing again")

        elif action.action_type == "CUT_SHIFTS":
            resolution.shift_cuts += 1
            # Affects workers at same location (same industry)
            workers_here = [
                aid for aid in colocated
                if agents[aid].social_role in ("factory_worker", "dock_worker", "office_worker", "bartender")
            ]
            for wid in workers_here[:8]:
                worker = agents[wid]
                income_loss = action.magnitude * 0.06
                worker.debt_pressure = _clamp(worker.debt_pressure + income_loss)
                worker.heart.tension = _clamp(worker.heart.tension + income_loss * 0.5)
                worker.heart.valence = _clamp(worker.heart.valence - income_loss * 0.3)
                affected_set.add(wid)
                worker.add_memory(tick, f"Shifts are being cut at {actor_location} — income is thinning")

        elif action.action_type == "LAYOFF":
            resolution.layoffs += 1
            workers_here = [
                aid for aid in colocated
                if agents[aid].social_role in ("factory_worker", "dock_worker", "office_worker")
            ]
            # Layoff hits 1-3 workers hard
            targets = workers_here[:min(3, max(1, int(action.magnitude * 4)))]
            for wid in targets:
                worker = agents[wid]
                worker.debt_pressure = _clamp(worker.debt_pressure + 0.3)
                worker.dread_pressure = _clamp(worker.dread_pressure + 0.15)
                worker.heart.tension = _clamp(worker.heart.tension + 0.2)
                worker.heart.valence = _clamp(worker.heart.valence - 0.15)
                worker.heart.wounds.append((0.08, 0.994))  # lasting emotional wound
                affected_set.add(wid)
                worker.add_memory(tick, f"Lost my position — income gone, scrambling")
            # Remaining workers feel survivor anxiety
            for wid in workers_here[len(targets):]:
                worker = agents[wid]
                worker.dread_pressure = _clamp(worker.dread_pressure + 0.06)
                worker.heart.tension = _clamp(worker.heart.tension + 0.05)
                affected_set.add(wid)
                worker.add_memory(tick, f"Others got laid off — am I next?")

        elif action.action_type == "REDUCE_SPENDING":
            resolution.spending_reductions += 1
            # Affects vendors/service workers at economic hub locations
            vendors_affected = [
                aid for aid in agents
                if agents[aid].social_role in ("market_vendor", "bartender")
                and agents[aid].location in ECONOMIC_HUB_LOCATIONS
            ]
            # Each consumer reduction is small but they compound
            impact_per_vendor = action.magnitude * 0.008
            for vid in vendors_affected[:6]:
                vendor = agents[vid]
                vendor.debt_pressure = _clamp(vendor.debt_pressure + impact_per_vendor)
                affected_set.add(vid)

        elif action.action_type == "HOARD_SUPPLIES":
            resolution.hoarding += 1
            # Creates scarcity pressure for everyone at the location
            for cid in colocated:
                other = agents[cid]
                scarcity = action.magnitude * 0.03
                other.debt_pressure = _clamp(other.debt_pressure + scarcity)
                other.dread_pressure = _clamp(other.dread_pressure + scarcity * 0.5)
                affected_set.add(cid)

        elif action.action_type == "DEFAULT_ON_DEBT":
            resolution.defaults += 1
            # Damages creditors (agents with positive debt relationship)
            for other_id, rel in relationships.get_agent_relationships(action.actor_id)[:5]:
                debt_owed = relationships.get_debt(action.actor_id, other_id)
                if debt_owed > 0.1 and other_id in agents:
                    creditor = agents[other_id]
                    loss = debt_owed * action.magnitude * 0.15
                    creditor.debt_pressure = _clamp(creditor.debt_pressure + loss)
                    relationships.adjust_debt(action.actor_id, other_id, -debt_owed * 0.5)
                    relationships.set_grievance(
                        other_id, action.actor_id,
                        relationships.get_grievance(other_id, action.actor_id) + 0.1,
                    )
                    rel.trust = max(-1.0, rel.trust - 0.08)
                    affected_set.add(other_id)
                    creditor.add_memory(tick, f"Someone defaulted on what they owed me — I absorb the loss")

        resolution.actions_taken += 1

    resolution.agents_affected = len(affected_set)
    return resolution


# ---------------------------------------------------------------------------
# Forward-looking expectations — agents form beliefs about the future
# ---------------------------------------------------------------------------

def update_expectations(agents: dict[str, "WorldAgent"], tick: int):
    """Update each agent's forward-looking expectation based on recent trajectory.

    Agents form beliefs about whether things will get BETTER or WORSE.
    This is what the real CCI measures — not current state, but expectations.

    expectation_pessimism: 0 = expects things to improve, 1 = expects disaster
    """
    for agent in agents.values():
        # Look at recent trajectory of debt_pressure and tension
        # If they're rising, expect worse. If falling, expect better.
        current_pressure = agent.debt_pressure + agent.dread_pressure + agent.heart.tension
        prev_pressure = getattr(agent, "_prev_total_pressure", current_pressure)

        # Pressure trend: positive = getting worse
        trend = current_pressure - prev_pressure
        agent._prev_total_pressure = current_pressure

        # Current pessimism
        current_pessimism = getattr(agent, "expectation_pessimism", 0.0)

        # Adjust pessimism based on trend + current state
        # Momentum: pessimism doesn't snap — it drifts
        if trend > 0.01:
            # Things getting worse → pessimism rises
            delta = trend * 0.8 + current_pressure * 0.1
            new_pessimism = current_pessimism * 0.85 + (current_pessimism + delta) * 0.15
        elif trend < -0.01:
            # Things improving → pessimism falls, but slowly (negativity bias)
            delta = trend * 0.4  # recovery is slower than decline (asymmetric)
            new_pessimism = current_pessimism * 0.92 + (current_pessimism + delta) * 0.08
        else:
            # Flat — slight decay toward baseline
            new_pessimism = current_pessimism * 0.995

        # Threat lens amplifies pessimism for aligned threats
        profile = agent.get_human_profile()
        if profile.get("threat_lens") == "scarcity" and agent.debt_pressure > 0.2:
            new_pessimism += 0.01  # scarcity-lens people stay pessimistic about money longer
        if profile.get("self_story") == "survivor":
            new_pessimism *= 0.97  # survivors recover expectations faster

        agent.expectation_pessimism = _clamp(new_pessimism, 0.0, 1.0)

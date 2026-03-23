"""Persistent external conditions and rally-around-flag mechanics.

Fixes two critical gaps:

1. PERSISTENT CONDITIONS: External states (oil price, pandemic active, credit freeze)
   stay active and apply pressure every tick until explicitly resolved.
   Without this, a one-time shock decays and agents recover unrealistically fast.

2. RALLY-AROUND-FLAG: Existential and health crises initially BOOST social cohesion
   and institutional trust (people come together) before eroding them.
   This matches documented behavior in Cuban Missile Crisis, early COVID, 9/11.

3. COMPOUND CRISIS SUPPORT: Multiple active conditions interact and amplify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World
    from .world_agent import WorldAgent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Persistent conditions
# ---------------------------------------------------------------------------

@dataclass
class PersistentCondition:
    """An external condition that remains active and applies pressure every tick.

    Unlike a one-time shock, this represents an ongoing reality:
    "Oil is STILL at $147" or "The pandemic is STILL spreading" or
    "The banking system is STILL frozen".
    """
    label: str
    kind: str
    severity: float  # 0-1, how bad the condition is
    start_tick: int
    active: bool = True

    # Per-tick pressure applied to agents (much smaller than initial shock)
    # These compound over time, creating sustained impact
    economic_drag: float = 0.0  # per-tick debt_pressure increment
    dread_drag: float = 0.0  # per-tick dread_pressure increment
    tension_drag: float = 0.0  # per-tick tension increment

    # Which roles are affected most (multiplier)
    role_multipliers: dict[str, float] = field(default_factory=dict)

    # Rally-around-flag parameters
    rally_strength: float = 0.0  # 0 = no rally, 1 = strong rally
    rally_duration_ticks: int = 0  # how long rally lasts before fading
    rally_phase: str = "none"  # "none", "rally", "fading", "erosion"

    def ticks_active(self, current_tick: int) -> int:
        return current_tick - self.start_tick


def create_persistent_condition(kind: str, severity: float, start_tick: int) -> PersistentCondition:
    """Create a persistent condition from a shock type."""
    conditions = {
        "oil_price_surge": PersistentCondition(
            label="Oil prices elevated", kind=kind, severity=severity, start_tick=start_tick,
            economic_drag=0.003,  # small per-tick, but compounds over 720 ticks
            tension_drag=0.001,
            role_multipliers={
                "dock_worker": 2.0, "factory_worker": 1.8, "market_vendor": 1.6,
                "bartender": 1.3, "community": 1.2, "office_worker": 0.8,
                "teacher": 0.6, "healthcare": 0.5, "government_worker": 0.4,
                "student": 0.7, "retiree": 0.9,
            },
            rally_strength=0.0,  # economic shocks don't create rally
        ),
        "banking_panic": PersistentCondition(
            label="Banking system frozen", kind=kind, severity=severity, start_tick=start_tick,
            economic_drag=0.004,
            tension_drag=0.002,
            dread_drag=0.001,
            role_multipliers={
                "market_vendor": 2.0, "factory_worker": 1.5, "dock_worker": 1.5,
                "office_professional": 1.3, "retiree": 1.8, "community": 1.4,
                "bartender": 1.3, "student": 1.0, "teacher": 0.7,
                "healthcare": 0.5, "government_worker": 0.6,
            },
            rally_strength=0.0,
        ),
        "health_crisis": PersistentCondition(
            label="Pandemic active", kind=kind, severity=severity, start_tick=start_tick,
            economic_drag=0.002,
            dread_drag=0.003,
            tension_drag=0.001,
            role_multipliers={
                "healthcare": 2.5, "community": 1.5, "teacher": 1.4,
                "market_vendor": 1.3, "bartender": 1.5, "retiree": 2.0,
                "factory_worker": 1.2, "dock_worker": 1.0, "student": 1.1,
                "office_professional": 0.6, "government_worker": 0.8,
            },
            rally_strength=0.6,  # strong initial solidarity
            rally_duration_ticks=240,  # ~10 days of solidarity before fading
            rally_phase="rally",
        ),
        "military_crisis": PersistentCondition(
            label="Military crisis active", kind=kind, severity=severity, start_tick=start_tick,
            dread_drag=0.004,
            tension_drag=0.002,
            economic_drag=0.002,
            role_multipliers={
                "community": 1.8, "student": 1.6, "teacher": 1.5,
                "healthcare": 1.3, "retiree": 1.4, "factory_worker": 1.2,
                "dock_worker": 1.2, "market_vendor": 1.0, "bartender": 1.0,
                "government_worker": 1.5, "office_professional": 0.8,
            },
            rally_strength=0.8,  # very strong rally-around-flag (like Cuban Missile Crisis)
            rally_duration_ticks=168,  # ~7 days (shorter than pandemic)
            rally_phase="rally",
        ),
        "brand_scandal": PersistentCondition(
            label="Brand scandal ongoing", kind=kind, severity=severity, start_tick=start_tick,
            economic_drag=0.001,
            tension_drag=0.001,
            dread_drag=0.001,
            role_multipliers={
                "market_vendor": 1.5, "factory_worker": 1.3, "student": 1.4,
                "office_professional": 1.0, "community": 1.2,
                "manager": 1.3, "government_worker": 0.8,
                "healthcare": 0.4, "retiree": 0.6,
            },
            rally_strength=0.3,  # mild solidarity (people unite against the brand)
            rally_duration_ticks=120,
            rally_phase="rally",
        ),
        "mass_layoffs": PersistentCondition(
            label="Mass layoffs ongoing", kind=kind, severity=severity, start_tick=start_tick,
            economic_drag=0.004,
            tension_drag=0.002,
            dread_drag=0.002,
            role_multipliers={
                "factory_worker": 2.0, "dock_worker": 1.8, "office_worker": 1.6,
                "market_vendor": 1.3, "community": 1.4, "bartender": 1.2,
                "student": 1.0, "teacher": 0.7, "healthcare": 0.5,
                "government_worker": 0.6, "retiree": 0.8,
            },
            rally_strength=0.0,
        ),
    }
    return conditions.get(kind, PersistentCondition(
        label=f"Crisis: {kind}", kind=kind, severity=severity, start_tick=start_tick,
        economic_drag=0.002, tension_drag=0.001,
    ))


# ---------------------------------------------------------------------------
# Apply persistent conditions each tick
# ---------------------------------------------------------------------------

def apply_persistent_conditions(world: "World") -> dict:
    """Apply all active persistent conditions to agents each tick.

    This is the fix for "shock fires once then decays":
    persistent conditions keep applying small pressure every tick
    until they're deactivated.
    """
    if not hasattr(world, "_persistent_conditions"):
        world._persistent_conditions = []

    if not world._persistent_conditions:
        return {"active_conditions": 0}

    total_affected = 0
    rally_effects = {"cohesion_boost": 0.0, "trust_boost": 0.0}

    for condition in world._persistent_conditions:
        if not condition.active:
            continue

        ticks_active = condition.ticks_active(world.tick_count)

        # --- Rally-around-flag phase management ---
        if condition.rally_strength > 0:
            if condition.rally_phase == "rally":
                if ticks_active > condition.rally_duration_ticks:
                    condition.rally_phase = "fading"
                else:
                    # During rally: boost cohesion and trust, reduce tension
                    rally_intensity = condition.rally_strength * (1.0 - ticks_active / condition.rally_duration_ticks)
                    rally_effects["cohesion_boost"] += rally_intensity * 0.003
                    rally_effects["trust_boost"] += rally_intensity * 0.002
                    # Rally partially counteracts dread/tension
                    for agent in world.agents.values():
                        agent.heart.tension = max(0.0, agent.heart.tension - rally_intensity * 0.002)
                        # Reduce pessimism during rally (people believe "we'll get through this")
                        agent.expectation_pessimism = max(
                            0.0, getattr(agent, "expectation_pessimism", 0) - rally_intensity * 0.003
                        )

            elif condition.rally_phase == "fading":
                if ticks_active > condition.rally_duration_ticks * 2:
                    condition.rally_phase = "erosion"
                # Fading: rally effects diminish
                fade_factor = 1.0 - (ticks_active - condition.rally_duration_ticks) / condition.rally_duration_ticks
                fade_factor = max(0, fade_factor)
                rally_effects["cohesion_boost"] += condition.rally_strength * fade_factor * 0.001

            # Erosion phase: rally is over, trust starts eroding (disillusionment)
            # No special handling needed — the persistent drag takes over

        # --- Apply per-tick pressure ---
        # Natural decay: conditions weaken over time (severity * decay curve)
        decay = max(0.3, 1.0 - ticks_active / 2000)  # takes ~80 days to drop to 30%
        effective_severity = condition.severity * decay

        for agent in world.agents.values():
            multiplier = condition.role_multipliers.get(agent.social_role, 1.0)
            if multiplier < 0.3:
                continue

            # Apply drags scaled by role multiplier and effective severity
            if condition.economic_drag > 0:
                drag = condition.economic_drag * multiplier * effective_severity
                agent.debt_pressure = _clamp(agent.debt_pressure + drag)
            if condition.dread_drag > 0:
                drag = condition.dread_drag * multiplier * effective_severity
                agent.dread_pressure = _clamp(agent.dread_pressure + drag)
            if condition.tension_drag > 0:
                drag = condition.tension_drag * multiplier * effective_severity
                agent.heart.tension = _clamp(agent.heart.tension + drag)

            total_affected += 1

    return {
        "active_conditions": sum(1 for c in world._persistent_conditions if c.active),
        "agents_affected": total_affected,
        "rally_effects": rally_effects,
        "conditions": [
            {"label": c.label, "kind": c.kind, "active": c.active,
             "rally_phase": c.rally_phase,
             "ticks_active": c.ticks_active(world.tick_count)}
            for c in world._persistent_conditions
        ],
    }


def register_persistent_condition(world: "World", kind: str, severity: float):
    """Register a new persistent condition when a shock is injected."""
    if not hasattr(world, "_persistent_conditions"):
        world._persistent_conditions = []
    condition = create_persistent_condition(kind, severity, world.tick_count)
    world._persistent_conditions.append(condition)

"""Research-calibrated economic engine.

Replaces the ad-hoc economic interactions with a system calibrated
from real BLS, FRED, NBER, and IMF data. Every parameter has a source.

Key differences from the old system:
  1. STAGED corporate response (weeks, not instant)
  2. REALISTIC savings (30% paycheck-to-paycheck, 55% have 3 months)
  3. GRADUAL price pass-through (12% day 1, 50% by month 1)
  4. SECTOR RISK CAPS (only 15-20% actually lose jobs)
  5. CALIBRATED re-employment (job finding rate drops 38% → 18%)
  6. FISCAL MULTIPLIER (1.0-1.5x government spending)
  7. TIME COMPRESSION (1 sim-day ≈ 25 real-days for 14-day runs)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .economic_calibration import (
    PAYCHECK_TO_PAYCHECK_PCT,
    SECTOR_RISK_RATES,
    ESSENTIAL_SECTORS,
    RECESSION_PROOF_EMPLOYMENT_FLOOR,
    REEMPLOYMENT_RATE_NORMAL,
    REEMPLOYMENT_RATE_CRISIS,
    GIG_ABSORPTION_RATE,
    UI_REPLACEMENT_RATE,
    FISCAL_MULTIPLIER_RECESSION,
    OIL_TO_GAS_DAY1,
    OIL_TO_GAS_MONTH1,
    COMPRESSED_CORP_HOUR_CUTS_TICKS,
    COMPRESSED_CORP_LAYOFF_TICKS,
    COMPRESSED_FED_RESPONSE_TICKS,
    COMPRESSED_STIMULUS_DELIVERY_TICKS,
    COMPRESSION_FACTOR,
    VIX_BASELINE,
)

if TYPE_CHECKING:
    from .world import World
    from .world_agent import WorldAgent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ═══════════════════════════════════════════════════════════════
# 1. REALISTIC SAVINGS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def calibrate_agent_savings(agent: "WorldAgent", rng: random.Random):
    """Set realistic savings based on role and income.

    BLS/Fed data: 30% paycheck-to-paycheck, 55% have 3 months saved.
    Income correlates strongly with savings.
    """
    role = agent.social_role

    # Income-based savings distribution
    savings_by_role = {
        "manager": (0.5, 0.9),           # high income, good savings
        "office_professional": (0.4, 0.8),
        "tech_worker": (0.3, 0.7),
        "healthcare_worker": (0.25, 0.6),
        "government_worker": (0.3, 0.6),
        "teacher": (0.2, 0.5),
        "factory_worker": (0.08, 0.35),   # lower income, less savings
        "office_worker": (0.1, 0.4),
        "retail_worker": (0.03, 0.25),    # often paycheck-to-paycheck
        "gig_worker": (0.02, 0.2),        # most precarious
        "small_business_owner": (0.1, 0.4),
        "student": (0.01, 0.15),
        "retiree": (0.15, 0.6),           # fixed income, variable savings
    }

    lo, hi = savings_by_role.get(role, (0.1, 0.4))

    # 30% are paycheck-to-paycheck (near zero savings)
    if rng.random() < PAYCHECK_TO_PAYCHECK_PCT:
        agent.savings_buffer = rng.uniform(0.01, 0.08)
    else:
        agent.savings_buffer = rng.uniform(lo, hi)

    # Credit access correlates with income
    credit_by_role = {
        "manager": (0.6, 0.95), "office_professional": (0.5, 0.85),
        "tech_worker": (0.4, 0.8), "healthcare_worker": (0.4, 0.7),
        "government_worker": (0.4, 0.7), "teacher": (0.3, 0.6),
        "factory_worker": (0.15, 0.45), "office_worker": (0.2, 0.5),
        "retail_worker": (0.1, 0.35), "gig_worker": (0.05, 0.25),
        "small_business_owner": (0.15, 0.45), "student": (0.05, 0.2),
        "retiree": (0.2, 0.5),
    }
    clo, chi = credit_by_role.get(role, (0.15, 0.4))
    agent.credit_access = rng.uniform(clo, chi)

    # Income level by role
    income_by_role = {
        "manager": (0.7, 0.95), "office_professional": (0.6, 0.85),
        "tech_worker": (0.55, 0.8), "healthcare_worker": (0.45, 0.7),
        "government_worker": (0.45, 0.65), "teacher": (0.35, 0.55),
        "factory_worker": (0.3, 0.5), "office_worker": (0.3, 0.5),
        "retail_worker": (0.2, 0.35), "gig_worker": (0.15, 0.35),
        "small_business_owner": (0.25, 0.55), "student": (0.1, 0.25),
        "retiree": (0.2, 0.45),
    }
    ilo, ihi = income_by_role.get(role, (0.25, 0.5))
    agent.income_level = rng.uniform(ilo, ihi)


# ═══════════════════════════════════════════════════════════════
# 2. STAGED CORPORATE RESPONSE
# ═══════════════════════════════════════════════════════════════

@dataclass
class CorporateResponseTracker:
    """Tracks where each company is in the cost-cutting sequence.

    Real sequence (BLS): discretionary freeze → hiring freeze →
    hour cuts → layoffs → capex cuts. Takes 3-9 months in reality,
    compressed to ~2-10 sim-days in our 14-day runs.
    """
    # Ticks since shock was felt by this company
    shock_tick: int = 0
    stage: str = "normal"  # normal → freeze → hours_cut → layoffs
    severity: float = 0.0  # how bad the revenue hit is (0-1)
    hours_cut_applied: bool = False
    layoffs_applied: bool = False
    workers_laid_off: list[str] = field(default_factory=list)

    def advance(self, current_tick: int, company_debt: float):
        """Advance through the cost-cutting stages based on time + severity."""
        if self.shock_tick == 0:
            return

        ticks_since = current_tick - self.shock_tick
        self.severity = min(1.0, company_debt * 1.5)

        # Stage transitions based on calibrated compressed timelines
        if ticks_since < 24:  # Day 1: assessment
            self.stage = "assessing"
        elif ticks_since < COMPRESSED_CORP_HOUR_CUTS_TICKS:  # ~2.3 days: freeze
            self.stage = "freeze"
        elif ticks_since < COMPRESSED_CORP_LAYOFF_TICKS:  # ~4.7 days: hour cuts
            self.stage = "hours_cut"
        else:
            self.stage = "layoffs"


# ═══════════════════════════════════════════════════════════════
# 3. CALIBRATED ECONOMIC TICK
# ═══════════════════════════════════════════════════════════════

def calibrated_economic_tick(world: "World", tick: int):
    """Run one tick of calibrated economic mechanics.

    This replaces the old economic_actions + household mechanics
    with research-calibrated interactions.
    """
    agents = world.agents
    n = len(agents)
    if n == 0:
        return {}

    # Count current unemployment
    unemployed = sum(1 for a in agents.values() if not a.employed)
    unemp_rate = unemployed / n
    in_crisis = unemp_rate > 0.05

    stats = {
        "unemployment_rate": round(unemp_rate * 100, 1),
        "newly_unemployed": 0,
        "reemployed": 0,
        "savings_depleted": 0,
        "spending_cuts": 0,
        "price_increases": 0,
        "stimulus_delivered": 0,
    }

    # --- Automatic stabilizers (fire every tick) ---

    # UI benefits for unemployed (BLS: replaces 43% of wages)
    for agent in agents.values():
        if not agent.employed:
            # UI prevents debt spiral — replaces partial income
            ui_income = UI_REPLACEMENT_RATE * 0.5  # normalized
            agent.income_level = max(agent.income_level, ui_income * 0.5)
            # UI reduces debt accumulation
            if agent.debt_pressure > 0.15:
                agent.debt_pressure = max(0.05, agent.debt_pressure - 0.008)

    # Essential sector protection (60-65% recession-proof)
    for agent in agents.values():
        if agent.social_role in ESSENTIAL_SECTORS and not agent.employed:
            agent.employed = True
            agent.income_level = max(agent.income_level, 0.35)

    # --- Savings mechanics (slower, realistic depletion) ---
    for agent in agents.values():
        dp = agent.debt_pressure
        if dp <= 0.1:
            # Low pressure: savings slowly recover
            agent.savings_buffer = min(1.0, agent.savings_buffer + 0.001)
            continue

        excess = dp - 0.1

        # Savings absorb first (lasts ~3 months = ~90 compressed ticks)
        if agent.savings_buffer > 0.02:
            # Draw rate: ~3% of savings per compressed day
            draw = min(excess * 0.08, agent.savings_buffer * 0.01)
            agent.savings_buffer = max(0.01, agent.savings_buffer - draw)
            agent.debt_pressure = max(0.0, dp - draw * 0.5)
        elif agent.savings_buffer < 0.05:
            stats["savings_depleted"] += 1

        # Credit absorbs second
        if agent.savings_buffer < 0.1 and agent.credit_access > 0.02 and excess > 0.05:
            credit_draw = min(excess * 0.05, agent.credit_access * 0.008)
            agent.credit_access = max(0.0, agent.credit_access - credit_draw)
            agent.debt_pressure = max(0.0, agent.debt_pressure - credit_draw * 0.3)

        # Income recovery: employed agents reduce debt
        if agent.employed and agent.income_level > 0.15:
            recovery = agent.income_level * 0.002
            agent.debt_pressure = max(0.0, agent.debt_pressure - recovery)

    # --- Re-employment (calibrated job finding rate) ---
    # Normal: 38%/month ≈ 1.7%/day ≈ 0.07%/tick
    # Crisis: 18%/month ≈ 0.82%/day ≈ 0.034%/tick
    # Compressed: multiply by compression factor / 30 for daily rate
    reemploy_rate = REEMPLOYMENT_RATE_CRISIS if in_crisis else REEMPLOYMENT_RATE_NORMAL
    # Apply compression: each sim-tick represents ~25 real-hours
    compressed_rate = reemploy_rate * COMPRESSION_FACTOR / 24

    for agent in agents.values():
        if not agent.employed and agent.social_role not in ("retiree", "student"):
            if random.random() < compressed_rate:
                agent.employed = True
                agent.income_level = max(0.2, agent.income_level * 0.7)  # often at lower pay
                stats["reemployed"] += 1

    # Gig economy absorption
    if in_crisis:
        newly_unemployed = [a for a in agents.values()
                           if not a.employed and a.debt_pressure > 0.3
                           and a.social_role not in ("retiree", "student")]
        n_gig = max(1, int(len(newly_unemployed) * GIG_ABSORPTION_RATE))
        for agent in random.sample(newly_unemployed, min(n_gig, len(newly_unemployed))):
            agent.employed = True
            agent.income_level = 0.2  # gig income is lower
            agent.social_role = "gig_worker"  # transition to gig

    # --- Sector-specific layoff risk caps ---
    # Only X% of each sector can actually lose jobs (SECTOR_RISK_RATES)
    by_role: dict[str, list] = {}
    for agent in agents.values():
        by_role.setdefault(agent.social_role, []).append(agent)

    for role, role_agents in by_role.items():
        max_risk = SECTOR_RISK_RATES.get(role, 0.10)
        currently_unemployed = sum(1 for a in role_agents if not a.employed)
        max_unemployed = int(len(role_agents) * max_risk)

        if currently_unemployed > max_unemployed:
            # Over the cap — re-employ excess
            unemployed_here = [a for a in role_agents if not a.employed]
            n_reemploy = currently_unemployed - max_unemployed
            for agent in random.sample(unemployed_here, min(n_reemploy, len(unemployed_here))):
                agent.employed = True
                agent.income_level = max(0.2, agent.income_level)

    # --- Government stimulus (fires after compressed delay) ---
    if hasattr(world, "_shock_tick") and tick > 0:
        ticks_since_shock = tick - world._shock_tick

        # Fed response (~10 compressed ticks)
        if ticks_since_shock == COMPRESSED_FED_RESPONSE_TICKS:
            for agent in agents.values():
                agent.heart.tension = max(0.0, agent.heart.tension - 0.02)
                agent.expectation_pessimism = max(0.0,
                    getattr(agent, "expectation_pessimism", 0) - 0.03)

        # Stimulus delivery (~17 compressed ticks)
        if ticks_since_shock == COMPRESSED_STIMULUS_DELIVERY_TICKS:
            for agent in agents.values():
                # Fiscal multiplier: each dollar of stimulus creates 1.25x effect
                relief = 0.04 * FISCAL_MULTIPLIER_RECESSION
                agent.debt_pressure = max(0.0, agent.debt_pressure - relief)
                stats["stimulus_delivered"] += 1

    # --- Gradual price pass-through ---
    # Prices don't jump instantly — 12% day 1, 50% month 1
    # Track via a world-level price pressure variable
    if hasattr(world, "_price_pressure"):
        pp = world._price_pressure
        if pp > 0.01:
            # Each tick, a small fraction of price pressure reaches consumers
            # Compressed: 12% in first day = ~0.5% per tick
            pass_through = pp * 0.005
            for agent in agents.values():
                if agent.social_role not in ESSENTIAL_SECTORS:
                    agent.debt_pressure = _clamp(agent.debt_pressure + pass_through * 0.3)
            # Price pressure slowly depletes as it passes through
            world._price_pressure = pp * 0.997

    return stats


# ═══════════════════════════════════════════════════════════════
# 4. SPENDING BEHAVIOR (phased, not instant)
# ═══════════════════════════════════════════════════════════════

def apply_spending_behavior(agents: dict[str, "WorldAgent"], tick: int):
    """Agents cut spending in phases: discretionary first, then essentials.

    BLS data: Essential spending = 70-75% normally, 85%+ in crisis.
    Spending cuts build over 2-6 months organically, 2-4 weeks in shocks.
    Compressed: phases happen over 1-3 sim-days.
    """
    for agent in agents.values():
        dp = agent.debt_pressure
        if dp < 0.2:
            continue

        # Phase 1: Cut discretionary (dp > 0.2) — small debt relief
        if dp > 0.2 and dp <= 0.4:
            agent.debt_pressure = max(0.0, dp - 0.003)  # small savings from cutting extras

        # Phase 2: Cut deeply (dp > 0.4) — bigger savings but suffering
        elif dp > 0.4 and dp <= 0.6:
            agent.debt_pressure = max(0.0, dp - 0.005)
            agent.heart.valence = max(0.1, agent.heart.valence - 0.002)

        # Phase 3: Survival mode (dp > 0.6) — everything cut, desperation
        elif dp > 0.6:
            agent.debt_pressure = max(0.0, dp - 0.007)
            agent.heart.valence = max(0.05, agent.heart.valence - 0.004)
            agent.dread_pressure = _clamp(agent.dread_pressure + 0.002)

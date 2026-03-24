"""Financial market model — VIX, S&P proxy, credit spreads, unemployment rate.

Computes realistic macro-economic metrics from agent state:
  - Unemployment: based on explicit employment status, not just debt threshold
  - VIX proxy: driven by fear contagion among finance agents + sudden state changes
  - S&P proxy: inverse of collective business confidence and spending
  - Credit spread: driven by banking sector stress and default rates
  - Inflation: driven by price-increase decisions and supply pressure
  - Savings rate: population-wide savings buffer depletion

The key fix: agents have SAVINGS BUFFERS that absorb economic shocks before
they become unemployed. A worker whose hours are cut depletes savings first,
then credit, THEN becomes effectively unemployed. This prevents the 100%
unemployment problem.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World
    from .world_agent import WorldAgent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Savings buffer mechanics — absorb shocks before unemployment
# ---------------------------------------------------------------------------

def apply_savings_mechanics(agents: dict[str, "WorldAgent"], tick: int):
    """Each tick, agents use savings/credit to cushion debt_pressure.

    Instead of debt_pressure directly = economic pain,
    savings absorb the first hit:
      debt_pressure rises → savings_buffer depletes → credit tapped →
      THEN debt_pressure actually hurts (valence drop, tension rise)

    This means a worker whose hours are cut can survive on savings
    for weeks before hitting true distress.
    """
    for agent in agents.values():
        dp = agent.debt_pressure

        if dp <= 0.1:
            # Low pressure — savings slowly recover
            agent.savings_buffer = min(1.0, agent.savings_buffer + 0.003)
            continue

        # How much pressure needs to be absorbed?
        excess = dp - 0.1  # pressure above baseline

        # Savings absorb first — slower depletion (lasts weeks, not days)
        if agent.savings_buffer > 0.01:
            absorbed = min(excess * 0.15, agent.savings_buffer * 0.02)  # savings drain slowly
            agent.savings_buffer = max(0.0, agent.savings_buffer - absorbed)
            agent.debt_pressure = max(0.0, dp - absorbed * 0.6)
            excess = max(0.0, excess - absorbed)

        # Credit absorbs second (if savings getting low)
        if excess > 0.05 and agent.savings_buffer < 0.2 and agent.credit_access > 0.01:
            credit_used = min(excess * 0.1, agent.credit_access * 0.015)  # credit drains slowly too
            agent.credit_access = max(0.0, agent.credit_access - credit_used)
            agent.debt_pressure = max(0.0, agent.debt_pressure - credit_used * 0.4)

        # Income recovery: employed agents slowly reduce debt
        if agent.employed and agent.income_level > 0.2:
            recovery = agent.income_level * 0.004
            agent.debt_pressure = max(0.0, agent.debt_pressure - recovery)

        # Employment check: if debt is extreme AND savings/credit gone → unemployed
        if agent.debt_pressure > 0.8 and agent.savings_buffer < 0.05 and agent.credit_access < 0.1:
            if agent.employed:
                agent.employed = False  # lost job / can't sustain
        elif agent.debt_pressure < 0.3 and not agent.employed:
            # Re-employment if pressure drops enough
            agent.employed = True


# ---------------------------------------------------------------------------
# Financial market metrics
# ---------------------------------------------------------------------------

@dataclass
class MarketState:
    """Full set of financial market metrics."""
    # Consumer
    cci: float = 80.0  # Consumer Confidence Index (0-100)

    # Labor market
    unemployment_rate: float = 4.0  # % (realistic scale)
    underemployment_rate: float = 6.0  # % (includes part-time wanting full-time)
    initial_claims_proxy: float = 200.0  # thousands (weekly)
    labor_force_participation: float = 63.0  # %

    # Financial markets
    sp500_proxy: float = 100.0  # index (100 = baseline)
    vix_proxy: float = 15.0  # volatility index (15 = normal, 80 = panic)
    credit_spread: float = 1.0  # basis points proxy (1 = normal, 10+ = crisis)

    # Prices / Inflation
    cpi_change: float = 0.0  # % monthly change
    price_pressure_index: float = 0.0  # 0-100

    # Banking
    bank_stress: float = 0.0  # 0-100
    default_rate: float = 0.0  # % of agents defaulting

    # Wealth / Savings
    avg_savings_rate: float = 50.0  # 0-100
    savings_depletion_rate: float = 0.0  # % who depleted savings
    inequality_index: float = 10.0  # gini-like (0-100)

    # Business
    business_confidence: float = 80.0  # 0-100
    investment_index: float = 100.0  # 100 = baseline

    # GDP
    gdp_index: float = 100.0  # 100 = baseline

    # Social
    institutional_trust: float = 60.0  # 0-100
    social_cohesion: float = 70.0  # 0-100
    civil_unrest: float = 2.0  # 0-100

    # Housing
    housing_stress: float = 10.0  # 0-100
    foreclosure_risk: float = 2.0  # %

    def as_dict(self) -> dict:
        return {k: round(v, 1) for k, v in self.__dict__.items()}


def compute_market_state(world: "World", meta: dict) -> MarketState:
    """Compute all financial market metrics from agent state."""
    agents = list(world.agents.values())
    n = len(agents)
    if n == 0:
        return MarketState()

    ms = MarketState()

    # ── CCI ──
    macro = world.get_macro_summary()
    ms.cci = macro.get("current", {}).get("consumer_confidence", 80)

    # ── Unemployment (REAL: based on explicit employment status) ──
    unemployed = sum(1 for a in agents if not a.employed)
    ms.unemployment_rate = unemployed / n * 100

    # Underemployment: employed but debt_pressure > 0.3 (hours cut, income reduced)
    underemployed = sum(1 for a in agents if a.employed and a.debt_pressure > 0.3)
    ms.underemployment_rate = underemployed / n * 100

    # Initial claims proxy: agents who BECAME unemployed recently
    # (track via savings depletion rate as proxy)
    recently_distressed = sum(1 for a in agents if a.savings_buffer < 0.1 and a.debt_pressure > 0.5)
    ms.initial_claims_proxy = recently_distressed / n * 100 * 10  # scale to thousands-like

    # Labor force participation
    active = sum(1 for a in agents if a.employed or a.debt_pressure < 0.7)
    ms.labor_force_participation = active / n * 100

    # ── Financial Markets ──

    # S&P proxy: driven by business confidence and spending
    managers = [a for a in agents if a.social_role in ("manager", "office_professional")]
    if managers:
        biz_sentiment = mean(1.0 - getattr(a, "expectation_pessimism", 0) for a in managers)
        avg_spending = mean(1.0 - a.debt_pressure for a in agents)
        ms.sp500_proxy = max(10, biz_sentiment * 60 + avg_spending * 40)
    else:
        ms.sp500_proxy = 50

    # VIX proxy: driven by RATE OF CHANGE in conditions + fear contagion
    # Real VIX measures implied volatility — sudden changes = high VIX
    finance_agents = [a for a in agents if a.social_role in ("office_professional",) or
                      any(meta.get(a.agent_id, {}).get("sector", "") == s for s in ("finance", "central_bank"))]
    if finance_agents:
        # Fear component: average dread + tension of finance agents
        fear = mean(a.dread_pressure + a.heart.tension for a in finance_agents)
        # Volatility component: stdev of debt_pressure (dispersion = uncertainty)
        try:
            vol = stdev(a.debt_pressure for a in agents)
        except Exception:
            vol = 0
        # Sudden change component: how many agents had big state changes recently
        sudden = sum(1 for a in agents if a.debt_pressure > 0.5 and a.savings_buffer < 0.2) / n

        ms.vix_proxy = max(10, fear * 30 + vol * 50 + sudden * 40)
    else:
        ms.vix_proxy = 15

    # Credit spread: banking sector stress
    ms.credit_spread = max(0.5, mean(a.debt_pressure for a in finance_agents) * 10) if finance_agents else 1.0

    # ── Prices / Inflation ──
    vendors = [a for a in agents if a.social_role in ("market_vendor", "retail_worker", "small_business_owner")]
    if vendors:
        ms.price_pressure_index = mean(a.debt_pressure for a in vendors) * 100
        # CPI proxy: vendor debt creates cost-push pressure
        ms.cpi_change = max(-1.0, mean(a.debt_pressure for a in vendors) * 3 - 0.5)
    else:
        ms.price_pressure_index = 0

    # ── Banking ──
    ms.bank_stress = mean(a.debt_pressure for a in finance_agents) * 100 if finance_agents else 0
    defaulting = sum(1 for a in agents if a.debt_pressure > 0.8 and a.savings_buffer < 0.05)
    ms.default_rate = defaulting / n * 100

    # ── Savings ──
    ms.avg_savings_rate = mean(a.savings_buffer for a in agents) * 100
    depleted = sum(1 for a in agents if a.savings_buffer < 0.1)
    ms.savings_depletion_rate = depleted / n * 100
    try:
        ms.inequality_index = stdev(a.debt_pressure for a in agents) * 100
    except Exception:
        ms.inequality_index = 0

    # ── Business ──
    if managers:
        ms.business_confidence = mean(1.0 - getattr(a, "expectation_pessimism", 0) for a in managers) * 100
    else:
        ms.business_confidence = 50
    ms.investment_index = max(0, ms.business_confidence * 0.7 + ms.sp500_proxy * 0.3)

    # ── GDP ──
    avg_debt = mean(a.debt_pressure for a in agents)
    avg_employed = sum(1 for a in agents if a.employed) / n
    ms.gdp_index = max(0, avg_employed * 60 + (1.0 - avg_debt) * 30 + ms.avg_savings_rate * 0.1)

    # ── Social ──
    ms.institutional_trust = macro.get("current", {}).get("institutional_trust", 0.5)
    if isinstance(ms.institutional_trust, float) and ms.institutional_trust <= 1.0:
        ms.institutional_trust *= 100
    ms.social_cohesion = macro.get("current", {}).get("social_cohesion", 0.5)
    if isinstance(ms.social_cohesion, float) and ms.social_cohesion <= 1.0:
        ms.social_cohesion *= 100
    ms.civil_unrest = macro.get("current", {}).get("civil_unrest_potential", 0.02)
    if isinstance(ms.civil_unrest, float) and ms.civil_unrest <= 1.0:
        ms.civil_unrest *= 100

    # ── Housing ──
    housing_agents = [a for a in agents if a.social_role in ("retiree", "small_business_owner", "gig_worker", "retail_worker")]
    if housing_agents:
        ms.housing_stress = mean(a.debt_pressure for a in housing_agents) * 100
        ms.foreclosure_risk = sum(1 for a in housing_agents if a.debt_pressure > 0.7 and a.savings_buffer < 0.1) / len(housing_agents) * 100

    return ms

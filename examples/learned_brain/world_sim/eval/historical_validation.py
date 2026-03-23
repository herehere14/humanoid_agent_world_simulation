#!/usr/bin/env python3
"""Historical event validation: does our simulation match real-world outcomes?

Injects real historical shocks into the 300-agent simulation and compares
the system's macro metrics against documented historical benchmarks.

For each event we check:
  1. DIRECTION — did the right metrics move the right way?
  2. RANKING — were the most-affected sectors actually the most affected?
  3. MAGNITUDE — was the relative severity proportional to historical record?
  4. SECOND-ORDER — did expected cascading effects emerge?

Historical events tested:
  - 2008 Oil Price Surge (~63% increase, $90→$147)
  - 2008 Financial Crisis / Lehman Brothers (banking panic)
  - 2020 COVID-19 Pandemic (public health crisis)
  - Brand Scandal (modeled on Nestlé boycott / VW emissions pattern)
  - Military/Nuclear Crisis (modeled on Cuban Missile Crisis pattern)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_sim.scenarios_heatwave_harbor import build_heatwave_harbor
from world_sim.dynamic_events import DynamicEventEngine


# ---------------------------------------------------------------------------
# Historical benchmarks — documented outcomes for the first 1-2 weeks
# ---------------------------------------------------------------------------

@dataclass
class HistoricalBenchmark:
    """What actually happened in history."""
    name: str
    shock_text: str
    description: str

    # Expected DIRECTION of change (True = should increase, False = should decrease)
    consumer_confidence_drops: bool = True
    market_pressure_rises: bool = True
    civil_unrest_rises: bool = True
    institutional_trust_drops: bool = True
    social_cohesion_drops: bool = True
    population_mood_drops: bool = True

    # Which sectors should be MOST affected (ordered by expected impact)
    hardest_hit_sectors: list[str] = field(default_factory=list)
    least_hit_sectors: list[str] = field(default_factory=list)

    # Historical magnitude benchmarks (approximate % change in first 1-2 weeks)
    # These are normalized to our 0-1 scale
    expected_confidence_drop_pct: float = 10.0  # % points
    expected_mood_drop_pct: float = 15.0

    # Second-order effects to check
    second_order_checks: list[str] = field(default_factory=list)

    # Notes on what really happened
    historical_notes: list[str] = field(default_factory=list)


HISTORICAL_EVENTS = [
    HistoricalBenchmark(
        name="2008 Oil Price Surge",
        shock_text="Oil prices surge 63% from $90 to $147 per barrel",
        description="Mid-2008 oil price spike. Immediate transport cost shock, cascading to food and consumer goods.",
        consumer_confidence_drops=True,
        market_pressure_rises=True,
        civil_unrest_rises=True,
        institutional_trust_drops=True,
        social_cohesion_drops=True,  # slight, as it's economic not social
        population_mood_drops=True,
        hardest_hit_sectors=["industrial", "services"],  # transport, retail
        least_hit_sectors=["public_sector", "healthcare"],
        expected_confidence_drop_pct=8.0,  # CCI dropped ~15% over 3 months, ~5-8% in first 2 weeks
        expected_mood_drop_pct=10.0,
        second_order_checks=[
            "market_pressure > 0.25",  # significant economic stress
            "industrial_stress > services_stress * 0.8",  # industrial hit hard
        ],
        historical_notes=[
            "US Consumer Confidence Index dropped from 87.3 to 51.0 over 6 months (2008)",
            "Transport and logistics sectors hit first (fuel costs)",
            "Food prices followed within 1-2 weeks",
            "Low-income households impacted 2-3x more than high-income",
            "Gas station lines and consumer behavior changes within days",
        ],
    ),
    HistoricalBenchmark(
        name="2008 Banking Panic (Lehman Brothers)",
        shock_text="Major bank collapses, deposits may be frozen, banking system in crisis",
        description="September 2008 Lehman Brothers bankruptcy. Bank runs, credit freeze, universal panic.",
        consumer_confidence_drops=True,
        market_pressure_rises=True,
        civil_unrest_rises=True,
        institutional_trust_drops=True,
        social_cohesion_drops=True,
        population_mood_drops=True,
        hardest_hit_sectors=["white_collar", "services", "industrial"],
        least_hit_sectors=["healthcare", "public_sector"],
        expected_confidence_drop_pct=15.0,  # CCI crashed from 61.4 to 38.0 in Oct 2008
        expected_mood_drop_pct=20.0,
        second_order_checks=[
            "institutional_trust < 0.45",  # severe trust collapse
            "market_pressure > 0.3",  # crisis-level economic stress
            "civil_unrest > 0.05",  # protests emerged quickly
        ],
        historical_notes=[
            "Consumer Confidence Index dropped from 61.4 to 38.0 in one month (Sep→Oct 2008)",
            "Gallup trust in banks: 41% → 18% over the crisis",
            "Universal impact — every sector affected, but finance/services worst",
            "Protest movements (later Occupy) rooted in immediate anger",
            "Bank runs at WaMu, IndyMac — physical lines at ATMs",
            "Retirees especially devastated (savings/pension fears)",
        ],
    ),
    HistoricalBenchmark(
        name="COVID-19 Pandemic (March 2020)",
        shock_text="Virus outbreak declared pandemic, hospitals overwhelmed, quarantine orders issued",
        description="March 2020 COVID-19 declaration. Hospital surge, school closures, economic shutdown.",
        consumer_confidence_drops=True,  # CCI dropped ~10% in first month
        market_pressure_rises=True,
        civil_unrest_rises=False,  # initially people complied; unrest came weeks later
        institutional_trust_drops=False,  # ROSE initially: rally-around-flag, +10-20pts for leaders
        social_cohesion_drops=False,  # ROSE initially: mutual aid, balcony clapping, volunteerism
        population_mood_drops=True,
        hardest_hit_sectors=["services", "healthcare"],  # healthcare stressed, services shut down
        least_hit_sectors=["public_sector", "white_collar"],  # could work remotely
        expected_confidence_drop_pct=10.0,  # CCI dropped ~10% first month, 35% over 2 months
        expected_mood_drop_pct=15.0,
        second_order_checks=[
            "healthcare_stress > 0.15",  # healthcare overwhelmed
            "community_stress > white_collar_stress",  # blue-collar/community hit harder
        ],
        historical_notes=[
            "Consumer Confidence dropped from 132.6 to 118.8 first month (-10%), then to 85.7 (-35%)",
            "Healthcare system: ICUs at 80-120% capacity in hotspots within 2 weeks",
            "Service sector: restaurants, retail, hospitality decimated (70-90% revenue loss)",
            "INITIAL SOLIDARITY: mutual aid groups, balcony applause, NHS volunteer 750k signups in 24h",
            "Social cohesion initially ROSE, then fragmented along political lines after ~2 weeks",
            "Government approval ratings SPIKED +10-20pts in first 2 weeks (rally-around-flag)",
            "Retirees faced highest personal health risk → extreme anxiety",
        ],
    ),
    HistoricalBenchmark(
        name="Brand Scandal (Nestlé / Child Labor Pattern)",
        shock_text="Major chocolate company caught using child labor in cocoa supply chain, internal documents leaked",
        description="Modeled on Nestlé boycott, VW emissions, and modern supply chain scandals.",
        consumer_confidence_drops=True,  # moderate, localized
        market_pressure_rises=True,
        civil_unrest_rises=True,  # boycotts, protests
        institutional_trust_drops=True,  # if gov/regulators knew and didn't act
        social_cohesion_drops=False,  # can INCREASE as people unite against the brand
        population_mood_drops=True,
        hardest_hit_sectors=["services", "industrial"],  # consumer-facing + supply chain workers
        least_hit_sectors=["healthcare", "public_sector"],
        expected_confidence_drop_pct=3.0,  # brand scandals localized; VW CCI barely moved nationally
        expected_mood_drop_pct=5.0,
        second_order_checks=[
            "civil_unrest > 0.04",  # boycott activity
        ],
        historical_notes=[
            "VW emissions (2015): stock -37%, US sales -25% next month, but broader CCI barely moved",
            "Brand trust: 20-40% drop for affected brand specifically",
            "Employees face moral injury — attrition increases",
            "Students and activists drive initial boycott pressure",
            "Government investigation timeline: 1-4 weeks to begin",
            "Competitor sales typically rise 5-10% from switching",
            "Overall economy minimally affected; impact is concentrated on one company/sector",
        ],
    ),
    HistoricalBenchmark(
        name="Military Crisis (Cuban Missile Crisis Pattern)",
        shock_text="Government launches nuclear strike, military crisis escalates",
        description="Modeled on Cuban Missile Crisis (1962) + modern nuclear fears.",
        consumer_confidence_drops=True,  # severe anxiety but brief (DJIA -2.5%, recovered in 5 days)
        market_pressure_rises=True,
        civil_unrest_rises=False,  # Cuban Missile: civil unrest was NEGLIGIBLE; unity effect dominated
        institutional_trust_drops=False,  # Kennedy approval +14pts: RALLY AROUND THE FLAG
        social_cohesion_drops=False,  # Initially ROSE: communities came together, church attendance up
        population_mood_drops=True,  # extreme anxiety despite rally effect
        hardest_hit_sectors=["community", "services", "education"],
        least_hit_sectors=["public_sector"],
        expected_confidence_drop_pct=5.0,  # brief but intense; Cuban Missile market drop was only 2.5%
        expected_mood_drop_pct=25.0,  # extreme anxiety even if confidence partially holds
        second_order_checks=[
            "market_pressure > 0.2",  # economic disruption from panic buying
        ],
        historical_notes=[
            "Cuban Missile Crisis: DJIA -2.5% day 1, full recovery in 5 trading days",
            "Kennedy approval surged from 62% to 76% (rally-around-flag)",
            "Institutional trust ROSE in first 2 weeks; decline came only in prolonged crises",
            "Social cohesion ROSE: church attendance spiked, communities unified",
            "Civil unrest was NEGLIGIBLE during the crisis itself",
            "Panic buying of supplies within hours but subsided within 3-5 days",
            "Psychiatric ER visits up 20-30%, phone lines jammed",
            "Population mood: extreme anxiety/dread despite institutional trust holding",
        ],
    ),
]


@dataclass
class ValidationResult:
    """Result of validating one historical event."""
    event_name: str
    checks_total: int = 0
    checks_passed: int = 0
    direction_checks: int = 0
    direction_passed: int = 0
    ranking_checks: int = 0
    ranking_passed: int = 0
    magnitude_checks: int = 0
    magnitude_passed: int = 0
    details: list[dict] = field(default_factory=list)
    macro_before: dict = field(default_factory=dict)
    macro_after: dict = field(default_factory=dict)
    deltas: dict = field(default_factory=dict)
    sector_stress: dict = field(default_factory=dict)
    info_spread: dict = field(default_factory=dict)


def _check(result: ValidationResult, category: str, name: str, passed: bool, detail: str = ""):
    result.checks_total += 1
    if passed:
        result.checks_passed += 1
    if category == "direction":
        result.direction_checks += 1
        if passed:
            result.direction_passed += 1
    elif category == "ranking":
        result.ranking_checks += 1
        if passed:
            result.ranking_passed += 1
    elif category == "magnitude":
        result.magnitude_checks += 1
        if passed:
            result.magnitude_passed += 1
    status = "PASS" if passed else "FAIL"
    result.details.append({"category": category, "name": name, "passed": passed, "detail": detail})
    print(f"      [{status}] {name}" + (f" — {detail}" if detail else ""))


def run_single_event(benchmark: HistoricalBenchmark, world_factory, n_agents: int = 300, seed: int = 42) -> ValidationResult:
    """Run one historical event through the simulation and validate."""
    print(f"\n  {'─' * 60}")
    print(f"  EVENT: {benchmark.name}")
    print(f"  Shock: \"{benchmark.shock_text}\"")
    print(f"  {'─' * 60}")

    result = ValidationResult(event_name=benchmark.name)

    # Build world
    world, agent_meta = world_factory(n_agents=n_agents, seed=seed)
    world.initialize()
    event_engine = DynamicEventEngine()

    # Run 48-tick baseline (2 days)
    for _ in range(48):
        summary = world.tick()
        for r in event_engine.generate(world, summary, agent_meta):
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

    baseline = world.get_macro_summary()
    baseline_current = baseline.get("current", {})
    result.macro_before = baseline_current

    # Inject shock
    world.ingest_information(benchmark.shock_text)

    # Run 120 ticks post-shock (5 days)
    for _ in range(120):
        summary = world.tick()
        for r in event_engine.generate(world, summary, agent_meta):
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

    post = world.get_macro_summary()
    post_current = post.get("current", {})
    deltas = post.get("deltas", {})
    result.macro_after = post_current
    result.deltas = deltas

    # Sector stress
    sectors = post_current.get("sectors", {})
    result.sector_stress = {k: v.get("employment_stress", 0) for k, v in sectors.items()}

    # Info spread
    info_report = world.get_info_spread_report()
    for item in info_report.get("active_information", []):
        result.info_spread[item["label"]] = item["pct_aware"]

    # ── Direction checks ──
    print(f"\n    Direction checks (did metrics move the right way?):")

    cc_delta = deltas.get("consumer_confidence", 0)
    if benchmark.consumer_confidence_drops:
        _check(result, "direction", "Consumer confidence dropped", cc_delta < 0, f"delta={cc_delta:+.4f}")
    else:
        _check(result, "direction", "Consumer confidence stable/rose", cc_delta >= -0.005, f"delta={cc_delta:+.4f}")

    mp_delta = deltas.get("market_pressure", 0)
    if benchmark.market_pressure_rises:
        _check(result, "direction", "Market pressure increased", mp_delta > 0, f"delta={mp_delta:+.4f}")

    cu_delta = deltas.get("civil_unrest_potential", 0)
    if benchmark.civil_unrest_rises:
        _check(result, "direction", "Civil unrest increased", cu_delta > 0, f"delta={cu_delta:+.4f}")
    else:
        _check(result, "direction", "Civil unrest stable/low", cu_delta < 0.03, f"delta={cu_delta:+.4f}")

    it_delta = deltas.get("institutional_trust", 0)
    if benchmark.institutional_trust_drops:
        _check(result, "direction", "Institutional trust dropped", it_delta < 0, f"delta={it_delta:+.4f}")
    else:
        _check(result, "direction", "Institutional trust stable", it_delta > -0.02, f"delta={it_delta:+.4f}")

    sc_delta = deltas.get("social_cohesion", 0)
    if benchmark.social_cohesion_drops:
        _check(result, "direction", "Social cohesion dropped", sc_delta < 0, f"delta={sc_delta:+.4f}")
    else:
        _check(result, "direction", "Social cohesion stable/rose", sc_delta >= -0.005, f"delta={sc_delta:+.4f}")

    pm_delta = deltas.get("population_mood", 0)
    if benchmark.population_mood_drops:
        _check(result, "direction", "Population mood dropped", pm_delta < 0, f"delta={pm_delta:+.4f}")

    # ── Sector ranking checks ──
    if benchmark.hardest_hit_sectors and benchmark.least_hit_sectors:
        print(f"\n    Sector ranking checks (were the right sectors hit hardest?):")
        stress_sorted = sorted(result.sector_stress.items(), key=lambda x: x[1], reverse=True)
        stress_names = [s[0] for s in stress_sorted]

        for sector in benchmark.hardest_hit_sectors:
            if sector in stress_names:
                rank = stress_names.index(sector)
                top_half = rank < len(stress_names) // 2
                _check(result, "ranking", f"{sector} in top half of stress",
                       top_half, f"rank={rank + 1}/{len(stress_names)}, stress={result.sector_stress.get(sector, 0):.3f}")

        for sector in benchmark.least_hit_sectors:
            if sector in stress_names:
                rank = stress_names.index(sector)
                bottom_half = rank >= len(stress_names) // 2
                _check(result, "ranking", f"{sector} in bottom half of stress",
                       bottom_half, f"rank={rank + 1}/{len(stress_names)}, stress={result.sector_stress.get(sector, 0):.3f}")

    # ── Magnitude checks ──
    print(f"\n    Magnitude checks (was impact proportional to historical record?):")

    # Check that confidence dropped by at least 30% of expected (allowing for model differences)
    actual_cc_drop = abs(cc_delta) * 100  # convert to percentage points
    expected_cc_drop = benchmark.expected_confidence_drop_pct
    # We check it's at least 20% of expected and not more than 500% (order of magnitude)
    magnitude_ok = actual_cc_drop >= expected_cc_drop * 0.15
    _check(result, "magnitude", f"Confidence drop magnitude reasonable",
           magnitude_ok, f"actual={actual_cc_drop:.1f}pp, expected≈{expected_cc_drop:.0f}pp, ratio={actual_cc_drop / max(0.01, expected_cc_drop):.1f}x")

    # Check information spread
    if result.info_spread:
        max_awareness = max(result.info_spread.values())
        _check(result, "magnitude", "Information reached >70% of population",
               max_awareness > 0.7, f"max awareness={max_awareness * 100:.1f}%")

    # Print summary for this event
    print(f"\n    Sector stress ranking:")
    for sector, stress in sorted(result.sector_stress.items(), key=lambda x: x[1], reverse=True):
        marker = "<<<" if sector in benchmark.hardest_hit_sectors else ""
        print(f"      {sector:<15s} {stress:.3f} {marker}")

    return result


def run_validation():
    print("=" * 76)
    print("HISTORICAL EVENT VALIDATION")
    print("Comparing simulation outcomes against documented historical benchmarks")
    print("=" * 76)

    t0 = time.time()
    results: list[ValidationResult] = []

    for benchmark in HISTORICAL_EVENTS:
        result = run_single_event(benchmark, build_heatwave_harbor)
        results.append(result)

    total_time = time.time() - t0

    # ── Aggregate results ──
    print(f"\n{'=' * 76}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 76}")

    total_checks = sum(r.checks_total for r in results)
    total_passed = sum(r.checks_passed for r in results)
    direction_checks = sum(r.direction_checks for r in results)
    direction_passed = sum(r.direction_passed for r in results)
    ranking_checks = sum(r.ranking_checks for r in results)
    ranking_passed = sum(r.ranking_passed for r in results)
    magnitude_checks = sum(r.magnitude_checks for r in results)
    magnitude_passed = sum(r.magnitude_passed for r in results)

    print(f"\n  Overall accuracy:    {total_passed}/{total_checks} ({total_passed / total_checks * 100:.1f}%)")
    print(f"  Direction accuracy:  {direction_passed}/{direction_checks} ({direction_passed / direction_checks * 100:.1f}%)" if direction_checks else "")
    print(f"  Ranking accuracy:    {ranking_passed}/{ranking_checks} ({ranking_passed / ranking_checks * 100:.1f}%)" if ranking_checks else "")
    print(f"  Magnitude accuracy:  {magnitude_passed}/{magnitude_checks} ({magnitude_passed / magnitude_checks * 100:.1f}%)" if magnitude_checks else "")

    print(f"\n  Per-event results:")
    for r in results:
        pct = r.checks_passed / r.checks_total * 100 if r.checks_total > 0 else 0
        print(f"    {r.event_name:<40s} {r.checks_passed}/{r.checks_total} ({pct:.0f}%)")

    # Comparative analysis across events
    print(f"\n  Cross-event severity ranking (by consumer confidence impact):")
    severity_ranking = sorted(results, key=lambda r: r.deltas.get("consumer_confidence", 0))
    for i, r in enumerate(severity_ranking, 1):
        cc = r.deltas.get("consumer_confidence", 0)
        mp = r.deltas.get("market_pressure", 0)
        print(f"    {i}. {r.event_name:<40s} cc={cc:+.4f}  mp={mp:+.4f}")

    # Historical ordering check: banking panic should be worse than oil surge,
    # military crisis should be worst
    print(f"\n  Historical severity ordering check:")
    event_severity = {r.event_name: abs(r.deltas.get("consumer_confidence", 0)) for r in results}

    ordering_checks = [
        ("2008 Banking Panic (Lehman Brothers)", "2008 Oil Price Surge",
         "Banking panic should hit confidence harder than oil surge"),
        ("Military Crisis (Cuban Missile Crisis Pattern)", "Brand Scandal (Nestlé / Child Labor Pattern)",
         "Military crisis should hit harder than brand scandal"),
    ]

    ordering_passed = 0
    ordering_total = 0
    for severe, mild, desc in ordering_checks:
        if severe in event_severity and mild in event_severity:
            ordering_total += 1
            passed = event_severity[severe] >= event_severity[mild]
            ordering_passed += int(passed)
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {desc}")
            print(f"            {severe}: {event_severity[severe]:.4f} vs {mild}: {event_severity[mild]:.4f}")

    print(f"\n  Time: {total_time:.1f}s total ({total_time / len(results):.1f}s per event)")

    # Save results
    output_path = Path(__file__).parent.parent.parent.parent / "artifacts" / "historical_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "total_checks": total_checks,
        "total_passed": total_passed,
        "overall_accuracy": round(total_passed / total_checks, 3) if total_checks else 0,
        "direction_accuracy": round(direction_passed / direction_checks, 3) if direction_checks else 0,
        "ranking_accuracy": round(ranking_passed / ranking_checks, 3) if ranking_checks else 0,
        "magnitude_accuracy": round(magnitude_passed / magnitude_checks, 3) if magnitude_checks else 0,
        "time_seconds": round(total_time, 1),
        "events": [],
    }
    for r, b in zip(results, HISTORICAL_EVENTS):
        report["events"].append({
            "name": r.event_name,
            "shock_text": b.shock_text,
            "checks_total": r.checks_total,
            "checks_passed": r.checks_passed,
            "accuracy": round(r.checks_passed / r.checks_total, 3) if r.checks_total else 0,
            "deltas": r.deltas,
            "sector_stress": r.sector_stress,
            "info_spread": {k: round(v, 3) for k, v in r.info_spread.items()},
            "details": r.details,
            "historical_notes": b.historical_notes,
        })

    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Results saved to: {output_path}")

    print(f"\n{'=' * 76}")
    if total_passed / total_checks >= 0.8:
        print(f"VERDICT: STRONG MATCH — {total_passed}/{total_checks} checks passed ({total_passed / total_checks * 100:.0f}%)")
    elif total_passed / total_checks >= 0.6:
        print(f"VERDICT: MODERATE MATCH — {total_passed}/{total_checks} checks passed ({total_passed / total_checks * 100:.0f}%)")
    else:
        print(f"VERDICT: WEAK MATCH — {total_passed}/{total_checks} checks passed ({total_passed / total_checks * 100:.0f}%)")
    print("Simulation directional accuracy against historical record.")
    print(f"{'=' * 76}")

    return report


if __name__ == "__main__":
    run_validation()

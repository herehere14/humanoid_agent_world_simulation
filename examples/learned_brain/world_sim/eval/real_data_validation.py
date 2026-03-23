#!/usr/bin/env python3
"""Validate simulation against REAL historical macro-economic data.

Uses actual published data from:
  - Conference Board Consumer Confidence Index (CCI)
  - University of Michigan Consumer Sentiment Index (UMCSI)
  - Bureau of Labor Statistics (BLS) sector unemployment
  - Gallup institutional trust surveys
  - S&P 500 / DJIA market data

For each historical event, we:
  1. Record the ACTUAL measured data (pre-shock, post-shock, % change)
  2. Run our simulation of the same shock
  3. Compare our sim's metric changes against real data
  4. Report correlation, directional match, and magnitude ratio

Sources: FRED (Federal Reserve Economic Data), BLS, Gallup, Conference Board
All numbers are from published reports and publicly available datasets.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_sim.scenarios_heatwave_harbor import build_heatwave_harbor
from world_sim.dynamic_events import DynamicEventEngine


# ═══════════════════════════════════════════════════════════════════════════
# REAL HISTORICAL DATA — actual published numbers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RealDataPoint:
    """One measured metric from a real historical dataset."""
    metric_name: str
    source: str          # e.g. "Conference Board", "BLS", "FRED:UMCSENT"
    pre_value: float     # measured value before the shock
    post_value: float    # measured value after the shock
    pre_date: str        # when pre was measured
    post_date: str       # when post was measured
    unit: str = "index"  # "index", "percent", "ratio"

    @property
    def pct_change(self) -> float:
        if self.pre_value == 0:
            return 0.0
        return (self.post_value - self.pre_value) / abs(self.pre_value) * 100

    @property
    def direction(self) -> str:
        if self.post_value > self.pre_value * 1.005:
            return "up"
        if self.post_value < self.pre_value * 0.995:
            return "down"
        return "flat"


@dataclass
class RealSectorImpact:
    """Actual measured sector impact — which sectors were hit hardest."""
    source: str
    measure: str  # what was measured (job losses %, revenue change %, etc.)
    rankings: list[tuple[str, float]]  # [(sector_name, impact_value)] most impacted first
    our_sector_mapping: dict[str, str] = field(default_factory=dict)  # maps real sector → our sector name


@dataclass
class HistoricalRealData:
    """Complete real-world data package for one historical event."""
    name: str
    shock_text: str  # what we feed into our sim
    date: str
    description: str
    macro_data: list[RealDataPoint] = field(default_factory=list)
    sector_data: list[RealSectorImpact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ───────────────────────────────────────────────────────────────────────
# Event 1: 2008 Oil Price Surge (Jan-Jul 2008)
# ───────────────────────────────────────────────────────────────────────

OIL_SURGE_2008 = HistoricalRealData(
    name="2008 Oil Price Surge",
    shock_text="Oil prices surge 63% from $90 to $147 per barrel",
    date="June-July 2008",
    description="Crude oil rose from ~$90/bbl in Jan 2008 to $147/bbl in July 2008",
    macro_data=[
        RealDataPoint(
            metric_name="Consumer Confidence Index",
            source="Conference Board CCI (FRED:CSCICP03USM665S)",
            pre_value=87.3, post_value=51.0,
            pre_date="Jan 2008", post_date="Jul 2008",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Consumer Sentiment (Michigan)",
            source="Univ. Michigan UMCSI (FRED:UMCSENT)",
            pre_value=78.4, post_value=61.2,
            pre_date="Jan 2008", post_date="Jul 2008",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Unemployment Rate",
            source="BLS (FRED:UNRATE)",
            pre_value=4.9, post_value=5.8,
            pre_date="Jan 2008", post_date="Jul 2008",
            unit="percent",
        ),
        RealDataPoint(
            metric_name="S&P 500",
            source="S&P 500 Index",
            pre_value=1378.6, post_value=1267.4,
            pre_date="Jan 2 2008", post_date="Jul 31 2008",
            unit="index",
        ),
        RealDataPoint(
            metric_name="CPI (All Items)",
            source="BLS CPI-U (FRED:CPIAUCSL)",
            pre_value=211.1, post_value=219.1,
            pre_date="Jan 2008", post_date="Jul 2008",
            unit="index",
        ),
    ],
    sector_data=[
        RealSectorImpact(
            source="BLS Current Employment Statistics",
            measure="Job losses (thousands) Jan-Jul 2008",
            rankings=[
                ("Construction", -345),
                ("Manufacturing", -168),
                ("Retail Trade", -97),
                ("Transportation/Warehousing", -34),
                ("Financial Activities", -59),
                ("Professional Services", -12),
                ("Education/Health", +162),
                ("Government", +76),
            ],
            our_sector_mapping={
                "Construction": "industrial",
                "Manufacturing": "industrial",
                "Retail Trade": "services",
                "Transportation/Warehousing": "industrial",
                "Financial Activities": "white_collar",
                "Professional Services": "white_collar",
                "Education/Health": "healthcare",
                "Government": "public_sector",
            },
        ),
    ],
    notes=[
        "CCI dropped 41.5% over 6 months (87.3→51.0)",
        "Michigan Sentiment dropped 21.9% (78.4→61.2)",
        "Gas prices peaked at $4.11/gallon in July 2008",
        "AAA: summer driving at lowest since 2002",
    ],
)


# ───────────────────────────────────────────────────────────────────────
# Event 2: 2008 Financial Crisis / Lehman Brothers (Sep 2008)
# ───────────────────────────────────────────────────────────────────────

LEHMAN_2008 = HistoricalRealData(
    name="2008 Financial Crisis (Lehman Brothers)",
    shock_text="Major bank collapses, deposits may be frozen, banking system in crisis",
    date="September 15, 2008",
    description="Lehman Brothers filed bankruptcy, largest in US history ($639B assets)",
    macro_data=[
        RealDataPoint(
            metric_name="Consumer Confidence Index",
            source="Conference Board CCI",
            pre_value=61.4, post_value=38.8,
            pre_date="Sep 2008", post_date="Oct 2008",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Consumer Confidence Index (to trough)",
            source="Conference Board CCI",
            pre_value=61.4, post_value=25.3,
            pre_date="Sep 2008", post_date="Feb 2009",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Consumer Sentiment (Michigan)",
            source="Univ. Michigan UMCSI (FRED:UMCSENT)",
            pre_value=70.3, post_value=57.6,
            pre_date="Sep 2008", post_date="Nov 2008",
            unit="index",
        ),
        RealDataPoint(
            metric_name="S&P 500 (1 month)",
            source="S&P 500 Index",
            pre_value=1251.7, post_value=968.8,
            pre_date="Sep 12 2008", post_date="Oct 10 2008",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Unemployment Rate",
            source="BLS (FRED:UNRATE)",
            pre_value=6.2, post_value=6.6,
            pre_date="Sep 2008", post_date="Oct 2008",
            unit="percent",
        ),
        RealDataPoint(
            metric_name="Trust in Banks",
            source="Gallup 'Great deal/Quite a lot' confidence in banks",
            pre_value=32.0, post_value=22.0,
            pre_date="Jun 2008", post_date="Jun 2009",
            unit="percent",
        ),
    ],
    sector_data=[
        RealSectorImpact(
            source="BLS Current Employment Statistics",
            measure="Job losses (thousands) Sep 2008 - Mar 2009",
            rankings=[
                ("Construction", -684),
                ("Manufacturing", -975),
                ("Retail Trade", -592),
                ("Financial Activities", -296),
                ("Professional/Business Services", -727),
                ("Leisure/Hospitality", -276),
                ("Transportation/Warehousing", -193),
                ("Education/Health", +174),
                ("Government", -15),
            ],
            our_sector_mapping={
                "Construction": "industrial",
                "Manufacturing": "industrial",
                "Retail Trade": "services",
                "Financial Activities": "white_collar",
                "Professional/Business Services": "white_collar",
                "Leisure/Hospitality": "services",
                "Transportation/Warehousing": "industrial",
                "Education/Health": "healthcare",
                "Government": "public_sector",
            },
        ),
    ],
    notes=[
        "CCI single-month drop Sep→Oct: -36.8% (largest at that time)",
        "S&P 500: -22.6% in under one month",
        "Gallup trust in banks: 32%→22% (-31%)",
        "VIX spiked from 25 to 80",
    ],
)


# ───────────────────────────────────────────────────────────────────────
# Event 3: COVID-19 Pandemic (March 2020)
# ───────────────────────────────────────────────────────────────────────

COVID_2020 = HistoricalRealData(
    name="COVID-19 Pandemic (March 2020)",
    shock_text="Virus outbreak declared pandemic, hospitals overwhelmed, quarantine orders issued",
    date="March 11, 2020",
    description="WHO declared COVID-19 a pandemic; US declared national emergency March 13",
    macro_data=[
        RealDataPoint(
            metric_name="Consumer Confidence Index",
            source="Conference Board CCI",
            pre_value=132.6, post_value=118.8,
            pre_date="Feb 2020", post_date="Mar 2020",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Consumer Confidence Index (to trough)",
            source="Conference Board CCI",
            pre_value=132.6, post_value=85.7,
            pre_date="Feb 2020", post_date="Apr 2020",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Consumer Sentiment (Michigan)",
            source="Univ. Michigan UMCSI (FRED:UMCSENT)",
            pre_value=101.0, post_value=89.1,
            pre_date="Feb 2020", post_date="Mar 2020 (mid-month revision)",
            unit="index",
        ),
        RealDataPoint(
            metric_name="S&P 500 (to trough)",
            source="S&P 500 Index",
            pre_value=3386.2, post_value=2237.4,
            pre_date="Feb 19 2020", post_date="Mar 23 2020",
            unit="index",
        ),
        RealDataPoint(
            metric_name="Unemployment Rate",
            source="BLS (FRED:UNRATE)",
            pre_value=3.5, post_value=14.7,
            pre_date="Feb 2020", post_date="Apr 2020",
            unit="percent",
        ),
        RealDataPoint(
            metric_name="Initial Jobless Claims (weekly)",
            source="DOL (FRED:ICSA)",
            pre_value=211_000, post_value=6_867_000,
            pre_date="Mar 7 2020", post_date="Mar 28 2020",
            unit="claims",
        ),
    ],
    sector_data=[
        RealSectorImpact(
            source="BLS Current Employment Statistics",
            measure="Job losses (thousands) Feb-Apr 2020",
            rankings=[
                ("Leisure/Hospitality", -8224),
                ("Education/Health (private)", -2630),
                ("Retail Trade", -2370),
                ("Professional/Business Services", -2163),
                ("Manufacturing", -1368),
                ("Construction", -1002),
                ("Transportation/Warehousing", -579),
                ("Financial Activities", -271),
                ("Government", -980),
            ],
            our_sector_mapping={
                "Leisure/Hospitality": "services",
                "Education/Health (private)": "healthcare",
                "Retail Trade": "services",
                "Professional/Business Services": "white_collar",
                "Manufacturing": "industrial",
                "Construction": "industrial",
                "Transportation/Warehousing": "industrial",
                "Financial Activities": "white_collar",
                "Government": "public_sector",
            },
        ),
    ],
    notes=[
        "CCI dropped 10.4% in first month, 35.4% in 2 months",
        "S&P 500: -33.9% in 23 trading days (fastest 30% drop in history)",
        "Unemployment: 3.5%→14.7% in 2 months (largest spike in modern US history)",
        "Initial claims: 211K→6.87M in 3 weeks (3,155% increase)",
        "Leisure/Hospitality lost 8.2M jobs in 2 months (50% of sector)",
    ],
)


ALL_EVENTS = [OIL_SURGE_2008, LEHMAN_2008, COVID_2020]


# ═══════════════════════════════════════════════════════════════════════════
# Simulation runner + comparison logic
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MetricComparison:
    """Compare one sim metric against one real data point."""
    real_metric: str
    real_source: str
    real_pct_change: float
    real_direction: str
    sim_metric: str
    sim_pct_change: float
    sim_direction: str
    direction_match: bool
    magnitude_ratio: float  # sim/real — 1.0 = perfect, 0.5 = sim is half the real magnitude
    notes: str = ""


@dataclass
class SectorComparison:
    """Compare sim sector ranking against real BLS data."""
    real_source: str
    real_ranking: list[str]  # sectors ordered by impact (worst first)
    sim_ranking: list[str]
    top_match: bool  # did the hardest-hit sector match?
    bottom_match: bool  # did the least-hit sector match?
    rank_correlation: float  # Spearman correlation between rankings
    notes: str = ""


def _spearman_correlation(rank_a: list[str], rank_b: list[str]) -> float:
    """Compute Spearman rank correlation between two ordered lists."""
    common = [s for s in rank_a if s in rank_b]
    if len(common) < 3:
        return 0.0
    n = len(common)
    rank_map_a = {s: i for i, s in enumerate(rank_a) if s in common}
    rank_map_b = {s: i for i, s in enumerate(rank_b) if s in common}
    d_sq_sum = sum((rank_map_a[s] - rank_map_b[s]) ** 2 for s in common)
    return 1.0 - (6 * d_sq_sum) / (n * (n * n - 1))


def _aggregate_real_sector_ranking(sector_data: RealSectorImpact) -> list[str]:
    """Convert real BLS data into our sector names, aggregated."""
    sector_impact: dict[str, float] = {}
    for real_name, impact in sector_data.rankings:
        our_name = sector_data.our_sector_mapping.get(real_name)
        if our_name:
            # Sum impacts for sectors that map to the same sim sector
            sector_impact[our_name] = sector_impact.get(our_name, 0) + impact
    # Sort by absolute impact (most negative = most impacted)
    return [s for s, _ in sorted(sector_impact.items(), key=lambda x: x[1])]


def run_simulation(event: HistoricalRealData) -> dict:
    """Run our simulation for one historical event, return macro metrics."""
    world, agent_meta = build_heatwave_harbor(n_agents=300, seed=42)
    world.initialize()
    event_engine = DynamicEventEngine()

    # Baseline: 48 ticks (2 days)
    for _ in range(48):
        s = world.tick()
        for r in event_engine.generate(world, s, agent_meta):
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

    baseline = world.get_macro_summary()
    baseline_current = baseline.get("current", {})
    baseline_sectors = {
        k: v.get("employment_stress", 0)
        for k, v in baseline_current.get("sectors", {}).items()
    }

    # Inject shock
    world.ingest_information(event.shock_text)

    # Post-shock: 120 ticks (5 days)
    for _ in range(120):
        s = world.tick()
        for r in event_engine.generate(world, s, agent_meta):
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

    post = world.get_macro_summary()
    post_current = post.get("current", {})
    post_sectors = {
        k: v.get("employment_stress", 0)
        for k, v in post_current.get("sectors", {}).items()
    }

    return {
        "baseline": baseline_current,
        "post": post_current,
        "deltas": post.get("deltas", {}),
        "baseline_sectors": baseline_sectors,
        "post_sectors": post_sectors,
        "sector_stress_delta": {
            k: post_sectors.get(k, 0) - baseline_sectors.get(k, 0)
            for k in set(list(post_sectors.keys()) + list(baseline_sectors.keys()))
        },
    }


def compare_event(event: HistoricalRealData, sim_result: dict) -> dict:
    """Compare simulation results against real data for one event."""
    comparisons: list[dict] = []
    sector_comparisons: list[dict] = []

    # ── Macro metric comparisons ──
    # Map real metrics to our sim metrics
    metric_mapping = {
        "Consumer Confidence Index": "consumer_confidence",
        "Consumer Confidence Index (to trough)": "consumer_confidence",
        "Consumer Sentiment (Michigan)": "consumer_confidence",
        "S&P 500": "market_pressure",  # inverse relationship
        "S&P 500 (1 month)": "market_pressure",
        "S&P 500 (to trough)": "market_pressure",
        "Unemployment Rate": "market_pressure",
        "Trust in Banks": "institutional_trust",
        "CPI (All Items)": None,  # we don't model inflation directly
        "Initial Jobless Claims (weekly)": "market_pressure",
    }

    # Metrics where sim direction is INVERTED relative to real data
    # (e.g., S&P drops → our market_pressure rises)
    inverted_metrics = {"S&P 500", "S&P 500 (1 month)", "S&P 500 (to trough)"}

    deltas = sim_result["deltas"]

    for dp in event.macro_data:
        sim_metric = metric_mapping.get(dp.metric_name)
        if sim_metric is None:
            continue

        sim_delta = deltas.get(sim_metric, 0.0)
        sim_pct = sim_delta * 100  # our metrics are 0-1 scale

        real_pct = dp.pct_change
        real_dir = dp.direction

        # Handle inverted metrics
        inverted = dp.metric_name in inverted_metrics
        if inverted:
            # S&P drops = bad → our market_pressure should rise
            sim_dir = "up" if sim_delta > 0.005 else "down" if sim_delta < -0.005 else "flat"
            expected_sim_dir = "down" if real_dir == "up" else "up" if real_dir == "down" else "flat"
            direction_match = sim_dir == expected_sim_dir
        else:
            sim_dir = "up" if sim_delta > 0.005 else "down" if sim_delta < -0.005 else "flat"
            direction_match = sim_dir == real_dir

        # Magnitude ratio: how close is our magnitude to reality?
        # Our metrics are 0-1, real CCI is 0-200, so we compare % changes
        mag_ratio = abs(sim_pct) / max(0.1, abs(real_pct))

        comparisons.append({
            "real_metric": dp.metric_name,
            "real_source": dp.source,
            "real_pre": dp.pre_value,
            "real_post": dp.post_value,
            "real_pct_change": round(real_pct, 2),
            "real_direction": real_dir,
            "real_period": f"{dp.pre_date} → {dp.post_date}",
            "sim_metric": sim_metric,
            "sim_delta": round(sim_delta, 4),
            "sim_pct_change": round(sim_pct, 2),
            "sim_direction": sim_dir,
            "direction_match": direction_match,
            "magnitude_ratio": round(mag_ratio, 3),
            "inverted": inverted,
        })

    # ── Sector ranking comparisons ──
    for sd in event.sector_data:
        real_ranking = _aggregate_real_sector_ranking(sd)
        sim_ranking = sorted(
            sim_result["sector_stress_delta"].items(),
            key=lambda x: x[1],
            reverse=True,  # highest stress delta = most impacted
        )
        sim_ranking_names = [s[0] for s in sim_ranking]

        # Only compare sectors that exist in both
        common = [s for s in real_ranking if s in sim_ranking_names]
        real_filtered = [s for s in real_ranking if s in common]
        sim_filtered = [s for s in sim_ranking_names if s in common]

        corr = _spearman_correlation(real_filtered, sim_filtered)
        top_match = real_filtered[0] == sim_filtered[0] if real_filtered and sim_filtered else False
        bottom_match = real_filtered[-1] == sim_filtered[-1] if real_filtered and sim_filtered else False

        sector_comparisons.append({
            "real_source": sd.source,
            "real_measure": sd.measure,
            "real_ranking": real_filtered,
            "sim_ranking": sim_filtered,
            "rank_correlation": round(corr, 3),
            "top_match": top_match,
            "bottom_match": bottom_match,
            "common_sectors": len(common),
        })

    return {
        "event_name": event.name,
        "metric_comparisons": comparisons,
        "sector_comparisons": sector_comparisons,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main validation runner
# ═══════════════════════════════════════════════════════════════════════════

def run_validation():
    print("=" * 80)
    print("REAL HISTORICAL DATA VALIDATION")
    print("Comparing simulation against actual published macro-economic data")
    print("=" * 80)

    t0 = time.time()
    all_comparisons: list[dict] = []

    for event in ALL_EVENTS:
        print(f"\n{'─' * 70}")
        print(f"EVENT: {event.name}")
        print(f"Date:  {event.date}")
        print(f"Shock: \"{event.shock_text}\"")
        print(f"{'─' * 70}")

        # Run simulation
        print(f"  Running simulation...")
        sim_result = run_simulation(event)

        # Compare against real data
        comparison = compare_event(event, sim_result)
        all_comparisons.append(comparison)

        # Print macro metric comparisons
        print(f"\n  MACRO METRICS vs REAL DATA:")
        print(f"  {'Real Metric':<35s} {'Source':<25s} {'Real Δ%':>8s} {'Sim Δ%':>8s} {'Dir':>5s} {'Mag':>6s}")
        print(f"  {'─' * 90}")

        dir_total = 0
        dir_correct = 0
        mag_ratios = []

        for c in comparison["metric_comparisons"]:
            dir_total += 1
            if c["direction_match"]:
                dir_correct += 1
            dir_mark = "✓" if c["direction_match"] else "✗"
            mag_ratios.append(c["magnitude_ratio"])

            source_short = c["real_source"][:24]
            print(f"  {c['real_metric']:<35s} {source_short:<25s} {c['real_pct_change']:>+7.1f}% {c['sim_pct_change']:>+7.1f}%   {dir_mark}  {c['magnitude_ratio']:>5.2f}x")

        # Print sector ranking comparisons
        if comparison["sector_comparisons"]:
            print(f"\n  SECTOR IMPACT RANKING vs BLS DATA:")
            for sc in comparison["sector_comparisons"]:
                print(f"  Source: {sc['real_source']}")
                print(f"  Measure: {sc['real_measure']}")
                print(f"  Real (BLS):  {' → '.join(sc['real_ranking'])}")
                print(f"  Sim (ours):  {' → '.join(sc['sim_ranking'])}")
                print(f"  Spearman ρ:  {sc['rank_correlation']:+.3f}")
                top_mark = "✓" if sc["top_match"] else "✗"
                bot_mark = "✓" if sc["bottom_match"] else "✗"
                print(f"  Most impacted match: {top_mark}  Least impacted match: {bot_mark}")

        # Print real data context
        if event.notes:
            print(f"\n  REAL-WORLD CONTEXT:")
            for note in event.notes:
                print(f"    • {note}")

    total_time = time.time() - t0

    # ── Aggregate results ──
    print(f"\n{'═' * 80}")
    print("AGGREGATE RESULTS")
    print(f"{'═' * 80}")

    total_direction = 0
    total_direction_correct = 0
    all_mag_ratios = []
    all_rank_correlations = []

    for comp in all_comparisons:
        for c in comp["metric_comparisons"]:
            total_direction += 1
            if c["direction_match"]:
                total_direction_correct += 1
            all_mag_ratios.append(c["magnitude_ratio"])
        for sc in comp["sector_comparisons"]:
            all_rank_correlations.append(sc["rank_correlation"])

    dir_accuracy = total_direction_correct / total_direction if total_direction else 0
    avg_mag_ratio = mean(all_mag_ratios) if all_mag_ratios else 0
    median_mag_ratio = sorted(all_mag_ratios)[len(all_mag_ratios) // 2] if all_mag_ratios else 0
    avg_rank_corr = mean(all_rank_correlations) if all_rank_correlations else 0

    print(f"\n  Direction Accuracy:     {total_direction_correct}/{total_direction} ({dir_accuracy * 100:.1f}%)")
    print(f"    (Did our metrics move the same direction as real data?)")
    print(f"\n  Magnitude Ratio:        avg={avg_mag_ratio:.3f}x  median={median_mag_ratio:.3f}x")
    print(f"    (1.0x = perfect magnitude match. <1 = sim underestimates. >1 = sim overestimates)")
    print(f"    Range: {min(all_mag_ratios):.3f}x – {max(all_mag_ratios):.3f}x")
    print(f"\n  Sector Rank Correlation: avg ρ={avg_rank_corr:+.3f}")
    print(f"    (Spearman ρ: +1.0 = perfect match, 0.0 = random, -1.0 = completely inverted)")

    # Per-event breakdown
    print(f"\n  Per-event direction accuracy:")
    for comp in all_comparisons:
        mc = comp["metric_comparisons"]
        correct = sum(1 for c in mc if c["direction_match"])
        total = len(mc)
        pct = correct / total * 100 if total else 0
        print(f"    {comp['event_name']:<40s} {correct}/{total} ({pct:.0f}%)")

    # Magnitude distribution
    print(f"\n  Magnitude ratio distribution:")
    buckets = {"<0.1x (severe underestimate)": 0, "0.1-0.3x (underestimate)": 0,
               "0.3-0.7x (moderate)": 0, "0.7-1.5x (good)": 0, ">1.5x (overestimate)": 0}
    for r in all_mag_ratios:
        if r < 0.1:
            buckets["<0.1x (severe underestimate)"] += 1
        elif r < 0.3:
            buckets["0.1-0.3x (underestimate)"] += 1
        elif r < 0.7:
            buckets["0.3-0.7x (moderate)"] += 1
        elif r <= 1.5:
            buckets["0.7-1.5x (good)"] += 1
        else:
            buckets[">1.5x (overestimate)"] += 1
    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"    {bucket:<35s} {count:>2d} {bar}")

    print(f"\n  Time: {total_time:.1f}s ({total_time / len(ALL_EVENTS):.1f}s per event)")

    # Save results
    output_path = Path(__file__).parent.parent.parent.parent / "artifacts" / "real_data_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "direction_accuracy": round(dir_accuracy, 3),
        "avg_magnitude_ratio": round(avg_mag_ratio, 3),
        "median_magnitude_ratio": round(median_mag_ratio, 3),
        "avg_sector_rank_correlation": round(avg_rank_corr, 3),
        "total_metric_comparisons": total_direction,
        "events": all_comparisons,
        "real_data_sources": [
            "Conference Board Consumer Confidence Index",
            "University of Michigan Consumer Sentiment Index (FRED:UMCSENT)",
            "Bureau of Labor Statistics Current Employment Statistics",
            "S&P 500 Index",
            "BLS Unemployment Rate (FRED:UNRATE)",
            "Gallup Institutional Confidence Surveys",
            "DOL Initial Jobless Claims (FRED:ICSA)",
            "BLS CPI-U (FRED:CPIAUCSL)",
        ],
        "time_seconds": round(total_time, 1),
    }

    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Results saved to: {output_path}")

    print(f"\n{'═' * 80}")
    if dir_accuracy >= 0.8:
        grade = "STRONG"
    elif dir_accuracy >= 0.65:
        grade = "MODERATE"
    else:
        grade = "WEAK"
    print(f"VERDICT: {grade} directional match against real macro-economic data")
    print(f"  Direction: {dir_accuracy * 100:.0f}% of metrics moved the correct way")
    print(f"  Magnitude: Our sim produces {avg_mag_ratio:.1f}x the real-world magnitude on average")
    print(f"  Sector ranking: ρ={avg_rank_corr:+.2f} correlation with BLS employment data")
    print(f"{'═' * 80}")

    return report


if __name__ == "__main__":
    run_validation()

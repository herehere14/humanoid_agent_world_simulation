#!/usr/bin/env python3
"""Integration test: inject shocks and verify macro metrics, info propagation, and outcomes.

This script validates the full pipeline:
  1. Build a town with 300 agents
  2. Run baseline ticks (no shock) and record macro baseline
  3. Inject an oil price surge
  4. Run post-shock ticks and verify:
     - Macro metrics change meaningfully
     - Information propagates through the population
     - Different sectors are impacted differently
     - Shock impact report shows pre/post deltas
  5. Inject a brand scandal and verify second shock compounds
  6. Print a human-readable summary of what happened

Usage:
    cd openclaw_closedsourcemodel_RL
    python -m world_sim.test_macro_integration
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

# Ensure the module can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from world_sim.scenarios_heatwave_harbor import build_heatwave_harbor
from world_sim.dynamic_events import DynamicEventEngine


def run_test():
    print("=" * 70)
    print("MACRO INTEGRATION TEST")
    print("=" * 70)

    # Build world
    print("\n[1] Building heatwave_harbor scenario (300 agents)...")
    t0 = time.time()
    world, agent_meta = build_heatwave_harbor(n_agents=300, seed=84)
    print(f"    Built in {time.time() - t0:.1f}s")

    print("[2] Initializing world (loading SBERT model)...")
    t0 = time.time()
    world.initialize()
    event_engine = DynamicEventEngine()
    print(f"    Initialized in {time.time() - t0:.1f}s")

    # Phase 1: Baseline (2 days = 48 ticks)
    print("\n[3] Running baseline (48 ticks / 2 days)...")
    t0 = time.time()
    for _ in range(48):
        summary = world.tick()
        ripples = event_engine.generate(world, summary, agent_meta)
        for r in ripples:
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)
    baseline_time = time.time() - t0
    print(f"    Done in {baseline_time:.1f}s")

    # Record baseline macro metrics
    baseline_macro = world.get_macro_summary()
    baseline_current = baseline_macro.get("current", {})
    print(f"\n    BASELINE MACRO METRICS:")
    print(f"    Consumer Confidence:  {baseline_current.get('consumer_confidence', 0):.3f}")
    print(f"    Social Cohesion:      {baseline_current.get('social_cohesion', 0):.3f}")
    print(f"    Institutional Trust:  {baseline_current.get('institutional_trust', 0):.3f}")
    print(f"    Civil Unrest:         {baseline_current.get('civil_unrest_potential', 0):.3f}")
    print(f"    Market Pressure:      {baseline_current.get('market_pressure', 0):.3f}")
    print(f"    Population Mood:      {baseline_current.get('population_mood', 0):+.3f}")

    # Phase 2: Inject oil price surge
    print("\n[4] Injecting shock: 'Oil prices surge 100%'...")
    result = world.ingest_information("Oil prices surge 100%")
    print(f"    Impacted agents: {result['impacted_agents']}")
    print(f"    Scheduled events: {result['scheduled_event_count']}")

    # Run 72 more ticks (3 days post-shock)
    print("\n[5] Running post-shock simulation (72 ticks / 3 days)...")
    t0 = time.time()
    for _ in range(72):
        summary = world.tick()
        ripples = event_engine.generate(world, summary, agent_meta)
        for r in ripples:
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)
    post_shock1_time = time.time() - t0
    print(f"    Done in {post_shock1_time:.1f}s")

    # Check macro metrics after oil shock
    post_oil_macro = world.get_macro_summary()
    post_oil_current = post_oil_macro.get("current", {})
    post_oil_deltas = post_oil_macro.get("deltas", {})
    print(f"\n    POST-OIL-SHOCK MACRO METRICS:")
    print(f"    Consumer Confidence:  {post_oil_current.get('consumer_confidence', 0):.3f} (delta: {post_oil_deltas.get('consumer_confidence', 0):+.4f})")
    print(f"    Social Cohesion:      {post_oil_current.get('social_cohesion', 0):.3f} (delta: {post_oil_deltas.get('social_cohesion', 0):+.4f})")
    print(f"    Institutional Trust:  {post_oil_current.get('institutional_trust', 0):.3f} (delta: {post_oil_deltas.get('institutional_trust', 0):+.4f})")
    print(f"    Civil Unrest:         {post_oil_current.get('civil_unrest_potential', 0):.3f} (delta: {post_oil_deltas.get('civil_unrest_potential', 0):+.4f})")
    print(f"    Market Pressure:      {post_oil_current.get('market_pressure', 0):.3f} (delta: {post_oil_deltas.get('market_pressure', 0):+.4f})")
    print(f"    Population Mood:      {post_oil_current.get('population_mood', 0):+.3f} (delta: {post_oil_deltas.get('population_mood', 0):+.4f})")

    # Information spread check
    info_report = world.get_info_spread_report()
    print(f"\n    INFORMATION SPREAD:")
    for item in info_report.get("active_information", []):
        print(f"    - {item['label']}: {item['pct_aware'] * 100:.1f}% aware ({item['total_aware']}/{info_report['total_agents']})")
        print(f"      Source breakdown: {item['source_breakdown']}")

    # Sector-level impact
    sectors = post_oil_current.get("sectors", {})
    print(f"\n    SECTOR EMPLOYMENT STRESS:")
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1].get("employment_stress", 0), reverse=True)
    for sector_name, sector_data in sorted_sectors:
        print(f"    - {sector_name}: stress={sector_data.get('employment_stress', 0):.3f}, "
              f"valence={sector_data.get('avg_valence', 0):.3f}, "
              f"tension={sector_data.get('avg_tension', 0):.3f}")

    # Phase 3: Inject brand scandal
    print("\n[6] Injecting second shock: 'Major chocolate company exposed using child labor in supply chain'...")
    result2 = world.ingest_information("Major chocolate company exposed using child labor in supply chain")
    print(f"    Impacted agents: {result2['impacted_agents']}")
    print(f"    Scheduled events: {result2['scheduled_event_count']}")

    # Run 48 more ticks (2 days)
    print("\n[7] Running post-scandal simulation (48 ticks / 2 days)...")
    t0 = time.time()
    for _ in range(48):
        summary = world.tick()
        ripples = event_engine.generate(world, summary, agent_meta)
        for r in ripples:
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)
    post_shock2_time = time.time() - t0
    print(f"    Done in {post_shock2_time:.1f}s")

    # Final macro metrics
    final_macro = world.get_macro_summary()
    final_current = final_macro.get("current", {})
    final_deltas = final_macro.get("deltas", {})
    print(f"\n    FINAL MACRO METRICS (after both shocks):")
    print(f"    Consumer Confidence:  {final_current.get('consumer_confidence', 0):.3f} (delta from baseline: {final_deltas.get('consumer_confidence', 0):+.4f})")
    print(f"    Social Cohesion:      {final_current.get('social_cohesion', 0):.3f} (delta: {final_deltas.get('social_cohesion', 0):+.4f})")
    print(f"    Institutional Trust:  {final_current.get('institutional_trust', 0):.3f} (delta: {final_deltas.get('institutional_trust', 0):+.4f})")
    print(f"    Civil Unrest:         {final_current.get('civil_unrest_potential', 0):.3f} (delta: {final_deltas.get('civil_unrest_potential', 0):+.4f})")
    print(f"    Market Pressure:      {final_current.get('market_pressure', 0):.3f} (delta: {final_deltas.get('market_pressure', 0):+.4f})")
    print(f"    Population Mood:      {final_current.get('population_mood', 0):+.3f} (delta: {final_deltas.get('population_mood', 0):+.4f})")

    # Full info spread report
    final_info = world.get_info_spread_report()
    print(f"\n    FINAL INFORMATION SPREAD:")
    for item in final_info.get("active_information", []):
        print(f"    - {item['label']}: {item['pct_aware'] * 100:.1f}% aware")
        print(f"      Sources: {item['source_breakdown']}")
        print(f"      Avg distortion: {item['avg_distortion']:.3f}")

    # Shock impact report
    shock_report = world.get_shock_impact_report()
    print(f"\n    SHOCK IMPACT ANALYSIS:")
    print(f"    Shock onset: {shock_report.get('shock_onset_time', 'unknown')}")
    if "impact" in shock_report:
        for metric, vals in shock_report["impact"].items():
            print(f"    - {metric}: {vals['pre']:.3f} → {vals['post']:.3f} ({vals['pct_change']:+.1f}%)")

    # Faction analysis
    factions = final_current.get("factions", {})
    if factions:
        print(f"\n    FACTION POWER BALANCE:")
        for fname, fdata in sorted(factions.items(), key=lambda x: x[1].get("power_index", 0), reverse=True):
            print(f"    - {fname}: power={fdata.get('power_index', 0):.3f}, "
                  f"cohesion={fdata.get('cohesion', 0):.3f}, "
                  f"members={fdata.get('member_count', 0)}, "
                  f"concern='{fdata.get('dominant_concern', '')}'")

    # Validation checks
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    checks_passed = 0
    checks_total = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    # Consumer confidence should drop after shocks
    cc_delta = final_deltas.get("consumer_confidence", 0)
    check("Consumer confidence dropped", cc_delta < -0.001,
          f"delta={cc_delta:+.4f}")

    # Market pressure should increase
    mp_delta = final_deltas.get("market_pressure", 0)
    check("Market pressure increased", mp_delta > 0.001,
          f"delta={mp_delta:+.4f}")

    # Civil unrest should increase (not necessarily a lot, but some)
    cu_delta = final_deltas.get("civil_unrest_potential", 0)
    check("Civil unrest potential changed", cu_delta != 0.0,
          f"delta={cu_delta:+.4f}")

    # Population mood should be lower
    pm_delta = final_deltas.get("population_mood", 0)
    check("Population mood decreased", pm_delta < 0,
          f"delta={pm_delta:+.4f}")

    # Information should have spread beyond directly impacted
    for item in final_info.get("active_information", []):
        check(f"Info '{item['label']}' spread to >50% of population",
              item["pct_aware"] > 0.5,
              f"{item['pct_aware'] * 100:.1f}%")

    # Sectors should have different stress levels (not all identical)
    if sorted_sectors:
        stress_values = [s[1].get("employment_stress", 0) for s in sorted_sectors]
        stress_range = max(stress_values) - min(stress_values) if len(stress_values) > 1 else 0
        check("Sectors have differential stress", stress_range > 0.01,
              f"range={stress_range:.4f}")

    # Macro timeline should exist and have entries
    timeline = world.get_macro_timeline()
    check("Macro timeline has data", len(timeline) > 100,
          f"entries={len(timeline)}")

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")

    # Save results
    output_path = Path(__file__).parent.parent / "artifacts" / "macro_integration_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_result = {
        "baseline_macro": baseline_current,
        "post_oil_macro": post_oil_current,
        "post_oil_deltas": post_oil_deltas,
        "final_macro": final_current,
        "final_deltas": final_deltas,
        "info_spread": final_info,
        "shock_impact": shock_report,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "timing": {
            "baseline_s": round(baseline_time, 1),
            "post_shock1_s": round(post_shock1_time, 1),
            "post_shock2_s": round(post_shock2_time, 1),
        },
    }
    output_path.write_text(json.dumps(test_result, indent=2, default=str))
    print(f"\n  Full results saved to: {output_path}")

    print("\n" + "=" * 70)
    if checks_passed == checks_total:
        print("ALL CHECKS PASSED — macro integration is working.")
    else:
        print(f"WARNING: {checks_total - checks_passed} check(s) failed.")
    print("=" * 70)

    return checks_passed == checks_total


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

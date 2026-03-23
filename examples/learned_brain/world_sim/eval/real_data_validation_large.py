#!/usr/bin/env python3
"""Large-scale real data validation: 1000 agents, 30 days, 100 LLM agents.

Runs each historical event for a full month with 1000 agents.
~100 agents get real GPT-5-mini LLM calls when promoted to high salience.
Compares macro outcomes against actual FRED/BLS/Conference Board data.

The longer timespan should capture peak impact that our 5-day sim missed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_sim.scenarios_heatwave_harbor import build_heatwave_harbor
from world_sim.dynamic_events import DynamicEventEngine
from world_sim.llm_chooser import LLMChooser


# Same real data as real_data_validation.py — importing it
from world_sim.eval.real_data_validation import (
    ALL_EVENTS,
    HistoricalRealData,
    RealDataPoint,
    _aggregate_real_sector_ranking,
    _spearman_correlation,
)


def run_single_event(
    event: HistoricalRealData,
    api_key: str,
    model: str = "gpt-5-mini",
    n_agents: int = 1000,
    days: int = 30,
    shock_day: int = 3,
    llm_max_per_tick: int = 8,
) -> dict:
    """Run one historical event for a full month with LLM agents."""
    total_ticks = days * 24
    shock_tick = shock_day * 24

    print(f"\n  Building world ({n_agents} agents)...")
    t0 = time.time()
    world, agent_meta = build_heatwave_harbor(n_agents=n_agents, seed=42)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")

    print(f"  Initializing SBERT...")
    t0 = time.time()
    world.initialize()
    init_time = time.time() - t0
    print(f"  Initialized in {init_time:.1f}s")

    event_engine = DynamicEventEngine()

    # Set up LLM chooser
    chooser = LLMChooser(
        api_key=api_key,
        model=model,
        max_per_tick=llm_max_per_tick,
        enabled=True,
    )

    # Tracking
    daily_macro: list[dict] = []
    peak_tracking = {
        "min_confidence": 1.0,
        "min_confidence_tick": 0,
        "max_market_pressure": 0.0,
        "max_market_pressure_tick": 0,
        "min_mood": 1.0,
        "min_mood_tick": 0,
        "max_civil_unrest": 0.0,
        "max_civil_unrest_tick": 0,
    }

    print(f"  Running {total_ticks} ticks ({days} days)...", flush=True)
    print(f"  Shock injects at tick {shock_tick} (day {shock_day})...", flush=True)
    t0 = time.time()

    llm_decisions_total = 0

    for tick in range(1, total_ticks + 1):
        # Inject shock at designated tick
        if tick == shock_tick:
            print(f"    → SHOCK INJECTED at tick {tick}: \"{event.shock_text}\"")
            result = world.ingest_information(event.shock_text)
            print(f"      Impacted: {result['impacted_agents']}, Avg severity: {result.get('avg_personal_severity', 'n/a')}")

        summary = world.tick()

        # Dynamic events
        ripples = event_engine.generate(world, summary, agent_meta)
        for r in ripples:
            r.description = f"[{r.kind}] {r.description}"
            world.schedule_event(r)

        # LLM chooser — only during high-impact windows, not every tick
        # Call LLM during: first 48h after shock, event ticks, and every 6h thereafter
        should_call_llm = (
            chooser.enabled and
            tick >= shock_tick and
            (
                (tick - shock_tick) <= 48 or  # first 2 days after shock: every tick
                len(summary.get("events", [])) > 0 or  # event ticks
                tick % 6 == 0  # every 6 hours otherwise
            )
        )
        if should_call_llm:
            from world_sim.action_table import Action
            actions_map = {}
            for aid, agent in world.agents.items():
                try:
                    actions_map[aid] = Action[agent.last_action]
                except (KeyError, ValueError):
                    actions_map[aid] = Action.IDLE
            decisions = chooser.execute_llm_decisions(world, actions_map, summary)
            llm_decisions_total += len(decisions)

        # Track daily macro at end of each day
        if world.hour_of_day == 23:
            macro = world.get_macro_summary()
            current = macro.get("current", {})

            cc = current.get("consumer_confidence", 1.0)
            mp = current.get("market_pressure", 0.0)
            mood = current.get("population_mood", 0.0)
            cu = current.get("civil_unrest_potential", 0.0)

            daily_macro.append({
                "day": world.day,
                "tick": tick,
                "consumer_confidence": round(cc, 4),
                "market_pressure": round(mp, 4),
                "population_mood": round(mood, 4),
                "civil_unrest_potential": round(cu, 4),
                "institutional_trust": round(current.get("institutional_trust", 0.5), 4),
                "social_cohesion": round(current.get("social_cohesion", 0.5), 4),
            })

            # Track peaks
            if cc < peak_tracking["min_confidence"]:
                peak_tracking["min_confidence"] = cc
                peak_tracking["min_confidence_tick"] = tick
            if mp > peak_tracking["max_market_pressure"]:
                peak_tracking["max_market_pressure"] = mp
                peak_tracking["max_market_pressure_tick"] = tick
            if mood < peak_tracking["min_mood"]:
                peak_tracking["min_mood"] = mood
                peak_tracking["min_mood_tick"] = tick
            if cu > peak_tracking["max_civil_unrest"]:
                peak_tracking["max_civil_unrest"] = cu
                peak_tracking["max_civil_unrest_tick"] = tick

            # Progress
            if world.day % 5 == 0 or world.day == shock_day or world.day == shock_day + 1:
                elapsed = time.time() - t0
                print(f"    Day {world.day:>2d}: cc={cc:.3f}  mp={mp:.3f}  mood={mood:+.3f}  unrest={cu:.3f}  "
                      f"LLM calls so far: {chooser.stats.total_calls}  [{elapsed:.0f}s]", flush=True)

    sim_time = time.time() - t0
    print(f"  Completed in {sim_time:.1f}s ({sim_time / days:.1f}s/day)")

    # Get final state
    final_macro = world.get_macro_summary()
    final_current = final_macro.get("current", {})
    deltas = final_macro.get("deltas", {})

    # Baseline = average of first shock_day days
    baseline_days = [d for d in daily_macro if d["day"] <= shock_day]
    baseline_cc = mean(d["consumer_confidence"] for d in baseline_days) if baseline_days else 0.7

    # Peak impact = worst value after shock
    post_shock_days = [d for d in daily_macro if d["day"] > shock_day]

    # Compute actual % changes using peak impact (not just final)
    peak_cc_drop_pct = (peak_tracking["min_confidence"] - baseline_cc) / baseline_cc * 100 if baseline_cc else 0

    # Sector stress
    sectors = final_current.get("sectors", {})
    baseline_sectors_data = daily_macro[shock_day - 1] if len(daily_macro) >= shock_day else {}

    sector_stress = {k: v.get("employment_stress", 0) for k, v in sectors.items()}
    sector_ranking = sorted(sector_stress.items(), key=lambda x: x[1], reverse=True)

    return {
        "event_name": event.name,
        "n_agents": n_agents,
        "days": days,
        "shock_day": shock_day,
        "sim_time_s": round(sim_time, 1),
        "llm_stats": chooser.get_stats(),
        "llm_decisions_total": llm_decisions_total,
        "baseline_cc": round(baseline_cc, 4),
        "peak_tracking": {k: round(v, 4) if isinstance(v, float) else v for k, v in peak_tracking.items()},
        "peak_cc_drop_pct": round(peak_cc_drop_pct, 2),
        "final_deltas": deltas,
        "final_macro": {k: round(v, 4) if isinstance(v, float) else v for k, v in final_current.items() if k not in ("sectors", "factions", "action_distribution", "emotion_distribution", "top_concerns")},
        "sector_ranking": [(k, round(v, 4)) for k, v in sector_ranking],
        "daily_macro": daily_macro,
        "info_spread": world.get_info_spread_report(),
    }


def compare_against_real(sim_result: dict, event: HistoricalRealData) -> dict:
    """Compare simulation peak impacts against real data."""
    comparisons = []

    metric_mapping = {
        "Consumer Confidence Index": ("consumer_confidence", False),
        "Consumer Confidence Index (to trough)": ("consumer_confidence", False),
        "Consumer Sentiment (Michigan)": ("consumer_confidence", False),
        "S&P 500": ("market_pressure", True),
        "S&P 500 (1 month)": ("market_pressure", True),
        "S&P 500 (to trough)": ("market_pressure", True),
        "Unemployment Rate": ("market_pressure", False),
        "Trust in Banks": ("institutional_trust", False),
        "Initial Jobless Claims (weekly)": ("market_pressure", False),
    }

    peak = sim_result["peak_tracking"]
    baseline_cc = sim_result["baseline_cc"]

    for dp in event.macro_data:
        mapping = metric_mapping.get(dp.metric_name)
        if mapping is None:
            continue
        sim_metric, inverted = mapping

        real_pct = dp.pct_change

        # Use PEAK impact for comparison, not final state
        if sim_metric == "consumer_confidence":
            sim_pct = sim_result["peak_cc_drop_pct"]
        elif sim_metric == "market_pressure":
            # peak market pressure as % above baseline
            baseline_mp = sim_result["daily_macro"][sim_result["shock_day"] - 1]["market_pressure"] if len(sim_result["daily_macro"]) >= sim_result["shock_day"] else 0.2
            if baseline_mp > 0.01:
                sim_pct = (peak["max_market_pressure"] - baseline_mp) / baseline_mp * 100
            else:
                sim_pct = peak["max_market_pressure"] * 100
        elif sim_metric == "institutional_trust":
            sim_pct = sim_result["final_deltas"].get("institutional_trust", 0) * 100
        else:
            sim_pct = sim_result["final_deltas"].get(sim_metric, 0) * 100

        # Direction check
        real_dir = "down" if real_pct < -0.5 else "up" if real_pct > 0.5 else "flat"
        if inverted:
            sim_dir = "up" if sim_pct > 0.5 else "down" if sim_pct < -0.5 else "flat"
            expected_dir = "up" if real_dir == "down" else "down" if real_dir == "up" else "flat"
            dir_match = sim_dir == expected_dir
            mag_ratio = abs(sim_pct) / max(0.1, abs(real_pct))
        else:
            sim_dir = "down" if sim_pct < -0.5 else "up" if sim_pct > 0.5 else "flat"
            dir_match = sim_dir == real_dir
            mag_ratio = abs(sim_pct) / max(0.1, abs(real_pct))

        comparisons.append({
            "real_metric": dp.metric_name,
            "real_source": dp.source,
            "real_period": f"{dp.pre_date} → {dp.post_date}",
            "real_pct": round(real_pct, 1),
            "sim_pct": round(sim_pct, 1),
            "direction_match": dir_match,
            "magnitude_ratio": round(mag_ratio, 3),
            "using_peak": sim_metric == "consumer_confidence",
        })

    # Sector comparison
    sector_comps = []
    for sd in event.sector_data:
        real_ranking = _aggregate_real_sector_ranking(sd)
        sim_ranking = [s[0] for s in sim_result["sector_ranking"]]
        common = [s for s in real_ranking if s in sim_ranking]
        real_f = [s for s in real_ranking if s in common]
        sim_f = [s for s in sim_ranking if s in common]
        corr = _spearman_correlation(real_f, sim_f)
        sector_comps.append({
            "real_ranking": real_f,
            "sim_ranking": sim_f,
            "correlation": round(corr, 3),
            "top_match": real_f[0] == sim_f[0] if real_f and sim_f else False,
        })

    return {"metric_comparisons": comparisons, "sector_comparisons": sector_comps}


def run_validation(api_key: str, model: str = "gpt-5-mini"):
    print("=" * 80)
    print("LARGE-SCALE REAL DATA VALIDATION")
    print(f"1000 agents | 30 days | ~100 LLM agents | {model}")
    print("=" * 80)

    t0_total = time.time()
    all_results = []

    for event in ALL_EVENTS:
        print(f"\n{'━' * 75}")
        print(f"EVENT: {event.name}")
        print(f"Date:  {event.date}")
        print(f"{'━' * 75}")

        sim_result = run_single_event(
            event, api_key=api_key, model=model,
            n_agents=1000, days=30, shock_day=3,
            llm_max_per_tick=3,
        )

        comparison = compare_against_real(sim_result, event)

        # Print results
        print(f"\n  SIMULATION vs REAL DATA (using PEAK impact):")
        print(f"  {'Real Metric':<38s} {'Real':>8s} {'Sim':>8s} {'Dir':>5s} {'Mag':>6s}")
        print(f"  {'─' * 70}")

        for c in comparison["metric_comparisons"]:
            mark = "✓" if c["direction_match"] else "✗"
            print(f"  {c['real_metric']:<38s} {c['real_pct']:>+7.1f}% {c['sim_pct']:>+7.1f}%   {mark}  {c['magnitude_ratio']:>5.2f}x")

        if comparison["sector_comparisons"]:
            for sc in comparison["sector_comparisons"]:
                print(f"\n  Sector ranking: real={sc['real_ranking']}")
                print(f"                  sim ={sc['sim_ranking']}")
                print(f"  Spearman ρ={sc['correlation']:+.3f}, top match={'✓' if sc['top_match'] else '✗'}")

        # Print peak tracking
        peak = sim_result["peak_tracking"]
        print(f"\n  Peak impacts:")
        print(f"    Min consumer confidence: {peak['min_confidence']:.3f} (day {peak['min_confidence_tick'] // 24 + 1})")
        print(f"    Max market pressure:     {peak['max_market_pressure']:.3f} (day {peak['max_market_pressure_tick'] // 24 + 1})")
        print(f"    Min population mood:     {peak['min_mood']:+.3f} (day {peak['min_mood_tick'] // 24 + 1})")
        print(f"    Peak CC drop from base:  {sim_result['peak_cc_drop_pct']:+.1f}%")

        # LLM stats
        llm = sim_result["llm_stats"]
        print(f"\n  LLM stats: {llm['total_calls']} calls, {llm['successful_calls']} succeeded, "
              f"{llm['actions_overridden']} actions overridden, avg {llm['avg_latency_ms']}ms")

        # Daily trajectory
        print(f"\n  Daily confidence trajectory:")
        for d in sim_result["daily_macro"]:
            bar_len = int(d["consumer_confidence"] * 40)
            shock_mark = " ← SHOCK" if d["day"] == 3 else ""
            print(f"    Day {d['day']:>2d}: {'█' * bar_len}{'░' * (40 - bar_len)} {d['consumer_confidence']:.3f}{shock_mark}")

        all_results.append({
            "sim_result": sim_result,
            "comparison": comparison,
        })

    total_time = time.time() - t0_total

    # Aggregate
    print(f"\n{'═' * 80}")
    print("AGGREGATE RESULTS")
    print(f"{'═' * 80}")

    all_comps = []
    all_mag = []
    all_corr = []
    dir_ok = 0
    dir_total = 0

    for r in all_results:
        for c in r["comparison"]["metric_comparisons"]:
            all_comps.append(c)
            dir_total += 1
            if c["direction_match"]:
                dir_ok += 1
            all_mag.append(c["magnitude_ratio"])
        for sc in r["comparison"]["sector_comparisons"]:
            all_corr.append(sc["correlation"])

    dir_acc = dir_ok / dir_total if dir_total else 0
    avg_mag = mean(all_mag) if all_mag else 0
    median_mag = sorted(all_mag)[len(all_mag) // 2] if all_mag else 0
    avg_corr = mean(all_corr) if all_corr else 0

    print(f"\n  Direction Accuracy:      {dir_ok}/{dir_total} ({dir_acc * 100:.1f}%)")
    print(f"  Magnitude Ratio:         avg={avg_mag:.3f}x  median={median_mag:.3f}x")
    print(f"  Sector Rank Correlation: ρ={avg_corr:+.3f}")

    print(f"\n  Magnitude distribution:")
    buckets = {"<0.1x (severe under)": 0, "0.1-0.3x (under)": 0,
               "0.3-0.7x (moderate)": 0, "0.7-1.5x (good)": 0, "1.5-3x (over)": 0, ">3x (severe over)": 0}
    for r in all_mag:
        if r < 0.1: buckets["<0.1x (severe under)"] += 1
        elif r < 0.3: buckets["0.1-0.3x (under)"] += 1
        elif r < 0.7: buckets["0.3-0.7x (moderate)"] += 1
        elif r <= 1.5: buckets["0.7-1.5x (good)"] += 1
        elif r <= 3.0: buckets["1.5-3x (over)"] += 1
        else: buckets[">3x (severe over)"] += 1
    for bucket, count in buckets.items():
        print(f"    {bucket:<25s} {count:>2d} {'█' * count}")

    total_llm = sum(r["sim_result"]["llm_stats"]["total_calls"] for r in all_results)
    print(f"\n  Total LLM calls: {total_llm}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    # Save
    output_path = Path(__file__).parent.parent.parent.parent / "artifacts" / "real_data_validation_large.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {"n_agents": 1000, "days": 30, "model": model},
        "direction_accuracy": round(dir_acc, 3),
        "avg_magnitude_ratio": round(avg_mag, 3),
        "median_magnitude_ratio": round(median_mag, 3),
        "avg_sector_correlation": round(avg_corr, 3),
        "total_llm_calls": total_llm,
        "total_time_s": round(total_time, 1),
        "events": [
            {
                "name": ALL_EVENTS[i].name,
                "comparisons": r["comparison"],
                "peak_tracking": r["sim_result"]["peak_tracking"],
                "peak_cc_drop_pct": r["sim_result"]["peak_cc_drop_pct"],
                "llm_stats": r["sim_result"]["llm_stats"],
                "daily_macro": r["sim_result"]["daily_macro"],
            }
            for i, r in enumerate(all_results)
        ],
    }
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved to: {output_path}")

    print(f"\n{'═' * 80}")
    if dir_acc >= 0.8 and avg_mag >= 0.3:
        grade = "STRONG"
    elif dir_acc >= 0.65:
        grade = "MODERATE"
    else:
        grade = "WEAK"
    print(f"VERDICT: {grade} match against real macro-economic data")
    print(f"  Direction: {dir_acc * 100:.0f}%  |  Magnitude: {avg_mag:.2f}x  |  Sector ρ: {avg_corr:+.2f}")
    print(f"  1000 agents, 30 days, {total_llm} LLM calls, {total_time / 60:.1f} minutes")
    print(f"{'═' * 80}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()
    run_validation(api_key=args.api_key, model=args.model)

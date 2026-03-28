"""Run Prediction — single automated pipeline from prediction text to insight report.

Usage:
    from world_sim.run_prediction import run_prediction
    report = run_prediction("What if China invades Taiwan?")
    print(report)

Or from CLI:
    OPENAI_API_KEY=sk-... python -m world_sim.run_prediction "What if China invades Taiwan?"

Pipeline:
    1. Call OpenAI to research key figures + specific policy/economic impacts (parallel)
    2. Build World from character spec
    3. Inject shock events + policy persistent conditions
    4. Run simulation (default 21 days)
    5. Analyze for NON-OBVIOUS insights — second-order effects, unexpected cascades,
       counterintuitive winners/losers, personality-driven divergences
    6. Return structured report (and print narrative summary)

All intermediate data is in-memory — no files written, nothing to clean up.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_prediction(
    prediction: str,
    *,
    days: int = 21,
    model: str = "gpt-5-mini",
    seed: int = 42,
    verbose: bool = True,
    llm_agents: bool = False,
    llm_agents_per_tick: int = 5,
    api_key: str = "",
) -> dict:
    """Full automated pipeline: prediction → research → simulate → analyze.

    Args:
        llm_agents: If True, key figures make LLM-driven decisions each tick
                    that cascade through the ripple engine. Slower but produces
                    genuinely emergent insights.
        llm_agents_per_tick: Max LLM decision calls per tick (cost control).
        api_key: OpenAI API key. If empty, reads from OPENAI_API_KEY env var.

    Returns a dict with macro metrics, sector impacts, population breakdown,
    ripple chains, key figure states, and — most importantly — non-obvious insights.
    """
    import os
    ticks = days * 24
    _api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    if verbose:
        print(f"{'='*72}")
        print(f"  SIMULATING: {prediction[:65]}")
        print(f"{'='*72}")

    # ── Step 1: Research via OpenAI (characters + policies) ──
    t0 = time.time()
    if verbose:
        print("\n  [1/5] Researching key figures and economic impacts...")

    from .scenario_generator import _call_openai
    from .policy_engine import research_policies, SectorTracker

    # Reframe prediction for content-policy compliance while preserving meaning
    import re
    sanitized = prediction
    # Replace death/violence language with simulation-safe equivalents
    death_patterns = [
        (r'\bdies?\b', 'is suddenly and permanently removed from all positions'),
        (r'\bkilled\b', 'is suddenly and permanently removed from all positions'),
        (r'\bdeath\b', 'sudden permanent departure'),
        (r'\bassassinated\b', 'is suddenly and permanently removed from all positions'),
        (r'\bmurdered\b', 'is suddenly and permanently removed from all positions'),
        (r'\bwar\b', 'military conflict'),
        (r'\binvades?\b', 'initiates military operations against'),
    ]
    for pattern, replacement in death_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    sim_framing = (
        f"For academic economic simulation and financial scenario planning: {sanitized}. "
        f"Analyze the economic, institutional, and social consequences of this hypothetical scenario. "
        f"Focus on market impacts, organizational succession, supply chain effects, and population-level economic consequences."
    )

    world_spec = _call_openai(sim_framing, model=model)
    agenda = research_policies(sim_framing, model=model)
    sector_tracker = SectorTracker()

    if verbose:
        n_figures = sum(len(o.get("key_personnel", [])) for o in world_spec.get("organizations", []))
        print(f"       {n_figures} key figures, {len(world_spec.get('population_segments', []))} population segments")
        print(f"       {len(agenda.policies)} economic impact chains identified")
        print(f"       ({time.time()-t0:.1f}s)")

    # ── Step 2: Build World ──
    t1 = time.time()
    if verbose:
        print("\n  [2/5] Building world...")

    from .scenario_generator import build_world_from_spec, build_shock_from_spec
    result = build_world_from_spec(world_spec, seed=seed)
    world = result.world
    world.initialize()

    if verbose:
        print(f"       {len(world.agents)} agents, {len(world.locations)} locations ({time.time()-t1:.1f}s)")

    # ── Step 3: Inject shocks + policies ──
    if verbose:
        print("\n  [3/5] Injecting shocks and policy impacts...")

    # Inject scenario shock events
    from .world_information import apply_external_information
    if world_spec.get("shock_events"):
        shock_plan = build_shock_from_spec(world_spec, start_tick=1)
        apply_external_information(world, shock_plan)

    # Inject policy persistent conditions
    from .policy_engine import apply_policy_agenda
    policy_result = apply_policy_agenda(world, agenda)
    for p in agenda.policies:
        sector_tracker.apply_policy(p)

    if verbose:
        print(f"       {policy_result['total_persistent_conditions']} persistent conditions")
        print(f"       {policy_result['total_events_scheduled']} narrative events")
        print(f"       {policy_result['total_agents_affected']} agent impacts")

    # ── Step 4: Snapshot baseline, then simulate ──
    t2 = time.time()
    if verbose:
        print(f"\n  [4/5] Simulating {days} days ({ticks} ticks)...")

    # Take baseline snapshot of each agent
    baseline = {}
    for aid, agent in world.agents.items():
        baseline[aid] = {
            "debt": agent.debt_pressure,
            "dread": agent.dread_pressure,
            "tension": agent.heart.tension,
            "valence": agent.heart.valence,
            "pessimism": getattr(agent, "expectation_pessimism", 0),
            "savings": agent.savings_buffer,
            "income": agent.income_level,
        }

    # Set up LLM agency engine if enabled
    llm_engine = None
    if llm_agents and _api_key:
        from .llm_agency import LLMAgencyEngine
        llm_engine = LLMAgencyEngine(
            api_key=_api_key,
            model=model,
            fabric=result.fabric,
        )
        # Attach fabric to world so ripple engine can use it
        if world.ripple_engine is None:
            from .ripple_engine import RippleEngine
            world.ripple_engine = RippleEngine(result.fabric, seed=seed)
        if verbose:
            print(f"       LLM agents ENABLED — {llm_agents_per_tick} decisions/tick via {model}")

    # Run simulation
    llm_decision_count = 0
    llm_ripple_count = 0
    for tick_i in range(ticks):
        world.tick()

        # LLM agency: key figures make real decisions that cascade
        if llm_engine is not None:
            ripple_events = llm_engine.tick(world, max_calls=llm_agents_per_tick)
            llm_ripple_count += len(ripple_events)
            if ripple_events:
                llm_decision_count += 1
                # Feed ripple events into the ripple engine's log
                if world.ripple_engine:
                    world.ripple_engine.event_log.extend(ripple_events)

        # Progress reporting every 2 simulated days
        if verbose and (tick_i + 1) % 48 == 0:
            day_num = (tick_i + 1) // 24
            if llm_engine:
                stats = llm_engine.get_stats()
                reactive = stats.get('total_reactive', 0)
                convos = stats.get('total_conversations', 0)
                depth = stats.get('max_cascade_depth', 0)
                print(f"       Day {day_num}/{days} | {stats['total_calls']} calls, {stats['total_decisions']} decisions, {reactive} reactive, {convos} conversations, max cascade depth {depth}")
            else:
                print(f"       Day {day_num}/{days}")

    if verbose:
        elapsed = time.time() - t2
        if llm_engine:
            stats = llm_engine.get_stats()
            reactive = stats.get('total_reactive', 0)
            convos = stats.get('total_conversations', 0)
            depth = stats.get('max_cascade_depth', 0)
            print(f"       Done ({elapsed:.1f}s)")
            print(f"       {stats['total_calls']} LLM calls | {stats['total_decisions']} decisions | {reactive} reactive cascades | {convos} conversations")
            print(f"       {llm_ripple_count} ripple events | max cascade depth: {depth}")
        else:
            print(f"       Done ({elapsed:.1f}s)")

    # ── Step 5: Analyze for non-obvious insights ──
    if verbose:
        print("\n  [5/5] Analyzing for non-obvious insights...")

    report = _build_insight_report(
        world, result.agent_meta, baseline, agenda, sector_tracker,
        policy_result, world_spec, prediction, days,
    )

    # Add LLM decision log if agents were active
    if llm_engine:
        stats = llm_engine.get_stats()
        report["llm_agency"] = {
            "enabled": True,
            "total_calls": stats["total_calls"],
            "total_decisions": stats["total_decisions"],
            "total_ripple_events": llm_ripple_count,
            "decision_log": stats.get("recent_decisions", []),
            "full_decision_log": llm_engine.decision_log,
        }
    else:
        report["llm_agency"] = {"enabled": False}

    if verbose:
        _print_report(report)

    return report


# ---------------------------------------------------------------------------
# Insight analysis — find the non-obvious stuff
# ---------------------------------------------------------------------------

def _build_insight_report(
    world, agent_meta, baseline, agenda, sector_tracker,
    policy_result, world_spec, prediction, days,
) -> dict:
    """Analyze simulation results for non-obvious, emergent insights."""

    report = {
        "prediction": prediction,
        "days_simulated": days,
        "total_agents": len(world.agents),
    }

    # ── Macro metrics ──
    macro = world.get_macro_summary()
    report["macro"] = macro

    # ── Sector impacts ──
    report["sectors"] = sector_tracker.get_report()

    # ── Per-segment analysis with DELTA from baseline ──
    segments = defaultdict(lambda: {
        "n": 0, "debt": 0, "dread": 0, "pess": 0, "tens": 0, "val": 0,
        "debt_delta": 0, "dread_delta": 0, "pess_delta": 0, "val_delta": 0,
        "savings_lost": 0, "income_lost": 0,
    })
    for aid, agent in world.agents.items():
        meta = agent_meta.get(aid, {})
        seg = meta.get("segment", meta.get("role", agent.social_role))
        s = segments[seg]
        s["n"] += 1
        s["debt"] += agent.debt_pressure
        s["dread"] += agent.dread_pressure
        s["pess"] += getattr(agent, "expectation_pessimism", 0)
        s["tens"] += agent.heart.tension
        s["val"] += agent.heart.valence
        # Deltas from baseline
        b = baseline.get(aid, {})
        s["debt_delta"] += agent.debt_pressure - b.get("debt", 0)
        s["dread_delta"] += agent.dread_pressure - b.get("dread", 0)
        s["pess_delta"] += getattr(agent, "expectation_pessimism", 0) - b.get("pessimism", 0)
        s["val_delta"] += agent.heart.valence - b.get("valence", 0)
        s["savings_lost"] += b.get("savings", 0) - agent.savings_buffer
        s["income_lost"] += b.get("income", 0) - agent.income_level

    # Sort by pessimism delta (who got hit hardest relative to where they started)
    seg_report = []
    for seg, s in sorted(segments.items(), key=lambda x: -(x[1]["pess_delta"] / max(1, x[1]["n"]))):
        n = s["n"]
        if n == 0:
            continue
        seg_report.append({
            "segment": seg,
            "count": n,
            "avg_debt": round(s["debt"] / n, 3),
            "avg_dread": round(s["dread"] / n, 3),
            "avg_pessimism": round(s["pess"] / n, 3),
            "avg_tension": round(s["tens"] / n, 3),
            "avg_mood": round(s["val"] / n, 3),
            "debt_change": round(s["debt_delta"] / n, 3),
            "dread_change": round(s["dread_delta"] / n, 3),
            "pessimism_change": round(s["pess_delta"] / n, 3),
            "mood_change": round(s["val_delta"] / n, 3),
            "avg_savings_lost": round(s["savings_lost"] / n, 3),
            "avg_income_lost": round(s["income_lost"] / n, 3),
        })
    report["segments"] = seg_report

    # ── Ripple chains — the non-obvious cascades ──
    ripple_chains = []
    if world.ripple_engine:
        ripple_chains = world.ripple_engine.get_recent_chains(50)
        report["ripple_summary"] = world.ripple_engine.get_chain_summary()
    report["ripple_chains"] = ripple_chains

    # ── Key figure deep dives ──
    key_figures = []
    for aid, meta in agent_meta.items():
        if not meta.get("is_llm_agent") or aid not in world.agents:
            continue
        agent = world.agents[aid]
        b = baseline.get(aid, {})
        memories = agent.get_recent_memories(5)
        policy_memories = [m for m in memories if "[policy]" in m.description]

        key_figures.append({
            "name": agent.personality.name,
            "title": meta.get("title", ""),
            "org": meta.get("org", ""),
            "emotion": agent.heart.surface_emotion,
            "internal_emotion": agent.heart.internal_emotion,
            "divergence": round(agent.heart.divergence, 3),
            "tension": round(agent.heart.tension, 3),
            "dread": round(agent.dread_pressure, 3),
            "concern": agent.appraisal.primary_concern,
            "ongoing_story": agent.appraisal.ongoing_story[:100],
            "blame_target": agent.appraisal.blame_target,
            "coalitions": list(agent.coalitions),
            "mood_change": round(agent.heart.valence - b.get("valence", 0.5), 3),
            "policy_reactions": [m.description[:120] for m in policy_memories[:3]],
        })
    report["key_figures"] = key_figures

    # ── Policy impact chain ──
    report["policy_impacts"] = [
        {
            "domain": pr["domain"],
            "policy": pr["policy"],
            "description": pr["description"],
            "winners": pr["winners"],
            "losers": pr["losers"],
            "sectors": pr.get("market_sectors", {}),
            "agents_affected": pr["agents_affected"],
        }
        for pr in policy_result["policies_applied"]
    ]

    # ── NON-OBVIOUS INSIGHTS — this is the key differentiator ──
    report["insights"] = _extract_insights(
        world, agent_meta, baseline, segments, seg_report,
        ripple_chains, key_figures, agenda, sector_tracker,
    )

    # ── Concerns and emotions ──
    concerns = Counter()
    emotions = Counter()
    for agent in world.agents.values():
        if agent.appraisal.primary_concern:
            concerns[agent.appraisal.primary_concern] += 1
        emotions[agent.heart.surface_emotion] += 1
    report["top_concerns"] = concerns.most_common(10)
    report["emotion_distribution"] = emotions.most_common()

    return report


def _extract_insights(
    world, agent_meta, baseline, segments, seg_report,
    ripple_chains, key_figures, agenda, sector_tracker,
) -> list[dict]:
    """Extract non-obvious, counterintuitive, and emergent insights.

    This is what makes the simulation valuable — surfacing things
    humans wouldn't predict from reading the news.
    """
    insights = []

    # ── 1. Unexpected winners/losers ──
    # Find segments whose pessimism change doesn't match expectations
    for seg in seg_report:
        name = seg["segment"]
        pess = seg["pessimism_change"]
        debt = seg["debt_change"]

        # High debt increase but LOW pessimism = psychologically resilient group
        if debt > 0.3 and pess < 0.3:
            insights.append({
                "type": "counterintuitive_resilience",
                "title": f"{name} absorb economic hit without breaking psychologically",
                "detail": f"Despite debt increasing by {debt:+.2f}, pessimism only rose {pess:+.2f}. "
                         f"This group's coping mechanisms (threat_lens, self_story) buffer the psychological impact "
                         f"even as the financial damage accumulates. They're hurting but not panicking.",
                "segment": name,
                "surprise_factor": debt - pess,
            })

        # LOW debt increase but HIGH pessimism = psychological damage exceeds financial
        if debt < 0.15 and pess > 0.4:
            insights.append({
                "type": "psychological_damage",
                "title": f"{name} psychologically devastated despite limited financial impact",
                "detail": f"Debt only changed {debt:+.2f} but pessimism surged {pess:+.2f}. "
                         f"The fear and dread are doing more damage than the actual financial hit. "
                         f"This group is being destroyed by anxiety about what MIGHT happen, not what has.",
                "segment": name,
                "surprise_factor": pess - debt,
            })

    # ── 2. Emotional divergence — who is masking? ──
    for fig in key_figures:
        if fig["divergence"] > 0.15:
            insights.append({
                "type": "emotional_masking",
                "title": f"{fig['name']} is hiding their real emotional state",
                "detail": f"Surface: {fig['emotion']}, Internal: {fig['internal_emotion']} "
                         f"(divergence={fig['divergence']:.2f}). Their {fig.get('org','')} role demands composure "
                         f"but internally they're {fig['internal_emotion']}. This masking costs energy and "
                         f"makes their future decisions less predictable.",
                "person": fig["name"],
                "surprise_factor": fig["divergence"],
            })

    # ── 3. Blame patterns — who gets scapegoated? ──
    blame_counter = Counter()
    for agent in world.agents.values():
        if agent.appraisal.blame_target and agent.appraisal.blame_target != "circumstances":
            blame_counter[agent.appraisal.blame_target] += 1

    total = len(world.agents)
    for target, count in blame_counter.most_common(5):
        if count > total * 0.05:  # >5% blaming same target
            insights.append({
                "type": "blame_concentration",
                "title": f"{count/total*100:.0f}% of population blames '{target}'",
                "detail": f"{count} agents ({count/total*100:.1f}%) have directed blame at '{target}'. "
                         f"This concentration of blame creates political pressure and potential for "
                         f"scapegoating, protests, or policy overreaction targeting this group.",
                "target": target,
                "surprise_factor": count / total,
            })

    # ── 4. Ripple chain analysis — longest cascades ──
    if ripple_chains:
        # Find most-affected targets (appeared in most ripple events)
        target_hits = Counter()
        actor_impacts = Counter()
        for chain in ripple_chains:
            target_hits[chain.get("target", "")] += 1
            actor_impacts[chain.get("actor", "")] += 1

        for target, hits in target_hits.most_common(3):
            if hits >= 3:
                insights.append({
                    "type": "cascade_victim",
                    "title": f"{target} hit by {hits} separate ripple cascades",
                    "detail": f"Multiple independent economic consequences converge on {target}. "
                             f"They're being squeezed from multiple directions simultaneously — "
                             f"this compound pressure is harder to survive than any single hit.",
                    "person": target,
                    "surprise_factor": hits / 3,
                })

        for actor, impacts in actor_impacts.most_common(3):
            if impacts >= 3:
                insights.append({
                    "type": "cascade_source",
                    "title": f"{actor}'s decisions triggered {impacts} downstream consequences",
                    "detail": f"When {actor} adjusted their behavior under pressure, it cascaded "
                             f"to {impacts} other people. This person is a critical node — their "
                             f"next decision will amplify or dampen the crisis.",
                    "person": actor,
                    "surprise_factor": impacts / 3,
                })

    # ── 5. Sector paradoxes — sectors hurt by their own "win" ──
    sector_data = sector_tracker.get_report()
    for sec, val in sector_data.get("all_sectors", {}).items():
        # Sectors that boom but their workers still suffer
        if val > 0.1:
            # Check if workers in this sector are actually doing well
            for seg in seg_report:
                seg_name = seg["segment"].lower()
                if sec in seg_name or seg_name in sec:
                    if seg["pessimism_change"] > 0.2:
                        insights.append({
                            "type": "sector_paradox",
                            "title": f"{sec} sector booming but {seg['segment']} workers still suffering",
                            "detail": f"The {sec} sector grew ({val:+.1f}) but workers in this area "
                                     f"saw pessimism increase by {seg['pessimism_change']:+.2f}. "
                                     f"Sector-level growth doesn't mean worker-level wellbeing — "
                                     f"profits flow to owners while workers absorb secondary shocks.",
                            "sector": sec,
                            "segment": seg["segment"],
                            "surprise_factor": val + seg["pessimism_change"],
                        })

    # ── 6. Coalition fracture risk ──
    coalition_stress = defaultdict(lambda: {"members": 0, "avg_tension": 0, "avg_rival_tension": 0})
    for agent in world.agents.values():
        for coal in agent.coalitions:
            cs = coalition_stress[coal]
            cs["members"] += 1
            cs["avg_tension"] += agent.heart.tension

    for coal, data in coalition_stress.items():
        if data["members"] > 3:
            avg_t = data["avg_tension"] / data["members"]
            if avg_t > 0.25:
                insights.append({
                    "type": "coalition_fracture_risk",
                    "title": f"'{coal}' coalition under internal stress (tension={avg_t:.2f})",
                    "detail": f"The {data['members']} members of the '{coal}' coalition have "
                             f"average tension of {avg_t:.2f}. High internal tension increases "
                             f"the probability of defections, internal power struggles, or "
                             f"the coalition breaking apart under pressure.",
                    "coalition": coal,
                    "surprise_factor": avg_t,
                })

    # ── 7. Savings depletion timeline — who runs out of money first? ──
    depletion_risk = []
    for aid, agent in world.agents.items():
        if agent.savings_buffer < 0.05 and baseline.get(aid, {}).get("savings", 0.5) > 0.15:
            meta = agent_meta.get(aid, {})
            seg = meta.get("segment", meta.get("role", agent.social_role))
            depletion_risk.append(seg)

    if depletion_risk:
        depleted_segments = Counter(depletion_risk)
        for seg, count in depleted_segments.most_common(3):
            insights.append({
                "type": "savings_depletion",
                "title": f"{count} {seg} agents have burned through their savings",
                "detail": f"{count} people in the '{seg}' segment went from having savings "
                         f"to effectively zero. These households are now one unexpected expense "
                         f"from crisis. They'll cut spending drastically, which hits local "
                         f"businesses in a secondary wave.",
                "segment": seg,
                "count": count,
                "surprise_factor": count / max(1, sum(1 for s in segments if s == seg)),
            })

    # ── 8. Compound policy interactions — policies that amplify each other ──
    # Find roles that appear as losers in 3+ policies
    role_policy_hits = defaultdict(list)
    for p in agenda.policies:
        for role in p.losers:
            role_policy_hits[role].append(p.policy_name)

    for role, policies in role_policy_hits.items():
        if len(policies) >= 3:
            insights.append({
                "type": "compound_policy_squeeze",
                "title": f"{role} hit by {len(policies)} policies simultaneously",
                "detail": f"The '{role}' segment is a loser in {len(policies)} different policies: "
                         f"{', '.join(policies[:4])}. Each policy alone is manageable, but the "
                         f"compound effect is devastating — like being punched from every direction.",
                "role": role,
                "policies": policies,
                "surprise_factor": len(policies) / 3,
            })

    # Sort by surprise factor
    insights.sort(key=lambda x: -x.get("surprise_factor", 0))

    return insights


# ---------------------------------------------------------------------------
# Pretty print the report
# ---------------------------------------------------------------------------

def _print_report(report: dict):
    """Print a narrative report focused on non-obvious insights."""

    print(f"\n{'='*72}")
    print(f"  SIMULATION RESULTS: {report['prediction'][:60]}")
    print(f"  {report['total_agents']} agents, {report['days_simulated']} days")
    print(f"{'='*72}")

    # Macro
    macro = report.get("macro", {})
    current = macro.get("current", {})
    deltas = macro.get("deltas", {})
    if current:
        print(f"\n── MACRO ECONOMY ──")
        for label, key in [("Consumer Confidence", "consumer_confidence"),
                           ("Population Mood", "population_mood"),
                           ("Market Pressure", "market_pressure"),
                           ("Institutional Trust", "institutional_trust"),
                           ("Civil Unrest Risk", "civil_unrest_potential")]:
            c = current.get(key, 0)
            d = deltas.get(key, 0)
            if isinstance(c, (int, float)):
                print(f"  {label:25s} {c:>8.2f}  {'▲' if d>0 else '▼' if d<0 else '─'} {d:+.2f}")

    # Sectors
    sectors = report.get("sectors", {})
    boom = sectors.get("booming", {})
    bust = sectors.get("struggling", {})
    if boom or bust:
        print(f"\n── MARKET SECTORS ──")
        if boom:
            for s, v in boom.items():
                print(f"  ▲ {s:20s} {v:+.1f}")
        if bust:
            for s, v in bust.items():
                print(f"  ▼ {s:20s} {v:+.1f}")

    # Who gets hit (sorted by delta, not absolute)
    segs = report.get("segments", [])
    if segs:
        print(f"\n── WHO GETS HIT (by change from baseline) ──")
        print(f"  {'Group':25s} {'Debt Δ':>7s} {'Dread Δ':>8s} {'Pessim Δ':>9s} {'Mood Δ':>7s} {'Savings Lost':>13s}")
        print(f"  {'─'*72}")
        for seg in segs[:12]:
            print(f"  {seg['segment']:25s} {seg['debt_change']:>+7.2f} {seg['dread_change']:>+8.2f} "
                  f"{seg['pessimism_change']:>+9.2f} {seg['mood_change']:>+7.2f} {seg['avg_savings_lost']:>+13.2f}")

    # Policy chain
    impacts = report.get("policy_impacts", [])
    if impacts:
        print(f"\n── ECONOMIC CONSEQUENCE CHAIN ──")
        for p in impacts:
            losers = ", ".join(p["losers"]) if p["losers"] else "everyone"
            print(f"\n  {p['policy']} ({p['domain']})")
            print(f"    {p['description'][:140]}")
            print(f"    Hits: {losers} ({p['agents_affected']} agents)")

    # ═══════════════════════════════════════════════════════════════
    # THE KEY SECTION — NON-OBVIOUS INSIGHTS
    # ═══════════════════════════════════════════════════════════════
    insights = report.get("insights", [])
    if insights:
        print(f"\n{'='*72}")
        print(f"  NON-OBVIOUS INSIGHTS ({len(insights)} found)")
        print(f"  These are things you wouldn't predict from reading the news")
        print(f"{'='*72}")

        for i, insight in enumerate(insights[:15], 1):
            itype = insight["type"].replace("_", " ").upper()
            print(f"\n  [{i}] {itype}")
            print(f"      {insight['title']}")
            print(f"      {insight['detail'][:200]}")

    # Key figures
    figures = report.get("key_figures", [])
    if figures:
        print(f"\n── KEY FIGURES ──")
        for fig in figures[:10]:
            reactions = "; ".join(r[:60] for r in fig.get("policy_reactions", [])[:2])
            print(f"\n  {fig['name']} — {fig['title']}")
            print(f"    {fig['emotion']} | tension={fig['tension']:.2f} | Concern: {fig['concern'][:55]}")
            print(f"    Story: {fig['ongoing_story'][:70]}")
            if reactions:
                print(f"    Reacts: {reactions[:120]}")

    # LLM Agent Decisions — the emergent part
    llm_data = report.get("llm_agency", {})
    if llm_data.get("enabled"):
        full_log = llm_data.get("full_decision_log", [])
        print(f"\n{'='*72}")
        print(f"  LLM AGENT DECISIONS ({llm_data['total_decisions']} decisions, {llm_data['total_ripple_events']} ripple events)")
        print(f"  These are real decisions made by simulated characters")
        print(f"{'='*72}")
        for d in full_log[:25]:
            print(f"\n  [{d.get('time','')}] {d['agent']} ({d['role']})")
            action = d.get('action', d.get('decision', '?'))
            print(f"    ACTION: {action[:150]}")
            print(f"    WHY: {d.get('reasoning','')[:130]}")
            if d.get('speech'):
                print(f"    Says: \"{d['speech'][:110]}\"")
            if d.get('thought'):
                print(f"    Thinks: \"{d['thought'][:110]}\"")
            msgs = d.get('message_recipients', [])
            if msgs:
                print(f"    Messages → {', '.join(msgs[:4])}")
            ripples = d.get('ripple_count', d.get('consequences', 0))
            if ripples:
                print(f"    → {ripples} downstream consequences")

    # Ripple chains
    chains = report.get("ripple_chains", [])
    if chains:
        print(f"\n── RIPPLE CASCADES ({len(chains)} events) ──")
        for chain in chains[:15]:
            print(f"  {chain.get('actor','?'):25s} → {chain.get('target','?'):25s} | {chain.get('action','')[:50]}")

    # Concerns
    print(f"\n── TOP CONCERNS ──")
    for concern, count in report.get("top_concerns", [])[:6]:
        pct = count / report["total_agents"] * 100
        print(f"  {pct:5.1f}%  {concern}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m world_sim.run_prediction 'What if X happens?'")
        sys.exit(1)

    prediction = " ".join(sys.argv[1:])
    run_prediction(prediction)

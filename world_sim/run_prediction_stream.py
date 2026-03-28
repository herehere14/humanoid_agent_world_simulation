"""Streaming prediction runner — yields events as the simulation runs.

Instead of returning a single report dict at the end, this yields
SSE-compatible events in real-time:
  - setup: key figures, policies, world built
  - decision: each LLM agent decision as it happens
  - message: agent-to-agent communication
  - day_summary: narrative + macro metrics at end of each day
  - ripple: cascade chain events
  - insight: non-obvious insights (at the end)
  - complete: final summary
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from typing import Generator


def stream_prediction(
    prediction: str,
    *,
    days: int = 7,
    model: str = "gpt-5-mini",
    seed: int = 42,
    llm_agents_per_tick: int = 4,
) -> Generator[dict, None, None]:
    """Yields simulation events as they happen.

    Each yield is a dict with {"type": "...", "data": {...}}
    """
    import re
    ticks = days * 24
    api_key = os.getenv("OPENAI_API_KEY", "")

    yield {"type": "status", "data": {"message": "Researching key figures and economic impacts..."}}

    # ── Step 1: Research ──
    from .scenario_generator import _call_openai
    from .policy_engine import research_policies, SectorTracker

    sanitized = prediction
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

    t0 = time.time()
    world_spec = _call_openai(sim_framing, model=model)
    agenda = research_policies(sim_framing, model=model)
    sector_tracker = SectorTracker()

    key_figure_names = []
    for org in world_spec.get("organizations", []):
        for p in org.get("key_personnel", []):
            key_figure_names.append({"name": p["name"], "title": p.get("title", ""), "org": org["name"]})

    yield {
        "type": "setup",
        "data": {
            "prediction": prediction,
            "key_figures": key_figure_names,
            "policies": [{"domain": p.domain, "name": p.policy_name, "description": p.description[:150]} for p in agenda.policies],
            "population_segments": [{"label": s.get("label", ""), "count": s.get("count", 0)} for s in world_spec.get("population_segments", [])],
            "research_time": round(time.time() - t0, 1),
        },
    }

    # ── Step 2: Build World ──
    yield {"type": "status", "data": {"message": "Building world with personality profiles..."}}

    from .scenario_generator import build_world_from_spec, build_shock_from_spec
    result = build_world_from_spec(world_spec, seed=seed)
    world = result.world
    world.initialize()

    yield {
        "type": "world_built",
        "data": {
            "total_agents": len(world.agents),
            "total_locations": len(world.locations),
            "locations": [{"id": lid, "name": loc.name} for lid, loc in world.locations.items()],
        },
    }

    # ── Step 3: Inject shocks + policies ──
    yield {"type": "status", "data": {"message": "Injecting shocks and policy conditions..."}}

    from .world_information import apply_external_information
    if world_spec.get("shock_events"):
        shock_plan = build_shock_from_spec(world_spec, start_tick=1)
        apply_external_information(world, shock_plan)

    from .policy_engine import apply_policy_agenda
    policy_result = apply_policy_agenda(world, agenda)
    for p in agenda.policies:
        sector_tracker.apply_policy(p)

    yield {
        "type": "policies_injected",
        "data": {
            "persistent_conditions": policy_result["total_persistent_conditions"],
            "narrative_events": policy_result["total_events_scheduled"],
            "agents_impacted": policy_result["total_agents_affected"],
        },
    }

    # ── Step 4: Set up LLM agency ──
    llm_engine = None
    if api_key:
        from .llm_agency import LLMAgencyEngine
        llm_engine = LLMAgencyEngine(api_key=api_key, model=model, fabric=result.fabric)
        if world.ripple_engine is None:
            from .ripple_engine import RippleEngine
            world.ripple_engine = RippleEngine(result.fabric, seed=seed)

    # Baseline snapshot
    baseline = {}
    for aid, agent in world.agents.items():
        baseline[aid] = {
            "debt": agent.debt_pressure, "dread": agent.dread_pressure,
            "tension": agent.heart.tension, "valence": agent.heart.valence,
            "pessimism": getattr(agent, "expectation_pessimism", 0),
            "savings": agent.savings_buffer, "income": agent.income_level,
        }

    yield {"type": "status", "data": {"message": f"Simulating {days} days with freeform LLM agents..."}}

    # ── Step 5: Run simulation, streaming events ──
    prev_decision_count = 0

    for tick_i in range(ticks):
        summary = world.tick()
        current_tick = tick_i + 1

        # Emit scheduled narrative events
        for evt in summary.get("events", []):
            yield {
                "type": "event",
                "data": {
                    "tick": current_tick,
                    "time": summary.get("time", ""),
                    "location": evt.get("location", ""),
                    "description": evt.get("description", ""),
                    "kind": evt.get("kind", ""),
                },
            }

        # LLM agent decisions
        if llm_engine:
            ripple_events = llm_engine.tick(world, max_calls=llm_agents_per_tick)
            if world.ripple_engine:
                world.ripple_engine.event_log.extend(ripple_events)

            # Stream new decisions since last check
            new_decisions = llm_engine.decision_log[prev_decision_count:]
            prev_decision_count = len(llm_engine.decision_log)

            for d in new_decisions:
                yield {
                    "type": "decision",
                    "data": {
                        "tick": current_tick,
                        "time": d.get("time", ""),
                        "agent": d.get("agent", ""),
                        "role": d.get("role", ""),
                        "action": d.get("action", ""),
                        "reasoning": d.get("reasoning", ""),
                        "speech": d.get("speech", ""),
                        "thought": d.get("thought", ""),
                        "messages_to": d.get("message_recipients", []),
                        "ripple_count": d.get("ripple_count", 0),
                        "trigger": d.get("trigger", "scheduled"),
                        "cascade_depth": d.get("cascade_depth", 0),
                        "triggered_by": d.get("triggered_by", ""),
                    },
                }

        # End of day summary
        if current_tick % 24 == 0:
            day_num = current_tick // 24
            macro = summary.get("macro", {})

            # Collect the day's decisions for narrative
            day_decisions = [
                d for d in llm_engine.decision_log if d.get("tick", 0) > (day_num - 1) * 24 and d.get("tick", 0) <= day_num * 24
            ] if llm_engine else []

            # Generate day narrative using GPT-5-mini
            narrative = _generate_day_narrative(
                prediction, day_num, macro, day_decisions, world, api_key, model,
            )

            # Segment stats
            seg_stats = defaultdict(lambda: {"n": 0, "debt": 0, "dread": 0, "pess": 0})
            for aid, agent in world.agents.items():
                meta = result.agent_meta.get(aid, {})
                seg = meta.get("segment", meta.get("role", agent.social_role))
                s = seg_stats[seg]
                s["n"] += 1
                s["debt"] += agent.debt_pressure
                s["dread"] += agent.dread_pressure
                s["pess"] += getattr(agent, "expectation_pessimism", 0)

            worst_segments = sorted(
                [(seg, s["pess"] / max(1, s["n"])) for seg, s in seg_stats.items()],
                key=lambda x: -x[1],
            )[:5]

            yield {
                "type": "day_summary",
                "data": {
                    "day": day_num,
                    "narrative": narrative,
                    "macro": {
                        "consumer_confidence": macro.get("consumer_confidence", 0),
                        "population_mood": macro.get("population_mood", 0),
                        "market_pressure": macro.get("market_pressure", 0),
                        "institutional_trust": macro.get("institutional_trust", 0),
                        "civil_unrest": macro.get("civil_unrest_potential", 0),
                    },
                    "decisions_today": len(day_decisions),
                    "total_decisions": len(llm_engine.decision_log) if llm_engine else 0,
                    "worst_hit": [{"segment": s, "pessimism": round(p, 3)} for s, p in worst_segments],
                },
            }

    # ── Step 6: Final analysis ──
    yield {"type": "status", "data": {"message": "Analyzing for non-obvious insights..."}}

    from .run_prediction import _build_insight_report, _extract_insights
    report = _build_insight_report(
        world, result.agent_meta, baseline, agenda, sector_tracker,
        policy_result, world_spec, prediction, days,
    )

    if llm_engine:
        stats = llm_engine.get_stats()
        report["llm_agency"] = {
            "enabled": True,
            "total_calls": stats["total_calls"],
            "total_decisions": stats["total_decisions"],
            "total_reactive": stats.get("total_reactive", 0),
            "total_conversations": stats.get("total_conversations", 0),
            "total_messages_sent": stats.get("total_messages_sent", 0),
            "max_cascade_depth": stats.get("max_cascade_depth", 0),
            "full_decision_log": llm_engine.decision_log,
        }

    # Stream insights one by one
    for insight in report.get("insights", []):
        yield {"type": "insight", "data": insight}

    # Final complete event with full report
    yield {"type": "complete", "data": report}


def _generate_day_narrative(
    prediction: str, day_num: int, macro: dict, day_decisions: list,
    world, api_key: str, model: str,
) -> str:
    """Use GPT-5-mini to generate a dramatic day narrative from the decisions."""
    if not api_key or not day_decisions:
        return f"Day {day_num} passed with {len(day_decisions)} decisions made."

    # Build context from the day's decisions
    decision_summaries = []
    for d in day_decisions[:15]:  # Cap to avoid token limits
        line = f"- {d.get('agent', '?')}: {d.get('action', '?')[:100]}"
        if d.get('speech'):
            line += f" (said: \"{d['speech'][:60]}\")"
        decision_summaries.append(line)

    prompt = f"""You are a dramatic narrator for a world simulation. Write a vivid 3-5 sentence narrative summary of Day {day_num} in this scenario:

SCENARIO: {prediction}

KEY EVENTS TODAY:
{chr(10).join(decision_summaries)}

MACRO STATE:
- Consumer confidence: {macro.get('consumer_confidence', '?')}
- Population mood: {macro.get('population_mood', '?')}
- Market pressure: {macro.get('market_pressure', '?')}

Write a dramatic, specific, and vivid narrative of what happened on Day {day_num}. Include specific names, numbers, and consequences. Make it read like a news report from inside the crisis. Be concrete — not "things got worse" but "Sydney's fresh produce supply ran out by 3pm, forcing Woolworths to ration bread to two loaves per household."

Return ONLY the narrative text, no JSON, no formatting."""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=400,
            reasoning={"effort": "low"},
        )
        return (resp.output_text or "").strip()
    except Exception:
        return f"Day {day_num}: {len(day_decisions)} decisions made across the simulation."

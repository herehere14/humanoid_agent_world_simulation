"""Dynamic scenario generator — turns any user prediction into a runnable world.

Pipeline:
  1. User submits a prediction (e.g., "Australian Albanese government loses election")
  2. We call OpenAI (with web search capability) to research all key figures
  3. OpenAI returns structured JSON with personalities, locations, orgs, population
  4. We build a World with those characters and return it ready to simulate

The LLM does the research and personality injection. The deterministic engine
does the physics.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .world import World, Location
from .world_agent import WorldAgent, Personality
from .human_profiles import assign_human_profile
from .ripple_engine import OrganizationalFabric, OrgLink


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT — tells OpenAI exactly what structure to return
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a world-building researcher for an academic agent-based economic simulation engine used for scenario planning, risk analysis, and policy research.

Given a hypothetical scenario for economic simulation (e.g., "What if a major political leader leaves office?"), you must:

1. RESEARCH all the key real-world figures who would be affected by or involved in this scenario
2. For each figure, create a psychologically rich personality profile based on their REAL public behavior, biography, and known temperament
3. Organize them into institutions/organizations
4. Define relevant locations where the action happens
5. Define population segments that represent ordinary people affected

You must return VALID JSON matching this exact schema. No markdown, no commentary — just the JSON object.

IMPORTANT RULES:
- Use REAL names of real people. This is a simulation of real-world dynamics.
- Personality profiles should be psychologically honest — how they ACTUALLY behave under pressure, not PR versions
- Include 15-30 key figures across government, opposition, regulators, business, unions, media, civil society
- Include 6-12 population segments with 20-50 agents each
- Include 15-30 locations relevant to the scenario
- Define realistic coalitions and rivalries
- The "shock_description" should describe the triggering event clearly for our shock engine

VALID VALUES for personality fields:
- threat_lens: "chaos", "humiliation", "abandonment", "scarcity", "betrayal", "exposure"
- core_need: "belonging", "control", "dignity", "safety", "usefulness", "truth", "justice"
- coping_style: "reach for connection", "perform competence", "disappear into work", "intellectualize", "confront head-on", "control the room", "command", "deflect with humor", "caretake first", "keep score quietly", "seek witnesses"
- conflict_style: "cool negotiation", "go sharp", "command", "appease first", "keep score", "moralize in public", "straight negotiation", "triangulate room"
- mask_tendency: "polished competence", "dutiful calm", "soft warmth", "joke through it", "emotional shutdown", "command presence", "righteous heat", "quiet shutdown"
- self_story: "survivor", "fixer", "witness", "provider", "guardian", "climber", "operator", "loyalist", "story carrier", "believer", "outsider"
- care_style: "steady presence", "practical fixing", "quiet encouragement", "strategic problem-solving", "protective provisioning", "emotional reassurance"
- social_role (for key figures): "government_worker", "manager", "office_professional", "community", "healthcare"
- social_role (for population): "factory_worker", "office_professional", "government_worker", "healthcare", "market_vendor", "community", "student", "dock_worker"

JSON SCHEMA:
{
  "scenario_name": "string — short name for this scenario",
  "scenario_description": "string — 2-3 sentence description",
  "shock_description": "string — the triggering event, written for our simulation engine",
  "shock_severity": 0.0-3.0,

  "locations": [
    {
      "id": "string — snake_case unique ID",
      "name": "string — human-readable name",
      "activity": "string — what normally happens here"
    }
  ],

  "organizations": [
    {
      "name": "string — org name",
      "sector": "string — sector type",
      "hq_location": "string — location ID from above",
      "employee_count": 10-50,
      "revenue_influence": 0.0-0.20,
      "supply_chain_to": ["other org names"],
      "regulated_by": ["regulator org names"],
      "coalition": "string — coalition name or empty",
      "rival_coalition": "string — rival coalition or empty",
      "key_personnel": [
        {
          "name": "string — real full name",
          "title": "string — their role/title",
          "background": "string — 2-3 sentences of real biography and career context",
          "temperament": "string — 2-3 sentences of honest psychological profile: how they actually behave, what drives them, their weaknesses",
          "social_role": "government_worker|manager|office_professional|community|healthcare",
          "threat_lens": "one of the valid values above",
          "core_need": "one of the valid values above",
          "coping_style": "one of the valid values above",
          "conflict_style": "one of the valid values above",
          "self_story": "one of the valid values above",
          "shame_trigger": "string — what makes them feel exposed or ashamed",
          "mask_tendency": "one of the valid values above",
          "care_style": "one of the valid values above",
          "longing": "string — what they deeply want but rarely say"
        }
      ]
    }
  ],

  "population_segments": [
    {
      "label": "string — descriptive label (e.g., 'public_servant', 'miner', 'retiree')",
      "social_role": "one of the valid social_role values for population",
      "count": 20-50,
      "work_location": "string — location ID",
      "home_location": "string — location ID",
      "backgrounds": ["string — 3 background templates with {years} placeholder"],
      "temperaments": ["string — 3 temperament descriptions"],
      "debt_range": [0.0, 1.0],
      "ambition_range": [0.0, 1.0],
      "coalition": "string — optional coalition affiliation or empty"
    }
  ],

  "key_relationships": [
    {
      "from": "person name",
      "to": "person name",
      "trust": -1.0 to 1.0,
      "warmth": -1.0 to 1.0,
      "rivalry": 0.0 to 1.0,
      "alliance": 0.0 to 1.0
    }
  ],

  "shock_events": [
    {
      "tick_offset": 0-48,
      "location": "string — location ID",
      "description": "string — what happens in the narrative",
      "emotional_text": "string — how it FEELS to people there",
      "severity": 0.0-3.0
    }
  ],

  "role_impacts": {
    "social_role_name": {
      "tension": 0.0-0.3,
      "debt_pressure": 0.0-0.3,
      "dread_pressure": 0.0-0.3,
      "valence": -0.3 to 0.0
    }
  }
}"""


USER_PROMPT_TEMPLATE = """Research and build a complete simulation world for this prediction/scenario:

"{prediction}"

Remember:
- Use REAL names of real people involved (politicians, business leaders, regulators, union leaders, etc.)
- Create psychologically honest personality profiles based on their known real-world behavior
- Include all sides: government, opposition, regulators, business, workers, media, civil society
- Population segments should represent the real demographic groups affected
- Locations should be real places relevant to this scenario
- Define coalitions and rivalries that reflect real political/economic alliances

Return ONLY the JSON object. No markdown fences, no commentary."""


# ═══════════════════════════════════════════════════════════════════════════
# OPENAI CALLER
# ═══════════════════════════════════════════════════════════════════════════

def _call_openai(
    prediction: str,
    model: str = "gpt-5-mini",
    temperature: float = 0.4,
    max_tokens: int = 16000,
    timeout: int = 300,
) -> dict:
    """Call OpenAI to research key figures and generate scenario JSON."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    body = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(prediction=prediction)},
        ],
    }
    # Only add temperature if model supports it (gpt-5 series does not)
    if not model.startswith("gpt-5"):
        body["temperature"] = temperature

    payload = json.dumps(body, ensure_ascii=True).encode("utf-8")
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error = ""
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"] or ""

            # Handle empty content or refusal
            if not content.strip():
                refusal = data["choices"][0]["message"].get("refusal", "")
                if refusal:
                    raise RuntimeError(f"Model refused: {refusal}")
                last_error = "Empty response from model"
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise RuntimeError(last_error)

            # Strip markdown fences if present
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            return json.loads(text)
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTPError {exc.code}: {body_text}"
            if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenAI API error: {last_error}")
        except RuntimeError:
            raise
        except Exception as e:
            last_error = str(e)
            if attempt < 2:
                time.sleep(1)
                continue
            raise RuntimeError(f"OpenAI call failed: {last_error}")

    raise RuntimeError(f"OpenAI call failed after retries: {last_error}")


# ═══════════════════════════════════════════════════════════════════════════
# WORLD BUILDER — converts OpenAI JSON into a runnable World
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratedScenario:
    """Result of dynamic scenario generation."""
    world: World
    agent_meta: dict[str, dict]
    fabric: OrganizationalFabric
    scenario_spec: dict  # raw OpenAI output for inspection
    shock_description: str


def build_world_from_spec(spec: dict, seed: int = 42) -> GeneratedScenario:
    """Build a World from OpenAI-generated scenario specification.

    This is the pure deterministic builder — no API calls.
    Can be used with cached/modified specs.
    """
    rng = random.Random(seed)
    world = World()
    agent_meta: dict[str, dict] = {}
    fabric = OrganizationalFabric()
    org_agents: dict[str, list[str]] = {}  # org_name → [agent_ids]
    name_to_id: dict[str, str] = {}  # person name → agent_id

    # ── 1. Add locations ──
    home_locations = []
    for loc in spec.get("locations", []):
        loc_id = loc["id"]
        world.add_location(Location(loc_id, loc["name"], loc.get("activity", "daily activity")))
        # Track locations that look like homes/residential
        if any(kw in loc_id.lower() for kw in ("suburb", "home", "residential", "regional", "downtown", "urban")):
            home_locations.append(loc_id)

    if not home_locations:
        # Fallback: create a generic home location
        world.add_location(Location("residential_area", "Residential Area", "daily life"))
        home_locations = ["residential_area"]

    # ── 2. Build organization agents (key figures) ──
    for org in spec.get("organizations", []):
        org_name = org["name"]
        org_agents[org_name] = []
        hq = org.get("hq_location", home_locations[0])

        # Ensure HQ location exists
        if hq not in world.locations:
            world.add_location(Location(hq, hq.replace("_", " ").title(), "organizational headquarters"))

        coalition = org.get("coalition", "")
        rival = org.get("rival_coalition", "")

        for person in org.get("key_personnel", []):
            agent_id = person["name"].lower().replace(" ", "_").replace(".", "").replace("'", "")
            if agent_id in world.agents:
                agent_id += f"_{org_name[:4].lower()}"

            personality = Personality(
                name=person["name"],
                background=f'{person.get("title", "")} at {org_name}. {person.get("background", "")}',
                temperament=person.get("temperament", "measured and professional"),
                attachment_style="",
                coping_style=person.get("coping_style", ""),
                threat_lens=person.get("threat_lens", ""),
                core_need=person.get("core_need", ""),
                conflict_style=person.get("conflict_style", ""),
                self_story=person.get("self_story", ""),
            )

            # Inject extended personality fields
            for extra in ("shame_trigger", "mask_tendency", "longing", "care_style"):
                if extra in person and hasattr(personality, extra):
                    setattr(personality, extra, person[extra])

            social_role = person.get("social_role", "office_professional")

            # Build schedule: at HQ during work hours
            schedule = {}
            home = rng.choice(home_locations)
            for h in range(0, 8):
                schedule[h] = home
            for h in range(8, 19):
                schedule[h] = hq
            for h in range(19, 24):
                schedule[h] = home

            agent = WorldAgent(
                agent_id=agent_id,
                personality=personality,
                schedule=schedule,
                social_role=social_role,
                debt_pressure=rng.uniform(0.02, 0.10),
                ambition=rng.uniform(0.5, 0.9),
                savings_buffer=rng.uniform(0.5, 0.9),
                credit_access=rng.uniform(0.6, 0.95),
                income_level=rng.uniform(0.7, 0.95),
            )

            # Set coalitions
            if coalition:
                agent.coalitions = tuple(c.strip() for c in coalition.split(",") if c.strip())
            if rival:
                agent.rival_coalitions = tuple(r.strip() for r in rival.split(",") if r.strip())

            world.add_agent(agent)
            org_agents[org_name].append(agent_id)
            name_to_id[person["name"]] = agent_id

            agent_meta[agent_id] = {
                "org": org_name,
                "title": person.get("title", ""),
                "sector": org.get("sector", "unknown"),
                "role": social_role,
                "is_llm_agent": True,
                "revenue_influence": org.get("revenue_influence", 0.05),
            }

        # Add non-LLM employees
        employee_count = org.get("employee_count", 10)
        name_pool = [
            "Alex", "Sam", "Jordan", "Casey", "Morgan", "Riley", "Taylor",
            "Drew", "Quinn", "Avery", "Harpreet", "Wei", "Mei", "Aditya",
            "Priya", "Ahmed", "Fatima", "Liam", "Chloe", "Isla", "Marcus",
            "Elena", "Daniel", "Sofia", "Carlos", "Rosa", "Yuki", "Nina",
        ]
        for i in range(employee_count):
            emp_id = f"{org_name.lower().replace(' ', '_')[:12]}_emp_{i}"
            personality = Personality(
                name=f"{rng.choice(name_pool)}_{org_name[:3]}_{i}",
                background=f"Staff at {org_name}, {rng.randint(1, 15)} years",
                temperament=rng.choice(["diligent and quiet", "ambitious mid-career", "steady team player"]),
            )
            schedule = {}
            home = rng.choice(home_locations)
            for h in range(8, 18):
                schedule[h] = hq
            for h in list(range(0, 8)) + list(range(18, 24)):
                schedule[h] = home

            emp_role = "office_worker"
            if org.get("sector", "") in ("mining", "energy", "energy_fossil", "construction"):
                emp_role = "factory_worker"
            elif org.get("sector", "") in ("executive", "legislature", "regulator", "central_bank"):
                emp_role = "government_worker"

            agent = WorldAgent(
                agent_id=emp_id,
                personality=personality,
                schedule=schedule,
                social_role=emp_role,
                debt_pressure=rng.uniform(0.08, 0.28),
                savings_buffer=rng.uniform(0.2, 0.6),
                credit_access=rng.uniform(0.3, 0.7),
                income_level=rng.uniform(0.4, 0.7),
            )

            if coalition:
                agent.coalitions = tuple(c.strip() for c in coalition.split(",") if c.strip())

            world.add_agent(agent)
            org_agents[org_name].append(emp_id)

            # Employee → managed by leaders
            for leader_id in org_agents[org_name][:len(org.get("key_personnel", []))]:
                fabric.add(OrgLink(leader_id, emp_id, "employs", strength=0.7))

    # ── 3. Build population segment agents ──
    for seg in spec.get("population_segments", []):
        work_loc = seg.get("work_location", home_locations[0])
        home_loc = seg.get("home_location", rng.choice(home_locations))

        # Ensure locations exist
        for loc_id in (work_loc, home_loc):
            if loc_id not in world.locations:
                world.add_location(Location(loc_id, loc_id.replace("_", " ").title(), "daily activity"))

        social_role = seg.get("social_role", "community")
        backgrounds = seg.get("backgrounds", [f"Worker, {{years}} years experience"])
        temperaments = seg.get("temperaments", ["steady and pragmatic"])
        debt_lo, debt_hi = seg.get("debt_range", [0.1, 0.3])
        amb_lo, amb_hi = seg.get("ambition_range", [0.2, 0.5])
        count = seg.get("count", 25)
        seg_coalition = seg.get("coalition", "")

        name_pool = [
            "James", "Sarah", "Michael", "Emma", "Daniel", "Olivia", "David", "Chloe",
            "Matt", "Jessica", "Chris", "Sophie", "Mark", "Laura", "Ben", "Mia",
            "Tom", "Hannah", "Luke", "Kate", "Ryan", "Holly", "Josh", "Amy",
            "Harpreet", "Priya", "Wei", "Li", "Ahmed", "Fatima", "Anh", "Minh",
            "Yusuf", "Amira", "Raj", "Mei", "Tariq", "Zara", "Carlos", "Rosa",
        ]

        savings_map = {
            "factory_worker": (0.10, 0.40), "government_worker": (0.25, 0.55),
            "healthcare": (0.20, 0.50), "market_vendor": (0.08, 0.30),
            "community": (0.10, 0.40), "student": (0.03, 0.15),
            "office_professional": (0.30, 0.65), "dock_worker": (0.15, 0.40),
        }
        income_map = {
            "factory_worker": (0.35, 0.65), "government_worker": (0.40, 0.60),
            "healthcare": (0.35, 0.60), "market_vendor": (0.20, 0.40),
            "community": (0.20, 0.45), "student": (0.10, 0.25),
            "office_professional": (0.45, 0.75), "dock_worker": (0.35, 0.55),
        }
        credit_map = {
            "factory_worker": (0.20, 0.50), "government_worker": (0.35, 0.60),
            "healthcare": (0.30, 0.55), "market_vendor": (0.15, 0.40),
            "community": (0.15, 0.40), "student": (0.05, 0.25),
            "office_professional": (0.40, 0.70), "dock_worker": (0.20, 0.45),
        }

        label = seg.get("label", social_role)

        for i in range(count):
            name = f"{rng.choice(name_pool)}_{label[:5]}_{i}"
            agent_id = name.lower().replace(" ", "_")

            years = rng.randint(1, 20)
            bg = rng.choice(backgrounds).format(years=years)
            temperament = rng.choice(temperaments)

            personality = Personality(name=name, background=bg, temperament=temperament)

            schedule = {}
            for h in range(0, 7):
                schedule[h] = home_loc
            for h in range(7, 9):
                schedule[h] = home_loc
            for h in range(9, 17):
                schedule[h] = work_loc
            for h in range(17, 22):
                schedule[h] = rng.choice([home_loc, work_loc])
            for h in range(22, 24):
                schedule[h] = home_loc

            sav_lo, sav_hi = savings_map.get(social_role, (0.15, 0.45))
            inc_lo, inc_hi = income_map.get(social_role, (0.30, 0.50))
            cred_lo, cred_hi = credit_map.get(social_role, (0.20, 0.50))

            agent = WorldAgent(
                agent_id=agent_id,
                personality=personality,
                schedule=schedule,
                social_role=social_role,
                debt_pressure=rng.uniform(debt_lo, debt_hi),
                ambition=rng.uniform(amb_lo, amb_hi),
                savings_buffer=rng.uniform(sav_lo, sav_hi),
                credit_access=rng.uniform(cred_lo, cred_hi),
                income_level=rng.uniform(inc_lo, inc_hi),
            )

            if seg_coalition:
                agent.coalitions = tuple(c.strip() for c in seg_coalition.split(",") if c.strip())

            assign_human_profile(personality, social_role, rng)
            world.add_agent(agent)

            agent_meta[agent_id] = {
                "role": social_role,
                "segment": label,
                "is_llm_agent": False,
            }

            # Link population to relevant orgs (supply/employment)
            for org_name, org_ids in org_agents.items():
                if org_ids and rng.random() < 0.15:
                    fabric.add(OrgLink(org_ids[0], agent_id, "supplies", strength=0.3))

    # ── 4. Supply chain and regulatory links ──
    for org in spec.get("organizations", []):
        org_name = org["name"]
        for target in org.get("supply_chain_to", []):
            if target in org_agents and org_name in org_agents:
                for from_id in org_agents[org_name][:2]:
                    for to_id in org_agents[target][:2]:
                        fabric.add(OrgLink(from_id, to_id, "supplies", strength=0.6))

        for reg in org.get("regulated_by", []):
            if reg in org_agents and org_name in org_agents:
                for reg_id in org_agents[reg][:2]:
                    for co_id in org_agents[org_name][:2]:
                        fabric.add(OrgLink(reg_id, co_id, "regulates", strength=0.5))

    # ── 5. Government → population links ──
    gov_orgs = [o for o in spec.get("organizations", [])
                if o.get("sector", "") in ("executive", "legislature", "government", "opposition")]
    gov_ids = []
    for go in gov_orgs:
        if go["name"] in org_agents:
            gov_ids.extend(org_agents[go["name"]][:4])
    all_pop_ids = [aid for aid, m in agent_meta.items() if not m.get("is_llm_agent")]
    for gid in gov_ids:
        constituents = rng.sample(all_pop_ids, min(50, len(all_pop_ids)))
        for cid in constituents:
            fabric.add(OrgLink(gid, cid, "governs", strength=0.3))

    # ── 6. Seed key relationships from spec ──
    for rel_spec in spec.get("key_relationships", []):
        from_name = rel_spec.get("from", "")
        to_name = rel_spec.get("to", "")
        from_id = name_to_id.get(from_name)
        to_id = name_to_id.get(to_name)
        if from_id and to_id and from_id in world.agents and to_id in world.agents:
            rel = world.relationships.get_or_create(from_id, to_id)
            rel.trust = rel_spec.get("trust", 0.0)
            rel.warmth = rel_spec.get("warmth", 0.0)
            rel.rivalry = rel_spec.get("rivalry", 0.0)
            if rel_spec.get("alliance", 0) > 0:
                rel.alliance_strength = rel_spec["alliance"]
            rel.familiarity = rng.randint(10, 60)

    # ── 7. Seed intra-org relationships ──
    for org_name, agent_ids in org_agents.items():
        for i, aid_a in enumerate(agent_ids[:8]):
            for aid_b in agent_ids[i + 1:8]:
                if rng.random() < 0.5:
                    rel = world.relationships.get_or_create(aid_a, aid_b)
                    rel.trust = rng.uniform(0.1, 0.5)
                    rel.warmth = rng.uniform(0.0, 0.4)
                    rel.familiarity = rng.randint(5, 50)

    # Apply calibrated savings
    try:
        from .calibrated_economy import calibrate_agent_savings
        for agent in world.agents.values():
            calibrate_agent_savings(agent, rng)
    except ImportError:
        pass  # calibrated_economy not available

    shock_desc = spec.get("shock_description", "")

    return GeneratedScenario(
        world=world,
        agent_meta=agent_meta,
        fabric=fabric,
        scenario_spec=spec,
        shock_description=shock_desc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SHOCK BUILDER — converts spec shock events into ExternalSignalPlan
# ═══════════════════════════════════════════════════════════════════════════

def build_shock_from_spec(spec: dict, start_tick: int = 1) -> "ExternalSignalPlan":
    """Build an ExternalSignalPlan from the generated scenario spec."""
    from .world import ScheduledEvent
    from .world_information import ExternalSignalPlan

    events = []
    for evt in spec.get("shock_events", []):
        events.append(ScheduledEvent(
            tick=start_tick + evt.get("tick_offset", 0),
            location=evt.get("location", ""),
            description=evt.get("description", ""),
            emotional_text=evt.get("emotional_text", ""),
            severity=evt.get("severity", 1.5),
            kind="generated_scenario_shock",
        ))

    role_impacts = {}
    for role, impacts in spec.get("role_impacts", {}).items():
        role_impacts[role] = {k: float(v) for k, v in impacts.items()}

    return ExternalSignalPlan(
        source_text=spec.get("shock_description", "Dynamic scenario shock"),
        label=spec.get("scenario_name", "Generated Scenario"),
        kind="generated_scenario_shock",
        severity=spec.get("shock_severity", 2.0),
        start_tick=start_tick,
        scheduled_events=events,
        role_impacts=role_impacts,
        notes=[f"Auto-generated from prediction: {spec.get('scenario_description', '')}"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL API — one call does everything
# ═══════════════════════════════════════════════════════════════════════════

def generate_and_build(
    prediction: str,
    *,
    model: str = "gpt-5-mini",
    seed: int = 42,
    auto_inject_shock: bool = True,
) -> GeneratedScenario:
    """Full pipeline: prediction → OpenAI research → World with agents.

    Args:
        prediction: Natural language prediction (e.g., "Albanese loses the election")
        model: OpenAI model to use for research
        seed: Random seed for deterministic agent generation
        auto_inject_shock: If True, automatically schedule the shock events

    Returns:
        GeneratedScenario with ready-to-simulate World
    """
    # Step 1: Call OpenAI to research and generate character specs
    spec = _call_openai(prediction, model=model)

    # Step 2: Build World from spec
    result = build_world_from_spec(spec, seed=seed)

    # Step 3: Initialize the world (loads SBERT etc.)
    result.world.initialize()

    # Step 4: Inject the shock if requested
    if auto_inject_shock and spec.get("shock_events"):
        from .world_information import apply_external_information
        shock_plan = build_shock_from_spec(spec, start_tick=result.world.tick_count + 1)
        apply_external_information(result.world, shock_plan)

    return result


def generate_spec_only(
    prediction: str,
    *,
    model: str = "gpt-5-mini",
) -> dict:
    """Just generate the scenario spec (for inspection/caching) without building."""
    return _call_openai(prediction, model=model)

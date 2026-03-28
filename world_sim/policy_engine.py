"""Policy Engine — translates government policy shifts into concrete economic effects.

When a new government takes power, they bring specific policy positions that
change the economic landscape for specific groups. This engine:

1. Takes structured policy shifts (from OpenAI research or manual input)
2. Creates PersistentConditions that apply pressure every tick
3. Generates ScheduledEvents showing the narrative consequences
4. Wires into the existing ripple engine so effects cascade person-to-person

Each policy is NOT a one-time shock — it's a sustained change in the rules
of the game. "Scrap the safeguard mechanism" means the renewables sector
feels pain every single day, not just on announcement day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World, ScheduledEvent

from .persistent_conditions import PersistentCondition


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Policy shift data structures
# ---------------------------------------------------------------------------

@dataclass
class PolicyShift:
    """One concrete policy change and its economic consequences."""
    domain: str              # "housing", "energy", "immigration", "labor", "fiscal", etc.
    policy_name: str         # short name: "Scrap renewables target"
    description: str         # what actually changes
    winners: dict[str, float]  # role → benefit magnitude (positive = good for them)
    losers: dict[str, float]   # role → harm magnitude (positive = bad for them)
    market_sectors: dict[str, float]  # sector → impact (-1 to +1)
    implementation_delay_ticks: int = 24  # hours until policy takes effect
    severity: float = 1.0   # how dramatic this shift is
    narrative_events: list[dict] = field(default_factory=list)  # scheduled narrative moments


@dataclass
class PolicyAgenda:
    """Complete policy agenda of an incoming government."""
    government_name: str
    leader_name: str
    policies: list[PolicyShift] = field(default_factory=list)
    overall_ideology: str = ""  # "center-right", "progressive", etc.


# ---------------------------------------------------------------------------
# LLM prompt for policy research
# ---------------------------------------------------------------------------

POLICY_SYSTEM_PROMPT = """You are a policy analyst for an academic economic simulation engine used for scenario planning, risk analysis, and financial research.

Given a political scenario (e.g., "Dutton's Coalition wins the Australian election"), research and return the SPECIFIC POLICY POSITIONS of the incoming government and their CONCRETE ECONOMIC IMPACTS on different groups.

Return VALID JSON matching this schema. No markdown, no commentary.

IMPORTANT:
- Be SPECIFIC about policies, not vague. "Scrap the 82% renewables target" not "change energy policy"
- For each policy, name the WINNERS and LOSERS by occupation/role
- Include the MECHANISM — how does this policy actually affect people's wallets?
- Include realistic magnitudes (0.0-0.3 scale where 0.1 = noticeable, 0.2 = significant, 0.3 = severe)
- Cover: housing, energy, climate, immigration, labor/IR, fiscal/budget, healthcare, education, defence, trade

Valid roles for winners/losers:
  factory_worker, office_professional, government_worker, healthcare, market_vendor,
  community, student, dock_worker, manager, office_worker

JSON SCHEMA:
{
  "government_name": "string",
  "leader_name": "string",
  "overall_ideology": "string",
  "policies": [
    {
      "domain": "housing|energy|climate|immigration|labor|fiscal|healthcare|education|defence|trade|welfare",
      "policy_name": "string — short, specific name",
      "description": "string — 2-3 sentences explaining what changes and WHY it matters economically",
      "winners": {
        "role_name": 0.0-0.3  // positive = how much they benefit (debt relief, income boost, etc.)
      },
      "losers": {
        "role_name": 0.0-0.3  // positive = how much they're harmed (debt increase, income loss, etc.)
      },
      "market_sectors": {
        "sector_name": -1.0 to 1.0  // negative = sector shrinks, positive = sector grows
        // sectors: mining, renewables, construction, banking, retail, healthcare, education, public_service, agriculture, tech, defence, property
      },
      "implementation_delay_ticks": 24-168,  // hours until policy bites (24=next day, 168=week)
      "severity": 0.5-2.5,
      "narrative_events": [
        {
          "tick_offset": 0-336,
          "location": "string — where this plays out",
          "description": "string — what happens in the real world",
          "emotional_text": "string — how it FEELS to people affected",
          "severity": 0.5-3.0
        }
      ]
    }
  ]
}"""


POLICY_USER_PROMPT = """Research the SPECIFIC policy positions and their concrete economic impacts for this scenario:

"{prediction}"

For EACH major policy domain (housing, energy, immigration, labor/IR, fiscal, healthcare, education, defence, trade), identify:
1. What SPECIFICALLY will change (name the actual policy)
2. WHO wins and who loses (by occupation)
3. HOW MUCH — in concrete economic terms (rent goes up X%, job losses of Y, etc.)
4. WHEN it hits (implementation timeline)
5. The narrative moments when people feel it

Be specific and honest about the economic mechanisms. "Scrap the safeguard mechanism" means renewable energy companies lose project certainty, workers in that sector face job risk, but coal miners keep their jobs. That's a concrete chain.

Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Call OpenAI for policy research
# ---------------------------------------------------------------------------

def research_policies(prediction: str, model: str = "gpt-5-mini") -> PolicyAgenda:
    """Call OpenAI to research specific policy positions and impacts."""
    import json
    import os
    import time
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    body = {
        "model": model,
        "max_completion_tokens": 16000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": POLICY_USER_PROMPT.format(prediction=prediction)},
        ],
    }
    if not model.startswith("gpt-5"):
        body["temperature"] = 0.3

    payload = json.dumps(body).encode("utf-8")
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(3):
        try:
            with urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if not text:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise RuntimeError("Empty response from model")
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            content = json.loads(text)
            return _parse_policy_agenda(content)
        except HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except RuntimeError:
            raise
        except Exception:
            if attempt < 2:
                time.sleep(1)
                continue
            raise

    raise RuntimeError("Policy research failed after retries")


def _parse_policy_agenda(raw: dict) -> PolicyAgenda:
    """Parse OpenAI response into PolicyAgenda."""
    agenda = PolicyAgenda(
        government_name=raw.get("government_name", ""),
        leader_name=raw.get("leader_name", ""),
        overall_ideology=raw.get("overall_ideology", ""),
    )
    for p in raw.get("policies", []):
        shift = PolicyShift(
            domain=p.get("domain", "unknown"),
            policy_name=p.get("policy_name", ""),
            description=p.get("description", ""),
            winners={k: float(v) for k, v in p.get("winners", {}).items()},
            losers={k: float(v) for k, v in p.get("losers", {}).items()},
            market_sectors={k: float(v) for k, v in p.get("market_sectors", {}).items()},
            implementation_delay_ticks=p.get("implementation_delay_ticks", 48),
            severity=p.get("severity", 1.0),
            narrative_events=p.get("narrative_events", []),
        )
        agenda.policies.append(shift)
    return agenda


# ---------------------------------------------------------------------------
# Apply policy agenda to the world
# ---------------------------------------------------------------------------

def apply_policy_agenda(world: "World", agenda: PolicyAgenda) -> dict:
    """Translate a policy agenda into persistent conditions, events, and agent impacts.

    Each policy becomes:
      1. A PersistentCondition applying per-tick pressure to affected roles
      2. ScheduledEvents at specific locations showing narrative consequences
      3. Direct agent state changes for winners and losers
    """
    from .world import ScheduledEvent

    results = {
        "policies_applied": [],
        "total_persistent_conditions": 0,
        "total_events_scheduled": 0,
        "total_agents_affected": 0,
    }

    for policy in agenda.policies:
        policy_result = _apply_single_policy(world, policy, agenda.leader_name)
        results["policies_applied"].append(policy_result)
        results["total_persistent_conditions"] += 1
        results["total_events_scheduled"] += policy_result["events_scheduled"]
        results["total_agents_affected"] += policy_result["agents_affected"]

    return results


def _apply_single_policy(world: "World", policy: PolicyShift, leader_name: str) -> dict:
    """Apply one policy shift to the world."""
    from .world import ScheduledEvent

    start_tick = world.tick_count + policy.implementation_delay_ticks

    # ── 1. Create persistent condition ──
    # Losers get economic_drag, tension_drag, dread_drag
    # Winners get negative drag (i.e., relief)
    role_multipliers = {}

    # Losers: positive multiplier = more pressure (reduced to prevent saturation)
    for role, magnitude in policy.losers.items():
        role_multipliers[role] = role_multipliers.get(role, 0.0) + magnitude * 2.0

    # Winners: negative multiplier = relief
    for role, magnitude in policy.winners.items():
        role_multipliers[role] = role_multipliers.get(role, 0.0) - magnitude * 1.5

    # Calculate drag rates based on severity
    # Reduced from original to prevent saturation — let ripple engine do the cascading
    base_economic = 0.0005 * policy.severity
    base_tension = 0.0003 * policy.severity
    base_dread = 0.0002 * policy.severity

    condition = PersistentCondition(
        label=f"Policy: {policy.policy_name}",
        kind=f"policy_{policy.domain}",
        severity=policy.severity,
        start_tick=start_tick,
        economic_drag=base_economic,
        tension_drag=base_tension,
        dread_drag=base_dread,
        role_multipliers=role_multipliers,
        rally_strength=0.0,
    )

    if not hasattr(world, "_persistent_conditions"):
        world._persistent_conditions = []
    world._persistent_conditions.append(condition)

    # ── 2. Schedule narrative events ──
    events_scheduled = 0
    for evt in policy.narrative_events:
        loc = evt.get("location", "")
        # Ensure location exists
        if loc and loc not in world.locations:
            from .world import Location
            world.add_location(Location(loc, loc.replace("_", " ").title(), "policy impact zone"))

        if loc:
            world.schedule_event(ScheduledEvent(
                tick=start_tick + evt.get("tick_offset", 0),
                location=loc,
                description=evt.get("description", ""),
                emotional_text=evt.get("emotional_text", ""),
                severity=evt.get("severity", 1.5),
                kind=f"policy_{policy.domain}",
            ))
            events_scheduled += 1

    # ── 3. Apply immediate impacts to agents ──
    agents_affected = 0

    for agent in world.agents.values():
        role = agent.social_role

        # Apply loser impacts (softened to leave room for LLM agent cascades)
        if role in policy.losers:
            magnitude = policy.losers[role]
            agent.debt_pressure = _clamp(agent.debt_pressure + magnitude * 0.15)
            agent.expectation_pessimism = _clamp(
                getattr(agent, "expectation_pessimism", 0) + magnitude * 0.25
            )
            agent.heart.tension = _clamp(agent.heart.tension + magnitude * 0.1)
            agent.dread_pressure = _clamp(agent.dread_pressure + magnitude * 0.12)
            agent.heart.valence = max(0.1, agent.heart.valence - magnitude * 0.08)

            # Add wound for significant impacts
            if magnitude > 0.15:
                agent.heart.wounds.append((magnitude * 0.04, 0.997))

            agent.add_memory(
                world.tick_count,
                f"[policy] {leader_name}'s {policy.policy_name}: {policy.description[:80]}",
            )
            agents_affected += 1

        # Apply winner impacts
        elif role in policy.winners:
            magnitude = policy.winners[role]
            agent.debt_pressure = _clamp(agent.debt_pressure - magnitude * 0.3)
            agent.expectation_pessimism = max(
                0, getattr(agent, "expectation_pessimism", 0) - magnitude * 0.5
            )
            agent.heart.tension = max(0, agent.heart.tension - magnitude * 0.2)
            agent.heart.valence = min(0.9, agent.heart.valence + magnitude * 0.15)

            agent.add_memory(
                world.tick_count,
                f"[policy] {leader_name}'s {policy.policy_name} is good for people like me",
            )
            agents_affected += 1

    return {
        "policy": policy.policy_name,
        "domain": policy.domain,
        "description": policy.description,
        "severity": policy.severity,
        "winners": list(policy.winners.keys()),
        "losers": list(policy.losers.keys()),
        "market_sectors": policy.market_sectors,
        "events_scheduled": events_scheduled,
        "agents_affected": agents_affected,
        "persistent_condition": condition.label,
    }


# ---------------------------------------------------------------------------
# Market sector tracker — tracks which sectors are growing/shrinking
# ---------------------------------------------------------------------------

@dataclass
class SectorTracker:
    """Tracks market sector health based on policy impacts."""
    sectors: dict[str, float] = field(default_factory=lambda: {
        "mining": 0.0, "renewables": 0.0, "construction": 0.0,
        "banking": 0.0, "retail": 0.0, "healthcare": 0.0,
        "education": 0.0, "public_service": 0.0, "agriculture": 0.0,
        "tech": 0.0, "defence": 0.0, "property": 0.0,
    })

    def apply_policy(self, policy: PolicyShift):
        for sector, impact in policy.market_sectors.items():
            if sector in self.sectors:
                self.sectors[sector] = _clamp(self.sectors[sector] + impact, -1.0, 1.0)

    def get_report(self) -> dict:
        booming = {k: v for k, v in self.sectors.items() if v > 0.1}
        struggling = {k: v for k, v in self.sectors.items() if v < -0.1}
        return {
            "booming": dict(sorted(booming.items(), key=lambda x: -x[1])),
            "struggling": dict(sorted(struggling.items(), key=lambda x: x[1])),
            "all_sectors": {k: round(v, 3) for k, v in sorted(self.sectors.items(), key=lambda x: -x[1])},
        }

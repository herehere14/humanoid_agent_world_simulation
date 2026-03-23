"""LLM Agency — LLM agents make REAL decisions that drive the simulation.

This replaces the old llm_chooser.py which treated the LLM as a narrator.
Now the LLM is the decision-maker for key agents in every organization:

  MANAGERS decide: raise prices, cut staff, invest, absorb losses, lobby government
  GOVERNMENT WORKERS decide: approve stimulus, impose restrictions, investigate, do nothing
  VENDORS decide: raise prices, find new suppliers, close shop, extend credit to loyal customers
  WORKERS decide: organize, accept cuts, quit, side-hustle, confront management
  COMMUNITY LEADERS decide: call mutual aid, organize protest, lobby officials, stockpile

The LLM's decision feeds directly into the ripple engine — when the LLM decides
"cut 3 workers", those specific workers lose income, reduce spending, vendors
lose revenue, and the cascade continues.

Architecture:
  - Each org-role gets a DECISION PROMPT with their real situation
  - The LLM returns a structured decision with CONCRETE CONSEQUENCES
  - The consequences are applied directly to connected agents via org links
  - The deterministic system handles agents who aren't LLM-promoted
  - LLM decisions are BOUNDED — they can't invent money or break physics
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World
    from .world_agent import WorldAgent
    from .ripple_engine import OrganizationalFabric, RippleEvent


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Decision schemas per organizational role
# ---------------------------------------------------------------------------

MANAGER_DECISIONS = [
    "CUT_WORKERS",      # lay off N workers → they lose income, spend less
    "CUT_HOURS",        # reduce hours → workers lose partial income
    "RAISE_PRICES",     # pass costs to customers → customers pay more
    "ABSORB_LOSSES",    # eat the cost → own debt_pressure rises, no cascade
    "LOBBY_GOVERNMENT", # pressure government for relief → may trigger policy response
    "INVEST_DESPITE",   # double down / hire → costs now but builds long-term
    "HOLD_STEADY",      # do nothing, wait and see
]

VENDOR_DECISIONS = [
    "RAISE_PRICES",       # pass costs to customers
    "FIND_NEW_SUPPLIER",  # absorb short-term cost for better margin later
    "EXTEND_CREDIT",      # give loyal customers credit → builds trust, takes risk
    "REDUCE_INVENTORY",   # stock less → fewer choices for customers
    "CLOSE_TEMPORARILY",  # shut down until conditions improve → workers idle
    "HOLD_STEADY",
]

GOVERNMENT_DECISIONS = [
    "APPROVE_STIMULUS",      # distribute relief → reduces debt for many, costs budget
    "IMPOSE_RESTRICTIONS",   # lockdowns/regulations → controls crisis but hurts economy
    "INVESTIGATE_COMPANIES", # accountability → builds public trust, spooks businesses
    "PUBLIC_REASSURANCE",    # messaging only → small trust boost, no material change
    "EMERGENCY_FUND",        # targeted aid to most distressed → helps few a lot
    "DO_NOTHING",            # bureaucratic inaction → trust erodes
]

WORKER_DECISIONS = [
    "ORGANIZE_UNION",     # build collective power → costs time, builds alliance
    "ACCEPT_CUTS",        # comply → less conflict, more debt
    "CONFRONT_MANAGEMENT",# demand better terms → may backfire or succeed
    "SIDE_HUSTLE",        # find alternate income → reduces dependence
    "QUIT",               # leave → lose income now, free later
    "HOLD_STEADY",
]

COMMUNITY_DECISIONS = [
    "ORGANIZE_MUTUAL_AID",  # pool resources → helps neighborhood, costs organizer
    "ORGANIZE_PROTEST",     # public action → draws attention, creates conflict
    "LOBBY_OFFICIALS",      # work channels → may trigger government response
    "STOCKPILE",            # hoard resources → helps self, hurts others
    "SUPPORT_NEIGHBORS",    # direct help → builds bonds
    "HOLD_STEADY",
]


def _get_decision_menu(role: str) -> list[str]:
    return {
        "manager": MANAGER_DECISIONS,
        "office_professional": MANAGER_DECISIONS,
        "market_vendor": VENDOR_DECISIONS,
        "bartender": VENDOR_DECISIONS,
        "government_worker": GOVERNMENT_DECISIONS,
        "factory_worker": WORKER_DECISIONS,
        "dock_worker": WORKER_DECISIONS,
        "office_worker": WORKER_DECISIONS,
        "community": COMMUNITY_DECISIONS,
        "student": WORKER_DECISIONS,
        "teacher": COMMUNITY_DECISIONS,
        "healthcare": COMMUNITY_DECISIONS,
        "retiree": COMMUNITY_DECISIONS,
    }.get(role, ["HOLD_STEADY"])


# ---------------------------------------------------------------------------
# Decision prompt builder
# ---------------------------------------------------------------------------

@dataclass
class AgentSituation:
    """Everything the LLM needs to make a decision."""
    agent_id: str
    name: str
    role: str
    background: str
    personality_summary: str
    financial_state: str
    emotional_state: str
    recent_events: str
    people_affected: str  # who depends on this agent
    decision_options: list[str] = field(default_factory=list)


def _build_situation(agent: "WorldAgent", fabric: "OrganizationalFabric", world: "World") -> AgentSituation:
    """Build the full situation context for an LLM decision."""
    profile = agent.get_human_profile()

    # Who depends on this agent?
    dependents = fabric.get_dependents(agent.agent_id)
    dep_lines = []
    for link in dependents[:8]:
        if link.to_id in world.agents:
            other = world.agents[link.to_id]
            dep_lines.append(
                f"  - {other.personality.name} ({other.social_role}): "
                f"you {link.link_type} them, "
                f"their debt={other.debt_pressure:.2f}, tension={other.heart.tension:.2f}"
            )

    # Recent memories
    memories = agent.get_recent_memories(5)
    mem_text = "\n".join(f"  - {m.description}" for m in memories) if memories else "Nothing notable."

    return AgentSituation(
        agent_id=agent.agent_id,
        name=agent.personality.name,
        role=agent.social_role,
        background=agent.personality.background,
        personality_summary=(
            f"Copes by: {profile['coping_style']}. "
            f"Fears: {profile['threat_lens']}. "
            f"Needs: {profile['core_need']}. "
            f"Self-story: {profile['self_story']}. "
            f"Fights by: {profile['conflict_style']}."
        ),
        financial_state=(
            f"Debt pressure: {agent.debt_pressure:.2f}/1.0 "
            f"({'critical' if agent.debt_pressure > 0.6 else 'high' if agent.debt_pressure > 0.3 else 'manageable'}). "
            f"Dread: {agent.dread_pressure:.2f}. "
            f"Expectations: {'pessimistic' if getattr(agent, 'expectation_pessimism', 0) > 0.2 else 'cautious' if getattr(agent, 'expectation_pessimism', 0) > 0.1 else 'neutral'}."
        ),
        emotional_state=(
            f"Feeling: {agent.heart.internal_emotion} (showing: {agent.heart.surface_emotion}). "
            f"Tension: {agent.heart.tension:.2f}. Energy: {agent.heart.energy:.2f}. "
            f"Wounds: {len(agent.heart.wounds)}."
        ),
        recent_events=mem_text,
        people_affected="\n".join(dep_lines) if dep_lines else "Nobody directly depends on your decisions.",
        decision_options=_get_decision_menu(agent.social_role),
    )


def _build_decision_prompt(situation: AgentSituation) -> str:
    options_text = "\n".join(f"  - {opt}" for opt in situation.decision_options)

    return f"""You are {situation.name}, a {situation.role}.
Background: {situation.background}
Personality: {situation.personality_summary}

YOUR SITUATION RIGHT NOW:
Financial: {situation.financial_state}
Emotional: {situation.emotional_state}

RECENT EVENTS IN YOUR LIFE:
{situation.recent_events}

PEOPLE WHO DEPEND ON YOUR DECISIONS:
{situation.people_affected}

You must make a CONCRETE DECISION about what to do. This is not hypothetical — your choice will directly affect the people listed above.

YOUR OPTIONS:
{options_text}

Think about:
- Your personality and coping style
- Your financial pressure vs. the impact on people who depend on you
- Whether you protect yourself or protect others
- Your self-story: are you a {situation.personality_summary.split('Self-story: ')[1].split('.')[0]}?

Return ONLY valid JSON:
{{
  "decision": "<one of the options above>",
  "magnitude": <0.1 to 1.0, how aggressively you act>,
  "reasoning": "<2-3 sentences: WHY you chose this, what you're protecting>",
  "who_you_help": "<name of someone you're trying to protect, or 'myself'>",
  "who_gets_hurt": "<name of someone who will suffer from this, or 'nobody'>",
  "what_you_say": "<what you actually say out loud to the people around you>",
  "what_you_think": "<what you're really thinking but not saying>"
}}"""


# ---------------------------------------------------------------------------
# Decision parser
# ---------------------------------------------------------------------------

@dataclass
class LLMAgentDecision:
    agent_id: str
    agent_name: str
    role: str
    decision: str = "HOLD_STEADY"
    magnitude: float = 0.5
    reasoning: str = ""
    who_helps: str = ""
    who_hurts: str = ""
    speech: str = ""
    private_thought: str = ""
    valid: bool = True
    latency_ms: int = 0


def _parse_decision(raw: str, agent_id: str, agent_name: str, role: str, allowed: list[str]) -> LLMAgentDecision:
    d = LLMAgentDecision(agent_id=agent_id, agent_name=agent_name, role=role)
    text = raw.strip()
    start = text.find("{")
    if start == -1:
        d.valid = False
        return d
    end = text.rfind("}")
    json_str = text[start:end + 1] if end > start else text[start:]

    # Try to repair truncated JSON (missing closing brace/quotes)
    if not json_str.endswith("}"):
        # Truncated — try adding closing quote + brace
        json_str = json_str.rstrip()
        if json_str.endswith(","):
            json_str = json_str[:-1]
        if not json_str.endswith('"'):
            json_str += '"'
        json_str += "}"

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try even more aggressive repair
        try:
            # Close any open strings and objects
            repaired = json_str
            open_quotes = repaired.count('"') % 2
            if open_quotes:
                repaired += '"'
            while repaired.count("{") > repaired.count("}"):
                repaired += "}"
            data = json.loads(repaired)
        except json.JSONDecodeError:
            d.valid = False
            return d

    decision = str(data.get("decision", "HOLD_STEADY")).strip().upper()
    if decision not in allowed:
        for a in allowed:
            if a in decision or decision in a:
                decision = a
                break
        else:
            decision = "HOLD_STEADY"

    d.decision = decision
    d.magnitude = max(0.1, min(1.0, float(data.get("magnitude", 0.5) or 0.5)))
    d.reasoning = str(data.get("reasoning", ""))[:300]
    d.who_helps = str(data.get("who_you_help", ""))[:100]
    d.who_hurts = str(data.get("who_gets_hurt", ""))[:100]
    d.speech = str(data.get("what_you_say", ""))[:300]
    d.private_thought = str(data.get("what_you_think", ""))[:300]
    return d


# ---------------------------------------------------------------------------
# Apply decisions — wire into ripple engine
# ---------------------------------------------------------------------------

def _apply_decision(
    decision: LLMAgentDecision,
    agent: "WorldAgent",
    fabric: "OrganizationalFabric",
    world: "World",
) -> list["RippleEvent"]:
    """Apply an LLM decision and return the ripple events it creates."""
    from .ripple_engine import RippleEvent

    events: list[RippleEvent] = []
    tick = world.tick_count
    dependents = fabric.get_dependents(agent.agent_id)
    mag = decision.magnitude

    # Record speech and thought
    agent.last_speech = decision.speech
    if decision.private_thought:
        agent.add_memory(tick, f"[decision:{decision.decision}] {decision.reasoning}")
        if agent.memory:
            agent.memory[-1].interpretation = decision.private_thought

    if decision.decision == "CUT_WORKERS":
        employees = [l for l in dependents if l.link_type == "employs"]
        n_cut = max(1, int(len(employees) * mag * 0.4))
        import random
        targets = random.sample(employees, min(n_cut, len(employees)))
        for link in targets:
            if link.to_id not in world.agents: continue
            target = world.agents[link.to_id]
            target.debt_pressure = _clamp(target.debt_pressure + 0.25 * mag)
            target.dread_pressure = _clamp(target.dread_pressure + 0.1 * mag)
            target.heart.wounds.append((0.06 * mag, 0.995))
            target.add_memory(tick, f"{agent.personality.name} laid me off — income gone")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=link.to_id, target_name=target.personality.name,
                action=f"laid off worker (LLM decision, magnitude={mag:.1f})",
                consequence=f"lost income, debt+{0.25*mag:.2f}, dread+{0.1*mag:.2f}",
                mechanism="LLM:employs", debt_delta=0.25 * mag,
            ))
        # Survivor anxiety
        for link in employees:
            if link.to_id in world.agents and link not in targets:
                world.agents[link.to_id].dread_pressure = _clamp(
                    world.agents[link.to_id].dread_pressure + 0.04 * mag)

    elif decision.decision == "CUT_HOURS":
        employees = [l for l in dependents if l.link_type == "employs"]
        for link in employees:
            if link.to_id not in world.agents: continue
            target = world.agents[link.to_id]
            cut = 0.08 * mag
            target.debt_pressure = _clamp(target.debt_pressure + cut)
            target.add_memory(tick, f"{agent.personality.name} cut hours — paycheck shrinking")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=link.to_id, target_name=target.personality.name,
                action=f"cut hours (LLM decision)",
                consequence=f"reduced income, debt+{cut:.3f}",
                mechanism="LLM:employs", debt_delta=cut,
            ))

    elif decision.decision == "RAISE_PRICES":
        customers = [l for l in dependents if l.link_type == "supplies"]
        for link in customers:
            if link.to_id not in world.agents: continue
            target = world.agents[link.to_id]
            hike = 0.04 * mag * link.strength
            target.debt_pressure = _clamp(target.debt_pressure + hike)
            target.add_memory(tick, f"Prices up at {agent.personality.name}'s — costs climbing")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=link.to_id, target_name=target.personality.name,
                action=f"raised prices (LLM decision)",
                consequence=f"paying more, debt+{hike:.3f}",
                mechanism="LLM:supplies", debt_delta=hike,
            ))

    elif decision.decision == "ABSORB_LOSSES":
        agent.debt_pressure = _clamp(agent.debt_pressure + 0.08 * mag)
        agent.add_memory(tick, "Absorbing losses to protect my people — but the pressure is building")

    elif decision.decision == "LOBBY_GOVERNMENT" or decision.decision == "LOBBY_OFFICIALS":
        gov_links = [l for l in fabric.links_of_type("governs") if l.from_id in world.agents]
        if gov_links:
            import random
            gov_link = random.choice(gov_links)
            gov = world.agents[gov_link.from_id]
            gov.heart.tension = _clamp(gov.heart.tension + 0.04 * mag)
            gov.add_memory(tick, f"{agent.personality.name} lobbied me for economic relief")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=gov_link.from_id, target_name=gov.personality.name,
                action=f"lobbied for relief (LLM decision)",
                consequence=f"government feels pressure to act",
                mechanism="LLM:lobbies",
            ))

    elif decision.decision == "APPROVE_STIMULUS":
        governed = [l for l in dependents if l.link_type == "governs"]
        distressed = [l for l in governed if l.to_id in world.agents and world.agents[l.to_id].debt_pressure > 0.25]
        relief = 0.06 * mag
        for link in distressed[:10]:
            target = world.agents[link.to_id]
            target.debt_pressure = _clamp(target.debt_pressure - relief)
            target.add_memory(tick, f"Government relief from {agent.personality.name} — some breathing room")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=link.to_id, target_name=target.personality.name,
                action=f"approved stimulus relief (LLM decision)",
                consequence=f"received relief, debt-{relief:.3f}",
                mechanism="LLM:governs", debt_delta=-relief,
            ))

    elif decision.decision == "IMPOSE_RESTRICTIONS":
        governed = [l for l in dependents if l.link_type == "governs"]
        for link in governed[:15]:
            if link.to_id not in world.agents: continue
            target = world.agents[link.to_id]
            if target.social_role in ("market_vendor", "bartender", "factory_worker"):
                target.debt_pressure = _clamp(target.debt_pressure + 0.04 * mag)
            target.dread_pressure = _clamp(target.dread_pressure - 0.02 * mag)
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=link.to_id, target_name=target.personality.name,
                action=f"imposed restrictions (LLM decision)",
                consequence=f"safer but economically squeezed",
                mechanism="LLM:governs",
            ))

    elif decision.decision == "INVESTIGATE_COMPANIES":
        managers = [a for a in world.agents.values() if a.social_role == "manager"]
        for mgr in managers[:5]:
            mgr.secret_pressure = _clamp(mgr.secret_pressure + 0.06 * mag)
            mgr.heart.tension = _clamp(mgr.heart.tension + 0.04 * mag)
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=mgr.agent_id, target_name=mgr.personality.name,
                action=f"launched investigation (LLM decision)",
                consequence=f"corporate executives under scrutiny",
                mechanism="LLM:investigates",
            ))

    elif decision.decision == "ORGANIZE_UNION" or decision.decision == "ORGANIZE_PROTEST":
        # Build alliance with nearby workers
        nearby = [
            a for a in world.agents.values()
            if a.location == agent.location and a.agent_id != agent.agent_id
            and a.social_role in ("factory_worker", "dock_worker", "office_worker", "community", "student")
        ][:8]
        for other in nearby:
            rel = world.relationships.get_or_create(agent.agent_id, other.agent_id)
            rel.alliance_strength = min(1.0, rel.alliance_strength + 0.06 * mag)
            rel.trust = min(1.0, rel.trust + 0.03 * mag)
            other.add_memory(tick, f"{agent.personality.name} is organizing — asked me to join")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=other.agent_id, target_name=other.personality.name,
                action=f"organizing collective action (LLM decision)",
                consequence=f"alliance strengthened, trust built",
                mechanism="LLM:organizes",
            ))

    elif decision.decision == "ORGANIZE_MUTUAL_AID":
        nearby_distressed = [
            a for a in world.agents.values()
            if a.location == agent.location and a.debt_pressure > 0.3 and a.agent_id != agent.agent_id
        ][:6]
        for other in nearby_distressed:
            relief = 0.03 * mag
            other.debt_pressure = _clamp(other.debt_pressure - relief)
            other.heart.tension = max(0.0, other.heart.tension - 0.02)
            rel = world.relationships.get_or_create(agent.agent_id, other.agent_id)
            rel.warmth = min(1.0, rel.warmth + 0.04)
            other.add_memory(tick, f"{agent.personality.name} organized mutual aid — I got help")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=other.agent_id, target_name=other.personality.name,
                action=f"organized mutual aid (LLM decision)",
                consequence=f"received help, debt-{relief:.3f}",
                mechanism="LLM:mutual_aid", debt_delta=-relief,
            ))

    elif decision.decision == "CONFRONT_MANAGEMENT":
        managers = [
            a for a in world.agents.values()
            if a.social_role in ("manager", "office_professional") and a.location == agent.location
        ][:2]
        for mgr in managers:
            rel = world.relationships.get_or_create(agent.agent_id, mgr.agent_id)
            world.relationships.set_grievance(mgr.agent_id, agent.agent_id,
                world.relationships.get_grievance(mgr.agent_id, agent.agent_id) + 0.05 * mag)
            mgr.heart.tension = _clamp(mgr.heart.tension + 0.05 * mag)
            mgr.add_memory(tick, f"{agent.personality.name} confronted me about conditions")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=mgr.agent_id, target_name=mgr.personality.name,
                action=f"confronted management (LLM decision)",
                consequence=f"management feels pressure, tension rises",
                mechanism="LLM:confronts",
            ))

    elif decision.decision == "EXTEND_CREDIT":
        customers = [l for l in dependents if l.link_type == "supplies"]
        distressed = [l for l in customers if l.to_id in world.agents and world.agents[l.to_id].debt_pressure > 0.3]
        for link in distressed[:4]:
            target = world.agents[link.to_id]
            target.debt_pressure = _clamp(target.debt_pressure - 0.03 * mag)
            agent.debt_pressure = _clamp(agent.debt_pressure + 0.02 * mag)  # vendor takes risk
            rel = world.relationships.get_or_create(agent.agent_id, link.to_id)
            rel.warmth = min(1.0, rel.warmth + 0.05)
            world.relationships.adjust_debt(link.to_id, agent.agent_id, 0.03 * mag)
            target.add_memory(tick, f"{agent.personality.name} extended me credit — I owe them")
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=link.to_id, target_name=target.personality.name,
                action=f"extended credit to loyal customer (LLM decision)",
                consequence=f"short-term relief but now in debt to vendor",
                mechanism="LLM:extends_credit", debt_delta=-0.03 * mag,
            ))

    elif decision.decision == "SUPPORT_NEIGHBORS":
        nearby = [
            a for a in world.agents.values()
            if a.location == agent.location and a.agent_id != agent.agent_id
        ][:4]
        for other in nearby:
            other.heart.tension = max(0.0, other.heart.tension - 0.02 * mag)
            rel = world.relationships.get_or_create(agent.agent_id, other.agent_id)
            rel.warmth = min(1.0, rel.warmth + 0.03)
            events.append(RippleEvent(
                tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                target_id=other.agent_id, target_name=other.personality.name,
                action=f"checked on neighbor (LLM decision)",
                consequence=f"tension eased, bond strengthened",
                mechanism="LLM:supports",
            ))

    return events


# ---------------------------------------------------------------------------
# LLM Agency Engine — the main orchestrator
# ---------------------------------------------------------------------------

class LLMAgencyEngine:
    """Makes LLM calls for key agents and applies their decisions to the world.

    Unlike the old LLM chooser, this:
    - Calls the LLM for agents in EVERY organization/role, not just top-salience
    - Gives role-specific decision menus (managers get different options than workers)
    - Wires decisions directly into the ripple engine
    - Spaces calls out (not every tick) to manage API costs
    """

    def __init__(self, api_key: str, model: str = "gpt-5-mini", fabric: "OrganizationalFabric" = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fabric = fabric
        self.total_calls = 0
        self.total_decisions = 0
        self.decision_log: list[dict] = []
        self._last_decision_tick: dict[str, int] = {}

    def tick(self, world: "World", max_calls: int = 5) -> list["RippleEvent"]:
        """Select key agents, get LLM decisions, apply to world.

        Called every tick but only actually calls the LLM for agents who:
        1. Are in a position of influence (have dependents in org fabric)
        2. Are under significant pressure (state changed recently)
        3. Haven't made a decision in the last 24 ticks
        """
        if self.fabric is None:
            return []

        from .ripple_engine import RippleEvent

        tick = world.tick_count
        candidates = []

        for agent_id, agent in world.agents.items():
            # Cooldown: 24 ticks (1 day) between decisions per agent
            if self._last_decision_tick.get(agent_id, 0) + 24 > tick:
                continue

            # Must have people who depend on them
            dependents = self.fabric.get_dependents(agent_id)
            if not dependents:
                continue

            # Must be under meaningful pressure
            pressure = agent.debt_pressure + agent.dread_pressure + agent.heart.tension
            if pressure < 0.4:
                continue

            # Priority: most dependents × most pressure
            priority = len(dependents) * pressure
            candidates.append((priority, agent_id))

        # Sort by priority, take top N
        candidates.sort(reverse=True)
        selected = candidates[:max_calls]

        all_events: list[RippleEvent] = []

        for _, agent_id in selected:
            agent = world.agents[agent_id]
            situation = _build_situation(agent, self.fabric, world)
            prompt = _build_decision_prompt(situation)

            t0 = time.time()
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=500,
                    reasoning={"effort": "low"},
                )
                raw = (resp.output_text or "").strip()
                latency = int((time.time() - t0) * 1000)
            except Exception as e:
                continue

            self.total_calls += 1
            decision = _parse_decision(
                raw, agent_id, agent.personality.name,
                agent.social_role, situation.decision_options,
            )
            decision.latency_ms = latency

            if not decision.valid:
                continue

            self.total_decisions += 1
            self._last_decision_tick[agent_id] = tick

            # Apply decision → get ripple events
            ripple_events = _apply_decision(decision, agent, self.fabric, world)
            all_events.extend(ripple_events)

            self.decision_log.append({
                "tick": tick,
                "time": world.time_str,
                "agent": decision.agent_name,
                "role": decision.role,
                "decision": decision.decision,
                "magnitude": round(decision.magnitude, 2),
                "reasoning": decision.reasoning[:100],
                "who_helps": decision.who_helps,
                "who_hurts": decision.who_hurts,
                "speech": decision.speech[:100],
                "thought": decision.private_thought[:100],
                "ripple_count": len(ripple_events),
                "latency_ms": latency,
            })

        return all_events

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_decisions": self.total_decisions,
            "recent_decisions": self.decision_log[-10:],
        }

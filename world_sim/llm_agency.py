"""LLM Agency — LLM agents make FREEFORM decisions and communicate with each other.

Agents are NOT given a menu of options. They describe what they do in their
own words, specify who they're talking to, and define the concrete consequences.

The engine extracts structured impacts from the freeform response and applies
them through the ripple engine. Agents can also send messages to specific other
agents — those messages arrive in the recipient's inbox and become part of their
context on their next decision tick.

This creates genuinely emergent behavior: agents invent actions nobody coded for,
coordinate with each other, form alliances, make threats, negotiate deals,
leak information, or do anything a real person would do.

Architecture:
  - Each agent gets their REAL situation + inbox of messages from other agents
  - The LLM returns a FREEFORM action + structured consequences + messages to others
  - Consequences are applied directly to connected agents via org links
  - Messages are delivered to recipients' inboxes for their next decision
  - The deterministic system handles agents who aren't LLM-promoted
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


# Legacy menus kept for backwards compat — no longer used in freeform mode
MANAGER_DECISIONS = ["CUT_WORKERS", "CUT_HOURS", "RAISE_PRICES", "ABSORB_LOSSES", "LOBBY_GOVERNMENT", "INVEST_DESPITE", "HOLD_STEADY"]
VENDOR_DECISIONS = ["RAISE_PRICES", "FIND_NEW_SUPPLIER", "EXTEND_CREDIT", "REDUCE_INVENTORY", "CLOSE_TEMPORARILY", "HOLD_STEADY"]
GOVERNMENT_DECISIONS = ["APPROVE_STIMULUS", "IMPOSE_RESTRICTIONS", "INVESTIGATE_COMPANIES", "PUBLIC_REASSURANCE", "EMERGENCY_FUND", "DO_NOTHING"]
WORKER_DECISIONS = ["ORGANIZE_UNION", "ACCEPT_CUTS", "CONFRONT_MANAGEMENT", "SIDE_HUSTLE", "QUIT", "HOLD_STEADY"]
COMMUNITY_DECISIONS = ["ORGANIZE_MUTUAL_AID", "ORGANIZE_PROTEST", "LOBBY_OFFICIALS", "STOCKPILE", "SUPPORT_NEIGHBORS", "HOLD_STEADY"]

def _get_decision_menu(role: str) -> list[str]:
    return {"manager": MANAGER_DECISIONS, "office_professional": MANAGER_DECISIONS,
            "market_vendor": VENDOR_DECISIONS, "bartender": VENDOR_DECISIONS,
            "government_worker": GOVERNMENT_DECISIONS, "factory_worker": WORKER_DECISIONS,
            "dock_worker": WORKER_DECISIONS, "office_worker": WORKER_DECISIONS,
            "community": COMMUNITY_DECISIONS, "student": WORKER_DECISIONS,
            "teacher": COMMUNITY_DECISIONS, "healthcare": COMMUNITY_DECISIONS,
            "retiree": COMMUNITY_DECISIONS}.get(role, ["HOLD_STEADY"])


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
    decision_options: list[str] = field(default_factory=list)  # legacy menu
    _depends_on: str = ""  # who this agent depends on
    _inbox: str = ""  # messages from other agents
    _nearby: str = ""  # who is at the same location


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

    # Who does this agent depend on?
    dependencies = fabric.get_dependencies(agent.agent_id)
    dep_on_lines = []
    for link in dependencies[:5]:
        if link.from_id in world.agents:
            other = world.agents[link.from_id]
            dep_on_lines.append(
                f"  - {other.personality.name} ({other.social_role}): "
                f"they {link.link_type} you"
            )

    # Recent memories
    memories = agent.get_recent_memories(5)
    mem_text = "\n".join(f"  - {m.description}" for m in memories) if memories else "Nothing notable."

    # Inbox: messages from other agents
    inbox = getattr(agent, "_inbox", [])
    inbox_text = ""
    if inbox:
        inbox_text = "\n".join(
            f"  - From {msg['from']}: \"{msg['message']}\""
            for msg in inbox[-5:]
        )

    # Nearby people at same location
    nearby_lines = []
    for aid, other in world.agents.items():
        if other.location == agent.location and aid != agent.agent_id:
            nearby_lines.append(f"{other.personality.name} ({other.social_role})")
            if len(nearby_lines) >= 6:
                break

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
            f"Savings: {agent.savings_buffer:.2f}. Income: {agent.income_level:.2f}. "
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
        decision_options=_get_decision_menu(agent.social_role),  # kept for legacy
        # New freeform fields
        _depends_on="\n".join(dep_on_lines) if dep_on_lines else "Nobody.",
        _inbox=inbox_text,
        _nearby=", ".join(nearby_lines) if nearby_lines else "Nobody nearby.",
    )


def _build_freeform_prompt(situation: AgentSituation, world: "World") -> str:
    """Build a freeform decision prompt — no menu, the agent decides what to do."""

    # Gather names the agent knows about for the messages section
    known_names = set()
    for line in situation.people_affected.split("\n"):
        if " - " in line:
            name = line.split(" - ")[1].split(" (")[0].strip()
            known_names.add(name)
    for line in getattr(situation, "_depends_on", "").split("\n"):
        if " - " in line:
            name = line.split(" - ")[1].split(" (")[0].strip()
            known_names.add(name)

    inbox_section = ""
    inbox_text = getattr(situation, "_inbox", "")
    if inbox_text:
        inbox_section = f"""
MESSAGES YOU RECEIVED FROM OTHER PEOPLE:
{inbox_text}
You may respond to these messages or ignore them.
"""

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

PEOPLE YOU DEPEND ON:
{getattr(situation, '_depends_on', 'Nobody.')}

PEOPLE NEAR YOU RIGHT NOW:
{getattr(situation, '_nearby', 'Nobody.')}
{inbox_section}
You are a real person in this situation. What do you ACTUALLY DO right now?

There is NO menu of options. Do whatever a real person in your position would do.
You can: negotiate, threaten, help, organize, cut costs, raise prices, hire, fire,
call someone, send a message, leak information, make a deal, protest, comfort someone,
stockpile, flee, confront, investigate, cooperate, betray, or anything else.

Be SPECIFIC and CONCRETE. Not "I help people" — WHO do you help, HOW, and what does it COST you?

Return ONLY valid JSON:
{{
  "action": "<1-2 sentences: what you specifically do right now>",
  "reasoning": "<2-3 sentences: WHY — what you're protecting, what you fear>",
  "consequences": [
    {{
      "target_name": "<name of a specific person affected>",
      "effect": "<what happens to them: loses income, gets help, feels pressure, etc.>",
      "debt_impact": <-0.15 to 0.15, negative=relief positive=pressure>,
      "tension_impact": <-0.1 to 0.1>,
      "dread_impact": <-0.1 to 0.1>,
      "relationship_change": "<trust+, trust-, warmth+, warmth-, rivalry+, alliance+, or none>"
    }}
  ],
  "messages": [
    {{
      "to": "<name of person you're communicating with>",
      "message": "<what you actually say or write to them, in your own voice>"
    }}
  ],
  "what_you_say_publicly": "<what you say out loud for everyone to hear, or nothing>",
  "what_you_think_privately": "<your internal monologue — fears, calculations, regrets>"
}}"""


def _build_decision_prompt(situation: AgentSituation) -> str:
    """Legacy menu-based prompt — kept for backwards compatibility."""
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

You must make a CONCRETE DECISION about what to do.

YOUR OPTIONS:
{options_text}

Return ONLY valid JSON:
{{
  "decision": "<one of the options above>",
  "magnitude": <0.1 to 1.0, how aggressively you act>,
  "reasoning": "<2-3 sentences>",
  "who_you_help": "<name or 'myself'>",
  "who_gets_hurt": "<name or 'nobody'>",
  "what_you_say": "<what you say out loud>",
  "what_you_think": "<what you're really thinking>"
}}"""


# ---------------------------------------------------------------------------
# Decision parser
# ---------------------------------------------------------------------------

@dataclass
class LLMAgentDecision:
    """Legacy menu-based decision."""
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


@dataclass
class FreeformDecision:
    """A freeform decision with self-described consequences and messages."""
    agent_id: str
    agent_name: str
    role: str
    action: str = ""
    reasoning: str = ""
    consequences: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    public_speech: str = ""
    private_thought: str = ""
    valid: bool = True
    latency_ms: int = 0


def _parse_freeform_decision(raw: str, agent_id: str, agent_name: str, role: str) -> FreeformDecision:
    """Parse a freeform LLM response into a FreeformDecision."""
    d = FreeformDecision(agent_id=agent_id, agent_name=agent_name, role=role)
    text = raw.strip()

    # Extract JSON
    start = text.find("{")
    if start == -1:
        d.valid = False
        return d
    end = text.rfind("}")
    json_str = text[start:end + 1] if end > start else text[start:]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Aggressively repair truncated JSON
        repaired = json_str.rstrip()
        # Close any open strings
        open_quotes = repaired.count('"') % 2
        if open_quotes:
            repaired += '"'
        # Close open arrays/objects
        open_brackets = repaired.count('[') - repaired.count(']')
        open_braces = repaired.count('{') - repaired.count('}')
        repaired = repaired.rstrip(',').rstrip()
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            # Last resort: try to extract just action and reasoning
            import re
            action_match = re.search(r'"action"\s*:\s*"([^"]*)"', json_str)
            if action_match:
                d.action = action_match.group(1)[:300]
                reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', json_str)
                if reason_match:
                    d.reasoning = reason_match.group(1)[:500]
                return d  # valid=True with partial data
            d.valid = False
            return d

    d.action = str(data.get("action", ""))[:300]
    d.reasoning = str(data.get("reasoning", ""))[:500]
    d.public_speech = str(data.get("what_you_say_publicly", ""))[:300]
    d.private_thought = str(data.get("what_you_think_privately", ""))[:300]

    # Parse consequences
    for cons in data.get("consequences", []):
        if isinstance(cons, dict) and cons.get("target_name"):
            d.consequences.append({
                "target_name": str(cons["target_name"]),
                "effect": str(cons.get("effect", ""))[:200],
                "debt_impact": max(-0.15, min(0.15, float(cons.get("debt_impact", 0)))),
                "tension_impact": max(-0.1, min(0.1, float(cons.get("tension_impact", 0)))),
                "dread_impact": max(-0.1, min(0.1, float(cons.get("dread_impact", 0)))),
                "relationship_change": str(cons.get("relationship_change", "none")),
            })

    # Parse messages
    for msg in data.get("messages", []):
        if isinstance(msg, dict) and msg.get("to") and msg.get("message"):
            d.messages.append({
                "to": str(msg["to"]),
                "message": str(msg["message"])[:300],
            })

    if not d.action:
        d.valid = False
    return d


def _apply_freeform_decision(
    decision: FreeformDecision,
    agent: "WorldAgent",
    fabric: "OrganizationalFabric",
    world: "World",
) -> list["RippleEvent"]:
    """Apply a freeform decision's self-described consequences to the world."""
    from .ripple_engine import RippleEvent

    events: list[RippleEvent] = []
    tick = world.tick_count

    # Record the action in agent memory
    agent.last_speech = decision.public_speech
    agent.add_memory(tick, f"[freeform] {decision.action[:100]}")
    if agent.memory and decision.private_thought:
        agent.memory[-1].interpretation = decision.private_thought[:150]

    # Build name→agent_id lookup for this world
    name_to_id: dict[str, str] = {}
    for aid, ag in world.agents.items():
        name_to_id[ag.personality.name] = aid
        # Also index by first name for fuzzy matching
        first = ag.personality.name.split("_")[0].split(" ")[0]
        if first not in name_to_id:
            name_to_id[first] = aid

    # Apply each consequence
    for cons in decision.consequences:
        target_name = cons["target_name"]
        target_id = name_to_id.get(target_name)

        # Fuzzy match: try partial name matching
        if not target_id:
            for name, aid in name_to_id.items():
                if target_name.lower() in name.lower() or name.lower() in target_name.lower():
                    target_id = aid
                    break

        if not target_id or target_id not in world.agents:
            continue

        target = world.agents[target_id]

        # Apply bounded impacts
        debt_d = cons.get("debt_impact", 0)
        tension_d = cons.get("tension_impact", 0)
        dread_d = cons.get("dread_impact", 0)

        if debt_d != 0:
            target.debt_pressure = _clamp(target.debt_pressure + debt_d)
        if tension_d != 0:
            target.heart.tension = _clamp(target.heart.tension + tension_d)
        if dread_d != 0:
            target.dread_pressure = _clamp(target.dread_pressure + dread_d)

        # Apply relationship changes — handle both structured and freeform descriptions
        rel_change = cons.get("relationship_change", "none").lower()
        if rel_change and rel_change != "none":
            rel = world.relationships.get_or_create(agent.agent_id, target_id)
            # Positive relationship signals
            if any(w in rel_change for w in ("trust+", "strengthen", "improve", "build", "warm", "support",
                                              "empower", "reassur", "solidif", "positive", "bond", "connect")):
                rel.trust = min(1.0, rel.trust + 0.05)
                rel.warmth = min(1.0, rel.warmth + 0.03)
            # Negative relationship signals
            if any(w in rel_change for w in ("trust-", "damage", "worsen", "betray", "hostile", "resent",
                                              "threaten", "undermine", "weaken", "erode", "distrust")):
                rel.trust = max(-1.0, rel.trust - 0.05)
                rel.warmth = max(-1.0, rel.warmth - 0.03)
            if any(w in rel_change for w in ("rivalry+", "rival", "compet", "oppose", "antagoni")):
                rel.rivalry = min(1.0, rel.rivalry + 0.05)
            if any(w in rel_change for w in ("alliance+", "alli", "coalit", "solidar", "unite")):
                rel.alliance_strength = min(1.0, rel.alliance_strength + 0.05)

        # Add wound for significant negative impact
        if debt_d > 0.08 or dread_d > 0.05:
            target.heart.wounds.append((abs(debt_d) * 0.3, 0.996))

        # Memory for target
        target.add_memory(tick, f"{agent.personality.name}: {cons['effect'][:80]}")

        events.append(RippleEvent(
            tick=tick,
            actor_id=agent.agent_id,
            actor_name=agent.personality.name,
            target_id=target_id,
            target_name=target.personality.name,
            action=decision.action[:80],
            consequence=cons["effect"][:80],
            mechanism="freeform",
            debt_delta=debt_d,
            tension_delta=tension_d,
            dread_delta=dread_d,
        ))

    # Deliver messages to recipients' inboxes
    for msg in decision.messages:
        recipient_name = msg["to"]
        recipient_id = name_to_id.get(recipient_name)
        if not recipient_id:
            for name, aid in name_to_id.items():
                if recipient_name.lower() in name.lower() or name.lower() in recipient_name.lower():
                    recipient_id = aid
                    break

        if recipient_id and recipient_id in world.agents:
            recipient = world.agents[recipient_id]
            if not hasattr(recipient, "_inbox"):
                recipient._inbox = []
            recipient._inbox.append({
                "from": agent.personality.name,
                "message": msg["message"],
                "tick": tick,
            })
            # Cap inbox size
            if len(recipient._inbox) > 10:
                recipient._inbox = recipient._inbox[-10:]
            # Also add as memory
            recipient.add_memory(tick, f"Message from {agent.personality.name}: {msg['message'][:60]}")

    return events


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
        # Only lay off employed workers
        employed_links = [l for l in employees if l.to_id in world.agents and world.agents[l.to_id].employed]
        n_cut = max(1, int(len(employed_links) * mag * 0.3))
        import random
        targets = random.sample(employed_links, min(n_cut, len(employed_links))) if employed_links else []
        for link in targets:
            target = world.agents[link.to_id]
            target.employed = False
            target.income_level = 0.05  # near zero income
            target.debt_pressure = _clamp(target.debt_pressure + 0.12 * mag)
            target.dread_pressure = _clamp(target.dread_pressure + 0.08 * mag)
            target.heart.wounds.append((0.04 * mag, 0.995))
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
            # Only apply if not already at reduced income
            # CUT_HOURS reduces income_level, which creates ONGOING lower recovery
            # instead of additive debt stacking every tick
            if target.income_level > 0.15:
                income_cut = min(0.15, target.income_level * 0.2 * mag)
                target.income_level = max(0.1, target.income_level - income_cut)
                # Small one-time debt bump (missed income this period)
                debt_bump = income_cut * 0.3
                target.debt_pressure = _clamp(target.debt_pressure + debt_bump)
                target.add_memory(tick, f"{agent.personality.name} cut hours — paycheck shrinking")
                events.append(RippleEvent(
                    tick=tick, actor_id=agent.agent_id, actor_name=agent.personality.name,
                    target_id=link.to_id, target_name=target.personality.name,
                    action=f"cut hours (LLM decision)",
                    consequence=f"income reduced {income_cut:.2f}, debt+{debt_bump:.3f}",
                    mechanism="LLM:employs", debt_delta=debt_bump,
                ))

    elif decision.decision == "RAISE_PRICES":
        customers = [l for l in dependents if l.link_type == "supplies"]
        for link in customers:
            if link.to_id not in world.agents: continue
            target = world.agents[link.to_id]
            # Realistic price hike: 2-5% cost increase, not 4-8%
            hike = 0.015 * mag * link.strength
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
    """Reactive cascade LLM agents — decisions trigger immediate reactions.

    When Agent A makes a decision that hits Agent B hard enough, Agent B
    gets an IMMEDIATE reactive LLM call in the same tick. B's reaction may
    cascade to C, who reacts to D, etc. — up to MAX_CASCADE_DEPTH levels.

    Agents at the same location can have multi-turn conversations within
    a single tick (up to CONVERSATION_MAX_TURNS exchanges).

    This creates real butterfly effects: one decision spirals through the
    network in ways nobody predicted.
    """

    # Tuning constants
    REACTIVE_THRESHOLD = 0.06      # min impact (debt+tension+dread) to trigger reactive call
    MAX_CASCADE_DEPTH = 6          # prevent infinite loops
    MAX_REACTIVE_PER_TICK = 12     # max reactive LLM calls per tick (cost control)
    CONVERSATION_MAX_TURNS = 3     # max back-and-forth exchanges per pair per tick
    SCHEDULED_COOLDOWN = 8         # ticks between scheduled decisions (8 hours, not 24)
    REACTIVE_COOLDOWN = 4          # ticks before same agent can react again

    def __init__(self, api_key: str, model: str = "gpt-5-mini", fabric: "OrganizationalFabric" = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fabric = fabric
        self.total_calls = 0
        self.total_decisions = 0
        self.total_reactive = 0
        self.total_messages_sent = 0
        self.total_conversations = 0
        self.max_cascade_depth_reached = 0
        self.decision_log: list[dict] = []
        self.cascade_chains: list[list[dict]] = []  # full chains for analysis
        self._last_decision_tick: dict[str, int] = {}
        self._conversation_pairs: dict[tuple, int] = {}  # (a,b) → turns this tick
        self._current_tick_reactive_count = 0

    def tick(self, world: "World", max_calls: int = 5) -> list["RippleEvent"]:
        """Run one tick: scheduled decisions → reactive cascades → conversations.

        Flow:
          1. Select top agents by pressure for SCHEDULED decisions
          2. For each decision, apply consequences
          3. Check if any consequence triggers a REACTIVE decision (immediate)
          4. Reactive decisions may trigger more reactive decisions (cascade)
          5. Messages trigger multi-turn CONVERSATIONS at same location
        """
        if self.fabric is None:
            return []

        from .ripple_engine import RippleEvent

        tick = world.tick_count
        self._current_tick_reactive_count = 0
        self._conversation_pairs = {}
        all_events: list[RippleEvent] = []

        # ── Phase 1: Scheduled decisions ──
        candidates = self._get_scheduled_candidates(world, tick)
        candidates.sort(reverse=True)
        selected = candidates[:max_calls]

        for _, agent_id in selected:
            decision, ripple_events = self._execute_decision(agent_id, world, tick, trigger="scheduled")
            if decision:
                all_events.extend(ripple_events)
                # Phase 2: Check for reactive cascades from this decision
                cascade_events = self._process_reactive_cascade(
                    ripple_events, decision, world, tick, depth=0,
                )
                all_events.extend(cascade_events)

        # ── Phase 3: Process any remaining inbox messages as conversations ──
        conversation_events = self._process_conversations(world, tick)
        all_events.extend(conversation_events)

        return all_events

    def _get_scheduled_candidates(self, world, tick):
        """Find agents due for a scheduled decision."""
        candidates = []
        for agent_id, agent in world.agents.items():
            if self._last_decision_tick.get(agent_id, 0) + self.SCHEDULED_COOLDOWN > tick:
                continue

            dependents = self.fabric.get_dependents(agent_id)
            dependencies = self.fabric.get_dependencies(agent_id)
            has_inbox = bool(getattr(agent, "_inbox", []))

            if not dependents and not dependencies and not has_inbox:
                continue

            pressure = agent.debt_pressure + agent.dread_pressure + agent.heart.tension
            if pressure < 0.2 and not has_inbox:
                continue

            n_connections = len(dependents) + len(dependencies)
            inbox_bonus = 3.0 if has_inbox else 0.0
            priority = max(1, n_connections) * pressure + inbox_bonus
            candidates.append((priority, agent_id))

        return candidates

    def _execute_decision(self, agent_id, world, tick, trigger="scheduled", cascade_context=""):
        """Execute one freeform LLM decision for an agent. Returns (decision, ripple_events)."""
        if agent_id not in world.agents:
            return None, []

        agent = world.agents[agent_id]
        situation = _build_situation(agent, self.fabric, world)

        # Add cascade context if this is a reactive decision
        prompt = _build_freeform_prompt(situation, world)
        if cascade_context:
            prompt += f"\n\nIMPORTANT CONTEXT — THIS JUST HAPPENED TO YOU:\n{cascade_context}\nYou must react to this RIGHT NOW."

        t0 = time.time()
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=2000,
                reasoning={"effort": "low"},
            )
            raw = (resp.output_text or "").strip()
            latency = int((time.time() - t0) * 1000)
        except Exception:
            return None, []

        self.total_calls += 1

        decision = _parse_freeform_decision(raw, agent_id, agent.personality.name, agent.social_role)
        if not decision.valid:
            return None, []

        self.total_decisions += 1
        if trigger == "reactive":
            self.total_reactive += 1
        self._last_decision_tick[agent_id] = tick

        # Clear inbox
        if hasattr(agent, "_inbox"):
            agent._inbox = []

        # Apply decision
        ripple_events = _apply_freeform_decision(decision, agent, self.fabric, world)
        self.total_messages_sent += len(decision.messages)

        self.decision_log.append({
            "tick": tick,
            "time": world.time_str,
            "agent": decision.agent_name,
            "role": decision.role,
            "action": decision.action[:150],
            "reasoning": decision.reasoning[:150],
            "speech": decision.public_speech[:120],
            "thought": decision.private_thought[:120],
            "consequences": len(decision.consequences),
            "messages_sent": len(decision.messages),
            "message_recipients": [m["to"] for m in decision.messages],
            "ripple_count": len(ripple_events),
            "latency_ms": latency,
            "trigger": trigger,
            "cascade_depth": 0,  # updated by cascade processor
        })

        return decision, ripple_events

    def _process_reactive_cascade(self, ripple_events, source_decision, world, tick, depth=0):
        """Check if any ripple event should trigger an immediate reactive decision.

        This is where butterfly effects happen:
          Source decision → consequence hits Target → Target reacts → Target's
          reaction hits another person → they react → cascade continues.
        """
        if depth >= self.MAX_CASCADE_DEPTH:
            self.max_cascade_depth_reached = max(self.max_cascade_depth_reached, depth)
            return []
        if self._current_tick_reactive_count >= self.MAX_REACTIVE_PER_TICK:
            return []

        all_cascade_events = []
        reactive_queue = []

        for event in ripple_events:
            target_id = event.target_id
            if target_id not in world.agents:
                continue

            # Check if this agent can react (cooldown)
            if self._last_decision_tick.get(target_id, 0) + self.REACTIVE_COOLDOWN > tick:
                continue

            # Calculate total impact magnitude
            impact = abs(event.debt_delta) + abs(event.tension_delta) + abs(event.dread_delta)

            # Also check if agent received a message (messages always trigger reaction)
            has_new_message = bool(getattr(world.agents[target_id], "_inbox", []))

            if impact >= self.REACTIVE_THRESHOLD or has_new_message:
                # Build context about what just happened to them
                context = (
                    f"{event.actor_name} just did this: {event.action[:100]}. "
                    f"The effect on you: {event.consequence[:100]}."
                )
                if has_new_message:
                    msgs = getattr(world.agents[target_id], "_inbox", [])
                    latest = msgs[-1] if msgs else {}
                    context += f"\nMessage from {latest.get('from','?')}: \"{latest.get('message','')[:100]}\""

                reactive_queue.append((impact, target_id, context))

        # Sort by impact severity — react to biggest hits first
        reactive_queue.sort(reverse=True)

        for _, target_id, context in reactive_queue:
            if self._current_tick_reactive_count >= self.MAX_REACTIVE_PER_TICK:
                break

            self._current_tick_reactive_count += 1

            decision, new_ripple_events = self._execute_decision(
                target_id, world, tick,
                trigger="reactive",
                cascade_context=context,
            )

            if decision:
                # Update the log entry with cascade depth
                if self.decision_log:
                    self.decision_log[-1]["cascade_depth"] = depth + 1
                    self.decision_log[-1]["trigger"] = f"reactive (depth {depth + 1})"
                    self.decision_log[-1]["triggered_by"] = source_decision.agent_name if source_decision else "?"

                all_cascade_events.extend(new_ripple_events)

                # Track cascade chain
                chain_entry = {
                    "depth": depth + 1,
                    "source": source_decision.agent_name if source_decision else "?",
                    "reactor": decision.agent_name,
                    "action": decision.action[:100],
                    "triggered_by_impact": context[:100],
                }

                # Recurse: this reactive decision may trigger MORE reactions
                deeper_events = self._process_reactive_cascade(
                    new_ripple_events, decision, world, tick, depth=depth + 1,
                )
                all_cascade_events.extend(deeper_events)

        self.max_cascade_depth_reached = max(self.max_cascade_depth_reached, depth)
        return all_cascade_events

    def _process_conversations(self, world, tick):
        """Process multi-turn conversations for agents who received messages.

        When Agent A messages Agent B and they're at the same location,
        allow up to CONVERSATION_MAX_TURNS back-and-forth exchanges.
        """
        all_events = []

        for agent_id, agent in world.agents.items():
            inbox = getattr(agent, "_inbox", [])
            if not inbox:
                continue
            if self._current_tick_reactive_count >= self.MAX_REACTIVE_PER_TICK:
                break

            # Check if any message sender is at the same location (face-to-face)
            for msg in inbox:
                sender_name = msg.get("from", "")
                # Find sender agent_id
                sender_id = None
                for aid, ag in world.agents.items():
                    if ag.personality.name == sender_name:
                        sender_id = aid
                        break

                if not sender_id or sender_id not in world.agents:
                    continue

                sender = world.agents[sender_id]

                # Same location = face-to-face conversation possible
                if sender.location != agent.location:
                    continue

                pair = tuple(sorted([agent_id, sender_id]))
                turns = self._conversation_pairs.get(pair, 0)
                if turns >= self.CONVERSATION_MAX_TURNS:
                    continue

                # Agent responds to the message
                context = f"{sender_name} is right here and just said to you: \"{msg['message'][:150]}\"\nYou must respond directly to them."

                self._current_tick_reactive_count += 1
                decision, ripple_events = self._execute_decision(
                    agent_id, world, tick,
                    trigger="conversation",
                    cascade_context=context,
                )

                if decision:
                    self.total_conversations += 1
                    self._conversation_pairs[pair] = turns + 1

                    if self.decision_log:
                        self.decision_log[-1]["trigger"] = f"conversation (turn {turns + 1})"
                        self.decision_log[-1]["conversation_with"] = sender_name

                    all_events.extend(ripple_events)

                    # Conversation response may trigger cascades too
                    cascade_events = self._process_reactive_cascade(
                        ripple_events, decision, world, tick, depth=0,
                    )
                    all_events.extend(cascade_events)

                if self._current_tick_reactive_count >= self.MAX_REACTIVE_PER_TICK:
                    break

        return all_events

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_decisions": self.total_decisions,
            "total_reactive": self.total_reactive,
            "total_conversations": self.total_conversations,
            "total_messages_sent": self.total_messages_sent,
            "max_cascade_depth": self.max_cascade_depth_reached,
            "recent_decisions": self.decision_log[-10:],
        }

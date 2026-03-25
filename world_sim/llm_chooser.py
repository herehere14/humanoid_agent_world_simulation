"""Phase 3 — LLM Chooser/Executor for hybrid agents.

This is the actual LLM execution loop. For agents promoted to llm_active:
  1. Takes the pre-built decision packet (from llm_packet.py Phase 2)
  2. Sends it to an LLM (GPT-5-mini or similar)
  3. Parses the structured JSON response
  4. Validates the chosen action against allowed_actions
  5. Applies the decision back into the simulation:
     - Overrides the deterministic action with the LLM's choice
     - Records the speech
     - Applies relationship delta hints
     - Updates the agent's memory with a richer interpretation

Design constraints:
  - LLM cannot invent world facts (validated against packet)
  - LLM can only choose from allowed_actions
  - LLM's relationship hints are bounded (±0.05 per tick)
  - If LLM fails/times out, deterministic action stands
  - Batch calls where possible for throughput
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action_table import Action
    from .relationship import RelationshipStore
    from .world import World
    from .world_agent import WorldAgent


@dataclass
class LLMDecision:
    """Parsed and validated LLM decision for one agent."""
    agent_id: str
    chosen_action: str = "WORK"
    target_agent_id: str | None = None
    intent: str = ""
    tactic: str = ""
    tone: str = ""
    surface_move: str = ""
    private_thought: str = ""
    speech: str = ""
    story_beat: str = ""
    trust_delta_hint: float = 0.0
    grievance_delta_hint: float = 0.0
    debt_delta_hint: float = 0.0
    valid: bool = True
    fallback_used: bool = False
    latency_ms: int = 0


@dataclass
class LLMChooserStats:
    """Tracking stats for the chooser across a simulation run."""
    total_calls: int = 0
    successful_calls: int = 0
    fallback_calls: int = 0
    parse_failures: int = 0
    validation_failures: int = 0
    total_latency_ms: int = 0
    total_tokens_approx: int = 0
    actions_overridden: int = 0

    def as_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "fallback_calls": self.fallback_calls,
            "parse_failures": self.parse_failures,
            "validation_failures": self.validation_failures,
            "avg_latency_ms": self.total_latency_ms // max(1, self.total_calls),
            "actions_overridden": self.actions_overridden,
        }


def _build_prompt(packet: dict) -> str:
    """Convert a decision packet into an LLM prompt."""
    profile = packet.get("private_state", {}).get("human_profile", {})
    heart = packet.get("private_state", {}).get("heart", {})
    appraisal = packet.get("private_state", {}).get("appraisal", {})
    motives = packet.get("private_state", {}).get("motives", {})
    brief = packet.get("private_state", {}).get("subjective_brief", "")

    # Format nearby people
    nearby_text = ""
    for p in packet.get("nearby_people", [])[:4]:
        nearby_text += f"  - {p['name']} ({p['role']}): trust={p.get('trust', 0):+.2f}, warmth={p.get('warmth', 0):+.2f}, surface={p.get('surface_emotion', '?')}\n"

    # Format allowed actions
    actions_text = ""
    for opt in packet.get("allowed_actions", [])[:5]:
        target_note = f" → {opt.get('likely_target_name', '?')}" if opt.get("likely_target_name") else ""
        actions_text += f"  - {opt['action']}{target_note}: {opt['reason']}\n"

    # Format recent memories
    memory_text = ""
    for m in packet.get("recent_memories", [])[:4]:
        memory_text += f"  - {m['description']}"
        if m.get("interpretation"):
            memory_text += f" (I felt: {m['interpretation']})"
        memory_text += "\n"

    # Format live events
    events_text = ""
    for e in packet.get("live_events", [])[:3]:
        events_text += f"  - [{e.get('kind', 'event')}] {e.get('description', '')}\n"

    return f"""You are {packet['name']}, a {packet['role']}.

YOUR PSYCHOLOGY:
- Attachment: {profile.get('attachment_style', '?')}
- Coping: {profile.get('coping_style', '?')}
- What threatens you: {profile.get('threat_lens', '?')}
- What you need: {profile.get('core_need', '?')}
- Your shame trigger: {profile.get('shame_trigger', '?')}
- How you fight: {profile.get('conflict_style', '?')}
- The mask you wear: {profile.get('mask_tendency', '?')}
- Your self-story: {profile.get('self_story', '?')}
- Your longing: {profile.get('longing', '?')}

YOUR CURRENT STATE:
- Inside: {heart.get('internal_emotion', '?')} (arousal={heart.get('arousal', 0):.2f}, valence={heart.get('valence', 0):.2f})
- Showing: {heart.get('surface_emotion', '?')} (tension={heart.get('tension', 0):.2f})
- Energy: {heart.get('energy', 0):.2f}, Vulnerability: {heart.get('vulnerability', 0):.2f}
- Impulse control: {heart.get('impulse_control', 0):.2f}

WHAT'S ON YOUR MIND:
{brief}

RECENT EVENTS:
{memory_text.strip() if memory_text.strip() else "Nothing notable."}

LIVE EVENTS HAPPENING NOW:
{events_text.strip() if events_text.strip() else "None."}

PEOPLE NEARBY:
{nearby_text.strip() if nearby_text.strip() else "Nobody."}

TIME: {packet.get('time', '?')}
LOCATION: {packet.get('location', '?')}
RECOMMENDED ACTION: {packet.get('recommended_action', '?')}

ALLOWED ACTIONS (pick ONE):
{actions_text.strip()}

Respond with ONLY valid JSON matching this exact structure:
{{
  "chosen_action": "<one of the allowed actions above>",
  "target_agent_id": "<agent_id from nearby people, or null>",
  "intent": "<what you're really trying to accomplish>",
  "tactic": "<how you're going about it>",
  "tone": "<emotional register: sharp, soft, deflecting, commanding, etc.>",
  "surface_move": "<what you visibly do>",
  "private_thought": "<what you're thinking but not saying>",
  "speech": "<what you actually say, 1-3 sentences, in character>",
  "self_update": {{
    "story_beat": "<one line: what this moment means in your story>",
    "trust_delta_hint": <-0.05 to +0.05>,
    "grievance_delta_hint": <-0.05 to +0.05>,
    "debt_delta_hint": <-0.05 to +0.05>
  }}
}}

Stay in character. Your speech must sound like YOU — your specific fears, your coping style, your way of talking. Do NOT sound generic."""


def _parse_decision(raw: str, agent_id: str, allowed_actions: list[str]) -> LLMDecision:
    """Parse and validate LLM response."""
    decision = LLMDecision(agent_id=agent_id)

    # Extract JSON
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        decision.valid = False
        decision.fallback_used = True
        return decision

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        decision.valid = False
        decision.fallback_used = True
        return decision

    # Validate chosen_action
    chosen = str(data.get("chosen_action", "")).strip().upper()
    if chosen not in allowed_actions:
        # Try fuzzy match
        for a in allowed_actions:
            if a in chosen or chosen in a:
                chosen = a
                break
        else:
            decision.valid = False
            decision.fallback_used = True
            return decision

    decision.chosen_action = chosen
    decision.target_agent_id = data.get("target_agent_id")
    decision.intent = str(data.get("intent", ""))[:200]
    decision.tactic = str(data.get("tactic", ""))[:200]
    decision.tone = str(data.get("tone", ""))[:100]
    decision.surface_move = str(data.get("surface_move", ""))[:200]
    decision.private_thought = str(data.get("private_thought", ""))[:300]
    decision.speech = str(data.get("speech", ""))[:500]

    # Parse self_update with bounds
    su = data.get("self_update", {})
    if isinstance(su, dict):
        decision.story_beat = str(su.get("story_beat", ""))[:200]
        decision.trust_delta_hint = max(-0.05, min(0.05, float(su.get("trust_delta_hint", 0) or 0)))
        decision.grievance_delta_hint = max(-0.05, min(0.05, float(su.get("grievance_delta_hint", 0) or 0)))
        decision.debt_delta_hint = max(-0.05, min(0.05, float(su.get("debt_delta_hint", 0) or 0)))

    return decision


class LLMChooser:
    """Calls an LLM for promoted agents and applies decisions back into the sim.

    Usage:
        chooser = LLMChooser(api_key="sk-...", model="gpt-5-mini")

        # In the tick loop, after actions are selected:
        chooser.execute_llm_decisions(world, actions, tick_contexts)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        max_per_tick: int = 15,
        enabled: bool = True,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_per_tick = max_per_tick
        self.enabled = enabled
        self.stats = LLMChooserStats()
        self._call_log: list[dict] = []

    def execute_llm_decisions(
        self,
        world: "World",
        actions: dict[str, "Action"],
        tick_summary: dict,
    ) -> list[LLMDecision]:
        """Call LLM for all llm_active agents and apply decisions.

        Returns list of LLM decisions made this tick.
        """
        if not self.enabled:
            return []

        from .action_table import Action as ActionEnum

        # Get agents that are llm_active and have packets
        candidates = [
            agent for agent in world.agents.values()
            if agent.llm_active and agent.llm_packet_preview is not None
        ]

        # Limit per tick
        candidates.sort(key=lambda a: a.llm_salience, reverse=True)
        candidates = candidates[:self.max_per_tick]

        if not candidates:
            return []

        decisions: list[LLMDecision] = []

        for agent in candidates:
            packet = agent.llm_packet_preview
            allowed = [opt["action"] for opt in packet.get("allowed_actions", [])]
            if not allowed:
                continue

            prompt = _build_prompt(packet)

            t0 = time.time()
            self.stats.total_calls += 1

            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=400,
                    reasoning={"effort": "low"},
                )
                raw = (resp.output_text or "").strip()
                latency = int((time.time() - t0) * 1000)
            except Exception as e:
                self.stats.fallback_calls += 1
                decisions.append(LLMDecision(
                    agent_id=agent.agent_id,
                    valid=False,
                    fallback_used=True,
                    chosen_action=packet.get("recommended_action", "WORK"),
                ))
                continue

            decision = _parse_decision(raw, agent.agent_id, allowed)
            decision.latency_ms = latency

            if not decision.valid:
                self.stats.parse_failures += 1
                self.stats.fallback_calls += 1
                decision.chosen_action = packet.get("recommended_action", "WORK")
                decision.fallback_used = True
            else:
                self.stats.successful_calls += 1

            decisions.append(decision)

            # Apply decision to agent
            self._apply_decision(world, agent, decision, actions)

            # Log for diagnostics
            self._call_log.append({
                "tick": world.tick_count,
                "agent_id": agent.agent_id,
                "name": agent.personality.name,
                "role": agent.social_role,
                "recommended": packet.get("recommended_action", ""),
                "chosen": decision.chosen_action,
                "overridden": decision.chosen_action != packet.get("recommended_action", ""),
                "speech": decision.speech[:100] if decision.speech else "",
                "private_thought": decision.private_thought[:100] if decision.private_thought else "",
                "valid": decision.valid,
                "latency_ms": decision.latency_ms,
            })

        self.stats.total_latency_ms += sum(d.latency_ms for d in decisions)
        return decisions

    def _apply_decision(
        self,
        world: "World",
        agent: "WorldAgent",
        decision: LLMDecision,
        actions: dict[str, "Action"],
    ):
        """Apply a validated LLM decision back into the simulation."""
        from .action_table import Action as ActionEnum

        # Override action if LLM chose differently
        try:
            new_action = ActionEnum[decision.chosen_action]
        except (KeyError, ValueError):
            return  # invalid action name, keep deterministic

        old_action = actions.get(agent.agent_id)
        if new_action != old_action:
            actions[agent.agent_id] = new_action
            agent.last_action = new_action.name
            self.stats.actions_overridden += 1

        # Record speech
        if decision.speech:
            agent.last_speech = decision.speech

        # Add richer memory with LLM-generated interpretation
        if decision.private_thought:
            agent.add_memory(
                world.tick_count,
                f"[llm] {decision.surface_move or decision.chosen_action}",
            )
            if agent.memory:
                agent.memory[-1].interpretation = decision.private_thought
                if decision.story_beat:
                    agent.memory[-1].story_beat = decision.story_beat

        # Apply bounded relationship hints
        if decision.target_agent_id and decision.target_agent_id in world.agents:
            target_id = decision.target_agent_id
            rel = world.relationships.get_or_create(agent.agent_id, target_id)

            if decision.trust_delta_hint:
                rel.trust = max(-1.0, min(1.0, rel.trust + decision.trust_delta_hint))
            if decision.grievance_delta_hint:
                world.relationships.set_grievance(
                    agent.agent_id, target_id,
                    world.relationships.get_grievance(agent.agent_id, target_id) + decision.grievance_delta_hint,
                )
            if decision.debt_delta_hint:
                world.relationships.adjust_debt(agent.agent_id, target_id, decision.debt_delta_hint)

    def get_stats(self) -> dict:
        return self.stats.as_dict()

    def get_call_log(self, last_n: int = 50) -> list[dict]:
        return self._call_log[-last_n:]

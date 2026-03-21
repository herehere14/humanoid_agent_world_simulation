"""World — simulation loop, agent registry, clock.

The tick loop:
  1. Advance clock
  2. Move agents to scheduled locations
  3. Inject any scheduled events (encode once, share embedding)
  4. Update each agent's heart state (pure numpy, fast)
  5. Select actions deterministically
  6. Resolve interactions (pairs who interact at same location)
  7. Apply emotional contagion
  8. Record state for dashboard
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .world_agent import WorldAgent, SharedBrain, HeartState, Personality, update_heart, rest_heart
from .relationship import RelationshipStore
from .action_table import Action, select_action, TickContext, get_action_description
from .contagion import apply_contagion


@dataclass
class ScheduledEvent:
    """An event injected into the world at a specific tick."""
    tick: int
    location: str
    description: str         # narrative text for display/memory
    emotional_text: str = "" # first-person emotional text for SBERT encoding
                             # if empty, description is used instead
    severity: float = 1.0    # 0-1, affects how strongly agents react
    target_agent_ids: list[str] | None = None  # None = affects all at location


@dataclass
class Location:
    """A place in the world."""
    location_id: str
    name: str
    default_activity: str  # what happens here normally ("working at desk", "drinking at bar")


class World:
    """The simulation world. Holds agents, locations, clock, and runs the tick loop."""

    def __init__(self):
        self.agents: dict[str, WorldAgent] = {}
        self.locations: dict[str, Location] = {}
        self.relationships = RelationshipStore()
        self.brain: SharedBrain | None = None  # initialized lazily
        self.tick_count: int = 0
        self.events: list[ScheduledEvent] = []
        self.tick_log: list[dict] = []  # per-tick summary for dashboard

    def initialize(self):
        """Load the shared brain (SBERT + anchors). Call once before simulation."""
        print("  Loading SharedBrain (SBERT model + anchor embeddings)...")
        t0 = time.time()
        self.brain = SharedBrain.get()
        print(f"  SharedBrain loaded in {time.time() - t0:.1f}s")

    def add_location(self, loc: Location):
        self.locations[loc.location_id] = loc

    def add_agent(self, agent: WorldAgent):
        self.agents[agent.agent_id] = agent

    def schedule_event(self, event: ScheduledEvent):
        self.events.append(event)

    @property
    def hour_of_day(self) -> int:
        return self.tick_count % 24

    @property
    def day(self) -> int:
        return self.tick_count // 24 + 1

    @property
    def time_str(self) -> str:
        return f"Day {self.day}, {self.hour_of_day:02d}:00"

    def tick(self) -> dict:
        """Advance one simulated hour. Returns tick summary."""
        assert self.brain is not None, "Call world.initialize() first"
        self.tick_count += 1
        summary = {
            "tick": self.tick_count,
            "time": self.time_str,
            "events": [],
            "actions": {},
            "interactions": [],
            "speeches": [],
        }

        # Phase 1: Move agents to scheduled locations (tick overrides take priority)
        for agent in self.agents.values():
            override = agent.location_overrides.get(self.tick_count)
            if override:
                agent.location = override
            elif self.hour_of_day in agent.schedule:
                agent.location = agent.schedule[self.hour_of_day]

        # Phase 2: Inject scheduled events
        tick_events = [e for e in self.events if e.tick == self.tick_count]
        # Multiple events can fire at same location/tick, so use a list
        event_list: list[tuple[str, np.ndarray, str, list[str] | None]] = []
        for event in tick_events:
            # Use emotional_text for SBERT encoding (first-person emotional),
            # description for display/memory (narrative)
            encode_text = event.emotional_text or event.description
            emb = self.brain.encode(encode_text)
            event_list.append((event.location, emb, event.description, event.target_agent_ids))
            summary["events"].append({
                "location": event.location,
                "description": event.description,
                "targets": event.target_agent_ids,
            })

        # Phase 3: Update heart state for each agent
        for agent in self.agents.values():
            # Rest during sleep hours
            if self.hour_of_day >= 22 or self.hour_of_day < 6:
                rest_heart(agent.heart)

            # Check if agent is affected by any events this tick
            agent_had_event = False
            for ev_loc, ev_emb, ev_desc, ev_targets in event_list:
                if agent.location != ev_loc:
                    continue
                if ev_targets is not None and agent.agent_id not in ev_targets:
                    continue
                # Targeted events (like being personally fired) hit harder
                severity = 3.0 if ev_targets and agent.agent_id in ev_targets else 2.0
                update_heart(agent.heart, ev_emb, self.brain, agent.personality,
                             event_severity=severity)
                agent.add_memory(self.tick_count, ev_desc)
                agent_had_event = True

            # Only process routine activity if no event this tick
            # Events should dominate the hour, not get diluted
            if not agent_had_event:
                loc = self.locations.get(agent.location)
                if loc:
                    routine_emb = self.brain.encode(loc.default_activity)
                    update_heart(agent.heart, routine_emb, self.brain, agent.personality)

        # Phase 4: Build nearby agent lists
        by_location: dict[str, list[str]] = defaultdict(list)
        for agent in self.agents.values():
            by_location[agent.location].append(agent.agent_id)

        # Phase 5: Select actions deterministically
        actions: dict[str, Action] = {}
        for agent in self.agents.values():
            nearby = [aid for aid in by_location[agent.location] if aid != agent.agent_id]
            nearby_agents = {aid: self.agents[aid] for aid in nearby}
            ctx = TickContext(
                tick=self.tick_count,
                hour_of_day=self.hour_of_day,
                scheduled_location=agent.schedule.get(self.hour_of_day, agent.location),
                current_location=agent.location,
                nearby_agent_ids=nearby,
                nearby_agents=nearby_agents,
                relationships=self.relationships,
                agent_id=agent.agent_id,
            )
            action = select_action(agent, ctx)
            actions[agent.agent_id] = action
            agent.last_action = action.name

            # Handle location changes
            if action == Action.WITHDRAW or action == Action.FLEE:
                agent.location = "home"
            elif action == Action.SEEK_COMFORT:
                # Move to closest warm relationship's location
                closest = self.relationships.get_closest(agent.agent_id, n=1)
                if closest:
                    target_id = closest[0][0]
                    target_agent = self.agents.get(target_id)
                    if target_agent:
                        agent.location = target_agent.location

            summary["actions"][agent.agent_id] = {
                "action": action.name,
                "description": get_action_description(action, agent),
            }

        # Phase 6: Resolve interactions
        interactions = self._resolve_interactions(actions, by_location)
        summary["interactions"] = interactions

        # Phase 7: Emotional contagion
        apply_contagion(self.agents, self.relationships)

        self.tick_log.append(summary)
        return summary

    def _resolve_interactions(
        self,
        actions: dict[str, Action],
        by_location: dict[str, list[str]],
    ) -> list[dict]:
        """Find pairs that interact and update relationships."""
        interactions = []

        for loc, agent_ids in by_location.items():
            if len(agent_ids) < 2:
                continue

            social_agents = [
                aid for aid in agent_ids
                if actions.get(aid, Action.IDLE).is_social
            ]

            # Pair up social agents at the same location
            paired = set()
            for i, aid_a in enumerate(social_agents):
                if aid_a in paired:
                    continue
                action_a = actions[aid_a]
                agent_a = self.agents[aid_a]

                # Find the best interaction partner
                best_partner = None
                best_score = -float("inf")

                for aid_b in social_agents[i+1:]:
                    if aid_b in paired:
                        continue
                    agent_b = self.agents[aid_b]
                    action_b = actions[aid_b]
                    rel = self.relationships.get(aid_a, aid_b)

                    # Score: familiarity + warmth, or resentment for confrontation
                    score = 0.0
                    if rel:
                        score = rel.familiarity * 0.01 + rel.warmth * 0.5
                        if action_a == Action.CONFRONT:
                            score = self.relationships.get_resentment(aid_a, aid_b) * 2
                    else:
                        score = 0.1  # slight preference for novel interactions

                    if score > best_score:
                        best_score = score
                        best_partner = aid_b

                if best_partner is None:
                    continue

                paired.add(aid_a)
                paired.add(best_partner)
                agent_b = self.agents[best_partner]
                action_b = actions[best_partner]

                # Determine interaction type
                if action_a in (Action.CONFRONT, Action.LASH_OUT) or action_b in (Action.CONFRONT, Action.LASH_OUT):
                    interaction_type = "conflict"
                elif action_a in (Action.SEEK_COMFORT, Action.HELP_OTHERS) or action_b in (Action.SEEK_COMFORT, Action.HELP_OTHERS):
                    interaction_type = "support"
                elif action_a == Action.CELEBRATE or action_b == Action.CELEBRATE:
                    interaction_type = "positive"
                elif action_a == Action.VENT or action_b == Action.VENT:
                    interaction_type = "neutral"
                else:
                    interaction_type = "neutral"

                # Update relationships
                self.relationships.update_after_interaction(
                    aid_a, best_partner, self.tick_count,
                    agent_a.heart.valence, agent_b.heart.valence,
                    interaction_type,
                )

                # Memory
                agent_a.add_memory(self.tick_count, f"{interaction_type} interaction with {agent_b.personality.name}", best_partner)
                agent_b.add_memory(self.tick_count, f"{interaction_type} interaction with {agent_a.personality.name}", aid_a)

                interactions.append({
                    "agent_a": aid_a,
                    "agent_b": best_partner,
                    "type": interaction_type,
                    "location": loc,
                    "action_a": action_a.name,
                    "action_b": action_b.name,
                })

        return interactions

    def get_agent_dashboard(self, agent_id: str) -> dict | None:
        """Full dashboard for one agent — for user exploration."""
        agent = self.agents.get(agent_id)
        if not agent:
            return None

        state = agent.get_dashboard_state()
        state["recent_memories"] = [
            {"tick": m.tick, "time": f"Day {m.tick // 24 + 1} {m.tick % 24:02d}:00",
             "description": m.description, "valence": round(m.valence_at_time, 2),
             "arousal": round(m.arousal_at_time, 2), "other": m.other_agent_id}
            for m in agent.get_recent_memories(15)
        ]
        state["relationships"] = [
            {"other_id": other_id, "other_name": self.agents[other_id].personality.name if other_id in self.agents else other_id,
             "trust": round(rel.trust, 2), "warmth": round(rel.warmth, 2),
             "resentment_toward": round(self.relationships.get_resentment(agent_id, other_id), 2),
             "resentment_from": round(self.relationships.get_resentment(other_id, agent_id), 2),
             "interactions": rel.familiarity}
            for other_id, rel in self.relationships.get_agent_relationships(agent_id)[:10]
        ]
        state["last_speech"] = agent.last_speech
        return state

    def get_world_summary(self) -> dict:
        """Summary for the world dashboard."""
        from statistics import mean
        agents_list = list(self.agents.values())

        action_counts: dict[str, int] = defaultdict(int)
        for a in agents_list:
            action_counts[a.last_action] += 1

        return {
            "tick": self.tick_count,
            "time": self.time_str,
            "agent_count": len(agents_list),
            "relationship_count": self.relationships.pair_count,
            "avg_energy": round(mean(a.heart.energy for a in agents_list), 2),
            "avg_valence": round(mean(a.heart.valence for a in agents_list), 2),
            "avg_arousal": round(mean(a.heart.arousal for a in agents_list), 2),
            "avg_tension": round(mean(a.heart.tension for a in agents_list), 2),
            "action_counts": dict(action_counts),
            "most_distressed": sorted(
                [a.get_dashboard_state() for a in agents_list],
                key=lambda x: x["vulnerability"], reverse=True
            )[:5],
        }

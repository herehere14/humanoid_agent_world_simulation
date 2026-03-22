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
    kind: str = "scheduled"


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
        self.scenario_name: str = "large_town"
        self.group_profiles: dict[str, dict] = {}

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
        event_context_by_location: dict[str, list[dict]] = defaultdict(list)
        for event in tick_events:
            # Use emotional_text for SBERT encoding (first-person emotional),
            # description for display/memory (narrative)
            encode_text = event.emotional_text or event.description
            emb = self.brain.encode(encode_text)
            event_list.append((event.location, emb, event.description, event.target_agent_ids))
            event_context_by_location[event.location].append(
                {"kind": event.kind, "targets": event.target_agent_ids}
            )
            summary["events"].append({
                "location": event.location,
                "description": event.description,
                "targets": event.target_agent_ids,
                "kind": event.kind,
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

        self._apply_group_event_effects(tick_events, by_location)

        # Refresh each agent's subjective read now that location/social context
        # is known for this tick.
        for agent in self.agents.values():
            nearby = [aid for aid in by_location[agent.location] if aid != agent.agent_id]
            nearby_agents = {aid: self.agents[aid] for aid in nearby}
            agent.refresh_subjective_state(nearby_agents, self.relationships)

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
                event_count=len(event_context_by_location.get(agent.location, [])),
                event_kinds=tuple(
                    item["kind"] for item in event_context_by_location.get(agent.location, [])
                ),
                is_event_target=any(
                    item["targets"] is not None and agent.agent_id in item["targets"]
                    for item in event_context_by_location.get(agent.location, [])
                ),
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
                "priority_motive": agent.motives.priority,
                "action_style": agent.motives.action_style,
            }

        # Phase 6: Resolve interactions
        interactions = self._resolve_interactions(actions, by_location)
        summary["interactions"] = interactions

        # Phase 7: Emotional contagion
        apply_contagion(self.agents, self.relationships)

        # Contagion and interaction outcomes can change the internal read even if
        # the macro action stays the same; refresh once more for dashboard/prompt use.
        for agent in self.agents.values():
            nearby = [aid for aid in by_location[agent.location] if aid != agent.agent_id]
            nearby_agents = {aid: self.agents[aid] for aid in nearby}
            agent.refresh_subjective_state(nearby_agents, self.relationships)

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
            max_pairs_here = max(1, min(10, len(social_agents) // 5 + 1))

            # Pair up social agents at the same location
            paired = set()
            pairs_made = 0
            for i, aid_a in enumerate(social_agents):
                if pairs_made >= max_pairs_here:
                    break
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
                    shared_groups = agent_a.shared_coalitions(agent_b)
                    rival_groups = agent_a.rival_overlap(agent_b)
                    grievance_ab = self.relationships.get_grievance(aid_a, aid_b)
                    grievance_ba = self.relationships.get_grievance(aid_b, aid_a)
                    debt_ab = self.relationships.get_debt(aid_a, aid_b)
                    debt_ba = self.relationships.get_debt(aid_b, aid_a)

                    score = 0.05
                    if rel:
                        if action_a in (Action.CONFRONT, Action.LASH_OUT):
                            score = (
                                self.relationships.get_resentment(aid_a, aid_b) * 1.4 +
                                grievance_ab * 1.8 +
                                rel.rivalry * 1.2 +
                                debt_ab * 0.5 +
                                len(rival_groups) * 0.7 -
                                max(0.0, rel.trust) * 0.2
                            )
                        elif action_a in (Action.HELP_OTHERS, Action.SEEK_COMFORT, Action.SOCIALIZE, Action.VENT):
                            score = (
                                rel.warmth * 0.35 +
                                rel.trust * 0.3 +
                                len(shared_groups) * 0.55 +
                                rel.practical_help_events * 0.06 +
                                debt_ba * 0.35 -
                                rel.rivalry * 0.35
                            )
                        else:
                            score = (
                                rel.familiarity * 0.01 +
                                rel.warmth * 0.15 +
                                len(shared_groups) * 0.2 +
                                debt_ab * 0.08 +
                                debt_ba * 0.08 -
                                len(rival_groups) * 0.2
                            )
                    else:
                        score = 0.12 + len(shared_groups) * 0.25 - len(rival_groups) * 0.15

                    if action_b in (Action.CONFRONT, Action.LASH_OUT):
                        score += grievance_ba * 0.35 + self.relationships.get_resentment(aid_b, aid_a) * 0.25
                    if action_b == Action.HELP_OTHERS and action_a != Action.CONFRONT:
                        score += 0.12
                    if debt_ab > 0.45 or debt_ba > 0.45:
                        score += 0.08

                    if score > best_score:
                        best_score = score
                        best_partner = aid_b

                if best_partner is None:
                    continue

                paired.add(aid_a)
                paired.add(best_partner)
                pairs_made += 1
                agent_b = self.agents[best_partner]
                action_b = actions[best_partner]
                shared_groups = agent_a.shared_coalitions(agent_b)
                rival_groups = agent_a.rival_overlap(agent_b)
                grievance_ab = self.relationships.get_grievance(aid_a, best_partner)
                grievance_ba = self.relationships.get_grievance(best_partner, aid_a)
                debt_ab = self.relationships.get_debt(aid_a, best_partner)
                debt_ba = self.relationships.get_debt(best_partner, aid_a)

                # Determine interaction type
                initiator_id = None
                if (
                    action_a in (Action.CONFRONT, Action.LASH_OUT) or
                    action_b in (Action.CONFRONT, Action.LASH_OUT) or
                    (
                        rival_groups and
                        max(agent_a.appraisal.injustice + grievance_ab, agent_b.appraisal.injustice + grievance_ba) > 0.42
                    ) or
                    (
                        max(debt_ab, debt_ba) > 0.5 and
                        max(agent_a.appraisal.economic_pressure, agent_b.appraisal.economic_pressure) > 0.5
                    )
                ):
                    interaction_type = "conflict"
                    if action_a in (Action.CONFRONT, Action.LASH_OUT):
                        initiator_id = aid_a
                    elif action_b in (Action.CONFRONT, Action.LASH_OUT):
                        initiator_id = best_partner
                    elif agent_a.appraisal.injustice + grievance_ab >= agent_b.appraisal.injustice + grievance_ba:
                        initiator_id = aid_a
                    else:
                        initiator_id = best_partner
                elif action_a in (Action.SEEK_COMFORT, Action.HELP_OTHERS) or action_b in (Action.SEEK_COMFORT, Action.HELP_OTHERS):
                    interaction_type = "support"
                elif action_a == Action.CELEBRATE or action_b == Action.CELEBRATE:
                    interaction_type = "positive"
                elif shared_groups and (action_a == Action.SOCIALIZE or action_b == Action.SOCIALIZE):
                    interaction_type = "support"
                elif action_a == Action.VENT or action_b == Action.VENT:
                    interaction_type = "neutral"
                else:
                    interaction_type = "neutral"

                issue = self._infer_interaction_issue(loc, interaction_type)
                practical_help = self._is_practical_help(loc, action_a, action_b, interaction_type)
                helper_id = None
                receiver_id = None
                if interaction_type == "support" and practical_help:
                    if action_a == Action.HELP_OTHERS and action_b != Action.HELP_OTHERS:
                        helper_id, receiver_id = aid_a, best_partner
                    elif action_b == Action.HELP_OTHERS and action_a != Action.HELP_OTHERS:
                        helper_id, receiver_id = best_partner, aid_a
                    elif agent_a.heart.vulnerability <= agent_b.heart.vulnerability:
                        helper_id, receiver_id = aid_a, best_partner
                    else:
                        helper_id, receiver_id = best_partner, aid_a

                # Update relationships
                self.relationships.update_after_interaction(
                    aid_a, best_partner, self.tick_count,
                    agent_a.heart.valence, agent_b.heart.valence,
                    interaction_type,
                    issue=issue,
                    practical_help=practical_help,
                    initiator_id=initiator_id,
                    helper_id=helper_id,
                    receiver_id=receiver_id,
                )

                # Memory
                coalition_note = ""
                if rival_groups:
                    coalition_note = f" under rivalry with {', '.join(rival_groups[:2])}"
                elif shared_groups:
                    coalition_note = f" within {', '.join(shared_groups[:2])}"
                agent_a.add_memory(
                    self.tick_count,
                    f"{interaction_type} interaction with {agent_b.personality.name} around {issue}{coalition_note}",
                    best_partner,
                )
                agent_b.add_memory(
                    self.tick_count,
                    f"{interaction_type} interaction with {agent_a.personality.name} around {issue}{coalition_note}",
                    aid_a,
                )

                interactions.append({
                    "agent_a": aid_a,
                    "agent_b": best_partner,
                    "type": interaction_type,
                    "location": loc,
                    "issue": issue,
                    "practical_help": practical_help,
                    "initiator_id": initiator_id,
                    "shared_groups": list(shared_groups),
                    "rival_groups": list(rival_groups),
                    "debt_ab": round(debt_ab, 3),
                    "debt_ba": round(debt_ba, 3),
                    "grievance_ab": round(grievance_ab, 3),
                    "grievance_ba": round(grievance_ba, 3),
                    "action_a": action_a.name,
                    "action_b": action_b.name,
                })

        return interactions

    def _apply_group_event_effects(
        self,
        tick_events: list[ScheduledEvent],
        by_location: dict[str, list[str]],
    ) -> None:
        """Large public events should reshape blocs, suspicion, and obligations.

        This keeps town-wide meetings, leaks, and debt scares from collapsing
        into a handful of pair interactions.
        """
        for event in tick_events:
            participants = self._event_participants(event, by_location)
            if len(participants) < 2:
                continue

            issue = self._infer_interaction_issue(event.location, "neutral")
            source_id = None
            if event.kind == "whistleblower_leak":
                source_id = max(
                    participants,
                    key=lambda aid: self.agents[aid].secret_pressure + (0.25 if self.agents[aid].private_burden else 0.0),
                )

            for aid_a, aid_b in self._select_group_pairs(participants):
                agent_a = self.agents[aid_a]
                agent_b = self.agents[aid_b]
                rel = self.relationships.get_or_create(aid_a, aid_b)
                rel.familiarity += 1
                rel.last_interaction = self.tick_count
                rel.last_issue = issue

                shared_groups = agent_a.shared_coalitions(agent_b)
                rival_groups = agent_a.rival_overlap(agent_b)
                same_side = bool(shared_groups)
                opposite_side = bool(rival_groups)

                if event.kind in {"mutual_aid_hub", "neighborhood_meeting", "organizing_meeting"}:
                    if same_side or not opposite_side:
                        rel.trust = min(1.0, rel.trust + 0.008 + (0.006 if same_side else 0.0))
                        rel.warmth = min(1.0, rel.warmth + 0.01 + (0.008 if event.kind == "mutual_aid_hub" else 0.0))
                        rel.alliance_strength = min(1.0, rel.alliance_strength + 0.012 + (0.012 if same_side else 0.0))
                    if event.kind == "mutual_aid_hub" and agent_a.debt_pressure > 0.45 and agent_b.debt_pressure < 0.35:
                        self.relationships.adjust_debt(aid_a, aid_b, 0.02)
                    elif event.kind == "mutual_aid_hub" and agent_b.debt_pressure > 0.45 and agent_a.debt_pressure < 0.35:
                        self.relationships.adjust_debt(aid_b, aid_a, 0.02)

                if event.kind in {"accountability_hearing", "conflict_flashpoint", "boycott_call"}:
                    if same_side:
                        rel.alliance_strength = min(1.0, rel.alliance_strength + 0.02)
                        rel.trust = min(1.0, rel.trust + 0.006)
                    if opposite_side:
                        rel.trust = max(-1.0, rel.trust - 0.025)
                        rel.warmth = max(-1.0, rel.warmth - 0.02)
                        rel.rivalry = min(1.0, rel.rivalry + 0.03)
                        self.relationships.set_grievance(aid_a, aid_b, self.relationships.get_grievance(aid_a, aid_b) + 0.03)
                        self.relationships.set_grievance(aid_b, aid_a, self.relationships.get_grievance(aid_b, aid_a) + 0.03)
                        self.relationships.set_resentment(aid_a, aid_b, self.relationships.get_resentment(aid_a, aid_b) + 0.02)
                        self.relationships.set_resentment(aid_b, aid_a, self.relationships.get_resentment(aid_b, aid_a) + 0.02)

                if event.kind == "coalition_caucus" and same_side:
                    rel.alliance_strength = min(1.0, rel.alliance_strength + 0.025)
                    rel.trust = min(1.0, rel.trust + 0.01)
                    if max(agent_a.secret_pressure, agent_b.secret_pressure, agent_a.ambition, agent_b.ambition) > 0.55:
                        rel.trust = max(-1.0, rel.trust - 0.012)
                        self.relationships.set_grievance(aid_a, aid_b, self.relationships.get_grievance(aid_a, aid_b) + 0.012)
                        self.relationships.set_grievance(aid_b, aid_a, self.relationships.get_grievance(aid_b, aid_a) + 0.012)

                if event.kind == "rumor_wave":
                    if opposite_side:
                        rel.trust = max(-1.0, rel.trust - 0.01)
                        self.relationships.set_resentment(aid_a, aid_b, self.relationships.get_resentment(aid_a, aid_b) + 0.015)
                        self.relationships.set_resentment(aid_b, aid_a, self.relationships.get_resentment(aid_b, aid_a) + 0.015)
                    elif not same_side and (agent_a.private_burden or agent_b.private_burden):
                        rel.trust = max(-1.0, rel.trust - 0.006)

                if event.kind == "whistleblower_leak":
                    if source_id in {aid_a, aid_b}:
                        other_id = aid_b if source_id == aid_a else aid_a
                        rel.trust = max(-1.0, rel.trust - 0.02)
                        self.relationships.set_grievance(other_id, source_id, self.relationships.get_grievance(other_id, source_id) + 0.045)
                        if self.agents[source_id].shared_coalitions(self.agents[other_id]):
                            rel.betrayal_events += 1
                    elif opposite_side:
                        rel.rivalry = min(1.0, rel.rivalry + 0.02)
                        self.relationships.set_grievance(aid_a, aid_b, self.relationships.get_grievance(aid_a, aid_b) + 0.02)
                        self.relationships.set_grievance(aid_b, aid_a, self.relationships.get_grievance(aid_b, aid_a) + 0.02)

                if event.kind == "debt_crunch":
                    debt_ab = self.relationships.get_debt(aid_a, aid_b)
                    debt_ba = self.relationships.get_debt(aid_b, aid_a)
                    if debt_ab > 0.0:
                        self.relationships.adjust_debt(aid_a, aid_b, 0.03 * (1.0 - debt_ab))
                        self.relationships.set_grievance(aid_b, aid_a, self.relationships.get_grievance(aid_b, aid_a) + 0.04)
                        self.relationships.set_grievance(aid_a, aid_b, self.relationships.get_grievance(aid_a, aid_b) + 0.015)
                    if debt_ba > 0.0:
                        self.relationships.adjust_debt(aid_b, aid_a, 0.03 * (1.0 - debt_ba))
                        self.relationships.set_grievance(aid_a, aid_b, self.relationships.get_grievance(aid_a, aid_b) + 0.04)
                        self.relationships.set_grievance(aid_b, aid_a, self.relationships.get_grievance(aid_b, aid_a) + 0.015)
                    if debt_ab > 0.0 or debt_ba > 0.0:
                        rel.trust = max(-1.0, rel.trust - 0.015)
                        rel.warmth = max(-1.0, rel.warmth - 0.012)

    def _event_participants(
        self,
        event: ScheduledEvent,
        by_location: dict[str, list[str]],
    ) -> list[str]:
        if event.target_agent_ids is not None:
            participants = [
                aid for aid in event.target_agent_ids
                if aid in self.agents and self.agents[aid].location == event.location
            ]
        else:
            participants = list(by_location.get(event.location, []))
        participants.sort(
            key=lambda aid: (
                self.agents[aid].heart.vulnerability +
                self.agents[aid].heart.tension +
                len(self.agents[aid].coalitions) * 0.05,
                aid,
            ),
            reverse=True,
        )
        return participants[:14]

    def _select_group_pairs(self, participants: list[str]) -> list[tuple[str, str]]:
        scored_pairs: list[tuple[float, str, str]] = []
        for i, aid_a in enumerate(participants):
            for aid_b in participants[i + 1:]:
                agent_a = self.agents[aid_a]
                agent_b = self.agents[aid_b]
                shared_groups = agent_a.shared_coalitions(agent_b)
                rival_groups = agent_a.rival_overlap(agent_b)
                score = (
                    len(shared_groups) * 1.4 +
                    len(rival_groups) * 1.6 +
                    self.relationships.get_debt(aid_a, aid_b) * 0.7 +
                    self.relationships.get_debt(aid_b, aid_a) * 0.7 +
                    self.relationships.get_grievance(aid_a, aid_b) * 0.8 +
                    self.relationships.get_grievance(aid_b, aid_a) * 0.8
                )
                scored_pairs.append((score, aid_a, aid_b))
        scored_pairs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [(aid_a, aid_b) for _, aid_a, aid_b in scored_pairs[:24]]

    def _infer_interaction_issue(self, location: str, interaction_type: str) -> str:
        issue_map = {
            "factory_floor": "industrial fallout",
            "warehouse": "industrial fallout",
            "workers_canteen": "industrial fallout",
            "office_tower": "job insecurity",
            "trading_floor": "status pressure",
            "downtown_cafe": "rumor pressure",
            "lecture_hall": "organizing and debate",
            "library": "organizing and research",
            "student_union": "public organizing",
            "main_market": "livelihood strain",
            "food_court": "livelihood strain",
            "artisan_alley": "livelihood strain",
            "docks": "waterfront survival",
            "fish_market": "waterfront survival",
            "harbor_bar": "after-hours processing",
            "city_hall": "public accountability",
            "courthouse": "public accountability",
            "gov_offices": "public accountability",
            "hospital": "medical overload",
            "north_school": "family safety",
            "north_homes": "family safety",
            "community_center": "community care",
            "south_homes": "family safety",
            "central_park": "public organizing",
            "central_bar": "after-hours processing",
        }
        base_issue = issue_map.get(location, "general strain")
        if interaction_type == "conflict" and base_issue in {"after-hours processing", "rumor pressure"}:
            return "rumor pressure"
        if interaction_type == "support" and base_issue in {"family safety", "community care"}:
            return "community care"
        return base_issue

    def _is_practical_help(
        self,
        location: str,
        action_a: Action,
        action_b: Action,
        interaction_type: str,
    ) -> bool:
        if Action.HELP_OTHERS in (action_a, action_b):
            return True
        if interaction_type != "support":
            return False
        return location in {
            "community_center",
            "hospital",
            "main_market",
            "food_court",
            "north_homes",
            "south_homes",
            "north_school",
            "workers_canteen",
        }

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
             "grievance_toward": round(self.relationships.get_grievance(agent_id, other_id), 2),
             "grievance_from": round(self.relationships.get_grievance(other_id, agent_id), 2),
             "debt_toward": round(self.relationships.get_debt(agent_id, other_id), 2),
             "debt_from": round(self.relationships.get_debt(other_id, agent_id), 2),
             "interactions": rel.familiarity,
             "support_events": rel.support_events,
             "conflict_events": rel.conflict_events,
             "practical_help_events": rel.practical_help_events,
             "alliance_strength": round(rel.alliance_strength, 2),
             "rivalry": round(rel.rivalry, 2),
             "betrayal_events": rel.betrayal_events,
             "last_issue": rel.last_issue}
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

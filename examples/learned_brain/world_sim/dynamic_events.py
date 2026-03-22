"""Persistent town-level event generation for large-world runs.

The old ripple mechanic was stateless and only reacted to a couple of extreme
actions. This module turns endogenous events into a proper simulation layer:
recent interactions, issue pressure, and relationship semantics can now spawn
follow-up meetings, flashpoints, rumor waves, and mutual-aid scenes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .world import ScheduledEvent, World
from .world_agent import WorldAgent


DISTRICT_MAP = {
    "factory_floor": "Industrial Quarter",
    "warehouse": "Industrial Quarter",
    "workers_canteen": "Industrial Quarter",
    "office_tower": "Downtown",
    "trading_floor": "Downtown",
    "downtown_cafe": "Downtown",
    "lecture_hall": "University District",
    "library": "University District",
    "student_union": "University District",
    "main_market": "Market District",
    "food_court": "Market District",
    "artisan_alley": "Market District",
    "docks": "Waterfront",
    "fish_market": "Waterfront",
    "harbor_bar": "Waterfront",
    "city_hall": "Government Hill",
    "courthouse": "Government Hill",
    "gov_offices": "Government Hill",
    "hospital": "Suburbs North",
    "north_school": "Suburbs North",
    "north_homes": "Suburbs North",
    "community_center": "Suburbs South",
    "south_homes": "Suburbs South",
    "central_park": "Central",
    "central_bar": "Central",
}

ISSUE_BY_LOCATION = {
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

COOLDOWN_HOURS = {
    "rumor_wave": 8,
    "conflict_flashpoint": 10,
    "mutual_aid_hub": 14,
    "organizing_meeting": 16,
    "accountability_hearing": 18,
    "hospital_surge": 12,
    "neighborhood_meeting": 16,
    "waterfront_watch": 16,
    "slow_burn_followup": 20,
    "whistleblower_leak": 24,
    "debt_crunch": 18,
    "boycott_call": 18,
    "coalition_caucus": 14,
}

LOW_ACTIVITY_KINDS = {
    "mutual_aid_hub",
    "organizing_meeting",
    "accountability_hearing",
    "hospital_surge",
    "neighborhood_meeting",
    "waterfront_watch",
    "slow_burn_followup",
    "coalition_caucus",
}


@dataclass
class DynamicEventCandidate:
    score: float
    event: ScheduledEvent
    participants: list[str] = field(default_factory=list)
    duration: int = 2


def compute_district_stats(world: World) -> dict[str, dict]:
    """Compute per-district aggregate statistics."""
    district_agents: dict[str, list[WorldAgent]] = defaultdict(list)
    for agent in world.agents.values():
        district = DISTRICT_MAP.get(agent.location, "Other")
        district_agents[district].append(agent)

    stats = {}
    for district, agents in district_agents.items():
        if not agents:
            continue
        n = len(agents)
        emotions = defaultdict(int)
        for agent in agents:
            emotions[agent.heart.internal_emotion] += 1
        top_emotion = max(emotions, key=emotions.get)

        stats[district] = {
            "count": n,
            "avg_valence": sum(agent.heart.valence for agent in agents) / n,
            "avg_energy": sum(agent.heart.energy for agent in agents) / n,
            "avg_vulnerability": sum(agent.heart.vulnerability for agent in agents) / n,
            "top_emotion": top_emotion,
        }
    return stats


class DynamicEventEngine:
    """Generate endogenous events from the social state of the town."""

    def __init__(self):
        self.cooldowns: dict[tuple[str, str], int] = {}
        self.generated_counts: Counter[str] = Counter()
        self.generated_history: list[dict[str, str | int | float]] = []

    def generate(self, world: World, summary: dict, agent_meta: dict[str, dict]) -> list[ScheduledEvent]:
        """Return newly scheduled dynamic events and gather participants."""
        next_tick = world.tick_count + 1
        next_hour = next_tick % 24
        if next_hour >= 22 or next_hour < 6 or world.day <= 1:
            return []

        recent_window = world.tick_log[-36:]
        issue_stats, location_stats, recent_actions, recent_events = self._recent_metrics(recent_window)
        relationship_issue_pressure = self._relationship_issue_pressure(world)
        group_pressures = self._group_pressures(world)
        candidates: list[DynamicEventCandidate] = []

        candidates.extend(
            self._event_echo_candidates(
                world,
                summary,
                agent_meta,
                next_tick,
                next_hour,
            )
        )
        candidates.extend(
            self._flashpoint_candidates(
                world,
                summary,
                agent_meta,
                next_tick,
            )
        )
        candidates.extend(
            self._support_candidates(
                world,
                agent_meta,
                issue_stats,
                relationship_issue_pressure,
                next_tick,
                next_hour,
            )
        )
        candidates.extend(
            self._organizing_candidates(
                world,
                agent_meta,
                issue_stats,
                relationship_issue_pressure,
                next_tick,
                next_hour,
            )
        )
        candidates.extend(
            self._coalition_candidates(
                world,
                agent_meta,
                group_pressures,
                next_tick,
                next_hour,
            )
        )
        candidates.extend(
            self._butterfly_candidates(
                world,
                agent_meta,
                issue_stats,
                relationship_issue_pressure,
                next_tick,
                next_hour,
            )
        )
        candidates.extend(
            self._backstop_candidates(
                world,
                agent_meta,
                issue_stats,
                relationship_issue_pressure,
                recent_events,
                recent_actions,
                location_stats,
                next_tick,
                next_hour,
            )
        )

        selected: list[ScheduledEvent] = []
        used_locations: set[str] = set()
        kind_counts: Counter[str] = Counter()
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            kind = candidate.event.kind
            location = candidate.event.location
            if kind == "rumor_wave" and kind_counts[kind] >= 1:
                continue
            if kind == "coalition_caucus" and kind_counts[kind] >= 1:
                continue
            if location in used_locations:
                continue
            if not self._off_cooldown(kind, location, next_tick):
                continue
            self._invite_participants(world, candidate.participants, location, next_tick, candidate.duration)
            self.cooldowns[(kind, location)] = next_tick + COOLDOWN_HOURS.get(kind, 12)
            self.generated_counts[kind] += 1
            self.generated_history.append(
                {
                    "tick": next_tick,
                    "kind": kind,
                    "location": location,
                    "score": round(candidate.score, 3),
                    "description": candidate.event.description,
                }
            )
            selected.append(candidate.event)
            used_locations.add(location)
            kind_counts[kind] += 1
            if len(selected) >= 3:
                break

        return selected

    def _recent_metrics(
        self,
        recent_window: list[dict],
    ) -> tuple[dict[str, Counter], dict[str, Counter], Counter[str], Counter[str]]:
        issue_stats: dict[str, Counter] = defaultdict(Counter)
        location_stats: dict[str, Counter] = defaultdict(Counter)
        recent_actions: Counter[str] = Counter()
        recent_events: Counter[str] = Counter()

        for tick_summary in recent_window:
            for action in tick_summary.get("actions", {}).values():
                recent_actions[action["action"]] += 1
            for event in tick_summary.get("events", []):
                recent_events[event.get("kind", "scheduled")] += 1
            for interaction in tick_summary.get("interactions", []):
                issue = interaction.get("issue", "general strain")
                location = interaction.get("location", "unknown")
                interaction_type = interaction.get("type", "neutral")
                issue_stats[issue][interaction_type] += 1
                location_stats[location][interaction_type] += 1
                if interaction.get("practical_help"):
                    issue_stats[issue]["practical_help"] += 1
                    location_stats[location]["practical_help"] += 1

        return issue_stats, location_stats, recent_actions, recent_events

    def _relationship_issue_pressure(self, world: World) -> Counter[str]:
        pressure: Counter[str] = Counter()
        for rel in world.relationships._pairs.values():  # noqa: SLF001
            issue = rel.last_issue or "general strain"
            pressure[issue] += (
                rel.support_events * 0.8 +
                rel.conflict_events * 1.2 +
                rel.practical_help_events * 1.0 +
                abs(rel.alliance_strength) * 2.0 +
                (rel.grievance_ab + rel.grievance_ba) * 2.5 +
                (rel.debt_ab + rel.debt_ba) * 1.8 +
                rel.rivalry * 2.4 +
                max(rel.resentment_ab, rel.resentment_ba) * 3.0
            )
        return pressure

    def _group_pressures(self, world: World) -> dict[str, dict[str, float | str]]:
        pressures: dict[str, dict[str, float | str]] = {}
        if not world.group_profiles:
            return pressures

        for group_name, profile in world.group_profiles.items():
            members = [agent for agent in world.agents.values() if group_name in agent.coalitions]
            if not members:
                continue
            count = len(members)
            pressures[group_name] = {
                "count": float(count),
                "loyalty": sum(agent.appraisal.loyalty_pressure for agent in members) / count,
                "injustice": sum(agent.appraisal.injustice for agent in members) / count,
                "economic": sum(agent.appraisal.economic_pressure for agent in members) / count,
                "secrecy": sum(agent.appraisal.secrecy_pressure for agent in members) / count,
                "opportunity": sum(agent.appraisal.opportunity_pressure for agent in members) / count,
                "support": sum(agent.motives.protect_others for agent in members) / count,
                "location": str(profile.get("home_location", "community_center")),
                "issue": str(profile.get("issue", "general strain")),
                "label": str(profile.get("label", group_name.replace("_", " ").title())),
                "rival_count": float(len(profile.get("rivals", []))),
            }
        return pressures

    def _event_echo_candidates(
        self,
        world: World,
        summary: dict,
        agent_meta: dict[str, dict],
        next_tick: int,
        next_hour: int,
    ) -> list[DynamicEventCandidate]:
        candidates: list[DynamicEventCandidate] = []
        social_hubs = (
            ["downtown_cafe", "main_market", "community_center", "student_union"]
            if 9 <= next_hour < 17 else
            ["community_center", "central_bar", "harbor_bar", "central_park"]
        )

        for event in summary.get("events", [])[:3]:
            if event.get("kind") == "rumor_wave":
                continue
            source_location = event["location"]
            source_district = DISTRICT_MAP.get(source_location, "Unknown")
            issue = ISSUE_BY_LOCATION.get(source_location, "general strain")
            for destination in social_hubs:
                if DISTRICT_MAP.get(destination) == source_district:
                    continue
                if not self._off_cooldown("rumor_wave", destination, next_tick):
                    continue
                participants = self._select_participants(
                    world,
                    agent_meta,
                    location=destination,
                    roles={"community", "student", "market_vendor", "government_worker"},
                    concern_terms=("push back", "find someone safe", "keep income", "stay on my feet", "contain"),
                    count=18,
                    seed_ids=self._current_location_ids(world, destination),
                    issue=issue,
                    helper_weight=0.1,
                    distress_weight=0.45,
                )
                if len(participants) < 8:
                    continue
                description = (
                    f"Rumors from {source_district} sweep through {destination}. "
                    f"People compare names, blame, and half-verified details about the town's {issue}."
                )
                emotional_text = (
                    "The story keeps mutating as people repeat it. I do not know what is true anymore, "
                    "but every version makes the town feel less stable."
                )
                candidates.append(
                    DynamicEventCandidate(
                        score=1.1 + len(summary.get("events", [])) * 0.2,
                        event=ScheduledEvent(
                            tick=next_tick,
                            location=destination,
                            description=description,
                            emotional_text=emotional_text,
                            kind="rumor_wave",
                        ),
                        participants=participants,
                        duration=2,
                    )
                )
                break

        return candidates

    def _flashpoint_candidates(
        self,
        world: World,
        summary: dict,
        agent_meta: dict[str, dict],
        next_tick: int,
    ) -> list[DynamicEventCandidate]:
        by_location: dict[str, list[dict]] = defaultdict(list)
        for interaction in summary.get("interactions", []):
            by_location[interaction["location"]].append(interaction)

        candidates: list[DynamicEventCandidate] = []
        for location, interactions in by_location.items():
            conflict_count = sum(1 for item in interactions if item["type"] == "conflict")
            support_count = sum(1 for item in interactions if item["type"] == "support")
            if conflict_count == 0:
                continue
            issue = interactions[0].get("issue", ISSUE_BY_LOCATION.get(location, "general strain"))
            score = conflict_count * 1.4 + support_count * 0.3
            if score < 1.8 or not self._off_cooldown("conflict_flashpoint", location, next_tick):
                continue
            seed_ids = {item["agent_a"] for item in interactions} | {item["agent_b"] for item in interactions}
            participants = self._select_participants(
                world,
                agent_meta,
                location=location,
                roles=None,
                concern_terms=("push back", "contain", "keep authority", "stay on my feet"),
                count=max(10, min(20, len(seed_ids) + 8)),
                seed_ids=seed_ids,
                issue=issue,
                helper_weight=0.1,
                distress_weight=0.55,
            )
            if len(participants) < 6:
                continue
            description = (
                f"A tense exchange at {location} stops being private. "
                f"Bystanders start choosing sides as the town's {issue} turns into open confrontation."
            )
            emotional_text = (
                "A private argument has turned into a room-wide standoff. I can feel everyone deciding who belongs with whom."
            )
            candidates.append(
                DynamicEventCandidate(
                    score=score,
                    event=ScheduledEvent(
                        tick=next_tick,
                        location=location,
                        description=description,
                        emotional_text=emotional_text,
                        kind="conflict_flashpoint",
                    ),
                    participants=participants,
                    duration=1,
                )
            )

        return candidates

    def _support_candidates(
        self,
        world: World,
        agent_meta: dict[str, dict],
        issue_stats: dict[str, Counter],
        relationship_issue_pressure: Counter[str],
        next_tick: int,
        next_hour: int,
    ) -> list[DynamicEventCandidate]:
        candidates: list[DynamicEventCandidate] = []
        issue_to_kind = {
            "community care": ("mutual_aid_hub", self._support_location("community care", next_hour)),
            "family safety": ("neighborhood_meeting", self._support_location("family safety", next_hour)),
            "industrial fallout": ("mutual_aid_hub", self._support_location("industrial fallout", next_hour)),
            "livelihood strain": ("mutual_aid_hub", self._support_location("livelihood strain", next_hour)),
            "medical overload": ("hospital_surge", self._support_location("medical overload", next_hour)),
            "waterfront survival": ("waterfront_watch", self._support_location("waterfront survival", next_hour)),
        }

        for issue, (kind, location) in issue_to_kind.items():
            recent_support = issue_stats.get(issue, Counter())
            support_score = (
                recent_support.get("support", 0) * 1.1 +
                recent_support.get("practical_help", 0) * 1.4 +
                relationship_issue_pressure.get(issue, 0) * 0.08
            )
            if support_score < 2.0 or not self._off_cooldown(kind, location, next_tick):
                continue
            participants = self._select_participants(
                world,
                agent_meta,
                location=location,
                roles=self._support_roles(issue),
                concern_terms=("find someone safe", "keep income", "keep other people steady", "stay on my feet", "keep the damage contained"),
                count=20,
                issue=issue,
                helper_weight=0.55,
                distress_weight=0.5,
            )
            if len(participants) < 8:
                continue
            description, emotional_text = self._support_copy(issue, location)
            candidates.append(
                DynamicEventCandidate(
                    score=support_score,
                    event=ScheduledEvent(
                        tick=next_tick,
                        location=location,
                        description=description,
                        emotional_text=emotional_text,
                        kind=kind,
                    ),
                    participants=participants,
                    duration=3 if kind in {"mutual_aid_hub", "neighborhood_meeting"} else 2,
                )
            )

        return candidates

    def _coalition_candidates(
        self,
        world: World,
        agent_meta: dict[str, dict],
        group_pressures: dict[str, dict[str, float | str]],
        next_tick: int,
        next_hour: int,
    ) -> list[DynamicEventCandidate]:
        candidates: list[DynamicEventCandidate] = []
        if next_hour not in {10, 18, 19}:
            return candidates
        for group_name, pressure in group_pressures.items():
            location = str(pressure["location"])
            label = str(pressure["label"])
            issue = str(pressure["issue"])
            loyalty = float(pressure["loyalty"])
            injustice = float(pressure["injustice"])
            support = float(pressure["support"])
            secrecy = float(pressure["secrecy"])
            economic = float(pressure["economic"])

            if loyalty + max(injustice, support) > 0.98 and self._off_cooldown("coalition_caucus", location, next_tick):
                participants = self._select_participants(
                    world,
                    agent_meta,
                    location=location,
                    roles=None,
                    concern_terms=("hold my bloc together", "force public accountability", "defend my neighborhood", "keep other people steady"),
                    count=18,
                    seed_ids={agent.agent_id for agent in world.agents.values() if group_name in agent.coalitions},
                    issue=issue,
                    helper_weight=0.25,
                    distress_weight=0.3,
                )
                if len(participants) >= 8:
                    candidates.append(
                        DynamicEventCandidate(
                            score=1.7 + loyalty + max(injustice, support),
                            event=ScheduledEvent(
                                tick=next_tick,
                                location=location,
                                description=(
                                    f"{label} pulls together a closed-door caucus at {location}. "
                                    "Members compare loyalties, trade names, and decide who can be trusted to speak for the bloc."
                                ),
                                emotional_text=(
                                    "This is no longer just a crowd. It is a side taking shape, testing who is in and who is already drifting."
                                ),
                                kind="coalition_caucus",
                            ),
                            participants=participants,
                            duration=2,
                        )
                    )

            if (
                injustice + float(pressure["rival_count"]) * 0.08 + max(secrecy, economic) > 1.05 and
                next_hour >= 10 and
                self._off_cooldown("boycott_call", location, next_tick)
            ):
                target_location = "city_hall" if issue == "public accountability" else location
                participants = self._select_participants(
                    world,
                    agent_meta,
                    location=target_location,
                    roles=None,
                    concern_terms=("make", "force public accountability", "defend my neighborhood", "cover what I owe"),
                    count=20,
                    seed_ids={agent.agent_id for agent in world.agents.values() if group_name in agent.coalitions},
                    issue=issue,
                    helper_weight=0.15,
                    distress_weight=0.35,
                )
                if len(participants) >= 8:
                    candidates.append(
                        DynamicEventCandidate(
                            score=1.9 + injustice + max(secrecy, economic),
                            event=ScheduledEvent(
                                tick=next_tick,
                                location=target_location,
                                description=(
                                    f"{label} circulates a boycott and pressure campaign tied to the town's {issue}. "
                                    "People are suddenly being asked to pick a side publicly, not just complain privately."
                                ),
                                emotional_text=(
                                    "A private grievance has turned into a public demand. Staying neutral now will look like a choice."
                                ),
                                kind="boycott_call",
                            ),
                            participants=participants,
                            duration=2,
                        )
                    )

        return candidates

    def _organizing_candidates(
        self,
        world: World,
        agent_meta: dict[str, dict],
        issue_stats: dict[str, Counter],
        relationship_issue_pressure: Counter[str],
        next_tick: int,
        next_hour: int,
    ) -> list[DynamicEventCandidate]:
        candidates: list[DynamicEventCandidate] = []
        if next_hour < 8 or next_hour > 20:
            return candidates

        pushback_agents = [
            agent.agent_id
            for agent in world.agents.values()
            if "push back" in agent.appraisal.primary_concern or "authority" in agent.appraisal.primary_concern
        ]
        accountability_score = (
            issue_stats.get("public accountability", Counter()).get("conflict", 0) * 1.3 +
            relationship_issue_pressure.get("public accountability", 0) * 0.08 +
            len(pushback_agents) * 0.06
        )
        if accountability_score >= 2.4:
            location = "courthouse" if next_hour < 17 else "city_hall"
            if self._off_cooldown("accountability_hearing", location, next_tick):
                participants = self._select_participants(
                    world,
                    agent_meta,
                    location=location,
                    roles={"government_worker", "factory_worker", "student", "community"},
                    concern_terms=("push back", "keep authority", "contain", "keep income"),
                    count=22,
                    seed_ids=pushback_agents,
                    issue="public accountability",
                    helper_weight=0.15,
                    distress_weight=0.45,
                )
                if len(participants) >= 10:
                    candidates.append(
                        DynamicEventCandidate(
                            score=accountability_score,
                            event=ScheduledEvent(
                                tick=next_tick,
                                location=location,
                                description=(
                                    "Residents, workers, and officials crowd into an impromptu accountability session. "
                                    "Documents, accusations, and old decisions are suddenly everybody's business."
                                ),
                                emotional_text=(
                                    "People are lining up facts and blame in the same room. Nobody believes this can stay procedural for long."
                                ),
                                kind="accountability_hearing",
                            ),
                            participants=participants,
                            duration=2,
                        )
                    )

        organizing_score = (
            issue_stats.get("public organizing", Counter()).get("support", 0) * 1.0 +
            issue_stats.get("industrial fallout", Counter()).get("conflict", 0) * 0.8 +
            relationship_issue_pressure.get("industrial fallout", 0) * 0.06
        )
        if organizing_score >= 2.0:
            location = "student_union" if next_hour < 17 else "central_park"
            if self._off_cooldown("organizing_meeting", location, next_tick):
                participants = self._select_participants(
                    world,
                    agent_meta,
                    location=location,
                    roles={"student", "factory_worker", "community", "market_vendor"},
                    concern_terms=("push back", "keep income", "find someone safe", "stay on my feet"),
                    count=24,
                    issue="industrial fallout",
                    helper_weight=0.25,
                    distress_weight=0.4,
                )
                if len(participants) >= 10:
                    candidates.append(
                        DynamicEventCandidate(
                            score=organizing_score,
                            event=ScheduledEvent(
                                tick=next_tick,
                                location=location,
                                description=(
                                    "An organizing meeting swells past its original purpose. "
                                    "Workers, students, and neighbors swap names, promises, and plans for what comes next."
                                ),
                                emotional_text=(
                                    "The room is shifting from grief into coordination. I can feel people deciding whether to stay scared or start moving together."
                                ),
                                kind="organizing_meeting",
                            ),
                            participants=participants,
                            duration=2,
                        )
                    )

        return candidates

    def _butterfly_candidates(
        self,
        world: World,
        agent_meta: dict[str, dict],
        issue_stats: dict[str, Counter],
        relationship_issue_pressure: Counter[str],
        next_tick: int,
        next_hour: int,
    ) -> list[DynamicEventCandidate]:
        candidates: list[DynamicEventCandidate] = []
        if next_hour < 8 or next_hour > 20 or world.day <= 2:
            return candidates

        secret_agents = sorted(
            (
                agent for agent in world.agents.values()
                if agent.private_burden and (agent.secret_pressure > 0.3 or agent.appraisal.secrecy_pressure > 0.35)
            ),
            key=lambda agent: agent.secret_pressure + agent.appraisal.secrecy_pressure + agent.appraisal.injustice,
            reverse=True,
        )
        if secret_agents:
            source = secret_agents[0]
            source_meta = agent_meta.get(source.agent_id, {})
            issue = ISSUE_BY_LOCATION.get(source_meta.get("work_loc", source.location), "public accountability")
            location = source_meta.get("work_loc", source.location)
            if self._off_cooldown("whistleblower_leak", location, next_tick):
                participants = self._select_participants(
                    world,
                    agent_meta,
                    location=location,
                    roles=None,
                    concern_terms=("control the story", "force public accountability", "make", "hold my bloc together"),
                    count=22,
                    seed_ids={source.agent_id},
                    issue=issue,
                    helper_weight=0.15,
                    distress_weight=0.35,
                )
                if len(participants) >= 8:
                    candidates.append(
                        DynamicEventCandidate(
                            score=2.1 + source.secret_pressure + source.appraisal.secrecy_pressure,
                            event=ScheduledEvent(
                                tick=next_tick,
                                location=location,
                                description=(
                                    f"A private note about '{source.private_burden}' leaks out of {location}. "
                                    "One person's attempt to keep a detail buried suddenly rearranges who looks exposed across town."
                                ),
                                emotional_text=(
                                    "Something small and private just went public. The damage is not only what happened, but who can no longer deny knowing it."
                                ),
                                kind="whistleblower_leak",
                            ),
                            participants=participants,
                            duration=2,
                        )
                    )

        debt_agents = [
            agent for agent in world.agents.values()
            if agent.debt_pressure > 0.45 or agent.appraisal.economic_pressure > 0.5
        ]
        debt_pressure = (
            len(debt_agents) * 0.03 +
            relationship_issue_pressure.get("livelihood strain", 0) * 0.03 +
            issue_stats.get("livelihood strain", Counter()).get("practical_help", 0) * 0.18
        )
        if debt_pressure > 1.75 and self._off_cooldown("debt_crunch", "community_center", next_tick):
            participants = self._select_participants(
                world,
                agent_meta,
                location="community_center" if next_hour >= 17 else "main_market",
                roles={"market_vendor", "community", "factory_worker", "dock_worker"},
                concern_terms=("cover what I owe", "keep income", "stay on my feet", "defend my neighborhood"),
                count=22,
                seed_ids={agent.agent_id for agent in debt_agents[:20]},
                issue="livelihood strain",
                helper_weight=0.2,
                distress_weight=0.5,
            )
            if len(participants) >= 8:
                location = "community_center" if next_hour >= 17 else "main_market"
                candidates.append(
                    DynamicEventCandidate(
                        score=1.8 + debt_pressure,
                        event=ScheduledEvent(
                            tick=next_tick,
                            location=location,
                            description=(
                                f"A single missed payment ripples into a town-wide debt crunch at {location}. "
                                "Favours get called in, tempers sharpen, and quiet money problems stop staying private."
                            ),
                            emotional_text=(
                                "What looked like one person's money problem is suddenly everybody's problem. Every favor now has a cost attached to it."
                            ),
                            kind="debt_crunch",
                        ),
                        participants=participants,
                        duration=2,
                    )
                )

        return candidates

    def _backstop_candidates(
        self,
        world: World,
        agent_meta: dict[str, dict],
        issue_stats: dict[str, Counter],
        relationship_issue_pressure: Counter[str],
        recent_events: Counter[str],
        recent_actions: Counter[str],
        location_stats: dict[str, Counter],
        next_tick: int,
        next_hour: int,
    ) -> list[DynamicEventCandidate]:
        recent_event_count = sum(len(tick.get("events", [])) for tick in world.tick_log[-24:])
        if world.day <= 10 or recent_event_count >= 2 or next_hour not in {10, 18, 19}:
            return []

        dominant_issue = self._dominant_issue(issue_stats, relationship_issue_pressure)
        if dominant_issue == "general strain":
            if recent_actions["HELP_OTHERS"] >= recent_actions["CONFRONT"]:
                dominant_issue = "community care"
            elif recent_actions["CONFRONT"] + recent_actions["LASH_OUT"] > 0:
                dominant_issue = "public accountability"
            else:
                dominant_issue = "after-hours processing"

        kind, location, description, emotional_text = self._slow_burn_copy(dominant_issue, next_hour)
        if not self._off_cooldown(kind, location, next_tick):
            return []

        strongest_location = max(
            location_stats.items(),
            key=lambda item: item[1].get("conflict", 0) + item[1].get("support", 0) + item[1].get("practical_help", 0),
            default=(location, Counter()),
        )[0]
        participants = self._select_participants(
            world,
            agent_meta,
            location=location,
            roles=self._support_roles(dominant_issue),
            concern_terms=("find someone safe", "push back", "keep income", "stay on my feet", "keep other people steady"),
            count=22,
            seed_ids=self._current_location_ids(world, strongest_location),
            issue=dominant_issue,
            helper_weight=0.35,
            distress_weight=0.45,
        )
        if len(participants) < 8:
            return []

        score = 2.2 + relationship_issue_pressure.get(dominant_issue, 0) * 0.05 + recent_events.get("rumor_wave", 0) * 0.1
        return [
            DynamicEventCandidate(
                score=score,
                event=ScheduledEvent(
                    tick=next_tick,
                    location=location,
                    description=description,
                    emotional_text=emotional_text,
                    kind=kind,
                ),
                participants=participants,
                duration=3 if kind in {"slow_burn_followup", "neighborhood_meeting", "mutual_aid_hub"} else 2,
            )
        ]

    def _dominant_issue(
        self,
        issue_stats: dict[str, Counter],
        relationship_issue_pressure: Counter[str],
    ) -> str:
        issue_scores: Counter[str] = Counter()
        for issue, counts in issue_stats.items():
            issue_scores[issue] += (
                counts.get("support", 0) +
                counts.get("conflict", 0) * 1.3 +
                counts.get("practical_help", 0) * 1.2
            )
        for issue, score in relationship_issue_pressure.items():
            issue_scores[issue] += score * 0.06
        if not issue_scores:
            return "general strain"
        return max(issue_scores.items(), key=lambda item: item[1])[0]

    def _select_participants(
        self,
        world: World,
        agent_meta: dict[str, dict],
        *,
        location: str,
        roles: set[str] | None,
        concern_terms: tuple[str, ...],
        count: int,
        issue: str,
        helper_weight: float,
        distress_weight: float,
        seed_ids: set[str] | list[str] | None = None,
    ) -> list[str]:
        seed_ids = set(seed_ids or [])
        ranked: list[tuple[float, str]] = []
        for agent_id, agent in world.agents.items():
            meta = agent_meta.get(agent_id, {})
            score = 0.0
            if agent_id in seed_ids:
                score += 1.6
            if agent.location == location:
                score += 0.9
            if location in {meta.get("work_loc"), meta.get("home_loc"), meta.get("evening_loc")}:
                score += 0.35
            if roles and meta.get("role") in roles:
                score += 1.0
            if any(term in agent.appraisal.primary_concern for term in concern_terms):
                score += 0.75
            if any(term in agent.appraisal.interpretation for term in concern_terms):
                score += 0.35
            if issue and ISSUE_BY_LOCATION.get(agent.location) == issue:
                score += 0.25
            score += agent.motives.protect_others * helper_weight
            score += agent.motives.repair_bonds * helper_weight * 0.6
            score += agent.heart.vulnerability * distress_weight
            score += agent.motives.seek_support * distress_weight * 0.5
            score += agent.appraisal.injustice * 0.3
            if score > 0.8:
                ranked.append((score, agent_id))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [agent_id for _, agent_id in ranked[:count]]

    def _current_location_ids(self, world: World, location: str) -> set[str]:
        return {agent.agent_id for agent in world.agents.values() if agent.location == location}

    def _support_roles(self, issue: str) -> set[str]:
        if issue in {"community care", "family safety"}:
            return {"community", "healthcare", "market_vendor"}
        if issue == "industrial fallout":
            return {"factory_worker", "community", "student"}
        if issue == "livelihood strain":
            return {"market_vendor", "community", "factory_worker"}
        if issue == "medical overload":
            return {"healthcare", "community"}
        if issue == "waterfront survival":
            return {"dock_worker", "market_vendor", "community"}
        if issue == "public accountability":
            return {"government_worker", "factory_worker", "student", "community"}
        return {"community", "student", "factory_worker"}

    def _support_location(self, issue: str, next_hour: int) -> str:
        if issue == "community care":
            return "community_center"
        if issue == "family safety":
            return "north_school" if 9 <= next_hour < 17 else "community_center"
        if issue == "industrial fallout":
            return "workers_canteen" if 9 <= next_hour < 17 else "community_center"
        if issue == "livelihood strain":
            return "main_market" if 9 <= next_hour < 17 else "community_center"
        if issue == "medical overload":
            return "hospital"
        if issue == "waterfront survival":
            return "fish_market" if 9 <= next_hour < 17 else "harbor_bar"
        return "community_center"

    def _support_copy(self, issue: str, location: str) -> tuple[str, str]:
        if issue == "community care":
            return (
                f"Residents turn {location} into an improvised aid hub. Food, childcare offers, and quiet check-ins spread faster than official guidance.",
                "People are helping each other before they have answers. The room feels exhausted, but not abandoned.",
            )
        if issue == "family safety":
            return (
                f"Neighbors gather at {location} to compare symptoms, school worries, and who can watch whose kids before the next shift starts.",
                "Every practical question hides a deeper fear: whether home is still safe enough to feel like home.",
            )
        if issue == "industrial fallout":
            return (
                f"Workers and relatives crowd into {location}, trading job leads, spare cash, and names of people who might actually come through.",
                "Losing work is bleeding into everything else. People are trying to build stability out of favors and memory.",
            )
        if issue == "livelihood strain":
            return (
                f"The town's economic strain becomes visible at {location}. Vendors, customers, and laid-off workers improvise a relief line out of ordinary business.",
                "Money, pride, and solidarity are all mixed together now. Nobody wants to ask for help, but everyone can see who needs it.",
            )
        if issue == "medical overload":
            return (
                f"A fresh screening surge hits {location}. Staff improvise overflow lines while families compare symptoms and rumors in the hallway.",
                "The queue itself has become a form of dread. Waiting now feels like its own diagnosis.",
            )
        return (
            f"People pull together at {location} to protect one another after another hard day.",
            "Support is real here, but it is carrying more weight than the town's systems can handle.",
        )

    def _slow_burn_copy(self, issue: str, next_hour: int) -> tuple[str, str, str, str]:
        if issue == "public accountability":
            return (
                "slow_burn_followup",
                "courthouse" if next_hour < 17 else "city_hall",
                "A lingering accountability meeting draws people back in. Fresh timelines, private notes, and old grudges suddenly share the same table.",
                "This is no longer a single scandal. It has become a running argument about who kept the town exposed and why.",
            )
        if issue == "medical overload":
            return (
                "hospital_surge",
                "hospital",
                "Another wave of residents shows up for screenings and second opinions, forcing the hospital into another improvised surge plan.",
                "Nobody trusts that the first wave caught everything. Fear keeps arriving in the body long after the headline passes.",
            )
        if issue in {"community care", "family safety"}:
            return (
                "neighborhood_meeting",
                "community_center",
                "The town's crisis turns domestic again. Parents, neighbors, and caretakers gather to trade schedules, medications, and the names of people who still need watching.",
                "The emergency now lives inside ordinary routines. Every ride, meal, and bedtime has become part of the response.",
            )
        if issue == "waterfront survival":
            return (
                "waterfront_watch",
                "harbor_bar" if next_hour >= 17 else "docks",
                "Waterfront workers regroup to compare contamination reports, missed pay, and who is still willing to push back publicly.",
                "The shoreline is starting to feel like both a livelihood and a witness.",
            )
        if issue in {"industrial fallout", "livelihood strain"}:
            return (
                "slow_burn_followup",
                "community_center" if next_hour >= 17 else "workers_canteen",
                "A slow-burn relief and organizing session forms around rent, debt, and who still has enough leverage to call in favors.",
                "The first blast may be over, but the financial aftershock is still moving through people one bill at a time.",
            )
        return (
            "slow_burn_followup",
            "community_center" if next_hour >= 17 else "downtown_cafe",
            "The town's tension condenses into another informal check-in, half support group and half planning session.",
            "People are no longer waiting for closure. They are learning how to live inside the unresolved version.",
        )

    def _invite_participants(
        self,
        world: World,
        participants: list[str],
        location: str,
        start_tick: int,
        duration: int,
    ) -> None:
        for agent_id in participants:
            agent = world.agents.get(agent_id)
            if agent is None:
                continue
            for offset in range(duration):
                agent.location_overrides[start_tick + offset] = location

    def _off_cooldown(self, kind: str, location: str, tick: int) -> bool:
        return self.cooldowns.get((kind, location), -1) < tick

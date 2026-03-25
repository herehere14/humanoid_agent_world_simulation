"""Information Propagation Network — how knowledge spreads through the population.

Emotional contagion (contagion.py) spreads feelings. This module spreads *information*.
An agent who witnesses an event or learns about a shock doesn't just feel different —
they TELL other people. This creates realistic information cascades:

  - Word-of-mouth: agents share information with co-located agents they trust
  - Faction messaging: coalition members relay information within the group
  - Media awareness: some events become "public knowledge" after a delay
  - Rumor distortion: information changes as it passes through the network

The propagation affects agent behavior because:
  - Awareness of a shock creates anticipatory stress (even before direct impact)
  - Hearing about something from a trusted source vs. rumor affects reaction intensity
  - Factions frame information differently, creating divergent interpretations
  - Information asymmetry creates power dynamics (who knows what, and when)

Integration: called once per tick in the world loop, after interactions.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relationship import RelationshipStore
    from .world import World
    from .world_agent import WorldAgent


@dataclass
class InfoItem:
    """One piece of information circulating in the population."""
    label: str          # matches ExternalSignalPlan.label
    kind: str           # shock kind (oil_price_surge, mass_layoffs, etc.)
    source_text: str    # original text
    origin_tick: int    # when the info first appeared
    severity: float     # how alarming the information is (affects spread rate)


@dataclass
class AgentInfoState:
    """What an agent knows and how they learned it."""
    source: str = "direct"  # "direct", "trusted_contact", "faction", "rumor", "media"
    learned_tick: int = 0
    distortion: float = 0.0  # 0 = accurate, higher = more distorted
    reaction_applied: bool = False  # whether anticipatory stress was already applied


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class InfoPropagationEngine:
    """Manages information spread through the agent population.

    Tracks which agents know about each piece of information, how they
    learned it, and applies anticipatory effects when agents first learn
    about a shock through social channels.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # info_label -> agent_id -> AgentInfoState
        self.awareness: dict[str, dict[str, AgentInfoState]] = {}
        # Active information items
        self.active_info: dict[str, InfoItem] = {}
        # Propagation history for diagnostics
        self.spread_log: list[dict] = []

    def register_info(self, label: str, kind: str, source_text: str, origin_tick: int, severity: float):
        """Register a new piece of information that can spread."""
        if label not in self.active_info:
            self.active_info[label] = InfoItem(
                label=label,
                kind=kind,
                source_text=source_text,
                origin_tick=origin_tick,
                severity=severity,
            )
            self.awareness[label] = {}

    def mark_aware(self, label: str, agent_id: str, source: str = "direct", tick: int = 0):
        """Mark an agent as aware of a piece of information."""
        if label not in self.awareness:
            self.awareness[label] = {}
        if agent_id not in self.awareness[label]:
            distortion = {"direct": 0.0, "trusted_contact": 0.1, "faction": 0.15, "rumor": 0.3, "media": 0.05}.get(source, 0.2)
            self.awareness[label][agent_id] = AgentInfoState(
                source=source,
                learned_tick=tick,
                distortion=distortion,
            )

    def propagate(self, world: "World") -> dict:
        """Run one tick of information propagation.

        Called once per tick after interactions. Returns a summary of spread.

        Propagation rules:
        1. Co-located agents share info with probability based on:
           - Relationship trust/warmth
           - Information severity (alarming news spreads faster)
           - Hours since origin (news value decays)
        2. Coalition members have a bonus to share within the group
        3. Agents with social actions (SOCIALIZE, VENT, CONFRONT) are more likely to share
        4. When an agent first learns about a shock, they get anticipatory stress
        """
        tick = world.tick_count
        total_new_aware = 0
        spread_events: list[dict] = []

        # Group agents by location
        by_location: dict[str, list[str]] = defaultdict(list)
        for agent in world.agents.values():
            by_location[agent.location].append(agent.agent_id)

        for label, info in self.active_info.items():
            aware_set = self.awareness.get(label, {})
            if not aware_set:
                continue

            hours_since = tick - info.origin_tick
            # News value decays over time but never fully dies
            freshness = max(0.1, 1.0 - hours_since / 168.0)  # 7 days to decay to 10%
            base_spread_prob = min(0.8, info.severity * 0.15 * freshness)

            new_aware_this_tick: dict[str, AgentInfoState] = {}

            # Word-of-mouth at each location
            for loc, agent_ids in by_location.items():
                aware_here = [aid for aid in agent_ids if aid in aware_set]
                unaware_here = [aid for aid in agent_ids if aid not in aware_set and aid not in new_aware_this_tick]

                if not aware_here or not unaware_here:
                    continue

                for spreader_id in aware_here:
                    spreader = world.agents[spreader_id]
                    # Social agents spread more
                    social_bonus = 0.15 if spreader.last_action in ("SOCIALIZE", "VENT", "CONFRONT", "HELP_OTHERS", "CELEBRATE") else 0.0

                    for target_id in unaware_here:
                        if target_id in new_aware_this_tick:
                            continue

                        target = world.agents[target_id]
                        rel = world.relationships.get(spreader_id, target_id)

                        # Trust and warmth increase spread probability
                        rel_bonus = 0.0
                        source_type = "rumor"
                        if rel:
                            if rel.trust > 0.3 and rel.warmth > 0.2:
                                rel_bonus = 0.2
                                source_type = "trusted_contact"
                            elif rel.familiarity > 3:
                                rel_bonus = 0.08
                                source_type = "rumor"

                        # Coalition bonus
                        shared_coalitions = spreader.shared_coalitions(target)
                        if shared_coalitions:
                            rel_bonus += 0.15
                            source_type = "faction"

                        spread_prob = base_spread_prob + social_bonus + rel_bonus
                        # Cap per-pair probability to avoid unrealistic instant spread
                        spread_prob = min(0.5, spread_prob)

                        if self.rng.random() < spread_prob:
                            new_aware_this_tick[target_id] = AgentInfoState(
                                source=source_type,
                                learned_tick=tick,
                                distortion={"trusted_contact": 0.1, "faction": 0.15, "rumor": 0.3}.get(source_type, 0.2),
                            )
                            spread_events.append({
                                "label": label,
                                "from": spreader_id,
                                "to": target_id,
                                "source": source_type,
                                "location": loc,
                            })

            # Media propagation: after 24h, random unaware agents learn via media
            if hours_since >= 24 and info.severity >= 1.5:
                all_unaware = [
                    aid for aid in world.agents
                    if aid not in aware_set and aid not in new_aware_this_tick
                ]
                media_count = max(1, int(len(all_unaware) * 0.05 * freshness))
                media_targets = self.rng.sample(all_unaware, min(media_count, len(all_unaware)))
                for aid in media_targets:
                    new_aware_this_tick[aid] = AgentInfoState(
                        source="media",
                        learned_tick=tick,
                        distortion=0.05,
                    )

            # Apply new awareness and anticipatory effects
            for agent_id, info_state in new_aware_this_tick.items():
                aware_set[agent_id] = info_state
                total_new_aware += 1
                self._apply_anticipatory_effect(world.agents[agent_id], info, info_state, tick)

            self.awareness[label] = aware_set

        # Sync awareness back to world for macro aggregator
        if not hasattr(world, "_info_awareness"):
            world._info_awareness = {}
        for label, aware_set in self.awareness.items():
            world._info_awareness[label] = set(aware_set.keys())

        summary = {
            "tick": tick,
            "new_aware": total_new_aware,
            "spread_events": len(spread_events),
            "awareness_totals": {
                label: len(aware_set)
                for label, aware_set in self.awareness.items()
            },
            "awareness_pct": {
                label: round(len(aware_set) / len(world.agents), 3) if world.agents else 0
                for label, aware_set in self.awareness.items()
            },
        }
        if spread_events:
            self.spread_log.append(summary)
        return summary

    def _apply_anticipatory_effect(
        self,
        agent: "WorldAgent",
        info: InfoItem,
        info_state: AgentInfoState,
        tick: int,
    ):
        """Apply anticipatory stress when an agent first learns about a shock.

        Hearing about a shock secondhand creates stress proportional to:
        - The shock's severity
        - The agent's vulnerability
        - Trust in the source (distortion inversely affects reaction)
        - The agent's threat lens alignment with the shock type
        """
        if info_state.reaction_applied:
            return
        info_state.reaction_applied = True

        # Base reaction: scaled by severity and reduced by distortion
        trust_factor = 1.0 - info_state.distortion
        base_tension = info.severity * 0.04 * trust_factor
        base_valence = -info.severity * 0.02 * trust_factor

        # Amplify for agents whose threat lens aligns with the shock
        profile = agent.get_human_profile()
        threat_lens = profile.get("threat_lens", "")

        threat_alignment = {
            "oil_price_surge": {"scarcity": 1.5, "chaos": 1.2},
            "mass_layoffs": {"scarcity": 1.6, "betrayal": 1.3, "abandonment": 1.4},
            "brand_scandal": {"betrayal": 1.5, "exposure": 1.3, "humiliation": 1.2},
            "banking_panic": {"scarcity": 1.7, "chaos": 1.4},
            "health_crisis": {"chaos": 1.5, "abandonment": 1.3},
            "military_crisis": {"chaos": 1.6, "scarcity": 1.3},
        }

        alignment = threat_alignment.get(info.kind, {})
        amplifier = alignment.get(threat_lens, 1.0)

        agent.heart.tension = _clamp(agent.heart.tension + base_tension * amplifier)
        agent.heart.valence = _clamp(agent.heart.valence + base_valence * amplifier)

        # Add memory of hearing about the shock
        source_text = {
            "direct": "heard directly",
            "trusted_contact": "heard from someone I trust",
            "faction": "heard through my group",
            "rumor": "heard a rumor about",
            "media": "saw in the news",
        }.get(info_state.source, "heard about")

        agent.add_memory(
            tick,
            f"[info:{info.kind}] {source_text}: {info.label}",
        )

    def get_spread_report(self, world: "World") -> dict:
        """Generate a report on information spread patterns."""
        n = len(world.agents) if world.agents else 1
        report: dict = {
            "active_information": [],
            "total_agents": n,
        }
        for label, info in self.active_info.items():
            aware = self.awareness.get(label, {})
            source_counts: dict[str, int] = defaultdict(int)
            for state in aware.values():
                source_counts[state.source] += 1

            report["active_information"].append({
                "label": label,
                "kind": info.kind,
                "origin_tick": info.origin_tick,
                "severity": round(info.severity, 3),
                "total_aware": len(aware),
                "pct_aware": round(len(aware) / n, 3),
                "source_breakdown": dict(source_counts),
                "avg_distortion": round(
                    sum(s.distortion for s in aware.values()) / max(1, len(aware)), 3
                ),
            })
        return report

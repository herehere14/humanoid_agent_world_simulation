"""Emotional contagion — propagates emotional state between co-located agents.

Groups agents by location, computes the emotional "field" at each location,
and each agent absorbs the field proportional to their vulnerability and
relationship warmth with the broadcasting agents.

Complexity: O(L * K²) where L = locations, K = avg group size.
For 1000 agents across 10 locations, K ≈ 100, so ~100K operations — fast.
"""

from __future__ import annotations

from collections import defaultdict

from .world_agent import WorldAgent
from .relationship import RelationshipStore


def apply_contagion(
    agents: dict[str, WorldAgent],
    relationships: RelationshipStore,
    strength: float = 0.1,
):
    """Propagate emotional state between co-located agents.

    More aroused agents broadcast more strongly.
    Vulnerable agents absorb more.
    Warm relationships amplify the effect.
    """
    # Group by location
    by_location: dict[str, list[WorldAgent]] = defaultdict(list)
    for agent in agents.values():
        by_location[agent.location].append(agent)

    for loc, group in by_location.items():
        if len(group) < 2:
            continue

        # Compute location's emotional "field" — weighted by broadcast strength
        total_arousal = 0.0
        total_valence = 0.0
        total_weight = 0.0

        for agent in group:
            # More aroused agents radiate more; suppressed agents radiate less
            broadcast = agent.heart.arousal * (1.0 - agent.heart.suppression_effort * 0.5)
            broadcast = max(0.01, broadcast)  # everyone broadcasts a little
            total_arousal += agent.heart.arousal * broadcast
            total_valence += agent.heart.valence * broadcast
            total_weight += broadcast

        if total_weight < 0.01:
            continue

        field_arousal = total_arousal / total_weight
        field_valence = total_valence / total_weight

        # Each agent absorbs proportional to vulnerability + relationships
        for agent in group:
            # Base susceptibility from vulnerability
            susceptibility = agent.heart.vulnerability * 0.3 + 0.05

            # Relationship modifier: warm relationships amplify, cold ones dampen
            rel_modifier = 0.0
            rel_count = 0
            for other in group:
                if other.agent_id == agent.agent_id:
                    continue
                rel = relationships.get(agent.agent_id, other.agent_id)
                if rel:
                    rel_modifier += rel.warmth * 0.2  # warm = more susceptible
                    rel_count += 1
            if rel_count > 0:
                rel_modifier /= rel_count
                susceptibility += rel_modifier

            susceptibility = max(0.01, min(0.5, susceptibility))

            # Apply contagion
            delta_arousal = (field_arousal - agent.heart.arousal) * susceptibility * strength
            delta_valence = (field_valence - agent.heart.valence) * susceptibility * strength * 0.5

            agent.heart.arousal = max(0.0, min(1.0, agent.heart.arousal + delta_arousal))
            agent.heart.valence = max(0.0, min(1.0, agent.heart.valence + delta_valence))

"""Relationship tracking — sparse storage for agent-to-agent emotional bonds.

For N=1000 agents, N*(N-1)/2 ≈ 500K potential pairs. Most never interact.
Only stores pairs that have had at least one interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RelationshipVector:
    """Emotional bond between two agents.

    Trust and warmth are symmetric (simplification).
    Resentment is asymmetric — stored per-direction.
    """
    trust: float = 0.0          # -1 (betrayed) to 1 (deep trust)
    warmth: float = 0.0         # -1 (cold/hostile) to 1 (close/caring)
    resentment_ab: float = 0.0  # A's resentment toward B (0-1)
    resentment_ba: float = 0.0  # B's resentment toward A (0-1)
    familiarity: int = 0        # interaction count
    last_interaction: int = 0   # sim tick


class RelationshipStore:
    """Sparse storage for all relationship pairs.

    Key: canonical pair (min_id, max_id) so A→B and B→A share one entry.
    """

    def __init__(self):
        self._pairs: dict[tuple[str, str], RelationshipVector] = {}

    def _key(self, id_a: str, id_b: str) -> tuple[str, str]:
        return (min(id_a, id_b), max(id_a, id_b))

    def get(self, id_a: str, id_b: str) -> RelationshipVector | None:
        """Get relationship between two agents, or None if never interacted."""
        return self._pairs.get(self._key(id_a, id_b))

    def get_or_create(self, id_a: str, id_b: str) -> RelationshipVector:
        """Get or create a relationship vector."""
        key = self._key(id_a, id_b)
        if key not in self._pairs:
            self._pairs[key] = RelationshipVector()
        return self._pairs[key]

    def get_resentment(self, from_id: str, toward_id: str) -> float:
        """Get directional resentment."""
        key = self._key(from_id, toward_id)
        rel = self._pairs.get(key)
        if rel is None:
            return 0.0
        if from_id == key[0]:
            return rel.resentment_ab
        return rel.resentment_ba

    def set_resentment(self, from_id: str, toward_id: str, value: float):
        """Set directional resentment."""
        rel = self.get_or_create(from_id, toward_id)
        key = self._key(from_id, toward_id)
        if from_id == key[0]:
            rel.resentment_ab = max(0.0, min(1.0, value))
        else:
            rel.resentment_ba = max(0.0, min(1.0, value))

    def update_after_interaction(
        self,
        id_a: str,
        id_b: str,
        tick: int,
        valence_a: float,
        valence_b: float,
        interaction_type: str = "neutral",
    ):
        """Update relationship after an interaction between two agents.

        interaction_type: "positive", "negative", "conflict", "support", "neutral"
        """
        rel = self.get_or_create(id_a, id_b)
        rel.familiarity += 1
        rel.last_interaction = tick

        # Trust and warmth shift based on interaction type
        if interaction_type == "positive" or interaction_type == "support":
            rel.trust = min(1.0, rel.trust + 0.05)
            rel.warmth = min(1.0, rel.warmth + 0.08)
        elif interaction_type == "negative" or interaction_type == "conflict":
            rel.trust = max(-1.0, rel.trust - 0.1)
            rel.warmth = max(-1.0, rel.warmth - 0.06)
            # Resentment builds from conflict
            key = self._key(id_a, id_b)
            if id_a == key[0]:
                rel.resentment_ab = min(1.0, rel.resentment_ab + 0.15)
            else:
                rel.resentment_ba = min(1.0, rel.resentment_ba + 0.15)
        else:
            # Neutral interactions slowly build familiarity/warmth
            rel.warmth = min(1.0, rel.warmth + 0.01)
            rel.trust = min(1.0, rel.trust + 0.005)

        # Resentment slowly decays over time (forgiveness)
        rel.resentment_ab = max(0.0, rel.resentment_ab - 0.002)
        rel.resentment_ba = max(0.0, rel.resentment_ba - 0.002)

    def get_agent_relationships(self, agent_id: str) -> list[tuple[str, RelationshipVector]]:
        """Get all relationships for a specific agent, sorted by familiarity."""
        results = []
        for (a, b), rel in self._pairs.items():
            if a == agent_id:
                results.append((b, rel))
            elif b == agent_id:
                results.append((a, rel))
        results.sort(key=lambda x: x[1].familiarity, reverse=True)
        return results

    def get_closest(self, agent_id: str, n: int = 5) -> list[tuple[str, RelationshipVector]]:
        """Get N closest relationships by warmth."""
        rels = self.get_agent_relationships(agent_id)
        rels.sort(key=lambda x: x[1].warmth, reverse=True)
        return rels[:n]

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

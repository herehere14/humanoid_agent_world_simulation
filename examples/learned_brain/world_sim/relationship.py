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
    support_events: int = 0
    conflict_events: int = 0
    practical_help_events: int = 0
    alliance_strength: float = 0.0
    last_issue: str = "general strain"
    grievance_ab: float = 0.0   # A's unresolved grievance toward B
    grievance_ba: float = 0.0   # B's unresolved grievance toward A
    debt_ab: float = 0.0        # A owes B practical reciprocity
    debt_ba: float = 0.0        # B owes A practical reciprocity
    rivalry: float = 0.0        # symmetric faction / status rivalry
    betrayal_events: int = 0


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

    def get_grievance(self, from_id: str, toward_id: str) -> float:
        """Get directional unresolved grievance."""
        key = self._key(from_id, toward_id)
        rel = self._pairs.get(key)
        if rel is None:
            return 0.0
        if from_id == key[0]:
            return rel.grievance_ab
        return rel.grievance_ba

    def set_grievance(self, from_id: str, toward_id: str, value: float):
        """Set directional grievance."""
        rel = self.get_or_create(from_id, toward_id)
        key = self._key(from_id, toward_id)
        value = max(0.0, min(1.0, value))
        if from_id == key[0]:
            rel.grievance_ab = value
        else:
            rel.grievance_ba = value

    def get_debt(self, owed_by: str, owed_to: str) -> float:
        """Get directional practical debt."""
        key = self._key(owed_by, owed_to)
        rel = self._pairs.get(key)
        if rel is None:
            return 0.0
        if owed_by == key[0]:
            return rel.debt_ab
        return rel.debt_ba

    def adjust_debt(self, owed_by: str, owed_to: str, delta: float):
        """Adjust directional practical debt."""
        rel = self.get_or_create(owed_by, owed_to)
        key = self._key(owed_by, owed_to)
        if owed_by == key[0]:
            rel.debt_ab = max(0.0, min(1.0, rel.debt_ab + delta))
        else:
            rel.debt_ba = max(0.0, min(1.0, rel.debt_ba + delta))

    def update_after_interaction(
        self,
        id_a: str,
        id_b: str,
        tick: int,
        valence_a: float,
        valence_b: float,
        interaction_type: str = "neutral",
        issue: str = "general strain",
        practical_help: bool = False,
        initiator_id: str | None = None,
        helper_id: str | None = None,
        receiver_id: str | None = None,
    ):
        """Update relationship after an interaction between two agents.

        interaction_type: "positive", "negative", "conflict", "support", "neutral"
        """
        rel = self.get_or_create(id_a, id_b)
        rel.familiarity += 1
        rel.last_interaction = tick
        rel.last_issue = issue

        # Trust and warmth shift based on interaction type
        if interaction_type == "positive" or interaction_type == "support":
            trust_gain = max(0.008, 0.035 * (1.0 - max(0.0, rel.trust)))
            warmth_gain = max(0.01, 0.05 * (1.0 - max(-0.2, rel.warmth)))
            rel.trust = min(1.0, rel.trust + trust_gain)
            rel.warmth = min(1.0, rel.warmth + warmth_gain)
            rel.support_events += 1
            rel.alliance_strength = min(1.0, rel.alliance_strength + 0.04 * (1.0 - max(0.0, rel.alliance_strength)))
            if practical_help:
                rel.practical_help_events += 1
                if helper_id and receiver_id and helper_id != receiver_id:
                    existing_debt = self.get_debt(receiver_id, helper_id)
                    self.adjust_debt(receiver_id, helper_id, 0.08 * (1.0 - existing_debt))
                    self.set_grievance(receiver_id, helper_id, self.get_grievance(receiver_id, helper_id) * 0.85)
        elif interaction_type == "negative" or interaction_type == "conflict":
            trust_loss = 0.04 + rel.rivalry * 0.02 + max(rel.grievance_ab, rel.grievance_ba) * 0.015
            warmth_loss = 0.03 + rel.rivalry * 0.018
            rel.trust = max(-1.0, rel.trust - trust_loss)
            rel.warmth = max(-1.0, rel.warmth - warmth_loss)
            rel.conflict_events += 1
            rel.alliance_strength = max(-1.0, rel.alliance_strength - 0.06)
            rel.rivalry = min(1.0, rel.rivalry + 0.05 * (1.0 - rel.rivalry))

            if initiator_id == id_a:
                self.set_resentment(id_b, id_a, self.get_resentment(id_b, id_a) + 0.12)
                self.set_grievance(id_b, id_a, self.get_grievance(id_b, id_a) + 0.14)
                self.set_grievance(id_a, id_b, self.get_grievance(id_a, id_b) + 0.05)
            elif initiator_id == id_b:
                self.set_resentment(id_a, id_b, self.get_resentment(id_a, id_b) + 0.12)
                self.set_grievance(id_a, id_b, self.get_grievance(id_a, id_b) + 0.14)
                self.set_grievance(id_b, id_a, self.get_grievance(id_b, id_a) + 0.05)
            else:
                self.set_resentment(id_a, id_b, self.get_resentment(id_a, id_b) + 0.08)
                self.set_resentment(id_b, id_a, self.get_resentment(id_b, id_a) + 0.08)
                self.set_grievance(id_a, id_b, self.get_grievance(id_a, id_b) + 0.08)
                self.set_grievance(id_b, id_a, self.get_grievance(id_b, id_a) + 0.08)

            if rel.support_events > 2 or rel.practical_help_events > 1 or rel.trust > 0.25:
                rel.betrayal_events += 1
        else:
            # Neutral interactions mostly build familiarity, not deep attachment.
            rel.warmth = min(1.0, rel.warmth + 0.003 * (1.0 - max(0.0, rel.warmth)))
            rel.trust = min(1.0, rel.trust + 0.0015 * (1.0 - max(0.0, rel.trust)))
            rel.alliance_strength = max(-1.0, min(1.0, rel.alliance_strength + 0.002))

        # Conflict and obligation linger; they decay, but slowly.
        rel.resentment_ab = max(0.0, rel.resentment_ab - 0.0006)
        rel.resentment_ba = max(0.0, rel.resentment_ba - 0.0006)
        rel.grievance_ab = max(0.0, rel.grievance_ab - 0.0008)
        rel.grievance_ba = max(0.0, rel.grievance_ba - 0.0008)
        rel.rivalry = max(0.0, rel.rivalry - 0.0004)

        # Debts fade only when the relationship becomes actively reciprocal.
        if interaction_type in {"support", "positive"} and helper_id and receiver_id:
            self.adjust_debt(helper_id, receiver_id, -0.04)

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

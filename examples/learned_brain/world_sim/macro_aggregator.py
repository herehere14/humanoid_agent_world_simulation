"""Macro Aggregation Engine — translates micro agent behavior into macro outcomes.

This is the critical missing layer: the sim produces rich per-agent behavior but
without aggregation there's no way to answer "what happened at scale?"

The aggregator reads agent states, actions, relationships, and factions each tick
and computes inspectable macro metrics that a user or API consumer can track over
time to see how a shock propagates into society-level outcomes.

Metrics computed:
  - Sector employment stress (by role / industry)
  - Consumer confidence index
  - Social cohesion index
  - Faction power balance
  - Public sentiment distribution
  - Market pressure indicator
  - Institutional trust
  - Civil unrest potential
  - Information awareness (what % of population knows about each shock)

All metrics are normalized to [0, 1] or [-1, 1] ranges for easy comparison.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World
    from .world_agent import WorldAgent


# ---------------------------------------------------------------------------
# Sector / role groupings for macro analysis
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    # Industrial / blue-collar
    "factory_worker": "industrial",
    "dock_worker": "industrial",
    # Services / market
    "market_vendor": "services",
    "bartender": "services",
    # White-collar
    "office_professional": "white_collar",
    "office_worker": "white_collar",
    "manager": "white_collar",
    # Public sector
    "government_worker": "public_sector",
    "teacher": "public_sector",
    # Healthcare
    "healthcare": "healthcare",
    # Community / other
    "community": "community",
    "student": "education",
    "retiree": "retired",
}


@dataclass
class SectorMetrics:
    """Aggregate metrics for one economic sector."""
    sector: str
    agent_count: int = 0
    avg_valence: float = 0.5
    avg_tension: float = 0.1
    avg_energy: float = 0.8
    avg_vulnerability: float = 0.1
    avg_debt_pressure: float = 0.0
    employment_stress: float = 0.0  # composite: tension + debt + low_energy
    dominant_emotion: str = "neutral"
    conflict_rate: float = 0.0  # conflicts / agent in recent window
    support_rate: float = 0.0  # support interactions / agent
    action_distribution: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sector": self.sector,
            "agent_count": self.agent_count,
            "avg_valence": round(self.avg_valence, 3),
            "avg_tension": round(self.avg_tension, 3),
            "avg_energy": round(self.avg_energy, 3),
            "avg_vulnerability": round(self.avg_vulnerability, 3),
            "avg_debt_pressure": round(self.avg_debt_pressure, 3),
            "employment_stress": round(self.employment_stress, 3),
            "dominant_emotion": self.dominant_emotion,
            "conflict_rate": round(self.conflict_rate, 3),
            "support_rate": round(self.support_rate, 3),
            "action_distribution": {k: round(v, 3) for k, v in self.action_distribution.items()},
        }


@dataclass
class FactionMetrics:
    """Aggregate metrics for one faction/coalition."""
    name: str
    member_count: int = 0
    avg_alliance_strength: float = 0.0
    avg_internal_trust: float = 0.0
    avg_external_rivalry: float = 0.0
    cohesion: float = 0.0  # internal trust - internal grievance
    power_index: float = 0.0  # members * cohesion * avg_alliance
    avg_valence: float = 0.5
    dominant_concern: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "member_count": self.member_count,
            "avg_alliance_strength": round(self.avg_alliance_strength, 3),
            "avg_internal_trust": round(self.avg_internal_trust, 3),
            "avg_external_rivalry": round(self.avg_external_rivalry, 3),
            "cohesion": round(self.cohesion, 3),
            "power_index": round(self.power_index, 3),
            "avg_valence": round(self.avg_valence, 3),
            "dominant_concern": self.dominant_concern,
        }


@dataclass
class MacroSnapshot:
    """One tick's worth of macro metrics."""
    tick: int
    time_str: str

    # Population-level
    consumer_confidence: float = 0.5  # 0 = panic, 1 = optimistic
    social_cohesion: float = 0.5  # 0 = fragmented, 1 = unified
    institutional_trust: float = 0.5  # 0 = no trust, 1 = full trust
    civil_unrest_potential: float = 0.0  # 0 = calm, 1 = riot threshold
    market_pressure: float = 0.0  # 0 = stable, 1 = crisis
    population_mood: float = 0.0  # -1 = miserable, +1 = thriving
    information_awareness: dict[str, float] = field(default_factory=dict)  # shock_label -> % aware

    # Per-sector
    sectors: dict[str, SectorMetrics] = field(default_factory=dict)

    # Per-faction
    factions: dict[str, FactionMetrics] = field(default_factory=dict)

    # Action distribution (population-wide)
    action_distribution: dict[str, float] = field(default_factory=dict)

    # Sentiment distribution
    emotion_distribution: dict[str, float] = field(default_factory=dict)

    # Top concerns across population
    top_concerns: list[tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "tick": self.tick,
            "time": self.time_str,
            "consumer_confidence": round(self.consumer_confidence, 3),
            "social_cohesion": round(self.social_cohesion, 3),
            "institutional_trust": round(self.institutional_trust, 3),
            "civil_unrest_potential": round(self.civil_unrest_potential, 3),
            "market_pressure": round(self.market_pressure, 3),
            "population_mood": round(self.population_mood, 3),
            "information_awareness": {k: round(v, 3) for k, v in self.information_awareness.items()},
            "sectors": {k: v.as_dict() for k, v in self.sectors.items()},
            "factions": {k: v.as_dict() for k, v in self.factions.items()},
            "action_distribution": {k: round(v, 3) for k, v in self.action_distribution.items()},
            "emotion_distribution": {k: round(v, 3) for k, v in self.emotion_distribution.items()},
            "top_concerns": [(c, round(s, 3)) for c, s in self.top_concerns],
        }


class MacroAggregator:
    """Computes macro-level metrics from the world state each tick.

    Usage:
        aggregator = MacroAggregator()
        # In tick loop:
        snapshot = aggregator.compute(world)
        # After simulation:
        timeline = aggregator.get_timeline()
        summary = aggregator.get_summary()
    """

    def __init__(self):
        self.timeline: list[MacroSnapshot] = []
        self._baseline: MacroSnapshot | None = None

    def compute(self, world: "World", recent_interactions: list[dict] | None = None) -> MacroSnapshot:
        """Compute macro metrics for the current tick."""
        agents = list(world.agents.values())
        n = len(agents)
        if n == 0:
            snap = MacroSnapshot(tick=world.tick_count, time_str=world.time_str)
            self.timeline.append(snap)
            return snap

        # --- Sector metrics ---
        sector_agents: dict[str, list["WorldAgent"]] = defaultdict(list)
        for agent in agents:
            sector = SECTOR_MAP.get(agent.social_role, "other")
            sector_agents[sector].append(agent)

        sectors: dict[str, SectorMetrics] = {}
        for sector, s_agents in sector_agents.items():
            sn = len(s_agents)
            emotions: Counter[str] = Counter()
            actions: Counter[str] = Counter()
            for a in s_agents:
                emotions[a.heart.internal_emotion] += 1
                actions[a.last_action] += 1

            avg_dread_s = mean(a.dread_pressure for a in s_agents)
            sm = SectorMetrics(
                sector=sector,
                agent_count=sn,
                avg_valence=mean(a.heart.valence for a in s_agents),
                avg_tension=mean(a.heart.tension for a in s_agents),
                avg_energy=mean(a.heart.energy for a in s_agents),
                avg_vulnerability=mean(a.heart.vulnerability for a in s_agents),
                avg_debt_pressure=mean(a.debt_pressure for a in s_agents),
                dominant_emotion=emotions.most_common(1)[0][0] if emotions else "neutral",
                action_distribution={k: v / sn for k, v in actions.items()},
            )
            # Employment stress composite (includes dread for non-economic crises)
            sm.employment_stress = min(1.0, (
                sm.avg_tension * 0.25 +
                sm.avg_debt_pressure * 0.3 +
                avg_dread_s * 0.2 +
                (1.0 - sm.avg_energy) * 0.1 +
                sm.avg_vulnerability * 0.15
            ))
            sectors[sector] = sm

        # Count interactions by sector if provided
        if recent_interactions:
            sector_conflicts: Counter[str] = Counter()
            sector_supports: Counter[str] = Counter()
            for interaction in recent_interactions:
                itype = interaction.get("type", "neutral")
                for aid_key in ("agent_a", "agent_b"):
                    aid = interaction.get(aid_key)
                    if aid and aid in world.agents:
                        sector = SECTOR_MAP.get(world.agents[aid].social_role, "other")
                        if itype == "conflict":
                            sector_conflicts[sector] += 1
                        elif itype == "support":
                            sector_supports[sector] += 1
            for sector, sm in sectors.items():
                if sm.agent_count > 0:
                    sm.conflict_rate = sector_conflicts[sector] / sm.agent_count
                    sm.support_rate = sector_supports[sector] / sm.agent_count

        # --- Faction metrics ---
        coalition_members: dict[str, list["WorldAgent"]] = defaultdict(list)
        for agent in agents:
            for c in agent.coalitions:
                coalition_members[c].append(agent)

        factions: dict[str, FactionMetrics] = {}
        for name, members in coalition_members.items():
            if len(members) < 2:
                continue
            fm = FactionMetrics(name=name, member_count=len(members))
            fm.avg_valence = mean(a.heart.valence for a in members)

            # Internal relationships
            trust_vals = []
            alliance_vals = []
            grievance_vals = []
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    rel = world.relationships.get(a.agent_id, b.agent_id)
                    if rel:
                        trust_vals.append(rel.trust)
                        alliance_vals.append(rel.alliance_strength)
                        grievance_vals.append(
                            max(rel.grievance_ab, rel.grievance_ba)
                        )

            fm.avg_internal_trust = mean(trust_vals) if trust_vals else 0.0
            fm.avg_alliance_strength = mean(alliance_vals) if alliance_vals else 0.0
            avg_grievance = mean(grievance_vals) if grievance_vals else 0.0
            fm.cohesion = max(0.0, min(1.0, fm.avg_internal_trust - avg_grievance + 0.5))

            # External rivalry
            rival_vals = []
            for a in members:
                for rc in a.rival_coalitions:
                    for b in coalition_members.get(rc, []):
                        rel = world.relationships.get(a.agent_id, b.agent_id)
                        if rel:
                            rival_vals.append(rel.rivalry)
            fm.avg_external_rivalry = mean(rival_vals) if rival_vals else 0.0

            fm.power_index = min(1.0, (len(members) / n) * fm.cohesion * (1 + fm.avg_alliance_strength))

            # Dominant concern
            concerns: Counter[str] = Counter()
            for a in members:
                concerns[a.appraisal.primary_concern] += 1
            fm.dominant_concern = concerns.most_common(1)[0][0] if concerns else ""

            factions[name] = fm

        # --- Population-level metrics ---
        all_valences = [a.heart.valence for a in agents]
        all_tensions = [a.heart.tension for a in agents]
        all_energies = [a.heart.energy for a in agents]
        all_vulnerabilities = [a.heart.vulnerability for a in agents]
        all_debt = [a.debt_pressure for a in agents]
        all_dread = [a.dread_pressure for a in agents]

        avg_valence = mean(all_valences)
        avg_tension = mean(all_tensions)
        avg_energy = mean(all_energies)
        avg_vulnerability = mean(all_vulnerabilities)
        avg_debt = mean(all_debt)
        avg_dread = mean(all_dread)

        # Forward-looking expectations (like real CCI which asks about the future)
        avg_pessimism = mean(
            getattr(a, "expectation_pessimism", 0.0) for a in agents
        )

        # Consumer confidence: current state + forward-looking expectations
        # Expectations component only kicks in when pessimism > 0 (i.e. after a shock)
        # At baseline (pessimism=0), confidence is purely state-based
        expectation_drag = avg_pessimism * 0.25  # 0 at baseline, grows after shocks
        consumer_confidence = max(0.0, min(1.0,
            avg_valence * 0.3 +
            avg_energy * 0.15 +
            (1.0 - avg_debt) * 0.2 +
            (1.0 - avg_dread) * 0.2 +
            (1.0 - avg_tension) * 0.15 -
            expectation_drag  # drags confidence down when agents expect worse
        ))

        # Social cohesion: warm relationships + low rivalry + faction cooperation
        all_trust = []
        all_warmth = []
        all_rivalry = []
        for rel in world.relationships._pairs.values():
            all_trust.append(rel.trust)
            all_warmth.append(rel.warmth)
            all_rivalry.append(rel.rivalry)

        avg_trust = mean(all_trust) if all_trust else 0.0
        avg_warmth = mean(all_warmth) if all_warmth else 0.0
        avg_rivalry = mean(all_rivalry) if all_rivalry else 0.0

        social_cohesion = max(0.0, min(1.0,
            (avg_trust + 1) / 2 * 0.35 +  # trust is -1..1
            (avg_warmth + 1) / 2 * 0.35 +
            (1.0 - avg_rivalry) * 0.3
        ))

        # Institutional trust: government workers' valence + general trust toward gov roles
        gov_agents = [a for a in agents if a.social_role in ("government_worker",)]
        institutional_trust = mean(a.heart.valence for a in gov_agents) if gov_agents else 0.5
        institutional_trust = max(0.0, min(1.0,
            institutional_trust * 0.4 + avg_trust * 0.3 + (1.0 - avg_tension) * 0.3
        ))

        # Civil unrest: high tension + low impulse_control + high vulnerability + conflicts
        avg_impulse = mean(a.heart.impulse_control for a in agents)
        action_counts: Counter[str] = Counter()
        for a in agents:
            action_counts[a.last_action] += 1
        conflict_actions = (
            action_counts.get("CONFRONT", 0) +
            action_counts.get("LASH_OUT", 0)
        ) / n

        civil_unrest = max(0.0, min(1.0,
            avg_tension * 0.25 +
            (1.0 - avg_impulse) * 0.2 +
            avg_vulnerability * 0.15 +
            conflict_actions * 2.0 +  # amplify conflict signal
            avg_rivalry * 0.15 +
            (action_counts.get("FLEE", 0) / n) * 0.5
        ))

        # Market pressure: debt + tension + employment stress across industrial/services
        market_sectors = [s for k, s in sectors.items() if k in ("industrial", "services", "white_collar")]
        market_pressure = mean(s.employment_stress for s in market_sectors) if market_sectors else 0.0
        market_pressure = max(0.0, min(1.0,
            market_pressure * 0.5 + avg_debt * 0.3 + avg_tension * 0.2
        ))

        # Population mood: composite of valence, tension, debt, dread, energy
        # Raw valence alone stays too flat because SBERT routine activities pull to 0.5
        # We include persistent pressure signals (debt, dread, tension) that reflect shocks
        mood_positive = (
            (avg_valence - 0.4) * 0.4 +  # valence above 0.4 is positive
            (avg_energy - 0.5) * 0.25 +   # energy above 0.5 is positive
            (0.5 - avg_debt) * 0.3        # low debt is positive
        )
        mood_negative = (
            avg_tension * 0.4 +
            avg_vulnerability * 0.2 +
            avg_debt * 0.35 +
            avg_dread * 0.45  # dread has strongest mood impact (existential fear)
        )
        population_mood = max(-1.0, min(1.0, mood_positive - mood_negative))

        # Emotion distribution
        emotions: Counter[str] = Counter()
        for a in agents:
            emotions[a.heart.internal_emotion] += 1
        emotion_dist = {k: v / n for k, v in emotions.most_common()}

        # Action distribution
        action_dist = {k: v / n for k, v in action_counts.most_common()}

        # Top concerns
        concerns: Counter[str] = Counter()
        for a in agents:
            concerns[a.appraisal.primary_concern] += 1
        top_concerns = [(k, v / n) for k, v in concerns.most_common(8)]

        # Information awareness
        info_awareness: dict[str, float] = {}
        if hasattr(world, "_info_awareness"):
            for label, aware_set in world._info_awareness.items():
                info_awareness[label] = len(aware_set) / n

        snap = MacroSnapshot(
            tick=world.tick_count,
            time_str=world.time_str,
            consumer_confidence=consumer_confidence,
            social_cohesion=social_cohesion,
            institutional_trust=institutional_trust,
            civil_unrest_potential=civil_unrest,
            market_pressure=market_pressure,
            population_mood=population_mood,
            information_awareness=info_awareness,
            sectors=sectors,
            factions=factions,
            action_distribution=action_dist,
            emotion_distribution=emotion_dist,
            top_concerns=top_concerns,
        )

        if self._baseline is None and world.tick_count <= 2:
            self._baseline = snap

        self.timeline.append(snap)
        return snap

    def get_timeline(self) -> list[dict]:
        """Return full timeline of macro snapshots."""
        return [s.as_dict() for s in self.timeline]

    def get_timeline_series(self, metric: str) -> list[tuple[int, float]]:
        """Extract a single metric as a time series [(tick, value), ...]."""
        result = []
        for s in self.timeline:
            val = getattr(s, metric, None)
            if val is not None and isinstance(val, (int, float)):
                result.append((s.tick, float(val)))
        return result

    def get_summary(self) -> dict:
        """Summary comparing current state to baseline."""
        if not self.timeline:
            return {"error": "no data"}

        current = self.timeline[-1]
        baseline = self._baseline or self.timeline[0]

        def delta(attr: str) -> float:
            c = getattr(current, attr, 0.0)
            b = getattr(baseline, attr, 0.0)
            if isinstance(c, (int, float)) and isinstance(b, (int, float)):
                return round(c - b, 4)
            return 0.0

        return {
            "current": current.as_dict(),
            "baseline_tick": baseline.tick,
            "deltas": {
                "consumer_confidence": delta("consumer_confidence"),
                "social_cohesion": delta("social_cohesion"),
                "institutional_trust": delta("institutional_trust"),
                "civil_unrest_potential": delta("civil_unrest_potential"),
                "market_pressure": delta("market_pressure"),
                "population_mood": delta("population_mood"),
            },
            "timeline_length": len(self.timeline),
            "sector_stress_ranking": sorted(
                [(k, v.employment_stress) for k, v in current.sectors.items()],
                key=lambda x: x[1],
                reverse=True,
            ),
            "faction_power_ranking": sorted(
                [(k, v.power_index) for k, v in current.factions.items()],
                key=lambda x: x[1],
                reverse=True,
            ),
        }

    def get_shock_impact_report(self, pre_ticks: int = 24, post_ticks: int = 48) -> dict:
        """Compare macro metrics before and after a shock.

        Assumes the shock happened somewhere in the timeline. Finds the point
        of maximum rate-of-change in consumer_confidence as the shock onset.
        """
        if len(self.timeline) < pre_ticks + post_ticks:
            return self.get_summary()

        # Find shock onset: largest single-tick drop in consumer confidence
        max_drop = 0.0
        shock_idx = pre_ticks
        for i in range(1, len(self.timeline)):
            drop = self.timeline[i - 1].consumer_confidence - self.timeline[i].consumer_confidence
            if drop > max_drop:
                max_drop = drop
                shock_idx = i

        pre_start = max(0, shock_idx - pre_ticks)
        post_end = min(len(self.timeline), shock_idx + post_ticks)

        pre_snaps = self.timeline[pre_start:shock_idx]
        post_snaps = self.timeline[shock_idx:post_end]

        def avg_metric(snaps: list[MacroSnapshot], attr: str) -> float:
            vals = [getattr(s, attr, 0.0) for s in snaps]
            return mean(vals) if vals else 0.0

        metrics = [
            "consumer_confidence", "social_cohesion", "institutional_trust",
            "civil_unrest_potential", "market_pressure", "population_mood",
        ]

        report = {
            "shock_onset_tick": self.timeline[shock_idx].tick,
            "shock_onset_time": self.timeline[shock_idx].time_str,
            "pre_period": f"tick {self.timeline[pre_start].tick}-{self.timeline[shock_idx - 1].tick}" if pre_snaps else "n/a",
            "post_period": f"tick {self.timeline[shock_idx].tick}-{self.timeline[min(post_end - 1, len(self.timeline) - 1)].tick}" if post_snaps else "n/a",
            "impact": {},
        }

        for m in metrics:
            pre_val = avg_metric(pre_snaps, m)
            post_val = avg_metric(post_snaps, m)
            report["impact"][m] = {
                "pre": round(pre_val, 4),
                "post": round(post_val, 4),
                "delta": round(post_val - pre_val, 4),
                "pct_change": round((post_val - pre_val) / (pre_val + 1e-9) * 100, 1),
            }

        # Sector-level impact
        if post_snaps:
            final = post_snaps[-1]
            baseline = pre_snaps[-1] if pre_snaps else self.timeline[0]
            sector_impact = {}
            for sector in final.sectors:
                if sector in baseline.sectors:
                    sector_impact[sector] = {
                        "employment_stress_delta": round(
                            final.sectors[sector].employment_stress -
                            baseline.sectors[sector].employment_stress, 4
                        ),
                        "valence_delta": round(
                            final.sectors[sector].avg_valence -
                            baseline.sectors[sector].avg_valence, 4
                        ),
                    }
            report["sector_impact"] = sector_impact

        return report

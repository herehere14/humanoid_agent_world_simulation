"""WorldAgent — individual agent with heart state + SharedBrain singleton.

SharedBrain holds the single SBERT model and precomputed anchor embeddings.
WorldAgent holds per-agent mutable state. The heart update is a standalone
function operating on (HeartState, embedding, anchors) — no per-agent model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .relationship import RelationshipStore


# ---------------------------------------------------------------------------
# Anchor texts (reused from heart_engine.py)
# ---------------------------------------------------------------------------

AROUSAL_HIGH_ANCHORS = [
    "I'm furious and I can't hold back anymore",
    "I'm panicking, my heart is racing, I can't breathe",
    "This is outrageous, how dare they do this to me",
    "I'm terrified, something terrible is happening",
    "I'm so angry I could scream",
    "I'm losing control of the situation and I'm scared",
    "She found out what I did and she's furious with me",
    "I just got called into a meeting about layoffs",
    "The argument escalated into yelling and accusations",
    "I gambled away our money and my wife is screaming at me",
    "I'm about to lose my job and I can't afford it",
    "My spouse confronted me about a terrible mistake I made",
]

AROUSAL_LOW_ANCHORS = [
    "I don't care anymore, nothing matters",
    "I'm too tired to fight, just let it be",
    "Everything feels flat and empty",
    "I've given up, there's no point",
    "I barely slept and I can't function today",
    "I'm going through the motions but nothing feels real",
]

NEGATIVE_ANCHORS = [
    "I lost everything and it's all my fault",
    "I feel ashamed and guilty about what I did",
    "My heart is broken, I can't stop crying",
    "I'm devastated, the grief is overwhelming",
    "I'm so disappointed in myself",
    "I feel worthless and hopeless",
    "I'm angry and bitter about how I've been treated",
    "I lost all our savings gambling and my wife left",
    "My mother died and I'm struggling to cope",
    "I might get fired and we can't pay our bills",
    "I destroyed my family's trust and I can't fix it",
    "I barely slept because of guilt and anxiety",
    "My wife can't look at me after what I did",
    "The funeral was last week and I'm not okay",
    "How could you do this to us, to our family",
    "That was our emergency fund, how could you",
]

POSITIVE_ANCHORS = [
    "I'm so happy, this is the best day ever",
    "I feel proud and accomplished",
    "I'm grateful and at peace",
    "I love spending time with the people I care about",
    "I feel confident and strong",
    "This moment is beautiful and I want to hold onto it",
    "We're at the park and my daughter is laughing",
    "My student won the competition and I'm so proud",
    "My family is together and things are getting better",
    "It feels like we're healing and finding our way back",
]

TENSION_ANCHORS = [
    "I'm under tremendous pressure and I might snap",
    "The stress is unbearable, my body is so tight",
    "I'm walking on eggshells, one wrong move and everything falls apart",
    "There's a confrontation coming and I'm bracing for it",
    "I can't relax, I'm waiting for the other shoe to drop",
    "My wife is furious and I don't know what to say",
    "I might lose my job on top of everything else",
    "I'm pretending to be fine at work but I'm falling apart inside",
]


# ---------------------------------------------------------------------------
# HeartState — per-agent mutable state
# ---------------------------------------------------------------------------

@dataclass
class HeartState:
    """Physiological emotional state — runs per agent, updated purely with numpy."""

    arousal: float = 0.15
    valence: float = 0.55
    tension: float = 0.1
    impulse_control: float = 0.85
    energy: float = 0.8
    vulnerability: float = 0.1
    suppression_effort: float = 0.0

    arousal_history: list[float] = field(default_factory=list)
    valence_history: list[float] = field(default_factory=list)
    tension_peaks: list[float] = field(default_factory=list)

    surface_emotion: str = "neutral"
    internal_emotion: str = "neutral"
    divergence: float = 0.0

    # Persistent emotional wounds — severe events leave lasting traces
    # Each wound: (valence_impact, decay_rate) — decays slowly over days
    wounds: list[tuple[float, float]] = field(default_factory=list)

    turn: int = 0

    def cap_history(self, max_len: int = 100):
        """Keep history bounded for long simulations."""
        if len(self.arousal_history) > max_len:
            self.arousal_history = self.arousal_history[-max_len:]
        if len(self.valence_history) > max_len:
            self.valence_history = self.valence_history[-max_len:]
        if len(self.tension_peaks) > max_len:
            self.tension_peaks = self.tension_peaks[-max_len:]


# ---------------------------------------------------------------------------
# Personality — modifies heart update coefficients
# ---------------------------------------------------------------------------

@dataclass
class Personality:
    """Personality parameters that bias heart dynamics.

    Higher arousal_rise_rate → gets activated faster.
    Lower impulse_restore_rate → slower to regain composure.
    Higher energy_drain_rate → burns out faster under stress.
    """
    name: str
    background: str
    temperament: str

    # Heart dynamics modifiers (defaults = baseline personality)
    arousal_rise_rate: float = 0.7      # how fast arousal rises (0.5-0.9)
    arousal_decay_rate: float = 0.88    # how slowly arousal falls (0.8-0.95)
    valence_momentum: float = 0.45      # how much mood lingers (0.3-0.6)
    impulse_drain_rate: float = 0.15    # how fast filter depletes (0.1-0.25)
    impulse_restore_rate: float = 0.008 # how fast filter recovers (0.004-0.015)
    energy_drain_rate: float = 0.07     # how fast energy drops under stress (0.04-0.12)
    energy_regen_rate: float = 0.01     # passive energy recovery (0.005-0.02)
    vulnerability_weight: float = 1.0   # how easily they become vulnerable (0.5-1.5)


# ---------------------------------------------------------------------------
# MemoryEntry — compressed event log
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """One remembered event — compact for long simulations."""
    tick: int
    description: str       # short text
    valence_at_time: float
    arousal_at_time: float
    other_agent_id: str | None = None  # who was involved


@dataclass
class AppraisalState:
    """How the agent currently interprets the world."""

    threat: float = 0.0
    injustice: float = 0.0
    control_loss: float = 0.0
    social_need: float = 0.0
    duty_pressure: float = 0.0
    status_pressure: float = 0.0
    economic_pressure: float = 0.0
    loyalty_pressure: float = 0.0
    secrecy_pressure: float = 0.0
    opportunity_pressure: float = 0.0
    primary_concern: str = "keep things steady"
    interpretation: str = "Nothing feels urgent right now."
    blame_target: str = "circumstances"
    support_target: str = "nobody"


@dataclass
class MotiveState:
    """What the agent is trying to do with their current state."""

    seek_safety: float = 0.0
    seek_support: float = 0.0
    regain_control: float = 0.0
    protect_others: float = 0.0
    protect_status: float = 0.0
    hide_weakness: float = 0.0
    discharge_pressure: float = 0.0
    repair_bonds: float = 0.0
    priority: str = "stay steady"
    mask_style: str = "little masking"
    action_style: str = "plainspoken honesty"
    inner_voice: str = "Keep moving and do not make this bigger than it is."


# ---------------------------------------------------------------------------
# SharedBrain — singleton SBERT + anchor embeddings
# ---------------------------------------------------------------------------

class SharedBrain:
    """One SBERT model shared across all agents.

    Holds precomputed anchor embeddings and an LRU encoding cache
    so repeated texts (routine activities) aren't re-encoded.
    """

    _instance: SharedBrain | None = None

    def __init__(self):
        self._sbert = SentenceTransformer("all-MiniLM-L6-v2")

        # Precompute all anchor embeddings
        self.arousal_high_embs = self._batch_encode(AROUSAL_HIGH_ANCHORS)
        self.arousal_low_embs = self._batch_encode(AROUSAL_LOW_ANCHORS)
        self.negative_embs = self._batch_encode(NEGATIVE_ANCHORS)
        self.positive_embs = self._batch_encode(POSITIVE_ANCHORS)
        self.tension_embs = self._batch_encode(TENSION_ANCHORS)

        # Cache for text → embedding (avoids re-encoding routine activities)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max = 10000

    @classmethod
    def get(cls) -> SharedBrain:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = SharedBrain()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def _batch_encode(self, texts: list[str]) -> np.ndarray:
        return self._sbert.encode(texts, show_progress_bar=False).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        """Encode text, with caching."""
        if text in self._cache:
            return self._cache[text]
        emb = self._sbert.encode(text, show_progress_bar=False).astype(np.float32)
        if len(self._cache) < self._cache_max:
            self._cache[text] = emb
        return emb


# ---------------------------------------------------------------------------
# Heart update — pure function operating on state + embedding
# ---------------------------------------------------------------------------

def _cosine_sim_max(emb: np.ndarray, anchors: np.ndarray) -> float:
    """Max cosine similarity between embedding and anchor set."""
    norms_a = np.linalg.norm(anchors, axis=1, keepdims=True)
    norm_e = np.linalg.norm(emb)
    if norm_e < 1e-8:
        return 0.0
    sims = (anchors @ emb) / (norms_a.squeeze() * norm_e)
    return float(np.max(sims))


def _cosine_sim_avg(emb: np.ndarray, anchors: np.ndarray) -> float:
    """Average cosine similarity between embedding and anchor set."""
    norms_a = np.linalg.norm(anchors, axis=1, keepdims=True)
    norm_e = np.linalg.norm(emb)
    if norm_e < 1e-8:
        return 0.0
    sims = (anchors @ emb) / (norms_a.squeeze() * norm_e)
    return float(np.mean(sims))


def update_heart(
    state: HeartState,
    emb: np.ndarray,
    brain: SharedBrain,
    personality: Personality,
    event_severity: float = 0.0,
):
    """Update heart state given a text embedding. Pure numpy, no model loading.

    This is the hot path — called once per agent per tick. Must be fast.
    Personality parameters modify the update coefficients for agent divergence.

    event_severity: 0 = routine, 2.0 = general event, 3.0 = targeted personal event
    """
    state.turn += 1
    is_event = event_severity > 0

    # Raw signals from anchor similarity
    high_arousal_sim = _cosine_sim_max(emb, brain.arousal_high_embs)
    low_arousal_sim = _cosine_sim_max(emb, brain.arousal_low_embs)
    negative_sim = _cosine_sim_avg(emb, brain.negative_embs)
    positive_sim = _cosine_sim_avg(emb, brain.positive_embs)
    tension_sim = _cosine_sim_max(emb, brain.tension_embs)

    def scale(sim, low=0.15, high=0.55):
        return max(0.0, min(1.0, (sim - low) / (high - low)))

    raw_arousal_high = scale(high_arousal_sim, 0.20, 0.50)
    raw_arousal_low = scale(low_arousal_sim, 0.20, 0.50)
    raw_negative = scale(negative_sim, 0.15, 0.45)
    raw_positive = scale(positive_sim, 0.15, 0.45)
    raw_tension = scale(tension_sim, 0.20, 0.50)

    # Events amplify signals; targeted events amplify much more
    weight = max(1.0, event_severity)

    # --- Wounds: apply persistent emotional damage ---
    wound_drag = 0.0
    surviving_wounds = []
    for impact, decay in state.wounds:
        wound_drag += impact
        new_impact = impact * decay  # decay each tick
        if abs(new_impact) > 0.01:
            surviving_wounds.append((new_impact, decay))
    state.wounds = surviving_wounds

    # --- Arousal ---
    raw_arousal = max(raw_arousal_high * weight, raw_arousal_low * 0.3 * weight)
    prev = state.arousal
    if raw_arousal > prev:
        state.arousal = prev + (raw_arousal - prev) * personality.arousal_rise_rate
    else:
        state.arousal = prev * personality.arousal_decay_rate + raw_arousal * (1 - personality.arousal_decay_rate)
    state.arousal = max(0.0, min(1.0, state.arousal))
    state.arousal_history.append(state.arousal)

    # --- Valence ---
    if raw_negative > raw_positive:
        raw_valence = max(0.05, 0.5 - (raw_negative - raw_positive) * 1.5 * weight)
    elif raw_positive > raw_negative:
        raw_valence = min(0.95, 0.5 + (raw_positive - raw_negative) * 1.2 * weight)
    else:
        raw_valence = 0.5

    state.valence = state.valence * personality.valence_momentum + raw_valence * (1 - personality.valence_momentum)
    # Apply wound drag to valence (wounds pull valence down persistently)
    state.valence = max(0.0, min(1.0, state.valence - wound_drag))
    state.valence_history.append(state.valence)

    # Create wound from severe negative events
    if is_event and raw_negative > 0.3:
        wound_impact = raw_negative * weight * 0.05  # per-tick valence drag
        wound_decay = 0.992  # loses ~1% per tick, lasts ~80 ticks (~3 days)
        if event_severity >= 3.0:
            wound_impact *= 2.0   # personal events wound deeper
            wound_decay = 0.996   # and last longer (~250 ticks, ~10 days)
        state.wounds.append((wound_impact, wound_decay))

    # --- Tension ---
    raw_t = raw_tension * weight
    stress_tension = max(0, raw_negative * state.arousal * 1.5)
    combined_tension = max(raw_t, stress_tension)

    prev_t = state.tension
    if combined_tension > prev_t:
        state.tension = prev_t + (combined_tension - prev_t) * 0.6
    else:
        state.tension = prev_t * 0.85 + combined_tension * 0.15
    # Wounds keep tension elevated
    state.tension = max(0.0, min(1.0, state.tension + wound_drag * 0.5))
    if state.tension > 0.4:
        state.tension_peaks.append(state.tension)

    # --- Impulse control ---
    stress = max(0, state.arousal - 0.2) * max(0, 0.5 - state.valence)
    exhaustion_drain = max(0, 0.4 - state.energy) * 0.08
    drain = stress * personality.impulse_drain_rate + exhaustion_drain
    restore = max(0, state.valence - 0.5) * max(0, 0.4 - state.arousal) * 0.04 + personality.impulse_restore_rate
    state.impulse_control = max(0.05, min(1.0, state.impulse_control - drain + restore))

    # --- Energy ---
    if state.arousal > 0.3 or raw_negative > 0.3:
        cost = max(state.arousal, raw_negative) * personality.energy_drain_rate
    else:
        cost = 0.015
    if state.valence > 0.6:
        cost *= 0.4
    # Wounds drain energy too
    cost += wound_drag * 0.02
    state.energy = max(0.05, min(1.0, state.energy - cost + personality.energy_regen_rate))

    # --- Vulnerability ---
    v = (
        (1.0 - state.energy) * 0.3 +
        (1.0 - state.impulse_control) * 0.25 +
        state.tension * 0.2 +
        max(0, 0.4 - state.valence) * 0.25 +
        len(state.wounds) * 0.1  # having open wounds increases vulnerability
    ) * personality.vulnerability_weight
    state.vulnerability = max(0.0, min(1.0, v))

    # --- Suppression ---
    if raw_negative > 0.2 and state.impulse_control > 0.3:
        state.suppression_effort = min(1.0, raw_negative * state.impulse_control)
    else:
        state.suppression_effort = max(0, state.suppression_effort - 0.05)

    # --- Internal vs surface emotion labels ---
    # Use effective valence (including wounds) for labeling
    effective_neg = raw_negative + wound_drag * 2  # wounds amplify perceived negativity
    if effective_neg > raw_positive + 0.15:
        if raw_arousal_high > 0.4 or (state.arousal > 0.5 and wound_drag > 0.05):
            primary = "angry"
        elif raw_arousal_low > 0.3 or state.energy < 0.3:
            primary = "defeated"
        else:
            primary = "sad"
    elif raw_positive > effective_neg + 0.15:
        if raw_arousal_high > 0.3:
            primary = "excited"
        else:
            primary = "content"
    elif raw_tension > 0.3 or wound_drag > 0.02:
        primary = "anxious"
    else:
        primary = "neutral"

    state.internal_emotion = primary

    is_neg = effective_neg > raw_positive + 0.1
    if is_neg and state.impulse_control > 0.4:
        state.surface_emotion = "composed" if state.valence > 0.3 else "forced calm"
        state.divergence = min(0.8, state.impulse_control * effective_neg * 2)
    elif not is_neg and len(state.valence_history) >= 3 and any(v < 0.3 for v in state.valence_history[-6:]):
        state.surface_emotion = primary
        state.internal_emotion = "bittersweet"
        state.divergence = 0.25
    else:
        state.surface_emotion = primary
        state.divergence = 0.0

    state.cap_history()


def rest_heart(state: HeartState, hours: float = 1.0):
    """Recovery during sleep/rest. Called during rest hours.

    Recovery is slower when agent has active emotional wounds.
    """
    wound_penalty = min(0.7, len(state.wounds) * 0.15)  # wounds impair recovery

    state.energy = min(1.0, state.energy + 0.08 * (1.0 - wound_penalty) * hours)
    state.impulse_control = min(1.0, state.impulse_control + 0.03 * (1.0 - wound_penalty) * hours)
    state.tension = max(0.0, state.tension - 0.04 * (1.0 - wound_penalty) * hours)
    state.arousal = max(0.05, state.arousal * (0.8 + wound_penalty * 0.15))

    # Wounds still decay during sleep
    surviving = []
    for impact, decay in state.wounds:
        new_impact = impact * decay
        if abs(new_impact) > 0.01:
            surviving.append((new_impact, decay))
    state.wounds = surviving


# ---------------------------------------------------------------------------
# WorldAgent — the individual agent
# ---------------------------------------------------------------------------

@dataclass
class WorldAgent:
    """One agent in the world simulation."""

    agent_id: str
    personality: Personality
    heart: HeartState = field(default_factory=HeartState)
    location: str = "home"
    schedule: dict[int, str] = field(default_factory=dict)  # hour_of_day → location
    location_overrides: dict[int, str] = field(default_factory=dict)  # tick → location (day-specific)
    memory: list[MemoryEntry] = field(default_factory=list)
    last_action: str = "idle"
    last_speech: str = ""
    appraisal: AppraisalState = field(default_factory=AppraisalState)
    motives: MotiveState = field(default_factory=MotiveState)
    identity_tags: tuple[str, ...] = ()
    coalitions: tuple[str, ...] = ()
    rival_coalitions: tuple[str, ...] = ()
    private_burden: str = ""
    debt_pressure: float = 0.0
    secret_pressure: float = 0.0
    ambition: float = 0.0

    # Ring buffer max
    _max_memory: int = 50

    def add_memory(self, tick: int, description: str, other_id: str | None = None):
        """Add a compressed memory entry."""
        self.memory.append(MemoryEntry(
            tick=tick,
            description=description,
            valence_at_time=self.heart.valence,
            arousal_at_time=self.heart.arousal,
            other_agent_id=other_id,
        ))
        if len(self.memory) > self._max_memory:
            self.memory = self.memory[-self._max_memory:]

    def get_recent_memories(self, n: int = 10) -> list[MemoryEntry]:
        return self.memory[-n:]

    def _trait_text(self) -> str:
        extras = " ".join([
            *self.identity_tags,
            *self.coalitions,
            *self.rival_coalitions,
            self.private_burden,
        ]).strip()
        return f"{self.personality.background} {self.personality.temperament} {extras}".lower()

    def _has_any(self, *keywords: str) -> bool:
        text = self._trait_text()
        return any(keyword in text for keyword in keywords)

    def shared_coalitions(self, other: WorldAgent) -> tuple[str, ...]:
        return tuple(sorted(set(self.coalitions) & set(other.coalitions)))

    def rival_overlap(self, other: WorldAgent) -> tuple[str, ...]:
        overlap = (
            (set(self.rival_coalitions) & set(other.coalitions)) |
            (set(other.rival_coalitions) & set(self.coalitions))
        )
        return tuple(sorted(overlap))

    def refresh_subjective_state(
        self,
        nearby_agents: dict[str, WorldAgent],
        relationships: RelationshipStore,
    ) -> None:
        """Derive a subjective read from heart + personality + social context.

        Heart answers "how activated/distressed are you?"
        This layer answers "what does it mean to you, what are you trying to
        protect, and how will you express it?"
        """
        s = self.heart
        recent = self.get_recent_memories(4)
        recent_text = " ".join(m.description.lower() for m in recent)
        rels = relationships.get_agent_relationships(self.agent_id)

        warm_target = "nobody"
        warm_strength = 0.0
        resentment_target = "circumstances"
        resentment_strength = 0.0
        grievance_target = "circumstances"
        grievance_strength = 0.0
        debt_target = "nobody"
        debt_load = max(0.0, self.debt_pressure)
        for other_id, rel in rels[:10]:
            if rel.warmth > warm_strength:
                warm_strength = rel.warmth
                warm_target = other_id
            resent = relationships.get_resentment(self.agent_id, other_id)
            if resent > resentment_strength:
                resentment_strength = resent
                resentment_target = other_id
            grievance = relationships.get_grievance(self.agent_id, other_id)
            if grievance > grievance_strength:
                grievance_strength = grievance
                grievance_target = other_id
            debt_here = relationships.get_debt(self.agent_id, other_id)
            if debt_here > debt_load:
                debt_load = debt_here
                debt_target = other_id

        nearby_distress = max(
            (other.heart.vulnerability for other in nearby_agents.values()),
            default=0.0,
        )
        nearby_allies = sum(1 for other in nearby_agents.values() if self.shared_coalitions(other))
        nearby_rivals = sum(1 for other in nearby_agents.values() if self.rival_overlap(other))

        caretaker = self._has_any(
            "empathetic", "compassionate", "protective", "kind",
            "company heart", "counselor", "teacher", "community connector",
            "carries everyone's burdens", "carries heavy emotional load",
            "triage first", "duty bound", "shelter volunteer", "community kitchen",
        )
        guarded = self._has_any(
            "guarded", "stoic", "analytical", "cold", "methodical",
            "pragmatic", "buries feelings", "matter-of-fact",
            "message discipline", "procedural loyalist", "paper trail keeper",
        )
        humor_defense = self._has_any(
            "humor", "carefree", "party", "free spirit", "joke",
            "drama-averse", "uses humor as defense",
        )
        authoritative = self._has_any(
            "authoritative", "ceo", "director", "manager",
            "assertive", "blunt", "operations manager",
            "hearing veteran", "office survivor",
        )
        guilt_driven = self._has_any(
            "guilt-ridden", "guilty", "responsible", "feels responsible",
        )
        provider = self._has_any(
            "provider", "family-first", "lifeline", "controls the budget",
            "savings", "poor", "kids in college", "three kids",
            "toddler", "job security", "first-gen",
            "rent stressed", "survival math", "cash flow watcher",
        )
        status_driven = self._has_any(
            "ambitious", "competitive", "proud", "recently promoted",
            "founded", "authoritative", "assertive", "strategic",
            "career climber", "deal facing", "polished operator", "message control",
        )
        fierce = self._has_any(
            "fierce", "slow-burn anger", "quick to escalate",
            "explosive", "speaks his mind", "distrusts management",
        )
        organizer = self._has_any(
            "organizer", "activist", "union", "campaign", "tenant defense",
            "mutual aid", "community board", "neighborhood captain",
            "network builder", "cause driven", "phone tree runner", "document hoarder",
        )
        storyteller = self._has_any(
            "journalist", "gossip", "story", "media", "always looking for the next story",
            "broker", "knows everyone's business", "story carrier", "story chaser",
            "local reporter", "message discipline",
        )
        territorial = self._has_any(
            "neighborhood", "waterfront", "local", "homeowners", "tenant", "community elder",
            "protective of neighborhood kids", "harbor loyalist", "family crew",
            "waterfront pride", "block realist", "neighborhood anchor",
        )
        striver = self._has_any(
            "upward", "promotion", "dealmaker", "broker", "operator",
            "networked", "builder", "ambitious", "career climber",
            "deal facing", "polished operator",
        )

        event_keywords = ("layoff", "cuts", "fired", "termination", "restructuring", "rumor")
        scandal_keywords = (
            "rumor", "leak", "memo", "recording", "documents", "hearing",
            "investigators", "buyout", "rezoning", "bid", "auction",
        )
        coalition_keywords = (
            "caucus", "boycott", "rally", "meeting", "vote", "union", "bloc",
            "organizing", "hearing", "strike", "campaign",
        )
        money_keywords = (
            "rent", "debt", "bill", "arrears", "eviction", "buyout",
            "closure", "job", "contract", "overdue",
        )
        event_hit = 0.18 if any(word in recent_text for word in event_keywords) else 0.0
        scandal_hit = 0.2 if any(word in recent_text for word in scandal_keywords) else 0.0
        coalition_alert = any(word in recent_text for word in coalition_keywords)
        money_hit = 0.18 if any(word in recent_text for word in money_keywords) else 0.0
        threat = (
            max(0.0, 0.55 - s.valence) * 0.9 +
            s.vulnerability * 0.6 +
            len(s.wounds) * 0.08 +
            event_hit
        )
        injustice = (
            resentment_strength * 0.7 +
            s.tension * 0.35 +
            event_hit +
            (0.08 if fierce else 0.0)
        )
        control_loss = (
            max(0.0, 0.55 - s.impulse_control) * 0.8 +
            s.tension * 0.35 +
            max(0.0, 0.35 - s.energy) * 0.25 +
            (0.12 if "rumor" in recent_text else 0.0)
        )
        social_need = (
            s.vulnerability * 0.55 +
            max(0.0, 0.45 - s.valence) * 0.25 +
            max(0.0, warm_strength) * 0.2
        )
        duty_pressure = (
            (0.32 if caretaker else 0.0) +
            (0.18 if provider else 0.0) +
            (0.15 if authoritative else 0.0) +
            nearby_distress * 0.2
        )
        status_pressure = (
            (0.28 if status_driven else 0.0) +
            (0.18 if authoritative else 0.0) +
            max(0.0, 0.45 - s.valence) * 0.12 +
            self.ambition * 0.25
        )
        economic_pressure = (
            debt_load * 0.6 +
            self.debt_pressure * 0.45 +
            (0.22 if provider else 0.0) +
            money_hit +
            threat * 0.18
        )
        secrecy_pressure = (
            self.secret_pressure * 0.75 +
            scandal_hit +
            (0.1 if guilt_driven or guarded else 0.0)
        )
        opportunity_pressure = (
            self.ambition * 0.75 +
            (0.14 if status_driven or striver else 0.0) +
            (0.08 if storyteller else 0.0) +
            max(0.0, 0.4 - threat) * 0.08
        )
        coalition_strain = (
            min(0.18, len(self.coalitions) * 0.03) +
            nearby_allies * 0.02 +
            nearby_rivals * 0.08 +
            grievance_strength * 0.24 +
            resentment_strength * 0.18 +
            min(0.18, max(secrecy_pressure, economic_pressure) * 0.25) +
            (0.12 if coalition_alert else 0.0) +
            (0.08 if organizer or territorial else 0.0)
        )
        loyalty_pressure = (
            min(0.12, len(self.coalitions) * 0.04) +
            nearby_allies * 0.02 +
            max(0.0, warm_strength) * 0.08 +
            coalition_strain * 0.6
        )
        grievance_pressure = (
            injustice +
            grievance_strength * 0.75 +
            min(0.24, nearby_rivals * 0.08)
        )

        threat = min(1.0, threat)
        injustice = min(1.0, injustice)
        control_loss = min(1.0, control_loss)
        social_need = min(1.0, social_need)
        duty_pressure = min(1.0, duty_pressure)
        status_pressure = min(1.0, status_pressure)
        economic_pressure = min(1.0, economic_pressure)
        coalition_strain = min(1.0, coalition_strain)
        loyalty_pressure = min(1.0, loyalty_pressure)
        secrecy_pressure = min(1.0, secrecy_pressure)
        opportunity_pressure = min(1.0, opportunity_pressure)
        grievance_pressure = min(1.0, grievance_pressure)

        if secrecy_pressure > 0.52 and self.private_burden:
            blame_target = "what I have been hiding"
        elif debt_load > 0.45 and debt_target != "nobody":
            blame_target = debt_target
        elif grievance_strength > 0.25:
            blame_target = grievance_target
        elif guilt_driven and any(word in recent_text for word in ("layoff", "cuts", "fired", "termination", "restructuring")):
            blame_target = "my own decision"
        elif resentment_strength > 0.25:
            blame_target = resentment_target
        elif any(word in recent_text for word in ("layoff", "cuts", "fired", "termination", "restructuring")):
            blame_target = "management"
        elif guarded and control_loss > threat:
            blame_target = "uncertainty"
        elif provider and threat > 0.45:
            blame_target = "the bills"
        else:
            blame_target = "circumstances"

        dominant_signal = max(
            [
                ("threat", threat),
                ("social", social_need + max(0.0, warm_strength) * 0.15),
                ("control", control_loss + (0.15 if guarded or authoritative else 0.0)),
                ("duty", duty_pressure + nearby_distress * 0.15),
                ("status", status_pressure + (0.12 if authoritative else 0.0)),
                ("injustice", grievance_pressure + resentment_strength * 0.2),
                ("economic", economic_pressure + debt_load * 0.2),
                ("loyalty", loyalty_pressure + nearby_allies * 0.02),
                ("secrecy", secrecy_pressure + (0.08 if self.private_burden else 0.0)),
                ("opportunity", opportunity_pressure),
            ],
            key=lambda item: item[1],
        )[0]

        if secrecy_pressure > max(threat * 0.8, control_loss * 0.9, loyalty_pressure * 0.95) and secrecy_pressure > 0.45:
            primary_concern = "keep a damaging secret buried"
            interpretation = "One loose version of the story could expose what I have been trying to keep contained."
        elif dominant_signal == "economic" and debt_target != "nobody":
            primary_concern = f"cover what I owe before {debt_target} calls it in"
            interpretation = "The next problem is not abstract anymore; it has a name, a due date, and somebody watching."
        elif dominant_signal == "injustice" and (grievance_strength > 0.22 or nearby_rivals > 0):
            primary_concern = f"make {blame_target} pay a price"
            interpretation = f"This no longer feels like absorbing the hit; it feels like keeping score against {blame_target}."
        elif territorial and max(economic_pressure, grievance_pressure) > 0.38:
            primary_concern = "defend my neighborhood from the spillover"
            interpretation = "What hits one block never stays there; if nobody draws a line, home becomes the next casualty."
        elif caretaker and duty_pressure >= max(threat * 0.6, control_loss * 0.8, loyalty_pressure):
            primary_concern = "keep other people steady"
            interpretation = "If I lose my composure, other people will pay for it."
        elif provider and economic_pressure >= max(threat * 0.78, loyalty_pressure, social_need * 0.9):
            primary_concern = "keep income and home secure"
            interpretation = "This is not abstract anymore; it threatens the life I'm holding together."
        elif dominant_signal == "opportunity" and opportunity_pressure > 0.45:
            primary_concern = "turn chaos into leverage"
            interpretation = "A vacuum is opening somewhere in this mess, and somebody is going to benefit from stepping into it."
        elif organizer and dominant_signal in {"injustice", "loyalty"}:
            primary_concern = "force public accountability"
            interpretation = "Private outrage only matters if it becomes collective enough that nobody can wave it away."
        elif storyteller and scandal_hit:
            primary_concern = "control the story before it controls me"
            interpretation = "Whoever frames this first decides who looks guilty, weak, or disposable."
        elif (
            coalition_strain > max(duty_pressure * 0.85, social_need * 0.9, status_pressure) and
            loyalty_pressure > 0.34 and
            nearby_allies > 0 and
            (nearby_rivals > 0 or coalition_alert or max(grievance_pressure, secrecy_pressure, economic_pressure) > 0.34)
        ):
            primary_concern = "hold my bloc together"
            interpretation = "If our side splinters now, people start cutting private deals and we lose leverage fast."
        elif dominant_signal == "status" and authoritative:
            primary_concern = "keep authority intact"
            interpretation = "If I look weak now, I lose the room and the story hardens around me."
        elif dominant_signal == "control" and (guarded or authoritative):
            primary_concern = "contain the situation"
            interpretation = "Chaos is the real danger; somebody has to impose order."
        elif dominant_signal == "social":
            primary_concern = "find someone safe"
            interpretation = "I need one steady person before this turns into a free fall."
        elif dominant_signal == "injustice":
            primary_concern = f"push back against {blame_target}"
            interpretation = f"This lands like betrayal, not bad luck; I keep coming back to {blame_target}."
        elif provider:
            primary_concern = "keep income and home secure"
            interpretation = "This is not abstract anymore; it threatens the life I'm holding together."
        elif authoritative:
            primary_concern = "keep the situation from splintering"
            interpretation = "If this breaks apart on my watch, everyone will remember the collapse more than the reason."
        elif guarded:
            primary_concern = "keep the damage contained"
            interpretation = "The priority is to stop this from spreading, not to perform feelings."
        elif humor_defense:
            primary_concern = "stay employable without looking desperate"
            interpretation = "If I sound scared, I lose leverage; if I joke too hard, I look unserious."
        else:
            primary_concern = "stay on my feet"
            interpretation = "This could turn ugly fast, so every move has to buy me a little footing."

        seek_safety = min(1.0, threat * (1.0 if provider else 0.68) + (0.18 if provider else 0.0))
        seek_support = min(1.0, social_need + (0.12 if not guarded else 0.0))
        regain_control = min(1.0, control_loss + (0.22 if guarded or authoritative else 0.0))
        protect_others = min(1.0, duty_pressure + (0.3 if caretaker else 0.0) + nearby_distress * 0.25)
        protect_status = min(1.0, status_pressure + (0.22 if authoritative else 0.0) + opportunity_pressure * 0.18)
        hide_weakness = min(
            1.0,
            s.divergence * 0.7 +
            s.suppression_effort * 0.35 +
            (0.18 if guarded else 0.0) +
            (0.12 if authoritative else 0.0) +
            (0.08 if humor_defense else 0.0) +
            secrecy_pressure * 0.4,
        )
        discharge_pressure = min(
            1.0,
            max(0.0, s.arousal - 0.45) * 0.7 +
            max(0.0, grievance_pressure - 0.25) * 0.45 +
            (0.18 if fierce else 0.0),
        )
        repair_bonds = min(
            1.0,
            max(0.0, warm_strength) * 0.3 +
            (0.1 if caretaker else 0.0) +
            loyalty_pressure * 0.12
        )

        motive_map = {
            "stay safe": seek_safety,
            "find support": seek_support,
            "take control": regain_control,
            "protect other people": protect_others,
            "save face": protect_status,
            "hide weakness": hide_weakness,
            "release pressure": discharge_pressure,
            "repair trust": repair_bonds,
            "hold the bloc": min(1.0, loyalty_pressure * 0.45 + coalition_strain * 0.55),
            "collect leverage": opportunity_pressure,
        }
        priority = max(motive_map, key=motive_map.get)

        if secrecy_pressure > 0.52:
            mask_style = "speaks in careful half-truths"
            action_style = "careful omission"
        elif (
            loyalty_pressure > 0.52 and
            coalition_strain > 0.42 and
            nearby_allies > 0 and
            not caretaker and
            economic_pressure < loyalty_pressure + 0.05
        ):
            mask_style = "keeps a united front even when trust is fraying"
            action_style = "bloc discipline"
        elif debt_load > 0.45:
            mask_style = "acts normal while counting every favor"
            action_style = "hustling restraint"
        elif grievance_pressure > 0.52:
            mask_style = "stops pretending this is fine"
            action_style = "score-settling focus"
        elif opportunity_pressure > 0.5:
            mask_style = "stays polished while scanning openings"
            action_style = "calculated positioning"
        elif humor_defense and hide_weakness >= max(seek_support, seek_safety):
            mask_style = "covers fear with jokes"
            action_style = "joking deflection"
        elif guilt_driven and authoritative:
            mask_style = "uses command voice to hide guilt"
            action_style = "apologetic authority"
        elif caretaker and protect_others >= max(regain_control, protect_status):
            mask_style = "stays useful so panic stays hidden"
            action_style = "protective caretaking"
        elif authoritative and hide_weakness > 0.35:
            mask_style = "uses command voice to hide guilt"
            action_style = "command-and-stabilize"
        elif guarded and regain_control >= seek_support:
            mask_style = "locks emotion behind procedures"
            action_style = "controlled precision"
        elif fierce and discharge_pressure > 0.4:
            mask_style = "drops the mask and goes sharp"
            action_style = "sharp escalation"
        elif seek_support > 0.45:
            mask_style = "shows just enough fear to pull someone close"
            action_style = "earnest reassurance-seeking"
        else:
            mask_style = "little masking"
            action_style = "plainspoken honesty"

        support_target = warm_target if warm_strength > 0.2 else "nobody"
        if support_target == "nobody" and coalition_strain > 0.36:
            for other_id in nearby_agents:
                if self.shared_coalitions(nearby_agents[other_id]):
                    support_target = other_id
                    break
        inner_voice = (
            f"I need to {primary_concern}. {interpretation} "
            f"Right now I am leaning toward {priority}."
        )

        self.appraisal = AppraisalState(
            threat=round(threat, 3),
            injustice=round(injustice, 3),
            control_loss=round(control_loss, 3),
            social_need=round(social_need, 3),
            duty_pressure=round(duty_pressure, 3),
            status_pressure=round(status_pressure, 3),
            economic_pressure=round(economic_pressure, 3),
            loyalty_pressure=round(loyalty_pressure, 3),
            secrecy_pressure=round(secrecy_pressure, 3),
            opportunity_pressure=round(opportunity_pressure, 3),
            primary_concern=primary_concern,
            interpretation=interpretation,
            blame_target=blame_target,
            support_target=support_target,
        )
        self.motives = MotiveState(
            seek_safety=round(seek_safety, 3),
            seek_support=round(seek_support, 3),
            regain_control=round(regain_control, 3),
            protect_others=round(protect_others, 3),
            protect_status=round(protect_status, 3),
            hide_weakness=round(hide_weakness, 3),
            discharge_pressure=round(discharge_pressure, 3),
            repair_bonds=round(repair_bonds, 3),
            priority=priority,
            mask_style=mask_style,
            action_style=action_style,
            inner_voice=inner_voice,
        )

    def get_future_branches(self) -> list[dict[str, str]]:
        """Three lightweight future branches for product-facing inspection."""
        concern = self.appraisal.primary_concern
        support = self.appraisal.support_target
        blame = self.appraisal.blame_target
        style = self.motives.action_style

        likely = f"If nothing changes, {self.personality.name} keeps trying to {concern} through {style}."
        pressure = (
            f"If pressure rises again, {self.personality.name} leans harder into "
            f"{self.motives.priority} and the mask shifts toward {self.motives.mask_style}."
        )
        if self.private_burden and self.appraisal.secrecy_pressure > 0.4:
            support_path = (
                f"If the secret around '{self.private_burden}' starts to surface, {self.personality.name} either confesses early or doubles down on {blame}."
            )
        elif self.debt_pressure > 0.35:
            support_path = (
                f"If someone buys {self.personality.name} breathing room on the money side, the next choice becomes less about panic and more about allegiance."
            )
        elif support != "nobody":
            support_path = (
                f"If {support} responds well, {self.personality.name} becomes more direct and the next choice is less about {blame}."
            )
        else:
            support_path = (
                f"If a trustworthy ally appears, {self.personality.name} is most likely to soften first before changing course."
            )
        return [
            {"label": "Likely path", "summary": likely},
            {"label": "Pressure path", "summary": pressure},
            {"label": "Support path", "summary": support_path},
        ]

    def render_subjective_brief(self) -> str:
        """Compact text brief for prompts and drilldown UI."""
        app = self.appraisal
        motives = self.motives
        coalition_text = ", ".join(self.coalitions[:3]) if self.coalitions else "none"
        return (
            f"Private read: {app.interpretation}\n"
            f"Primary concern: {app.primary_concern}\n"
            f"Blame focus: {app.blame_target}\n"
            f"Likely support target: {app.support_target}\n"
            f"Coalitions: {coalition_text}\n"
            f"Economic pressure: {app.economic_pressure:.2f} | Loyalty pressure: {app.loyalty_pressure:.2f} | "
            f"Secrecy pressure: {app.secrecy_pressure:.2f}\n"
            f"Private burden: {self.private_burden or 'none'}\n"
            f"Priority motive: {motives.priority}\n"
            f"Mask: {motives.mask_style}\n"
            f"Action style: {motives.action_style}\n"
            f"Inner voice: {motives.inner_voice}"
        )

    def get_dashboard_state(self) -> dict:
        """Return state dict for dashboard display."""
        return {
            "id": self.agent_id,
            "name": self.personality.name,
            "location": self.location,
            "action": self.last_action,
            "arousal": round(self.heart.arousal, 2),
            "valence": round(self.heart.valence, 2),
            "tension": round(self.heart.tension, 2),
            "impulse_control": round(self.heart.impulse_control, 2),
            "energy": round(self.heart.energy, 2),
            "vulnerability": round(self.heart.vulnerability, 2),
            "surface": self.heart.surface_emotion,
            "internal": self.heart.internal_emotion,
            "divergence": round(self.heart.divergence, 2),
            "primary_concern": self.appraisal.primary_concern,
            "interpretation": self.appraisal.interpretation,
            "blame_target": self.appraisal.blame_target,
            "support_target": self.appraisal.support_target,
            "economic_pressure": round(self.appraisal.economic_pressure, 2),
            "loyalty_pressure": round(self.appraisal.loyalty_pressure, 2),
            "secrecy_pressure": round(self.appraisal.secrecy_pressure, 2),
            "opportunity_pressure": round(self.appraisal.opportunity_pressure, 2),
            "priority_motive": self.motives.priority,
            "mask_style": self.motives.mask_style,
            "action_style": self.motives.action_style,
            "inner_voice": self.motives.inner_voice,
            "future_branches": self.get_future_branches(),
            "subjective_brief": self.render_subjective_brief(),
            "coalitions": list(self.coalitions),
            "identity_tags": list(self.identity_tags),
            "private_burden": self.private_burden,
            "debt_pressure": round(self.debt_pressure, 2),
            "secret_pressure": round(self.secret_pressure, 2),
            "ambition": round(self.ambition, 2),
        }

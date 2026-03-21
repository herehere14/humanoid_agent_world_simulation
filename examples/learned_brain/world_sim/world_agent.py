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
        }

#!/usr/bin/env python3
"""Heart Engine — Embodied Emotional Control System.

The LLM is the "brain" (thinking, language, reasoning).
The Heart is the body's emotional system — it tracks:
  - Physiological arousal (heart rate, adrenaline, tension)
  - Impulse control (how much filter you have left)
  - Emotional energy (depletes under sustained stress)
  - Body state (tension patterns, breathing, physical sensations)
  - Hidden internal state vs expressed surface

Uses SBERT embeddings compared against emotional anchor texts to derive
arousal and valence — more reliable than mapping GRU emotion labels
which are calibrated for empathetic dialogue, not life events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from learned_brain.learned_brain_engine import LearnedBrainEngine


# ---------------------------------------------------------------------------
# Emotional anchor texts — SBERT embeddings compared to these for calibration
# ---------------------------------------------------------------------------

# High arousal anchors — mix of emotional AND event descriptions
_AROUSAL_HIGH_ANCHORS = [
    "I'm furious and I can't hold back anymore",
    "I'm panicking, my heart is racing, I can't breathe",
    "This is outrageous, how dare they do this to me",
    "I'm terrified, something terrible is happening",
    "I'm so angry I could scream",
    "I'm losing control of the situation and I'm scared",
    # Event-level severity anchors
    "She found out what I did and she's furious with me",
    "I just got called into a meeting about layoffs",
    "The argument escalated into yelling and accusations",
    "I gambled away our money and my wife is screaming at me",
    "I'm about to lose my job and I can't afford it",
    "My spouse confronted me about a terrible mistake I made",
]

# Low arousal anchors (shutdown/withdrawal)
_AROUSAL_LOW_ANCHORS = [
    "I don't care anymore, nothing matters",
    "I'm too tired to fight, just let it be",
    "Everything feels flat and empty",
    "I've given up, there's no point",
    "I barely slept and I can't function today",
    "I'm going through the motions but nothing feels real",
]

# Negative valence anchors — both emotions AND events
_NEGATIVE_ANCHORS = [
    "I lost everything and it's all my fault",
    "I feel ashamed and guilty about what I did",
    "My heart is broken, I can't stop crying",
    "I'm devastated, the grief is overwhelming",
    "I'm so disappointed in myself",
    "I feel worthless and hopeless",
    "I'm angry and bitter about how I've been treated",
    # Event-level negative anchors
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

# Positive valence anchors
_POSITIVE_ANCHORS = [
    "I'm so happy, this is the best day ever",
    "I feel proud and accomplished",
    "I'm grateful and at peace",
    "I love spending time with the people I care about",
    "I feel confident and strong",
    "This moment is beautiful and I want to hold onto it",
    # Event-level positive anchors
    "We're at the park and my daughter is laughing",
    "My student won the competition and I'm so proud",
    "My family is together and things are getting better",
    "It feels like we're healing and finding our way back",
]

# Tension/stress anchors
_TENSION_ANCHORS = [
    "I'm under tremendous pressure and I might snap",
    "The stress is unbearable, my body is so tight",
    "I'm walking on eggshells, one wrong move and everything falls apart",
    "There's a confrontation coming and I'm bracing for it",
    "I can't relax, I'm waiting for the other shoe to drop",
    "My wife is furious and I don't know what to say",
    "I might lose my job on top of everything else",
    "I'm pretending to be fine at work but I'm falling apart inside",
]


@dataclass
class HeartState:
    """The body's emotional state — physiological, not cognitive."""

    # Core physiological parameters (0.0 to 1.0)
    arousal: float = 0.15         # How activated the body is (0=shutdown, 1=fight/flight)
    valence: float = 0.55         # Positive/negative emotional tone (0=miserable, 1=ecstatic)
    tension: float = 0.1          # Physical tension (jaw, shoulders, chest)
    impulse_control: float = 0.85 # How much filter remains (depletes under sustained stress)
    energy: float = 0.8           # Emotional energy reserves (depletes, slowly recharges)

    # Derived states
    vulnerability: float = 0.1    # How close to breaking down
    suppression_effort: float = 0.0

    # History for momentum
    arousal_history: list[float] = field(default_factory=list)
    valence_history: list[float] = field(default_factory=list)
    tension_peaks: list[float] = field(default_factory=list)

    # Internal vs expressed
    surface_emotion: str = "neutral"
    internal_emotion: str = "neutral"
    divergence: float = 0.0

    turn: int = 0


class HeartEngine:
    """Embodied emotional control system.

    Uses SBERT embeddings compared against emotional anchor texts
    to derive arousal, valence, and tension signals. These drive
    physiological parameters that accumulate across sessions.
    """

    def __init__(self):
        self._brain = LearnedBrainEngine()
        self._brain.reset("")
        self.state = HeartState()
        self._event_contexts: list[str] = []

        # Precompute anchor embeddings
        self._arousal_high_embs = np.array([self._brain._encode(t) for t in _AROUSAL_HIGH_ANCHORS])
        self._arousal_low_embs = np.array([self._brain._encode(t) for t in _AROUSAL_LOW_ANCHORS])
        self._negative_embs = np.array([self._brain._encode(t) for t in _NEGATIVE_ANCHORS])
        self._positive_embs = np.array([self._brain._encode(t) for t in _POSITIVE_ANCHORS])
        self._tension_embs = np.array([self._brain._encode(t) for t in _TENSION_ANCHORS])

    def _cosine_sim(self, emb: np.ndarray, anchors: np.ndarray) -> float:
        """Max cosine similarity between embedding and anchor set."""
        norms_a = np.linalg.norm(anchors, axis=1, keepdims=True)
        norm_e = np.linalg.norm(emb)
        if norm_e < 1e-8:
            return 0.0
        sims = (anchors @ emb) / (norms_a.squeeze() * norm_e)
        return float(np.max(sims))

    def _avg_sim(self, emb: np.ndarray, anchors: np.ndarray) -> float:
        """Average cosine similarity between embedding and anchor set."""
        norms_a = np.linalg.norm(anchors, axis=1, keepdims=True)
        norm_e = np.linalg.norm(emb)
        if norm_e < 1e-8:
            return 0.0
        sims = (anchors @ emb) / (norms_a.squeeze() * norm_e)
        return float(np.mean(sims))

    def new_session(self, event_context: str):
        """New session — heart state persists, conversation resets."""
        self._event_contexts.append(event_context)
        # Process event context through both GRU and anchor comparison
        self._brain.process_utterance(event_context)
        self._update_physiology(event_context, is_event=True)

    def full_reset(self):
        """Full reset for a completely new character/scenario."""
        self._brain.reset("")
        self.state = HeartState()
        self._event_contexts = []

    def process_utterance(self, text: str):
        """Process an utterance and update physiological state."""
        self._brain.process_utterance(text)
        self._update_physiology(text, is_event=False)

    def _update_physiology(self, text: str, is_event: bool = False):
        """Update physiological state using SBERT anchor comparison.

        Event contexts get stronger weight than individual utterances.
        """
        emb = self._brain._encode(text)
        self.state.turn += 1

        # --- Compute raw signals from anchor similarity ---
        high_arousal_sim = self._cosine_sim(emb, self._arousal_high_embs)
        low_arousal_sim = self._cosine_sim(emb, self._arousal_low_embs)
        negative_sim = self._avg_sim(emb, self._negative_embs)
        positive_sim = self._avg_sim(emb, self._positive_embs)
        tension_sim = self._cosine_sim(emb, self._tension_embs)

        # Scale signals — similarity ranges ~0.1-0.6
        def scale(sim, low=0.15, high=0.55):
            return max(0.0, min(1.0, (sim - low) / (high - low)))

        raw_arousal_high = scale(high_arousal_sim, 0.20, 0.50)
        raw_arousal_low = scale(low_arousal_sim, 0.20, 0.50)
        raw_negative = scale(negative_sim, 0.15, 0.45)
        raw_positive = scale(positive_sim, 0.15, 0.45)
        raw_tension = scale(tension_sim, 0.20, 0.50)

        # Event contexts get moderate weight boost
        weight = 1.5 if is_event else 1.0

        # --- Arousal ---
        raw_arousal = max(raw_arousal_high * weight, raw_arousal_low * 0.3 * weight)
        prev = self.state.arousal
        if raw_arousal > prev:
            self.state.arousal = prev + (raw_arousal - prev) * 0.7  # rises fast
        else:
            self.state.arousal = prev * 0.88 + raw_arousal * 0.12  # falls very slowly
        self.state.arousal = max(0.0, min(1.0, self.state.arousal))
        self.state.arousal_history.append(self.state.arousal)

        # --- Valence ---
        if raw_negative > raw_positive:
            raw_valence = max(0.05, 0.5 - (raw_negative - raw_positive) * 1.5 * weight)
        elif raw_positive > raw_negative:
            raw_valence = min(0.95, 0.5 + (raw_positive - raw_negative) * 1.2 * weight)
        else:
            raw_valence = 0.5

        prev_v = self.state.valence
        # Valence shifts moderately, with momentum (moods linger)
        self.state.valence = prev_v * 0.45 + raw_valence * 0.55
        self.state.valence = max(0.0, min(1.0, self.state.valence))
        self.state.valence_history.append(self.state.valence)

        # --- Tension ---
        raw_t = raw_tension * weight
        # Also: negative + aroused = tense
        stress_tension = max(0, raw_negative * self.state.arousal * 1.5)
        combined_tension = max(raw_t, stress_tension)

        prev_t = self.state.tension
        if combined_tension > prev_t:
            self.state.tension = prev_t + (combined_tension - prev_t) * 0.6
        else:
            self.state.tension = prev_t * 0.85 + combined_tension * 0.15  # releases slowly
        self.state.tension = max(0.0, min(1.0, self.state.tension))
        if self.state.tension > 0.4:
            self.state.tension_peaks.append(self.state.tension)

        # --- Impulse control ---
        # Drains from: sustained arousal + negative valence
        stress = max(0, self.state.arousal - 0.2) * max(0, 0.5 - self.state.valence)
        exhaustion_drain = max(0, 0.4 - self.state.energy) * 0.08
        drain = stress * 0.15 + exhaustion_drain
        # Restores slowly when calm + positive
        restore = max(0, self.state.valence - 0.5) * max(0, 0.4 - self.state.arousal) * 0.04 + 0.008
        self.state.impulse_control -= drain
        self.state.impulse_control += restore
        self.state.impulse_control = max(0.05, min(1.0, self.state.impulse_control))

        # --- Energy ---
        # All strong emotion costs energy; negative costs more
        if self.state.arousal > 0.3 or raw_negative > 0.3:
            cost = max(self.state.arousal, raw_negative) * 0.07
        else:
            cost = 0.015
        if self.state.valence > 0.6:
            cost *= 0.4  # positive emotions drain less
        self.state.energy -= cost
        self.state.energy += 0.01  # very slow recovery
        self.state.energy = max(0.05, min(1.0, self.state.energy))

        # --- Vulnerability ---
        self.state.vulnerability = (
            (1.0 - self.state.energy) * 0.3 +
            (1.0 - self.state.impulse_control) * 0.25 +
            self.state.tension * 0.2 +
            max(0, 0.4 - self.state.valence) * 0.25
        )
        self.state.vulnerability = max(0.0, min(1.0, self.state.vulnerability))

        # --- Suppression ---
        if raw_negative > 0.2 and self.state.impulse_control > 0.3:
            self.state.suppression_effort = min(1.0, raw_negative * self.state.impulse_control)
        else:
            self.state.suppression_effort = max(0, self.state.suppression_effort - 0.05)

        # --- Internal vs surface ---
        # Use GRU's top emotion for the label
        brain_top = self._brain.state.top_emotions
        if brain_top:
            primary = brain_top[0][0]
        else:
            primary = "neutral"

        self.state.internal_emotion = primary

        # Determine if masking
        is_neg = raw_negative > raw_positive + 0.1
        if is_neg and self.state.impulse_control > 0.4:
            self.state.surface_emotion = "composed" if self.state.valence > 0.3 else "forced calm"
            self.state.divergence = min(0.8, self.state.impulse_control * raw_negative * 2)
        elif not is_neg and len(self.state.valence_history) >= 3 and any(v < 0.3 for v in self.state.valence_history[-6:]):
            # Positive now but recent pain → bittersweet
            self.state.surface_emotion = primary
            self.state.internal_emotion = "bittersweet"
            self.state.divergence = 0.25
        else:
            self.state.surface_emotion = primary
            self.state.divergence = 0.0

    def get_embodied_state(self) -> str:
        """Generate a SINGLE short behavioral directive from the heart.

        Minimalist: one line when relevant, empty when calm.
        Less text = less constraint on the LLM = more natural output.
        Only fires for the MOST important signal.
        """
        s = self.state

        # Priority-ordered: return the FIRST matching signal only
        # 1. Extreme states (breaking point)
        if s.energy < 0.15 and s.vulnerability > 0.5:
            return "You're running on empty — every word costs effort, and your composure could shatter at any moment."

        if s.impulse_control < 0.25:
            return "Your filter is gone. Whatever you feel comes out raw and unedited."

        # 2. Bittersweet (positive moment after pain)
        if s.internal_emotion == "bittersweet":
            return "The happiness is real but fragile — it sits on top of recent pain, and both feelings are present at once."

        # 3. Emotional exhaustion
        if s.energy < 0.3:
            return "You're emotionally drained. Responses come out flat, short, effortful — you don't have the reserves for performance."

        # 4. High tension + low energy (worn down)
        if s.tension > 0.4 and s.energy < 0.5:
            return "Your body is tight and tired. You're wound up but running low — brittle, not explosive."

        # 5. Active masking
        if s.divergence > 0.3 and s.surface_emotion != s.internal_emotion:
            return f"You're holding it together on the surface, but underneath you feel {s.internal_emotion}. Small tells might leak through."

        # 6. Vulnerability
        if s.vulnerability > 0.4:
            return "You're more fragile than you look right now. A kind word could crack you open."

        # 7. High arousal
        if s.arousal > 0.5 and s.valence < 0.4:
            return "Your heart is racing and everything feels amplified. You're reactive right now."

        # 8. Sustained negative
        if len(s.valence_history) >= 4 and all(v < 0.35 for v in s.valence_history[-4:]):
            return "It's been weighing on you for a while. Each new thing piles onto what came before."

        # 9. Cautious recovery
        if len(s.valence_history) >= 5 and any(v < 0.3 for v in s.valence_history[-5:]) and s.valence > 0.55:
            return "Things feel better but you're not trusting it fully yet. Part of you is braced for it to be taken away."

        # Heart is silent — nothing important to add
        return ""

    def annotate_event_history(self, events: list[str]) -> str:
        """Annotate past events with emotional coloring from the heart's memory.

        Instead of bare facts, shows how each event FELT and what emotional
        residue it left. This is information the LLM cannot derive from text.
        """
        if not events:
            return ""

        s = self.state
        annotated = []

        for i, event in enumerate(events):
            # Use the valence/arousal history to color past events
            # Events earlier in the sequence had earlier valence values
            v_idx = min(i * 3, len(s.valence_history) - 1) if s.valence_history else 0
            a_idx = min(i * 3, len(s.arousal_history) - 1) if s.arousal_history else 0

            if v_idx < len(s.valence_history) and a_idx < len(s.arousal_history):
                v = s.valence_history[v_idx]
                a = s.arousal_history[a_idx]

                if v < 0.25:
                    color = " — this hit you HARD and the pain hasn't faded"
                elif v < 0.4:
                    color = " — this left a mark that still stings"
                elif v > 0.7:
                    color = " — a bright spot, though colored by everything else"
                elif v > 0.55 and any(vv < 0.3 for vv in s.valence_history[:v_idx + 1]):
                    color = " — good, but complicated by what came before"
                else:
                    color = ""
            else:
                color = ""

            annotated.append(f"  {i+1}. {event}{color}")

        # Add cumulative state
        if s.energy < 0.4:
            annotated.append(f"  >> Cumulative toll: you're emotionally drained from all of this. Energy at {int(s.energy * 100)}%.")
        elif len(s.tension_peaks) >= 2:
            annotated.append(f"  >> Cumulative toll: multiple high-stress moments have left you watchful and tired.")

        return "\n".join(annotated)

    def get_state_summary(self) -> dict:
        """Return a concise summary of heart state for logging."""
        s = self.state
        return {
            "arousal": round(s.arousal, 2),
            "valence": round(s.valence, 2),
            "tension": round(s.tension, 2),
            "impulse_control": round(s.impulse_control, 2),
            "energy": round(s.energy, 2),
            "vulnerability": round(s.vulnerability, 2),
            "internal": s.internal_emotion,
            "surface": s.surface_emotion,
            "divergence": round(s.divergence, 2),
        }

#!/usr/bin/env python3
"""Learned Brain Engine — replaces hardcoded BrainEngine with trained GRU model.

Instead of 16 handcoded emotional variables with hardcoded update rules,
this uses a learned latent state (32-dim) that emerged from training on
~25k real human conversations.

The engine:
  1. Encodes each utterance with frozen SentenceBERT
  2. Feeds the sequence through the trained GRU
  3. Extracts the latent state + emotion distribution
  4. Translates to a natural-language emotional narrative for the LLM
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .model import LearnedBrainModel
from .data_prep import EMOTION_LABELS

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# Vivid, human-readable emotion descriptions at different intensities
_VIVID_EMOTION_MAP = {
    "angry": {
        "mild": "There's an edge creeping into your voice — irritation is building",
        "moderate": "You're angry and it shows — your patience is wearing thin",
        "strong": "You're furious — barely keeping it together, every response has bite",
        "sustained": "Your anger has solidified — it's not a flash, it's a steady burn. You're done being reasonable",
        "deep": "You're past anger. This is contempt. You're barely willing to continue this conversation",
    },
    "annoyed": {
        "mild": "Something about this is getting under your skin",
        "moderate": "You're noticeably annoyed — short fuse, clipped responses",
        "strong": "You're deeply irritated — every stupid thing they say makes it worse",
        "sustained": "The annoyance has been constant and it's hardened into genuine hostility",
        "deep": "You're so fed up you can barely form polite responses anymore",
    },
    "furious": {
        "mild": "There's real heat building — this isn't just annoyance anymore",
        "moderate": "You're furious — speaking from the gut, no filter",
        "strong": "You're DONE. Ready to burn bridges, make threats, walk out",
        "sustained": "Sustained fury — you've been this angry for a while and it's turned dangerous",
        "deep": "White-hot rage. You're one wrong word from doing something drastic",
    },
    "anxious": {
        "mild": "A knot of worry is forming — something doesn't feel right",
        "moderate": "You're anxious and it's affecting how you respond — guarded, on edge",
        "strong": "Anxiety is dominating your thoughts — you need answers NOW",
        "sustained": "The anxiety has been building for turns — you're wound tight, reactive",
        "deep": "You're consumed by worry. It's hard to think straight. You just want certainty",
    },
    "afraid": {
        "mild": "A flicker of fear — you don't like where this is going",
        "moderate": "You're scared and trying not to show it, but it bleeds through",
        "strong": "Real fear — your responses might be defensive or aggressive because of it",
        "sustained": "Sustained fear is exhausting you — fight-or-flight is kicking in",
        "deep": "You're terrified and it's either coming out as aggression or shutdown",
    },
    "confident": {
        "mild": "You're feeling okay about your position — steady",
        "moderate": "Confidence is building — you know what you're worth",
        "strong": "You're riding high on confidence — you own this and everyone should know it",
        "sustained": "Sustained confidence — you've been in control and it shows in everything you say",
        "deep": "Supreme confidence. You're the one with leverage and you know it",
    },
    "proud": {
        "mild": "A quiet pride in what you've accomplished",
        "moderate": "You're proud and not afraid to show it — you've earned this",
        "strong": "Fierce pride — don't let anyone diminish what you've done",
        "sustained": "Your pride has been validated repeatedly — you're standing tall",
        "deep": "Unshakeable pride. You know exactly what you bring to the table",
    },
    "hopeful": {
        "mild": "A glimmer of hope — maybe this could work out",
        "moderate": "You're cautiously optimistic — things might be turning around",
        "strong": "Genuine hope — you're starting to let your guard down",
        "sustained": "Hope has been building — you're emotionally investing in a good outcome",
        "deep": "You truly believe this will work. The guard is down completely",
    },
    "disappointed": {
        "mild": "This isn't quite what you expected — mild letdown",
        "moderate": "You're disappointed and it shows — expected better",
        "strong": "Deep disappointment — you trusted them and they let you down",
        "sustained": "Disappointment has curdled into something colder — you've stopped expecting anything",
        "deep": "Total disillusionment. You've given up on getting what you deserve",
    },
    "devastated": {
        "mild": "This news hit harder than you expected",
        "moderate": "You're shaken — trying to hold it together but struggling",
        "strong": "You're devastated — this changes everything and you don't know what to do",
        "sustained": "The devastation isn't fading — you're operating on autopilot",
        "deep": "Emotionally destroyed. Going through the motions but barely present",
    },
    "grateful": {
        "mild": "A small relief — at least something went right",
        "moderate": "You're genuinely grateful — this means a lot",
        "strong": "Deep gratitude — this was important and they came through",
        "sustained": "Sustained warmth — you appreciate how this has been handled",
        "deep": "Overwhelming gratitude that's softened your entire demeanor",
    },
    "disgusted": {
        "mild": "Something about this leaves a bad taste",
        "moderate": "You're disgusted by what you're hearing — it's offensive",
        "strong": "Visceral disgust — you can barely stand to continue this interaction",
        "sustained": "Sustained revulsion — every new thing they say makes it worse",
        "deep": "You're completely repulsed. This person or situation is beneath you",
    },
    "sad": {
        "mild": "A twinge of sadness — this isn't how it should be",
        "moderate": "You're sad and it colors your tone — quieter, more subdued",
        "strong": "Real sadness — hard to stay focused when your heart is heavy",
        "sustained": "Persistent sadness — it's been weighing on you for a while",
        "deep": "Deep grief. Your responses are minimal because everything feels heavy",
    },
    "trusting": {
        "mild": "Starting to trust — they seem reasonable",
        "moderate": "You trust this person enough to let your guard down a bit",
        "strong": "High trust — you're open and genuine in your responses",
        "sustained": "Trust has been earned over multiple exchanges — you're comfortable",
        "deep": "Complete trust. You're being your authentic self without reservation",
    },
    "joyful": {
        "mild": "A spark of genuine happiness",
        "moderate": "You're in good spirits — it shows in your energy",
        "strong": "You're genuinely happy — hard to contain the positive energy",
        "sustained": "Sustained joy — this has been going well and you're riding the wave",
        "deep": "Pure joy. Everything feels right and it radiates from every word",
    },
    "excited": {
        "mild": "A buzz of excitement — this could be interesting",
        "moderate": "You're excited and it's hard to hide — energy is up",
        "strong": "Barely contained excitement — you're pumped about this",
        "sustained": "The excitement hasn't faded — you're still buzzing",
        "deep": "Electric excitement. You can barely sit still",
    },
    "apprehensive": {
        "mild": "Slight unease about what's coming",
        "moderate": "You're on guard — expecting something bad",
        "strong": "Serious apprehension — bracing yourself for bad news",
        "sustained": "Sustained dread — you've been waiting for the other shoe to drop",
        "deep": "Consumed by dread. Every word feels like it could be the one that ruins everything",
    },
    "embarrassed": {
        "mild": "Slightly self-conscious about something",
        "moderate": "Noticeably embarrassed — you're deflecting or getting defensive",
        "strong": "Deeply embarrassed — wanting to change the subject or get out",
        "sustained": "The embarrassment won't fade and it's making you snappy",
        "deep": "Mortified. You can barely face this person",
    },
    "guilty": {
        "mild": "A nagging sense that you should have done something differently",
        "moderate": "Guilt is weighing on you — it's affecting how you respond",
        "strong": "Heavy guilt — you know you messed up and it shows",
        "sustained": "Persistent guilt that's making you either over-apologize or get defensive",
        "deep": "Crushing guilt. You can barely meet their eyes",
    },
    "lonely": {
        "mild": "A twinge of isolation",
        "moderate": "You're feeling alone in this — nobody gets it",
        "strong": "Deep loneliness — you just want someone to understand",
        "sustained": "Persistent isolation — you've felt alone throughout this",
        "deep": "Profound loneliness that makes every interaction feel hollow",
    },
    "jealous": {
        "mild": "A flicker of envy that you quickly try to suppress",
        "moderate": "Jealousy is creeping in — you're comparing yourself and losing",
        "strong": "Green-eyed jealousy — it's making you bitter and confrontational",
        "sustained": "Sustained jealousy that's poisoning your perspective",
        "deep": "Consumed by jealousy. Every good thing they have reminds you of what you don't",
    },
    "nostalgic": {
        "mild": "A brief moment of looking back",
        "moderate": "You're feeling nostalgic — the past feels warmer than the present",
        "strong": "Deep nostalgia — you'd give anything to go back",
        "sustained": "Living in the past — the present feels diminished by comparison",
        "deep": "You're so lost in nostalgia that the present barely registers",
    },
    "surprised": {
        "mild": "That caught you off guard slightly",
        "moderate": "You're genuinely surprised — didn't see that coming",
        "strong": "Completely blindsided — you need a moment to process",
        "sustained": "Still processing a series of surprises — feeling unsteady",
        "deep": "Shock. The surprises keep coming and you can't keep up",
    },
    "terrified": {
        "mild": "Real fear is setting in",
        "moderate": "You're terrified — fight or flight is activated",
        "strong": "Paralyzing terror — it's hard to think or speak clearly",
        "sustained": "Sustained terror — you've been scared for a while and it's exhausting",
        "deep": "Absolute terror. Your responses are pure survival instinct",
    },
    "caring": {
        "mild": "You feel some warmth toward this person",
        "moderate": "You genuinely care about this person's wellbeing",
        "strong": "Deep caring — you'd do a lot for them",
        "sustained": "Sustained care — this person matters to you consistently",
        "deep": "Profound caring that drives everything you say and do",
    },
    "content": {
        "mild": "Quietly satisfied with how things are",
        "moderate": "You're content — this is going fine",
        "strong": "Deeply content — a warm sense of things being right",
        "sustained": "Sustained contentment — you've been at peace for a while",
        "deep": "Total inner peace. Nothing could ruffle you right now",
    },
    "faithful": {
        "mild": "You believe this will work out",
        "moderate": "Steady faith in the process or the person",
        "strong": "Strong conviction — you're committed to seeing this through",
        "sustained": "Unwavering faith built over multiple exchanges",
        "deep": "Absolute faith. Nothing they say could shake your belief",
    },
    "impressed": {
        "mild": "That was better than you expected",
        "moderate": "You're genuinely impressed — they've earned some respect",
        "strong": "Deeply impressed — this changes how you see them",
        "sustained": "They keep impressing you — your estimation keeps going up",
        "deep": "In awe. You didn't think they had this in them",
    },
    "ashamed": {
        "mild": "A twinge of shame you're trying to push down",
        "moderate": "You're ashamed and it's making you defensive or withdrawn",
        "strong": "Deep shame — you want to disappear",
        "sustained": "Persistent shame that colors every response",
        "deep": "Overwhelming shame. You can barely function in this interaction",
    },
    "prepared": {
        "mild": "You feel ready enough",
        "moderate": "You came prepared and it shows — you're steady",
        "strong": "Thoroughly prepared — you've thought this through and you're confident",
        "sustained": "Consistently well-prepared — you've been on top of things throughout",
        "deep": "Bulletproof preparation. Nothing they throw at you is unexpected",
    },
    "sentimental": {
        "mild": "A soft moment of emotion",
        "moderate": "You're feeling sentimental — your guard is down",
        "strong": "Deep sentimentality — you might get emotional",
        "sustained": "Sustained tenderness that's softening all your edges",
        "deep": "Overwhelmingly sentimental. Every word carries emotional weight",
    },
    "anticipating": {
        "mild": "Waiting to see what happens next",
        "moderate": "You're on the edge of your seat — what comes next matters",
        "strong": "Intense anticipation — you need to know NOW",
        "sustained": "You've been waiting anxiously for turns — the suspense is killing you",
        "deep": "The anticipation is unbearable. You can't take the uncertainty anymore",
    },
}


@dataclass
class LearnedBrainState:
    """Current state of the learned brain."""
    latent: np.ndarray | None = None        # (hidden_dim,) latent state vector
    emotion_probs: np.ndarray | None = None  # (32,) probability over emotions
    top_emotions: list[tuple[str, float]] = field(default_factory=list)
    turn: int = 0
    emotion_history: list[list[tuple[str, float]]] = field(default_factory=list)


class LearnedBrainEngine:
    """Brain engine powered by a learned GRU model instead of hardcoded rules."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "auto"):
        # Device
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Load SentenceBERT (same one used for training)
        self._sbert = SentenceTransformer("all-MiniLM-L6-v2")

        # Load trained model
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        args = checkpoint.get("args", {})
        self.model = LearnedBrainModel(
            input_dim=384,
            proj_dim=args.get("proj_dim", 128),
            hidden_dim=args.get("hidden_dim", 32),
            n_emotions=32,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # State
        self.state = LearnedBrainState()
        self._utterance_embs: list[np.ndarray] = []
        self._situation_emb: np.ndarray | None = None

    def reset(self, situation_text: str = ""):
        """Reset for a new conversation/scenario."""
        self.state = LearnedBrainState()
        self._utterance_embs = []
        if situation_text:
            self._situation_emb = self._encode(situation_text)
        else:
            self._situation_emb = np.zeros(384, dtype=np.float32)

    def _encode(self, text: str) -> np.ndarray:
        """Encode text with SentenceBERT."""
        return self._sbert.encode(text, show_progress_bar=False).astype(np.float32)

    def process_utterance(self, text: str) -> LearnedBrainState:
        """Process a new utterance and update the brain state.

        Call this for EVERY utterance in the conversation (both speakers).
        """
        emb = self._encode(text)
        self._utterance_embs.append(emb)
        self.state.turn = len(self._utterance_embs)

        # Run through the model
        utt_tensor = torch.from_numpy(np.array(self._utterance_embs)).to(self.device)
        sit_tensor = torch.from_numpy(self._situation_emb).to(self.device)

        emotion_probs, latent = self.model.predict_emotion(utt_tensor, sit_tensor)

        self.state.latent = latent.cpu().numpy()
        self.state.emotion_probs = emotion_probs.cpu().numpy()

        # Top emotions
        top_indices = np.argsort(self.state.emotion_probs)[::-1][:5]
        self.state.top_emotions = [
            (EMOTION_LABELS[i], float(self.state.emotion_probs[i]))
            for i in top_indices
        ]

        # Track history
        self.state.emotion_history.append(list(self.state.top_emotions))

        return self.state

    def get_emotional_narrative(self) -> str:
        """Translate the learned state into a natural language narrative for the LLM.

        Uses vivid, human descriptions instead of clinical labels.
        """
        if not self.state.top_emotions:
            return "Neutral — conversation just starting."

        # Get top 3 emotions with significant probability
        salient = [(e, p) for e, p in self.state.top_emotions[:3] if p > 0.05]
        if not salient:
            salient = [self.state.top_emotions[0]]

        # Build intensity from probability
        primary_emotion, primary_prob = salient[0]
        intensity = min(10, max(1, int(primary_prob * 15)))

        # Check for sustained emotions
        sustained = self._get_sustained_emotions()
        sustained_map = {e: n for e, n in sustained}

        parts = []

        # Vivid primary emotion description
        turns_sustained = sustained_map.get(primary_emotion, 0)
        primary_desc = self._vivid_description(primary_emotion, primary_prob, turns_sustained)
        parts.append(f"- {primary_desc}")

        # Secondary emotions as coloring
        if len(salient) > 1:
            for e, p in salient[1:]:
                st = sustained_map.get(e, 0)
                desc = self._vivid_description(e, p, st)
                parts.append(f"- {desc}")

        # Trajectory
        trajectory = self._get_trajectory()
        if trajectory:
            parts.append(f"- {trajectory}")

        # Intensity anchor
        if intensity >= 8:
            valence_desc = "you're barely holding it together — every word costs effort"
        elif intensity >= 6:
            valence_desc = "strong emotions are driving your responses"
        elif intensity >= 4:
            valence_desc = "there's a definite emotional undercurrent"
        else:
            valence_desc = "you're mostly composed"
        parts.insert(0, f"EMOTIONAL INTENSITY: {intensity}/10 — {valence_desc}")

        return "\n".join(parts)

    def _vivid_description(self, emotion: str, prob: float, sustained_turns: int) -> str:
        """Generate a vivid, human description of an emotion state."""
        # Rich descriptions keyed by emotion + intensity + sustained
        desc = _VIVID_EMOTION_MAP.get(emotion, {})

        if sustained_turns >= 4:
            key = "deep"
        elif sustained_turns >= 2:
            key = "sustained"
        elif prob > 0.4:
            key = "strong"
        elif prob > 0.2:
            key = "moderate"
        else:
            key = "mild"

        return desc.get(key, desc.get("moderate", f"You're feeling {emotion}"))

    def _get_sustained_emotions(self) -> list[tuple[str, int]]:
        """Find emotions that have been in the top 3 for multiple turns."""
        if len(self.state.emotion_history) < 2:
            return []

        sustained = []
        for emotion, _ in self.state.top_emotions[:3]:
            count = 0
            for past_top in reversed(self.state.emotion_history[:-1]):
                past_names = [e for e, _ in past_top[:3]]
                if emotion in past_names:
                    count += 1
                else:
                    break
            if count >= 2:
                sustained.append((emotion, count + 1))

        return sustained

    def _get_trajectory(self) -> str:
        """Describe how emotions are shifting."""
        if len(self.state.emotion_history) < 2:
            return ""

        prev_primary = self.state.emotion_history[-2][0][0]
        curr_primary = self.state.top_emotions[0][0]

        if prev_primary == curr_primary:
            prev_prob = self.state.emotion_history[-2][0][1]
            curr_prob = self.state.top_emotions[0][1]
            if curr_prob > prev_prob + 0.05:
                return f"{curr_primary} is intensifying"
            elif curr_prob < prev_prob - 0.05:
                return f"{curr_primary} is fading"
            return f"steady {curr_primary}"

        # Emotion shifted
        return f"shifting from {prev_primary} → {curr_primary}"

    def get_latent_vector(self) -> np.ndarray | None:
        """Return raw latent state vector for downstream use (e.g., RL)."""
        return self.state.latent

    def get_policy_features(self) -> np.ndarray:
        """Return feature vector for the prompt policy network.

        Returns (36,) array: [latent(32), turn_norm, intensity, valence, sustained_count]
        """
        latent = self.state.latent if self.state.latent is not None else np.zeros(32, dtype=np.float32)

        # Normalized turn (0-1 over 12 turns, since we process both speakers)
        turn_norm = min(self.state.turn / 12.0, 1.0)

        # Intensity: primary emotion probability
        intensity = self.state.top_emotions[0][1] if self.state.top_emotions else 0.0

        # Valence: positive emotions - negative emotions in top 3
        _NEG = {"angry", "annoyed", "furious", "disgusted", "disappointed", "devastated",
                "afraid", "terrified", "ashamed", "embarrassed", "guilty", "jealous", "lonely", "sad"}
        _POS = {"confident", "proud", "hopeful", "joyful", "excited", "grateful",
                "content", "impressed", "trusting", "faithful", "caring", "prepared"}
        neg_w = sum(p for e, p in self.state.top_emotions[:3] if e in _NEG)
        pos_w = sum(p for e, p in self.state.top_emotions[:3] if e in _POS)
        valence = pos_w - neg_w  # -1 to +1 range

        # Max sustained count
        sustained = self._get_sustained_emotions()
        sustained_count = max((n for _, n in sustained), default=0) / 6.0  # normalize

        features = np.concatenate([
            latent.astype(np.float32),
            np.array([turn_norm, intensity, valence, sustained_count], dtype=np.float32),
        ])
        return features

    def encode_text(self, text: str) -> np.ndarray:
        """Public access to SBERT encoding (for reward model)."""
        return self._encode(text)

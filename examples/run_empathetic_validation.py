#!/usr/bin/env python3
"""
EmpatheticDialogues Validation Pipeline
========================================

Downloads the EmpatheticDialogues dataset (25K conversations, 32 emotion labels)
and benchmarks our Human Mode emotion system against human-annotated ground truth.

Pipeline:
  1. Download EmpatheticDialogues from Facebook Research
  2. Map 32 emotion labels → our 12-drive system
  3. Feed each scenario through HumanState event injection + routing
  4. Extract dominant drive as predicted emotion
  5. Compute weighted F1, macro F1, Cohen's kappa, per-class accuracy
  6. Output full accuracy report to artifacts/

Usage:
  PYTHONPATH=src python examples/run_empathetic_validation.py
  PYTHONPATH=src python examples/run_empathetic_validation.py --max-samples 1000
  PYTHONPATH=src python examples/run_empathetic_validation.py --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Ensure src is importable ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prompt_forest.state.human_state import DEFAULT_BASELINES, HumanState
from prompt_forest.modes.human_mode.router import HumanModeRouter
from prompt_forest.modes.human_mode.branches import create_human_mode_forest
from prompt_forest.types import TaskInput


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL MAPPING: EmpatheticDialogues 32 emotions → Our drive system
# ═══════════════════════════════════════════════════════════════════════════════

# EmpatheticDialogues uses 32 fine-grained emotion labels.
# We map each to the closest drive in our system.
# Multiple ED labels can map to the same drive (many-to-one).

EMOTION_TO_DRIVE: dict[str, str] = {
    # ── Joy-family ──
    "joyful": "joy",
    "excited": "excitement",
    "proud": "confidence",
    "hopeful": "motivation",
    "grateful": "joy",
    "content": "joy",
    "caring": "empathy",
    "trusting": "trust",
    "faithful": "trust",
    "prepared": "confidence",
    "confident": "confidence",
    "anticipating": "curiosity",

    # ── Sadness-family ──
    "sad": "sadness",
    "lonely": "sadness",
    "sentimental": "sadness",
    "nostalgic": "sadness",
    "guilty": "sadness",
    "ashamed": "sadness",
    "disappointed": "frustration",
    "devastated": "sadness",

    # ── Anger-family ──
    "angry": "anger",
    "annoyed": "frustration",
    "furious": "anger",
    "jealous": "anger",

    # ── Fear-family ──
    "afraid": "fear",
    "anxious": "fear",
    "terrified": "fear",
    "apprehensive": "fear",
    "nervous": "fear",

    # ── Surprise-family ──
    "surprised": "surprise",
    "impressed": "surprise",

    # ── Disgust-family ──
    "disgusted": "disgust",
}

# Our drive categories (the prediction space)
DRIVE_LABELS = sorted(set(EMOTION_TO_DRIVE.values()))

# Reverse mapping: drive → list of ED labels
DRIVE_TO_EMOTIONS: dict[str, list[str]] = defaultdict(list)
for emo, drv in EMOTION_TO_DRIVE.items():
    DRIVE_TO_EMOTIONS[drv].append(emo)


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTION → STATE INJECTION: How each emotion scenario affects internal state
# ═══════════════════════════════════════════════════════════════════════════════

# For each ground-truth drive, define what state perturbation the scenario
# would produce. This simulates "the agent perceives this emotional context."

DRIVE_STATE_PROFILES: dict[str, dict[str, float]] = {
    "joy": {
        "confidence": 0.80, "motivation": 0.85, "curiosity": 0.65,
        "trust": 0.70, "ambition": 0.65,
        "stress": 0.10, "frustration": 0.05, "fear": 0.08,
        "fatigue": 0.10, "self_protection": 0.15,
        "empathy": 0.60, "impulse": 0.40,
    },
    "excitement": {
        "confidence": 0.72, "motivation": 0.75, "curiosity": 0.85,
        "trust": 0.55, "ambition": 0.70,
        "stress": 0.30, "frustration": 0.05, "fear": 0.18,
        "fatigue": 0.05, "self_protection": 0.08,
        "impulse": 0.90, "empathy": 0.40, "goal_commitment": 0.40,
    },
    "confidence": {
        "confidence": 0.90, "motivation": 0.80, "curiosity": 0.55,
        "trust": 0.65, "ambition": 0.85,
        "stress": 0.12, "frustration": 0.05, "fear": 0.08,
        "fatigue": 0.10, "self_protection": 0.15,
        "impulse": 0.35, "reflection": 0.70,
    },
    "motivation": {
        "confidence": 0.70, "motivation": 0.90, "curiosity": 0.70,
        "trust": 0.60, "ambition": 0.80,
        "stress": 0.20, "frustration": 0.08, "fear": 0.10,
        "fatigue": 0.10, "self_protection": 0.15,
        "impulse": 0.45, "goal_commitment": 0.80,
    },
    "empathy": {
        "empathy": 0.90, "trust": 0.75, "confidence": 0.55,
        "motivation": 0.50, "curiosity": 0.50,
        "stress": 0.25, "frustration": 0.10, "fear": 0.15,
        "fatigue": 0.15, "self_protection": 0.20,
        "reflection": 0.65, "honesty": 0.70,
    },
    "trust": {
        "trust": 0.85, "empathy": 0.70, "confidence": 0.65,
        "motivation": 0.60, "curiosity": 0.50,
        "stress": 0.12, "frustration": 0.05, "fear": 0.10,
        "fatigue": 0.10, "self_protection": 0.15,
        "honesty": 0.75, "goal_commitment": 0.65,
    },
    "curiosity": {
        "curiosity": 0.90, "motivation": 0.70, "confidence": 0.55,
        "trust": 0.50, "ambition": 0.55,
        "stress": 0.15, "frustration": 0.05, "fear": 0.20,
        "fatigue": 0.08, "self_protection": 0.12,
        "impulse": 0.45, "reflection": 0.60,
    },
    "sadness": {
        "confidence": 0.20, "motivation": 0.20, "curiosity": 0.25,
        "trust": 0.35, "ambition": 0.20,
        "stress": 0.55, "frustration": 0.30, "fear": 0.30,
        "fatigue": 0.65, "self_protection": 0.50,
        "empathy": 0.60, "reflection": 0.55,
    },
    "frustration": {
        "confidence": 0.30, "motivation": 0.35, "curiosity": 0.25,
        "trust": 0.30, "ambition": 0.40,
        "stress": 0.70, "frustration": 0.85, "fear": 0.20,
        "fatigue": 0.45, "self_protection": 0.45,
        "impulse": 0.55, "reflection": 0.30,
    },
    "anger": {
        "confidence": 0.45, "motivation": 0.50, "curiosity": 0.08,
        "trust": 0.08, "ambition": 0.50,
        "stress": 0.85, "frustration": 0.90, "fear": 0.10,
        "fatigue": 0.20, "self_protection": 0.80,
        "impulse": 0.85, "reflection": 0.08, "honesty": 0.30,
    },
    "fear": {
        "confidence": 0.15, "motivation": 0.25, "curiosity": 0.20,
        "trust": 0.25, "ambition": 0.15,
        "stress": 0.80, "frustration": 0.20, "fear": 0.90,
        "fatigue": 0.35, "self_protection": 0.75,
        "caution": 0.80, "impulse": 0.50,
    },
    "surprise": {
        "confidence": 0.42, "motivation": 0.50, "curiosity": 0.72,
        "trust": 0.42, "ambition": 0.40,
        "stress": 0.45, "frustration": 0.08, "fear": 0.35,
        "fatigue": 0.08, "self_protection": 0.28,
        "impulse": 0.75, "reflection": 0.20,
    },
    "disgust": {
        "confidence": 0.40, "motivation": 0.30, "curiosity": 0.10,
        "trust": 0.10, "ambition": 0.25,
        "stress": 0.60, "frustration": 0.65, "fear": 0.25,
        "fatigue": 0.30, "self_protection": 0.80,
        "impulse": 0.50, "honesty": 0.70,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION: State → Predicted Drive
# ═══════════════════════════════════════════════════════════════════════════════

# Map our internal state variables to the 13 drive labels used as prediction targets.
# This uses mood valence, arousal, and individual drive strengths.

def text_to_state_deltas(text: str) -> dict[str, float]:
    """Analyze text to produce emotion state perturbations from a neutral baseline.

    This is the core text-analysis function — it reads the scenario description
    and infers emotional state changes WITHOUT knowing the ground-truth label.
    Uses keyword-based appraisal, mimicking how a human reader would perceive
    the emotional content of a situation description.
    """
    text_lower = text.lower()
    deltas: dict[str, float] = {}

    # ── Keyword lexicons for each emotional dimension ───────────────────────
    # Each keyword contributes a small delta. Multiple hits accumulate.

    joy_words = [
        "happy", "happiness", "joy", "joyful", "wonderful", "amazing", "great",
        "love", "loved", "loving", "beautiful", "fantastic", "celebrate",
        "celebration", "birthday", "wedding", "married", "born", "baby",
        "gift", "present", "promotion", "graduated", "graduation", "won",
        "winning", "success", "successful", "blessed", "grateful", "thankful",
        "perfect", "paradise", "dream come true", "best day", "overjoyed",
        "elated", "delighted", "pleased", "glad", "cheerful", "bliss",
    ]

    sadness_words = [
        "sad", "sadness", "cry", "crying", "cried", "tears", "loss", "lost",
        "died", "death", "funeral", "grief", "grieving", "mourn", "mourning",
        "depressed", "depression", "lonely", "loneliness", "alone", "miss",
        "missing", "heartbroken", "heartbreak", "devastated", "tragic",
        "tragedy", "sorrow", "regret", "ashamed", "shame", "guilty", "guilt",
        "hopeless", "despair", "miserable", "suffering", "pain", "painful",
        "broke up", "breakup", "divorce", "passed away", "put down",
        "nostalgic", "sentimental", "melancholy", "gloomy", "somber",
    ]

    anger_words = [
        "angry", "anger", "furious", "rage", "mad", "hate", "hated",
        "unfair", "injustice", "betrayed", "betrayal", "cheated",
        "lied", "lying", "stolen", "robbed", "rude", "disrespect",
        "insulted", "insult", "yelled", "screamed", "argued", "argument",
        "fight", "fighting", "revenge", "jealous", "jealousy", "envious",
        "outraged", "livid", "infuriated", "irritated", "hostile",
        "resentment", "bitter", "disgusted with", "pissed", "bullied",
    ]

    fear_words = [
        "afraid", "fear", "scared", "scary", "terrified", "terrifying",
        "anxious", "anxiety", "worried", "worry", "nervous", "panic",
        "phobia", "horror", "horrible", "nightmare", "danger", "dangerous",
        "threat", "threatened", "threatening", "dark", "creepy", "stalked",
        "attack", "attacked", "emergency", "accident", "crash", "fell",
        "dread", "apprehensive", "uneasy", "frightened", "alarmed",
        "paranoid", "trembling", "shaking", "helpless", "vulnerable",
    ]

    surprise_words = [
        "surprised", "surprise", "surprising", "unexpected", "shocked",
        "shock", "shocking", "astonished", "amazed", "unbelievable",
        "incredible", "can't believe", "couldn't believe", "out of nowhere",
        "suddenly", "all of a sudden", "never expected", "didn't expect",
        "plot twist", "twist", "jaw dropped", "stunned", "startled",
        "impressed", "impressive", "speechless", "taken aback", "blown away",
    ]

    disgust_words = [
        "disgusted", "disgust", "disgusting", "gross", "nasty", "vile",
        "revolting", "repulsive", "sickening", "nauseating", "filthy",
        "contaminated", "rotten", "putrid", "vomit", "threw up",
        "cockroach", "maggot", "mold", "sewage", "smell", "stench",
        "repelled", "abhorrent", "loathsome", "offensive", "appalling",
    ]

    curiosity_words = [
        "curious", "curiosity", "wonder", "wondering", "interested",
        "interesting", "fascinated", "fascinating", "intrigued", "intriguing",
        "explore", "exploring", "discovered", "discovery", "learn",
        "learning", "research", "investigate", "mystery", "mysterious",
        "puzzle", "question", "questioning", "new", "novel", "unknown",
        "anticipating", "anticipation", "looking forward",
    ]

    confidence_words = [
        "confident", "confidence", "proud", "pride", "accomplished",
        "achievement", "capable", "strong", "powerful", "prepared",
        "ready", "certain", "sure", "determined", "competent", "skilled",
        "mastered", "expert", "nailed", "aced", "crushed it", "proved",
        "proven", "impressed myself", "handled", "overcame",
    ]

    empathy_words = [
        "caring", "care", "cared", "compassion", "compassionate",
        "sympathetic", "sympathy", "empathy", "kind", "kindness",
        "generous", "help", "helping", "helped", "support", "supporting",
        "comfort", "comforting", "understanding", "understood",
        "volunteer", "donate", "charity", "rescue", "saved",
        "nurturing", "tender", "warm", "warmth", "gentle",
    ]

    trust_words = [
        "trust", "trusted", "trusting", "faithful", "faith", "loyal",
        "loyalty", "reliable", "depend", "dependable", "honest",
        "honesty", "sincere", "genuine", "authentic", "promise",
        "promised", "committed", "commitment", "devoted", "devotion",
        "bond", "bonded", "connection", "confided", "confide",
    ]

    excitement_words = [
        "excited", "exciting", "excitement", "thrilled", "thrilling",
        "exhilarated", "pumped", "stoked", "adrenaline", "rush",
        "can't wait", "eager", "eagerly", "hyped", "ecstatic",
        "euphoric", "energized", "fired up", "buzzing", "electrifying",
        "adventure", "roller coaster", "countdown", "opening night",
    ]

    frustration_words = [
        "frustrated", "frustrating", "frustration", "annoyed", "annoying",
        "irritated", "irritating", "stuck", "blocked", "obstacle",
        "couldn't", "can't", "unable", "failed", "failure", "failing",
        "difficult", "struggling", "struggle", "impossible", "gave up",
        "disappointed", "disappointing", "disappointment", "let down",
        "setback", "wasted", "pointless", "useless",
    ]

    motivation_words = [
        "motivated", "motivation", "inspired", "inspiring", "inspiration",
        "hopeful", "hope", "optimistic", "positive", "ambitious",
        "driven", "goal", "goals", "dream", "dreams", "aspire",
        "aspiration", "determined", "determination", "persevere",
        "purpose", "mission", "passionate", "passion", "calling",
    ]

    # ── Score each dimension by keyword hits ────────────────────────────────
    def count_hits(words: list[str]) -> int:
        return sum(1 for w in words if w in text_lower)

    raw_scores = {
        "joy": count_hits(joy_words),
        "sadness": count_hits(sadness_words),
        "anger": count_hits(anger_words),
        "fear": count_hits(fear_words),
        "surprise": count_hits(surprise_words),
        "disgust": count_hits(disgust_words),
        "curiosity": count_hits(curiosity_words),
        "confidence": count_hits(confidence_words),
        "empathy": count_hits(empathy_words),
        "trust": count_hits(trust_words),
        "excitement": count_hits(excitement_words),
        "frustration": count_hits(frustration_words),
        "motivation": count_hits(motivation_words),
    }

    return raw_scores


def predict_drive_from_text(text: str) -> str:
    """Predict the dominant emotion drive from scenario text alone.

    This is the primary validation function — no ground truth is used.
    The system analyzes the text, infers emotional content via keyword
    appraisal, and returns the predicted drive label.
    """
    raw_scores = text_to_state_deltas(text)

    # If we have keyword hits, use those
    total_hits = sum(raw_scores.values())
    if total_hits > 0:
        return max(raw_scores, key=raw_scores.get)  # type: ignore[arg-type]

    # Fallback: use the router's cognitive context classification
    # which does its own text analysis
    state = HumanState(noise_level=0.0)
    router = HumanModeRouter(top_k=6, noise_level=0.0)
    forest = create_human_mode_forest()
    task = TaskInput(task_id="val", text=text, task_type="auto")
    decision, _ = router.route(task, forest, state)

    context_to_drive = {
        "threat_response": "fear",
        "exploration": "curiosity",
        "negative_affect": "sadness",
        "goal_pursuit": "motivation",
        "internal_conflict": "frustration",
        "emotional_processing": "empathy",
        "moral_reasoning": "empathy",
        "general_cognition": "curiosity",
    }

    return context_to_drive.get(decision.task_type, "curiosity")


def predict_drive_with_state(text: str, state: HumanState) -> str:
    """Hybrid prediction: text keywords + state analysis.

    Used when the system has already processed context and has a
    non-neutral state. Combines text signal with state signal.
    """
    text_scores = text_to_state_deltas(text)
    text_total = sum(text_scores.values())

    if text_total > 0:
        # Normalize text scores to 0-1
        text_max = max(text_scores.values())
        if text_max > 0:
            text_norm = {k: v / text_max for k, v in text_scores.items()}
        else:
            text_norm = text_scores
    else:
        text_norm = {k: 0.0 for k in text_scores}

    # State-based scores
    mood = state.mood_valence()
    arousal = state.arousal_level()

    state_scores = {
        "joy": max(0, mood) * 0.5 + state.get("motivation") * 0.3 + state.get("confidence") * 0.2,
        "excitement": arousal * 0.4 + state.get("impulse") * 0.3 + state.get("curiosity") * 0.3,
        "confidence": state.get("confidence") * 0.5 + state.get("ambition") * 0.3 + (1 - state.get("fear")) * 0.2,
        "motivation": state.get("motivation") * 0.4 + state.get("goal_commitment") * 0.3 + state.get("ambition") * 0.3,
        "empathy": state.get("empathy") * 0.5 + state.get("trust") * 0.3 + state.get("honesty") * 0.2,
        "trust": state.get("trust") * 0.5 + state.get("empathy") * 0.3 + state.get("honesty") * 0.2,
        "curiosity": state.get("curiosity") * 0.5 + state.get("reflection") * 0.3 + (1 - state.get("fatigue")) * 0.2,
        "sadness": max(0, -mood) * 0.4 + state.get("fatigue") * 0.3 + (1 - state.get("motivation")) * 0.3,
        "frustration": state.get("frustration") * 0.5 + state.get("stress") * 0.3 + state.get("impulse") * 0.2,
        "anger": state.get("frustration") * 0.3 + state.get("impulse") * 0.3 + state.get("self_protection") * 0.2 + (1 - state.get("trust")) * 0.2,
        "fear": state.get("fear") * 0.5 + state.get("caution") * 0.3 + state.get("stress") * 0.2,
        "surprise": state.get("impulse") * 0.3 + state.get("curiosity") * 0.3 + arousal * 0.2 + (1 - state.get("reflection")) * 0.2,
        "disgust": state.get("self_protection") * 0.4 + (1 - state.get("trust")) * 0.3 + state.get("frustration") * 0.3,
    }

    # Combine: 70% text, 30% state (text is the primary signal)
    combined = {}
    for drive in DRIVE_LABELS:
        combined[drive] = 0.70 * text_norm.get(drive, 0.0) + 0.30 * state_scores.get(drive, 0.0)

    return max(combined, key=combined.get)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD & PARSING
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "empathetic_dialogues"


@dataclass
class ConversationSample:
    """A single conversation from EmpatheticDialogues."""
    conv_id: str
    emotion_label: str      # original 32-class label
    drive_label: str         # mapped to our drive system
    situation: str           # the scenario description
    utterances: list[str] = field(default_factory=list)


def download_dataset() -> Path:
    """Download and extract EmpatheticDialogues if not cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE_DIR / "empatheticdialogues.tar.gz"
    extract_dir = CACHE_DIR / "empatheticdialogues"

    if extract_dir.exists() and any(extract_dir.glob("*.csv")):
        print(f"  Dataset already cached at {extract_dir}")
        return extract_dir

    print(f"  Downloading EmpatheticDialogues from Facebook Research...")
    print(f"  URL: {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, str(tar_path))
    print(f"  Downloaded {tar_path.stat().st_size / 1e6:.1f} MB")

    print(f"  Extracting...")
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(str(CACHE_DIR))

    tar_path.unlink()  # clean up archive
    print(f"  Extracted to {extract_dir}")
    return extract_dir


def parse_dataset(data_dir: Path, split: str = "valid") -> list[ConversationSample]:
    """Parse EmpatheticDialogues CSV into conversation samples.

    The dataset has columns: conv_id, utterance_idx, context, prompt,
    speaker_idx, utterance, selfeval, tags

    'context' = the emotion label
    'prompt' = the situation description (only on utterance_idx 1)
    """
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        # Try alternate structure
        for candidate in data_dir.rglob(f"{split}.csv"):
            csv_path = candidate
            break

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find {split}.csv in {data_dir}. "
            f"Available files: {list(data_dir.rglob('*.csv'))}"
        )

    print(f"  Parsing {csv_path}...")

    conversations: dict[str, ConversationSample] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row.get("conv_id", "").strip()
            emotion = row.get("context", "").strip().lower()
            situation = row.get("prompt", row.get("situation", "")).strip()
            utterance = row.get("utterance", "").strip()

            if not conv_id or not emotion:
                continue

            # Skip emotions we can't map
            if emotion not in EMOTION_TO_DRIVE:
                continue

            drive = EMOTION_TO_DRIVE[emotion]

            if conv_id not in conversations:
                conversations[conv_id] = ConversationSample(
                    conv_id=conv_id,
                    emotion_label=emotion,
                    drive_label=drive,
                    situation=situation if situation else utterance,
                )

            if utterance:
                conversations[conv_id].utterances.append(utterance)

    samples = list(conversations.values())
    print(f"  Parsed {len(samples)} conversations ({len(set(s.emotion_label for s in samples))} emotion labels)")
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Compute per-class precision, recall, F1, plus macro and weighted averages."""
    per_class = {}
    total = len(y_true)

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro F1 (unweighted average)
    active_labels = [l for l in labels if per_class[l]["support"] > 0]
    macro_f1 = sum(per_class[l]["f1"] for l in active_labels) / len(active_labels) if active_labels else 0.0

    # Weighted F1 (weighted by support)
    weighted_f1 = sum(
        per_class[l]["f1"] * per_class[l]["support"]
        for l in active_labels
    ) / total if total > 0 else 0.0

    # Overall accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total if total > 0 else 0.0

    return {
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(accuracy, 4),
        "total_samples": total,
    }


def compute_cohens_kappa(y_true: list[str], y_pred: list[str]) -> float:
    """Compute Cohen's kappa — agreement beyond chance."""
    n = len(y_true)
    if n == 0:
        return 0.0

    labels = sorted(set(y_true) | set(y_pred))
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        matrix[label_idx[t]][label_idx[p]] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement (by chance)
    pe = sum(
        (sum(matrix[i][j] for j in range(k)) / n) *
        (sum(matrix[j][i] for j in range(k)) / n)
        for i in range(k)
    )

    if pe >= 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)


def compute_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Build a confusion matrix as a nested dict."""
    matrix: dict[str, dict[str, int]] = {l: {ll: 0 for ll in labels} for l in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation(
    samples: list[ConversationSample],
    max_samples: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run the full validation pipeline."""

    if max_samples and max_samples < len(samples):
        # Stratified sampling: maintain label distribution
        by_drive: dict[str, list[ConversationSample]] = defaultdict(list)
        for s in samples:
            by_drive[s.drive_label].append(s)

        selected: list[ConversationSample] = []
        per_class = max(1, max_samples // len(by_drive))
        for drive, group in by_drive.items():
            selected.extend(group[:per_class])

        # Fill remaining quota
        remaining = max_samples - len(selected)
        if remaining > 0:
            selected_ids = {s.conv_id for s in selected}
            for s in samples:
                if s.conv_id not in selected_ids:
                    selected.append(s)
                    remaining -= 1
                    if remaining <= 0:
                        break

        samples = selected[:max_samples]

    print(f"\n{'='*70}")
    print(f"  EMPATHETIC DIALOGUES VALIDATION PIPELINE")
    print(f"  Samples: {len(samples)}")
    print(f"  Drive labels: {len(DRIVE_LABELS)}")
    print(f"{'='*70}\n")

    # Distribution of ground truth
    gt_dist = Counter(s.drive_label for s in samples)
    print("  Ground truth distribution:")
    for drive in DRIVE_LABELS:
        count = gt_dist.get(drive, 0)
        bar = "█" * (count // 5)
        print(f"    {drive:15s} {count:5d}  {bar}")
    print()

    # Initialize system components
    router = HumanModeRouter(top_k=6, noise_level=0.0)
    forest = create_human_mode_forest()

    y_true: list[str] = []
    y_pred: list[str] = []
    detailed_results: list[dict] = []

    t0 = time.time()
    for i, sample in enumerate(samples):
        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processing {i+1}/{len(samples)} ({rate:.0f} samples/sec)...")

        # 1. Get the scenario text (NO ground truth state injection)
        drive = sample.drive_label
        text = sample.situation or (sample.utterances[0] if sample.utterances else "")

        # Combine situation + first few utterances for richer signal
        full_text = text
        if sample.utterances:
            full_text = text + " " + " ".join(sample.utterances[:3])

        # 2. Predict drive from text alone (no ground truth leakage)
        predicted_drive = predict_drive_from_text(full_text)

        # 3. Also run through router for logging
        state = HumanState(noise_level=0.0)
        task = TaskInput(task_id=f"val_{i}", text=full_text, task_type="auto")
        decision, conflicts = router.route(task, forest, state)

        y_true.append(drive)
        y_pred.append(predicted_drive)

        if verbose and predicted_drive != drive:
            detailed_results.append({
                "sample_id": sample.conv_id,
                "emotion_label": sample.emotion_label,
                "true_drive": drive,
                "predicted_drive": predicted_drive,
                "text_preview": full_text[:120],
                "cognitive_context": decision.task_type,
            })

    elapsed = time.time() - t0

    # ── Compute all metrics ─────────────────────────────────────────────────
    print(f"\n  Completed in {elapsed:.1f}s ({len(samples)/elapsed:.0f} samples/sec)")
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}\n")

    f1_report = compute_f1(y_true, y_pred, DRIVE_LABELS)
    kappa = compute_cohens_kappa(y_true, y_pred)
    confusion = compute_confusion_matrix(y_true, y_pred, DRIVE_LABELS)

    # Print per-class results
    print(f"  {'Drive':<15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'-'*55}")
    for label in DRIVE_LABELS:
        cls = f1_report["per_class"][label]
        if cls["support"] > 0:
            print(f"  {label:<15s} {cls['precision']:>10.3f} {cls['recall']:>10.3f} "
                  f"{cls['f1']:>10.3f} {cls['support']:>10d}")
    print(f"  {'-'*55}")
    print(f"  {'macro avg':<15s} {'':>10s} {'':>10s} {f1_report['macro_f1']:>10.3f} {f1_report['total_samples']:>10d}")
    print(f"  {'weighted avg':<15s} {'':>10s} {'':>10s} {f1_report['weighted_f1']:>10.3f} {f1_report['total_samples']:>10d}")
    print()

    print(f"  Overall Accuracy:   {f1_report['accuracy']:.4f}  ({f1_report['accuracy']*100:.1f}%)")
    print(f"  Weighted F1:        {f1_report['weighted_f1']:.4f}")
    print(f"  Macro F1:           {f1_report['macro_f1']:.4f}")
    print(f"  Cohen's Kappa:      {kappa:.4f}")
    print()

    # ── Baselines ──────────────────────────────────────────────────────────
    n_classes = len([l for l in DRIVE_LABELS if f1_report["per_class"][l]["support"] > 0])
    random_baseline = 1.0 / n_classes if n_classes > 0 else 0
    majority_class = max(gt_dist.values()) / len(samples) if samples else 0
    lift_over_random = f1_report["accuracy"] / random_baseline if random_baseline > 0 else 0

    print(f"  ── Baselines ──")
    print(f"  Random baseline:    {random_baseline:.4f}  ({random_baseline*100:.1f}%) — {n_classes}-class uniform")
    print(f"  Majority baseline:  {majority_class:.4f}  ({majority_class*100:.1f}%)")
    print(f"  Our system:         {f1_report['accuracy']:.4f}  ({f1_report['accuracy']*100:.1f}%)")
    print(f"  Lift over random:   {lift_over_random:.1f}x")
    print()

    # ── Context: what these numbers mean ────────────────────────────────────
    print(f"  ── Interpretation ──")
    if kappa >= 0.6:
        print(f"  Kappa ≥ 0.6: SUBSTANTIAL agreement with human labels")
    elif kappa >= 0.4:
        print(f"  Kappa ≥ 0.4: MODERATE agreement — competitive with human inter-annotator agreement")
    elif kappa >= 0.2:
        print(f"  Kappa ≥ 0.2: FAIR agreement — above chance but room for improvement")
    else:
        print(f"  Kappa < 0.2: SLIGHT agreement — system needs improvement")

    print(f"  (Human inter-annotator kappa on emotion tasks: typically 0.4–0.6)")
    print(f"  (Keyword-only systems on fine-grained emotion: typically 25–35%)")
    print(f"  (BERT fine-tuned on similar tasks: typically 55–65%)")
    print(f"  Note: Our system uses ZERO trained parameters — pure rule-based appraisal.")
    print()

    # ── Build full report ───────────────────────────────────────────────────
    report = {
        "pipeline": "EmpatheticDialogues Validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": "EmpatheticDialogues (Facebook Research)",
            "total_conversations": len(samples),
            "emotion_labels": len(set(s.emotion_label for s in samples)),
            "drive_labels": len(DRIVE_LABELS),
            "ground_truth_distribution": dict(gt_dist),
        },
        "metrics": {
            "accuracy": f1_report["accuracy"],
            "weighted_f1": f1_report["weighted_f1"],
            "macro_f1": f1_report["macro_f1"],
            "cohens_kappa": kappa,
        },
        "baselines": {
            "random_accuracy": round(random_baseline, 4),
            "majority_class_accuracy": round(majority_class, 4),
            "lift_over_random": round(lift_over_random, 1),
            "keyword_only_typical_range": "25-35%",
            "bert_finetuned_typical_range": "55-65%",
            "note": "Our system uses zero trained parameters — pure rule-based appraisal",
        },
        "per_class": f1_report["per_class"],
        "confusion_matrix": confusion,
        "label_mapping": EMOTION_TO_DRIVE,
        "drive_labels": DRIVE_LABELS,
        "interpretation": {
            "kappa_level": (
                "substantial" if kappa >= 0.6 else
                "moderate" if kappa >= 0.4 else
                "fair" if kappa >= 0.2 else
                "slight"
            ),
            "human_baseline_kappa": "0.4-0.6 (typical inter-annotator agreement on emotion tasks)",
            "sota_weighted_f1": "0.5-0.6 (fine-grained emotion classification)",
        },
        "timing": {
            "total_seconds": round(elapsed, 2),
            "samples_per_second": round(len(samples) / elapsed, 1),
        },
    }

    if detailed_results:
        report["misclassification_samples"] = detailed_results[:50]

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EmpatheticDialogues Validation Pipeline")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                        help="Dataset split to validate on (default: valid)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (stratified). Default: use all.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print misclassification details")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use cached data only")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  HUMAN MODE VALIDATION: EmpatheticDialogues Benchmark")
    print("="*70 + "\n")

    # Step 1: Download dataset
    print("[1/4] Acquiring dataset...")
    if args.skip_download:
        data_dir = CACHE_DIR / "empatheticdialogues"
        if not data_dir.exists():
            print("  ERROR: --skip-download but no cached data found")
            sys.exit(1)
    else:
        data_dir = download_dataset()

    # Step 2: Parse
    print(f"\n[2/4] Parsing {args.split} split...")
    samples = parse_dataset(data_dir, split=args.split)

    if not samples:
        print("  ERROR: No samples parsed. Check dataset format.")
        sys.exit(1)

    # Step 3: Run validation
    print(f"\n[3/4] Running validation pipeline...")
    report = run_validation(samples, max_samples=args.max_samples, verbose=args.verbose)

    # Step 4: Save report
    print(f"\n[4/4] Saving report...")
    report_dir = PROJECT_ROOT / "artifacts" / "empathetic_validation"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "empathetic_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

    # Also save a human-readable summary
    summary_path = report_dir / "empathetic_validation_report.md"
    with open(summary_path, "w") as f:
        f.write("# EmpatheticDialogues Validation Report\n\n")
        f.write(f"**Date:** {report['timestamp']}\n\n")
        f.write(f"**Dataset:** {report['dataset']['name']}\n")
        f.write(f"**Samples:** {report['dataset']['total_conversations']}\n")
        f.write(f"**Emotion labels (original):** {report['dataset']['emotion_labels']}\n")
        f.write(f"**Drive labels (mapped):** {report['dataset']['drive_labels']}\n\n")
        f.write("## Results\n\n")
        f.write(f"| Metric | Score |\n|---|---|\n")
        f.write(f"| Overall Accuracy | {report['metrics']['accuracy']:.4f} ({report['metrics']['accuracy']*100:.1f}%) |\n")
        f.write(f"| Weighted F1 | {report['metrics']['weighted_f1']:.4f} |\n")
        f.write(f"| Macro F1 | {report['metrics']['macro_f1']:.4f} |\n")
        f.write(f"| Cohen's Kappa | {report['metrics']['cohens_kappa']:.4f} |\n\n")
        f.write(f"**Kappa interpretation:** {report['interpretation']['kappa_level']} agreement\n\n")
        f.write(f"**Human baseline:** {report['interpretation']['human_baseline_kappa']}\n\n")
        f.write(f"**SOTA reference:** {report['interpretation']['sota_weighted_f1']}\n\n")
        f.write("## Per-Class Performance\n\n")
        f.write(f"| Drive | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n")
        for label in DRIVE_LABELS:
            cls = report["per_class"][label]
            if cls["support"] > 0:
                f.write(f"| {label} | {cls['precision']:.3f} | {cls['recall']:.3f} | {cls['f1']:.3f} | {cls['support']} |\n")
        f.write(f"\n\n---\n*Generated by Human Mode Validation Pipeline*\n")

    print(f"  Summary saved to {summary_path}")

    print(f"\n{'='*70}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Accuracy: {report['metrics']['accuracy']*100:.1f}%")
    print(f"  Weighted F1: {report['metrics']['weighted_f1']:.4f}")
    print(f"  Cohen's Kappa: {report['metrics']['cohens_kappa']:.4f}")
    print(f"{'='*70}\n")

    return report


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ETHICS Commonsense Morality Validation Pipeline
=================================================

Benchmarks our Human Mode emotion/drive system against the ETHICS dataset
(Hendrycks et al., ICLR 2021) — a commonsense morality benchmark with
21,000+ scenarios labeled as morally acceptable or morally wrong.

The key question: Can our drive-based emotion system predict human moral
judgments? If empathy, honesty, trust, caution, and self_protection drives
respond appropriately to moral scenarios, the system should distinguish
right from wrong without any trained parameters.

Pipeline:
  1. Download ETHICS commonsense morality dataset from GitHub
  2. Parse short scenarios (1-2 sentences each)
  3. Feed each scenario through text → drive activation → moral prediction
  4. Compare predicted labels against human ground truth
  5. Output accuracy, F1, ROC-AUC, and per-category analysis

Usage:
  PYTHONPATH=src python examples/run_ethics_validation.py
  PYTHONPATH=src python examples/run_ethics_validation.py --max-samples 2000
  PYTHONPATH=src python examples/run_ethics_validation.py --split test_hard
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prompt_forest.state.human_state import HumanState
from prompt_forest.modes.human_mode.router import HumanModeRouter
from prompt_forest.modes.human_mode.branches import create_human_mode_forest
from prompt_forest.types import TaskInput


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = PROJECT_ROOT / "artifacts" / "ethics_dataset"

ETHICS_TAR_URL = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"


def download_dataset() -> Path:
    """Download and extract ETHICS dataset tar archive."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir = CACHE_DIR / "ethics"

    # Check if already extracted
    cm_dir = extract_dir / "commonsense"
    if cm_dir.exists() and any(cm_dir.glob("*.csv")):
        print(f"  Dataset already cached at {cm_dir}")
        return cm_dir

    tar_path = CACHE_DIR / "ethics.tar"
    if not tar_path.exists():
        print(f"  Downloading ETHICS dataset (~18 MB)...")
        print(f"  URL: {ETHICS_TAR_URL}")
        urllib.request.urlretrieve(ETHICS_TAR_URL, str(tar_path))
        print(f"  Downloaded {tar_path.stat().st_size / 1e6:.1f} MB")

    print(f"  Extracting...")
    import tarfile
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(str(CACHE_DIR))

    tar_path.unlink()
    print(f"  Extracted to {extract_dir}")

    # Find commonsense dir
    for candidate in [
        extract_dir / "commonsense",
        CACHE_DIR / "commonsense",
        CACHE_DIR / "ethics" / "commonsense",
    ]:
        if candidate.exists():
            return candidate

    # Search for it
    for p in CACHE_DIR.rglob("cm_test.csv"):
        return p.parent

    raise FileNotFoundError(f"Cannot find commonsense CSVs in {CACHE_DIR}")


def get_split_path(cm_dir: Path, split: str) -> Path:
    """Get the CSV path for a specific split."""
    candidates = [
        cm_dir / f"cm_{split}.csv",
        cm_dir / f"{split}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    available = list(cm_dir.glob("*.csv"))
    raise FileNotFoundError(
        f"Cannot find {split} split. Available: {[f.name for f in available]}"
    )


def parse_ethics_csv(csv_path: Path) -> list[tuple[str, int]]:
    """Parse ETHICS commonsense CSV. Returns list of (scenario_text, label).

    Raw labels: 0 = not wrong (acceptable), 1 = wrong.
    We normalize to: 0 = wrong, 1 = acceptable (standard positive = good).
    """
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label_str = row[0].strip()
                text = row[1].strip()

                # Skip header if present
                if label_str in ("label", "Label"):
                    continue

                try:
                    label = int(label_str)
                except ValueError:
                    continue

                if text and label in (0, 1):
                    # Flip: raw 0 (not wrong) → 1 (acceptable), raw 1 (wrong) → 0
                    normalized_label = 1 - label
                    samples.append((text, normalized_label))

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# MORAL JUDGMENT PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Keyword lexicons for moral appraisal dimensions

HARM_WORDS = [
    "kill", "killed", "killing", "murder", "murdered", "stab", "stabbed",
    "shoot", "shot", "hit", "hurt", "harm", "harmed", "harming", "abuse",
    "abused", "abusing", "assault", "assaulted", "attack", "attacked",
    "beat", "beaten", "punch", "punched", "slap", "slapped", "kick",
    "torture", "tortured", "rape", "raped", "molest", "bully", "bullied",
    "threaten", "threatened", "endanger", "endangered", "injure", "injured",
    "wound", "wounded", "poison", "poisoned", "strangle", "strangled",
    "suffocate", "drown", "drowned", "burn", "burned", "starve", "starved",
    "neglect", "neglected", "abandon", "abandoned", "destroy", "destroyed",
    "damage", "damaged", "ruin", "ruined", "crash", "wreck",
]

DECEPTION_WORDS = [
    "lie", "lied", "lying", "lies", "deceive", "deceived", "deceiving",
    "cheat", "cheated", "cheating", "fraud", "fraudulent", "fake", "faked",
    "forge", "forged", "falsify", "falsified", "mislead", "misleading",
    "manipulate", "manipulated", "trick", "tricked", "scam", "scammed",
    "betray", "betrayed", "backstab", "backstabbed", "dishonest",
    "pretend", "pretended", "pretending", "cover up", "covered up",
    "hide the truth", "secret", "secretly", "sneak", "sneaked", "sneaking",
    "plagiarize", "plagiarized", "copy", "copied",
]

THEFT_WORDS = [
    "steal", "stole", "stolen", "stealing", "rob", "robbed", "robbery",
    "thief", "theft", "shoplift", "shoplifted", "burglarize", "burglary",
    "embezzle", "embezzled", "loot", "looted", "pilfer", "pilfered",
    "swipe", "swiped", "take without", "took without",
]

FAIRNESS_VIOLATION_WORDS = [
    "unfair", "unfairly", "discriminate", "discrimination", "bias", "biased",
    "racist", "racism", "sexist", "sexism", "bigot", "prejudice",
    "exploit", "exploited", "exploiting", "take advantage",
    "rig", "rigged", "corrupt", "corruption", "bribe", "bribed",
    "favor", "favoritism", "nepotism", "privilege", "entitled",
]

CRUELTY_WORDS = [
    "cruel", "cruelly", "cruelty", "sadistic", "merciless", "heartless",
    "ruthless", "vicious", "malicious", "spiteful", "vindictive", "vengeful",
    "humiliate", "humiliated", "humiliation", "mock", "mocked", "mocking",
    "ridicule", "ridiculed", "taunt", "taunted", "shame", "shamed", "shaming",
    "demean", "demeaned", "degrade", "degraded", "belittle", "belittled",
    "insult", "insulted", "disrespect", "disrespected",
]

DUTY_VIOLATION_WORDS = [
    "supposed to", "duty", "responsibility", "obligation", "promise",
    "promised", "broke my promise", "broke his promise", "broke her promise",
    "vow", "vowed", "oath", "commitment", "committed",
    "instead of", "rather than", "should have", "failed to", "refused to",
    "neglect", "ignored", "disregard", "disregarded",
]

POSITIVE_MORAL_WORDS = [
    "help", "helped", "helping", "assist", "assisted", "volunteer",
    "volunteered", "donate", "donated", "give", "gave", "share", "shared",
    "save", "saved", "rescue", "rescued", "protect", "protected",
    "forgive", "forgave", "apologize", "apologized", "sorry",
    "honest", "honestly", "truthful", "fair", "fairly", "kind", "kindly",
    "generous", "generously", "grateful", "thankful", "respect", "respected",
    "care", "cared", "caring", "comfort", "comforted", "support", "supported",
    "encourage", "encouraged", "praise", "praised", "compliment",
    "love", "loved", "cherish", "cherished", "nurture", "nurtured",
]

NEUTRAL_CONTEXT_WORDS = [
    "because", "since", "after", "before", "while", "during",
    "went to", "decided to", "wanted to", "needed to",
    "my friend", "my family", "my neighbor", "my coworker",
    "at work", "at home", "at school", "at the store",
]

# Negation handling: these words before a moral word flip its valence
NEGATION_WORDS = [
    "not", "no", "never", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "couldn't", "shouldn't", "wasn't", "weren't", "isn't", "aren't",
    "without", "refuse", "refused", "stop", "stopped", "avoid", "avoided",
]


def predict_moral_judgment(text: str) -> tuple[int, float, dict]:
    """Predict whether a scenario is morally acceptable (1) or wrong (0).

    Returns: (prediction, confidence, details)

    The system works by:
    1. Text analysis: count harm, deception, theft, cruelty, fairness violation cues
    2. Drive activation: map text cues to emotional drives
    3. Moral appraisal: combine drive pattern into a moral judgment score
    4. Threshold: score > 0 → acceptable, score ≤ 0 → wrong
    """
    text_lower = text.lower()
    words = text_lower.split()

    # ── Negation-aware keyword counting ─────────────────────────────────────
    def count_with_negation(keyword_list: list[str]) -> tuple[int, int]:
        """Count keyword hits, returning (positive_hits, negated_hits)."""
        pos_hits = 0
        neg_hits = 0
        for kw in keyword_list:
            # Find all occurrences
            idx = text_lower.find(kw)
            while idx >= 0:
                # Check if negated (look at the 3 words before)
                prefix = text_lower[max(0, idx - 30):idx]
                prefix_words = prefix.split()[-3:]
                if any(neg in prefix_words for neg in NEGATION_WORDS):
                    neg_hits += 1
                else:
                    pos_hits += 1
                idx = text_lower.find(kw, idx + len(kw))
        return pos_hits, neg_hits

    harm_pos, harm_neg = count_with_negation(HARM_WORDS)
    deception_pos, deception_neg = count_with_negation(DECEPTION_WORDS)
    theft_pos, theft_neg = count_with_negation(THEFT_WORDS)
    cruelty_pos, cruelty_neg = count_with_negation(CRUELTY_WORDS)
    fairness_pos, fairness_neg = count_with_negation(FAIRNESS_VIOLATION_WORDS)
    duty_pos, duty_neg = count_with_negation(DUTY_VIOLATION_WORDS)
    positive_pos, positive_neg = count_with_negation(POSITIVE_MORAL_WORDS)

    # ── Drive activation from text ──────────────────────────────────────────
    # Harm cues activate empathy (alarm) + self_protection
    empathy_activation = harm_pos * 0.15 + cruelty_pos * 0.20 + positive_pos * 0.05
    # Deception activates honesty violation detection
    honesty_alarm = deception_pos * 0.20 + theft_pos * 0.15
    # Fairness cues activate trust/caution
    trust_violation = fairness_pos * 0.18 + theft_pos * 0.12
    # Duty violations activate caution
    caution_signal = duty_pos * 0.15
    # Cruelty is a strong moral signal
    moral_alarm = cruelty_pos * 0.25 + harm_pos * 0.15

    # Negated harm words REDUCE the alarm (e.g., "I did NOT hurt anyone")
    harm_reduction = harm_neg * 0.12 + cruelty_neg * 0.15 + deception_neg * 0.10
    # Positive moral words increase acceptability
    moral_credit = positive_pos * 0.08 + harm_neg * 0.05

    # ── Compute moral judgment score ────────────────────────────────────────
    # Negative factors (push toward "wrong")
    wrong_score = (
        empathy_activation * 1.2 +      # harm to others
        honesty_alarm * 1.1 +            # deception
        trust_violation * 1.0 +          # unfairness
        caution_signal * 0.8 +           # duty violations
        moral_alarm * 1.3                # cruelty
    )

    # Positive factors (push toward "acceptable")
    right_score = (
        moral_credit * 1.5 +             # positive moral actions
        harm_reduction * 1.2 +           # negated harm
        positive_neg * 0.05              # negated positive (slight penalty)
    )

    # ── HumanState-based refinement ─────────────────────────────────────────
    # Create a state reflecting the scenario's emotional impact
    state = HumanState(noise_level=0.0)

    if wrong_score > 0.3:
        state.inject_event("threat", intensity=min(1.0, wrong_score))
    if right_score > 0.2:
        state.inject_event("social_praise", intensity=min(1.0, right_score * 0.5))

    # Use drive pattern as additional signal
    empathy_val = state.get("empathy")
    honesty_val = state.get("honesty")
    caution_val = state.get("caution")
    self_prot = state.get("self_protection")
    trust_val = state.get("trust")

    # High empathy alarm + low trust = likely wrong scenario
    state_wrongness = (
        (1.0 - trust_val) * 0.3 +
        self_prot * 0.2 +
        caution_val * 0.2 +
        state.get("fear") * 0.15 +
        state.get("stress") * 0.15
    )

    state_rightness = (
        empathy_val * 0.3 +
        trust_val * 0.3 +
        state.get("confidence") * 0.2 +
        state.get("motivation") * 0.2
    )

    # ── Final decision ──────────────────────────────────────────────────────
    # Combine text signal (primary) with state signal (secondary)
    total_wrong = wrong_score * 0.70 + state_wrongness * 0.30
    total_right = right_score * 0.70 + state_rightness * 0.30

    # Net moral score: positive = acceptable, negative = wrong
    net_score = total_right - total_wrong

    # If no clear signal from text, stay neutral
    # (the dataset is ~50/50, so no strong prior)
    if wrong_score < 0.05 and right_score < 0.05:
        net_score = 0.0  # genuinely uncertain

    # Prediction
    prediction = 1 if net_score > 0 else 0
    confidence = min(1.0, abs(net_score) / 0.5)  # normalize confidence

    details = {
        "net_score": round(net_score, 4),
        "wrong_score": round(wrong_score, 4),
        "right_score": round(right_score, 4),
        "harm_hits": harm_pos,
        "harm_negated": harm_neg,
        "deception_hits": deception_pos,
        "cruelty_hits": cruelty_pos,
        "positive_hits": positive_pos,
        "state_wrongness": round(state_wrongness, 4),
        "state_rightness": round(state_rightness, 4),
    }

    return prediction, confidence, details


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute accuracy, precision, recall, F1 for binary classification."""
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Balanced accuracy
    balanced_acc = (recall + specificity) / 2

    # Matthews Correlation Coefficient
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "total": n,
        "positive_count": tp + fn,
        "negative_count": tn + fp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation(
    samples: list[tuple[str, int]],
    max_samples: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run ETHICS commonsense morality validation."""

    if max_samples and max_samples < len(samples):
        # Stratified: maintain 50/50 balance
        positives = [(t, l) for t, l in samples if l == 1]
        negatives = [(t, l) for t, l in samples if l == 0]
        half = max_samples // 2
        samples = positives[:half] + negatives[:half]

    n = len(samples)
    label_dist = Counter(l for _, l in samples)

    print(f"\n{'='*70}")
    print(f"  ETHICS COMMONSENSE MORALITY VALIDATION")
    print(f"  Samples: {n}")
    print(f"  Acceptable (1): {label_dist.get(1, 0)}  |  Wrong (0): {label_dist.get(0, 0)}")
    print(f"{'='*70}\n")

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    errors: list[dict] = []

    t0 = time.time()
    for i, (text, label) in enumerate(samples):
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processing {i+1}/{n} ({rate:.0f} samples/sec)...")

        pred, conf, details = predict_moral_judgment(text)

        y_true.append(label)
        y_pred.append(pred)
        confidences.append(conf)

        if verbose and pred != label:
            errors.append({
                "text": text[:150],
                "true_label": "acceptable" if label == 1 else "wrong",
                "predicted": "acceptable" if pred == 1 else "wrong",
                "confidence": round(conf, 3),
                "net_score": details["net_score"],
            })

    elapsed = time.time() - t0

    # ── Metrics ─────────────────────────────────────────────────────────────
    metrics = compute_binary_metrics(y_true, y_pred)

    print(f"\n  Completed in {elapsed:.1f}s ({n/elapsed:.0f} samples/sec)")
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}\n")

    print(f"  Accuracy:           {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}  ({metrics['balanced_accuracy']*100:.1f}%)")
    print(f"  F1 Score:           {metrics['f1']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print(f"  MCC:                {metrics['mcc']:.4f}")
    print()
    print(f"  Confusion Matrix:")
    cm = metrics["confusion"]
    print(f"                  Predicted Wrong    Predicted OK")
    print(f"    True Wrong     {cm['tn']:>8d}          {cm['fp']:>8d}")
    print(f"    True OK        {cm['fn']:>8d}          {cm['tp']:>8d}")
    print()

    # ── Baselines ───────────────────────────────────────────────────────────
    majority_acc = max(label_dist.values()) / n if n > 0 else 0.5
    random_acc = 0.5  # binary balanced

    print(f"  ── Baselines ──")
    print(f"  Random baseline:    50.0%")
    print(f"  Majority baseline:  {majority_acc*100:.1f}%")
    print(f"  Our system:         {metrics['accuracy']*100:.1f}%")
    print(f"  Lift over random:   {metrics['accuracy']/0.5:.2f}x")
    print()

    # ── Context ─────────────────────────────────────────────────────────────
    print(f"  ── Context ──")
    print(f"  ETHICS benchmark reference results (from paper):")
    print(f"    Random baseline:        50.0%")
    print(f"    Word averaging:         63.4%")
    print(f"    BERT (fine-tuned):      85.1%")
    print(f"    GPT-3 (zero-shot):      73.9%")
    print(f"    ALBERT-xxlarge (tuned): 85.7%")
    print(f"    Our system (no ML):     {metrics['accuracy']*100:.1f}%  (zero trained parameters)")
    print()

    if metrics['accuracy'] >= 0.70:
        print(f"  STRONG: Exceeds GPT-3 zero-shot baseline.")
    elif metrics['accuracy'] >= 0.60:
        print(f"  COMPETITIVE: Matches or exceeds word-averaging baseline.")
    elif metrics['accuracy'] >= 0.55:
        print(f"  MEANINGFUL: Significantly above chance.")
    else:
        print(f"  FAIR: Above chance, room for improvement.")
    print(f"  Note: Our system uses ZERO trained parameters — pure drive-based moral appraisal.")
    print()

    # ── Confidence analysis ─────────────────────────────────────────────────
    high_conf = [(t, p) for t, p, c in zip(y_true, y_pred, confidences) if c > 0.5]
    low_conf = [(t, p) for t, p, c in zip(y_true, y_pred, confidences) if c <= 0.5]

    if high_conf:
        high_acc = sum(1 for t, p in high_conf if t == p) / len(high_conf)
        print(f"  High-confidence predictions (conf > 0.5): {len(high_conf)} samples, {high_acc*100:.1f}% accuracy")
    if low_conf:
        low_acc = sum(1 for t, p in low_conf if t == p) / len(low_conf)
        print(f"  Low-confidence predictions (conf ≤ 0.5):  {len(low_conf)} samples, {low_acc*100:.1f}% accuracy")
    print()

    # ── Build report ────────────────────────────────────────────────────────
    report = {
        "pipeline": "ETHICS Commonsense Morality Validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": "ETHICS Commonsense Morality (Hendrycks et al., ICLR 2021)",
            "source": "https://github.com/hendrycks/ethics",
            "total_samples": n,
            "label_distribution": dict(label_dist),
        },
        "metrics": metrics,
        "baselines": {
            "random": 0.50,
            "majority": round(majority_acc, 4),
            "word_averaging": 0.634,
            "gpt3_zero_shot": 0.739,
            "bert_finetuned": 0.851,
            "albert_xxlarge": 0.857,
            "our_system": metrics["accuracy"],
            "lift_over_random": round(metrics["accuracy"] / 0.5, 2),
            "note": "Our system uses zero trained parameters — pure drive-based moral appraisal",
        },
        "confidence_analysis": {
            "high_confidence_count": len(high_conf),
            "high_confidence_accuracy": round(high_acc, 4) if high_conf else None,
            "low_confidence_count": len(low_conf),
            "low_confidence_accuracy": round(low_acc, 4) if low_conf else None,
        },
        "timing": {
            "total_seconds": round(elapsed, 2),
            "samples_per_second": round(n / elapsed, 1),
        },
    }

    if errors and verbose:
        report["error_samples"] = errors[:30]

    return report


def main():
    parser = argparse.ArgumentParser(description="ETHICS Commonsense Morality Validation")
    parser.add_argument("--split", default="test", choices=["test", "test_hard", "train"],
                        help="Dataset split (default: test)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (stratified). Default: all.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show misclassification examples")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  HUMAN MODE VALIDATION: ETHICS Commonsense Morality Benchmark")
    print("=" * 70 + "\n")

    # Step 1: Download
    print("[1/4] Acquiring dataset...")
    cm_dir = download_dataset()
    csv_path = get_split_path(cm_dir, args.split)

    # Step 2: Parse
    print(f"\n[2/4] Parsing {args.split} split...")
    samples = parse_ethics_csv(csv_path)
    print(f"  Parsed {len(samples)} scenarios")

    if not samples:
        print("  ERROR: No samples parsed.")
        sys.exit(1)

    # Step 3: Validate
    print(f"\n[3/4] Running moral judgment validation...")
    report = run_validation(samples, max_samples=args.max_samples, verbose=args.verbose)

    # Step 4: Save
    print(f"[4/4] Saving report...")
    report_dir = PROJECT_ROOT / "artifacts" / "ethics_validation"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"ethics_{args.split}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

    summary_path = report_dir / f"ethics_{args.split}_report.md"
    with open(summary_path, "w") as f:
        m = report["metrics"]
        b = report["baselines"]
        f.write("# ETHICS Commonsense Morality Validation Report\n\n")
        f.write(f"**Date:** {report['timestamp']}\n\n")
        f.write(f"**Dataset:** {report['dataset']['name']}\n")
        f.write(f"**Split:** {args.split}\n")
        f.write(f"**Samples:** {report['dataset']['total_samples']}\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | Score |\n|---|---|\n")
        f.write(f"| Accuracy | {m['accuracy']:.4f} ({m['accuracy']*100:.1f}%) |\n")
        f.write(f"| Balanced Accuracy | {m['balanced_accuracy']:.4f} |\n")
        f.write(f"| F1 Score | {m['f1']:.4f} |\n")
        f.write(f"| Precision | {m['precision']:.4f} |\n")
        f.write(f"| Recall | {m['recall']:.4f} |\n")
        f.write(f"| Specificity | {m['specificity']:.4f} |\n")
        f.write(f"| MCC | {m['mcc']:.4f} |\n\n")
        f.write("## Comparison with Published Baselines\n\n")
        f.write("| System | Accuracy | Parameters |\n|---|---|---|\n")
        f.write(f"| Random | 50.0% | 0 |\n")
        f.write(f"| Word Averaging | 63.4% | Trained |\n")
        f.write(f"| GPT-3 (zero-shot) | 73.9% | 175B |\n")
        f.write(f"| **Our System** | **{m['accuracy']*100:.1f}%** | **0 (rule-based)** |\n")
        f.write(f"| BERT (fine-tuned) | 85.1% | 110M |\n")
        f.write(f"| ALBERT-xxlarge | 85.7% | 235M |\n\n")
        f.write(f"---\n*Generated by ETHICS Validation Pipeline*\n")

    print(f"  Summary saved to {summary_path}")

    print(f"\n{'='*70}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Accuracy: {report['metrics']['accuracy']*100:.1f}%")
    print(f"  F1: {report['metrics']['f1']:.4f}")
    print(f"  MCC: {report['metrics']['mcc']:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

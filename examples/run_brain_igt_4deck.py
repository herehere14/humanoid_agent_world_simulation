#!/usr/bin/env python3
"""Brain-First IGT Validation — Full 4-Deck (A/B/C/D).

Uses the original 4 decks instead of collapsing to safe/risky.
With 4 actions, history-only base rate is weaker, giving the brain's
11-dimensional state space more room to differentiate.

Dataset: 504 real humans x 100 trials, ALL participants,
         train on first 70 trials, predict last 30.
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.brain.brain_predictor import BrainPredictor
from src.prompt_forest.brain.controller import BrainController
from src.prompt_forest.brain.ensemble_predictor import AdaptiveEnsemblePredictor
from src.prompt_forest.brain.latent_state import LatentStatePredictor
from src.prompt_forest.brain.prospect_learner import ProspectTheoryLearner
from src.prompt_forest.brain.rl_adapter import BrainRLAdapter
from src.prompt_forest.brain.sequence_predictor import SequencePredictor
from src.prompt_forest.brain.transition_model import LearnedTransitionModel
from src.prompt_forest.modes.human_mode.branches import create_human_mode_forest
from src.prompt_forest.modes.human_mode.memory import HumanModeMemory
from src.prompt_forest.modes.human_mode.router import HumanModeRouter
from src.prompt_forest.state.human_state import HumanState
from src.prompt_forest.types import TaskInput

# Reuse data loader
import run_adaptive_learning_validation as base_val

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECK_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}
ACTIONS = ["A", "B", "C", "D"]
SAFE_DECKS = {3, 4}  # For reporting only
WINDOW_SIZE = 10
TRAIN_FRAC = 0.7


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Deck4Result:
    participant_id: str
    study: str
    safe_fraction: float
    deck_fractions: dict[str, float]
    holdout_accuracy: float
    random_accuracy: float
    majority_accuracy: float
    window_accuracies: list[float]
    final_priors: dict[str, float]


# ---------------------------------------------------------------------------
# Warm-start transfer
# ---------------------------------------------------------------------------

def _warm_start_from_similar(
    target_uid: str,
    predictor: BrainPredictor,
    min_donor_obs: int = 40,
    top_k: int = 5,
    transfer_weight: float = 0.35,
    min_similarity: float = 0.5,
) -> None:
    """Transfer outcome biases and transitions from similar completed users."""
    import math

    target = predictor._users.get(target_uid)
    if target is None or target.n_observations < 5:
        return

    def _sig(model: Any) -> dict[str, float]:
        s: dict[str, float] = {}
        for a, v in model.prior.items():
            s[f"p_{a}"] = v
        for a in predictor.actions:
            for vbin in ["positive", "negative"]:
                s[f"ob_{a}_{vbin}"] = model.outcome_bias.get(a, {}).get(vbin, 0.0)
        for src in predictor.actions:
            for vbin in ["positive", "negative"]:
                for dst in predictor.actions:
                    s[f"tr_{src}_{vbin}_{dst}"] = (
                        model.transitions.get(src, {}).get(vbin, {}).get(dst, 0.0)
                    )
        return s

    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        ma = math.sqrt(sum(a.get(k, 0) ** 2 for k in keys))
        mb = math.sqrt(sum(b.get(k, 0) ** 2 for k in keys))
        if ma < 1e-8 or mb < 1e-8:
            return 0.0
        return dot / (ma * mb)

    target_sig = _sig(target)
    donors: list[tuple[str, float]] = []
    for uid, model in predictor._users.items():
        if uid == target_uid or model.n_observations < min_donor_obs:
            continue
        sim = _cosine(target_sig, _sig(model))
        if sim >= min_similarity:
            donors.append((uid, sim))

    donors.sort(key=lambda x: x[1], reverse=True)
    donors = donors[:top_k]
    if not donors:
        return

    total_sim = sum(s for _, s in donors)
    tw = transfer_weight

    for action in predictor.actions:
        for vbin in ["positive", "negative", "neutral"]:
            donor_avg = 0.0
            for uid, sim in donors:
                d = predictor._users[uid]
                donor_avg += (sim / total_sim) * d.outcome_bias.get(action, {}).get(vbin, 0.0)
            cur = target.outcome_bias[action].get(vbin, 0.0)
            target.outcome_bias[action][vbin] = (1 - tw) * cur + tw * donor_avg

    for src in predictor.actions:
        for vbin in ["positive", "negative", "neutral"]:
            for dst in predictor.actions:
                donor_avg = 0.0
                for uid, sim in donors:
                    d = predictor._users[uid]
                    donor_avg += (sim / total_sim) * d.transitions.get(src, {}).get(vbin, {}).get(dst, 0.0)
                cur = target.transitions[src][vbin].get(dst, 0.0)
                target.transitions[src][vbin][dst] = (1 - tw) * cur + tw * donor_avg

    for action in predictor.actions:
        donor_ctx: dict[str, float] = {}
        for uid, sim in donors:
            d = predictor._users[uid]
            for feat, w in d.context_weights.get(action, {}).items():
                donor_ctx[feat] = donor_ctx.get(feat, 0.0) + (sim / total_sim) * w
        for feat, avg_w in donor_ctx.items():
            cur = target.context_weights[action].get(feat, 0.0)
            target.context_weights[action][feat] = (1 - tw) * cur + tw * avg_w


# ---------------------------------------------------------------------------
# Run one participant
# ---------------------------------------------------------------------------

def run_brain_participant(
    participant: base_val.RealIGTParticipant,
    transition_model: LearnedTransitionModel,
    brain_predictor: BrainPredictor,
    rl_adapter: BrainRLAdapter,
    prospect_learner: ProspectTheoryLearner,
    ensemble: AdaptiveEnsemblePredictor,
    latent_predictor: LatentStatePredictor | None = None,
    sequence_predictor: SequencePredictor | None = None,
) -> Deck4Result:
    """Run one participant through the ensemble pipeline with 4 deck actions."""
    n_train = int(participant.n_trials * TRAIN_FRAC)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]
    uid = participant.participant_id

    state = HumanState(
        decay_rate=0.05,
        momentum=0.45,
        noise_level=0.0,
        adaptive_baselines=True,
        baseline_lr=0.012,
        baseline_warmup=40,
    )
    forest = create_human_mode_forest()
    router = HumanModeRouter(top_k=4, noise_level=0.0)
    memory = HumanModeMemory(
        emotional_decay=0.92,
        trauma_amplification=1.5,
        experience_bias_strength=0.4,
    )
    controller = BrainController()

    correct_per_window: dict[int, list[bool]] = {}
    prev_outcome: float | None = None
    prev_action: str | None = None
    cumulative_score = 0.0
    deck_outcomes: dict[str, list[float]] = {d: [] for d in ACTIONS}
    consecutive_same = 0
    last_action: str | None = None

    # ------- Training phase -------
    for idx, deck_choice in enumerate(train_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[idx] + participant.losses[idx]
        cumulative_score += net_outcome

        # 1. Update brain state from previous outcome
        if prev_outcome is not None and prev_action is not None:
            deltas = transition_model.compute_deltas(
                user_id=uid,
                state_vars=state.variables,
                outcome=prev_outcome,
                action=prev_action,
            )
            state.update(deltas)

        # 2. Route through cognitive forest
        progress = idx / max(1, n_train)
        task = TaskInput(
            task_id=f"{uid}_t{idx}",
            text=f"IGT trial {idx+1}",
            task_type="decision",
            metadata={},
        )
        route, conflicts = router.route(task, forest, state)
        for conflict in conflicts:
            state.resolve_conflict(conflict, "weighted_compromise")

        # 3. Brain output
        brain_output = controller.build_output(
            state=state,
            route=route,
            conflicts=state.active_conflicts,
            human_memory=memory,
            branch_weights={
                name: b.state.weight
                for name, b in forest.branches.items()
                if not b.state.metadata.get("category_node")
            },
        )

        # Track consecutive same action
        if last_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_action = actual

        # 4. Build context features
        deck_counts = {d: 0 for d in ACTIONS}
        for c in train_choices[:idx + 1]:
            deck_counts[DECK_MAP[c]] += 1
        context: dict[str, float] = {
            "trial_progress": progress,
            "cumulative_score_norm": cumulative_score
            / max(1.0, abs(cumulative_score) + 100.0),
            "consecutive_same": min(1.0, consecutive_same / 5.0),
        }
        # Per-deck rates
        for d in ACTIONS:
            context[f"deck_rate::{d}"] = deck_counts[d] / max(1, idx + 1)
        # Per-deck outcome features
        for d in ACTIONS:
            recent_do = deck_outcomes[d][-5:]
            if recent_do:
                m = sum(recent_do) / len(recent_do)
                context[f"deck_outcome_mean::{d}"] = m / (abs(m) + 100.0)
                context[f"deck_outcome_neg::{d}"] = sum(
                    1 for o in recent_do if o < 0
                ) / len(recent_do)
        # Brain-derived features
        context.update(transition_model.get_ev_features(uid))
        context.update(transition_model.get_outcome_features(uid))
        context.update(transition_model.get_per_action_features(uid))
        context.update(brain_predictor.compute_sequence_features(uid))

        # 5. Predict via ensemble
        # Component 1: Prospect theory
        prospect_probs = prospect_learner.predict(uid)
        prospect_pred = max(prospect_probs, key=prospect_probs.get)

        # Component 2: History (brain predictor)
        history_probs = brain_predictor.predict_probs(uid, brain_output, context)
        history_pred = max(history_probs, key=history_probs.get)

        # Additional components
        component_probs = [prospect_probs, history_probs]
        component_preds = [prospect_pred, history_pred]
        if latent_predictor is not None:
            ev_est = {a: transition_model._user_ev.get(uid, {}).get(a, 0.0) for a in ACTIONS}
            latent_probs = latent_predictor.predict(
                uid, last_action=prev_action, last_outcome=prev_outcome, ev_estimates=ev_est,
            )
            latent_pred = max(latent_probs, key=latent_probs.get)
            component_probs.append(latent_probs)
            component_preds.append(latent_pred)
        if sequence_predictor is not None:
            seq_probs = sequence_predictor.predict(uid)
            seq_pred = max(seq_probs, key=seq_probs.get)
            component_probs.append(seq_probs)
            component_preds.append(seq_pred)

        ensemble_probs = ensemble.predict(uid, component_probs)
        ensemble_pred = max(ensemble_probs, key=ensemble_probs.get)

        is_correct = ensemble_pred == actual
        correct_per_window.setdefault(idx // WINDOW_SIZE, []).append(is_correct)

        # Update ensemble weights
        ensemble.update_weights(uid, component_preds, actual)

        # 6. RL adapt (brain predictor + transition model)
        rl_adapter.adapt(
            user_id=uid,
            brain_output=brain_output,
            predicted_action=ensemble_pred,
            actual_action=actual,
            outcome=net_outcome,
            context=context,
        )

        # Update prospect theory learner
        prospect_learner.adapt_parameters(uid, prospect_pred, actual)
        prospect_learner.update(uid, actual, net_outcome)

        # Update latent state predictor
        if latent_predictor is not None:
            ev_est = {a: transition_model._user_ev.get(uid, {}).get(a, 0.0) for a in ACTIONS}
            latent_predictor.update(
                uid, actual, last_action=prev_action,
                last_outcome=prev_outcome, ev_estimates=ev_est,
            )

        # Update sequence predictor
        if sequence_predictor is not None:
            sequence_predictor.update(uid, actual, net_outcome)

        deck_outcomes[actual].append(net_outcome)

        # Transfer learning after 15 trials
        if idx == 14 and brain_predictor.user_count > 3:
            _warm_start_from_similar(uid, brain_predictor, min_donor_obs=40, top_k=5)

        # 7. Record memory
        memory.record(
            event_id=task.task_id,
            task=task,
            state=state,
            reward=brain_predictor.prediction_accuracy_reward(
                ensemble_pred, actual, 0.5
            ),
            selected_branch=(
                route.activated_branches[0] if route.activated_branches else ""
            ),
            active_branches=list(route.activated_branches),
        )

        prev_outcome = net_outcome
        prev_action = actual

    # ------- Holdout phase -------
    holdout_correct = 0
    for idx, deck_choice in enumerate(holdout_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[n_train + idx] + participant.losses[n_train + idx]

        if prev_outcome is not None and prev_action is not None:
            deltas = transition_model.compute_deltas(
                user_id=uid,
                state_vars=state.variables,
                outcome=prev_outcome,
                action=prev_action,
            )
            state.update(deltas)

        task = TaskInput(
            task_id=f"{uid}_h{idx}",
            text=f"IGT holdout trial {n_train+idx+1}",
            task_type="decision",
            metadata={},
        )
        route, conflicts = router.route(task, forest, state)
        for conflict in conflicts:
            state.resolve_conflict(conflict, "weighted_compromise")

        brain_output = controller.build_output(
            state=state,
            route=route,
            conflicts=state.active_conflicts,
            human_memory=memory,
            branch_weights={
                name: b.state.weight
                for name, b in forest.branches.items()
                if not b.state.metadata.get("category_node")
            },
        )

        # Context
        deck_counts = {d: 0 for d in ACTIONS}
        for c in train_choices:
            deck_counts[DECK_MAP[c]] += 1
        for c in holdout_choices[:idx]:
            deck_counts[DECK_MAP[c]] += 1
        total_trials = n_train + idx
        context = {
            "trial_progress": 1.0,
            "cumulative_score_norm": cumulative_score
            / max(1.0, abs(cumulative_score) + 100.0),
            "consecutive_same": min(1.0, consecutive_same / 5.0),
        }
        for d in ACTIONS:
            context[f"deck_rate::{d}"] = deck_counts[d] / max(1, total_trials)
        for d in ACTIONS:
            recent_do = deck_outcomes[d][-5:]
            if recent_do:
                m = sum(recent_do) / len(recent_do)
                context[f"deck_outcome_mean::{d}"] = m / (abs(m) + 100.0)
                context[f"deck_outcome_neg::{d}"] = sum(
                    1 for o in recent_do if o < 0
                ) / len(recent_do)
        context.update(transition_model.get_ev_features(uid))
        context.update(transition_model.get_outcome_features(uid))
        context.update(transition_model.get_per_action_features(uid))
        context.update(brain_predictor.compute_sequence_features(uid))

        # Predict BEFORE observing ground truth — via ensemble
        prospect_probs = prospect_learner.predict(uid)
        prospect_pred = max(prospect_probs, key=prospect_probs.get)
        history_probs = brain_predictor.predict_probs(uid, brain_output, context)
        history_pred = max(history_probs, key=history_probs.get)

        component_probs = [prospect_probs, history_probs]
        component_preds = [prospect_pred, history_pred]
        if latent_predictor is not None:
            ev_est = {a: transition_model._user_ev.get(uid, {}).get(a, 0.0) for a in ACTIONS}
            latent_probs = latent_predictor.predict(
                uid, last_action=prev_action, last_outcome=prev_outcome, ev_estimates=ev_est,
            )
            latent_pred = max(latent_probs, key=latent_probs.get)
            component_probs.append(latent_probs)
            component_preds.append(latent_pred)
        if sequence_predictor is not None:
            seq_probs = sequence_predictor.predict(uid)
            seq_pred = max(seq_probs, key=seq_probs.get)
            component_probs.append(seq_probs)
            component_preds.append(seq_pred)

        ensemble_probs = ensemble.predict(uid, component_probs)
        ensemble_pred = max(ensemble_probs, key=ensemble_probs.get)

        if ensemble_pred == actual:
            holdout_correct += 1

        # Update ensemble weights from holdout observations
        ensemble.update_weights(uid, component_preds, actual)

        # Observe ground truth (lightweight updates)
        brain_predictor.observe(
            uid, actual, net_outcome,
            update_prior=True,
            update_outcome_bias=True,
            regime=brain_output.regime,
            brain_state_summary=brain_output.state_summary,
        )
        prospect_learner.adapt_parameters(uid, prospect_pred, actual)
        prospect_learner.update(uid, actual, net_outcome)
        if latent_predictor is not None:
            ev_est = {a: transition_model._user_ev.get(uid, {}).get(a, 0.0) for a in ACTIONS}
            latent_predictor.update(
                uid, actual, last_action=prev_action,
                last_outcome=prev_outcome, ev_estimates=ev_est,
            )
        if sequence_predictor is not None:
            sequence_predictor.update(uid, actual, net_outcome)
        transition_model.compute_deltas(
            user_id=uid,
            state_vars=state.variables,
            outcome=net_outcome,
            action=actual,
        )

        cumulative_score += net_outcome
        deck_outcomes[actual].append(net_outcome)
        if last_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_action = actual

        prev_outcome = net_outcome
        prev_action = actual

    # Random baseline (4-way)
    rng = random.Random(hash(uid) % 100000 + 7777)
    random_correct = sum(
        1
        for c in holdout_choices
        if rng.choice(ACTIONS) == DECK_MAP[c]
    )

    # Majority-class baseline (most frequent deck from training)
    train_deck_counts = {d: 0 for d in ACTIONS}
    for c in train_choices:
        train_deck_counts[DECK_MAP[c]] += 1
    majority_deck = max(train_deck_counts, key=train_deck_counts.get)
    majority_correct = sum(1 for c in holdout_choices if DECK_MAP[c] == majority_deck)

    # Deck fractions
    total_choices = len(participant.choices)
    deck_fracs = {}
    for d in ACTIONS:
        deck_fracs[d] = sum(1 for c in participant.choices if DECK_MAP[c] == d) / total_choices

    window_accs = [
        sum(items) / len(items) if items else 0.0
        for _, items in sorted(correct_per_window.items())
    ]

    return Deck4Result(
        participant_id=uid,
        study=participant.study,
        safe_fraction=participant.safe_fraction,
        deck_fractions=deck_fracs,
        holdout_accuracy=holdout_correct / max(1, n_holdout),
        random_accuracy=random_correct / max(1, n_holdout),
        majority_accuracy=majority_correct / max(1, n_holdout),
        window_accuracies=window_accs,
        final_priors=brain_predictor.get_user_summary(uid).get("priors", {}),
    )


# ---------------------------------------------------------------------------
# History-only baseline (no brain)
# ---------------------------------------------------------------------------

def run_history_only_participant(
    participant: base_val.RealIGTParticipant,
    transition_model: LearnedTransitionModel,
    predictor: BrainPredictor,
) -> float:
    """Run one participant with history-only (no brain output)."""
    from src.prompt_forest.brain.output import BrainOutput, BrainControlSignals, BrainActionTendencies

    n_train = int(participant.n_trials * TRAIN_FRAC)
    n_holdout = participant.n_trials - n_train
    train_choices = participant.choices[:n_train]
    holdout_choices = participant.choices[n_train:]
    uid = participant.participant_id + "_hist"

    prev_outcome: float | None = None
    prev_action: str | None = None
    deck_outcomes: dict[str, list[float]] = {d: [] for d in ACTIONS}
    cumulative_score = 0.0
    consecutive_same = 0
    last_action: str | None = None

    # Null brain output
    null_brain = BrainOutput(
        regime="stable",
        state={},
        dominant_drives=[],
        branch_activations={},
        active_branches=[],
        conflicts=[],
        control_signals=BrainControlSignals(
            approach_drive=0.0, avoidance_drive=0.0, exploration_drive=0.0,
            switch_pressure=0.0, persistence_drive=0.0, self_protection=0.0,
            social_openness=0.0, cognitive_effort=0.0,
        ),
        action_tendencies=BrainActionTendencies(
            act=0.0, inhibit=0.0, explore=0.0, exploit=0.0, reflect=0.0, react=0.0,
        ),
    )

    # Training
    for idx, deck_choice in enumerate(train_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[idx] + participant.losses[idx]
        cumulative_score += net_outcome

        if prev_outcome is not None and prev_action is not None:
            transition_model.compute_deltas(
                user_id=uid,
                state_vars={},
                outcome=prev_outcome,
                action=prev_action,
            )

        if last_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_action = actual

        deck_counts = {d: 0 for d in ACTIONS}
        for c in train_choices[:idx + 1]:
            deck_counts[DECK_MAP[c]] += 1
        context: dict[str, float] = {
            "trial_progress": idx / max(1, n_train),
            "cumulative_score_norm": cumulative_score / max(1.0, abs(cumulative_score) + 100.0),
            "consecutive_same": min(1.0, consecutive_same / 5.0),
        }
        for d in ACTIONS:
            context[f"deck_rate::{d}"] = deck_counts[d] / max(1, idx + 1)
        for d in ACTIONS:
            recent_do = deck_outcomes[d][-5:]
            if recent_do:
                m = sum(recent_do) / len(recent_do)
                context[f"deck_outcome_mean::{d}"] = m / (abs(m) + 100.0)
                context[f"deck_outcome_neg::{d}"] = sum(1 for o in recent_do if o < 0) / len(recent_do)
        context.update(transition_model.get_ev_features(uid))
        context.update(transition_model.get_outcome_features(uid))
        context.update(transition_model.get_per_action_features(uid))
        context.update(predictor.compute_sequence_features(uid))

        prediction = predictor.predict(uid, null_brain, context)
        predictor.update(
            uid, null_brain, actual,
            context=context, outcome=net_outcome,
        )
        deck_outcomes[actual].append(net_outcome)
        prev_outcome = net_outcome
        prev_action = actual

    # Holdout
    holdout_correct = 0
    for idx, deck_choice in enumerate(holdout_choices):
        actual = DECK_MAP[deck_choice]
        net_outcome = participant.wins[n_train + idx] + participant.losses[n_train + idx]

        if prev_outcome is not None and prev_action is not None:
            transition_model.compute_deltas(
                user_id=uid, state_vars={}, outcome=prev_outcome, action=prev_action,
            )

        deck_counts = {d: 0 for d in ACTIONS}
        for c in train_choices:
            deck_counts[DECK_MAP[c]] += 1
        for c in holdout_choices[:idx]:
            deck_counts[DECK_MAP[c]] += 1
        total_trials = n_train + idx
        context = {
            "trial_progress": 1.0,
            "cumulative_score_norm": cumulative_score / max(1.0, abs(cumulative_score) + 100.0),
            "consecutive_same": min(1.0, consecutive_same / 5.0),
        }
        for d in ACTIONS:
            context[f"deck_rate::{d}"] = deck_counts[d] / max(1, total_trials)
        for d in ACTIONS:
            recent_do = deck_outcomes[d][-5:]
            if recent_do:
                m = sum(recent_do) / len(recent_do)
                context[f"deck_outcome_mean::{d}"] = m / (abs(m) + 100.0)
                context[f"deck_outcome_neg::{d}"] = sum(1 for o in recent_do if o < 0) / len(recent_do)
        context.update(transition_model.get_ev_features(uid))
        context.update(transition_model.get_outcome_features(uid))
        context.update(transition_model.get_per_action_features(uid))
        context.update(predictor.compute_sequence_features(uid))

        prediction = predictor.predict(uid, null_brain, context)
        if prediction.predicted_action == actual:
            holdout_correct += 1

        predictor.observe(uid, actual, net_outcome, update_prior=True, update_outcome_bias=True)
        transition_model.compute_deltas(
            user_id=uid, state_vars={}, outcome=net_outcome, action=actual,
        )
        cumulative_score += net_outcome
        deck_outcomes[actual].append(net_outcome)
        if last_action == actual:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_action = actual
        prev_outcome = net_outcome
        prev_action = actual

    return holdout_correct / max(1, n_holdout)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(
    results: list[Deck4Result],
    history_accs: list[float],
) -> None:
    n = len(results)
    mean_acc = mean(r.holdout_accuracy for r in results)
    mean_rand = mean(r.random_accuracy for r in results)
    mean_maj = mean(r.majority_accuracy for r in results)
    mean_hist = mean(history_accs)
    std_acc = stdev(r.holdout_accuracy for r in results) if n > 1 else 0.0
    std_hist = stdev(history_accs) if n > 1 else 0.0

    learners = [i for i, r in enumerate(results) if r.safe_fraction > 0.65]
    mixed = [i for i, r in enumerate(results) if 0.35 <= r.safe_fraction <= 0.65]
    risk_seekers = [i for i, r in enumerate(results) if r.safe_fraction < 0.35]

    brain_contrib = mean_acc - mean_hist

    print()
    print("=" * 90)
    print("  4-DECK IGT VALIDATION — BRAIN vs HISTORY-ONLY")
    print("  Steingroever et al. (2015) — All 504 participants, 4 actions (A/B/C/D)")
    print("=" * 90)
    print()
    print(f"  Participants:     {n}")
    print(f"  Actions:          4 decks (A, B, C, D)")
    print(f"  Train/Holdout:    70/30 trials per participant")
    print(f"  Random baseline:  25% (1/4 decks)")
    print()
    print("  --- Overall Results ---")
    print()
    print(f"  Full Brain:       {mean_acc:.1%} (+/- {std_acc:.1%})")
    print(f"  History-Only:     {mean_hist:.1%} (+/- {std_hist:.1%})")
    print(f"  Majority-class:   {mean_maj:.1%}")
    print(f"  Random:           {mean_rand:.1%}")
    print()
    print(f"  Brain contribution: {brain_contrib:+.1%}")
    print(f"  Lift over random:   {mean_acc - mean_rand:+.1%}")
    print(f"  Lift over majority: {mean_acc - mean_maj:+.1%}")
    print()

    # Per-group breakdown
    print("  --- Per-Group Results ---")
    print(f"  {'Group':>30s}    {'Brain':>8s}  {'History':>8s}  {'Contrib':>8s}  {'Random':>8s}")
    print("  " + "-" * 76)
    for label, idxs in [
        ("ALL", list(range(n))),
        ("Learners (>65% safe)", learners),
        ("Mixed (35-65% safe)", mixed),
        ("Risk-seekers (<35%)", risk_seekers),
    ]:
        if not idxs:
            continue
        g_brain = mean(results[i].holdout_accuracy for i in idxs)
        g_hist = mean(history_accs[i] for i in idxs)
        g_rand = mean(results[i].random_accuracy for i in idxs)
        g_contrib = g_brain - g_hist
        print(
            f"  {label:>30s} (n={len(idxs):>3d})  "
            f"{g_brain:>7.1%}  {g_hist:>7.1%}  {g_contrib:>+7.1%}  {g_rand:>7.1%}"
        )
    print()

    # Learning curve
    max_windows = max(len(r.window_accuracies) for r in results)
    print("  --- Learning Curve (training windows) ---")
    for w in range(min(max_windows, 7)):
        accs_at_w = [
            r.window_accuracies[w]
            for r in results
            if w < len(r.window_accuracies)
        ]
        if accs_at_w:
            print(f"    Window {w+1}: {mean(accs_at_w):.1%}")
    print()

    # Top brain helps
    contribs = [(results[i], history_accs[i]) for i in range(n)]
    contribs.sort(key=lambda x: x[0].holdout_accuracy - x[1], reverse=True)
    print("  --- Top 10 brain helps ---")
    print(f"  {'ID':>10s}  {'safe%':>5s}  {'Brain':>6s}  {'Hist':>6s}  {'Contrib':>7s}  {'Deck priors':>30s}")
    for r, h in contribs[:10]:
        c = r.holdout_accuracy - h
        priors_str = " ".join(f"{d}={r.final_priors.get(d, 0):.2f}" for d in ACTIONS)
        print(f"  {r.participant_id:>10s}  {r.safe_fraction:>5.0%}  {r.holdout_accuracy:>5.0%}  {h:>5.0%}  {c:>+6.0%}  {priors_str}")
    print()

    # Top brain hurts
    print("  --- Top 10 brain hurts ---")
    for r, h in contribs[-10:]:
        c = r.holdout_accuracy - h
        priors_str = " ".join(f"{d}={r.final_priors.get(d, 0):.2f}" for d in ACTIONS)
        print(f"  {r.participant_id:>10s}  {r.safe_fraction:>5.0%}  {r.holdout_accuracy:>5.0%}  {h:>5.0%}  {c:>+6.0%}  {priors_str}")
    print()

    n_helps = sum(1 for r, h in contribs if r.holdout_accuracy > h)
    n_hurts = sum(1 for r, h in contribs if r.holdout_accuracy < h)
    n_ties = n - n_helps - n_hurts
    print(f"  Brain helps: {n_helps}/{n}  |  Hurts: {n_hurts}/{n}  |  Ties: {n_ties}/{n}")
    print()

    # Verdicts
    print("  --- Verdicts ---")
    verdicts = [
        ("Brain > random (25%)", mean_acc > 0.25),
        ("Brain > majority-class", mean_acc > mean_maj),
        ("Brain > history-only", mean_acc > mean_hist),
        ("Brain contribution > 0%", brain_contrib > 0),
        ("Brain accuracy >= 40%", mean_acc >= 0.40),
    ]
    for label, passed in verdicts:
        marker = "[+]" if passed else "[-]"
        print(f"    {marker} {label}: {'PASS' if passed else 'FAIL'}")
    print()
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 90)
    print("  4-DECK IGT VALIDATION")
    print("  Loading Steingroever et al. (2015) — ALL 504 participants, 4 decks")
    print("=" * 90)
    print()

    # Load ALL participants (no cap)
    participants = base_val.load_steingroever_data(
        max_participants=9999,  # effectively all
        seed=0,
    )

    safe_fracs = [p.safe_fraction for p in participants]
    n_learn = sum(1 for f in safe_fracs if f > 0.65)
    n_mixed = sum(1 for f in safe_fracs if 0.35 <= f <= 0.65)
    n_risk = sum(1 for f in safe_fracs if f < 0.35)
    print(
        f"  Loaded {len(participants)} participants "
        f"(learners={n_learn}, mixed={n_mixed}, risk-seekers={n_risk})"
    )
    print()

    # ===== ENSEMBLE (Prospect Theory + Brain Predictor) =====
    print("  === Running ENSEMBLE (Prospect + Brain, 4 decks) ===")
    transition_model = LearnedTransitionModel(lr=0.02)
    brain_predictor = BrainPredictor(
        actions=ACTIONS,
        learning_rate=0.14,
        prior_lr=0.08,
        context_lr=0.06,
        min_lr=0.025,
        anneal_rate=0.005,
        outcome_lr=0.16,
        transition_lr=0.14,
    )
    rl_adapter = BrainRLAdapter(
        transition_model=transition_model,
        brain_predictor=brain_predictor,
    )
    prospect_learner = ProspectTheoryLearner(
        actions=ACTIONS,
        adapt_lr=0.04,
        anneal_rate=0.995,
    )
    latent_predictor = None  # Dropped — too little signal with 70 trials
    sequence_predictor = SequencePredictor(
        actions=ACTIONS,
        alpha=0.3,
        recency_decay=0.98,
    )
    ensemble = AdaptiveEnsemblePredictor(
        n_components=3,
        component_names=["prospect", "brain_history", "sequence"],
        init_weights=[0.20, 0.05, 0.75],
        eta=0.10,
        min_weight=0.03,
    )

    results: list[Deck4Result] = []
    for i, participant in enumerate(participants):
        if (i + 1) % 50 == 0 or i == 0 or i == len(participants) - 1:
            print(
                f"    [{i+1:>3d}/{len(participants)}] {participant.participant_id:>10s} "
                f"({participant.study:>18s}, safe={participant.safe_fraction:.0%})",
                flush=True,
            )
        result = run_brain_participant(
            participant=participant,
            transition_model=transition_model,
            brain_predictor=brain_predictor,
            rl_adapter=rl_adapter,
            prospect_learner=prospect_learner,
            ensemble=ensemble,
            latent_predictor=latent_predictor,
            sequence_predictor=sequence_predictor,
        )
        results.append(result)
        # Accumulate population statistics for transfer learning
        if sequence_predictor is not None:
            sequence_predictor.finalize_user(participant.participant_id)
        prospect_learner.finalize_user(participant.participant_id)
    print()

    # ===== HISTORY-ONLY =====
    print("  === Running HISTORY-ONLY (4 decks) ===")
    hist_transition = LearnedTransitionModel(lr=0.02)
    hist_predictor = BrainPredictor(
        actions=ACTIONS,
        learning_rate=0.14,
        prior_lr=0.08,
        context_lr=0.06,
        min_lr=0.025,
        anneal_rate=0.005,
        outcome_lr=0.16,
        transition_lr=0.14,
    )

    history_accs: list[float] = []
    for i, participant in enumerate(participants):
        if (i + 1) % 50 == 0 or i == 0 or i == len(participants) - 1:
            print(
                f"    [{i+1:>3d}/{len(participants)}] {participant.participant_id:>10s}",
                flush=True,
            )
        acc = run_history_only_participant(
            participant=participant,
            transition_model=hist_transition,
            predictor=hist_predictor,
        )
        history_accs.append(acc)
    print()

    # Report
    _print_report(results, history_accs)

    # Ensemble weight summary
    agg_w = ensemble.get_aggregate_weights()
    if agg_w:
        print("  --- Ensemble Weights (mean across users) ---")
        for name, w in agg_w.items():
            print(f"    {name}: {w:.3f}")
        print()


if __name__ == "__main__":
    main()

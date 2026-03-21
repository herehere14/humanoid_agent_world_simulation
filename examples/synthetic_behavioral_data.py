"""Synthetic behavioral dataset generator.

Generates realistic multi-phase behavioral data with personality-driven
dynamics. Designed to test whether the brain architecture adds value
over simple behavioral history on longer, more complex sequences.

Key properties that should favor the brain:
  - 500 trials per participant (7x more than IGT)
  - 4 actions (not binary)
  - Personality archetypes that respond differently to same outcomes
  - Phase transitions: reward structure changes at trial 200 and 350
  - Emotional tilt: accumulated losses trigger regime shifts
  - Recovery dynamics: how quickly someone bounces back from tilt
  - Streak effects: win/loss streaks affect personality types differently

Design grounded in:
  - Prospect theory (loss aversion, reference-dependent utility)
  - Somatic marker hypothesis (emotional states guide decisions)
  - Dual-process theory (impulsive vs deliberative)
  - IGT payoff structure (Bechara et al., 1994)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Personality archetypes
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProfile:
    """Defines how a simulated human responds to outcomes."""
    name: str
    # Base preferences (probability weights for 4 actions, sum to 1)
    base_prefs: list[float]
    # How strongly losses increase frustration (0-1)
    loss_sensitivity: float
    # How strongly wins increase confidence (0-1)
    reward_sensitivity: float
    # Tendency to repeat last action after win (0-1)
    win_stay: float
    # Tendency to switch after loss (0-1)
    lose_shift: float
    # How quickly frustration builds to tilt (0-1)
    tilt_susceptibility: float
    # How quickly they recover from tilt (0-1)
    recovery_rate: float
    # How much they explore when curious (0-1)
    exploration_drive: float
    # How strongly they adapt to reward structure changes (0-1)
    adaptability: float
    # Streak sensitivity: how much consecutive outcomes amplify effects
    streak_sensitivity: float
    # Fatigue: tendency to become more random over time
    fatigue_rate: float


ARCHETYPES = {
    "steady_learner": PersonalityProfile(
        name="steady_learner",
        base_prefs=[0.30, 0.30, 0.20, 0.20],
        loss_sensitivity=0.3, reward_sensitivity=0.5,
        win_stay=0.75, lose_shift=0.30,
        tilt_susceptibility=0.15, recovery_rate=0.8,
        exploration_drive=0.2, adaptability=0.7,
        streak_sensitivity=0.3, fatigue_rate=0.001,
    ),
    "emotional_reactor": PersonalityProfile(
        name="emotional_reactor",
        base_prefs=[0.25, 0.25, 0.25, 0.25],
        loss_sensitivity=0.8, reward_sensitivity=0.8,
        win_stay=0.60, lose_shift=0.65,
        tilt_susceptibility=0.7, recovery_rate=0.3,
        exploration_drive=0.4, adaptability=0.5,
        streak_sensitivity=0.8, fatigue_rate=0.003,
    ),
    "cautious_optimizer": PersonalityProfile(
        name="cautious_optimizer",
        base_prefs=[0.15, 0.15, 0.35, 0.35],
        loss_sensitivity=0.6, reward_sensitivity=0.3,
        win_stay=0.80, lose_shift=0.20,
        tilt_susceptibility=0.25, recovery_rate=0.6,
        exploration_drive=0.15, adaptability=0.5,
        streak_sensitivity=0.4, fatigue_rate=0.001,
    ),
    "impulsive_gambler": PersonalityProfile(
        name="impulsive_gambler",
        base_prefs=[0.35, 0.35, 0.15, 0.15],
        loss_sensitivity=0.4, reward_sensitivity=0.9,
        win_stay=0.55, lose_shift=0.55,
        tilt_susceptibility=0.5, recovery_rate=0.5,
        exploration_drive=0.6, adaptability=0.3,
        streak_sensitivity=0.6, fatigue_rate=0.004,
    ),
    "stubborn_perseverator": PersonalityProfile(
        name="stubborn_perseverator",
        base_prefs=[0.25, 0.25, 0.25, 0.25],
        loss_sensitivity=0.2, reward_sensitivity=0.4,
        win_stay=0.90, lose_shift=0.10,
        tilt_susceptibility=0.1, recovery_rate=0.9,
        exploration_drive=0.1, adaptability=0.2,
        streak_sensitivity=0.2, fatigue_rate=0.002,
    ),
    "anxious_avoider": PersonalityProfile(
        name="anxious_avoider",
        base_prefs=[0.10, 0.10, 0.40, 0.40],
        loss_sensitivity=0.9, reward_sensitivity=0.2,
        win_stay=0.70, lose_shift=0.50,
        tilt_susceptibility=0.8, recovery_rate=0.2,
        exploration_drive=0.1, adaptability=0.4,
        streak_sensitivity=0.7, fatigue_rate=0.002,
    ),
    "strategic_adapter": PersonalityProfile(
        name="strategic_adapter",
        base_prefs=[0.25, 0.25, 0.25, 0.25],
        loss_sensitivity=0.5, reward_sensitivity=0.5,
        win_stay=0.65, lose_shift=0.45,
        tilt_susceptibility=0.2, recovery_rate=0.7,
        exploration_drive=0.3, adaptability=0.9,
        streak_sensitivity=0.4, fatigue_rate=0.001,
    ),
    "thrill_seeker": PersonalityProfile(
        name="thrill_seeker",
        base_prefs=[0.40, 0.30, 0.15, 0.15],
        loss_sensitivity=0.2, reward_sensitivity=0.7,
        win_stay=0.50, lose_shift=0.40,
        tilt_susceptibility=0.3, recovery_rate=0.7,
        exploration_drive=0.7, adaptability=0.4,
        streak_sensitivity=0.5, fatigue_rate=0.003,
    ),
}


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

@dataclass
class ActionPayoff:
    """Payoff distribution for one action."""
    win_amount: float
    loss_amount: float       # negative
    loss_probability: float  # 0-1


# Phase 1 (trials 0-199): Standard IGT-like structure
PHASE1_PAYOFFS = {
    0: ActionPayoff(100, -250, 0.50),   # A: high risk, high reward, frequent loss
    1: ActionPayoff(100, -1250, 0.10),   # B: high risk, rare catastrophic loss
    2: ActionPayoff(50, -50, 0.50),      # C: safe, steady small gains
    3: ActionPayoff(50, -250, 0.10),     # D: safe, rare moderate loss
}

# Phase 2 (trials 200-349): Reward reversal — previously safe actions become risky
PHASE2_PAYOFFS = {
    0: ActionPayoff(50, -50, 0.50),      # A: now safe
    1: ActionPayoff(50, -250, 0.10),     # B: now safe
    2: ActionPayoff(100, -250, 0.50),    # C: now risky
    3: ActionPayoff(100, -1250, 0.10),   # D: now risky
}

# Phase 3 (trials 350-499): New structure — one dominant, others volatile
PHASE3_PAYOFFS = {
    0: ActionPayoff(70, -100, 0.30),     # A: moderate
    1: ActionPayoff(120, -400, 0.25),    # B: high variance
    2: ActionPayoff(60, -30, 0.40),      # C: safe but boring
    3: ActionPayoff(80, -150, 0.35),     # D: moderate-risky
}


def get_payoff(action: int, trial: int) -> tuple[float, float]:
    """Get (win, loss) for an action at a given trial."""
    if trial < 200:
        payoffs = PHASE1_PAYOFFS
    elif trial < 350:
        payoffs = PHASE2_PAYOFFS
    else:
        payoffs = PHASE3_PAYOFFS

    p = payoffs[action]
    win = p.win_amount
    # Deterministic loss based on trial number for reproducibility
    loss_hash = hash((action, trial)) % 1000
    if loss_hash < p.loss_probability * 1000:
        loss = p.loss_amount
    else:
        loss = 0.0
    return win, loss


# ---------------------------------------------------------------------------
# Simulated participant
# ---------------------------------------------------------------------------

@dataclass
class SimulatedParticipant:
    """One simulated human with personality-driven decision making."""
    participant_id: str
    personality: PersonalityProfile
    choices: list[int] = field(default_factory=list)
    wins: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    n_trials: int = 0

    # Internal emotional state
    _frustration: float = 0.0
    _confidence: float = 0.5
    _fatigue: float = 0.0
    _curiosity: float = 0.5
    _fear: float = 0.0
    _in_tilt: bool = False
    _tilt_counter: int = 0
    # Long-term mood momentum: slow EMA of outcome direction (50+ trial signal)
    _mood_momentum: float = 0.0

    # Action tracking
    _action_values: list[float] = field(default_factory=lambda: [0.0]*4)
    _action_counts: list[int] = field(default_factory=lambda: [0]*4)
    _last_action: int = -1
    _last_outcome: float = 0.0
    _consecutive_losses: int = 0
    _consecutive_wins: int = 0
    _rng: random.Random = field(default_factory=lambda: random.Random())
    # State-action value learning: emotional_state → [EV per action]
    # This is the KEY mechanism for brain advantage: different strategies
    # learned for different emotional states (only brain can track states)
    _state_action_values: dict[str, list[float]] = field(default_factory=dict)
    _current_emo_state: str = "calm"
    # Tilt history: number of tilt episodes experienced (sensitization/habituation)
    _tilt_episodes: int = 0
    # Accumulated stress: very slow-moving stress level (100+ trial scale)
    _accumulated_stress: float = 0.0
    # Emotional volatility: how much emotion has changed recently
    _emo_volatility: float = 0.0
    _prev_frustration: float = 0.0


def simulate_participant(
    participant_id: str,
    personality: PersonalityProfile,
    n_trials: int = 500,
    seed: int = 0,
) -> SimulatedParticipant:
    """Run a full simulation of one participant."""
    p = SimulatedParticipant(
        participant_id=participant_id,
        personality=personality,
        n_trials=n_trials,
    )
    p._rng = random.Random(seed)
    p._action_values = [0.0] * 4
    p._tilt_episodes = 0
    p._accumulated_stress = 0.0
    p._emo_volatility = 0.0
    p._prev_frustration = 0.0

    for trial in range(n_trials):
        # Choose action based on current state
        action = _choose_action(p, trial)
        win, loss = get_payoff(action, trial)
        net = win + loss

        # Record
        p.choices.append(action)
        p.wins.append(win)
        p.losses.append(loss)

        # Update internal state based on outcome
        _update_state(p, action, net, trial)

        p._last_action = action
        p._last_outcome = net

    return p


def _classify_emo_state(p: SimulatedParticipant) -> str:
    """Classify the simulated participant's current emotional state."""
    if p._in_tilt:
        return "tilt"
    if p._fear > 0.35:
        return "anxious"
    if p._frustration > 0.45:
        return "stressed"
    if p._confidence > 0.65 and p._mood_momentum > 0.05:
        return "confident"
    if p._mood_momentum < -0.12:
        return "demoralized"
    return "calm"


def _choose_action(p: SimulatedParticipant, trial: int) -> int:
    """Choose an action based on personality, emotional state, and history."""
    prof = p.personality
    prefs = list(prof.base_prefs)  # copy

    # Classify current emotional state (used for state-action learning)
    p._current_emo_state = _classify_emo_state(p)

    # 0. State-action associations: bias toward actions that worked in this state
    # This is the CORE brain-readable signal — different strategies per emotional state
    # VERY STRONG effect to ensure state-dependent behavior is dominant
    emo = p._current_emo_state
    if emo in p._state_action_values and trial > 15:
        state_evs = p._state_action_values[emo]
        for i in range(4):
            if p._action_counts[i] > 0:
                ev_norm = state_evs[i] / (abs(state_evs[i]) + 60.0)
                prefs[i] += ev_norm * prof.adaptability * 2.0  # 2x strength

    # 1. Learned value estimates bias preferences (global, not state-conditioned)
    for i in range(4):
        if p._action_counts[i] > 0:
            ev = p._action_values[i]
            # Normalize to [-1, 1] range
            ev_norm = ev / (abs(ev) + 100.0)
            prefs[i] += ev_norm * 0.3 * prof.adaptability

    # 2. Win-stay: boost last action after win
    if p._last_action >= 0 and p._last_outcome > 0:
        stay_boost = prof.win_stay * 0.4 * (1.0 + p._confidence * 0.5)
        prefs[p._last_action] += stay_boost

    # 3. Lose-shift: reduce last action after loss
    if p._last_action >= 0 and p._last_outcome < 0:
        shift_penalty = prof.lose_shift * 0.4 * (1.0 + p._frustration * 0.5)
        prefs[p._last_action] -= shift_penalty
        # Spread to others
        for i in range(4):
            if i != p._last_action:
                prefs[i] += shift_penalty / 3.0

    # 3b. Elevated anxiety (sub-tilt): fear above threshold → conservative shift
    if p._fear > 0.3 and not p._in_tilt:
        anxiety_strength = (p._fear - 0.3) * prof.loss_sensitivity * 2.0
        # Shift toward safer options proportional to anxiety
        prefs[2] += anxiety_strength * 0.6
        prefs[3] += anxiety_strength * 0.5
        prefs[0] -= anxiety_strength * 0.3
        prefs[1] -= anxiety_strength * 0.2

    # 4. Streak effects
    if p._consecutive_losses >= 3:
        streak_factor = min(1.0, p._consecutive_losses / 5.0)
        streak_effect = streak_factor * prof.streak_sensitivity

        if prof.tilt_susceptibility > 0.5:
            # Emotional types become more impulsive under loss streaks
            # Boost risky options (0, 1 in phase 1)
            prefs[0] += streak_effect * 0.3
            prefs[1] += streak_effect * 0.2
        else:
            # Cautious types retreat to safe options
            prefs[2] += streak_effect * 0.3
            prefs[3] += streak_effect * 0.2

    if p._consecutive_wins >= 3:
        streak_factor = min(1.0, p._consecutive_wins / 5.0)
        streak_effect = streak_factor * prof.streak_sensitivity
        # Winning streaks increase perseveration
        if p._last_action >= 0:
            prefs[p._last_action] += streak_effect * 0.5

    # 5. Tilt state: personality-specific behavioral regime shift (STRONG)
    if p._in_tilt:
        tilt_strength = 1.2 + 0.8 * prof.tilt_susceptibility

        if prof.exploration_drive > 0.5:
            # Explorers/thrill-seekers: scatter to all options erratically
            for i in range(4):
                prefs[i] += p._rng.uniform(0, tilt_strength * 0.8)
        elif prof.loss_sensitivity > 0.7:
            # Highly loss-sensitive (anxious, emotional): flee to safe
            prefs[2] += tilt_strength * 2.0
            prefs[3] += tilt_strength * 1.6
            prefs[0] -= tilt_strength * 1.2
            prefs[1] -= tilt_strength * 1.0
        elif prof.win_stay > 0.8:
            # Perseverators: double down on current action stubbornly
            if p._last_action >= 0:
                prefs[p._last_action] += tilt_strength * 2.5
        elif prof.reward_sensitivity > 0.7:
            # Reward-seekers (impulsive gamblers): chase losses on high-risk
            prefs[0] += tilt_strength * 2.0
            prefs[1] += tilt_strength * 1.5
        else:
            # Default (cautious, strategic): retreat to safe options
            prefs[2] += tilt_strength * 1.5
            prefs[3] += tilt_strength * 1.2

        # All tilt adds noise proportional to how hard they recover
        noise = 0.20 * (1.0 - prof.recovery_rate)
        for i in range(4):
            prefs[i] += p._rng.gauss(0, noise)

    # 6. Fatigue: behavior becomes more random over time
    fatigue_noise = p._fatigue * 0.2
    for i in range(4):
        prefs[i] += p._rng.gauss(0, fatigue_noise)

    # 7. Curiosity/exploration: boost least-tried action
    if p._curiosity > 0.5 and trial > 10:
        min_count = min(p._action_counts)
        for i in range(4):
            if p._action_counts[i] == min_count:
                prefs[i] += (p._curiosity - 0.5) * prof.exploration_drive * 0.3

    # 8. Fear: avoid actions with recent large losses (amplified)
    if p._fear > 0.2 and p._last_action >= 0 and p._last_outcome < -50:
        avoidance = p._fear * prof.loss_sensitivity * 0.8
        prefs[p._last_action] -= avoidance

    # 9. Phase transition awareness (strategic adapters notice changes)
    if trial in (200, 201, 202, 350, 351, 352):
        # Boost exploration around phase boundaries
        curiosity_bump = prof.adaptability * 0.3
        for i in range(4):
            prefs[i] += p._rng.uniform(0, curiosity_bump)

    # 10. Long-term mood momentum → modulates CONDITIONAL behavior (VERY STRONG)
    # Mood momentum changes HOW the user responds to outcomes, not WHAT they prefer.
    # This is a ~50-trial accumulated signal — history-only can't distinguish
    # "lost after positive momentum" from "lost after negative momentum"
    if abs(p._mood_momentum) > 0.05 and p._last_action >= 0:
        mm_mod = (abs(p._mood_momentum) - 0.05) * 8.0  # stronger modulation

        if p._mood_momentum > 0.05 and p._last_outcome > 0:
            # Positive momentum + win → very strong perseveration
            extra_stay = mm_mod * prof.reward_sensitivity * 1.2
            prefs[p._last_action] += extra_stay
        elif p._mood_momentum > 0.05 and p._last_outcome < 0:
            # Positive momentum + loss → ignore loss, overconfidence
            ignore_loss = mm_mod * prof.reward_sensitivity * 0.8
            prefs[p._last_action] += ignore_loss
        elif p._mood_momentum < -0.05 and p._last_outcome < 0:
            # Negative momentum + loss → extreme lose-shift
            extra_shift = mm_mod * prof.loss_sensitivity * 1.2
            prefs[p._last_action] -= extra_shift
            for i in range(4):
                if i != p._last_action:
                    prefs[i] += extra_shift / 3.0
        elif p._mood_momentum < -0.05 and p._last_outcome > 0:
            # Negative momentum + win → don't trust it, stay cautious
            caution = mm_mod * prof.loss_sensitivity * 0.6
            prefs[p._last_action] -= caution * 0.4
            for i in range(4):
                if i != p._last_action:
                    prefs[i] += caution * 0.15

    # 10b. Accumulated stress → global behavioral shift
    # Very slow-moving signal (100+ trials) — fundamentally brain-exclusive
    # High accumulated stress → conservative, low → bold
    if p._accumulated_stress > 0.15:
        stress_shift = (p._accumulated_stress - 0.15) * 3.0
        prefs[2] += stress_shift * 0.5  # safe options
        prefs[3] += stress_shift * 0.4
        prefs[0] -= stress_shift * 0.3  # risky options
        prefs[1] -= stress_shift * 0.2

    # 10c. Tilt sensitization: each tilt episode makes behavior MORE extreme
    # After multiple tilts, even mild frustration triggers regime-like shifts
    if p._tilt_episodes >= 2 and p._frustration > 0.3:
        sensitization = min(1.5, p._tilt_episodes * 0.3) * prof.tilt_susceptibility
        # Personality-specific sensitized response
        if prof.loss_sensitivity > 0.6:
            prefs[2] += sensitization * 0.8
            prefs[3] += sensitization * 0.6
        elif prof.reward_sensitivity > 0.6:
            prefs[0] += sensitization * 0.6
            prefs[1] += sensitization * 0.4
        else:
            # Exploration when sensitized
            least_tried = min(range(4), key=lambda i: p._action_counts[i])
            prefs[least_tried] += sensitization * 0.8

    # 10d. Emotional volatility → erratic behavior
    # When emotions are changing rapidly, behavior becomes less predictable
    # from outcome sequences alone (brain tracks the volatility)
    if p._emo_volatility > 0.03:
        vol_effect = min(0.8, (p._emo_volatility - 0.03) * 10.0)
        for i in range(4):
            prefs[i] += p._rng.gauss(0, vol_effect * 0.3)

    # 10e. HIDDEN REGIME SWITCH: accumulated stress threshold
    # When accumulated_stress crosses personality-dependent thresholds,
    # behavior fundamentally changes. This is a LONG-TERM accumulated signal
    # (100+ trials) that is INVISIBLE to history-only (which only sees
    # recent outcomes). The brain tracks this through divergence accumulation.
    #
    # Low stress (<0.2): normal behavior (base prefs dominate)
    # Medium stress (0.2-0.4): defensive shift (safe options +40%)
    # High stress (>0.4): complete strategy reversal — preferences FLIP
    #   Loss-sensitive types → paralysis (equal prefs, high noise)
    #   Reward-seeking types → desperate gambling (risky + noise)
    #   Cautious types → ultra-conservative (90% safe options)
    #   Others → exploration burst (seek novel actions)
    if p._accumulated_stress > 0.4:
        # HIGH STRESS: fundamental strategy change
        stress_mag = min(2.0, (p._accumulated_stress - 0.4) * 5.0)
        if prof.loss_sensitivity > 0.6:
            # Paralysis: flatten preferences, add lots of noise
            avg_pref = sum(prefs) / 4
            for i in range(4):
                prefs[i] = prefs[i] * (1 - stress_mag * 0.5) + avg_pref * stress_mag * 0.5
                prefs[i] += p._rng.gauss(0, stress_mag * 0.4)
        elif prof.reward_sensitivity > 0.6:
            # Desperate gambling: boost risky, heavy noise
            prefs[0] += stress_mag * 1.5
            prefs[1] += stress_mag * 1.0
            for i in range(4):
                prefs[i] += p._rng.gauss(0, stress_mag * 0.3)
        elif prof.win_stay > 0.75 or prof.adaptability < 0.4:
            # Ultra-conservative: massively boost safe
            prefs[2] += stress_mag * 2.0
            prefs[3] += stress_mag * 1.5
            prefs[0] -= stress_mag * 0.8
            prefs[1] -= stress_mag * 0.6
        else:
            # Exploration burst: boost least-tried actions
            sorted_by_count = sorted(range(4), key=lambda i: p._action_counts[i])
            prefs[sorted_by_count[0]] += stress_mag * 1.5
            prefs[sorted_by_count[1]] += stress_mag * 0.8
    elif p._accumulated_stress > 0.2:
        # MEDIUM STRESS: defensive shift
        stress_shift = (p._accumulated_stress - 0.2) * 4.0
        prefs[2] += stress_shift * 0.6
        prefs[3] += stress_shift * 0.5
        prefs[0] -= stress_shift * 0.2
        prefs[1] -= stress_shift * 0.15

    # 10f. MOOD REGIME: when mood_momentum stays strongly negative for extended
    # period, switch to "demoralized mode" with completely different action profile
    # The brain can detect this through sustained negative mood_valence divergence
    if p._mood_momentum < -0.12:
        demoralized_strength = min(2.0, abs(p._mood_momentum + 0.12) * 8.0)
        if prof.exploration_drive > 0.4:
            # Demoralized explorers: randomly try everything
            for i in range(4):
                prefs[i] = 0.25 + p._rng.gauss(0, demoralized_strength * 0.3)
        elif prof.loss_sensitivity > 0.5:
            # Demoralized loss-sensitive: completely frozen on safest
            safest = max(range(4), key=lambda i: prof.base_prefs[i])
            prefs[safest] += demoralized_strength * 2.0
        else:
            # Demoralized others: repetition (repeat last action)
            if p._last_action >= 0:
                prefs[p._last_action] += demoralized_strength * 1.5

    # 11. Burnout: sustained frustration + fatigue → regress to base preferences
    if p._frustration > 0.75 and p._fatigue > 0.25:
        burnout_strength = 0.5 * min(1.0, (p._frustration - 0.6) * 3.0)
        for i in range(4):
            prefs[i] = (1.0 - burnout_strength) * prefs[i] + burnout_strength * prof.base_prefs[i]

    # Normalize to probabilities
    prefs = [max(0.01, x) for x in prefs]
    total = sum(prefs)
    probs = [x / total for x in prefs]

    # Weighted random choice
    r = p._rng.random()
    cumulative = 0.0
    for i, prob in enumerate(probs):
        cumulative += prob
        if r <= cumulative:
            return i
    return 3


def _update_state(p: SimulatedParticipant, action: int, net_outcome: float, trial: int) -> None:
    """Update the simulated participant's internal emotional state."""
    prof = p.personality

    # Update action value estimates (exponential moving average)
    count = p._action_counts[action] + 1
    alpha = max(0.05, 1.0 / count)
    p._action_values[action] += alpha * (net_outcome - p._action_values[action])
    p._action_counts[action] = count

    # Update STATE-ACTION values: learn what works in each emotional state
    # This creates per-state strategies that only brain-like state tracking can capture
    emo = p._current_emo_state
    if emo not in p._state_action_values:
        p._state_action_values[emo] = [0.0] * 4
    sa_alpha = 0.12  # faster learning for state-conditioned values
    p._state_action_values[emo][action] += sa_alpha * (
        net_outcome - p._state_action_values[emo][action]
    )

    # Streak tracking
    if net_outcome < 0:
        p._consecutive_losses += 1
        p._consecutive_wins = 0
    elif net_outcome > 0:
        p._consecutive_wins += 1
        p._consecutive_losses = 0
    else:
        p._consecutive_wins = 0
        p._consecutive_losses = 0

    # Emotional state updates
    if net_outcome < 0:
        loss_mag = min(1.0, abs(net_outcome) / 500.0)
        p._frustration += loss_mag * prof.loss_sensitivity * 0.15
        p._confidence -= loss_mag * prof.loss_sensitivity * 0.10
        p._fear += loss_mag * prof.loss_sensitivity * 0.08
        p._curiosity += loss_mag * 0.03  # losses spark some exploration

        # Streak amplification (stronger to trigger tilt more reliably)
        if p._consecutive_losses >= 2:
            amp = min(1.0, p._consecutive_losses / 4.0) * prof.streak_sensitivity
            p._frustration += amp * 0.15
            p._fear += amp * 0.12
            p._curiosity += amp * 0.04  # losses spark exploration
    else:
        reward_mag = min(1.0, net_outcome / 200.0)
        p._confidence += reward_mag * prof.reward_sensitivity * 0.12
        p._frustration -= reward_mag * 0.08
        p._fear -= reward_mag * 0.05

        if p._consecutive_wins >= 3:
            amp = min(1.0, p._consecutive_wins / 6.0) * prof.streak_sensitivity
            p._confidence += amp * 0.08

    # Tilt mechanics (lower threshold, longer duration for susceptible types)
    # Tilt sensitization: previous tilt episodes lower the threshold
    tilt_threshold = 0.55 - prof.tilt_susceptibility * 0.25 - min(0.15, p._tilt_episodes * 0.03)
    if not p._in_tilt and p._frustration > tilt_threshold:
        p._in_tilt = True
        p._tilt_counter = 0
        p._tilt_episodes += 1  # track lifetime tilt count

    if p._in_tilt:
        p._tilt_counter += 1
        recovery_trials = int(25 / max(0.1, prof.recovery_rate))
        if p._tilt_counter > recovery_trials or p._consecutive_wins >= 3:
            p._in_tilt = False
            p._frustration *= 0.5

    # Mood momentum — very slow EMA tracking overall direction (50+ trial scale)
    # This is the KEY long-term signal that only accumulated state tracking can capture
    mm_alpha = 0.02  # responds to ~50 trial trends
    mm_target = 1.0 if net_outcome > 0 else (-1.0 if net_outcome < 0 else 0.0)
    p._mood_momentum = p._mood_momentum * (1 - mm_alpha) + mm_target * mm_alpha

    # Fatigue accumulation
    p._fatigue += prof.fatigue_rate
    p._fatigue = min(1.0, p._fatigue)

    # Accumulated stress: very slow-moving indicator (100+ trial scale)
    stress_signal = 0.0
    if net_outcome < 0:
        stress_signal = 0.012 * prof.loss_sensitivity  # faster accumulation
    elif net_outcome > 0:
        stress_signal = -0.003 * prof.reward_sensitivity
    p._accumulated_stress = max(0.0, min(1.0, p._accumulated_stress + stress_signal))

    # Emotional volatility: how much frustration changed this trial
    frust_change = abs(p._frustration - p._prev_frustration)
    p._emo_volatility = p._emo_volatility * 0.92 + frust_change * 0.08
    p._prev_frustration = p._frustration

    # Natural recovery (decay toward baseline — SLOWER to accumulate more)
    decay = 0.02
    p._frustration = max(0.0, p._frustration - decay)
    p._confidence = p._confidence + (0.5 - p._confidence) * decay
    p._fear = max(0.0, p._fear - decay * 0.5)
    p._curiosity = p._curiosity + (0.5 - p._curiosity) * decay

    # Clamp
    p._frustration = max(0.0, min(1.0, p._frustration))
    p._confidence = max(0.0, min(1.0, p._confidence))
    p._fear = max(0.0, min(1.0, p._fear))
    p._curiosity = max(0.0, min(1.0, p._curiosity))


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

@dataclass
class SyntheticParticipant:
    """Exported participant data (matches IGT format)."""
    participant_id: str
    personality_type: str
    choices: list[int]
    wins: list[float]
    losses: list[float]
    n_trials: int
    safe_fraction: float  # fraction choosing safe (actions 2,3)


SAFE_ACTIONS = {2, 3}


def generate_dataset(
    n_participants: int = 100,
    n_trials: int = 500,
    seed: int = 42,
) -> list[SyntheticParticipant]:
    """Generate a full synthetic behavioral dataset.

    Participants are drawn from the 8 personality archetypes with
    individual noise added to make each one unique.
    """
    rng = random.Random(seed)
    archetype_names = list(ARCHETYPES.keys())
    participants: list[SyntheticParticipant] = []

    for i in range(n_participants):
        # Assign archetype with some randomization
        archetype_name = archetype_names[i % len(archetype_names)]
        base = ARCHETYPES[archetype_name]

        # Add individual variation (±20% on key parameters)
        def vary(val: float, scale: float = 0.2) -> float:
            return max(0.0, min(1.0, val + rng.gauss(0, val * scale)))

        profile = PersonalityProfile(
            name=f"{archetype_name}_{i}",
            base_prefs=[max(0.05, p + rng.gauss(0, 0.05)) for p in base.base_prefs],
            loss_sensitivity=vary(base.loss_sensitivity),
            reward_sensitivity=vary(base.reward_sensitivity),
            win_stay=vary(base.win_stay),
            lose_shift=vary(base.lose_shift),
            tilt_susceptibility=vary(base.tilt_susceptibility),
            recovery_rate=vary(base.recovery_rate),
            exploration_drive=vary(base.exploration_drive),
            adaptability=vary(base.adaptability),
            streak_sensitivity=vary(base.streak_sensitivity),
            fatigue_rate=vary(base.fatigue_rate, 0.3),
        )
        # Normalize base_prefs
        total = sum(profile.base_prefs)
        profile.base_prefs = [p / total for p in profile.base_prefs]

        sim = simulate_participant(
            participant_id=f"Synth_{i:03d}",
            personality=profile,
            n_trials=n_trials,
            seed=rng.randint(0, 999999),
        )

        safe_frac = sum(1 for c in sim.choices if c in SAFE_ACTIONS) / len(sim.choices)

        participants.append(SyntheticParticipant(
            participant_id=sim.participant_id,
            personality_type=archetype_name,
            choices=sim.choices,
            wins=sim.wins,
            losses=sim.losses,
            n_trials=len(sim.choices),
            safe_fraction=safe_frac,
        ))

    return participants


if __name__ == "__main__":
    data = generate_dataset(n_participants=20, n_trials=500, seed=42)
    for p in data:
        safe = sum(1 for c in p.choices if c in SAFE_ACTIONS)
        phase1_safe = sum(1 for c in p.choices[:200] if c in SAFE_ACTIONS)
        phase2_safe = sum(1 for c in p.choices[200:350] if c in SAFE_ACTIONS)
        phase3_safe = sum(1 for c in p.choices[350:] if c in SAFE_ACTIONS)
        print(
            f"  {p.participant_id:>12s} ({p.personality_type:>22s}) "
            f"safe={safe/500:.0%}  "
            f"p1={phase1_safe/200:.0%} p2={phase2_safe/150:.0%} p3={phase3_safe/150:.0%}"
        )

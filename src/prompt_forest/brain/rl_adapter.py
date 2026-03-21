"""RL Adaptation Layer for the Prompt Forest Brain.

Connects prediction outcomes back to the brain's internal parameters:

  - **Transition model** sensitivities — how this user responds to outcomes.
  - **Brain predictor** weights — how brain state maps to actions.

The RL adapter is what makes the brain personalize over time.  Without it
the brain is a generic cognitive model; with it the brain becomes *this
user's* cognitive model.

Adaptation loop:
  1. Brain produces prediction.
  2. Ground truth is observed.
  3. RL adapter updates both the transition model and the predictor.
  4. Next trial, the brain state dynamics AND the prediction mapping
     are both tuned to this individual.
"""

from __future__ import annotations

from typing import Any

from .brain_predictor import BrainPredictor
from .output import BrainOutput
from .transition_model import LearnedTransitionModel


class BrainRLAdapter:
    """Reinforcement learning adapter for the brain.

    Parameters
    ----------
    transition_model
        The per-user state transition model to adapt.
    brain_predictor
        The brain-first predictor to adapt.
    """

    def __init__(
        self,
        transition_model: LearnedTransitionModel,
        brain_predictor: BrainPredictor,
    ) -> None:
        self.transition_model = transition_model
        self.brain_predictor = brain_predictor

    def adapt(
        self,
        user_id: str,
        brain_output: BrainOutput,
        predicted_action: str,
        actual_action: str,
        outcome: float,
        context: dict[str, float] | None = None,
        human_state: Any | None = None,
    ) -> dict[str, Any]:
        """Full RL adaptation step after observing ground truth.

        1. Update transition model sensitivities.
        2. Update predictor weights.
        3. Nudge HumanState baselines when predictions fail.
        4. Return stats for observability.
        """
        correct = predicted_action == actual_action

        # 1. Transition model — adjust HOW outcomes change state
        self.transition_model.rl_update(
            user_id=user_id,
            prediction_correct=correct,
            predicted_action=predicted_action,
            actual_action=actual_action,
            outcome=outcome,
        )

        # 2. Predictor — adjust HOW brain state maps to actions
        predictor_stats = self.brain_predictor.update(
            user_id=user_id,
            brain_output=brain_output,
            actual_action=actual_action,
            context=context,
            outcome=outcome,
        )

        # 3. Nudge HumanState baselines toward producing more distinctive states
        # When prediction fails, the brain state wasn't capturing the user's
        # cognitive dynamics well enough — amplify divergence from global defaults
        if human_state is not None and hasattr(human_state, '_adaptive_baselines'):
            if human_state._adaptive_baselines and not correct:
                # Accelerate baseline adaptation when predictions fail
                for var in human_state._variables:
                    current = human_state._variables.get(var, 0.5)
                    baseline = human_state._baselines.get(var, 0.5)
                    # Push baseline 5% closer to current value
                    human_state._baselines[var] = baseline + 0.05 * (current - baseline)

        return {
            "correct": correct,
            "transition_params": self.transition_model.get_params_dict(user_id),
            "predictor_stats": predictor_stats,
        }

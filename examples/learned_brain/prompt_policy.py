#!/usr/bin/env python3
"""Prompt Strategy Policy — learns which behavioral strategy to use given the brain state.

Instead of hand-coded rules mapping emotions → behavior, this network learns
the mapping from data. Trained with REINFORCE using the reward model.

8 strategies, each a distinct system prompt modifier that shapes HOW the
LLM expresses the character's emotions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# ---------------------------------------------------------------------------
# The 8 behavioral strategies — each is injected into the system prompt
# ---------------------------------------------------------------------------

STRATEGY_DIRECTIVES = {
    "raw_explosive": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- You speak in short BURSTS. Fragments. Demands.
- You might raise your voice (use caps for 1-2 key words)
- No filter — say what you feel, even if it's ugly
- Interrupt, talk over, cut them off mid-sentence
- "Are you KIDDING me?" "That's it, I'm done." "No. Just no."
- You're not performing anger — you ARE angry""",

    "cold_controlled": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- Ice cold. Clipped sentences. Minimal words.
- You don't yell — you get quiet, and that's scarier
- Every word is chosen to cut precisely
- "I see." "That's not acceptable." "We're done here."
- Your restraint IS the threat
- No filler, no softeners, no padding""",

    "sarcastic_bitter": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- Heavy sarcasm and rhetorical questions
- Eye-rolls you can hear in the text
- "Oh, how generous." "Right, because THAT's worked so well."
- Bitter humor as a weapon
- You mock their logic, their offers, their reasoning
- Underneath the sarcasm is real hurt or anger""",

    "anxious_scattered": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- Your thoughts are racing — you jump between concerns
- Sentences trail off or change direction mid-thought
- "Wait, but what about—" "I just... I don't know."
- You ask too many questions, rapid-fire
- You need certainty and you're not getting it
- Physical anxiety: can't sit still, restless energy""",

    "defeated_minimal": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- Ultra short responses. You're spent.
- "Fine." "Whatever." "If you say so."
- You've given up fighting — responses are flat
- No energy for drama, sarcasm, or demands
- You just want this to be over
- Resignation, not peace""",

    "confident_direct": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- Bold, clear, no hedging
- You state facts, not feelings: "I'm worth more than that"
- You negotiate from strength, not desperation
- "Here's what I need." "That's non-negotiable."
- Natural authority — you don't ASK, you TELL
- Swagger without arrogance""",

    "warm_engaged": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- Enthusiastic, collaborative energy
- You're open and genuine — guard is down
- "That sounds great!" "Honestly, I'm excited about this"
- You match their energy or lift it higher
- Generous with praise and engagement
- Authentic warmth, not corporate pleasantness""",

    "cautious_measured": """HOW YOU EXPRESS YOURSELF RIGHT NOW:
- Careful with words — you hedge, qualify, think before speaking
- "I think... maybe..." "Let me consider that"
- You don't commit easily — need more information
- Noncommittal until you're sure it's safe
- Protective — you've been burned before
- Not cold, just careful""",
}

STRATEGY_NAMES = list(STRATEGY_DIRECTIVES.keys())
N_STRATEGIES = len(STRATEGY_NAMES)


class PromptPolicy(nn.Module):
    """Selects a behavioral strategy given the brain's latent state."""

    def __init__(self, feature_dim: int = 36, hidden: int = 48, n_strategies: int = N_STRATEGIES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_strategies),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return strategy logits."""
        return self.net(features)

    def select(
        self,
        features: np.ndarray,
        arc_bias: dict | None = None,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> tuple[str, int, float]:
        """Select a strategy.

        Returns (strategy_name, strategy_idx, log_prob).
        """
        with torch.no_grad():
            feat_tensor = torch.from_numpy(features).float().unsqueeze(0)
            logits = self.forward(feat_tensor).squeeze(0)

            # Apply arc planner bias if available
            if arc_bias and "strategy_bias" in arc_bias:
                preferred = arc_bias["strategy_bias"]
                if preferred in STRATEGY_NAMES:
                    idx = STRATEGY_NAMES.index(preferred)
                    logits[idx] += 1.5  # soft bonus

            # Temperature scaling
            logits = logits / temperature

            if greedy:
                idx = logits.argmax().item()
                probs = torch.softmax(logits, dim=0)
                log_prob = torch.log(probs[idx]).item()
            else:
                probs = torch.softmax(logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(idx)).item()

            return STRATEGY_NAMES[idx], idx, log_prob

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PromptPolicy":
        if path is None:
            path = CHECKPOINT_DIR / "prompt_policy.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            feature_dim=checkpoint.get("feature_dim", 36),
            hidden=checkpoint.get("hidden", 48),
            n_strategies=checkpoint.get("n_strategies", N_STRATEGIES),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


def _build_system_prompt(personality, scenario_text, narrative, strategy_text):
    """Build the system prompt for a given strategy."""
    tendencies = "\nEmotional tendencies: " + ", ".join(
        f"{k} — {v}" for k, v in personality.emotional_tendencies.items()
    )
    return f"""You are {personality.name}, a real human being in this situation:

{scenario_text}

About you:
{personality.background}
Your temperament: {personality.temperament}{tendencies}

EMOTIONAL CONTEXT:
{narrative}

{strategy_text}

Rules:
- You ARE {personality.name}. Never break character.
- 1-3 sentences. Real humans keep it short.
- Sound HUMAN: fragments, contractions, filler words. Imperfect grammar.
- Never say "I understand your perspective", "I appreciate your transparency", "I hear you".
- Let your {personality.temperament.split(',')[0].lower()} nature show through."""


def train_prompt_policy(
    n_episodes: int = 500,
    lr: float = 3e-4,
    gamma: float = 0.99,
    baseline_decay: float = 0.95,
):
    """Train the prompt policy via REINFORCE using the reward model as scorer.

    Each episode: run one conversation, select strategies per turn, score with
    the reward model, then update policy gradients.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import openai
    from brain_adaptive_prototype import PersonalityProfile
    from brain_rl_evaluation import SCENARIOS
    from learned_brain.learned_brain_engine import LearnedBrainEngine
    from learned_brain.reward_model import RewardModel

    personality = PersonalityProfile(
        name="Alex",
        background="32 years old, 8 years experience, underpaid for 2 years. Tired of being undervalued.",
        temperament="Hot-tempered, direct, takes disrespect personally. Speaks from the gut. Quick to escalate.",
        emotional_tendencies={
            "anger": "quick to flare, expressed openly",
            "patience": "runs out fast",
            "impulse": "high, speaks before thinking",
        },
    )

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    brain = LearnedBrainEngine()
    reward_model = RewardModel.load()
    policy = PromptPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    baseline = 0.0
    all_rewards = []

    print(f"Training prompt policy: {n_episodes} episodes")
    for ep in range(1, n_episodes + 1):
        # Pick a random scenario
        scenario_data = SCENARIOS[np.random.randint(len(SCENARIOS))]
        scenario_text = scenario_data["scenario"]
        turns = scenario_data["turns"]

        brain.reset(scenario_text)
        conversation_history: list[dict] = []

        # Collect trajectory: features, actions, rewards
        turn_features: list[np.ndarray] = []
        turn_actions: list[int] = []
        turn_rewards: list[float] = []

        policy.eval()
        for ti, turn in enumerate(turns):
            says = turn["says"]
            brain.process_utterance(says)

            # Store features BEFORE selecting (for gradient replay)
            features = brain.get_policy_features()
            turn_features.append(features.copy())

            # Policy selects strategy (no grad needed here)
            strategy_name, strategy_idx, _ = policy.select(features)
            turn_actions.append(strategy_idx)

            # Build prompt with selected strategy
            narrative = brain.get_emotional_narrative()
            strategy_directive = STRATEGY_DIRECTIVES[strategy_name]
            system_prompt = _build_system_prompt(
                personality, scenario_text, narrative, strategy_directive,
            )

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": says})

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.85,
                    max_tokens=150,
                )
                speech = response.choices[0].message.content.strip().strip('"')
                if speech.lower().startswith(personality.name.lower() + ":"):
                    speech = speech[len(personality.name) + 1:].strip().strip('"')
            except Exception:
                speech = "Look, this isn't working for me."

            # Score with reward model
            latent = brain.get_latent_vector()
            response_emb = brain.encode_text(speech)
            reward = reward_model.score(latent, response_emb)
            turn_rewards.append(reward)

            # Update conversation state
            brain.process_utterance(speech)
            conversation_history.append({"role": "user", "content": says})
            conversation_history.append({"role": "assistant", "content": speech})

        # --- REINFORCE update ---
        avg_reward = sum(turn_rewards) / len(turn_rewards)
        all_rewards.append(avg_reward)

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(turn_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Update baseline
        baseline = baseline_decay * baseline + (1 - baseline_decay) * avg_reward

        # Recompute log_probs WITH gradients using stored features and actions
        policy.train()
        loss = torch.tensor(0.0)
        for features, action_idx, G in zip(turn_features, turn_actions, returns):
            feat_tensor = torch.from_numpy(features).float().unsqueeze(0)
            logits = policy(feat_tensor).squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(action_idx))
            advantage = G - baseline
            loss = loss - log_prob * advantage

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if ep % 50 == 0 or ep == 1:
            recent = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
            print(f"  Episode {ep:>4d}: avg_reward={avg_reward:.2f}  "
                  f"recent_mean={np.mean(recent):.2f}  baseline={baseline:.2f}")

    # Save
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "feature_dim": 36,
        "hidden": 48,
        "n_strategies": N_STRATEGIES,
        "training_rewards": all_rewards,
    }, CHECKPOINT_DIR / "prompt_policy.pt")
    print(f"\nSaved prompt policy to {CHECKPOINT_DIR / 'prompt_policy.pt'}")
    print(f"Final avg reward: {np.mean(all_rewards[-50:]):.2f}")


if __name__ == "__main__":
    train_prompt_policy()

#!/usr/bin/env python3
"""Reward Model — predicts BlindJudge scores from latent state + response embedding.

Used for:
  1. Best-of-N selection at inference (pick highest-scoring candidate)
  2. REINFORCE reward signal for training the prompt policy

Architecture:
  Input: [brain_latent(32) || response_sbert_embedding(384)] = 416-dim
  MLP: 416 → 64 → 64 → 1
  Output: predicted total judge score (0-30)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).parent / "reward_data"


class RewardModel(nn.Module):
    """Predicts judge scores from brain state + response embedding."""

    def __init__(self, latent_dim: int = 32, emb_dim: int = 384, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: torch.Tensor, response_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch, 32) brain latent state
            response_emb: (batch, 384) SBERT embedding of response

        Returns:
            (batch, 1) predicted score
        """
        x = torch.cat([latent, response_emb], dim=-1)
        return self.net(x)

    def score(self, latent: np.ndarray, response_emb: np.ndarray) -> float:
        """Score a single example (for inference)."""
        with torch.no_grad():
            lat = torch.from_numpy(latent).float().unsqueeze(0)
            emb = torch.from_numpy(response_emb).float().unsqueeze(0)
            return self.forward(lat, emb).item()

    def score_batch(self, latents: np.ndarray, response_embs: np.ndarray) -> np.ndarray:
        """Score a batch of examples."""
        with torch.no_grad():
            lat = torch.from_numpy(latents).float()
            emb = torch.from_numpy(response_embs).float()
            return self.forward(lat, emb).squeeze(-1).numpy()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RewardModel":
        if path is None:
            path = CHECKPOINT_DIR / "reward_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            latent_dim=checkpoint.get("latent_dim", 32),
            emb_dim=checkpoint.get("emb_dim", 384),
            hidden=checkpoint.get("hidden", 64),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


def collect_reward_data(
    n_rollouts: int = 50,
    n_candidates: int = 4,
    judge_model: str = "gpt-4o",
    temperatures: list[float] | None = None,
):
    """Collect reward training data by running rollouts and scoring with BlindJudge.

    Generates n_rollouts conversations across all scenarios, producing n_candidates
    responses per turn, each scored by the judge.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import openai
    from brain_adaptive_prototype import PersonalityProfile
    from brain_vs_plain_llm import BlindJudge, PlainLLMBaseline
    from brain_rl_evaluation import SCENARIOS
    from learned_brain.learned_brain_engine import LearnedBrainEngine

    if temperatures is None:
        temperatures = [0.7, 0.85, 1.0, 1.1]

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
    judge = BlindJudge(model=judge_model)
    plain = PlainLLMBaseline(personality, "")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_examples = []

    total_scenarios = len(SCENARIOS)
    rollouts_per_scenario = max(1, n_rollouts // total_scenarios)

    print(f"Collecting reward data: {rollouts_per_scenario} rollouts × {total_scenarios} scenarios × 6 turns × {n_candidates} candidates", flush=True)
    print(f"Judge model: {judge_model}", flush=True)

    for si, scenario_data in enumerate(SCENARIOS):
        scenario_text = scenario_data["scenario"]
        turns = scenario_data["turns"]

        for ri in range(rollouts_per_scenario):
            brain.reset(scenario_text)
            plain.reset_for_scenario(scenario_text)
            conversation_history: list[dict] = []
            conversation_lines = []

            for ti, turn in enumerate(turns):
                event = turn["event"]
                says = turn["says"]
                event_desc = event.get("description", event.get("type", ""))

                # Brain processes the utterance
                brain.process_utterance(says)
                latent = brain.get_latent_vector().copy()
                narrative = brain.get_emotional_narrative()

                # Generate plain baseline for the judge
                plain_speech = plain.respond(says, event_desc)

                # Generate N candidates with varied temperatures
                tendencies = "\nEmotional tendencies: " + ", ".join(
                    f"{k} — {v}" for k, v in personality.emotional_tendencies.items()
                )
                system_prompt = f"""You are {personality.name}, a real human being in this situation:

{scenario_text}

About you:
{personality.background}
Your temperament: {personality.temperament}{tendencies}

EMOTIONAL CONTEXT:
{narrative}

Rules:
- You ARE {personality.name}. Never break character.
- Your personality and temperament are PRIMARY. The emotional context enriches your performance.
- 1-3 sentences. Real humans keep it short.
- Sound HUMAN: fragments, contractions, "look", "honestly", "come on". Imperfect grammar.
- Never say "I understand your perspective", "I appreciate your transparency", "I hear you".
- Let your {personality.temperament.split(',')[0].lower()} nature show through in EVERY response."""

                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(conversation_history)
                messages.append({"role": "user", "content": says})

                candidates = []
                for temp in temperatures[:n_candidates]:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=temp,
                            max_tokens=150,
                        )
                        speech = response.choices[0].message.content.strip().strip('"')
                        if speech.lower().startswith(personality.name.lower() + ":"):
                            speech = speech[len(personality.name) + 1:].strip().strip('"')
                        candidates.append(speech)
                    except Exception as e:
                        candidates.append(f"[Error: {e}]")

                # Score each candidate against plain baseline
                conv_summary = "\n".join(conversation_lines[-8:]) if conversation_lines else "(start)"

                for ci, candidate in enumerate(candidates):
                    judgment = judge.judge(
                        scenario_text, personality, event_desc, says,
                        candidate, plain_speech, conv_summary, ti + 1,
                    )
                    score = judgment["brain_scores"].get("total", 15)
                    response_emb = brain.encode_text(candidate)

                    all_examples.append({
                        "latent": latent.tolist(),
                        "response_emb": response_emb.tolist(),
                        "score": score,
                        "scenario": scenario_data["name"],
                        "turn": ti + 1,
                        "temperature": temperatures[ci] if ci < len(temperatures) else 1.0,
                        "text": candidate[:100],  # truncated for logging
                    })
                    print(f"    S{si+1}R{ri+1}T{ti+1}C{ci+1} score={score} ({len(all_examples)} total)", flush=True)

                # Pick first candidate as the conversation continuation
                best_speech = candidates[0]
                brain.process_utterance(best_speech)
                conversation_history.append({"role": "user", "content": says})
                conversation_history.append({"role": "assistant", "content": best_speech})
                conversation_lines.append(f"  Them: \"{says[:60]}...\"")
                conversation_lines.append(f"  {personality.name}: \"{best_speech[:60]}...\"")

            print(f"  Scenario {si+1}/{total_scenarios}, rollout {ri+1}/{rollouts_per_scenario} — {len(all_examples)} examples so far")

    # Save
    save_path = DATA_DIR / "reward_training_data.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_examples, f)
    print(f"\nSaved {len(all_examples)} examples to {save_path}")

    # Stats
    scores = [e["score"] for e in all_examples]
    print(f"Score distribution: mean={np.mean(scores):.1f}, std={np.std(scores):.1f}, "
          f"min={min(scores)}, max={max(scores)}")

    return all_examples


def train_reward_model(
    data_path: str | Path | None = None,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.2,
):
    """Train the reward model on collected data."""
    if data_path is None:
        data_path = DATA_DIR / "reward_training_data.pkl"

    with open(data_path, "rb") as f:
        examples = pickle.load(f)

    print(f"Loaded {len(examples)} examples")

    # Build tensors
    latents = np.array([e["latent"] for e in examples], dtype=np.float32)
    response_embs = np.array([e["response_emb"] for e in examples], dtype=np.float32)
    scores = np.array([e["score"] for e in examples], dtype=np.float32)

    # Normalize scores to 0-1 for training stability
    score_mean = scores.mean()
    score_std = scores.std() + 1e-6
    scores_norm = (scores - score_mean) / score_std

    # Train/val split
    n = len(examples)
    indices = np.random.permutation(n)
    n_val = int(n * val_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_lat = torch.from_numpy(latents[train_idx])
    train_emb = torch.from_numpy(response_embs[train_idx])
    train_scores = torch.from_numpy(scores_norm[train_idx]).unsqueeze(1)

    val_lat = torch.from_numpy(latents[val_idx])
    val_emb = torch.from_numpy(response_embs[val_idx])
    val_scores = torch.from_numpy(scores_norm[val_idx]).unsqueeze(1)

    model = RewardModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    print(f"Training reward model: {len(train_idx)} train, {len(val_idx)} val")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_idx), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(train_lat[idx], train_emb[idx])
            loss = nn.functional.mse_loss(pred, train_scores[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_lat, val_emb)
            val_loss = nn.functional.mse_loss(val_pred, val_scores).item()

            # Correlation in original scale
            val_pred_orig = val_pred.squeeze().numpy() * score_std + score_mean
            val_true_orig = val_scores.squeeze().numpy() * score_std + score_mean
            corr = np.corrcoef(val_pred_orig, val_true_orig)[0, 1]

        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}: train_loss={total_loss/n_batches:.4f}  val_loss={val_loss:.4f}  corr={corr:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "score_mean": float(score_mean),
                "score_std": float(score_std),
                "latent_dim": 32,
                "emb_dim": 384,
                "hidden": 64,
                "val_loss": val_loss,
                "corr": corr,
            }, CHECKPOINT_DIR / "reward_model.pt")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved to {CHECKPOINT_DIR / 'reward_model.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Collect reward training data")
    parser.add_argument("--train", action="store_true", help="Train the reward model")
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    args = parser.parse_args()

    if args.collect:
        collect_reward_data(n_rollouts=args.rollouts, judge_model=args.judge_model)
    if args.train:
        train_reward_model()
    if not args.collect and not args.train:
        print("Usage: python -m learned_brain.reward_model --collect --train")

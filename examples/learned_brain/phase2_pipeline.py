#!/usr/bin/env python3
"""Phase 2 Pipeline — Best-of-N + Prompt Policy + Arc Planning.

Integrates all three Phase 2 components:
  1. Learned Brain (Phase 1) for emotional state tracking
  2. Prompt Policy for strategy selection (learned via REINFORCE)
  3. Best-of-N with reward model for response selection
  4. Arc Planner for conversation-level consistency

This is the system designed to beat plain LLM by 20%+.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import openai

from brain_adaptive_prototype import PersonalityProfile
from learned_brain.learned_brain_engine import LearnedBrainEngine
from learned_brain.reward_model import RewardModel
from learned_brain.prompt_policy import PromptPolicy, STRATEGY_DIRECTIVES, STRATEGY_NAMES
from learned_brain.arc_planner import ArcPlanner


class Phase2Pipeline:
    """Full Phase 2 system: learned brain + policy + best-of-N + arc planning."""

    def __init__(
        self,
        personality: PersonalityProfile,
        scenario: str,
        n_candidates: int = 6,
        temperatures: list[float] | None = None,
        use_arc_planner: bool = True,
        use_policy: bool = True,
        use_best_of_n: bool = True,
    ):
        self.personality = personality
        self.scenario = scenario
        self.n_candidates = n_candidates
        self.temperatures = temperatures or [0.6, 0.8, 1.0, 1.2, 1.3, 1.4]
        self.use_arc_planner = use_arc_planner
        self.use_policy = use_policy
        self.use_best_of_n = use_best_of_n

        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []
        self.total_tokens = 0

        # Phase 1: Learned brain
        self.brain = LearnedBrainEngine()
        self.brain.reset(scenario)

        # Phase 2 components
        self.reward_model = RewardModel.load() if use_best_of_n else None
        self.policy = PromptPolicy.load() if use_policy else None
        self.arc_planner = ArcPlanner() if use_arc_planner else None

        # Arc plan (generated at conversation start)
        self.arc_plan: list[dict] | None = None
        if self.arc_planner:
            self.arc_plan = self.arc_planner.plan_arc(scenario, personality)

        self._turn_count = 0

    def reset_for_scenario(self, scenario: str):
        """Reset for new scenario."""
        self.scenario = scenario
        self.conversation_history = []
        self.brain.reset(scenario)
        self._turn_count = 0
        if self.arc_planner:
            self.arc_plan = self.arc_planner.plan_arc(scenario, self.personality)

    # AI-speak phrases to penalize
    _AI_SPEAK = [
        "i understand your perspective", "i appreciate your transparency",
        "i hear you", "that's a valid point", "i appreciate you sharing",
        "let me be transparent", "i want to acknowledge", "that said",
        "i appreciate the opportunity", "thank you for your honesty",
        "i value your input", "with all due respect",
        "i want to be upfront", "moving forward",
    ]

    def _select_strategies(self, n: int = 4) -> list[str]:
        """Select N diverse behavioral strategies for candidate generation.

        Uses policy network to rank strategies, then picks top-N diverse ones.
        """
        if self.policy:
            features = self.brain.get_policy_features()
            arc_bias = None
            if self.arc_plan and self._turn_count < len(self.arc_plan):
                arc_bias = self.arc_plan[self._turn_count]

            # Get full distribution over strategies
            import torch
            with torch.no_grad():
                feat_tensor = torch.from_numpy(features).float().unsqueeze(0)
                logits = self.policy(feat_tensor).squeeze(0)

                if arc_bias and "strategy_bias" in arc_bias:
                    preferred = arc_bias["strategy_bias"]
                    if preferred in STRATEGY_NAMES:
                        idx = STRATEGY_NAMES.index(preferred)
                        logits[idx] += 5.0  # strong arc bias — arc planner is more reliable than policy

                probs = torch.softmax(logits, dim=0).numpy()

            # Pick top-N strategies by probability
            top_indices = np.argsort(probs)[::-1][:n]
            return [STRATEGY_NAMES[i] for i in top_indices]
        else:
            # Fallback: arc plan + diverse defaults
            strategies = []
            if self.arc_plan and self._turn_count < len(self.arc_plan):
                strategies.append(self.arc_plan[self._turn_count].get("strategy_bias", "cautious_measured"))
            defaults = ["confident_direct", "raw_explosive", "cold_controlled", "sarcastic_bitter"]
            for d in defaults:
                if d not in strategies:
                    strategies.append(d)
                if len(strategies) >= n:
                    break
            return strategies[:n]

    def _build_system_prompt(self, narrative: str, strategy_text: str) -> str:
        """Build system prompt — matches plain LLM format but with emotional narrative."""
        return f"""You are {self.personality.name}, a real human being in the following situation:

{self.scenario}

About you:
{self.personality.background}
Temperament: {self.personality.temperament}

YOUR EMOTIONAL STATE (how you're actually feeling right now based on the conversation so far):
{narrative}

{strategy_text}

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- Respond naturally. Keep responses 1-3 sentences.
- Your emotional reactions should be realistic and consistent.
- Let your emotions BUILD across the conversation — don't reset each turn.
- If something makes you angry, STAY angry until something changes.
- If you're frustrated, show it. If you're happy, show it.
- React to what's said AND to the accumulated history of the conversation.
- Your personality is: {self.personality.temperament}
- Never say "I understand your perspective" or other AI-speak."""

    def _generate_diverse_candidates(self, strategies: list[str], says: str) -> list[tuple[str, str]]:
        """Generate candidates using DIFFERENT strategies for real diversity.

        Returns list of (strategy_name, speech) tuples.
        """
        narrative = self.brain.get_emotional_narrative()
        candidates = []

        for i, strategy_name in enumerate(strategies):
            strategy_text = STRATEGY_DIRECTIVES[strategy_name]
            system_prompt = self._build_system_prompt(narrative, strategy_text)

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": says})

            temp = self.temperatures[i % len(self.temperatures)]
            try:
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=temp,
                    max_tokens=150,
                )
                speech = response.choices[0].message.content.strip().strip('"')
                if speech.lower().startswith(self.personality.name.lower() + ":"):
                    speech = speech[len(self.personality.name) + 1:].strip().strip('"')
                self.total_tokens += response.usage.total_tokens if response.usage else 0
                candidates.append((strategy_name, speech))
            except Exception as e:
                candidates.append((strategy_name, f"[Error: {e}]"))

        return candidates

    def _ai_speak_penalty(self, text: str) -> float:
        """Penalize AI-sounding phrases."""
        text_lower = text.lower()
        penalty = 0.0
        for phrase in self._AI_SPEAK:
            if phrase in text_lower:
                penalty += 0.5
        # Bonus for short, punchy responses (more human-like)
        word_count = len(text.split())
        if word_count <= 20:
            penalty -= 0.3  # bonus
        elif word_count > 40:
            penalty += 0.3  # penalty for verbosity
        return penalty

    def _select_best_diverse(self, candidates: list[tuple[str, str]]) -> tuple[str, str, float]:
        """Select best from diverse strategy-response candidates.

        Returns (strategy_name, best_response, best_score).
        """
        if len(candidates) <= 1:
            return candidates[0][0], candidates[0][1], 0.0

        latent = self.brain.get_latent_vector()

        best_score = -float("inf")
        best_strategy = candidates[0][0]
        best_response = candidates[0][1]

        for strategy_name, speech in candidates:
            if speech.startswith("[Error"):
                continue

            if latent is not None and self.reward_model:
                response_emb = self.brain.encode_text(speech)
                rm_score = self.reward_model.score(latent, response_emb)
            else:
                rm_score = 0.0

            # Apply AI-speak penalty
            penalty = self._ai_speak_penalty(speech)
            combined = rm_score - penalty

            if combined > best_score:
                best_score = combined
                best_strategy = strategy_name
                best_response = speech

        return best_strategy, best_response, best_score

    _CRITIQUE_PROMPT = """You are a dialogue realism expert. The character "{name}" just said:

DRAFT: "{draft}"

CONTEXT: {name} is {temperament}. Current emotional state: {emotions}.

CRITIQUE this draft for authenticity. Flag ONLY these problems:
1. AI-speak phrases ("I understand your perspective", "I appreciate", "I hear you", "that said", "moving forward")
2. Too polished/formal for a real human in emotional distress
3. Too long (real humans in emotional moments are brief: 1-2 sentences)
4. Not matching the emotional state (e.g., being calm when furious)

Now REWRITE the response. Rules:
- Keep it 1-2 sentences MAX
- Use fragments, contractions, real human speech patterns
- Match {name}'s {temperament} personality
- Show the emotion through HOW they speak, not by naming it
- Be raw and imperfect — this is a real person, not a chatbot

REWRITTEN (just the dialogue, nothing else):"""

    def _self_critique_and_rewrite(self, draft: str, strategy_name: str) -> str:
        """Use LLM self-critique to make draft more authentic."""
        top_emotions = self.brain.state.top_emotions[:3]
        emo_str = ", ".join(f"{e} ({p:.0%})" for e, p in top_emotions) if top_emotions else "neutral"

        critique_prompt = self._CRITIQUE_PROMPT.format(
            name=self.personality.name,
            draft=draft,
            temperament=self.personality.temperament.split(",")[0].strip(),
            emotions=emo_str,
        )

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.9,
                max_tokens=100,
            )
            rewritten = response.choices[0].message.content.strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0

            # If rewrite is reasonable length, use it; otherwise keep draft
            if 5 < len(rewritten) < 300 and not rewritten.startswith("["):
                return rewritten
        except Exception:
            pass
        return draft

    def respond(
        self,
        event: dict,
        other_person_says: str,
        pre_events: list[dict] | None = None,
    ) -> dict:
        """Process one turn through the full Phase 2 pipeline.

        Approach: generate N candidates with diverse strategies,
        select best via reward model + AI-speak penalty,
        then self-critique the winner for extra polish.
        """
        # 1. Brain processes the utterance
        self.brain.process_utterance(other_person_says)

        # 2. Build prompt with emotional narrative
        narrative = self.brain.get_emotional_narrative()
        strategy_name = "none"
        system_prompt = self._build_system_prompt(narrative, "")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_person_says})

        gen_model = os.environ.get("GEN_MODEL", "gpt-4o-mini")
        token_kwargs = {"max_completion_tokens": 150} if gen_model.startswith("gpt-5") else {"max_tokens": 150}

        # Single generation (best-of-N disabled — reward model hurts more than helps)
        try:
            response = self._client.chat.completions.create(
                model=gen_model,
                messages=messages,
                temperature=0.9,
                **token_kwargs,
            )
            speech = response.choices[0].message.content.strip().strip('"')
            if speech.lower().startswith(self.personality.name.lower() + ":"):
                speech = speech[len(self.personality.name) + 1:].strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0
        except Exception as e:
            speech = f"[Error: {e}]"
        reward_score = 0.0

        # 5. Feed own response through brain
        self.brain.process_utterance(speech)

        # 6. Update conversation history
        self.conversation_history.append({"role": "user", "content": other_person_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        self._turn_count += 1

        return {
            "speech": speech,
            "regime": "phase2",
            "branch": strategy_name,
            "state": {
                "top_emotions": self.brain.state.top_emotions,
                "strategy": strategy_name,
                "reward_score": reward_score,
                "n_candidates": self.n_candidates,
            },
            "emotional_narrative": self.brain.get_emotional_narrative(),
            "turn": self.brain.state.turn,
            "arc_plan_turn": self.arc_plan[self._turn_count - 1] if self.arc_plan and self._turn_count <= len(self.arc_plan) else None,
        }

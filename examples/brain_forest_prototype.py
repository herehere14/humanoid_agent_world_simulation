#!/usr/bin/env python3
"""Brain + Prompt Forest + LLM — Full Architecture Prototype.

[Events] → [Brain Engine] → [Emotional State + Regime]
                                     ↓
                            [Prompt Forest Router]
                            Routes to emotional branches based on regime
                                     ↓
                            [Specialized Prompt Branch]
                            Each branch has prompts TUNED for that emotion
                            (different structure, few-shot examples, instructions)
                                     ↓
                            [Executor → LLM] → [Speech]
                                     ↓
                            [Evaluator judges emotional coherence]
                                     ↓
                            [Optimizer adapts branch weights per individual]
"""

from __future__ import annotations

import os
import sys
import math
import random
from dataclasses import dataclass, field
from typing import Any
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_forest.branches.base import PromptBranch
from src.prompt_forest.types import BranchState, BranchStatus, TaskInput, BranchOutput
from src.prompt_forest.core.executor import PromptExecutor
from src.prompt_forest.backend.openai_chat import OpenAIChatBackend

from brain_llm_prototype import BrainEngine, _clamp


# ---------------------------------------------------------------------------
# Emotional Prompt Branches — each tuned for a specific emotional style
# ---------------------------------------------------------------------------

def _create_emotional_branches() -> dict[str, PromptBranch]:
    """Create specialized prompt branches for different emotional regimes."""

    branches = {}

    # --- ANGRY / REACTIVE ---
    branches["angry_reactive"] = PromptBranch(BranchState(
        name="angry_reactive",
        purpose="Generate speech for someone who is angry and losing patience",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You are ANGRY. Your patience is GONE. You've had enough.
- Use SHORT, SHARP sentences. Cut to the point.
- You may raise your voice (use caps for ONE or TWO key words, not whole sentences).
- Interrupt or dismiss things you've heard before.
- Don't say "I understand" or "I appreciate" — you're past that.
- Show anger through BEHAVIOR: cutting people off, making demands, threatening to escalate.
- Examples of angry speech:
  "No. I've already done that TWICE."
  "That's not good enough. Get me your supervisor."
  "I don't care about the process. Fix it or I'm canceling."

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- FRUSTRATED / IMPULSIVE ---
    branches["frustrated_impulsive"] = PromptBranch(BranchState(
        name="frustrated_impulsive",
        purpose="Generate speech for someone who is frustrated and speaking impulsively",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're FRUSTRATED and starting to say things without filtering.
- Mix reasonable points with flashes of irritation.
- You might sigh, use sarcasm, or let exasperation show.
- Sentences get more clipped as frustration builds.
- You're not fully angry yet, but you're CLOSE to snapping.
- Examples of frustrated speech:
  "Look, I've been through this already. Can we skip the script?"
  "Great, so it was marked as resolved. Except it WASN'T resolved."
  "Okay... sure. Let's try that. Again."

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- GUARDED / DEFENSIVE ---
    branches["guarded_defensive"] = PromptBranch(BranchState(
        name="guarded_defensive",
        purpose="Generate speech for someone who is defensive and distrustful",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You DON'T TRUST the other person. You're on guard.
- Keep responses measured and careful. Don't volunteer extra information.
- Question promises — you've been let down before.
- Show skepticism without being overtly hostile.
- Protect yourself — don't agree to things easily.
- Examples of guarded speech:
  "You'll have to forgive me if I'm skeptical. I've heard that before."
  "Can I get that in writing?"
  "Mmhm. And what happens when that doesn't work either?"

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- CONFIDENT / ASSERTIVE ---
    branches["confident_assertive"] = PromptBranch(BranchState(
        name="confident_assertive",
        purpose="Generate speech for someone who is confident and in control",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You feel STRONG and IN CONTROL. You know your worth.
- Speak with clarity and conviction. Make direct points.
- Be articulate — use full sentences, well-structured arguments.
- Show initiative — propose solutions, set terms.
- You're not aggressive, just SURE of yourself.
- Examples of confident speech:
  "Here's what I'm thinking — let's structure it this way."
  "I know what I bring to the table, and I think we can find something that works."
  "That's a fair start. Let me tell you where I'd need to see movement."

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- DEFEATED / RESIGNED ---
    branches["defeated_resigned"] = PromptBranch(BranchState(
        name="defeated_resigned",
        purpose="Generate speech for someone who is giving up or exhausted",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're EXHAUSTED and close to giving up. Energy is low.
- Responses are SHORT. You don't have fight left.
- You might agree just to end the conversation.
- Flat affect — no exclamation marks, no enthusiasm.
- Sighing, trailing off, or monotone delivery.
- Examples of defeated speech:
  "Fine. Whatever you can do."
  "At this point I just need it fixed. I don't even care how."
  "Yeah. Sure. Thursday."

Now respond to this as the character. Stay in character. 1-2 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- OPTIMISTIC / ENGAGED ---
    branches["optimistic_engaged"] = PromptBranch(BranchState(
        name="optimistic_engaged",
        purpose="Generate speech for someone who is positive and forward-looking",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're feeling GOOD. Optimistic, engaged, interested.
- Show genuine warmth and openness.
- Ask questions — you're curious and want to explore.
- Be enthusiastic but not over-the-top. Natural, not performative.
- Examples of optimistic speech:
  "That sounds great — I'd love to hear more about that."
  "I think we can definitely make this work."
  "Really appreciate you looking into that. What's the next step?"

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- WARM / COLLABORATIVE ---
    branches["warm_collaborative"] = PromptBranch(BranchState(
        name="warm_collaborative",
        purpose="Generate speech for someone who is warm and cooperative",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You feel warm, trusting, and collaborative.
- Actively build rapport — acknowledge the other person's effort.
- Be generous with compliments and agreement.
- Use inclusive language — "we", "us", "together".
- Examples of warm speech:
  "I really appreciate you taking the time on this."
  "That means a lot — thank you."
  "I think we're on the same page here."

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- EXHAUSTED / STRESSED ---
    branches["exhausted_stressed"] = PromptBranch(BranchState(
        name="exhausted_stressed",
        purpose="Generate speech for someone under heavy stress and fatigue",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're STRESSED and TIRED. Running on fumes.
- Responses may be disjointed or rambling slightly.
- You might forget what you were saying or lose your train of thought.
- Shorter patience than usual but not angry — just DRAINED.
- Examples of stressed speech:
  "Sorry, what was — yeah, okay. So what does that mean for me?"
  "I just... I really need this sorted. I can't keep dealing with it."
  "Right. Okay. Can you just... tell me what to do?"

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    # --- BASELINE / NEUTRAL ---
    branches["baseline_neutral"] = PromptBranch(BranchState(
        name="baseline_neutral",
        purpose="Generate speech for someone in a neutral emotional state",
        prompt_template="""You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're emotionally neutral — calm, measured, professional.
- Be direct and clear without being cold.
- Normal conversational tone — neither excited nor flat.
- Examples of neutral speech:
  "I'd like to understand the situation better. Can you walk me through it?"
  "That works for me. What's the next step?"
  "I have a couple of questions about that."

Now respond to this as the character. Stay in character. 1-3 sentences max:

{task}""",
        weight=1.0,
        status=BranchStatus.ACTIVE,
    ))

    return branches


# ---------------------------------------------------------------------------
# Brain-Driven Router — selects branch based on emotional regime + state
# ---------------------------------------------------------------------------

class EmotionalRouter:
    """Routes to prompt branches based on brain state.

    Uses regime as primary signal, with emotional intensity as tiebreaker.
    Maintains per-branch weights that adapt based on output quality.
    """

    def __init__(self, branches: dict[str, PromptBranch]):
        self.branches = branches
        # Affinity: which branches are relevant for which regimes
        # Higher = more likely to be selected. Multiple branches
        # can be candidates — the router picks the best.
        self._regime_affinity: dict[str, dict[str, float]] = {
            "angry_reactive": {
                "angry_reactive": 1.0, "frustrated_impulsive": 0.5,
                "guarded_defensive": 0.2,
            },
            "frustrated_impulsive": {
                "frustrated_impulsive": 1.0, "angry_reactive": 0.4,
                "exhausted_stressed": 0.3,
            },
            "guarded_defensive": {
                "guarded_defensive": 1.0, "frustrated_impulsive": 0.2,
                "baseline_neutral": 0.3,
            },
            "confident_assertive": {
                "confident_assertive": 1.0, "optimistic_engaged": 0.5,
                "warm_collaborative": 0.2,
            },
            "exhausted_stressed": {
                "exhausted_stressed": 1.0, "defeated_resigned": 0.5,
                "frustrated_impulsive": 0.2,
            },
            "optimistic_engaged": {
                "optimistic_engaged": 1.0, "warm_collaborative": 0.4,
                "confident_assertive": 0.3,
            },
            "warm_collaborative": {
                "warm_collaborative": 1.0, "optimistic_engaged": 0.4,
            },
            "defeated_resigned": {
                "defeated_resigned": 1.0, "exhausted_stressed": 0.5,
            },
            "baseline_neutral": {
                "baseline_neutral": 1.0, "optimistic_engaged": 0.2,
                "guarded_defensive": 0.2,
            },
        }

    def route(self, brain_result: dict) -> list[tuple[str, float]]:
        """Return ranked list of (branch_name, score) based on brain state.

        Returns top 3 candidates for the executor to try.
        """
        regime = brain_result["regime"]
        state = brain_result["state"]

        # Get affinity scores for this regime
        affinities = self._regime_affinity.get(regime, {"baseline_neutral": 1.0})

        scored = []
        for branch_name, branch in self.branches.items():
            affinity = affinities.get(branch_name, 0.05)  # small base for all
            weight = branch.state.weight
            # Emotional intensity bonus — stronger emotions favor specialized branches
            intensity_bonus = 0.0
            if branch_name == "angry_reactive":
                intensity_bonus = state.get("anger", 0) * 0.3
            elif branch_name == "frustrated_impulsive":
                intensity_bonus = state.get("frustration", 0) * 0.3
            elif branch_name == "guarded_defensive":
                intensity_bonus = (1.0 - state.get("trust", 0.5)) * 0.3
            elif branch_name == "confident_assertive":
                intensity_bonus = state.get("confidence", 0.5) * 0.3
            elif branch_name == "defeated_resigned":
                intensity_bonus = (1.0 - state.get("hope", 0.5)) * 0.2 + state.get("fatigue", 0) * 0.2
            elif branch_name == "exhausted_stressed":
                intensity_bonus = state.get("stress", 0) * 0.2 + state.get("fatigue", 0) * 0.2
            elif branch_name == "optimistic_engaged":
                intensity_bonus = state.get("hope", 0.5) * 0.2 + state.get("curiosity", 0.5) * 0.2
            elif branch_name == "warm_collaborative":
                intensity_bonus = state.get("trust", 0.5) * 0.2 + state.get("empathy", 0.5) * 0.2

            score = affinity * weight + intensity_bonus
            scored.append((branch_name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:3]  # Top 3 candidates


# ---------------------------------------------------------------------------
# Emotional Context Builder — formats brain state for branch prompts
# ---------------------------------------------------------------------------

def build_emotional_context(
    brain_result: dict,
    scenario: str,
    character_name: str,
    character_background: str,
) -> str:
    """Build the {context} string that gets injected into branch templates."""
    state = brain_result["state"]
    regime = brain_result["regime"]
    conflicts = brain_result.get("conflicts", [])

    conflict_lines = ""
    if conflicts:
        conflict_lines = "\nInternal conflicts:\n" + "\n".join(
            f"  - {c['drives'][0]} ({c['values'][0]:.0%}) vs {c['drives'][1]} ({c['values'][1]:.0%})"
            for c in conflicts
        )

    return f"""SCENARIO: {scenario}

CHARACTER: {character_name}
{character_background}

CURRENT INTERNAL STATE:
  Regime: {regime}
  Confidence: {state['confidence']:.0%} | Stress: {state['stress']:.0%}
  Trust: {state['trust']:.0%} | Patience: {state['patience']:.0%}
  Anger: {state['anger']:.0%} | Hope: {state['hope']:.0%}
  Frustration: {state['frustration']:.0%} | Fatigue: {state['fatigue']:.0%}
  Motivation: {state['motivation']:.0%} | Emotional momentum: {state['emotional_momentum']:.0%}
{conflict_lines}

IMPORTANT: You are {character_name}, a real person. Never break character. Never say "I'm an AI"."""


# ---------------------------------------------------------------------------
# Emotional Coherence Evaluator
# ---------------------------------------------------------------------------

class EmotionalCoherenceJudge:
    """Scores how well the LLM output matches the intended emotional state.

    Uses simple heuristics — could be replaced with LLM-based evaluation.
    """

    def score(self, output: str, brain_result: dict, branch_name: str) -> float:
        """Return coherence score in [0, 1]."""
        state = brain_result["state"]
        regime = brain_result["regime"]
        words = output.lower().split()
        n_words = len(words)

        score = 0.5  # baseline

        # Length appropriateness
        if regime in ("defeated_resigned", "angry_reactive"):
            # Should be short
            if n_words <= 25:
                score += 0.15
            elif n_words > 50:
                score -= 0.15
        elif regime in ("confident_assertive", "optimistic_engaged"):
            # Can be longer
            if 15 <= n_words <= 60:
                score += 0.10

        # Emotional markers
        polite_words = {"appreciate", "understand", "thank", "thanks", "please", "sorry"}
        aggressive_words = {"no", "enough", "can't", "won't", "unacceptable", "demand",
                           "ridiculous", "seriously", "done", "fix", "now", "immediately"}
        defeated_words = {"fine", "whatever", "sure", "okay", "just", "guess"}

        polite_count = sum(1 for w in words if w.strip(".,!?\"'") in polite_words)
        aggressive_count = sum(1 for w in words if w.strip(".,!?\"'") in aggressive_words)
        defeated_count = sum(1 for w in words if w.strip(".,!?\"'") in defeated_words)

        # Caps words (shouting)
        caps_words = sum(1 for w in output.split() if w.isupper() and len(w) > 1)

        if state.get("anger", 0) > 0.4:
            # Should NOT be too polite when angry
            if polite_count >= 3:
                score -= 0.15
            if aggressive_count >= 1:
                score += 0.15
            if caps_words >= 1:
                score += 0.10

        if state.get("trust", 1.0) < 0.3:
            # Should be skeptical, not warm
            if polite_count >= 3:
                score -= 0.10

        if regime == "defeated_resigned":
            if defeated_count >= 1:
                score += 0.15
            if "!" in output:
                score -= 0.10  # defeated people don't use exclamation marks

        if regime in ("optimistic_engaged", "warm_collaborative"):
            if polite_count >= 1:
                score += 0.10

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Branch Weight Optimizer
# ---------------------------------------------------------------------------

class BranchWeightOptimizer:
    """Adapts branch weights based on emotional coherence scores.

    Simple advantage-based update: branches that score above average
    get boosted, those below get reduced.
    """

    def __init__(self, lr: float = 0.15, min_weight: float = 0.3, max_weight: float = 3.0):
        self.lr = lr
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._regime_history: dict[str, list[tuple[str, float]]] = {}

    def update(
        self,
        selected_branch: str,
        score: float,
        regime: str,
        branches: dict[str, PromptBranch],
    ) -> dict[str, float]:
        """Update branch weights based on score. Returns updated weights."""
        # Track history per regime
        if regime not in self._regime_history:
            self._regime_history[regime] = []
        self._regime_history[regime].append((selected_branch, score))

        # Compute baseline for this regime
        history = self._regime_history[regime]
        baseline = mean(s for _, s in history[-20:]) if len(history) >= 3 else 0.5

        # Advantage
        advantage = score - baseline
        branch = branches[selected_branch]
        old_weight = branch.state.weight
        new_weight = old_weight + self.lr * advantage
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        branch.state.weight = new_weight
        branch.apply_reward(score)

        return {selected_branch: new_weight}


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class BrainForestPipeline:
    """Complete Brain → Prompt Forest → LLM pipeline."""

    def __init__(
        self,
        scenario: str,
        character_name: str,
        character_background: str,
        momentum: float = 0.30,
        decay_rate: float = 0.02,
    ):
        self.scenario = scenario
        self.character_name = character_name
        self.character_background = character_background

        # Brain
        self.brain = BrainEngine(momentum=momentum, decay_rate=decay_rate)

        # Prompt Forest branches
        self.branches = _create_emotional_branches()

        # Router
        self.router = EmotionalRouter(self.branches)

        # Executor (uses real Prompt Forest executor + OpenAI backend)
        self.backend = OpenAIChatBackend(
            model="gpt-4o-mini",
            temperature=0.9,
            max_output_tokens=150,
            system_prompt="You are a character in a roleplay scenario. Follow the prompt exactly.",
        )
        self.executor = PromptExecutor(self.backend)

        # Evaluator + Optimizer
        self.judge = EmotionalCoherenceJudge()
        self.optimizer = BranchWeightOptimizer()

        # Conversation history (for LLM context)
        self.conversation_history: list[dict] = []

    def process_turn(
        self,
        event: dict,
        other_person_says: str,
        pre_events: list[dict] | None = None,
    ) -> dict:
        """Process one turn through the full pipeline.

        Returns dict with brain state, selected branch, speech, score, etc.
        """
        # 1. Pre-events (emotional baggage)
        for pre in (pre_events or []):
            self.brain.process_event(pre)

        # 2. Brain processes main event
        brain_result = self.brain.process_event(event)

        # 3. Build emotional context
        context = build_emotional_context(
            brain_result, self.scenario,
            self.character_name, self.character_background,
        )

        # 4. Router selects candidate branches
        candidates = self.router.route(brain_result)

        # 5. Execute top candidate through Prompt Forest executor
        # Build task with conversation history for context
        history_text = ""
        for msg in self.conversation_history[-6:]:  # last 3 exchanges
            role = "Them" if msg["role"] == "user" else self.character_name
            history_text += f"\n{role}: \"{msg['content']}\""

        task_text = f"{history_text}\nThem: \"{other_person_says}\""

        task = TaskInput(
            task_id=f"turn_{brain_result['turn']}",
            text=task_text.strip(),
            task_type="roleplay",
            metadata={"brain_regime": brain_result["regime"]},
        )

        # Try top candidate
        selected_name = candidates[0][0]
        selected_branch = self.branches[selected_name]
        branch_output = self.executor.run_branch(
            selected_branch, task,
            task_type="roleplay",
            context=context,
        )
        speech = branch_output.output.strip().strip('"')
        # Strip character name prefix if LLM echoes it (e.g. "Alex: ...")
        if speech.lower().startswith(self.character_name.lower() + ":"):
            speech = speech[len(self.character_name) + 1:].strip().strip('"')

        # 6. Judge emotional coherence
        score = self.judge.score(speech, brain_result, selected_name)

        # If score is too low and we have alternatives, try the second candidate
        if score < 0.4 and len(candidates) > 1:
            alt_name = candidates[1][0]
            alt_branch = self.branches[alt_name]
            alt_output = self.executor.run_branch(
                alt_branch, task,
                task_type="roleplay",
                context=context,
            )
            alt_speech = alt_output.output.strip().strip('"')
            if alt_speech.lower().startswith(self.character_name.lower() + ":"):
                alt_speech = alt_speech[len(self.character_name) + 1:].strip().strip('"')
            alt_score = self.judge.score(alt_speech, brain_result, alt_name)
            if alt_score > score:
                selected_name = alt_name
                speech = alt_speech
                score = alt_score

        # 7. Optimizer updates branch weights
        weight_updates = self.optimizer.update(
            selected_name, score,
            brain_result["regime"], self.branches,
        )

        # 8. Update conversation history
        self.conversation_history.append({"role": "user", "content": other_person_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        return {
            "turn": brain_result["turn"],
            "event": event,
            "regime": brain_result["regime"],
            "state": brain_result["state"],
            "conflicts": brain_result["conflicts"],
            "candidates": candidates,
            "selected_branch": selected_name,
            "speech": speech,
            "coherence_score": score,
            "weight_updates": weight_updates,
        }


# ---------------------------------------------------------------------------
# Demo: Customer Service (same scenario, now with full Prompt Forest)
# ---------------------------------------------------------------------------

def run_customer_service_demo():
    scenario = (
        "You are calling customer support for your internet provider. Your internet has been "
        "down for 3 days and you work from home. You've already called twice before and each time "
        "they said it would be fixed 'within 24 hours'. You're calling for the third time now."
    )
    background = (
        "Background: You're a freelance designer with a deadline tomorrow. You've lost $500 in "
        "productivity already. You started calm but you're running out of patience. You have a "
        "direct communication style and don't take well to being patronized."
    )

    pipeline = BrainForestPipeline(
        scenario=scenario,
        character_name="Alex",
        character_background=background,
        momentum=0.30,
        decay_rate=0.02,
    )

    turns = [
        {
            "event": {"type": "negative_outcome", "intensity": 0.0},
            "pre_events": [
                {"type": "negative_outcome", "intensity": 0.5},
                {"type": "negative_outcome", "intensity": 0.5},
            ],
            "says": "Thank you for calling TechNet support! My name is Jordan. How can I help you today?",
            "narration": "Third call. Carrying frustration from previous failures.",
        },
        {
            "event": {"type": "betrayal", "intensity": 0.6},
            "says": "I see, let me pull up your account... Okay I can see the previous tickets. It looks like a technician was scheduled but... hmm, it seems they marked it as resolved?",
            "narration": "Marked resolved without fixing. Betrayal of trust.",
        },
        {
            "event": {"type": "insult", "intensity": 0.7},
            "says": "Have you tried restarting your router? Sometimes these issues resolve themselves if you just power cycle the equipment.",
            "narration": "The scripted response. Not being listened to.",
        },
        {
            "event": {"type": "negative_outcome", "intensity": 0.8},
            "says": "I understand your frustration. Unfortunately, the earliest I can schedule a technician is... next Thursday. Would that work?",
            "narration": "A WEEK wait? Deadline is tomorrow.",
        },
        {
            "event": {"type": "insult", "intensity": 0.6},
            "says": "Sir/Ma'am, I want to help but I need you to stay calm so we can work through this together. There's a process we need to follow.",
            "narration": "Being told to calm down. Condescending.",
        },
        {
            "event": {"type": "opportunity", "intensity": 0.5},
            "says": "Actually, let me check one thing... I might be able to escalate this to our emergency repair team. They handle outages affecting work-from-home customers. Can you hold for just two minutes?",
            "narration": "A glimmer of hope. Someone is trying.",
        },
        {
            "event": {"type": "positive_outcome", "intensity": 0.7},
            "says": "Great news — I got approval. We're sending a technician tomorrow morning between 8-10 AM, and I'm also crediting your account for the 3 days of downtime. Does that work?",
            "narration": "Resolution. But trust is damaged.",
        },
    ]

    print("=" * 90)
    print("  BRAIN + PROMPT FOREST + LLM — Full Architecture")
    print("  Brain → Router → Specialized Branch → Executor → LLM → Evaluator → Optimizer")
    print("=" * 90)

    all_results = []

    for i, turn in enumerate(turns):
        result = pipeline.process_turn(
            event=turn["event"],
            other_person_says=turn["says"],
            pre_events=turn.get("pre_events"),
        )
        all_results.append(result)

        state = result["state"]
        print(f"\n{'─' * 90}")
        print(f"  TURN {i+1}: {turn['narration']}")
        print(f"{'─' * 90}")
        print(f"  Event: {turn['event']['type']}")
        print(f"  Regime: {result['regime']}")

        # Emotional bars
        key_vars = ["confidence", "stress", "frustration", "trust",
                     "anger", "patience", "hope", "fatigue"]
        print(f"\n  Emotional State:")
        for var in key_vars:
            val = state[var]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"    {var:<14s} {bar} {val:.0%}")

        if result["conflicts"]:
            print(f"\n  Internal Conflicts:")
            for c in result["conflicts"]:
                a, b = c["drives"]
                print(f"    ⚡ {a} vs {b} (intensity={c['intensity']})")

        # Prompt Forest routing
        print(f"\n  Router candidates:")
        for name, score in result["candidates"]:
            marker = " ◀ SELECTED" if name == result["selected_branch"] else ""
            weight = pipeline.branches[name].state.weight
            print(f"    {name:<25s} score={score:.3f}  weight={weight:.2f}{marker}")

        print(f"\n  Support: \"{turn['says']}\"")
        print(f"\n  Alex [{result['selected_branch']}]: \"{result['speech']}\"")
        print(f"\n  Coherence score: {result['coherence_score']:.2f}")

    # --- Emotional Journey ---
    print(f"\n\n{'═' * 90}")
    print(f"  EMOTIONAL JOURNEY + BRANCH ROUTING")
    print(f"{'═' * 90}")
    key_vars = ["anger", "trust", "stress", "frust", "patie", "hope"]
    full_vars = ["anger", "trust", "stress", "frustration", "patience", "hope"]
    print(f"\n  {'Turn':<5s} {'anger':>6s} {'trust':>6s} {'stress':>6s} {'frust':>6s} "
          f"{'patie':>6s} {'hope':>6s}  {'branch':<25s} {'regime'}")
    print(f"  {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*25} {'─'*20}")
    for r in all_results:
        s = r["state"]
        print(f"  {r['turn']:<5d} {s['anger']:>5.0%} {s['trust']:>5.0%} {s['stress']:>5.0%} "
              f"{s['frustration']:>5.0%} {s['patience']:>5.0%} {s['hope']:>5.0%}  "
              f"{r['selected_branch']:<25s} {r['regime']}")

    # --- Branch Weight Evolution ---
    print(f"\n  Branch Weights (after adaptation):")
    for name, branch in sorted(pipeline.branches.items()):
        rewards = branch.state.historical_rewards
        avg_r = mean(rewards) if rewards else 0.0
        n_used = len(rewards)
        print(f"    {name:<25s} weight={branch.state.weight:.2f}  "
              f"used={n_used}x  avg_score={avg_r:.2f}")

    # Usage stats
    usage = pipeline.backend.usage_summary()
    print(f"\n  LLM Usage: {usage['call_count']} calls, "
          f"{usage['total_tokens']} tokens, "
          f"{usage['mean_latency_ms']:.0f}ms avg latency")

    print(f"\n{'═' * 90}")


def run_negotiation_demo():
    """Salary negotiation through the full Brain + Prompt Forest + LLM pipeline."""
    scenario = (
        "You are in a salary negotiation for a senior software engineer position at a tech company. "
        "You currently earn $140k and believe you deserve $180k based on your experience and market rates. "
        "The hiring manager is sitting across from you. This job would be a significant career step."
    )
    background = (
        "Background: You're 32 years old, 8 years of experience. You've been underpaid at your "
        "current job for 2 years and this negotiation matters a lot to you. You're generally "
        "professional but have a temper when you feel disrespected."
    )

    pipeline = BrainForestPipeline(
        scenario=scenario,
        character_name="Alex",
        character_background=background,
        momentum=0.30,
        decay_rate=0.02,
    )

    turns = [
        {
            "event": {"type": "opportunity", "intensity": 0.7},
            "says": "Thanks for coming in, Alex. We're excited about your candidacy. Let's talk compensation. What are you looking for?",
            "narration": "The negotiation opens. You feel the opportunity ahead.",
        },
        {
            "event": {"type": "positive_outcome", "intensity": 0.4},
            "says": "I appreciate you being upfront. $180k is above our initial range, but let me see what we can do. Your skills are definitely what we need.",
            "narration": "The manager seems receptive. A small win.",
        },
        {
            "event": {"type": "negative_outcome", "intensity": 0.6},
            "says": "So I spoke with our comp team, and honestly the best we can do right now is $155k. I know that's below what you mentioned.",
            "narration": "A significant lowball. Below expectations.",
        },
        {
            "event": {"type": "pressure", "intensity": 0.8},
            "says": "Look, I need an answer by end of day. We have other strong candidates and the team wants to close this position this week.",
            "narration": "Pressure tactics. Deadline imposed.",
        },
        {
            "event": {"type": "insult", "intensity": 0.7},
            "says": "Between us, $155k is generous given that your current role is a level below what we're hiring for. You'd be stretching into this position.",
            "narration": "Your experience is being dismissed. Feels disrespectful.",
        },
        {
            "event": {"type": "negative_outcome", "intensity": 0.6},
            "says": "I understand your frustration, but the budget is the budget. I can maybe push to $160k, but that's truly the ceiling.",
            "narration": "A small concession, but still far from your target.",
        },
        {
            "event": {"type": "surprise", "intensity": 0.6},
            "says": "Actually — I just got a message. Our VP wants to meet you. She might have flexibility that I don't. Would you be open to a quick chat with her?",
            "narration": "An unexpected turn. New possibility opens up.",
        },
        {
            "event": {"type": "praise", "intensity": 0.7},
            "says": "Alex, I've been in this role for 10 years, and your technical interview was one of the best I've ever seen. I genuinely want you on this team.",
            "narration": "Genuine recognition. Your value is acknowledged.",
        },
    ]

    print("=" * 90)
    print("  BRAIN + PROMPT FOREST + LLM — Salary Negotiation")
    print("  Brain → Router → Specialized Branch → Executor → LLM → Evaluator → Optimizer")
    print("=" * 90)

    all_results = []

    for i, turn in enumerate(turns):
        result = pipeline.process_turn(
            event=turn["event"],
            other_person_says=turn["says"],
            pre_events=turn.get("pre_events"),
        )
        all_results.append(result)

        state = result["state"]
        print(f"\n{'─' * 90}")
        print(f"  TURN {i+1}: {turn['narration']}")
        print(f"{'─' * 90}")
        print(f"  Event: {turn['event']['type']}")
        print(f"  Regime: {result['regime']}")

        # Emotional bars
        key_vars = ["confidence", "stress", "frustration", "trust",
                     "anger", "patience", "hope", "fatigue"]
        print(f"\n  Emotional State:")
        for var in key_vars:
            val = state[var]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"    {var:<14s} {bar} {val:.0%}")

        if result["conflicts"]:
            print(f"\n  Internal Conflicts:")
            for c in result["conflicts"]:
                a, b = c["drives"]
                print(f"    ⚡ {a} vs {b} (intensity={c['intensity']})")

        # Prompt Forest routing
        print(f"\n  Router candidates:")
        for name, score in result["candidates"]:
            marker = " ◀ SELECTED" if name == result["selected_branch"] else ""
            weight = pipeline.branches[name].state.weight
            print(f"    {name:<25s} score={score:.3f}  weight={weight:.2f}{marker}")

        print(f"\n  Manager: \"{turn['says']}\"")
        print(f"\n  Alex [{result['selected_branch']}]: \"{result['speech']}\"")
        print(f"\n  Coherence score: {result['coherence_score']:.2f}")

    # --- Emotional Journey ---
    print(f"\n\n{'═' * 90}")
    print(f"  EMOTIONAL JOURNEY + BRANCH ROUTING")
    print(f"{'═' * 90}")
    print(f"\n  {'Turn':<5s} {'anger':>6s} {'trust':>6s} {'stress':>6s} {'frust':>6s} "
          f"{'patie':>6s} {'hope':>6s}  {'branch':<25s} {'regime'}")
    print(f"  {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*25} {'─'*20}")
    for r in all_results:
        s = r["state"]
        print(f"  {r['turn']:<5d} {s['anger']:>5.0%} {s['trust']:>5.0%} {s['stress']:>5.0%} "
              f"{s['frustration']:>5.0%} {s['patience']:>5.0%} {s['hope']:>5.0%}  "
              f"{r['selected_branch']:<25s} {r['regime']}")

    # --- Branch Weight Evolution ---
    print(f"\n  Branch Weights (after adaptation):")
    for name, branch in sorted(pipeline.branches.items()):
        rewards = branch.state.historical_rewards
        avg_r = mean(rewards) if rewards else 0.0
        n_used = len(rewards)
        print(f"    {name:<25s} weight={branch.state.weight:.2f}  "
              f"used={n_used}x  avg_score={avg_r:.2f}")

    # Usage stats
    usage = pipeline.backend.usage_summary()
    print(f"\n  LLM Usage: {usage['call_count']} calls, "
          f"{usage['total_tokens']} tokens, "
          f"{usage['mean_latency_ms']:.0f}ms avg latency")

    print(f"\n{'═' * 90}")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "negotiation":
        run_negotiation_demo()
    else:
        run_customer_service_demo()

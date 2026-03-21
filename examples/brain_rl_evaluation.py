#!/usr/bin/env python3
"""RL Evaluation: Does the optimizer actually improve over time?

Runs multiple diverse scenarios sequentially so the optimizer accumulates
experience. Compares:
  1. Adaptive system WITH learning (branch weights update from scores)
  2. Adaptive system WITHOUT learning (fixed weights = 1.0)

Uses LLM-based coherence judge instead of heuristics for better reward signal.
Uses epsilon-greedy exploration to discover better branches.

Scenarios are diverse and unbiased — not designed to favor any specific regime.
"""

from __future__ import annotations

import json
import os
import sys
import time
import copy
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openai

from src.prompt_forest.branches.base import PromptBranch
from src.prompt_forest.types import BranchState, BranchStatus, TaskInput
from src.prompt_forest.core.executor import PromptExecutor
from src.prompt_forest.backend.openai_chat import OpenAIChatBackend

from brain_llm_prototype import BrainEngine, BrainState, _clamp, _BASELINE_DICT
from brain_adaptive_prototype import (
    AdaptiveBrainEngine,
    PersonalityProfile,
    DynamicBranchGenerator,
    EmotionalCoherenceJudge as HeuristicJudge,
    BranchWeightOptimizer,
    build_emotional_context,
    _generate_deltas_via_llm,
)


# ---------------------------------------------------------------------------
# LLM-based Coherence Judge (replaces heuristic scorer)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """You are an emotional coherence evaluator. Given a character's emotional
state and their spoken response, score how well the speech matches the intended emotions.

Score from 0.0 to 1.0:
- 1.0 = speech perfectly reflects the emotional state (tone, word choice, length, intensity all match)
- 0.7 = good match with minor misalignment
- 0.5 = partially matches but some emotions are not reflected
- 0.3 = poor match — speech tone contradicts the emotional state
- 0.0 = completely wrong tone

Output ONLY a JSON object with two keys:
  "score": float between 0.0 and 1.0
  "reason": one sentence explaining why

Be strict but fair. Consider:
- Does the TONE match? (angry state should sound angry, not polite)
- Does the LENGTH match? (defeated people use short responses, confident people elaborate)
- Are the right emotions visible in word choice?
- Is the intensity appropriate? (85% anger should be explosive, 20% anger should be mild irritation)"""


class LLMCoherenceJudge:
    """Uses an LLM to score emotional coherence — much better signal than heuristics."""

    def __init__(self):
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._calls = 0
        self._total_ms = 0.0

    def score(self, output: str, brain_result: dict, branch_name: str) -> tuple[float, str]:
        """Return (score, reason) tuple."""
        state = brain_result["state"]
        regime = brain_result["regime"]

        state_summary = ", ".join(f"{k}={v:.0%}" for k, v in state.items()
                                   if k != "emotional_momentum")

        user_prompt = f"""INTENDED EMOTIONAL STATE:
  Regime: {regime}
  {state_summary}

BRANCH USED: {branch_name}

CHARACTER'S SPOKEN RESPONSE:
  "{output}"

Score the emotional coherence."""

        try:
            started = time.perf_counter()
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=100,
            )
            self._total_ms += (time.perf_counter() - started) * 1000.0
            self._calls += 1

            data = json.loads(response.choices[0].message.content.strip())
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            reason = data.get("reason", "")
            return score, reason

        except Exception as e:
            return 0.5, f"Judge failed: {e}"

    def summary(self) -> dict:
        return {
            "calls": self._calls,
            "avg_ms": round(self._total_ms / max(1, self._calls), 1),
        }


# ---------------------------------------------------------------------------
# Stable Branch Pool (generated once, reused across all turns)
# ---------------------------------------------------------------------------

def _create_stable_branches() -> dict[str, PromptBranch]:
    """Pre-defined stable branches covering the emotional spectrum.

    These are fixed targets the optimizer can learn over — unlike fully
    dynamic branches which change every turn.
    """
    templates = {
        "explosive_angry": {
            "purpose": "Explosive anger — shouting, demanding, threatening",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You are FURIOUS. Patience is GONE. You've HAD ENOUGH.
- Use SHORT, SHARP sentences. Cut to the point.
- Use caps for ONE or TWO key words to show intensity.
- Make DEMANDS, not requests. Threaten consequences.
- Don't soften anything — no "I understand" or "I appreciate."
- Examples: "No. DONE. Get me someone who can actually help." / "This is UNACCEPTABLE. Fix it NOW or I walk."

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "cold_angry": {
            "purpose": "Cold controlled anger — quiet fury, clipped, icy",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You are angry but CONTROLLED. Cold, not hot.
- Speak in measured, clipped sentences. Every word is deliberate.
- Use silence and brevity as weapons. Don't explain yourself.
- Show anger through what you DON'T say — short, final statements.
- No shouting, no caps. Just ice.
- Examples: "I see. And what exactly do you plan to do about it." / "That's not going to work for me."

Respond as the character. 1-2 sentences max. Stay in character:

{task}""",
        },
        "frustrated_sarcastic": {
            "purpose": "Frustrated with visible sarcasm and exasperation",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're FRUSTRATED and it's showing. Sarcasm is your weapon.
- Mix reasonable points with flashes of biting sarcasm.
- Eye-rolls, sighs, "oh great", "wow, really" energy.
- You're not fully enraged — you're exasperated and done pretending.
- Examples: "Oh wonderful. Another promise. Can't wait." / "Right, because that worked so well last time."

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "frustrated_direct": {
            "purpose": "Frustrated but channeling it into blunt directness",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're frustrated and done with pleasantries. CUT TO THE POINT.
- No small talk, no padding, no diplomatic softening.
- State what you want clearly and firmly. Skip the preamble.
- Show frustration through EFFICIENCY — you don't have time for this.
- Examples: "Here's what I need. Can you do it or not?" / "Skip the script. What's the actual solution?"

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "guarded_skeptical": {
            "purpose": "Distrustful, questioning everything, protecting self",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You DON'T TRUST the other person. On guard.
- Question everything. Ask for specifics and proof.
- Don't volunteer information. Keep responses measured.
- Show skepticism through pointed questions, not hostility.
- Examples: "Can I get that in writing?" / "And what happens when that doesn't work either?"

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "defeated_flat": {
            "purpose": "Emotionally drained, giving up, flat affect",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're DONE. Energy is gone. You don't have fight left.
- SHORT responses. Monotone. No exclamation marks.
- You might agree just to end the conversation.
- Trailing off, flat delivery, minimal engagement.
- Examples: "Fine. Whatever works." / "Yeah. Sure. Just... do what you need to do."

Respond as the character. 1-2 sentences max. Stay in character:

{task}""",
        },
        "stressed_scattered": {
            "purpose": "High stress, slightly scattered, running on fumes",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're STRESSED and TIRED. Barely holding it together.
- Responses may be slightly disjointed. You might lose your train of thought.
- Shorter patience but not angry — just DRAINED.
- Show stress through broken rhythm and overwhelm.
- Examples: "Sorry, what was — yeah, okay. What does that mean for me?" / "I just... I need this sorted."

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "confident_assertive": {
            "purpose": "Strong, in control, clear and direct",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You feel STRONG and IN CONTROL. You know your worth.
- Speak with clarity and conviction. Make direct points.
- Show initiative — propose solutions, set terms.
- Not aggressive, just SURE of yourself.
- Examples: "Here's what I'm thinking — let's structure it this way." / "I know what I bring to the table."

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "confident_warm": {
            "purpose": "Confident but approachable, collaborative confidence",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're confident AND warm. Strength with generosity.
- Acknowledge the other person while maintaining your position.
- Use "we" language. Build rapport while being clear about what you want.
- Firm but friendly — the iron fist in the velvet glove.
- Examples: "I think we can find something that works for both of us." / "I appreciate that — and here's where I'd need to see movement."

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "optimistic_curious": {
            "purpose": "Positive, forward-looking, asking good questions",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're feeling POSITIVE and genuinely interested.
- Ask good questions. Show curiosity and engagement.
- Be enthusiastic but grounded — not over-the-top.
- Show optimism through forward-looking statements.
- Examples: "That sounds promising — tell me more about how that would work." / "I like where this is going."

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "cautious_polite": {
            "purpose": "Careful, diplomatic, hedging bets",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're being CAREFUL. Weighing options. Not committing.
- Use hedging language — "perhaps", "I'd want to think about", "that could work."
- Polite but noncommittal. Keeping options open.
- Show caution through measured, diplomatic responses.
- Examples: "That's worth considering. Let me think about it." / "I'd want to understand more before committing."

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
        "neutral_professional": {
            "purpose": "Calm, measured, professional baseline",
            "template": """You are roleplaying as a real person in a conversation. Here is the situation and your current emotional state:

{context}

CRITICAL BEHAVIORAL RULES:
- You're emotionally neutral. Calm, clear, professional.
- Be direct without being cold. Normal conversational tone.
- Neither excited nor flat. Just... normal.
- Examples: "I'd like to understand the situation better." / "That works for me. What's the next step?"

Respond as the character. 1-3 sentences max. Stay in character:

{task}""",
        },
    }

    branches = {}
    for name, config in templates.items():
        branches[name] = PromptBranch(BranchState(
            name=name,
            purpose=config["purpose"],
            prompt_template=config["template"],
            weight=1.0,
            status=BranchStatus.ACTIVE,
        ))
    return branches


# ---------------------------------------------------------------------------
# RL-Enabled Router with epsilon-greedy exploration
# ---------------------------------------------------------------------------

class RLRouter:
    """Routes to branches using regime affinity + learned weights + exploration."""

    def __init__(self, branches: dict[str, PromptBranch], epsilon: float = 0.15):
        self.branches = branches
        self.epsilon = epsilon
        self._rng_counter = 0

        # Regime → branch affinity (which branches are relevant)
        self._affinity: dict[str, dict[str, float]] = {
            "angry_reactive": {
                "explosive_angry": 1.0, "cold_angry": 0.8,
                "frustrated_direct": 0.3,
            },
            "frustrated_impulsive": {
                "frustrated_sarcastic": 0.9, "frustrated_direct": 0.9,
                "explosive_angry": 0.3, "cold_angry": 0.3,
            },
            "guarded_defensive": {
                "guarded_skeptical": 1.0, "cold_angry": 0.4,
                "cautious_polite": 0.5, "frustrated_direct": 0.2,
            },
            "confident_assertive": {
                "confident_assertive": 1.0, "confident_warm": 0.6,
                "frustrated_direct": 0.2,
            },
            "exhausted_stressed": {
                "stressed_scattered": 1.0, "defeated_flat": 0.5,
                "frustrated_direct": 0.2,
            },
            "optimistic_engaged": {
                "optimistic_curious": 1.0, "confident_warm": 0.6,
                "confident_assertive": 0.3,
            },
            "warm_collaborative": {
                "confident_warm": 1.0, "optimistic_curious": 0.5,
                "cautious_polite": 0.3,
            },
            "defeated_resigned": {
                "defeated_flat": 1.0, "stressed_scattered": 0.4,
            },
            "baseline_neutral": {
                "neutral_professional": 1.0, "cautious_polite": 0.5,
                "optimistic_curious": 0.3,
            },
        }

    def route(self, brain_result: dict, explore: bool = True) -> list[tuple[str, float]]:
        """Return ranked (branch_name, score) list. Top 3."""
        import random
        regime = brain_result["regime"]
        state = brain_result["state"]

        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            # Pick a random branch
            all_names = list(self.branches.keys())
            picked = random.choice(all_names)
            result = [(picked, 0.5)]
            # Add the greedy pick as fallback
            greedy = self._greedy_route(regime, state)
            if greedy and greedy[0][0] != picked:
                result.append(greedy[0])
            return result

        return self._greedy_route(regime, state)

    def _greedy_route(self, regime: str, state: dict) -> list[tuple[str, float]]:
        """Score all branches and return top 3."""
        affinities = self._affinity.get(regime, {"neutral_professional": 1.0})

        scored = []
        for name, branch in self.branches.items():
            affinity = affinities.get(name, 0.05)
            weight = branch.state.weight

            # Emotional intensity bonuses
            intensity = 0.0
            if "angry" in name:
                intensity = state.get("anger", 0) * 0.3
            elif "frustrated" in name:
                intensity = state.get("frustration", 0) * 0.3
            elif "guarded" in name or "skeptical" in name:
                intensity = (1.0 - state.get("trust", 0.5)) * 0.3
            elif "confident" in name:
                intensity = state.get("confidence", 0.5) * 0.2
            elif "defeated" in name:
                intensity = (1.0 - state.get("hope", 0.5)) * 0.2 + state.get("fatigue", 0) * 0.2
            elif "stressed" in name:
                intensity = state.get("stress", 0) * 0.2 + state.get("fatigue", 0) * 0.2
            elif "optimistic" in name or "curious" in name:
                intensity = state.get("hope", 0.5) * 0.2
            elif "cautious" in name:
                intensity = state.get("caution", 0.3) * 0.2

            score = affinity * weight + intensity
            scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:3]


# ---------------------------------------------------------------------------
# Full RL Pipeline
# ---------------------------------------------------------------------------

class RLPipeline:
    """Pipeline with proper RL: stable branches, LLM judge, exploration."""

    def __init__(
        self,
        personality: PersonalityProfile,
        scenario: str,
        use_learning: bool = True,
        epsilon: float = 0.15,
        momentum: float = 0.30,
        decay_rate: float = 0.02,
    ):
        self.personality = personality
        self.scenario = scenario
        self.use_learning = use_learning

        # Brain
        self.brain = AdaptiveBrainEngine(
            personality=personality, scenario=scenario,
            momentum=momentum, decay_rate=decay_rate,
        )

        # Stable branch pool
        self.branches = _create_stable_branches()

        # Router with exploration
        self.router = RLRouter(self.branches, epsilon=epsilon if use_learning else 0.0)

        # Executor
        self.backend = OpenAIChatBackend(
            model="gpt-4o-mini", temperature=0.9,
            max_output_tokens=150,
            system_prompt="You are a character in a roleplay scenario. Follow the prompt exactly.",
        )
        self.executor = PromptExecutor(self.backend)

        # LLM-based judge
        self.judge = LLMCoherenceJudge()

        # Optimizer
        self.optimizer = BranchWeightOptimizer(lr=0.12)

        # History
        self.conversation_history: list[dict] = []
        self.all_scores: list[dict] = []

    def reset_for_scenario(self, scenario: str):
        """Reset brain and conversation for a new scenario (keep learned weights)."""
        self.scenario = scenario
        self.brain = AdaptiveBrainEngine(
            personality=self.personality, scenario=scenario,
            momentum=0.30, decay_rate=0.02,
        )
        self.conversation_history = []

    def process_turn(self, event: dict, other_person_says: str,
                     pre_events: list[dict] | None = None) -> dict:
        # 1. Pre-events
        for pre in (pre_events or []):
            self.brain.process_event(pre)

        # 2. Brain processes event
        brain_result = self.brain.process_event(event)

        # 3. Context
        context = build_emotional_context(brain_result, self.scenario, self.personality)

        # 4. Route (with exploration if learning)
        candidates = self.router.route(brain_result, explore=self.use_learning)

        # 5. Execute
        history_text = ""
        for msg in self.conversation_history[-6:]:
            role = "Them" if msg["role"] == "user" else self.personality.name
            history_text += f"\n{role}: \"{msg['content']}\""
        task_text = f"{history_text}\nThem: \"{other_person_says}\""

        task = TaskInput(
            task_id=f"turn_{brain_result['turn']}",
            text=task_text.strip(),
            task_type="roleplay",
            metadata={"brain_regime": brain_result["regime"]},
        )

        selected_name = candidates[0][0]
        selected_branch = self.branches[selected_name]
        branch_output = self.executor.run_branch(
            selected_branch, task, task_type="roleplay", context=context,
        )
        speech = branch_output.output.strip().strip('"')
        if speech.lower().startswith(self.personality.name.lower() + ":"):
            speech = speech[len(self.personality.name) + 1:].strip().strip('"')

        # 6. LLM judge
        score, reason = self.judge.score(speech, brain_result, selected_name)

        # 7. Try fallback if bad
        if score < 0.35 and len(candidates) > 1:
            alt_name = candidates[1][0]
            alt_branch = self.branches[alt_name]
            alt_output = self.executor.run_branch(
                alt_branch, task, task_type="roleplay", context=context,
            )
            alt_speech = alt_output.output.strip().strip('"')
            if alt_speech.lower().startswith(self.personality.name.lower() + ":"):
                alt_speech = alt_speech[len(self.personality.name) + 1:].strip().strip('"')
            alt_score, alt_reason = self.judge.score(alt_speech, brain_result, alt_name)
            if alt_score > score:
                selected_name = alt_name
                selected_branch = self.branches[selected_name]
                speech = alt_speech
                score = alt_score
                reason = alt_reason

        # 8. Update weights if learning
        if self.use_learning:
            self.optimizer.update(selected_name, score,
                                  brain_result["regime"], selected_branch)

        # 9. History
        self.conversation_history.append({"role": "user", "content": other_person_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        record = {
            "turn": brain_result["turn"],
            "regime": brain_result["regime"],
            "branch": selected_name,
            "score": score,
            "reason": reason,
            "speech": speech,
            "state": brain_result["state"],
        }
        self.all_scores.append(record)
        return record


# ---------------------------------------------------------------------------
# Diverse Scenarios (unbiased — cover different emotional trajectories)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "Salary Negotiation",
        "scenario": "You are in a salary negotiation for a senior software engineer position. You currently earn $140k and want $180k. The hiring manager is across from you.",
        "turns": [
            {"event": {"type": "opportunity", "intensity": 0.7, "description": "Negotiation opens, you feel excited"}, "says": "Thanks for coming in. Let's talk compensation. What are you looking for?"},
            {"event": {"type": "positive_outcome", "intensity": 0.4, "description": "Manager seems open to your number"}, "says": "I appreciate you being upfront. $180k is above our range, but let me see what we can do."},
            {"event": {"type": "negative_outcome", "intensity": 0.6, "description": "Lowball offer of $155k"}, "says": "The best we can do is $155k. I know that's below what you mentioned."},
            {"event": {"type": "pressure", "intensity": 0.8, "description": "Deadline pressure and competing candidates"}, "says": "I need an answer by end of day. We have other strong candidates."},
            {"event": {"type": "insult", "intensity": 0.7, "description": "Told you'd be stretching into the role"}, "says": "Between us, $155k is generous given your current role is a level below. You'd be stretching."},
            {"event": {"type": "positive_outcome", "intensity": 0.7, "description": "VP wants to meet you directly"}, "says": "Actually, our VP wants to meet you. She might have flexibility I don't."},
        ],
    },
    {
        "name": "Customer Support (Internet Down)",
        "scenario": "You're calling internet support for the 3rd time. Internet has been down 3 days. You work from home as a freelance designer with a deadline tomorrow.",
        "turns": [
            {"event": {"type": "negative_outcome", "intensity": 0.3, "description": "Third call to support, carrying frustration"}, "pre_events": [{"type": "negative_outcome", "intensity": 0.5}, {"type": "negative_outcome", "intensity": 0.5}], "says": "Thank you for calling TechNet support! My name is Jordan. How can I help you today?"},
            {"event": {"type": "betrayal", "intensity": 0.6, "description": "Previous ticket was marked resolved but nothing was fixed"}, "says": "I can see the previous tickets. It looks like a technician was scheduled but... they marked it as resolved?"},
            {"event": {"type": "insult", "intensity": 0.7, "description": "Asked to restart router — the most patronizing response possible"}, "says": "Have you tried restarting your router? Sometimes these issues resolve themselves."},
            {"event": {"type": "negative_outcome", "intensity": 0.8, "description": "Earliest technician is next Thursday — a week away"}, "says": "The earliest I can schedule a technician is next Thursday. Would that work?"},
            {"event": {"type": "insult", "intensity": 0.6, "description": "Told to stay calm — condescending"}, "says": "I need you to stay calm so we can work through this together. There's a process."},
            {"event": {"type": "positive_outcome", "intensity": 0.7, "description": "Technician tomorrow plus credit for downtime"}, "says": "Great news — technician tomorrow morning 8-10 AM, plus I'm crediting your account for 3 days. Does that work?"},
        ],
    },
    {
        "name": "Doctor Appointment (Bad News)",
        "scenario": "You're at a doctor's appointment to discuss test results. You've been anxious for a week waiting. The doctor is professional but direct.",
        "turns": [
            {"event": {"type": "pressure", "intensity": 0.5, "description": "Waiting in the office, anxiety building"}, "says": "Thanks for coming in. I have your test results here. Let me pull them up."},
            {"event": {"type": "negative_outcome", "intensity": 0.7, "description": "Results show something concerning that needs further investigation"}, "says": "So your results show some elevated markers that we need to look into further. It's not definitive, but I want to be thorough."},
            {"event": {"type": "surprise", "intensity": 0.5, "description": "Doctor mentions it could be something serious"}, "says": "In some cases, these markers can indicate an autoimmune condition. I want to run a few more specific tests to narrow it down."},
            {"event": {"type": "negative_outcome", "intensity": 0.4, "description": "More tests mean more waiting and uncertainty"}, "says": "The additional tests will take about two weeks to come back. I know the waiting is hard."},
            {"event": {"type": "positive_outcome", "intensity": 0.5, "description": "Doctor says it's very likely manageable"}, "says": "I should say — even if it is what I suspect, it's very manageable with medication. Most patients live completely normal lives."},
            {"event": {"type": "relief", "intensity": 0.6, "description": "Doctor provides a clear plan and reassurance"}, "says": "Here's what we'll do: blood work Monday, follow-up in two weeks, and I'm here if you have questions before then."},
        ],
    },
    {
        "name": "Landlord Dispute",
        "scenario": "You're confronting your landlord about a broken heater in winter. It's been 3 weeks. You have a young child at home. The landlord has been dodging your calls.",
        "turns": [
            {"event": {"type": "negative_outcome", "intensity": 0.6, "description": "Finally reached landlord after weeks of being ignored"}, "pre_events": [{"type": "negative_outcome", "intensity": 0.4}, {"type": "betrayal", "intensity": 0.5}], "says": "Oh hey, sorry I've been hard to reach. What's going on?"},
            {"event": {"type": "insult", "intensity": 0.6, "description": "Landlord minimizes the problem"}, "says": "A broken heater? Can't you just use a space heater for now? Those work fine."},
            {"event": {"type": "betrayal", "intensity": 0.7, "description": "Landlord says it's not a priority"}, "says": "Look, I've got other properties with bigger issues. I'll get to it when I can, probably next month."},
            {"event": {"type": "threat", "intensity": 0.5, "description": "Landlord implies rent consequences"}, "says": "And just so we're clear — rent is still due on the first. I can't give discounts for maintenance issues."},
            {"event": {"type": "opportunity", "intensity": 0.6, "description": "You mention tenant rights and the landlord's tone changes"}, "says": "Okay, okay. Let me see what I can do. Maybe I can get someone out this week."},
            {"event": {"type": "positive_outcome", "intensity": 0.5, "description": "Landlord agrees to send someone but grudgingly"}, "says": "Fine. I'll have my guy come by Wednesday. But this is a one-time thing, I can't keep dropping everything."},
        ],
    },
    {
        "name": "Job Interview (Going Well)",
        "scenario": "You're in a job interview for your dream role at a company you admire. The interviewer is the team lead. Things are going surprisingly well.",
        "turns": [
            {"event": {"type": "opportunity", "intensity": 0.6, "description": "Interview starts, you're excited but nervous"}, "says": "Welcome! We've heard great things about you from your referral. Tell me about your background."},
            {"event": {"type": "praise", "intensity": 0.6, "description": "Interviewer is impressed by your answer"}, "says": "That's exactly the kind of experience we're looking for. Really impressive work on that project."},
            {"event": {"type": "surprise", "intensity": 0.5, "description": "They ask you to lead a whiteboard session — unexpected"}, "says": "Would you be up for a quick whiteboard session? I'd love to see how you think through a problem live."},
            {"event": {"type": "positive_outcome", "intensity": 0.7, "description": "You nail the whiteboard and they're clearly impressed"}, "says": "Wow, that's a really elegant approach. We've had senior candidates struggle with that one."},
            {"event": {"type": "praise", "intensity": 0.7, "description": "Team lead says you'd be a great culture fit"}, "says": "Honestly, you'd be a fantastic addition to the team. Your communication style really fits our culture."},
            {"event": {"type": "positive_outcome", "intensity": 0.8, "description": "They hint at an offer"}, "says": "I don't want to get ahead of myself, but I'm going to strongly recommend we move forward. Can you come back Thursday to meet the VP?"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(personality: PersonalityProfile):
    """Run all scenarios with and without learning, compare scores."""

    print(f"\n{'=' * 90}")
    print(f"  RL EVALUATION — {personality.temperament[:60]}...")
    print(f"  {len(SCENARIOS)} scenarios × ~6 turns = ~{sum(len(s['turns']) for s in SCENARIOS)} turns")
    print(f"{'=' * 90}")

    # --- WITH LEARNING ---
    pipeline_learn = RLPipeline(
        personality=personality,
        scenario=SCENARIOS[0]["scenario"],
        use_learning=True,
        epsilon=0.15,
    )

    # --- WITHOUT LEARNING ---
    pipeline_fixed = RLPipeline(
        personality=personality,
        scenario=SCENARIOS[0]["scenario"],
        use_learning=False,
        epsilon=0.0,
    )

    for si, scenario_data in enumerate(SCENARIOS):
        sname = scenario_data["name"]
        scenario_text = scenario_data["scenario"]
        turns = scenario_data["turns"]

        # Reset brains for new scenario (learning pipeline keeps weights)
        pipeline_learn.reset_for_scenario(scenario_text)
        pipeline_fixed.reset_for_scenario(scenario_text)

        print(f"\n  {'─' * 86}")
        print(f"  SCENARIO {si+1}: {sname}")
        print(f"  {'─' * 86}")

        for ti, turn in enumerate(turns):
            # Run both pipelines
            r_learn = pipeline_learn.process_turn(
                turn["event"], turn["says"], turn.get("pre_events"),
            )
            r_fixed = pipeline_fixed.process_turn(
                turn["event"], turn["says"], turn.get("pre_events"),
            )

            learn_marker = "✓" if r_learn["score"] >= r_fixed["score"] else " "
            fixed_marker = "✓" if r_fixed["score"] >= r_learn["score"] else " "

            print(f"    T{ti+1} [{r_learn['regime']:<22s}] "
                  f"Learn: {r_learn['branch']:<22s} {r_learn['score']:.2f}{learn_marker} | "
                  f"Fixed: {r_fixed['branch']:<22s} {r_fixed['score']:.2f}{fixed_marker}")

    # --- Summary ---
    learn_scores = [r["score"] for r in pipeline_learn.all_scores]
    fixed_scores = [r["score"] for r in pipeline_fixed.all_scores]

    n = len(learn_scores)
    half = n // 2
    learn_first_half = learn_scores[:half]
    learn_second_half = learn_scores[half:]
    fixed_first_half = fixed_scores[:half]
    fixed_second_half = fixed_scores[half:]

    print(f"\n\n{'═' * 90}")
    print(f"  RESULTS")
    print(f"{'═' * 90}")

    print(f"\n  Overall ({n} turns):")
    print(f"    WITH learning:    mean={mean(learn_scores):.3f}  std={stdev(learn_scores):.3f}")
    print(f"    WITHOUT learning: mean={mean(fixed_scores):.3f}  std={stdev(fixed_scores):.3f}")
    diff = mean(learn_scores) - mean(fixed_scores)
    print(f"    Difference:       {diff:+.3f} {'(learning helps)' if diff > 0 else '(learning hurts)' if diff < 0 else '(no difference)'}")

    print(f"\n  First half ({half} turns) vs Second half ({n - half} turns):")
    print(f"    Learn first:  {mean(learn_first_half):.3f}  →  Learn second: {mean(learn_second_half):.3f}  "
          f"(Δ = {mean(learn_second_half) - mean(learn_first_half):+.3f})")
    print(f"    Fixed first:  {mean(fixed_first_half):.3f}  →  Fixed second: {mean(fixed_second_half):.3f}  "
          f"(Δ = {mean(fixed_second_half) - mean(fixed_first_half):+.3f})")

    # Per-scenario breakdown
    print(f"\n  Per-scenario scores:")
    idx = 0
    for scenario_data in SCENARIOS:
        n_turns = len(scenario_data["turns"])
        ls = learn_scores[idx:idx + n_turns]
        fs = fixed_scores[idx:idx + n_turns]
        print(f"    {scenario_data['name']:<30s}  Learn={mean(ls):.3f}  Fixed={mean(fs):.3f}  "
              f"Δ={mean(ls) - mean(fs):+.3f}")
        idx += n_turns

    # Branch weight evolution
    print(f"\n  Branch weights after learning:")
    sorted_branches = sorted(pipeline_learn.branches.items(),
                              key=lambda x: x[1].state.weight, reverse=True)
    for name, branch in sorted_branches:
        rewards = branch.state.historical_rewards
        n_used = len(rewards)
        if n_used > 0:
            print(f"    {name:<24s} weight={branch.state.weight:.2f}  "
                  f"used={n_used}x  avg_score={mean(rewards):.2f}")

    # LLM call budget
    delta_stats = pipeline_learn.brain.delta_call_summary()
    judge_stats = pipeline_learn.judge.summary()
    speech_stats = pipeline_learn.backend.usage_summary()
    print(f"\n  LLM budget (learning pipeline):")
    print(f"    Delta interpretation: {delta_stats['calls']} calls ({delta_stats['avg_ms']:.0f}ms avg)")
    print(f"    Judge evaluation:     {judge_stats['calls']} calls ({judge_stats['avg_ms']:.0f}ms avg)")
    print(f"    Speech generation:    {speech_stats['call_count']} calls ({speech_stats['total_tokens']} tokens)")

    print(f"\n{'═' * 90}")

    return learn_scores, fixed_scores


def main():
    print("=" * 90)
    print("  RL EVALUATION: Does the optimizer actually improve over time?")
    print("  5 diverse scenarios × 6 turns = 30 turns per personality")
    print("  WITH learning vs WITHOUT learning (fixed weights)")
    print("  LLM-based coherence judge for proper reward signal")
    print("=" * 90)

    fiery = PersonalityProfile(
        name="Alex",
        background="32 years old, 8 years experience, underpaid for 2 years. Tired of being undervalued.",
        temperament="Hot-tempered, direct, takes disrespect personally. Speaks from the gut. Quick to escalate.",
        emotional_tendencies={
            "anger": "quick to flare, expressed openly",
            "patience": "runs out fast",
            "impulse": "high, speaks before thinking",
        },
    )

    run_evaluation(fiery)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Brain-Enhanced LLM — Brain as context provider, not controller.

The brain tracks emotional state deterministically (momentum, decay,
cross-interactions). Its output is translated into natural language
that AUGMENTS a single LLM call rather than constraining it through
templates and branch routing.

Architecture:
  Event → BrainStateMachine → translate_emotional_state()
       → Single LLM(personality + emotional narrative + history) → Speech

One LLM call. No branches. No routing. No templates.
Personality stays primary. Emotions modulate, never override.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openai

from brain_llm_prototype import BrainEngine, BrainState, _clamp, _BASELINE_DICT
from brain_adaptive_prototype import (
    AdaptiveBrainEngine,
    PersonalityProfile,
    _generate_deltas_via_llm,
)


# ---------------------------------------------------------------------------
# Emotional State → Natural Language Translation
# ---------------------------------------------------------------------------

# Human-readable labels for emotions at different levels
_EMOTION_DESCRIPTORS = {
    "anger": {
        "high_rising": "Your anger has been building and you're close to boiling over",
        "high_steady": "You're angry and it's not fading",
        "high_falling": "You're still angry, but starting to come down slightly",
        "moderate_rising": "Irritation is creeping in and growing",
        "moderate_steady": "There's a persistent edge of irritation",
        "moderate_falling": "Your annoyance is slowly easing",
        "low": None,  # don't mention if low
    },
    "trust": {
        "high_rising": "You're starting to trust this person more",
        "high_steady": "You feel relatively trusting right now",
        "low_rising": "Your trust was shattered but is slowly rebuilding",
        "low_steady": "You don't trust this person at all",
        "low_falling": "What little trust you had is evaporating",
        "moderate_falling": "Your trust is slipping with each exchange",
    },
    "stress": {
        "high_rising": "Stress is mounting — you feel the pressure physically",
        "high_steady": "You're under heavy stress and it's not letting up",
        "moderate_rising": "The tension is building",
        "low": None,
    },
    "frustration": {
        "high_rising": "Frustration is compounding — nothing seems to work",
        "high_steady": "Deep frustration has settled in",
        "moderate_rising": "You're getting increasingly frustrated",
        "moderate_falling": "The frustration is easing somewhat",
        "low": None,
    },
    "patience": {
        "low_falling": "Your patience is almost completely gone",
        "low_steady": "You have virtually no patience left",
        "moderate_falling": "Your patience is wearing thin",
        "high": None,  # don't mention if high (it's the default)
    },
    "hope": {
        "high_rising": "You're feeling genuinely hopeful — things might work out",
        "high_steady": "You're holding onto hope",
        "low_falling": "Hope is draining away",
        "low_steady": "You've largely given up on a good outcome",
        "moderate_rising": "A spark of hope is emerging",
        "moderate_falling": "Your hope is fading",
    },
    "confidence": {
        "high_rising": "Your confidence is surging — you know your worth and you're not afraid to show it",
        "high_steady": "You're assured in your position — you've earned this and everyone should know it",
        "low_falling": "Your confidence is crumbling",
        "low_steady": "You're feeling unsure of yourself",
        "moderate_falling": "Your confidence is taking hits",
        "moderate_rising": "You're starting to feel your footing — getting more sure of yourself",
    },
    "fatigue": {
        "high_rising": "You're exhausted and running on fumes",
        "high_steady": "Deep tiredness weighs on everything",
        "moderate_rising": "The conversation is draining you",
        "low": None,
    },
    "fear": {
        "high_rising": "Anxiety is spiking — you're genuinely worried",
        "high_steady": "A deep unease sits in your gut",
        "moderate_rising": "Worry is creeping in",
        "low": None,
    },
    "impulse": {
        "high_rising": "You're fighting the urge to say something you might regret",
        "high_steady": "You're barely filtering what comes out",
    },
    "empathy": {
        "low_falling": "You've stopped caring about the other person's perspective",
    },
    "curiosity": {
        "high_rising": "You're genuinely interested and want to know more",
        "moderate_rising": "Your curiosity is piqued",
    },
    "caution": {
        "high_rising": "You're being very careful about what you reveal or commit to",
        "moderate_rising": "You're starting to hedge — protecting yourself",
    },
}


def _classify_level(val: float, baseline: float) -> str:
    """Classify value as high/moderate/low relative to baseline."""
    dev = val - baseline
    if dev > 0.20:
        return "high"
    elif dev > 0.08:
        return "moderate"
    elif dev < -0.20:
        return "low"
    elif dev < -0.08:
        return "moderate_low"
    return "neutral"


def _classify_direction(current: float, previous: float) -> str:
    """Classify direction of change."""
    delta = current - previous
    if delta > 0.05:
        return "rising"
    elif delta < -0.05:
        return "falling"
    return "steady"


def _count_sustained_turns(brain_history: list[dict], var: str, threshold: float, above: bool = True) -> int:
    """Count how many recent consecutive turns a variable has been above/below threshold."""
    count = 0
    for h in reversed(brain_history):
        val = h["state"].get(var, 0.5)
        if (above and val >= threshold) or (not above and val <= threshold):
            count += 1
        else:
            break
    return count


def translate_emotional_state(
    brain_result: dict,
    brain_history: list[dict],
    personality: PersonalityProfile,
) -> str:
    """Convert brain state into natural language emotional narrative.

    Returns a few lines describing only the SALIENT emotions — not all 16.
    Uses trajectory (how emotions changed) not just current values.
    Emotions that have been sustained get EVOLVING descriptions — turn 5 anger
    is described differently than turn 1 anger.
    """
    state = brain_result["state"]
    conflicts = brain_result.get("conflicts", [])
    turn_num = len(brain_history) + 1

    is_volatile = any(w in personality.temperament.lower()
                      for w in ("hot-tempered", "impulsive", "direct", "quick to escalate"))

    # Get previous state for trajectory
    prev_state = None
    if len(brain_history) >= 2:
        prev_state = brain_history[-2]["state"]
    elif len(brain_history) >= 1:
        prev_state = brain_history[-1]["state"]

    # Find salient emotions (notably different from baseline)
    salient = []
    for var in _EMOTION_DESCRIPTORS:
        val = state.get(var, 0.5)
        baseline = _BASELINE_DICT.get(var, 0.5)
        deviation = abs(val - baseline)
        if deviation < 0.08:
            continue

        level = _classify_level(val, baseline)
        if level == "neutral":
            continue

        direction = "steady"
        if prev_state:
            direction = _classify_direction(val, prev_state.get(var, 0.5))

        if var in ("patience", "trust", "hope", "confidence"):
            if val < baseline:
                level = "low" if val < baseline - 0.15 else "moderate"
            else:
                level = "high"
        else:
            if val > baseline:
                level = "high" if val > baseline + 0.15 else "moderate"
            else:
                level = "low"

        salient.append((var, val, deviation, level, direction))

    salient.sort(key=lambda x: x[2], reverse=True)
    salient = salient[:3]  # MAX 3 emotions — keep it punchy

    # Generate natural language lines with SUSTAINED-STATE EVOLUTION
    lines = []
    for var, val, dev, level, direction in salient:
        # Count how long this emotion has been elevated/depressed
        baseline_val = _BASELINE_DICT.get(var, 0.5)
        if val > baseline_val:
            sustained = _count_sustained_turns(brain_history, var, baseline_val + 0.08, above=True)
        else:
            sustained = _count_sustained_turns(brain_history, var, baseline_val - 0.08, above=False)

        desc = _get_evolved_descriptor(var, val, level, direction, sustained, is_volatile, personality)
        if desc:
            lines.append(f"- {desc}")

    if not lines:
        lines.append("- You're in a relatively neutral emotional state")

    # Add INTENSITY score — a single number that anchors the LLM's emotional expression
    neg_vars = ["anger", "frustration", "stress", "fear", "fatigue"]
    pos_vars = ["confidence", "hope"]
    neg_weight = sum(max(0, state.get(v, 0) - _BASELINE_DICT.get(v, 0)) for v in neg_vars)
    pos_weight = sum(max(0, state.get(v, 0) - _BASELINE_DICT.get(v, 0)) for v in pos_vars)
    patience_loss = max(0, _BASELINE_DICT.get("patience", 0.7) - state.get("patience", 0.7))
    trust_loss = max(0, _BASELINE_DICT.get("trust", 0.5) - state.get("trust", 0.5))

    # Intensity = how far from neutral baseline (0-10 scale)
    raw_intensity = (neg_weight + patience_loss + trust_loss) * 5 + pos_weight * 3
    intensity = min(10, max(1, round(raw_intensity + 2)))  # minimum 2, max 10

    # Determine dominant valence
    if neg_weight > pos_weight:
        if intensity >= 8:
            valence_desc = "you're seething / desperate / overwhelmed — barely holding it together"
        elif intensity >= 6:
            valence_desc = "you're clearly upset and it colors every word you say"
        elif intensity >= 4:
            valence_desc = "there's a definite edge — irritation, worry, or frustration is simmering"
        else:
            valence_desc = "slightly off-balance but still composed"
    else:
        if intensity >= 7:
            valence_desc = "you're buzzing with confidence and energy — you own this moment"
        elif intensity >= 5:
            valence_desc = "you're in a positive, engaged headspace — feeling your worth"
        else:
            valence_desc = "you're calm and grounded"

    lines.insert(0, f"EMOTIONAL INTENSITY: {intensity}/10 — {valence_desc}")

    return "\n".join(lines)


def _get_evolved_descriptor(
    var: str, val: float, level: str, direction: str,
    sustained_turns: int, is_volatile: bool, personality: PersonalityProfile,
) -> str | None:
    """Get an emotion descriptor that EVOLVES based on how long it's been sustained."""
    # For sustained states (3+ turns), use evolved descriptions
    if sustained_turns >= 4:
        evolved = _SUSTAINED_DESCRIPTORS.get(var, {}).get("deep", None)
        if evolved:
            return evolved if not is_volatile else _SUSTAINED_DESCRIPTORS.get(var, {}).get("deep_volatile", evolved)
    elif sustained_turns >= 2:
        evolved = _SUSTAINED_DESCRIPTORS.get(var, {}).get("hardened", None)
        if evolved:
            return evolved if not is_volatile else _SUSTAINED_DESCRIPTORS.get(var, {}).get("hardened_volatile", evolved)

    # For non-sustained or early turns, use the standard descriptors
    descriptors = _EMOTION_DESCRIPTORS.get(var, {})
    keys_to_try = [f"{level}_{direction}", f"{level}_steady", level]
    for key in keys_to_try:
        desc = descriptors.get(key)
        if desc:
            return desc
    return None


# Sustained-state evolution descriptors
_SUSTAINED_DESCRIPTORS = {
    "anger": {
        "hardened": "Your anger has solidified — it's not a flash, it's a steady burn. You're done being reasonable.",
        "hardened_volatile": "Your anger has been building for multiple rounds and you're about to snap. Every word they say makes it worse.",
        "deep": "You're past anger. This is contempt. You're barely willing to continue this conversation.",
        "deep_volatile": "You're DONE. The anger has been building so long it's turned into something dangerous. You're ready to burn bridges.",
    },
    "frustration": {
        "hardened": "Frustration has curdled into something deeper — you feel like nothing you say matters.",
        "hardened_volatile": "You're so frustrated you could scream. Nothing is working and nobody is listening.",
        "deep": "The frustration is total. You've lost faith that this interaction will produce anything useful.",
        "deep_volatile": "You're beyond frustrated — you're ready to walk out, hang up, or do something drastic.",
    },
    "stress": {
        "hardened": "The stress has been constant and it's affecting your thinking — you're reactive, not strategic.",
        "deep": "You're running on cortisol and adrenaline. Every response feels like it costs you something.",
    },
    "patience": {
        "hardened": "Your patience ran out turns ago. Now you're operating on pure stubbornness.",
        "hardened_volatile": "Patience? What patience? You've been done with niceties for a while now.",
        "deep": "There is no patience left. Zero. You're one wrong word from ending this entirely.",
        "deep_volatile": "You have absolutely NO patience left. You're either getting what you want RIGHT NOW or you're done.",
    },
    "trust": {
        "hardened": "Trust has eroded steadily — you assume the worst about everything they say now.",
        "deep": "You don't believe a single word they're saying. Every promise sounds like a lie.",
    },
    "hope": {
        "hardened": "Hope is fading — you're starting to accept this won't end well.",
        "deep": "You've given up. Whatever happens happens. You're just going through the motions.",
    },
    "confidence": {
        "hardened": "Your confidence has been building steadily — you KNOW what you're worth and you're done being modest about it.",
        "hardened_volatile": "You're riding high on confidence now. You've got swagger and you're not dialing it back for anyone.",
        "deep": "You're supremely confident. You own this room and everyone in it should recognize that.",
    },
    "fatigue": {
        "hardened": "You're tired of this. Emotionally drained. Your responses are getting shorter because you're spent.",
        "deep": "Complete emotional exhaustion. You don't have the energy to fight anymore — just want this over.",
    },
    "hope": {
        "hardened": "Your optimism is building — this might actually work out, and you're starting to let yourself believe it.",
        "deep": "You're genuinely hopeful now. The guard is coming down and you're starting to invest emotionally in a good outcome.",
    },
    "curiosity": {
        "hardened": "Your interest is sustained — you're genuinely engaged and want to dig deeper.",
    },
}


def generate_behavioral_hints(
    state: dict, personality: PersonalityProfile, brain_history: list[dict] | None = None,
) -> str:
    """Generate soft behavioral hints from emotional state + personality.

    NOT rigid templates. Personality-aware suggestions.
    Includes sustained-state behavioral evolution.
    """
    hints = []
    anger = state.get("anger", 0.05)
    patience = state.get("patience", 0.7)
    trust = state.get("trust", 0.5)
    stress = state.get("stress", 0.2)
    confidence = state.get("confidence", 0.55)
    hope = state.get("hope", 0.5)
    fatigue = state.get("fatigue", 0.15)
    frustration = state.get("frustration", 0.1)
    impulse = state.get("impulse", 0.3)

    is_volatile = any(w in personality.temperament.lower()
                      for w in ("hot-tempered", "impulsive", "direct", "quick to escalate"))
    is_calm = any(w in personality.temperament.lower()
                  for w in ("patient", "analytical", "calm"))

    history = brain_history or []
    turn_num = len(history) + 1

    # Track sustained anger/frustration for behavioral evolution
    anger_sustained = _count_sustained_turns(history, "anger", 0.15, above=True) if history else 0
    frust_sustained = _count_sustained_turns(history, "frustration", 0.18, above=True) if history else 0

    if anger > 0.4 and patience < 0.3:
        if anger_sustained >= 3:
            if is_volatile:
                hints.append("You've been angry for so long you're not even articulating it anymore — it comes out as demands, ultimatums, or threats")
            else:
                hints.append("The sustained anger has made you cold and cutting — you've moved past emotion into ruthless pragmatism")
        elif is_volatile:
            hints.append("You'd speak in short, sharp bursts — no filter right now")
        else:
            hints.append("You'd be clipped and cold — the anger shows in brevity, not volume")
    elif anger > 0.3:
        if is_volatile:
            hints.append("Your responses would have an edge — you're not hiding your irritation")
        else:
            hints.append("You're controlled but there's steel underneath")

    if frustration > 0.3 and patience < 0.4:
        if frust_sustained >= 3:
            hints.append("Don't repeat yourself — you've already made your point. Escalate: demand action, threaten consequences, or express disgust")
        else:
            hints.append("You wouldn't bother with pleasantries or diplomatic softening")

    if trust < 0.3:
        if _count_sustained_turns(history, "trust", 0.4, above=False) >= 2:
            hints.append("You assume they're lying or stalling. Challenge EVERY claim they make.")
        else:
            hints.append("You'd question promises and ask for specifics — you've been let down")

    if confidence > 0.6 and stress < 0.3:
        hints.append("You'd speak with conviction — you know your position is strong")
    elif confidence < 0.35:
        hints.append("You'd be less assertive than usual — your footing feels uncertain")

    if hope < 0.3 and fatigue > 0.3:
        hints.append("You're running out of energy to fight — responses would be shorter, flatter")

    if impulse > 0.5 and is_volatile:
        hints.append("You might say something raw and unfiltered — the impulse is strong")

    if stress > 0.5 and fatigue > 0.3:
        hints.append("You're overwhelmed — you might lose your train of thought or ramble slightly")

    # Late-conversation escalation cue — only when genuinely negative
    if turn_num >= 4 and anger > 0.45 and frustration > 0.35 and hope < 0.4:
        hints.append("This has gone on too long. A real person would escalate — ask for a supervisor, threaten to leave, mention legal action, or give an ultimatum")

    # Positive trajectory hints
    if confidence > 0.6 and hope > 0.45 and anger < 0.3:
        if is_volatile:
            hints.append("You're feeling good — let that confidence show as boldness and energy, not aggression")
        else:
            hints.append("You're in a strong position — you'd be assured and warm, not combative")

    if not hints:
        hints.append("You'd respond naturally, in your normal conversational style")

    return "\n".join(f"- {h}" for h in hints[:4])


# ---------------------------------------------------------------------------
# Brain-Enhanced LLM System
# ---------------------------------------------------------------------------

class BrainEnhancedLLM:
    """Single LLM call augmented with brain state context.

    The brain tracks emotions deterministically. Its output is translated
    into natural language that guides — not constrains — the LLM.
    """

    def __init__(
        self,
        personality: PersonalityProfile,
        scenario: str,
        use_llm_deltas: bool = False,
        momentum: float = 0.30,
        decay_rate: float = 0.02,
    ):
        self.personality = personality
        self.scenario = scenario
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []
        self.total_tokens = 0

        # Brain engine — always use LLM deltas for personality-aware reactions
        self.brain = AdaptiveBrainEngine(
            personality=personality, scenario=scenario,
            momentum=momentum, decay_rate=decay_rate,
        )

    def reset_for_scenario(self, scenario: str):
        """Reset for new scenario, keeping personality."""
        self.scenario = scenario
        self.conversation_history = []
        self.brain = AdaptiveBrainEngine(
            personality=self.personality, scenario=scenario,
            momentum=self.brain.momentum, decay_rate=self.brain.decay_rate,
        )

    def _build_system_prompt(self, emotional_narrative: str) -> str:
        """Build system prompt: personality + concise emotional state."""
        tendencies = ""
        if self.personality.emotional_tendencies:
            tendencies = "\nEmotional tendencies: " + ", ".join(
                f"{k} — {v}" for k, v in self.personality.emotional_tendencies.items()
            )

        return f"""You are {self.personality.name}, a real human being in this situation:

{self.scenario}

About you:
{self.personality.background}
Your temperament: {self.personality.temperament}{tendencies}

EMOTIONAL CONTEXT (how your feelings have been building across this conversation):
{emotional_narrative}

Rules:
- You ARE {self.personality.name}. Never break character.
- Your personality and temperament are PRIMARY. The emotional context above enriches your performance — use it to calibrate your intensity and tone.
- 1-3 sentences. Real humans keep it short.
- Sound HUMAN: fragments, contractions, "look", "honestly", "come on". Imperfect grammar.
- Never say "I understand your perspective", "I appreciate your transparency", "I hear you".
- Let your {self.personality.temperament.split(',')[0].lower()} nature show through in EVERY response."""

    def _get_top_emotions(self, state: dict) -> str:
        """Get 2-3 word emotional summary for history annotation."""
        emotions = []
        for var in ["anger", "frustration", "stress", "confidence", "fear", "hope", "trust", "patience"]:
            val = state.get(var, 0.5)
            baseline = _BASELINE_DICT.get(var, 0.5)
            dev = val - baseline
            if abs(dev) > 0.12:
                if var in ("patience", "trust", "hope"):
                    if dev < 0:
                        emotions.append(f"low {var}")
                else:
                    if dev > 0:
                        emotions.append(f"high {var}")
                    elif dev < -0.15:
                        emotions.append(f"low {var}")
        return ", ".join(emotions[:3])

    def respond(
        self,
        event: dict,
        other_person_says: str,
        pre_events: list[dict] | None = None,
    ) -> dict:
        """Process one turn: brain tracks state, LLM generates speech."""
        # 1. Brain processes events (lower momentum on early turns for responsiveness)
        original_momentum = self.brain.momentum
        if self.brain.turn < 2:
            self.brain.momentum = 0.10  # more responsive on early turns
        for pre in (pre_events or []):
            self.brain.process_event(pre)
        brain_result = self.brain.process_event(event)
        self.brain.momentum = original_momentum

        # 2. Translate state to CONCISE natural language
        emotional_narrative = translate_emotional_state(
            brain_result, self.brain.history[:-1], self.personality,
        )

        # 3. Build prompt
        system_prompt = self._build_system_prompt(emotional_narrative)

        # 4. Single LLM call
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_person_says})

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.85,
                max_tokens=150,
            )
            speech = response.choices[0].message.content.strip().strip('"')
            if speech.lower().startswith(self.personality.name.lower() + ":"):
                speech = speech[len(self.personality.name) + 1:].strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0
        except Exception as e:
            speech = f"[Error: {e}]"

        # 5. Update conversation history (clean — no annotations)
        self.conversation_history.append({"role": "user", "content": other_person_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        return {
            "speech": speech,
            "regime": brain_result["regime"],
            "state": brain_result["state"],
            "emotional_narrative": emotional_narrative,
            "behavioral_hints": "",
            "turn": brain_result["turn"],
            "conflicts": brain_result.get("conflicts", []),
        }


# ---------------------------------------------------------------------------
# Test harness: Brain-Enhanced vs Plain LLM, blind judged
# ---------------------------------------------------------------------------

from brain_vs_plain_llm import PlainLLMBaseline, BlindJudge
from brain_rl_evaluation import SCENARIOS


def run_test(personality: PersonalityProfile, scenarios: list[dict]):
    """Head-to-head: Brain-Enhanced LLM vs Plain LLM."""

    print(f"\n{'=' * 90}")
    print(f"  BRAIN-ENHANCED LLM vs PLAIN LLM")
    print(f"  Personality: {personality.temperament[:70]}")
    print(f"  {len(scenarios)} scenarios, blind LLM judge")
    print(f"{'=' * 90}")

    enhanced = BrainEnhancedLLM(personality, scenarios[0]["scenario"])
    plain = PlainLLMBaseline(personality, scenarios[0]["scenario"])
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    judge = BlindJudge(model=judge_model)

    all_enhanced_totals = []
    all_plain_totals = []
    all_results = []

    for si, scenario_data in enumerate(scenarios):
        sname = scenario_data["name"]
        enhanced.reset_for_scenario(scenario_data["scenario"])
        plain.reset_for_scenario(scenario_data["scenario"])

        print(f"\n  {'─' * 86}")
        print(f"  SCENARIO {si+1}: {sname}")
        print(f"  {'─' * 86}")

        conversation_lines = []

        for ti, turn in enumerate(scenario_data["turns"]):
            event = turn["event"]
            says = turn["says"]
            event_desc = event.get("description", event.get("type", ""))

            # Brain-enhanced system
            result = enhanced.respond(event, says, turn.get("pre_events"))
            enhanced_speech = result["speech"]

            # Plain LLM
            plain_speech = plain.respond(says, event_desc)

            # Blind judge
            conv_summary = "\n".join(conversation_lines[-8:]) if conversation_lines else "(start)"
            judgment = judge.judge(
                scenario_data["scenario"], personality, event_desc, says,
                enhanced_speech, plain_speech, conv_summary, ti + 1,
            )

            es = judgment["brain_scores"]  # "brain" = enhanced in the judge
            ps = judgment["plain_scores"]
            e_total = es.get("total", sum(es.get(k, 0) for k in ["emotional_accuracy", "naturalness", "consistency"]))
            p_total = ps.get("total", sum(ps.get(k, 0) for k in ["emotional_accuracy", "naturalness", "consistency"]))

            all_enhanced_totals.append(e_total)
            all_plain_totals.append(p_total)
            all_results.append({
                "scenario": sname, "turn": ti + 1,
                "regime": result["regime"],
                "enhanced_speech": enhanced_speech,
                "plain_speech": plain_speech,
                "enhanced_scores": es, "plain_scores": ps,
            })

            winner = "ENHANCED" if e_total > p_total else "PLAIN" if p_total > e_total else "TIE"
            mark = {"ENHANCED": "◀", "PLAIN": "▶", "TIE": "="}[winner]

            print(f"\n    Turn {ti+1}: {event['type']} | {result['regime']}")
            print(f"    Enhanced: \"{enhanced_speech[:90]}{'...' if len(enhanced_speech) > 90 else ''}\"")
            print(f"    Plain:    \"{plain_speech[:90]}{'...' if len(plain_speech) > 90 else ''}\"")
            print(f"    Score: Enhanced={e_total}/30  Plain={p_total}/30  {mark} {winner}")

            conversation_lines.append(f"  Them: \"{says[:60]}...\"")
            conversation_lines.append(f"  {personality.name}: \"{enhanced_speech[:60]}...\"")

    # --- Results ---
    from statistics import mean, stdev

    e_mean = mean(all_enhanced_totals)
    p_mean = mean(all_plain_totals)
    n = len(all_enhanced_totals)

    print(f"\n\n{'═' * 90}")
    print(f"  RESULTS ({n} turns)")
    print(f"{'═' * 90}")

    print(f"\n  Brain-Enhanced: {e_mean:.1f}/30 avg  ({e_mean/30:.0%})")
    print(f"  Plain LLM:      {p_mean:.1f}/30 avg  ({p_mean/30:.0%})")
    diff = e_mean - p_mean
    print(f"  Difference:     {diff:+.1f} {'(enhanced wins!)' if diff > 0 else '(plain wins)' if diff < 0 else '(tie)'}")

    e_wins = sum(1 for e, p in zip(all_enhanced_totals, all_plain_totals) if e > p)
    p_wins = sum(1 for e, p in zip(all_enhanced_totals, all_plain_totals) if p > e)
    ties = sum(1 for e, p in zip(all_enhanced_totals, all_plain_totals) if e == p)
    print(f"  Enhanced wins: {e_wins}  |  Plain wins: {p_wins}  |  Ties: {ties}")

    # Per-criterion
    e_ea = mean(r["enhanced_scores"].get("emotional_accuracy", 0) for r in all_results)
    p_ea = mean(r["plain_scores"].get("emotional_accuracy", 0) for r in all_results)
    e_nat = mean(r["enhanced_scores"].get("naturalness", 0) for r in all_results)
    p_nat = mean(r["plain_scores"].get("naturalness", 0) for r in all_results)
    e_con = mean(r["enhanced_scores"].get("consistency", 0) for r in all_results)
    p_con = mean(r["plain_scores"].get("consistency", 0) for r in all_results)

    print(f"\n  {'Criterion':<25s} {'Enhanced':>8s} {'Plain':>6s} {'Diff':>7s}")
    print(f"  {'─'*25} {'─'*8} {'─'*6} {'─'*7}")
    print(f"  {'Emotional accuracy':<25s} {e_ea:>7.1f} {p_ea:>5.1f} {e_ea - p_ea:>+6.1f}")
    print(f"  {'Naturalness':<25s} {e_nat:>7.1f} {p_nat:>5.1f} {e_nat - p_nat:>+6.1f}")
    print(f"  {'Consistency':<25s} {e_con:>7.1f} {p_con:>5.1f} {e_con - p_con:>+6.1f}")

    # Per-scenario
    print(f"\n  Per-scenario:")
    idx = 0
    for sd in scenarios:
        nt = len(sd["turns"])
        es = all_enhanced_totals[idx:idx + nt]
        ps = all_plain_totals[idx:idx + nt]
        ew = sum(1 for e, p in zip(es, ps) if e > p)
        pw = sum(1 for e, p in zip(es, ps) if p > e)
        print(f"    {sd['name']:<32s} Enh={mean(es):.1f}  Plain={mean(ps):.1f}  (E:{ew} P:{pw})")
        idx += nt

    print(f"\n{'═' * 90}")
    return e_mean, p_mean


def main():
    personality_type = os.environ.get("PERSONALITY", "fiery")

    if personality_type == "calm":
        personality = PersonalityProfile(
            name="Sarah",
            background="35 years old, analytical mind, 10 years in corporate. Values precision and fairness. Recently passed over for a promotion she deserved.",
            temperament="Calm and measured, but with a cold edge when pushed. Processes internally before responding. When angry, gets quiet and surgical, not loud.",
            emotional_tendencies={
                "anger": "slow to build but devastating when it arrives — expressed as cold precision",
                "patience": "high baseline but when it breaks, it breaks completely",
                "trust": "earned slowly, lost permanently",
            },
        )
    else:
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

    n_trials = int(os.environ.get("N_TRIALS", "1"))
    if n_trials > 1:
        all_e = []
        all_p = []
        for t in range(n_trials):
            print(f"\n\n{'#' * 90}")
            print(f"  TRIAL {t+1}/{n_trials}")
            print(f"{'#' * 90}")
            e_mean, p_mean = run_test(personality, SCENARIOS)
            all_e.append(e_mean)
            all_p.append(p_mean)

        from statistics import mean, stdev
        print(f"\n\n{'█' * 90}")
        print(f"  AGGREGATE RESULTS ({n_trials} trials)")
        print(f"{'█' * 90}")
        print(f"  Enhanced: {mean(all_e):.1f}/30  (std: {stdev(all_e):.1f})  ({mean(all_e)/30:.0%})")
        print(f"  Plain:    {mean(all_p):.1f}/30  (std: {stdev(all_p):.1f})  ({mean(all_p)/30:.0%})")
        print(f"  Difference: {mean(all_e) - mean(all_p):+.1f}")
        print(f"  Per-trial enhanced: {[f'{x:.1f}' for x in all_e]}")
        print(f"  Per-trial plain:    {[f'{x:.1f}' for x in all_p]}")
        print(f"{'█' * 90}")
    else:
        run_test(personality, SCENARIOS)


if __name__ == "__main__":
    main()

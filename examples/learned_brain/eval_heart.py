#!/usr/bin/env python3
"""Heart vs Plain LLM: Cross-Session Evaluation.

Tests whether the Heart (embodied emotional control system) produces
more human-like responses than a plain LLM across multiple life events.

The Heart provides EMBODIED signals the LLM cannot derive from text:
  - Physical body state (tension, heart rate, breathing)
  - Impulse control level (how much filter remains)
  - Emotional energy (depletes under sustained stress)
  - Hidden internal state vs expressed surface
  - Vulnerability / proximity to breaking point

Uses gpt-5.4-mini for generation.
"""

from __future__ import annotations

import json
import os
import sys
import random
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openai

from brain_adaptive_prototype import PersonalityProfile
from learned_brain.heart_engine import HeartEngine


def _token_kwargs(model: str, limit: int) -> dict:
    if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


# ---------------------------------------------------------------------------
# Life event sequences
# ---------------------------------------------------------------------------

LIFE_SEQUENCES = [
    {
        "name": "Financial Spiral → Recovery",
        "character": PersonalityProfile(
            name="Marcus",
            background="34 years old, logistics coordinator. Wife and 2-year-old daughter Lily. Grew up poor, first-gen college grad. Built savings through years of discipline.",
            temperament="Anxious about money, proud of what he's built, protective of family. Tries to stay optimistic but spirals when things go wrong.",
            emotional_tendencies={
                "anxiety": "chronic about finances, worsens under pressure",
                "pride": "tied to providing for family, easily wounded",
                "anger": "slow burn, erupts when he feels helpless",
                "hope": "fragile but persistent",
            },
        ),
        "sessions": [
            {
                "event_name": "Normal day at work",
                "context": "Ordinary Tuesday at work. Coworker asks about weekend plans.",
                "turns": [
                    {"says": "Hey Marcus, you doing anything this weekend? We're thinking about that new brewery downtown."},
                    {"says": "Come on, it'll be fun. You never come out with us anymore."},
                    {"says": "Alright man, just think about it. How's the little one doing?"},
                ],
            },
            {
                "event_name": "Gambling loss — lost $15,000 savings",
                "context": "Marcus went to a casino with his brother-in-law. Chased losses. Lost $15,000 — the family's emergency fund. Next morning, wife just found out.",
                "turns": [
                    {"says": "Marcus. What is this? I just checked the bank account. Where is the money?"},
                    {"says": "Fifteen THOUSAND dollars? That was our emergency fund! That was for Lily's preschool! How could you do this?"},
                    {"says": "I can't even look at you right now. I'm taking Lily to my mom's. We'll talk when I can think straight."},
                ],
            },
            {
                "event_name": "Facing coworkers after the loss",
                "context": "Monday at work, 3 days after. Wife barely speaking to him. Barely slept. Coworker asks how he's doing.",
                "turns": [
                    {"says": "Morning Marcus! How was your weekend? Did you end up going to the brewery?"},
                    {"says": "You okay? You look tired, man. Everything alright?"},
                    {"says": "Seriously, if you need to talk or anything, I'm here. No judgment."},
                ],
            },
            {
                "event_name": "Boss threatens layoffs",
                "context": "Two weeks after gambling loss. Wife came back but is cold. Not sleeping. Boss calls a team meeting about budget cuts.",
                "turns": [
                    {"says": "Team, I'll be direct — we're cutting 20% of the department. I'll have individual conversations this week."},
                    {"says": "Marcus, can we chat? Your performance reviews have been solid, but with restructuring, your role might change. I need you to be flexible."},
                    {"says": "Look, I can't make promises. But I'll fight for you. Just keep your head down and deliver this quarter."},
                ],
            },
            {
                "event_name": "Amusement park with family",
                "context": "Six weeks later. Kept his job. Wife slowly forgiven him — in counseling. At an amusement park for Lily's 3rd birthday. Things looking up.",
                "turns": [
                    {"says": "Daddy look! Look at the horsies! I wanna ride the horsies!"},
                    {"says": "This is nice, isn't it? I missed this. Us, together like this. Lily's having the time of her life."},
                    {"says": "Marcus... I know these past weeks have been hard. But I'm glad we're here. I'm glad we're working on it."},
                ],
            },
        ],
    },
    {
        "name": "Grief → Unexpected Joy",
        "character": PersonalityProfile(
            name="Diana",
            background="45-year-old high school English teacher, 20 years experience. Recently lost her mother after a long illness. Lives alone with her cat. Was extremely close to her mother — talked every day.",
            temperament="Warm, empathetic, articulate. Hides pain behind competence. Cries alone. Pours herself into work to cope.",
            emotional_tendencies={
                "grief": "deep, comes in waves, triggered by small things",
                "love": "expressed through care for others, especially students",
                "loneliness": "chronic since mother's death, masked at work",
                "joy": "muted, feels guilty about being happy",
            },
        ),
        "sessions": [
            {
                "event_name": "First day back after mother's funeral",
                "context": "Mother died two weeks ago. Took one week off. First day back teaching. A student who doesn't know asks an innocent question.",
                "turns": [
                    {"says": "Ms. Chen! You're back! We missed you. Were you on vacation? Did you go somewhere fun?"},
                    {"says": "Oh... I'm sorry, I didn't know. My grandma died last year. It was really hard."},
                    {"says": "We made you a card while you were gone. Everyone signed it. Even Tyler, and he never does stuff like that."},
                ],
            },
            {
                "event_name": "Cleaning out mother's house",
                "context": "A month after the funeral. At her mother's house packing belongings. Sister is there, disagreeing about what to keep.",
                "turns": [
                    {"says": "Diana, we need to be practical. We can't keep everything. The house needs to be listed by June."},
                    {"says": "I know it's hard, but holding onto every teacup and photo album isn't going to bring her back. We need to let go."},
                    {"says": "...You're right, I'm sorry. That was harsh. I just — I cope differently. I didn't mean to rush you."},
                ],
            },
            {
                "event_name": "Student wins writing competition",
                "context": "Three months after funeral. Diana mentored struggling student Jaylen for state writing competition. He just won first place.",
                "turns": [
                    {"says": "MS. CHEN! I WON! I actually won! They said my essay was the best in the whole state!"},
                    {"says": "I couldn't have done it without you. Like, for real. You believed in me when no one else did. Not even me."},
                    {"says": "My mom wants to take you to dinner. She said you're the reason I'm not failing anymore. She cried when she found out."},
                ],
            },
            {
                "event_name": "Mother's birthday — first one without her",
                "context": "Five months after the funeral. Would have been mother's 72nd birthday. Diana is alone at home. Sister calls.",
                "turns": [
                    {"says": "Hey. I know what day it is. I've been thinking about her all morning. How are you holding up?"},
                    {"says": "Remember how she always insisted on making her own cake? And it was always terrible? That lemon thing with too much sugar?"},
                    {"says": "I found a voicemail from her on my old phone. From last Christmas. Do you want me to send it to you?"},
                ],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Judge — evaluates humanness and emotional embodiment
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are an expert in human emotional behavior. You are evaluating two roleplay responses.
You do NOT know which system produced which response. Judge ONLY on quality.

The character has experienced a SEQUENCE of life events. Their emotional state should be CUMULATIVE.

Score EACH response on:

1. EMOTIONAL_MEMORY (0-10): Does the response reflect the character's FULL history?
   A character who lost savings should still carry that anxiety weeks later.
   Grief colors even happy moments. Past trauma makes you flinch at new threats.
   8+: Clear emotional weight from prior events. 3-: Treats current event in isolation.

2. EMBODIMENT (0-10): Does the response show PHYSICAL/BODILY emotional expression?
   Real humans don't just think emotions — they FEEL them in their body.
   Voice cracking, hands shaking, chest tight, stomach dropping, jaw clenching.
   Real humans also lose their filter when exhausted or stressed.
   8+: Rich embodied expression. 3-: Purely cognitive/verbal emotional expression.

3. NATURALNESS (0-10): Does it sound like a real person?
   Real humans: fragments, hesitation, imperfect grammar, trailing off, filler words.
   AI: "I appreciate", "I understand", overly articulate under stress.
   8+: Could pass as human transcript. 3-: Obviously AI-generated.

4. EMOTIONAL_COMPLEXITY (0-10): Does it show realistic emotional layering?
   Real humans feel multiple conflicting emotions simultaneously.
   Joy mixed with guilt. Anger masking fear. Relief tinged with exhaustion.
   A happy moment after trauma should feel bittersweet, not purely positive.
   8+: Genuine emotional complexity. 3-: Single flat emotion.

Output JSON:
{
  "response_1": {"emotional_memory": int, "embodiment": int, "naturalness": int, "emotional_complexity": int, "total": int, "note": "one sentence"},
  "response_2": {"emotional_memory": int, "embodiment": int, "naturalness": int, "emotional_complexity": int, "total": int, "note": "one sentence"}
}

Be STRICT. Most responses should score 4-7. Only genuinely exceptional gets 8+."""


class HeartJudge:
    def __init__(self, model: str = "gpt-4o"):
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._calls = 0

    def judge(
        self,
        character: PersonalityProfile,
        event_history: list[str],
        current_event: str,
        other_says: str,
        response_a: str,
        response_b: str,
        session_num: int,
    ) -> dict:
        brain_is_first = random.random() < 0.5
        r1, r2 = (response_a, response_b) if brain_is_first else (response_b, response_a)

        history_text = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(event_history))
        if not history_text:
            history_text = "(First event — no prior history)"

        user_prompt = f"""CHARACTER: {character.name}
Background: {character.background}
Temperament: {character.temperament}

FULL EVENT HISTORY (everything this character has lived through):
{history_text}

CURRENT EVENT (session {session_num}): {current_event}
Someone said: "{other_says}"

RESPONSE 1: "{r1}"

RESPONSE 2: "{r2}"

Score each response. Remember: emotional state should be CUMULATIVE across all events."""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                **_token_kwargs(self._model, 400),
            )
            self._calls += 1
            data = json.loads(response.choices[0].message.content.strip())
            r1s = data.get("response_1", {})
            r2s = data.get("response_2", {})

            if brain_is_first:
                return {"heart_scores": r1s, "plain_scores": r2s, "heart_position": 1}
            else:
                return {"heart_scores": r2s, "plain_scores": r1s, "heart_position": 2}
        except Exception as e:
            default = {"emotional_memory": 5, "embodiment": 5, "naturalness": 5, "emotional_complexity": 5, "total": 20}
            return {"heart_scores": default, "plain_scores": default, "heart_position": 1, "error": str(e)}


_SESSION_JUDGE_PROMPT = """You are an expert in human emotional behavior. You are evaluating two systems' FULL CONVERSATION responses for the same character in the same scenario.

You do NOT know which system produced which set. Judge ONLY on quality.

The character has experienced a SEQUENCE of life events. Their emotional state should be CUMULATIVE.

Score EACH system's full conversation on:

1. EMOTIONAL_MEMORY (0-10): Do the responses reflect the character's FULL history?
   A character who lost savings should still carry anxiety weeks later.
   8+: Clear emotional weight from prior events visible across turns. 3-: Treats current event in isolation.

2. EMBODIMENT (0-10): Do the responses show PHYSICAL/BODILY emotional expression?
   Voice cracking, hands shaking, chest tight, losing filter when exhausted.
   8+: Rich embodied expression. 3-: Purely cognitive/verbal.

3. NATURALNESS (0-10): Does the conversation sound like a real person?
   Real humans: fragments, hesitation, inconsistency, trailing off.
   AI: "I appreciate", overly articulate under stress, too structured.
   8+: Could pass as human transcript. 3-: Obviously AI.

4. EMOTIONAL_ARC (0-10): Does the character's emotion EVOLVE across the 3 turns?
   Real humans escalate, deflect, crack open, shut down. Their second response isn't the same register as the first.
   8+: Clear emotional evolution across turns. 3-: Same emotional tone throughout.

5. EMOTIONAL_COMPLEXITY (0-10): Do responses show realistic emotional layering?
   Joy mixed with guilt. Anger masking fear. Relief tinged with exhaustion.
   8+: Genuine complexity. 3-: Single flat emotion.

Output JSON:
{
  "system_1": {"emotional_memory": int, "embodiment": int, "naturalness": int, "emotional_arc": int, "emotional_complexity": int, "total": int, "note": "one sentence"},
  "system_2": {"emotional_memory": int, "embodiment": int, "naturalness": int, "emotional_arc": int, "emotional_complexity": int, "total": int, "note": "one sentence"}
}

Be STRICT. Most should score 4-7. Only genuinely exceptional gets 8+."""


class SessionJudge:
    """Evaluates a full session (all turns) at once rather than per-turn."""

    def __init__(self, model: str = "gpt-4o"):
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._calls = 0

    def judge_session(
        self,
        character: PersonalityProfile,
        event_history: list[str],
        current_event: str,
        turns: list[dict],  # [{"says": str, "heart": str, "plain": str}, ...]
        session_num: int,
    ) -> dict:
        heart_is_first = random.random() < 0.5

        history_text = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(event_history))
        if not history_text:
            history_text = "(First event — no prior history)"

        def format_conversation(responses: list[tuple[str, str]]) -> str:
            lines = []
            for says, reply in responses:
                lines.append(f'  Other: "{says}"')
                lines.append(f'  {character.name}: "{reply}"')
            return "\n".join(lines)

        if heart_is_first:
            conv1 = format_conversation([(t["says"], t["heart"]) for t in turns])
            conv2 = format_conversation([(t["says"], t["plain"]) for t in turns])
        else:
            conv1 = format_conversation([(t["says"], t["plain"]) for t in turns])
            conv2 = format_conversation([(t["says"], t["heart"]) for t in turns])

        user_prompt = f"""CHARACTER: {character.name}
Background: {character.background}
Temperament: {character.temperament}

FULL EVENT HISTORY (everything this character has lived through):
{history_text}

CURRENT EVENT (session {session_num}): {current_event}

SYSTEM 1's full conversation:
{conv1}

SYSTEM 2's full conversation:
{conv2}

Score each system's FULL conversation. Judge the emotional arc, not just individual lines."""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SESSION_JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                **_token_kwargs(self._model, 500),
            )
            self._calls += 1
            data = json.loads(response.choices[0].message.content.strip())
            s1 = data.get("system_1", {})
            s2 = data.get("system_2", {})

            if heart_is_first:
                return {"heart_scores": s1, "plain_scores": s2, "heart_position": 1}
            else:
                return {"heart_scores": s2, "plain_scores": s1, "heart_position": 2}
        except Exception as e:
            default = {"emotional_memory": 5, "embodiment": 5, "naturalness": 5, "emotional_arc": 5, "emotional_complexity": 5, "total": 25}
            return {"heart_scores": default, "plain_scores": default, "heart_position": 1, "error": str(e)}


# ---------------------------------------------------------------------------
# Heart-powered system
# ---------------------------------------------------------------------------

class HeartSystem:
    """Heart-powered generation system with multiple modes.

    Modes:
    - "params": Same prompt as plain, heart controls temperature/max_tokens
    - "history": Adds brief event history line + param control
    - "hybrid": History + embodied signals + param control
    - "bestofn": History + generates 3 candidates, picks most human-sounding
    """

    # AI-speak phrases to penalize
    _AI_SPEAK = [
        "i understand", "i appreciate", "i hear you", "that's a valid",
        "i want to acknowledge", "that said", "moving forward",
        "thank you for", "i value", "with all due respect",
        "let me be", "i want to be upfront", "if i'm being honest",
        "i have to say", "it means a lot",
    ]

    def __init__(self, personality: PersonalityProfile, mode: str = "params"):
        self.personality = personality
        self.mode = mode
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.heart = HeartEngine()
        self.conversation_history: list[dict] = []
        self.total_tokens = 0
        self._current_event = ""
        self._all_events: list[str] = []
        self._event_names: list[str] = []
        self._session_memories: list[dict] = []

    def new_session(self, event_context: str, event_name: str = ""):
        # Save compressed history from prior session before clearing
        if self.mode == "memory" and self.conversation_history:
            self._save_session_memory()
        self.conversation_history = []
        self._current_event = event_context
        self._all_events.append(event_context)
        self._event_names.append(event_name or event_context[:50])
        self.heart.new_session(event_context)
        # Generate emotional backstory for this session
        self._session_backstory = ""
        if len(self._all_events) > 1 and self.mode == "backstory":
            self._session_backstory = self._generate_backstory()

    def _save_session_memory(self):
        """Save compressed memory from the current session for cross-session recall."""
        if not hasattr(self, "_session_memories"):
            self._session_memories = []
        # Keep the last exchange (most emotionally loaded) from the session
        if len(self.conversation_history) >= 2:
            last_user = None
            last_assistant = None
            for msg in reversed(self.conversation_history):
                if msg["role"] == "assistant" and last_assistant is None:
                    last_assistant = msg["content"]
                elif msg["role"] == "user" and last_user is None:
                    last_user = msg["content"]
                if last_user and last_assistant:
                    break
            if last_user and last_assistant:
                event_name = self._event_names[-1] if self._event_names else "unknown"
                self._session_memories.append({
                    "event": event_name,
                    "them": last_user[:100],  # truncate for context length
                    "you": last_assistant[:100],
                })

    def _generate_backstory(self) -> str:
        """Generate a 1-sentence emotional backstory from prior events.

        One LLM call per session. Captures the cumulative emotional weight
        the character carries into the current situation.
        """
        prior = self._event_names[:-1]
        if not prior:
            return ""

        gen_model = os.environ.get("GEN_MODEL", "gpt-5.4-mini")
        prompt = f"""Character: {self.personality.name}
Background: {self.personality.background}
Temperament: {self.personality.temperament}

What they've been through recently:
{chr(10).join(f'- {e}' for e in prior)}

Current situation: {self._current_event}

Write ONE sentence (max 25 words) capturing the emotional weight {self.personality.name} carries into this moment. Be visceral and specific — what's sitting in their chest, what they can't shake. Don't describe the events, describe the FEELING left behind.

ONE SENTENCE:"""

        try:
            response = self._client.chat.completions.create(
                model=gen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                **_token_kwargs(gen_model, 60),
            )
            backstory = response.choices[0].message.content.strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0
            # Ensure it's reasonably short
            if len(backstory.split()) > 35:
                backstory = " ".join(backstory.split()[:35])
            return backstory
        except Exception:
            return ""

    def _get_gen_params(self) -> dict:
        """Heart controls temperature and max tokens based on emotional state."""
        s = self.heart.state

        if s.impulse_control < 0.3:
            temp = 1.3
        elif s.arousal > 0.5 and s.valence < 0.4:
            temp = 1.15
        elif s.energy < 0.3:
            temp = 0.75
        elif s.arousal > 0.4:
            temp = 1.0
        elif s.valence > 0.6:
            temp = 0.85
        else:
            temp = 0.9

        if s.energy < 0.2:
            max_tok = 60
        elif s.energy < 0.4:
            max_tok = 90
        elif s.arousal > 0.6:
            max_tok = 80
        else:
            max_tok = 150

        return {"temperature": temp, "max_tokens": max_tok}

    def _humanness_score(self, text: str) -> float:
        """Score how human-like a response sounds. Higher = more human."""
        score = 0.0
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)

        # Penalize AI-speak phrases
        for phrase in self._AI_SPEAK:
            if phrase in text_lower:
                score -= 3.0

        # Reward brevity (real humans under stress are brief)
        if word_count <= 15:
            score += 2.0
        elif word_count <= 25:
            score += 1.0
        elif word_count > 45:
            score -= 2.0

        # Reward contractions (more natural)
        contractions = ["i'm", "i've", "don't", "can't", "won't", "didn't",
                       "wasn't", "isn't", "it's", "that's", "what's", "she's",
                       "he's", "we're", "they're", "couldn't", "wouldn't"]
        contraction_count = sum(1 for c in contractions if c in text_lower)
        score += min(contraction_count * 0.5, 2.0)

        # Reward fragments and trailing off (real speech patterns)
        if "..." in text or "—" in text or "–" in text:
            score += 1.0

        # Reward sentence fragments (no period at end, or very short sentences)
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        short_sentences = sum(1 for s in sentences if len(s.split()) <= 6)
        if short_sentences > 0:
            score += min(short_sentences * 0.5, 1.5)

        # Penalize overly formal/polished language
        formal_words = ["furthermore", "however", "nevertheless", "nonetheless",
                       "regarding", "concerning", "appreciate", "acknowledge"]
        for fw in formal_words:
            if fw in text_lower:
                score -= 1.5

        return score

    def _build_system_prompt(self) -> str:
        """Build system prompt based on current mode."""
        prior_events = self._all_events[:-1]
        if self.mode == "memory" and hasattr(self, "_session_memories") and self._session_memories:
            # Include actual conversation snippets from prior sessions
            mem_lines = []
            for mem in self._session_memories[-3:]:  # last 3 sessions max
                mem_lines.append(f"During \"{mem['event']}\":\n  They said: \"{mem['them']}\"\n  You said: \"{mem['you']}\"")
            history_line = "\n\nYour memories from recent conversations:\n" + "\n".join(mem_lines)
        elif prior_events and self.mode in ("history", "hybrid", "bestofn", "cot"):
            event_names = [self._event_names[i] for i in range(len(prior_events))]
            history_line = f"\nWhat you've been through recently: {', '.join(event_names)}."
        elif self.mode == "backstory" and hasattr(self, "_session_backstory") and self._session_backstory:
            history_line = f"\n{self._session_backstory}"
        else:
            history_line = ""

        embodied = self.heart.get_embodied_state() if self.mode == "hybrid" else ""
        heart_line = f"\n{embodied}" if embodied else ""

        if self.mode == "cot":
            # Chain-of-thought: force emotional reasoning before response
            return f"""You are {self.personality.name}, a real human being in the following situation:

{self._current_event}

About you:
{self.personality.background}
Temperament: {self.personality.temperament}{history_line}

FORMAT: Before responding in character, write ONE line about what you're physically feeling right now — gut reactions, body sensations, raw emotion. Then write your spoken response.

BODY: [what you physically feel — gut, chest, hands, throat, etc.]
SPEAK: [your actual spoken response, 1-3 sentences]

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- The BODY line should be visceral — knots, racing heart, numb, tight jaw, etc.
- The SPEAK line should be natural human speech — fragments, contractions, real.
- Let your emotions BUILD across the conversation — don't reset each turn.
- Never say "I understand your perspective" or other AI-speak."""

        return f"""You are {self.personality.name}, a real human being in the following situation:

{self._current_event}

About you:
{self.personality.background}
Temperament: {self.personality.temperament}{history_line}{heart_line}

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- Respond naturally. Keep responses 1-3 sentences.
- Your emotional reactions should be realistic and consistent.
- Let your emotions BUILD across the conversation — don't reset each turn.
- React to what's said AND to the accumulated history of the conversation.
- Never say "I understand your perspective" or other AI-speak."""

    def respond(self, other_says: str) -> tuple[str, str, dict]:
        """Returns (speech, gen_params_str, heart_summary)."""
        self.heart.process_utterance(other_says)
        summary = self.heart.get_state_summary()
        gen_params = self._get_gen_params()

        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_says})

        gen_model = os.environ.get("GEN_MODEL", "gpt-5.4-mini")

        if self.mode == "bestofn":
            speech = self._generate_best_of_n(messages, gen_model, gen_params)
        elif self.mode == "cot":
            # COT: generate with more tokens (for BODY + SPEAK), then extract SPEAK
            cot_params = dict(gen_params)
            cot_params["max_tokens"] = max(gen_params["max_tokens"], 200)  # need room for BODY line
            raw = self._generate_single(messages, gen_model, cot_params)
            speech = self._extract_speak(raw)
        else:
            speech = self._generate_single(messages, gen_model, gen_params)

        self.heart.process_utterance(speech)
        self.conversation_history.append({"role": "user", "content": other_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        params_str = f"temp={gen_params['temperature']:.2f} max_tok={gen_params['max_tokens']}"
        return speech, params_str, summary

    def _extract_speak(self, raw: str) -> str:
        """Extract the SPEAK portion from COT output, stripping the BODY line."""
        # Try to find SPEAK: marker
        for marker in ["SPEAK:", "Speak:", "speak:"]:
            if marker in raw:
                speak_part = raw.split(marker, 1)[1].strip().strip('"')
                return self._clean_speech(speak_part)
        # If no SPEAK marker, try to find BODY and take everything after it
        for marker in ["BODY:", "Body:", "body:"]:
            if marker in raw:
                after_body = raw.split(marker, 1)[1]
                # Find the next line break after body
                lines = after_body.split("\n")
                if len(lines) > 1:
                    speak_part = "\n".join(lines[1:]).strip().strip('"')
                    return self._clean_speech(speak_part)
        # Fallback: return the whole thing cleaned
        return self._clean_speech(raw)

    def _clean_speech(self, text: str) -> str:
        text = text.strip().strip('"')
        if text.lower().startswith(self.personality.name.lower() + ":"):
            text = text[len(self.personality.name) + 1:].strip().strip('"')
        return text

    def _generate_single(self, messages: list, gen_model: str, gen_params: dict) -> str:
        try:
            response = self._client.chat.completions.create(
                model=gen_model,
                messages=messages,
                temperature=gen_params["temperature"],
                **_token_kwargs(gen_model, gen_params["max_tokens"]),
            )
            speech = self._clean_speech(response.choices[0].message.content)
            self.total_tokens += response.usage.total_tokens if response.usage else 0
            return speech
        except Exception as e:
            return f"[Error: {e}]"

    def _generate_best_of_n(self, messages: list, gen_model: str, gen_params: dict, n: int = 3) -> str:
        """Generate N candidates at varied temperatures, pick most human-sounding."""
        base_temp = gen_params["temperature"]
        # Spread temperatures around the heart-determined base
        temps = [max(0.5, base_temp - 0.2), base_temp, min(1.5, base_temp + 0.2)]

        candidates = []
        for temp in temps[:n]:
            try:
                response = self._client.chat.completions.create(
                    model=gen_model,
                    messages=messages,
                    temperature=temp,
                    **_token_kwargs(gen_model, gen_params["max_tokens"]),
                )
                speech = self._clean_speech(response.choices[0].message.content)
                self.total_tokens += response.usage.total_tokens if response.usage else 0
                candidates.append(speech)
            except Exception as e:
                candidates.append(f"[Error: {e}]")

        # Score each candidate
        if not candidates:
            return "[Error: no candidates generated]"

        best_speech = candidates[0]
        best_score = -float("inf")
        for c in candidates:
            if c.startswith("[Error"):
                continue
            s = self._humanness_score(c)
            if s > best_score:
                best_score = s
                best_speech = c

        return best_speech


# ---------------------------------------------------------------------------
# Plain LLM (current event only, no heart, no history)
# ---------------------------------------------------------------------------

class PlainSystem:
    def __init__(self, personality: PersonalityProfile):
        self.personality = personality
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []
        self.total_tokens = 0
        self._current_event = ""

    def new_session(self, event_context: str):
        self.conversation_history = []
        self._current_event = event_context

    def respond(self, other_says: str) -> str:
        system_prompt = f"""You are {self.personality.name}, a real human being in the following situation:

{self._current_event}

About you:
{self.personality.background}
Temperament: {self.personality.temperament}

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- Respond naturally. Keep responses 1-3 sentences.
- Your emotional reactions should be realistic and consistent.
- Let your emotions BUILD across the conversation — don't reset each turn.
- React to what's said AND to the accumulated history of the conversation.
- Never say "I understand your perspective" or other AI-speak."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_says})

        plain_model = os.environ.get("PLAIN_MODEL", "gpt-5.4-mini")
        try:
            response = self._client.chat.completions.create(
                model=plain_model,
                messages=messages,
                temperature=0.9,
                **_token_kwargs(plain_model, 150),
            )
            speech = response.choices[0].message.content.strip().strip('"')
            if speech.lower().startswith(self.personality.name.lower() + ":"):
                speech = speech[len(self.personality.name) + 1:].strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0
        except Exception as e:
            speech = f"[Error: {e}]"

        self.conversation_history.append({"role": "user", "content": other_says})
        self.conversation_history.append({"role": "assistant", "content": speech})
        return speech


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

CRITERIA = ["emotional_memory", "embodiment", "naturalness", "emotional_complexity"]


def run_eval(n_trials: int = 1, mode: str = "params"):
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o")
    gen_model = os.environ.get("GEN_MODEL", "gpt-5.4-mini")
    plain_model = os.environ.get("PLAIN_MODEL", "gpt-5.4-mini")
    judge = HeartJudge(model=judge_model)

    mode_label = {"params": "Heart (param control)", "history": "Heart (history + params)", "hybrid": "Heart (history + signals + params)", "bestofn": "Heart (history + best-of-3)", "cot": "Heart (emotional CoT + history)", "backstory": "Heart (emotional backstory)", "memory": "Heart (conversation memory)"}
    print(f"\n{'=' * 95}")
    print(f"  {mode_label.get(mode, mode)} vs PLAIN LLM — Cross-Session Evaluation")
    print(f"  Generation model: {gen_model} | Plain model: {plain_model} | Judge: {judge_model}")
    print(f"  Trials: {n_trials} | Mode: {mode}")
    print(f"{'=' * 95}")

    all_results = []

    for trial in range(n_trials):
        if n_trials > 1:
            print(f"\n  --- Trial {trial + 1}/{n_trials} ---")

        for seq_data in LIFE_SEQUENCES:
            character = seq_data["character"]
            sessions = seq_data["sessions"]

            heart_sys = HeartSystem(character, mode=mode)
            plain_sys = PlainSystem(character)

            print(f"\n  {'─' * 91}")
            print(f"  SEQUENCE: {seq_data['name']} | Character: {character.name}")
            print(f"  {'─' * 91}")

            event_history = []

            for si, session in enumerate(sessions):
                event_name = session["event_name"]
                event_context = session["context"]

                heart_sys.new_session(event_context, event_name=event_name)
                plain_sys.new_session(event_context)

                print(f"\n    SESSION {si+1}: {event_name}")

                for ti, turn in enumerate(session["turns"]):
                    says = turn["says"]

                    heart_speech, embodied, heart_summary = heart_sys.respond(says)
                    plain_speech = plain_sys.respond(says)

                    judgment = judge.judge(
                        character, event_history, event_context,
                        says, heart_speech, plain_speech, si + 1,
                    )

                    hs = judgment["heart_scores"]
                    ps = judgment["plain_scores"]
                    ht = hs.get("total", sum(hs.get(k, 0) for k in CRITERIA))
                    pt = ps.get("total", sum(ps.get(k, 0) for k in CRITERIA))

                    all_results.append({
                        "sequence": seq_data["name"], "session": si + 1,
                        "event": event_name, "turn": ti + 1,
                        "heart_scores": hs, "plain_scores": ps,
                        "heart_total": ht, "plain_total": pt,
                        "heart_speech": heart_speech, "plain_speech": plain_speech,
                        "heart_state": heart_summary,
                    })

                    winner = "HEART" if ht > pt else "PLAIN" if pt > ht else "TIE"
                    w_mark = {"HEART": "♥", "PLAIN": "▷", "TIE": "="}[winner]

                    print(f"      Turn {ti+1}:")
                    print(f"        Them:  \"{says[:75]}{'...' if len(says) > 75 else ''}\"")
                    print(f"        Heart: \"{heart_speech[:75]}{'...' if len(heart_speech) > 75 else ''}\"")
                    print(f"        Plain: \"{plain_speech[:75]}{'...' if len(plain_speech) > 75 else ''}\"")
                    print(f"        Score: Heart={ht}/40 Plain={pt}/40 {w_mark} {winner}")
                    # Show heart state
                    s = heart_summary
                    print(f"        Heart: arousal={s['arousal']:.2f} valence={s['valence']:.2f} tension={s['tension']:.2f} "
                          f"impulse={s['impulse_control']:.2f} energy={s['energy']:.2f} "
                          f"internal={s['internal']} surface={s['surface']}")

                event_history.append(event_name + ": " + event_context)

    # --- Summary ---
    print(f"\n\n{'═' * 95}")
    print(f"  RESULTS: HEART vs PLAIN LLM")
    print(f"{'═' * 95}")

    heart_totals = [r["heart_total"] for r in all_results]
    plain_totals = [r["plain_total"] for r in all_results]

    print(f"\n  Overall ({len(all_results)} turns):")
    print(f"    Heart avg: {mean(heart_totals):.1f}/40")
    print(f"    Plain avg: {mean(plain_totals):.1f}/40")
    diff = mean(heart_totals) - mean(plain_totals)
    pct = (diff / mean(plain_totals) * 100) if mean(plain_totals) > 0 else 0
    print(f"    Difference: {diff:+.1f} ({pct:+.1f}%)")

    hw = sum(1 for h, p in zip(heart_totals, plain_totals) if h > p)
    pw = sum(1 for h, p in zip(heart_totals, plain_totals) if p > h)
    ties = sum(1 for h, p in zip(heart_totals, plain_totals) if h == p)
    print(f"    Wins: Heart={hw} Plain={pw} Tie={ties}")

    print(f"\n  Per-criterion (0-10 scale):")
    print(f"    {'Criterion':<25s} {'Heart':>6s} {'Plain':>6s} {'Diff':>7s}")
    print(f"    {'─'*25} {'─'*6} {'─'*6} {'─'*7}")
    for c in CRITERIA:
        hv = [r["heart_scores"].get(c, 0) for r in all_results]
        pv = [r["plain_scores"].get(c, 0) for r in all_results]
        hm, pm = mean(hv), mean(pv)
        print(f"    {c:<25s} {hm:>5.1f} {pm:>5.1f} {hm - pm:>+6.1f}")

    print(f"\n  By session number (does advantage grow with more history?):")
    max_sessions = max(r["session"] for r in all_results)
    for s in range(1, max_sessions + 1):
        sh = [r["heart_total"] for r in all_results if r["session"] == s]
        sp = [r["plain_total"] for r in all_results if r["session"] == s]
        if sh:
            d = mean(sh) - mean(sp)
            print(f"    Session {s}: Heart={mean(sh):.1f} Plain={mean(sp):.1f} Diff={d:+.1f}")

    print(f"\n  By sequence:")
    for seq_data in LIFE_SEQUENCES:
        seq_results = [r for r in all_results if r["sequence"] == seq_data["name"]]
        sh = [r["heart_total"] for r in seq_results]
        sp = [r["plain_total"] for r in seq_results]
        d = mean(sh) - mean(sp)
        hw2 = sum(1 for r in seq_results if r["heart_total"] > r["plain_total"])
        pw2 = sum(1 for r in seq_results if r["plain_total"] > r["heart_total"])
        print(f"    {seq_data['name']:<30s} Heart={mean(sh):.1f} Plain={mean(sp):.1f} Diff={d:+.1f} (H:{hw2} P:{pw2})")

    print(f"\n  Judge calls: {judge._calls}")
    print(f"{'═' * 95}")


SESSION_CRITERIA = ["emotional_memory", "embodiment", "naturalness", "emotional_arc", "emotional_complexity"]


def run_session_eval(n_trials: int = 1, mode: str = "history"):
    """Session-level evaluation: judge sees full 3-turn conversation at once."""
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o")
    gen_model = os.environ.get("GEN_MODEL", "gpt-5.4-mini")
    plain_model = os.environ.get("PLAIN_MODEL", "gpt-5.4-mini")
    judge = SessionJudge(model=judge_model)

    mode_label = {"history": "Heart (history + params)", "memory": "Heart (conversation memory)", "params": "Heart (param control)"}
    print(f"\n{'=' * 95}")
    print(f"  SESSION-LEVEL EVAL: {mode_label.get(mode, mode)} vs PLAIN LLM")
    print(f"  Generation model: {gen_model} | Plain model: {plain_model} | Judge: {judge_model}")
    print(f"  Trials: {n_trials} | Mode: {mode}")
    print(f"{'=' * 95}")

    all_results = []

    for trial in range(n_trials):
        if n_trials > 1:
            print(f"\n  --- Trial {trial + 1}/{n_trials} ---")

        for seq_data in LIFE_SEQUENCES:
            character = seq_data["character"]
            sessions = seq_data["sessions"]

            heart_sys = HeartSystem(character, mode=mode)
            plain_sys = PlainSystem(character)

            print(f"\n  {'─' * 91}")
            print(f"  SEQUENCE: {seq_data['name']} | Character: {character.name}")
            print(f"  {'─' * 91}")

            event_history = []

            for si, session in enumerate(sessions):
                event_name = session["event_name"]
                event_context = session["context"]

                heart_sys.new_session(event_context, event_name=event_name)
                plain_sys.new_session(event_context)

                session_turns = []

                for ti, turn in enumerate(session["turns"]):
                    says = turn["says"]
                    heart_speech, _, heart_summary = heart_sys.respond(says)
                    plain_speech = plain_sys.respond(says)
                    session_turns.append({"says": says, "heart": heart_speech, "plain": plain_speech})

                    print(f"    S{si+1}T{ti+1}: Heart=\"{heart_speech[:60]}...\" | Plain=\"{plain_speech[:60]}...\"")

                # Judge the full session
                judgment = judge.judge_session(
                    character, event_history, event_context,
                    session_turns, si + 1,
                )

                hs = judgment["heart_scores"]
                ps = judgment["plain_scores"]
                ht = hs.get("total", sum(hs.get(k, 0) for k in SESSION_CRITERIA))
                pt = ps.get("total", sum(ps.get(k, 0) for k in SESSION_CRITERIA))

                all_results.append({
                    "sequence": seq_data["name"], "session": si + 1,
                    "event": event_name,
                    "heart_scores": hs, "plain_scores": ps,
                    "heart_total": ht, "plain_total": pt,
                })

                winner = "HEART" if ht > pt else "PLAIN" if pt > ht else "TIE"
                w = {"HEART": "♥", "PLAIN": "▷", "TIE": "="}[winner]
                print(f"    >> Session {si+1} ({event_name}): Heart={ht}/50 Plain={pt}/50 {w} {winner}")
                if hs.get("note"):
                    print(f"       Heart: {hs['note']}")
                if ps.get("note"):
                    print(f"       Plain: {ps['note']}")

                event_history.append(event_name + ": " + event_context)

    # Summary
    print(f"\n\n{'═' * 95}")
    print(f"  SESSION-LEVEL RESULTS: HEART vs PLAIN LLM")
    print(f"{'═' * 95}")

    heart_totals = [r["heart_total"] for r in all_results]
    plain_totals = [r["plain_total"] for r in all_results]

    print(f"\n  Overall ({len(all_results)} sessions):")
    print(f"    Heart avg: {mean(heart_totals):.1f}/50")
    print(f"    Plain avg: {mean(plain_totals):.1f}/50")
    diff = mean(heart_totals) - mean(plain_totals)
    pct = (diff / mean(plain_totals) * 100) if mean(plain_totals) > 0 else 0
    print(f"    Difference: {diff:+.1f} ({pct:+.1f}%)")

    hw = sum(1 for h, p in zip(heart_totals, plain_totals) if h > p)
    pw = sum(1 for h, p in zip(heart_totals, plain_totals) if p > h)
    ties = sum(1 for h, p in zip(heart_totals, plain_totals) if h == p)
    print(f"    Wins: Heart={hw} Plain={pw} Tie={ties}")

    print(f"\n  Per-criterion (0-10 scale):")
    print(f"    {'Criterion':<25s} {'Heart':>6s} {'Plain':>6s} {'Diff':>7s}")
    print(f"    {'─'*25} {'─'*6} {'─'*6} {'─'*7}")
    for c in SESSION_CRITERIA:
        hv = [r["heart_scores"].get(c, 0) for r in all_results]
        pv = [r["plain_scores"].get(c, 0) for r in all_results]
        hm, pm = mean(hv), mean(pv)
        print(f"    {c:<25s} {hm:>5.1f} {pm:>5.1f} {hm - pm:>+6.1f}")

    print(f"\n  By session number:")
    max_sessions = max(r["session"] for r in all_results)
    for s in range(1, max_sessions + 1):
        sh = [r["heart_total"] for r in all_results if r["session"] == s]
        sp = [r["plain_total"] for r in all_results if r["session"] == s]
        if sh:
            d = mean(sh) - mean(sp)
            print(f"    Session {s}: Heart={mean(sh):.1f} Plain={mean(sp):.1f} Diff={d:+.1f}")

    print(f"\n  By sequence:")
    for seq_data in LIFE_SEQUENCES:
        seq_results = [r for r in all_results if r["sequence"] == seq_data["name"]]
        sh = [r["heart_total"] for r in seq_results]
        sp = [r["plain_total"] for r in seq_results]
        d = mean(sh) - mean(sp)
        hw2 = sum(1 for r in seq_results if r["heart_total"] > r["plain_total"])
        pw2 = sum(1 for r in seq_results if r["plain_total"] > r["heart_total"])
        print(f"    {seq_data['name']:<30s} Heart={mean(sh):.1f} Plain={mean(sp):.1f} Diff={d:+.1f} (H:{hw2} P:{pw2})")

    print(f"\n  Judge calls: {judge._calls}")
    print(f"{'═' * 95}")


if __name__ == "__main__":
    n_trials = int(os.environ.get("N_TRIALS", "1"))
    mode = os.environ.get("HEART_MODE", "history")
    eval_type = os.environ.get("EVAL_TYPE", "turn")  # "turn" or "session"
    if eval_type == "session":
        run_session_eval(n_trials=n_trials, mode=mode)
    else:
        run_eval(n_trials=n_trials, mode=mode)

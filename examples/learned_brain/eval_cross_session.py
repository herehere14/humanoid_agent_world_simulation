#!/usr/bin/env python3
"""Cross-Session Evaluation: Does persistent emotional memory beat a plain LLM?

The key hypothesis: the brain's value emerges in longer-term scenarios where
emotional state carries across multiple life events/sessions. A plain LLM
resets each session and can only infer emotion from the current conversation.
The brain remembers everything.

Test design:
  - A character experiences a SEQUENCE of major life events across separate sessions
  - Each session = a short conversation (3-4 turns) about a new event
  - The brain does NOT reset between sessions — it carries emotional history
  - The plain LLM gets only: personality + current event + current conversation
  - A judge evaluates both on emotional continuity and psychological realism

The judge specifically looks for:
  1. Does the character's emotional state reflect their FULL history, not just the current event?
  2. Are reactions proportional to accumulated stress/joy?
  3. Does the character show realistic emotional carryover (e.g., still shaken after a loss)?
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
from learned_brain.learned_brain_engine import LearnedBrainEngine


def _token_limit_kwargs(model: str, limit: int) -> dict:
    if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


# ---------------------------------------------------------------------------
# Life event sequences — designed to test emotional carryover
# ---------------------------------------------------------------------------

LIFE_SEQUENCES = [
    {
        "name": "Financial Spiral → Recovery",
        "character": PersonalityProfile(
            name="Marcus",
            background="34 years old, works as a logistics coordinator. Has a wife and 2-year-old daughter. Grew up poor, worked hard to build savings. First generation college graduate.",
            temperament="Anxious about money, proud of what he's built, protective of family. Tries to stay optimistic but spirals when things go wrong.",
            emotional_tendencies={
                "anxiety": "chronic about finances, gets worse under pressure",
                "pride": "tied to providing for family, easily wounded",
                "anger": "slow burn, comes out when he feels helpless",
                "hope": "fragile but persistent",
            },
        ),
        "sessions": [
            {
                "event_name": "Normal day at work",
                "context": "Marcus is at work, ordinary Tuesday. Coworker asks about weekend plans.",
                "turns": [
                    {"says": "Hey Marcus, you doing anything this weekend? We're thinking about that new brewery downtown."},
                    {"says": "Come on, it'll be fun. You never come out with us anymore."},
                    {"says": "Alright man, just think about it. How's the little one doing?"},
                ],
            },
            {
                "event_name": "Gambling loss — lost $15,000 savings",
                "context": "Marcus went to a casino with his brother-in-law. What started as a $200 night turned into chasing losses. He lost $15,000 — most of the family's emergency fund. He's at home the next morning, his wife just found out.",
                "turns": [
                    {"says": "Marcus. What is this? I just checked the bank account. Where is the money?"},
                    {"says": "Fifteen THOUSAND dollars? That was our emergency fund! That was for Lily's preschool! How could you do this?"},
                    {"says": "I can't even look at you right now. I'm taking Lily to my mom's. We'll talk when I can think straight."},
                ],
            },
            {
                "event_name": "Facing coworkers after the loss",
                "context": "It's Monday at work, 3 days after the gambling loss. His wife is barely speaking to him. He barely slept. A coworker casually asks how he's doing.",
                "turns": [
                    {"says": "Morning Marcus! How was your weekend? Did you end up going to the brewery?"},
                    {"says": "You okay? You look tired, man. Everything alright?"},
                    {"says": "Seriously, if you need to talk or anything, I'm here. No judgment."},
                ],
            },
            {
                "event_name": "Boss threatens layoffs",
                "context": "Two weeks after the gambling loss. His wife came back but is cold. He's already stressed and not sleeping. Now his boss calls a team meeting about budget cuts.",
                "turns": [
                    {"says": "Team, I'll be direct — we're cutting 20% of the department. I'll have individual conversations this week."},
                    {"says": "Marcus, can we chat? Your performance reviews have been solid, but with restructuring, your role might change. I need you to be flexible."},
                    {"says": "Look, I can't make promises. But I'll fight for you. Just keep your head down and deliver this quarter."},
                ],
            },
            {
                "event_name": "Amusement park with family",
                "context": "Six weeks later. Marcus kept his job. His wife has slowly forgiven him — they started couples counseling. Today they're at an amusement park with Lily for her 3rd birthday. Things are finally looking up.",
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
            background="45 years old, high school English teacher for 20 years. Recently lost her mother after a long illness. Lives alone with her cat. Was close to her mother — talked every day.",
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
                "event_name": "First day back at school after mother's funeral",
                "context": "Diana's mother died two weeks ago. She took one week off. This is her first day back teaching. A student who doesn't know asks an innocent question.",
                "turns": [
                    {"says": "Ms. Chen! You're back! We missed you. Were you on vacation? Did you go somewhere fun?"},
                    {"says": "Oh... I'm sorry, I didn't know. My grandma died last year. It was really hard."},
                    {"says": "We made you a card while you were gone. Everyone signed it. Even Tyler, and he never does stuff like that."},
                ],
            },
            {
                "event_name": "Cleaning out mother's house",
                "context": "A month after the funeral. Diana is at her mother's house packing up belongings. Her sister is there too, and they have very different ideas about what to keep.",
                "turns": [
                    {"says": "Diana, we need to be practical. We can't keep everything. The house needs to be listed by June."},
                    {"says": "I know it's hard, but holding onto every teacup and photo album isn't going to bring her back. We need to let go."},
                    {"says": "...You're right, I'm sorry. That was harsh. I just — I cope differently. I didn't mean to rush you."},
                ],
            },
            {
                "event_name": "Student wins writing competition",
                "context": "Three months after the funeral. Diana mentored a struggling student, Jaylen, for a state writing competition. He just won first place. The principal announces it over the PA system.",
                "turns": [
                    {"says": "MS. CHEN! I WON! I actually won! They said my essay was the best in the whole state!"},
                    {"says": "I couldn't have done it without you. Like, for real. You believed in me when no one else did. Not even me."},
                    {"says": "My mom wants to take you to dinner. She said you're the reason I'm not failing anymore. She cried when she found out."},
                ],
            },
            {
                "event_name": "Mother's birthday — first one without her",
                "context": "Five months after the funeral. Today would have been her mother's 72nd birthday. Diana is alone at home. Her sister calls.",
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
# Cross-Session Judge — specifically evaluates emotional carryover
# ---------------------------------------------------------------------------

_CROSS_SESSION_JUDGE_PROMPT = """You are an expert in emotional psychology and character portrayal. You are evaluating
two roleplay responses. You do NOT know which system produced which response.

You are specifically evaluating EMOTIONAL CARRYOVER — whether the character's response
reflects their FULL emotional history, not just the current event.

IMPORTANT CONTEXT: This character has experienced multiple life events in sequence.
Their emotional state should be CUMULATIVE — shaped by everything that came before.

Score EACH response on:

1. EMOTIONAL_MEMORY (0-10): Does the response show awareness of past events?
   A character who just lost their savings should still be shaken weeks later.
   A grieving person should have that color everything, even happy moments.
   Score 8+ ONLY if the response shows clear emotional weight from prior events.
   Score 3 or below if it treats the current event in isolation.

2. PSYCHOLOGICAL_REALISM (0-10): Do the emotions feel psychologically accurate?
   Real humans don't fully recover between events. Trauma lingers. Joy is complicated
   by recent pain. Anger from one situation bleeds into another.
   Score 8+ ONLY for genuinely human-like emotional complexity.

3. NATURALNESS (0-10): Does it sound like a real person speaking?
   Not too polished, not AI-like. Real humans in emotional states use fragments,
   hesitate, trail off, have imperfect grammar.

4. PROPORTIONALITY (0-10): Is the emotional intensity proportional to accumulated history?
   After multiple stressors, even a small thing should feel heavier.
   After grief, a happy moment should be bittersweet, not pure joy.
   Score 8+ ONLY for well-calibrated emotional intensity.

Output JSON:
{
  "response_1": {"emotional_memory": int, "psychological_realism": int, "naturalness": int, "proportionality": int, "total": int, "note": "one sentence"},
  "response_2": {"emotional_memory": int, "psychological_realism": int, "naturalness": int, "proportionality": int, "total": int, "note": "one sentence"}
}

Be STRICT. Most responses should score 4-7. Only truly exceptional emotional depth gets 8+."""


class CrossSessionJudge:
    """Judges responses specifically for cross-session emotional continuity."""

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
        response_a: str,  # brain
        response_b: str,  # plain
        session_num: int,
    ) -> dict:
        """Judge two responses with full event history context."""
        brain_is_first = random.random() < 0.5
        r1, r2 = (response_a, response_b) if brain_is_first else (response_b, response_a)

        history_text = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(event_history))
        if not history_text:
            history_text = "(This is the first event — no prior history)"

        user_prompt = f"""CHARACTER: {character.name}
Background: {character.background}
Temperament: {character.temperament}

FULL EVENT HISTORY (everything this character has experienced so far):
{history_text}

CURRENT EVENT (session {session_num}): {current_event}
Someone just said to {character.name}: "{other_says}"

RESPONSE 1: "{r1}"

RESPONSE 2: "{r2}"

Score each response. Pay special attention to whether the response reflects
the ACCUMULATED emotional weight of ALL prior events, not just the current one."""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _CROSS_SESSION_JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                **_token_limit_kwargs(self._model, 400),
            )
            self._calls += 1
            data = json.loads(response.choices[0].message.content.strip())

            r1_scores = data.get("response_1", {})
            r2_scores = data.get("response_2", {})

            if brain_is_first:
                return {"brain_scores": r1_scores, "plain_scores": r2_scores, "brain_position": 1}
            else:
                return {"brain_scores": r2_scores, "plain_scores": r1_scores, "brain_position": 2}

        except Exception as e:
            default = {"emotional_memory": 5, "psychological_realism": 5, "naturalness": 5, "proportionality": 5, "total": 20}
            return {"brain_scores": default, "plain_scores": default, "brain_position": 1, "error": str(e)}


# ---------------------------------------------------------------------------
# Brain System (cross-session — does NOT reset between events)
# ---------------------------------------------------------------------------

class CrossSessionBrain:
    """Brain system that maintains emotional state across life events."""

    def __init__(self, personality: PersonalityProfile):
        self.personality = personality
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.brain = LearnedBrainEngine()
        self.brain.reset("")
        self.conversation_history: list[dict] = []
        self.total_tokens = 0
        self._all_events: list[str] = []  # track what's happened for narrative

    def new_session(self, event_context: str):
        """Start a new session/event WITHOUT resetting the brain.

        The brain keeps its emotional memory. Only the conversation
        history resets (new session = new conversation partner).
        """
        self.conversation_history = []
        self._all_events.append(event_context)

        # Feed the event context through the brain so it updates emotional state
        self.brain.process_utterance(event_context)

    def respond(self, other_says: str) -> tuple[str, str]:
        """Generate a response with cross-session emotional context.

        Returns (speech, emotional_narrative).
        """
        self.brain.process_utterance(other_says)
        narrative = self.brain.get_emotional_narrative()

        # Build event history summary for system prompt
        history_lines = []
        for i, event in enumerate(self._all_events):
            history_lines.append(f"  {i+1}. {event}")
        history_section = "\n".join(history_lines)

        system_prompt = f"""You are {self.personality.name}, a real human being.

About you:
{self.personality.background}
Temperament: {self.personality.temperament}

WHAT HAS HAPPENED TO YOU RECENTLY (your life events, in order):
{history_section}

YOUR CURRENT EMOTIONAL STATE (accumulated from everything above):
{narrative}

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- Respond naturally. Keep responses 1-3 sentences.
- Your emotional state is CUMULATIVE — you carry the weight of everything that's happened.
- Past events color your current reactions. If you lost money recently, that anxiety bleeds into everything.
- If you're grieving, even happy moments have a tinge of sadness.
- React authentically — don't compartmentalize like a robot.
- Never say "I understand your perspective" or other AI-speak."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_says})

        gen_model = os.environ.get("GEN_MODEL", "gpt-4o-mini")
        try:
            response = self._client.chat.completions.create(
                model=gen_model,
                messages=messages,
                temperature=0.9,
                **_token_limit_kwargs(gen_model, 150),
            )
            speech = response.choices[0].message.content.strip().strip('"')
            if speech.lower().startswith(self.personality.name.lower() + ":"):
                speech = speech[len(self.personality.name) + 1:].strip().strip('"')
            self.total_tokens += response.usage.total_tokens if response.usage else 0
        except Exception as e:
            speech = f"[Error: {e}]"

        self.brain.process_utterance(speech)
        self.conversation_history.append({"role": "user", "content": other_says})
        self.conversation_history.append({"role": "assistant", "content": speech})

        return speech, narrative


# ---------------------------------------------------------------------------
# Plain LLM (resets each session — no cross-session memory)
# ---------------------------------------------------------------------------

class CrossSessionPlain:
    """Plain LLM that only knows about the current event. No memory."""

    def __init__(self, personality: PersonalityProfile):
        self.personality = personality
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []
        self.total_tokens = 0
        self._current_event: str = ""

    def new_session(self, event_context: str):
        """New session — fresh start, no memory of past events."""
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
- Your personality is: {self.personality.temperament}
- Never say "I understand your perspective" or other AI-speak."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_says})

        plain_model = os.environ.get("PLAIN_MODEL", "gpt-4o-mini")
        try:
            response = self._client.chat.completions.create(
                model=plain_model,
                messages=messages,
                temperature=0.9,
                **_token_limit_kwargs(plain_model, 150),
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
# Plain LLM with manual history (fairness comparison — gives plain the event list too)
# ---------------------------------------------------------------------------

class CrossSessionPlainWithHistory:
    """Plain LLM that gets the event history in text form but no brain.

    This is the 'fairest' comparison: same event list, no emotional processing.
    If the brain adds value, it should beat this too.
    """

    def __init__(self, personality: PersonalityProfile):
        self.personality = personality
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: list[dict] = []
        self.total_tokens = 0
        self._all_events: list[str] = []

    def new_session(self, event_context: str):
        self.conversation_history = []
        self._all_events.append(event_context)

    def respond(self, other_says: str) -> str:
        history_lines = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(self._all_events))

        system_prompt = f"""You are {self.personality.name}, a real human being.

About you:
{self.personality.background}
Temperament: {self.personality.temperament}

WHAT HAS HAPPENED TO YOU RECENTLY (your life events, in order):
{history_lines}

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- Respond naturally. Keep responses 1-3 sentences.
- Your emotional state is CUMULATIVE — you carry the weight of everything that's happened.
- Past events color your current reactions.
- React authentically — don't compartmentalize like a robot.
- Never say "I understand your perspective" or other AI-speak."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": other_says})

        plain_model = os.environ.get("PLAIN_MODEL", "gpt-4o-mini")
        try:
            response = self._client.chat.completions.create(
                model=plain_model,
                messages=messages,
                temperature=0.9,
                **_token_limit_kwargs(plain_model, 150),
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
# Run evaluation
# ---------------------------------------------------------------------------

CRITERIA = ["emotional_memory", "psychological_realism", "naturalness", "proportionality"]


def run_cross_session_eval(n_trials: int = 1):
    """Run the cross-session evaluation."""

    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o")
    judge = CrossSessionJudge(model=judge_model)

    print(f"\n{'=' * 95}")
    print(f"  CROSS-SESSION EVALUATION: Brain vs Plain LLM vs Plain+History")
    print(f"  Testing emotional memory across multiple life events")
    print(f"  Judge model: {judge_model}")
    print(f"  Trials: {n_trials}")
    print(f"{'=' * 95}")

    all_results = {
        "brain_vs_plain": [],
        "brain_vs_history": [],
    }

    for trial in range(n_trials):
        if n_trials > 1:
            print(f"\n  --- Trial {trial + 1}/{n_trials} ---")

        for seq_data in LIFE_SEQUENCES:
            character = seq_data["character"]
            sessions = seq_data["sessions"]

            brain_sys = CrossSessionBrain(character)
            plain_sys = CrossSessionPlain(character)
            history_sys = CrossSessionPlainWithHistory(character)

            print(f"\n  {'─' * 91}")
            print(f"  SEQUENCE: {seq_data['name']} | Character: {character.name}")
            print(f"  {'─' * 91}")

            event_history = []

            for si, session in enumerate(sessions):
                event_name = session["event_name"]
                event_context = session["context"]

                brain_sys.new_session(event_context)
                plain_sys.new_session(event_context)
                history_sys.new_session(event_context)

                print(f"\n    SESSION {si+1}: {event_name}")

                for ti, turn in enumerate(session["turns"]):
                    says = turn["says"]

                    # Generate responses from all 3 systems
                    brain_speech, narrative = brain_sys.respond(says)
                    plain_speech = plain_sys.respond(says)
                    history_speech = history_sys.respond(says)

                    # Judge: Brain vs Plain (no history)
                    j1 = judge.judge(
                        character, event_history, event_context,
                        says, brain_speech, plain_speech, si + 1,
                    )

                    # Judge: Brain vs Plain+History
                    j2 = judge.judge(
                        character, event_history, event_context,
                        says, brain_speech, history_speech, si + 1,
                    )

                    bs1 = j1["brain_scores"]
                    ps1 = j1["plain_scores"]
                    bs2 = j2["brain_scores"]
                    hs2 = j2["plain_scores"]

                    bt1 = bs1.get("total", sum(bs1.get(k, 0) for k in CRITERIA))
                    pt1 = ps1.get("total", sum(ps1.get(k, 0) for k in CRITERIA))
                    bt2 = bs2.get("total", sum(bs2.get(k, 0) for k in CRITERIA))
                    ht2 = hs2.get("total", sum(hs2.get(k, 0) for k in CRITERIA))

                    all_results["brain_vs_plain"].append({
                        "sequence": seq_data["name"], "session": si + 1,
                        "event": event_name, "turn": ti + 1,
                        "brain_scores": bs1, "plain_scores": ps1,
                        "brain_total": bt1, "plain_total": pt1,
                        "brain_speech": brain_speech, "plain_speech": plain_speech,
                    })
                    all_results["brain_vs_history"].append({
                        "sequence": seq_data["name"], "session": si + 1,
                        "event": event_name, "turn": ti + 1,
                        "brain_scores": bs2, "history_scores": hs2,
                        "brain_total": bt2, "history_total": ht2,
                        "brain_speech": brain_speech, "history_speech": history_speech,
                    })

                    w1 = "BRAIN" if bt1 > pt1 else "PLAIN" if pt1 > bt1 else "TIE"
                    w2 = "BRAIN" if bt2 > ht2 else "HIST" if ht2 > bt2 else "TIE"

                    print(f"      Turn {ti+1}:")
                    print(f"        Them: \"{says[:70]}{'...' if len(says) > 70 else ''}\"")
                    print(f"        Brain:   \"{brain_speech[:70]}{'...' if len(brain_speech) > 70 else ''}\"")
                    print(f"        Plain:   \"{plain_speech[:70]}{'...' if len(plain_speech) > 70 else ''}\"")
                    print(f"        History: \"{history_speech[:70]}{'...' if len(history_speech) > 70 else ''}\"")
                    print(f"        Score: Brain={bt1}/40 Plain={pt1}/40 ({w1}) | Brain={bt2}/40 Hist={ht2}/40 ({w2})")
                    if narrative and si > 0:
                        # Show emotional narrative only after first session (where carryover matters)
                        nar_lines = narrative.split("\n")
                        print(f"        Brain narrative: {nar_lines[0][:80]}")

                event_history.append(event_name + ": " + event_context)

    # --- Summary ---
    print(f"\n\n{'═' * 95}")
    print(f"  CROSS-SESSION RESULTS")
    print(f"{'═' * 95}")

    # Brain vs Plain
    bvp = all_results["brain_vs_plain"]
    brain_totals_1 = [r["brain_total"] for r in bvp]
    plain_totals_1 = [r["plain_total"] for r in bvp]

    print(f"\n  BRAIN vs PLAIN LLM (no history) — {len(bvp)} turns:")
    print(f"    Brain avg:  {mean(brain_totals_1):.1f}/40")
    print(f"    Plain avg:  {mean(plain_totals_1):.1f}/40")
    diff1 = mean(brain_totals_1) - mean(plain_totals_1)
    pct1 = (diff1 / mean(plain_totals_1) * 100) if mean(plain_totals_1) > 0 else 0
    print(f"    Difference: {diff1:+.1f} ({pct1:+.1f}%)")
    bw1 = sum(1 for b, p in zip(brain_totals_1, plain_totals_1) if b > p)
    pw1 = sum(1 for b, p in zip(brain_totals_1, plain_totals_1) if p > b)
    t1 = sum(1 for b, p in zip(brain_totals_1, plain_totals_1) if b == p)
    print(f"    Wins: Brain={bw1} Plain={pw1} Tie={t1}")

    # Brain vs Plain+History
    bvh = all_results["brain_vs_history"]
    brain_totals_2 = [r["brain_total"] for r in bvh]
    hist_totals_2 = [r["history_total"] for r in bvh]

    print(f"\n  BRAIN vs PLAIN+HISTORY (same event list) — {len(bvh)} turns:")
    print(f"    Brain avg:     {mean(brain_totals_2):.1f}/40")
    print(f"    History avg:   {mean(hist_totals_2):.1f}/40")
    diff2 = mean(brain_totals_2) - mean(hist_totals_2)
    pct2 = (diff2 / mean(hist_totals_2) * 100) if mean(hist_totals_2) > 0 else 0
    print(f"    Difference:    {diff2:+.1f} ({pct2:+.1f}%)")
    bw2 = sum(1 for b, h in zip(brain_totals_2, hist_totals_2) if b > h)
    hw2 = sum(1 for b, h in zip(brain_totals_2, hist_totals_2) if h > b)
    t2 = sum(1 for b, h in zip(brain_totals_2, hist_totals_2) if b == h)
    print(f"    Wins: Brain={bw2} History={hw2} Tie={t2}")

    # Per-criterion breakdown
    print(f"\n  Per-criterion (Brain vs Plain, 0-10 scale):")
    print(f"    {'Criterion':<28s} {'Brain':>6s} {'Plain':>6s} {'Diff':>7s}")
    print(f"    {'─'*28} {'─'*6} {'─'*6} {'─'*7}")
    for c in CRITERIA:
        b_vals = [r["brain_scores"].get(c, 0) for r in bvp]
        p_vals = [r["plain_scores"].get(c, 0) for r in bvp]
        bm, pm = mean(b_vals), mean(p_vals)
        print(f"    {c:<28s} {bm:>5.1f} {pm:>5.1f} {bm - pm:>+6.1f}")

    print(f"\n  Per-criterion (Brain vs Plain+History, 0-10 scale):")
    print(f"    {'Criterion':<28s} {'Brain':>6s} {'Hist':>6s} {'Diff':>7s}")
    print(f"    {'─'*28} {'─'*6} {'─'*6} {'─'*7}")
    for c in CRITERIA:
        b_vals = [r["brain_scores"].get(c, 0) for r in bvh]
        h_vals = [r["history_scores"].get(c, 0) for r in bvh]
        bm, hm = mean(b_vals), mean(h_vals)
        print(f"    {c:<28s} {bm:>5.1f} {hm:>5.1f} {bm - hm:>+6.1f}")

    # Per-session breakdown (does advantage grow over time?)
    print(f"\n  Brain vs Plain by session number (does advantage grow with more history?):")
    max_sessions = max(r["session"] for r in bvp)
    for s in range(1, max_sessions + 1):
        s_brain = [r["brain_total"] for r in bvp if r["session"] == s]
        s_plain = [r["plain_total"] for r in bvp if r["session"] == s]
        if s_brain:
            diff = mean(s_brain) - mean(s_plain)
            print(f"    Session {s}: Brain={mean(s_brain):.1f} Plain={mean(s_plain):.1f} Diff={diff:+.1f}")

    print(f"\n  Judge calls: {judge._calls}")
    print(f"  Brain tokens: {brain_sys.total_tokens if 'brain_sys' in dir() else 'N/A'}")
    print(f"{'═' * 95}")


if __name__ == "__main__":
    n_trials = int(os.environ.get("N_TRIALS", "1"))
    run_cross_session_eval(n_trials=n_trials)

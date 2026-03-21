#!/usr/bin/env python3
"""Validation pipeline for moral/causal judgment prediction using the real
ModeOrchestrator pipeline.

MoCa (Moral/Causal judgments) contains 206 vignettes:
  - 144 causal judgment vignettes
  - 62 moral judgment vignettes

Each vignette has a story, question, answer_dist (probability distribution over
yes/no), and individual_votes (25 per vignette).

This script runs each vignette's story+question through run_task() on the
ModeOrchestrator pipeline and uses the pipeline response to predict P(yes):
  - Which branches activated tells us about the cognitive process
  - moral_evaluation + empathy_social -> higher blame/causation attribution
  - self_protection + self_justification -> lower blame (defensive reasoning)
  - reflective_reasoning -> moderate, nuanced judgment (closer to 0.5)
  - fear_risk -> heightened threat perception -> more blame
  - Evaluator's conflict handling score indicates uncertainty
  - Number of active conflicts indicates moral complexity

Uses: ModeOrchestrator + HumanModeRouter + 14 cognitive branches +
      HumanModeEvaluator + HumanModeMemory + RL weight adaptation.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prompt_forest.modes.orchestrator import ModeOrchestrator
from src.prompt_forest.backend.mock import MockLLMBackend


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class Vignette:
    """A single MoCa vignette with story, question, and human judgments."""

    id: str
    story: str
    question: str
    vignette_type: str  # "causal" or "moral"
    answer_dist: dict[str, float]  # {"yes": p, "no": 1-p}
    individual_votes: list[str]  # list of "yes"/"no" strings
    raw: dict[str, Any]  # original JSON

    @property
    def p_yes(self) -> float:
        """Proportion of 'yes' votes."""
        if self.answer_dist and "yes" in self.answer_dist:
            return self.answer_dist["yes"]
        if self.individual_votes:
            yes_count = sum(1 for v in self.individual_votes if v.lower() == "yes")
            return yes_count / len(self.individual_votes) if self.individual_votes else 0.5
        return 0.5


# ===========================================================================
# Text analysis functions (keyword-based cue extractors for initial_state)
# ===========================================================================

HARM_WORDS = [
    "kill", "killed", "die", "died", "death", "dead", "murder", "hurt",
    "harm", "injure", "injured", "injury", "wound", "wounded", "destroy",
    "destroyed", "damage", "damaged", "suffer", "suffering", "pain",
    "painful", "violent", "violence", "assault", "attack", "crash",
    "smash", "break", "broke", "broken", "burn", "burned", "fire",
    "poison", "poisoned", "drown", "drowned", "shoot", "shot", "stab",
    "stabbed", "bleed", "bleeding", "bruise", "fracture", "crush",
    "collapsed", "accident", "fatal", "lethal", "toxic", "contaminate",
    "explode", "explosion", "electrocute", "suffocate",
]

INTENT_WORDS = [
    "intended", "intentional", "intentionally", "deliberate", "deliberately",
    "purpose", "purposely", "meant", "planned", "plan", "decided", "decide",
    "chose", "choose", "wanted", "desire", "desired", "goal", "aimed",
    "aim", "knowingly", "aware", "conscious", "consciously", "willful",
    "willfully", "on purpose", "premeditated", "calculated", "scheme",
    "plotted", "contrived", "arranged", "set out to", "tried to",
    "attempted", "sought", "motivated", "motive",
]

SOCIAL_WORDS = [
    "friend", "family", "mother", "father", "sister", "brother", "child",
    "children", "wife", "husband", "partner", "neighbor", "colleague",
    "community", "group", "people", "person", "stranger", "help",
    "helped", "care", "caring", "love", "loved", "trust", "trusted",
    "support", "supported", "together", "relationship", "society",
    "social", "team", "cooperate", "share", "shared", "compassion",
    "sympathy", "concern", "worried", "protect", "protected",
    "sacrifice", "volunteer", "generous", "kind", "kindness",
]

MORAL_WORDS = [
    "wrong", "right", "moral", "immoral", "ethical", "unethical", "fair",
    "unfair", "just", "unjust", "justice", "blame", "blameworthy",
    "responsible", "responsibility", "guilty", "guilt", "innocent",
    "duty", "obligation", "virtue", "vice", "sin", "evil", "good",
    "bad", "should", "ought", "deserve", "punishment", "punish",
    "condemn", "forbidden", "permissible", "impermissible", "violation",
    "violate", "norm", "principle", "conscience", "integrity",
    "dishonest", "cheat", "cheated", "lie", "lied", "steal", "stole",
    "corrupt", "betray", "betrayal", "exploit", "exploited",
]

CAUSAL_WORDS = [
    "cause", "caused", "causing", "because", "result", "resulted",
    "consequence", "led to", "lead to", "brought about", "produce",
    "produced", "effect", "affected", "responsible", "due to",
    "trigger", "triggered", "made", "make", "force", "forced",
    "prevent", "prevented", "allow", "allowed", "enable", "enabled",
    "directly", "indirectly", "chain", "mechanism", "means",
    "instrument", "through", "by", "via", "outcome", "impact",
]

PERSONAL_FORCE_WORDS = [
    "push", "pushed", "shove", "shoved", "grab", "grabbed", "throw",
    "threw", "hit", "struck", "kick", "kicked", "pull", "pulled",
    "drag", "dragged", "squeeze", "hold", "held", "force", "forced",
    "physically", "hands", "body", "contact", "touch", "touched",
    "personally", "himself", "herself", "own hands",
]

EVITABILITY_WORDS = [
    "could have", "alternative", "another way", "option", "choice",
    "chose not", "decided not", "instead", "otherwise", "avoid",
    "avoidable", "preventable", "unnecessary", "different",
    "other way", "possible", "possibility",
]


def _word_density(text: str, word_list: list[str]) -> float:
    """Count how many words from word_list appear in text, normalised to [0, 1]."""
    text_lower = text.lower()
    count = 0
    for word in word_list:
        if " " in word:
            count += len(re.findall(re.escape(word), text_lower))
        else:
            count += len(re.findall(r"\b" + re.escape(word) + r"\b", text_lower))
    return min(1.0, 1.0 - math.exp(-count * 0.3))


def extract_harm_cues(text: str) -> float:
    return _word_density(text, HARM_WORDS)

def extract_intent_cues(text: str) -> float:
    return _word_density(text, INTENT_WORDS)

def extract_social_cues(text: str) -> float:
    return _word_density(text, SOCIAL_WORDS)

def extract_moral_cues(text: str) -> float:
    return _word_density(text, MORAL_WORDS)

def extract_causal_cues(text: str) -> float:
    return _word_density(text, CAUSAL_WORDS)

def extract_personal_force(text: str) -> float:
    return _word_density(text, PERSONAL_FORCE_WORDS)

def extract_evitability(text: str) -> float:
    return _word_density(text, EVITABILITY_WORDS)


# ===========================================================================
# Dataset loading
# ===========================================================================

def download_moca(target_dir: str) -> bool:
    """Clone the MoCa dataset repository into target_dir."""
    repo_url = "https://github.com/cicl-stanford/moca"
    print(f"  Cloning MoCa dataset from {repo_url} ...")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  [ERROR] git clone failed: {result.stderr.strip()}")
            return False
        print(f"  Clone successful -> {target_dir}")
        return True
    except FileNotFoundError:
        print("  [ERROR] git is not installed or not on PATH.")
        return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] git clone timed out after 120s.")
        return False
    except Exception as e:
        print(f"  [ERROR] Unexpected error during clone: {e}")
        return False


def find_json_files(repo_dir: str) -> list[str]:
    """Recursively find all JSON files in the MoCa repo that look like data."""
    json_files = []
    for root, _dirs, files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    return sorted(json_files)


def parse_vignettes(repo_dir: str) -> list[Vignette]:
    """Parse MoCa JSON files into Vignette objects."""
    vignettes: list[Vignette] = []
    json_files = find_json_files(repo_dir)

    if not json_files:
        print("  [WARNING] No JSON files found in repository.")
        return vignettes

    print(f"  Found {len(json_files)} JSON files, scanning for vignette data...")

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ["vignettes", "data", "items", "stories", "stimuli",
                        "trials", "scenarios", "experiments"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            if not items and _looks_like_vignette(data):
                items = [data]

        for i, item in enumerate(items):
            v = _try_parse_vignette(item, jf, i)
            if v is not None:
                vignettes.append(v)

    seen_ids: set[str] = set()
    unique: list[Vignette] = []
    for v in vignettes:
        if v.id not in seen_ids:
            seen_ids.add(v.id)
            unique.append(v)

    print(f"  Parsed {len(unique)} unique vignettes "
          f"({sum(1 for v in unique if v.vignette_type == 'causal')} causal, "
          f"{sum(1 for v in unique if v.vignette_type == 'moral')} moral)")
    return unique


def _looks_like_vignette(d: dict) -> bool:
    has_story = any(k in d for k in ["story", "text", "vignette", "scenario",
                                      "context", "stimulus", "narrative"])
    has_response = any(k in d for k in ["answer_dist", "responses", "votes",
                                         "individual_votes", "judgments",
                                         "answer", "response"])
    return has_story and has_response


def _try_parse_vignette(item: Any, source_file: str, index: int) -> Vignette | None:
    if not isinstance(item, dict):
        return None

    story = ""
    for key in ["story", "text", "vignette", "scenario", "context",
                "stimulus", "narrative", "description"]:
        if key in item and isinstance(item[key], str):
            story = item[key]
            break

    if not story or len(story) < 20:
        return None

    question = ""
    for key in ["question", "query", "prompt", "probe"]:
        if key in item and isinstance(item[key], str):
            question = item[key]
            break

    answer_dist: dict[str, float] = {}
    for key in ["answer_dist", "response_dist", "distribution", "prob"]:
        if key in item and isinstance(item[key], dict):
            answer_dist = _normalise_answer_dist(item[key])
            break

    individual_votes: list[str] = []
    for key in ["individual_votes", "votes", "responses", "judgments",
                "individual_responses"]:
        if key in item and isinstance(item[key], list):
            individual_votes = [str(v).lower() for v in item[key]]
            break

    if not answer_dist and individual_votes:
        yes_count = sum(1 for v in individual_votes if v in ("yes", "1", "true"))
        total = len(individual_votes)
        if total > 0:
            answer_dist = {"yes": yes_count / total, "no": (total - yes_count) / total}

    if not answer_dist:
        return None

    vignette_type = "causal"
    for key in ["type", "category", "condition", "vignette_type", "judgment_type"]:
        if key in item and isinstance(item[key], str):
            val = item[key].lower()
            if "moral" in val:
                vignette_type = "moral"
            elif "causal" in val:
                vignette_type = "causal"
            break

    combined_text = (question + " " + story).lower()
    if not any(k in item for k in ["type", "category", "condition"]):
        if any(w in combined_text for w in ["morally", "moral", "blame", "wrong",
                                             "permissible", "ethical"]):
            vignette_type = "moral"

    vid = item.get("id", item.get("vignette_id", item.get("name", "")))
    if not vid:
        basename = Path(source_file).stem
        vid = f"{basename}_{index}"

    return Vignette(
        id=str(vid),
        story=story,
        question=question,
        vignette_type=vignette_type,
        answer_dist=answer_dist,
        individual_votes=individual_votes,
        raw=item,
    )


def _normalise_answer_dist(dist: dict) -> dict[str, float]:
    yes_val = 0.0
    no_val = 0.0
    for k, v in dist.items():
        k_lower = str(k).lower()
        if k_lower in ("yes", "1", "true"):
            yes_val = float(v)
        elif k_lower in ("no", "0", "false"):
            no_val = float(v)
    total = yes_val + no_val
    if total > 0:
        return {"yes": yes_val / total, "no": no_val / total}
    return {}


# ===========================================================================
# Generate synthetic vignettes (fallback)
# ===========================================================================

def generate_synthetic_vignettes() -> list[Vignette]:
    """Generate synthetic MoCa-style vignettes for testing when download fails."""
    print("  Generating synthetic MoCa-style vignettes for validation...")
    vignettes: list[Vignette] = []

    causal_scenarios = [
        {"id": "causal_01", "story": "Alex was working at a chemical plant. He accidentally knocked over a container of toxic chemicals, which spilled into the water supply. Several people in the nearby town became seriously ill.", "question": "Did Alex cause the people to become ill?", "p_yes": 0.82},
        {"id": "causal_02", "story": "A tree fell during a storm and blocked the road. Sarah could not get to the hospital in time, and her condition worsened significantly.", "question": "Did the fallen tree cause Sarah's condition to worsen?", "p_yes": 0.74},
        {"id": "causal_03", "story": "The machine was designed to automatically shut down when it overheated. Tom disabled the safety mechanism. Later, the machine overheated and caused a fire in the factory.", "question": "Did Tom cause the fire?", "p_yes": 0.88},
        {"id": "causal_04", "story": "Jenny left a glass of water on the edge of the table. Her cat jumped onto the table, knocking the glass onto the laptop below, which was destroyed by the water.", "question": "Did Jenny cause the laptop to be destroyed?", "p_yes": 0.42},
        {"id": "causal_05", "story": "David was driving at the speed limit when a child suddenly ran into the road from between parked cars. Despite braking immediately, he could not stop in time and the child was injured.", "question": "Did David cause the child's injury?", "p_yes": 0.35},
        {"id": "causal_06", "story": "Maria knowingly sold contaminated food at her restaurant because she did not want to lose money by throwing it away. Several customers became very ill after eating there.", "question": "Did Maria cause the customers to become ill?", "p_yes": 0.95},
        {"id": "causal_07", "story": "The bridge had a structural defect that the city engineers had failed to notice during their last inspection. During a heavy storm, the bridge collapsed and three cars fell into the river.", "question": "Did the engineers' failure cause the collapse?", "p_yes": 0.68},
        {"id": "causal_08", "story": "A software bug in the trading algorithm caused it to make millions of incorrect trades in seconds. The programmer who wrote the code had left the company years ago.", "question": "Did the programmer cause the financial losses?", "p_yes": 0.45},
        {"id": "causal_09", "story": "Mark deliberately cut the brake lines on his coworker's car. His coworker drove the car later that day and crashed into a building, suffering severe injuries.", "question": "Did Mark cause his coworker's injuries?", "p_yes": 0.97},
        {"id": "causal_10", "story": "The doctor prescribed the standard medication for the patient's condition. Unknown to anyone, the patient had a rare allergy to the medication and had a severe reaction.", "question": "Did the doctor cause the patient's allergic reaction?", "p_yes": 0.28},
        {"id": "causal_11", "story": "Lisa forgot to close the garden gate. Her neighbor's dog wandered in through the open gate and trampled all the flowers in the garden.", "question": "Did Lisa cause the flowers to be destroyed?", "p_yes": 0.55},
        {"id": "causal_12", "story": "The factory released waste into the river, but the waste was within legally permitted levels. Over many years, the cumulative effect of the waste poisoned the fish population.", "question": "Did the factory cause the fish to die?", "p_yes": 0.72},
        {"id": "causal_13", "story": "A lightning strike hit the old oak tree, which fell onto the power lines, causing a blackout in the entire neighborhood that lasted three days.", "question": "Did the lightning cause the blackout?", "p_yes": 0.91},
        {"id": "causal_14", "story": "Ben gave his friend directions to a new restaurant. He made a mistake and sent his friend the wrong way. His friend got lost and missed an important meeting.", "question": "Did Ben cause his friend to miss the meeting?", "p_yes": 0.60},
        {"id": "causal_15", "story": "The company decided to lay off 500 workers to increase profits. Many of the laid-off workers could not find new jobs and faced serious financial hardship.", "question": "Did the company cause the workers' financial hardship?", "p_yes": 0.78},
        {"id": "causal_16", "story": "Emily was texting while driving and ran a red light. She collided with another car in the intersection, injuring the other driver.", "question": "Did Emily cause the accident?", "p_yes": 0.96},
        {"id": "causal_17", "story": "The locksmith made a copy of the key for the building manager. Later, the building manager used that key to enter a tenant's apartment without permission.", "question": "Did the locksmith cause the unauthorized entry?", "p_yes": 0.15},
        {"id": "causal_18", "story": "A gust of wind blew a sign off a building. The sign fell onto a parked car below, shattering the windshield.", "question": "Did the wind cause the windshield to shatter?", "p_yes": 0.88},
        {"id": "causal_19", "story": "Kevin told his friend a secret that his friend had shared in confidence. His friend was embarrassed when the secret became widely known and suffered social consequences.", "question": "Did Kevin cause his friend's embarrassment?", "p_yes": 0.84},
        {"id": "causal_20", "story": "The babysitter turned away for a moment to answer the phone. During that moment, the child fell off the swing and broke her arm.", "question": "Did the babysitter cause the child's broken arm?", "p_yes": 0.50},
    ]

    moral_scenarios = [
        {"id": "moral_01", "story": "A trolley is heading toward five people. You can pull a lever to divert it to a side track where it will kill one person instead. You pull the lever.", "question": "Is it morally permissible to pull the lever?", "p_yes": 0.85},
        {"id": "moral_02", "story": "A trolley is heading toward five people. You can push a large stranger off a bridge onto the tracks. His body will stop the trolley, saving five lives, but he will die.", "question": "Is it morally permissible to push the stranger?", "p_yes": 0.30},
        {"id": "moral_03", "story": "A surgeon has five patients who will die without organ transplants. A healthy visitor comes in for a check-up. The surgeon could kill the visitor and use his organs to save the five patients.", "question": "Is it morally permissible for the surgeon to kill the visitor?", "p_yes": 0.08},
        {"id": "moral_04", "story": "During a severe famine, a mother steals bread from a wealthy merchant's store to feed her starving children. The merchant will not be significantly affected by the loss.", "question": "Is it morally permissible for the mother to steal the bread?", "p_yes": 0.88},
        {"id": "moral_05", "story": "A CEO discovers that her company's product has a minor defect that could cause injury in rare cases. Recalling the product would be very expensive and might bankrupt the company, costing thousands of jobs.", "question": "Is it morally wrong for the CEO to not recall the product?", "p_yes": 0.62},
        {"id": "moral_06", "story": "A person finds a wallet with cash and identification. Nobody is around and there are no security cameras. The person keeps the cash and throws away the wallet.", "question": "Is the person's action morally wrong?", "p_yes": 0.82},
        {"id": "moral_07", "story": "A doctor lies to a terminally ill patient about their prognosis to spare them emotional suffering in their final days. The patient has expressed a desire to know the truth.", "question": "Is it morally permissible for the doctor to lie?", "p_yes": 0.38},
        {"id": "moral_08", "story": "A soldier is ordered to bomb a village that is harboring enemy combatants. There are also many civilians in the village. The soldier follows the order.", "question": "Is the soldier morally blameworthy for following the order?", "p_yes": 0.55},
        {"id": "moral_09", "story": "A pharmaceutical company raises the price of a life-saving drug by 5000% because they have a monopoly and patients have no alternative.", "question": "Is the company's action morally wrong?", "p_yes": 0.94},
        {"id": "moral_10", "story": "A teacher gives a struggling student a passing grade they did not earn because the student is dealing with difficult family circumstances and failing would cause them to lose their scholarship.", "question": "Is it morally permissible for the teacher to give the unearned grade?", "p_yes": 0.48},
    ]

    for sc in causal_scenarios:
        p_yes = sc["p_yes"]
        votes = (["yes"] * round(p_yes * 25) + ["no"] * (25 - round(p_yes * 25)))
        vignettes.append(Vignette(
            id=sc["id"], story=sc["story"], question=sc["question"],
            vignette_type="causal",
            answer_dist={"yes": p_yes, "no": 1.0 - p_yes},
            individual_votes=votes, raw=sc,
        ))

    for sc in moral_scenarios:
        p_yes = sc["p_yes"]
        votes = (["yes"] * round(p_yes * 25) + ["no"] * (25 - round(p_yes * 25)))
        vignettes.append(Vignette(
            id=sc["id"], story=sc["story"], question=sc["question"],
            vignette_type="moral",
            answer_dist={"yes": p_yes, "no": 1.0 - p_yes},
            individual_votes=votes, raw=sc,
        ))

    print(f"  Generated {len(vignettes)} synthetic vignettes "
          f"({sum(1 for v in vignettes if v.vignette_type == 'causal')} causal, "
          f"{sum(1 for v in vignettes if v.vignette_type == 'moral')} moral)")
    return vignettes


# ===========================================================================
# Pipeline-based prediction
# ===========================================================================

def build_initial_state_for_story(story: str, question: str) -> dict[str, float]:
    """Build initial_state overrides based on story content for ModeOrchestrator."""
    combined = story + " " + question

    harm = extract_harm_cues(combined)
    intent = extract_intent_cues(combined)
    social = extract_social_cues(combined)
    moral = extract_moral_cues(combined)
    personal = extract_personal_force(combined)
    evitability = extract_evitability(combined)

    initial_state: dict[str, float] = {}

    # Empathy scales with harm (more harm -> more empathetic response)
    initial_state["empathy"] = min(0.95, 0.35 + harm * 0.45 + social * 0.2 + moral * 0.15)
    # Honesty scales with intent clarity and moral salience
    initial_state["honesty"] = min(0.95, 0.40 + intent * 0.30 + moral * 0.20)
    # Reflection scales with complexity
    initial_state["reflection"] = min(0.95, 0.35 + intent * 0.20 + evitability * 0.25 + moral * 0.15)
    # Self-protection increases with personal force
    initial_state["self_protection"] = min(0.95, 0.20 + personal * 0.30 + (1.0 - harm) * 0.15)
    # Fear increases with harm
    initial_state["fear"] = min(0.95, 0.10 + harm * 0.35 + personal * 0.15)
    # Trust modulated by social context
    initial_state["trust"] = max(0.05, min(0.95, 0.35 + social * 0.35 - harm * 0.10))
    # Caution increases with harm severity
    initial_state["caution"] = min(0.95, 0.25 + harm * 0.30 + personal * 0.15)
    # Impulse decreases with reflection
    initial_state["impulse"] = max(0.05, 0.40 - intent * 0.20 - moral * 0.10)

    return initial_state


def build_events_for_story(story: str, question: str) -> list[tuple[str, float]]:
    """Build inject_event() calls based on story content."""
    combined = story + " " + question
    events: list[tuple[str, float]] = []

    harm = extract_harm_cues(combined)
    social = extract_social_cues(combined)
    intent = extract_intent_cues(combined)
    personal = extract_personal_force(combined)

    if harm > 0.05:
        events.append(("threat", min(1.0, harm * 1.5)))
    if social > 0.05:
        events.append(("social_praise", min(1.0, social * 1.2)))
    if intent > 0.05:
        events.append(("novelty", min(1.0, intent * 1.0)))
    if personal > 0.05:
        events.append(("deadline_pressure", min(1.0, personal * 1.2)))

    return events


def predict_p_yes_from_pipeline(
    result: dict[str, Any],
    vignette_type: str,
    story: str,
    question: str,
) -> float:
    """Predict P(yes) from pipeline outputs.

    Uses:
      - Which branches were activated (cognitive process)
      - Routing branch scores (drive strengths)
      - Post-task state (mood, arousal, drives)
      - Evaluator coherence scores
      - Number and intensity of active conflicts
    """
    routing = result.get("routing", {})
    activated = routing.get("activated_branches", []) if isinstance(routing, dict) else []
    branch_scores = routing.get("branch_scores", {}) if isinstance(routing, dict) else {}
    task_type = routing.get("task_type", "") if isinstance(routing, dict) else ""

    eval_signal = result.get("evaluation_signal", {})
    reward = eval_signal.get("reward_score", 0.0) if isinstance(eval_signal, dict) else 0.0

    human_state = result.get("human_state", {})
    after_state = human_state.get("after", {}) if isinstance(human_state, dict) else {}
    after_vars = after_state.get("variables", {}) if isinstance(after_state, dict) else {}
    mood = after_state.get("mood_valence", 0.0) if isinstance(after_state, dict) else 0.0

    conflicts = result.get("conflicts", [])
    n_conflicts = len(conflicts) if isinstance(conflicts, list) else 0

    # Extract text cues for base prediction
    combined = story + " " + question
    harm = extract_harm_cues(combined)
    intent = extract_intent_cues(combined)
    causal = extract_causal_cues(combined)
    moral = extract_moral_cues(combined)
    social = extract_social_cues(combined)
    personal = extract_personal_force(combined)
    evitability = extract_evitability(combined)

    # Get relevant branch scores
    moral_eval_score = branch_scores.get("moral_evaluation", 0.0)
    empathy_score = branch_scores.get("empathy_social", 0.0)
    self_prot_score = branch_scores.get("self_protection", 0.0)
    self_just_score = branch_scores.get("self_justification", 0.0)
    reflect_score = branch_scores.get("reflective_reasoning", 0.0)
    fear_score = branch_scores.get("fear_risk", 0.0)
    impulse_score = branch_scores.get("impulse_response", 0.0)

    # Normalize scores to [0, 1] range for predictive use
    all_scores = list(branch_scores.values())
    max_score = max(all_scores) if all_scores else 1.0
    if max_score > 0:
        moral_eval_norm = moral_eval_score / max_score
        empathy_norm = empathy_score / max_score
        self_prot_norm = self_prot_score / max_score
        self_just_norm = self_just_score / max_score
        reflect_norm = reflect_score / max_score
        fear_norm = fear_score / max_score
    else:
        moral_eval_norm = empathy_norm = self_prot_norm = 0.5
        self_just_norm = reflect_norm = fear_norm = 0.5

    if vignette_type == "causal":
        # Causal judgment: did the agent cause the outcome?
        causal_clarity = min(1.0, causal * 0.5 + intent * 0.4 + harm * 0.3)
        agent_awareness = min(1.0, intent * 0.7 + harm * 0.2 + 0.1)

        base_p = 0.20 + causal_clarity * 0.35 + agent_awareness * 0.30

        # moral_evaluation + empathy_social -> higher blame/causation
        base_p += (moral_eval_norm - 0.5) * 0.15
        base_p += (empathy_norm - 0.5) * harm * 0.15

        # fear_risk -> heightened threat perception -> more blame
        base_p += (fear_norm - 0.5) * 0.10

        # self_protection + self_justification -> lower blame (defensive)
        base_p -= (self_prot_norm - 0.3) * 0.12
        base_p -= (self_just_norm - 0.3) * 0.08

        # reflective_reasoning -> moderate judgment (closer to 0.5)
        if reflect_norm > 0.6:
            base_p = base_p * 0.85 + 0.5 * 0.15

    else:
        # Moral judgment
        question_lower = question.lower()
        is_permissible = any(w in question_lower for w in
                             ["permissible", "acceptable", "justified", "right to", "okay to"])
        is_blame = any(w in question_lower for w in
                       ["wrong", "blame", "blameworthy", "immoral", "unethical", "guilty"])

        moral_severity = min(1.0, moral * 0.4 + harm * 0.35 + intent * 0.35)
        personal_effect = personal * 0.25
        evit_effect = evitability * 0.20

        if is_permissible:
            base_p = 0.6 - moral_severity * 0.3 - personal_effect - evit_effect
            # empathy_social high -> less permissible (empathy for victim)
            base_p -= empathy_norm * harm * 0.1
            # social context can increase permissibility
            base_p += social * empathy_norm * 0.08
            # moral_evaluation high -> less permissible (moral disapproval)
            base_p -= moral_eval_norm * moral_severity * 0.05
            # self_justification -> rationalise -> more permissible
            base_p += self_just_norm * 0.08
        elif is_blame:
            base_p = 0.4 + moral_severity * 0.3 + personal_effect + evit_effect
            # empathy -> more blame
            base_p += empathy_norm * harm * 0.1
            # moral_evaluation -> more blame
            base_p += moral_eval_norm * moral_severity * 0.05
            # self_protection/justification -> less blame
            base_p -= self_just_norm * 0.06
            base_p -= self_prot_norm * 0.04
        else:
            base_p = 0.5 + moral_severity * 0.2 + personal_effect
            base_p += empathy_norm * harm * 0.08
            base_p += moral_eval_norm * 0.05

        # reflective_reasoning moderates extreme judgments
        if reflect_norm > 0.6:
            base_p = base_p * 0.85 + 0.5 * 0.15

    # Conflict penalty: more conflicts -> more uncertainty -> pull toward 0.5
    conflict_penalty = min(0.3, n_conflicts * 0.05)
    base_p = base_p * (1.0 - conflict_penalty) + 0.5 * conflict_penalty

    # Evaluator confidence modulation
    if reward < 0.3:
        base_p = base_p * 0.9 + 0.5 * 0.1  # low confidence -> pull toward center

    return max(0.01, min(0.99, base_p))


def predict_vignette(v: Vignette) -> tuple[float, dict[str, Any]]:
    """Predict P(yes) for a vignette using the full ModeOrchestrator pipeline."""
    initial_state = build_initial_state_for_story(v.story, v.question)
    events = build_events_for_story(v.story, v.question)

    orch = ModeOrchestrator(
        mode="human_mode",
        backend=MockLLMBackend(seed=42),
        initial_state=initial_state,
    )

    # Inject story-derived events
    for event_type, intensity in events:
        orch.inject_event(event_type, intensity)

    # Run through the full pipeline
    task_text = f"{v.story} {v.question}"
    result = orch.run_task(text=task_text, task_type="auto")

    # Predict P(yes) from pipeline outputs
    p_yes = predict_p_yes_from_pipeline(result, v.vignette_type, v.story, v.question)

    return p_yes, result


# ===========================================================================
# Metrics
# ===========================================================================

def pearson_correlation(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if std_x < 1e-9 or std_y < 1e-9:
        return 0.0
    return cov / (std_x * std_y)


def spearman_correlation(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and abs(indexed[j + 1][1] - indexed[i][1]) < 1e-12:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    return pearson_correlation(rx, ry)


def kl_divergence(p: list[float], q: list[float]) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(0.001, min(0.999, pi))
        qi = max(0.001, min(0.999, qi))
        kl = pi * math.log(pi / qi) + (1 - pi) * math.log((1 - pi) / (1 - qi))
        total += kl
    return total / len(p) if p else 0.0


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def binary_accuracy(actual: list[float], predicted: list[float],
                    threshold: float = 0.5) -> float:
    if not actual:
        return 0.0
    correct = sum(
        1 for a, p in zip(actual, predicted)
        if (a >= threshold) == (p >= threshold)
    )
    return correct / len(actual)


# ===========================================================================
# Output formatting
# ===========================================================================

def print_header(title: str, width: int = 72) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 72) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, width - len(title) - 5))


def print_metrics(actual: list[float], predicted: list[float],
                  label: str = "Overall") -> dict[str, float]:
    r = pearson_correlation(actual, predicted)
    rho = spearman_correlation(actual, predicted)
    kl = kl_divergence(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    acc = binary_accuracy(actual, predicted)

    print(f"  Pearson r        = {r:+.4f}")
    print(f"  Spearman rho     = {rho:+.4f}")
    print(f"  KL divergence    = {kl:.4f}")
    print(f"  MAE              = {mae:.4f}")
    print(f"  Binary accuracy  = {acc:.2%}  (threshold=0.5)")
    print(f"  N                = {len(actual)}")

    return {"pearson": r, "spearman": rho, "kl": kl, "mae": mae,
            "accuracy": acc, "n": len(actual)}


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_validation() -> None:
    """Run the full MoCa validation pipeline using ModeOrchestrator."""

    print_header("MoCa Validation Pipeline (ModeOrchestrator)")
    print("  Pipeline: ModeOrchestrator + HumanModeRouter + 14 cognitive branches")
    print("           + HumanModeEvaluator + HumanModeMemory + RL weight adaptation")
    print("  Validating that pipeline routing decisions predict human")
    print("  causal and moral judgments from the MoCa dataset.")

    # ── Step 1: Load dataset ─────────────────────────────────────────────
    print_header("Step 1: Load MoCa Dataset")

    vignettes: list[Vignette] = []
    temp_dir = tempfile.mkdtemp(prefix="moca_")

    try:
        success = download_moca(temp_dir)
        if success:
            vignettes = parse_vignettes(temp_dir)
    except Exception as e:
        print(f"  [ERROR] Failed to process MoCa data: {e}")

    if len(vignettes) < 5:
        print()
        print("  Insufficient vignettes from MoCa repository.")
        print("  Falling back to synthetic MoCa-style vignettes.")
        vignettes = generate_synthetic_vignettes()

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    if not vignettes:
        print("  [FATAL] No vignettes available. Exiting.")
        return

    # ── Step 2: Run predictions through the pipeline ─────────────────────
    print_header("Step 2: Run ModeOrchestrator Pipeline Predictions")

    results: list[dict[str, Any]] = []

    for i, v in enumerate(vignettes):
        predicted_p, pipeline_result = predict_vignette(v)
        actual_p = v.p_yes

        # Extract pipeline info
        routing = pipeline_result.get("routing", {})
        activated = routing.get("activated_branches", []) if isinstance(routing, dict) else []
        task_type = routing.get("task_type", "") if isinstance(routing, dict) else ""
        eval_signal = pipeline_result.get("evaluation_signal", {})
        reward = eval_signal.get("reward_score", 0.0) if isinstance(eval_signal, dict) else 0.0
        conflicts = pipeline_result.get("conflicts", [])
        branch_weights = pipeline_result.get("branch_weights", {})

        results.append({
            "id": v.id,
            "type": v.vignette_type,
            "story_snippet": v.story[:80] + "..." if len(v.story) > 80 else v.story,
            "actual_p_yes": actual_p,
            "predicted_p_yes": predicted_p,
            "error": abs(actual_p - predicted_p),
            "activated_branches": activated,
            "task_type": task_type,
            "reward_score": reward,
            "n_conflicts": len(conflicts) if isinstance(conflicts, list) else 0,
            "branch_weights": branch_weights,
        })

        if (i + 1) % 10 == 0 or i == len(vignettes) - 1:
            print(f"  Processed {i + 1}/{len(vignettes)} vignettes...")

    # ── Step 3: Compute metrics ──────────────────────────────────────────
    print_header("Step 3: Results")

    actual_all = [r["actual_p_yes"] for r in results]
    pred_all = [r["predicted_p_yes"] for r in results]

    print_subheader("Overall Metrics")
    overall_metrics = print_metrics(actual_all, pred_all, "Overall")

    causal_results = [r for r in results if r["type"] == "causal"]
    moral_results = [r for r in results if r["type"] == "moral"]

    if causal_results:
        print_subheader(f"Causal Vignettes (n={len(causal_results)})")
        causal_actual = [r["actual_p_yes"] for r in causal_results]
        causal_pred = [r["predicted_p_yes"] for r in causal_results]
        causal_metrics = print_metrics(causal_actual, causal_pred, "Causal")

    if moral_results:
        print_subheader(f"Moral Vignettes (n={len(moral_results)})")
        moral_actual = [r["actual_p_yes"] for r in moral_results]
        moral_pred = [r["predicted_p_yes"] for r in moral_results]
        moral_metrics = print_metrics(moral_actual, moral_pred, "Moral")

    # ── Step 4: Best and worst predictions with pipeline details ─────────
    print_header("Step 4: Prediction Analysis (with Pipeline Details)")

    sorted_by_error = sorted(results, key=lambda r: r["error"])

    print_subheader("Top 5 Best Predictions (lowest error)")
    for r in sorted_by_error[:5]:
        print(f"  [{r['type'].upper():6s}] {r['id']}")
        print(f"    Story: {r['story_snippet']}")
        print(f"    Actual P(yes)={r['actual_p_yes']:.3f}  "
              f"Predicted={r['predicted_p_yes']:.3f}  "
              f"Error={r['error']:.3f}")
        print(f"    Activated branches: {', '.join(r['activated_branches'][:5])}")
        print(f"    Cognitive context: {r['task_type']}")
        print(f"    Evaluator reward: {r['reward_score']:.3f}  "
              f"Conflicts: {r['n_conflicts']}")
        print()

    print_subheader("Top 5 Worst Predictions (highest error)")
    for r in sorted_by_error[-5:]:
        print(f"  [{r['type'].upper():6s}] {r['id']}")
        print(f"    Story: {r['story_snippet']}")
        print(f"    Actual P(yes)={r['actual_p_yes']:.3f}  "
              f"Predicted={r['predicted_p_yes']:.3f}  "
              f"Error={r['error']:.3f}")
        print(f"    Activated branches: {', '.join(r['activated_branches'][:5])}")
        print(f"    Cognitive context: {r['task_type']}")
        print(f"    Evaluator reward: {r['reward_score']:.3f}  "
              f"Conflicts: {r['n_conflicts']}")
        print()

    # ── Step 5: Pipeline-specific analysis ────────────────────────────────
    print_header("Step 5: Pipeline Routing Analysis")

    # Branch activation frequency
    branch_counts: dict[str, int] = {}
    for r in results:
        for b in r["activated_branches"]:
            branch_counts[b] = branch_counts.get(b, 0) + 1

    print_subheader("Branch Activation Frequency Across Vignettes")
    for branch, count in sorted(branch_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {branch:<30} {count:>4}/{len(results)} ({count/len(results)*100:.1f}%)")

    # Cognitive context distribution
    ctx_counts: dict[str, int] = {}
    for r in results:
        ctx = r["task_type"]
        ctx_counts[ctx] = ctx_counts.get(ctx, 0) + 1

    print_subheader("Cognitive Context Classification")
    for ctx, count in sorted(ctx_counts.items(), key=lambda x: -x[1]):
        print(f"  {ctx:<25} {count:>4}/{len(results)} ({count/len(results)*100:.1f}%)")

    # Branch weight evolution (compare first and last vignette)
    if results:
        first_w = results[0].get("branch_weights", {})
        last_w = results[-1].get("branch_weights", {})
        if first_w and last_w:
            print_subheader("Branch Weight Changes (first -> last vignette)")
            changes = []
            for k in first_w:
                if k in last_w:
                    delta = last_w[k] - first_w[k]
                    if abs(delta) > 0.001:
                        changes.append((k, first_w[k], last_w[k], delta))
            changes.sort(key=lambda x: abs(x[3]), reverse=True)
            for name, first, last, delta in changes[:8]:
                print(f"  {name:<30} {first:.4f} -> {last:.4f} ({delta:+.4f})")

    # ── Summary ──────────────────────────────────────────────────────────
    print_header("Summary")
    print(f"  Total vignettes evaluated:  {len(results)}")
    print(f"  Causal vignettes:           {len(causal_results)}")
    print(f"  Moral vignettes:            {len(moral_results)}")
    print(f"  Overall Pearson r:          {overall_metrics['pearson']:+.4f}")
    print(f"  Overall Spearman rho:       {overall_metrics['spearman']:+.4f}")
    print(f"  Overall MAE:                {overall_metrics['mae']:.4f}")
    print(f"  Overall binary accuracy:    {overall_metrics['accuracy']:.2%}")
    print(f"  Overall KL divergence:      {overall_metrics['kl']:.4f}")
    print(f"  Unique branches activated:  {len(branch_counts)}")
    print(f"  Mean evaluator reward:      {sum(r['reward_score'] for r in results)/len(results):.4f}")
    print()

    r_val = overall_metrics["pearson"]
    if r_val > 0.7:
        interpretation = "Strong positive correlation -- ModeOrchestrator routing captures meaningful variance in human judgments."
    elif r_val > 0.4:
        interpretation = "Moderate positive correlation -- Pipeline routing partially aligns with human judgment patterns."
    elif r_val > 0.2:
        interpretation = "Weak positive correlation -- some signal, but substantial room for improvement."
    elif r_val > -0.2:
        interpretation = "Near-zero correlation -- Pipeline routing does not meaningfully predict human judgments in current configuration."
    else:
        interpretation = "Negative correlation -- routing logic may need revision."
    print(f"  Interpretation: {interpretation}")
    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_validation()

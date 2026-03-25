#!/usr/bin/env python3
"""
Generate a realistic mock simulation snapshot JSON file.
No ML models or heavy dependencies required — stdlib only.
"""

import json
import math
import random
import os

random.seed(42)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "frontend", "public", "mock_snapshot.json",
)
OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)

TOTAL_TICKS = 360
TICKS_PER_DAY = 24  # 1 tick = 1 hour

LOCATIONS = {
    "office": {"id": "office", "name": "Meridian Corp Office", "default_activity": "Working at desk"},
    "home": {"id": "home", "name": "Home", "default_activity": "At home, relaxing"},
    "bar": {"id": "bar", "name": "The Tap Bar & Grill", "default_activity": "Drinking, socializing"},
    "park": {"id": "park", "name": "Riverside Park", "default_activity": "Walking, sitting"},
    "church": {"id": "church", "name": "Community Church", "default_activity": "Quiet reflection"},
    "school": {"id": "school", "name": "Lincoln High School", "default_activity": "Teaching, studying"},
    "hospital": {"id": "hospital", "name": "General Hospital", "default_activity": "Medical care"},
    "market": {"id": "market", "name": "Town Market", "default_activity": "Shopping, trading"},
}

ACTIONS = [
    "COLLAPSE", "LASH_OUT", "CONFRONT", "FLEE", "WITHDRAW", "SEEK_COMFORT",
    "RUMINATE", "VENT", "SOCIALIZE", "CELEBRATE", "HELP_OTHERS", "WORK", "REST", "IDLE",
]

LAID_OFF_IDS = {"marcus", "rosa", "jake", "mika", "greg", "nadia", "carlos"}

# ── Agent definitions ────────────────────────────────────────────────────────

AGENT_DEFS = [
    {"id": "marcus", "name": "Marcus", "background": "34, logistics coordinator. Wife Lisa and toddler Lily.", "temperament": "Anxious about money, proud provider", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "worried about mortgage", "role": "office_worker", "laid_off": True, "base_valence": 0.55, "base_arousal": 0.2, "base_tension": 0.15, "base_impulse": 0.85, "base_energy": 0.8, "base_vulnerability": 0.15, "relationships_spec": [("lisa", "Lisa", 0.85, 0.8), ("tom", "Tom", 0.3, 0.4), ("sarah", "Sarah", 0.4, 0.45), ("rosa", "Rosa", 0.5, 0.5)]},
    {"id": "sarah", "name": "Sarah", "background": "29, ambitious analyst at Meridian Corp. Single, career-focused.", "temperament": "Driven, competitive, secretly insecure", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "impostor syndrome", "role": "office_worker", "laid_off": False, "base_valence": 0.6, "base_arousal": 0.3, "base_tension": 0.2, "base_impulse": 0.7, "base_energy": 0.85, "base_vulnerability": 0.2, "relationships_spec": [("tom", "Tom", 0.5, 0.5), ("priya", "Priya", 0.6, 0.55), ("lena", "Lena", 0.45, 0.4)]},
    {"id": "tom", "name": "Tom", "background": "52, senior engineer at Meridian. Divorced, two adult kids.", "temperament": "Dry humor, quietly loyal, tired", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "loneliness after divorce", "role": "office_worker", "laid_off": False, "base_valence": 0.45, "base_arousal": 0.15, "base_tension": 0.15, "base_impulse": 0.9, "base_energy": 0.6, "base_vulnerability": 0.25, "relationships_spec": [("ray", "Ray", 0.75, 0.7), ("marcus", "Marcus", 0.3, 0.4), ("sarah", "Sarah", 0.5, 0.5)]},
    {"id": "priya", "name": "Priya", "background": "27, data analyst at Meridian. Engaged to Omar.", "temperament": "Optimistic, social, empathetic", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "pressure from family about wedding", "role": "office_worker", "laid_off": False, "base_valence": 0.7, "base_arousal": 0.25, "base_tension": 0.1, "base_impulse": 0.8, "base_energy": 0.85, "base_vulnerability": 0.15, "relationships_spec": [("omar", "Omar", 0.9, 0.85), ("sarah", "Sarah", 0.6, 0.55), ("nadia", "Nadia", 0.55, 0.5)]},
    {"id": "jake", "name": "Jake", "background": "25, junior sales rep at Meridian. Roommate Jenny.", "temperament": "Carefree, charming, avoids responsibility", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "student loan debt", "role": "office_worker", "laid_off": True, "base_valence": 0.65, "base_arousal": 0.3, "base_tension": 0.1, "base_impulse": 0.55, "base_energy": 0.9, "base_vulnerability": 0.1, "relationships_spec": [("jenny", "Jenny", 0.7, 0.65), ("andre", "Andre", 0.5, 0.55), ("mika", "Mika", 0.45, 0.4)]},
    {"id": "diana", "name": "Diana", "background": "45, HR director at Meridian. Single mom to Sophie.", "temperament": "Empathetic, burdened by difficult decisions", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "guilt about layoffs she helped plan", "role": "office_worker", "laid_off": False, "base_valence": 0.5, "base_arousal": 0.2, "base_tension": 0.25, "base_impulse": 0.85, "base_energy": 0.65, "base_vulnerability": 0.3, "relationships_spec": [("sophie", "Sophie", 0.9, 0.9), ("richard", "Richard", 0.4, 0.35), ("rosa", "Rosa", 0.55, 0.5)]},
    {"id": "chen", "name": "Chen", "background": "40, operations manager at Meridian. Married to Yuki.", "temperament": "Calm, methodical, quietly caring", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "Yuki's exhaustion worries him", "role": "office_worker", "laid_off": False, "base_valence": 0.55, "base_arousal": 0.15, "base_tension": 0.15, "base_impulse": 0.9, "base_energy": 0.7, "base_vulnerability": 0.15, "relationships_spec": [("yuki", "Yuki", 0.9, 0.85), ("kevin", "Kevin", 0.6, 0.55), ("tom", "Tom", 0.5, 0.5)]},
    {"id": "rosa", "name": "Rosa", "background": "31, admin assistant at Meridian. Dating David.", "temperament": "Anxious, people-pleasing, quietly resentful", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "fear of being perceived as incompetent", "role": "office_worker", "laid_off": True, "base_valence": 0.5, "base_arousal": 0.25, "base_tension": 0.2, "base_impulse": 0.75, "base_energy": 0.7, "base_vulnerability": 0.3, "relationships_spec": [("david", "David", 0.8, 0.75), ("diana", "Diana", 0.55, 0.5), ("marcus", "Marcus", 0.5, 0.5)]},
    {"id": "kevin", "name": "Kevin", "background": "38, IT specialist at Meridian. Single, introverted.", "temperament": "Stoic, observant, emotionally guarded", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "social anxiety", "role": "office_worker", "laid_off": False, "base_valence": 0.5, "base_arousal": 0.1, "base_tension": 0.1, "base_impulse": 0.9, "base_energy": 0.65, "base_vulnerability": 0.1, "relationships_spec": [("chen", "Chen", 0.6, 0.55), ("tom", "Tom", 0.45, 0.4)]},
    {"id": "lena", "name": "Lena", "background": "36, project manager at Meridian. Single, athletic.", "temperament": "Assertive, direct, secretly kind", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "burnout she won't admit", "role": "office_worker", "laid_off": False, "base_valence": 0.55, "base_arousal": 0.3, "base_tension": 0.2, "base_impulse": 0.75, "base_energy": 0.8, "base_vulnerability": 0.15, "relationships_spec": [("sarah", "Sarah", 0.45, 0.4), ("andre", "Andre", 0.5, 0.45), ("priya", "Priya", 0.5, 0.5)]},
    {"id": "andre", "name": "Andre", "background": "33, sales lead at Meridian. Married, two kids.", "temperament": "Confident, gregarious, territorial", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "marriage tension", "role": "office_worker", "laid_off": False, "base_valence": 0.6, "base_arousal": 0.3, "base_tension": 0.15, "base_impulse": 0.65, "base_energy": 0.85, "base_vulnerability": 0.15, "relationships_spec": [("jake", "Jake", 0.5, 0.55), ("lena", "Lena", 0.5, 0.45), ("richard", "Richard", 0.35, 0.3)]},
    {"id": "mika", "name": "Mika", "background": "23, junior developer at Meridian. First real job.", "temperament": "Nervous, eager to please, creative", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "fear of failure", "role": "office_worker", "laid_off": True, "base_valence": 0.6, "base_arousal": 0.25, "base_tension": 0.2, "base_impulse": 0.7, "base_energy": 0.8, "base_vulnerability": 0.3, "relationships_spec": [("jake", "Jake", 0.45, 0.4), ("kevin", "Kevin", 0.4, 0.35), ("nadia", "Nadia", 0.5, 0.45)]},
    {"id": "greg", "name": "Greg", "background": "48, facilities manager at Meridian. Divorced, lives alone.", "temperament": "Blunt, gruff, secretly insecure", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "drinking problem", "role": "office_worker", "laid_off": True, "base_valence": 0.4, "base_arousal": 0.2, "base_tension": 0.25, "base_impulse": 0.55, "base_energy": 0.6, "base_vulnerability": 0.25, "relationships_spec": [("frank", "Frank", 0.5, 0.45), ("hank", "Hank", 0.45, 0.4), ("tom", "Tom", 0.35, 0.3)]},
    {"id": "nadia", "name": "Nadia", "background": "28, graphic designer at Meridian. Introverted artist.", "temperament": "Sensitive, creative, prone to spiraling", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "chronic self-doubt", "role": "office_worker", "laid_off": True, "base_valence": 0.55, "base_arousal": 0.2, "base_tension": 0.2, "base_impulse": 0.8, "base_energy": 0.65, "base_vulnerability": 0.35, "relationships_spec": [("priya", "Priya", 0.55, 0.5), ("mika", "Mika", 0.5, 0.45), ("elena", "Elena", 0.45, 0.4)]},
    {"id": "carlos", "name": "Carlos", "background": "50, warehouse supervisor at Meridian. Widower, father of Mike.", "temperament": "Steady, stoic, deeply principled", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "grief for late wife", "role": "office_worker", "laid_off": True, "base_valence": 0.45, "base_arousal": 0.15, "base_tension": 0.15, "base_impulse": 0.9, "base_energy": 0.6, "base_vulnerability": 0.2, "relationships_spec": [("mike", "Mike", 0.9, 0.85), ("hank", "Hank", 0.5, 0.45), ("frank", "Frank", 0.55, 0.5)]},
    {"id": "richard", "name": "Richard", "background": "55, CEO of Meridian Corp. Married, wealthy, conflicted.", "temperament": "Authoritative, guilt-ridden, defensive", "identity_tags": ["manager"], "coalitions": ["meridian_leadership"], "rival_coalitions": [], "private_burden": "knows the layoffs were avoidable", "role": "manager", "laid_off": False, "base_valence": 0.5, "base_arousal": 0.2, "base_tension": 0.3, "base_impulse": 0.8, "base_energy": 0.7, "base_vulnerability": 0.2, "relationships_spec": [("victoria", "Victoria", 0.6, 0.5), ("diana", "Diana", 0.4, 0.35), ("tom", "Tom", 0.35, 0.3)]},
    {"id": "victoria", "name": "Victoria", "background": "48, CFO of Meridian Corp. Divorced, pragmatic.", "temperament": "Pragmatic, cold exterior, sharp mind", "identity_tags": ["manager"], "coalitions": ["meridian_leadership"], "rival_coalitions": [], "private_burden": "fears being seen as heartless", "role": "manager", "laid_off": False, "base_valence": 0.5, "base_arousal": 0.15, "base_tension": 0.2, "base_impulse": 0.85, "base_energy": 0.75, "base_vulnerability": 0.1, "relationships_spec": [("richard", "Richard", 0.6, 0.5), ("diana", "Diana", 0.35, 0.3)]},
    {"id": "frank", "name": "Frank", "background": "60, owner and bartender at The Tap. Widower, community pillar.", "temperament": "Wise, warm, seen-it-all calm", "identity_tags": ["bartender", "community"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "misses his wife", "role": "bartender", "laid_off": False, "base_valence": 0.6, "base_arousal": 0.15, "base_tension": 0.1, "base_impulse": 0.9, "base_energy": 0.65, "base_vulnerability": 0.15, "relationships_spec": [("greg", "Greg", 0.5, 0.45), ("carlos", "Carlos", 0.55, 0.5), ("jenny", "Jenny", 0.6, 0.55), ("hank", "Hank", 0.55, 0.5)]},
    {"id": "maria", "name": "Maria", "background": "72, retired schoolteacher. Lives alone, knows everyone.", "temperament": "Kind but nosy, gossips with good intentions", "identity_tags": ["retiree", "community"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "loneliness", "role": "retiree", "laid_off": False, "base_valence": 0.6, "base_arousal": 0.2, "base_tension": 0.1, "base_impulse": 0.8, "base_energy": 0.5, "base_vulnerability": 0.2, "relationships_spec": [("pastor_james", "Pastor James", 0.7, 0.65), ("elena", "Elena", 0.6, 0.55), ("lisa", "Lisa", 0.55, 0.5)]},
    {"id": "pastor_james", "name": "Pastor James", "background": "65, pastor of Community Church for 30 years.", "temperament": "Compassionate, steady, quietly burdened", "identity_tags": ["retiree", "community", "spiritual"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "doubts about his own faith", "role": "retiree", "laid_off": False, "base_valence": 0.6, "base_arousal": 0.1, "base_tension": 0.1, "base_impulse": 0.95, "base_energy": 0.55, "base_vulnerability": 0.15, "relationships_spec": [("maria", "Maria", 0.7, 0.65), ("frank", "Frank", 0.6, 0.55), ("carlos", "Carlos", 0.5, 0.45)]},
    {"id": "lisa", "name": "Lisa", "background": "32, stay-at-home mom. Marcus's wife, toddler Lily.", "temperament": "Worried, nurturing, increasingly frustrated", "identity_tags": ["retiree", "family"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "feels trapped by motherhood", "role": "retiree", "laid_off": False, "base_valence": 0.55, "base_arousal": 0.2, "base_tension": 0.15, "base_impulse": 0.8, "base_energy": 0.6, "base_vulnerability": 0.25, "relationships_spec": [("marcus", "Marcus", 0.85, 0.8), ("maria", "Maria", 0.55, 0.5), ("sophie", "Sophie", 0.45, 0.4)]},
    {"id": "david", "name": "David", "background": "33, account manager at Meridian. Rosa's boyfriend.", "temperament": "Protective, loyal, quick to anger", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "fear of abandonment", "role": "office_worker", "laid_off": False, "base_valence": 0.55, "base_arousal": 0.25, "base_tension": 0.2, "base_impulse": 0.6, "base_energy": 0.8, "base_vulnerability": 0.2, "relationships_spec": [("rosa", "Rosa", 0.8, 0.75), ("richard", "Richard", 0.2, 0.2), ("chen", "Chen", 0.45, 0.4)]},
    {"id": "sophie", "name": "Sophie", "background": "22, student teacher at Lincoln High. Diana's daughter.", "temperament": "Perceptive, idealistic, anxious about the future", "identity_tags": ["teacher"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "pressure to live up to her mother", "role": "teacher", "laid_off": False, "base_valence": 0.65, "base_arousal": 0.2, "base_tension": 0.15, "base_impulse": 0.75, "base_energy": 0.8, "base_vulnerability": 0.2, "relationships_spec": [("diana", "Diana", 0.9, 0.9), ("elena", "Elena", 0.6, 0.55), ("lisa", "Lisa", 0.45, 0.4)]},
    {"id": "ray", "name": "Ray", "background": "50, QA engineer at Meridian. Tom's best friend.", "temperament": "Supportive, easygoing, conflict-avoidant", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "regrets about missed opportunities", "role": "office_worker", "laid_off": False, "base_valence": 0.55, "base_arousal": 0.15, "base_tension": 0.1, "base_impulse": 0.85, "base_energy": 0.65, "base_vulnerability": 0.15, "relationships_spec": [("tom", "Tom", 0.75, 0.7), ("chen", "Chen", 0.5, 0.45), ("kevin", "Kevin", 0.45, 0.4)]},
    {"id": "jenny", "name": "Jenny", "background": "26, bartender at The Tap. Jake's roommate.", "temperament": "Easygoing, funny, avoids deep emotion", "identity_tags": ["bartender", "community"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "lack of direction in life", "role": "bartender", "laid_off": False, "base_valence": 0.65, "base_arousal": 0.25, "base_tension": 0.1, "base_impulse": 0.6, "base_energy": 0.85, "base_vulnerability": 0.1, "relationships_spec": [("jake", "Jake", 0.7, 0.65), ("frank", "Frank", 0.6, 0.55), ("andre", "Andre", 0.4, 0.35)]},
    {"id": "omar", "name": "Omar", "background": "28, financial analyst at Meridian. Engaged to Priya.", "temperament": "Calm, analytical, emotionally reserved", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "cultural expectations about being provider", "role": "office_worker", "laid_off": False, "base_valence": 0.6, "base_arousal": 0.15, "base_tension": 0.1, "base_impulse": 0.85, "base_energy": 0.75, "base_vulnerability": 0.1, "relationships_spec": [("priya", "Priya", 0.9, 0.85), ("chen", "Chen", 0.5, 0.45), ("sarah", "Sarah", 0.4, 0.35)]},
    {"id": "elena", "name": "Elena", "background": "42, English teacher at Lincoln High. Divorced, warm.", "temperament": "Warm, articulate, quietly lonely", "identity_tags": ["teacher"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "misses having a partner", "role": "teacher", "laid_off": False, "base_valence": 0.6, "base_arousal": 0.15, "base_tension": 0.1, "base_impulse": 0.85, "base_energy": 0.7, "base_vulnerability": 0.2, "relationships_spec": [("sophie", "Sophie", 0.6, 0.55), ("maria", "Maria", 0.6, 0.55), ("nadia", "Nadia", 0.45, 0.4)]},
    {"id": "hank", "name": "Hank", "background": "70, retired mechanic. Gruff exterior, soft heart.", "temperament": "Gruff, opinionated, secretly sentimental", "identity_tags": ["retiree", "community"], "coalitions": ["community"], "rival_coalitions": [], "private_burden": "declining health he won't admit", "role": "retiree", "laid_off": False, "base_valence": 0.45, "base_arousal": 0.15, "base_tension": 0.15, "base_impulse": 0.7, "base_energy": 0.45, "base_vulnerability": 0.2, "relationships_spec": [("frank", "Frank", 0.55, 0.5), ("carlos", "Carlos", 0.5, 0.45), ("greg", "Greg", 0.45, 0.4)]},
    {"id": "yuki", "name": "Yuki", "background": "38, accountant at Meridian. Chen's wife, exhausted.", "temperament": "Diligent, exhausted, internalizes stress", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "burnout and resentment about workload", "role": "office_worker", "laid_off": False, "base_valence": 0.45, "base_arousal": 0.2, "base_tension": 0.25, "base_impulse": 0.8, "base_energy": 0.5, "base_vulnerability": 0.3, "relationships_spec": [("chen", "Chen", 0.9, 0.85), ("lena", "Lena", 0.4, 0.35), ("priya", "Priya", 0.45, 0.4)]},
    {"id": "mike", "name": "Mike", "background": "22, intern at Meridian. Carlos's son, idealistic.", "temperament": "Idealistic, passionate, naive", "identity_tags": ["office_worker"], "coalitions": ["meridian_staff"], "rival_coalitions": [], "private_burden": "wants to prove himself to his father", "role": "office_worker", "laid_off": False, "base_valence": 0.65, "base_arousal": 0.25, "base_tension": 0.1, "base_impulse": 0.65, "base_energy": 0.9, "base_vulnerability": 0.2, "relationships_spec": [("carlos", "Carlos", 0.9, 0.85), ("mika", "Mika", 0.45, 0.4), ("jake", "Jake", 0.4, 0.35)]},
]

# ── Helper utilities ─────────────────────────────────────────────────────────

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def jitter(v, scale=0.02):
    return clamp(v + random.gauss(0, scale))


def smooth_toward(current, target, rate=0.05):
    return current + (target - current) * rate


def tick_to_time(tick):
    """tick 1 = Day 1 01:00, tick 24 = Day 1 00:00 (midnight), tick 25 = Day 2 01:00"""
    day = (tick - 1) // TICKS_PER_DAY + 1
    hour = (tick - 1) % TICKS_PER_DAY + 1
    if hour == 24:
        hour = 0
        day += 1
    return day, hour


def time_label(day, hour):
    return f"Day {day}, {hour:02d}:00"


# ── Location scheduling ─────────────────────────────────────────────────────

def get_location(agent_def, day, hour, is_laid_off_now):
    role = agent_def["role"]
    aid = agent_def["id"]

    # Night: everyone home (22-6)
    if hour >= 22 or hour <= 5:
        return "home"

    if role == "bartender":
        if 16 <= hour <= 23:
            return "bar"
        elif 10 <= hour <= 15:
            r = random.random()
            if r < 0.3:
                return "market"
            elif r < 0.5:
                return "park"
            return "home"
        return "home"

    if role == "teacher":
        if 7 <= hour <= 15:
            return "school"
        elif 16 <= hour <= 18:
            r = random.random()
            if r < 0.3:
                return "park"
            elif r < 0.15:
                return "market"
            return "home"
        return "home"

    if role == "retiree":
        if aid == "pastor_james":
            if 8 <= hour <= 12:
                return "church"
            elif 13 <= hour <= 15:
                r = random.random()
                if r < 0.3:
                    return "market"
                elif r < 0.5:
                    return "park"
                return "home"
            elif 18 <= hour <= 20:
                if random.random() < 0.2:
                    return "bar"
                return "home"
            return "home"
        # other retirees
        if 9 <= hour <= 11:
            r = random.random()
            if r < 0.3:
                return "market"
            elif r < 0.5:
                return "park"
            elif r < 0.6:
                return "church"
            return "home"
        elif 14 <= hour <= 16:
            return random.choice(["park", "home", "market"])
        elif 18 <= hour <= 20:
            if random.random() < 0.15:
                return "bar"
            return "home"
        return "home"

    # Office workers and managers
    if is_laid_off_now:
        # Laid-off workers: stay home more, go to bar/park/church
        if 9 <= hour <= 12:
            r = random.random()
            if r < 0.4:
                return "home"
            elif r < 0.55:
                return "park"
            elif r < 0.65:
                return "church"
            elif r < 0.75:
                return "market"
            return "home"
        elif 13 <= hour <= 17:
            r = random.random()
            if r < 0.45:
                return "home"
            elif r < 0.6:
                return "park"
            elif r < 0.7:
                return "bar"
            elif r < 0.8:
                return "church"
            return "market"
        elif 18 <= hour <= 21:
            r = random.random()
            if r < 0.35:
                return "bar"
            elif r < 0.55:
                return "home"
            elif r < 0.65:
                return "park"
            return "home"
        return "home"
    else:
        # Working office workers
        if 8 <= hour <= 17:
            return "office"
        elif 18 <= hour <= 20:
            r = random.random()
            if r < 0.25:
                return "bar"
            elif r < 0.35:
                return "market"
            return "home"
        return "home"


# ── Action selection ─────────────────────────────────────────────────────────

def pick_action(location, valence, arousal, tension, energy, hour, is_laid_off_now):
    if hour >= 22 or hour <= 5:
        return "REST"

    if tension > 0.7 and arousal > 0.6:
        r = random.random()
        if r < 0.25:
            return "LASH_OUT"
        elif r < 0.45:
            return "CONFRONT"
        elif r < 0.6:
            return "VENT"
        elif r < 0.75:
            return "WITHDRAW"
        return "RUMINATE"

    if valence < 0.2 and energy < 0.3:
        r = random.random()
        if r < 0.3:
            return "COLLAPSE"
        elif r < 0.6:
            return "WITHDRAW"
        return "SEEK_COMFORT"

    if valence < 0.3:
        r = random.random()
        if r < 0.3:
            return "RUMINATE"
        elif r < 0.5:
            return "SEEK_COMFORT"
        elif r < 0.65:
            return "WITHDRAW"
        elif r < 0.75:
            return "VENT"
        return "IDLE"

    if location == "office":
        return "WORK"
    if location == "bar":
        r = random.random()
        if r < 0.6:
            return "SOCIALIZE"
        elif r < 0.8:
            return "VENT" if valence < 0.45 else "SOCIALIZE"
        return "IDLE"
    if location == "park":
        return random.choice(["IDLE", "SOCIALIZE", "REST"])
    if location == "church":
        return random.choice(["REST", "IDLE", "SEEK_COMFORT"])
    if location == "school":
        return "WORK"
    if location == "market":
        return random.choice(["SOCIALIZE", "IDLE"])
    if location == "home":
        if energy < 0.4:
            return "REST"
        if is_laid_off_now and valence < 0.4:
            return random.choice(["RUMINATE", "IDLE", "SEEK_COMFORT"])
        return random.choice(["REST", "IDLE", "SOCIALIZE"])

    return "IDLE"


# ── Emotional surface/internal labels ────────────────────────────────────────

def surface_label(valence, arousal, tension, mask_level):
    if mask_level > 0.5 and tension > 0.3:
        return random.choice(["forced cheerfulness", "stiff politeness", "professional calm"])
    if valence > 0.65:
        return random.choice(["warm", "cheerful", "bright", "relaxed"])
    if valence > 0.45:
        return random.choice(["neutral", "composed", "polite", "calm"])
    if valence > 0.3:
        return random.choice(["subdued", "quiet", "withdrawn", "flat"])
    return random.choice(["tense", "strained", "hollow", "agitated"])


def internal_label(valence, arousal, tension):
    if tension > 0.6:
        return random.choice(["anxious", "angry", "overwhelmed", "desperate"])
    if valence < 0.25:
        return random.choice(["despairing", "numb", "bitter", "crushed"])
    if valence < 0.4:
        return random.choice(["worried", "uneasy", "sad", "restless"])
    if valence > 0.65:
        return random.choice(["content", "hopeful", "steady", "grateful"])
    return random.choice(["neutral", "okay", "calm", "thoughtful"])


# ── Inner voice / concern / interpretation generators ────────────────────────

CONCERNS_NORMAL = [
    "keep things steady", "do good work", "stay connected", "find balance",
    "make progress", "support the team", "stay on track",
]
CONCERNS_STRESSED = [
    "hold it together", "figure out what happens next", "protect the family",
    "find a new path", "not fall apart", "keep some dignity", "survive this",
]
CONCERNS_LAID_OFF = [
    "find work before savings run out", "not drag everyone down",
    "figure out what went wrong", "keep the family afloat",
    "prove I'm still worth something", "find a way forward",
]

INNER_VOICES_NORMAL = [
    "Keep moving and do not make this bigger than it is.",
    "One day at a time.", "Just focus on what's in front of you.",
    "Things are okay. Stay steady.", "Nothing to panic about yet.",
]
INNER_VOICES_STRESSED = [
    "This isn't sustainable.", "Something has to give.",
    "I can feel it coming apart.", "Don't show how scared you are.",
    "Breathe. You've been through worse.", "Hold on a little longer.",
]
INNER_VOICES_LAID_OFF = [
    "What am I even worth now?", "I gave them everything.",
    "How do I face everyone?", "The bills don't stop.",
    "I should have seen this coming.", "Start over. Again.",
    "Keep it together for the family.",
]

INTERPS_NORMAL = [
    "Nothing feels urgent right now.", "A regular day, all things considered.",
    "Things are moving along.", "Quiet for now.", "Routine holds.",
]
INTERPS_STRESSED = [
    "Something feels off.", "The tension is hard to ignore.",
    "People are worried and it shows.", "Can't shake the unease.",
    "The mood at work has shifted.",
]
INTERPS_LAID_OFF = [
    "Everything changed overnight.", "I'm on the outside now.",
    "The silence is deafening.", "They moved on without me.",
    "Each day feels heavier than the last.",
]


def pick_concern(is_laid_off_now, valence, tension):
    if is_laid_off_now:
        return random.choice(CONCERNS_LAID_OFF)
    if tension > 0.4 or valence < 0.35:
        return random.choice(CONCERNS_STRESSED)
    return random.choice(CONCERNS_NORMAL)


def pick_inner_voice(is_laid_off_now, valence, tension):
    if is_laid_off_now:
        return random.choice(INNER_VOICES_LAID_OFF)
    if tension > 0.4 or valence < 0.35:
        return random.choice(INNER_VOICES_STRESSED)
    return random.choice(INNER_VOICES_NORMAL)


def pick_interpretation(is_laid_off_now, valence, tension):
    if is_laid_off_now:
        return random.choice(INTERPS_LAID_OFF)
    if tension > 0.4 or valence < 0.35:
        return random.choice(INTERPS_STRESSED)
    return random.choice(INTERPS_NORMAL)


# ── Future branches ──────────────────────────────────────────────────────────

def make_branches(agent_def, is_laid_off_now, valence, tension):
    name = agent_def["name"]
    if is_laid_off_now:
        return [
            {"label": "Likely path", "summary": f"If nothing changes, {name} sinks further into anxiety and withdrawal."},
            {"label": "Recovery path", "summary": f"If {name} gets support or a lead, slow recovery becomes possible."},
            {"label": "Crisis path", "summary": f"If pressure mounts, {name} risks lashing out or collapsing."},
        ]
    if tension > 0.4:
        return [
            {"label": "Likely path", "summary": f"{name} stays tense but functional, watching for the next shoe to drop."},
            {"label": "Pressure path", "summary": f"If stress continues, {name} may confront someone or withdraw."},
            {"label": "Relief path", "summary": f"If things settle, {name} gradually returns to normal."},
        ]
    return [
        {"label": "Likely path", "summary": f"If nothing changes, {name} keeps trying to keep things steady."},
        {"label": "Pressure path", "summary": f"If pressure rises, {name} adjusts coping strategies."},
        {"label": "Support path", "summary": f"If connections deepen, {name} finds more stability."},
    ]


# ── Blame / support / mask / action-style ────────────────────────────────────

BLAME_TARGETS = ["circumstances", "self", "management", "the economy", "bad luck", "Richard", "the system"]
MASK_STYLES = ["little masking", "polite facade", "professional distance", "deliberate cheerfulness", "stoic front", "no mask"]
ACTION_STYLES = ["plainspoken honesty", "careful diplomacy", "blunt directness", "quiet observation", "nervous energy", "warm openness"]


def pick_blame(is_laid_off_now, tension, agent_id):
    if is_laid_off_now:
        r = random.random()
        if r < 0.3:
            return "management"
        if r < 0.5:
            return "Richard"
        if r < 0.65:
            return "the system"
        if r < 0.8:
            return "self"
        return "circumstances"
    if tension > 0.4:
        return random.choice(["circumstances", "management", "self", "the economy"])
    return "circumstances"


# ── Build agent state map (per-agent mutable state) ──────────────────────────

def init_agent_states():
    states = {}
    for ad in AGENT_DEFS:
        rels = []
        for (oid, oname, trust, warmth) in ad["relationships_spec"]:
            rels.append({
                "other_id": oid, "other_name": oname,
                "trust": trust, "warmth": warmth,
                "resentment_toward": 0.0, "resentment_from": 0.0,
                "grievance_toward": 0.0, "grievance_from": 0.0,
                "debt_toward": 0.0, "debt_from": 0.0,
                "alliance_strength": 0.0, "rivalry": 0.0,
                "support_events": 0, "conflict_events": 0, "betrayal_events": 0,
            })
        states[ad["id"]] = {
            "valence": ad["base_valence"],
            "arousal": ad["base_arousal"],
            "tension": ad["base_tension"],
            "impulse_control": ad["base_impulse"],
            "energy": ad["base_energy"],
            "vulnerability": ad["base_vulnerability"],
            "relationships": rels,
            "recent_memories": [],
            "economic_pressure": 0.1 if ad["role"] == "office_worker" else 0.05,
            "loyalty_pressure": 0.05,
            "secrecy_pressure": 0.0,
            "opportunity_pressure": 0.0,
            "debt_pressure": 0.0,
            "secret_pressure": 0.0,
            "ambition": 0.3 if ad["role"] == "office_worker" else 0.1,
        }
    return states


# ── Event definitions ────────────────────────────────────────────────────────

def get_events(tick, day, hour):
    events = []
    # Day 4, 10:00 — rumors
    if day == 4 and hour == 10:
        events.append({
            "type": "rumor",
            "description": "Whispers of layoffs circulate through the Meridian Corp office. No official announcement yet.",
            "affected": ["all_office"],
        })
    # Day 4, 14:00 — HR meeting
    if day == 4 and hour == 14:
        events.append({
            "type": "announcement",
            "description": "Diana is seen entering Richard's office with a stack of folders. The rumor mill accelerates.",
            "affected": ["all_office"],
        })
    # Day 5, 9:00 — layoffs
    if day == 5 and hour == 9:
        events.append({
            "type": "layoff",
            "description": "Meridian Corp announces restructuring. Marcus, Rosa, Jake, Mika, Greg, Nadia, and Carlos are terminated effective immediately.",
            "affected": list(LAID_OFF_IDS),
        })
    # Day 5, 12:00 — aftermath
    if day == 5 and hour == 12:
        events.append({
            "type": "emotional_wave",
            "description": "The office is stunned. Remaining workers barely speak. The laid-off employees collect their things.",
            "affected": ["all_office"],
        })
    # Day 6, 19:00 — bar gathering
    if day == 6 and hour == 19:
        events.append({
            "type": "gathering",
            "description": "Several laid-off workers and friends gather at The Tap. Frank pours drinks on the house.",
            "affected": ["marcus", "jake", "greg", "frank", "jenny", "tom", "ray"],
        })
    # Day 7, 10:00 — church
    if day == 7 and hour == 10:
        events.append({
            "type": "community",
            "description": "Pastor James holds a special community gathering at the church. Several affected families attend.",
            "affected": ["pastor_james", "marcus", "lisa", "carlos", "mike", "rosa", "nadia", "maria"],
        })
    # Day 8, 14:00 — David confronts Richard
    if day == 8 and hour == 14:
        events.append({
            "type": "confrontation",
            "description": "David confronts Richard in the office about Rosa's layoff. Security has to intervene.",
            "affected": ["david", "richard", "diana"],
        })
    # Day 10, 11:00 — market community support
    if day == 10 and hour == 11:
        events.append({
            "type": "community",
            "description": "Elena and Maria organize a community support drive at the Town Market for affected families.",
            "affected": ["elena", "maria", "lisa", "rosa", "nadia", "carlos"],
        })
    # Day 12, 18:00 — Mike protests
    if day == 12 and hour == 18:
        events.append({
            "type": "protest",
            "description": "Mike organizes a small protest outside Meridian Corp. A handful of people show up.",
            "affected": ["mike", "carlos", "greg", "mika", "jake"],
        })
    # Day 14, 15:00 — Marcus gets a lead
    if day == 14 and hour == 15:
        events.append({
            "type": "opportunity",
            "description": "Marcus gets a call about a potential logistics position in the next town over.",
            "affected": ["marcus", "lisa"],
        })
    return events


# ── Interaction generation ───────────────────────────────────────────────────

def generate_interactions(agent_locations, agent_states_map, day, hour, is_laid_off_map):
    interactions = []
    if hour < 6 or hour > 22:
        return interactions

    # Group agents by location
    loc_groups = {}
    for aid, loc in agent_locations.items():
        loc_groups.setdefault(loc, []).append(aid)

    agent_map = {ad["id"]: ad for ad in AGENT_DEFS}

    for loc, agents_here in loc_groups.items():
        if len(agents_here) < 2:
            continue
        # Generate a few interactions per location
        n = min(len(agents_here) // 2, 3)
        pairs_tried = set()
        for _ in range(n):
            if len(agents_here) < 2:
                break
            a1 = random.choice(agents_here)
            a2 = random.choice(agents_here)
            if a1 == a2:
                continue
            pair = tuple(sorted([a1, a2]))
            if pair in pairs_tried:
                continue
            pairs_tried.add(pair)

            s1 = agent_states_map[a1]
            s2 = agent_states_map[a2]

            avg_tension = (s1["tension"] + s2["tension"]) / 2
            avg_valence = (s1["valence"] + s2["valence"]) / 2

            if avg_tension > 0.5 and random.random() < 0.3:
                itype = "conflict"
                desc = f"{agent_map[a1]['name']} and {agent_map[a2]['name']} exchange heated words."
            elif avg_valence < 0.35 and random.random() < 0.4:
                itype = "support"
                desc = f"{agent_map[a1]['name']} offers quiet support to {agent_map[a2]['name']}."
            elif avg_valence > 0.55:
                itype = "neutral"
                desc = f"{agent_map[a1]['name']} and {agent_map[a2]['name']} chat casually."
            else:
                itype = "neutral"
                desc = f"{agent_map[a1]['name']} and {agent_map[a2]['name']} acknowledge each other."

            interactions.append({
                "agents": [a1, a2],
                "type": itype,
                "location": loc,
                "description": desc,
            })

            # Update relationship states
            for s, other in [(s1, a2), (s2, a1)]:
                for rel in s["relationships"]:
                    if rel["other_id"] == other:
                        if itype == "conflict":
                            rel["conflict_events"] += 1
                            rel["trust"] = clamp(rel["trust"] - 0.02)
                            rel["resentment_toward"] = clamp(rel["resentment_toward"] + 0.05)
                        elif itype == "support":
                            rel["support_events"] += 1
                            rel["trust"] = clamp(rel["trust"] + 0.01)
                            rel["warmth"] = clamp(rel["warmth"] + 0.01)
                        break

    return interactions


# ── Main generation ──────────────────────────────────────────────────────────

def generate():
    print("Generating mock snapshot...")

    agent_map = {ad["id"]: ad for ad in AGENT_DEFS}
    agents_json = {}
    for ad in AGENT_DEFS:
        agents_json[ad["id"]] = {
            "id": ad["id"],
            "name": ad["name"],
            "background": ad["background"],
            "temperament": ad["temperament"],
            "identity_tags": ad["identity_tags"],
            "coalitions": ad["coalitions"],
            "rival_coalitions": ad["rival_coalitions"],
            "private_burden": ad["private_burden"],
        }

    states = init_agent_states()
    ticks = []

    layoff_day = 5
    rumor_day = 4

    for tick in range(1, TOTAL_TICKS + 1):
        day, hour = tick_to_time(tick)

        # ── Phase effects on emotional baselines ──
        is_rumor_phase = (day == rumor_day and hour >= 10) or (day == rumor_day and hour < 10 and False)  # only after 10am Day 4
        is_rumor_phase = day == rumor_day and hour >= 10
        is_layoff_tick = day == layoff_day and hour == 9
        days_since_layoff = max(0, (tick - (layoff_day - 1) * 24 - 9)) / 24.0 if tick >= (layoff_day - 1) * 24 + 9 else 0
        is_post_layoff = tick > (layoff_day - 1) * 24 + 9

        events = get_events(tick, day, hour)

        # Compute agent locations and states
        agent_locations = {}
        agent_states_tick = {}

        for ad in AGENT_DEFS:
            aid = ad["id"]
            s = states[aid]
            is_laid_off_now = ad["laid_off"] and is_post_layoff
            is_office_worker = ad["role"] in ("office_worker", "manager")

            # ── Emotional dynamics ──

            # Baseline drift
            base_v = ad["base_valence"]
            base_a = ad["base_arousal"]
            base_t = ad["base_tension"]
            base_e = ad["base_energy"]
            base_vul = ad["base_vulnerability"]

            # Rumor phase: office workers get anxious
            if is_rumor_phase and is_office_worker:
                base_a += 0.15
                base_t += 0.1
                base_v -= 0.1

            # Layoff moment
            if is_layoff_tick and ad["laid_off"]:
                s["valence"] = clamp(s["valence"] - 0.35)
                s["arousal"] = clamp(s["arousal"] + 0.4)
                s["tension"] = clamp(s["tension"] + 0.4)
                s["vulnerability"] = clamp(s["vulnerability"] + 0.3)
                s["energy"] = clamp(s["energy"] - 0.2)
                s["economic_pressure"] = clamp(s["economic_pressure"] + 0.4)
                s["recent_memories"].append(f"Day {day}: Laid off from Meridian Corp.")

            # Layoff moment for survivors: guilt, tension
            if is_layoff_tick and is_office_worker and not ad["laid_off"]:
                s["tension"] = clamp(s["tension"] + 0.15)
                s["valence"] = clamp(s["valence"] - 0.1)
                s["arousal"] = clamp(s["arousal"] + 0.1)

            # Special: Richard and Victoria on layoff day
            if is_layoff_tick and aid == "richard":
                s["tension"] = clamp(s["tension"] + 0.25)
                s["valence"] = clamp(s["valence"] - 0.2)
                s["vulnerability"] = clamp(s["vulnerability"] + 0.15)
            if is_layoff_tick and aid == "victoria":
                s["tension"] = clamp(s["tension"] + 0.1)

            # Special: Diana on layoff day (guilt)
            if is_layoff_tick and aid == "diana":
                s["tension"] = clamp(s["tension"] + 0.2)
                s["valence"] = clamp(s["valence"] - 0.15)
                s["vulnerability"] = clamp(s["vulnerability"] + 0.2)

            # Special: Lisa when Marcus laid off
            if is_layoff_tick and aid == "lisa":
                s["tension"] = clamp(s["tension"] + 0.2)
                s["valence"] = clamp(s["valence"] - 0.2)
                s["vulnerability"] = clamp(s["vulnerability"] + 0.15)

            # Special: David when Rosa laid off
            if is_layoff_tick and aid == "david":
                s["tension"] = clamp(s["tension"] + 0.25)
                s["valence"] = clamp(s["valence"] - 0.15)
                s["arousal"] = clamp(s["arousal"] + 0.2)

            # Special: Mike when Carlos laid off
            if is_layoff_tick and aid == "mike":
                s["tension"] = clamp(s["tension"] + 0.2)
                s["valence"] = clamp(s["valence"] - 0.15)
                s["arousal"] = clamp(s["arousal"] + 0.15)

            # Post-layoff recovery / sustained distress
            if is_post_layoff:
                if ad["laid_off"]:
                    # Slow, painful drift with some recovery over days
                    recovery_rate = 0.003 + days_since_layoff * 0.0003  # very slow
                    recovery_rate = min(recovery_rate, 0.015)
                    target_v = max(ad["base_valence"] - 0.2, 0.2)
                    target_t = min(ad["base_tension"] + 0.15, 0.6)
                    target_a = ad["base_arousal"] + 0.05
                    target_e = max(ad["base_energy"] - 0.15, 0.3)
                    target_vul = min(ad["base_vulnerability"] + 0.15, 0.7)

                    # Day 14 Marcus gets a lead
                    if aid == "marcus" and day >= 14:
                        target_v += 0.1
                        target_t -= 0.05
                        recovery_rate *= 1.5

                    s["valence"] = smooth_toward(s["valence"], target_v, recovery_rate)
                    s["tension"] = smooth_toward(s["tension"], target_t, recovery_rate)
                    s["arousal"] = smooth_toward(s["arousal"], target_a, recovery_rate * 0.5)
                    s["energy"] = smooth_toward(s["energy"], target_e, recovery_rate * 0.8)
                    s["vulnerability"] = smooth_toward(s["vulnerability"], target_vul, recovery_rate * 0.5)
                    s["economic_pressure"] = clamp(s["economic_pressure"] + 0.001)

                elif is_office_worker:
                    # Survivor guilt fades slowly
                    survivor_recovery = 0.008
                    s["tension"] = smooth_toward(s["tension"], ad["base_tension"] + 0.05, survivor_recovery)
                    s["valence"] = smooth_toward(s["valence"], ad["base_valence"] - 0.05, survivor_recovery)

                # Family/partner effects
                if aid == "lisa" and is_post_layoff:
                    target_v = max(0.3, ad["base_valence"] - 0.15)
                    s["valence"] = smooth_toward(s["valence"], target_v, 0.005)
                    s["tension"] = smooth_toward(s["tension"], ad["base_tension"] + 0.1, 0.005)
                if aid == "david" and is_post_layoff:
                    s["tension"] = smooth_toward(s["tension"], ad["base_tension"] + 0.1, 0.006)
                if aid == "mike" and is_post_layoff:
                    s["tension"] = smooth_toward(s["tension"], ad["base_tension"] + 0.08, 0.005)
                    s["valence"] = smooth_toward(s["valence"], ad["base_valence"] - 0.1, 0.005)

            # Normal drift toward baseline (when not in special phase)
            if not is_post_layoff and not is_rumor_phase:
                s["valence"] = smooth_toward(s["valence"], base_v, 0.02)
                s["arousal"] = smooth_toward(s["arousal"], base_a, 0.03)
                s["tension"] = smooth_toward(s["tension"], base_t, 0.02)
                s["energy"] = smooth_toward(s["energy"], base_e, 0.02)
                s["vulnerability"] = smooth_toward(s["vulnerability"], base_vul, 0.02)
            elif is_rumor_phase and not is_post_layoff:
                s["valence"] = smooth_toward(s["valence"], base_v, 0.03)
                s["arousal"] = smooth_toward(s["arousal"], base_a, 0.04)
                s["tension"] = smooth_toward(s["tension"], base_t, 0.03)

            # Diurnal energy cycle
            if 6 <= hour <= 10:
                s["energy"] = clamp(s["energy"] + 0.01)
            elif 13 <= hour <= 14:
                s["energy"] = clamp(s["energy"] - 0.01)  # post-lunch dip
            elif hour >= 20:
                s["energy"] = clamp(s["energy"] - 0.015)
            elif hour <= 5:
                s["energy"] = clamp(s["energy"] + 0.02)  # sleep recovery

            # Jitter all values
            s["valence"] = jitter(s["valence"], 0.012)
            s["arousal"] = jitter(s["arousal"], 0.01)
            s["tension"] = jitter(s["tension"], 0.01)
            s["energy"] = jitter(s["energy"], 0.008)
            s["vulnerability"] = jitter(s["vulnerability"], 0.005)
            s["impulse_control"] = jitter(s["impulse_control"], 0.005)

            # ── Event-specific emotional shocks ──
            for ev in events:
                affected = ev.get("affected", [])
                applies = (aid in affected) or ("all_office" in affected and is_office_worker)
                if not applies:
                    continue

                if ev["type"] == "rumor":
                    s["arousal"] = clamp(s["arousal"] + 0.08)
                    s["tension"] = clamp(s["tension"] + 0.06)
                    s["valence"] = clamp(s["valence"] - 0.05)
                    if len(s["recent_memories"]) < 10:
                        s["recent_memories"].append(f"Day {day}: Heard layoff rumors at work.")

                elif ev["type"] == "gathering":
                    s["valence"] = clamp(s["valence"] + 0.05)
                    s["tension"] = clamp(s["tension"] - 0.03)

                elif ev["type"] == "community":
                    s["valence"] = clamp(s["valence"] + 0.06)
                    s["vulnerability"] = clamp(s["vulnerability"] - 0.02)
                    s["tension"] = clamp(s["tension"] - 0.03)

                elif ev["type"] == "confrontation":
                    if aid == "david":
                        s["arousal"] = clamp(s["arousal"] + 0.3)
                        s["tension"] = clamp(s["tension"] + 0.2)
                        if len(s["recent_memories"]) < 10:
                            s["recent_memories"].append(f"Day {day}: Confronted Richard about Rosa's layoff.")
                    elif aid == "richard":
                        s["tension"] = clamp(s["tension"] + 0.15)
                        s["vulnerability"] = clamp(s["vulnerability"] + 0.1)
                        if len(s["recent_memories"]) < 10:
                            s["recent_memories"].append(f"Day {day}: David confronted me about the layoffs.")
                    elif aid == "diana":
                        s["tension"] = clamp(s["tension"] + 0.1)

                elif ev["type"] == "protest":
                    if aid == "mike":
                        s["arousal"] = clamp(s["arousal"] + 0.2)
                        s["valence"] = clamp(s["valence"] + 0.05)  # feels empowered
                        if len(s["recent_memories"]) < 10:
                            s["recent_memories"].append(f"Day {day}: Organized protest outside Meridian.")
                    else:
                        s["arousal"] = clamp(s["arousal"] + 0.1)

                elif ev["type"] == "opportunity":
                    if aid == "marcus":
                        s["valence"] = clamp(s["valence"] + 0.15)
                        s["tension"] = clamp(s["tension"] - 0.08)
                        s["energy"] = clamp(s["energy"] + 0.1)
                        if len(s["recent_memories"]) < 10:
                            s["recent_memories"].append(f"Day {day}: Got a call about a logistics job.")
                    elif aid == "lisa":
                        s["valence"] = clamp(s["valence"] + 0.1)
                        s["tension"] = clamp(s["tension"] - 0.05)

            # ── Location ──
            location = get_location(ad, day, hour, is_laid_off_now)
            agent_locations[aid] = location

            # ── Action ──
            action = pick_action(location, s["valence"], s["arousal"], s["tension"], s["energy"], hour, is_laid_off_now)

            # ── Derived labels ──
            mask_level = 0.0
            if s["tension"] > 0.3 and s["impulse_control"] > 0.7:
                mask_level = 0.5 + (s["impulse_control"] - 0.7) * 1.5

            divergence = abs(s["valence"] - 0.5) * mask_level
            sur = surface_label(s["valence"], s["arousal"], s["tension"], mask_level)
            internal = internal_label(s["valence"], s["arousal"], s["tension"])

            concern = pick_concern(is_laid_off_now, s["valence"], s["tension"])
            interpretation = pick_interpretation(is_laid_off_now, s["valence"], s["tension"])
            blame = pick_blame(is_laid_off_now, s["tension"], aid)
            inner_voice = pick_inner_voice(is_laid_off_now, s["valence"], s["tension"])

            # Support target: pick from relationships
            support_target = ""
            if s["relationships"]:
                best_rel = max(s["relationships"], key=lambda r: r["trust"] + r["warmth"])
                support_target = best_rel["other_id"]

            mask_style = random.choice(MASK_STYLES) if mask_level > 0.3 else "little masking"
            action_style = random.choice(ACTION_STYLES)

            priority_motive = "stay steady"
            if is_laid_off_now:
                priority_motive = random.choice(["find work", "protect family", "survive", "keep dignity"])
            elif s["tension"] > 0.4:
                priority_motive = random.choice(["hold it together", "stay alert", "protect what matters"])

            branches = make_branches(ad, is_laid_off_now, s["valence"], s["tension"])

            # Round all floats
            def r2(v):
                return round(v, 2)

            rels_snapshot = []
            for rel in s["relationships"]:
                rels_snapshot.append({
                    "other_id": rel["other_id"],
                    "other_name": rel["other_name"],
                    "trust": r2(rel["trust"]),
                    "warmth": r2(rel["warmth"]),
                    "resentment_toward": r2(rel["resentment_toward"]),
                    "resentment_from": r2(rel["resentment_from"]),
                    "grievance_toward": r2(rel["grievance_toward"]),
                    "grievance_from": r2(rel["grievance_from"]),
                    "debt_toward": r2(rel["debt_toward"]),
                    "debt_from": r2(rel["debt_from"]),
                    "alliance_strength": r2(rel["alliance_strength"]),
                    "rivalry": r2(rel["rivalry"]),
                    "support_events": rel["support_events"],
                    "conflict_events": rel["conflict_events"],
                    "betrayal_events": rel["betrayal_events"],
                })

            agent_states_tick[aid] = {
                "id": aid,
                "name": ad["name"],
                "location": location,
                "action": action,
                "arousal": r2(s["arousal"]),
                "valence": r2(s["valence"]),
                "tension": r2(s["tension"]),
                "impulse_control": r2(s["impulse_control"]),
                "energy": r2(s["energy"]),
                "vulnerability": r2(s["vulnerability"]),
                "surface": sur,
                "internal": internal,
                "divergence": r2(divergence),
                "primary_concern": concern,
                "interpretation": interpretation,
                "blame_target": blame,
                "support_target": support_target,
                "economic_pressure": r2(s["economic_pressure"]),
                "loyalty_pressure": r2(s["loyalty_pressure"]),
                "secrecy_pressure": r2(s["secrecy_pressure"]),
                "opportunity_pressure": r2(s["opportunity_pressure"]),
                "priority_motive": priority_motive,
                "mask_style": mask_style,
                "action_style": action_style,
                "inner_voice": inner_voice,
                "future_branches": branches,
                "coalitions": ad["coalitions"],
                "identity_tags": ad["identity_tags"],
                "private_burden": ad["private_burden"],
                "debt_pressure": r2(s["debt_pressure"]),
                "secret_pressure": r2(s["secret_pressure"]),
                "ambition": r2(s["ambition"]),
                "relationships": rels_snapshot,
                "recent_memories": list(s["recent_memories"]),
            }

        # ── Interactions ──
        interactions = generate_interactions(agent_locations, states, day, hour, {ad["id"]: ad["laid_off"] for ad in AGENT_DEFS})

        # Format events for output
        events_out = []
        for ev in events:
            events_out.append({
                "type": ev["type"],
                "description": ev["description"],
            })

        ticks.append({
            "tick": tick,
            "time": time_label(day, hour),
            "events": events_out,
            "interactions": interactions,
            "agent_states": agent_states_tick,
        })

        if tick % 50 == 0:
            print(f"  Generated tick {tick}/{TOTAL_TICKS}")

    snapshot = {
        "scenario": "small_town",
        "total_ticks": TOTAL_TICKS,
        "locations": LOCATIONS,
        "agents": agents_json,
        "ticks": ticks,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(snapshot, f, indent=None, separators=(",", ":"))

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Done! Written to {OUTPUT_PATH}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Ticks: {TOTAL_TICKS}, Agents: {len(AGENT_DEFS)}")


if __name__ == "__main__":
    generate()

#!/usr/bin/env python3
"""Large-scale town simulation: 300 agents across 8 districts.

Three simultaneous crisis chains cascade across the town:
  1. Industrial: chemical leak → factory shutdown → explosion
  2. Political: corruption revealed → protests → mayor resigns
  3. Community: contamination scare → school closure → mutual aid

Cross-district dynamics: factory workers live in suburbs, students
join worker protests, market vendors lose customers, healthcare
workers see contamination patients.

Usage:
    from learned_brain.world_sim.scenarios_large import build_large_town
    world, meta = build_large_town()
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .human_profiles import assign_human_profile
from .world import World, Location, ScheduledEvent
from .world_agent import WorldAgent, Personality


# ─── Locations (25 locations across 8 districts + 3 shared) ──────────────────

ALL_LOCATIONS = [
    # Industrial Quarter
    Location("factory_floor", "Consolidated Chemical Factory",
             "Operating machinery, handling chemicals, assembly line work"),
    Location("warehouse", "Factory Warehouse",
             "Loading inventory, checking shipments, stacking crates"),
    Location("workers_canteen", "Workers' Canteen",
             "Eating lunch, chatting about shifts, playing cards"),

    # Downtown
    Location("office_tower", "Central Business Tower",
             "Working at desks, attending meetings, reviewing reports"),
    Location("trading_floor", "Financial Trading Floor",
             "Monitoring markets, making calls, analyzing data"),
    Location("downtown_cafe", "The Corner Grind Café",
             "Sipping coffee, working on laptops, casual meetings"),

    # University
    Location("lecture_hall", "Morrison Lecture Hall",
             "Attending lectures, taking notes, discussing theories"),
    Location("library", "University Library",
             "Studying quietly, researching, writing papers"),
    Location("student_union", "Student Union Building",
             "Socializing, organizing events, grabbing food"),

    # Market District
    Location("main_market", "Riverside Market",
             "Selling goods, haggling, arranging displays"),
    Location("food_court", "Market Food Court",
             "Cooking, serving food, eating lunch"),
    Location("artisan_alley", "Artisan Alley",
             "Creating crafts, displaying art, talking to visitors"),

    # Waterfront
    Location("docks", "Harbor Docks",
             "Loading boats, repairing nets, hauling catch"),
    Location("fish_market", "Fish Market",
             "Selling fresh catch, negotiating prices"),
    Location("harbor_bar", "The Rusty Anchor Bar",
             "Drinking beer, telling stories, watching sports"),

    # Government Hill
    Location("city_hall", "City Hall",
             "Attending council meetings, filing paperwork"),
    Location("courthouse", "County Courthouse",
             "Hearing cases, filing motions, waiting in hallways"),
    Location("gov_offices", "Government Office Complex",
             "Processing applications, answering phones"),

    # Suburbs North
    Location("hospital", "St. Mary's Hospital",
             "Treating patients, checking vitals, updating charts"),
    Location("north_school", "Northside Elementary",
             "Teaching classes, grading papers, managing students"),
    Location("north_homes", "Suburbs North Residential",
             "Home routine, cooking, caring for family"),

    # Suburbs South
    Location("community_center", "Southside Community Center",
             "Running programs, organizing events, helping residents"),
    Location("south_homes", "Suburbs South Residential",
             "Home routine, gardening, quiet neighborhood life"),

    # Shared / Cross-District
    Location("central_park", "Central Park",
             "Walking dogs, jogging, sitting on benches"),
    Location("central_bar", "Mahoney's Pub",
             "Drinking, socializing, unwinding after work"),
]


# ─── Name Pool (320 diverse names) ───────────────────────────────────────────

NAME_POOL = [
    # Western
    "Marcus", "Rosa", "Jake", "Diana", "Tom", "Sarah", "Greg", "Kevin",
    "Lena", "Mike", "Andre", "Mika", "David", "Ray", "Richard",
    "James", "Maria", "Robert", "Elena", "Daniel", "Sofia", "William",
    "Anna", "Joseph", "Clara", "Thomas", "Laura", "George", "Emma",
    "Frank", "Nina", "Henry", "Vera", "Paul", "Rita", "Simon", "Greta",
    "Patrick", "Ingrid", "Oscar", "Ivan", "Petra", "Kurt", "Hans",
    "Brian", "Chloe", "Derek", "Fiona", "Grant", "Holly", "Ian",
    "Julia", "Keith", "Lisa", "Nathan", "Olivia", "Peter", "Quinn",
    "Russell", "Tina", "Victor", "Wendy", "Xavier", "Yvonne",
    "Adam", "Bella", "Caleb", "Daphne", "Ethan", "Flora", "Gavin",
    "Helen", "Isaac", "Jenny", "Karl", "Lydia", "Martin", "Nora",
    "Owen", "Penelope", "Trevor", "Uma", "Wade", "Zach", "Amber",
    "Blake", "Crystal", "Darren", "Eve", "Finn", "Grace", "Hugo",
    "Iris", "Joel", "Kira", "Leo", "Mona", "Nico", "Opal", "Reese",
    "Theo", "Violet", "Wyatt", "Abel", "Bianca", "Cedric", "Donna",
    "Ellis", "Freya", "Jasper", "Lance", "Noel", "Pearl", "Seth",
    "Tara", "Vince", "Wren", "Aurora", "Coral", "Dorian",
    # Latin American
    "Eduardo", "Camila", "Alejandro", "Valentina", "Diego", "Isabella",
    "Santiago", "Lucia", "Mateo", "Carmen", "Andres", "Paula",
    "Felipe", "Gabriela", "Ricardo", "Daniela", "Javier", "Adriana",
    "Miguel", "Pilar", "Rafael", "Marisol", "Hector", "Emilio",
    "Teresa", "Lorenzo", "Beatriz",
    # Asian
    "Wei", "Mei", "Jun", "Ling", "Hiroshi", "Akiko", "Kenji", "Sakura",
    "Takeshi", "Haruki", "Sanjay", "Anita", "Raj", "Deepa", "Arjun",
    "Sunita", "Vikram", "Neha", "Ravi", "Kavita", "Jin", "Hana",
    "Sung", "Yong", "Taro", "Noriko", "Koji", "Yumiko",
    "Amit", "Priti", "Rahul", "Seema", "Arun", "Lakshmi", "Priya",
    "Omar", "Chen", "Yuki",
    # African
    "Kwame", "Amara", "Kofi", "Fatima", "Jabari", "Zara", "Tendai",
    "Aisha", "Obinna", "Nneka", "Sekou", "Mariama", "Chidi", "Adaeze",
    "Yusuf", "Halima", "Bakari", "Safiya", "Emeka", "Chiamaka",
    "Jelani", "Wanjiku", "Tariq", "Layla", "Malik", "Sanaa",
    "Dayo", "Funke", "Olumide", "Ngozi",
    # Middle Eastern
    "Hassan", "Leila", "Karim", "Yasmin", "Ali", "Samira", "Farid",
    "Soraya", "Nabil", "Dina", "Samir", "Rania", "Khalid", "Maryam",
    "Mustafa", "Noura",
    # Eastern European
    "Nikolai", "Natasha", "Dmitri", "Olga", "Sergei", "Katya", "Alexei",
    "Irina", "Boris", "Svetlana", "Yuri", "Anya", "Mikhail", "Daria",
    "Vlad", "Sonya", "Victoria", "Nadia", "Carlos",
    # Extra batch 1
    "Sage", "Barrett", "Hazel", "Gideon", "Keiko", "Mabel", "Roma",
    "Uriel", "Yara", "Zeke", "Elara", "Gage", "Quincy",
    # Extra batch 2
    "Alden", "Brynn", "Caspian", "Delia", "Ezra", "Faye", "Glenn",
    "Hilda", "Idris", "Juno", "Knox", "Lila", "Myles", "Odette",
    "Penn", "Rhea", "Sterling", "Thea", "Ulric", "Vera_B", "Wilder",
    "Xander", "Yolanda", "Zuri", "Ansel", "Bria", "Cyrus", "Dove",
    "Eamon", "Gemma", "Ines", "Jude", "Kali", "Lev", "Maeve",
    "Orla", "Pierre", "Remy",
]


# ─── Role Archetypes ─────────────────────────────────────────────────────────

ROLE_ARCHETYPES = [
    {
        "role": "factory_worker",
        "count": 45,
        "work_locs": ["factory_floor", "warehouse"],
        "home_locs": ["north_homes", "south_homes"],
        "evening_locs": ["central_bar", "harbor_bar", "workers_canteen"],
        "backgrounds": [
            "assembly line operator at Consolidated Chemical, {years} years, supporting a family of {fam}",
            "warehouse handler at the chemical plant, {years} years of heavy lifting",
            "machine operator at the factory, {years} years of shift work, saving for a house",
            "quality inspector at Consolidated Chemical, {years} years, sole breadwinner",
            "forklift driver at the plant, {years} years, sends money to aging parents",
            "maintenance technician at the factory, {years} years keeping machines running",
        ],
        "temperaments": [
            "steady and hardworking, slow to anger but fierce when pushed",
            "rough around the edges but loyal, protective of coworkers",
            "quiet and methodical, bottles up frustration until it explodes",
            "gregarious and loud, uses humor to cope with stress",
            "stoic and resilient, believes in earning respect through work",
        ],
        "base_params": {
            "arousal_rise_rate": 0.72, "arousal_decay_rate": 0.86,
            "valence_momentum": 0.48, "impulse_drain_rate": 0.18,
            "impulse_restore_rate": 0.007, "energy_drain_rate": 0.08,
            "energy_regen_rate": 0.012, "vulnerability_weight": 1.1,
        },
    },
    {
        "role": "office_professional",
        "count": 35,
        "work_locs": ["office_tower", "trading_floor"],
        "home_locs": ["north_homes", "south_homes"],
        "evening_locs": ["downtown_cafe", "central_bar", "central_park"],
        "backgrounds": [
            "financial analyst at First Regional Bank, {years} years, recently divorced",
            "corporate accountant, {years} years, anxious about quarterly reviews",
            "marketing director, {years} years, two kids in private school",
            "HR specialist, {years} years, carries the weight of everyone's problems",
            "IT systems admin, {years} years keeping servers running, quiet introvert",
            "project manager, {years} years of deadline pressure, runs marathons",
        ],
        "temperaments": [
            "composed and professional, masks anxiety behind efficiency",
            "ambitious and driven, competitive but fair",
            "empathetic and thoughtful, absorbs others' stress easily",
            "analytical and reserved, processes emotions through logic",
            "outgoing and networked, uses social connections to cope",
        ],
        "base_params": {
            "arousal_rise_rate": 0.65, "arousal_decay_rate": 0.90,
            "valence_momentum": 0.50, "impulse_drain_rate": 0.12,
            "impulse_restore_rate": 0.010, "energy_drain_rate": 0.06,
            "energy_regen_rate": 0.011, "vulnerability_weight": 0.9,
        },
    },
    {
        "role": "student",
        "count": 30,
        "work_locs": ["lecture_hall", "library", "student_union"],
        "home_locs": ["south_homes", "north_homes"],
        "evening_locs": ["central_bar", "student_union", "central_park"],
        "backgrounds": [
            "chemistry major, junior year, first in family to attend college",
            "environmental science student, passionate about justice",
            "engineering student working part-time, {years} years, drowning in debt",
            "sociology grad student, researching inequality, idealistic",
            "pre-med student, {years} years of study, immigrant family expectations",
            "journalism student, always looking for the next story",
        ],
        "temperaments": [
            "idealistic and passionate, quick to mobilize for causes",
            "anxious but determined, channels stress into activism",
            "rebellious and outspoken, challenges authority reflexively",
            "quiet and observant, processes the world through writing",
            "energetic and social, the glue that holds friend groups together",
        ],
        "base_params": {
            "arousal_rise_rate": 0.80, "arousal_decay_rate": 0.84,
            "valence_momentum": 0.38, "impulse_drain_rate": 0.20,
            "impulse_restore_rate": 0.009, "energy_drain_rate": 0.05,
            "energy_regen_rate": 0.015, "vulnerability_weight": 1.2,
        },
    },
    {
        "role": "market_vendor",
        "count": 30,
        "work_locs": ["main_market", "food_court", "artisan_alley"],
        "home_locs": ["south_homes", "north_homes"],
        "evening_locs": ["central_bar", "harbor_bar", "central_park"],
        "backgrounds": [
            "fruit and vegetable vendor, {years} years, family business",
            "food stall owner, {years} years cooking for the community",
            "artisan jewelry maker, {years} years, selling handmade pieces",
            "fish monger, {years} years, buys catch fresh from docks every morning",
            "bakery owner, {years} years of 4am starts, known for sourdough",
            "secondhand bookshop owner, {years} years, the market's therapist",
        ],
        "temperaments": [
            "warm and chatty, knows everyone's business",
            "shrewd and pragmatic, can smell trouble from a mile away",
            "generous and nurturing, feeds anyone who looks hungry",
            "anxious about money but hides it behind hospitality",
            "proud and independent, built everything from scratch",
        ],
        "base_params": {
            "arousal_rise_rate": 0.70, "arousal_decay_rate": 0.87,
            "valence_momentum": 0.45, "impulse_drain_rate": 0.15,
            "impulse_restore_rate": 0.008, "energy_drain_rate": 0.07,
            "energy_regen_rate": 0.013, "vulnerability_weight": 1.0,
        },
    },
    {
        "role": "dock_worker",
        "count": 20,
        "work_locs": ["docks", "fish_market"],
        "home_locs": ["south_homes"],
        "evening_locs": ["harbor_bar", "central_bar"],
        "backgrounds": [
            "longshoreman, {years} years on the docks, union member, family man",
            "boat mechanic, {years} years keeping vessels running, quiet loner",
            "dock supervisor, {years} years, respected by the crew",
            "fisherman, {years} years on the water, weathered by salt and sun",
        ],
        "temperaments": [
            "tough and weathered, rarely shows vulnerability",
            "loyal to crew, protective of waterfront community",
            "gruff exterior but deeply caring, expresses love through action",
            "superstitious and cautious, reads the wind for trouble",
        ],
        "base_params": {
            "arousal_rise_rate": 0.68, "arousal_decay_rate": 0.88,
            "valence_momentum": 0.50, "impulse_drain_rate": 0.16,
            "impulse_restore_rate": 0.006, "energy_drain_rate": 0.09,
            "energy_regen_rate": 0.014, "vulnerability_weight": 0.85,
        },
    },
    {
        "role": "government_worker",
        "count": 25,
        "work_locs": ["city_hall", "courthouse", "gov_offices"],
        "home_locs": ["north_homes", "south_homes"],
        "evening_locs": ["downtown_cafe", "central_park"],
        "backgrounds": [
            "city clerk, {years} years processing permits, seen three mayors",
            "public defender, {years} years fighting for the underdog",
            "council aide, {years} years navigating politics, idealistic but jaded",
            "building inspector, {years} years — the one who approved the factory",
            "environmental compliance officer, {years} years, ignored warnings",
        ],
        "temperaments": [
            "by-the-book and cautious, carries guilt about past decisions",
            "idealistic but worn down by bureaucracy",
            "politically savvy, always calculating angles",
            "empathetic public servant, genuinely wants to help",
            "cynical veteran, has seen too many promises broken",
        ],
        "base_params": {
            "arousal_rise_rate": 0.60, "arousal_decay_rate": 0.92,
            "valence_momentum": 0.52, "impulse_drain_rate": 0.10,
            "impulse_restore_rate": 0.012, "energy_drain_rate": 0.05,
            "energy_regen_rate": 0.010, "vulnerability_weight": 0.80,
        },
    },
    {
        "role": "healthcare",
        "count": 25,
        "work_locs": ["hospital"],
        "home_locs": ["north_homes"],
        "evening_locs": ["central_park", "central_bar"],
        "backgrounds": [
            "ER nurse, {years} years of night shifts, single parent",
            "hospital administrator, {years} years managing chaos",
            "paramedic, {years} years on the road, burning out",
            "family doctor, {years} years, knows half the town by name",
            "lab technician, {years} years — first to see contamination data",
        ],
        "temperaments": [
            "compassionate but exhausted, running on fumes and duty",
            "clinically detached at work, falls apart at home",
            "fiercely protective of patients, confrontational with bureaucracy",
            "calm under pressure, the one everyone leans on",
            "anxious and hypervigilant, sees danger everywhere",
        ],
        "base_params": {
            "arousal_rise_rate": 0.70, "arousal_decay_rate": 0.85,
            "valence_momentum": 0.42, "impulse_drain_rate": 0.14,
            "impulse_restore_rate": 0.009, "energy_drain_rate": 0.10,
            "energy_regen_rate": 0.008, "vulnerability_weight": 1.15,
        },
    },
    {
        "role": "community",
        "count": 90,
        "work_locs": ["north_school", "community_center", "main_market",
                       "downtown_cafe", "central_park"],
        "home_locs": ["north_homes", "south_homes"],
        "evening_locs": ["central_bar", "central_park", "community_center"],
        "backgrounds": [
            "elementary school teacher, {years} years shaping young minds",
            "retired factory worker, {years} years of service, watching from the porch",
            "stay-at-home parent, {years} years raising kids, neighborhood eyes and ears",
            "freelance carpenter, {years} years, built half the houses on the block",
            "church organist, {years} years of Sunday services, the town's conscience",
            "youth soccer coach, {years} years volunteering, keeps kids off streets",
            "community center director, {years} years organizing everything",
            "local journalist, {years} years chasing the truth for the town paper",
            "small restaurant owner, {years} years, the town's gathering place",
            "mechanic, {years} years fixing cars, knows everyone's vehicle",
            "librarian, {years} years of quiet service, deeply observant",
            "postal carrier, {years} years, knows every doorstep in town",
        ],
        "temperaments": [
            "warm and community-minded, always organizing something",
            "worried but practical, acts when others freeze",
            "gossip-prone but good-hearted, information flows through them",
            "protective of neighborhood kids, fierce when threatened",
            "quiet pillar of stability, people gravitate toward in crisis",
            "outspoken and opinionated, not afraid to make waves",
            "empathetic listener, absorbs community pain like a sponge",
        ],
        "base_params": {
            "arousal_rise_rate": 0.70, "arousal_decay_rate": 0.88,
            "valence_momentum": 0.46, "impulse_drain_rate": 0.14,
            "impulse_restore_rate": 0.008, "energy_drain_rate": 0.06,
            "energy_regen_rate": 0.012, "vulnerability_weight": 1.0,
        },
    },
]


# ─── Helper functions ─────────────────────────────────────────────────────────

def _perturb_params(base: dict, rng: random.Random, noise: float = 0.08) -> dict:
    """Add random perturbation to personality parameters."""
    result = {}
    bounds = {
        "arousal_rise_rate": (0.50, 0.90),
        "arousal_decay_rate": (0.80, 0.95),
        "valence_momentum": (0.30, 0.60),
        "impulse_drain_rate": (0.08, 0.25),
        "impulse_restore_rate": (0.004, 0.015),
        "energy_drain_rate": (0.04, 0.12),
        "energy_regen_rate": (0.005, 0.020),
        "vulnerability_weight": (0.50, 1.50),
    }
    for key, val in base.items():
        lo, hi = bounds[key]
        perturbed = val + rng.gauss(0, noise * (hi - lo))
        result[key] = max(lo, min(hi, perturbed))
    return result


def _make_schedule(work_loc: str, home_loc: str, evening_loc: str) -> dict[int, str]:
    """Generate a daily schedule: sleep at home, work, evening out."""
    schedule = {}
    for h in range(0, 7):
        schedule[h] = home_loc
    for h in range(7, 8):
        schedule[h] = home_loc  # getting ready
    for h in range(8, 17):
        schedule[h] = work_loc
    schedule[17] = home_loc  # home first
    for h in range(18, 21):
        schedule[h] = evening_loc
    for h in range(21, 24):
        schedule[h] = home_loc
    return schedule


def _seed_relationships(world: World, agent_meta: dict, rng: random.Random):
    """Seed initial relationships between agents."""
    agents_by_work = {}
    agents_by_home = {}
    for aid, meta in agent_meta.items():
        wl = meta["work_loc"]
        hl = meta["home_loc"]
        agents_by_work.setdefault(wl, []).append(aid)
        agents_by_home.setdefault(hl, []).append(aid)

    # Colleagues: same work location → warmth, trust, familiarity
    for loc, aids in agents_by_work.items():
        pairs = [(aids[i], aids[j]) for i in range(len(aids))
                 for j in range(i + 1, len(aids))]
        # Sample subset to keep sparse — ~30% of possible pairs
        sample_size = max(1, len(pairs) // 3)
        for a, b in rng.sample(pairs, min(sample_size, len(pairs))):
            rel = world.relationships.get_or_create(a, b)
            rel.warmth = rng.uniform(0.05, 0.25)
            rel.trust = rng.uniform(0.05, 0.15)
            rel.familiarity = rng.randint(5, 30)

    # Neighbors: same home location → mild warmth
    for loc, aids in agents_by_home.items():
        pairs = [(aids[i], aids[j]) for i in range(len(aids))
                 for j in range(i + 1, len(aids))]
        sample_size = max(1, len(pairs) // 8)
        for a, b in rng.sample(pairs, min(sample_size, len(pairs))):
            rel = world.relationships.get_or_create(a, b)
            rel.warmth = max(rel.warmth, rng.uniform(0.02, 0.12))
            rel.trust = max(rel.trust, rng.uniform(0.02, 0.08))
            rel.familiarity = max(rel.familiarity, rng.randint(2, 10))

    # Cross-district family ties: 40 random close pairs
    all_ids = list(agent_meta.keys())
    for _ in range(40):
        a, b = rng.sample(all_ids, 2)
        rel = world.relationships.get_or_create(a, b)
        rel.warmth = rng.uniform(0.4, 0.8)
        rel.trust = rng.uniform(0.4, 0.7)
        rel.familiarity = rng.randint(20, 50)

    # Some rival pairs with resentment
    for _ in range(15):
        a, b = rng.sample(all_ids, 2)
        rel = world.relationships.get_or_create(a, b)
        world.relationships.set_resentment(a, b, rng.uniform(0.2, 0.5))
        rel.warmth = min(rel.warmth, rng.uniform(-0.3, 0.0))


def _schedule_events(world: World, factory_worker_ids: list[str],
                     agent_meta: dict):
    """Schedule the multi-crisis event chain across 10 days."""

    # Helper: find a specific agent by role substring in background
    def _find_agent(keyword: str) -> str | None:
        for aid, meta in agent_meta.items():
            if keyword.lower() in meta.get("background", "").lower():
                return aid
        return None

    inspector_id = _find_agent("building inspector")
    compliance_id = _find_agent("compliance officer")
    lab_tech_id = _find_agent("contamination data")

    events = [
        # ═══ DAY 2 — INDUSTRIAL CRISIS BEGINS ═══
        # 09:00 Chemical leak at factory
        (2 * 24 + 9, "factory_floor",
         "Emergency alarm sounds — chemical leak detected in Sector B. Workers ordered to evacuate.",
         "I smell chemicals and alarms are going off. I'm terrified for my life.",
         None),
        # 10:00 Warehouse workers see emergency vehicles
        (2 * 24 + 10, "warehouse",
         "Hazmat teams arrive at the factory. Emergency vehicles everywhere.",
         "Something terrible happened at the plant. I'm worried about my coworkers.",
         None),
        # 14:00 News reaches market
        (2 * 24 + 14, "main_market",
         "News of chemical leak reaches the market. Customers start leaving.",
         "People are scared of contamination. My livelihood depends on customers.",
         None),
        # 14:00 Harbor warned
        (2 * 24 + 14, "docks",
         "Harbor patrol warns: chemical runoff may be reaching the waterfront.",
         "The water might be contaminated. Everything I depend on comes from the harbor.",
         None),
        # 18:00 Evening news
        (2 * 24 + 18, "central_bar",
         "Everyone at the bar is talking about the factory leak. Fear and anger mix.",
         "The whole town is buzzing about the leak. I'm angry and scared.",
         None),

        # ═══ DAY 3 — FACTORY SHUTDOWN + POLITICAL RESPONSE ═══
        # 09:00 Factory shutdown — targets all factory workers
        (3 * 24 + 9, "factory_floor",
         "Factory management announces indefinite shutdown. All workers told to go home.",
         "I just lost my job. The factory is closing. How will I feed my family?",
         factory_worker_ids),
        # 10:00 Emergency council session
        (3 * 24 + 10, "city_hall",
         "Emergency council session called. Officials scramble to respond to the crisis.",
         "The city is in crisis and everyone is looking to us for answers.",
         None),
        # 12:00 First contamination patients
        (3 * 24 + 12, "hospital",
         "First contamination patients arrive at ER. Three workers with respiratory symptoms.",
         "I'm treating people poisoned by chemicals. This is worse than we thought.",
         None),
        # 14:00 Students organize
        (3 * 24 + 14, "student_union",
         "Students organize emergency meeting about the chemical leak and worker safety.",
         "This is environmental injustice. We need to take action now.",
         None),
        # 18:00 Workers at bar
        (3 * 24 + 18, "central_bar",
         "Laid-off factory workers gather at the bar. The mood is desperate and angry.",
         "I lost everything today. I need a drink and someone who understands.",
         None),
        # 19:00 Harbor bar
        (3 * 24 + 19, "harbor_bar",
         "Dock workers discuss the contamination. Fishing boats sit idle.",
         "No fishing, no work. The factory's poison is killing our livelihood too.",
         None),

        # ═══ DAY 4 — CORRUPTION SCANDAL + PROTESTS ═══
        # 09:00 Corruption revealed
        (4 * 24 + 9, "city_hall",
         "Leaked documents reveal city officials ignored safety warnings about the factory for years.",
         "They knew. They all knew and did nothing. I'm furious at the betrayal.",
         None),
        # 10:00 Inspector targeted
        (4 * 24 + 10, "gov_offices",
         "The building inspector who approved the factory faces media and investigators.",
         "Everyone is pointing fingers at me. I was just following orders. I feel trapped.",
         [inspector_id] if inspector_id else None),
        # 11:00 Market sales crash
        (4 * 24 + 11, "main_market",
         "Market sales drop 60%. Customers fear contamination in local produce.",
         "Nobody's buying. My family depends on this stall. I'm desperate.",
         None),
        # 18:00 Protest march — moves people to central_park
        (4 * 24 + 18, "central_park",
         "Workers and students march through town to Central Park. Hundreds attend.",
         "We're marching for justice. The energy of the crowd is electric.",
         None),
        # 19:00 Confrontation at city hall
        (4 * 24 + 19, "city_hall",
         "Protesters confront city officials. Shouting and pushing on the steps.",
         "The crowd is angry and pushing forward. Someone could get hurt.",
         None),

        # ═══ DAY 5 — EXPLOSION ═══
        # 06:00 Factory explosion
        (5 * 24 + 6, "factory_floor",
         "EXPLOSION at the factory. A ruptured chemical tank ignites. Fireball visible across town.",
         "Oh my god, the factory exploded. I can hear the blast from here. I'm terrified.",
         None),
        # 07:00 Hospital mass casualty
        (5 * 24 + 7, "hospital",
         "Mass casualty event. Hospital calls in all staff. 12 injured arriving by ambulance.",
         "The ER is chaos. Burns, smoke inhalation, screaming. I've never seen this.",
         None),
        # 08:00 School lockdown
        (5 * 24 + 8, "north_school",
         "School lockdown. Parents see smoke from the factory. Children are frightened.",
         "The children are scared. I can see smoke from the window. Parents are frantic.",
         None),
        # 09:00 Market evacuated
        (5 * 24 + 9, "main_market",
         "Market evacuated as toxic smoke drifts over the district. Chemical smell in the air.",
         "They're evacuating us. The air smells toxic. Everything I built could be gone.",
         None),
        # 10:00 Harbor closed
        (5 * 24 + 10, "docks",
         "Harbor closed indefinitely. Chemical contamination confirmed in the water.",
         "The harbor is closed. No fishing, no work. Our way of life is dying.",
         None),
        # 12:00 Emergency council
        (5 * 24 + 12, "city_hall",
         "Emergency city council session. Mayor declares state of emergency.",
         "The city is falling apart. People look at us like we failed them. Because we did.",
         None),
        # 18:00 Evening processing
        (5 * 24 + 18, "central_bar",
         "The bar is packed. Workers, students, residents all processing the explosion.",
         "Everyone's here tonight. Shock, anger, fear — it's all mixing together.",
         None),

        # ═══ DAY 6 — AFTERMATH ═══
        # 09:00 Critical patients
        (6 * 24 + 9, "hospital",
         "Two factory workers in critical condition. Vigil forms outside the hospital.",
         "People are dying because of corporate greed. I feel helpless and furious.",
         None),
        # 10:00 State investigators
        (6 * 24 + 10, "gov_offices",
         "State investigators arrive. All government workers ordered to preserve documents.",
         "The investigators are here. Everyone is scared. Careers are on the line.",
         None),
        # 10:00 Compliance officer targeted
        (6 * 24 + 10, "gov_offices",
         "Environmental compliance officer questioned about ignored safety reports.",
         "They're asking me about the reports I filed. I warned them. Nobody listened.",
         [compliance_id] if compliance_id else None),
        # 14:00 Student occupation vote
        (6 * 24 + 14, "student_union",
         "Students vote to occupy City Hall until responsible officials resign.",
         "We voted to take action. I'm scared but we can't let them get away with it.",
         None),
        # 17:00 Occupation begins
        (6 * 24 + 17, "city_hall",
         "Students occupy City Hall lobby. Police arrive but hold back.",
         "We're inside City Hall. Police are watching. My heart is pounding.",
         None),

        # ═══ DAY 7 — COMMUNITY RALLY ═══
        # 10:00 (Saturday) — rally at central park
        (7 * 24 + 10, "central_park",
         "Massive community rally. All districts represented. Speeches, songs, and tears.",
         "The whole town is here. We're standing together. I feel hope for the first time.",
         None),
        # 14:00 CEO press conference
        (7 * 24 + 14, "city_hall",
         "CEO of Consolidated Chemical arrives for press conference. Angry crowd gathered.",
         "The CEO is here. I want to scream at him. He destroyed our community.",
         None),
        # 15:00 Settlement offer
        (7 * 24 + 15, "city_hall",
         "CEO offers $2M settlement fund. Crowd splits — some outraged, others cautiously relieved.",
         "Two million? That's an insult. Or maybe it's a start. I don't know what to feel.",
         None),
        # 19:00 Community dinner
        (7 * 24 + 19, "community_center",
         "Community potluck dinner. People share food, stories, and support.",
         "We're feeding each other. The warmth here is real. We'll get through this.",
         None),

        # ═══ DAY 8 — ENVIRONMENTAL REPORT ═══
        # 09:00 Contamination wider than thought
        (8 * 24 + 9, "gov_offices",
         "Environmental report: contamination extends 2 miles from factory. Suburbs affected.",
         "The contamination is in our neighborhood. Our kids played in that soil.",
         None),
        # 09:00 Lab tech sees data first
        (8 * 24 + 9, "hospital",
         "Lab results confirm elevated toxin levels in 23 residents. Public health emergency.",
         "The numbers are worse than I feared. People have been breathing this for years.",
         [lab_tech_id] if lab_tech_id else None),
        # 10:00 School closed
        (8 * 24 + 10, "north_school",
         "School closed indefinitely for contamination testing. Parents scramble for childcare.",
         "The school is closed. Parents are panicking. The children don't understand.",
         None),
        # 12:00 Property values crash
        (8 * 24 + 12, "south_homes",
         "Real estate values plummet. Homes near factory zone lost 40% of value.",
         "My home — my biggest investment — just lost almost half its value.",
         None),
        # 14:00 Health screening
        (8 * 24 + 14, "hospital",
         "Free health screening announced. Lines wrap around the block.",
         "Everyone is scared they've been poisoned. The fear in their eyes is haunting.",
         None),

        # ═══ DAY 9 — ORGANIZING ═══
        # 09:00 Mutual aid launches
        (9 * 24 + 9, "community_center",
         "Mutual aid network launches. Volunteers organize food, legal aid, housing.",
         "We're building something together. People are helping each other.",
         None),
        # 12:00 Class action filed
        (9 * 24 + 12, "courthouse",
         "Class action lawsuit filed against Consolidated Chemical and the city.",
         "We filed the lawsuit. Justice might actually happen. I feel empowered.",
         None),
        # 14:00 Market vendors feed community
        (9 * 24 + 14, "main_market",
         "Market vendors offer free food to affected families. Long lines form.",
         "We're feeding people because nobody else will. This community is ours.",
         None),
        # 18:00 Hope at the bar
        (9 * 24 + 18, "central_bar",
         "For the first time in a week, there's laughter at the bar alongside the grief.",
         "We're still here. Still together. Maybe that's enough for now.",
         None),

        # ═══ DAY 10 — TOWN HALL SHOWDOWN ═══
        # 10:00 Town hall meeting
        (10 * 24 + 10, "city_hall",
         "Town hall meeting. Every seat filled. Overflow crowd outside with speakers.",
         "Everyone is here. Workers, students, parents, vendors. Moment of truth.",
         None),
        # 12:00 Confrontation erupts
        (10 * 24 + 12, "city_hall",
         "Shouting match erupts. Factory workers confront government officials directly.",
         "They lied to us for years. I can barely contain my rage.",
         None),
        # 14:00 Mayor resigns
        (10 * 24 + 14, "city_hall",
         "Mayor announces resignation. Promises independent investigation.",
         "The mayor resigned. Is this justice or just another politician running away?",
         None),
        # 18:00 Town processes
        (10 * 24 + 18, "central_bar",
         "Town processes the day. Some celebrate, others mourn. It's over — or just beginning.",
         "It's over. Or maybe just beginning. I raise my glass but my hand is shaking.",
         None),
    ]

    for tick, loc, desc, emo_text, targets in events:
        world.schedule_event(ScheduledEvent(
            tick=tick,
            location=loc,
            description=desc,
            emotional_text=emo_text,
            target_agent_ids=targets,
        ))


def _set_rally_overrides(world: World, agent_meta: dict, rng: random.Random):
    """Override agent locations for major rally/march events."""
    all_ids = list(agent_meta.keys())

    # Day 4, 18:00 — Protest march: move ~80 factory workers + students to central_park
    march_tick = 4 * 24 + 18
    march_participants = []
    for aid, meta in agent_meta.items():
        if meta["role"] in ("factory_worker", "student"):
            march_participants.append(aid)
        elif rng.random() < 0.15:  # 15% of others join
            march_participants.append(aid)
    for aid in march_participants:
        world.agents[aid].location_overrides[march_tick] = "central_park"
        world.agents[aid].location_overrides[march_tick + 1] = "central_park"

    # Day 7, 10:00 — Community rally: move ~150 agents to central_park
    rally_tick = 7 * 24 + 10
    rally_participants = rng.sample(all_ids, min(150, len(all_ids)))
    for aid in rally_participants:
        for t in range(rally_tick, rally_tick + 3):  # 3 hours
            world.agents[aid].location_overrides[t] = "central_park"

    # Day 10, 10:00 — Town hall: move ~120 to city_hall
    townhall_tick = 10 * 24 + 10
    townhall_participants = rng.sample(all_ids, min(120, len(all_ids)))
    for aid in townhall_participants:
        for t in range(townhall_tick, townhall_tick + 5):  # 5 hours
            world.agents[aid].location_overrides[t] = "city_hall"


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_large_town(n_agents: int = 300, seed: int = 42) -> tuple[World, dict]:
    """Build a 300-agent town across 8 districts with crisis event chains.

    Returns (World, agent_meta) where agent_meta maps agent_id to role info.
    """
    rng = random.Random(seed)
    world = World()

    # Add all locations
    for loc in ALL_LOCATIONS:
        world.add_location(loc)

    # Generate agents from role archetypes
    names = list(NAME_POOL)
    rng.shuffle(names)
    if len(names) < n_agents:
        raise ValueError(f"Need {n_agents} names, only have {len(names)}")

    agent_meta: dict[str, dict] = {}
    factory_worker_ids: list[str] = []
    name_idx = 0

    for archetype in ROLE_ARCHETYPES:
        count = archetype["count"]
        # Scale count proportionally if n_agents != 300
        if n_agents != 300:
            count = max(1, round(count * n_agents / 300))

        for i in range(count):
            if name_idx >= len(names):
                break
            name = names[name_idx]
            name_idx += 1
            agent_id = name.lower()

            # Ensure unique IDs
            if agent_id in world.agents:
                agent_id = f"{agent_id}_{name_idx}"

            # Perturb personality
            params = _perturb_params(archetype["base_params"], rng)

            # Pick background and temperament
            years = rng.randint(2, 25)
            fam = rng.randint(2, 5)
            bg_template = rng.choice(archetype["backgrounds"])
            bg = bg_template.format(years=years, fam=fam) if "{" in bg_template else bg_template
            temperament = rng.choice(archetype["temperaments"])

            personality = Personality(
                name=name,
                background=bg,
                temperament=temperament,
                **params,
            )
            assign_human_profile(personality, archetype["role"], rng)

            # Build schedule
            work_loc = rng.choice(archetype["work_locs"])
            home_loc = rng.choice(archetype["home_locs"])
            evening_loc = rng.choice(archetype["evening_locs"])
            schedule = _make_schedule(work_loc, home_loc, evening_loc)

            agent = WorldAgent(
                agent_id=agent_id,
                personality=personality,
                schedule=schedule,
                social_role=archetype["role"],
            )
            world.add_agent(agent)

            meta = {
                "role": archetype["role"],
                "work_loc": work_loc,
                "home_loc": home_loc,
                "evening_loc": evening_loc,
                "background": bg,
            }
            agent_meta[agent_id] = meta

            if archetype["role"] == "factory_worker":
                factory_worker_ids.append(agent_id)

    # Seed relationships
    _seed_relationships(world, agent_meta, rng)

    # Schedule events
    _schedule_events(world, factory_worker_ids, agent_meta)

    # Set rally location overrides
    _set_rally_overrides(world, agent_meta, rng)

    return world, agent_meta

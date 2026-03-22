#!/usr/bin/env python3
"""Alternative 300-agent environment: heatwave blackouts and waterfront buyouts.

The same simulation skeleton runs in a different town crisis:
  1. Rolling blackouts and utility triage
  2. Waterfront redevelopment and buyout pressure
  3. Tenant, harbor, and campus coalitions hardening over weeks
"""

from __future__ import annotations

import random

from .scenarios_large import NAME_POOL, ROLE_ARCHETYPES, _make_schedule, _perturb_params, _seed_relationships
from .world import Location, ScheduledEvent, World
from .world_agent import Personality, WorldAgent


ALL_LOCATIONS = [
    Location("factory_floor", "GridSwitch Battery Works",
             "Maintaining battery racks, replacing coolant lines, stabilizing damaged cells"),
    Location("warehouse", "North Freight Logistics Yard",
             "Sorting generators, emergency supplies, and delayed shipments"),
    Location("workers_canteen", "Union Dispatch Hall",
             "Trading outage stories, charging phones, and comparing shift rumors"),
    Location("office_tower", "Lighthouse Utility Headquarters",
             "Reviewing outage maps, investor decks, and emergency response plans"),
    Location("trading_floor", "Redevelopment Finance Suite",
             "Negotiating bridge loans, property options, and emergency acquisition models"),
    Location("downtown_cafe", "Current & Steam Cafe",
             "Meeting reporters, freelancers, and exhausted office staff under humming fans"),
    Location("lecture_hall", "Harbor State Lecture Hall",
             "Debating climate policy, redevelopment, and public accountability"),
    Location("library", "Harbor State Library",
             "Researching zoning records, policy memos, and legal filings"),
    Location("student_union", "Harbor State Student Union",
             "Printing flyers, organizing testimony, and feeding volunteers"),
    Location("main_market", "Boardwalk Market",
             "Selling spoiled-down inventory fast and improvising shade for customers"),
    Location("food_court", "Boardwalk Food Arcade",
             "Running generators, handing out ice, and arguing over supply costs"),
    Location("artisan_alley", "Saltline Artisan Row",
             "Packing up handmade goods before another outage or inspection"),
    Location("docks", "Harbor Ferry Terminal",
             "Loading ferries, checking heat-buckled ramps, and fighting schedule cuts"),
    Location("fish_market", "Pier Fish Exchange",
             "Trying to save catch before the ice melts and the power cuts again"),
    Location("harbor_bar", "Low Tide Tavern",
             "Harbor crews and renters trade rumors over warm beer and charger cables"),
    Location("city_hall", "Harbor City Hall",
             "Running emergency hearings, redevelopment votes, and damage-control briefings"),
    Location("courthouse", "Harbor County Courthouse",
             "Filing injunctions, tenant claims, and emergency motions"),
    Location("gov_offices", "Planning and Utilities Annex",
             "Processing permits, outage appeals, and rezoning packets"),
    Location("hospital", "Mercy Harbor Medical Center",
             "Treating heatstroke, dehydration, and respiratory flareups"),
    Location("north_school", "Northside K-8",
             "Managing early dismissals, cooling shelters, and frightened parents"),
    Location("north_homes", "North Harbor Apartments",
             "Cooking around outages, rationing medicine, and checking on elders"),
    Location("community_center", "South Harbor Mutual Aid Hall",
             "Coordinating water drops, rent support, and legal clinics"),
    Location("south_homes", "South Harbor Homes",
             "Tracking rent notices, spoiled food, and generator noise all night"),
    Location("central_park", "Harbor Commons",
             "Holding vigils, rallies, and open-air meetings when buildings are too hot"),
    Location("central_bar", "Breaker Street Taproom",
             "Cooling off, charging devices, and arguing about who sold the town out"),
]


ROLE_TEXT_OVERRIDES = {
    "factory_worker": {
        "backgrounds": [
            "battery line technician at GridSwitch, {years} years, still covering a cousin's rent",
            "coolant systems mechanic at the battery works, {years} years, paying off storm repairs",
            "warehouse loader moving emergency generators, {years} years, keeping three relatives afloat",
            "shift electrician at GridSwitch, {years} years, one missed paycheck from trouble",
            "dispatch mechanic in the union hall, {years} years, supporting an aging parent and a nephew",
        ],
        "temperaments": [
            "union-minded and practical, keeps score when management lies",
            "quiet until pushed, then relentless about fairness",
            "protective of crew, suspicious of polished explanations",
            "hardworking and exhausted, carries money stress in the body",
        ],
    },
    "office_professional": {
        "backgrounds": [
            "utility analyst, {years} years, reviewing sacrifice-zone outage models",
            "redevelopment associate, {years} years, told this project could make a career",
            "communications manager at the utility, {years} years, living inside talking points",
            "property finance analyst, {years} years, helping price distressed parcels",
            "operations planner, {years} years, juggling emergency briefings and investor calls",
        ],
        "temperaments": [
            "polished and ambitious, constantly reading where power is moving",
            "efficient and guarded, knows too much and says too little",
            "privately conflicted, publicly composed",
            "networked and strategic, sees chaos as both danger and opening",
        ],
    },
    "student": {
        "backgrounds": [
            "urban policy student, organizing testimony on redlined outage maps",
            "marine science major, furious about harbor redevelopment and heat discharge",
            "journalism student, {years} years freelancing and chasing leaks",
            "public health student, {years} years of study, working cooling-center shifts",
            "labor studies student, helping ferry workers and tenants share a playbook",
        ],
        "temperaments": [
            "idealistic and tactical, always turning anger into plans",
            "restless and outspoken, hates closed-door deals",
            "observant and political, stores names and contradictions",
            "social and persuasive, good at keeping groups together under stress",
        ],
    },
    "market_vendor": {
        "backgrounds": [
            "boardwalk grocer, {years} years, losing inventory to every outage",
            "food stall owner, {years} years, feeding neighbors off a failing generator",
            "artisan seller, {years} years, one rent hike away from losing the stall",
            "ice and bait vendor, {years} years, caught between the docks and the tourists",
            "bakery owner, {years} years, waking at 3am to beat the heat and the outages",
        ],
        "temperaments": [
            "community-rooted and outspoken about survival math",
            "generous with food, ruthless about broken promises",
            "warm with customers, hard-edged with officials",
            "resourceful and proud, hates being made to beg",
        ],
    },
    "dock_worker": {
        "backgrounds": [
            "ferry deckhand, {years} years, family has fished the harbor for generations",
            "pier mechanic, {years} years, trying to keep warped ramps serviceable",
            "dock dispatcher, {years} years, hearing the buyout rumors before sunrise",
            "fish pier loader, {years} years, behind on boat payments after the last storm",
        ],
        "temperaments": [
            "territorial and loyal, takes the harbor personally",
            "gruff, watchful, and hard to bluff",
            "protective of family crews, quick to read disrespect",
            "superstitious about the water and furious at land deals made over it",
        ],
    },
    "government_worker": {
        "backgrounds": [
            "planning clerk, {years} years, routing rezoning packets nobody wants to own",
            "city attorney, {years} years, drafting emergency powers language on no sleep",
            "utility regulator, {years} years, buried under outage appeals and quiet pressure",
            "permit inspector, {years} years, signed more waivers than feels defensible",
            "council aide, {years} years, carrying two phones and three contradictory stories",
        ],
        "temperaments": [
            "procedural and tense, knows paper trails can become weapons",
            "cautious, guilt-prone, and always calculating fallout",
            "publicly loyal, privately fraying",
            "politically nimble, hates losing control of the narrative",
        ],
    },
    "healthcare": {
        "backgrounds": [
            "ER nurse, {years} years, triaging heatstroke and inhaler failures back to back",
            "hospital administrator, {years} years, turning hallways into overflow units",
            "paramedic, {years} years, carrying collapsed residents out of stalled elevators",
            "respiratory therapist, {years} years, watching outages turn into admissions",
            "lab tech, {years} years, logging the pattern before the city admits it",
        ],
        "temperaments": [
            "duty-driven and spent, keeps showing up anyway",
            "clinically calm until somebody lies about the numbers",
            "protective, blunt, and past the point of politeness",
            "hypervigilant, compassionate, and close to burning out",
        ],
    },
    "community": {
        "backgrounds": [
            "tenant organizer, {years} years, living in the same block under threat",
            "elementary teacher, {years} years, running cooling check-ins after school",
            "retired ferry worker, {years} years on the water, refuses to leave the neighborhood",
            "church volunteer, {years} years, turning phone trees into survival systems",
            "mechanic, {years} years, charging phones and jump-starting generators for neighbors",
            "local reporter, {years} years, tracking who profited from the emergency",
            "single parent, {years} years, juggling outages, rent, and a kid with asthma",
        ],
        "temperaments": [
            "community-minded and stubborn, remembers every slight",
            "protective of neighbors, good at getting people moving",
            "worried but practical, counts everything that can disappear",
            "good-hearted and political, hates watching quiet people get cornered",
        ],
    },
}


GROUP_PROFILES = {
    "grid_union": {
        "label": "Grid Union",
        "issue": "industrial fallout",
        "home_location": "workers_canteen",
        "rivals": ["redevelopment_board", "city_hall_caucus"],
    },
    "tenant_defense_network": {
        "label": "Tenant Defense Network",
        "issue": "family safety",
        "home_location": "community_center",
        "rivals": ["redevelopment_board"],
    },
    "harbor_families": {
        "label": "Harbor Families",
        "issue": "waterfront survival",
        "home_location": "harbor_bar",
        "rivals": ["redevelopment_board"],
    },
    "campus_action_network": {
        "label": "Campus Action Network",
        "issue": "public organizing",
        "home_location": "student_union",
        "rivals": ["city_hall_caucus", "redevelopment_board"],
    },
    "care_network": {
        "label": "Care Network",
        "issue": "medical overload",
        "home_location": "hospital",
        "rivals": [],
    },
    "small_business_circle": {
        "label": "Small Business Circle",
        "issue": "livelihood strain",
        "home_location": "main_market",
        "rivals": ["redevelopment_board"],
    },
    "city_hall_caucus": {
        "label": "City Hall Caucus",
        "issue": "public accountability",
        "home_location": "city_hall",
        "rivals": ["campus_action_network", "grid_union"],
    },
    "redevelopment_board": {
        "label": "Redevelopment Board",
        "issue": "public accountability",
        "home_location": "office_tower",
        "rivals": ["tenant_defense_network", "harbor_families", "grid_union", "campus_action_network", "small_business_circle"],
    },
    "mutual_aid_ring": {
        "label": "Mutual Aid Ring",
        "issue": "community care",
        "home_location": "community_center",
        "rivals": [],
    },
}


SOCIAL_PROFILES = {
    "factory_worker": {
        "identity_tags": ("provider", "crew loyalist", "rent stressed", "neighborhood rooted", "union shop"),
        "coalitions": (("grid_union", 0.85), ("mutual_aid_ring", 0.35)),
        "rivals": ("redevelopment_board", "city_hall_caucus"),
        "burdens": (
            "covered a cousin's bill with money meant for my own landlord",
            "kept quiet about a safety shortcut on a brutal week",
            "borrowed from a coworker and still have not squared it",
        ),
        "burden_chance": 0.35,
        "debt_range": (0.18, 0.72),
        "secret_range": (0.05, 0.35),
        "ambition_range": (0.05, 0.28),
    },
    "office_professional": {
        "identity_tags": ("message discipline", "career climber", "knows the numbers", "polished operator", "deal facing"),
        "coalitions": (("redevelopment_board", 0.55), ("city_hall_caucus", 0.25)),
        "rivals": ("grid_union", "tenant_defense_network", "campus_action_network"),
        "burdens": (
            "sat on an outage map that protected wealthier blocks first",
            "helped price distressed waterfront parcels before the buyout notices went out",
            "rewrote a briefing to hide how political the outage plan really was",
        ),
        "burden_chance": 0.45,
        "debt_range": (0.05, 0.32),
        "secret_range": (0.18, 0.7),
        "ambition_range": (0.28, 0.85),
    },
    "student": {
        "identity_tags": ("organizer", "document hoarder", "cause driven", "story chaser", "network builder"),
        "coalitions": (("campus_action_network", 0.9), ("mutual_aid_ring", 0.28)),
        "rivals": ("city_hall_caucus", "redevelopment_board"),
        "burdens": (
            "promised to protect a source and already repeated too much",
            "lost grant money and is pretending activism did not cost tuition",
            "is sitting on screenshots that could break a coalition or save it",
        ),
        "burden_chance": 0.32,
        "debt_range": (0.12, 0.5),
        "secret_range": (0.1, 0.58),
        "ambition_range": (0.22, 0.75),
    },
    "market_vendor": {
        "identity_tags": ("cash flow watcher", "community kitchen", "stall owner", "survival math", "block realist"),
        "coalitions": (("small_business_circle", 0.8), ("mutual_aid_ring", 0.3), ("tenant_defense_network", 0.22)),
        "rivals": ("redevelopment_board",),
        "burdens": (
            "is three bad market days from missing the stall lease",
            "has been quietly feeding families from inventory that should have been sold",
            "took a predatory bridge loan to keep the generator running",
        ),
        "burden_chance": 0.42,
        "debt_range": (0.24, 0.78),
        "secret_range": (0.06, 0.32),
        "ambition_range": (0.1, 0.42),
    },
    "dock_worker": {
        "identity_tags": ("harbor loyalist", "family crew", "territorial", "union memory", "waterfront pride"),
        "coalitions": (("harbor_families", 0.88), ("grid_union", 0.18)),
        "rivals": ("redevelopment_board",),
        "burdens": (
            "is behind on a boat payment nobody else in the family knows about",
            "helped move gear off a pier slated for buyout before the notice was public",
            "owes a favor to someone negotiating against the harbor",
        ),
        "burden_chance": 0.38,
        "debt_range": (0.16, 0.68),
        "secret_range": (0.08, 0.38),
        "ambition_range": (0.06, 0.26),
    },
    "government_worker": {
        "identity_tags": ("paper trail keeper", "procedural loyalist", "hearing veteran", "message control", "office survivor"),
        "coalitions": (("city_hall_caucus", 0.72), ("redevelopment_board", 0.18)),
        "rivals": ("campus_action_network", "grid_union"),
        "burdens": (
            "signed the emergency rezoning waiver after being told it was harmless",
            "deleted an embarrassing thread but knows copies still exist",
            "promised a neighborhood hearing that was never going to happen",
        ),
        "burden_chance": 0.5,
        "debt_range": (0.04, 0.24),
        "secret_range": (0.22, 0.82),
        "ambition_range": (0.2, 0.7),
    },
    "healthcare": {
        "identity_tags": ("triage first", "numbers watcher", "duty bound", "burnout edge", "shelter volunteer"),
        "coalitions": (("care_network", 0.94), ("mutual_aid_ring", 0.22)),
        "rivals": (),
        "burdens": (
            "already knows the admissions curve is worse than the city admits",
            "skipped a family emergency because the ward could not spare another body",
            "has been rationing supplies before anyone approved it",
        ),
        "burden_chance": 0.34,
        "debt_range": (0.05, 0.26),
        "secret_range": (0.14, 0.52),
        "ambition_range": (0.02, 0.22),
    },
    "community": {
        "identity_tags": ("neighborhood anchor", "rent stressed", "phone tree runner", "story carrier", "mutual aid"),
        "coalitions": (("tenant_defense_network", 0.45), ("mutual_aid_ring", 0.52), ("small_business_circle", 0.12)),
        "rivals": ("redevelopment_board",),
        "burdens": (
            "is behind on rent and hiding notices from the kids",
            "told one neighbor who the buyer's agent was and now that rumor is everywhere",
            "borrowed grocery money and keeps promising next week",
        ),
        "burden_chance": 0.46,
        "debt_range": (0.2, 0.82),
        "secret_range": (0.08, 0.34),
        "ambition_range": (0.05, 0.3),
    },
}


def _assign_social_profile(agent: WorldAgent, role: str, rng: random.Random):
    profile = SOCIAL_PROFILES[role]
    identity_count = 2 if rng.random() < 0.55 else 1
    identity_tags = tuple(rng.sample(list(profile["identity_tags"]), k=identity_count))
    coalitions = [name for name, chance in profile["coalitions"] if rng.random() < chance]
    burden = ""
    secret_pressure = rng.uniform(0.0, profile["secret_range"][1] * 0.4)
    if rng.random() < profile["burden_chance"]:
        burden = rng.choice(profile["burdens"])
        secret_pressure = rng.uniform(*profile["secret_range"])

    agent.identity_tags = identity_tags
    agent.coalitions = tuple(sorted(set(coalitions)))
    agent.rival_coalitions = tuple(sorted(set(profile["rivals"])))
    agent.private_burden = burden
    agent.debt_pressure = round(rng.uniform(*profile["debt_range"]), 3)
    agent.secret_pressure = round(secret_pressure, 3)
    agent.ambition = round(rng.uniform(*profile["ambition_range"]), 3)


def _seed_heatwave_relationships(world: World, agent_meta: dict[str, dict], rng: random.Random):
    _seed_relationships(world, agent_meta, rng)

    agents_by_coalition: dict[str, list[str]] = {}
    for aid, agent in world.agents.items():
        for coalition in agent.coalitions:
            agents_by_coalition.setdefault(coalition, []).append(aid)

    for coalition, aids in agents_by_coalition.items():
        if len(aids) < 2:
            continue
        pairs = [(aids[i], aids[j]) for i in range(len(aids)) for j in range(i + 1, len(aids))]
        sample_size = max(1, len(pairs) // 4)
        for a, b in rng.sample(pairs, min(sample_size, len(pairs))):
            rel = world.relationships.get_or_create(a, b)
            rel.trust = max(rel.trust, rng.uniform(0.08, 0.22))
            rel.warmth = max(rel.warmth, rng.uniform(0.08, 0.28))
            rel.familiarity = max(rel.familiarity, rng.randint(6, 28))
            rel.alliance_strength = max(rel.alliance_strength, rng.uniform(0.18, 0.45))

    for coalition, profile in GROUP_PROFILES.items():
        member_ids = agents_by_coalition.get(coalition, [])
        if len(member_ids) < 2:
            continue
        for rival in profile["rivals"]:
            rival_ids = agents_by_coalition.get(rival, [])
            if not rival_ids:
                continue
            pair_count = max(3, min(15, len(member_ids), len(rival_ids)))
            for _ in range(pair_count):
                a = rng.choice(member_ids)
                b = rng.choice(rival_ids)
                rel = world.relationships.get_or_create(a, b)
                rel.warmth = min(rel.warmth, rng.uniform(-0.18, 0.04))
                rel.trust = min(rel.trust, rng.uniform(-0.12, 0.05))
                rel.rivalry = max(rel.rivalry, rng.uniform(0.18, 0.45))
                world.relationships.set_resentment(a, b, rng.uniform(0.12, 0.34))
                world.relationships.set_grievance(a, b, rng.uniform(0.08, 0.26))

    stressed = [aid for aid, agent in world.agents.items() if agent.debt_pressure > 0.45]
    stabilizers = [
        aid for aid, agent in world.agents.items()
        if {"mutual_aid_ring", "care_network", "small_business_circle"} & set(agent.coalitions)
    ]
    for debtor in rng.sample(stressed, min(len(stressed), 45)):
        if not stabilizers:
            break
        creditor = rng.choice(stabilizers)
        if creditor == debtor:
            continue
        rel = world.relationships.get_or_create(debtor, creditor)
        world.relationships.adjust_debt(debtor, creditor, rng.uniform(0.18, 0.5))
        rel.trust = max(rel.trust, rng.uniform(0.02, 0.18))
        rel.warmth = max(rel.warmth, rng.uniform(0.0, 0.16))
        rel.practical_help_events += rng.randint(1, 3)
        rel.last_issue = rng.choice(["livelihood strain", "community care", "family safety"])


def _schedule_heatwave_events(world: World, agent_meta: dict[str, dict], rng: random.Random):
    secret_holders = [aid for aid, agent in world.agents.items() if agent.private_burden]
    city_hall_secret = next((aid for aid in secret_holders if agent_meta[aid]["role"] == "government_worker"), None)
    office_secret = next((aid for aid in secret_holders if agent_meta[aid]["role"] == "office_professional"), None)

    events = [
        (2 * 24 + 8, "north_homes",
         "Rolling blackout slams older apartment blocks before breakfast. Elevators stall, fridges warm, and tempers spike in the hallways.",
         "The power is out again and the heat is already unbearable. Home does not feel safe or stable anymore.",
         None),
        (2 * 24 + 11, "office_tower",
         "Internal outage memo circulates upstairs: wealthier districts will be protected first if the grid buckles again.",
         "The plan is uglier than anyone will say out loud. People will call it triage, but it feels like sacrifice by zip code.",
         [office_secret] if office_secret else None),
        (2 * 24 + 18, "central_bar",
         "At Breaker Street, people compare outage maps, dead refrigerators, and who always seems to stay lit.",
         "Everyone has a different version of the plan, but every version says someone like us is expected to take the hit.",
         None),
        (3 * 24 + 9, "factory_floor",
         "Transformer fire halts GridSwitch production. Contractors are sent home while managers insist the shutdown is temporary.",
         "Work just stopped under the worst possible conditions. Temporary is the word people use when they do not want to tell the truth.",
         None),
        (3 * 24 + 12, "hospital",
         "Mercy Harbor ER fills with heatstroke, asthma flareups, and residents whose medical devices failed during the outages.",
         "The heat and the power cuts are turning ordinary vulnerability into crisis after crisis.",
         None),
        (3 * 24 + 17, "community_center",
         "Neighbors crowd the mutual aid hall to ask for ice, inhaler rides, extension cords, and rent advice.",
         "People are not asking for luxuries. They are asking for the things that keep a household from tipping over.",
         None),
        (4 * 24 + 9, "gov_offices",
         "Rezoning packets for three waterfront blocks are filed under emergency powers before most residents hear the word buyout.",
         "They are trying to move whole blocks with paperwork and heat before people can organize against it.",
         [city_hall_secret] if city_hall_secret else None),
        (4 * 24 + 12, "docks",
         "Harbor ferry schedule is cut again. Commuters and dock crews hear that the reductions may become permanent.",
         "They are treating the harbor like a line item instead of a living part of the town.",
         None),
        (4 * 24 + 18, "student_union",
         "Students host a packed teach-in on sacrifice-zone blackouts and quiet waterfront buyouts.",
         "The same people keeping the lights on for investors are letting whole neighborhoods simmer in the dark.",
         None),
        (5 * 24 + 7, "north_school",
         "Northside K-8 closes after air systems fail. Families scramble for childcare and a cool room before noon.",
         "The school was supposed to be the safe fallback. Now even that is failing under the heat.",
         None),
        (5 * 24 + 10, "fish_market",
         "Spoiled ice and failing coolers ruin part of the catch. Vendors accuse the city of writing off the waterfront.",
         "One more outage and weeks of work disappear into the heat. Nobody who made this plan has to smell the loss.",
         None),
        (5 * 24 + 18, "central_park",
         "A candlelight vigil for heat deaths fills Harbor Commons. Names spread through the crowd faster than official statements.",
         "This is not a bad week anymore. It is a list of people the city failed in sequence.",
         None),
        (6 * 24 + 9, "city_hall",
         "A leaked color-coded map shows outage priority tracked property values almost perfectly.",
         "They turned neighborhoods into tiers and hoped nobody would see the pattern.",
         None),
        (6 * 24 + 14, "courthouse",
         "Residents file for an emergency injunction against waterfront buyouts and forced relocations.",
         "Going to court feels like the last thin wall between us and people who think panic is a buying opportunity.",
         None),
        (6 * 24 + 19, "harbor_bar",
         "Harbor families argue over whether to blockade equipment trucks or keep the fight inside the courts.",
         "Every path risks something we cannot get back: work, housing, or the trust that we are still on the same side.",
         None),
        (7 * 24 + 10, "central_park",
         "Tenant groups, harbor crews, and campus organizers rally together under sun tents and borrowed megaphones.",
         "This is the first time the town's separate hurts feel like one fight instead of parallel losses.",
         None),
        (7 * 24 + 15, "city_hall",
         "Utility executives blame sabotage and heat instead of admitting the map was political.",
         "The official story keeps shifting, which only proves they are still trying to manage the blame instead of the harm.",
         None),
        (7 * 24 + 20, "community_center",
         "The mutual aid kitchen runs short on water, propane, and patience.",
         "Even solidarity has a supply line. We are starting to feel where it bends.",
         None),
        (8 * 24 + 9, "office_tower",
         "Investors are briefed on 'recovery opportunities' in blocks still waiting for steady power.",
         "Somebody is already pricing the town we are losing while we are still living in it.",
         None),
        (8 * 24 + 13, "main_market",
         "Boardwalk merchants refuse new outage surcharges and begin talking openly about a payment strike.",
         "If every emergency cost gets pushed downward, eventually saying yes becomes the same as disappearing.",
         None),
        (8 * 24 + 18, "city_hall",
         "Private security contractors appear around parcels rumored to be in the first redevelopment wave.",
         "Once security shows up before the explanation, everyone understands whose fears counted first.",
         None),
        (9 * 24 + 10, "docks",
         "An unofficial ferry slowdown begins. Harbor crews insist it is safety; city hall calls it economic blackmail.",
         "People who keep the place moving are finally using that fact as leverage.",
         None),
        (9 * 24 + 14, "gov_offices",
         "A planning staffer is questioned over deleted texts tied to rezoning and outage sequencing.",
         "The paper trail is no longer staying on paper.",
         [city_hall_secret] if city_hall_secret else None),
        (9 * 24 + 19, "central_bar",
         "Rumor races through the taproom that buyout offers were timed to the blackout map on purpose.",
         "The town is done treating coincidence as innocence.",
         None),
        (10 * 24 + 10, "city_hall",
         "Council opens a tense vote on emergency redevelopment powers and neighborhood relocation authority.",
         "They want to call this recovery while people are still counting what the emergency took from them.",
         None),
        (10 * 24 + 13, "city_hall",
         "A whistleblower recording interrupts the vote and throws the chamber into shouting.",
         "One person's private recording just changed what the whole town thinks is provable.",
         None),
        (10 * 24 + 18, "central_park",
         "The town regroups in the heat, not sure whether the vote collapse was a victory, a delay, or the start of something rougher.",
         "Nothing is settled, but nobody believes they can go back to being only neighbors now.",
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


def _set_heatwave_overrides(world: World, agent_meta: dict[str, dict], rng: random.Random):
    all_ids = list(agent_meta.keys())

    rally_tick = 7 * 24 + 10
    rally_participants = []
    for aid, agent in world.agents.items():
        if {"tenant_defense_network", "harbor_families", "campus_action_network", "mutual_aid_ring"} & set(agent.coalitions):
            rally_participants.append(aid)
        elif rng.random() < 0.12:
            rally_participants.append(aid)
    for aid in rally_participants:
        for t in range(rally_tick, rally_tick + 3):
            world.agents[aid].location_overrides[t] = "central_park"

    vote_tick = 10 * 24 + 10
    vote_participants = rng.sample(all_ids, min(130, len(all_ids)))
    for aid in vote_participants:
        for t in range(vote_tick, vote_tick + 5):
            world.agents[aid].location_overrides[t] = "city_hall"


def build_heatwave_harbor(n_agents: int = 300, seed: int = 84) -> tuple[World, dict]:
    rng = random.Random(seed)
    world = World()
    world.scenario_name = "heatwave_harbor"
    world.group_profiles = GROUP_PROFILES

    for loc in ALL_LOCATIONS:
        world.add_location(loc)

    names = list(NAME_POOL)
    rng.shuffle(names)
    if len(names) < n_agents:
        raise ValueError(f"Need {n_agents} names, only have {len(names)}")

    base_roles = {item["role"]: item for item in ROLE_ARCHETYPES}
    agent_meta: dict[str, dict] = {}
    name_idx = 0

    for role, base_archetype in base_roles.items():
        text_override = ROLE_TEXT_OVERRIDES[role]
        count = base_archetype["count"]
        if n_agents != 300:
            count = max(1, round(count * n_agents / 300))

        for _ in range(count):
            if name_idx >= len(names):
                break
            name = names[name_idx]
            name_idx += 1
            agent_id = name.lower()
            if agent_id in world.agents:
                agent_id = f"{agent_id}_{name_idx}"

            params = _perturb_params(base_archetype["base_params"], rng)
            years = rng.randint(2, 25)
            fam = rng.randint(2, 5)
            bg_template = rng.choice(text_override["backgrounds"])
            background = bg_template.format(years=years, fam=fam) if "{" in bg_template else bg_template
            temperament = rng.choice(text_override["temperaments"])

            personality = Personality(
                name=name,
                background=background,
                temperament=temperament,
                **params,
            )

            work_loc = rng.choice(base_archetype["work_locs"])
            home_loc = rng.choice(base_archetype["home_locs"])
            evening_loc = rng.choice(base_archetype["evening_locs"])
            schedule = _make_schedule(work_loc, home_loc, evening_loc)

            agent = WorldAgent(
                agent_id=agent_id,
                personality=personality,
                schedule=schedule,
            )
            _assign_social_profile(agent, role, rng)
            world.add_agent(agent)

            agent_meta[agent_id] = {
                "role": role,
                "work_loc": work_loc,
                "home_loc": home_loc,
                "evening_loc": evening_loc,
                "background": background,
                "coalitions": list(agent.coalitions),
                "private_burden": agent.private_burden,
            }

    _seed_heatwave_relationships(world, agent_meta, rng)
    _schedule_heatwave_events(world, agent_meta, rng)
    _set_heatwave_overrides(world, agent_meta, rng)
    return world, agent_meta

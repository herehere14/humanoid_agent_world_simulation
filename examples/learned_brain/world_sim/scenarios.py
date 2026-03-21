"""Demo scenario: Small Town, Bad News.

30 agents at a small company — office, homes, bar, park, church.
Day 1-3: Normal work life. Day 4: Layoff rumors. Day 5: Actual layoffs.
Day 6-15: Ripple effects — anger, grief, drinking, support networks.
"""

from __future__ import annotations

from .world_agent import WorldAgent, HeartState, Personality
from .world import World, Location, ScheduledEvent


def _make_schedule(role: str) -> dict[int, str]:
    """Generate hourly schedule based on role."""
    schedule = {}
    for h in range(0, 6):
        schedule[h] = "home"
    for h in range(6, 8):
        schedule[h] = "home"  # morning routine

    if role == "office_worker":
        for h in range(8, 17):
            schedule[h] = "office"
        for h in range(17, 19):
            schedule[h] = "home"
        schedule[19] = "bar"  # some evenings
        for h in range(20, 22):
            schedule[h] = "home"
    elif role == "manager":
        for h in range(7, 18):
            schedule[h] = "office"
        for h in range(18, 22):
            schedule[h] = "home"
    elif role == "bartender":
        for h in range(8, 14):
            schedule[h] = "home"
        for h in range(14, 23):
            schedule[h] = "bar"
    elif role == "teacher":
        for h in range(7, 15):
            schedule[h] = "school"
        for h in range(15, 18):
            schedule[h] = "home"
        schedule[18] = "park"
        for h in range(19, 22):
            schedule[h] = "home"
    elif role == "retiree":
        for h in range(8, 11):
            schedule[h] = "park"
        for h in range(11, 14):
            schedule[h] = "home"
        schedule[14] = "church"
        for h in range(15, 22):
            schedule[h] = "home"
    else:
        for h in range(8, 17):
            schedule[h] = "office"
        for h in range(17, 22):
            schedule[h] = "home"

    for h in range(22, 24):
        schedule[h] = "home"

    return schedule


# ---------------------------------------------------------------------------
# Agent definitions — 30 characters with distinct personalities
# ---------------------------------------------------------------------------

AGENTS = [
    # --- Office workers (will face layoffs) ---
    Personality(
        name="Marcus", background="34, logistics coordinator. Wife and toddler Lily. Built savings through discipline.",
        temperament="Anxious about money, proud provider, slow-burn anger",
        arousal_rise_rate=0.75, impulse_drain_rate=0.18, energy_drain_rate=0.08,
        vulnerability_weight=1.2,
    ),
    Personality(
        name="Sarah", background="28, marketing analyst. Single, ambitious. First real job after grad school.",
        temperament="Driven, competitive, masks insecurity with confidence",
        arousal_rise_rate=0.65, valence_momentum=0.35, impulse_drain_rate=0.12,
    ),
    Personality(
        name="Tom", background="52, senior accountant. 25 years at the company. Two kids in college.",
        temperament="Steady, loyal, internalizes stress, fears change",
        arousal_rise_rate=0.5, arousal_decay_rate=0.92, energy_drain_rate=0.05,
        impulse_restore_rate=0.012, vulnerability_weight=0.8,
    ),
    Personality(
        name="Priya", background="31, software developer. Engaged, planning wedding. Immigrant, first-gen professional.",
        temperament="Optimistic but carries family pressure, people-pleaser",
        arousal_rise_rate=0.6, valence_momentum=0.5, impulse_drain_rate=0.1,
        energy_regen_rate=0.015,
    ),
    Personality(
        name="Jake", background="26, junior sales rep. Lives with roommates. Party guy, doesn't take work seriously.",
        temperament="Carefree, avoidant, uses humor as defense",
        arousal_rise_rate=0.55, arousal_decay_rate=0.82, impulse_drain_rate=0.08,
        vulnerability_weight=0.6,
    ),
    Personality(
        name="Diana", background="45, HR director. Single mom, two teens. Known as the 'company heart'.",
        temperament="Empathetic, strong, carries everyone's burdens",
        arousal_rise_rate=0.6, energy_drain_rate=0.09, vulnerability_weight=1.3,
        impulse_restore_rate=0.01,
    ),
    Personality(
        name="Chen", background="38, operations manager. Methodical, efficient. Wife is a doctor.",
        temperament="Calm under pressure, analytical, emotionally guarded",
        arousal_rise_rate=0.45, arousal_decay_rate=0.9, impulse_drain_rate=0.08,
        vulnerability_weight=0.5,
    ),
    Personality(
        name="Rosa", background="29, office administrator. Grew up poor, this job is her lifeline.",
        temperament="Warm, anxious, fierce when cornered",
        arousal_rise_rate=0.8, impulse_drain_rate=0.2, energy_drain_rate=0.1,
        vulnerability_weight=1.4,
    ),
    Personality(
        name="Kevin", background="41, IT support. Divorced, sees kids on weekends. Quiet, reliable.",
        temperament="Stoic, lonely, buries feelings in work",
        arousal_rise_rate=0.5, valence_momentum=0.55, energy_drain_rate=0.06,
        impulse_restore_rate=0.006,
    ),
    Personality(
        name="Lena", background="33, project manager. Married, no kids by choice. Organized, ambitious.",
        temperament="Assertive, strategic, doesn't suffer fools",
        arousal_rise_rate=0.65, impulse_drain_rate=0.1, vulnerability_weight=0.7,
    ),
    # More office workers
    Personality(
        name="Andre", background="36, sales lead. Charismatic, recently promoted.",
        temperament="Confident, protective of team, competitive",
        arousal_rise_rate=0.7, impulse_drain_rate=0.12,
    ),
    Personality(
        name="Mika", background="24, intern turned junior analyst. Eager, nervous.",
        temperament="Eager to please, anxious about job security",
        arousal_rise_rate=0.8, vulnerability_weight=1.5, impulse_drain_rate=0.2,
    ),
    Personality(
        name="Greg", background="48, facilities manager. Union guy, speaks his mind.",
        temperament="Blunt, loyal to coworkers, distrusts management",
        arousal_rise_rate=0.75, impulse_drain_rate=0.22, impulse_restore_rate=0.005,
    ),
    Personality(
        name="Nadia", background="30, graphic designer. Creative, introverted.",
        temperament="Sensitive, artistic, withdraws under stress",
        arousal_rise_rate=0.55, energy_drain_rate=0.1, vulnerability_weight=1.3,
    ),
    Personality(
        name="Carlos", background="39, warehouse supervisor. Three kids, coaches little league.",
        temperament="Steady, family-first, slow to anger but explosive",
        arousal_rise_rate=0.5, arousal_decay_rate=0.93, impulse_drain_rate=0.25,
    ),

    # --- Management (delivers bad news) ---
    Personality(
        name="Richard", background="55, CEO. Founded the company 20 years ago. Feels responsible.",
        temperament="Authoritative, guilt-ridden about layoffs, drinks too much",
        arousal_rise_rate=0.6, energy_drain_rate=0.08, vulnerability_weight=1.1,
        impulse_restore_rate=0.004,
    ),
    Personality(
        name="Victoria", background="42, CFO. MBA, sharp, pragmatic. Recommended the cuts.",
        temperament="Cold analytical exterior, private empathy",
        arousal_rise_rate=0.4, arousal_decay_rate=0.85, impulse_drain_rate=0.06,
        vulnerability_weight=0.4,
    ),

    # --- Community (not directly affected but feel ripples) ---
    Personality(
        name="Frank", background="62, bartender at The Tap. Knows everyone, hears everything.",
        temperament="Wise, patient, carries others' stories",
        arousal_rise_rate=0.4, energy_drain_rate=0.04, vulnerability_weight=0.6,
        impulse_restore_rate=0.015,
    ),
    Personality(
        name="Maria", background="58, retired teacher. Volunteers at church. Town gossip.",
        temperament="Kind but nosy, worries about everyone",
        arousal_rise_rate=0.65, vulnerability_weight=1.0,
    ),
    Personality(
        name="Pastor James", background="50, church pastor. Counselor for the community.",
        temperament="Compassionate, patient, carries heavy emotional load",
        arousal_rise_rate=0.45, energy_drain_rate=0.07, vulnerability_weight=0.9,
        impulse_restore_rate=0.012,
    ),

    # --- Families (feel the impact through their partners) ---
    Personality(
        name="Lisa", background="33, Marcus's wife. Stay-at-home mom. Controls the budget.",
        temperament="Practical, worried, fierce protector of family",
        arousal_rise_rate=0.75, impulse_drain_rate=0.18, vulnerability_weight=1.2,
    ),
    Personality(
        name="David", background="30, Rosa's boyfriend. Works construction. Protective.",
        temperament="Gentle giant, gets angry when Rosa is threatened",
        arousal_rise_rate=0.7, impulse_drain_rate=0.2, vulnerability_weight=0.8,
    ),
    Personality(
        name="Sophie", background="16, Diana's daughter. Perceptive, worried about mom.",
        temperament="Mature for her age, anxious, protective of her mom",
        arousal_rise_rate=0.7, vulnerability_weight=1.4,
    ),
    # Additional community members
    Personality(
        name="Ray", background="44, Tom's best friend. Works at the hardware store.",
        temperament="Supportive, practical, good listener",
        arousal_rise_rate=0.5, vulnerability_weight=0.7,
    ),
    Personality(
        name="Jenny", background="27, Jake's roommate. Barista, free spirit.",
        temperament="Easygoing, supportive, drama-averse",
        arousal_rise_rate=0.5, vulnerability_weight=0.6,
    ),
    Personality(
        name="Omar", background="35, Priya's fiance. Accountant at another firm.",
        temperament="Calm, supportive, practical problem-solver",
        arousal_rise_rate=0.45, vulnerability_weight=0.5,
    ),
    Personality(
        name="Elena", background="40, school teacher. Colleagues with Diana's kids.",
        temperament="Warm, organized, community connector",
        arousal_rise_rate=0.55, vulnerability_weight=0.8,
    ),
    Personality(
        name="Hank", background="60, retired factory worker. Regular at The Tap.",
        temperament="Gruff, experienced, seen layoffs before",
        arousal_rise_rate=0.4, vulnerability_weight=0.5,
        impulse_restore_rate=0.015,
    ),
    Personality(
        name="Yuki", background="32, Chen's wife. ER doctor. Practical, stretched thin.",
        temperament="Compassionate but exhausted, matter-of-fact",
        arousal_rise_rate=0.5, energy_drain_rate=0.06, vulnerability_weight=0.7,
    ),
    Personality(
        name="Mike", background="22, Carlos's eldest son. College sophomore, home for summer.",
        temperament="Idealistic, worried about family finances",
        arousal_rise_rate=0.7, vulnerability_weight=1.2,
    ),
]

# Roles determine schedules and who gets laid off
AGENT_ROLES = {
    "Marcus": "office_worker", "Sarah": "office_worker", "Tom": "office_worker",
    "Priya": "office_worker", "Jake": "office_worker", "Diana": "office_worker",
    "Chen": "office_worker", "Rosa": "office_worker", "Kevin": "office_worker",
    "Lena": "office_worker", "Andre": "office_worker", "Mika": "office_worker",
    "Greg": "office_worker", "Nadia": "office_worker", "Carlos": "office_worker",
    "Richard": "manager", "Victoria": "manager",
    "Frank": "bartender",
    "Maria": "retiree", "Pastor James": "retiree",
    "Elena": "teacher", "Sophie": "teacher",
    "Lisa": "retiree", "David": "office_worker",  # construction → similar schedule
    "Ray": "office_worker",  # hardware store
    "Jenny": "bartender",  # barista
    "Omar": "office_worker",  # accountant at another firm
    "Yuki": "office_worker",  # doctor, similar hours
    "Hank": "retiree",
    "Mike": "office_worker",  # summer job
}

# Who gets laid off on Day 5
LAYOFF_TARGETS = ["Marcus", "Rosa", "Jake", "Mika", "Greg", "Nadia", "Carlos"]


def build_small_town() -> World:
    """Build the 'Small Town, Bad News' scenario."""
    world = World()

    # Locations
    for loc in [
        Location("office", "Meridian Corp Office", "Working at desk, meetings, routine office work"),
        Location("home", "Home", "At home, family time, relaxing"),
        Location("bar", "The Tap Bar & Grill", "Drinking, socializing, unwinding after work"),
        Location("park", "Riverside Park", "Walking, sitting on benches, children playing"),
        Location("church", "Community Church", "Quiet reflection, community gatherings"),
        Location("school", "Lincoln High School", "Teaching, studying, school activities"),
    ]:
        world.add_location(loc)

    # Agents
    for personality in AGENTS:
        role = AGENT_ROLES.get(personality.name, "office_worker")
        agent = WorldAgent(
            agent_id=personality.name.lower().replace(" ", "_"),
            personality=personality,
            schedule=_make_schedule(role),
        )
        world.add_agent(agent)

    # Pre-set some relationships (family, close friends)
    _setup_relationships(world)

    # Schedule events
    _schedule_events(world)

    return world


def _setup_relationships(world: World):
    """Set up initial relationships — family bonds, friendships, work ties."""
    rs = world.relationships

    # Family bonds (high trust + warmth)
    family_pairs = [
        ("marcus", "lisa"), ("rosa", "david"), ("diana", "sophie"),
        ("carlos", "mike"), ("chen", "yuki"), ("priya", "omar"),
        ("jake", "jenny"), ("tom", "ray"),
    ]
    for a, b in family_pairs:
        rel = rs.get_or_create(a, b)
        rel.trust = 0.8
        rel.warmth = 0.7
        rel.familiarity = 100

    # Work friendships
    work_friends = [
        ("marcus", "tom"), ("sarah", "lena"), ("priya", "nadia"),
        ("jake", "andre"), ("rosa", "diana"), ("kevin", "greg"),
        ("chen", "victoria"), ("mika", "sarah"),
    ]
    for a, b in work_friends:
        rel = rs.get_or_create(a, b)
        rel.trust = 0.3
        rel.warmth = 0.4
        rel.familiarity = 50

    # Community ties
    community_ties = [
        ("frank", "hank"), ("maria", "pastor_james"), ("frank", "greg"),
        ("elena", "sophie"), ("maria", "diana"),
    ]
    for a, b in community_ties:
        rel = rs.get_or_create(a, b)
        rel.trust = 0.4
        rel.warmth = 0.5
        rel.familiarity = 80

    # Management-employee (lower trust, formal)
    for emp in ["marcus", "sarah", "tom", "diana", "chen", "rosa"]:
        for mgr in ["richard", "victoria"]:
            rel = rs.get_or_create(emp, mgr)
            rel.trust = 0.1
            rel.warmth = 0.0
            rel.familiarity = 30


def _schedule_events(world: World):
    """Schedule the drama — rumors, layoffs, aftermath."""

    # Event tick helper: day 1 starts at tick 0
    def dt(day: int, hour: int) -> int:
        return (day - 1) * 24 + hour

    # Day 1-3: Normal life (no events — just routine)

    # Day 4, 10am: Layoff RUMORS start at the office
    world.schedule_event(ScheduledEvent(
        tick=dt(4, 10),
        location="office",
        description="Whispers about budget cuts and potential layoffs are circulating through the office.",
        emotional_text="I'm about to lose my job and I can't afford it. I'm panicking, my heart is racing.",
    ))

    # Day 4, 3pm: Rumor intensifies
    world.schedule_event(ScheduledEvent(
        tick=dt(4, 15),
        location="office",
        description="Someone saw Victoria carrying a folder labeled 'restructuring plan'. The office is buzzing with fear.",
        emotional_text="I'm terrified, something terrible is happening. I might get fired and we can't pay our bills.",
    ))

    # Day 4, 7pm: Rumors reach the bar
    world.schedule_event(ScheduledEvent(
        tick=dt(4, 19),
        location="bar",
        description="Word about Meridian Corp layoffs has reached the bar. Regulars are discussing who might be affected.",
        emotional_text="I'm worried about my friends at work. There's a confrontation coming and I'm bracing for it.",
    ))

    # Day 5, 9am: Official announcement
    world.schedule_event(ScheduledEvent(
        tick=dt(5, 9),
        location="office",
        description="Richard calls an all-hands meeting. 'We're eliminating seven positions effective immediately.'",
        emotional_text="I'm panicking, my heart is racing, I can't breathe. I'm about to lose my job and I can't afford it. This is outrageous.",
    ))

    # Day 5, 10am-12pm: Individual notifications (targeted events)
    for i, name in enumerate(LAYOFF_TARGETS):
        aid = name.lower()
        world.schedule_event(ScheduledEvent(
            tick=dt(5, 10) + i // 3,  # staggered
            location="office",
            description=f"{name} is called into the conference room. 'Your position has been eliminated.'",
            emotional_text="I just got fired and I don't know what to do. I feel worthless and hopeless. I lost everything and it's all my fault. How could they do this to me after everything I gave them.",
            target_agent_ids=[aid],
        ))

    # Day 5, 5pm: Laid-off workers gather at the bar
    world.schedule_event(ScheduledEvent(
        tick=dt(5, 17),
        location="bar",
        description="The laid-off workers are gathering at The Tap. Shock, anger, and dark humor.",
        emotional_text="I'm angry and bitter about how I've been treated. I'm so angry I could scream. How dare they do this to us.",
    ))

    # Day 6, 8am: The morning after — homes
    world.schedule_event(ScheduledEvent(
        tick=dt(6, 8),
        location="home",
        description="The morning after the layoffs. Staring at the ceiling. The weight of unemployment.",
        emotional_text="I barely slept because of guilt and anxiety. I feel worthless and hopeless. I don't care anymore, nothing matters. Everything feels flat and empty.",
        target_agent_ids=[name.lower() for name in LAYOFF_TARGETS],
    ))

    # Day 7: Community rallies
    world.schedule_event(ScheduledEvent(
        tick=dt(7, 14),
        location="church",
        description="Pastor James organizes a community support gathering. Tears, hugs, and offers of help.",
        emotional_text="I'm grateful and at peace. I love spending time with the people I care about. It feels like we're healing and finding our way back.",
    ))

    # Day 8: Greg confronts management at the bar
    world.schedule_event(ScheduledEvent(
        tick=dt(8, 19),
        location="bar",
        description="Greg has been drinking since 5pm and is loudly criticizing Richard and Victoria. 'Twenty years I gave them!'",
        emotional_text="I'm furious and I can't hold back anymore. This is outrageous, how dare they do this to me. I'm so angry I could scream. I'm angry and bitter about how I've been treated.",
    ))

    # Day 10: Small victory — Rosa gets a job lead
    world.schedule_event(ScheduledEvent(
        tick=dt(10, 14),
        location="home",
        description="Rosa gets a call from a contact — there might be an opening at a company across town.",
        emotional_text="I feel proud and accomplished. I'm grateful and at peace. This moment is beautiful and I want to hold onto it.",
        target_agent_ids=["rosa"],
    ))

    # Day 12: Marcus has a breakdown
    world.schedule_event(ScheduledEvent(
        tick=dt(12, 22),
        location="home",
        description="Marcus sits alone in the dark kitchen. No job responses. Savings running out. Lisa crying.",
        emotional_text="I lost everything and it's all my fault. I feel worthless and hopeless. I'm devastated, the grief is overwhelming. I destroyed my family's trust and I can't fix it.",
        target_agent_ids=["marcus"],
    ))

    # Day 14: Company picnic for remaining employees (survivors guilt)
    world.schedule_event(ScheduledEvent(
        tick=dt(14, 12),
        location="park",
        description="Meridian Corp holds a 'team building' picnic. The remaining employees feel guilty.",
        emotional_text="I feel ashamed and guilty about what I did. I'm so disappointed in myself. I'm pretending to be fine at work but I'm falling apart inside.",
        target_agent_ids=["sarah", "tom", "priya", "diana", "chen", "kevin", "lena", "andre", "richard", "victoria"],
    ))

"""External information ingestion for world what-if scenarios."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import ScheduledEvent, World


@dataclass
class ExternalSignalPlan:
    source_text: str
    label: str
    kind: str
    severity: float
    start_tick: int
    scheduled_events: list["ScheduledEvent"] = field(default_factory=list)
    role_impacts: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source_text": self.source_text,
            "label": self.label,
            "kind": self.kind,
            "severity": round(self.severity, 3),
            "start_tick": self.start_tick,
            "notes": list(self.notes),
            "role_impacts": {
                role: {k: round(v, 3) for k, v in values.items()}
                for role, values in self.role_impacts.items()
            },
            "scheduled_events": [
                {
                    "tick": event.tick,
                    "location": event.location,
                    "description": event.description,
                    "emotional_text": event.emotional_text,
                    "severity": event.severity,
                    "kind": event.kind,
                }
                for event in self.scheduled_events
            ],
        }


def _percent_multiplier(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return 1.0
    return max(1.0, 1.0 + float(match.group(1)) / 100.0)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _oil_price_plan(text: str, start_tick: int) -> ExternalSignalPlan:
    from .world import ScheduledEvent

    surge = _percent_multiplier(text)
    severity = min(3.0, 1.2 + surge)
    role_impacts = {
        "dock_worker": {"debt_pressure": 0.22 * surge, "tension": 0.12 * surge, "valence": -0.08 * surge},
        "market_vendor": {"debt_pressure": 0.18 * surge, "tension": 0.1 * surge, "valence": -0.07 * surge},
        "factory_worker": {"debt_pressure": 0.14 * surge, "tension": 0.09 * surge, "valence": -0.05 * surge},
        "community": {"debt_pressure": 0.12 * surge, "tension": 0.08 * surge, "valence": -0.05 * surge},
        "office_professional": {"debt_pressure": 0.06 * surge, "tension": 0.05 * surge, "valence": -0.03 * surge},
        "government_worker": {"secret_pressure": 0.03 * surge, "tension": 0.05 * surge},
        "office_worker": {"debt_pressure": 0.08 * surge, "tension": 0.06 * surge, "valence": -0.04 * surge},
        "manager": {"debt_pressure": 0.05 * surge, "secret_pressure": 0.03 * surge, "tension": 0.05 * surge},
        "bartender": {"debt_pressure": 0.14 * surge, "tension": 0.08 * surge, "valence": -0.05 * surge},
        "teacher": {"debt_pressure": 0.09 * surge, "tension": 0.06 * surge, "valence": -0.04 * surge},
        "retiree": {"debt_pressure": 0.07 * surge, "tension": 0.05 * surge, "valence": -0.03 * surge},
    }
    events = [
        ScheduledEvent(
            tick=start_tick,
            location="docks",
            description="Fuel costs double overnight. Dock crews hear that shipping margins have evaporated and overtime may vanish first.",
            emotional_text="Every haul now costs more to move. Work that felt hard but survivable suddenly feels like it could stop making sense at all.",
            severity=severity,
            kind="macro_cost_shock",
        ),
        ScheduledEvent(
            tick=start_tick,
            location="warehouse",
            description="Suppliers push emergency surcharges through the logistics chain. Dispatchers start rerouting shipments and cutting lower-margin runs.",
            emotional_text="The math changed while we were sleeping. Every route, every delivery, every promise now looks thinner and more fragile.",
            severity=severity,
            kind="macro_cost_shock",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="main_market",
            description="Vendors wake up to higher transport costs and thinner stock. Customers complain before stall owners have answers.",
            emotional_text="I can already hear the blame before I even finish unpacking. Prices are jumping and everyone wants somebody close enough to yell at.",
            severity=severity,
            kind="macro_cost_shock",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="office_tower",
            description="Executives and analysts circulate emergency memos about freight surcharges, pricing pressure, and which costs will be passed on first.",
            emotional_text="The spreadsheets still look tidy, but everyone in the room knows this kind of jump starts moving pain outward fast.",
            severity=severity * 0.85,
            kind="macro_cost_shock",
        ),
        ScheduledEvent(
            tick=start_tick + 2,
            location="community_center",
            description="Families compare grocery, transport, and utility spikes. Small debts that felt containable last week suddenly look like a longer slide.",
            emotional_text="None of this is one bill anymore. It feels like the whole floor of daily life just moved upward at once.",
            severity=severity * 0.9,
            kind="macro_cost_shock",
        ),
    ]
    return ExternalSignalPlan(
        source_text=text,
        label="Oil price surge",
        kind="oil_price_surge",
        severity=severity,
        start_tick=start_tick,
        scheduled_events=events,
        role_impacts=role_impacts,
        notes=[
            "Transport-heavy roles absorb the first impact through debt and tension.",
            "Market and community roles pick up second-order cost pressure within 24-48 hours.",
        ],
    )


def _mass_layoffs_plan(text: str, start_tick: int) -> ExternalSignalPlan:
    from .world import ScheduledEvent

    pct = _percent_multiplier(text)
    severity = min(3.0, 1.5 + pct * 0.5)
    role_impacts = {
        "factory_worker": {"debt_pressure": 0.28 * pct, "tension": 0.2 * pct, "valence": -0.12 * pct},
        "dock_worker": {"debt_pressure": 0.22 * pct, "tension": 0.16 * pct, "valence": -0.1 * pct},
        "office_worker": {"debt_pressure": 0.2 * pct, "tension": 0.18 * pct, "valence": -0.1 * pct},
        "office_professional": {"debt_pressure": 0.15 * pct, "tension": 0.14 * pct, "valence": -0.08 * pct},
        "manager": {"debt_pressure": 0.1 * pct, "secret_pressure": 0.08 * pct, "tension": 0.12 * pct, "valence": -0.05 * pct},
        "market_vendor": {"debt_pressure": 0.12 * pct, "tension": 0.1 * pct, "valence": -0.06 * pct},
        "bartender": {"debt_pressure": 0.1 * pct, "tension": 0.08 * pct, "valence": -0.05 * pct},
        "community": {"debt_pressure": 0.08 * pct, "tension": 0.1 * pct, "valence": -0.06 * pct},
        "teacher": {"tension": 0.08 * pct, "valence": -0.04 * pct},
        "healthcare": {"tension": 0.06 * pct, "valence": -0.03 * pct},
        "government_worker": {"secret_pressure": 0.06 * pct, "tension": 0.08 * pct},
        "student": {"tension": 0.1 * pct, "valence": -0.06 * pct},
        "retiree": {"tension": 0.05 * pct, "valence": -0.03 * pct},
    }
    events = [
        ScheduledEvent(
            tick=start_tick,
            location="office_tower",
            description="Management announces mass layoffs effective immediately. Security escorts workers out of the building with boxes.",
            emotional_text="The email went out at 8am. By 9am people were carrying boxes past my desk. Nobody looked at each other. The silence was louder than shouting.",
            severity=severity,
            kind="mass_layoffs",
        ),
        ScheduledEvent(
            tick=start_tick,
            location="factory_floor",
            description="Factory floor supervisors gather workers for an emergency meeting. Shifts are being eliminated. Seniority offers no protection this time.",
            emotional_text="Twenty years and they read our names off a list like inventory. My hands were still dirty from the morning run when they told me.",
            severity=severity,
            kind="mass_layoffs",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="workers_canteen",
            description="Remaining workers sit in stunned silence. Some cry. Others are already calculating how many weeks of savings they have left.",
            emotional_text="I keep doing the math. Rent. Car payment. Insurance. The numbers stop adding up before I finish the list.",
            severity=severity * 0.9,
            kind="mass_layoffs",
        ),
        ScheduledEvent(
            tick=start_tick + 2,
            location="community_center",
            description="Laid-off workers and families gather. Some come for information about unemployment benefits. Others just need to not be alone.",
            emotional_text="My kids asked why I was home on a Tuesday. I said it was a day off. They will find out soon enough.",
            severity=severity * 0.85,
            kind="mass_layoffs",
        ),
        ScheduledEvent(
            tick=start_tick + 3,
            location="main_market",
            description="Market vendors notice spending dropping immediately. Customers walk past stalls they used to stop at. Credit requests go up.",
            emotional_text="The regulars are not buying. Some of them cannot look at me when they walk past. I know that look. I have worn it.",
            severity=severity * 0.7,
            kind="mass_layoffs_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 5,
            location="central_bar",
            description="After-hours conversations turn dark. People who never drank alone are here alone. Arguments break out about who knew what and when.",
            emotional_text="The bar is full of people pretending they came for one drink. Nobody is leaving. Nobody is talking about tomorrow.",
            severity=severity * 0.65,
            kind="mass_layoffs_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 8,
            location="city_hall",
            description="Town council holds an emergency session on the layoffs. Citizens pack the gallery demanding answers about job programs and emergency relief.",
            emotional_text="They sat behind their desks and used words like 'transition' and 'restructuring'. Nobody in the gallery was restructuring. They were drowning.",
            severity=severity * 0.75,
            kind="institutional_response",
        ),
    ]
    return ExternalSignalPlan(
        source_text=text,
        label="Mass layoffs",
        kind="mass_layoffs",
        severity=severity,
        start_tick=start_tick,
        scheduled_events=events,
        role_impacts=role_impacts,
        notes=[
            "Direct job loss creates immediate debt pressure and existential fear.",
            "Second-order: consumer spending drops within 24h, service sector feels it within 48h.",
            "Third-order: community fabric strains as mutual aid demand exceeds supply.",
            "Institutional trust erodes as government response is perceived as slow.",
        ],
    )


def _brand_scandal_plan(text: str, start_tick: int) -> ExternalSignalPlan:
    from .world import ScheduledEvent

    severity = 2.2
    # Brand scandal affects consumer-facing roles and creates moral pressure
    # dread_pressure = moral injury / ethical distress (non-economic persistent stress)
    role_impacts = {
        "market_vendor": {"tension": 0.15, "debt_pressure": 0.12, "dread_pressure": 0.1, "valence": -0.08},
        "factory_worker": {"tension": 0.18, "dread_pressure": 0.15, "secret_pressure": 0.08, "valence": -0.1},
        "office_professional": {"tension": 0.12, "dread_pressure": 0.12, "secret_pressure": 0.1, "valence": -0.06},
        "office_worker": {"tension": 0.1, "dread_pressure": 0.1, "secret_pressure": 0.06, "valence": -0.05},
        "manager": {"tension": 0.14, "dread_pressure": 0.14, "secret_pressure": 0.15, "valence": -0.08},
        "community": {"tension": 0.12, "dread_pressure": 0.12, "valence": -0.08},
        "student": {"tension": 0.14, "dread_pressure": 0.16, "valence": -0.1},
        "teacher": {"tension": 0.1, "dread_pressure": 0.1, "valence": -0.06},
        "healthcare": {"tension": 0.06, "dread_pressure": 0.06, "valence": -0.04},
        "government_worker": {"tension": 0.08, "dread_pressure": 0.08, "secret_pressure": 0.06},
        "dock_worker": {"tension": 0.08, "dread_pressure": 0.06, "valence": -0.04},
    }
    events = [
        ScheduledEvent(
            tick=start_tick,
            location="office_tower",
            description="News breaks that a major local employer has been using child labor in its overseas supply chain. Internal emails show executives knew for years.",
            emotional_text="The company I work for knew. They knew and they kept shipping. Every paycheck I cashed came from that supply chain.",
            severity=severity,
            kind="brand_scandal",
        ),
        ScheduledEvent(
            tick=start_tick,
            location="main_market",
            description="Customers start confronting vendors who sell the brand's products. Some vendors remove products preemptively. Others defend their livelihood.",
            emotional_text="A woman held up a chocolate bar and asked me how it tasted knowing a child made it. I did not have an answer. I still have inventory to move.",
            severity=severity,
            kind="brand_scandal",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="student_union",
            description="Student organizations call for an immediate campus-wide boycott. Protest signs go up overnight. Social media campaigns explode.",
            emotional_text="This is exactly what we have been saying. The system grinds children and calls it a supply chain. We are not buying anymore.",
            severity=severity * 0.9,
            kind="boycott_call",
        ),
        ScheduledEvent(
            tick=start_tick + 2,
            location="factory_floor",
            description="Factory workers who make components for the brand face a moral crisis. Some refuse shifts. Others fear losing their jobs if they speak up.",
            emotional_text="I cannot quit. I have a family. But I also cannot pretend I do not know. Every part I assemble now feels like it weighs more.",
            severity=severity * 0.85,
            kind="brand_scandal_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 3,
            location="community_center",
            description="Community meeting about the scandal draws a crowd. Parents who bought the products feel guilty. Workers who depend on the company feel trapped.",
            emotional_text="The parents are angry at the company. The workers are angry at the parents for wanting to shut it down. Everyone is angry and nobody is wrong.",
            severity=severity * 0.8,
            kind="brand_scandal_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 5,
            location="city_hall",
            description="City council debates whether to terminate municipal contracts with the company. The company employs 400 people locally.",
            emotional_text="If the city cuts them off, hundreds of families lose income. If they do not, the city is complicit. There is no clean answer in this room.",
            severity=severity * 0.75,
            kind="institutional_response",
        ),
        ScheduledEvent(
            tick=start_tick + 10,
            location="main_market",
            description="Sales of the brand's products have dropped sharply. Some vendors have switched suppliers. Others cannot afford to and are losing customers to those who did.",
            emotional_text="The ones who switched are getting all the foot traffic now. I cannot afford new suppliers and I cannot afford to keep the old ones.",
            severity=severity * 0.6,
            kind="brand_scandal_aftermath",
        ),
    ]
    return ExternalSignalPlan(
        source_text=text,
        label="Brand scandal (child labor)",
        kind="brand_scandal",
        severity=severity,
        start_tick=start_tick,
        scheduled_events=events,
        role_impacts=role_impacts,
        notes=[
            "Consumer-facing roles face immediate revenue pressure from boycotts.",
            "Workers at the company face moral injury alongside job insecurity.",
            "Students and community activists drive the boycott pressure.",
            "Government faces a dilemma between moral stance and economic reality.",
            "Second-order: supply chain workers feel trapped between conscience and livelihood.",
        ],
    )


def _banking_panic_plan(text: str, start_tick: int) -> ExternalSignalPlan:
    from .world import ScheduledEvent

    severity = 2.8
    role_impacts = {
        "factory_worker": {"debt_pressure": 0.25, "tension": 0.18, "valence": -0.12},
        "dock_worker": {"debt_pressure": 0.22, "tension": 0.16, "valence": -0.1},
        "market_vendor": {"debt_pressure": 0.28, "tension": 0.2, "valence": -0.14},
        "office_professional": {"debt_pressure": 0.18, "tension": 0.15, "valence": -0.1},
        "office_worker": {"debt_pressure": 0.2, "tension": 0.14, "valence": -0.09},
        "manager": {"debt_pressure": 0.15, "tension": 0.12, "secret_pressure": 0.08, "valence": -0.08},
        "community": {"debt_pressure": 0.2, "tension": 0.15, "valence": -0.1},
        "bartender": {"debt_pressure": 0.18, "tension": 0.12, "valence": -0.08},
        "teacher": {"debt_pressure": 0.12, "tension": 0.1, "valence": -0.06},
        "healthcare": {"debt_pressure": 0.1, "tension": 0.08, "valence": -0.05},
        "government_worker": {"tension": 0.12, "secret_pressure": 0.1},
        "student": {"debt_pressure": 0.15, "tension": 0.12, "valence": -0.08},
        "retiree": {"debt_pressure": 0.22, "tension": 0.18, "valence": -0.14},
    }
    events = [
        ScheduledEvent(
            tick=start_tick,
            location="office_tower",
            description="Local bank announces emergency measures. ATM withdrawals are limited. Lines form at every branch as word spreads.",
            emotional_text="I saw the line at the bank before I heard the news. By the time I got there, the ATM said 'limit reached'. My rent is due Friday.",
            severity=severity,
            kind="banking_panic",
        ),
        ScheduledEvent(
            tick=start_tick,
            location="main_market",
            description="Cash registers stop accepting card payments intermittently. Vendors switch to cash-only. Customers without cash cannot buy groceries.",
            emotional_text="The card machine blinked twice and died. The woman holding formula and diapers did not have cash. Neither did I. We just stood there.",
            severity=severity,
            kind="banking_panic",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="community_center",
            description="Rumors spread that savings are frozen. Retirees and families with thin margins panic. Mutual aid networks activate but resources are thin.",
            emotional_text="My mother called crying. Her pension deposit did not clear. She has been saving for forty years and now she cannot buy groceries.",
            severity=severity * 0.9,
            kind="banking_panic",
        ),
        ScheduledEvent(
            tick=start_tick + 2,
            location="docks",
            description="Shipping companies cannot process payroll. Dock workers are told wages may be delayed indefinitely. Some walk off the job.",
            emotional_text="They said 'temporary delay'. Last time someone said temporary, my cousin lost his house. I am not waiting to find out.",
            severity=severity * 0.85,
            kind="banking_panic_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 3,
            location="city_hall",
            description="Mayor holds emergency press conference promising the situation is contained. Nobody in the audience believes it.",
            emotional_text="The mayor used the word 'contained' four times. That is how you know it is not contained.",
            severity=severity * 0.7,
            kind="institutional_response",
        ),
        ScheduledEvent(
            tick=start_tick + 6,
            location="central_park",
            description="Protest march from bank to city hall. Signs read 'Where is our money?' and 'Bail us out like you bailed them out'.",
            emotional_text="We are standing in the street because the institutions that were supposed to stand for us sat down.",
            severity=severity * 0.75,
            kind="civil_unrest",
        ),
    ]
    return ExternalSignalPlan(
        source_text=text,
        label="Banking panic",
        kind="banking_panic",
        severity=severity,
        start_tick=start_tick,
        scheduled_events=events,
        role_impacts=role_impacts,
        notes=[
            "Banking panic hits everyone simultaneously — universal debt pressure.",
            "Retirees and low-income families are the most vulnerable immediately.",
            "Market vendors lose both customers and payment infrastructure.",
            "Institutional trust collapses faster than in other shocks.",
            "Civil unrest potential is high because the betrayal feels personal.",
        ],
    )


def _health_crisis_plan(text: str, start_tick: int) -> ExternalSignalPlan:
    from .world import ScheduledEvent

    severity = 2.4
    # dread_pressure = health anxiety, fear of illness/death (non-economic persistent stress)
    role_impacts = {
        "healthcare": {"tension": 0.25, "dread_pressure": 0.22, "valence": -0.15, "debt_pressure": 0.05},
        "community": {"tension": 0.18, "dread_pressure": 0.16, "valence": -0.12},
        "teacher": {"tension": 0.15, "dread_pressure": 0.14, "valence": -0.1},
        "factory_worker": {"tension": 0.12, "dread_pressure": 0.12, "valence": -0.08},
        "dock_worker": {"tension": 0.1, "dread_pressure": 0.1, "valence": -0.06},
        "market_vendor": {"tension": 0.14, "dread_pressure": 0.12, "debt_pressure": 0.1, "valence": -0.08},
        "office_professional": {"tension": 0.08, "dread_pressure": 0.08, "valence": -0.04},
        "office_worker": {"tension": 0.08, "dread_pressure": 0.08, "valence": -0.04},
        "manager": {"tension": 0.06, "dread_pressure": 0.06, "valence": -0.03},
        "government_worker": {"tension": 0.12, "dread_pressure": 0.1, "secret_pressure": 0.08},
        "student": {"tension": 0.1, "dread_pressure": 0.1, "valence": -0.06},
        "retiree": {"tension": 0.2, "dread_pressure": 0.22, "valence": -0.14},
        "bartender": {"tension": 0.1, "dread_pressure": 0.1, "debt_pressure": 0.08, "valence": -0.06},
    }
    events = [
        ScheduledEvent(
            tick=start_tick,
            location="hospital",
            description="Hospital declares emergency overflow. Waiting rooms are full. Staff are pulled from every department. Non-emergency procedures cancelled.",
            emotional_text="We ran out of beds at 6am. By noon I had triaged more people than I normally see in a week. Some of them looked at me like I could save them. I am not sure I can.",
            severity=severity,
            kind="health_crisis",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="north_school",
            description="Schools announce closures as cases surge. Parents scramble for childcare. Some cannot miss work. Others have no choice.",
            emotional_text="The school sent the text at 7am. By 7:15 I was calling everyone I know. I cannot stay home and I cannot send her anywhere.",
            severity=severity * 0.8,
            kind="health_crisis",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="main_market",
            description="Market traffic drops sharply as people avoid crowded spaces. Vendors watch empty aisles. Some start wearing masks. Others refuse.",
            emotional_text="The market feels like a ghost town with a few stubborn ghosts. Everyone who is here is watching everyone else, wondering who is carrying what.",
            severity=severity * 0.7,
            kind="health_crisis_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 3,
            location="community_center",
            description="Mutual aid groups organize supply drops for quarantined families. Volunteers are overwhelmed. Some volunteers get sick.",
            emotional_text="We are delivering food to people who cannot leave their homes. Three of our volunteers are now in those homes. The circle keeps shrinking.",
            severity=severity * 0.75,
            kind="health_crisis_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 5,
            location="factory_floor",
            description="Factory runs at 60% capacity as workers call out sick or refuse to come in. Management threatens disciplinary action. Union pushes back.",
            emotional_text="They want us to choose between our health and our paycheck. That is not a choice. That is a trap.",
            severity=severity * 0.7,
            kind="health_crisis_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 8,
            location="city_hall",
            description="Public health officials brief city council on containment measures. Mandatory closures discussed. Business owners protest outside.",
            emotional_text="The scientists say close everything. The business owners say they will go bankrupt. The politicians are trying to find a middle that does not exist.",
            severity=severity * 0.65,
            kind="institutional_response",
        ),
    ]
    return ExternalSignalPlan(
        source_text=text,
        label="Public health crisis",
        kind="health_crisis",
        severity=severity,
        start_tick=start_tick,
        scheduled_events=events,
        role_impacts=role_impacts,
        notes=[
            "Healthcare workers absorb the first physical and emotional impact.",
            "Schools closing creates cascading childcare pressure for working parents.",
            "Market and service sectors lose foot traffic immediately.",
            "Retirees face highest personal risk, driving anxiety.",
            "Institutional trust depends on speed and honesty of government response.",
        ],
    )


def _military_crisis_plan(text: str, start_tick: int) -> ExternalSignalPlan:
    from .world import ScheduledEvent

    severity = 3.0
    # dread_pressure = existential fear, survival anxiety (highest of any shock type)
    role_impacts = {
        "factory_worker": {"tension": 0.2, "dread_pressure": 0.25, "valence": -0.15},
        "dock_worker": {"tension": 0.18, "dread_pressure": 0.22, "valence": -0.12},
        "market_vendor": {"tension": 0.15, "dread_pressure": 0.2, "debt_pressure": 0.1, "valence": -0.1},
        "office_professional": {"tension": 0.14, "dread_pressure": 0.2, "valence": -0.1},
        "office_worker": {"tension": 0.14, "dread_pressure": 0.2, "valence": -0.1},
        "manager": {"tension": 0.12, "dread_pressure": 0.18, "secret_pressure": 0.06, "valence": -0.08},
        "community": {"tension": 0.22, "dread_pressure": 0.28, "valence": -0.18},
        "teacher": {"tension": 0.18, "dread_pressure": 0.24, "valence": -0.14},
        "healthcare": {"tension": 0.2, "dread_pressure": 0.22, "valence": -0.12},
        "government_worker": {"tension": 0.2, "dread_pressure": 0.2, "secret_pressure": 0.15, "valence": -0.1},
        "student": {"tension": 0.2, "dread_pressure": 0.26, "valence": -0.16},
        "retiree": {"tension": 0.18, "dread_pressure": 0.24, "valence": -0.15},
        "bartender": {"tension": 0.14, "dread_pressure": 0.18, "valence": -0.1},
    }
    events = [
        ScheduledEvent(
            tick=start_tick,
            location="office_tower",
            description="Breaking news: military strike launched. Screens in every office show missile trajectories. Markets crash. Phone lines jam.",
            emotional_text="The news hit every screen at once. Nobody moved. Somebody's phone rang and nobody answered. The world just changed and we are sitting in cubicles.",
            severity=severity,
            kind="military_crisis",
        ),
        ScheduledEvent(
            tick=start_tick,
            location="community_center",
            description="Families gather around TVs and phones. Children sense the fear. Some parents try to explain. Others cannot find the words.",
            emotional_text="My daughter asked if the bombs would come here. I said no. I do not know if I was lying.",
            severity=severity,
            kind="military_crisis",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="main_market",
            description="Panic buying begins. Shelves empty within hours. Water, canned food, batteries — everything people think they need for the end.",
            emotional_text="The market turned into a stampede. People were grabbing things they do not even eat. It is not about food. It is about feeling like you did something.",
            severity=severity * 0.9,
            kind="panic_buying",
        ),
        ScheduledEvent(
            tick=start_tick + 1,
            location="student_union",
            description="Students organize emergency anti-war protest. Some are terrified. Others are angry. Draft rumors circulate wildly.",
            emotional_text="Half of us are protesting and half of us are wondering if we are going to get drafted. The anger and the fear taste the same right now.",
            severity=severity * 0.85,
            kind="civil_unrest",
        ),
        ScheduledEvent(
            tick=start_tick + 2,
            location="hospital",
            description="Hospital activates emergency protocols. Staff are called back from leave. Psych ward sees surge in anxiety and panic attacks.",
            emotional_text="We are preparing for something we hope never arrives. But the preparation itself is breaking people. The waiting room is full of panic attacks.",
            severity=severity * 0.8,
            kind="military_crisis_ripple",
        ),
        ScheduledEvent(
            tick=start_tick + 3,
            location="city_hall",
            description="Emergency council session. Civil defense plans dusted off. Officials look scared. Citizens demand evacuation plans that do not exist.",
            emotional_text="The officials read from a binder that was last updated in 1987. Everyone in the room knows those plans assume a world that no longer exists.",
            severity=severity * 0.85,
            kind="institutional_response",
        ),
        ScheduledEvent(
            tick=start_tick + 5,
            location="central_park",
            description="Vigil and protest merge into one crowd. Candles and angry signs side by side. Nobody knows if this is grief or resistance.",
            emotional_text="We are holding candles and shouting at the same time. Some people are praying. Some are cursing. The sky looks exactly the same as yesterday.",
            severity=severity * 0.7,
            kind="civil_unrest",
        ),
        ScheduledEvent(
            tick=start_tick + 12,
            location="community_center",
            description="After days of crisis, exhaustion sets in. Children are still not in school. Some families have left town. Those remaining feel abandoned.",
            emotional_text="The people who left had somewhere to go. The rest of us are here because here is all we have.",
            severity=severity * 0.6,
            kind="military_crisis_aftermath",
        ),
    ]
    return ExternalSignalPlan(
        source_text=text,
        label="Military/nuclear crisis",
        kind="military_crisis",
        severity=severity,
        start_tick=start_tick,
        scheduled_events=events,
        role_impacts=role_impacts,
        notes=[
            "Military crisis is universal — every role feels existential threat simultaneously.",
            "Unlike economic shocks, the fear is about survival, not livelihood.",
            "Panic buying creates immediate supply disruption.",
            "Students and young people drive both protest and draft anxiety.",
            "Institutional trust depends on perceived competence of emergency response.",
            "Community fragmentation as some flee and others cannot.",
        ],
    )


def _generic_macro_plan(text: str, start_tick: int) -> ExternalSignalPlan:
    from .world import ScheduledEvent

    return ExternalSignalPlan(
        source_text=text,
        label="External macro signal",
        kind="external_macro_signal",
        severity=1.6,
        start_tick=start_tick,
        scheduled_events=[
            ScheduledEvent(
                tick=start_tick,
                location="office_tower",
                description=f"External news hits the town: {text}",
                emotional_text="Something outside the town just changed the stakes here. People do not know the full shape of it yet, but everyone feels the floor move.",
                severity=1.6,
                kind="external_macro_signal",
            ),
            ScheduledEvent(
                tick=start_tick + 1,
                location="community_center",
                description=f"Residents and workers start asking what the new development means locally: {text}",
                emotional_text="The event is no longer distant. People are trying to translate a big headline into rent, safety, jobs, and who gets squeezed first.",
                severity=1.5,
                kind="external_macro_signal",
            ),
        ],
        role_impacts={
            "community": {"tension": 0.06, "debt_pressure": 0.05},
            "market_vendor": {"tension": 0.05, "debt_pressure": 0.05},
            "office_professional": {"tension": 0.04},
        },
        notes=["Generic external signals route first through uncertainty, then through household interpretation."],
    )


def interpret_external_information(text: str, world: "World", start_tick: int | None = None) -> ExternalSignalPlan:
    lowered = text.lower().strip()
    tick = world.tick_count + 1 if start_tick is None else max(1, start_tick)

    # Oil / energy price shock
    if any(token in lowered for token in ("oil", "fuel", "diesel", "gasoline", "energy price")) and any(token in lowered for token in ("surge", "spike", "jump", "up", "increase")):
        return _oil_price_plan(text, tick)

    # Mass layoffs
    if any(token in lowered for token in ("layoff", "lay off", "laid off", "mass firing", "downsiz", "redundanc", "job cut")):
        return _mass_layoffs_plan(text, tick)

    # Brand scandal / child labor / ethical violation
    if any(token in lowered for token in ("child labo", "child slave", "scandal", "sweatshop", "forced labo")) or (
        any(token in lowered for token in ("brand", "company", "corporation", "firm")) and
        any(token in lowered for token in ("scandal", "exposed", "caught", "reveal", "investigat"))
    ):
        return _brand_scandal_plan(text, tick)

    # Banking panic / financial crisis
    if any(token in lowered for token in ("bank run", "banking panic", "bank fail", "bank collaps", "financial crisis", "credit freeze", "liquidity", "deposit freeze", "savings frozen")):
        return _banking_panic_plan(text, tick)

    # Public health crisis
    if any(token in lowered for token in ("pandemic", "epidemic", "outbreak", "virus", "contagion", "quarantine", "health crisis", "health emergency", "hospital overflow")):
        return _health_crisis_plan(text, tick)

    # Military / nuclear crisis
    if any(token in lowered for token in ("nuclear", "military strike", "war", "missile", "invasion", "bombing", "airstrike", "military crisis", "nuclear strike", "declaration of war")):
        return _military_crisis_plan(text, tick)

    return _generic_macro_plan(text, tick)


def apply_external_information(world: "World", plan: ExternalSignalPlan) -> dict:
    """Apply a world-information plan using individual agent appraisal.

    Each agent interprets the shock for themselves based on their personality,
    financial situation, threat lens, and current emotional state.
    Macro outcomes emerge from aggregating individual reactions.
    """
    from .shock_appraisal import classify_shock, appraise_shock_for_all, apply_reactions

    # Schedule narrative events (the physical scenes — these stay hardcoded
    # because they're world facts, not individual reactions)
    for event in plan.scheduled_events:
        world.schedule_event(event)

    # Classify what KIND of threat this shock represents
    shock_profile = classify_shock(plan.kind, plan.source_text, plan.severity)

    # Each agent individually appraises the shock
    reactions = appraise_shock_for_all(world.agents, shock_profile)

    # Apply individual reactions to agent state
    result = apply_reactions(
        world.agents, reactions, world.tick_count,
        shock_label=plan.label,
        shock_kind=plan.kind,
        source_text=plan.source_text,
    )

    # Initialize information awareness tracking
    if not hasattr(world, "_info_awareness"):
        world._info_awareness: dict[str, set[str]] = {}
    if plan.label not in world._info_awareness:
        world._info_awareness[plan.label] = set()
    world._info_awareness[plan.label].update(result["awareness_set"])

    if not hasattr(world, "external_signals"):
        world.external_signals = []
    world.external_signals.append(plan.as_dict())

    return {
        "signal": plan.as_dict(),
        "impacted_agents": result["impacted_agents"],
        "avg_personal_severity": result["avg_severity"],
        "scheduled_event_count": len(plan.scheduled_events),
    }

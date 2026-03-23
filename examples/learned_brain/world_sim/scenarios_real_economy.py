"""Real-world economic scenario: named companies, real leaders, structured economy.

Builds a simulation with:
  - Real companies (NVIDIA, Apple, Microsoft, Amazon, Walmart, etc.)
  - Real institutions (Federal Reserve, Congress, WHO, etc.)
  - Named leaders with personality profiles mirrored from real behavior
  - Individual consumers/workers representing population segments
  - Structural economic relationships (supply chains, employment, regulation)

Each company has 15+ LLM agents: CEO, CFO, board members, division heads,
key employees. Their decisions ripple through supply chains, employment,
and consumer behavior.

The three pillars of the economy:
  1. GOVERNMENT — Fed, Congress, regulators, central banks
  2. FIRMS — tech, retail, energy, finance, healthcare companies
  3. INDIVIDUALS — workers, consumers, investors, unemployed
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .world import World, Location
from .world_agent import WorldAgent, Personality, HeartState
from .relationship import RelationshipStore
from .human_profiles import assign_human_profile
from .ripple_engine import OrganizationalFabric, OrgLink, build_organizational_fabric


# ═══════════════════════════════════════════════════════════════════════════
# LOCATIONS — sectors of the real economy
# ═══════════════════════════════════════════════════════════════════════════

REAL_LOCATIONS = [
    # Tech sector
    Location("nvidia_hq", "NVIDIA Headquarters", "designing AI chips and GPUs"),
    Location("apple_hq", "Apple Campus", "product design and supply chain management"),
    Location("microsoft_hq", "Microsoft Campus", "cloud services and enterprise software"),
    Location("amazon_hq", "Amazon HQ", "e-commerce operations and AWS cloud"),
    Location("google_hq", "Google Campus", "search, ads, and AI research"),
    Location("meta_hq", "Meta Headquarters", "social media and VR platforms"),
    Location("tesla_hq", "Tesla Factory", "EV manufacturing and energy"),

    # Retail / Consumer
    Location("walmart_hq", "Walmart Corporate", "retail operations and supply chain"),
    Location("costco_hq", "Costco Corporate", "wholesale retail operations"),
    Location("walmart_store", "Walmart Supercenter", "selling groceries and goods"),
    Location("costco_store", "Costco Warehouse", "bulk retail sales"),
    Location("main_street", "Main Street Shops", "small business retail"),

    # Finance
    Location("jpmorgan_hq", "JPMorgan Chase HQ", "banking and financial services"),
    Location("goldman_hq", "Goldman Sachs HQ", "investment banking and trading"),
    Location("fed_building", "Federal Reserve Building", "monetary policy and bank regulation"),
    Location("wall_street", "Wall Street Trading Floor", "securities trading"),

    # Energy
    Location("exxon_hq", "ExxonMobil HQ", "oil and gas operations"),
    Location("chevron_hq", "Chevron HQ", "energy production"),

    # Healthcare / Pharma
    Location("pfizer_hq", "Pfizer HQ", "pharmaceutical development"),
    Location("hospital_system", "Regional Hospital Network", "patient care"),

    # Government
    Location("white_house", "The White House", "executive branch decisions"),
    Location("capitol", "US Capitol", "legislative deliberation"),
    Location("treasury", "US Treasury", "fiscal policy"),
    Location("cdc_hq", "CDC Headquarters", "public health guidance"),
    Location("pentagon", "The Pentagon", "defense operations"),

    # International
    Location("who_hq", "WHO Geneva", "global health coordination"),
    Location("ecb_hq", "European Central Bank", "eurozone monetary policy"),
    Location("opec_hq", "OPEC Headquarters", "oil production coordination"),

    # Workers / Consumers
    Location("factory_district", "Industrial District", "manufacturing and warehousing"),
    Location("office_park", "Corporate Office Park", "white-collar work"),
    Location("residential_suburb", "Suburban Neighborhoods", "family life and commuting"),
    Location("urban_downtown", "Downtown District", "urban living and services"),
    Location("university", "State University", "education and research"),
    Location("gig_economy", "Gig Economy Hub", "freelance and contract work"),
]


# ═══════════════════════════════════════════════════════════════════════════
# COMPANY AND INSTITUTION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CompanyDef:
    """Definition of a company or institution with its key personnel."""
    name: str
    sector: str
    hq_location: str
    employee_count: int  # non-LLM workers at this company
    revenue_influence: float  # 0-1, how much this company affects the economy
    supply_chain_to: list[str] = field(default_factory=list)  # companies it supplies
    regulated_by: list[str] = field(default_factory=list)
    key_personnel: list[dict] = field(default_factory=list)  # LLM agents


# Each person: name, title, background, temperament + personality traits
# These are fictional analogs inspired by real archetypes, NOT real people

COMPANIES = [
    # ── TECH ──────────────────────────────────────────────────
    CompanyDef(
        name="NovaTech", sector="tech_chips", hq_location="nvidia_hq",
        employee_count=30, revenue_influence=0.08,
        supply_chain_to=["ApexDevices", "CloudScale", "VoltMotors"],
        regulated_by=["SEC", "Commerce_Dept"],
        key_personnel=[
            {"name": "Jensen Huang", "title": "CEO", "background": "Founded NovaTech, built it from graphics cards to AI dominance. Engineer-turned-visionary who bets big on technology waves.", "temperament": "Relentless optimist, data-driven, speaks in product roadmaps. Leather jacket energy.", "threat_lens": "humiliation", "core_need": "dignity", "self_story": "climber", "coping_style": "perform competence", "conflict_style": "command"},
            {"name": "Colette Kress", "title": "CFO", "background": "Wall Street veteran turned tech CFO. Guards the balance sheet like a fortress.", "temperament": "Precise, cautious with numbers, protective of margins", "threat_lens": "scarcity", "core_need": "control", "self_story": "guardian", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
            {"name": "Ajay Puri", "title": "VP Sales", "background": "Sells billions in chips to every major cloud provider. Lives on planes and handshakes.", "temperament": "Charismatic dealmaker, reads rooms instantly, always closing", "threat_lens": "betrayal", "core_need": "dignity", "self_story": "climber", "coping_style": "perform competence", "conflict_style": "cool negotiation"},
        ],
    ),
    CompanyDef(
        name="ApexDevices", sector="tech_consumer", hq_location="apple_hq",
        employee_count=40, revenue_influence=0.10,
        supply_chain_to=["RetailGiant", "CostPlus"],
        regulated_by=["SEC", "FTC"],
        key_personnel=[
            {"name": "Tim Park", "title": "CEO", "background": "Operations genius who inherited the company from a legendary founder. Calm, methodical, values privacy above all.", "temperament": "Quiet authority, deliberate speaker, believes in doing fewer things better", "threat_lens": "exposure", "core_need": "control", "self_story": "guardian", "coping_style": "control the room", "conflict_style": "cool negotiation"},
            {"name": "Luca Torres", "title": "CFO", "background": "Italian-American finance executive. Manages the world's largest cash pile with conservative discipline.", "temperament": "Numbers-first, risk-averse, speaks in quarters and margins", "threat_lens": "chaos", "core_need": "control", "self_story": "guardian", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
            {"name": "Craig Rivera", "title": "VP Hardware", "background": "Engineer who designs the flagship products. Lives in prototyping labs.", "temperament": "Detail-obsessed perfectionist, hates shipping late, values craft", "threat_lens": "humiliation", "core_need": "dignity", "self_story": "fixer", "coping_style": "disappear into work", "conflict_style": "keep score"},
        ],
    ),
    CompanyDef(
        name="CloudScale", sector="tech_cloud", hq_location="microsoft_hq",
        employee_count=45, revenue_influence=0.09,
        supply_chain_to=["ApexDevices", "RetailGiant", "PharmaCore"],
        regulated_by=["SEC", "FTC", "EU_Commission"],
        key_personnel=[
            {"name": "Satya Reddy", "title": "CEO", "background": "Born in India, rose through cloud division. Transformed a legacy software company into a cloud and AI powerhouse.", "temperament": "Empathetic listener, growth mindset evangelist, quotes cricket analogies", "threat_lens": "abandonment", "core_need": "belonging", "self_story": "fixer", "coping_style": "reach for connection", "conflict_style": "cool negotiation"},
            {"name": "Amy Zhao", "title": "CFO", "background": "Goldman Sachs veteran. Runs the financial engine of the world's second-most-valuable company.", "temperament": "Sharp, fast-talking, sees patterns in revenue data others miss", "threat_lens": "scarcity", "core_need": "control", "self_story": "climber", "coping_style": "intellectualize", "conflict_style": "command"},
            {"name": "Kevin Turner", "title": "COO", "background": "Walmart executive turned tech COO. Brings retail-scale operational discipline to cloud services.", "temperament": "Blunt Midwesterner, measures everything, expects results yesterday", "threat_lens": "chaos", "core_need": "control", "self_story": "operator", "coping_style": "control the room", "conflict_style": "go sharp"},
        ],
    ),
    CompanyDef(
        name="MegaMart", sector="tech_ecommerce", hq_location="amazon_hq",
        employee_count=60, revenue_influence=0.12,
        supply_chain_to=["RetailGiant", "CostPlus"],
        regulated_by=["SEC", "FTC", "Labor_Board"],
        key_personnel=[
            {"name": "Andy Burke", "title": "CEO", "background": "Rose from AWS engineer to CEO after the founder stepped down. Quiet operator who runs the machine.", "temperament": "Data-obsessed, frugal even at scale, customer-metrics-first", "threat_lens": "chaos", "core_need": "control", "self_story": "operator", "coping_style": "intellectualize", "conflict_style": "command"},
            {"name": "Brian Nash", "title": "CFO", "background": "Managed Amazon's shift to profitability after years of growth-first spending.", "temperament": "Careful, margin-conscious, challenges every budget line", "threat_lens": "scarcity", "core_need": "safety", "self_story": "guardian", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
            {"name": "Dave Cho", "title": "VP Logistics", "background": "Built the delivery network that ships 10 billion packages a year. Former Marine.", "temperament": "Military precision, hates waste, respects execution over ideas", "threat_lens": "betrayal", "core_need": "dignity", "self_story": "loyalist", "coping_style": "confront head-on", "conflict_style": "go sharp"},
        ],
    ),

    # ── RETAIL ────────────────────────────────────────────────
    CompanyDef(
        name="RetailGiant", sector="retail", hq_location="walmart_hq",
        employee_count=80, revenue_influence=0.10,
        supply_chain_to=[],
        regulated_by=["FTC", "Labor_Board"],
        key_personnel=[
            {"name": "Doug Whitfield", "title": "CEO", "background": "Third-generation retail. Grew up in stores, knows every aisle. Modernized a legacy retailer for the e-commerce age.", "temperament": "Folksy but sharp, speaks in customer stories, competitive with MegaMart", "threat_lens": "betrayal", "core_need": "belonging", "self_story": "loyalist", "coping_style": "reach for connection", "conflict_style": "straight negotiation"},
            {"name": "John Furner", "title": "US CEO", "background": "Started as an hourly associate, worked up to running US operations. Knows the front lines.", "temperament": "Empathetic to workers, pragmatic about costs, walks stores weekly", "threat_lens": "scarcity", "core_need": "usefulness", "self_story": "provider", "coping_style": "caretake first", "conflict_style": "appease first"},
            {"name": "Judith McKenna", "title": "International CEO", "background": "British executive running global operations across 23 countries.", "temperament": "Strategic, culturally aware, thinks in supply chains not products", "threat_lens": "chaos", "core_need": "control", "self_story": "operator", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
        ],
    ),
    CompanyDef(
        name="CostPlus", sector="retail_wholesale", hq_location="costco_hq",
        employee_count=50, revenue_influence=0.06,
        supply_chain_to=[],
        regulated_by=["FTC"],
        key_personnel=[
            {"name": "Ron Vachris", "title": "CEO", "background": "Started as a forklift driver, rose to CEO. Believes in paying workers well and keeping margins razor-thin.", "temperament": "Blue-collar executive, worker-first philosophy, stubborn on wages", "threat_lens": "betrayal", "core_need": "dignity", "self_story": "loyalist", "coping_style": "confront head-on", "conflict_style": "straight negotiation"},
            {"name": "Gary Millerchip", "title": "CFO", "background": "Kroger veteran who manages the financial discipline of a low-margin, high-volume model.", "temperament": "Conservative spender, obsesses over member retention metrics", "threat_lens": "scarcity", "core_need": "safety", "self_story": "guardian", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
        ],
    ),

    # ── FINANCE ───────────────────────────────────────────────
    CompanyDef(
        name="FirstBank", sector="finance", hq_location="jpmorgan_hq",
        employee_count=40, revenue_influence=0.09,
        supply_chain_to=["RetailGiant", "MegaMart", "NovaTech"],
        regulated_by=["Federal_Reserve", "SEC", "Treasury"],
        key_personnel=[
            {"name": "Jamie Stone", "title": "CEO", "background": "Longest-serving big bank CEO. Survived 2008, COVID, and every crisis in between. The banker presidents call.", "temperament": "Alpha personality, speaks bluntly to Congress, believes banks are essential infrastructure", "threat_lens": "chaos", "core_need": "control", "self_story": "guardian", "coping_style": "command", "conflict_style": "command"},
            {"name": "Marianne Lake", "title": "Co-President", "background": "Former CFO, now runs consumer banking. Data-driven with a human touch.", "temperament": "Analytical but warm, translates numbers into stories", "threat_lens": "exposure", "core_need": "dignity", "self_story": "fixer", "coping_style": "perform competence", "conflict_style": "cool negotiation"},
            {"name": "Daniel Pinto", "title": "Co-President", "background": "Runs corporate and investment banking. Born in Argentina, rose through London trading floors.", "temperament": "Globally minded, risk-aware, speaks three languages", "threat_lens": "scarcity", "core_need": "control", "self_story": "survivor", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
        ],
    ),

    # ── ENERGY ────────────────────────────────────────────────
    CompanyDef(
        name="PetroMax", sector="energy", hq_location="exxon_hq",
        employee_count=35, revenue_influence=0.08,
        supply_chain_to=["MegaMart", "RetailGiant", "CostPlus"],
        regulated_by=["EPA", "Commerce_Dept"],
        key_personnel=[
            {"name": "Darren Foster", "title": "CEO", "background": "Lifer who rose from petroleum engineer to CEO. Believes in oil's role for decades to come.", "temperament": "Steady, unapologetic about fossil fuels, thinks in 30-year cycles", "threat_lens": "betrayal", "core_need": "dignity", "self_story": "loyalist", "coping_style": "perform competence", "conflict_style": "command"},
            {"name": "Kathy Mikells", "title": "CFO", "background": "Joined from Diageo. Manages capital allocation for the world's most capital-intensive industry.", "temperament": "Disciplined, dividend-first, protects shareholder returns", "threat_lens": "scarcity", "core_need": "safety", "self_story": "guardian", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
        ],
    ),

    # ── PHARMA ────────────────────────────────────────────────
    CompanyDef(
        name="PharmaCore", sector="pharma", hq_location="pfizer_hq",
        employee_count=30, revenue_influence=0.06,
        supply_chain_to=["RetailGiant", "CostPlus"],
        regulated_by=["FDA", "CDC"],
        key_personnel=[
            {"name": "Albert Bourla", "title": "CEO", "background": "Greek-born veterinarian turned pharma CEO. Led the fastest vaccine development in history.", "temperament": "Passionate, mission-driven, speaks about patients not products", "threat_lens": "abandonment", "core_need": "usefulness", "self_story": "guardian", "coping_style": "caretake first", "conflict_style": "command"},
            {"name": "David Denton", "title": "CFO", "background": "CVS veteran managing the financial transition from blockbuster drugs to mRNA platform.", "temperament": "Pragmatic, cost-conscious, balances R&D spend with shareholder returns", "threat_lens": "scarcity", "core_need": "control", "self_story": "fixer", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
        ],
    ),
]

# ── GOVERNMENT INSTITUTIONS ─────────────────────────────────

GOVERNMENT_INSTITUTIONS = [
    CompanyDef(
        name="Federal_Reserve", sector="central_bank", hq_location="fed_building",
        employee_count=15, revenue_influence=0.15,
        supply_chain_to=[],
        regulated_by=[],
        key_personnel=[
            {"name": "Jerome Mitchell", "title": "Fed Chair", "background": "Lawyer-turned-central-banker. Navigated COVID, inflation surge, and rate hike cycle. Every word he says moves markets.", "temperament": "Measured, chooses words like a surgeon, hates surprising markets", "threat_lens": "chaos", "core_need": "control", "self_story": "guardian", "coping_style": "intellectualize", "conflict_style": "cool negotiation"},
            {"name": "Lael Foster", "title": "Vice Chair", "background": "Economist who bridges academic theory and practical policy. Dove on rates, hawk on financial stability.", "temperament": "Thoughtful, consensus-builder, speaks in careful conditionals", "threat_lens": "scarcity", "core_need": "belonging", "self_story": "fixer", "coping_style": "reach for connection", "conflict_style": "appease first"},
            {"name": "Christopher Waller", "title": "Governor", "background": "Hawk on inflation. Academic economist who believes in letting markets clear.", "temperament": "Direct, data-driven, willing to dissent from the majority", "threat_lens": "chaos", "core_need": "truth", "self_story": "witness", "coping_style": "confront head-on", "conflict_style": "go sharp"},
        ],
    ),
    CompanyDef(
        name="US_Congress", sector="legislature", hq_location="capitol",
        employee_count=20, revenue_influence=0.12,
        supply_chain_to=[],
        regulated_by=[],
        key_personnel=[
            {"name": "Speaker Morrison", "title": "Speaker of the House", "background": "Career politician from a swing state. Master of legislative horse-trading.", "temperament": "Calculating, counts votes before speaking, trades favors like currency", "threat_lens": "betrayal", "core_need": "control", "self_story": "operator", "coping_style": "control the room", "conflict_style": "keep score"},
            {"name": "Senator Chen", "title": "Senate Majority Leader", "background": "Former prosecutor turned senator. Known for late-night deals and iron discipline.", "temperament": "Poker-faced, never shows his hand, rewards loyalty", "threat_lens": "exposure", "core_need": "control", "self_story": "operator", "coping_style": "control the room", "conflict_style": "command"},
            {"name": "Rep. Williams", "title": "Budget Committee Chair", "background": "Fiscal hawk from the heartland. Fights every spending bill on principle.", "temperament": "Moralistic about debt, speaks in kitchen-table economics", "threat_lens": "scarcity", "core_need": "justice", "self_story": "witness", "coping_style": "confront head-on", "conflict_style": "moralize in public"},
        ],
    ),
    CompanyDef(
        name="Treasury_Dept", sector="fiscal_policy", hq_location="treasury",
        employee_count=10, revenue_influence=0.10,
        supply_chain_to=[],
        regulated_by=["US_Congress"],
        key_personnel=[
            {"name": "Janet Morrison", "title": "Treasury Secretary", "background": "Former Fed Chair turned Treasury Secretary. The most experienced economic policymaker alive.", "temperament": "Soft-spoken authority, sees systemic risk others miss, maternal concern for workers", "threat_lens": "scarcity", "core_need": "usefulness", "self_story": "guardian", "coping_style": "caretake first", "conflict_style": "cool negotiation"},
        ],
    ),
    CompanyDef(
        name="CDC", sector="public_health", hq_location="cdc_hq",
        employee_count=10, revenue_influence=0.04,
        supply_chain_to=[],
        regulated_by=["US_Congress"],
        key_personnel=[
            {"name": "Dr. Mandy Cohen", "title": "CDC Director", "background": "Physician-administrator who must balance scientific truth with political reality.", "temperament": "Empathetic communicator, data-first, frustrated by politicization of science", "threat_lens": "chaos", "core_need": "truth", "self_story": "guardian", "coping_style": "caretake first", "conflict_style": "appease first"},
        ],
    ),
]

# ── INDIVIDUAL POPULATION SEGMENTS ──────────────────────────

POPULATION_SEGMENTS = [
    {"role": "tech_worker", "count": 60, "location": "office_park", "home": "residential_suburb",
     "backgrounds": [
         "Software engineer at a mid-size company, {years} years in tech, worries about layoffs",
         "Data scientist supporting cloud infrastructure, {years} years experience, stock options are underwater",
         "Product manager juggling roadmaps and budget cuts, {years} years in the industry",
     ],
     "temperaments": ["analytical and cautious", "ambitious but anxious about market", "burned out from crunch cycles"],
     "debt_range": (0.05, 0.25), "ambition_range": (0.3, 0.7)},

    {"role": "factory_worker", "count": 50, "location": "factory_district", "home": "residential_suburb",
     "backgrounds": [
         "Assembly line worker at an auto parts plant, {years} years, union member",
         "Warehouse picker at MegaMart fulfillment center, {years} years, sore back",
         "Machine operator at a food processing plant, {years} years, three kids",
     ],
     "temperaments": ["steady and pragmatic", "frustrated with management", "loyal but tired"],
     "debt_range": (0.1, 0.35), "ambition_range": (0.1, 0.4)},

    {"role": "retail_worker", "count": 60, "location": "walmart_store", "home": "urban_downtown",
     "backgrounds": [
         "Cashier at RetailGiant, {years} years, saving for community college",
         "Stock clerk at CostPlus, {years} years, single parent",
         "Shift manager at a chain restaurant, {years} years, dreams of opening own place",
     ],
     "temperaments": ["cheerful but underpaid", "quiet and enduring", "restless and looking for a way out"],
     "debt_range": (0.15, 0.45), "ambition_range": (0.2, 0.5)},

    {"role": "healthcare_worker", "count": 30, "location": "hospital_system", "home": "residential_suburb",
     "backgrounds": [
         "ER nurse with {years} years experience, still haunted by COVID shifts",
         "Physician assistant at a community clinic, {years} years, student loans",
         "Hospital administrator juggling budgets and bed counts, {years} years",
     ],
     "temperaments": ["compassionate but exhausted", "methodical under pressure", "frustrated by system failures"],
     "debt_range": (0.05, 0.3), "ambition_range": (0.2, 0.5)},

    {"role": "gig_worker", "count": 40, "location": "gig_economy", "home": "urban_downtown",
     "backgrounds": [
         "Rideshare driver, {years} years, no benefits, car payment due",
         "Food delivery courier, {years} years, saving for trade school",
         "Freelance graphic designer, {years} years, feast-or-famine income",
     ],
     "temperaments": ["hustling and anxious", "independent but precarious", "creative but broke"],
     "debt_range": (0.2, 0.5), "ambition_range": (0.3, 0.6)},

    {"role": "student", "count": 30, "location": "university", "home": "urban_downtown",
     "backgrounds": [
         "Undergrad in computer science, {years} years from graduation, $40k in loans",
         "MBA student, career-switching from teaching, {years} semesters left",
         "Graduate researcher in public health, {years} years, worried about job market",
     ],
     "temperaments": ["idealistic but anxious about future", "driven by debt clock", "politically engaged"],
     "debt_range": (0.15, 0.4), "ambition_range": (0.4, 0.8)},

    {"role": "retiree", "count": 25, "location": "residential_suburb", "home": "residential_suburb",
     "backgrounds": [
         "Retired teacher, {years} years of pension, watches savings nervously",
         "Retired auto worker, {years} years on fixed income, Medicare dependent",
         "Retired small business owner, sold shop {years} years ago, lives on investments",
     ],
     "temperaments": ["cautious with every dollar", "worried about outliving savings", "nostalgic for simpler times"],
     "debt_range": (0.05, 0.2), "ambition_range": (0.05, 0.2)},

    {"role": "small_business_owner", "count": 25, "location": "main_street", "home": "residential_suburb",
     "backgrounds": [
         "Owns a family restaurant, {years} years, barely survived COVID",
         "Runs a hardware store, {years} years, competing with MegaMart",
         "Hair salon owner, {years} years, three employees depend on her",
     ],
     "temperaments": ["scrappy and worried", "proud but stretched thin", "community anchor, personally exhausted"],
     "debt_range": (0.15, 0.45), "ambition_range": (0.3, 0.6)},
]


# ═══════════════════════════════════════════════════════════════════════════
# BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_real_economy(seed: int = 42) -> tuple[World, dict, OrganizationalFabric]:
    """Build a real-economy simulation with named companies and leaders.

    Returns (world, agent_meta, organizational_fabric).
    """
    rng = random.Random(seed)
    world = World()
    agent_meta: dict[str, dict] = {}

    # Add locations
    for loc in REAL_LOCATIONS:
        world.add_location(loc)

    fabric = OrganizationalFabric()
    company_agents: dict[str, list[str]] = {}  # company_name → [agent_ids]

    # ── Build company/institution LLM agents ──
    all_orgs = COMPANIES + GOVERNMENT_INSTITUTIONS
    for org in all_orgs:
        company_agents[org.name] = []

        for person in org.key_personnel:
            agent_id = person["name"].lower().replace(" ", "_").replace(".", "")
            if agent_id in world.agents:
                agent_id += f"_{org.name[:4].lower()}"

            personality = Personality(
                name=person["name"],
                background=f'{person["title"]} at {org.name}. {person["background"]}',
                temperament=person["temperament"],
                attachment_style=person.get("attachment_style", ""),
                coping_style=person.get("coping_style", ""),
                threat_lens=person.get("threat_lens", ""),
                core_need=person.get("core_need", ""),
                conflict_style=person.get("conflict_style", ""),
                self_story=person.get("self_story", ""),
            )

            role_map = {
                "tech_chips": "manager", "tech_consumer": "manager", "tech_cloud": "manager",
                "tech_ecommerce": "manager", "retail": "manager", "retail_wholesale": "manager",
                "finance": "office_professional", "energy": "manager",
                "pharma": "manager", "central_bank": "government_worker",
                "legislature": "government_worker", "fiscal_policy": "government_worker",
                "public_health": "healthcare",
            }
            social_role = role_map.get(org.sector, "office_professional")

            # Schedule: at HQ during work hours
            schedule = {}
            for h in range(0, 7):
                schedule[h] = "residential_suburb"
            for h in range(7, 9):
                schedule[h] = "residential_suburb"
            for h in range(9, 18):
                schedule[h] = org.hq_location
            for h in range(18, 22):
                schedule[h] = "residential_suburb"
            for h in range(22, 24):
                schedule[h] = "residential_suburb"

            agent = WorldAgent(
                agent_id=agent_id,
                personality=personality,
                schedule=schedule,
                social_role=social_role,
                debt_pressure=rng.uniform(0.02, 0.12),
                ambition=rng.uniform(0.5, 0.9),
            )

            world.add_agent(agent)
            company_agents[org.name].append(agent_id)
            agent_meta[agent_id] = {
                "org": org.name,
                "title": person["title"],
                "sector": org.sector,
                "role": social_role,
                "is_llm_agent": True,
                "revenue_influence": org.revenue_influence,
            }

        # Add non-LLM employees
        for i in range(org.employee_count):
            emp_id = f"{org.name.lower()}_emp_{i}"
            bg_templates = [
                f"Employee at {org.name}, works in operations",
                f"Mid-level staff at {org.name}, {rng.randint(2,15)} years",
                f"Junior analyst at {org.name}, recently hired",
            ]
            personality = Personality(
                name=f"{rng.choice(['Alex','Sam','Jordan','Casey','Morgan','Riley','Taylor','Drew','Quinn','Avery'])}_{org.name[:3]}_{i}",
                background=rng.choice(bg_templates),
                temperament=rng.choice(["diligent and quiet", "ambitious mid-career", "steady team player"]),
            )
            schedule = {}
            for h in range(9, 18):
                schedule[h] = org.hq_location
            for h in list(range(0, 9)) + list(range(18, 24)):
                schedule[h] = rng.choice(["residential_suburb", "urban_downtown"])

            agent = WorldAgent(
                agent_id=emp_id,
                personality=personality,
                schedule=schedule,
                social_role="office_worker" if org.sector != "retail" else "retail_worker",
                debt_pressure=rng.uniform(0.05, 0.25),
            )
            world.add_agent(agent)
            company_agents[org.name].append(emp_id)

            # Employee → managed by company leaders
            for leader_id in company_agents[org.name][:len(org.key_personnel)]:
                fabric.add(OrgLink(leader_id, emp_id, "employs", strength=0.7))

    # ── Build population segment agents ──
    for seg in POPULATION_SEGMENTS:
        for i in range(seg["count"]):
            name_pool = [
                "James","Maria","Robert","Elena","Daniel","Sofia","William","Anna",
                "Michael","Sarah","David","Laura","Thomas","Emma","Frank","Nina",
                "Brian","Chloe","Derek","Fiona","Grant","Holly","Ian","Julia",
                "Keith","Lisa","Nathan","Olivia","Peter","Quinn","Kevin","Mika",
                "Carlos","Rosa","Ahmed","Priya","Yuki","Marcus","Lena","Andre",
            ]
            name = f"{rng.choice(name_pool)}_{seg['role'][:3]}_{i}"
            agent_id = name.lower().replace(" ", "_")

            years = rng.randint(1, 20)
            bg = rng.choice(seg["backgrounds"]).format(years=years)
            temperament = rng.choice(seg["temperaments"])

            personality = Personality(name=name, background=bg, temperament=temperament)

            schedule = {}
            for h in range(0, 7):
                schedule[h] = seg["home"]
            for h in range(7, 9):
                schedule[h] = seg["home"]
            for h in range(9, 17):
                schedule[h] = seg["location"]
            for h in range(17, 22):
                schedule[h] = rng.choice([seg["home"], "urban_downtown", "main_street"])
            for h in range(22, 24):
                schedule[h] = seg["home"]

            agent = WorldAgent(
                agent_id=agent_id,
                personality=personality,
                schedule=schedule,
                social_role=seg["role"],
                debt_pressure=rng.uniform(*seg["debt_range"]),
                ambition=rng.uniform(*seg["ambition_range"]),
            )
            assign_human_profile(personality, seg["role"], rng)
            world.add_agent(agent)

            agent_meta[agent_id] = {
                "role": seg["role"],
                "segment": seg["role"],
                "is_llm_agent": False,
            }

            # Population → supplied by retail companies
            for retail_co in ["RetailGiant", "CostPlus", "MegaMart"]:
                if retail_co in company_agents and rng.random() < 0.3:
                    for leader_id in company_agents[retail_co][:2]:
                        fabric.add(OrgLink(leader_id, agent_id, "supplies", strength=0.3))

            # Population → employed by companies (factory/office workers)
            if seg["role"] in ("factory_worker", "tech_worker", "retail_worker"):
                employer_cos = [c for c in COMPANIES if c.employee_count > 20]
                if employer_cos:
                    employer = rng.choice(employer_cos)
                    if employer.name in company_agents:
                        for leader_id in company_agents[employer.name][:2]:
                            fabric.add(OrgLink(leader_id, agent_id, "employs", strength=0.5))

    # ── Supply chain links between companies ──
    for org in all_orgs:
        for target_name in org.supply_chain_to:
            if target_name in company_agents and org.name in company_agents:
                for from_id in company_agents[org.name][:2]:
                    for to_id in company_agents[target_name][:2]:
                        fabric.add(OrgLink(from_id, to_id, "supplies", strength=0.6))

    # ── Regulatory links ──
    for org in all_orgs:
        for reg_name in org.regulated_by:
            if reg_name in company_agents and org.name in company_agents:
                for reg_id in company_agents[reg_name][:2]:
                    for co_id in company_agents[org.name][:2]:
                        fabric.add(OrgLink(reg_id, co_id, "regulates", strength=0.5))

    # ── Government → population ──
    gov_ids = []
    for gov_org in GOVERNMENT_INSTITUTIONS:
        if gov_org.name in company_agents:
            gov_ids.extend(company_agents[gov_org.name][:3])
    all_pop_ids = [aid for aid, m in agent_meta.items() if not m.get("is_llm_agent")]
    for gid in gov_ids:
        constituents = rng.sample(all_pop_ids, min(40, len(all_pop_ids)))
        for cid in constituents:
            fabric.add(OrgLink(gid, cid, "governs", strength=0.3))

    # Seed basic relationships
    _seed_basic_relationships(world, company_agents, rng)

    return world, agent_meta, fabric


def _seed_basic_relationships(world: World, company_agents: dict[str, list[str]], rng: random.Random):
    """Seed initial relationships within companies and between connected agents."""
    for co_name, agent_ids in company_agents.items():
        for i, aid_a in enumerate(agent_ids[:10]):
            for aid_b in agent_ids[i + 1:10]:
                if rng.random() < 0.5:
                    rel = world.relationships.get_or_create(aid_a, aid_b)
                    rel.trust = rng.uniform(0.1, 0.5)
                    rel.warmth = rng.uniform(0.0, 0.4)
                    rel.familiarity = rng.randint(5, 50)

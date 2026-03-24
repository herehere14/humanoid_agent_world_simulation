"""Research-calibrated economic parameters.

All numbers sourced from BLS, FRED, NBER, IMF, and peer-reviewed studies.
See data/economic_simulation_research.md for full citations.

This module replaces hardcoded guesses with real-world numbers.
"""

# ═══════════════════════════════════════════════════════════════
# HOUSEHOLD PARAMETERS (BLS Consumer Expenditure Survey 2023)
# ═══════════════════════════════════════════════════════════════

# Essential spending as fraction of income
ESSENTIAL_SPENDING_NORMAL = 0.73  # housing 33% + transport 17% + food 13% + health 8%
ESSENTIAL_SPENDING_CRISIS = 0.85  # discretionary collapses, essentials dominate

# Savings (Fed SHED Survey 2024, BEA)
SAVINGS_RATE_NORMAL = 0.06  # 5-8% of income saved normally
SAVINGS_3_MONTHS_COVERAGE = 0.55  # 55% of adults have 3 months saved
PAYCHECK_TO_PAYCHECK_PCT = 0.30  # 24-34% live paycheck-to-paycheck
# Savings drawdown: 80%+ of middle-income drew down within 3 months (2022 data)
SAVINGS_DRAWDOWN_MONTHS = 3  # savings last ~3 months under crisis pressure

# Emergency expense response sequence (Fed 2024)
# When facing $1000 emergency:
RESPONSE_USE_SAVINGS = 0.30       # 30% use savings first
RESPONSE_CURRENT_INCOME = 0.17   # 17% absorb from cashflow
RESPONSE_CREDIT_CARD = 0.17     # 17% put on credit
RESPONSE_FAMILY_FRIENDS = 0.12  # 12% borrow from network
RESPONSE_LOAN = 0.03            # 3% take a loan

# ═══════════════════════════════════════════════════════════════
# CORPORATE DECISION TIMELINES (BLS Mass Layoffs, NBER)
# ═══════════════════════════════════════════════════════════════

# Timeline from revenue drop to action (in TICKS, 1 tick = 1 hour)
# Normal recession (organic):
CORP_DISCRETIONARY_FREEZE_TICKS = 24 * 7    # Week 1-2: travel/perks cut
CORP_HIRING_FREEZE_TICKS = 24 * 14          # Week 2-4: stop hiring
CORP_HOUR_CUTS_TICKS = 24 * 60             # Month 2-3: reduce hours
CORP_LAYOFF_TICKS = 24 * 120               # Month 3-6: actual layoffs
CORP_CAPEX_CUT_TICKS = 24 * 180            # Month 3-9: investment cuts

# Shock recession (COVID-like): compressed to near-zero for hospitality/retail
CORP_SHOCK_MULTIPLIER = 0.1  # 10% of normal timeline during acute shocks

# Who gets cut (2026 survey): high-salary 48%, no-AI-skills 46%, recent hires 42%

# ═══════════════════════════════════════════════════════════════
# LABOR MARKET (BLS, St. Louis Fed, CPS)
# ═══════════════════════════════════════════════════════════════

JOB_FINDING_RATE_NORMAL = 0.38    # 38-40% of unemployed find work each month
JOB_FINDING_RATE_CRISIS = 0.18    # 18% during 2009 trough
MEDIAN_UNEMPLOYMENT_WEEKS_NORMAL = 8.6
MEDIAN_UNEMPLOYMENT_WEEKS_CRISIS = 25.2  # tripled in 2009-2010

# Re-employment per tick (convert monthly to hourly ticks)
# Normal: 38% per month = ~1.7% per day = ~0.07% per tick
REEMPLOYMENT_RATE_NORMAL = 0.0017  # per tick (hourly)
REEMPLOYMENT_RATE_CRISIS = 0.0008  # per tick during crisis

# Gig economy absorption: +1% county unemployment = +21.8 more gig workers
GIG_ABSORPTION_RATE = 0.02  # fraction of newly unemployed who find gig work

# Which sectors are at risk during recession (BLS data):
# Only 15-20% of total workforce actually loses jobs in a severe recession
SECTOR_RISK_RATES = {
    "retail_worker": 0.25,         # 25% at risk (leisure/hospitality: 47% in COVID)
    "gig_worker": 0.20,            # 20% lose primary gig
    "factory_worker": 0.18,        # 18% (manufacturing)
    "small_business_owner": 0.15,  # 15% close temporarily
    "tech_worker": 0.12,           # 12% (professional services)
    "office_worker": 0.10,         # 10%
    "healthcare_worker": 0.03,     # 3% (recession-proof)
    "government_worker": 0.02,     # 2% (almost immune)
    "teacher": 0.02,               # 2% (protected by contracts)
    "retiree": 0.0,                # not in labor force
    "student": 0.0,                # not in labor force (but employment declines)
}

# Essential sectors that keep running (BLS):
# ~60-65% of economy is recession-proof
ESSENTIAL_SECTORS = {"healthcare_worker", "government_worker", "teacher"}
RECESSION_PROOF_EMPLOYMENT_FLOOR = 0.60  # 60% of jobs survive any recession

# ═══════════════════════════════════════════════════════════════
# GOVERNMENT RESPONSE (Historical timelines)
# ═══════════════════════════════════════════════════════════════

# Fed can act in 4-43 days from crisis onset
FED_RESPONSE_TICKS = 24 * 10  # ~10 days average

# Congress: CARES Act took 67 days from first COVID case, 16 days from WHO declaration
CONGRESS_STIMULUS_TICKS = 24 * 16  # ~16 days for emergency legislation

# Stimulus checks reach households 17-21 days after bill signing (direct deposit)
STIMULUS_DELIVERY_TICKS = 24 * 18

# UI replacement rate (DOL)
UI_REPLACEMENT_RATE = 0.43  # 43% of prior wages on average (21-55% by state)
UI_FEDERAL_SUPPLEMENT = 600  # $600/week during CARES Act (later $300)

# Fiscal multiplier (IMF, NBER)
FISCAL_MULTIPLIER_NORMAL = 0.85  # 0.7-0.9 in normal times
FISCAL_MULTIPLIER_RECESSION = 1.25  # 1.0-1.5 during recession (more effective)

# ═══════════════════════════════════════════════════════════════
# PRICE TRANSMISSION (Dallas Fed, BLS CPI)
# ═══════════════════════════════════════════════════════════════

# Oil → gas: 12% same day, 50% by 1 month, ~100% by 3 months
OIL_TO_GAS_DAY1 = 0.12
OIL_TO_GAS_MONTH1 = 0.50
OIL_TO_GAS_MONTH3 = 1.00

# Oil → food: builds over 8 quarters (2 years!)
OIL_TO_FOOD_MONTH1 = 0.05
OIL_TO_FOOD_MONTH6 = 0.25
OIL_TO_FOOD_YEAR2 = 0.60

# Wage response to prices: 6-18 months normally, 3-6 months in high inflation
WAGE_LAG_MONTHS_NORMAL = 12
WAGE_LAG_MONTHS_HIGH_INFLATION = 4

# ═══════════════════════════════════════════════════════════════
# FINANCIAL MARKETS (CBOE, IMF, NBER)
# ═══════════════════════════════════════════════════════════════

VIX_BASELINE = 15.0  # normal 12-20
VIX_ELEVATED = 30.0  # concern
VIX_CRISIS = 55.0    # panic
VIX_EXTREME = 80.0   # all-time crisis peaks (2008: 80, 2020: 82)

# CCI → spending coefficient (Chicago Fed)
CCI_SPENDING_COEFFICIENT = 0.0038  # +1pt CCI → +0.38pt consumption growth

# Credit tightening propagation timeline:
# Interbank: hours. Corporate: days-weeks. Consumer: months.
CREDIT_TIGHTENING_INTERBANK_TICKS = 4    # hours
CREDIT_TIGHTENING_CORPORATE_TICKS = 24 * 14  # 2 weeks
CREDIT_TIGHTENING_CONSUMER_TICKS = 24 * 60   # 2 months

# ═══════════════════════════════════════════════════════════════
# SECTOR INTERDEPENDENCIES (BEA Input-Output, EPI)
# ═══════════════════════════════════════════════════════════════

# Employment multipliers: each job in sector X supports Y total jobs
EMPLOYMENT_MULTIPLIER = {
    "manufacturing": 2.91,     # 1 manufacturing job → 2.91 total (EPI)
    "construction": 1.97,
    "retail": 0.88,            # below 1.0 because low wages
    "healthcare": 1.50,
    "professional_services": 1.86,
    "finance": 1.50,
    "government": 1.30,
    "hospitality": 0.78,      # lowest multiplier
}

# Leading vs lagging indicators:
# LEADING: housing starts (6-18 month lead), manufacturing PMI, stock market
# LAGGING: unemployment rate (peaks 6-12 months AFTER recession ends), CPI

# ═══════════════════════════════════════════════════════════════
# SOCIAL/BEHAVIORAL (Lancet, PMC, Political Science)
# ═══════════════════════════════════════════════════════════════

# Unemployment → suicide: +1pp unemployment → +1.0-1.6% suicide rate increase
UNEMPLOYMENT_SUICIDE_COEFFICIENT = 1.3  # per percentage point

# Rally-around-flag duration
RALLY_DURATION_ECONOMIC_WEEKS = 3    # 2-4 weeks for economic crises
RALLY_DURATION_SECURITY_MONTHS = 6   # 2-14 months for security crises

# Mutual aid formation: 1-2 weeks after crisis onset
MUTUAL_AID_FORMATION_TICKS = 24 * 10  # ~10 days

# ═══════════════════════════════════════════════════════════════
# TIME SCALING
# ═══════════════════════════════════════════════════════════════

# Our simulation uses 1 tick = 1 hour
# Real economic decisions happen on these timescales:
# - Financial markets: minutes to hours (1-4 ticks)
# - Consumer spending: days to weeks (24-168 ticks)
# - Corporate decisions: weeks to months (168-4320 ticks)
# - Government policy: days to months (96-4320 ticks)
# - Full recession cycle: 6-18 months (4320-12960 ticks)

# For a 14-day simulation (336 ticks), we need to COMPRESS these timelines
# to see meaningful dynamics. The compression factor:
SIM_DAYS = 14
REAL_RECESSION_MONTHS = 12
COMPRESSION_FACTOR = (REAL_RECESSION_MONTHS * 30) / SIM_DAYS  # ~25.7x

# This means 1 sim-day ≈ 25 real-days
# So corporate decisions that take 3 months in reality should take ~3.5 sim-days
COMPRESSED_CORP_LAYOFF_TICKS = int(CORP_LAYOFF_TICKS / COMPRESSION_FACTOR)  # ~112 ticks ≈ 4.7 days
COMPRESSED_CORP_HOUR_CUTS_TICKS = int(CORP_HOUR_CUTS_TICKS / COMPRESSION_FACTOR)  # ~56 ticks ≈ 2.3 days
COMPRESSED_FED_RESPONSE_TICKS = int(FED_RESPONSE_TICKS / COMPRESSION_FACTOR)  # ~9 ticks
COMPRESSED_STIMULUS_DELIVERY_TICKS = int(STIMULUS_DELIVERY_TICKS / COMPRESSION_FACTOR)  # ~17 ticks

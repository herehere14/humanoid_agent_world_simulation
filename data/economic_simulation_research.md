# How the Real Economy Actually Works: Comprehensive Research for Economic Simulation Calibration

Research compiled: March 2026
Sources: BLS, Federal Reserve (FRED), BEA, IMF, NBER, WTO, EPI, peer-reviewed studies

---

## 1. HOUSEHOLD SPENDING BEHAVIOR DURING CRISES

### Essential vs Discretionary Spending Breakdown (Normal Times)

**BLS Consumer Expenditure Survey 2023:**
- Housing: 32.9% of total expenditures
- Transportation: 17.0%
- Food: 12.9%
- Personal insurance/pensions: 12.0%
- Healthcare: 8.4%
- Top 5 categories = 83.2 cents of every dollar spent
- Total average annual expenditure: $77,280

**During crisis:** Housing share rose from 32.8% (2019) to 34.9% (2020) as discretionary collapsed. Lower-income households spent 40% on housing alone; renters in the bottom third spent nearly 50% of income on housing.

**Essential spending (housing + food + healthcare + transport) = ~70-75% of income for median households in normal times, rising to 85%+ during crises as discretionary collapses.**

### Speed of Spending Cuts

**2008 Great Recession:**
- Gradual: Monthly PCE changes ranged -1.5% to +1.2%
- By Q4 2008, consumers cut spending by over $200 billion from prior year
- The full contraction played out over 18 months (Dec 2007 - Jun 2009)
- Food spending fell from ~$720B to $690B over 2008-2009

**2020 COVID:**
- Extremely rapid: PCE dropped 6.9% annualized in March, 11.4% in April
- Bounced: +8.3% in May, +6.0% in June
- Recession lasted only 2 months (shortest on record)

**Simulation parameter:** Organic recession = spending cuts build over 2-6 months. Shock event = spending collapses within 2-4 weeks.

### Savings Rate and Buffer

**Personal savings rate (BEA):** 4.9% as of early 2025 (historically 6-8% normal, spiked to 33.8% in April 2020 during lockdowns)

**Emergency savings coverage:**
- 55% of adults have 3 months of expenses saved (2024 Fed survey)
- 30% cannot cover 3 months by any means
- By income: only 24% of those earning <$25K have 3 months saved; 75% of $100K+ households do

### Paycheck-to-Paycheck Living

- 24% of US households live paycheck-to-paycheck (Bank of America Institute, 2025)
- 34% of workers self-report feeling paycheck-to-paycheck (Bankrate)
- These percentages rise 5-10 points during recessions

### Sequence: Savings → Credit → Family → Distress

When facing a $1,000 emergency expense (2024 Fed data):
1. **Savings:** 30% would use savings
2. **Current income/cashflow:** 17% would absorb from regular income
3. **Credit card:** 17% would finance on credit
4. **Family/friends:** 12% would borrow from social network
5. **Personal loan:** 3% would take out a loan

During extended recession, the sequence shifts:
- 80%+ of middle-income households drew down savings within 3 months (2022 data)
- Credit card debt surges as savings deplete
- Payday/installment lenders become last resort — usage spiked post-2008 among credit-constrained households

---

## 2. CORPORATE DECISION CASCADES

### The Cost-Cutting Sequence (When Revenue Drops ~15%)

Typical timeline from revenue decline to full layoffs:

| Stage | Timing from Revenue Drop | Action |
|-------|--------------------------|--------|
| 1. Discretionary freeze | Week 1-2 | Travel budgets, conferences, perks eliminated |
| 2. Hiring freeze | Week 2-4 | All non-critical hiring paused |
| 3. Promotion/raise freeze | Month 1-2 | Compensation adjustments halted |
| 4. Hours reduction | Month 2-3 | Reduced schedules, furlough days |
| 5. Wage/bonus cuts | Month 2-4 | 29% of revenue-hit firms cut pay in 2020 (15% cut wages, 19% cut bonuses) |
| 6. Layoffs | Month 3-6 | Non-revenue-generating roles cut first |
| 7. Capex cuts | Month 3-9 | Capital expenditure projects deferred/cancelled |

**Key data point:** Hiring freezes are ANTICIPATORY (signal concern). Layoffs are REACTIVE (confirm losses). The gap is typically 4-12 weeks.

**Who gets cut first (2026 survey):**
- High-salary employees: 48% most at risk
- Employees lacking AI/tech skills: 46%
- Recently hired workers: 42%
- Entry-level employees: 41%

### Lag Between Revenue Drop and Layoffs

**2008:** The crisis accelerated dramatically after September 2008 (Lehman). Extended mass layoff events rose 46% year-over-year; demand-related layoffs nearly doubled (248,056 → 476,302 separations). The lag from financial crisis onset (mid-2007) to mass layoffs (late 2008) was ~12-15 months for the gradual build, then compressed to weeks during the acute phase.

**COVID 2020:** Near-zero lag. Layoff rate hit record 2.4%. March-April 2020 saw immediate mass layoffs — revenue disappeared overnight for hospitality/retail.

**Average 700,000 jobs lost per month from Oct 2008 to Apr 2009.**

### Hours Cut vs Layoffs

- In 2020, active employment dropped 11% but paid employment dropped 21%, meaning ~10% of affected workers were on temporary furlough rather than permanently laid off
- Germany's Kurzarbeit (short-time work) saved 400,000 jobs with 1.4 million workers participating at peak (May 2009)
- US work-sharing saved only ~165,000 jobs in 2009 due to minimal programs
- Approximately 57% of companies pass costs to consumers, 55% reduce margins, 46% switch suppliers before cutting headcount

### Supply Chain Cost Propagation

- Supplier cost increases reach retail prices in **weeks to months** depending on contract structures
- Companies with index-linked pricing contracts adjust automatically
- During 2021-2022 inflation: 80% of companies reported passing rising costs to consumers (Fed survey)
- Markups at each supply chain stage fluctuate, but total markup (raw materials → retail) remains surprisingly stable over time

---

## 3. FINANCIAL MARKET MECHANICS

### Consumer Spending → Stock Prices

- Stock prices LEAD consumer confidence (Granger causality runs from stocks → confidence, not reverse)
- The mechanism: stock gains → wealth effect + confidence channel → spending increase
- A 1 percentage point increase in year-over-year confidence change → 0.38 percentage point increase in consumption spending growth
- Consumer spending data CAN predict earnings surprises → post-announcement stock returns

**PCE represents ~70% of US GDP**, so sustained spending decline = corporate earnings decline = stock price adjustment. But markets typically price this in 1-3 months before the spending data confirms it.

### VIX Mechanics

**Calculation:** Weighted average of implied volatilities from S&P 500 options (puts and calls) with 23-37 days to expiration, annualized.

**Historical levels:**
- Normal/calm: 12-15
- Average: ~20
- Elevated uncertainty: 20-30
- Crisis: 30-50
- Extreme panic: 50-80+

**Crisis peaks:**
- 2008 GFC: intraday high 89.53 (Oct 24, 2008), daily close peak 80.86 (Nov 20, 2008)
- 2020 COVID: peak 82.69 (March 2020), monthly average 57.74
- Since 1990, only 14 episodes above 32.7 (Bank of America)
- ALL 69 highest VIX closes occurred during only two periods: Oct 2008-Mar 2009 and Mar-Apr 2020

**What drives VIX:** Demand for protective put options. During stress, sellers demand higher premiums → inflated implied volatility. VIX is inversely correlated with S&P 500. VIX does NOT predict direction, only expected magnitude of moves.

### Credit Tightening Propagation

**2007-2010 timeline:**
- Summer 2007: Subprime market collapse begins (Crisis I)
- Aug 2008: CDS spreads on financial institutions begin rising (Crisis II)
- Sep 2008: Lehman bankruptcy — massive credit freeze
- End 2008: Peak tightening of lending standards
- 2008-2010: Average loan spread increased ~1 percentage point
- Standards remained tight through 2010+ even during recovery

**Propagation order:** Interbank lending freezes (hours) → Corporate credit tightens (days-weeks) → Small business credit restricted (weeks-months) → Consumer credit tightened (months) → Housing credit frozen (months)

### Fiscal Multiplier

**Empirical estimates (central range):** 0.50 to 0.90 in normal times

**During recessions:**
- Auerbach & Gorodnichenko (2012): ~1.4x in first year of recession
- Baum et al. (2012): ~0.9x overall downturn multiplier
- At zero lower bound (ZLB): evidence for >1.0x is scarce but theoretically expected

**Full literature range:** -3.8 to +3.8 (extreme estimates in both directions)

**Simulation parameter:** Use 0.7-0.9 for normal times, 1.0-1.5 for deep recession with monetary accommodation.

---

## 4. LABOR MARKET DYNAMICS

### Employment → Unemployment Flow

**Job Finding Rate (monthly probability an unemployed worker finds a job):**
- Normal times: ~38-40%
- Great Recession trough (Q3 2009): 18%
- COVID trough: 29%
- Recovery to pre-recession levels takes 5-10 years

**Unemployment duration (median weeks):**
- Pre-recession (Nov 2007): 8.6 weeks
- Peak recession (Jun 2010): 25.2 weeks (nearly 6 months)
- Long-term unemployed (27+ weeks): quadrupled from 1.6M to 6.8M (Apr 2010), representing 45.5% of total unemployment

**Exit hazard from unemployment:**
- Mid-2007: ~40% monthly exit rate
- 2009-2010: ~25% monthly exit rate

### Sector Vulnerability in Recession

**2008 Great Recession losses:**
- Manufacturing: -18% (-2.5M jobs)
- Construction: -20% (still hadn't recovered by 2018)
- Financial activities: significant losses
- Healthcare: GAINED 861,000 jobs (counter-cyclical)

**2020 COVID losses (Feb-Apr 2020):**
- Leisure & hospitality: -48.6% (-8.2M jobs in March-April)
  - Food service alone: -5.5M
- Retail: -2.1M in April alone
- Leisure/hospitality = 40% of ALL job losses
- Total April 2020: -20.5M jobs

**Not all workers at risk:** In a typical recession, ~15-20% of workers in vulnerable sectors face actual job loss risk. Essential services, healthcare, government, and utilities are largely insulated.

### Re-employment and Recovery

- Oct 2008-Apr 2009: average 700,000 jobs lost per month
- Employers didn't begin net job additions until 2010
- Took until mid-2014 to recover the 8.7M jobs lost (4+ years)
- Re-employed workers often earn significantly less than before
- Job-to-job flow rate hit lowest level in 4 decades during 2008-2009

### Gig Economy as Shock Absorber

- 1% increase in county unemployment → 21.8 increase in volume of residents working on gig platforms
- Uber's arrival in a city reduced unemployment rate by 0.2-0.5 percentage points
- Displaced workers were 79% more likely to enter ridesharing (4 years after Uber entry) vs 29% for other self-employment
- Laid-off workers with gig access were less likely to file for UI benefits
- Gig work serves as BRIDGE employment during job search, not permanent replacement

---

## 5. GOVERNMENT RESPONSE TIMING

### Crisis → Response Timeline

**2008 Financial Crisis:**

| Date | Event | Days from Lehman |
|------|-------|------------------|
| Sep 15, 2008 | Lehman Brothers bankruptcy | Day 0 |
| Sep 19, 2008 | Treasury announces $50B money market guarantee | Day 4 |
| Oct 3, 2008 | TARP signed ($700B) | Day 18 |
| Oct-Dec 2008 | Fed cuts rate to 0-0.25% | Days 30-90 |
| Feb 17, 2009 | ARRA signed ($787B, 5.5% of GDP) | Day 155 |

**BUT:** The crisis actually started building in mid-2007. The Feb 2008 stimulus ($152B tax rebates) came ~6 months into the subprime crisis. Full response took 18+ months.

**2020 COVID Crisis (dramatically faster):**

| Date | Event | Days from First Case |
|------|-------|---------------------|
| Jan 20, 2020 | First US COVID case | Day 0 |
| Mar 3, 2020 | Fed cuts rate 50bp (to 1-1.25%) | Day 43 |
| Mar 6, 2020 | First COVID spending bill signed | Day 46 |
| Mar 15, 2020 | Fed cuts to 0-0.25% | Day 55 |
| Mar 18, 2020 | Families First Act signed | Day 58 |
| Mar 27, 2020 | CARES Act signed ($2.2T) | Day 67 |
| Apr 13, 2020 | First direct deposit stimulus checks arrive | Day 84 |
| Apr 17, 2020 | 90M of 150M eligible had received payment | Day 88 |
| Apr 24, 2020 | PPP Enhancement Act signed | Day 95 |
| Dec 31, 2020 | All first-round checks distributed | Day 346 |

**Key parameter:** CARES Act signed → first checks in bank accounts = ~17 days. But paper checks took months.

### Stimulus Check Distribution Speed

**2008:** Signed Feb 13 → direct deposits mid-May → paper checks May-July (3-5 months)
**2020:** Signed Mar 27 → direct deposits Apr 13 → 90M received by Apr 17 (17-21 days for direct deposit recipients)

### Unemployment Insurance Effectiveness

**Average replacement rate:** 43% of prior wages nationally (2023)

**State variation:**
- Lowest: ~21% (DC)
- Highest: ~55% (Hawaii)
- Massachusetts: ~45%
- Most states: 35-45% range

**During COVID (with $600/week federal supplement):** Median replacement rate exceeded 100% of prior wages for low-wage workers — this was by design to encourage compliance with stay-at-home orders.

---

## 6. SECTOR INTERDEPENDENCIES

### Manufacturing Employment Multiplier

**EPI estimates:**
- 1 manufacturing job supports 1.4 additional jobs elsewhere in the economy (conservative)
- Alternative estimate: 1 new manufacturing job creates 7.4 new jobs in other industries (including induced effects)
- Comparison: 1 retail job creates only 1.2 new jobs

**Supplier jobs per $1M demand change:**
- Durable manufacturing: 16.5 indirect jobs lost per $1M demand drop
- Retail trade: 10.6 indirect jobs lost per $1M demand drop

**Full employment multiplier (direct + indirect + induced, per 100 direct jobs):**
- Manufacturing: 291 total jobs
- Personal/business services: 154 total jobs
- Health services: 117 total jobs
- Retail trade: 88 total jobs

### Leading vs Lagging Sectors

**Leading indicators (turn down first):**
- Building permits / housing starts (6-18 months lead)
- Manufacturing PMI
- Yield curve (6-24 months lead — inverted curve preceded every US recession since 1970s)
- Stock market
- Consumer expectations

**Coincident indicators:**
- Industrial production
- Personal income
- Retail sales
- Nonfarm payrolls

**Lagging indicators (turn down last):**
- Unemployment rate (peaks 6-12 months AFTER recession ends)
- Corporate profits (reported with quarter lag)
- Bank lending rates
- CPI (inflation slows late)

### Sector Cascade: If Manufacturing Drops 10%

Approximate propagation:
- Transportation/warehousing: -3 to -5% (ships fewer goods)
- Wholesale trade: -2 to -4%
- Business services: -1 to -3%
- Retail (durable goods): -2 to -4%
- Mining/raw materials: -3 to -6% (reduced input demand)

---

## 7. CONSUMER CONFIDENCE MECHANICS

### What Moves CCI

**Conference Board CCI components:**
1. Present Situation Index (current business/employment conditions)
2. Expectations Index (6-month outlook for business, employment, income)

**Key drivers:**
- **Unemployment:** Negative correlation — higher unemployment → lower confidence. Historically weak association until post-pandemic period when it strengthened
- **Inflation expectations:** Consistently strong negative association with confidence, especially during 2022-2023 high-inflation period
- **Stock market:** Positive correlation — rising markets → wealth effect → higher confidence
- **Gas prices:** Immediate psychological impact on consumer mood

### CCI → Spending Relationship

- **1 percentage point increase in YoY confidence change → 0.38 percentage point increase in consumption spending growth** (Chicago Fed research)
- CCI is a leading indicator — drops in confidence precede spending declines by 1-3 months
- However, the predictive power is modest — confidence explains roughly 15-25% of spending variation

### CCI → GDP

- Consumer spending = ~70% of GDP
- CCI predicts direction but not magnitude well
- Sharp CCI drops (>20 points) have historically preceded recessions within 6-12 months
- CCI recovery typically leads GDP recovery by 1-2 quarters

---

## 8. PRICE TRANSMISSION

### Oil → Gasoline (Days)

- **Same day:** 12% of crude oil price increase passes through to retail gasoline
- **20 working days (1 month):** ~50% pass-through
- **Long-run:** ~55% pass-through
- Alternative measurement: 13% after 1 week, 37% after 3 months, 50% long-run
- **Asymmetry:** Prices rise faster than they fall ("rockets and feathers")

### Oil → Food/Consumer Prices (Weeks to Months)

- After 1 week: 0.5% pass-through to consumer prices
- After 3 months: 1.5% pass-through
- Long-run: 4.2% pass-through
- Food specifically: 10% oil price increase → 0.3% food CPI increase, building over ~8 quarters (2 years)

### Wage Response to Prices (Months to Years)

- Wages use a 3-year weighted lag of CPI inflation as proxy for expectations
- Pre-pandemic: wage persistence coefficient 0.55 (slow adjustment)
- Post-pandemic: wage persistence dropped to 0.34 with near 1:1 inflation pass-through
- Wage Phillips curve slope: -0.45 (1991-2008) → weakened to -0.14 (2009-2015)
- **Key implication:** Wages lag prices by 6-18 months in normal times, compressing to 3-6 months in high-inflation regimes

### Deflation Risk Mechanism

**Trigger conditions:**
1. Demand destruction creates excess capacity (output gap goes negative)
2. Firms cut prices to maintain sales volume
3. Sticky nominal wages mean real wages rise → firms cut labor → more unemployment
4. Unemployed consume less → further demand destruction
5. Falling prices → consumers postpone purchases (especially durables) → more demand loss
6. Debt deflation: real value of debt rises → defaults increase → banks tighten further

**Key insight for simulation:** Deflation risk emerges when unemployment exceeds ~8% AND capacity utilization falls below ~70% AND credit is contracting simultaneously. The 2008 crisis nearly triggered it; massive Fed intervention prevented it.

---

## 9. SOCIAL/BEHAVIORAL ECONOMICS

### Unemployment → Mental Health/Suicide

**Specific correlations:**
- 1 percentage point increase in unemployment → 1.0-1.6% increase in suicide rate
- Unemployed individuals are 87% more likely to die by suicide (meta-analysis, 43 studies)
- Individuals with financial stress: 74% more likely (23 studies)
- Long-term unemployment (>52 weeks): significantly larger negative mental health effects

**2008 Great Recession specific:**
- US unemployment rose from 5.8% to 9.6% (2007-2010)
- Associated with 3.8% increase in suicide rate → ~1,330 additional suicides
- Total excess suicides during recessionary period: ~4,750

### Unemployment → Crime (Counterintuitive)

**Theory:** 1% decrease in unemployment → 1.6-2.4% decrease in property crime, 0.5% decrease in violent crime (historical 1971-1997 data)

**BUT during 2008 Great Recession (anomalous):**
- Unemployment doubled (5% → 10%)
- Property crime FELL: robbery down 8%, auto theft down 17% (2009)
- Violent crime fell 6% in 2010
- NYC: robbery -4%, burglary -10% (2008-2010)

**Explanation:** Other factors (demographics, policing, incarceration rates, technology) can dominate the unemployment-crime relationship.

### Community Self-Organization (Mutual Aid)

**COVID-19 (2020):**
- Thousands of grassroots mutual aid groups formed within weeks of lockdowns (Spring 2020)
- Scale: from street-level groups to city-wide networks
- Activities: food distribution, PPE procurement, medicine delivery, financial support
- More people organized mutual aid during COVID than in previous decades combined
- Groups often became semi-permanent community social services

**Timeline:** Crisis onset → first mutual aid groups: 1-2 weeks. Peak organizing: 3-6 weeks into crisis.

### Rally Around the Flag Effect

**Duration by crisis type:**
- Military/security crisis: 2-6 months (Bush post-9/11: 14+ months — largest ever recorded)
- Economic crisis: 2-4 weeks (much shorter, quickly replaced by blame)
- Pandemic: initial 2-3 week boost, then rapid decay (Austria COVID data)
- Targeted event (bin Laden killing): ~10 weeks back to baseline

**For economic simulation:** Model a 2-4 week social cohesion boost at crisis onset, then decaying by ~20% per week.

---

## 10. INTERNATIONAL SPILLOVER

### US Recession → Global Trade

**2008-2009 Great Trade Collapse:**
- Real world trade fell ~15% from 2008Q1 to 2009Q1
- Full-year 2009: world merchandise trade volume fell 12.2%
- Major bilateral trade flows dropped 20-30% from 2008Q2 to 2009Q2
- Sharpest drops: automobiles, durable industrial supplies, capital goods
- ALL 104 WTO-reporting nations experienced import AND export declines
- Speed: equivalent of 24 months of Great Depression decline compressed into 9 months (Nov 2008 onward)

### Supply Chain Shock Propagation

**Timeline:**
- Financial shock: Hours to days (interbank/wholesale funding)
- Trade finance freeze: Days to weeks (letters of credit, export guarantees)
- Order cancellations: Weeks (1-4 weeks)
- Production adjustments: Weeks to months (1-3 months)
- Employment effects abroad: Months (2-6 months)

### Financial Market Contagion

**Speed:** Near-instantaneous across connected markets.

**Lehman Brothers collapse (Sep 15, 2008):**
- DJIA dropped 504.48 points (4.42%) on Day 1
- European/Asian markets followed within hours (next trading session)
- Cross-market correlations surged: Indonesia-Korea correlation jumped from <60% to >70%
- Fed had to arrange emergency dollar swap lines within DAYS
- European banks that had no direct US housing exposure faced funding difficulties within weeks
- Equity prices in Taiwan dropped 38.5% in three months
- Full global contagion: ~5 years to work through all economies

**Modern interconnection:** Financial contagion now spreads at "internet speed" — essentially same-day across all major markets that are open.

---

## SIMULATION CALIBRATION SUMMARY

### Critical Parameters for Economic Engine

| Parameter | Normal Value | Crisis Value | Source |
|-----------|-------------|-------------|--------|
| Essential spending share | 70-75% of income | 85%+ | BLS CEX |
| Personal savings rate | 5-8% | 2-4% (drawdown) or 30%+ (lockdown) | BEA |
| Paycheck-to-paycheck households | 24-34% | 35-45% | Fed, Bankrate |
| Job finding rate (monthly) | 38-40% | 18-29% | CPS, St. Louis Fed |
| Median unemployment duration | 8-9 weeks | 25+ weeks | BLS |
| Revenue drop → layoffs lag | N/A | 3-6 months (organic), 0-4 weeks (shock) | BLS Mass Layoffs |
| Fiscal multiplier | 0.7-0.9 | 1.0-1.5 | IMF, NBER |
| UI replacement rate | 43% of wages | 43% + federal supplement | DOL |
| Manufacturing job multiplier | 1:1.4 (conservative) to 1:7.4 (total) | Same ratios apply | EPI |
| Oil → gas pass-through | 12% day 1, 50% by month 1 | Same | Dallas Fed |
| VIX normal/crisis | 12-20 / 50-80+ | Peak: 82-89 | CBOE |
| CCI → spending coefficient | +1pt CCI → +0.38pt spending growth | Same | Chicago Fed |
| Crisis → Fed response | N/A | 4-43 days | Historical |
| Crisis → stimulus checks | N/A | 17-21 days (direct deposit) | IRS, GAO |
| Global trade decline | N/A | -12 to -15% year-over-year | WTO |
| Financial contagion speed | N/A | Hours (markets), days (funding), weeks (real economy) | NBER |
| Unemployment → suicide | Baseline | +1pp unemployment → +1-1.6% suicide rate | Lancet, PMC |
| Rally-around-flag duration | N/A | 2-4 weeks (economic), 2-14 months (security) | Political science lit |

---

## KEY SOURCES

- BLS Consumer Expenditure Survey 2023: https://www.bls.gov/opub/reports/consumer-expenditures/2023/
- Federal Reserve SHED Survey 2024: https://www.federalreserve.gov/publications/2025-economic-well-being-of-us-households-in-2024-savings-and-investments.htm
- BLS Extended Mass Layoffs 2008: https://www.bls.gov/opub/reports/mass-layoffs/archive/extended_mass_layoffs2008.pdf
- FRED Personal Savings Rate: https://fred.stlouisfed.org/series/PSAVERT
- EPI Employment Multipliers: https://www.epi.org/publication/updated-employment-multipliers-for-the-u-s-economy/
- Dallas Fed Oil Price Pass-Through: https://www.dallasfed.org/research/economics/2019/1001
- Chicago Fed Consumer Sentiment & Spending: https://www.chicagofed.org/publications/chicago-fed-letter/2009/may-262
- St. Louis Fed Job Finding Rates: https://www.stlouisfed.org/on-the-economy/2021/september/evolution-job-finding-rate-covid19-recession
- IMF Fiscal Multipliers: https://www.imf.org/external/pubs/ft/tnm/2014/tnm1404.pdf
- WTO Great Trade Collapse: https://www.wto.org/english/news_e/pres09_e/pr554_e.htm
- GAO Stimulus Checks: https://www.gao.gov/products/gao-22-106044
- NBER Great Recession Labor Market: https://www.nber.org/system/files/working_papers/w15979/w15979.pdf
- PMC Unemployment & Suicide: https://pmc.ncbi.nlm.nih.gov/articles/PMC9298506/
- SF Fed Credit Tightening Effects: https://www.frbsf.org/research-and-insights/publications/economic-letter/2024/05/economic-effects-of-tighter-lending-by-banks/
- Macroption VIX Historical: https://www.macroption.com/vix-all-time-high/
- NBER Great Trade Collapse: https://www.nber.org/system/files/working_papers/w18632/w18632.pdf

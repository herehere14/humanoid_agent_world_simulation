# 300-Agent 50-Day Diagnostic

Scenario: Harbor Heatwave and Buyout Crisis
Simulated 300 agents for 50 days (1200 ticks) in a single town with 8 districts.

## Headline Findings
- Runtime: 239.17s
- Final relationship pairs: 7525 (from 5548 seeded pairs)
- Fired events: 1494
- Ripple events added during sim: 1467
- Generated dynamic event mix: {'neighborhood_meeting': 98, 'mutual_aid_hub': 155, 'rumor_wave': 334, 'coalition_caucus': 142, 'hospital_surge': 81, 'debt_crunch': 96, 'whistleblower_leak': 45, 'accountability_hearing': 96, 'organizing_meeting': 94, 'boycott_call': 99, 'conflict_flashpoint': 227}
- Fired dynamic event mix: {'neighborhood_meeting': 98, 'mutual_aid_hub': 155, 'rumor_wave': 334, 'coalition_caucus': 142, 'hospital_surge': 81, 'debt_crunch': 96, 'whistleblower_leak': 45, 'accountability_hearing': 96, 'organizing_meeting': 94, 'boycott_call': 99, 'conflict_flashpoint': 227}
- Total resolved pair interactions: 6458
- Final dominant concerns: {'defend my neighborhood from the spillover': 142, 'force public accountability': 93, 'hold my bloc together': 37, 'control the story before it controls me': 10, 'keep income and home secure': 8, 'stay on my feet': 3, 'make jasper pay a price': 1, 'make adam pay a price': 1}
- Final dominant action styles: {'bloc discipline': 169, 'hustling restraint': 61, 'careful omission': 18, 'protective caretaking': 17, 'score-settling focus': 13, 'plainspoken honesty': 10, 'calculated positioning': 10, 'controlled precision': 1}

## Daily Arc
- Day 1: events=0, interactions=0, avg_valence=0.50, avg_vulnerability=0.01
- Day 2: events=10, interactions=70, avg_valence=0.50, avg_vulnerability=0.04
- Day 3: events=12, interactions=97, avg_valence=0.50, avg_vulnerability=0.04
- Day 4: events=13, interactions=118, avg_valence=0.50, avg_vulnerability=0.07
- Day 5: events=13, interactions=115, avg_valence=0.50, avg_vulnerability=0.05
- Day 6: events=11, interactions=108, avg_valence=0.50, avg_vulnerability=0.06
- Day 7: events=14, interactions=125, avg_valence=0.50, avg_vulnerability=0.07
- Day 8: events=15, interactions=121, avg_valence=0.50, avg_vulnerability=0.08
- Day 9: events=13, interactions=119, avg_valence=0.50, avg_vulnerability=0.07
- Day 10: events=14, interactions=130, avg_valence=0.50, avg_vulnerability=0.08
- ...
- Day 46: events=16, interactions=140, avg_valence=0.50, avg_vulnerability=0.08
- Day 47: events=14, interactions=134, avg_valence=0.50, avg_vulnerability=0.08
- Day 48: events=12, interactions=126, avg_valence=0.50, avg_vulnerability=0.08
- Day 49: events=13, interactions=130, avg_valence=0.50, avg_vulnerability=0.09
- Day 50: events=13, interactions=134, avg_valence=0.50, avg_vulnerability=0.08

## Sample Communications
- No LLM dialogue samples generated.
## Representative Dynamic Events
- Day 2, 06:00 [neighborhood_meeting] community_center: Neighbors gather at community_center to compare symptoms, school worries, and who can watch whose kids before the next shift starts.
- Day 2, 07:00 [mutual_aid_hub] community_center: Residents turn community_center into an improvised aid hub. Food, childcare offers, and quiet check-ins spread faster than official guidance.
- Day 2, 07:00 [rumor_wave] central_bar: Rumors from Suburbs South sweep through central_bar. People compare names, blame, and half-verified details about the town's community care.
- Day 2, 08:00 [rumor_wave] harbor_bar: Rumors from Suburbs South sweep through harbor_bar. People compare names, blame, and half-verified details about the town's community care.
- Day 2, 09:00 [neighborhood_meeting] north_school: Neighbors gather at north_school to compare symptoms, school worries, and who can watch whose kids before the next shift starts.
- Day 2, 09:00 [mutual_aid_hub] main_market: The town's economic strain becomes visible at main_market. Vendors, customers, and laid-off workers improvise a relief line out of ordinary business.
- Day 2, 10:00 [coalition_caucus] hospital: Care Network pulls together a closed-door caucus at hospital. Members compare loyalties, trade names, and decide who can be trusted to speak for the bloc.
- Day 2, 10:00 [rumor_wave] downtown_cafe: Rumors from Suburbs North sweep through downtown_cafe. People compare names, blame, and half-verified details about the town's family safety.
- Day 2, 11:00 [hospital_surge] hospital: A fresh screening surge hits hospital. Staff improvise overflow lines while families compare symptoms and rumors in the hallway.
- Day 2, 11:00 [rumor_wave] main_market: Rumors from Suburbs North sweep through main_market. People compare names, blame, and half-verified details about the town's medical overload.
## Relationship Formation
### Highest Trust
- Felipe × Teresa: trust=+1.00, warmth=+1.00, familiarity=141
- Anita × Orla: trust=+1.00, warmth=+1.00, familiarity=84
- Faye × Idris: trust=+1.00, warmth=+1.00, familiarity=129
- Lisa × Noriko: trust=+1.00, warmth=+1.00, familiarity=114
- Lance × Wren: trust=+1.00, warmth=+1.00, familiarity=159
- Ngozi × Yasmin: trust=+1.00, warmth=+1.00, familiarity=316
- Lila × Olumide: trust=+1.00, warmth=+1.00, familiarity=138
- Lance × Yasmin: trust=+1.00, warmth=+1.00, familiarity=195
- Amit × Lance: trust=+1.00, warmth=+1.00, familiarity=111
- Gavin × Lev: trust=+1.00, warmth=+1.00, familiarity=202
### Highest Resentment
- Arun × Pierre: resentment_ab=1.00, resentment_ba=0.64, trust=-1.00
- Aisha × Tendai: resentment_ab=0.10, resentment_ba=1.00, trust=-1.00
- Kavita × Obinna: resentment_ab=0.98, resentment_ba=1.00, trust=-1.00
- Haruki × Keiko: resentment_ab=1.00, resentment_ba=1.00, trust=-1.00
- Bianca × Jude: resentment_ab=0.00, resentment_ba=1.00, trust=+0.05
- Daria × Sung: resentment_ab=0.00, resentment_ba=1.00, trust=-0.88
- Bianca × Haruki: resentment_ab=1.00, resentment_ba=0.69, trust=-0.99
- Akiko × Joseph: resentment_ab=0.22, resentment_ba=1.00, trust=-0.99
- Neha × Zara: resentment_ab=1.00, resentment_ba=0.94, trust=-1.00
- Elara × Trevor: resentment_ab=0.95, resentment_ba=1.00, trust=-0.63
### Biggest Changes
- Lila × Yasmin: Δtrust=+0.71, Δwarmth=+0.80, Δfamiliarity=373, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Aurora × Simon: Δtrust=-1.17, Δwarmth=-1.22, Δfamiliarity=315, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.69
- Lance × Lila: Δtrust=+0.37, Δwarmth=+0.63, Δfamiliarity=330, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Adaeze × Adriana: Δtrust=-1.17, Δwarmth=-1.11, Δfamiliarity=284, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.63
- Lila × Ngozi: Δtrust=+0.33, Δwarmth=+0.75, Δfamiliarity=333, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.81
- Boris × Lila: Δtrust=-1.19, Δwarmth=-1.24, Δfamiliarity=247, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.82
- Anya × Gavin: Δtrust=-1.20, Δwarmth=-1.23, Δfamiliarity=239, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.78
- Jude × Sanaa: Δtrust=-0.77, Δwarmth=-0.60, Δfamiliarity=300, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.86
- Thomas × Xavier: Δtrust=-0.57, Δwarmth=-0.34, Δfamiliarity=271, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.76
- Bianca × Jude: Δtrust=-0.09, Δwarmth=-0.31, Δfamiliarity=278, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
## Coalitions
- Mutual Aid Ring: members=91, issue=community care, avg_loyalty=1.00, avg_injustice=0.44, avg_economic=0.65, avg_secrecy=0.27
- Tenant Defense Network: members=55, issue=family safety, avg_loyalty=1.00, avg_injustice=0.54, avg_economic=0.76, avg_secrecy=0.26
- Grid Union: members=39, issue=industrial fallout, avg_loyalty=1.00, avg_injustice=0.23, avg_economic=0.60, avg_secrecy=0.20
- Small Business Circle: members=31, issue=livelihood strain, avg_loyalty=1.00, avg_injustice=0.58, avg_economic=0.78, avg_secrecy=0.27
- Campus Action Network: members=27, issue=public organizing, avg_loyalty=1.00, avg_injustice=0.16, avg_economic=0.33, avg_secrecy=0.28
- Care Network: members=24, issue=medical overload, avg_loyalty=0.98, avg_injustice=0.64, avg_economic=0.40, avg_secrecy=0.33
- City Hall Caucus: members=24, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.25, avg_economic=0.17, avg_secrecy=0.47
- Redevelopment Board: members=23, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.17, avg_economic=0.21, avg_secrecy=0.38
- Harbor Families: members=17, issue=waterfront survival, avg_loyalty=1.00, avg_injustice=0.19, avg_economic=0.39, avg_secrecy=0.31

## Representative Final Minds
### Adaeze [healthcare]
- Action: REST at north_homes
- Thought: Private read: If our side splinters now, people start cutting private deals and we lose leverage fast.
Primary concern: hold my bloc together
Blame focus: adriana
Likely support target: seema
Coalitions: care_network, mutual_aid_ring
Economic pressure: 0.51 | Loyalty pressure: 1.00 | Secrecy pressure: 0.36
Private burden: skipped a family emergency because the ward could not spare another body
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to hold my bloc together. If our side splinters now, people start cutting private deals and we lose leverage fast. Right now I am leaning toward hold the bloc.
- Strongest tie: Adriana (trust=-0.98, warmth=-0.85, resentment=1.00)
- Futures: If nothing changes, Adaeze keeps trying to hold my bloc together through hustling restraint.
### Thomas [community]
- Action: REST at north_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Blame focus: xavier
Likely support target: ngozi
Coalitions: mutual_aid_ring, tenant_defense_network
Economic pressure: 0.84 | Loyalty pressure: 1.00 | Secrecy pressure: 0.41
Private burden: borrowed grocery money and keeps promising next week
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. Right now I am leaning toward hold the bloc.
- Strongest tie: Xavier (trust=-0.48, warmth=-0.11, resentment=1.00)
- Futures: If nothing changes, Thomas keeps trying to defend my neighborhood from the spillover through hustling restraint.
### Khalid [factory_worker]
- Action: REST at south_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: circumstances
Likely support target: wanjiku
Coalitions: none
Economic pressure: 0.74 | Loyalty pressure: 1.00 | Secrecy pressure: 0.10
Private burden: none
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. Right now I am leaning toward hold the bloc.
- Strongest tie: Wanjiku (trust=+0.41, warmth=+0.65, resentment=0.00)
- Futures: If nothing changes, Khalid keeps trying to force public accountability through hustling restraint.
### Lev [market_vendor]
- Action: REST at south_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Blame focus: circumstances
Likely support target: thea
Coalitions: mutual_aid_ring, small_business_circle
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.27
Private burden: has been quietly feeding families from inventory that should have been sold
Priority motive: hold the bloc
Mask: keeps a united front even when trust is fraying
Action style: bloc discipline
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. Right now I am leaning toward hold the bloc.
- Strongest tie: Thea (trust=+1.00, warmth=+1.00, resentment=0.00)
- Futures: If nothing changes, Lev keeps trying to keep income and home secure through bloc discipline.
### Donna [office_professional]
- Action: REST at south_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: circumstances
Likely support target: nina
Coalitions: none
Economic pressure: 0.28 | Loyalty pressure: 1.00 | Secrecy pressure: 0.33
Private burden: helped price distressed waterfront parcels before the buyout notices went out
Priority motive: hold the bloc
Mask: stays polished while scanning openings
Action style: calculated positioning
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. Right now I am leaning toward hold the bloc.
- Strongest tie: Ricardo (trust=+0.07, warmth=+0.17, resentment=0.00)
- Futures: If nothing changes, Donna keeps trying to force public accountability through calculated positioning.
### Yuki [government_worker]
- Action: RUMINATE at south_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: what I have been hiding
Likely support target: jabari
Coalitions: city_hall_caucus
Economic pressure: 0.19 | Loyalty pressure: 1.00 | Secrecy pressure: 0.61
Private burden: deleted an embarrassing thread but knows copies still exist
Priority motive: hold the bloc
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. Right now I am leaning toward hold the bloc.
- Strongest tie: Jasper (trust=+0.17, warmth=+0.00, resentment=0.00)
- Futures: If nothing changes, Yuki keeps trying to force public accountability through careful omission.
### Orla [student]
- Action: REST at south_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: circumstances
Likely support target: anita
Coalitions: campus_action_network
Economic pressure: 0.31 | Loyalty pressure: 1.00 | Secrecy pressure: 0.44
Private burden: promised to protect a source and already repeated too much
Priority motive: hold the bloc
Mask: keeps a united front even when trust is fraying
Action style: bloc discipline
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. Right now I am leaning toward hold the bloc.
- Strongest tie: Anita (trust=+1.00, warmth=+1.00, resentment=0.00)
- Futures: If nothing changes, Orla keeps trying to force public accountability through bloc discipline.
### George [dock_worker]
- Action: REST at south_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Blame focus: circumstances
Likely support target: sakura
Coalitions: harbor_families
Economic pressure: 0.38 | Loyalty pressure: 1.00 | Secrecy pressure: 0.38
Private burden: owes a favor to someone negotiating against the harbor
Priority motive: hold the bloc
Mask: keeps a united front even when trust is fraying
Action style: bloc discipline
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. Right now I am leaning toward hold the bloc.
- Strongest tie: Victor (trust=+0.61, warmth=+0.68, resentment=0.00)
- Futures: If nothing changes, George keeps trying to defend my neighborhood from the spillover through bloc discipline.

## Where We Lack
- Most behavior is still routine (`WORK`/`REST`/`IDLE`). Emotional individuality is stronger now, but action diversity is still too low for a world this large.
- Too many agents collapse into a small set of concerns. The new subjective layer helps, but archetypes still bunch together under the same crisis.
- Communication is still mostly unrendered. The sim forms relationships numerically, but only a tiny fraction of interactions become actual dialogue.
# 300-Agent 20-Day Diagnostic

Scenario: Harbor Heatwave and Buyout Crisis
Simulated 300 agents for 20 days (480 ticks) in a single town with 8 districts.

## Headline Findings
- Runtime: 88.23s
- Final relationship pairs: 6847 (from 5548 seeded pairs)
- Fired events: 560
- Ripple events added during sim: 533
- Generated dynamic event mix: {'neighborhood_meeting': 38, 'mutual_aid_hub': 50, 'rumor_wave': 137, 'coalition_caucus': 57, 'hospital_surge': 31, 'debt_crunch': 36, 'whistleblower_leak': 17, 'accountability_hearing': 36, 'organizing_meeting': 34, 'boycott_call': 30, 'conflict_flashpoint': 67}
- Fired dynamic event mix: {'neighborhood_meeting': 38, 'mutual_aid_hub': 50, 'rumor_wave': 137, 'coalition_caucus': 57, 'hospital_surge': 31, 'debt_crunch': 36, 'whistleblower_leak': 17, 'accountability_hearing': 36, 'organizing_meeting': 34, 'boycott_call': 30, 'conflict_flashpoint': 67}
- Total resolved pair interactions: 2319
- Final dominant concerns: {'defend my neighborhood from the spillover': 141, 'force public accountability': 94, 'hold my bloc together': 36, 'control the story before it controls me': 12, 'keep income and home secure': 7, 'stay on my feet': 6, 'make jasper pay a price': 1, 'make adam pay a price': 1}
- Final dominant action styles: {'bloc discipline': 169, 'hustling restraint': 50, 'protective caretaking': 24, 'score-settling focus': 20, 'careful omission': 19, 'plainspoken honesty': 9, 'calculated positioning': 7, 'controlled precision': 1}

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
- Day 16: events=13, interactions=145, avg_valence=0.50, avg_vulnerability=0.08
- Day 17: events=13, interactions=144, avg_valence=0.50, avg_vulnerability=0.09
- Day 18: events=12, interactions=125, avg_valence=0.50, avg_vulnerability=0.08
- Day 19: events=15, interactions=135, avg_valence=0.50, avg_vulnerability=0.08
- Day 20: events=13, interactions=139, avg_valence=0.50, avg_vulnerability=0.08

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
- Lev × Thea: trust=+1.00, warmth=+1.00, familiarity=146
- Lance × Thomas: trust=+0.84, warmth=+0.55, familiarity=91
- Lance × Wren: trust=+0.77, warmth=+0.72, familiarity=70
- Lance × Lila: trust=+0.70, warmth=+0.73, familiarity=127
- Lisa × Noriko: trust=+0.70, warmth=+0.76, familiarity=57
- Ali × Sage: trust=+0.70, warmth=+0.70, familiarity=34
- George × Sakura: trust=+0.69, warmth=+0.69, familiarity=29
- Noura × Patrick: trust=+0.68, warmth=+0.44, familiarity=45
- Nneka × Yasmin: trust=+0.68, warmth=+0.41, familiarity=79
- Anita × Orla: trust=+0.67, warmth=+0.81, familiarity=33
### Highest Resentment
- Jasper × Sekou: resentment_ab=1.00, resentment_ba=0.30, trust=-0.72
- Holly × Javier: resentment_ab=1.00, resentment_ba=0.71, trust=-0.38
- Anya × Gavin: resentment_ab=1.00, resentment_ba=0.99, trust=-0.99
- Thomas × Xavier: resentment_ab=1.00, resentment_ba=0.99, trust=-0.96
- Ethan × Katya: resentment_ab=1.00, resentment_ba=0.59, trust=-0.44
- Jude × Sanaa: resentment_ab=1.00, resentment_ba=0.95, trust=-0.52
- Alejandro × Joseph: resentment_ab=0.21, resentment_ba=1.00, trust=-0.73
- Keiko × Ulric: resentment_ab=1.00, resentment_ba=0.94, trust=-0.99
- Jasper × Kofi: resentment_ab=1.00, resentment_ba=0.32, trust=-0.89
- Sanaa × Victor: resentment_ab=1.00, resentment_ba=0.23, trust=-0.47
### Biggest Changes
- Anya × Gavin: Δtrust=-1.19, Δwarmth=-1.07, Δfamiliarity=115, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.65
- Thomas × Xavier: Δtrust=-1.05, Δwarmth=-0.53, Δfamiliarity=127, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.63
- Jude × Sanaa: Δtrust=-0.52, Δwarmth=-0.25, Δfamiliarity=122, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.67
- Keiko × Ulric: Δtrust=-0.99, Δwarmth=-0.81, Δfamiliarity=54, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.56
- Lila × Yasmin: Δtrust=+0.20, Δwarmth=+0.26, Δfamiliarity=130, Δresentment=+0.71, Δgrievance=+1.00, Δdebt=+0.60
- Brynn × Leila: Δtrust=-0.67, Δwarmth=-0.11, Δfamiliarity=57, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.54
- Adaeze × Adriana: Δtrust=-0.03, Δwarmth=+0.09, Δfamiliarity=106, Δresentment=+0.83, Δgrievance=+1.00, Δdebt=+0.63
- Ethan × Katya: Δtrust=-0.63, Δwarmth=-0.25, Δfamiliarity=50, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.53
- Boris × Lila: Δtrust=-0.28, Δwarmth=+0.00, Δfamiliarity=88, Δresentment=+0.95, Δgrievance=+1.00, Δdebt=+0.56
- Holly × Javier: Δtrust=-0.60, Δwarmth=-0.23, Δfamiliarity=41, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.56
## Coalitions
- Mutual Aid Ring: members=91, issue=community care, avg_loyalty=1.00, avg_injustice=0.32, avg_economic=0.64, avg_secrecy=0.26
- Tenant Defense Network: members=55, issue=family safety, avg_loyalty=1.00, avg_injustice=0.36, avg_economic=0.74, avg_secrecy=0.22
- Grid Union: members=39, issue=industrial fallout, avg_loyalty=1.00, avg_injustice=0.13, avg_economic=0.62, avg_secrecy=0.19
- Small Business Circle: members=31, issue=livelihood strain, avg_loyalty=1.00, avg_injustice=0.40, avg_economic=0.78, avg_secrecy=0.23
- Campus Action Network: members=27, issue=public organizing, avg_loyalty=1.00, avg_injustice=0.16, avg_economic=0.33, avg_secrecy=0.28
- Care Network: members=24, issue=medical overload, avg_loyalty=0.97, avg_injustice=0.38, avg_economic=0.35, avg_secrecy=0.36
- City Hall Caucus: members=24, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.23, avg_economic=0.18, avg_secrecy=0.47
- Redevelopment Board: members=23, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.18, avg_economic=0.21, avg_secrecy=0.38
- Harbor Families: members=17, issue=waterfront survival, avg_loyalty=1.00, avg_injustice=0.19, avg_economic=0.39, avg_secrecy=0.31

## Representative Final Minds
### Nikolai [healthcare]
- Action: REST at north_homes
- Thought: Private read: If our side splinters now, people start cutting private deals and we lose leverage fast.
Primary concern: hold my bloc together
Blame focus: circumstances
Likely support target: elara
Coalitions: care_network
Economic pressure: 0.26 | Loyalty pressure: 0.95 | Secrecy pressure: 0.30
Private burden: none
Priority motive: hold the bloc
Mask: stays useful so panic stays hidden
Action style: protective caretaking
Inner voice: I need to hold my bloc together. If our side splinters now, people start cutting private deals and we lose leverage fast. Right now I am leaning toward hold the bloc.
- Strongest tie: Elara (trust=+0.43, warmth=+0.52, resentment=0.00)
- Futures: If nothing changes, Nikolai keeps trying to hold my bloc together through protective caretaking.
### Maria [community]
- Action: REST at south_homes
- Thought: Private read: If our side splinters now, people start cutting private deals and we lose leverage fast.
Primary concern: hold my bloc together
Blame focus: amara
Likely support target: amara
Coalitions: mutual_aid_ring
Economic pressure: 0.97 | Loyalty pressure: 1.00 | Secrecy pressure: 0.43
Private burden: is behind on rent and hiding notices from the kids
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to hold my bloc together. If our side splinters now, people start cutting private deals and we lose leverage fast. Right now I am leaning toward hold the bloc.
- Strongest tie: Olumide (trust=+0.41, warmth=+0.34, resentment=0.00)
- Futures: If nothing changes, Maria keeps trying to hold my bloc together through hustling restraint.
### Sanaa [market_vendor]
- Action: REST at south_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Blame focus: bianca
Likely support target: takeshi
Coalitions: mutual_aid_ring, small_business_circle, tenant_defense_network
Economic pressure: 0.94 | Loyalty pressure: 1.00 | Secrecy pressure: 0.22
Private burden: none
Priority motive: hold the bloc
Mask: keeps a united front even when trust is fraying
Action style: bloc discipline
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. Right now I am leaning toward hold the bloc.
- Strongest tie: Jude (trust=-0.52, warmth=-0.25, resentment=0.95)
- Futures: If nothing changes, Sanaa keeps trying to defend my neighborhood from the spillover through bloc discipline.
### Adam [government_worker]
- Action: RUMINATE at north_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Blame focus: what I have been hiding
Likely support target: jun
Coalitions: city_hall_caucus
Economic pressure: 0.09 | Loyalty pressure: 1.00 | Secrecy pressure: 0.80
Private burden: promised a neighborhood hearing that was never going to happen
Priority motive: hold the bloc
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. Right now I am leaning toward hold the bloc.
- Strongest tie: Mabel (trust=+0.11, warmth=+0.09, resentment=0.00)
- Futures: If nothing changes, Adam keeps trying to defend my neighborhood from the spillover through careful omission.
### Tendai [factory_worker]
- Action: REST at south_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Blame focus: circumstances
Likely support target: aisha
Coalitions: grid_union
Economic pressure: 0.92 | Loyalty pressure: 1.00 | Secrecy pressure: 0.06
Private burden: borrowed from a coworker and still have not squared it
Priority motive: hold the bloc
Mask: keeps a united front even when trust is fraying
Action style: bloc discipline
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. Right now I am leaning toward hold the bloc.
- Strongest tie: Aisha (trust=+0.34, warmth=+0.40, resentment=0.00)
- Futures: If nothing changes, Tendai keeps trying to defend my neighborhood from the spillover through bloc discipline.
### Henry [student]
- Action: REST at north_homes
- Thought: Private read: Whoever frames this first decides who looks guilty, weak, or disposable.
Primary concern: control the story before it controls me
Blame focus: circumstances
Likely support target: ines
Coalitions: campus_action_network
Economic pressure: 0.42 | Loyalty pressure: 1.00 | Secrecy pressure: 0.36
Private burden: none
Priority motive: hold the bloc
Mask: keeps a united front even when trust is fraying
Action style: bloc discipline
Inner voice: I need to control the story before it controls me. Whoever frames this first decides who looks guilty, weak, or disposable. Right now I am leaning toward hold the bloc.
- Strongest tie: Teresa (trust=+0.34, warmth=+0.28, resentment=0.00)
- Futures: If nothing changes, Henry keeps trying to control the story before it controls me through bloc discipline.
### Nina [office_professional]
- Action: REST at south_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Blame focus: circumstances
Likely support target: olga
Coalitions: redevelopment_board
Economic pressure: 0.15 | Loyalty pressure: 1.00 | Secrecy pressure: 0.50
Private burden: none
Priority motive: hold the bloc
Mask: keeps a united front even when trust is fraying
Action style: bloc discipline
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. Right now I am leaning toward hold the bloc.
- Strongest tie: Delia (trust=+0.12, warmth=+0.09, resentment=0.00)
- Futures: If nothing changes, Nina keeps trying to defend my neighborhood from the spillover through bloc discipline.
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
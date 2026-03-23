# 300-Agent 50-Day Diagnostic

Scenario: Harbor Heatwave and Buyout Crisis
Simulated 300 agents for 50 days (1200 ticks) in a single town with 8 districts.

## Headline Findings
- Runtime: 241.23s
- Final relationship pairs: 7567 (from 5279 seeded pairs)
- Fired events: 1569
- Ripple events added during sim: 1542
- Generated dynamic event mix: {'neighborhood_meeting': 98, 'mutual_aid_hub': 171, 'rumor_wave': 286, 'coalition_caucus': 141, 'boycott_call': 203, 'debt_crunch': 96, 'whistleblower_leak': 57, 'hospital_surge': 80, 'organizing_meeting': 94, 'accountability_hearing': 92, 'conflict_flashpoint': 224}
- Fired dynamic event mix: {'neighborhood_meeting': 98, 'mutual_aid_hub': 171, 'rumor_wave': 286, 'coalition_caucus': 141, 'boycott_call': 203, 'debt_crunch': 96, 'whistleblower_leak': 57, 'hospital_surge': 80, 'organizing_meeting': 94, 'accountability_hearing': 92, 'conflict_flashpoint': 224}
- Total resolved pair interactions: 5952
- Final dominant concerns: {'defend my neighborhood from the spillover': 67, 'turn private anger into organized pressure': 52, "make the neighborhood's losses impossible to wave away": 42, 'test who is still solid when it counts': 26, 'keep income and home secure': 19, 'keep the block from fraying into private panic': 18, 'keep this mess from reaching my block and my bills': 17, 'force public accountability': 17}
- Final dominant action styles: {'hustling restraint': 104, 'careful omission': 64, 'cold bookkeeping': 21, 'double-shift numbness': 17, 'score-settling focus': 15, 'sharp escalation': 11, 'career triangulation': 10, 'narrative positioning': 9}

## Daily Arc
- Day 1: events=0, interactions=0, avg_valence=0.50, avg_vulnerability=0.01
- Day 2: events=10, interactions=63, avg_valence=0.50, avg_vulnerability=0.04
- Day 3: events=12, interactions=85, avg_valence=0.50, avg_vulnerability=0.05
- Day 4: events=11, interactions=83, avg_valence=0.50, avg_vulnerability=0.06
- Day 5: events=15, interactions=113, avg_valence=0.50, avg_vulnerability=0.07
- Day 6: events=15, interactions=111, avg_valence=0.50, avg_vulnerability=0.07
- Day 7: events=16, interactions=130, avg_valence=0.50, avg_vulnerability=0.07
- Day 8: events=15, interactions=119, avg_valence=0.50, avg_vulnerability=0.09
- Day 9: events=14, interactions=114, avg_valence=0.50, avg_vulnerability=0.08
- Day 10: events=13, interactions=106, avg_valence=0.50, avg_vulnerability=0.07
- ...
- Day 46: events=12, interactions=118, avg_valence=0.50, avg_vulnerability=0.09
- Day 47: events=15, interactions=126, avg_valence=0.50, avg_vulnerability=0.09
- Day 48: events=13, interactions=107, avg_valence=0.50, avg_vulnerability=0.10
- Day 49: events=14, interactions=127, avg_valence=0.50, avg_vulnerability=0.09
- Day 50: events=15, interactions=147, avg_valence=0.50, avg_vulnerability=0.12

## Sample Communications
- No LLM dialogue samples generated.
## Representative Dynamic Events
- Day 2, 06:00 [neighborhood_meeting] community_center: Neighbors gather at community_center to compare symptoms, school worries, and who can watch whose kids before the next shift starts.
- Day 2, 07:00 [mutual_aid_hub] community_center: Residents turn community_center into an improvised aid hub. Food, childcare offers, and quiet check-ins spread faster than official guidance.
- Day 2, 07:00 [rumor_wave] central_bar: Rumors from Suburbs South sweep through central_bar. People compare names, blame, and half-verified details about the town's community care.
- Day 2, 08:00 [rumor_wave] harbor_bar: Rumors from Suburbs South sweep through harbor_bar. People compare names, blame, and half-verified details about the town's community care.
- Day 2, 09:00 [neighborhood_meeting] north_school: Neighbors gather at north_school to compare symptoms, school worries, and who can watch whose kids before the next shift starts.
- Day 2, 09:00 [mutual_aid_hub] main_market: The town's economic strain becomes visible at main_market. Vendors, customers, and laid-off workers improvise a relief line out of ordinary business.
- Day 2, 10:00 [coalition_caucus] workers_canteen: Grid Union pulls together a closed-door caucus at workers_canteen. Members compare loyalties, trade names, and decide who can be trusted to speak for the bloc.
- Day 2, 10:00 [boycott_call] community_center: Tenant Defense Network circulates a boycott and pressure campaign tied to the town's family safety. People are suddenly being asked to pick a side publicly, not just complain privately.
- Day 2, 10:00 [boycott_call] main_market: Small Business Circle circulates a boycott and pressure campaign tied to the town's livelihood strain. People are suddenly being asked to pick a side publicly, not just complain privately.
- Day 2, 11:00 [mutual_aid_hub] workers_canteen: Workers and relatives crowd into workers_canteen, trading job leads, spare cash, and names of people who might actually come through.
## Relationship Formation
### Highest Trust
- Ezra × Rosa: trust=+1.00, warmth=+0.87, familiarity=129
- Rafael × Tina: trust=+1.00, warmth=+0.96, familiarity=167
- Pearl × Rafael: trust=+1.00, warmth=+0.92, familiarity=172
- Anya × Lakshmi: trust=+1.00, warmth=+1.00, familiarity=168
- Sergei × Wren: trust=+1.00, warmth=+1.00, familiarity=155
- Elena × Sung: trust=+1.00, warmth=+1.00, familiarity=98
- Sofia × Sung: trust=+1.00, warmth=+1.00, familiarity=275
- Dmitri × Kofi: trust=+1.00, warmth=+0.86, familiarity=119
- Kofi × Rosa: trust=+1.00, warmth=+0.95, familiarity=158
- Khalid × Kofi: trust=+1.00, warmth=+0.64, familiarity=115
### Highest Resentment
- Joseph × Paula: resentment_ab=1.00, resentment_ba=0.58, trust=-0.97
- Aisha × Khalid: resentment_ab=0.00, resentment_ba=1.00, trust=-1.00
- Vera × Yuri: resentment_ab=1.00, resentment_ba=0.23, trust=-0.38
- Tendai × Yuri: resentment_ab=0.00, resentment_ba=1.00, trust=-0.31
- Funke × Rahul: resentment_ab=1.00, resentment_ba=0.59, trust=+0.25
- Alejandro × Rahul: resentment_ab=0.57, resentment_ba=1.00, trust=-0.99
- Elara × Nikolai: resentment_ab=0.82, resentment_ba=1.00, trust=-0.98
- Lakshmi × Mariama: resentment_ab=1.00, resentment_ba=0.21, trust=-0.36
- Mariama × Sergei: resentment_ab=0.99, resentment_ba=1.00, trust=-0.99
- Katya × Ling: resentment_ab=1.00, resentment_ba=1.00, trust=-1.00
### Biggest Changes
- Daniel × Dmitri: Δtrust=-1.11, Δwarmth=-1.27, Δfamiliarity=373, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.87
- Amit × Lakshmi: Δtrust=-1.11, Δwarmth=-1.24, Δfamiliarity=268, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Lakshmi × Mariama: Δtrust=-0.48, Δwarmth=-0.64, Δfamiliarity=273, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.88
- Anya × Sung: Δtrust=-0.97, Δwarmth=-0.56, Δfamiliarity=243, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Alejandro × Rahul: Δtrust=-1.14, Δwarmth=-1.19, Δfamiliarity=198, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.96
- Rahul × Sanjay: Δtrust=-1.00, Δwarmth=-1.00, Δfamiliarity=206, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.81
- Dmitri × Tendai: Δtrust=-1.12, Δwarmth=-0.80, Δfamiliarity=219, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.84
- Mariama × Sergei: Δtrust=-1.10, Δwarmth=-0.38, Δfamiliarity=223, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.70
- Funke × Rahul: Δtrust=+0.14, Δwarmth=-0.10, Δfamiliarity=285, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.87
- Bianca × Leo: Δtrust=-1.10, Δwarmth=-1.28, Δfamiliarity=143, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.54
## Coalitions
- Mutual Aid Ring: members=76, issue=community care, avg_loyalty=1.00, avg_injustice=0.50, avg_economic=0.82, avg_secrecy=0.32
- Tenant Defense Network: members=50, issue=family safety, avg_loyalty=1.00, avg_injustice=0.50, avg_economic=0.84, avg_secrecy=0.35
- Grid Union: members=45, issue=industrial fallout, avg_loyalty=1.00, avg_injustice=0.55, avg_economic=0.89, avg_secrecy=0.32
- Small Business Circle: members=28, issue=livelihood strain, avg_loyalty=1.00, avg_injustice=0.46, avg_economic=0.84, avg_secrecy=0.23
- City Hall Caucus: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.28, avg_economic=0.21, avg_secrecy=0.61
- Redevelopment Board: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.15, avg_economic=0.23, avg_secrecy=0.51
- Campus Action Network: members=26, issue=public organizing, avg_loyalty=1.00, avg_injustice=0.36, avg_economic=0.41, avg_secrecy=0.52
- Care Network: members=25, issue=medical overload, avg_loyalty=1.00, avg_injustice=0.86, avg_economic=0.49, avg_secrecy=0.40
- Harbor Families: members=17, issue=waterfront survival, avg_loyalty=1.00, avg_injustice=0.38, avg_economic=0.82, avg_secrecy=0.38

## Representative Final Minds
### Sage [healthcare]
- Action: REST at north_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Ongoing story: the story can be stolen if I do not frame it
Blame focus: bakari
Likely support target: bakari
Attachment: secure attachment | Coping: disappear into work | Threat lens: scarcity
Core need: control | Shame trigger: freezing when people count on me
Care style: steady presence | Conflict style: appease first
Mask tendency: emotional shutdown | Self-story: guardian | Longing: get one quiet hour where nobody needs triage
Coalitions: care_network
Economic pressure: 0.66 | Loyalty pressure: 1.00 | Secrecy pressure: 0.43
Private burden: none
Priority motive: hold the bloc
Mask: tries to outrun debt by becoming more useful per hour
Action style: double-shift numbness
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. What would really undo me is freezing when people count on me. Under that, I just want to get one quiet hour where nobody needs triage. Right now I am leaning toward hold the bloc.
- Strongest tie: Adriana (trust=-0.38, warmth=-0.06, resentment=1.00)
- Futures: If nothing changes, Sage keeps trying to defend my neighborhood from the spillover through double-shift numbness.
### Daniel [factory_worker]
- Action: REST at south_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Ongoing story: help always comes with a price
Blame focus: dmitri
Likely support target: gemma
Attachment: anxious attachment | Coping: confront head-on | Threat lens: scarcity
Core need: safety | Shame trigger: failing family
Care style: steady presence | Conflict style: straight negotiation
Mask tendency: quiet shutdown | Self-story: provider | Longing: keep the house standing
Coalitions: grid_union, mutual_aid_ring
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.23
Private burden: none
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. What would really undo me is failing family. Under that, I just want to keep the house standing. Right now I am leaning toward protect other people.
- Strongest tie: Dmitri (trust=-0.98, warmth=-1.00, resentment=1.00)
- Futures: If nothing changes, Daniel keeps trying to keep income and home secure through hustling restraint.
### Sofia [community]
- Action: REST at north_homes
- Thought: Private read: If nobody keeps the harm visible, the block gets renamed as collateral and the damage disappears on paper.
Primary concern: make the neighborhood's losses impossible to wave away
Ongoing story: the story can be stolen if I do not frame it
Blame focus: what I have been hiding
Likely support target: sung
Attachment: anxious attachment | Coping: seek witnesses | Threat lens: scarcity
Core need: truth | Shame trigger: failing family
Care style: steady presence | Conflict style: keep score
Mask tendency: joke through it | Self-story: witness | Longing: be held without having to earn it
Coalitions: mutual_aid_ring, tenant_defense_network
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.52
Private burden: is behind on rent and hiding notices from the kids
Priority motive: protect other people
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to make the neighborhood's losses impossible to wave away. If nobody keeps the harm visible, the block gets renamed as collateral and the damage disappears on paper. What would really undo me is failing family. Under that, I just want to be held without having to earn it. Right now I am leaning toward protect other people.
- Strongest tie: Sung (trust=+1.00, warmth=+1.00, resentment=0.00)
- Futures: If nothing changes, Sofia keeps trying to make the neighborhood's losses impossible to wave away through careful omission.
### Tom [market_vendor]
- Action: REST at north_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Ongoing story: the story can be stolen if I do not frame it
Blame focus: sung
Likely support target: samira
Attachment: anxious attachment | Coping: caretake first | Threat lens: scarcity
Core need: usefulness | Shame trigger: failing family
Care style: practical fixing | Conflict style: keep score
Mask tendency: dutiful calm | Self-story: provider | Longing: make it through the month intact
Coalitions: mutual_aid_ring
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.38
Private burden: none
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. What would really undo me is failing family. Under that, I just want to make it through the month intact. Right now I am leaning toward protect other people.
- Strongest tie: Samira (trust=+1.00, warmth=+0.89, resentment=0.00)
- Futures: If nothing changes, Tom keeps trying to keep income and home secure through hustling restraint.
### Roma [dock_worker]
- Action: REST at south_homes
- Thought: Private read: I am not just looking for comfort; I am updating the map of who actually stays present under stress.
Primary concern: test who is still solid when it counts
Ongoing story: who showed up is now part of the map
Blame focus: circumstances
Likely support target: yong
Attachment: secure attachment | Coping: disappear into work | Threat lens: humiliation
Core need: belonging | Shame trigger: owing the wrong person
Care style: protective provisioning | Conflict style: straight negotiation
Mask tendency: quiet shutdown | Self-story: guardian | Longing: keep home from being swallowed
Coalitions: harbor_families
Economic pressure: 0.80 | Loyalty pressure: 1.00 | Secrecy pressure: 0.44
Private burden: helped move gear off a pier slated for buyout before the notice was public
Priority motive: protect other people
Mask: tries to outrun debt by becoming more useful per hour
Action style: double-shift numbness
Inner voice: I need to test who is still solid when it counts. I am not just looking for comfort; I am updating the map of who actually stays present under stress. What would really undo me is owing the wrong person. Under that, I just want to keep home from being swallowed. Right now I am leaning toward protect other people.
- Strongest tie: Halima (trust=+0.33, warmth=+0.08, resentment=0.00)
- Futures: If nothing changes, Roma keeps trying to test who is still solid when it counts through double-shift numbness.
### Caleb [student]
- Action: RUMINATE at south_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Ongoing story: help always comes with a price
Blame focus: what I have been hiding
Likely support target: julia
Attachment: self-protective attachment | Coping: seek witnesses | Threat lens: scarcity
Core need: truth | Shame trigger: failing family
Care style: practical fixing | Conflict style: moralize in public
Mask tendency: righteous heat | Self-story: witness | Longing: force the truth into daylight
Coalitions: campus_action_network
Economic pressure: 0.94 | Loyalty pressure: 1.00 | Secrecy pressure: 0.79
Private burden: lost grant money and is pretending activism did not cost tuition
Priority motive: hold the bloc
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. What would really undo me is failing family. Under that, I just want to force the truth into daylight. Right now I am leaning toward hold the bloc.
- Strongest tie: Julia (trust=+0.91, warmth=+0.79, resentment=0.00)
- Futures: If nothing changes, Caleb keeps trying to force public accountability through careful omission.
### Sonya [government_worker]
- Action: RUMINATE at south_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Ongoing story: the story can be stolen if I do not frame it
Blame focus: what I have been hiding
Likely support target: jasper
Attachment: secure attachment | Coping: perform competence | Threat lens: exposure
Core need: truth | Shame trigger: looking naive about power
Care style: strategic problem-solving | Conflict style: command
Mask tendency: polished competence | Self-story: fixer | Longing: get through this without becoming the scapegoat
Coalitions: city_hall_caucus
Economic pressure: 0.23 | Loyalty pressure: 1.00 | Secrecy pressure: 0.88
Private burden: deleted an embarrassing thread but knows copies still exist
Priority motive: save face
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. What would really undo me is looking naive about power. Under that, I just want to get through this without becoming the scapegoat. Right now I am leaning toward save face.
- Strongest tie: Alden (trust=+0.28, warmth=+0.09, resentment=0.00)
- Futures: If nothing changes, Sonya keeps trying to force public accountability through careful omission.
### Zeke [office_professional]
- Action: RUMINATE at south_homes
- Thought: Private read: One loose version of the story could expose what I have been trying to keep contained.
Primary concern: keep a damaging secret buried
Ongoing story: the story can be stolen if I do not frame it
Blame focus: what I have been hiding
Likely support target: crystal
Attachment: secure attachment | Coping: perform competence | Threat lens: exposure
Core need: control | Shame trigger: looking incompetent
Care style: strategic problem-solving | Conflict style: command
Mask tendency: polished competence | Self-story: climber | Longing: keep options open
Coalitions: redevelopment_board
Economic pressure: 0.33 | Loyalty pressure: 1.00 | Secrecy pressure: 1.00
Private burden: sat on an outage map that protected wealthier blocks first
Priority motive: save face
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to keep a damaging secret buried. One loose version of the story could expose what I have been trying to keep contained. What would really undo me is looking incompetent. Under that, I just want to keep options open. Right now I am leaning toward save face.
- Strongest tie: Malik (trust=+0.11, warmth=+0.23, resentment=0.00)
- Futures: If nothing changes, Zeke keeps trying to keep a damaging secret buried through careful omission.

## Where We Lack
- Most behavior is still routine (`WORK`/`REST`/`IDLE`). Emotional individuality is stronger now, but action diversity is still too low for a world this large.
- Communication is still mostly unrendered. The sim forms relationships numerically, but only a tiny fraction of interactions become actual dialogue.
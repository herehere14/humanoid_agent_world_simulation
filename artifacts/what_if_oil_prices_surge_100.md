# 300-Agent 3-Day Diagnostic

Scenario: Harbor Heatwave and Buyout Crisis
Simulated 300 agents for 3 days (72 ticks) in a single town with 8 districts.

## Headline Findings
- Runtime: 17.56s
- Final relationship pairs: 5601 (from 5279 seeded pairs)
- Fired events: 47
- Ripple events added during sim: 39
- Generated dynamic event mix: {'neighborhood_meeting': 4, 'mutual_aid_hub': 7, 'rumor_wave': 10, 'coalition_caucus': 6, 'boycott_call': 8, 'debt_crunch': 2, 'whistleblower_leak': 1, 'hospital_surge': 1}
- Fired dynamic event mix: {'macro_cost_shock': 5, 'neighborhood_meeting': 4, 'mutual_aid_hub': 7, 'rumor_wave': 10, 'coalition_caucus': 6, 'boycott_call': 8, 'debt_crunch': 2, 'whistleblower_leak': 1, 'hospital_surge': 1}
- Total resolved pair interactions: 137
- Final dominant concerns: {'defend my neighborhood from the spillover': 58, 'turn private anger into organized pressure': 48, "make the neighborhood's losses impossible to wave away": 39, 'keep income and home secure': 27, 'test who is still solid when it counts': 23, 'keep the block from fraying into private panic': 23, 'force public accountability': 22, 'keep this mess from reaching my block and my bills': 12}
- Final dominant action styles: {'hustling restraint': 148, 'careful omission': 49, 'double-shift numbness': 17, 'narrative positioning': 16, 'career triangulation': 13, 'smiling solvency theater': 13, 'bloc discipline': 9, 'task-anesthetizing focus': 8}

## Daily Arc
- Day 1: events=3, interactions=0, avg_valence=0.50, avg_vulnerability=0.01
- Day 2: events=10, interactions=62, avg_valence=0.50, avg_vulnerability=0.04
- Day 3: events=12, interactions=75, avg_valence=0.50, avg_vulnerability=0.04
- ...
- Day 1: events=3, interactions=0, avg_valence=0.50, avg_vulnerability=0.01
- Day 2: events=10, interactions=62, avg_valence=0.50, avg_vulnerability=0.04
- Day 3: events=12, interactions=75, avg_valence=0.50, avg_vulnerability=0.04

## Sample Communications
- No LLM dialogue samples generated.
## Representative Dynamic Events
- Day 2, 06:00 [neighborhood_meeting] community_center: Neighbors gather at community_center to compare symptoms, school worries, and who can watch whose kids before the next shift starts.
- Day 2, 07:00 [mutual_aid_hub] community_center: Residents turn community_center into an improvised aid hub. Food, childcare offers, and quiet check-ins spread faster than official guidance.
- Day 2, 07:00 [rumor_wave] central_bar: Rumors from Suburbs South sweep through central_bar. People compare names, blame, and half-verified details about the town's community care.
- Day 2, 08:00 [rumor_wave] harbor_bar: Rumors from Suburbs South sweep through harbor_bar. People compare names, blame, and half-verified details about the town's community care.
- Day 2, 09:00 [mutual_aid_hub] main_market: The town's economic strain becomes visible at main_market. Vendors, customers, and laid-off workers improvise a relief line out of ordinary business.
- Day 2, 09:00 [neighborhood_meeting] north_school: Neighbors gather at north_school to compare symptoms, school worries, and who can watch whose kids before the next shift starts.
- Day 2, 10:00 [coalition_caucus] workers_canteen: Grid Union pulls together a closed-door caucus at workers_canteen. Members compare loyalties, trade names, and decide who can be trusted to speak for the bloc.
- Day 2, 10:00 [boycott_call] community_center: Tenant Defense Network circulates a boycott and pressure campaign tied to the town's family safety. People are suddenly being asked to pick a side publicly, not just complain privately.
- Day 2, 10:00 [boycott_call] main_market: Small Business Circle circulates a boycott and pressure campaign tied to the town's livelihood strain. People are suddenly being asked to pick a side publicly, not just complain privately.
- Day 2, 11:00 [mutual_aid_hub] workers_canteen: Workers and relatives crowd into workers_canteen, trading job leads, spare cash, and names of people who might actually come through.
## Relationship Formation
### Highest Trust
- Omar × Quincy: trust=+0.70, warmth=+0.59, familiarity=27
- Halima × Nina: trust=+0.70, warmth=+0.41, familiarity=27
- Deepa × Remy: trust=+0.69, warmth=+0.80, familiarity=32
- Ling × Xavier: trust=+0.69, warmth=+0.75, familiarity=42
- Mateo × Takeshi: trust=+0.69, warmth=+0.71, familiarity=49
- Deepa × Khalid: trust=+0.68, warmth=+0.67, familiarity=24
- Richard × Seth: trust=+0.66, warmth=+0.70, familiarity=38
- Nico × Taro: trust=+0.66, warmth=+0.65, familiarity=27
- Khalid × Kurt: trust=+0.66, warmth=+0.44, familiarity=33
- Kali × Wanjiku: trust=+0.65, warmth=+0.54, familiarity=22
### Highest Resentment
- Eduardo × Priya: resentment_ab=0.00, resentment_ba=0.49, trust=+0.18
- Violet × Yasmin: resentment_ab=0.00, resentment_ba=0.47, trust=+0.00
- Eve × Knox: resentment_ab=0.00, resentment_ba=0.42, trust=+0.00
- Bakari × Donna: resentment_ab=0.00, resentment_ba=0.42, trust=+0.00
- Nathan × Vlad: resentment_ab=0.00, resentment_ba=0.42, trust=+0.00
- Daniela × Eve: resentment_ab=0.41, resentment_ba=0.00, trust=+0.00
- Coral × Thea: resentment_ab=0.40, resentment_ba=0.00, trust=+0.00
- James × Wren: resentment_ab=0.38, resentment_ba=0.00, trust=+0.16
- Chloe × Mariama: resentment_ab=0.36, resentment_ba=0.00, trust=+0.00
- Hiroshi × Isaac: resentment_ab=0.00, resentment_ba=0.35, trust=+0.00
### Biggest Changes
- Emilio × Wyatt: Δtrust=+0.14, Δwarmth=+0.17, Δfamiliarity=6, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.28
- Maeve × Yasmin: Δtrust=-0.04, Δwarmth=-0.03, Δfamiliarity=1, Δresentment=+0.12, Δgrievance=+0.14, Δdebt=+0.00
- Nathan × Simon: Δtrust=+0.09, Δwarmth=+0.10, Δfamiliarity=4, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
- Caspian × Neha: Δtrust=+0.09, Δwarmth=+0.09, Δfamiliarity=4, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
- Bianca × Maryam: Δtrust=+0.08, Δwarmth=+0.09, Δfamiliarity=4, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
- Adriana × Nathan: Δtrust=+0.05, Δwarmth=+0.07, Δfamiliarity=4, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
- Katya × Ling: Δtrust=+0.06, Δwarmth=+0.09, Δfamiliarity=2, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
- Daphne × Jin: Δtrust=+0.06, Δwarmth=+0.08, Δfamiliarity=2, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
- Anya × Rafael: Δtrust=+0.06, Δwarmth=+0.07, Δfamiliarity=2, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
- Elara × Lorenzo: Δtrust=+0.06, Δwarmth=+0.07, Δfamiliarity=2, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.15
## Coalitions
- Mutual Aid Ring: members=76, issue=community care, avg_loyalty=1.00, avg_injustice=0.21, avg_economic=0.92, avg_secrecy=0.28
- Tenant Defense Network: members=50, issue=family safety, avg_loyalty=1.00, avg_injustice=0.24, avg_economic=0.93, avg_secrecy=0.31
- Grid Union: members=45, issue=industrial fallout, avg_loyalty=1.00, avg_injustice=0.18, avg_economic=0.98, avg_secrecy=0.22
- Small Business Circle: members=28, issue=livelihood strain, avg_loyalty=1.00, avg_injustice=0.22, avg_economic=0.96, avg_secrecy=0.30
- City Hall Caucus: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.10, avg_economic=0.20, avg_secrecy=0.56
- Redevelopment Board: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.11, avg_economic=0.33, avg_secrecy=0.46
- Campus Action Network: members=26, issue=public organizing, avg_loyalty=1.00, avg_injustice=0.27, avg_economic=0.40, avg_secrecy=0.44
- Care Network: members=25, issue=medical overload, avg_loyalty=0.98, avg_injustice=0.25, avg_economic=0.31, avg_secrecy=0.43
- Harbor Families: members=17, issue=waterfront survival, avg_loyalty=1.00, avg_injustice=0.26, avg_economic=0.98, avg_secrecy=0.30

## Representative Final Minds
### Caspian [healthcare]
- Action: REST at north_homes
- Thought: Private read: The real danger is not only the opposition; it is what fear makes our own people agree to in private.
Primary concern: keep our side from panicking into bad bargains
Ongoing story: help always comes with a price
Blame focus: circumstances
Likely support target: gage
Attachment: anxious attachment | Coping: caretake first | Threat lens: scarcity
Core need: usefulness | Shame trigger: freezing when people count on me
Care style: practical fixing | Conflict style: go sharp
Mask tendency: dutiful calm | Self-story: guardian | Longing: keep people alive without losing myself
Coalitions: care_network
Economic pressure: 0.38 | Loyalty pressure: 0.98 | Secrecy pressure: 0.28
Private burden: none
Priority motive: protect other people
Mask: stays busy with triage so nobody notices the shake
Action style: steady triage
Inner voice: I need to keep our side from panicking into bad bargains. The real danger is not only the opposition; it is what fear makes our own people agree to in private. What would really undo me is freezing when people count on me. Under that, I just want to keep people alive without losing myself. Right now I am leaning toward protect other people.
- Strongest tie: Gage (trust=+0.47, warmth=+0.55, resentment=0.00)
- Futures: If nothing changes, Caspian keeps trying to keep our side from panicking into bad bargains through steady triage.
### Katya [community]
- Action: REST at south_homes
- Thought: Private read: What matters now is time: one more paycheck, one more bill delayed, one more reason the walls do not move yet.
Primary concern: buy one more month before the floor gives way
Ongoing story: help always comes with a price
Blame focus: circumstances
Likely support target: anna
Attachment: secure attachment | Coping: seek witnesses | Threat lens: scarcity
Core need: justice | Shame trigger: failing family
Care style: protective provisioning | Conflict style: appease first
Mask tendency: soft warmth | Self-story: witness | Longing: keep my people connected
Coalitions: none
Economic pressure: 1.00 | Loyalty pressure: 0.99 | Secrecy pressure: 0.08
Private burden: borrowed grocery money and keeps promising next week
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to buy one more month before the floor gives way. What matters now is time: one more paycheck, one more bill delayed, one more reason the walls do not move yet. What would really undo me is failing family. Under that, I just want to keep my people connected. Right now I am leaning toward hold the bloc.
- Strongest tie: Anna (trust=+0.48, warmth=+0.57, resentment=0.00)
- Futures: If nothing changes, Katya keeps trying to buy one more month before the floor gives way through hustling restraint.
### Tom [market_vendor]
- Action: REST at north_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Ongoing story: help always comes with a price
Blame focus: circumstances
Likely support target: hugo
Attachment: anxious attachment | Coping: caretake first | Threat lens: scarcity
Core need: usefulness | Shame trigger: failing family
Care style: practical fixing | Conflict style: keep score
Mask tendency: dutiful calm | Self-story: provider | Longing: make it through the month intact
Coalitions: mutual_aid_ring
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.29
Private burden: none
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. What would really undo me is failing family. Under that, I just want to make it through the month intact. Right now I am leaning toward protect other people.
- Strongest tie: Samira (trust=+0.26, warmth=+0.25, resentment=0.00)
- Futures: If nothing changes, Tom keeps trying to keep income and home secure through hustling restraint.
### Kofi [factory_worker]
- Action: REST at north_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Ongoing story: security is one bad week from collapse
Blame focus: circumstances
Likely support target: anita
Attachment: anxious attachment | Coping: disappear into work | Threat lens: scarcity
Core need: belonging | Shame trigger: being talked down to
Care style: practical fixing | Conflict style: go sharp
Mask tendency: dutiful calm | Self-story: guardian | Longing: make it through without bending
Coalitions: grid_union, mutual_aid_ring
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.19
Private burden: none
Priority motive: hold the bloc
Mask: tries to outrun debt by becoming more useful per hour
Action style: double-shift numbness
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. What would really undo me is being talked down to. Under that, I just want to make it through without bending. Right now I am leaning toward hold the bloc.
- Strongest tie: Hassan (trust=+0.18, warmth=+0.10, resentment=0.00)
- Futures: If nothing changes, Kofi keeps trying to defend my neighborhood from the spillover through double-shift numbness.
### Halima [dock_worker]
- Action: REST at south_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Ongoing story: help always comes with a price
Blame focus: the bills
Likely support target: nina
Attachment: guarded attachment | Coping: keep score quietly | Threat lens: scarcity
Core need: safety | Shame trigger: failing family
Care style: practical fixing | Conflict style: straight negotiation
Mask tendency: command presence | Self-story: guardian | Longing: hold the line without selling anyone out
Coalitions: harbor_families
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.51
Private burden: is behind on a boat payment nobody else in the family knows about
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. What would really undo me is failing family. Under that, I just want to hold the line without selling anyone out. Right now I am leaning toward protect other people.
- Strongest tie: Nina (trust=+0.70, warmth=+0.41, resentment=0.00)
- Futures: If nothing changes, Halima keeps trying to defend my neighborhood from the spillover through hustling restraint.
### Akiko [government_worker]
- Action: REST at north_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Ongoing story: help always comes with a price
Blame focus: uncertainty
Likely support target: nobody
Attachment: secure attachment | Coping: perform competence | Threat lens: chaos
Core need: truth | Shame trigger: losing bureaucratic control
Care style: practical fixing | Conflict style: command
Mask tendency: polished competence | Self-story: fixer | Longing: keep the paperwork from turning into a verdict
Coalitions: none
Economic pressure: 0.07 | Loyalty pressure: 1.00 | Secrecy pressure: 0.53
Private burden: none
Priority motive: hold the bloc
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. What would really undo me is losing bureaucratic control. Under that, I just want to keep the paperwork from turning into a verdict. Right now I am leaning toward hold the bloc.
- Strongest tie: Hassan (trust=+0.06, warmth=+0.09, resentment=0.00)
- Futures: If nothing changes, Akiko keeps trying to force public accountability through careful omission.
### Penn [office_professional]
- Action: REST at north_homes
- Thought: Private read: Feeling it is not enough; it has to become numbers, witnesses, and something the room cannot privately dismiss.
Primary concern: turn private anger into organized pressure
Ongoing story: if I stop holding the room, it tilts
Blame focus: uncertainty
Likely support target: rania
Attachment: guarded attachment | Coping: perform competence | Threat lens: chaos
Core need: control | Shame trigger: needing help in public
Care style: practical fixing | Conflict style: command
Mask tendency: polished competence | Self-story: climber | Longing: be taken seriously
Coalitions: none
Economic pressure: 0.24 | Loyalty pressure: 1.00 | Secrecy pressure: 0.18
Private burden: none
Priority motive: save face
Mask: stays polished while scanning openings
Action style: career triangulation
Inner voice: I need to turn private anger into organized pressure. Feeling it is not enough; it has to become numbers, witnesses, and something the room cannot privately dismiss. What would really undo me is needing help in public. Under that, I just want to be taken seriously. Right now I am leaning toward save face.
- Strongest tie: Rania (trust=+0.14, warmth=+0.23, resentment=0.00)
- Futures: If nothing changes, Penn keeps trying to turn private anger into organized pressure through career triangulation.
### Remy [student]
- Action: REST at south_homes
- Thought: Private read: Feeling it is not enough; it has to become numbers, witnesses, and something the room cannot privately dismiss.
Primary concern: turn private anger into organized pressure
Ongoing story: the story can be stolen if I do not frame it
Blame focus: circumstances
Likely support target: deepa
Attachment: anxious attachment | Coping: reach for connection | Threat lens: abandonment
Core need: belonging | Shame trigger: looking naive
Care style: emotional reassurance | Conflict style: triangulate the room
Mask tendency: joke through it | Self-story: witness | Longing: not be disposable
Coalitions: none
Economic pressure: 0.21 | Loyalty pressure: 1.00 | Secrecy pressure: 0.44
Private burden: none
Priority motive: hold the bloc
Mask: looks curious rather than hungry while testing what can be said aloud
Action style: narrative positioning
Inner voice: I need to turn private anger into organized pressure. Feeling it is not enough; it has to become numbers, witnesses, and something the room cannot privately dismiss. What would really undo me is looking naive. Under that, I just want to not be disposable. Right now I am leaning toward hold the bloc.
- Strongest tie: Deepa (trust=+0.69, warmth=+0.80, resentment=0.00)
- Futures: If nothing changes, Remy keeps trying to turn private anger into organized pressure through narrative positioning.

## Where We Lack
- After the scripted crisis arc, the town still does not generate enough fresh macro situations on its own. The new event engine extends the tail, but the society still needs stronger institutional and faction mechanics.
- Most behavior is still routine (`WORK`/`REST`/`IDLE`). Emotional individuality is stronger now, but action diversity is still too low for a world this large.
- Conflict loops are stronger than before, but they still resolve too politely. Coalitions harden, yet too little of that pressure converts into durable interpersonal hostility.
- Communication is still mostly unrendered. The sim forms relationships numerically, but only a tiny fraction of interactions become actual dialogue.
- Relationships now track grievances, debts, rivalry, and betrayal, but they still need stronger promise-keeping and named shared-history objects to feel fully lived in.
- Group dynamics are present, but they still need stronger leadership turnover, defection, and faction-level bargaining to feel fully societal.
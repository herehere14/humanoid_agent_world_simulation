# 300-Agent 10-Day Diagnostic

Scenario: Harbor Heatwave and Buyout Crisis
Simulated 300 agents for 10 days (240 ticks) in a single town with 8 districts.

## Headline Findings
- Runtime: 59.37s
- Final relationship pairs: 6242 (from 5279 seeded pairs)
- Fired events: 257
- Ripple events added during sim: 228
- Generated dynamic event mix: {'neighborhood_meeting': 18, 'mutual_aid_hub': 31, 'rumor_wave': 52, 'coalition_caucus': 27, 'boycott_call': 37, 'debt_crunch': 16, 'whistleblower_leak': 8, 'hospital_surge': 13, 'organizing_meeting': 14, 'conflict_flashpoint': 6, 'accountability_hearing': 6}
- Fired dynamic event mix: {'macro_cost_shock': 5, 'neighborhood_meeting': 18, 'mutual_aid_hub': 31, 'rumor_wave': 52, 'coalition_caucus': 27, 'boycott_call': 37, 'debt_crunch': 16, 'whistleblower_leak': 8, 'hospital_surge': 13, 'organizing_meeting': 14, 'conflict_flashpoint': 6, 'accountability_hearing': 6}
- Total resolved pair interactions: 870
- Final dominant concerns: {'defend my neighborhood from the spillover': 59, 'turn private anger into organized pressure': 45, "make the neighborhood's losses impossible to wave away": 40, 'keep the block from fraying into private panic': 26, 'test who is still solid when it counts': 24, 'keep income and home secure': 24, 'force public accountability': 16, 'keep this mess from reaching my block and my bills': 14}
- Final dominant action styles: {'hustling restraint': 146, 'careful omission': 60, 'double-shift numbness': 18, 'career triangulation': 14, 'smiling solvency theater': 12, 'narrative positioning': 11, 'bloc discipline': 8, 'steady triage': 6}

## Daily Arc
- Day 1: events=3, interactions=0, avg_valence=0.50, avg_vulnerability=0.01
- Day 2: events=10, interactions=62, avg_valence=0.50, avg_vulnerability=0.04
- Day 3: events=12, interactions=75, avg_valence=0.50, avg_vulnerability=0.04
- Day 4: events=11, interactions=81, avg_valence=0.50, avg_vulnerability=0.07
- Day 5: events=13, interactions=95, avg_valence=0.50, avg_vulnerability=0.07
- Day 6: events=12, interactions=89, avg_valence=0.50, avg_vulnerability=0.06
- Day 7: events=14, interactions=117, avg_valence=0.50, avg_vulnerability=0.07
- Day 8: events=15, interactions=127, avg_valence=0.50, avg_vulnerability=0.09
- Day 9: events=15, interactions=114, avg_valence=0.50, avg_vulnerability=0.08
- Day 10: events=15, interactions=110, avg_valence=0.50, avg_vulnerability=0.07
- ...
- Day 6: events=12, interactions=89, avg_valence=0.50, avg_vulnerability=0.06
- Day 7: events=14, interactions=117, avg_valence=0.50, avg_vulnerability=0.07
- Day 8: events=15, interactions=127, avg_valence=0.50, avg_vulnerability=0.09
- Day 9: events=15, interactions=114, avg_valence=0.50, avg_vulnerability=0.08
- Day 10: events=15, interactions=110, avg_valence=0.50, avg_vulnerability=0.07

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
- Rahul × Vera: resentment_ab=0.48, resentment_ba=0.00, trust=+0.11
- Emilio × Wyatt: resentment_ab=0.36, resentment_ba=0.47, trust=+0.13
- Violet × Yasmin: resentment_ab=0.00, resentment_ba=0.47, trust=+0.00
- Eve × Knox: resentment_ab=0.00, resentment_ba=0.42, trust=+0.00
- Bakari × Donna: resentment_ab=0.00, resentment_ba=0.42, trust=+0.00
- Nathan × Vlad: resentment_ab=0.00, resentment_ba=0.42, trust=+0.00
- Daniela × Eve: resentment_ab=0.41, resentment_ba=0.00, trust=+0.00
- Coral × Thea: resentment_ab=0.40, resentment_ba=0.00, trust=+0.00
- James × Wren: resentment_ab=0.38, resentment_ba=0.00, trust=+0.16
### Biggest Changes
- Rahul × Vera: Δtrust=+0.05, Δwarmth=+0.14, Δfamiliarity=38, Δresentment=+0.48, Δgrievance=+0.99, Δdebt=+0.36
- Emilio × Wyatt: Δtrust=-0.03, Δwarmth=+0.15, Δfamiliarity=22, Δresentment=+0.47, Δgrievance=+0.70, Δdebt=+0.53
- Greta × Simon: Δtrust=+0.09, Δwarmth=+0.23, Δfamiliarity=19, Δresentment=+0.36, Δgrievance=+0.52, Δdebt=+0.59
- Daniel × Dmitri: Δtrust=+0.18, Δwarmth=+0.22, Δfamiliarity=50, Δresentment=+0.12, Δgrievance=+0.54, Δdebt=+0.41
- Ezra × Rosa: Δtrust=+0.30, Δwarmth=+0.25, Δfamiliarity=35, Δresentment=+0.24, Δgrievance=+0.28, Δdebt=+0.00
- Daphne × Jin: Δtrust=+0.33, Δwarmth=+0.36, Δfamiliarity=34, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.60
- Wilder × Zara: Δtrust=+0.18, Δwarmth=+0.24, Δfamiliarity=10, Δresentment=+0.12, Δgrievance=+0.14, Δdebt=+0.53
- Daniel × Ezra: Δtrust=+0.30, Δwarmth=+0.28, Δfamiliarity=28, Δresentment=+0.12, Δgrievance=+0.14, Δdebt=+0.00
- Boris × Sofia: Δtrust=+0.30, Δwarmth=+0.24, Δfamiliarity=26, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.44
- Katya × Ling: Δtrust=+0.24, Δwarmth=+0.35, Δfamiliarity=16, Δresentment=+0.00, Δgrievance=+0.00, Δdebt=+0.53
## Coalitions
- Mutual Aid Ring: members=76, issue=community care, avg_loyalty=1.00, avg_injustice=0.24, avg_economic=0.93, avg_secrecy=0.30
- Tenant Defense Network: members=50, issue=family safety, avg_loyalty=1.00, avg_injustice=0.27, avg_economic=0.95, avg_secrecy=0.33
- Grid Union: members=45, issue=industrial fallout, avg_loyalty=1.00, avg_injustice=0.27, avg_economic=0.98, avg_secrecy=0.30
- Small Business Circle: members=28, issue=livelihood strain, avg_loyalty=1.00, avg_injustice=0.21, avg_economic=0.97, avg_secrecy=0.29
- City Hall Caucus: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.14, avg_economic=0.27, avg_secrecy=0.62
- Redevelopment Board: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.11, avg_economic=0.37, avg_secrecy=0.47
- Campus Action Network: members=26, issue=public organizing, avg_loyalty=1.00, avg_injustice=0.33, avg_economic=0.48, avg_secrecy=0.56
- Care Network: members=25, issue=medical overload, avg_loyalty=0.99, avg_injustice=0.32, avg_economic=0.43, avg_secrecy=0.41
- Harbor Families: members=17, issue=waterfront survival, avg_loyalty=1.00, avg_injustice=0.33, avg_economic=0.98, avg_secrecy=0.38

## Representative Final Minds
### Sage [healthcare]
- Action: REST at north_homes
- Thought: Private read: The real danger is not only the opposition; it is what fear makes our own people agree to in private.
Primary concern: keep our side from panicking into bad bargains
Ongoing story: the story can be stolen if I do not frame it
Blame focus: circumstances
Likely support target: ulric
Attachment: secure attachment | Coping: disappear into work | Threat lens: scarcity
Core need: control | Shame trigger: freezing when people count on me
Care style: steady presence | Conflict style: appease first
Mask tendency: emotional shutdown | Self-story: guardian | Longing: get one quiet hour where nobody needs triage
Coalitions: care_network
Economic pressure: 0.46 | Loyalty pressure: 1.00 | Secrecy pressure: 0.43
Private burden: none
Priority motive: protect other people
Mask: hides panic inside useful tasks
Action style: task-anesthetizing focus
Inner voice: I need to keep our side from panicking into bad bargains. The real danger is not only the opposition; it is what fear makes our own people agree to in private. What would really undo me is freezing when people count on me. Under that, I just want to get one quiet hour where nobody needs triage. Right now I am leaning toward protect other people.
- Strongest tie: Jin (trust=+0.19, warmth=+0.16, resentment=0.00)
- Futures: If nothing changes, Sage keeps trying to keep our side from panicking into bad bargains through task-anesthetizing focus.
### Paula [factory_worker]
- Action: REST at south_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Ongoing story: security is one bad week from collapse
Blame focus: the bills
Likely support target: darren
Attachment: self-protective attachment | Coping: keep score quietly | Threat lens: scarcity
Core need: safety | Shame trigger: failing family
Care style: steady presence | Conflict style: go sharp
Mask tendency: dutiful calm | Self-story: provider | Longing: be respected without begging
Coalitions: grid_union
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.39
Private burden: covered a cousin's bill with money meant for my own landlord
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. What would really undo me is failing family. Under that, I just want to be respected without begging. Right now I am leaning toward hold the bloc.
- Strongest tie: Juno (trust=+0.10, warmth=+0.20, resentment=0.00)
- Futures: If nothing changes, Paula keeps trying to keep income and home secure through hustling restraint.
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
Economic pressure: 1.00 | Loyalty pressure: 0.99 | Secrecy pressure: 0.28
Private burden: borrowed grocery money and keeps promising next week
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to buy one more month before the floor gives way. What matters now is time: one more paycheck, one more bill delayed, one more reason the walls do not move yet. What would really undo me is failing family. Under that, I just want to keep my people connected. Right now I am leaning toward hold the bloc.
- Strongest tie: Ling (trust=+0.33, warmth=+0.43, resentment=0.00)
- Futures: If nothing changes, Katya keeps trying to buy one more month before the floor gives way through hustling restraint.
### Leo [market_vendor]
- Action: REST at south_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Ongoing story: help always comes with a price
Blame focus: circumstances
Likely support target: bianca
Attachment: anxious attachment | Coping: perform competence | Threat lens: scarcity
Core need: safety | Shame trigger: failing family
Care style: practical fixing | Conflict style: appease first
Mask tendency: dutiful calm | Self-story: provider | Longing: have one season without scrambling
Coalitions: mutual_aid_ring, small_business_circle
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.17
Private burden: is three bad market days from missing the stall lease
Priority motive: hold the bloc
Mask: looks solvent while quietly rebuilding the runway
Action style: smiling solvency theater
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. What would really undo me is failing family. Under that, I just want to have one season without scrambling. Right now I am leaning toward hold the bloc.
- Strongest tie: Bria (trust=+0.36, warmth=+0.23, resentment=0.00)
- Futures: If nothing changes, Leo keeps trying to keep income and home secure through smiling solvency theater.
### Emeka [dock_worker]
- Action: REST at south_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Ongoing story: help always comes with a price
Blame focus: circumstances
Likely support target: fatima
Attachment: self-protective attachment | Coping: keep score quietly | Threat lens: scarcity
Core need: safety | Shame trigger: being pushed off the waterfront
Care style: protective provisioning | Conflict style: keep score
Mask tendency: dutiful calm | Self-story: guardian | Longing: stay rooted where my people know me
Coalitions: grid_union, harbor_families
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.33
Private burden: owes a favor to someone negotiating against the harbor
Priority motive: hold the bloc
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. What would really undo me is being pushed off the waterfront. Under that, I just want to stay rooted where my people know me. Right now I am leaning toward hold the bloc.
- Strongest tie: Ezra (trust=+0.13, warmth=+0.15, resentment=0.00)
- Futures: If nothing changes, Emeka keeps trying to defend my neighborhood from the spillover through hustling restraint.
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
Priority motive: protect other people
Mask: stays polished while scanning openings
Action style: career triangulation
Inner voice: I need to turn private anger into organized pressure. Feeling it is not enough; it has to become numbers, witnesses, and something the room cannot privately dismiss. What would really undo me is needing help in public. Under that, I just want to be taken seriously. Right now I am leaning toward protect other people.
- Strongest tie: Rania (trust=+0.14, warmth=+0.23, resentment=0.00)
- Futures: If nothing changes, Penn keeps trying to turn private anger into organized pressure through career triangulation.
### Emma [government_worker]
- Action: RUMINATE at north_homes
- Thought: Private read: The danger is not only the event itself; it is what happens when everyone starts protecting their own door alone.
Primary concern: keep the block from fraying into private panic
Ongoing story: the story can be stolen if I do not frame it
Blame focus: what I have been hiding
Likely support target: ines
Attachment: guarded attachment | Coping: perform competence | Threat lens: chaos
Core need: truth | Shame trigger: being publicly tied to the cover-up
Care style: practical fixing | Conflict style: keep score
Mask tendency: polished competence | Self-story: fixer | Longing: get through this without becoming the scapegoat
Coalitions: city_hall_caucus
Economic pressure: 0.12 | Loyalty pressure: 1.00 | Secrecy pressure: 0.80
Private burden: promised a neighborhood hearing that was never going to happen
Priority motive: hold the bloc
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to keep the block from fraying into private panic. The danger is not only the event itself; it is what happens when everyone starts protecting their own door alone. What would really undo me is being publicly tied to the cover-up. Under that, I just want to get through this without becoming the scapegoat. Right now I am leaning toward hold the bloc.
- Strongest tie: Thomas (trust=+0.59, warmth=+0.43, resentment=0.00)
- Futures: If nothing changes, Emma keeps trying to keep the block from fraying into private panic through careful omission.
### Blake [student]
- Action: RUMINATE at north_homes
- Thought: Private read: One loose version of the story could expose what I have been trying to keep contained.
Primary concern: keep a damaging secret buried
Ongoing story: the story can be stolen if I do not frame it
Blame focus: what I have been hiding
Likely support target: mikhail
Attachment: self-protective attachment | Coping: seek witnesses | Threat lens: exposure
Core need: truth | Shame trigger: selling out
Care style: strategic problem-solving | Conflict style: moralize in public
Mask tendency: righteous heat | Self-story: witness | Longing: be part of something real
Coalitions: campus_action_network
Economic pressure: 0.49 | Loyalty pressure: 1.00 | Secrecy pressure: 1.00
Private burden: is sitting on screenshots that could break a coalition or save it
Priority motive: hold the bloc
Mask: speaks in careful half-truths
Action style: careful omission
Inner voice: I need to keep a damaging secret buried. One loose version of the story could expose what I have been trying to keep contained. What would really undo me is selling out. Under that, I just want to be part of something real. Right now I am leaning toward hold the bloc.
- Strongest tie: Caleb (trust=+0.13, warmth=+0.21, resentment=0.00)
- Futures: If nothing changes, Blake keeps trying to keep a damaging secret buried through careful omission.

## Where We Lack
- After the scripted crisis arc, the town still does not generate enough fresh macro situations on its own. The new event engine extends the tail, but the society still needs stronger institutional and faction mechanics.
- Most behavior is still routine (`WORK`/`REST`/`IDLE`). Emotional individuality is stronger now, but action diversity is still too low for a world this large.
- Conflict loops are stronger than before, but they still resolve too politely. Coalitions harden, yet too little of that pressure converts into durable interpersonal hostility.
- Communication is still mostly unrendered. The sim forms relationships numerically, but only a tiny fraction of interactions become actual dialogue.
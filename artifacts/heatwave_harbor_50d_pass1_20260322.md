# 300-Agent 50-Day Diagnostic

Scenario: Harbor Heatwave and Buyout Crisis
Simulated 300 agents for 50 days (1200 ticks) in a single town with 8 districts.

## Headline Findings
- Runtime: 237.77s
- Final relationship pairs: 7407 (from 5279 seeded pairs)
- Fired events: 1533
- Ripple events added during sim: 1506
- Generated dynamic event mix: {'neighborhood_meeting': 98, 'mutual_aid_hub': 171, 'rumor_wave': 282, 'coalition_caucus': 133, 'boycott_call': 136, 'debt_crunch': 96, 'whistleblower_leak': 56, 'hospital_surge': 80, 'organizing_meeting': 96, 'accountability_hearing': 92, 'conflict_flashpoint': 266}
- Fired dynamic event mix: {'neighborhood_meeting': 98, 'mutual_aid_hub': 171, 'rumor_wave': 282, 'coalition_caucus': 133, 'boycott_call': 136, 'debt_crunch': 96, 'whistleblower_leak': 56, 'hospital_surge': 80, 'organizing_meeting': 96, 'accountability_hearing': 92, 'conflict_flashpoint': 266}
- Total resolved pair interactions: 6888
- Final dominant concerns: {'defend my neighborhood from the spillover': 157, 'force public accountability': 81, 'keep income and home secure': 26, 'hold my bloc together': 10, 'stop our side from splintering into self-protection': 5, 'keep a damaging secret buried': 5, 'keep the house from feeling the dominoes': 3, 'make sure this cannot be quietly rewritten': 3}
- Final dominant action styles: {'hustling restraint': 137, 'careful omission': 53, 'score-settling focus': 45, 'calculated positioning': 24, 'task-anesthetizing focus': 8, 'bloc discipline': 8, 'witness-seeking candor': 5, 'overfunctioning care': 5}

## Daily Arc
- Day 1: events=0, interactions=0, avg_valence=0.50, avg_vulnerability=0.01
- Day 2: events=10, interactions=72, avg_valence=0.50, avg_vulnerability=0.03
- Day 3: events=12, interactions=118, avg_valence=0.50, avg_vulnerability=0.05
- Day 4: events=12, interactions=108, avg_valence=0.50, avg_vulnerability=0.07
- Day 5: events=13, interactions=137, avg_valence=0.50, avg_vulnerability=0.08
- Day 6: events=13, interactions=130, avg_valence=0.50, avg_vulnerability=0.08
- Day 7: events=15, interactions=144, avg_valence=0.50, avg_vulnerability=0.08
- Day 8: events=16, interactions=160, avg_valence=0.50, avg_vulnerability=0.08
- Day 9: events=16, interactions=147, avg_valence=0.50, avg_vulnerability=0.10
- Day 10: events=15, interactions=127, avg_valence=0.50, avg_vulnerability=0.08
- ...
- Day 46: events=14, interactions=145, avg_valence=0.50, avg_vulnerability=0.09
- Day 47: events=15, interactions=159, avg_valence=0.50, avg_vulnerability=0.09
- Day 48: events=14, interactions=144, avg_valence=0.50, avg_vulnerability=0.09
- Day 49: events=14, interactions=142, avg_valence=0.50, avg_vulnerability=0.09
- Day 50: events=15, interactions=141, avg_valence=0.50, avg_vulnerability=0.10

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
- Gemma × Yuri: trust=+1.00, warmth=+0.90, familiarity=132
- Rafael × Tina: trust=+1.00, warmth=+1.00, familiarity=141
- Sergei × Wren: trust=+1.00, warmth=+1.00, familiarity=226
- Elena × Sung: trust=+1.00, warmth=+1.00, familiarity=102
- Sofia × Sung: trust=+1.00, warmth=+0.69, familiarity=149
- Daniel × Gemma: trust=+1.00, warmth=+1.00, familiarity=165
- Mustafa × Takeshi: trust=+1.00, warmth=+0.79, familiarity=99
- Lakshmi × Wren: trust=+1.00, warmth=+1.00, familiarity=238
- Sterling × Yasmin: trust=+1.00, warmth=+0.69, familiarity=102
- Pearl × Tina: trust=+1.00, warmth=+1.00, familiarity=182
### Highest Resentment
- Freya × Wei: resentment_ab=1.00, resentment_ba=0.39, trust=-1.00
- Andres × Yusuf: resentment_ab=1.00, resentment_ba=0.51, trust=-1.00
- Joseph × Paula: resentment_ab=1.00, resentment_ba=0.34, trust=-0.98
- Arjun × Hiroshi: resentment_ab=1.00, resentment_ba=1.00, trust=-0.92
- Aisha × Khalid: resentment_ab=1.00, resentment_ba=1.00, trust=-1.00
- Tendai × Yuri: resentment_ab=0.35, resentment_ba=1.00, trust=-0.40
- Funke × Rahul: resentment_ab=1.00, resentment_ba=0.83, trust=+0.27
- Rahul × Vera: resentment_ab=1.00, resentment_ba=0.82, trust=-0.96
- Maria × Victor: resentment_ab=1.00, resentment_ba=0.99, trust=-1.00
- Layla × Leo: resentment_ab=0.99, resentment_ba=1.00, trust=-1.00
### Biggest Changes
- Daniel × Dmitri: Δtrust=-1.12, Δwarmth=-1.27, Δfamiliarity=377, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.91
- Sanjay × Tendai: Δtrust=-0.97, Δwarmth=-0.98, Δfamiliarity=379, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.89
- Anya × Wren: Δtrust=-1.14, Δwarmth=-0.88, Δfamiliarity=339, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.86
- Rahul × Yuri: Δtrust=-1.04, Δwarmth=-0.90, Δfamiliarity=312, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Lakshmi × Mariama: Δtrust=-1.10, Δwarmth=-1.18, Δfamiliarity=275, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Rahul × Vera: Δtrust=-1.02, Δwarmth=-0.60, Δfamiliarity=277, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.85
- Amit × Sergei: Δtrust=-0.97, Δwarmth=-0.97, Δfamiliarity=233, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Amit × Lakshmi: Δtrust=-1.12, Δwarmth=-1.12, Δfamiliarity=211, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+1.00
- Elara × Lucia: Δtrust=-1.11, Δwarmth=-1.08, Δfamiliarity=228, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.70
- Tendai × Vera: Δtrust=-0.82, Δwarmth=-0.31, Δfamiliarity=274, Δresentment=+1.00, Δgrievance=+1.00, Δdebt=+0.75
## Coalitions
- Mutual Aid Ring: members=76, issue=community care, avg_loyalty=1.00, avg_injustice=0.53, avg_economic=0.82, avg_secrecy=0.28
- Tenant Defense Network: members=50, issue=family safety, avg_loyalty=1.00, avg_injustice=0.53, avg_economic=0.84, avg_secrecy=0.31
- Grid Union: members=45, issue=industrial fallout, avg_loyalty=1.00, avg_injustice=0.51, avg_economic=0.87, avg_secrecy=0.24
- Small Business Circle: members=28, issue=livelihood strain, avg_loyalty=1.00, avg_injustice=0.46, avg_economic=0.86, avg_secrecy=0.24
- City Hall Caucus: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.23, avg_economic=0.18, avg_secrecy=0.60
- Redevelopment Board: members=27, issue=public accountability, avg_loyalty=1.00, avg_injustice=0.14, avg_economic=0.22, avg_secrecy=0.56
- Campus Action Network: members=26, issue=public organizing, avg_loyalty=1.00, avg_injustice=0.33, avg_economic=0.41, avg_secrecy=0.52
- Care Network: members=25, issue=medical overload, avg_loyalty=1.00, avg_injustice=0.90, avg_economic=0.49, avg_secrecy=0.39
- Harbor Families: members=17, issue=waterfront survival, avg_loyalty=1.00, avg_injustice=0.24, avg_economic=0.80, avg_secrecy=0.31

## Representative Final Minds
### Elara [healthcare]
- Action: REST at north_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: lucia
Likely support target: emilio
Attachment: anxious attachment | Coping: caretake first | Threat lens: abandonment
Core need: control | Shame trigger: failing a patient
Care style: practical fixing | Conflict style: go sharp
Mask tendency: soft warmth | Self-story: witness | Longing: get one quiet hour where nobody needs triage
Coalitions: care_network
Economic pressure: 0.54 | Loyalty pressure: 1.00 | Secrecy pressure: 0.35
Private burden: none
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. What would really undo me is failing a patient. Under that, I just want to get one quiet hour where nobody needs triage. Right now I am leaning toward protect other people.
- Strongest tie: Lucia (trust=-0.98, warmth=-0.81, resentment=0.70)
- Futures: If nothing changes, Elara keeps trying to force public accountability through hustling restraint.
### Dorian [community]
- Action: REST at south_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Blame focus: the bills
Likely support target: james
Attachment: secure attachment | Coping: caretake first | Threat lens: scarcity
Core need: usefulness | Shame trigger: failing family
Care style: practical fixing | Conflict style: keep score
Mask tendency: dutiful calm | Self-story: witness | Longing: keep my people connected
Coalitions: none
Economic pressure: 1.00 | Loyalty pressure: 0.97 | Secrecy pressure: 0.41
Private burden: borrowed grocery money and keeps promising next week
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. What would really undo me is failing family. Under that, I just want to keep my people connected. Right now I am leaning toward protect other people.
- Strongest tie: James (trust=+0.08, warmth=+0.25, resentment=0.00)
- Futures: If nothing changes, Dorian keeps trying to keep income and home secure through hustling restraint.
### Yuri [factory_worker]
- Action: REST at south_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Blame focus: daniel
Likely support target: gemma
Attachment: anxious attachment | Coping: disappear into work | Threat lens: scarcity
Core need: safety | Shame trigger: failing family
Care style: practical fixing | Conflict style: straight negotiation
Mask tendency: quiet shutdown | Self-story: provider | Longing: be respected without begging
Coalitions: grid_union
Economic pressure: 1.00 | Loyalty pressure: 1.00 | Secrecy pressure: 0.34
Private burden: none
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to keep income and home secure. This is not abstract anymore; it threatens the life I'm holding together. What would really undo me is failing family. Under that, I just want to be respected without begging. Right now I am leaning toward protect other people.
- Strongest tie: Rahul (trust=-0.92, warmth=-0.65, resentment=0.00)
- Futures: If nothing changes, Yuri keeps trying to keep income and home secure through hustling restraint.
### Tom [market_vendor]
- Action: REST at north_homes
- Thought: Private read: This is not abstract anymore; it threatens the life I'm holding together.
Primary concern: keep income and home secure
Blame focus: bria
Likely support target: samira
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
- Strongest tie: Samira (trust=+0.82, warmth=+0.67, resentment=0.71)
- Futures: If nothing changes, Tom keeps trying to keep income and home secure through hustling restraint.
### Akiko [government_worker]
- Action: REST at north_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: joseph
Likely support target: joseph
Attachment: secure attachment | Coping: perform competence | Threat lens: chaos
Core need: truth | Shame trigger: losing bureaucratic control
Care style: practical fixing | Conflict style: command
Mask tendency: polished competence | Self-story: fixer | Longing: keep the paperwork from turning into a verdict
Coalitions: none
Economic pressure: 0.49 | Loyalty pressure: 1.00 | Secrecy pressure: 0.49
Private burden: none
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. What would really undo me is losing bureaucratic control. Under that, I just want to keep the paperwork from turning into a verdict. Right now I am leaning toward protect other people.
- Strongest tie: Joseph (trust=+0.48, warmth=+0.62, resentment=0.12)
- Futures: If nothing changes, Akiko keeps trying to force public accountability through hustling restraint.
### Penn [office_professional]
- Action: REST at north_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: uncertainty
Likely support target: rania
Attachment: guarded attachment | Coping: perform competence | Threat lens: chaos
Core need: control | Shame trigger: needing help in public
Care style: practical fixing | Conflict style: command
Mask tendency: polished competence | Self-story: climber | Longing: be taken seriously
Coalitions: none
Economic pressure: 0.12 | Loyalty pressure: 1.00 | Secrecy pressure: 0.18
Private burden: none
Priority motive: save face
Mask: stays polished while scanning openings
Action style: calculated positioning
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. What would really undo me is needing help in public. Under that, I just want to be taken seriously. Right now I am leaning toward save face.
- Strongest tie: Rania (trust=+0.14, warmth=+0.23, resentment=0.00)
- Futures: If nothing changes, Penn keeps trying to force public accountability through calculated positioning.
### Roma [dock_worker]
- Action: REST at south_homes
- Thought: Private read: What hits one block never stays there; if nobody draws a line, home becomes the next casualty.
Primary concern: defend my neighborhood from the spillover
Blame focus: circumstances
Likely support target: yong
Attachment: secure attachment | Coping: disappear into work | Threat lens: humiliation
Core need: belonging | Shame trigger: owing the wrong person
Care style: protective provisioning | Conflict style: straight negotiation
Mask tendency: quiet shutdown | Self-story: guardian | Longing: keep home from being swallowed
Coalitions: harbor_families
Economic pressure: 0.79 | Loyalty pressure: 1.00 | Secrecy pressure: 0.44
Private burden: helped move gear off a pier slated for buyout before the notice was public
Priority motive: protect other people
Mask: acts normal while counting every favor
Action style: hustling restraint
Inner voice: I need to defend my neighborhood from the spillover. What hits one block never stays there; if nobody draws a line, home becomes the next casualty. What would really undo me is owing the wrong person. Under that, I just want to keep home from being swallowed. Right now I am leaning toward protect other people.
- Strongest tie: Halima (trust=+0.26, warmth=+0.08, resentment=0.00)
- Futures: If nothing changes, Roma keeps trying to defend my neighborhood from the spillover through hustling restraint.
### Remy [student]
- Action: REST at south_homes
- Thought: Private read: Private outrage only matters if it becomes collective enough that nobody can wave it away.
Primary concern: force public accountability
Blame focus: circumstances
Likely support target: deepa
Attachment: anxious attachment | Coping: reach for connection | Threat lens: abandonment
Core need: belonging | Shame trigger: looking naive
Care style: emotional reassurance | Conflict style: triangulate the room
Mask tendency: joke through it | Self-story: witness | Longing: not be disposable
Coalitions: none
Economic pressure: 0.21 | Loyalty pressure: 1.00 | Secrecy pressure: 0.36
Private burden: none
Priority motive: hold the bloc
Mask: lets need leak through before pride can stop it
Action style: unguarded checking-in
Inner voice: I need to force public accountability. Private outrage only matters if it becomes collective enough that nobody can wave it away. What would really undo me is looking naive. Under that, I just want to not be disposable. Right now I am leaning toward hold the bloc.
- Strongest tie: Mikhail (trust=+0.12, warmth=+0.29, resentment=0.00)
- Futures: If nothing changes, Remy keeps trying to force public accountability through unguarded checking-in.

## Where We Lack
- Most behavior is still routine (`WORK`/`REST`/`IDLE`). Emotional individuality is stronger now, but action diversity is still too low for a world this large.
- Too many agents collapse into a small set of concerns. The new subjective layer helps, but archetypes still bunch together under the same crisis.
- Communication is still mostly unrendered. The sim forms relationships numerically, but only a tiny fraction of interactions become actual dialogue.
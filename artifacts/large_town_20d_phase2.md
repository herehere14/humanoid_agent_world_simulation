# 300-Agent 50-Day Diagnostic

Simulated 300 agents for 20 days (480 ticks) in a single town with 8 districts.

## Headline Findings
- Runtime: 44.27s
- Final relationship pairs: 4242 (from 3627 seeded pairs)
- Fired events: 227
- Ripple events added during sim: 182
- Generated dynamic event mix: {'rumor_wave': 96, 'accountability_hearing': 28, 'organizing_meeting': 26, 'mutual_aid_hub': 32}
- Fired dynamic event mix: {'rumor_wave': 96, 'accountability_hearing': 28, 'organizing_meeting': 26, 'mutual_aid_hub': 32}
- Total resolved pair interactions: 2002
- Final dominant concerns: {'stay on my feet': 92, 'keep other people steady': 75, 'find someone safe': 46, 'push back against circumstances': 42, 'contain the situation': 26, 'stay employable without looking desperate': 7, 'keep the damage contained': 4, 'keep the situation from splintering': 4}
- Final dominant action styles: {'plainspoken honesty': 196, 'protective caretaking': 75, 'controlled precision': 28, 'earnest reassurance-seeking': 1}

## Daily Arc
- Day 1: events=0, interactions=0, avg_valence=0.50, avg_vulnerability=0.04
- Day 2: events=0, interactions=0, avg_valence=0.50, avg_vulnerability=0.04
- Day 3: events=9, interactions=32, avg_valence=0.50, avg_vulnerability=0.07
- Day 4: events=9, interactions=74, avg_valence=0.47, avg_vulnerability=0.09
- Day 5: events=7, interactions=69, avg_valence=0.47, avg_vulnerability=0.10
- Day 6: events=10, interactions=114, avg_valence=0.48, avg_vulnerability=0.10
- Day 7: events=7, interactions=31, avg_valence=0.48, avg_vulnerability=0.09
- Day 8: events=10, interactions=146, avg_valence=0.49, avg_vulnerability=0.10
- Day 9: events=11, interactions=88, avg_valence=0.49, avg_vulnerability=0.10
- Day 10: events=13, interactions=189, avg_valence=0.50, avg_vulnerability=0.05
- ...
- Day 16: events=10, interactions=126, avg_valence=0.50, avg_vulnerability=0.05
- Day 17: events=9, interactions=114, avg_valence=0.50, avg_vulnerability=0.05
- Day 18: events=10, interactions=119, avg_valence=0.50, avg_vulnerability=0.05
- Day 19: events=9, interactions=114, avg_valence=0.50, avg_vulnerability=0.05
- Day 20: events=10, interactions=128, avg_valence=0.50, avg_vulnerability=0.05

## Sample Communications
- No LLM dialogue samples generated.
## Relationship Formation
### Highest Trust
- Xander × Yuri: trust=+1.00, warmth=+1.00, familiarity=81
- Helen × Peter: trust=+1.00, warmth=+1.00, familiarity=56
- Gemma × Yusuf: trust=+1.00, warmth=+1.00, familiarity=60
- Arjun × Vera: trust=+0.78, warmth=+1.00, familiarity=21
- Alejandro × Nadia: trust=+0.77, warmth=+1.00, familiarity=19
- Finn × Olivia: trust=+0.74, warmth=+1.00, familiarity=38
- Dmitri × Jenny: trust=+0.74, warmth=+1.00, familiarity=45
- Lance × Robert: trust=+0.72, warmth=+0.82, familiarity=33
- Khalid × Xander: trust=+0.72, warmth=+0.82, familiarity=37
- Javier × Simon: trust=+0.70, warmth=+0.63, familiarity=28
### Highest Resentment
- Amara × Mika: resentment_ab=0.00, resentment_ba=0.48, trust=+0.00
- Iris × William: resentment_ab=0.47, resentment_ba=0.00, trust=+0.00
- Lev × Violet: resentment_ab=0.00, resentment_ba=0.47, trust=+0.00
- Lev × Nico: resentment_ab=0.00, resentment_ba=0.43, trust=+0.00
- Rhea × Sakura: resentment_ab=0.43, resentment_ba=0.00, trust=+0.07
- Abel × Lakshmi: resentment_ab=0.00, resentment_ba=0.43, trust=+0.00
- Dorian × Faye: resentment_ab=0.41, resentment_ba=0.00, trust=+0.00
- Haruki × Vikram: resentment_ab=0.33, resentment_ba=0.00, trust=+0.00
- Ellis × Wren: resentment_ab=0.32, resentment_ba=0.00, trust=+0.00
- Rania × Theo: resentment_ab=0.00, resentment_ba=0.32, trust=+0.10
### Biggest Changes
- Gemma × Yusuf: Δtrust=+1.00, Δwarmth=+1.00, Δfamiliarity=60, Δresentment=+0.00
- Xander × Yuri: Δtrust=+0.93, Δwarmth=+0.89, Δfamiliarity=68, Δresentment=+0.00
- Helen × Peter: Δtrust=+0.88, Δwarmth=+0.83, Δfamiliarity=32, Δresentment=+0.00
- Dmitri × Jenny: Δtrust=+0.68, Δwarmth=+0.78, Δfamiliarity=18, Δresentment=+0.00
- Finn × Olivia: Δtrust=+0.66, Δwarmth=+0.84, Δfamiliarity=15, Δresentment=+0.00
- Alejandro × Nadia: Δtrust=+0.65, Δwarmth=+0.83, Δfamiliarity=13, Δresentment=+0.00
- Arjun × Vera: Δtrust=+0.65, Δwarmth=+0.81, Δfamiliarity=13, Δresentment=+0.00
- Yasmin × Yumiko: Δtrust=+0.55, Δwarmth=+0.79, Δfamiliarity=19, Δresentment=+0.00
- Amit × Pearl: Δtrust=+0.45, Δwarmth=+0.72, Δfamiliarity=9, Δresentment=+0.00
- Eamon × Hector: Δtrust=+0.42, Δwarmth=+0.68, Δfamiliarity=12, Δresentment=+0.00

## Representative Final Minds
### Gideon [factory_worker]
- Action: REST at south_homes
- Thought: Private read: This could turn ugly fast, so every move has to buy me a little footing.
Primary concern: stay on my feet
Blame focus: circumstances
Likely support target: victoria
Priority motive: find support
Mask: little masking
Action style: plainspoken honesty
Inner voice: I need to stay on my feet. This could turn ugly fast, so every move has to buy me a little footing. Right now I am leaning toward find support.
- Strongest tie: Victoria (trust=+0.15, warmth=+0.24, resentment=0.00)
- Futures: If nothing changes, Gideon keeps trying to stay on my feet through plainspoken honesty.
### Remy [student]
- Action: REST at north_homes
- Thought: Private read: This lands like betrayal, not bad luck; I keep coming back to circumstances.
Primary concern: push back against circumstances
Blame focus: circumstances
Likely support target: gemma
Priority motive: find support
Mask: little masking
Action style: plainspoken honesty
Inner voice: I need to push back against circumstances. This lands like betrayal, not bad luck; I keep coming back to circumstances. Right now I am leaning toward find support.
- Strongest tie: Gemma (trust=+0.60, warmth=+0.69, resentment=0.00)
- Futures: If nothing changes, Remy keeps trying to push back against circumstances through plainspoken honesty.
### Maryam [market_vendor]
- Action: REST at north_homes
- Thought: Private read: Chaos is the real danger; somebody has to impose order.
Primary concern: contain the situation
Blame focus: circumstances
Likely support target: sakura
Priority motive: take control
Mask: locks emotion behind procedures
Action style: controlled precision
Inner voice: I need to contain the situation. Chaos is the real danger; somebody has to impose order. Right now I am leaning toward take control.
- Strongest tie: Sakura (trust=+0.13, warmth=+0.33, resentment=0.00)
- Futures: If nothing changes, Maryam keeps trying to contain the situation through controlled precision.
### Yumiko [community]
- Action: REST at north_homes
- Thought: Private read: If I lose my composure, other people will pay for it.
Primary concern: keep other people steady
Blame focus: circumstances
Likely support target: yasmin
Priority motive: protect other people
Mask: stays useful so panic stays hidden
Action style: protective caretaking
Inner voice: I need to keep other people steady. If I lose my composure, other people will pay for it. Right now I am leaning toward protect other people.
- Strongest tie: Yasmin (trust=+0.68, warmth=+1.00, resentment=0.00)
- Futures: If nothing changes, Yumiko keeps trying to keep other people steady through protective caretaking.
### Lorenzo [healthcare]
- Action: REST at north_homes
- Thought: Private read: If I lose my composure, other people will pay for it.
Primary concern: keep other people steady
Blame focus: circumstances
Likely support target: sage
Priority motive: protect other people
Mask: stays useful so panic stays hidden
Action style: protective caretaking
Inner voice: I need to keep other people steady. If I lose my composure, other people will pay for it. Right now I am leaning toward protect other people.
- Strongest tie: Adaeze (trust=+0.09, warmth=+0.16, resentment=0.00)
- Futures: If nothing changes, Lorenzo keeps trying to keep other people steady through protective caretaking.
### Seth [office_professional]
- Action: REST at north_homes
- Thought: Private read: If I lose my composure, other people will pay for it.
Primary concern: keep other people steady
Blame focus: circumstances
Likely support target: caleb
Priority motive: protect other people
Mask: stays useful so panic stays hidden
Action style: protective caretaking
Inner voice: I need to keep other people steady. If I lose my composure, other people will pay for it. Right now I am leaning toward protect other people.
- Strongest tie: Miguel (trust=+0.16, warmth=+0.27, resentment=0.00)
- Futures: If nothing changes, Seth keeps trying to keep other people steady through protective caretaking.
### George [dock_worker]
- Action: REST at south_homes
- Thought: Private read: This lands like betrayal, not bad luck; I keep coming back to circumstances.
Primary concern: push back against circumstances
Blame focus: circumstances
Likely support target: samir
Priority motive: protect other people
Mask: little masking
Action style: plainspoken honesty
Inner voice: I need to push back against circumstances. This lands like betrayal, not bad luck; I keep coming back to circumstances. Right now I am leaning toward protect other people.
- Strongest tie: Jake (trust=+0.07, warmth=+0.10, resentment=0.00)
- Futures: If nothing changes, George keeps trying to push back against circumstances through plainspoken honesty.
### Oscar [government_worker]
- Action: REST at north_homes
- Thought: Private read: If I lose my composure, other people will pay for it.
Primary concern: keep other people steady
Blame focus: circumstances
Likely support target: marisol
Priority motive: protect other people
Mask: stays useful so panic stays hidden
Action style: protective caretaking
Inner voice: I need to keep other people steady. If I lose my composure, other people will pay for it. Right now I am leaning toward protect other people.
- Strongest tie: Marisol (trust=+0.11, warmth=+0.20, resentment=0.00)
- Futures: If nothing changes, Oscar keeps trying to keep other people steady through protective caretaking.

## Where We Lack
- Most behavior is still routine (`WORK`/`REST`/`IDLE`). Emotional individuality is stronger now, but action diversity is still too low for a world this large.
- Too many agents collapse into a small set of concerns. The new subjective layer helps, but archetypes still bunch together under the same crisis.
- Communication is still mostly unrendered. The sim forms relationships numerically, but only a tiny fraction of interactions become actual dialogue.
- Relationships now track issue-specific support, conflict, and practical help, but they still lack explicit promises, secrets, debts, and named shared history.
- Group dynamics are shallow. The core sim handles pair interactions, not meetings, factions, family systems, rumor chains, or coalition politics as first-class mechanics.
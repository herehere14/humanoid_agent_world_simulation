---
name: prompt-forest
description: Adaptive prompt routing for planning, verification, critique, code review, and confidence-qualified responses via the Prompt Forest plugin tools.
---

# Prompt Forest

Use `prompt_forest_assist` when the user asks for work that benefits from structured branch routing rather than a single raw draft.

Good triggers:
- planning, rollout, roadmap, milestones, owners, risks, tradeoffs
- verification, auditing, contradiction checks, constraint satisfaction, evidence checks
- code review, bug triage, refactor planning, algorithm comparison
- critique, compare-and-choose, structured brainstorming, confidence-qualified answers

Avoid it for:
- trivial chit-chat
- one-line factual replies when no structure or validation is needed
- cases where the user only wants you to restate or lightly rewrite their text

How to call `prompt_forest_assist`:
- Put the full user request into `task`.
- Use `taskType` only when obvious; otherwise keep `auto`.
- Pass `requiredChecks` for hard requirements such as `confidence`, `owners`, `risks`, or `citations`.
- Pass `expectedKeywords` only for the most important content anchors.
- Put only high-signal context into `metadata` or `contextSeed`.

After the tool returns:
- answer naturally for the user instead of dumping raw metadata
- use the Prompt Forest metadata when it helps, especially `selected_branch`, `reward`, `confidence`, and the trace
- if the user explicitly accepts, rejects, or corrects the result, call `prompt_forest_feedback`

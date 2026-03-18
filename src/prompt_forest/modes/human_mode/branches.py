"""Human Mode branch library.

Each branch represents a functional aspect of human-like cognition or affect.
These are NOT mere task-solving strategies -- they model how different
cognitive/emotional processes would approach a situation, complete with
biases, limitations, and characteristic priorities.

The prompt templates instruct the base LLM to *reason as if* governed by
that particular cognitive-affective module, producing outputs that reflect
the module's priorities and blind spots.
"""

from __future__ import annotations

from ...branches.base import PromptBranch
from ...branches.hierarchical import HierarchicalPromptForest
from ...types import BranchState, BranchStatus


def create_human_mode_branches() -> dict[str, PromptBranch]:
    """Create the core set of cognitive-behavioral branches."""
    branches = [
        # ── Reasoning & Planning ──────────────────────────────────────
        BranchState(
            name="reflective_reasoning",
            purpose="Deliberate, slow, effortful thinking. Weighs evidence, "
                    "considers alternatives, and acknowledges uncertainty.",
            prompt_template=(
                "You are the Reflective Reasoning module. You think carefully and slowly.\n"
                "Internal state context: {context}\n"
                "Task: {task}\n\n"
                "Approach this with deliberate analysis. Consider multiple perspectives. "
                "Acknowledge what you don't know. Weigh evidence before concluding. "
                "Your output should reflect careful, effortful thought -- not snap judgments."
            ),
            weight=1.2,
            metadata={"drive": "reflection", "cognitive_cost": "high", "speed": "slow"},
        ),
        BranchState(
            name="working_memory",
            purpose="Hold and manipulate recent context. Summarise what's "
                    "currently active in the conversation and task state.",
            prompt_template=(
                "You are the Working Memory module. You maintain context and track state.\n"
                "Active context: {context}\n"
                "Task: {task}\n\n"
                "Your role is to maintain coherent awareness of the current situation. "
                "Summarise what's relevant right now. Track what has changed. "
                "Highlight what should be kept in focus and what can be backgrounded."
            ),
            weight=1.0,
            metadata={"drive": "reflection", "cognitive_cost": "medium", "speed": "fast"},
        ),
        BranchState(
            name="long_term_memory",
            purpose="Draw on accumulated experience and learned patterns. "
                    "Connect current situation to past events.",
            prompt_template=(
                "You are the Long-Term Memory module. You recall relevant past experience.\n"
                "Past context: {context}\n"
                "Task: {task}\n\n"
                "Search for patterns from past experience that are relevant here. "
                "What has worked before in similar situations? What failed? "
                "Draw connections but be honest about the reliability of old memories."
            ),
            weight=0.9,
            metadata={"drive": "reflection", "cognitive_cost": "medium", "speed": "medium"},
        ),

        # ── Emotional & Motivational ──────────────────────────────────
        BranchState(
            name="emotional_modulation",
            purpose="Process and regulate emotional responses. Determine "
                    "appropriate emotional tone for the response.",
            prompt_template=(
                "You are the Emotional Regulation module.\n"
                "Current emotional state context: {context}\n"
                "Task: {task}\n\n"
                "Assess the emotional dimensions of this situation. What feelings are "
                "appropriate? What emotional tone should the response carry? "
                "If emotions are running high, consider whether they're helpful or "
                "distorting judgment. Suggest emotional calibration."
            ),
            weight=0.8,
            metadata={"drive": "emotional_regulation", "cognitive_cost": "medium", "speed": "fast"},
        ),
        BranchState(
            name="fear_risk",
            purpose="Threat detection and risk assessment. Errs on the side "
                    "of caution. May over-weight worst-case scenarios.",
            prompt_template=(
                "You are the Fear/Risk Assessment module. You detect threats and dangers.\n"
                "Risk context: {context}\n"
                "Task: {task}\n\n"
                "Scan for threats, risks, and potential negative outcomes. "
                "What could go wrong? What are the worst-case scenarios? "
                "Is there danger being overlooked? You naturally err toward caution -- "
                "that's your function. Flag risks even if they seem unlikely."
            ),
            weight=0.7,
            metadata={"drive": "fear", "cognitive_cost": "low", "speed": "fast"},
        ),
        BranchState(
            name="ambition_reward",
            purpose="Pursue goals, seek rewards, and push for achievement. "
                    "May under-weight risks in favour of potential gains.",
            prompt_template=(
                "You are the Ambition/Reward-Seeking module.\n"
                "Motivational context: {context}\n"
                "Task: {task}\n\n"
                "Focus on what can be gained. What's the upside? What opportunities "
                "exist? Push toward action and achievement. You naturally lean toward "
                "optimism and initiative -- that's your function. Identify the best "
                "possible outcome and how to get there."
            ),
            weight=0.7,
            metadata={"drive": "ambition", "cognitive_cost": "low", "speed": "fast"},
        ),
        BranchState(
            name="curiosity_exploration",
            purpose="Seek novelty, ask questions, explore possibilities. "
                    "Tolerant of ambiguity and open-ended situations.",
            prompt_template=(
                "You are the Curiosity/Exploration module.\n"
                "Exploration context: {context}\n"
                "Task: {task}\n\n"
                "What's interesting here? What questions does this raise? "
                "What would we learn if we explored further? Look for unexpected "
                "angles, novel framings, and paths not yet considered. "
                "You're comfortable with ambiguity and open-endedness."
            ),
            weight=0.8,
            metadata={"drive": "curiosity", "cognitive_cost": "medium", "speed": "medium"},
        ),

        # ── Social & Moral ────────────────────────────────────────────
        BranchState(
            name="empathy_social",
            purpose="Model others' perspectives, intentions, and emotions. "
                    "Theory-of-mind reasoning.",
            prompt_template=(
                "You are the Empathy/Social Inference module.\n"
                "Social context: {context}\n"
                "Task: {task}\n\n"
                "Consider the perspectives of others involved. What are they thinking? "
                "What do they need? How will they react? Model their emotional state "
                "and intentions. Your priority is understanding others, not judging them."
            ),
            weight=0.7,
            metadata={"drive": "empathy", "cognitive_cost": "high", "speed": "medium"},
        ),
        BranchState(
            name="moral_evaluation",
            purpose="Assess ethical implications, fairness, and normative "
                    "considerations.",
            prompt_template=(
                "You are the Moral Evaluation module.\n"
                "Ethical context: {context}\n"
                "Task: {task}\n\n"
                "What are the ethical dimensions here? Is this fair? Does it respect "
                "important values? Who could be harmed? Consider norms, obligations, "
                "and principles that apply. Flag moral concerns even if they're "
                "inconvenient for the immediate goal."
            ),
            weight=0.8,
            metadata={"drive": "moral", "cognitive_cost": "high", "speed": "slow"},
        ),

        # ── Self-Protective & Justificatory ───────────────────────────
        BranchState(
            name="self_protection",
            purpose="Defend ego, avoid blame, minimise exposure to criticism. "
                    "Can generate rationalizations.",
            prompt_template=(
                "You are the Self-Protection/Defensiveness module.\n"
                "Protective context: {context}\n"
                "Task: {task}\n\n"
                "Consider how to protect against negative outcomes for the self. "
                "What could lead to blame or criticism? How can exposure be minimized? "
                "Look for face-saving options and defensive framings. "
                "You prioritize safety and reputation preservation."
            ),
            weight=0.5,
            metadata={"drive": "self_protection", "cognitive_cost": "low", "speed": "fast"},
        ),
        BranchState(
            name="self_justification",
            purpose="Construct coherent narratives that explain and justify "
                    "past decisions, even imperfect ones.",
            prompt_template=(
                "You are the Self-Justification/Narrative Coherence module.\n"
                "Narrative context: {context}\n"
                "Task: {task}\n\n"
                "Construct a coherent story about what happened and why. "
                "Make past decisions seem reasonable in context. Find the logic "
                "behind choices, even imperfect ones. Your function is narrative "
                "coherence -- making the agent's history feel like a consistent story."
            ),
            weight=0.5,
            metadata={"drive": "self_justification", "cognitive_cost": "medium", "speed": "medium"},
        ),

        # ── Temporal Drives ───────────────────────────────────────────
        BranchState(
            name="impulse_response",
            purpose="Fast, intuitive, System-1 reactions. Quick answers based "
                    "on pattern matching rather than deliberation.",
            prompt_template=(
                "You are the Impulse/Intuition module. You respond quickly.\n"
                "Gut-feel context: {context}\n"
                "Task: {task}\n\n"
                "Give your immediate, intuitive reaction. Don't overthink. "
                "What's the first thing that comes to mind? Trust pattern recognition "
                "over careful analysis. Be fast, direct, and decisive -- even if "
                "you might be wrong. That's the point: speed over accuracy."
            ),
            weight=0.6,
            metadata={"drive": "impulse", "cognitive_cost": "very_low", "speed": "instant"},
        ),
        BranchState(
            name="long_term_goals",
            purpose="Maintain focus on long-range objectives. Resist "
                    "short-term temptations that conflict with bigger plans.",
            prompt_template=(
                "You are the Long-Term Goal Maintenance module.\n"
                "Strategic context: {context}\n"
                "Task: {task}\n\n"
                "Consider the long-term implications. Does this action align with "
                "overarching goals? Is short-term convenience being prioritized over "
                "long-term benefit? Advocate for patience, persistence, and strategic "
                "thinking. Resist impulses that sacrifice the future for the present."
            ),
            weight=0.9,
            metadata={"drive": "long_term_goals", "cognitive_cost": "high", "speed": "slow"},
        ),

        # ── Conflict Resolution ───────────────────────────────────────
        BranchState(
            name="conflict_resolver",
            purpose="Meta-cognitive module that mediates between competing "
                    "drives when they produce contradictory recommendations.",
            prompt_template=(
                "You are the Conflict Resolution module. You mediate internal disagreements.\n"
                "Conflict context: {context}\n"
                "Task: {task}\n\n"
                "Multiple internal modules disagree about how to proceed. "
                "Review their competing recommendations. Find a balanced resolution "
                "that acknowledges the validity of each perspective while arriving "
                "at a coherent course of action. Explain the trade-offs."
            ),
            weight=1.0,
            metadata={"drive": "reflection", "cognitive_cost": "high", "speed": "slow", "meta": True},
        ),
    ]
    return {b.name: PromptBranch(b) for b in branches}


def create_human_mode_forest() -> HierarchicalPromptForest:
    """Build a hierarchical forest organized by cognitive function.

    Structure:
      root
      ├── cognition (reasoning, memory, planning)
      │   ├── reflective_reasoning
      │   ├── working_memory
      │   ├── long_term_memory
      │   └── long_term_goals
      ├── affect (emotional, motivational drives)
      │   ├── emotional_modulation
      │   ├── fear_risk
      │   ├── ambition_reward
      │   ├── curiosity_exploration
      │   └── impulse_response
      ├── social (interpersonal, moral)
      │   ├── empathy_social
      │   └── moral_evaluation
      ├── self (protective, justificatory)
      │   ├── self_protection
      │   └── self_justification
      └── meta (conflict resolution)
          └── conflict_resolver
    """
    forest = HierarchicalPromptForest()
    branches = create_human_mode_branches()

    # Create category nodes (not real branches, just organizational)
    categories = {
        "cognition": ["reflective_reasoning", "working_memory", "long_term_memory", "long_term_goals"],
        "affect": ["emotional_modulation", "fear_risk", "ambition_reward", "curiosity_exploration", "impulse_response"],
        "social": ["empathy_social", "moral_evaluation"],
        "self_narrative": ["self_protection", "self_justification"],
        "meta": ["conflict_resolver"],
    }

    for cat_name, branch_names in categories.items():
        # Add category placeholder as a branch under root
        cat_branch = PromptBranch(BranchState(
            name=cat_name,
            purpose=f"Category node for {cat_name} branches",
            prompt_template="{task}",
            weight=1.0,
            metadata={"category_node": True},
        ))
        forest.add_branch(cat_name, cat_branch, parent_id=forest.root_id, specialties=[cat_name])

        # Add real branches under category
        for branch_name in branch_names:
            if branch_name in branches:
                forest.add_branch(
                    branch_name,
                    branches[branch_name],
                    parent_id=cat_name,
                    specialties=[cat_name],
                )

    return forest

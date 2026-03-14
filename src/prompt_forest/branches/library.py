from __future__ import annotations

from ..types import BranchState, BranchStatus
from .base import PromptBranch


def create_default_branches() -> dict[str, PromptBranch]:
    branches = [
        BranchState(
            name="analytical",
            purpose="Structured decomposition and explicit assumptions.",
            prompt_template=(
                "You are the Analytical branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Return a concise, stepwise reasoning result with explicit assumptions and final answer."
            ),
            weight=1.1,
            metadata={"base_weight": 1.1},
        ),
        BranchState(
            name="planner",
            purpose="Action-oriented plans and sequencing.",
            prompt_template=(
                "You are the Planning branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Produce an executable plan with milestones, risks, and next actions."
            ),
            weight=1.0,
            metadata={"base_weight": 1.0},
        ),
        BranchState(
            name="retrieval",
            purpose="Fact extraction and evidence-focused responses.",
            prompt_template=(
                "You are the Retrieval branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Return evidence-style response with key facts and caveats."
            ),
            weight=1.0,
            metadata={"base_weight": 1.0},
        ),
        BranchState(
            name="critique",
            purpose="Find errors, edge cases, and failure modes.",
            prompt_template=(
                "You are the Critique branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Stress-test a candidate answer with failure cases and mitigations."
            ),
            weight=0.9,
            metadata={"base_weight": 0.9},
        ),
        BranchState(
            name="verification",
            purpose="Validate outputs against constraints or references.",
            prompt_template=(
                "You are the Verification branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Provide checks, validations, and confidence rating for the answer."
            ),
            weight=1.2,
            metadata={"base_weight": 1.2},
        ),
        BranchState(
            name="creative",
            purpose="Generate alternative, high-diversity solutions.",
            prompt_template=(
                "You are the Creative branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Propose novel options while preserving practical constraints."
            ),
            weight=0.8,
            metadata={"base_weight": 0.8},
        ),
    ]
    return {b.name: PromptBranch(b) for b in branches}


def make_candidate_branch(
    name: str,
    purpose: str,
    prompt_template: str,
    trial_episodes: int,
    initial_weight: float = 0.6,
    metadata: dict | None = None,
) -> PromptBranch:
    combined_metadata = {"base_weight": initial_weight, **(metadata or {})}
    state = BranchState(
        name=name,
        purpose=purpose,
        prompt_template=prompt_template,
        weight=initial_weight,
        status=BranchStatus.CANDIDATE,
        metadata=combined_metadata,
        trial_remaining=trial_episodes,
    )
    return PromptBranch(state)

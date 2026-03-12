from __future__ import annotations

from dataclasses import dataclass

from ..branches.base import PromptBranch
from ..branches.library import make_candidate_branch
from ..config import OptimizerConfig
from ..memory.store import MemoryStore
from ..types import BranchStatus, EvaluationSignal, RoutingDecision, TaskInput


@dataclass
class OptimizationEvent:
    updated_weights: dict[str, float]
    rewritten_prompts: list[str]
    promoted_candidates: list[str]
    archived_candidates: list[str]
    created_candidates: list[str]


class OptimizerAgent:
    """Agent 2: constrained local updates over active branches only."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def optimize(
        self,
        task: TaskInput,
        route: RoutingDecision,
        signal: EvaluationSignal,
        branches: dict[str, PromptBranch],
        memory: MemoryStore,
    ) -> OptimizationEvent:
        updated: dict[str, float] = {}
        rewritten: list[str] = []
        promoted: list[str] = []
        archived: list[str] = []
        created: list[str] = []

        for branch_name in route.activated_branches:
            branch = branches.get(branch_name)
            if branch is None or not branch.is_active:
                continue

            fb = signal.branch_feedback.get(branch_name)
            reward = fb.reward if fb else 0.0
            branch.apply_reward(reward)

            delta = self.config.learning_rate * (reward - 0.5)
            decay = self.config.weight_decay * (branch.state.weight - 1.0)
            new_weight = branch.state.weight + delta - decay
            branch.state.weight = max(self.config.min_weight, min(self.config.max_weight, new_weight))
            updated[branch_name] = round(branch.state.weight, 4)

            if reward < self.config.prompt_rewrite_threshold and fb:
                branch.rewrite_prompt(fb.suggested_improvement_direction, self.config.max_prompt_variants)
                rewritten.append(branch_name)

            if branch.state.status == BranchStatus.CANDIDATE:
                branch.state.trial_remaining -= 1
                if branch.state.trial_remaining <= 0:
                    avg = branch.state.avg_reward()
                    if avg >= self.config.candidate_promote_threshold:
                        branch.state.status = BranchStatus.ACTIVE
                        branch.state.weight = max(1.0, branch.state.weight)
                        promoted.append(branch_name)
                    else:
                        branch.state.status = BranchStatus.ARCHIVED
                        archived.append(branch_name)

        self._try_create_candidate(task, route, signal, branches, memory, created)

        return OptimizationEvent(
            updated_weights=updated,
            rewritten_prompts=rewritten,
            promoted_candidates=promoted,
            archived_candidates=archived,
            created_candidates=created,
        )

    def _try_create_candidate(
        self,
        task: TaskInput,
        route: RoutingDecision,
        signal: EvaluationSignal,
        branches: dict[str, PromptBranch],
        memory: MemoryStore,
        created: list[str],
    ) -> None:
        failure_map = memory.repeated_failures(min_count=self.config.candidate_failure_trigger)
        if not failure_map:
            return

        active_candidates = [b for b in branches.values() if b.state.status == BranchStatus.CANDIDATE]
        if active_candidates:
            return

        active_rewards = [signal.branch_feedback[b].reward for b in route.activated_branches if b in signal.branch_feedback]
        if not active_rewards or max(active_rewards) > 0.55:
            return

        if len([b for b in branches.values() if b.state.status != BranchStatus.ARCHIVED]) >= self.config.max_active_branches:
            return

        dominant_failure = sorted(failure_map.items(), key=lambda x: x[1], reverse=True)[0][0]
        candidate_name, purpose, template = self._candidate_spec_from_failure(dominant_failure, task.task_type)

        if candidate_name in branches:
            return

        # Duplication guard: avoid creating a branch with near-identical role.
        if any(candidate_name.split("_")[0] in b.state.name for b in branches.values()):
            return

        candidate = make_candidate_branch(
            name=candidate_name,
            purpose=purpose,
            prompt_template=template,
            trial_episodes=self.config.candidate_trial_episodes,
        )
        branches[candidate_name] = candidate
        created.append(candidate_name)

    def _candidate_spec_from_failure(self, failure_reason: str, task_type: str) -> tuple[str, str, str]:
        fr = failure_reason.lower()
        if "keyword_coverage" in fr or "low_quality" in fr:
            return (
                f"evidence_fuser_{task_type}",
                "Fuse retrieval evidence with strict task-keyword anchoring.",
                (
                    "You are the Evidence Fuser candidate branch. Task type: {task_type}.\n"
                    "Task: {task}\n"
                    "Prioritize traceable evidence snippets and guarantee key requirement coverage."
                ),
            )
        if "rule_miss" in fr:
            return (
                f"constraint_solver_{task_type}",
                "Satisfy explicit constraints before producing final response.",
                (
                    "You are the Constraint Solver candidate branch. Task type: {task_type}.\n"
                    "Task: {task}\n"
                    "First enumerate constraints, then produce an answer that verifies each one."
                ),
            )

        return (
            f"verifier_plus_{task_type}",
            "Deep verification and self-check loops for reliability.",
            (
                "You are the Verifier Plus candidate branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Run a strict self-checklist and only return conclusions that pass all checks."
            ),
        )

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
    task_baseline_before: float = 0.5
    task_baseline_after: float = 0.5


class OptimizerAgent:
    """Agent 2: constrained local updates over active branches only."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self._task_baselines: dict[str, float] = {}

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
        task_baseline_before = self._task_baselines.get(route.task_type, 0.5)

        for branch_name in route.activated_branches:
            branch = branches.get(branch_name)
            if branch is None or not branch.is_active:
                continue

            fb = signal.branch_feedback.get(branch_name)
            reward = fb.reward if fb else 0.0
            branch.apply_reward(reward)

            advantage = reward - task_baseline_before
            delta = self.config.learning_rate * advantage
            decay = self.config.weight_decay * (branch.state.weight - 1.0)
            new_weight = branch.state.weight + delta - decay
            branch.state.weight = max(self.config.min_weight, min(self.config.max_weight, new_weight))
            branch.state.metadata["last_advantage"] = round(advantage, 6)
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
                    elif self._should_extend_candidate_trial(avg, branch):
                        branch.state.trial_remaining = self.config.candidate_extension_episodes
                    else:
                        branch.state.status = BranchStatus.ARCHIVED
                        archived.append(branch_name)

        self._try_create_candidate(task, route, signal, branches, memory, created)
        task_baseline_after = self._update_task_baseline(route.task_type, signal.reward_score)

        return OptimizationEvent(
            updated_weights=updated,
            rewritten_prompts=rewritten,
            promoted_candidates=promoted,
            archived_candidates=archived,
            created_candidates=created,
            task_baseline_before=round(task_baseline_before, 4),
            task_baseline_after=round(task_baseline_after, 4),
        )

    def _update_task_baseline(self, task_type: str, reward: float) -> float:
        old = self._task_baselines.get(task_type, 0.5)
        beta = self.config.advantage_baseline_beta
        new = (1.0 - beta) * old + beta * reward
        self._task_baselines[task_type] = new
        return new

    def _should_extend_candidate_trial(self, avg_reward: float, branch: PromptBranch) -> bool:
        lower = self.config.candidate_promote_threshold - self.config.candidate_neutral_band
        upper = self.config.candidate_promote_threshold + self.config.candidate_neutral_band
        if not (lower <= avg_reward <= upper):
            return False

        used = int(branch.state.metadata.get("trial_extensions_used", 0))
        if used >= self.config.candidate_max_extensions:
            return False

        branch.state.metadata["trial_extensions_used"] = used + 1
        return True

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
        if len(active_candidates) >= self.config.max_active_candidates:
            return

        active_rewards = [signal.branch_feedback[b].reward for b in route.activated_branches if b in signal.branch_feedback]
        if not active_rewards or max(active_rewards) > 0.55:
            return

        if len([b for b in branches.values() if b.state.status != BranchStatus.ARCHIVED]) >= self.config.max_active_branches:
            return

        dominant_failure = sorted(failure_map.items(), key=lambda x: x[1], reverse=True)[0][0]
        base_name, capability_tag, purpose, template = self._candidate_spec_from_failure(dominant_failure, task.task_type)
        parent_hint = route.activated_branches[-1] if route.activated_branches else ""

        if self._capability_exists_under_parent(branches, capability_tag, parent_hint):
            return

        candidate_name = self._unique_candidate_name(base_name, branches)
        candidate = make_candidate_branch(
            name=candidate_name,
            purpose=purpose,
            prompt_template=template,
            trial_episodes=self.config.candidate_trial_episodes,
            initial_weight=self.config.candidate_initial_weight,
            metadata={
                "capability_tag": capability_tag,
                "creation_reason": dominant_failure,
                "task_type": task.task_type,
                "parent_hint": parent_hint,
            },
        )
        branches[candidate_name] = candidate
        created.append(candidate_name)

    def _candidate_spec_from_failure(self, failure_reason: str, task_type: str) -> tuple[str, str, str, str]:
        fr = failure_reason.lower()
        if "keyword_coverage" in fr or "low_quality" in fr:
            return (
                f"evidence_fuser_{task_type}",
                "evidence_fuser",
                "Fuse retrieval evidence with strict task-keyword anchoring.",
                (
                    "You are the Evidence Fuser candidate branch. Task type: {task_type}.\n"
                    "Task: {task}\n"
                    "Parent context: {context}\n"
                    "Prioritize traceable evidence snippets and guarantee key requirement coverage."
                ),
            )
        if "rule_miss" in fr:
            return (
                f"constraint_solver_{task_type}",
                "constraint_solver",
                "Satisfy explicit constraints before producing final response.",
                (
                    "You are the Constraint Solver candidate branch. Task type: {task_type}.\n"
                    "Task: {task}\n"
                    "Parent context: {context}\n"
                    "First enumerate constraints, then produce an answer that verifies each one."
                ),
            )

        return (
            f"verifier_plus_{task_type}",
            "verifier_plus",
            "Deep verification and self-check loops for reliability.",
            (
                "You are the Verifier Plus candidate branch. Task type: {task_type}.\n"
                "Task: {task}\n"
                "Parent context: {context}\n"
                "Run a strict self-checklist and only return conclusions that pass all checks."
            ),
        )

    @staticmethod
    def _unique_candidate_name(base: str, branches: dict[str, PromptBranch]) -> str:
        if base not in branches:
            return base
        i = 2
        while f"{base}_v{i}" in branches:
            i += 1
        return f"{base}_v{i}"

    @staticmethod
    def _capability_exists_under_parent(
        branches: dict[str, PromptBranch],
        capability_tag: str,
        parent_hint: str,
    ) -> bool:
        for branch in branches.values():
            if branch.state.status == BranchStatus.ARCHIVED:
                continue
            meta = branch.state.metadata
            if meta.get("capability_tag") != capability_tag:
                continue
            if parent_hint and meta.get("parent_hint") != parent_hint:
                continue
            return True
        return False

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..branches.base import PromptBranch
from ..branches.library import make_candidate_branch
from ..config import OptimizerConfig
from ..memory.store import MemoryStore
from ..types import BranchStatus, EvaluationSignal, RoutingDecision, TaskInput
from .llm_optimizer_advisor import LLMOptimizerAdvisor


@dataclass
class OptimizationEvent:
    updated_weights: dict[str, float]
    update_details: dict[str, dict[str, Any]]
    rewritten_prompts: list[str]
    promoted_candidates: list[str]
    archived_candidates: list[str]
    created_candidates: list[str]
    task_baseline_before: float = 0.5
    task_baseline_after: float = 0.5
    advisor_used: bool = False
    advisor_error: str = ""


class OptimizerAgent:
    """Agent 2: constrained local updates over active branches only."""

    def __init__(self, config: OptimizerConfig, advisor: LLMOptimizerAdvisor | None = None) -> None:
        self.config = config
        self._task_baselines: dict[str, float] = {}
        self._branch_task_baselines: dict[str, float] = {}
        self._advisor = advisor
        self._episode_idx = 0

    def optimize(
        self,
        task: TaskInput,
        route: RoutingDecision,
        signal: EvaluationSignal,
        branches: dict[str, PromptBranch],
        memory: MemoryStore,
    ) -> OptimizationEvent:
        self._episode_idx += 1
        updated: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        rewritten: list[str] = []
        promoted: list[str] = []
        archived: list[str] = []
        created: list[str] = []
        task_baseline_before = self._task_baselines.get(route.task_type, 0.5)
        user_id = str(task.metadata.get("user_id", "global")).strip() or "global"
        llm_runtime_active = bool(task.metadata.get("llm_runtime_active", False))

        advisor_directives: dict[str, dict[str, Any]] = {}
        advisor_proposals: list[dict[str, str]] = []
        advisor_error = ""
        advisor_used = False
        if self._advisor and self._advisor.is_enabled():
            advisor_used = True
            try:
                advisor_payload = self._advisor.advise(task, route, signal, branches)
                advisor_directives = self._extract_advisor_directives(advisor_payload, route)
                advisor_proposals = self._extract_advisor_candidate_proposals(advisor_payload, route)
            except Exception as exc:  # pragma: no cover - depends on external API runtime
                advisor_error = str(exc)

        for branch_name in route.activated_branches:
            branch = branches.get(branch_name)
            if branch is None or not branch.is_active:
                continue

            fb = signal.branch_feedback.get(branch_name)
            reward = fb.reward if fb else 0.0
            branch.apply_reward(reward)

            status_before = branch.state.status.value
            old_weight = branch.state.weight
            advice = advisor_directives.get(branch_name, {})

            branch_baseline_key = self._branch_baseline_key(route.task_type, branch_name)
            branch_baseline_before = self._branch_task_baselines.get(branch_baseline_key, task_baseline_before)
            baseline_mix = self._clamp01(self.config.branch_advantage_mix)
            combined_baseline_before = ((1.0 - baseline_mix) * task_baseline_before) + (
                baseline_mix * branch_baseline_before
            )

            advantage = reward - combined_baseline_before
            fb_conf = self._clamp01(fb.confidence if fb else 0.5)
            confidence_scale = max(0.0, min(1.0, (fb_conf - 0.4) / 0.6))
            variance_scale, reward_variance, reward_count = self._variance_scale_for_update(
                memory=memory,
                branch_name=branch_name,
                task_type=route.task_type,
                user_id=user_id,
                llm_runtime_active=llm_runtime_active,
            )

            # Damp noisy updates from low-confidence branch feedback.
            delta = self.config.learning_rate * advantage * confidence_scale * variance_scale

            advice_conf = self._clamp01(advice.get("confidence", 0.0))
            advice_weight_threshold = max(0.55, self.config.advisor_rewrite_confidence_threshold - 0.15)
            advisory_extra_delta = self._bounded_advisory_delta(advice.get("extra_weight_delta", 0.0))
            if advice_conf < advice_weight_threshold:
                advisory_extra_delta = 0.0
            else:
                advisory_extra_delta *= ((advice_conf - advice_weight_threshold) / max(1e-8, 1.0 - advice_weight_threshold))
                advisory_extra_delta *= confidence_scale
                advisory_extra_delta *= variance_scale

            total_delta = delta + advisory_extra_delta
            decay = self.config.weight_decay * (old_weight - 1.0)
            raw_weight = old_weight + total_delta - decay
            new_weight = raw_weight
            branch.state.weight = max(self.config.min_weight, min(self.config.max_weight, new_weight))
            branch.state.metadata["last_advantage"] = round(advantage, 6)

            prompt_rewritten = False
            advice_rewrite_hint = str(advice.get("rewrite_hint", "")).strip()
            rewrite_hint = fb.suggested_improvement_direction if fb else "improve_clarity_and_verification"
            if advice_rewrite_hint and advice_conf >= self.config.advisor_rewrite_confidence_threshold:
                rewrite_hint = advice_rewrite_hint

            should_rewrite = self._should_rewrite_branch(
                branch=branch,
                reward=reward,
                advice_rewrite_hint=advice_rewrite_hint,
                advice_conf=advice_conf,
            )

            if should_rewrite:
                branch.rewrite_prompt(rewrite_hint, self.config.max_prompt_variants)
                rewritten.append(branch_name)
                prompt_rewritten = True

            accepted_update = self._passes_acceptance_gate(
                memory=memory,
                branch_name=branch_name,
                task_type=route.task_type,
                user_id=user_id,
                old_weight=old_weight,
                new_weight=branch.state.weight,
                prompt_rewritten=prompt_rewritten,
            )
            if not accepted_update:
                branch.state.weight = old_weight
                if prompt_rewritten:
                    branch.rollback_prompt()
                    if branch_name in rewritten:
                        rewritten.remove(branch_name)
                prompt_rewritten = False
            updated[branch_name] = round(branch.state.weight, 4)
            branch_baseline_after = self._update_branch_baseline(
                key=branch_baseline_key,
                reward=reward,
                default=combined_baseline_before,
            )

            if branch.state.status == BranchStatus.CANDIDATE:
                self._update_candidate_parent_comparison(branch_name, branch, route, signal)
                branch.state.trial_remaining -= 1
                if branch.state.trial_remaining <= 0:
                    avg = branch.state.avg_reward()
                    if avg >= self.config.candidate_promote_threshold and self._passes_parent_gate(branch):
                        branch.state.status = BranchStatus.ACTIVE
                        branch.state.weight = max(1.0, branch.state.weight)
                        promoted.append(branch_name)
                    elif self._should_extend_candidate_trial(avg, branch):
                        branch.state.trial_remaining = self.config.candidate_extension_episodes
                    else:
                        branch.state.status = BranchStatus.ARCHIVED
                        archived.append(branch_name)

            details[branch_name] = {
                "status_before": status_before,
                "status_after": branch.state.status.value,
                "reward": round(reward, 4),
                "task_baseline_before": round(task_baseline_before, 4),
                "branch_baseline_before": round(branch_baseline_before, 4),
                "combined_baseline_before": round(combined_baseline_before, 4),
                "branch_baseline_after": round(branch_baseline_after, 4),
                "advantage": round(advantage, 4),
                "delta": round(delta, 4),
                "advisory_extra_delta": round(advisory_extra_delta, 4),
                "total_delta": round(total_delta, 4),
                "decay": round(decay, 4),
                "feedback_confidence": round(fb_conf, 4),
                "confidence_scale": round(confidence_scale, 4),
                "variance_scale": round(variance_scale, 4),
                "reward_variance": round(reward_variance, 6),
                "reward_count": round(reward_count, 4),
                "llm_runtime_active": llm_runtime_active,
                "advisor_confidence": round(advice_conf, 4),
                "old_weight": round(old_weight, 4),
                "raw_weight_after_update": round(raw_weight, 4),
                "new_weight": round(branch.state.weight, 4),
                "prompt_rewritten": prompt_rewritten,
                "advisory_rewrite_hint_used": bool(advice_rewrite_hint and rewrite_hint == advice_rewrite_hint),
                "update_accepted": accepted_update,
                "low_reward_streak": int(branch.state.metadata.get("low_reward_streak", 0)),
                "trial_remaining": branch.state.trial_remaining,
            }

        self._try_create_candidates(
            task=task,
            route=route,
            signal=signal,
            branches=branches,
            memory=memory,
            created=created,
            advisor_proposals=advisor_proposals,
        )
        task_baseline_after = self._update_task_baseline(route.task_type, signal.reward_score)

        return OptimizationEvent(
            updated_weights=updated,
            update_details=details,
            rewritten_prompts=rewritten,
            promoted_candidates=promoted,
            archived_candidates=archived,
            created_candidates=created,
            task_baseline_before=round(task_baseline_before, 4),
            task_baseline_after=round(task_baseline_after, 4),
            advisor_used=advisor_used,
            advisor_error=advisor_error,
        )

    def _update_task_baseline(self, task_type: str, reward: float) -> float:
        old = self._task_baselines.get(task_type, 0.5)
        beta = self.config.advantage_baseline_beta
        new = (1.0 - beta) * old + beta * reward
        self._task_baselines[task_type] = new
        return new

    def _update_branch_baseline(self, key: str, reward: float, default: float) -> float:
        old = self._branch_task_baselines.get(key, default)
        beta = self.config.branch_baseline_beta
        new = (1.0 - beta) * old + beta * reward
        self._branch_task_baselines[key] = new
        return new

    @staticmethod
    def _branch_baseline_key(task_type: str, branch_name: str) -> str:
        return f"{task_type}::{branch_name}"

    def _variance_scale_for_update(
        self,
        memory: MemoryStore,
        branch_name: str,
        task_type: str,
        user_id: str,
        llm_runtime_active: bool,
    ) -> tuple[float, float, float]:
        if not llm_runtime_active:
            return 1.0, 0.0, 0.0

        moments = memory.branch_reward_moments(
            branch_name=branch_name,
            task_type=task_type,
            user_id=user_id,
            limit=self.config.update_acceptance_window * 3,
        )
        variance = max(0.0, float(moments.get("variance", 0.0)))
        count = max(0.0, float(moments.get("count", 0.0)))
        if count <= 0.0:
            return 1.0, variance, count

        raw_scale = 1.0 / (1.0 + (self.config.llm_variance_sensitivity * variance))
        raw_scale = max(self.config.llm_min_variance_scale, min(1.0, raw_scale))
        shrink = count / (count + 8.0)
        scale = (1.0 - shrink) + (shrink * raw_scale)
        return max(self.config.llm_min_variance_scale, min(1.0, scale)), variance, count

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

    def _try_create_candidates(
        self,
        task: TaskInput,
        route: RoutingDecision,
        signal: EvaluationSignal,
        branches: dict[str, PromptBranch],
        memory: MemoryStore,
        created: list[str],
        advisor_proposals: list[dict[str, str]],
    ) -> None:
        failure_map = memory.repeated_failures(min_count=self.config.candidate_failure_trigger)
        if not failure_map:
            return
        if not self._has_clustered_failures(failure_map):
            return

        active_candidates = [b for b in branches.values() if b.state.status == BranchStatus.CANDIDATE]
        if len(active_candidates) >= self.config.max_active_candidates:
            return

        active_rewards = [signal.branch_feedback[b].reward for b in route.activated_branches if b in signal.branch_feedback]
        if not active_rewards or max(active_rewards) > 0.55:
            return

        active_non_archived = [b for b in branches.values() if b.state.status != BranchStatus.ARCHIVED]
        if len(active_non_archived) >= self.config.max_active_branches:
            return

        parent_hint = route.activated_branches[-1] if route.activated_branches else ""
        dominant_failure = sorted(failure_map.items(), key=lambda x: x[1], reverse=True)[0][0]
        specs = advisor_proposals or self._candidate_specs_from_failure(dominant_failure, task.task_type, parent_hint)
        if not advisor_proposals and self.config.candidate_spawn_per_event <= 1 and specs:
            # Conservative default: keep legacy behavior (one candidate family per failure regime).
            specs = [specs[0]]
        max_spawn = max(1, self.config.candidate_spawn_per_event)

        for spec in specs:
            if len(created) >= max_spawn:
                break

            active_candidates_now = [b for b in branches.values() if b.state.status == BranchStatus.CANDIDATE]
            if len(active_candidates_now) >= self.config.max_active_candidates:
                break
            active_non_archived_now = [b for b in branches.values() if b.state.status != BranchStatus.ARCHIVED]
            if len(active_non_archived_now) >= self.config.max_active_branches:
                break

            base_name = spec["base_name"]
            capability_tag = spec["capability_tag"]
            purpose = spec["purpose"]
            template = spec["prompt_template"]
            parent_for_candidate = spec.get("parent_hint", parent_hint) or parent_hint

            if self._children_under_parent(branches, parent_for_candidate) >= self.config.candidate_max_children_per_parent:
                continue
            if self._capability_exists_under_parent(branches, capability_tag, parent_for_candidate):
                continue

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
                    "parent_hint": parent_for_candidate,
                },
            )
            branches[candidate_name] = candidate
            created.append(candidate_name)

    def _candidate_specs_from_failure(
        self,
        failure_reason: str,
        task_type: str,
        parent_hint: str,
    ) -> list[dict[str, str]]:
        fr = failure_reason.lower()
        specs: list[dict[str, str]] = []

        if "rule_miss" in fr:
            specs.extend(
                [
                    {
                        "base_name": f"constraint_solver_{task_type}",
                        "capability_tag": "constraint_solver",
                        "purpose": "Satisfy explicit constraints before producing final response.",
                        "prompt_template": (
                            "You are the Constraint Solver candidate branch. Task type: {task_type}.\n"
                            "Task: {task}\n"
                            "Parent context: {context}\n"
                            "First enumerate constraints, then verify each before the answer."
                        ),
                        "parent_hint": parent_hint,
                    },
                    {
                        "base_name": f"rule_auditor_{task_type}",
                        "capability_tag": "rule_auditor",
                        "purpose": "Audit rule coverage and detect missing constraints.",
                        "prompt_template": (
                            "You are the Rule Auditor candidate branch. Task type: {task_type}.\n"
                            "Task: {task}\n"
                            "Parent context: {context}\n"
                            "List every rule requirement and provide pass/fail evidence."
                        ),
                        "parent_hint": parent_hint,
                    },
                ]
            )
        elif "keyword_coverage" in fr or "low_quality" in fr:
            specs.extend(
                [
                    {
                        "base_name": f"evidence_fuser_{task_type}",
                        "capability_tag": "evidence_fuser",
                        "purpose": "Fuse evidence with explicit requirement anchoring.",
                        "prompt_template": (
                            "You are the Evidence Fuser candidate branch. Task type: {task_type}.\n"
                            "Task: {task}\n"
                            "Parent context: {context}\n"
                            "Extract evidence snippets and map each requirement to an evidence-backed statement."
                        ),
                        "parent_hint": parent_hint,
                    },
                    {
                        "base_name": f"coverage_planner_{task_type}",
                        "capability_tag": "coverage_planner",
                        "purpose": "Plan full requirement coverage before final output.",
                        "prompt_template": (
                            "You are the Coverage Planner candidate branch. Task type: {task_type}.\n"
                            "Task: {task}\n"
                            "Parent context: {context}\n"
                            "Create a coverage checklist and ensure all required keywords/constraints are addressed."
                        ),
                        "parent_hint": parent_hint,
                    },
                ]
            )

        specs.append(
            {
                "base_name": f"verifier_plus_{task_type}",
                "capability_tag": "verifier_plus",
                "purpose": "Deep verification and self-check loops for reliability.",
                "prompt_template": (
                    "You are the Verifier Plus candidate branch. Task type: {task_type}.\n"
                    "Task: {task}\n"
                    "Parent context: {context}\n"
                    "Run a strict self-checklist and only return conclusions that pass all checks."
                ),
                "parent_hint": parent_hint,
            }
        )
        return specs

    def _extract_advisor_directives(
        self,
        payload: dict[str, Any],
        route: RoutingDecision,
    ) -> dict[str, dict[str, Any]]:
        directives: dict[str, dict[str, Any]] = {}
        items = payload.get("branch_directives", [])
        if not isinstance(items, list):
            return directives
        active = set(route.activated_branches)
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("branch_name", "")).strip()
            if not name or name not in active:
                continue
            directives[name] = item
        return directives

    def _extract_advisor_candidate_proposals(
        self,
        payload: dict[str, Any],
        route: RoutingDecision,
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        items = payload.get("candidate_proposals", [])
        if not isinstance(items, list):
            return out
        fallback_parent = route.activated_branches[-1] if route.activated_branches else ""
        active_set = set(route.activated_branches)
        for item in items:
            if not isinstance(item, dict):
                continue
            base_name = str(item.get("base_name", "")).strip().replace(" ", "_")
            capability_tag = str(item.get("capability_tag", "")).strip().replace(" ", "_")
            purpose = str(item.get("purpose", "")).strip()
            template = str(item.get("prompt_template", "")).strip()
            parent_hint = str(item.get("parent_hint", fallback_parent)).strip() or fallback_parent
            if parent_hint not in active_set:
                parent_hint = fallback_parent
            if not base_name or not capability_tag or not purpose or not template:
                continue
            if "{task}" not in template or "{task_type}" not in template or "{context}" not in template:
                continue
            out.append(
                {
                    "base_name": base_name,
                    "capability_tag": capability_tag,
                    "purpose": purpose,
                    "prompt_template": template,
                    "parent_hint": parent_hint,
                }
            )
            if len(out) >= 2:
                break
        return out

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

    @staticmethod
    def _children_under_parent(branches: dict[str, PromptBranch], parent_hint: str) -> int:
        if not parent_hint:
            return 0
        count = 0
        for branch in branches.values():
            if branch.state.status == BranchStatus.ARCHIVED:
                continue
            if branch.state.metadata.get("parent_hint") == parent_hint:
                count += 1
        return count

    def _should_rewrite_branch(
        self,
        branch: PromptBranch,
        reward: float,
        advice_rewrite_hint: str,
        advice_conf: float,
    ) -> bool:
        streak = int(branch.state.metadata.get("low_reward_streak", 0))
        if reward < self.config.prompt_rewrite_threshold:
            streak += 1
        else:
            streak = 0
        branch.state.metadata["low_reward_streak"] = streak

        last_rewrite_episode = int(branch.state.metadata.get("last_rewrite_episode", -10**9))
        cooldown_ok = (self._episode_idx - last_rewrite_episode) >= max(1, self.config.rewrite_cooldown_episodes)
        repeated_failures_ok = streak >= max(1, self.config.rewrite_failure_streak_trigger)
        advisor_ok = bool(advice_rewrite_hint and advice_conf >= self.config.advisor_rewrite_confidence_threshold)

        should = cooldown_ok and repeated_failures_ok and (reward < self.config.prompt_rewrite_threshold or advisor_ok)
        if should:
            branch.state.metadata["last_rewrite_episode"] = self._episode_idx
        return should

    def _passes_acceptance_gate(
        self,
        memory: MemoryStore,
        branch_name: str,
        task_type: str,
        user_id: str,
        old_weight: float,
        new_weight: float,
        prompt_rewritten: bool,
    ) -> bool:
        before_est = memory.branch_expected_reward(
            branch_name=branch_name,
            task_type=task_type,
            user_id=user_id,
            limit=self.config.update_acceptance_window,
        )
        if before_est is None:
            return True

        weight_effect = 0.4 * (new_weight - old_weight)
        rewrite_effect = 0.015 if prompt_rewritten else 0.0
        after_est = before_est + weight_effect + rewrite_effect
        return after_est >= (before_est + self.config.update_acceptance_min_gain)

    @staticmethod
    def _update_candidate_parent_comparison(
        branch_name: str,
        branch: PromptBranch,
        route: RoutingDecision,
        signal: EvaluationSignal,
    ) -> None:
        parent_hint = str(branch.state.metadata.get("parent_hint", "")).strip()
        if not parent_hint or parent_hint not in route.activated_branches:
            return
        candidate_fb = signal.branch_feedback.get(branch_name)
        parent_fb = signal.branch_feedback.get(parent_hint)
        if not candidate_fb or not parent_fb:
            return

        compare_n = int(branch.state.metadata.get("parent_compare_count", 0)) + 1
        win_n = int(branch.state.metadata.get("parent_win_count", 0))
        if candidate_fb.reward > parent_fb.reward + 0.02:
            win_n += 1
        branch.state.metadata["parent_compare_count"] = compare_n
        branch.state.metadata["parent_win_count"] = win_n

    def _passes_parent_gate(self, branch: PromptBranch) -> bool:
        parent_hint = str(branch.state.metadata.get("parent_hint", "")).strip()
        if not parent_hint:
            return True
        compares = int(branch.state.metadata.get("parent_compare_count", 0))
        wins = int(branch.state.metadata.get("parent_win_count", 0))
        if compares < self.config.candidate_parent_min_comparisons:
            return False
        win_rate = wins / max(1, compares)
        return win_rate >= self.config.candidate_parent_win_rate_threshold

    def _has_clustered_failures(self, failure_map: dict[str, int]) -> bool:
        if not failure_map:
            return False
        ordered = sorted(failure_map.values(), reverse=True)
        if ordered[0] < self.config.candidate_failure_trigger:
            return False
        if len(ordered) == 1:
            return ordered[0] >= self.config.candidate_failure_trigger + 1
        return ordered[1] >= max(2, self.config.candidate_failure_trigger // 2)

    def _bounded_advisory_delta(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        cap = max(0.0, self.config.advisor_weight_delta_cap)
        return max(-cap, min(cap, parsed))

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, parsed))

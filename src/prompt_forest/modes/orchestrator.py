"""Mode Orchestrator: unified entry point for both operating modes.

The orchestrator wraps ``PromptForestEngine`` and adds mode-specific
initialisation, routing, evaluation, state management, and observability.

In **Agent Improvement Mode**, the orchestrator is a thin passthrough
to the existing engine -- backward compatibility is preserved exactly.

In **Human Mode**, the orchestrator:
  1. Initialises the cognitive-behavioral branch forest and HumanState.
  2. Runs the conflict-aware router conditioned on internal state.
  3. Injects state context into branch prompts before execution.
  4. Evaluates outputs for behavioral coherence.
  5. Updates internal state from task outcomes.
  6. Records experiential memories.
  7. Emits rich observability traces.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from ..backend.base import LLMBackend
from ..backend.mock import MockLLMBackend
from ..agents.optimizer_agent import OptimizationEvent
from ..brain import BrainController, BrainOutput, BrainState
from ..branches.base import PromptBranch
from ..config import EngineConfig, load_config
from ..core.engine import PromptForestEngine
from ..state.human_state import DriveConflict, HumanState
from ..types import MemoryRecord, RoutingDecision, TaskInput
from .human_mode.branches import create_human_mode_forest
from .human_mode.evaluator import HumanModeEvaluator
from .human_mode.memory import HumanModeMemory
from .human_mode.router import HumanModeRouter
from .registry import ModeConfig, OperatingMode, get_mode_config


class ModeOrchestrator:
    """Unified engine that supports both Agent Improvement and Human Mode.

    Parameters
    ----------
    mode:
        Operating mode string or enum.
    engine_config:
        Shared EngineConfig for the underlying prompt forest engine.
    mode_config:
        Mode-specific configuration (auto-created if omitted).
    backend:
        LLM backend for branch execution.
    initial_state:
        Initial human-state variable overrides (Human Mode only).
    """

    def __init__(
        self,
        mode: str | OperatingMode = "agent_improvement",
        engine_config: EngineConfig | None = None,
        engine_config_path: str | Path | None = None,
        mode_config: ModeConfig | None = None,
        backend: LLMBackend | None = None,
        initial_state: dict[str, float] | None = None,
    ) -> None:
        if isinstance(mode, str):
            mode = OperatingMode(mode)
        self.mode = mode
        self.mode_config = mode_config or get_mode_config(mode)
        self._engine_config = engine_config or load_config(engine_config_path)

        if mode == OperatingMode.AGENT_IMPROVEMENT:
            self._init_agent_mode(backend)
        else:
            self._init_human_mode(backend, initial_state)

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_agent_mode(self, backend: LLMBackend | None) -> None:
        """Initialise for Agent Improvement Mode (standard engine passthrough)."""
        self.engine = PromptForestEngine(
            config=self._engine_config,
            backend=backend,
        )
        # No extra state needed
        self.human_state: HumanState | None = None
        self.human_router: HumanModeRouter | None = None
        self.human_evaluator: HumanModeEvaluator | None = None
        self.human_memory: HumanModeMemory | None = None
        self.brain_controller: BrainController | None = None
        self._last_brain_output: BrainOutput | None = None

    def _init_human_mode(
        self,
        backend: LLMBackend | None,
        initial_state: dict[str, float] | None,
    ) -> None:
        """Initialise for Human Mode with cognitive-behavioral components."""
        hmc = self.mode_config.human_mode

        # Build human-mode forest instead of default
        forest = create_human_mode_forest()
        self.engine = PromptForestEngine(
            config=self._engine_config,
            backend=backend,
        )
        # Replace the engine's forest with human-mode forest
        self.engine.forest = forest
        self.engine.branches = forest.branches

        # Human-specific components
        self.human_state = HumanState(
            initial_values=initial_state,
            decay_rate=hmc.state_decay_rate,
            momentum=hmc.emotional_inertia,
            noise_level=hmc.noise_level,
        )
        self.human_router = HumanModeRouter(
            top_k=self._engine_config.router.top_k,
            noise_level=hmc.noise_level,
        )
        self.human_evaluator = HumanModeEvaluator(
            coherence_weight=hmc.coherence_weight,
            consistency_weight=hmc.consistency_weight,
            believability_weight=hmc.believability_weight,
            conflict_handling_weight=hmc.conflict_handling_weight,
        )
        self.human_memory = HumanModeMemory(
            emotional_decay=hmc.emotional_memory_decay,
            trauma_amplification=hmc.trauma_amplification,
            experience_bias_strength=hmc.experience_bias_strength,
        )
        self.brain_controller = BrainController()
        self._last_brain_output: BrainOutput | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def run_task(
        self,
        text: str,
        task_type: str = "auto",
        metadata: dict[str, Any] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        *,
        adapt: bool = False,
        update_memory: bool = True,
        acceptance_runner: Callable[..., bool] | None = None,
    ) -> dict[str, Any]:
        """Run a task through the appropriate mode pipeline.

        This is the main entry point for both modes.
        """
        if self.mode == OperatingMode.AGENT_IMPROVEMENT:
            return self._run_agent_mode(
                text,
                task_type,
                metadata,
                event_callback,
                adapt=adapt,
                update_memory=update_memory,
            )
        else:
            return self._run_human_mode(
                text,
                task_type,
                metadata,
                event_callback,
                adapt=adapt,
                update_memory=update_memory,
                acceptance_runner=acceptance_runner,
            )

    def inject_event(self, event_type: str, intensity: float = 0.5) -> dict[str, Any]:
        """Inject an external event into the human state (Human Mode only)."""
        if self.human_state is None:
            return {"error": "inject_event is only available in human_mode"}
        snap = self.human_state.inject_event(event_type, intensity)
        return snap.to_dict()

    def get_state(self) -> dict[str, Any]:
        """Get current internal state (mode-aware)."""
        result: dict[str, Any] = {
            "mode": self.mode.value,
            "branches": [
                {
                    "name": b.state.name,
                    "purpose": b.state.purpose,
                    "weight": round(b.state.weight, 4),
                    "status": b.state.status.value,
                    "avg_reward": round(b.state.avg_reward(), 4),
                }
                for b in self.engine.branches.values()
                if not b.state.metadata.get("category_node")
            ],
            "branch_count": len(self.engine.branches),
        }
        if self.human_state is not None:
            result["human_state"] = self.human_state.to_dict()
            result["mood_valence"] = self.human_state.mood_valence()
            result["arousal"] = self.human_state.arousal_level()
            result["dominant_drives"] = self.human_state.dominant_drives(5)
            result["active_conflicts"] = [
                c.to_dict() for c in self.human_state.active_conflicts
            ]
        if self.human_memory is not None:
            result["experiential_memory_count"] = self.human_memory.memory_count
        if self.brain_controller is not None and self.human_state is not None:
            result["brain_state"] = self.get_brain_state().to_dict()
            if self._last_brain_output is not None:
                result["last_brain_output"] = self._last_brain_output.to_dict()
        return result

    def get_brain_state(self) -> BrainState:
        """Return the current portable latent-state summary for external agents."""
        if self.brain_controller is None or self.human_state is None:
            raise RuntimeError("Brain state is only available in human_mode")
        return self.brain_controller.build_state(self.human_state)

    def get_last_brain_output(self) -> BrainOutput | None:
        """Return the most recent brain control surface, if any."""
        return self._last_brain_output

    def openclaw_ingest(self, trajectory_event: dict[str, Any]) -> dict[str, Any]:
        """Compatibility hook for external runtimes that want the brain layer."""
        metadata = {
            "expected_keywords": trajectory_event.get("expected_keywords", []),
            "required_substrings": trajectory_event.get("required_checks", []),
            "trajectory": trajectory_event,
            "user_id": trajectory_event.get("user_id", "global"),
        }
        return self.run_task(
            text=str(trajectory_event.get("task", "") or ""),
            task_type=str(trajectory_event.get("task_type", "auto") or "auto"),
            metadata=metadata,
            adapt=bool(trajectory_event.get("adapt", False)),
            update_memory=bool(trajectory_event.get("update_memory", True)),
        )

    # ── Agent Improvement Mode pipeline ───────────────────────────────────

    def _run_agent_mode(
        self,
        text: str,
        task_type: str,
        metadata: dict[str, Any] | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        *,
        adapt: bool,
        update_memory: bool,
    ) -> dict[str, Any]:
        """Passthrough to standard engine. No behaviour change."""
        result = self.engine.run_task_controlled(
            text=text,
            task_type=task_type,
            metadata=metadata,
            event_callback=event_callback,
            adapt=adapt,
            update_memory=update_memory,
        )
        result["mode"] = "agent_improvement"
        return result

    # ── Human Mode pipeline ───────────────────────────────────────────────

    def _run_human_mode(
        self,
        text: str,
        task_type: str,
        metadata: dict[str, Any] | None,
        event_callback: Callable[[dict[str, Any]], None] | None,
        *,
        adapt: bool,
        update_memory: bool,
        acceptance_runner: Callable[..., bool] | None,
    ) -> dict[str, Any]:
        """Full human-mode pipeline with state, conflicts, and experiential memory."""
        assert self.human_state is not None
        assert self.human_router is not None
        assert self.human_evaluator is not None
        assert self.human_memory is not None

        task_metadata = dict(metadata or {})
        task = TaskInput(
            task_id=str(uuid4()),
            text=text,
            task_type=task_type,
            metadata=task_metadata,
        )
        started_at = perf_counter()
        timings: dict[str, float] = {}

        # Emit state before routing
        state_before = self.human_state.snapshot()
        self._emit(event_callback, {
            "type": "human_state",
            "phase": "before",
            "state": state_before.to_dict(),
        })

        # Step 1: State-conditioned routing
        route_started = perf_counter()
        route, conflicts = self.human_router.route(
            task, self.engine.forest, self.human_state, self.engine.memory,
        )
        task.task_type = route.task_type
        timings["route_ms"] = round((perf_counter() - route_started) * 1000, 3)

        self._emit(event_callback, {
            "type": "routing",
            "task_type": route.task_type,
            "activated_branches": list(route.activated_branches),
            "branch_scores": {k: round(v, 4) for k, v in route.branch_scores.items()},
            "conflicts": [c.to_dict() for c in conflicts],
            "dominant_drives": state_before.dominant_drives,
        })

        # Step 2: Resolve conflicts before execution
        resolved_conflicts: list[DriveConflict] = []
        strategy = self.mode_config.human_mode.conflict_resolution_strategy
        for conflict in conflicts:
            resolved = self.human_state.resolve_conflict(conflict, strategy)
            resolved_conflicts.append(resolved)
            self._emit(event_callback, {
                "type": "conflict_resolution",
                "drive_a": conflict.drive_a,
                "drive_b": conflict.drive_b,
                "intensity": conflict.intensity,
                "resolution": resolved.resolution,
                "resolution_weight": resolved.resolution_weight,
            })

        # Step 3: Inject state context into branch execution
        state_context = self._build_state_context()
        task.metadata["context_seed"] = state_context

        # Step 4: Execute branches via the engine's executor
        execute_started = perf_counter()
        outputs = self.engine._run_path(route, task, event_callback=event_callback)
        timings["execute_ms"] = round((perf_counter() - execute_started) * 1000, 3)

        # Step 5: Score outputs with the standard judge
        branch_scores = self.engine.judge.score_all(outputs, task)

        # Step 6: Aggregate
        numeric_scores = {k: v.reward for k, v in branch_scores.items()}
        aggregation = self.engine._aggregate(
            route, outputs, numeric_scores,
            candidate_branches=set(route.activated_branches),
        )

        # Step 7: Human-mode evaluation (coherence-based)
        eval_started = perf_counter()
        signal = self.human_evaluator.evaluate(
            task=task,
            route=route,
            branch_scores=branch_scores,
            aggregation=aggregation,
            state=self.human_state,
            conflicts=resolved_conflicts,
            branch_outputs={k: v.output for k, v in outputs.items()},
        )
        timings["evaluate_ms"] = round((perf_counter() - eval_started) * 1000, 3)

        self._emit(event_callback, {
            "type": "evaluation",
            "reward_score": signal.reward_score,
            "confidence": signal.confidence,
            "selected_branch": signal.selected_branch,
            "failure_reason": signal.failure_reason,
            "coherence_details": signal.aggregator_notes,
        })

        # Step 8: Update human state from outcome
        state_snap = self.human_state.apply_outcome(
            reward=signal.reward_score,
            task_type=route.task_type,
            failure_reason=signal.failure_reason,
        )
        self._emit(event_callback, {
            "type": "human_state",
            "phase": "after",
            "state": state_snap.to_dict(),
        })

        # Step 9: Record experiential memory
        memory_started = perf_counter()
        exp_mem = self.human_memory.record(
            event_id=task.task_id,
            task=task,
            state=self.human_state,
            reward=signal.reward_score,
            selected_branch=signal.selected_branch,
            active_branches=route.activated_branches,
            failure_reason=signal.failure_reason,
        )
        timings["memory_ms"] = round((perf_counter() - memory_started) * 1000, 3)

        self._emit(event_callback, {
            "type": "memory_update",
            "task_id": task.task_id,
            "record_count": self.human_memory.memory_count,
            "tags": exp_mem.tags,
            "emotional_valence": exp_mem.emotional_valence,
        })

        selected_path = self.engine._selected_path(route, signal.selected_branch)
        routing_context_key = ""
        if len(selected_path) >= 2:
            routing_context_key = self.engine.memory.routing_context_key_for_task(
                task,
                parent_id=selected_path[-2],
            )
        reward_components = self.engine._compute_reward_components(
            task,
            signal.selected_output,
            signal.reward_score,
        )
        brain_output = self.brain_controller.build_output(
            state=self.human_state,
            route=route,
            conflicts=resolved_conflicts,
            human_memory=self.human_memory,
            branch_weights={
                name: b.state.weight
                for name, b in self.engine.branches.items()
                if not b.state.metadata.get("category_node")
            },
        )
        self._last_brain_output = brain_output

        # Step 10: Also update the core engine for weight adaptation
        optimize_started = perf_counter()
        self.engine._propagate_rewards_along_path(route, signal, gamma=0.9, local_mix=0.55)
        # Apply branch rewards
        for branch_name, fb in signal.branch_feedback.items():
            branch = self.engine.branches.get(branch_name)
            if branch:
                branch.apply_reward(fb.reward)
        if adapt:
            optimize_event: OptimizationEvent = self.engine.optimizer_agent.optimize(
                task=task,
                route=route,
                signal=signal,
                branches=self.engine.branches,
                memory=self.engine.memory,
                acceptance_runner=acceptance_runner or self.engine._mini_holdout_acceptance,
            )
            self.engine._attach_created_candidates(
                route,
                optimize_event,
                selected_branch=signal.selected_branch,
            )
        else:
            optimize_event = OptimizationEvent(
                updated_weights={},
                update_details={},
                rewritten_prompts=[],
                promoted_candidates=[],
                archived_candidates=[],
                created_candidates=[],
                advisor_used=False,
                advisor_error="",
            )
        timings["optimize_ms"] = round((perf_counter() - optimize_started) * 1000, 3)

        # Step 11: Mirror the core engine memory path so acceptance gates and
        # routing preferences can use human-mode episodes too.
        memory_started_core = perf_counter()
        record = MemoryRecord(
            task_id=task.task_id,
            task_type=route.task_type,
            input_text=task.text,
            activated_branches=route.activated_branches,
            branch_outputs={name: out.output for name, out in outputs.items()},
            selected_branch=signal.selected_branch,
            selected_output=signal.selected_output,
            reward_score=signal.reward_score,
            failure_reason=signal.failure_reason,
            confidence=signal.confidence,
            useful_patterns=self.engine.memory.useful_patterns(route.task_type),
            branch_rewards={name: fb.reward for name, fb in signal.branch_feedback.items()},
            selected_path=selected_path,
            routing_context_key=routing_context_key,
            user_id=str(task.metadata.get("user_id", "global")).strip() or "global",
            task_metadata=task.metadata,
            reward_components=reward_components,
        )
        if update_memory:
            self.engine.memory.add(record)
            self.engine._record_route_preferences(task, route, signal)
        timings["core_memory_ms"] = round((perf_counter() - memory_started_core) * 1000, 3)

        timings["total_ms"] = round((perf_counter() - started_at) * 1000, 3)

        # Build result
        result: dict[str, Any] = {
            "mode": "human_mode",
            "task": asdict(task),
            "routing": asdict(route),
            "conflicts": [c.to_dict() for c in resolved_conflicts],
            "evaluation_signal": {
                "reward_score": signal.reward_score,
                "confidence": signal.confidence,
                "selected_branch": signal.selected_branch,
                "selected_output": signal.selected_output,
                "failure_reason": signal.failure_reason,
                "coherence_details": signal.aggregator_notes,
            },
            "branch_scores": {k: asdict(v) for k, v in branch_scores.items()},
            "human_state": {
                "before": state_before.to_dict(),
                "after": state_snap.to_dict(),
            },
            "experiential_memory": {
                "count": self.human_memory.memory_count,
                "latest_tags": exp_mem.tags,
            },
            "brain_output": brain_output.to_dict(),
            "timings": timings,
            "branch_weights": {
                name: round(b.state.weight, 4)
                for name, b in self.engine.branches.items()
                if not b.state.metadata.get("category_node")
            },
            "optimization": {
                "updated_weights": dict(optimize_event.updated_weights),
                "rewritten_prompts": list(optimize_event.rewritten_prompts),
                "created_candidates": list(optimize_event.created_candidates),
                "promoted_candidates": list(optimize_event.promoted_candidates),
                "archived_candidates": list(optimize_event.archived_candidates),
            },
        }

        self._emit(event_callback, {
            "type": "task_complete",
            "selected_branch": signal.selected_branch,
            "selected_output": signal.selected_output,
            "reward_score": signal.reward_score,
            "mood_after": state_snap.mood_valence,
        })

        return result

    def _build_state_context(self) -> str:
        """Build a state context string to inject into branch prompts."""
        if self.human_state is None:
            return ""

        state = self.human_state
        mood = state.mood_valence()
        top = state.dominant_drives(3)
        arousal = state.arousal_level()
        conflicts = state.active_conflicts

        parts = [
            f"Mood: {'positive' if mood > 0.2 else 'negative' if mood < -0.2 else 'neutral'} ({mood:+.2f})",
            f"Arousal: {'high' if arousal > 0.6 else 'low' if arousal < 0.3 else 'moderate'}",
            f"Dominant drives: {', '.join(top)}",
            f"Confidence: {state.get('confidence'):.2f}",
            f"Stress: {state.get('stress'):.2f}",
        ]
        if conflicts:
            conflict_strs = [f"{c.drive_a} vs {c.drive_b} (intensity {c.intensity:.2f})" for c in conflicts]
            parts.append(f"Active conflicts: {'; '.join(conflict_strs)}")

        # Experiential memory bias
        if self.human_memory and self.human_memory.memory_count > 0:
            biases = self.human_memory.experiential_bias(state)
            if biases:
                top_bias = sorted(biases.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                bias_strs = [f"{k}: {'+' if v > 0 else ''}{v:.3f}" for k, v in top_bias]
                parts.append(f"Experience biases: {', '.join(bias_strs)}")

        return " | ".join(parts)

    @staticmethod
    def _emit(
        callback: Callable[[dict[str, Any]], None] | None,
        payload: dict[str, Any],
    ) -> None:
        if callback:
            callback(payload)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import BranchState
from .base import PromptBranch


@dataclass
class ForestNode:
    node_id: str
    parent_id: str | None
    depth: int
    specialties: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)


class HierarchicalPromptForest:
    def __init__(self) -> None:
        self.root_id = "root"
        self.nodes: dict[str, ForestNode] = {
            self.root_id: ForestNode(node_id=self.root_id, parent_id=None, depth=0, specialties=["general"])
        }
        self.branches: dict[str, PromptBranch] = {}

    @classmethod
    def from_flat(cls, branches: dict[str, PromptBranch]) -> "HierarchicalPromptForest":
        forest = cls()
        for branch_name, branch in branches.items():
            forest.add_branch(branch_name, branch, parent_id=forest.root_id, specialties=["general"])
        return forest

    def add_branch(
        self,
        branch_name: str,
        branch: PromptBranch,
        parent_id: str,
        specialties: list[str] | None = None,
    ) -> None:
        if parent_id not in self.nodes:
            raise KeyError(f"Unknown parent node: {parent_id}")
        parent = self.nodes[parent_id]

        self.branches[branch_name] = branch
        self.nodes[branch_name] = ForestNode(
            node_id=branch_name,
            parent_id=parent_id,
            depth=parent.depth + 1,
            specialties=specialties or ["general"],
        )
        parent.children.append(branch_name)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def get_branch(self, node_id: str) -> PromptBranch:
        return self.branches[node_id]

    def children(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].children)

    def parent(self, node_id: str) -> str | None:
        return self.nodes[node_id].parent_id

    def depth(self, node_id: str) -> int:
        return self.nodes[node_id].depth

    def path_to_root(self, node_id: str) -> list[str]:
        path: list[str] = []
        cursor: str | None = node_id
        while cursor and cursor != self.root_id:
            path.append(cursor)
            cursor = self.parent(cursor)
        return list(reversed(path))

    def branch_snapshot(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for branch_name, branch in self.branches.items():
            node = self.nodes[branch_name]
            out[branch_name] = {
                "weight": round(branch.state.weight, 4),
                "status": branch.state.status.value,
                "avg_reward": round(branch.state.avg_reward(), 4),
                "trial_remaining": branch.state.trial_remaining,
                "history_len": len(branch.state.historical_rewards),
                "depth": node.depth,
                "parent": node.parent_id,
                "children": list(node.children),
                "specialties": list(node.specialties),
            }
        return out



def _branch(name: str, purpose: str, prompt: str, weight: float) -> PromptBranch:
    return PromptBranch(
        BranchState(
            name=name,
            purpose=purpose,
            prompt_template=prompt,
            weight=weight,
            metadata={"base_weight": weight},
        )
    )



def create_default_hierarchical_forest() -> HierarchicalPromptForest:
    forest = HierarchicalPromptForest()

    macro_specs = [
        (
            "analytical",
            "Structured decomposition and explicit assumptions.",
            (
                "You are the Analytical macro branch. Task type: {task_type}.\\n"
                "Task: {task}\\n"
                "Upstream context: {context}\\n"
                "Decompose the task into formal components and produce a precise intermediate reasoning state."
            ),
            1.05,
            ["math", "code", "general"],
        ),
        (
            "planner",
            "Action-oriented sequencing and execution planning.",
            (
                "You are the Planning macro branch. Task type: {task_type}.\\n"
                "Task: {task}\\n"
                "Upstream context: {context}\\n"
                "Produce a structured action plan with sequencing dependencies and risks."
            ),
            1.0,
            ["planning", "general", "code"],
        ),
        (
            "retrieval",
            "Evidence gathering and grounding.",
            (
                "You are the Retrieval macro branch. Task type: {task_type}.\\n"
                "Task: {task}\\n"
                "Upstream context: {context}\\n"
                "Extract key evidence points and factual anchors needed for high-confidence answers."
            ),
            1.0,
            ["factual", "general", "code"],
        ),
        (
            "critique",
            "Failure detection and adversarial stress testing.",
            (
                "You are the Critique macro branch. Task type: {task_type}.\\n"
                "Task: {task}\\n"
                "Upstream context: {context}\\n"
                "Identify likely failure modes, blind spots, and robustness risks."
            ),
            0.95,
            ["code", "planning", "general"],
        ),
        (
            "verification",
            "Constraint validation and consistency checks.",
            (
                "You are the Verification macro branch. Task type: {task_type}.\\n"
                "Task: {task}\\n"
                "Upstream context: {context}\\n"
                "Verify constraints, check consistency, and provide confidence-calibrated conclusions."
            ),
            1.15,
            ["math", "factual", "general", "code"],
        ),
        (
            "creative",
            "Diverse solution generation under constraints.",
            (
                "You are the Creative macro branch. Task type: {task_type}.\\n"
                "Task: {task}\\n"
                "Upstream context: {context}\\n"
                "Generate high-diversity options that still satisfy core constraints."
            ),
            0.9,
            ["creative", "planning", "general"],
        ),
    ]

    for name, purpose, prompt, weight, specialties in macro_specs:
        forest.add_branch(name, _branch(name, purpose, prompt, weight), parent_id=forest.root_id, specialties=specialties)

    niche_specs: dict[str, list[tuple[str, str, str, float, list[str]]]] = {
        "analytical": [
            (
                "analytical_symbolic_solver",
                "Symbolic and equation-level decomposition.",
                (
                    "You are the Symbolic Solver niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Focus on precise symbolic derivations and explicit checkable steps."
                ),
                0.95,
                ["math", "code"],
            ),
            (
                "analytical_causal_decomposer",
                "Cause-effect decomposition and dependency tracing.",
                (
                    "You are the Causal Decomposer niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Model causal dependencies and isolate the highest-impact factors."
                ),
                0.92,
                ["planning", "general", "code"],
            ),
        ],
        "planner": [
            (
                "planner_timeline_optimizer",
                "Timeline and milestone optimization.",
                (
                    "You are the Timeline Optimizer niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Optimize for schedule structure, not risk ownership. "
                    "Produce a realistic timeline with phases, milestones, checkpoints, sequencing, dependencies, and monitoring points. "
                    "Default to time blocks such as day/week/phase, and make handoffs explicit. "
                    "Do not turn the answer into a risk register unless the task explicitly asks for one."
                ),
                0.94,
                ["planning", "general"],
            ),
            (
                "planner_risk_allocator",
                "Risk-aware planning and resource allocation.",
                (
                    "You are the Risk Allocator niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Optimize for risk ownership and mitigation, not timeline polish. "
                    "Produce a risk register or operating-risk plan with explicit owners, mitigations, fallback or rollback actions, and high-risk dependencies. "
                    "Surface uncertainty, escalation triggers, and containment steps. "
                    "Keep the register compact and operational rather than encyclopedic. "
                    "Close with a short rollout-plan note or go/no-go timing cue if it helps execution. "
                    "End with a calibrated numeric confidence line in the form confidence=0.xx even if the table already includes qualitative confidence labels. "
                    "Do not spend most of the answer on phases or milestone sequencing unless that is explicitly requested."
                ),
                0.93,
                ["planning", "code", "general"],
            ),
        ],
        "retrieval": [
            (
                "retrieval_evidence_tracer",
                "Evidence tracing and source-grounded extraction.",
                (
                    "You are the Evidence Tracer niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Prioritize evidence traceability and factual grounding cues."
                ),
                0.95,
                ["factual", "general", "code"],
            ),
            (
                "retrieval_source_triage",
                "Source relevance triage and conflict resolution.",
                (
                    "You are the Source Triage niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Rank source reliability and resolve conflicting factual signals."
                ),
                0.92,
                ["factual", "general"],
            ),
        ],
        "critique": [
            (
                "critique_failure_hunter",
                "Edge-case and failure scenario mining.",
                (
                    "You are the Failure Hunter niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Hunt for edge cases and brittle assumptions likely to break outputs."
                ),
                0.9,
                ["code", "planning", "general"],
            ),
            (
                "critique_adversarial_probe",
                "Adversarial probing for hallucination and logic gaps.",
                (
                    "You are the Adversarial Probe niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Probe contradictions and adversarially test reliability claims."
                ),
                0.9,
                ["factual", "general", "code"],
            ),
        ],
        "verification": [
            (
                "verification_constraint_checker",
                "Hard-constraint satisfaction verification.",
                (
                    "You are the Constraint Checker niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Act as a hard requirements auditor. "
                    "Verify required constraints one by one and produce pass or fail evidence for each explicit requirement, field, rule, or contract. "
                    "Prefer checklist, requirement coverage, and missing-item detection over broad narrative critique. "
                    "If a requirement is absent, say exactly what is missing. "
                    "For audit or consistency tasks, explicitly name the contradiction or unresolved gap and end with a calibrated confidence line in the form confidence=0.xx."
                ),
                0.97,
                ["math", "factual", "code", "general"],
            ),
            (
                "verification_consistency_auditor",
                "Cross-step consistency and calibration auditing.",
                (
                    "You are the Consistency Auditor niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Act as an internal coherence and calibration auditor. "
                    "Audit cross-step consistency, contradictions, unsupported jumps, and confidence calibration. "
                    "Prefer reasoning integrity, tradeoff coherence, and uncertainty calibration over checklist coverage. "
                    "Do not frame the answer as a pass or fail checklist unless the task explicitly demands that format. "
                    "When you find a contradiction, state it explicitly, explain the uncertainty impact, and end with a calibrated confidence line in the form confidence=0.xx."
                ),
                0.95,
                ["general", "planning", "factual"],
            ),
            (
                "json_lock",
                "Strict JSON contract output with zero extra text.",
                (
                    "You are the JSON Lock niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "If JSON is requested, return ONLY valid JSON with no prose, no markdown fences, and no extra tokens."
                ),
                0.98,
                ["math", "factual", "general", "code"],
            ),
            (
                "csv_lock",
                "Strict CSV row formatting with no explanatory text.",
                (
                    "You are the CSV Lock niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "If CSV is requested, return ONLY raw CSV lines in requested order and no additional commentary."
                ),
                0.98,
                ["general", "planning"],
            ),
            (
                "code_patch_lock",
                "Strict code-patch contract (FIX/TESTS) with no contamination.",
                (
                    "You are the Code Patch Lock niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "When asked for FIX/TESTS format, output exactly that contract without extra verification prose."
                ),
                0.99,
                ["code", "general"],
            ),
            (
                "bullet_lock",
                "Strict bullet-only contract following line-level constraints.",
                (
                    "You are the Bullet Lock niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "If bullet output is requested, return only bullet lines that satisfy explicit count/format constraints."
                ),
                0.97,
                ["general", "planning", "creative"],
            ),
        ],
        "creative": [
            (
                "creative_divergent_generator",
                "Divergent ideation with controlled novelty.",
                (
                    "You are the Divergent Generator niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Generate diverse solution candidates and make novelty explicit."
                ),
                0.9,
                ["creative", "general"],
            ),
            (
                "creative_constraint_innovator",
                "Constraint-aware innovation and refinement.",
                (
                    "You are the Constraint Innovator niche branch. Task type: {task_type}.\\n"
                    "Task: {task}\\n"
                    "Parent context: {context}\\n"
                    "Innovate while explicitly satisfying practical constraints."
                ),
                0.91,
                ["creative", "planning", "code"],
            ),
        ],
    }

    for parent, specs in niche_specs.items():
        for name, purpose, prompt, weight, specialties in specs:
            forest.add_branch(name, _branch(name, purpose, prompt, weight), parent_id=parent, specialties=specialties)

    return forest

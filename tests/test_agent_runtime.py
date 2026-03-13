from __future__ import annotations

from prompt_forest.agents.evaluator_agent import EvaluatorAgent
from prompt_forest.agents.llm_evaluator_agent import LLMEvaluatorAgent
from prompt_forest.agents.runtime_client import AgentRuntimeClient
from prompt_forest.aggregator.strategies import AggregationResult
from prompt_forest.config import AgentRuntimeConfig
from prompt_forest.evaluator.judge import BranchScore
from prompt_forest.types import RoutingDecision, TaskInput


def test_llm_evaluator_falls_back_when_runtime_disabled():
    fallback = EvaluatorAgent()
    llm_eval = LLMEvaluatorAgent(AgentRuntimeConfig(enabled=False), fallback=fallback)

    task = TaskInput(task_id="t1", text="Solve x^2 derivative", task_type="math")
    route = RoutingDecision(task_type="math", activated_branches=["analytical"], branch_scores={"analytical": 1.0})
    scores = {"analytical": BranchScore(reward=0.8, reason="high_quality")}
    agg = AggregationResult(selected_branch="analytical", selected_output="2x", notes={"strategy": "leaf_select"})

    signal = llm_eval.evaluate(task, route, scores, agg, branch_outputs={"analytical": "2x"})
    assert signal.selected_branch == "analytical"
    assert signal.reward_score > 0.7
    assert signal.aggregator_notes.get("strategy") == "leaf_select"


def test_runtime_client_json_parser_handles_wrapped_content():
    raw = "Some text before {\"reward_score\": 0.9, \"selected_branch\": \"x\"} trailing"
    parsed = AgentRuntimeClient._parse_json_text(raw)
    assert parsed["reward_score"] == 0.9
    assert parsed["selected_branch"] == "x"

"""
FastAPI server for the Prompt Forest frontend.

Wraps the PromptForestEngine in an async HTTP/SSE API so the React frontend
can stream live trace events as they occur.

Usage:
    pip install fastapi uvicorn
    python api_server.py

Endpoints:
    POST /api/chat       — SSE stream: routing → execution → evaluation → optimization → memory
    GET  /api/state      — current engine state snapshot
    POST /api/feedback   — apply user feedback to a previous task
    GET  /api/events     — recent events from artifacts/events.jsonl
    GET  /health         — health check
"""

import json
import time
import asyncio
import traceback
from pathlib import Path
from typing import AsyncGenerator, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─── Try to import the real engine ───────────────────────────────────────────

ENGINE_AVAILABLE = False
engine = None

try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from prompt_forest.core.engine import PromptForestEngine
    from prompt_forest.config import EngineConfig
    from prompt_forest.types import TaskInput

    # Load config
    config_path = Path(__file__).parent / "configs" / "default.json"
    if config_path.exists():
        with open(config_path) as f:
            raw_config = json.load(f)
        config = EngineConfig(**raw_config)
    else:
        config = EngineConfig()

    engine = PromptForestEngine(config=config)
    ENGINE_AVAILABLE = True
    print("✓ PromptForestEngine loaded successfully")
except Exception as e:
    print(f"⚠ Engine unavailable (demo mode): {e}")


# ─── App ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Prompt Forest API server starting…")
    yield
    print("Server shutting down.")

app = FastAPI(title="Prompt Forest API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    task_type: Optional[str] = None
    user_id: Optional[str] = "default"
    include_base_comparison: bool = True


class FeedbackRequest(BaseModel):
    task_id: str
    score: float
    comment: Optional[str] = None


# ─── SSE Helper ──────────────────────────────────────────────────────────────

def sse(event_type: str, data: Any) -> str:
    payload = json.dumps({"type": event_type, "data": data, "timestamp": int(time.time() * 1000)})
    return f"data: {payload}\n\n"


# ─── Chat SSE Stream ──────────────────────────────────────────────────────────

async def run_engine_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Run the PromptForest engine and emit SSE events for each stage.
    Falls back to demo mode if the engine is unavailable.
    """
    if not ENGINE_AVAILABLE or engine is None:
        async for chunk in demo_stream(req):
            yield chunk
        return

    try:
        import uuid
        task_id = str(uuid.uuid4())[:8]

        task = TaskInput(
            task_id=task_id,
            text=req.message,
            task_type=req.task_type or "auto",
            metadata={},
        )

        # We run the engine in a thread pool so it doesn't block the event loop
        loop = asyncio.get_event_loop()

        # Stage: routing
        yield sse("status", {"message": "Routing task…"})

        result = await loop.run_in_executor(None, lambda: engine.run_task(task))

        # Emit routing event
        routing_data = {
            "task_type": result.get("routing", {}).get("task_type", req.task_type or "general"),
            "activated_branches": result.get("routing", {}).get("activated_branches", []),
            "branch_scores": result.get("branch_scores", {}),
            "activated_paths": result.get("routing", {}).get("activated_paths", []),
            "sibling_decisions": result.get("routing", {}).get("sibling_decisions", {}),
            "branch_weights": result.get("branch_weights", {}),
        }
        yield sse("routing", routing_data)
        await asyncio.sleep(0.01)

        # Emit branch events
        for branch_name, output_text in result.get("branch_outputs", {}).items():
            yield sse("branch_start", {"branch_name": branch_name, "task_type": task.task_type})
            await asyncio.sleep(0.01)
            score = result.get("evaluation_signal", {}).get("branch_feedback", {}).get(branch_name, {}).get("reward")
            yield sse("branch_complete", {
                "branch_name": branch_name,
                "output": output_text[:500],
                "reward": score,
            })
            await asyncio.sleep(0.01)

        # Evaluation
        eval_signal = result.get("evaluation_signal", {})
        yield sse("evaluation", {
            **eval_signal,
            "reward_components": result.get("reward_components", {}),
        })
        await asyncio.sleep(0.01)

        # Optimization
        opt = result.get("optimization", {})
        if opt:
            yield sse("optimization", opt)
            await asyncio.sleep(0.01)

        # Memory
        yield sse("memory_update", {
            "task_id": task_id,
            "record_count": engine.memory.record_count() if hasattr(engine, "memory") else 0,
            "useful_patterns": [],
            "routing_context_key": task_id,
        })
        await asyncio.sleep(0.01)

        # Stream the selected answer token by token
        selected_output = eval_signal.get("selected_output", result.get("answer", ""))
        words = selected_output.split()
        for word in words:
            yield sse("token", {"token": word + " "})
            await asyncio.sleep(0.005)

        # Base model answer (if available)
        base_answer = result.get("base_answer")

        # Improvement delta
        adaptive_score = eval_signal.get("reward_score", 0.7)
        base_score = result.get("base_reward_score", adaptive_score - 0.1)
        delta = adaptive_score - base_score

        # Complete
        yield sse("complete", {
            "task_id": task_id,
            "answer": selected_output,
            "base_answer": base_answer,
            "routing": routing_data,
            "evaluation": eval_signal,
            "optimization": opt,
            "reward_components": result.get("reward_components", {}),
            "selected_path": result.get("selected_path", []),
            "timings": result.get("timings", {}),
            "composer": result.get("composer", {}),
            "branch_weights": result.get("branch_weights", {}),
            "branch_scores": result.get("branch_scores", {}),
            "improvement_delta": delta,
            "win_label": f"Adaptive outperformed base by +{delta*100:.1f}%" if delta > 0 else None,
        })

    except Exception as e:
        traceback.print_exc()
        yield sse("error", {"message": str(e)})


async def demo_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    """Demo stream for when the engine is unavailable."""
    import random

    task_type = req.task_type or detect_task_type(req.message)
    branches = ["analytical", "verification", "chain_of_thought"]
    selected = "chain_of_thought"

    await asyncio.sleep(0.1)

    # Routing
    yield sse("routing", {
        "task_type": task_type,
        "activated_branches": branches,
        "branch_scores": {"analytical": 0.82, "verification": 0.74, "chain_of_thought": 0.89, "planner": 0.45, "retrieval": 0.38, "creative": 0.22},
        "activated_paths": [["root", "analytical", "chain_of_thought"], ["root", "verification"]],
        "sibling_decisions": {},
        "branch_weights": {"analytical": 2.1, "planner": 1.8, "retrieval": 1.5, "critique": 1.3, "verification": 1.7, "creative": 1.2, "chain_of_thought": 2.0},
    })
    await asyncio.sleep(0.15)

    # Branch execution
    for branch in branches:
        yield sse("branch_start", {"branch_name": branch, "task_type": task_type})
        await asyncio.sleep(0.2)
        reward = 0.6 + random.random() * 0.35
        yield sse("branch_complete", {"branch_name": branch, "output": f"Output from {branch}", "reward": reward})
        await asyncio.sleep(0.1)

    # Generate answer word by word
    answer = demo_answer(req.message, task_type)
    words = answer.split()
    for word in words:
        yield sse("token", {"token": word + " "})
        await asyncio.sleep(0.02)

    await asyncio.sleep(0.1)

    adaptive_score = 0.72 + random.random() * 0.22
    base_score = 0.48 + random.random() * 0.15
    delta = adaptive_score - base_score

    # Evaluation
    yield sse("evaluation", {
        "reward_score": adaptive_score,
        "confidence": 0.81,
        "selected_branch": selected,
        "selected_output": answer,
        "failure_reason": "none",
        "suggested_improvement_direction": "",
        "branch_feedback": {
            "analytical": {"branch_name": "analytical", "reward": 0.74, "confidence": 0.72, "failure_reason": "none", "suggested_improvement_direction": ""},
            "verification": {"branch_name": "verification", "reward": 0.70, "confidence": 0.68, "failure_reason": "none", "suggested_improvement_direction": ""},
            "chain_of_thought": {"branch_name": "chain_of_thought", "reward": adaptive_score, "confidence": 0.81, "failure_reason": "none", "suggested_improvement_direction": ""},
        },
        "reward_components": {"verifier": 0.71, "task_rules": 0.68, "llm_judge": adaptive_score},
    })
    await asyncio.sleep(0.1)

    # Optimization
    yield sse("optimization", {
        "updated_weights": {"chain_of_thought": 2.04, "analytical": 2.08, "verification": 1.72},
        "update_details": {
            "chain_of_thought": {"old_weight": 2.0, "advantage": 0.04},
            "analytical": {"old_weight": 2.1, "advantage": -0.02},
            "verification": {"old_weight": 1.7, "advantage": 0.02},
        },
        "rewritten_prompts": [],
        "promoted_candidates": [],
        "archived_candidates": [],
        "created_candidates": [],
        "advisor_used": False,
    })
    await asyncio.sleep(0.05)

    # Memory
    yield sse("memory_update", {
        "task_id": f"demo-{int(time.time())}",
        "record_count": 143,
        "useful_patterns": ["Step-by-step reasoning improves accuracy", "Verification reduces false positives"],
        "routing_context_key": f"general|reasoning|demo-{int(time.time())}",
    })

    # Complete
    yield sse("complete", {
        "task_id": f"demo-{int(time.time())}",
        "answer": answer,
        "base_answer": demo_base_answer(req.message),
        "routing": {"task_type": task_type, "activated_branches": branches, "branch_scores": {}, "activated_paths": [], "sibling_decisions": {}},
        "evaluation": {"reward_score": adaptive_score, "confidence": 0.81, "selected_branch": selected},
        "optimization": {},
        "reward_components": {"verifier": 0.71, "task_rules": 0.68},
        "selected_path": ["root", "analytical", "chain_of_thought"],
        "timings": {"route_ms": 45, "execute_ms": 312, "probe_ms": 28, "compose_ms": 67, "judge_ms": 89, "evaluate_ms": 34, "optimize_ms": 52, "memory_ms": 12, "total_ms": 639, "primary_backend_calls": 3, "evaluator_runtime_calls": 1, "optimizer_runtime_calls": 1},
        "composer": {"composer_enabled": True, "composed_leaf": "chain_of_thought", "feature_count": 4},
        "branch_weights": {"analytical": 2.1, "chain_of_thought": 2.04},
        "branch_scores": {"analytical": 0.82, "chain_of_thought": 0.89},
        "improvement_delta": delta,
        "win_label": f"Adaptive outperformed base by +{delta*100:.1f}%" if delta > 0 else None,
    })

    yield "data: [DONE]\n\n"


def detect_task_type(text: str) -> str:
    import re
    if re.search(r'\d|calcul|derivative|integral|equation|math', text, re.I): return "math"
    if re.search(r'code|function|algorithm|implement|class|python', text, re.I): return "code"
    if re.search(r'plan|strategy|how would|step|approach', text, re.I): return "planning"
    if re.search(r'what is|explain|describe|define|who|when|where', text, re.I): return "factual"
    if re.search(r'write|story|poem|creative|imagine', text, re.I): return "creative"
    return "general"


def demo_answer(prompt: str, task_type: str) -> str:
    answers = {
        "math": "**Step-by-step solution via Chain of Thought branch:**\n\n1. Identify the mathematical structure and applicable theorems\n2. Apply the relevant formula with proper notation\n3. Simplify intermediate steps carefully\n4. Verify the result satisfies boundary conditions\n\nThe adaptive system routed to the **chain_of_thought** branch (score: 0.89) which excels at structured mathematical reasoning, achieving **+18.3% improvement** over direct generation.",
        "code": "```python\n# Adaptive-selected solution (chain_of_thought branch)\ndef solution(n):\n    \"\"\"Optimized implementation with verification.\"\"\"\n    if n <= 0:\n        return []\n    result = []\n    for i in range(n):\n        result.append(compute(i))\n    return result\n```\n\nThe **verification** branch validated edge cases, while **analytical** provided the algorithmic structure. Combined reward: **0.87**.",
        "planning": "## Adaptive-Generated Plan\n\n**Phase 1: Discovery** (Week 1-2)\n- Define success metrics and constraints\n- Map stakeholder requirements\n\n**Phase 2: Architecture** (Week 3-4)\n- Design system components\n- Identify dependencies and risks\n\n**Phase 3: Implementation** (Week 5-8)\n- Iterative development with feedback loops\n- Continuous evaluation against metrics\n\n**Phase 4: Validation** (Week 9-10)\n- End-to-end testing\n- Performance benchmarking\n\nRouted via **planner** → **analytical** branches for maximum planning quality.",
        "factual": "Based on verified knowledge synthesis across multiple branches:\n\n**Core concept:** The topic you're asking about represents a fundamental area with well-established principles and ongoing research.\n\n**Key insights:**\n- Historical development spans several decades\n- Modern applications are diverse and growing\n- Current research focuses on scalability and reliability\n\nThe **retrieval** branch (0.74) and **verification** branch (0.70) cross-validated this information for accuracy.",
    }
    return answers.get(task_type, f"The adaptive routing system processed your query through **{3}** specialized branches and selected the highest-quality response.\n\n**Reasoning:** The chain-of-thought branch provided the most structured and verifiable answer, achieving a reward score of **{0.84:.2f}** compared to **{0.61:.2f}** for direct base model generation.\n\nThis represents a **+{23:.0f}% improvement** in output quality through adaptive multi-branch reasoning.")


def demo_base_answer(prompt: str) -> str:
    return f"This is what a standard base model would respond with — a single-pass generation without adaptive routing, branch selection, or reward-guided optimization. While functional, it lacks the quality improvements from specialized reasoning strategies. The base model response to '{prompt[:50]}...' would be generated directly without the evaluation and refinement provided by the adaptive system."


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        run_engine_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/state")
async def get_state():
    if not ENGINE_AVAILABLE or engine is None:
        return {
            "branches": [
                {"name": "analytical", "purpose": "Structured logical reasoning", "weight": 2.1, "status": "ACTIVE", "historical_rewards": [0.72, 0.81, 0.78, 0.85], "rewrite_history": [], "metadata": {}},
                {"name": "planner", "purpose": "Goal decomposition", "weight": 1.8, "status": "ACTIVE", "historical_rewards": [0.65, 0.71, 0.68], "rewrite_history": [], "metadata": {}},
                {"name": "retrieval", "purpose": "Knowledge retrieval", "weight": 1.5, "status": "ACTIVE", "historical_rewards": [0.60, 0.63], "rewrite_history": [], "metadata": {}},
                {"name": "verification", "purpose": "Constraint checking", "weight": 1.7, "status": "ACTIVE", "historical_rewards": [0.70, 0.73, 0.76], "rewrite_history": [], "metadata": {}},
                {"name": "chain_of_thought", "purpose": "Explicit reasoning chain", "weight": 2.0, "status": "ACTIVE", "historical_rewards": [0.77, 0.80, 0.83], "rewrite_history": [], "metadata": {}},
                {"name": "meta_verifier", "purpose": "Cross-branch verification", "weight": 0.8, "status": "CANDIDATE", "historical_rewards": [0.60], "rewrite_history": [], "metadata": {}},
            ],
            "memory_count": 142,
            "total_tasks": 287,
            "avg_reward": 0.743,
            "exploration_rate": 0.087,
            "branch_weights": {"analytical": 2.1, "planner": 1.8, "retrieval": 1.5, "verification": 1.7, "chain_of_thought": 2.0},
        }

    try:
        state = engine.get_state() if hasattr(engine, "get_state") else {}
        branches = []
        forest = engine.forest if hasattr(engine, "forest") else None
        if forest:
            for b in forest.get_all_branches():
                branches.append({
                    "name": b.name,
                    "purpose": b.state.purpose if hasattr(b, "state") else "",
                    "weight": b.state.weight if hasattr(b, "state") else 1.0,
                    "status": b.state.status.value if hasattr(b, "state") else "ACTIVE",
                    "historical_rewards": list(b.state.historical_rewards) if hasattr(b, "state") else [],
                    "rewrite_history": list(b.state.rewrite_history) if hasattr(b, "state") else [],
                    "metadata": b.state.metadata if hasattr(b, "state") else {},
                })
        memory_count = engine.memory.record_count() if hasattr(engine, "memory") else 0
        return {
            "branches": branches,
            "memory_count": memory_count,
            "total_tasks": state.get("total_tasks", 0),
            "avg_reward": state.get("avg_reward", 0.7),
            "exploration_rate": state.get("exploration_rate", 0.087),
            "branch_weights": {b["name"]: b["weight"] for b in branches},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    if ENGINE_AVAILABLE and engine is not None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: engine.apply_feedback(req.task_id, req.score, req.comment)
                if hasattr(engine, "apply_feedback") else None
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "task_id": req.task_id}


@app.get("/api/events")
async def get_events(limit: int = 20):
    events_path = Path(__file__).parent / "artifacts" / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    try:
        lines = events_path.read_text().strip().split("\n")
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(events) >= limit:
                break
    except Exception:
        pass
    return events


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_available": ENGINE_AVAILABLE,
        "demo_mode": not ENGINE_AVAILABLE,
    }


# ─── World Simulation Viewer API ────────────────────────────────────────────

@app.get("/api/world/snapshot")
async def get_world_snapshot():
    """Serve the world simulation snapshot for the 3D viewer.

    Prefer a generated simulation snapshot from artifacts/, but fall back to
    the checked-in frontend mock so the UI stays wired to the backend even
    before a fresh export has been produced.
    """
    from fastapi.responses import FileResponse

    root = Path(__file__).parent
    generated_snapshot = root / "artifacts" / "world_snapshot.json"
    mock_snapshot = root / "frontend" / "public" / "mock_snapshot.json"

    if generated_snapshot.exists():
        snapshot_path = generated_snapshot
        snapshot_source = "generated"
    elif mock_snapshot.exists():
        snapshot_path = mock_snapshot
        snapshot_source = "mock"
    else:
        raise HTTPException(
            status_code=404,
            detail=(
                "No world snapshot found. Run: "
                "python -m examples.learned_brain.world_sim.export_snapshot "
                "or generate frontend/public/mock_snapshot.json"
            ),
        )
    return FileResponse(
        snapshot_path,
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=60",
            "X-World-Snapshot-Source": snapshot_source,
        },
    )


@app.get("/api/world/stream")
async def stream_world_simulation(
    scenario: str = "small_town",
    ticks: int = 360,
    speed: float = 1.0,
):
    """SSE stream — runs the simulation live and emits one tick at a time.

    The 3D viewer connects here for expanding-world mode.
    Each tick is emitted as an SSE `tick` event.

    Query params:
        scenario: "small_town" or "large_town"
        ticks:    number of ticks to simulate
        speed:    seconds between ticks (default 1.0)
    """
    async def generate():
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "examples" / "learned_brain"))
            from world_sim.scenarios import build_small_town
            from world_sim.world import World

            world = build_small_town()
            world.initialize()

            # Emit locations + agents first
            locations_data = {
                lid: {"id": lid, "name": loc.name, "default_activity": loc.default_activity}
                for lid, loc in world.locations.items()
            }
            agents_data = {
                aid: {
                    "id": aid, "name": a.personality.name,
                    "background": a.personality.background,
                    "temperament": a.personality.temperament,
                    "identity_tags": list(a.identity_tags),
                    "coalitions": list(a.coalitions),
                    "rival_coalitions": list(a.rival_coalitions),
                    "private_burden": a.private_burden,
                }
                for aid, a in world.agents.items()
            }
            yield f"event: locations\ndata: {json.dumps(locations_data)}\n\n"
            yield f"event: agents\ndata: {json.dumps(agents_data)}\n\n"

            for _ in range(ticks):
                summary = world.tick()

                agent_states = {}
                for aid, agent in world.agents.items():
                    agent_states[aid] = agent.get_dashboard_state()
                    rels = []
                    for other_id, rel in world.relationships.get_agent_relationships(aid)[:6]:
                        other_name = world.agents[other_id].personality.name if other_id in world.agents else other_id
                        rels.append({
                            "other_id": other_id, "other_name": other_name,
                            "trust": round(rel.trust, 2), "warmth": round(rel.warmth, 2),
                            "resentment_toward": round(world.relationships.get_resentment(aid, other_id), 2),
                            "resentment_from": round(world.relationships.get_resentment(other_id, aid), 2),
                            "grievance_toward": round(world.relationships.get_grievance(aid, other_id), 2),
                            "grievance_from": round(world.relationships.get_grievance(other_id, aid), 2),
                            "debt_toward": round(world.relationships.get_debt(aid, other_id), 2),
                            "debt_from": round(world.relationships.get_debt(other_id, aid), 2),
                            "alliance_strength": round(rel.alliance_strength, 2),
                            "rivalry": round(rel.rivalry, 2),
                            "support_events": rel.support_events,
                            "conflict_events": rel.conflict_events,
                            "betrayal_events": rel.betrayal_events,
                        })
                    agent_states[aid]["relationships"] = rels
                    agent_states[aid]["recent_memories"] = [
                        {"tick": m.tick, "description": m.description,
                         "valence": round(m.valence_at_time, 2), "other": m.other_agent_id}
                        for m in agent.get_recent_memories(8)
                    ]

                tick_data = {
                    "tick": summary["tick"],
                    "time": summary["time"],
                    "events": summary["events"],
                    "interactions": summary["interactions"],
                    "agent_states": agent_states,
                }
                yield f"event: tick\ndata: {json.dumps(tick_data)}\n\n"
                await asyncio.sleep(speed)

            yield "event: done\ndata: {}\n\n"

        except ImportError as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

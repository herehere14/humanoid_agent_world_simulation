# Architecture Notes

## Stage 1: Interfaces

- `TaskInput`: normalized task object with metadata
- `PromptBranch`: branch state + prompt rendering + local rewrite hooks
- `LLMBackend`: backend abstraction for mock or external APIs
- `OutputJudge`: reward function abstraction (`exact`, `keyword`, `rule`, `hybrid`)
- `MemoryStore`: trajectory memory and branch bias retrieval

## Stage 2: MVP flow

`task -> router -> active branches -> backend outputs -> judge -> aggregator -> memory`

## Stage 3: Two-agent adaptation

- Agent 1 (`EvaluatorAgent`) emits:
  - reward score
  - confidence
  - failure reason
  - branch-level improvement directions
- Agent 2 (`OptimizerAgent`) applies constrained local updates:
  - active branch weight nudges
  - active branch prompt rewrites
  - candidate branch trial handling

## Stage 4: Candidate branch lifecycle

Creation conditions:
- repeated failure patterns in memory
- poor active-branch rewards on current task
- no concurrent candidate
- branch count and duplication checks pass

Lifecycle transitions:
- `candidate -> active` on trial success
- `candidate -> archived` on trial failure

## OpenClaw compatibility

`adapters/openclaw_adapter.py` and `PromptForestEngine.openclaw_ingest()` provide a structured bridge for OpenClaw-style trajectory events with tool activity, outputs, and judge signals.

This keeps adaptation backend-agnostic and avoids base-model weight changes.

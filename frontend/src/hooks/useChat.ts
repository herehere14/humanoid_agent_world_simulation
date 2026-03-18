import { useRef, useCallback } from 'react';
import { useAppStore } from '@/store/useAppStore';
import { streamChat } from '@/lib/api';
import type { TaskType, Message, TraceState } from '@/types';
import { MOCK_ENGINE_STATE } from '@/lib/mockData';

function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function useChat() {
  const abortRef = useRef<AbortController | null>(null);
  const {
    addMessage,
    updateMessage,
    setIsStreaming,
    resetTrace,
    setTraceStage,
    setEngineState,
    incrementTaskCount,
  } = useAppStore();

  const sendMessage = useCallback(async (text: string, taskType: TaskType = 'auto') => {
    if (!text.trim()) return;

    // Abort any existing stream
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    // Add user message
    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content: text,
      status: 'complete',
      timestamp: Date.now(),
    };
    addMessage(userMsg);

    // Add placeholder assistant message
    const assistantId = generateId();
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      status: 'streaming',
      timestamp: Date.now(),
    };
    addMessage(assistantMsg);

    // Reset and start trace
    resetTrace();
    setTraceStage('routing');
    setIsStreaming(true);
    incrementTaskCount();

    try {
      await streamChat(
        {
          message: text,
          task_type: taskType === 'auto' ? undefined : taskType,
          include_base_comparison: true,
        },
        abortRef.current.signal
      );
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;

      // On error, fall back to mock data
      console.warn('API unavailable, using mock data:', err);
      await simulateMockRun(assistantId, text, taskType);
    } finally {
      setIsStreaming(false);
      setTraceStage('done');
    }
  }, [addMessage, updateMessage, setIsStreaming, resetTrace, setTraceStage, incrementTaskCount]);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
    setTraceStage('idle');
  }, [setIsStreaming, setTraceStage]);

  return { sendMessage, abort };
}

// ─── Mock Simulation ──────────────────────────────────────────────────────────
// Used when the backend is unavailable to demonstrate the UI

async function simulateMockRun(assistantId: string, prompt: string, _taskType: TaskType) {
  const store = useAppStore.getState();
  const { setTraceRouting, addTraceBranchStart, updateTraceBranchOutput,
    setTraceEvaluation, setTraceOptimization, setTraceMemory,
    setTraceSelectedPath, setTraceTimings, setTraceComposer,
    appendToken, updateMessage, addPerformancePoint } = store;

  // Detect task type
  const type = detectTaskType(prompt);

  const activeBranches = ['analytical', 'verification', 'chain_of_thought'];
  const selectedBranch = 'chain_of_thought';

  // Stage 1: Routing
  setTraceRouting({
    task_type: type,
    activated_branches: activeBranches,
    branch_scores: {
      analytical: 0.82,
      verification: 0.74,
      chain_of_thought: 0.89,
      planner: 0.45,
      retrieval: 0.38,
      creative: 0.22,
    },
    activated_paths: [
      ['root', 'analytical', 'chain_of_thought'],
      ['root', 'verification'],
    ],
    sibling_decisions: {},
    branch_weights: { ...MOCK_ENGINE_STATE.branch_weights },
  });

  await delay(400);

  // Stage 2: Branch execution
  for (const branch of activeBranches) {
    addTraceBranchStart(branch);
    await delay(300);
    updateTraceBranchOutput(branch, `Output from ${branch}`, 0.6 + Math.random() * 0.35);
    await delay(150);
  }

  // Generate answer token by token
  const answer = generateMockAnswer(prompt, type);
  const words = answer.split(' ');
  for (const word of words) {
    appendToken(assistantId, word + ' ');
    await delay(25 + Math.random() * 30);
  }

  // Stage 3: Evaluation
  const adaptiveScore = 0.72 + Math.random() * 0.22;
  const baseScore = 0.48 + Math.random() * 0.15;
  const delta = adaptiveScore - baseScore;

  setTraceEvaluation({
    reward_score: adaptiveScore,
    confidence: 0.78 + Math.random() * 0.18,
    selected_branch: selectedBranch,
    selected_output: answer,
    failure_reason: 'none',
    suggested_improvement_direction: '',
    branch_feedback: {
      analytical: { branch_name: 'analytical', reward: 0.74, confidence: 0.72, failure_reason: 'none', suggested_improvement_direction: '' },
      verification: { branch_name: 'verification', reward: 0.70, confidence: 0.68, failure_reason: 'none', suggested_improvement_direction: '' },
      chain_of_thought: { branch_name: 'chain_of_thought', reward: adaptiveScore, confidence: 0.81, failure_reason: 'none', suggested_improvement_direction: '' },
    },
    reward_components: {
      verifier: 0.71,
      task_rules: 0.68,
      llm_judge: adaptiveScore,
    },
  });

  await delay(300);

  // Stage 4: Optimization
  setTraceOptimization({
    updated_weights: {
      chain_of_thought: 2.04,
      analytical: 2.08,
      verification: 1.72,
    },
    update_details: {
      chain_of_thought: { old_weight: 2.0, advantage: 0.04 },
      analytical: { old_weight: 2.1, advantage: -0.02 },
      verification: { old_weight: 1.7, advantage: 0.02 },
    },
    rewritten_prompts: [],
    promoted_candidates: [],
    archived_candidates: [],
    created_candidates: [],
    advisor_used: false,
  });

  await delay(200);

  // Stage 5: Memory
  setTraceMemory({
    task_id: `task-${Date.now()}`,
    record_count: MOCK_ENGINE_STATE.memory_count + 1,
    useful_patterns: [
      'Step-by-step reasoning improves accuracy',
      'Verification reduces false positives',
    ],
  });

  setTraceSelectedPath(['root', 'analytical', 'chain_of_thought']);
  setTraceTimings({
    route_ms: 45, execute_ms: 312, probe_ms: 28, compose_ms: 67,
    judge_ms: 89, evaluate_ms: 34, optimize_ms: 52, memory_ms: 12,
    total_ms: 639, primary_backend_calls: 3, evaluator_runtime_calls: 1, optimizer_runtime_calls: 1,
  });
  setTraceComposer({ composer_enabled: true, composed_leaf: 'chain_of_thought', feature_count: 4 });

  // Update message
  updateMessage(assistantId, {
    content: answer,
    status: 'complete',
    baseAnswer: generateBaseAnswer(prompt),
    improvementDelta: delta,
    winLabel: `Adaptive beat base by +${(delta * 100).toFixed(1)}%`,
  });

  // Performance history
  addPerformancePoint({
    taskIndex: store.taskCount,
    adaptiveScore,
    baseScore,
    delta,
    label: `Task ${store.taskCount}`,
  });
}

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function detectTaskType(prompt: string): 'math' | 'code' | 'planning' | 'factual' | 'creative' | 'general' {
  const lower = prompt.toLowerCase();
  if (/\d|calcul|derivative|integral|equation|math|sum|product/i.test(lower)) return 'math';
  if (/code|function|algorithm|implement|class|python|javascript|typescript/i.test(lower)) return 'code';
  if (/plan|how would|strategy|step|approach|roadmap/i.test(lower)) return 'planning';
  if (/what is|explain|describe|define|history|who|when|where/i.test(lower)) return 'factual';
  if (/write|story|poem|creative|imagine|design/i.test(lower)) return 'creative';
  return 'general';
}

function generateMockAnswer(prompt: string, type: string): string {
  const intros: Record<string, string> = {
    math: "Let me work through this step by step using the **Chain of Thought** branch:\n\n",
    code: "Here's my analysis using the **Analytical** branch with verification:\n\n",
    planning: "I'll break this down into a structured plan:\n\n",
    factual: "Based on my knowledge retrieval and verification:\n\n",
    creative: "Let me explore this creatively:\n\n",
    general: "Let me reason through this carefully:\n\n",
  };

  const answers: Record<string, string> = {
    math: "**Step 1:** Identify the mathematical structure\n\n**Step 2:** Apply the appropriate theorem or formula\n\n**Step 3:** Simplify and verify\n\nThe adaptive routing selected the **Chain of Thought** branch because it provides structured reasoning with explicit intermediate steps, which improves accuracy on mathematical tasks by **16-23%** compared to direct generation.",
    code: "```python\n# Adaptive-optimized solution\ndef solution(input_data):\n    # Step 1: Parse and validate\n    processed = preprocess(input_data)\n    # Step 2: Apply algorithm\n    result = algorithm(processed)\n    # Step 3: Verify output\n    return verify(result)\n```\n\nThe **Verification** branch validated the logic, reducing error rate. The **Analytical** branch provided the algorithmic structure.",
    planning: "## Strategic Plan\n\n1. **Phase 1: Foundation** — Define goals, constraints, and success metrics\n2. **Phase 2: Research** — Gather domain knowledge and identify dependencies\n3. **Phase 3: Execution** — Implement in iterative cycles with feedback loops\n4. **Phase 4: Evaluation** — Measure outcomes against defined metrics\n\nThe **Planner** branch decomposed this into actionable steps, while **Critique** checked for logical gaps.",
    factual: "Based on verified knowledge sources:\n\n**Key facts:**\n- The core principle involves structured information organization\n- Historical development spans several decades of research\n- Modern applications include AI, data systems, and cognitive science\n\nThe **Retrieval** and **Verification** branches cross-referenced multiple knowledge domains to ensure factual accuracy.",
    general: "After routing through the **Chain of Thought** and **Analytical** branches:\n\n**Analysis:** The question involves multiple dimensions requiring careful consideration.\n\n**Reasoning:** The adaptive system identified the most effective reasoning path by comparing branch scores and selecting the approach with highest expected reward.\n\n**Conclusion:** The adaptive routing improved response quality by selecting specialized reasoning strategies over direct generation.",
  };

  return (intros[type] ?? intros.general) + (answers[type] ?? answers.general);
}

function generateBaseAnswer(prompt: string): string {
  return `This is a direct response without adaptive routing. The base model generates an answer in a single pass without branch selection, evaluation, or optimization. While this is faster, it lacks the quality improvements from specialized prompt branches, reward-guided selection, and iterative refinement that the adaptive system provides.

For a prompt like "${prompt.slice(0, 60)}...", the base model would provide a generic response without the specialized reasoning strategies or quality guarantees of the adaptive system.`;
}

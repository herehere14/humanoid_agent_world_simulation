import type { PerformancePoint, EngineState, BranchState } from '@/types';

export const MOCK_BRANCHES: BranchState[] = [
  { name: 'analytical', purpose: 'Structured logical reasoning and step-by-step analysis', weight: 2.1, status: 'ACTIVE', historical_rewards: [0.72, 0.81, 0.78, 0.85, 0.79, 0.83, 0.87, 0.84], rewrite_history: [], metadata: { task_affinity: ['math', 'factual'], latency_avg_ms: 340, token_avg: 420, success_rate: 0.87, memory_bias: 0.72 } },
  { name: 'planner', purpose: 'Goal decomposition and sequential planning', weight: 1.8, status: 'ACTIVE', historical_rewards: [0.65, 0.71, 0.68, 0.74, 0.77, 0.79, 0.76, 0.80], rewrite_history: [], metadata: { task_affinity: ['planning'], latency_avg_ms: 290, token_avg: 380, success_rate: 0.79, memory_bias: 0.65 } },
  { name: 'retrieval', purpose: 'Knowledge retrieval and fact extraction', weight: 1.5, status: 'ACTIVE', historical_rewards: [0.60, 0.63, 0.67, 0.64, 0.70, 0.72, 0.69, 0.74], rewrite_history: [], metadata: { task_affinity: ['factual'], latency_avg_ms: 180, token_avg: 250, success_rate: 0.74, memory_bias: 0.58 } },
  { name: 'critique', purpose: 'Self-evaluation and answer verification', weight: 1.3, status: 'ACTIVE', historical_rewards: [0.55, 0.58, 0.62, 0.60, 0.65, 0.68, 0.66, 0.71], rewrite_history: [], metadata: { task_affinity: ['general'], latency_avg_ms: 310, token_avg: 350, success_rate: 0.68, memory_bias: 0.52 } },
  { name: 'verification', purpose: 'Constraint checking and answer validation', weight: 1.7, status: 'ACTIVE', historical_rewards: [0.70, 0.73, 0.76, 0.74, 0.80, 0.78, 0.82, 0.84], rewrite_history: [], metadata: { task_affinity: ['math', 'code'], latency_avg_ms: 260, token_avg: 300, success_rate: 0.82, memory_bias: 0.68 } },
  { name: 'creative', purpose: 'Open-ended exploration and novel generation', weight: 1.2, status: 'ACTIVE', historical_rewards: [0.50, 0.54, 0.57, 0.55, 0.60, 0.63, 0.61, 0.66], rewrite_history: [], metadata: { task_affinity: ['creative'], latency_avg_ms: 400, token_avg: 520, success_rate: 0.63, memory_bias: 0.45 } },
  { name: 'deep_math', purpose: 'Advanced mathematical reasoning', weight: 2.4, status: 'ACTIVE', historical_rewards: [0.80, 0.84, 0.82, 0.87, 0.90, 0.88, 0.91, 0.93], rewrite_history: [], metadata: { task_affinity: ['math'], latency_avg_ms: 450, token_avg: 600, success_rate: 0.91, memory_bias: 0.85 } },
  { name: 'code_solver', purpose: 'Algorithmic problem solving and code generation', weight: 1.9, status: 'ACTIVE', historical_rewards: [0.75, 0.78, 0.76, 0.82, 0.85, 0.83, 0.87, 0.89], rewrite_history: [], metadata: { task_affinity: ['code'], latency_avg_ms: 380, token_avg: 550, success_rate: 0.86, memory_bias: 0.78 } },
  { name: 'meta_verifier', purpose: 'Cross-branch answer verification', weight: 0.8, status: 'CANDIDATE', historical_rewards: [0.60, 0.63, 0.67], rewrite_history: [], metadata: { task_affinity: ['math', 'factual'], latency_avg_ms: 200, token_avg: 180, success_rate: 0.65, memory_bias: 0.40 } },
  { name: 'chain_of_thought', purpose: 'Explicit reasoning chain generation', weight: 2.0, status: 'ACTIVE', historical_rewards: [0.77, 0.80, 0.83, 0.81, 0.86, 0.84, 0.88, 0.90], rewrite_history: [], metadata: { task_affinity: ['math', 'planning'], latency_avg_ms: 420, token_avg: 580, success_rate: 0.88, memory_bias: 0.80 } },
];

export const MOCK_ENGINE_STATE: EngineState = {
  branches: MOCK_BRANCHES,
  memory_count: 142,
  total_tasks: 287,
  avg_reward: 0.743,
  exploration_rate: 0.087,
  branch_weights: Object.fromEntries(MOCK_BRANCHES.map(b => [b.name, b.weight])),
};

export const MOCK_PERFORMANCE_HISTORY: PerformancePoint[] = Array.from({ length: 20 }, (_, i) => {
  const base = 0.52 + Math.random() * 0.12;
  const adaptive = base + 0.08 + Math.random() * 0.15;
  return {
    taskIndex: i + 1,
    adaptiveScore: Math.min(0.97, adaptive),
    baseScore: base,
    delta: adaptive - base,
    label: `Task ${i + 1}`,
  };
});

export const MACRO_BRANCHES = [
  'analytical', 'planner', 'retrieval', 'critique', 'verification', 'creative'
];

export const NICHE_BRANCHES: Record<string, string[]> = {
  analytical: ['deep_math', 'logical_chain', 'step_decompose'],
  planner: ['goal_tree', 'sequential_plan'],
  retrieval: ['fact_extract', 'knowledge_scan'],
  critique: ['self_eval', 'contradiction_check'],
  verification: ['constraint_verify', 'answer_validate'],
  creative: ['open_explore', 'analogy_map'],
};

export const BRANCH_COLORS: Record<string, string> = {
  analytical: '#0066ff',
  planner: '#7c3aed',
  retrieval: '#10b981',
  critique: '#f59e0b',
  verification: '#3b82f6',
  creative: '#ec4899',
  deep_math: '#0066ff',
  code_solver: '#22d3ee',
  chain_of_thought: '#6366f1',
  meta_verifier: '#f97316',
  default: '#71717a',
};

export function getBranchColor(name: string): string {
  for (const [key, color] of Object.entries(BRANCH_COLORS)) {
    if (name.toLowerCase().includes(key.toLowerCase())) return color;
  }
  return BRANCH_COLORS.default;
}

export function formatBranchName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function getRewardColor(reward: number): string {
  if (reward >= 0.75) return '#10b981';
  if (reward >= 0.5) return '#f59e0b';
  return '#ef4444';
}

export function getRewardLabel(reward: number): string {
  if (reward >= 0.75) return 'HIGH';
  if (reward >= 0.5) return 'MED';
  return 'LOW';
}

export function getStageLabel(stage: string): string {
  const labels: Record<string, string> = {
    idle: 'Idle',
    routing: 'Routing',
    executing: 'Executing Branches',
    evaluating: 'Evaluating',
    optimizing: 'Optimizing',
    done: 'Complete',
  };
  return labels[stage] ?? stage;
}

// ─── Benchmark mock data ───────────────────────────────────────────────────
export const MOCK_BENCHMARK_DATA = {
  rewardOverEpisodes: Array.from({ length: 50 }, (_, i) => ({
    episode: i + 1,
    reward: 0.45 + (0.45 * (1 - Math.exp(-i / 15))) + (Math.random() * 0.06 - 0.03),
    baseline: 0.52 + (Math.random() * 0.04 - 0.02),
  })),
  winRateOverTime: Array.from({ length: 30 }, (_, i) => ({
    task: i + 1,
    winRate: Math.min(0.95, 0.55 + (0.35 * (1 - Math.exp(-i / 10))) + (Math.random() * 0.05 - 0.025)),
  })),
  branchEntropy: Array.from({ length: 30 }, (_, i) => ({
    task: i + 1,
    entropy: 2.2 - (0.8 * (1 - Math.exp(-i / 12))) + (Math.random() * 0.15 - 0.075),
  })),
  latencyVsQuality: MOCK_BRANCHES.filter(b => b.status === 'ACTIVE').map(b => ({
    name: formatBranchName(b.name),
    latency: (b.metadata as Record<string, number>).latency_avg_ms ?? 300,
    quality: b.historical_rewards[b.historical_rewards.length - 1],
    weight: b.weight,
  })),
  shiftComparison: Array.from({ length: 20 }, (_, i) => {
    const shifted = i >= 10;
    return {
      task: i + 1,
      adaptive: shifted
        ? 0.72 + (0.15 * (1 - Math.exp(-(i - 10) / 5))) + (Math.random() * 0.04)
        : 0.80 + (Math.random() * 0.08),
      baseline: shifted
        ? 0.55 - (0.1 * (1 - Math.exp(-(i - 10) / 8))) + (Math.random() * 0.04)
        : 0.58 + (Math.random() * 0.06),
      shifted,
    };
  }),
};

// ─── Preset queries ────────────────────────────────────────────────────────
export const PRESET_QUERIES = {
  coding: [
    'Write a Python function to find the longest palindromic substring',
    'Implement a binary search tree with insert, delete, and balance operations',
    'Create a Redis-like cache with TTL expiry in TypeScript',
  ],
  planning: [
    'Plan a migration from monolith to microservices for an e-commerce platform',
    'Design a CI/CD pipeline for a team of 15 developers',
    'Create a 6-month roadmap for launching a fintech startup',
  ],
  analysis: [
    'Analyze the trade-offs between SQL and NoSQL for a real-time analytics system',
    'Compare event-driven vs request-driven architecture for IoT',
    'Evaluate the security implications of moving to a zero-trust network model',
  ],
  extraction: [
    'Extract all entities, relationships, and key metrics from this quarterly report',
    'Parse and structure the following unstructured medical notes into FHIR format',
    'Extract API endpoints, parameters, and response schemas from this documentation',
  ],
  strategy: [
    'Develop a pricing strategy for a B2B SaaS product entering the enterprise market',
    'Create a go-to-market strategy for an open-source developer tool',
    'Design an A/B testing strategy for a recommendation engine',
  ],
  shift: [
    'Solve this calculus problem step by step, then switch to writing a poem about math',
    'First write Python code, then explain it in Spanish, then critique it',
    'Start with a factual question, then a creative one, then a planning task',
  ],
};

// ─── Memory mock data ──────────────────────────────────────────────────────
export const MOCK_MEMORY_ENTRIES = [
  { id: 'm1', task_type: 'math', pattern: 'step-by-step decomposition', branch_preference: 'deep_math', bias_strength: 0.85, timestamp: Date.now() - 120000, feedback_effect: '+12% reward when using chain_of_thought alongside' },
  { id: 'm2', task_type: 'code', pattern: 'algorithmic optimization', branch_preference: 'code_solver', bias_strength: 0.78, timestamp: Date.now() - 240000, feedback_effect: 'Verification branch catches 23% more edge cases' },
  { id: 'm3', task_type: 'planning', pattern: 'goal decomposition with constraints', branch_preference: 'planner', bias_strength: 0.72, timestamp: Date.now() - 360000, feedback_effect: 'Critique branch improves feasibility score by 18%' },
  { id: 'm4', task_type: 'factual', pattern: 'multi-source fact synthesis', branch_preference: 'retrieval', bias_strength: 0.68, timestamp: Date.now() - 480000, feedback_effect: 'Analytical branch adds 15% confidence to factual claims' },
  { id: 'm5', task_type: 'creative', pattern: 'analogical reasoning', branch_preference: 'creative', bias_strength: 0.55, timestamp: Date.now() - 600000, feedback_effect: 'Lower constraint = higher novelty score' },
  { id: 'm6', task_type: 'math', pattern: 'proof construction', branch_preference: 'chain_of_thought', bias_strength: 0.90, timestamp: Date.now() - 60000, feedback_effect: 'Chain of thought + verification = 94% accuracy' },
];

// ─── Architecture flow steps ───────────────────────────────────────────────
export const ARCHITECTURE_STEPS = [
  { id: 'input', label: 'User Query', icon: 'MessageSquare', description: 'Natural language input from user' },
  { id: 'classify', label: 'Task Understanding', icon: 'Brain', description: 'Classify task type, extract intent and constraints' },
  { id: 'score', label: 'Branch Scoring', icon: 'BarChart3', description: 'Score all branches using weights, affinity, memory' },
  { id: 'route', label: 'Route Selection', icon: 'GitBranch', description: 'Select top-K branches via hierarchical beam search' },
  { id: 'execute', label: 'Backend Execution', icon: 'Cpu', description: 'Execute branch-specific prompts against LLM backend' },
  { id: 'evaluate', label: 'Evaluation', icon: 'CheckCircle', description: 'Score outputs using hybrid reward (keyword, rule, LLM judge)' },
  { id: 'reward', label: 'Reward Update', icon: 'TrendingUp', description: 'Compute advantage signal, update branch weights' },
  { id: 'memory', label: 'Memory Write-back', icon: 'Database', description: 'Store routing context, patterns, and preferences' },
  { id: 'bias', label: 'Future Route Biasing', icon: 'Compass', description: 'Memory-based routing bias for future similar queries' },
];

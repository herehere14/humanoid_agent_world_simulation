// ─── Core Domain Types ───────────────────────────────────────────────────────

export type TaskType =
  | 'math'
  | 'planning'
  | 'factual'
  | 'code'
  | 'creative'
  | 'general'
  | 'auto';

export type BranchStatus = 'ACTIVE' | 'CANDIDATE' | 'ARCHIVED';

export interface BranchState {
  name: string;
  purpose: string;
  weight: number;
  status: BranchStatus;
  historical_rewards: number[];
  rewrite_history: string[];
  metadata: Record<string, unknown>;
}

export interface RoutingDecision {
  task_type: TaskType;
  activated_branches: string[];
  branch_scores: Record<string, number>;
  activated_paths: string[][];
  sibling_decisions: Record<string, Record<string, unknown>>;
}

export interface BranchFeedback {
  branch_name: string;
  reward: number;
  confidence: number;
  failure_reason: string;
  suggested_improvement_direction: string;
}

export interface EvaluationSignal {
  reward_score: number;
  confidence: number;
  selected_branch: string;
  selected_output: string;
  failure_reason: string;
  suggested_improvement_direction: string;
  branch_feedback: Record<string, BranchFeedback>;
}

export interface OptimizationEvent {
  updated_weights: Record<string, number>;
  update_details: Record<string, Record<string, unknown>>;
  rewritten_prompts: string[];
  promoted_candidates: string[];
  archived_candidates: string[];
  created_candidates: string[];
  advisor_used: boolean;
  advisor_error?: string;
}

export interface RewardComponents {
  user_feedback?: number;
  verifier?: number;
  task_rules?: number;
  llm_judge?: number;
}

export interface TimingInfo {
  route_ms: number;
  execute_ms: number;
  probe_ms: number;
  compose_ms: number;
  judge_ms: number;
  evaluate_ms: number;
  optimize_ms: number;
  memory_ms: number;
  total_ms: number;
  primary_backend_calls: number;
  evaluator_runtime_calls: number;
  optimizer_runtime_calls: number;
}

export interface ComposerInfo {
  composer_enabled: boolean;
  composed_leaf: string;
  feature_count: number;
}

// ─── SSE / Streaming Event Types ─────────────────────────────────────────────

export type SSEEventType =
  | 'routing'
  | 'branch_start'
  | 'branch_complete'
  | 'evaluation'
  | 'optimization'
  | 'memory_update'
  | 'token'
  | 'complete'
  | 'error'
  | 'base_complete';

export interface SSEEvent {
  type: SSEEventType;
  data: unknown;
  timestamp: number;
}

export interface RoutingSSEEvent extends SSEEvent {
  type: 'routing';
  data: RoutingDecision & { branch_weights: Record<string, number> };
}

export interface BranchStartSSEEvent extends SSEEvent {
  type: 'branch_start';
  data: { branch_name: string; task_type: TaskType };
}

export interface BranchCompleteSSEEvent extends SSEEvent {
  type: 'branch_complete';
  data: {
    branch_name: string;
    output: string;
    reward?: number;
    reason?: string;
  };
}

export interface EvaluationSSEEvent extends SSEEvent {
  type: 'evaluation';
  data: EvaluationSignal & { reward_components?: RewardComponents };
}

export interface OptimizationSSEEvent extends SSEEvent {
  type: 'optimization';
  data: OptimizationEvent;
}

export interface MemoryUpdateSSEEvent extends SSEEvent {
  type: 'memory_update';
  data: {
    task_id: string;
    record_count: number;
    useful_patterns: string[];
    routing_context_key: string;
  };
}

export interface TokenSSEEvent extends SSEEvent {
  type: 'token';
  data: { token: string };
}

export interface CompleteSSEEvent extends SSEEvent {
  type: 'complete';
  data: {
    task_id: string;
    answer: string;
    base_answer?: string;
    routing: RoutingDecision;
    evaluation: EvaluationSignal;
    optimization: OptimizationEvent;
    reward_components: RewardComponents;
    selected_path: string[];
    timings: TimingInfo;
    composer: ComposerInfo;
    branch_weights: Record<string, number>;
    branch_scores: Record<string, number>;
    improvement_delta?: number;
    win_label?: string;
  };
}

// ─── UI State Types ───────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant';
export type MessageStatus = 'pending' | 'streaming' | 'complete' | 'error';
export type UIMode = 'simple' | 'advanced';
export type PanelView = 'trace' | 'comparison' | 'memory';

export interface TraceState {
  routing?: RoutingDecision & { branch_weights: Record<string, number> };
  activeBranches: string[];
  branchOutputs: Record<string, { output: string; reward?: number; status: 'running' | 'done' | 'pending' }>;
  evaluation?: EvaluationSignal & { reward_components?: RewardComponents };
  optimization?: OptimizationEvent;
  memory?: { task_id: string; record_count: number; useful_patterns: string[] };
  selectedPath: string[];
  timings?: TimingInfo;
  composer?: ComposerInfo;
  stage: 'idle' | 'routing' | 'executing' | 'evaluating' | 'optimizing' | 'done';
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  timestamp: number;
  taskId?: string;
  trace?: TraceState;
  baseAnswer?: string;
  improvementDelta?: number;
  winLabel?: string;
}

export interface PerformancePoint {
  taskIndex: number;
  adaptiveScore: number;
  baseScore: number;
  delta: number;
  label: string;
}

export interface EngineState {
  branches: BranchState[];
  memory_count: number;
  total_tasks: number;
  avg_reward: number;
  exploration_rate: number;
  branch_weights: Record<string, number>;
}

/** Types for the prediction simulation viewer */

export interface SimDecision {
  tick: number;
  time: string;
  agent: string;
  role: string;
  action: string;
  reasoning: string;
  speech: string;
  thought: string;
  consequences: number;
  messages_sent: number;
  message_recipients: string[];
  ripple_count: number;
  latency_ms: number;
  trigger?: string;
  cascade_depth?: number;
  triggered_by?: string;
  conversation_with?: string;
}

export interface RippleChain {
  tick: number;
  actor: string;
  target: string;
  action: string;
  consequence: string;
  mechanism: string;
}

export interface SegmentData {
  segment: string;
  count: number;
  avg_debt: number;
  avg_dread: number;
  avg_pessimism: number;
  avg_tension: number;
  avg_mood: number;
  debt_change: number;
  dread_change: number;
  pessimism_change: number;
  mood_change: number;
  avg_savings_lost: number;
  avg_income_lost: number;
}

export interface KeyFigure {
  name: string;
  title: string;
  org: string;
  emotion: string;
  internal_emotion: string;
  divergence: number;
  tension: number;
  dread: number;
  concern: string;
  ongoing_story: string;
  blame_target: string;
  coalitions: string[];
  mood_change: number;
  policy_reactions: string[];
}

export interface Insight {
  type: string;
  title: string;
  detail: string;
  surprise_factor: number;
  segment?: string;
  person?: string;
  coalition?: string;
  target?: string;
  sector?: string;
  role?: string;
  count?: number;
  policies?: string[];
}

export interface PolicyImpact {
  domain: string;
  policy: string;
  description: string;
  winners: string[];
  losers: string[];
  sectors: Record<string, number>;
  agents_affected: number;
}

export interface MacroData {
  current: Record<string, number>;
  baseline: Record<string, number>;
  deltas: Record<string, number>;
}

export interface SectorReport {
  booming: Record<string, number>;
  struggling: Record<string, number>;
  all_sectors: Record<string, number>;
}

export interface LLMAgencyData {
  enabled: boolean;
  total_calls: number;
  total_decisions: number;
  total_reactive?: number;
  total_conversations?: number;
  total_messages_sent?: number;
  max_cascade_depth?: number;
  full_decision_log: SimDecision[];
}

export interface SimulationReport {
  prediction: string;
  days_simulated: number;
  total_agents: number;
  macro: MacroData;
  sectors: SectorReport;
  segments: SegmentData[];
  key_figures: KeyFigure[];
  insights: Insight[];
  policy_impacts: PolicyImpact[];
  ripple_chains: RippleChain[];
  ripple_summary: { total_ripple_events: number; by_mechanism: Record<string, number> };
  llm_agency: LLMAgencyData;
  top_concerns: [string, number][];
  emotion_distribution: [string, number][];
}

/** Types mirroring the simulation data model */

export interface LocationMeta {
  id: string;
  name: string;
  default_activity: string;
}

export interface AgentMeta {
  id: string;
  name: string;
  background: string;
  temperament: string;
  identity_tags: string[];
  coalitions: string[];
  rival_coalitions: string[];
  private_burden: string;
}

export interface RelationshipEntry {
  other_id: string;
  other_name: string;
  trust: number;
  warmth: number;
  resentment_toward: number;
  resentment_from: number;
  grievance_toward: number;
  grievance_from: number;
  debt_toward: number;
  debt_from: number;
  alliance_strength: number;
  rivalry: number;
  support_events: number;
  conflict_events: number;
  betrayal_events: number;
}

export interface MemoryEntry {
  tick: number;
  description: string;
  valence: number;
  other: string | null;
}

export interface FutureBranch {
  label: string;
  summary: string;
}

export interface AgentState {
  id: string;
  name: string;
  location: string;
  action: string;
  arousal: number;
  valence: number;
  tension: number;
  impulse_control: number;
  energy: number;
  vulnerability: number;
  surface: string;
  internal: string;
  divergence: number;
  primary_concern: string;
  interpretation: string;
  blame_target: string;
  support_target: string;
  economic_pressure: number;
  loyalty_pressure: number;
  secrecy_pressure: number;
  opportunity_pressure: number;
  priority_motive: string;
  mask_style: string;
  action_style: string;
  inner_voice: string;
  future_branches: FutureBranch[];
  coalitions: string[];
  identity_tags: string[];
  private_burden: string;
  debt_pressure: number;
  secret_pressure: number;
  ambition: number;
  relationships: RelationshipEntry[];
  recent_memories: MemoryEntry[];
}

export interface InteractionRecord {
  agent_a: string;
  agent_b: string;
  type: 'conflict' | 'support' | 'positive' | 'neutral';
  location: string;
  issue?: string;
  action_a?: string;
  action_b?: string;
}

export interface EventRecord {
  location: string;
  description: string;
  targets: string[] | null;
  kind: string;
}

export interface TickData {
  tick: number;
  time: string;
  events: EventRecord[];
  interactions: InteractionRecord[];
  agent_states: Record<string, AgentState>;
}

export interface WorldSnapshot {
  scenario: string;
  total_ticks: number;
  locations: Record<string, LocationMeta>;
  agents: Record<string, AgentMeta>;
  ticks: TickData[];
}

/** Action types from the simulation */
export type AgentAction =
  | 'COLLAPSE' | 'LASH_OUT' | 'CONFRONT' | 'FLEE'
  | 'WITHDRAW' | 'SEEK_COMFORT' | 'RUMINATE' | 'VENT'
  | 'SOCIALIZE' | 'CELEBRATE' | 'HELP_OTHERS'
  | 'WORK' | 'REST' | 'IDLE';

/** 3D world layout positions for locations */
export interface LocationLayout {
  id: string;
  position: [number, number, number];
  size: [number, number, number];
  color: string;
  label: string;
}

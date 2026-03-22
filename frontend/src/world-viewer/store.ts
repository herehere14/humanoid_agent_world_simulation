/** Zustand store for the world viewer.
 *
 *  Supports expanding worlds — new ticks, locations, and agents can be
 *  appended at any time via appendTicks / addLocations / addAgents.
 *  The layout regenerates automatically when new locations appear.
 */

import { create } from 'zustand';
import type {
  WorldSnapshot, TickData, AgentState, AgentMeta, LocationMeta,
} from './types';
import { initLayout } from './layout';

interface WorldStore {
  // Data
  snapshot: WorldSnapshot | null;
  loading: boolean;
  error: string | null;

  // Playback
  currentTick: number;
  playing: boolean;
  speed: number;
  maxTick: number;

  // Selection
  selectedAgentId: string | null;
  hoveredAgentId: string | null;
  inspectorOpen: boolean;

  // Derived getters
  getCurrentTickData: () => TickData | null;
  getAgentState: (id: string) => AgentState | null;
  getAgentMeta: (id: string) => AgentMeta | null;
  getLocationMeta: (id: string) => LocationMeta | null;
  getSelectedAgent: () => AgentState | null;

  // Core actions
  loadSnapshot: (data: WorldSnapshot) => void;
  setTick: (tick: number) => void;
  advanceTick: () => void;
  togglePlay: () => void;
  setSpeed: (speed: number) => void;
  selectAgent: (id: string | null) => void;
  hoverAgent: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // ─── Expanding world API ───────────────────────────────
  /** Append new tick data — the world grows forward in time */
  appendTicks: (ticks: TickData[]) => void;
  /** Register new locations — the town layout regenerates to accommodate them */
  addLocations: (locations: Record<string, LocationMeta>) => void;
  /** Register new agents — they appear in the world on their first tick */
  addAgents: (agents: Record<string, AgentMeta>) => void;
}

export const useWorldStore = create<WorldStore>((set, get) => ({
  snapshot: null,
  loading: false,
  error: null,

  currentTick: 0,
  playing: false,
  speed: 2,
  maxTick: 0,

  selectedAgentId: null,
  hoveredAgentId: null,
  inspectorOpen: false,

  getCurrentTickData: () => {
    const { snapshot, currentTick } = get();
    if (!snapshot || currentTick < 0 || currentTick >= snapshot.ticks.length) return null;
    return snapshot.ticks[currentTick];
  },

  getAgentState: (id) => {
    const tick = get().getCurrentTickData();
    return tick?.agent_states[id] ?? null;
  },

  getAgentMeta: (id) => get().snapshot?.agents[id] ?? null,
  getLocationMeta: (id) => get().snapshot?.locations[id] ?? null,

  getSelectedAgent: () => {
    const { selectedAgentId } = get();
    return selectedAgentId ? get().getAgentState(selectedAgentId) : null;
  },

  // ─── load full snapshot ───────────────────────────────
  loadSnapshot: (data) => {
    initLayout(data.locations);
    set({
      snapshot: data,
      maxTick: data.ticks.length - 1,
      currentTick: 0,
      loading: false,
      error: null,
    });
  },

  // ─── expanding world: append ticks ────────────────────
  appendTicks: (newTicks) => {
    const { snapshot } = get();
    if (!snapshot) return;

    // Scan new ticks for any previously-unseen locations or agents
    // and auto-register them so the world grows naturally.
    const newLocations: Record<string, LocationMeta> = {};
    const newAgents: Record<string, AgentMeta> = {};

    for (const tick of newTicks) {
      for (const [id, state] of Object.entries(tick.agent_states)) {
        // new location?
        if (state.location && !snapshot.locations[state.location] && !newLocations[state.location]) {
          newLocations[state.location] = {
            id: state.location,
            name: state.location.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            default_activity: 'Activity',
          };
        }
        // new agent?
        if (!snapshot.agents[id] && !newAgents[id]) {
          newAgents[id] = {
            id,
            name: state.name ?? id,
            background: '',
            temperament: '',
            identity_tags: state.identity_tags ?? [],
            coalitions: state.coalitions ?? [],
            rival_coalitions: [],
            private_burden: state.private_burden ?? '',
          };
        }
      }
    }

    const mergedLocations = { ...snapshot.locations, ...newLocations };
    const mergedAgents = { ...snapshot.agents, ...newAgents };
    const mergedTicks = [...snapshot.ticks, ...newTicks];

    // Regenerate layout if new locations appeared
    if (Object.keys(newLocations).length > 0) {
      initLayout(mergedLocations);
    }

    set({
      snapshot: {
        ...snapshot,
        locations: mergedLocations,
        agents: mergedAgents,
        ticks: mergedTicks,
        total_ticks: mergedTicks.length,
      },
      maxTick: mergedTicks.length - 1,
    });
  },

  // ─── expanding world: add locations ───────────────────
  addLocations: (locations) => {
    const { snapshot } = get();
    if (!snapshot) return;
    const merged = { ...snapshot.locations, ...locations };
    initLayout(merged);
    set({
      snapshot: { ...snapshot, locations: merged },
    });
  },

  // ─── expanding world: add agents ──────────────────────
  addAgents: (agents) => {
    const { snapshot } = get();
    if (!snapshot) return;
    set({
      snapshot: { ...snapshot, agents: { ...snapshot.agents, ...agents } },
    });
  },

  // ─── playback ─────────────────────────────────────────
  setTick: (tick) => set({ currentTick: Math.max(0, Math.min(tick, get().maxTick)) }),

  advanceTick: () => {
    const { currentTick, maxTick, playing } = get();
    if (!playing) return;
    if (currentTick >= maxTick) {
      set({ playing: false });
      return;
    }
    set({ currentTick: currentTick + 1 });
  },

  togglePlay: () => set(s => ({ playing: !s.playing })),
  setSpeed: (speed) => set({ speed: Math.max(0.5, Math.min(30, speed)) }),

  selectAgent: (id) => set({ selectedAgentId: id, inspectorOpen: id !== null }),
  hoverAgent: (id) => set({ hoveredAgentId: id }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error, loading: false }),
}));

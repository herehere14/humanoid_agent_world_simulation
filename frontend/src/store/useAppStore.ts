import { create } from 'zustand';
import type {
  Message,
  TraceState,
  UIMode,
  PanelView,
  PerformancePoint,
  EngineState,
  RoutingDecision,
  EvaluationSignal,
  OptimizationEvent,
  RewardComponents,
  TimingInfo,
  ComposerInfo,
} from '@/types';

interface AppState {
  // Messages
  messages: Message[];
  addMessage: (msg: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  appendToken: (id: string, token: string) => void;

  // Active trace (current processing)
  activeTrace: TraceState;
  setTraceStage: (stage: TraceState['stage']) => void;
  setTraceRouting: (routing: RoutingDecision & { branch_weights: Record<string, number> }) => void;
  addTraceBranchStart: (branchName: string) => void;
  updateTraceBranchOutput: (branchName: string, output: string, reward?: number) => void;
  setTraceEvaluation: (evaluation: EvaluationSignal & { reward_components?: RewardComponents }) => void;
  setTraceOptimization: (optimization: OptimizationEvent) => void;
  setTraceMemory: (memory: { task_id: string; record_count: number; useful_patterns: string[] }) => void;
  setTraceSelectedPath: (path: string[]) => void;
  setTraceTimings: (timings: TimingInfo) => void;
  setTraceComposer: (composer: ComposerInfo) => void;
  resetTrace: () => void;

  // UI state
  uiMode: UIMode;
  setUIMode: (mode: UIMode) => void;
  panelView: PanelView;
  setPanelView: (view: PanelView) => void;
  traceExpanded: boolean;
  setTraceExpanded: (v: boolean) => void;
  comparisonExpanded: boolean;
  setComparisonExpanded: (v: boolean) => void;
  isStreaming: boolean;
  setIsStreaming: (v: boolean) => void;

  // Performance history
  performanceHistory: PerformancePoint[];
  addPerformancePoint: (point: PerformancePoint) => void;

  // Engine state
  engineState: EngineState | null;
  setEngineState: (state: EngineState) => void;

  // Task counter
  taskCount: number;
  incrementTaskCount: () => void;
}

const defaultTrace: TraceState = {
  activeBranches: [],
  branchOutputs: {},
  selectedPath: [],
  stage: 'idle',
};

export const useAppStore = create<AppState>((set) => ({
  messages: [],
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  updateMessage: (id, updates) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, ...updates } : m)),
    })),
  appendToken: (id, token) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m
      ),
    })),

  activeTrace: defaultTrace,
  setTraceStage: (stage) =>
    set((s) => ({ activeTrace: { ...s.activeTrace, stage } })),
  setTraceRouting: (routing) =>
    set((s) => ({
      activeTrace: {
        ...s.activeTrace,
        routing,
        activeBranches: routing.activated_branches,
        stage: 'executing',
      },
    })),
  addTraceBranchStart: (branchName) =>
    set((s) => ({
      activeTrace: {
        ...s.activeTrace,
        branchOutputs: {
          ...s.activeTrace.branchOutputs,
          [branchName]: { output: '', status: 'running' },
        },
      },
    })),
  updateTraceBranchOutput: (branchName, output, reward) =>
    set((s) => ({
      activeTrace: {
        ...s.activeTrace,
        branchOutputs: {
          ...s.activeTrace.branchOutputs,
          [branchName]: { output, reward, status: 'done' },
        },
      },
    })),
  setTraceEvaluation: (evaluation) =>
    set((s) => ({
      activeTrace: { ...s.activeTrace, evaluation, stage: 'evaluating' },
    })),
  setTraceOptimization: (optimization) =>
    set((s) => ({
      activeTrace: { ...s.activeTrace, optimization, stage: 'optimizing' },
    })),
  setTraceMemory: (memory) =>
    set((s) => ({ activeTrace: { ...s.activeTrace, memory } })),
  setTraceSelectedPath: (selectedPath) =>
    set((s) => ({ activeTrace: { ...s.activeTrace, selectedPath } })),
  setTraceTimings: (timings) =>
    set((s) => ({ activeTrace: { ...s.activeTrace, timings } })),
  setTraceComposer: (composer) =>
    set((s) => ({ activeTrace: { ...s.activeTrace, composer } })),
  resetTrace: () => set({ activeTrace: defaultTrace }),

  uiMode: 'advanced',
  setUIMode: (uiMode) => set({ uiMode }),
  panelView: 'trace',
  setPanelView: (panelView) => set({ panelView }),
  traceExpanded: true,
  setTraceExpanded: (traceExpanded) => set({ traceExpanded }),
  comparisonExpanded: true,
  setComparisonExpanded: (comparisonExpanded) => set({ comparisonExpanded }),
  isStreaming: false,
  setIsStreaming: (isStreaming) => set({ isStreaming }),

  performanceHistory: [],
  addPerformancePoint: (point) =>
    set((s) => ({
      performanceHistory: [...s.performanceHistory.slice(-19), point],
    })),

  engineState: null,
  setEngineState: (engineState) => set({ engineState }),

  taskCount: 0,
  incrementTaskCount: () => set((s) => ({ taskCount: s.taskCount + 1 })),
}));

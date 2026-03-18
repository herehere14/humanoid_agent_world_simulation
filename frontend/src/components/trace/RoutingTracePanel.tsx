import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare, Brain, BarChart3, GitBranch, Cpu,
  CheckCircle, TrendingUp, Database, ChevronDown, ChevronRight
} from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { SectionHeading, Chip, Bar } from '@/components/ui';
import { getBranchColor, formatBranchName, getRewardColor } from '@/lib/mockData';

interface TraceStep {
  id: string;
  label: string;
  icon: React.ElementType;
  color: string;
  stage: string;
  getDetails: () => React.ReactNode;
}

export function RoutingTracePanel() {
  const activeTrace = useAppStore((s) => s.activeTrace);
  const messages = useAppStore((s) => s.messages);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [visibleSteps, setVisibleSteps] = useState(0);

  const lastMessage = [...messages].reverse().find((m) => m.role === 'assistant');

  // Animate steps appearing sequentially
  useEffect(() => {
    if (activeTrace.stage === 'idle') {
      setVisibleSteps(0);
      return;
    }
    const stageOrder = ['routing', 'executing', 'evaluating', 'optimizing', 'done'];
    const currentIdx = stageOrder.indexOf(activeTrace.stage);
    const target = Math.max(0, currentIdx + 2); // Show current + 1 step
    if (target > visibleSteps) {
      const timer = setTimeout(() => setVisibleSteps(target), 200);
      return () => clearTimeout(timer);
    }
  }, [activeTrace.stage, visibleSteps]);

  const toggleStep = (id: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const traceSteps: TraceStep[] = [
    {
      id: 'input',
      label: 'Input Received',
      icon: MessageSquare,
      color: '#e4e4e7',
      stage: 'routing',
      getDetails: () => {
        const lastUser = [...messages].reverse().find((m) => m.role === 'user');
        return lastUser ? (
          <p className="text-xs text-text-secondary">{lastUser.content.slice(0, 120)}{lastUser.content.length > 120 ? '...' : ''}</p>
        ) : <p className="text-xs text-text-muted">Waiting for input...</p>;
      },
    },
    {
      id: 'classify',
      label: 'Task Classified',
      icon: Brain,
      color: '#7c3aed',
      stage: 'routing',
      getDetails: () => activeTrace.routing ? (
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted">Type:</span>
            <Chip color="purple">{activeTrace.routing.task_type}</Chip>
          </div>
        </div>
      ) : null,
    },
    {
      id: 'score',
      label: 'Candidate Branches Scored',
      icon: BarChart3,
      color: '#3b82f6',
      stage: 'routing',
      getDetails: () => activeTrace.routing ? (
        <div className="space-y-2">
          {Object.entries(activeTrace.routing.branch_scores)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 6)
            .map(([name, score]) => (
              <div key={name} className="flex items-center gap-2">
                <span className="text-[10px] text-text-secondary w-24 truncate">{formatBranchName(name)}</span>
                <Bar value={score} color={getBranchColor(name)} height={3} className="flex-1" />
                <span className="text-[10px] font-mono text-text-muted w-8 text-right">{score.toFixed(2)}</span>
              </div>
            ))}
        </div>
      ) : null,
    },
    {
      id: 'route',
      label: 'Route Selected',
      icon: GitBranch,
      color: '#0066ff',
      stage: 'executing',
      getDetails: () => activeTrace.routing ? (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-1">
            {activeTrace.activeBranches.map((b) => (
              <Chip key={b} color="cyan" dot>{formatBranchName(b)}</Chip>
            ))}
          </div>
          {activeTrace.routing.activated_paths.length > 0 && (
            <div className="text-[10px] text-text-muted">
              Path: {activeTrace.routing.activated_paths[0]?.join(' → ')}
            </div>
          )}
        </div>
      ) : null,
    },
    {
      id: 'execute',
      label: 'Branch Execution',
      icon: Cpu,
      color: '#22d3ee',
      stage: 'executing',
      getDetails: () => (
        <div className="space-y-2">
          {Object.entries(activeTrace.branchOutputs).map(([name, info]) => (
            <div key={name} className="flex items-center gap-2">
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: info.status === 'running' ? '#f59e0b' : info.status === 'done' ? '#10b981' : '#71717a' }}
              />
              <span className="text-[10px] text-text-secondary flex-1">{formatBranchName(name)}</span>
              <span className="text-[10px] font-mono text-text-muted">
                {info.status === 'running' ? 'Running...' : info.status === 'done' ? `${info.output.length} chars` : 'Pending'}
              </span>
              {info.reward !== undefined && (
                <span className="text-[10px] font-mono" style={{ color: getRewardColor(info.reward) }}>
                  {info.reward.toFixed(2)}
                </span>
              )}
            </div>
          ))}
        </div>
      ),
    },
    {
      id: 'evaluate',
      label: 'Evaluator Score Returned',
      icon: CheckCircle,
      color: '#10b981',
      stage: 'evaluating',
      getDetails: () => activeTrace.evaluation ? (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div>
              <span className="text-[10px] text-text-muted">Reward</span>
              <div className="text-sm font-mono font-semibold" style={{ color: getRewardColor(activeTrace.evaluation.reward_score) }}>
                {activeTrace.evaluation.reward_score.toFixed(3)}
              </div>
            </div>
            <div>
              <span className="text-[10px] text-text-muted">Confidence</span>
              <div className="text-sm font-mono font-semibold text-text-base">
                {activeTrace.evaluation.confidence.toFixed(2)}
              </div>
            </div>
            <div>
              <span className="text-[10px] text-text-muted">Selected</span>
              <div className="text-sm font-medium text-primary">
                {formatBranchName(activeTrace.evaluation.selected_branch)}
              </div>
            </div>
          </div>
          {activeTrace.evaluation.reward_components && (
            <div className="grid grid-cols-2 gap-1 text-[10px]">
              {Object.entries(activeTrace.evaluation.reward_components).map(([key, val]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-text-muted">{key}</span>
                  <span className="font-mono text-text-secondary">{(val as number)?.toFixed(2) ?? '—'}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : null,
    },
    {
      id: 'reward',
      label: 'Reward Update Applied',
      icon: TrendingUp,
      color: '#f59e0b',
      stage: 'optimizing',
      getDetails: () => activeTrace.optimization ? (
        <div className="space-y-2">
          {Object.entries(activeTrace.optimization.updated_weights).map(([name, weight]) => {
            const detail = activeTrace.optimization!.update_details[name] as Record<string, number> | undefined;
            const delta = detail?.delta ?? 0;
            return (
              <div key={name} className="flex items-center gap-2 text-[10px]">
                <span className="text-text-secondary w-24 truncate">{formatBranchName(name)}</span>
                <span className="font-mono text-text-base">{(weight as number).toFixed(2)}</span>
                {delta !== 0 && (
                  <span className={`font-mono ${delta > 0 ? 'text-success-DEFAULT' : 'text-danger-DEFAULT'}`}>
                    {delta > 0 ? '+' : ''}{delta.toFixed(3)}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      ) : null,
    },
    {
      id: 'memory',
      label: 'Memory Written',
      icon: Database,
      color: '#6366f1',
      stage: 'done',
      getDetails: () => activeTrace.memory ? (
        <div className="space-y-1 text-[10px]">
          <div className="flex justify-between">
            <span className="text-text-muted">Records</span>
            <span className="font-mono text-text-base">{activeTrace.memory.record_count}</span>
          </div>
          {activeTrace.memory.useful_patterns.length > 0 && (
            <div>
              <span className="text-text-muted">Patterns:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {activeTrace.memory.useful_patterns.map((p, i) => (
                  <Chip key={i} color="purple">{p}</Chip>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : null,
    },
  ];

  const completionTime = lastMessage?.trace?.timings?.total_ms;

  return (
    <section id="trace" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Real-Time Trace"
          title="Routing Decision Pipeline"
          subtitle="Watch every step of the adaptive routing process unfold in real-time as your query is processed."
          center
        />

        <div className="mt-12 max-w-2xl mx-auto">
          {/* Completion time */}
          {completionTime && (
            <div className="text-center mb-6">
              <Chip color="green" dot>Completed in {completionTime.toFixed(0)}ms</Chip>
            </div>
          )}

          {/* Steps */}
          <div className="relative">
            {/* Vertical line */}
            <div className="absolute left-[19px] top-0 bottom-0 w-px bg-border" />

            <div className="space-y-1">
              {traceSteps.map((step, idx) => {
                const stageOrder = ['routing', 'executing', 'evaluating', 'optimizing', 'done'];
                const stepStageIdx = stageOrder.indexOf(step.stage);
                const currentStageIdx = stageOrder.indexOf(activeTrace.stage);
                const isReached = activeTrace.stage !== 'idle' && stepStageIdx <= currentStageIdx;
                const isCurrent = step.stage === activeTrace.stage;
                const isExpanded = expandedSteps.has(step.id);
                const isVisible = activeTrace.stage === 'idle' || idx < visibleSteps;

                if (!isVisible && activeTrace.stage !== 'idle') return null;

                return (
                  <motion.div
                    key={step.id}
                    initial={activeTrace.stage !== 'idle' ? { opacity: 0, x: -10 } : false}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="relative"
                  >
                    <button
                      onClick={() => toggleStep(step.id)}
                      className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-[rgba(0,0,0,0.02)] transition-colors text-left group"
                    >
                      {/* Icon circle */}
                      <div
                        className="w-10 h-10 rounded-full flex items-center justify-center shrink-0 relative z-10 transition-all"
                        style={{
                          background: isReached ? step.color + '20' : 'rgba(0,0,0,0.03)',
                          boxShadow: isCurrent ? `0 0 12px ${step.color}30` : undefined,
                        }}
                      >
                        <step.icon
                          size={16}
                          style={{ color: isReached ? step.color : '#71717a' }}
                        />
                        {isCurrent && (
                          <span className="absolute inset-0 rounded-full animate-ping opacity-20"
                            style={{ background: step.color }} />
                        )}
                      </div>

                      {/* Label */}
                      <span className={`flex-1 text-sm font-medium ${isReached ? 'text-text-base' : 'text-text-muted'}`}>
                        {step.label}
                      </span>

                      {/* Status */}
                      {isReached && (
                        <span className="text-[10px] font-mono text-text-muted">
                          {isCurrent ? 'Active' : 'Done'}
                        </span>
                      )}

                      {/* Expand icon */}
                      {isReached && (
                        isExpanded
                          ? <ChevronDown size={14} className="text-text-muted" />
                          : <ChevronRight size={14} className="text-text-muted" />
                      )}
                    </button>

                    {/* Expanded detail */}
                    <AnimatePresence>
                      {isExpanded && isReached && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="overflow-hidden"
                        >
                          <div className="ml-[52px] mr-3 mb-3 p-3 rounded-lg bg-[rgba(0,0,0,0.02)] border border-border">
                            {step.getDetails()}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Empty state */}
          {activeTrace.stage === 'idle' && (
            <div className="text-center py-8">
              <p className="text-sm text-text-muted">Run a query above to see the routing trace in real-time</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

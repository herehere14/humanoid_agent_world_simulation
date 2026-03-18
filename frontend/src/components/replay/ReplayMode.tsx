import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Rewind, FastForward, Clock, GitBranch, Brain, BarChart3 } from 'lucide-react';
import { SectionHeading, Card, GlowButton, Chip, Bar } from '@/components/ui';
import { useAppStore } from '@/store/useAppStore';
import { formatBranchName, getBranchColor, getRewardColor } from '@/lib/mockData';

type ReplayTab = 'routing' | 'timeline' | 'memory' | 'alternate';

export function ReplayMode() {
  const [activeTab, setActiveTab] = useState<ReplayTab>('routing');
  const [replayStep, setReplayStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const messages = useAppStore((s) => s.messages);

  const lastComplete = [...messages].reverse().find(
    (m) => m.role === 'assistant' && m.status === 'complete' && m.trace
  );
  const trace = lastComplete?.trace;

  const tabs: { key: ReplayTab; label: string; icon: React.ElementType }[] = [
    { key: 'routing', label: 'Replay Routing', icon: Play },
    { key: 'timeline', label: 'Activation Timeline', icon: Clock },
    { key: 'memory', label: 'Memory Effect', icon: Brain },
    { key: 'alternate', label: 'Alternate Path', icon: GitBranch },
  ];

  const startReplay = () => {
    setIsPlaying(true);
    setReplayStep(0);
    let s = 0;
    const steps = ['input', 'classify', 'score', 'route', 'execute', 'evaluate', 'optimize', 'memory'];
    const interval = setInterval(() => {
      s++;
      setReplayStep(s);
      if (s >= steps.length - 1) {
        clearInterval(interval);
        setIsPlaying(false);
      }
    }, 800);
  };

  const replayStages = [
    { label: 'Input Received', color: '#e4e4e7' },
    { label: 'Task Classified', color: '#7c3aed' },
    { label: 'Branches Scored', color: '#3b82f6' },
    { label: 'Route Selected', color: '#0066ff' },
    { label: 'Branches Executing', color: '#22d3ee' },
    { label: 'Evaluation Complete', color: '#10b981' },
    { label: 'Weights Optimized', color: '#f59e0b' },
    { label: 'Memory Updated', color: '#6366f1' },
  ];

  return (
    <section id="replay" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Deep Inspection"
          title="Replay & Inspect"
          subtitle="Replay any routing decision, inspect the activation timeline, analyze memory effects, and explore alternate paths."
          center
        />

        <div className="mt-12 max-w-3xl mx-auto space-y-6">
          {/* Tabs */}
          <div className="flex justify-center gap-2">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
                  activeTab === tab.key
                    ? 'bg-primary-light text-primary border border-primary/20'
                    : 'text-text-muted hover:text-text-secondary hover:bg-[rgba(0,0,0,0.03)]'
                }`}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </div>

          {!trace ? (
            <Card className="p-8 text-center">
              <BarChart3 size={28} className="mx-auto text-text-muted mb-3" />
              <p className="text-sm text-text-secondary">Run a query first to replay its routing decision</p>
            </Card>
          ) : (
            <Card className="p-6">
              {/* Routing Replay */}
              {activeTab === 'routing' && (
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-text-base">Routing Replay</h3>
                    <GlowButton onClick={startReplay} disabled={isPlaying} size="sm">
                      <Play size={12} /> {isPlaying ? 'Replaying...' : 'Replay'}
                    </GlowButton>
                  </div>

                  {/* Timeline bar */}
                  <div className="flex items-center gap-1">
                    {replayStages.map((stage, idx) => (
                      <div key={idx} className="flex-1 flex flex-col items-center gap-1">
                        <motion.div
                          className="w-full h-2 rounded-full"
                          animate={{
                            background: idx <= replayStep ? stage.color : 'rgba(0,0,0,0.03)',
                            boxShadow: idx === replayStep ? `0 0 8px ${stage.color}40` : 'none',
                          }}
                          transition={{ duration: 0.3 }}
                        />
                        <span className={`text-[8px] transition-colors ${
                          idx <= replayStep ? 'text-text-secondary' : 'text-text-placeholder'
                        }`}>
                          {stage.label}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Current step detail */}
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={replayStep}
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -5 }}
                      className="p-4 rounded-lg bg-[rgba(0,0,0,0.02)] border border-border"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ background: replayStages[replayStep]?.color }}
                        />
                        <span className="text-xs font-medium text-text-base">
                          {replayStages[replayStep]?.label}
                        </span>
                      </div>
                      <p className="text-xs text-text-muted">
                        Step {replayStep + 1} of {replayStages.length}
                        {trace.routing && replayStep >= 3 && (
                          <span> — Activated: {trace.activeBranches.map(formatBranchName).join(', ')}</span>
                        )}
                      </p>
                    </motion.div>
                  </AnimatePresence>
                </div>
              )}

              {/* Activation Timeline */}
              {activeTab === 'timeline' && (
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-text-base">Activation Timeline</h3>
                  {trace.timings && (
                    <div className="space-y-3">
                      {[
                        { label: 'Route', ms: trace.timings.route_ms, color: '#0066ff' },
                        { label: 'Execute', ms: trace.timings.execute_ms, color: '#22d3ee' },
                        { label: 'Evaluate', ms: trace.timings.evaluate_ms, color: '#10b981' },
                        { label: 'Optimize', ms: trace.timings.optimize_ms, color: '#f59e0b' },
                        { label: 'Memory', ms: trace.timings.memory_ms, color: '#6366f1' },
                      ].map((t) => (
                        <div key={t.label} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-text-secondary">{t.label}</span>
                            <span className="font-mono text-text-muted">{t.ms.toFixed(0)}ms</span>
                          </div>
                          <Bar value={t.ms} max={trace.timings!.total_ms} color={t.color} height={4} />
                        </div>
                      ))}
                      <div className="pt-2 border-t border-border flex justify-between text-xs">
                        <span className="text-text-base font-medium">Total</span>
                        <span className="font-mono text-primary">{trace.timings.total_ms.toFixed(0)}ms</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Memory Effect */}
              {activeTab === 'memory' && (
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-text-base">Memory Effect on Routing</h3>
                  {trace.memory ? (
                    <div className="space-y-3">
                      <div className="flex justify-between text-xs">
                        <span className="text-text-muted">Records accessed</span>
                        <span className="font-mono text-text-base">{trace.memory.record_count}</span>
                      </div>
                      {trace.memory.useful_patterns.length > 0 && (
                        <div>
                          <span className="text-xs text-text-muted">Matched patterns:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {trace.memory.useful_patterns.map((p, i) => (
                              <Chip key={i} color="purple">{p}</Chip>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="p-3 rounded-lg bg-success-light border border-success-DEFAULT/20">
                        <p className="text-xs text-success-DEFAULT">
                          Memory contributed to branch selection by biasing toward previously successful routes for similar task patterns.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-text-muted">No memory data available for this query</p>
                  )}
                </div>
              )}

              {/* Alternate Path */}
              {activeTab === 'alternate' && (
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-text-base">What If: Alternate Routes</h3>
                  {trace.routing ? (
                    <div className="space-y-3">
                      <p className="text-xs text-text-muted">
                        Comparing the chosen route against the next-best alternatives:
                      </p>
                      {Object.entries(trace.routing.branch_scores)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 5)
                        .map(([name, score], idx) => {
                          const isSelected = trace.activeBranches.includes(name);
                          return (
                            <div
                              key={name}
                              className={`p-3 rounded-lg border ${
                                isSelected ? 'border-primary/30 bg-primary-light' : 'border-border bg-[rgba(0,0,0,0.02)]'
                              }`}
                            >
                              <div className="flex items-center gap-2">
                                <span
                                  className="w-2 h-2 rounded-full"
                                  style={{ background: getBranchColor(name) }}
                                />
                                <span className={`text-xs font-medium flex-1 ${isSelected ? 'text-primary' : 'text-text-secondary'}`}>
                                  {formatBranchName(name)}
                                </span>
                                {isSelected && <Chip color="cyan" dot>Selected</Chip>}
                                <span className="text-xs font-mono text-text-muted">{score.toFixed(3)}</span>
                              </div>
                              {!isSelected && idx < 3 && (
                                <p className="text-[10px] text-text-muted mt-1 ml-4">
                                  Score gap: {(Object.values(trace.routing!.branch_scores).sort((a, b) => b - a)[0] - score).toFixed(3)} below top pick
                                </p>
                              )}
                            </div>
                          );
                        })}
                    </div>
                  ) : (
                    <p className="text-xs text-text-muted">No routing data available</p>
                  )}
                </div>
              )}
            </Card>
          )}
        </div>
      </div>
    </section>
  );
}

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Database, ArrowRight, Sparkles, ChevronRight } from 'lucide-react';
import { SectionHeading, Card, Chip, Bar } from '@/components/ui';
import { MOCK_MEMORY_ENTRIES, getBranchColor, formatBranchName } from '@/lib/mockData';
import { useAppStore } from '@/store/useAppStore';

export function MemoryPanel() {
  const [selectedMemory, setSelectedMemory] = useState<string | null>(null);
  const activeTrace = useAppStore((s) => s.activeTrace);

  const memories = MOCK_MEMORY_ENTRIES;

  return (
    <section id="memory" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Persistent Intelligence"
          title="Memory System"
          subtitle="The system remembers routing patterns, branch preferences, and task-specific biases across queries."
          center
        />

        <div className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Memory entries */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-text-base mb-3">Recent Memory Entries</h3>
            {memories.map((mem, idx) => (
              <motion.div
                key={mem.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.08 }}
              >
                <Card
                  hover
                  className="p-4"
                  onClick={() => setSelectedMemory(selectedMemory === mem.id ? null : mem.id)}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                      style={{ background: getBranchColor(mem.branch_preference) + '15' }}
                    >
                      <Database size={14} style={{ color: getBranchColor(mem.branch_preference) }} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-text-base truncate">{mem.pattern}</span>
                        <Chip color="gray">{mem.task_type}</Chip>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-[10px] text-text-muted">Prefers</span>
                        <ArrowRight size={10} className="text-text-muted" />
                        <span className="text-[10px] font-medium" style={{ color: getBranchColor(mem.branch_preference) }}>
                          {formatBranchName(mem.branch_preference)}
                        </span>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <Bar value={mem.bias_strength} color={getBranchColor(mem.branch_preference)} height={3} className="w-16" />
                      <span className="text-[10px] text-text-muted">{(mem.bias_strength * 100).toFixed(0)}% bias</span>
                    </div>
                    <ChevronRight size={14} className="text-text-muted" />
                  </div>

                  <AnimatePresence>
                    {selectedMemory === mem.id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-3 pt-3 border-t border-border space-y-2">
                          <div className="flex items-start gap-2">
                            <Sparkles size={12} className="text-accent-amber mt-0.5" />
                            <p className="text-xs text-text-secondary">{mem.feedback_effect}</p>
                          </div>
                          <div className="text-[10px] text-text-muted">
                            {new Date(mem.timestamp).toLocaleTimeString()} · Bias strength: {mem.bias_strength.toFixed(2)}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Why this route was chosen */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-text-base">Why This Route Was Chosen</h3>
            <Card className="p-5 space-y-4">
              {activeTrace.routing ? (
                <>
                  <div className="space-y-3">
                    <ReasonItem
                      label="Task Type Affinity"
                      description={`Task classified as "${activeTrace.routing.task_type}" — branches with matching affinity scored higher`}
                      strength={0.85}
                      color="#0066ff"
                    />
                    <ReasonItem
                      label="Historical Reward"
                      description="Branch weights reflect accumulated reward signals from past evaluations"
                      strength={0.72}
                      color="#10b981"
                    />
                    <ReasonItem
                      label="Memory Preference"
                      description="Similar past queries routed successfully through this branch combination"
                      strength={0.65}
                      color="#6366f1"
                    />
                    <ReasonItem
                      label="Exploration Bonus"
                      description={`Exploration rate: ${((useAppStore.getState().engineState?.exploration_rate ?? 0.087) * 100).toFixed(1)}% — occasionally tries less-used branches`}
                      strength={0.15}
                      color="#f59e0b"
                    />
                  </div>
                  <div className="pt-3 border-t border-border">
                    <span className="text-[10px] text-text-muted">Selected branches:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {activeTrace.activeBranches.map((b) => (
                        <Chip key={b} color="cyan" dot>{formatBranchName(b)}</Chip>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-8">
                  <Database size={24} className="mx-auto text-text-muted mb-2" />
                  <p className="text-sm text-text-muted">Run a query to see routing reasoning</p>
                </div>
              )}
            </Card>

            {/* Branch preferences learned */}
            <Card className="p-5">
              <h4 className="text-xs font-semibold text-text-base mb-3">Branch Preferences Learned</h4>
              <div className="space-y-2">
                {['math → deep_math + chain_of_thought', 'code → code_solver + verification', 'planning → planner + critique', 'factual → retrieval + analytical'].map((pref, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className="w-1 h-1 rounded-full bg-primary" />
                    <span className="text-text-secondary font-mono">{pref}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
}

function ReasonItem({ label, description, strength, color }: {
  label: string; description: string; strength: number; color: string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-text-base">{label}</span>
        <span className="text-[10px] font-mono" style={{ color }}>{(strength * 100).toFixed(0)}%</span>
      </div>
      <Bar value={strength} color={color} height={3} />
      <p className="text-[10px] text-text-muted">{description}</p>
    </div>
  );
}

import { motion } from 'framer-motion';
import { TrendingUp, Award, Gauge, BarChart3, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { SectionHeading, Card, Bar, Chip } from '@/components/ui';
import { useAppStore } from '@/store/useAppStore';
import { formatBranchName, getBranchColor, getRewardColor } from '@/lib/mockData';

export function RewardPanel() {
  const activeTrace = useAppStore((s) => s.activeTrace);
  const engineState = useAppStore((s) => s.engineState);

  const evaluation = activeTrace.evaluation;
  const optimization = activeTrace.optimization;

  return (
    <section id="rewards" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="RL Control Board"
          title="Reward & Optimization"
          subtitle="Watch the reinforcement learning loop in action — evaluator scores, reward deltas, weight updates, and branch lifecycle."
          center
        />

        <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Evaluator Score */}
          <Card className="p-5">
            <div className="flex items-center gap-2 mb-4">
              <Award size={16} className="text-primary" />
              <h3 className="text-sm font-semibold text-text-base">Evaluator Score</h3>
            </div>
            {evaluation ? (
              <div className="space-y-4">
                <div className="text-center">
                  <span
                    className="text-4xl font-bold font-mono"
                    style={{ color: getRewardColor(evaluation.reward_score) }}
                  >
                    {evaluation.reward_score.toFixed(3)}
                  </span>
                  <div className="text-xs text-text-muted mt-1">Composite Reward Score</div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-text-muted">Confidence</span>
                    <span className="font-mono text-text-base">{evaluation.confidence.toFixed(2)}</span>
                  </div>
                  <Bar value={evaluation.confidence} color="#0066ff" height={3} />
                </div>
                {evaluation.reward_components && (
                  <div className="space-y-1.5 pt-2 border-t border-border">
                    <span className="text-[10px] text-text-muted uppercase tracking-wider">Components</span>
                    {Object.entries(evaluation.reward_components).map(([key, val]) => val != null && (
                      <div key={key} className="flex justify-between text-xs">
                        <span className="text-text-secondary">{key.replace(/_/g, ' ')}</span>
                        <span className="font-mono" style={{ color: getRewardColor(val as number) }}>
                          {(val as number).toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <EmptyReward label="Run a query to see evaluator scores" />
            )}
          </Card>

          {/* Weight Changes */}
          <Card className="p-5">
            <div className="flex items-center gap-2 mb-4">
              <Gauge size={16} className="text-accent-purple" />
              <h3 className="text-sm font-semibold text-text-base">Branch Weight Updates</h3>
            </div>
            {optimization ? (
              <div className="space-y-3">
                {Object.entries(optimization.updated_weights)
                  .sort(([, a], [, b]) => (b as number) - (a as number))
                  .map(([name, weight]) => {
                    const detail = optimization.update_details[name] as Record<string, number> | undefined;
                    const delta = detail?.delta ?? 0;
                    const oldWeight = detail?.old_weight ?? (weight as number);
                    return (
                      <motion.div
                        key={name}
                        initial={{ opacity: 0, x: -5 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center gap-2"
                      >
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ background: getBranchColor(name) }}
                        />
                        <span className="text-xs text-text-secondary flex-1 truncate">
                          {formatBranchName(name)}
                        </span>
                        <span className="text-[10px] font-mono text-text-muted">
                          {oldWeight.toFixed(2)}
                        </span>
                        <span className="text-[10px] text-text-muted">→</span>
                        <span className="text-xs font-mono font-medium text-text-base">
                          {(weight as number).toFixed(2)}
                        </span>
                        {delta !== 0 && (
                          <span className={`text-[10px] font-mono flex items-center gap-0.5 ${delta > 0 ? 'text-success-DEFAULT' : 'text-danger-DEFAULT'}`}>
                            {delta > 0 ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                            {Math.abs(delta).toFixed(3)}
                          </span>
                        )}
                      </motion.div>
                    );
                  })}
              </div>
            ) : (
              <EmptyReward label="Weight updates appear after optimization" />
            )}
          </Card>

          {/* Optimizer Events */}
          <Card className="p-5">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp size={16} className="text-accent-green" />
              <h3 className="text-sm font-semibold text-text-base">Optimizer Event Log</h3>
            </div>
            {optimization ? (
              <div className="space-y-3">
                {optimization.rewritten_prompts.length > 0 && (
                  <EventItem
                    label="Prompt Rewrites"
                    items={optimization.rewritten_prompts.map(formatBranchName)}
                    color="amber"
                    icon="rewrite"
                  />
                )}
                {optimization.promoted_candidates.length > 0 && (
                  <EventItem
                    label="Promoted to Active"
                    items={optimization.promoted_candidates.map(formatBranchName)}
                    color="green"
                    icon="promote"
                  />
                )}
                {optimization.archived_candidates.length > 0 && (
                  <EventItem
                    label="Archived"
                    items={optimization.archived_candidates.map(formatBranchName)}
                    color="red"
                    icon="archive"
                  />
                )}
                {optimization.created_candidates.length > 0 && (
                  <EventItem
                    label="New Candidates"
                    items={optimization.created_candidates.map(formatBranchName)}
                    color="cyan"
                    icon="create"
                  />
                )}
                <div className="flex items-center gap-2 text-xs pt-2 border-t border-border">
                  <span className="text-text-muted">LLM Advisor:</span>
                  <Chip color={optimization.advisor_used ? 'green' : 'gray'} dot>
                    {optimization.advisor_used ? 'Active' : 'Skipped'}
                  </Chip>
                </div>

                {/* Branch ranking */}
                <div className="pt-2 border-t border-border">
                  <span className="text-[10px] text-text-muted uppercase tracking-wider">Branch Ranking</span>
                  <div className="mt-2 space-y-1">
                    {Object.entries(optimization.updated_weights)
                      .sort(([, a], [, b]) => (b as number) - (a as number))
                      .slice(0, 5)
                      .map(([name, weight], i) => (
                        <div key={name} className="flex items-center gap-2 text-[10px]">
                          <span className="text-text-muted w-4">#{i + 1}</span>
                          <span className="text-text-secondary flex-1 truncate">{formatBranchName(name)}</span>
                          <span className="font-mono text-text-base">{(weight as number).toFixed(2)}</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            ) : (
              <EmptyReward label="Optimizer events appear after a full run" />
            )}
          </Card>
        </div>
      </div>
    </section>
  );
}

function EventItem({ label, items, color, icon }: {
  label: string; items: string[]; color: string; icon: string;
}) {
  return (
    <div className="space-y-1">
      <span className="text-xs font-medium text-text-base">{label}</span>
      <div className="flex flex-wrap gap-1">
        {items.map((item, i) => (
          <Chip key={i} color={color}>{item}</Chip>
        ))}
      </div>
    </div>
  );
}

function EmptyReward({ label }: { label: string }) {
  return (
    <div className="text-center py-6">
      <BarChart3 size={20} className="mx-auto text-text-muted mb-2" />
      <p className="text-xs text-text-muted">{label}</p>
    </div>
  );
}

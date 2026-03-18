import { motion } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { MOCK_ENGINE_STATE, formatBranchName } from '@/lib/mockData';
import { Bar } from '@/components/ui';
import { clsx } from 'clsx';

export function LiveMetrics() {
  const { activeTrace, engineState } = useAppStore();
  const state = engineState ?? MOCK_ENGINE_STATE;
  const { evaluation, optimization, timings } = activeTrace;

  return (
    <div className="px-4 pb-4 space-y-4">

      {/* Reward */}
      {evaluation && (
        <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}>
          <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted mb-2">Reward</div>
          <div className="space-y-2">
            <Row
              label="Adaptive score"
              value={`${(evaluation.reward_score * 100).toFixed(1)}%`}
              bar={evaluation.reward_score}
              barColor={evaluation.reward_score >= 0.75 ? '#059669' : evaluation.reward_score >= 0.5 ? '#D97706' : '#DC2626'}
            />
            <Row
              label="Confidence"
              value={`${(evaluation.confidence * 100).toFixed(0)}%`}
              bar={evaluation.confidence}
              barColor="#2563EB"
            />
            {evaluation.reward_components && Object.entries(evaluation.reward_components).map(([k, v]) => {
              if (v == null) return null;
              const labels: Record<string,string> = { verifier:'Verifier', task_rules:'Task rules', llm_judge:'LLM judge', user_feedback:'User feedback' };
              return (
                <Row key={k} label={labels[k] ?? k} value={`${((v as number)*100).toFixed(0)}%`} bar={v as number} barColor="#9CA3AF" />
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Optimizer */}
      {optimization && Object.keys(optimization.updated_weights).length > 0 && (
        <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted mb-2">Weight Δ</div>
          <div className="space-y-2">
            {Object.entries(optimization.updated_weights).slice(0, 5).map(([name, newW]) => {
              const det   = optimization.update_details?.[name];
              const oldW  = (det?.old_weight as number) ?? newW;
              const delta = newW - oldW;
              return (
                <div key={name} className="flex items-center gap-2">
                  <span className="text-xs text-text-secondary w-28 truncate shrink-0">{formatBranchName(name)}</span>
                  <div className="flex-1">
                    <Bar value={newW / 3} color="#2563EB" height={4} />
                  </div>
                  <span className={clsx('text-[11px] font-mono w-12 text-right shrink-0',
                    delta > 0 ? 'text-success' : delta < 0 ? 'text-danger' : 'text-text-muted'
                  )}>
                    {delta > 0 ? '+' : ''}{delta.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Timing */}
      {timings && (
        <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
          <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted mb-2">Timing</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
            {[
              ['Route',    timings.route_ms],
              ['Execute',  timings.execute_ms],
              ['Evaluate', timings.evaluate_ms],
              ['Optimize', timings.optimize_ms],
              ['Total',    timings.total_ms],
            ].map(([label, ms]) => (
              <div key={label as string} className="flex items-center justify-between">
                <span className="text-xs text-text-muted">{label}</span>
                <span className="text-xs font-mono text-text-secondary">{ms}ms</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Engine stats — always shown */}
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted mb-2">Engine</div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
          {[
            ['Memory records', state.memory_count],
            ['Exploration', `${(state.exploration_rate * 100).toFixed(1)}%`],
            ['Avg reward', `${(state.avg_reward * 100).toFixed(1)}%`],
            ['Total tasks', state.total_tasks],
          ].map(([label, val]) => (
            <div key={label as string} className="flex items-center justify-between">
              <span className="text-xs text-text-muted">{label}</span>
              <span className="text-xs font-mono text-text-secondary font-medium">{val}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Optimizer events */}
      {optimization && (
        <OptimizerEvents opt={optimization} />
      )}
    </div>
  );
}

function Row({ label, value, bar, barColor }: { label: string; value: string; bar: number; barColor: string }) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-text-muted">{label}</span>
        <span className="text-xs font-mono font-medium text-text-secondary">{value}</span>
      </div>
      <Bar value={bar} color={barColor} height={3} />
    </div>
  );
}

function OptimizerEvents({ opt }: { opt: ReturnType<typeof useAppStore.getState>['activeTrace']['optimization'] }) {
  if (!opt) return null;
  const events: { label: string; color: string }[] = [];
  if (opt.rewritten_prompts.length)    events.push({ label: `${opt.rewritten_prompts.length} prompt rewrite${opt.rewritten_prompts.length > 1 ? 's' : ''}`, color: '#D97706' });
  if (opt.promoted_candidates.length)  events.push({ label: `Promoted: ${opt.promoted_candidates.join(', ')}`, color: '#059669' });
  if (opt.archived_candidates.length)  events.push({ label: `Archived: ${opt.archived_candidates.join(', ')}`, color: '#6B7280' });
  if (opt.created_candidates.length)   events.push({ label: `New branch: ${opt.created_candidates.join(', ')}`, color: '#7C3AED' });
  if (opt.advisor_used)                events.push({ label: 'LLM advisor applied', color: '#2563EB' });
  if (events.length === 0) return null;

  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted mb-2">Optimizer Events</div>
      <div className="space-y-1.5">
        {events.map((e, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: e.color }} />
            <span className="text-xs text-text-secondary">{e.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

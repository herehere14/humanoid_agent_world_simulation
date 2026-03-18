import { motion } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { MOCK_ENGINE_STATE, getBranchColor, formatBranchName } from '@/lib/mockData';
import { Bar } from '@/components/ui';
import { clsx } from 'clsx';

export function BranchMatrix() {
  const { activeTrace, engineState } = useAppStore();
  const state = engineState ?? MOCK_ENGINE_STATE;
  const { activeBranches, selectedPath, branchOutputs, routing } = activeTrace;

  // Merge engine branches with live trace data
  const branches = state.branches.slice().sort((a, b) => b.weight - a.weight);
  const maxWeight = Math.max(...branches.map(b => b.weight), 3);

  return (
    <div className="px-4 pb-4">
      {/* Column headers */}
      <div className="grid grid-cols-[1fr_52px_52px_52px] gap-x-3 px-1 mb-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-text-muted">Branch</span>
        <span className="text-[10px] font-semibold uppercase tracking-widest text-text-muted text-right">Weight</span>
        <span className="text-[10px] font-semibold uppercase tracking-widest text-text-muted text-right">Score</span>
        <span className="text-[10px] font-semibold uppercase tracking-widest text-text-muted text-right">Status</span>
      </div>

      <div className="space-y-0.5">
        {branches.map((branch, i) => {
          const isActive   = activeBranches.includes(branch.name);
          const isSelected = selectedPath.includes(branch.name);
          const reward     = branchOutputs[branch.name]?.reward
                            ?? routing?.branch_scores?.[branch.name]
                            ?? (branch.historical_rewards.at(-1) ?? 0);
          const color      = getBranchColor(branch.name);
          const isCandidate = branch.status === 'CANDIDATE';
          const isRunning  = branchOutputs[branch.name]?.status === 'running';

          return (
            <motion.div
              key={branch.name}
              initial={i === 0 ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              className={clsx(
                'grid grid-cols-[1fr_52px_52px_52px] gap-x-3 items-center px-2 py-2 rounded-xl transition-all duration-150',
                isSelected
                  ? 'bg-primary-light border border-primary-mid'
                  : isActive
                  ? 'bg-elevated'
                  : 'hover:bg-elevated/60'
              )}
            >
              {/* Name + bar */}
              <div className="flex flex-col gap-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <div
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{
                      backgroundColor: isActive ? color : '#D1D5DB',
                      boxShadow: isSelected ? `0 0 5px ${color}80` : undefined,
                    }}
                  />
                  <span className={clsx(
                    'text-xs truncate',
                    isSelected ? 'font-semibold text-primary' : isActive ? 'font-medium text-text-base' : 'text-text-secondary'
                  )}>
                    {formatBranchName(branch.name)}
                  </span>
                  {isRunning && (
                    <span className="w-1 h-1 rounded-full bg-primary animate-pulse shrink-0" />
                  )}
                  {isCandidate && (
                    <span className="text-[8px] font-bold text-warn border border-warn-light bg-warn-light px-1 rounded uppercase tracking-wide shrink-0">
                      new
                    </span>
                  )}
                </div>
                {/* Weight bar */}
                <Bar
                  value={branch.weight / maxWeight}
                  color={isActive ? color : '#E5E7EB'}
                  height={3}
                />
              </div>

              {/* Weight */}
              <span className={clsx(
                'text-xs font-mono text-right',
                isActive ? 'text-text-base font-medium' : 'text-text-muted'
              )}>
                {branch.weight.toFixed(2)}
              </span>

              {/* Score */}
              <span className={clsx(
                'text-xs font-mono text-right',
                reward >= 0.75 ? 'text-success' : reward >= 0.5 ? 'text-warn' : 'text-text-muted'
              )}>
                {(reward * 100).toFixed(0)}%
              </span>

              {/* Status dot */}
              <div className="flex justify-end">
                <span className={clsx(
                  'inline-block w-2 h-2 rounded-full',
                  isSelected ? 'bg-primary' :
                  isRunning  ? 'bg-warn animate-pulse' :
                  isActive   ? 'bg-success' :
                  isCandidate? 'bg-warn' :
                  'bg-border-strong'
                )} />
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Selected path breadcrumb */}
      {selectedPath.length > 0 && (
        <div className="mt-3 flex items-center gap-1.5 flex-wrap px-1">
          <span className="text-[10px] text-text-muted font-medium uppercase tracking-wider">Path</span>
          {selectedPath.map((step, i) => (
            <span key={step} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-text-muted text-[10px]">›</span>}
              <span className="text-[11px] font-medium text-primary bg-primary-light px-2 py-0.5 rounded-full">
                {formatBranchName(step)}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

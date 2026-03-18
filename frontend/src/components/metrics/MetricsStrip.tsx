import { motion } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { MOCK_ENGINE_STATE } from '@/lib/mockData';

interface MetricItem {
  label: string;
  value: string | number;
  color: string;
  suffix?: string;
}

export function MetricsStrip() {
  const engineState = useAppStore((s) => s.engineState) ?? MOCK_ENGINE_STATE;
  const activeTrace = useAppStore((s) => s.activeTrace);
  const performanceHistory = useAppStore((s) => s.performanceHistory);

  const activeBranches = engineState.branches.filter((b) => b.status === 'ACTIVE').length;
  const winRate = performanceHistory.length > 0
    ? performanceHistory.filter((p) => p.delta > 0).length / performanceHistory.length
    : 0.82;
  const avgReward = activeTrace.evaluation?.reward_score ?? engineState.avg_reward;
  const memoryHits = activeTrace.memory?.record_count ?? engineState.memory_count;
  const latency = activeTrace.timings?.total_ms ?? 0;
  const adaptationScore = Math.min(1, avgReward * 1.1);
  const tokenUsage = activeTrace.timings?.primary_backend_calls ?? 0;
  const routeDepth = activeTrace.selectedPath.length || 2;

  const metrics: MetricItem[] = [
    { label: 'Active Branches', value: activeBranches, color: '#0066ff' },
    { label: 'Route Depth', value: routeDepth, color: '#7c3aed' },
    { label: 'Avg Reward', value: avgReward.toFixed(3), color: '#059669' },
    { label: 'Memory Hits', value: memoryHits, color: '#4f46e5' },
    { label: 'Token Usage', value: tokenUsage || '—', color: '#0891b2' },
    { label: 'Latency', value: latency ? `${latency.toFixed(0)}ms` : '—', color: '#d97706' },
    { label: 'Adaptation', value: `${(adaptationScore * 100).toFixed(0)}%`, color: '#ec4899' },
    { label: 'Shift Resilience', value: `${(winRate * 100).toFixed(0)}%`, color: '#2563eb' },
    { label: 'Win Rate', value: `${(winRate * 100).toFixed(0)}%`, color: '#059669' },
  ];

  return (
    <div className="w-full overflow-hidden border-y border-border bg-surface/50 py-3">
      <div className="flex animate-[scroll_30s_linear_infinite] hover:[animation-play-state:paused]" style={{ width: 'max-content' }}>
        {[...metrics, ...metrics].map((m, i) => (
          <motion.div
            key={i}
            className="flex items-center gap-3 px-6 border-r border-border last:border-r-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: (i % metrics.length) * 0.05 }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: m.color }} />
            <span className="text-[10px] uppercase tracking-wider text-text-muted whitespace-nowrap">{m.label}</span>
            <span className="text-sm font-mono font-semibold whitespace-nowrap" style={{ color: m.color }}>
              {m.value}{m.suffix ?? ''}
            </span>
          </motion.div>
        ))}
      </div>
      <style>{`
        @keyframes scroll {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
      `}</style>
    </div>
  );
}

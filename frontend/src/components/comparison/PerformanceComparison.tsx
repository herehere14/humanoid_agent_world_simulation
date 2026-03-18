import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, ChevronUp, ChevronDown, Award } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { PerformanceTrend } from '@/components/charts/PerformanceTrend';
import { Bar } from '@/components/ui';
import { MOCK_PERFORMANCE_HISTORY } from '@/lib/mockData';
import { clsx } from 'clsx';

export function PerformanceComparison() {
  const { performanceHistory, messages, isStreaming } = useAppStore();
  const [collapsed, setCollapsed] = useState(false);

  const history = performanceHistory.length > 0 ? performanceHistory : MOCK_PERFORMANCE_HISTORY;

  const lastMsg  = [...messages].reverse().find(m => m.role === 'assistant' && m.status === 'complete');
  const latestDelta = lastMsg?.improvementDelta;
  const latestScore = lastMsg?.trace?.evaluation?.reward_score;
  const baseScore   = latestScore !== undefined && latestDelta !== undefined
    ? latestScore - latestDelta : undefined;

  const avgAdaptive = history.reduce((s,p) => s + p.adaptiveScore, 0) / history.length;
  const avgBase     = history.reduce((s,p) => s + p.baseScore,     0) / history.length;
  const avgDelta    = avgAdaptive - avgBase;
  const winRate     = history.filter(p => p.adaptiveScore > p.baseScore).length / history.length;

  return (
    <div className={clsx(
      'border-t border-border bg-surface transition-all duration-200 shrink-0',
      collapsed ? 'h-11' : 'h-[180px]'
    )}>
      {/* Header */}
      <div className="h-11 flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-1.5 text-sm font-medium text-text-base">
            <TrendingUp size={14} className="text-success" />
            Performance
          </div>
          {!collapsed && (
            <div className="hidden sm:flex items-center gap-4 text-xs text-text-secondary">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-primary" />
                Adaptive <strong className="text-text-base ml-1">{(avgAdaptive*100).toFixed(1)}%</strong>
              </span>
              <span className="text-border">·</span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-border-strong" />
                Base <strong className="text-text-base ml-1">{(avgBase*100).toFixed(1)}%</strong>
              </span>
              <span className={clsx(
                'font-medium',
                avgDelta > 0 ? 'text-success' : 'text-danger'
              )}>
                {avgDelta > 0 ? '+' : ''}{(avgDelta*100).toFixed(1)}% avg
              </span>
              <span className="flex items-center gap-1 text-text-secondary">
                <Award size={11} />
                {(winRate*100).toFixed(0)}% win rate
              </span>
              {isStreaming && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-1 text-primary"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                  Live
                </motion.span>
              )}
            </div>
          )}
        </div>
        <button onClick={() => setCollapsed(c => !c)} className="p-1 rounded-lg hover:bg-elevated text-text-muted transition-colors">
          {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {/* Body */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex gap-0 h-[calc(180px-44px)]"
          >
            {/* Chart */}
            <div className="flex-1 border-r border-border px-3 py-2">
              <PerformanceTrend data={history} />
            </div>

            {/* Cards */}
            <div className="w-72 shrink-0 flex flex-col gap-2 p-3">
              {latestScore !== undefined && baseScore !== undefined ? (
                <LatestCards adaptive={latestScore} base={baseScore} delta={latestDelta ?? 0} winLabel={lastMsg?.winLabel} />
              ) : (
                <AggregateCards adaptive={avgAdaptive} base={avgBase} delta={avgDelta} winRate={winRate} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function LatestCards({ adaptive, base, delta, winLabel }: { adaptive: number; base: number; delta: number; winLabel?: string }) {
  const won = adaptive > base;
  return (
    <>
      <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted">Latest Round</div>
      <div className="grid grid-cols-2 gap-2">
        <ScoreCard label="Adaptive" score={adaptive} color="#2563EB" highlight={won} />
        <ScoreCard label="Base"     score={base}     color="#9CA3AF" />
      </div>
      <div className={clsx(
        'flex items-center justify-center gap-1.5 py-1.5 rounded-xl text-xs font-medium border',
        won
          ? 'bg-success-light border-success-mid text-success'
          : 'bg-danger-light border-red-200 text-danger'
      )}>
        {won ? <TrendingUp size={11} /> : <ChevronDown size={11} />}
        {won ? 'Adaptive beat base by ' : 'Base ahead by '}
        <strong>{won?'+':''}{(Math.abs(delta)*100).toFixed(1)}%</strong>
      </div>
      {winLabel && <p className="text-[10px] text-text-muted text-center">{winLabel}</p>}
    </>
  );
}

function AggregateCards({ adaptive, base, delta, winRate }: { adaptive:number; base:number; delta:number; winRate:number }) {
  return (
    <>
      <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted">All-time Average</div>
      <div className="grid grid-cols-2 gap-2">
        <ScoreCard label="Adaptive" score={adaptive} color="#2563EB" highlight />
        <ScoreCard label="Base"     score={base}     color="#9CA3AF" />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <MiniStat label="Improvement" value={`+${(delta*100).toFixed(1)}%`} color="text-success" />
        <MiniStat label="Win Rate"    value={`${(winRate*100).toFixed(0)}%`} color="text-primary" />
      </div>
    </>
  );
}

function ScoreCard({ label, score, color, highlight }: { label:string; score:number; color:string; highlight?:boolean }) {
  return (
    <div className={clsx(
      'border rounded-xl p-2.5',
      highlight ? 'border-primary-mid bg-primary-light' : 'border-border bg-elevated'
    )}>
      <div className="text-[9px] font-semibold uppercase tracking-wider text-text-muted mb-1">{label}</div>
      <div className="text-base font-semibold mb-1.5" style={{ color }}>{(score*100).toFixed(1)}%</div>
      <Bar value={score} color={color} height={3} />
    </div>
  );
}

function MiniStat({ label, value, color }: { label:string; value:string; color:string }) {
  return (
    <div className="border border-border rounded-xl p-2.5 bg-elevated">
      <div className="text-[9px] font-semibold uppercase tracking-wider text-text-muted mb-1">{label}</div>
      <div className={clsx('text-sm font-semibold', color)}>{value}</div>
    </div>
  );
}

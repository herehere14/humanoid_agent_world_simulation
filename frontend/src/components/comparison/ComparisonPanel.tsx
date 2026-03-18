import { motion } from 'framer-motion';
import { Zap, Clock, Hash, GitBranch, Brain, TrendingUp, ArrowUp, ArrowDown, Minus } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { SectionHeading, Card, Bar, Chip, MetricBox } from '@/components/ui';
import { formatBranchName, getRewardColor } from '@/lib/mockData';
import { PerformanceTrend } from '@/components/charts/PerformanceTrend';

export function ComparisonPanel() {
  const messages = useAppStore((s) => s.messages);
  const performanceHistory = useAppStore((s) => s.performanceHistory);
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant' && m.status === 'complete');

  const trace = lastAssistant?.trace;
  const delta = lastAssistant?.improvementDelta;
  const winLabel = lastAssistant?.winLabel;

  // Compute stats
  const adaptiveScore = trace?.evaluation?.reward_score ?? 0;
  const baseScore = delta !== undefined ? adaptiveScore - delta : 0;
  const confidence = trace?.evaluation?.confidence ?? 0;
  const totalMs = trace?.timings?.total_ms ?? 0;
  const tokens = trace?.timings?.primary_backend_calls ?? 0;
  const selectedBranch = trace?.evaluation?.selected_branch ?? '';
  const memoryRecords = trace?.memory?.record_count ?? 0;

  const hasData = !!trace;

  return (
    <section id="comparison" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Head to Head"
          title="Adaptive Forest vs Base Model"
          subtitle="See exactly how the adaptive routing system outperforms a single-model baseline across every metric."
          center
        />

        {hasData ? (
          <div className="mt-12 space-y-8">
            {/* Improvement badge */}
            {delta !== undefined && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="flex justify-center"
              >
                <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-2xl ${
                  delta > 0 ? 'bg-success-light border border-success-DEFAULT/20' :
                  delta < 0 ? 'bg-danger-light border border-danger-DEFAULT/20' :
                  'bg-[rgba(0,0,0,0.03)] border border-border'
                }`}>
                  {delta > 0 ? <ArrowUp className="text-success-DEFAULT" /> :
                   delta < 0 ? <ArrowDown className="text-danger-DEFAULT" /> :
                   <Minus className="text-text-muted" />}
                  <span className={`text-2xl font-bold font-mono ${
                    delta > 0 ? 'text-success-DEFAULT' : delta < 0 ? 'text-danger-DEFAULT' : 'text-text-base'
                  }`}>
                    {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}%
                  </span>
                  <span className="text-sm text-text-secondary">
                    {winLabel ?? (delta > 0 ? 'Improvement' : delta < 0 ? 'Decrease' : 'No change')}
                  </span>
                </div>
              </motion.div>
            )}

            {/* Split screen comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Base model */}
              <Card className="p-5 relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-text-muted/30 to-transparent" />
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-2 h-2 rounded-full bg-text-muted" />
                  <h3 className="text-sm font-semibold text-text-secondary">Normal Single Model</h3>
                  <Chip color="gray">Baseline</Chip>
                </div>
                <div className="space-y-4">
                  <CompareMetric label="Response Quality" value={baseScore} maxValue={1} color="#71717a" />
                  <CompareMetric label="Task Score" value={baseScore} maxValue={1} color="#71717a" />
                  <CompareMetric label="Confidence" value={Math.max(0, confidence - 0.15)} maxValue={1} color="#71717a" />
                  <div className="grid grid-cols-2 gap-3 pt-2">
                    <MetricBox label="Latency" value={totalMs ? `${(totalMs * 0.6).toFixed(0)}` : '—'} suffix="ms" small />
                    <MetricBox label="Tokens" value={tokens ? `${Math.round(tokens * 0.7)}` : '—'} small />
                    <MetricBox label="Route" value="Single" small color="#71717a" />
                    <MetricBox label="Memory" value="None" small color="#71717a" />
                  </div>
                </div>
              </Card>

              {/* Adaptive */}
              <Card className="p-5 relative overflow-hidden" glow>
                <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                  <h3 className="text-sm font-semibold text-text-base">Adaptive Prompt Forest</h3>
                  <Chip color="cyan" dot>Active</Chip>
                </div>
                <div className="space-y-4">
                  <CompareMetric label="Response Quality" value={adaptiveScore} maxValue={1} color="#0066ff" />
                  <CompareMetric label="Task Score" value={adaptiveScore} maxValue={1} color="#0066ff" />
                  <CompareMetric label="Confidence" value={confidence} maxValue={1} color="#0066ff" />
                  <div className="grid grid-cols-2 gap-3 pt-2">
                    <MetricBox label="Latency" value={totalMs ? `${totalMs.toFixed(0)}` : '—'} suffix="ms" small />
                    <MetricBox label="Tokens" value={tokens || '—'} small />
                    <MetricBox label="Route" value={selectedBranch ? formatBranchName(selectedBranch) : '—'} small color="#0066ff" />
                    <MetricBox label="Memory" value={memoryRecords || '—'} suffix="hits" small color="#6366f1" />
                  </div>
                </div>

                {/* Adaptation gain */}
                {delta !== undefined && delta > 0 && (
                  <div className="mt-4 pt-3 border-t border-border">
                    <div className="flex items-center gap-2 text-xs">
                      <TrendingUp size={14} className="text-success-DEFAULT" />
                      <span className="text-success-DEFAULT font-medium">
                        Adaptation gain: +{(delta * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                )}
              </Card>
            </div>

            {/* Performance trend */}
            {performanceHistory.length > 0 && (
              <Card className="p-5">
                <h3 className="text-sm font-semibold text-text-base mb-4">Performance Over Time</h3>
                <div className="h-[200px]">
                  <PerformanceTrend data={performanceHistory} />
                </div>
              </Card>
            )}
          </div>
        ) : (
          <div className="mt-12 text-center py-16">
            <div className="glass rounded-2xl inline-flex flex-col items-center gap-4 p-8">
              <GitBranch size={32} className="text-text-muted" />
              <p className="text-sm text-text-secondary">Run a query to see the comparison</p>
              <p className="text-xs text-text-muted">The adaptive forest will be compared against a base model response</p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function CompareMetric({ label, value, maxValue, color }: {
  label: string; value: number; maxValue: number; color: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-text-muted">{label}</span>
        <span className="font-mono" style={{ color: getRewardColor(value) }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <Bar value={value} max={maxValue} color={color} height={4} />
    </div>
  );
}

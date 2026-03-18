import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Shuffle, Play, RotateCcw, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { SectionHeading, Card, GlowButton, Chip, Bar } from '@/components/ui';
import { MOCK_BENCHMARK_DATA, getBranchColor, formatBranchName, MOCK_BRANCHES } from '@/lib/mockData';
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis, ReferenceLine, Tooltip, CartesianGrid } from 'recharts';

type ShiftPhase = 'pre' | 'shifting' | 'adapting' | 'recovered';

export function DomainShiftChallenge() {
  const [shiftActive, setShiftActive] = useState(false);
  const [phase, setPhase] = useState<ShiftPhase>('pre');
  const [step, setStep] = useState(0);
  const [branchWeights, setBranchWeights] = useState<Record<string, number>>(
    Object.fromEntries(MOCK_BRANCHES.map((b) => [b.name, b.weight]))
  );

  const data = MOCK_BENCHMARK_DATA.shiftComparison;

  const startShift = useCallback(() => {
    setShiftActive(true);
    setPhase('pre');
    setStep(0);

    // Animate through steps
    let s = 0;
    const interval = setInterval(() => {
      s++;
      setStep(s);
      if (s < 10) setPhase('pre');
      else if (s < 13) {
        setPhase('shifting');
        // Shuffle weights during shift
        setBranchWeights((prev) => {
          const next = { ...prev };
          Object.keys(next).forEach((k) => {
            next[k] = Math.max(0.3, next[k] + (Math.random() - 0.6) * 0.3);
          });
          return next;
        });
      } else if (s < 17) {
        setPhase('adapting');
        // Weights start recovering
        setBranchWeights((prev) => {
          const next = { ...prev };
          const orig = Object.fromEntries(MOCK_BRANCHES.map((b) => [b.name, b.weight]));
          Object.keys(next).forEach((k) => {
            next[k] = next[k] + (orig[k] - next[k]) * 0.3 + (Math.random() - 0.5) * 0.1;
          });
          return next;
        });
      } else {
        setPhase('recovered');
        setBranchWeights(Object.fromEntries(MOCK_BRANCHES.map((b) => [b.name, b.weight * (1 + Math.random() * 0.1)])));
      }

      if (s >= data.length - 1) {
        clearInterval(interval);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [data.length]);

  const reset = () => {
    setShiftActive(false);
    setPhase('pre');
    setStep(0);
    setBranchWeights(Object.fromEntries(MOCK_BRANCHES.map((b) => [b.name, b.weight])));
  };

  const phaseColors: Record<ShiftPhase, string> = {
    pre: '#10b981',
    shifting: '#ef4444',
    adapting: '#f59e0b',
    recovered: '#0066ff',
  };

  const phaseLabels: Record<ShiftPhase, string> = {
    pre: 'Stable Distribution',
    shifting: 'Distribution Shift Detected',
    adapting: 'Re-adapting...',
    recovered: 'Adapted to New Distribution',
  };

  return (
    <section id="shift" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Strongest Differentiator"
          title="Domain Shift Challenge"
          subtitle="Watch the adaptive system detect, survive, and recover from distribution shift — while the baseline collapses."
          center
        />

        <div className="mt-12 space-y-6">
          {/* Controls */}
          <div className="flex justify-center gap-3">
            {!shiftActive ? (
              <GlowButton onClick={startShift} size="lg">
                <Play size={16} /> Start Domain Shift Challenge
              </GlowButton>
            ) : (
              <GlowButton onClick={reset} variant="secondary">
                <RotateCcw size={16} /> Reset
              </GlowButton>
            )}
          </div>

          {/* Phase indicator */}
          {shiftActive && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-center"
            >
              <Chip
                color={phase === 'pre' ? 'green' : phase === 'shifting' ? 'red' : phase === 'adapting' ? 'amber' : 'cyan'}
                dot
              >
                {phase === 'shifting' && <AlertTriangle size={10} />}
                {phaseLabels[phase]}
              </Chip>
            </motion.div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Chart */}
            <Card className="p-5 lg:col-span-2">
              <h3 className="text-sm font-semibold text-text-base mb-4">Adaptive vs Baseline Under Shift</h3>
              <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data.slice(0, shiftActive ? step + 1 : data.length)}>
                    <defs>
                      <linearGradient id="gradAdaptShift" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#0066ff" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#0066ff" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="gradBaseShift" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#71717a" stopOpacity={0.15} />
                        <stop offset="100%" stopColor="#71717a" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
                    <XAxis dataKey="task" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0.3, 1]} tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <ReferenceLine x={10} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: 'SHIFT', fill: '#ef4444', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: '#ffffff', border: '1px solid rgba(0,0,0,0.06)', borderRadius: 8, fontSize: 11 }}
                      labelStyle={{ color: '#71717a' }}
                    />
                    <Area type="monotone" dataKey="adaptive" stroke="#0066ff" fill="url(#gradAdaptShift)" strokeWidth={2} name="Adaptive" />
                    <Area type="monotone" dataKey="baseline" stroke="#71717a" fill="url(#gradBaseShift)" strokeWidth={1.5} strokeDasharray="4 4" name="Baseline" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Branch weight changes */}
            <Card className="p-5">
              <h3 className="text-sm font-semibold text-text-base mb-4">Branch Weight Reshuffling</h3>
              <div className="space-y-3">
                {Object.entries(branchWeights)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 8)
                  .map(([name, weight]) => {
                    const original = MOCK_BRANCHES.find((b) => b.name === name)?.weight ?? 1;
                    const delta = weight - original;
                    return (
                      <motion.div
                        key={name}
                        layout
                        className="space-y-1"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-text-secondary">{formatBranchName(name)}</span>
                          <div className="flex items-center gap-1">
                            <span className="text-[10px] font-mono text-text-base">{weight.toFixed(2)}</span>
                            {Math.abs(delta) > 0.05 && (
                              <span className={`text-[9px] font-mono flex items-center ${delta > 0 ? 'text-success-DEFAULT' : 'text-danger-DEFAULT'}`}>
                                {delta > 0 ? <TrendingUp size={8} /> : <TrendingDown size={8} />}
                              </span>
                            )}
                          </div>
                        </div>
                        <Bar
                          value={weight}
                          max={3}
                          color={phase === 'shifting' ? '#ef4444' : getBranchColor(name)}
                          height={3}
                        />
                      </motion.div>
                    );
                  })}
              </div>

              {phase === 'recovered' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4 p-3 rounded-lg bg-success-light border border-success-DEFAULT/20"
                >
                  <div className="flex items-center gap-2 text-xs text-success-DEFAULT">
                    <TrendingUp size={14} />
                    <span className="font-medium">System adapted to new distribution</span>
                  </div>
                </motion.div>
              )}
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
}

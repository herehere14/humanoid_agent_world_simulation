import { useState } from 'react';
import { motion } from 'framer-motion';
import { SectionHeading, Card } from '@/components/ui';
import { MOCK_BENCHMARK_DATA } from '@/lib/mockData';
import {
  Area, AreaChart, Line, LineChart, ScatterChart, Scatter,
  ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, ZAxis
} from 'recharts';

type ChartKey = 'reward' | 'winRate' | 'entropy' | 'latency' | 'shift';

const chartTabs: { key: ChartKey; label: string }[] = [
  { key: 'reward', label: 'Reward / Episodes' },
  { key: 'winRate', label: 'Win Rate' },
  { key: 'entropy', label: 'Branch Entropy' },
  { key: 'latency', label: 'Latency vs Quality' },
  { key: 'shift', label: 'Pre vs Post Shift' },
];

export function BenchmarkSection() {
  const [activeChart, setActiveChart] = useState<ChartKey>('reward');

  const tooltipStyle = {
    contentStyle: { background: '#ffffff', border: '1px solid rgba(0,0,0,0.06)', borderRadius: 8, fontSize: 11 },
    labelStyle: { color: '#71717a' },
  };

  return (
    <section id="benchmarks" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Research-Backed"
          title="Benchmark Results"
          subtitle="Quantitative evidence that the adaptive prompt forest consistently outperforms static baselines."
          center
        />

        <div className="mt-12">
          {/* Chart tabs */}
          <div className="flex flex-wrap justify-center gap-2 mb-6">
            {chartTabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveChart(tab.key)}
                className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
                  activeChart === tab.key
                    ? 'bg-primary-light text-primary border border-primary/20'
                    : 'text-text-muted hover:text-text-secondary hover:bg-[rgba(0,0,0,0.03)]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <Card className="p-6">
            <div className="h-[320px]">
              {activeChart === 'reward' && (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={MOCK_BENCHMARK_DATA.rewardOverEpisodes}>
                    <defs>
                      <linearGradient id="gradReward" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#0066ff" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#0066ff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
                    <XAxis dataKey="episode" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} label={{ value: 'Episode', fill: '#71717a', fontSize: 10, position: 'insideBottom', offset: -5 }} />
                    <YAxis domain={[0.3, 1]} tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip {...tooltipStyle} />
                    <ReferenceLine y={0.5} stroke="rgba(0,0,0,0.06)" strokeDasharray="4 4" />
                    <Area type="monotone" dataKey="reward" stroke="#0066ff" fill="url(#gradReward)" strokeWidth={2} name="Adaptive" />
                    <Line type="monotone" dataKey="baseline" stroke="#71717a" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Baseline" />
                  </AreaChart>
                </ResponsiveContainer>
              )}

              {activeChart === 'winRate' && (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={MOCK_BENCHMARK_DATA.winRateOverTime}>
                    <defs>
                      <linearGradient id="gradWin" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10b981" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
                    <XAxis dataKey="task" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0.4, 1]} tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip {...tooltipStyle} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                    <ReferenceLine y={0.75} stroke="rgba(16,185,129,0.2)" strokeDasharray="4 4" />
                    <Area type="monotone" dataKey="winRate" stroke="#10b981" fill="url(#gradWin)" strokeWidth={2} name="Win Rate" />
                  </AreaChart>
                </ResponsiveContainer>
              )}

              {activeChart === 'entropy' && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={MOCK_BENCHMARK_DATA.branchEntropy}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
                    <XAxis dataKey="task" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip {...tooltipStyle} />
                    <Line type="monotone" dataKey="entropy" stroke="#7c3aed" strokeWidth={2} dot={false} name="Selection Entropy" />
                  </LineChart>
                </ResponsiveContainer>
              )}

              {activeChart === 'latency' && (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
                    <XAxis dataKey="latency" name="Latency (ms)" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} label={{ value: 'Latency (ms)', fill: '#71717a', fontSize: 10, position: 'insideBottom', offset: -5 }} />
                    <YAxis dataKey="quality" name="Quality" domain={[0.5, 1]} tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} label={{ value: 'Quality', fill: '#71717a', fontSize: 10, angle: -90, position: 'insideLeft' }} />
                    <ZAxis dataKey="weight" range={[40, 200]} name="Weight" />
                    <Tooltip {...tooltipStyle} />
                    <Scatter data={MOCK_BENCHMARK_DATA.latencyVsQuality} fill="#0066ff" name="Branches" />
                  </ScatterChart>
                </ResponsiveContainer>
              )}

              {activeChart === 'shift' && (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={MOCK_BENCHMARK_DATA.shiftComparison}>
                    <defs>
                      <linearGradient id="gradAdaptB" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#0066ff" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#0066ff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
                    <XAxis dataKey="task" tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0.3, 1]} tick={{ fill: '#71717a', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip {...tooltipStyle} />
                    <ReferenceLine x={10} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.5} />
                    <Area type="monotone" dataKey="adaptive" stroke="#0066ff" fill="url(#gradAdaptB)" strokeWidth={2} name="Adaptive" />
                    <Line type="monotone" dataKey="baseline" stroke="#71717a" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Baseline" />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
}

import type { SimulationReport } from '../types';

export function MacroPanel({ report }: { report: SimulationReport }) {
  const sectors = report.sectors ?? { booming: {}, struggling: {}, all_sectors: {} };
  const segments = report.segments ?? [];
  const concerns = report.top_concerns ?? [];
  const emotions = report.emotion_distribution ?? [];

  return (
    <div className="space-y-8">
      {/* Sector Impacts */}
      <section>
        <h3 className="text-lg font-bold mb-4">Market Sector Impacts</h3>
        <div className="grid md:grid-cols-2 gap-4">
          {/* Booming */}
          <div className="bg-emerald-50 rounded-xl p-5 border border-emerald-200">
            <h4 className="text-sm font-semibold text-emerald-800 mb-3">Booming Sectors</h4>
            {Object.entries(sectors.booming ?? {}).map(([sec, val]) => (
              <div key={sec} className="flex items-center gap-3 mb-2">
                <span className="text-sm font-medium text-text-base w-32">{sec}</span>
                <div className="flex-1 h-5 bg-emerald-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all"
                    style={{ width: `${Math.min(100, Math.abs(val) * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-emerald-700 w-12 text-right">+{val.toFixed(1)}</span>
              </div>
            ))}
            {Object.keys(sectors.booming ?? {}).length === 0 && (
              <p className="text-sm text-emerald-600">No sectors booming</p>
            )}
          </div>

          {/* Struggling */}
          <div className="bg-red-50 rounded-xl p-5 border border-red-200">
            <h4 className="text-sm font-semibold text-red-800 mb-3">Struggling Sectors</h4>
            {Object.entries(sectors.struggling ?? {}).map(([sec, val]) => (
              <div key={sec} className="flex items-center gap-3 mb-2">
                <span className="text-sm font-medium text-text-base w-32">{sec}</span>
                <div className="flex-1 h-5 bg-red-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 rounded-full transition-all"
                    style={{ width: `${Math.min(100, Math.abs(val) * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-red-700 w-12 text-right">{val.toFixed(1)}</span>
              </div>
            ))}
            {Object.keys(sectors.struggling ?? {}).length === 0 && (
              <p className="text-sm text-red-600">No sectors struggling</p>
            )}
          </div>
        </div>
      </section>

      {/* Population Segments */}
      <section>
        <h3 className="text-lg font-bold mb-4">Who Gets Hit (Change from Baseline)</h3>
        <div className="bg-white rounded-xl border border-black/5 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 text-[11px] uppercase tracking-wider text-text-muted">
                <th className="text-left px-4 py-2.5">Segment</th>
                <th className="text-right px-3 py-2.5">Count</th>
                <th className="text-right px-3 py-2.5">Debt &Delta;</th>
                <th className="text-right px-3 py-2.5">Dread &Delta;</th>
                <th className="text-right px-3 py-2.5">Pessimism &Delta;</th>
                <th className="text-right px-3 py-2.5">Mood &Delta;</th>
                <th className="text-right px-3 py-2.5">Savings Lost</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((seg, idx) => (
                <tr key={seg.segment} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                  <td className="px-4 py-2 font-medium">{seg.segment}</td>
                  <td className="px-3 py-2 text-right text-text-muted">{seg.count}</td>
                  <td className={`px-3 py-2 text-right font-mono text-xs ${seg.debt_change > 0.1 ? 'text-red-600 font-semibold' : 'text-text-secondary'}`}>
                    {seg.debt_change > 0 ? '+' : ''}{seg.debt_change.toFixed(2)}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono text-xs ${seg.dread_change > 0.2 ? 'text-red-600 font-semibold' : 'text-text-secondary'}`}>
                    {seg.dread_change > 0 ? '+' : ''}{seg.dread_change.toFixed(2)}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono text-xs ${seg.pessimism_change > 0.3 ? 'text-red-600 font-semibold' : 'text-text-secondary'}`}>
                    {seg.pessimism_change > 0 ? '+' : ''}{seg.pessimism_change.toFixed(2)}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono text-xs ${seg.mood_change < -0.1 ? 'text-red-600' : 'text-emerald-600'}`}>
                    {seg.mood_change > 0 ? '+' : ''}{seg.mood_change.toFixed(2)}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono text-xs ${seg.avg_savings_lost > 0.2 ? 'text-red-600 font-semibold' : 'text-text-secondary'}`}>
                    {seg.avg_savings_lost > 0 ? '+' : ''}{seg.avg_savings_lost.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Concerns + Emotions side by side */}
      <div className="grid md:grid-cols-2 gap-6">
        <section>
          <h3 className="text-lg font-bold mb-3">Top Concerns</h3>
          <div className="space-y-2">
            {concerns.slice(0, 8).map(([concern, count], idx) => {
              const pct = (count / report.total_agents * 100);
              return (
                <div key={idx} className="flex items-center gap-3">
                  <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden relative">
                    <div className="h-full bg-blue-200 rounded-full" style={{ width: `${pct}%` }} />
                    <span className="absolute inset-0 flex items-center px-3 text-xs text-text-base truncate">
                      {concern}
                    </span>
                  </div>
                  <span className="text-xs text-text-muted w-12 text-right">{pct.toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </section>

        <section>
          <h3 className="text-lg font-bold mb-3">Emotional State</h3>
          <div className="space-y-2">
            {emotions.map(([emotion, count], idx) => {
              const pct = (count / report.total_agents * 100);
              const color = emotion === 'content' ? 'bg-emerald-300'
                : emotion === 'neutral' ? 'bg-gray-300'
                : emotion === 'composed' ? 'bg-blue-300'
                : emotion === 'defeated' ? 'bg-red-300'
                : emotion === 'forced calm' ? 'bg-amber-300'
                : 'bg-gray-300';
              return (
                <div key={idx} className="flex items-center gap-3">
                  <span className="text-sm w-24 text-text-base">{emotion}</span>
                  <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden">
                    <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-xs text-text-muted w-12 text-right">{pct.toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}

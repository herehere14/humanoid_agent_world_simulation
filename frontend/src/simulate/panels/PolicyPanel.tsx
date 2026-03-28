import { motion } from 'framer-motion';
import type { SimulationReport } from '../types';

const DOMAIN_COLORS: Record<string, string> = {
  housing: 'border-l-orange-400',
  energy: 'border-l-amber-400',
  climate: 'border-l-emerald-400',
  immigration: 'border-l-blue-400',
  labor: 'border-l-red-400',
  fiscal: 'border-l-purple-400',
  healthcare: 'border-l-pink-400',
  education: 'border-l-cyan-400',
  defence: 'border-l-slate-500',
  trade: 'border-l-indigo-400',
  welfare: 'border-l-teal-400',
};

export function PolicyPanel({ report }: { report: SimulationReport }) {
  const policies = report.policy_impacts ?? [];

  return (
    <div>
      <h3 className="text-lg font-bold mb-1">Economic Consequence Chain</h3>
      <p className="text-sm text-text-muted mb-6">
        Each policy/consequence and who it affects. These were identified by the LLM research phase
        and injected as persistent conditions into the simulation.
      </p>

      <div className="space-y-4">
        {policies.map((p, idx) => {
          const borderClass = DOMAIN_COLORS[p.domain] ?? 'border-l-gray-300';
          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
              className={`bg-white rounded-xl border border-black/5 border-l-4 ${borderClass} overflow-hidden`}
            >
              <div className="p-5">
                {/* Header */}
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <span className="text-[10px] uppercase tracking-wider text-text-muted">
                      {p.domain}
                    </span>
                    <h4 className="text-sm font-semibold text-text-base mt-0.5">{p.policy}</h4>
                  </div>
                  <span className="text-xs text-text-muted whitespace-nowrap">
                    {p.agents_affected} agents
                  </span>
                </div>

                {/* Description */}
                <p className="text-sm text-text-secondary mt-2 leading-relaxed">{p.description}</p>

                {/* Winners and Losers */}
                <div className="flex gap-4 mt-3">
                  {p.winners?.length > 0 && (
                    <div>
                      <span className="text-[10px] text-emerald-600 uppercase tracking-wider">Winners</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {p.winners.map((w, i) => (
                          <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                            {w}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {p.losers?.length > 0 && (
                    <div>
                      <span className="text-[10px] text-red-600 uppercase tracking-wider">Losers</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {p.losers.map((l, i) => (
                          <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200">
                            {l}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Sector impacts */}
                {p.sectors && Object.keys(p.sectors).length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {Object.entries(p.sectors).sort((a, b) => b[1] - a[1]).map(([sec, val]) => (
                      <span key={sec} className={`text-[10px] px-2 py-0.5 rounded-full border ${
                        val > 0
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : 'bg-red-50 text-red-700 border-red-200'
                      }`}>
                        {sec} {val > 0 ? '+' : ''}{val.toFixed(1)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {policies.length === 0 && (
        <div className="text-center py-12 text-text-muted">No policy impacts recorded.</div>
      )}
    </div>
  );
}

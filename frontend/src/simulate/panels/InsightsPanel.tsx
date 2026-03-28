import { motion } from 'framer-motion';
import type { SimulationReport } from '../types';

const TYPE_STYLES: Record<string, { bg: string; border: string; icon: string }> = {
  savings_depletion: { bg: 'bg-red-50', border: 'border-red-200', icon: '💸' },
  counterintuitive_resilience: { bg: 'bg-emerald-50', border: 'border-emerald-200', icon: '🛡' },
  psychological_damage: { bg: 'bg-purple-50', border: 'border-purple-200', icon: '🧠' },
  emotional_masking: { bg: 'bg-amber-50', border: 'border-amber-200', icon: '🎭' },
  blame_concentration: { bg: 'bg-orange-50', border: 'border-orange-200', icon: '🎯' },
  cascade_victim: { bg: 'bg-red-50', border: 'border-red-200', icon: '🌊' },
  cascade_source: { bg: 'bg-blue-50', border: 'border-blue-200', icon: '⚡' },
  sector_paradox: { bg: 'bg-yellow-50', border: 'border-yellow-200', icon: '🔄' },
  compound_policy_squeeze: { bg: 'bg-pink-50', border: 'border-pink-200', icon: '🗜' },
  coalition_fracture_risk: { bg: 'bg-slate-50', border: 'border-slate-200', icon: '💔' },
};

export function InsightsPanel({ report }: { report: SimulationReport }) {
  const insights = report.insights ?? [];

  return (
    <div>
      <h3 className="text-lg font-bold mb-1">Non-Obvious Insights</h3>
      <p className="text-sm text-text-muted mb-6">
        These are things you wouldn't predict from reading the news — emergent patterns from the simulation.
      </p>

      <div className="grid gap-4">
        {insights.map((insight, idx) => {
          const style = TYPE_STYLES[insight.type] ?? { bg: 'bg-gray-50', border: 'border-gray-200', icon: '🔍' };
          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.05 }}
              className={`${style.bg} border ${style.border} rounded-xl p-5`}
            >
              <div className="flex items-start gap-3">
                <span className="text-xl">{style.icon}</span>
                <div className="min-w-0">
                  <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1">
                    {insight.type.replace(/_/g, ' ')}
                  </div>
                  <h4 className="text-sm font-semibold text-text-base mb-2">
                    {insight.title}
                  </h4>
                  <p className="text-sm text-text-secondary leading-relaxed">
                    {insight.detail}
                  </p>
                  {insight.policies && insight.policies.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {insight.policies.map((p, i) => (
                        <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-white/80 text-text-secondary border border-black/5">
                          {p}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <span className="text-[10px] text-text-muted whitespace-nowrap ml-auto">
                  surprise: {insight.surprise_factor?.toFixed(2)}
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>

      {insights.length === 0 && (
        <div className="text-center py-12 text-text-muted">No non-obvious insights found.</div>
      )}
    </div>
  );
}

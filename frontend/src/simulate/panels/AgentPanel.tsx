import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SimulationReport, KeyFigure } from '../types';

export function AgentPanel({ report }: { report: SimulationReport }) {
  const figures = report.key_figures ?? [];
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  // Get decisions for a specific agent
  const getAgentDecisions = (name: string) => {
    return (report.llm_agency?.full_decision_log ?? []).filter(d => d.agent === name);
  };

  return (
    <div>
      <h3 className="text-lg font-bold mb-4">Key Figures</h3>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {figures.map((fig, idx) => {
          const isSelected = selectedIdx === idx;
          const decisions = getAgentDecisions(fig.name);

          return (
            <motion.div
              key={idx}
              layout
              className={`bg-white rounded-xl border overflow-hidden cursor-pointer transition-all ${
                isSelected ? 'border-primary shadow-lg col-span-full' : 'border-black/5 hover:shadow-sm'
              }`}
              onClick={() => setSelectedIdx(isSelected ? null : idx)}
            >
              {/* Card header */}
              <div className="p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="text-sm font-semibold text-text-base">{fig.name}</h4>
                    <p className="text-xs text-text-muted">{fig.title}</p>
                    {fig.org && <p className="text-[10px] text-text-muted mt-0.5">{fig.org}</p>}
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    fig.emotion === 'content' ? 'bg-emerald-50 text-emerald-700' :
                    fig.emotion === 'neutral' ? 'bg-gray-100 text-gray-600' :
                    fig.emotion === 'composed' ? 'bg-blue-50 text-blue-700' :
                    fig.emotion === 'defeated' ? 'bg-red-50 text-red-700' :
                    'bg-gray-100 text-gray-600'
                  }`}>
                    {fig.emotion}
                  </span>
                </div>

                {/* Stats bar */}
                <div className="grid grid-cols-3 gap-2 mt-3">
                  <div className="text-center">
                    <div className="text-[10px] text-text-muted">Tension</div>
                    <div className="text-sm font-mono font-medium">{fig.tension.toFixed(2)}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-[10px] text-text-muted">Dread</div>
                    <div className="text-sm font-mono font-medium">{fig.dread.toFixed(2)}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-[10px] text-text-muted">Mood &Delta;</div>
                    <div className={`text-sm font-mono font-medium ${fig.mood_change < 0 ? 'text-red-600' : 'text-emerald-600'}`}>
                      {fig.mood_change > 0 ? '+' : ''}{fig.mood_change.toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Concern */}
                <div className="mt-3 text-xs text-text-secondary">
                  <span className="text-text-muted">Concern:</span> {fig.concern}
                </div>

                {/* Coalitions */}
                {fig.coalitions?.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {fig.coalitions.map((c, i) => (
                      <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-600 border border-blue-100">{c}</span>
                    ))}
                  </div>
                )}
              </div>

              {/* Expanded detail */}
              <AnimatePresence>
                {isSelected && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden border-t border-black/5"
                  >
                    <div className="p-4 space-y-4">
                      {/* Story */}
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Ongoing Story</div>
                        <p className="text-sm text-text-secondary">{fig.ongoing_story}</p>
                      </div>

                      {/* Blame */}
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Blames</div>
                        <p className="text-sm text-text-secondary">{fig.blame_target}</p>
                      </div>

                      {/* Emotional divergence */}
                      {fig.divergence > 0.05 && (
                        <div className="bg-amber-50 rounded-lg p-3">
                          <div className="text-[10px] uppercase tracking-wider text-amber-600 mb-1">Emotional Masking</div>
                          <p className="text-sm text-text-secondary">
                            Shows: <strong>{fig.emotion}</strong> &middot;
                            Feels: <strong>{fig.internal_emotion}</strong> &middot;
                            Divergence: <strong>{fig.divergence.toFixed(2)}</strong>
                          </p>
                        </div>
                      )}

                      {/* Policy reactions */}
                      {fig.policy_reactions?.length > 0 && (
                        <div>
                          <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Policy Reactions</div>
                          {fig.policy_reactions.map((r, i) => (
                            <p key={i} className="text-xs text-text-secondary mb-1">{r}</p>
                          ))}
                        </div>
                      )}

                      {/* This agent's decisions */}
                      {decisions.length > 0 && (
                        <div>
                          <div className="text-[10px] uppercase tracking-wider text-text-muted mb-2">
                            Decisions Made ({decisions.length})
                          </div>
                          <div className="space-y-2 max-h-60 overflow-y-auto">
                            {decisions.map((d, i) => (
                              <div key={i} className="bg-gray-50 rounded-lg p-3">
                                <div className="text-[10px] text-text-muted mb-1">{d.time}</div>
                                <div className="text-xs text-text-base">{d.action}</div>
                                {d.message_recipients?.length > 0 && (
                                  <div className="flex gap-1 mt-1.5 flex-wrap">
                                    {d.message_recipients.map((r, j) => (
                                      <span key={j} className="text-[10px] px-1.5 py-0.5 rounded bg-sky-50 text-sky-600 border border-sky-100">{r}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

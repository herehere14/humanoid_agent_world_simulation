import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SimulationReport, SimDecision } from '../types';

export function TimelinePanel({ report }: { report: SimulationReport }) {
  const decisions = report.llm_agency?.full_decision_log ?? [];
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [filter, setFilter] = useState<string>('all');

  // Group decisions by simulated day
  const byDay: Record<string, SimDecision[]> = {};
  for (const d of decisions) {
    const day = d.time?.split(',')[0] ?? 'Unknown';
    if (!byDay[day]) byDay[day] = [];
    byDay[day].push(d);
  }

  // Get unique agent names for filter
  const agentNames = [...new Set(decisions.map(d => d.agent))].sort();

  const filtered = filter === 'all'
    ? decisions
    : decisions.filter(d => d.agent === filter);

  const triggerColor = (trigger?: string) => {
    if (!trigger) return 'bg-blue-100 text-blue-700';
    if (trigger.includes('reactive')) return 'bg-orange-100 text-orange-700';
    if (trigger.includes('conversation')) return 'bg-purple-100 text-purple-700';
    return 'bg-blue-100 text-blue-700';
  };

  const triggerLabel = (trigger?: string) => {
    if (!trigger || trigger === 'scheduled') return 'scheduled';
    if (trigger.includes('reactive')) return 'reactive cascade';
    if (trigger.includes('conversation')) return 'conversation';
    return trigger;
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold">
          Decision Timeline
          <span className="text-sm font-normal text-text-muted ml-2">
            {filtered.length} decisions
          </span>
        </h3>
        <select
          value={filter}
          onChange={e => setFilter(e.target.value)}
          className="text-sm px-3 py-1.5 rounded-lg border border-black/10 bg-white"
        >
          <option value="all">All agents</option>
          {agentNames.map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      <div className="space-y-2">
        {filtered.map((d, idx) => (
          <motion.div
            key={idx}
            layout
            className="bg-white rounded-xl border border-black/5 overflow-hidden hover:shadow-sm transition-shadow"
          >
            {/* Header row — always visible */}
            <button
              onClick={() => setExpandedIdx(expandedIdx === idx ? null : idx)}
              className="w-full px-4 py-3 flex items-start gap-3 text-left"
            >
              {/* Time */}
              <span className="text-[11px] font-mono text-text-muted whitespace-nowrap pt-0.5 min-w-[80px]">
                {d.time}
              </span>

              {/* Trigger badge */}
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${triggerColor(d.trigger)}`}>
                {triggerLabel(d.trigger)}
              </span>

              {/* Agent name + role */}
              <div className="min-w-0">
                <span className="font-semibold text-sm text-text-base">{d.agent}</span>
                <span className="text-xs text-text-muted ml-1.5">({d.role})</span>
                {d.triggered_by && (
                  <span className="text-[10px] text-orange-600 ml-2">
                    reacting to {d.triggered_by}
                  </span>
                )}
                {d.conversation_with && (
                  <span className="text-[10px] text-purple-600 ml-2">
                    talking with {d.conversation_with}
                  </span>
                )}
              </div>

              {/* Cascade indicators */}
              <div className="ml-auto flex items-center gap-2 shrink-0">
                {d.ripple_count > 0 && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
                    {d.ripple_count} ripples
                  </span>
                )}
                {d.messages_sent > 0 && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200">
                    {d.messages_sent} msgs
                  </span>
                )}
                {(d.cascade_depth ?? 0) > 0 && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200">
                    depth {d.cascade_depth}
                  </span>
                )}
                <svg className={`w-4 h-4 text-text-muted transition-transform ${expandedIdx === idx ? 'rotate-180' : ''}`}
                  viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" />
                </svg>
              </div>
            </button>

            {/* Expanded detail */}
            <AnimatePresence>
              {expandedIdx === idx && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4 pt-1 border-t border-black/5 space-y-3">
                    {/* Action */}
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Action</div>
                      <div className="text-sm text-text-base">{d.action}</div>
                    </div>

                    {/* Reasoning */}
                    {d.reasoning && (
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Why</div>
                        <div className="text-sm text-text-secondary">{d.reasoning}</div>
                      </div>
                    )}

                    {/* Public speech */}
                    {d.speech && (
                      <div className="bg-blue-50/50 rounded-lg p-3">
                        <div className="text-[10px] uppercase tracking-wider text-blue-600 mb-1">Says publicly</div>
                        <div className="text-sm text-text-base italic">"{d.speech}"</div>
                      </div>
                    )}

                    {/* Private thought */}
                    {d.thought && (
                      <div className="bg-gray-50 rounded-lg p-3">
                        <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Thinks privately</div>
                        <div className="text-sm text-text-secondary italic">"{d.thought}"</div>
                      </div>
                    )}

                    {/* Messages sent */}
                    {d.message_recipients?.length > 0 && (
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Messages sent to</div>
                        <div className="flex flex-wrap gap-1.5">
                          {d.message_recipients.map((r, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200">
                              {r}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Meta */}
                    <div className="flex gap-4 text-[11px] text-text-muted pt-1">
                      <span>Latency: {d.latency_ms}ms</span>
                      <span>Consequences: {d.consequences ?? d.ripple_count}</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {decisions.length === 0 && (
        <div className="text-center py-12 text-text-muted">
          No LLM agent decisions recorded. Run with <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded">llm_agents=True</code> to see agent interactions.
        </div>
      )}
    </div>
  );
}

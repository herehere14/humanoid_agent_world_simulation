import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SimulationReport, SimDecision } from './types';
import { TimelinePanel } from './panels/TimelinePanel';
import { InsightsPanel } from './panels/InsightsPanel';
import { MacroPanel } from './panels/MacroPanel';
import { AgentPanel } from './panels/AgentPanel';
import { NetworkPanel } from './panels/NetworkPanel';
import { PolicyPanel } from './panels/PolicyPanel';

const API_BASE = '/api';

type TabId = 'live' | 'timeline' | 'insights' | 'macro' | 'agents' | 'network' | 'policy';

const TABS: { id: TabId; label: string }[] = [
  { id: 'live', label: 'Live Feed' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'insights', label: 'Insights' },
  { id: 'macro', label: 'Economy' },
  { id: 'agents', label: 'Key Figures' },
  { id: 'network', label: 'Network' },
  { id: 'policy', label: 'Policy Chain' },
];

// ═══════════════════════════════════════════════════════════
// Live Feed Event Types
// ═══════════════════════════════════════════════════════════

interface LiveEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: number;
}

// ═══════════════════════════════════════════════════════════
// Grid Background
// ═══════════════════════════════════════════════════════════

function GridBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(rgba(0,102,255,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(0,102,255,0.4) 1px, transparent 1px)`,
          backgroundSize: '60px 60px',
        }}
      />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] opacity-30"
        style={{ background: 'radial-gradient(ellipse, rgba(0,102,255,0.08), transparent 70%)' }}
      />
    </div>
  );
}

function PulseDot({ color = '#10b981' }: { color?: string }) {
  return (
    <span className="relative flex h-2 w-2">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-40" style={{ backgroundColor: color }} />
      <span className="relative inline-flex rounded-full h-2 w-2" style={{ backgroundColor: color }} />
    </span>
  );
}

// ═══════════════════════════════════════════════════════════
// Live Feed Panel — shows events as they stream in
// ═══════════════════════════════════════════════════════════

function LiveFeedPanel({ events, running }: { events: LiveEvent[]; running: boolean }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  return (
    <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-2">
      {events.map((evt, idx) => (
        <LiveEventCard key={idx} event={evt} />
      ))}
      {running && (
        <div className="flex items-center gap-2 py-3 px-4 text-sm text-slate-400">
          <div className="w-3 h-3 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin" />
          Simulation running...
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}

function LiveEventCard({ event }: { event: LiveEvent }) {
  const { type, data } = event;

  if (type === 'status') {
    return (
      <div className="flex items-center gap-2 py-2 px-4 text-xs text-slate-400">
        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
        {data.message as string}
      </div>
    );
  }

  if (type === 'setup') {
    const figures = (data.key_figures as Array<Record<string, string>>) ?? [];
    const policies = (data.policies as Array<Record<string, string>>) ?? [];
    return (
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-100 p-4">
        <div className="text-[10px] uppercase tracking-wider text-blue-500 font-semibold mb-2">Research Complete</div>
        <div className="text-sm font-medium text-slate-800 mb-3">{data.prediction as string}</div>
        <div className="grid md:grid-cols-2 gap-3">
          <div>
            <div className="text-[10px] text-slate-400 uppercase mb-1">Key Figures ({figures.length})</div>
            <div className="flex flex-wrap gap-1">
              {figures.map((f, i) => (
                <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-white border border-blue-100 text-slate-700">{f.name}</span>
              ))}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400 uppercase mb-1">Policies ({policies.length})</div>
            <div className="flex flex-wrap gap-1">
              {policies.map((p, i) => (
                <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-white border border-purple-100 text-slate-700">{p.name}</span>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'decision') {
    const trigger = data.trigger as string;
    const isReactive = trigger?.includes('reactive');
    const isConversation = trigger?.includes('conversation');
    const borderColor = isReactive ? 'border-l-orange-400' : isConversation ? 'border-l-purple-400' : 'border-l-blue-400';
    const badgeColor = isReactive ? 'bg-orange-100 text-orange-700' : isConversation ? 'bg-purple-100 text-purple-700' : 'bg-blue-50 text-blue-600';
    const badgeLabel = isReactive ? 'reactive' : isConversation ? 'conversation' : 'scheduled';

    return (
      <div className={`bg-white rounded-xl border border-black/5 border-l-4 ${borderColor} p-4`}>
        <div className="flex items-start gap-2 mb-2">
          <span className="text-[10px] font-mono text-slate-400 pt-0.5">{data.time as string}</span>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${badgeColor}`}>{badgeLabel}</span>
          <span className="text-sm font-semibold text-slate-800">{data.agent as string}</span>
          <span className="text-xs text-slate-400">({data.role as string})</span>
          {String(data.triggered_by ?? '') !== '' && (
            <span className="text-[10px] text-orange-500 ml-auto">reacting to {String(data.triggered_by)}</span>
          )}
        </div>
        <div className="text-sm text-slate-700 mb-2">{String(data.action ?? '')}</div>
        {String(data.speech ?? '') !== '' && (
          <div className="bg-blue-50/50 rounded-lg px-3 py-2 mb-2">
            <span className="text-[10px] text-blue-500 uppercase">Says: </span>
            <span className="text-xs text-slate-700 italic">"{String(data.speech)}"</span>
          </div>
        )}
        {String(data.thought ?? '') !== '' && (
          <div className="bg-slate-50 rounded-lg px-3 py-2 mb-2">
            <span className="text-[10px] text-slate-400 uppercase">Thinks: </span>
            <span className="text-xs text-slate-500 italic">"{String(data.thought)}"</span>
          </div>
        )}
        <div className="flex items-center gap-2 flex-wrap">
          {((data.messages_to as string[]) ?? []).map((r, i) => (
            <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200">
              msg → {r}
            </span>
          ))}
          {(data.ripple_count as number) > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
              {data.ripple_count as number} ripples
            </span>
          )}
          {((data.cascade_depth as number) ?? 0) > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200">
              cascade depth {data.cascade_depth as number}
            </span>
          )}
        </div>
      </div>
    );
  }

  if (type === 'day_summary') {
    const macro = data.macro as Record<string, number> | undefined;
    const worst = (data.worst_hit as Array<Record<string, unknown>>) ?? [];
    return (
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-xl p-5 text-white">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-lg font-bold">Day {data.day as number}</span>
          <span className="text-xs text-slate-400">{data.decisions_today as number} decisions</span>
          <span className="text-xs text-slate-500 ml-auto">{data.total_decisions as number} total</span>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed mb-4">{data.narrative as string}</p>
        {macro && (
          <div className="grid grid-cols-5 gap-3">
            {[
              { key: 'consumer_confidence', label: 'Confidence', fmt: (v: number) => v.toFixed(0) },
              { key: 'population_mood', label: 'Mood', fmt: (v: number) => v.toFixed(2) },
              { key: 'market_pressure', label: 'Mkt Pressure', fmt: (v: number) => v.toFixed(2) },
              { key: 'institutional_trust', label: 'Trust', fmt: (v: number) => v.toFixed(2) },
              { key: 'civil_unrest', label: 'Unrest', fmt: (v: number) => v.toFixed(3) },
            ].map(m => (
              <div key={m.key} className="text-center">
                <div className="text-[10px] text-slate-500 uppercase">{m.label}</div>
                <div className="text-lg font-bold text-white">{m.fmt(macro[m.key] ?? 0)}</div>
              </div>
            ))}
          </div>
        )}
        {worst.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            <span className="text-[10px] text-slate-500">Hardest hit:</span>
            {worst.map((w, i) => (
              <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/20 text-red-300">
                {w.segment as string} ({(w.pessimism as number).toFixed(2)})
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (type === 'insight') {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
        <div className="text-[10px] uppercase tracking-wider text-amber-600 mb-1">
          Insight: {(data.type as string ?? '').replace(/_/g, ' ')}
        </div>
        <div className="text-sm font-medium text-slate-800">{data.title as string}</div>
        <div className="text-xs text-slate-600 mt-1">{data.detail as string}</div>
      </div>
    );
  }

  if (type === 'event') {
    return (
      <div className="flex items-start gap-2 py-2 px-4 bg-white/50 rounded-lg border border-black/[0.03]">
        <span className="text-[10px] font-mono text-slate-400 pt-0.5 shrink-0">{data.time as string}</span>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-600 shrink-0">{data.kind as string}</span>
        <span className="text-xs text-slate-600">{data.description as string}</span>
      </div>
    );
  }

  if (type === 'world_built') {
    return (
      <div className="flex items-center gap-3 py-2 px-4 text-xs text-slate-500">
        <span className="text-emerald-500 font-medium">World built:</span>
        {data.total_agents as number} agents, {data.total_locations as number} locations
      </div>
    );
  }

  return null;
}


// ═══════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════

export default function SimulationViewer() {
  const [prediction, setPrediction] = useState('');
  const [report, setReport] = useState<SimulationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('live');
  const [liveEvents, setLiveEvents] = useState<LiveEvent[]>([]);
  const [dayNarratives, setDayNarratives] = useState<Array<{ day: number; narrative: string; macro: Record<string, number> }>>([]);
  const [stats, setStats] = useState({ agents: 0, decisions: 0, day: 0, ripples: 0 });

  const runSimulation = useCallback(async () => {
    if (!prediction.trim()) return;
    setLoading(true);
    setError(null);
    setReport(null);
    setLiveEvents([]);
    setDayNarratives([]);
    setActiveTab('live');
    setStats({ agents: 0, decisions: 0, day: 0, ripples: 0 });

    try {
      const res = await fetch(`${API_BASE}/world/predict/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction: prediction.trim(),
          ticks: 168,
          model: 'gpt-5-mini',
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;

          try {
            const event = JSON.parse(trimmed.slice(6)) as LiveEvent;
            event.timestamp = Date.now();

            if (event.type === 'done') continue;
            if (event.type === 'error') {
              setError((event.data as { message: string }).message);
              continue;
            }

            // Add to live feed
            setLiveEvents(prev => [...prev, event]);

            // Update stats
            if (event.type === 'world_built') {
              setStats(s => ({ ...s, agents: (event.data.total_agents as number) ?? 0 }));
            }
            if (event.type === 'decision') {
              setStats(s => ({
                ...s,
                decisions: s.decisions + 1,
                ripples: s.ripples + ((event.data.ripple_count as number) ?? 0),
              }));
            }
            if (event.type === 'day_summary') {
              const d = event.data as { day: number; narrative: string; macro: Record<string, number>; total_decisions: number };
              setStats(s => ({ ...s, day: d.day, decisions: d.total_decisions }));
              setDayNarratives(prev => [...prev, { day: d.day, narrative: d.narrative, macro: d.macro }]);
            }

            // Final report
            if (event.type === 'complete') {
              setReport(event.data as unknown as SimulationReport);
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [prediction]);

  return (
    <div className="min-h-screen bg-[#f8f9fb] relative">
      <GridBackground />

      {/* ═══ Header ═══ */}
      <header className="sticky top-0 z-50 glass-strong border-b border-black/[0.06]">
        <div className="max-w-[1440px] mx-auto px-6 h-12 flex items-center gap-4">
          <a href="#/" className="text-xs text-slate-400 hover:text-slate-600 transition-colors font-medium">&larr; Back</a>
          <div className="w-px h-4 bg-slate-200" />
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <svg viewBox="0 0 16 16" className="w-3 h-3 text-white" fill="currentColor">
                <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 2.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM5.5 9a2.5 2.5 0 015 0v.5a.5.5 0 01-.5.5h-4a.5.5 0 01-.5-.5V9z"/>
              </svg>
            </div>
            <span className="text-[13px] font-semibold text-slate-800">World Simulation</span>
          </div>
          {(loading || report) && (
            <div className="ml-auto flex items-center gap-4 text-[11px] text-slate-400">
              {loading && <PulseDot color="#f59e0b" />}
              {!loading && report && <PulseDot color="#10b981" />}
              <span>{stats.agents} agents</span>
              <span className="text-slate-300">|</span>
              <span>Day {stats.day}</span>
              <span className="text-slate-300">|</span>
              <span>{stats.decisions} decisions</span>
              <span className="text-slate-300">|</span>
              <span>{stats.ripples} ripples</span>
            </div>
          )}
        </div>
      </header>

      {/* ═══ Input ═══ */}
      <div className="relative">
        <div className="max-w-[1440px] mx-auto px-6 pt-12 pb-8">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto text-center">
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 mb-2">
              What happens if<span className="gradient-text">...</span>
            </h1>
            <p className="text-sm text-slate-500 mb-6 max-w-lg mx-auto leading-relaxed">
              Real people. Freeform decisions. Reactive cascades. Live streaming.
            </p>
            <div className="relative">
              <input
                type="text"
                value={prediction}
                onChange={e => setPrediction(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !loading && runSimulation()}
                placeholder="e.g. Zombie virus outbreak in rural Australia..."
                className="w-full px-5 py-4 pr-32 rounded-2xl border border-black/[0.08] bg-white text-[15px] text-slate-900 placeholder:text-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/30 shadow-[0_2px_8px_rgba(0,0,0,0.04)] transition-all"
                disabled={loading}
              />
              <button
                onClick={runSimulation}
                disabled={loading || !prediction.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white transition-all disabled:opacity-30"
                style={{
                  background: loading ? '#94a3b8' : 'linear-gradient(135deg, #0066ff, #5b21b6)',
                  boxShadow: loading ? 'none' : '0 2px 8px rgba(0,102,255,0.3)',
                }}
              >
                {loading ? 'Simulating...' : 'Simulate'}
              </button>
            </div>
            {!report && !loading && (
              <div className="flex flex-wrap justify-center gap-2 mt-4">
                {[
                  'Zombie virus in rural Australia',
                  'Elon Musk dies suddenly',
                  'Nuclear weapon hits Shanghai',
                  'US biggest property developer bankrupt',
                  'Albanese loses Australian election',
                ].map(s => (
                  <button key={s} onClick={() => setPrediction(s)}
                    className="text-[11px] px-3 py-1.5 rounded-full border border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300 hover:bg-white transition-all">
                    {s}
                  </button>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>

      {error && (
        <div className="max-w-2xl mx-auto px-6 mb-6">
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-sm text-red-700">{error}</div>
        </div>
      )}

      {/* ═══ Results ═══ */}
      <AnimatePresence>
        {(liveEvents.length > 0 || report) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            {/* Tab Navigation */}
            <div className="border-t border-b border-black/[0.04] bg-white sticky top-12 z-40">
              <div className="max-w-[1440px] mx-auto px-6">
                <div className="flex">
                  {TABS.map(tab => {
                    const disabled = tab.id !== 'live' && !report;
                    return (
                      <button
                        key={tab.id}
                        onClick={() => !disabled && setActiveTab(tab.id)}
                        disabled={disabled}
                        className={`relative px-5 py-3.5 text-[13px] font-medium transition-all ${
                          disabled ? 'text-slate-300 cursor-not-allowed' :
                          activeTab === tab.id ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'
                        }`}
                      >
                        {tab.id === 'live' && loading && <PulseDot color="#f59e0b" />}
                        {' '}{tab.label}
                        {activeTab === tab.id && (
                          <motion.div
                            layoutId="activeTab"
                            className="absolute bottom-0 left-0 right-0 h-[2px] rounded-full"
                            style={{ background: 'linear-gradient(90deg, #0066ff, #7c3aed)' }}
                          />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Tab Content */}
            <div className="max-w-[1440px] mx-auto px-6 py-6 relative z-10">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.15 }}
                >
                  {activeTab === 'live' && <LiveFeedPanel events={liveEvents} running={loading} />}
                  {activeTab === 'timeline' && report && <TimelinePanel report={report} />}
                  {activeTab === 'insights' && report && <InsightsPanel report={report} />}
                  {activeTab === 'macro' && report && <MacroPanel report={report} />}
                  {activeTab === 'agents' && report && <AgentPanel report={report} />}
                  {activeTab === 'network' && report && <NetworkPanel report={report} />}
                  {activeTab === 'policy' && report && <PolicyPanel report={report} />}
                </motion.div>
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

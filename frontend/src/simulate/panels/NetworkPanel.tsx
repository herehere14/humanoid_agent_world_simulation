import { useMemo } from 'react';
import type { SimulationReport } from '../types';

export function NetworkPanel({ report }: { report: SimulationReport }) {
  const decisions = report.llm_agency?.full_decision_log ?? [];
  const ripples = report.ripple_chains ?? [];

  // Build communication network: who messaged/affected who
  const network = useMemo(() => {
    const edges: Record<string, { count: number; types: Set<string> }> = {};
    const nodeActivity: Record<string, { sent: number; received: number; decisions: number }> = {};

    const ensureNode = (name: string) => {
      if (!nodeActivity[name]) nodeActivity[name] = { sent: 0, received: 0, decisions: 0 };
    };

    // Messages from decisions
    for (const d of decisions) {
      ensureNode(d.agent);
      nodeActivity[d.agent].decisions++;

      for (const recipient of (d.message_recipients ?? [])) {
        ensureNode(recipient);
        const key = `${d.agent}→${recipient}`;
        if (!edges[key]) edges[key] = { count: 0, types: new Set() };
        edges[key].count++;
        edges[key].types.add('message');
        nodeActivity[d.agent].sent++;
        nodeActivity[recipient].received++;
      }
    }

    // Ripple cascades
    for (const r of ripples) {
      ensureNode(r.actor);
      ensureNode(r.target);
      const key = `${r.actor}→${r.target}`;
      if (!edges[key]) edges[key] = { count: 0, types: new Set() };
      edges[key].count++;
      edges[key].types.add('ripple');
    }

    // Sort nodes by total activity
    const sortedNodes = Object.entries(nodeActivity)
      .sort((a, b) => (b[1].sent + b[1].received + b[1].decisions) - (a[1].sent + a[1].received + a[1].decisions));

    // Sort edges by count
    const sortedEdges = Object.entries(edges)
      .sort((a, b) => b[1].count - a[1].count);

    return { nodes: sortedNodes, edges: sortedEdges };
  }, [decisions, ripples]);

  // Most messaged people
  const mostMessaged = network.nodes
    .sort((a, b) => b[1].received - a[1].received)
    .slice(0, 15);

  // Most active senders
  const mostActive = network.nodes
    .sort((a, b) => (b[1].sent + b[1].decisions) - (a[1].sent + a[1].decisions))
    .slice(0, 15);

  return (
    <div className="space-y-8">
      {/* Communication Flow */}
      <section>
        <h3 className="text-lg font-bold mb-4">Communication Network</h3>
        <div className="grid md:grid-cols-2 gap-6">
          {/* Most messaged */}
          <div className="bg-white rounded-xl border border-black/5 p-5">
            <h4 className="text-sm font-semibold text-text-base mb-3">Most Contacted People</h4>
            <div className="space-y-2">
              {mostMessaged.map(([name, data], idx) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="text-[10px] text-text-muted w-4">{idx + 1}</span>
                  <span className="text-sm text-text-base flex-1 truncate">{name}</span>
                  <div className="w-32 h-4 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-sky-400 rounded-full"
                      style={{ width: `${Math.min(100, (data.received / Math.max(1, mostMessaged[0]?.[1]?.received ?? 1)) * 100)}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-text-muted w-8 text-right">{data.received}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Most active */}
          <div className="bg-white rounded-xl border border-black/5 p-5">
            <h4 className="text-sm font-semibold text-text-base mb-3">Most Active Decision-Makers</h4>
            <div className="space-y-2">
              {mostActive.map(([name, data], idx) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="text-[10px] text-text-muted w-4">{idx + 1}</span>
                  <span className="text-sm text-text-base flex-1 truncate">{name}</span>
                  <div className="flex gap-1.5">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-600">
                      {data.decisions} decisions
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-50 text-sky-600">
                      {data.sent} sent
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Top Communication Links */}
      <section>
        <h3 className="text-lg font-bold mb-4">Strongest Connections</h3>
        <div className="bg-white rounded-xl border border-black/5 overflow-hidden">
          <div className="max-h-[400px] overflow-y-auto">
            {network.edges.slice(0, 30).map(([key, data], idx) => {
              const [from, to] = key.split('→');
              const types = [...data.types];
              return (
                <div key={key} className={`flex items-center gap-3 px-4 py-2.5 ${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                  <span className="text-sm font-medium text-text-base w-1/3 truncate text-right">{from}</span>
                  <span className="text-text-muted">→</span>
                  <span className="text-sm text-text-base w-1/3 truncate">{to}</span>
                  <div className="flex gap-1.5 ml-auto">
                    {types.map(t => (
                      <span key={t} className={`text-[10px] px-1.5 py-0.5 rounded ${
                        t === 'message' ? 'bg-sky-50 text-sky-600' : 'bg-amber-50 text-amber-600'
                      }`}>
                        {t}
                      </span>
                    ))}
                    <span className="text-xs font-mono text-text-muted">×{data.count}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Ripple Cascades */}
      {ripples.length > 0 && (
        <section>
          <h3 className="text-lg font-bold mb-4">Ripple Cascades ({ripples.length} events)</h3>
          <div className="bg-white rounded-xl border border-black/5 overflow-hidden">
            <div className="max-h-[400px] overflow-y-auto">
              {ripples.slice(0, 50).map((r, idx) => (
                <div key={idx} className={`flex items-center gap-3 px-4 py-2 text-sm ${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                  <span className="font-medium text-text-base w-1/4 truncate text-right">{r.actor}</span>
                  <span className="text-amber-500">→</span>
                  <span className="text-text-base w-1/4 truncate">{r.target}</span>
                  <span className="text-xs text-text-secondary flex-1 truncate">{r.action}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-text-muted whitespace-nowrap">
                    {r.mechanism}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

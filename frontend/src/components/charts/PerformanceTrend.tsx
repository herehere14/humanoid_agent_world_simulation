import {
  ResponsiveContainer, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';
import type { PerformancePoint } from '@/types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function Tip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const a = payload.find((p: any) => p.dataKey === 'adaptiveScore');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const b = payload.find((p: any) => p.dataKey === 'baseScore');
  const delta = (a?.value ?? 0) - (b?.value ?? 0);
  return (
    <div className="bg-surface border border-border rounded-xl shadow-card-md px-3 py-2 text-xs">
      <div className="space-y-1">
        <div className="flex justify-between gap-5">
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-primary" />Adaptive</span>
          <strong className="text-text-base">{((a?.value ?? 0)*100).toFixed(1)}%</strong>
        </div>
        <div className="flex justify-between gap-5">
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-text-muted" />Base</span>
          <strong className="text-text-secondary">{((b?.value ?? 0)*100).toFixed(1)}%</strong>
        </div>
        <div className="flex justify-between gap-5 pt-1 border-t border-border">
          <span className="text-text-muted">Delta</span>
          <strong className={delta >= 0 ? 'text-success-DEFAULT' : 'text-danger-DEFAULT'}>
            {delta >= 0 ? '+' : ''}{(delta*100).toFixed(1)}%
          </strong>
        </div>
      </div>
    </div>
  );
}

export function PerformanceTrend({ data }: { data: PerformancePoint[] }) {
  if (!data.length) return <div className="flex items-center justify-center h-full text-xs text-text-muted">No data yet</div>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 6, right: 8, left: -24, bottom: 0 }}>
        <defs>
          <linearGradient id="gradA" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0066ff" stopOpacity={0.15} />
            <stop offset="100%" stopColor="#0066ff" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gradB" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#71717a" stopOpacity={0.08} />
            <stop offset="100%" stopColor="#71717a" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.04)" />
        <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#71717a' }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
        <YAxis domain={[0.3, 1]} tickFormatter={(v: number) => `${(v*100).toFixed(0)}%`} tick={{ fontSize: 9, fill: '#71717a' }} tickLine={false} axisLine={false} />
        <Tooltip content={<Tip />} />
        <ReferenceLine y={0.75} stroke="#10b981" strokeDasharray="4 3" strokeOpacity={0.2} />
        <ReferenceLine y={0.5}  stroke="#f59e0b" strokeDasharray="4 3" strokeOpacity={0.15} />
        <Area type="monotone" dataKey="baseScore"     stroke="#71717a" strokeWidth={1.5} fill="url(#gradB)" strokeDasharray="4 3" dot={false} activeDot={{ r:3, fill:'#71717a' }} />
        <Area type="monotone" dataKey="adaptiveScore" stroke="#0066ff" strokeWidth={2}   fill="url(#gradA)"                      dot={false} activeDot={{ r:4, fill:'#0066ff', stroke:'#ffffff', strokeWidth:2 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

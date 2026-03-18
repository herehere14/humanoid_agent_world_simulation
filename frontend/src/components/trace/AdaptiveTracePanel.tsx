import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { BranchSphere } from './BranchSphere';
import { BranchMatrix } from './BranchMatrix';
import { LiveMetrics } from './LiveMetrics';
import { Divider } from '@/components/ui';
import { getStageLabel } from '@/lib/mockData';
import { clsx } from 'clsx';

const STAGE_COLORS: Record<string, string> = {
  routing:    'text-primary',
  executing:  'text-violet-600',
  evaluating: 'text-success',
  optimizing: 'text-warn',
  done:       'text-success',
  idle:       'text-text-muted',
};

export function AdaptiveTracePanel() {
  const { activeTrace, isStreaming } = useAppStore();
  const { stage } = activeTrace;
  const [matrixOpen, setMatrixOpen] = useState(true);
  const [metricsOpen, setMetricsOpen] = useState(true);

  return (
    <div className="flex flex-col h-full bg-surface border-l border-border overflow-hidden">
      {/* Panel title */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border shrink-0">
        <div>
          <h2 className="text-sm font-semibold text-text-base">Adaptive Trace</h2>
          <p className="text-xs text-text-muted mt-0.5">Live internal state</p>
        </div>
        <div className="flex items-center gap-2">
          {isStreaming && (
            <span className="flex items-center gap-1.5 text-xs text-primary">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              Live
            </span>
          )}
          {stage !== 'idle' && (
            <span className={clsx('text-xs font-medium', STAGE_COLORS[stage] ?? 'text-text-muted')}>
              {getStageLabel(stage)}
            </span>
          )}
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto">

        {/* ── Sphere ─────────────────────────────────────────── */}
        <div className="px-4 pt-5 pb-3">
          <SectionLabel>Branch Network</SectionLabel>
          <div className="flex justify-center">
            <BranchSphere />
          </div>
        </div>

        <Divider />

        {/* ── Branch Matrix ───────────────────────────────────── */}
        <section>
          <CollapseHeader
            label="Branch Matrix"
            open={matrixOpen}
            onToggle={() => setMatrixOpen(o => !o)}
            badge={activeTrace.activeBranches.length > 0
              ? `${activeTrace.activeBranches.length} active`
              : undefined}
          />
          <AnimatePresence initial={false}>
            {matrixOpen && (
              <Collapsible>
                <BranchMatrix />
              </Collapsible>
            )}
          </AnimatePresence>
        </section>

        <Divider />

        {/* ── Live Metrics ────────────────────────────────────── */}
        <section>
          <CollapseHeader
            label="Live Metrics"
            open={metricsOpen}
            onToggle={() => setMetricsOpen(o => !o)}
          />
          <AnimatePresence initial={false}>
            {metricsOpen && (
              <Collapsible>
                <LiveMetrics />
              </Collapsible>
            )}
          </AnimatePresence>
        </section>

        <div className="h-8" />
      </div>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted mb-3">
      {children}
    </div>
  );
}

function CollapseHeader({
  label, open, onToggle, badge
}: { label: string; open: boolean; onToggle: () => void; badge?: string }) {
  return (
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between px-5 py-3 hover:bg-elevated transition-colors"
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-text-base">{label}</span>
        {badge && (
          <span className="text-[10px] font-medium bg-primary-light text-primary border border-primary-mid px-1.5 py-0.5 rounded-full">
            {badge}
          </span>
        )}
      </div>
      <motion.div animate={{ rotate: open ? 0 : -90 }} transition={{ duration: 0.18 }}>
        <ChevronDown size={14} className="text-text-muted" />
      </motion.div>
    </button>
  );
}

function Collapsible({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: 'auto', opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
      style={{ overflow: 'hidden' }}
    >
      {children}
    </motion.div>
  );
}

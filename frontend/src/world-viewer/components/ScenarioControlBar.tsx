import { useEffect, useMemo, useState } from 'react';

interface ScenarioControlBarProps {
  loading: boolean;
  currentEventCount: number;
  activeInformation: string[];
  onApply: (information: string[]) => Promise<void>;
  onReset: () => Promise<void>;
  error: string | null;
}

const EXAMPLES = [
  'oil prices surge 100%',
  'major employer announces a 30% layoff',
  'heatwave pushes power demand past grid capacity',
];

export function ScenarioControlBar({
  loading,
  currentEventCount,
  activeInformation,
  onApply,
  onReset,
  error,
}: ScenarioControlBarProps) {
  const [slotCount, setSlotCount] = useState(Math.max(1, activeInformation.length || 1));
  const [entries, setEntries] = useState<string[]>(
    activeInformation.length > 0 ? activeInformation : [''],
  );

  useEffect(() => {
    const nextCount = Math.max(1, activeInformation.length || slotCount);
    setSlotCount(nextCount);
    setEntries(() => {
      const seed = activeInformation.length > 0 ? [...activeInformation] : [''];
      while (seed.length < nextCount) seed.push('');
      return seed.slice(0, nextCount);
    });
  }, [activeInformation]);

  useEffect(() => {
    setEntries(prev => {
      const next = [...prev];
      while (next.length < slotCount) next.push('');
      return next.slice(0, slotCount);
    });
  }, [slotCount]);

  const trimmedEntries = useMemo(
    () => entries.map(item => item.trim()).filter(Boolean),
    [entries],
  );

  return (
    <div className="scenario-bar">
      <div className="scenario-bar-header">
        <div>
          <div className="scenario-bar-title">Inject World Events</div>
          <div className="scenario-bar-meta">
            Current tick events {currentEventCount} · loaded shocks {activeInformation.length}
          </div>
        </div>
        <label className="scenario-slot-count">
          <span>Event slots</span>
          <input
            type="number"
            min={1}
            max={5}
            value={slotCount}
            onChange={(e) => setSlotCount(Math.max(1, Math.min(5, Number(e.target.value) || 1)))}
          />
        </label>
      </div>

      <div className="scenario-input-grid">
        {entries.map((value, index) => (
          <input
            key={index}
            type="text"
            className="scenario-input"
            value={value}
            onChange={(e) => {
              const next = [...entries];
              next[index] = e.target.value;
              setEntries(next);
            }}
            placeholder={EXAMPLES[index % EXAMPLES.length]}
          />
        ))}
      </div>

      <div className="scenario-bar-footer">
        <div className="scenario-examples">
          Examples: {EXAMPLES.join(' · ')}
        </div>
        <div className="scenario-actions">
          <button
            className="scenario-btn scenario-btn-secondary"
            onClick={() => void onReset()}
            disabled={loading}
          >
            Reset World
          </button>
          <button
            className="scenario-btn scenario-btn-primary"
            onClick={() => void onApply(trimmedEntries)}
            disabled={loading || trimmedEntries.length === 0}
          >
            {loading ? 'Rebuilding…' : 'Inject And Rerun'}
          </button>
        </div>
      </div>

      {error && <div className="scenario-error">{error}</div>}
    </div>
  );
}

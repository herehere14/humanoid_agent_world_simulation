/** Event overlay — shows current events and world pulse at top of screen */

import { useMemo } from 'react';
import { useWorldStore } from '../store';

export function EventOverlay() {
  const currentTick = useWorldStore(s => s.currentTick);
  const tickData = useWorldStore(s => s.snapshot?.ticks[s.currentTick] ?? null);

  const worldPulse = useMemo(() => {
    if (!tickData) return null;

    const agents = Object.values(tickData.agent_states);
    const n = agents.length;
    if (n === 0) return null;

    const avgValence = agents.reduce((s, a) => s + a.valence, 0) / n;
    const avgArousal = agents.reduce((s, a) => s + a.arousal, 0) / n;
    const avgEnergy = agents.reduce((s, a) => s + a.energy, 0) / n;
    const avgTension = agents.reduce((s, a) => s + a.tension, 0) / n;

    // Action distribution
    const actionCounts: Record<string, number> = {};
    agents.forEach(a => {
      actionCounts[a.action] = (actionCounts[a.action] || 0) + 1;
    });

    // Most distressed
    const mostDistressed = [...agents]
      .sort((a, b) => b.vulnerability - a.vulnerability)
      .slice(0, 3);

    return { avgValence, avgArousal, avgEnergy, avgTension, actionCounts, mostDistressed, total: n };
  }, [tickData, currentTick]);

  if (!tickData) return null;

  return (
    <>
      {/* Event banner */}
      {tickData.events.length > 0 && (
        <div className="event-banner">
          {tickData.events.map((evt, i) => (
            <div key={i} className="event-item">
              <span className="event-icon">⚡</span>
              <span className="event-desc">{evt.description}</span>
              <span className="event-location">@ {evt.location}</span>
            </div>
          ))}
        </div>
      )}

      {/* World pulse (top-right) */}
      {worldPulse && (
        <div className="world-pulse">
          <div className="pulse-title">Town Pulse</div>
          <div className="pulse-meters">
            <PulseMeter label="Mood" value={worldPulse.avgValence} color="#60a5fa" />
            <PulseMeter label="Energy" value={worldPulse.avgEnergy} color="#22c55e" />
            <PulseMeter label="Arousal" value={worldPulse.avgArousal} color="#f59e0b" />
            <PulseMeter label="Tension" value={worldPulse.avgTension} color="#ef4444" />
          </div>
          <div className="pulse-actions">
            {Object.entries(worldPulse.actionCounts)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 4)
              .map(([action, count]) => (
                <span key={action} className="action-count">
                  {action.replace(/_/g, ' ')} <b>{count}</b>
                </span>
              ))}
          </div>
          {worldPulse.mostDistressed.some(a => a.vulnerability > 0.4) && (
            <div className="pulse-distressed">
              <span className="distressed-label">Most vulnerable:</span>
              {worldPulse.mostDistressed
                .filter(a => a.vulnerability > 0.3)
                .map(a => (
                  <span key={a.id} className="distressed-name">{a.name}</span>
                ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}

function PulseMeter({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="pulse-meter">
      <span className="pulse-meter-label">{label}</span>
      <div className="pulse-meter-track">
        <div
          className="pulse-meter-fill"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
      <span className="pulse-meter-value">{(value * 100).toFixed(0)}</span>
    </div>
  );
}

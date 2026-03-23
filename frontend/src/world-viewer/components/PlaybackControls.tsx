/** Playback controls — timeline, play/pause, speed, event markers */

import { useEffect, useRef, useMemo } from 'react';
import { useWorldStore } from '../store';

export function PlaybackControls() {
  const currentTick = useWorldStore(s => s.currentTick);
  const maxTick = useWorldStore(s => s.maxTick);
  const playing = useWorldStore(s => s.playing);
  const speed = useWorldStore(s => s.speed);
  const snapshot = useWorldStore(s => s.snapshot);
  const setTick = useWorldStore(s => s.setTick);
  const togglePlay = useWorldStore(s => s.togglePlay);
  const setSpeed = useWorldStore(s => s.setSpeed);
  const advanceTick = useWorldStore(s => s.advanceTick);

  const tickData = snapshot?.ticks[currentTick] ?? null;

  // Pre-scan all ticks for event markers + conflict hotspots
  const markers = useMemo(() => {
    if (!snapshot) return [];
    const m: Array<{ tick: number; type: 'event' | 'conflict' | 'collapse'; label: string }> = [];
    for (const t of snapshot.ticks) {
      if (t.events.length > 0) {
        m.push({ tick: t.tick - 1, type: 'event', label: t.events[0].description.slice(0, 40) });
      }
      const conflicts = t.interactions.filter(i => i.type === 'conflict').length;
      if (conflicts >= 3) {
        m.push({ tick: t.tick - 1, type: 'conflict', label: `${conflicts} conflicts` });
      }
      const collapses = Object.values(t.agent_states).filter((a: any) => a.action === 'COLLAPSE').length;
      if (collapses > 0) {
        m.push({ tick: t.tick - 1, type: 'collapse', label: `${collapses} collapsed` });
      }
    }
    return m;
  }, [snapshot]);
  const intervalRef = useRef<number | null>(null);

  // Playback timer
  useEffect(() => {
    if (playing) {
      intervalRef.current = window.setInterval(() => {
        advanceTick();
      }, 1000 / speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, speed, advanceTick]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          togglePlay();
          break;
        case 'ArrowRight':
          setTick(currentTick + 1);
          break;
        case 'ArrowLeft':
          setTick(currentTick - 1);
          break;
        case 'ArrowUp':
          setSpeed(speed + 1);
          break;
        case 'ArrowDown':
          setSpeed(speed - 1);
          break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentTick, speed, togglePlay, setTick, setSpeed]);

  const day = Math.floor(currentTick / 24) + 1;
  const hour = currentTick % 24;
  const timeStr = `Day ${day}, ${String(hour).padStart(2, '0')}:00`;

  const eventCount = tickData?.events.length ?? 0;
  const interactionCount = tickData?.interactions.length ?? 0;

  return (
    <div className="playback-controls">
      {/* Time display */}
      <div className="playback-time">
        <span className="time-display">{timeStr}</span>
        <span className="tick-display">tick {currentTick + 1}/{maxTick + 1}</span>
      </div>

      {/* Transport controls */}
      <div className="playback-transport">
        <button
          className="transport-btn"
          onClick={() => setTick(0)}
          title="Go to start"
        >
          ⏮
        </button>
        <button
          className="transport-btn"
          onClick={() => setTick(currentTick - 1)}
          title="Previous tick"
        >
          ◀
        </button>
        <button
          className="transport-btn transport-play"
          onClick={togglePlay}
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? '⏸' : '▶'}
        </button>
        <button
          className="transport-btn"
          onClick={() => setTick(currentTick + 1)}
          title="Next tick"
        >
          ▶
        </button>
        <button
          className="transport-btn"
          onClick={() => setTick(maxTick)}
          title="Go to end"
        >
          ⏭
        </button>
      </div>

      {/* Speed control */}
      <div className="playback-speed">
        <button
          className="speed-btn"
          onClick={() => setSpeed(speed - 1)}
          disabled={speed <= 0.5}
        >
          −
        </button>
        <span className="speed-display">{speed}x</span>
        <button
          className="speed-btn"
          onClick={() => setSpeed(speed + 1)}
        >
          +
        </button>
      </div>

      {/* Timeline slider */}
      <div className="playback-timeline">
        <input
          type="range"
          min={0}
          max={maxTick}
          value={currentTick}
          onChange={(e) => setTick(Number(e.target.value))}
          className="timeline-slider"
        />
        {/* Day markers */}
        <div className="timeline-markers">
          {Array.from({ length: Math.ceil((maxTick + 1) / 24) }).map((_, i) => (
            <div
              key={i}
              className="timeline-day-marker"
              style={{ left: `${(i * 24 / (maxTick + 1)) * 100}%` }}
            >
              D{i + 1}
            </div>
          ))}
        </div>
        {/* Event/conflict/collapse markers on the timeline */}
        <div className="timeline-events">
          {markers.map((m, i) => (
            <div
              key={`em-${i}`}
              className={`timeline-event-dot timeline-event-${m.type}`}
              style={{ left: `${(m.tick / (maxTick + 1)) * 100}%` }}
              title={m.label}
              onClick={(e) => { e.stopPropagation(); setTick(m.tick); }}
            />
          ))}
        </div>
      </div>

      {/* Status indicators */}
      <div className="playback-status">
        {eventCount > 0 && (
          <span className="status-badge status-event">
            ⚡ {eventCount} event{eventCount > 1 ? 's' : ''}
          </span>
        )}
        {interactionCount > 0 && (
          <span className="status-badge status-interaction">
            💬 {interactionCount} interaction{interactionCount > 1 ? 's' : ''}
          </span>
        )}
      </div>
    </div>
  );
}

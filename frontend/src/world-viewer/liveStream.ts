/** Live simulation stream — connects to the backend SSE endpoint
 *  and feeds expanding tick data into the store in real-time.
 *
 *  Usage:
 *    const stream = connectLiveSimulation('/api/world/stream');
 *    // later:
 *    stream.disconnect();
 *
 *  The backend emits SSE events:
 *    event: tick        — one tick of simulation data
 *    event: locations   — new locations to add
 *    event: agents      — new agents to add
 *    event: done        — simulation complete
 */

import { useWorldStore } from './store';
import type { TickData, LocationMeta, AgentMeta, WorldSnapshot } from './types';

export interface LiveStream {
  disconnect: () => void;
  readonly connected: boolean;
}

export function connectLiveSimulation(url: string): LiveStream {
  const store = useWorldStore.getState();
  let es: EventSource | null = null;
  let connected = true;

  // Ensure a base snapshot exists
  if (!store.snapshot) {
    store.loadSnapshot({
      scenario: 'live',
      total_ticks: 0,
      locations: {},
      agents: {},
      ticks: [],
    });
  }

  es = new EventSource(url);

  es.addEventListener('tick', (e) => {
    try {
      const tick: TickData = JSON.parse(e.data);
      useWorldStore.getState().appendTicks([tick]);

      // Auto-advance to latest tick if we're near the end
      const { currentTick, maxTick, playing } = useWorldStore.getState();
      if (!playing && currentTick >= maxTick - 1) {
        useWorldStore.getState().setTick(maxTick);
      }
    } catch {
      // ignore malformed ticks
    }
  });

  es.addEventListener('locations', (e) => {
    try {
      const locations: Record<string, LocationMeta> = JSON.parse(e.data);
      useWorldStore.getState().addLocations(locations);
    } catch {
      // ignore
    }
  });

  es.addEventListener('agents', (e) => {
    try {
      const agents: Record<string, AgentMeta> = JSON.parse(e.data);
      useWorldStore.getState().addAgents(agents);
    } catch {
      // ignore
    }
  });

  es.addEventListener('done', () => {
    connected = false;
    es?.close();
  });

  es.onerror = () => {
    // SSE will auto-reconnect, but mark disconnected temporarily
    connected = false;
  };

  es.onopen = () => {
    connected = true;
  };

  return {
    disconnect: () => {
      connected = false;
      es?.close();
      es = null;
    },
    get connected() { return connected; },
  };
}

/** Hook-friendly: try connecting to live stream, fall back gracefully */
export async function tryLiveConnection(): Promise<LiveStream | null> {
  try {
    const res = await fetch('/api/world/stream', { method: 'HEAD' });
    if (res.ok || res.status === 405) {
      return connectLiveSimulation('/api/world/stream');
    }
  } catch {
    // endpoint not available
  }
  return null;
}

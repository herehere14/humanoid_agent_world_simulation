/** WorldViewer — main 3D world viewer page */

import { useCallback, useEffect, useMemo, Suspense, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Sky } from '@react-three/drei';
import * as THREE from 'three';
import './world-viewer.css';
import { useWorldStore } from './store';
import { TownEnvironment } from './components/TownEnvironment';
import { AgentLayer } from './components/AgentLayer';
import { InspectorPanel } from './components/InspectorPanel';
import { PlaybackControls } from './components/PlaybackControls';
import { EventOverlay } from './components/EventOverlay';
import { PostProcessing } from './components/PostProcessing';
import { EnvironmentEffects } from './components/EnvironmentEffects';
import { DayNightCycle, useSkyInclination } from './components/DayNightCycle';
import { ScenarioControlBar } from './components/ScenarioControlBar';
import { getLayouts, getTownBounds } from './layout';
import { generateUniverseSeed, getUniverseParams, applyUniverseToSnapshot } from './universe';
import type { WorldSnapshot } from './types';

function Scene() {
  const selectAgent = useWorldStore(s => s.selectAgent);
  const skyInclination = useSkyInclination();

  // Camera and controls adapt to the generated town size
  const layouts = getLayouts();
  const bounds = getTownBounds(layouts);
  const camDist = bounds.radius * 1.4 + 20;

  return (
    <>
      <PerspectiveCamera
        makeDefault
        position={[bounds.centerX, camDist * 0.8, bounds.centerZ + camDist]}
        fov={50}
      />
      <OrbitControls
        maxPolarAngle={Math.PI / 2.2}
        minDistance={3}
        maxDistance={camDist * 3}
        enableDamping
        dampingFactor={0.05}
        target={[bounds.centerX, 0, bounds.centerZ]}
      />
      <Sky
        distance={450000}
        sunPosition={[100, 60, -50]}
        inclination={skyInclination}
        azimuth={0.25}
        rayleigh={0.4}
        turbidity={4}
        mieCoefficient={0.003}
        mieDirectionalG={0.8}
      />
      {/* Dynamic day/night lighting — replaces static lights in TownEnvironment */}
      <DayNightCycle radius={bounds.radius} />
      <TownEnvironment />
      <Suspense fallback={null}>
        <AgentLayer />
        <EnvironmentEffects />
      </Suspense>
      <PostProcessing />
      {/* Click on empty space to deselect */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[bounds.centerX, -0.1, bounds.centerZ]}
        onClick={() => selectAgent(null)}
        visible={false}
      >
        <planeGeometry args={[bounds.radius * 4, bounds.radius * 4]} />
        <meshBasicMaterial />
      </mesh>
    </>
  );
}

function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="loading-spinner" />
        <h2>Loading World Simulation</h2>
        <p>Preparing agent data and town layout...</p>
      </div>
    </div>
  );
}

function ErrorScreen({ error }: { error: string }) {
  return (
    <div className="loading-screen">
      <div className="loading-content">
        <h2 style={{ color: '#dc2626' }}>Failed to Load</h2>
        <p>{error}</p>
        <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>
          Run the snapshot exporter first, or check that mock_snapshot.json exists.
        </p>
      </div>
    </div>
  );
}

export default function WorldViewer() {
  const loading = useWorldStore(s => s.loading);
  const error = useWorldStore(s => s.error);
  const snapshot = useWorldStore(s => s.snapshot);
  const loadSnapshot = useWorldStore(s => s.loadSnapshot);
  const setLoading = useWorldStore(s => s.setLoading);
  const setError = useWorldStore(s => s.setError);
  const currentTick = useWorldStore(s => s.currentTick);
  const tickData = useWorldStore(s => s.snapshot?.ticks[s.currentTick] ?? null);
  const [controlError, setControlError] = useState<string | null>(null);
  const [activeInformation, setActiveInformation] = useState<string[]>([]);

  // Parse universe seed from URL: #/world?seed=12345
  const universeSeed = useMemo(() => {
    const match = window.location.hash.match(/[?&]seed=(\d+)/);
    return match ? Number(match[1]) : generateUniverseSeed();
  }, []);
  const universeParams = useMemo(() => getUniverseParams(universeSeed), [universeSeed]);

  const loadWorldData = useCallback(async (
    options: {
      information?: string[];
      keepCurrentOnError?: boolean;
    } = {},
  ) => {
    const cleanInformation = (options.information ?? []).map(item => item.trim()).filter(Boolean);
    const keepCurrentOnError = options.keepCurrentOnError ?? false;
    const currentSnapshot = useWorldStore.getState().snapshot;

    setLoading(true);
    setControlError(null);
    if (!keepCurrentOnError) {
      setError(null);
    }

    try {
      let data: WorldSnapshot | null = null;

      if (cleanInformation.length > 0) {
        const res = await fetch('/api/world/snapshot/custom', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scenario: currentSnapshot?.generation_meta?.scenario || currentSnapshot?.scenario || 'small_town',
            ticks: currentSnapshot?.generation_meta?.ticks || currentSnapshot?.total_ticks || 240,
            information: cleanInformation,
          }),
        });
        if (!res.ok) {
          const detail = await res.text();
          throw new Error(detail || 'Failed to build custom world snapshot');
        }
        data = await res.json();
      } else {
        try {
          const res = await fetch('/api/world/snapshot');
          if (res.ok) {
            const text = await res.text();
            data = JSON.parse(text);
          }
        } catch {
          // Backend not available
        }

        if (!data) {
          try {
            const res = await fetch('/mock_snapshot.json');
            if (res.ok) {
              const text = await res.text();
              data = JSON.parse(text);
            }
          } catch {
            // Static file not available either
          }
        }
      }

      if (!data) {
        throw new Error('Could not load simulation data from /api/world/snapshot or /mock_snapshot.json');
      }

      applyUniverseToSnapshot(data, universeParams);
      loadSnapshot(data);
      setActiveInformation(data.generation_meta?.external_information || cleanInformation);
      setError(null);
    } catch (err: any) {
      const message = err.message || 'Unknown error loading simulation data';
      if (keepCurrentOnError && currentSnapshot) {
        setControlError(message);
        setLoading(false);
      } else {
        setError(message);
      }
    }
  }, [loadSnapshot, setLoading, setError, universeParams]);

  // Load snapshot data and apply universe divergence
  useEffect(() => {
    void loadWorldData();
  }, [loadWorldData, universeSeed]);

  if (loading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;
  if (!snapshot) return <LoadingScreen />;

  return (
    <div className="world-viewer">
      {/* 3D Canvas */}
      <div className="world-canvas">
        <Canvas
          shadows="soft"
          gl={{
            antialias: false, // handled by SMAA postprocessing
            alpha: false,
            powerPreference: 'high-performance',
            stencil: false,
          }}
          dpr={[1, 2]}
          onCreated={({ gl }) => {
            gl.setClearColor('#e8e0d8');
            gl.shadowMap.type = THREE.PCFSoftShadowMap;
            gl.toneMapping = THREE.NoToneMapping; // postprocessing handles this
          }}
        >
          <Scene />
        </Canvas>
      </div>

      {/* UI Overlays */}
      <EventOverlay />
      <ScenarioControlBar
        loading={loading}
        currentEventCount={tickData?.events.length ?? 0}
        activeInformation={activeInformation}
        onApply={async (information) => {
          await loadWorldData({ information, keepCurrentOnError: true });
        }}
        onReset={async () => {
          await loadWorldData({ information: [], keepCurrentOnError: true });
        }}
        error={controlError}
      />
      <PlaybackControls />
      <InspectorPanel />

      {/* Universe badge */}
      <div className="universe-badge">
        <div className="universe-label">{universeParams.label}</div>
        <div className="universe-meta">
          seed {universeSeed} · chaos {(universeParams.chaosFactor * 100).toFixed(0)}% · {universeParams.weatherMood}
        </div>
        <button
          className="universe-btn"
          onClick={() => {
            const newSeed = generateUniverseSeed();
            window.location.hash = `#/world?seed=${newSeed}`;
            window.location.reload();
          }}
        >
          New Universe
        </button>
      </div>

      {/* Help hint */}
      <div className="help-hint">
        Click agent to inspect · Space to play/pause · Arrow keys to navigate · Each reload = new parallel universe
      </div>
    </div>
  );
}

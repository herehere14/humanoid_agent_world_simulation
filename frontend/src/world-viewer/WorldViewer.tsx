/** WorldViewer — main 3D world viewer page */

import { useEffect, Suspense } from 'react';
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
import { getLayouts, getTownBounds } from './layout';
import type { WorldSnapshot } from './types';

function Scene() {
  const selectAgent = useWorldStore(s => s.selectAgent);

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
        minDistance={5}
        maxDistance={camDist * 3}
        enableDamping
        dampingFactor={0.05}
        target={[bounds.centerX, 0, bounds.centerZ]}
      />
      <Sky
        distance={450000}
        sunPosition={[100, 60, -50]}
        inclination={0.49}
        azimuth={0.25}
        rayleigh={0.4}
        turbidity={4}
        mieCoefficient={0.003}
        mieDirectionalG={0.8}
      />
      <TownEnvironment />
      <Suspense fallback={null}>
        <AgentLayer />
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

  // Load snapshot data
  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        // Prefer the backend snapshot so the viewer reflects live/exported
        // simulation data. Fall back to the bundled mock if the API is not up.
        let data: WorldSnapshot | null = null;

        // 1. Try backend API
        try {
          const res = await fetch('/api/world/snapshot');
          if (res.ok) {
            const text = await res.text();
            data = JSON.parse(text);
          }
        } catch {
          // Backend not available
        }

        // 2. Fall back to bundled mock snapshot
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

        if (data) {
          loadSnapshot(data);
        } else {
          throw new Error('Could not load simulation data from /api/world/snapshot or /mock_snapshot.json');
        }
      } catch (err: any) {
        setError(err.message || 'Unknown error loading simulation data');
      }
    }
    load();
  }, [loadSnapshot, setLoading, setError]);

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
      <PlaybackControls />
      <InspectorPanel />

      {/* Help hint */}
      <div className="help-hint">
        Click agent to inspect · Space to play/pause · Arrow keys to navigate
      </div>
    </div>
  );
}

/**
 * Day/night cycle — shifts sky, lighting, fog, and atmosphere based
 * on the simulated hour of day from the current tick.
 *
 * Hours map to visual phases:
 *   0-5   deep night  — dark blue sky, moonlight, dim ambient
 *   5-7   dawn        — orange/pink horizon, warm directional
 *   7-10  morning     — bright, cool white sun, light fog
 *   10-16 midday      — full brightness, minimal fog
 *   16-18 golden hour — warm amber sun, long shadows
 *   18-20 sunset      — deep orange, purple sky
 *   20-22 dusk        — dim, blue-grey
 *   22-24 night       — same as 0-5
 */

import { useRef, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useWorldStore } from '../store';

interface TimeOfDay {
  sunPosition: [number, number, number];
  sunColor: string;
  sunIntensity: number;
  ambientIntensity: number;
  ambientColor: string;
  fogColor: string;
  fogNear: number;
  fogFar: number;
  skyInclination: number;
  clearColor: string;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpColor(a: string, b: string, t: number): string {
  const ca = new THREE.Color(a);
  const cb = new THREE.Color(b);
  ca.lerp(cb, t);
  return '#' + ca.getHexString();
}

const PHASES: Record<string, TimeOfDay> = {
  night: {
    sunPosition: [0, -20, -50],
    sunColor: '#4466aa',
    sunIntensity: 0.08,
    ambientIntensity: 0.12,
    ambientColor: '#1a1a3e',
    fogColor: '#0a0a1e',
    fogNear: 20,
    fogFar: 80,
    skyInclination: 0.52,
    clearColor: '#080818',
  },
  dawn: {
    sunPosition: [-60, 8, -30],
    sunColor: '#ff8855',
    sunIntensity: 0.5,
    ambientIntensity: 0.25,
    ambientColor: '#553322',
    fogColor: '#443322',
    fogNear: 30,
    fogFar: 100,
    skyInclination: 0.505,
    clearColor: '#2a1a10',
  },
  morning: {
    sunPosition: [-40, 25, -20],
    sunColor: '#fff5e0',
    sunIntensity: 0.9,
    ambientIntensity: 0.35,
    ambientColor: '#8899bb',
    fogColor: '#d8d0c0',
    fogNear: 50,
    fogFar: 150,
    skyInclination: 0.49,
    clearColor: '#d8d0c0',
  },
  midday: {
    sunPosition: [20, 50, 10],
    sunColor: '#ffffff',
    sunIntensity: 1.3,
    ambientIntensity: 0.4,
    ambientColor: '#87ceeb',
    fogColor: '#e8e0d8',
    fogNear: 60,
    fogFar: 180,
    skyInclination: 0.49,
    clearColor: '#e8e0d8',
  },
  golden: {
    sunPosition: [60, 12, 30],
    sunColor: '#ffaa44',
    sunIntensity: 1.0,
    ambientIntensity: 0.3,
    ambientColor: '#aa7744',
    fogColor: '#c8a060',
    fogNear: 40,
    fogFar: 120,
    skyInclination: 0.495,
    clearColor: '#c8a060',
  },
  sunset: {
    sunPosition: [70, 4, 40],
    sunColor: '#ff5522',
    sunIntensity: 0.6,
    ambientIntensity: 0.2,
    ambientColor: '#663322',
    fogColor: '#553020',
    fogNear: 30,
    fogFar: 100,
    skyInclination: 0.505,
    clearColor: '#3a1a0a',
  },
  dusk: {
    sunPosition: [50, -5, 30],
    sunColor: '#5555aa',
    sunIntensity: 0.15,
    ambientIntensity: 0.15,
    ambientColor: '#2a2a4e',
    fogColor: '#1a1a2e',
    fogNear: 25,
    fogFar: 90,
    skyInclination: 0.515,
    clearColor: '#121228',
  },
};

function getPhaseAndBlend(hour: number): { from: TimeOfDay; to: TimeOfDay; t: number } {
  if (hour < 5)  return { from: PHASES.night,   to: PHASES.night,   t: 0 };
  if (hour < 6)  return { from: PHASES.night,   to: PHASES.dawn,    t: hour - 5 };
  if (hour < 7)  return { from: PHASES.dawn,    to: PHASES.morning, t: hour - 6 };
  if (hour < 10) return { from: PHASES.morning, to: PHASES.midday,  t: (hour - 7) / 3 };
  if (hour < 16) return { from: PHASES.midday,  to: PHASES.midday,  t: 0 };
  if (hour < 17) return { from: PHASES.midday,  to: PHASES.golden,  t: hour - 16 };
  if (hour < 18) return { from: PHASES.golden,  to: PHASES.sunset,  t: hour - 17 };
  if (hour < 20) return { from: PHASES.sunset,  to: PHASES.dusk,    t: (hour - 18) / 2 };
  if (hour < 22) return { from: PHASES.dusk,    to: PHASES.night,   t: (hour - 20) / 2 };
  return { from: PHASES.night, to: PHASES.night, t: 0 };
}

export function DayNightCycle({ radius }: { radius: number }) {
  const currentTick = useWorldStore(s => s.currentTick);
  const hour = currentTick % 24;

  const sunRef = useRef<THREE.DirectionalLight>(null);
  const ambientRef = useRef<THREE.AmbientLight>(null);
  const { gl, scene } = useThree();

  // Interpolate between phases
  const state = useMemo(() => {
    const { from, to, t } = getPhaseAndBlend(hour);
    return {
      sunPosition: [
        lerp(from.sunPosition[0], to.sunPosition[0], t),
        lerp(from.sunPosition[1], to.sunPosition[1], t),
        lerp(from.sunPosition[2], to.sunPosition[2], t),
      ] as [number, number, number],
      sunColor: lerpColor(from.sunColor, to.sunColor, t),
      sunIntensity: lerp(from.sunIntensity, to.sunIntensity, t),
      ambientIntensity: lerp(from.ambientIntensity, to.ambientIntensity, t),
      ambientColor: lerpColor(from.ambientColor, to.ambientColor, t),
      fogColor: lerpColor(from.fogColor, to.fogColor, t),
      fogNear: lerp(from.fogNear, to.fogNear, t),
      fogFar: lerp(from.fogFar, to.fogFar, t),
      skyInclination: lerp(from.skyInclination, to.skyInclination, t),
      clearColor: lerpColor(from.clearColor, to.clearColor, t),
    };
  }, [hour]);

  // Apply to scene
  useFrame(() => {
    gl.setClearColor(state.clearColor);
    if (scene.fog && scene.fog instanceof THREE.Fog) {
      scene.fog.color.set(state.fogColor);
      scene.fog.near = state.fogNear * (radius / 50);
      scene.fog.far = state.fogFar * (radius / 50);
    }
    if (sunRef.current) {
      sunRef.current.position.set(...state.sunPosition);
      sunRef.current.color.set(state.sunColor);
      sunRef.current.intensity = state.sunIntensity;
    }
    if (ambientRef.current) {
      ambientRef.current.color.set(state.ambientColor);
      ambientRef.current.intensity = state.ambientIntensity;
    }
  });

  const isNight = hour >= 20 || hour < 6;

  return (
    <>
      <directionalLight
        ref={sunRef}
        position={state.sunPosition}
        intensity={state.sunIntensity}
        color={state.sunColor}
        castShadow
        shadow-mapSize-width={4096}
        shadow-mapSize-height={4096}
        shadow-camera-far={radius * 3.5}
        shadow-camera-left={-radius * 1.8}
        shadow-camera-right={radius * 1.8}
        shadow-camera-top={radius * 1.8}
        shadow-camera-bottom={-radius * 1.8}
        shadow-bias={-0.0003}
      />
      <ambientLight
        ref={ambientRef}
        intensity={state.ambientIntensity}
        color={state.ambientColor}
      />

      {/* Moon during night */}
      {isNight && (
        <mesh position={[-30, 35, -40]}>
          <sphereGeometry args={[2, 16, 12]} />
          <meshBasicMaterial color="#e8e0c8" />
          <pointLight color="#8888cc" intensity={0.2} distance={80} />
        </mesh>
      )}

      {/* Stars during night */}
      {isNight && <Stars />}
    </>
  );
}

function Stars() {
  const positions = useMemo(() => {
    const arr = new Float32Array(200 * 3);
    for (let i = 0; i < 200; i++) {
      // hemisphere above
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI * 0.45; // upper hemisphere
      const r = 80 + Math.random() * 20;
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.cos(phi);
      arr[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    return arr;
  }, []);

  const ref = useRef<THREE.Points>(null);
  useFrame(({ clock }) => {
    if (ref.current) {
      // twinkle
      const mat = ref.current.material as THREE.PointsMaterial;
      mat.opacity = 0.6 + Math.sin(clock.elapsedTime * 0.5) * 0.2;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={200} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial color="#ffffff" size={0.3} transparent opacity={0.7} sizeAttenuation={false} />
    </points>
  );
}

/** Get sky inclination for the Sky component based on current tick */
export function useSkyInclination(): number {
  const currentTick = useWorldStore(s => s.currentTick);
  const hour = currentTick % 24;
  const { from, to, t } = getPhaseAndBlend(hour);
  return lerp(from.skyInclination, to.skyInclination, t);
}

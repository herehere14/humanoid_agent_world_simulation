/**
 * Environment effects layer — the world reacts to what agents do.
 *
 * Reads the current tick's agent states and interactions to produce:
 *  - Damage marks on buildings where conflicts happen
 *  - Debris / scattered objects near fight locations
 *  - Fire / smoke at extreme event locations
 *  - Dark atmosphere zones around collapsed agents
 *  - Celebration confetti near celebrating agents
 *  - Tension haze when average town arousal is high
 *  - Trampled ground near large gatherings
 *  - Graffiti / protest signs at locations with repeated conflict
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useWorldStore } from '../store';
import { getAgentPosition, getLocationLayout } from '../layout';

interface LocationImpact {
  locationId: string;
  conflictIntensity: number;   // 0-1 how much fighting happened here
  collapseCount: number;       // agents collapsed here
  crowdSize: number;           // total agents present
  celebrationLevel: number;    // 0-1
  avgArousal: number;
  avgValence: number;
  position: [number, number, number];
}

function computeLocationImpacts(tickData: any): LocationImpact[] {
  if (!tickData) return [];

  const agents = Object.values(tickData.agent_states) as any[];
  const byLoc = new Map<string, any[]>();
  for (const a of agents) {
    if (!byLoc.has(a.location)) byLoc.set(a.location, []);
    byLoc.get(a.location)!.push(a);
  }

  // count conflicts per location
  const conflictsAt = new Map<string, number>();
  for (const ix of tickData.interactions) {
    if (ix.type === 'conflict') {
      conflictsAt.set(ix.location, (conflictsAt.get(ix.location) ?? 0) + 1);
    }
  }

  const impacts: LocationImpact[] = [];
  for (const [locId, locAgents] of byLoc) {
    const layout = getLocationLayout(locId);
    if (!layout) continue;

    const n = locAgents.length;
    const conflicts = conflictsAt.get(locId) ?? 0;
    const collapses = locAgents.filter((a: any) => a.action === 'COLLAPSE').length;
    const celebs = locAgents.filter((a: any) => a.action === 'CELEBRATE').length;
    const avgArousal = locAgents.reduce((s: number, a: any) => s + a.arousal, 0) / n;
    const avgValence = locAgents.reduce((s: number, a: any) => s + a.valence, 0) / n;

    impacts.push({
      locationId: locId,
      conflictIntensity: Math.min(1, conflicts * 0.4),
      collapseCount: collapses,
      crowdSize: n,
      celebrationLevel: Math.min(1, celebs * 0.3),
      avgArousal,
      avgValence,
      position: layout.position as [number, number, number],
    });
  }

  return impacts;
}

export function EnvironmentEffects() {
  const currentTick = useWorldStore(s => s.currentTick);
  const tickData = useWorldStore(s => s.snapshot?.ticks[s.currentTick] ?? null);

  const impacts = useMemo(() => computeLocationImpacts(tickData), [tickData, currentTick]);

  // Accumulate damage over time
  const damageMap = useRef(new Map<string, number>());
  useMemo(() => {
    for (const imp of impacts) {
      const prev = damageMap.current.get(imp.locationId) ?? 0;
      // damage accumulates from conflicts, decays slowly
      const newDamage = Math.min(1, prev * 0.98 + imp.conflictIntensity * 0.15);
      damageMap.current.set(imp.locationId, newDamage);
    }
  }, [impacts]);

  // Town-wide atmosphere
  const townMood = useMemo(() => {
    if (!tickData) return { avgArousal: 0.2, avgValence: 0.5, conflictCount: 0 };
    const agents = Object.values(tickData.agent_states) as any[];
    const n = agents.length || 1;
    return {
      avgArousal: agents.reduce((s, a: any) => s + a.arousal, 0) / n,
      avgValence: agents.reduce((s, a: any) => s + a.valence, 0) / n,
      conflictCount: tickData.interactions.filter((i: any) => i.type === 'conflict').length,
    };
  }, [tickData, currentTick]);

  return (
    <>
      {/* Per-location effects */}
      {impacts.map(imp => (
        <LocationEffects
          key={imp.locationId}
          impact={imp}
          damage={damageMap.current.get(imp.locationId) ?? 0}
        />
      ))}

      {/* Town-wide tension haze */}
      {townMood.avgArousal > 0.35 && (
        <TensionHaze intensity={townMood.avgArousal} valence={townMood.avgValence} />
      )}

      {/* Event-driven atmospheric shift */}
      {tickData && tickData.events.length > 0 && (
        <EventAtmosphere events={tickData.events} />
      )}
    </>
  );
}

// ── per-location VFX ────────────────────────────────────────
function LocationEffects({ impact, damage }: { impact: LocationImpact; damage: number }) {
  const [px, , pz] = impact.position;

  return (
    <group position={[px, 0, pz]}>
      {/* Debris from fights */}
      {damage > 0.1 && (
        <DebrisField intensity={damage} />
      )}

      {/* Fire/smoke for extreme damage */}
      {damage > 0.5 && (
        <FireEffect intensity={damage} />
      )}

      {/* Dark aura around collapsed agents */}
      {impact.collapseCount > 0 && (
        <DarkAura count={impact.collapseCount} />
      )}

      {/* Celebration confetti */}
      {impact.celebrationLevel > 0.1 && (
        <ConfettiEffect intensity={impact.celebrationLevel} />
      )}

      {/* Crowd dust when many agents gather */}
      {impact.crowdSize > 8 && (
        <CrowdDust count={impact.crowdSize} />
      )}

      {/* Scorch marks on ground from heavy conflict */}
      {damage > 0.3 && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
          <circleGeometry args={[2 + damage * 3, 12]} />
          <meshStandardMaterial
            color="#2d1f0e"
            roughness={1}
            transparent
            opacity={damage * 0.4}
          />
        </mesh>
      )}

      {/* Conflict location red tint on ground */}
      {impact.conflictIntensity > 0 && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
          <circleGeometry args={[3, 16]} />
          <meshBasicMaterial
            color="#ef4444"
            transparent
            opacity={impact.conflictIntensity * 0.12}
          />
        </mesh>
      )}
    </group>
  );
}

// ── debris ──────────────────────────────────────────────────
function DebrisField({ intensity }: { intensity: number }) {
  const pieces = useMemo(() => {
    const count = Math.floor(intensity * 12) + 2;
    return Array.from({ length: count }).map((_, i) => ({
      x: (Math.sin(i * 2.7) * 2 + Math.cos(i * 1.3) * 1.5) * intensity,
      z: (Math.cos(i * 3.1) * 2 + Math.sin(i * 1.7) * 1.5) * intensity,
      rot: i * 0.8,
      scale: 0.08 + (i % 3) * 0.06,
      type: i % 4, // 0=cube, 1=slab, 2=cylinder, 3=wedge
    }));
  }, [intensity]);

  return (
    <>
      {pieces.map((p, i) => (
        <mesh
          key={i}
          position={[p.x, p.scale * 0.5, p.z]}
          rotation={[p.rot, p.rot * 0.5, p.rot * 0.3]}
          castShadow
        >
          {p.type === 0 && <boxGeometry args={[p.scale * 2, p.scale, p.scale * 1.5]} />}
          {p.type === 1 && <boxGeometry args={[p.scale * 3, p.scale * 0.3, p.scale * 2]} />}
          {p.type === 2 && <cylinderGeometry args={[p.scale * 0.5, p.scale * 0.5, p.scale * 2, 5]} />}
          {p.type === 3 && <boxGeometry args={[p.scale * 1.5, p.scale * 1.2, p.scale]} />}
          <meshStandardMaterial
            color={i % 2 === 0 ? '#8c7a6b' : '#6b6b6b'}
            roughness={0.9}
          />
        </mesh>
      ))}
    </>
  );
}

// ── fire/smoke ──────────────────────────────────────────────
function FireEffect({ intensity }: { intensity: number }) {
  const fireRef = useRef<THREE.Group>(null);
  const particleCount = Math.floor(intensity * 8) + 3;

  useFrame(() => {
    if (!fireRef.current) return;
    const t = performance.now() * 0.001;
    fireRef.current.children.forEach((c, i) => {
      const speed = 1.5 + (i % 3) * 0.5;
      const life = (t * speed + i * 0.4) % 2;
      c.position.y = life * 1.5;
      c.position.x = Math.sin(t * 3 + i * 1.1) * 0.3 * intensity;
      c.position.z = Math.cos(t * 2.5 + i * 0.9) * 0.3 * intensity;
      const fade = life < 1 ? life : 2 - life;
      const s = (0.2 + fade * 0.5) * intensity;
      c.scale.setScalar(s);
      const mat = (c as THREE.Mesh).material as THREE.MeshBasicMaterial;
      mat.opacity = fade * 0.7;
      // colour shifts from orange to grey (smoke) as it rises
      if (life > 1) {
        mat.color.setStyle('#555');
      } else {
        mat.color.setStyle(life < 0.5 ? '#ff6b00' : '#ff9500');
      }
    });
  });

  return (
    <group ref={fireRef}>
      {Array.from({ length: particleCount }).map(i => (
        <mesh key={`fire-${i}`}>
          <sphereGeometry args={[0.15, 6, 5]} />
          <meshBasicMaterial color="#ff6b00" transparent opacity={0.6} />
        </mesh>
      ))}
    </group>
  );
}

// ── dark aura ───────────────────────────────────────────────
function DarkAura({ count }: { count: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => {
    if (!ref.current) return;
    const t = performance.now() * 0.001;
    const scale = 2 + Math.sin(t * 0.8) * 0.3;
    ref.current.scale.set(scale, 1, scale);
    (ref.current.material as THREE.MeshBasicMaterial).opacity =
      0.08 + Math.sin(t * 1.2) * 0.03 * count;
  });

  return (
    <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
      <circleGeometry args={[2, 24]} />
      <meshBasicMaterial color="#1a1a2e" transparent opacity={0.1} side={THREE.DoubleSide} />
    </mesh>
  );
}

// ── confetti ────────────────────────────────────────────────
function ConfettiEffect({ intensity }: { intensity: number }) {
  const count = Math.floor(intensity * 20) + 5;
  const ref = useRef<THREE.Group>(null);

  useFrame(() => {
    if (!ref.current) return;
    const t = performance.now() * 0.001;
    ref.current.children.forEach((c, i) => {
      const fall = (t * 0.8 + i * 0.3) % 3;
      c.position.y = 3 - fall;
      c.position.x = Math.sin(t + i * 1.7) * 2;
      c.position.z = Math.cos(t * 0.7 + i * 2.1) * 2;
      c.rotation.x = t * 3 + i;
      c.rotation.z = t * 2 + i * 0.5;
      const mat = (c as THREE.Mesh).material as THREE.MeshBasicMaterial;
      mat.opacity = fall < 2.5 ? 0.8 : (3 - fall) * 3.2;
    });
  });

  const colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ec4899'];

  return (
    <group ref={ref}>
      {Array.from({ length: count }).map((_, i) => (
        <mesh key={i}>
          <planeGeometry args={[0.08, 0.12]} />
          <meshBasicMaterial
            color={colors[i % colors.length]}
            transparent
            opacity={0.8}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

// ── crowd dust ──────────────────────────────────────────────
function CrowdDust({ count }: { count: number }) {
  const ref = useRef<THREE.Points>(null);
  const n = Math.min(count, 25);
  const positions = useMemo(() => {
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 6;
      arr[i * 3 + 1] = Math.random() * 0.5;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 6;
    }
    return arr;
  }, [n]);

  useFrame(() => {
    if (!ref.current) return;
    const pos = ref.current.geometry.attributes.position.array as Float32Array;
    const t = performance.now() * 0.0002;
    for (let i = 0; i < n; i++) {
      pos[i * 3 + 1] = Math.abs(Math.sin(t + i * 0.5)) * 0.4;
      pos[i * 3] += Math.sin(t * 3 + i) * 0.003;
    }
    ref.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={n} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial color="#c2b280" size={0.12} transparent opacity={0.3} sizeAttenuation />
    </points>
  );
}

// ── town-wide tension haze ──────────────────────────────────
function TensionHaze({ intensity, valence }: { intensity: number; valence: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const haze = Math.max(0, intensity - 0.3) * 2; // only visible at high arousal

  useFrame(() => {
    if (!ref.current) return;
    const t = performance.now() * 0.0005;
    const mat = ref.current.material as THREE.MeshBasicMaterial;
    mat.opacity = haze * 0.06 * (1 + Math.sin(t) * 0.3);
  });

  const color = valence < 0.35 ? '#4a0000' : valence < 0.5 ? '#4a3000' : '#3a3a00';

  return (
    <mesh ref={ref} position={[0, 8, 0]}>
      <sphereGeometry args={[80, 12, 8]} />
      <meshBasicMaterial color={color} transparent opacity={0.03} side={THREE.BackSide} />
    </mesh>
  );
}

// ── event atmosphere ────────────────────────────────────────
function EventAtmosphere({ events }: { events: any[] }) {
  // Dramatic events tint the sky
  const hasLayoff = events.some((e: any) => /layoff|fired|eliminated/i.test(e.description));
  const hasCommunity = events.some((e: any) => /community|support|gathering/i.test(e.description));
  const hasConfrontation = events.some((e: any) => /confront|anger|critic/i.test(e.description));

  if (!hasLayoff && !hasCommunity && !hasConfrontation) return null;

  const color = hasLayoff ? '#1a1a3e' : hasConfrontation ? '#3e1a1a' : '#1a3e1a';
  const opacity = hasLayoff ? 0.04 : 0.03;

  return (
    <mesh position={[0, 12, 0]}>
      <sphereGeometry args={[90, 8, 6]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} side={THREE.BackSide} />
    </mesh>
  );
}

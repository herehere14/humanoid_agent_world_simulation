/** Agent layer — positions all agents, draws interaction arcs and VFX */

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useWorldStore } from '../store';
import { getAgentPosition } from '../layout';
import { AgentAvatar } from './AgentAvatar';
import type { AgentState } from '../types';

export function AgentLayer() {
  const snapshot    = useWorldStore(s => s.snapshot);
  const currentTick = useWorldStore(s => s.currentTick);
  const tickData    = useWorldStore(s => s.snapshot?.ticks[s.currentTick] ?? null);

  const agentPositions = useMemo(() => {
    if (!tickData) return new Map<string, [number, number, number]>();
    const byLoc = new Map<string, string[]>();
    for (const [id, state] of Object.entries(tickData.agent_states)) {
      if (!byLoc.has(state.location)) byLoc.set(state.location, []);
      byLoc.get(state.location)!.push(id);
    }
    const positions = new Map<string, [number, number, number]>();
    for (const [loc, ids] of byLoc) {
      ids.forEach((id, i) => positions.set(id, getAgentPosition(loc, i, ids.length)));
    }
    return positions;
  }, [tickData, currentTick]);

  const interactionMap = useMemo(() => {
    if (!tickData) return new Map<string, { partner: string; type: string }>();
    const map = new Map<string, { partner: string; type: string }>();
    for (const ix of tickData.interactions) {
      map.set(ix.agent_a, { partner: ix.agent_b, type: ix.type });
      map.set(ix.agent_b, { partner: ix.agent_a, type: ix.type });
    }
    return map;
  }, [tickData, currentTick]);

  if (!tickData || !snapshot) return null;

  const agents = Object.values(tickData.agent_states);

  return (
    <>
      {agents.map((agent: AgentState) => {
        const pos = agentPositions.get(agent.id) ?? [0, 0, 0] as [number, number, number];
        const ix = interactionMap.get(agent.id);
        const pPos = ix ? (agentPositions.get(ix.partner) ?? null) : null;
        return (
          <AgentAvatar
            key={agent.id}
            agent={agent}
            targetPosition={pos}
            interactionPartner={ix?.partner ?? null}
            interactionType={ix?.type ?? null}
            partnerPosition={pPos}
          />
        );
      })}

      {/* Interaction arcs */}
      {tickData.interactions.map((ix, i) => {
        const posA = agentPositions.get(ix.agent_a);
        const posB = agentPositions.get(ix.agent_b);
        if (!posA || !posB) return null;
        return (
          <InteractionArc
            key={`arc-${i}-${currentTick}`}
            from={posA}
            to={posB}
            type={ix.type}
          />
        );
      })}
    </>
  );
}

// ─── interaction arc with animated particles ─────────────────
function InteractionArc({
  from, to, type,
}: {
  from: [number, number, number];
  to: [number, number, number];
  type: string;
}) {
  const groupRef = useRef<THREE.Group>(null);

  const color = type === 'conflict' ? '#ef4444' :
    type === 'support' ? '#22c55e' :
    type === 'positive' ? '#f59e0b' : '#94a3b8';

  // build a quadratic bezier curve
  const curve = useMemo(() => {
    const a = new THREE.Vector3(from[0], from[1] + 1.2, from[2]);
    const b = new THREE.Vector3(to[0],   to[1] + 1.2,   to[2]);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    mid.y += 1.5 + a.distanceTo(b) * 0.12;
    return new THREE.QuadraticBezierCurve3(a, mid, b);
  }, [from, to]);

  const tubeGeo = useMemo(() => {
    return new THREE.TubeGeometry(curve, 20, type === 'conflict' ? 0.06 : 0.035, 6, false);
  }, [curve, type]);

  // animate particles along the arc
  const particleCount = type === 'conflict' ? 6 : 3;
  const particleRefs = useRef<(THREE.Mesh | null)[]>(new Array(particleCount).fill(null));

  useFrame(() => {
    const t = performance.now() * 0.001;
    for (let i = 0; i < particleCount; i++) {
      const mesh = particleRefs.current[i];
      if (!mesh) continue;
      const progress = ((t * (type === 'conflict' ? 1.5 : 0.8) + i / particleCount) % 1);
      const pos = curve.getPointAt(progress);
      mesh.position.copy(pos);
      const scale = 0.6 + Math.sin(t * 6 + i) * 0.3;
      mesh.scale.setScalar(scale);
    }
  });

  return (
    <group ref={groupRef}>
      {/* arc tube */}
      <mesh geometry={tubeGeo}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={type === 'conflict' ? 0.35 : 0.2}
        />
      </mesh>
      {/* travelling particles */}
      {Array.from({ length: particleCount }).map((_, i) => (
        <mesh
          key={i}
          ref={el => { particleRefs.current[i] = el; }}
        >
          <sphereGeometry args={[type === 'conflict' ? 0.08 : 0.05, 6, 6]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}
      {/* conflict: ground shockwave ring at midpoint */}
      {type === 'conflict' && (
        <ShockwaveRing
          position={[
            (from[0] + to[0]) / 2,
            0.05,
            (from[2] + to[2]) / 2,
          ]}
        />
      )}
    </group>
  );
}

function ShockwaveRing({ position }: { position: [number, number, number] }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => {
    if (!ref.current) return;
    const t = (performance.now() * 0.001 * 2) % 1;
    const scale = 0.5 + t * 2;
    ref.current.scale.set(scale, scale, 1);
    (ref.current.material as THREE.MeshBasicMaterial).opacity = (1 - t) * 0.3;
  });
  return (
    <mesh ref={ref} position={position} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.4, 0.6, 24]} />
      <meshBasicMaterial color="#ef4444" transparent opacity={0.3} side={THREE.DoubleSide} />
    </mesh>
  );
}

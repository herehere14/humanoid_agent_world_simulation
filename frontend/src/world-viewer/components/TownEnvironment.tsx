/**
 * Stylized 3D town environment — Fortnite-tier visual quality.
 *
 * Key techniques:
 *  - Chunky stylized geometry with beveled edges
 *  - Warm saturated color palette
 *  - Animated foliage (wind sway on trees/bushes)
 *  - Detailed building facades (awnings, signs, planters, AC units)
 *  - Grass patches with vertex displacement
 *  - Animated water with wave distortion
 *  - Volumetric-style clouds (billboard sprites)
 *  - Street furniture (lamps, benches, bins, signs)
 *  - Floating dust/pollen particles
 *  - Hemisphere + directional lighting with color temperature
 */

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, RoundedBox, Sparkles } from '@react-three/drei';
import * as THREE from 'three';
import {
  getLayouts,
  getTownBounds,
  generateTreePositions,
  generateRoads,
  classifyArchetype,
  type BuildingArchetype,
} from '../layout';
import { useWorldStore } from '../store';
import type { LocationLayout } from '../types';

/** Returns how strongly windows should glow (0 = day, 1 = deep night) */
function useNightGlow(): number {
  const tick = useWorldStore(s => s.currentTick);
  const h = tick % 24;
  if (h >= 7 && h < 18) return 0;      // day
  if (h >= 20 || h < 5) return 1;       // night
  if (h >= 18) return (h - 18) / 2;     // sunset transition
  return 1 - (h - 5) / 2;               // dawn transition
}

// ── colour palette ──────────────────────────────────────────
const PAL = {
  ground:    '#d4c9a8',
  road:      '#8c8c8c',
  sidewalk:  '#c8bfaa',
  grass:     '#5cb85c',
  grassDark: '#3d8b3d',
  water:     '#4db8e8',
  waterDeep: '#2980b9',
  wood:      '#8B5E3C',
  woodDark:  '#5C3A1E',
  metal:     '#6b7280',
  skin:      '#e8b88a',
  sky:       '#87ceeb',
  cloud:     '#f8f8ff',
  lamp:      '#fef3c7',
  concrete:  '#b0a99a',
};

// ════════════════════════════════════════════════════════════
//  BUILDING
// ════════════════════════════════════════════════════════════

function Building({ id, position, size, color, label }: LocationLayout) {
  const [sx, sy, sz] = size;
  const [px, , pz] = position;
  const isFlat = sy < 1;
  const archetype = classifyArchetype(id, label, '');

  return (
    <group position={[px, 0, pz]}>
      {isFlat ? (
        <ParkArea size={size} />
      ) : (
        <StylisedBuilding size={size} color={color} archetype={archetype} />
      )}
      {/* floating label */}
      <Text
        position={[0, isFlat ? 3.5 : sy + 2, 0]}
        fontSize={Math.min(1.6, Math.max(0.9, sx * 0.1))}
        color="#1e293b"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.08}
        outlineColor="#ffffff"
        maxWidth={sx * 1.5}
      >
        {label}
      </Text>
    </group>
  );
}

// ── stylised building ───────────────────────────────────────
function StylisedBuilding({ size, color, archetype }: {
  size: [number, number, number]; color: string; archetype: BuildingArchetype;
}) {
  const [sx, sy, sz] = size;
  const roofCol   = new THREE.Color(color).multiplyScalar(0.72).getStyle();
  const trimCol   = new THREE.Color(color).multiplyScalar(1.15).getStyle();
  const darkCol   = new THREE.Color(color).multiplyScalar(0.55).getStyle();

  const nightGlow = useNightGlow();
  const winCols = Math.max(1, Math.floor(sx / 2.8));
  const winRows = Math.max(1, Math.floor(sy / 3.5));
  const baseGlow = archetype === 'bar' || archetype === 'food' ? 0.4 : 0.15;
  const winGlow = baseGlow + nightGlow * 0.6; // windows glow much more at night
  const winCol  = archetype === 'bar' ? '#fde68a' : archetype === 'food' ? '#fed7aa'
    : nightGlow > 0.5 ? '#ffe4b5' : '#d4e8ff'; // warm glow at night vs cool blue by day

  return (
    <group position={[0, sy / 2, 0]}>
      {/* MAIN BODY — rounded for stylised look */}
      <RoundedBox args={[sx, sy, sz]} radius={0.4} smoothness={4} castShadow receiveShadow>
        <meshStandardMaterial color={color} roughness={0.65} metalness={0.02} />
      </RoundedBox>

      {/* FOUNDATION STRIP */}
      <mesh position={[0, -sy / 2 + 0.15, 0]} castShadow>
        <boxGeometry args={[sx + 0.3, 0.3, sz + 0.3]} />
        <meshStandardMaterial color={darkCol} roughness={0.8} />
      </mesh>

      {/* ROOF — varies by archetype */}
      {archetype === 'worship' ? (
        <group position={[0, sy / 2, 0]}>
          <mesh castShadow>
            <coneGeometry args={[sx * 0.35, 4, 4]} />
            <meshStandardMaterial color={roofCol} roughness={0.55} />
          </mesh>
          {/* cross/spire */}
          <mesh position={[0, 2.5, 0]} castShadow>
            <cylinderGeometry args={[0.06, 0.06, 1.5, 6]} />
            <meshStandardMaterial color="#d4af37" roughness={0.3} metalness={0.7} />
          </mesh>
        </group>
      ) : archetype === 'factory' ? (
        <group position={[0, sy / 2, 0]}>
          {/* sawtooth roof */}
          {Array.from({ length: Math.max(1, Math.floor(sz / 4)) }).map((_, i) => (
            <mesh key={i} position={[0, 0.3, -sz / 2 + 2 + i * 4]} castShadow>
              <boxGeometry args={[sx + 0.2, 0.6, 3.5]} />
              <meshStandardMaterial color={roofCol} roughness={0.75} />
            </mesh>
          ))}
          {/* chimneys */}
          {[0.3, -0.3].map((xf, i) => (
            <mesh key={`chim-${i}`} position={[sx * xf, 2, sz * 0.2 * (i === 0 ? 1 : -1)]} castShadow>
              <cylinderGeometry args={[0.35, 0.45, 4, 8]} />
              <meshStandardMaterial color="#57534e" roughness={0.8} />
            </mesh>
          ))}
        </group>
      ) : (
        <group position={[0, sy / 2, 0]}>
          {/* flat roof with parapet */}
          <mesh castShadow>
            <boxGeometry args={[sx + 0.5, 0.35, sz + 0.5]} />
            <meshStandardMaterial color={roofCol} roughness={0.6} />
          </mesh>
          {/* parapet walls */}
          {[[-1, 0], [1, 0], [0, -1], [0, 1]].map(([dx, dz], i) => {
            const isX = dx !== 0;
            return (
              <mesh key={`par-${i}`} position={[dx * (sx / 2 + 0.1), 0.4, dz * (sz / 2 + 0.1)]}>
                <boxGeometry args={[isX ? 0.15 : sx + 0.5, 0.5, isX ? sz + 0.5 : 0.15]} />
                <meshStandardMaterial color={roofCol} roughness={0.7} />
              </mesh>
            );
          })}
          {/* rooftop AC unit */}
          <mesh position={[sx * 0.25, 0.65, -sz * 0.2]} castShadow>
            <boxGeometry args={[1.2, 0.8, 0.8]} />
            <meshStandardMaterial color={PAL.metal} roughness={0.5} metalness={0.3} />
          </mesh>
        </group>
      )}

      {/* DOOR */}
      <group position={[0, -sy / 2 + 1.3, sz / 2 + 0.02]}>
        <mesh>
          <planeGeometry args={[1.6, 2.6]} />
          <meshStandardMaterial color={PAL.woodDark} roughness={0.85} />
        </mesh>
        {/* door frame */}
        <mesh position={[0, 0, -0.01]}>
          <planeGeometry args={[1.9, 2.9]} />
          <meshStandardMaterial color={trimCol} roughness={0.7} />
        </mesh>
        {/* handle */}
        <mesh position={[0.55, -0.2, 0.02]}>
          <sphereGeometry args={[0.05, 6, 6]} />
          <meshStandardMaterial color="#d4af37" roughness={0.3} metalness={0.7} />
        </mesh>
      </group>

      {/* WINDOWS — both faces */}
      {['front', 'side'].map(face => {
        const isFront = face === 'front';
        const cols = isFront ? winCols : Math.max(1, Math.floor(sz / 3.5));
        const span = isFront ? sx : sz;
        return Array.from({ length: cols }).map((_, col) =>
          Array.from({ length: winRows }).map((_, row) => {
            const xPos = isFront
              ? -span / 2 + 1.5 + col * ((span - 3) / Math.max(cols - 1, 1))
              : sx / 2 + 0.02;
            const zPos = isFront
              ? sz / 2 + 0.02
              : -span / 2 + 1.5 + col * ((span - 3) / Math.max(cols - 1, 1));
            const yPos = -sy / 2 + 2 + row * ((sy - 3) / Math.max(winRows, 1));
            return (
              <group
                key={`${face}-${col}-${row}`}
                position={isFront ? [xPos, yPos, zPos] : [xPos, yPos, zPos]}
                rotation={isFront ? [0, 0, 0] : [0, Math.PI / 2, 0]}
              >
                {/* window pane */}
                <mesh>
                  <planeGeometry args={[1.0, 1.4]} />
                  <meshStandardMaterial
                    color={winCol}
                    emissive={winCol}
                    emissiveIntensity={winGlow}
                    roughness={0.2}
                    metalness={0.4}
                    transparent
                    opacity={0.85}
                  />
                </mesh>
                {/* frame */}
                <mesh position={[0, 0, -0.01]}>
                  <planeGeometry args={[1.2, 1.6]} />
                  <meshStandardMaterial color={trimCol} roughness={0.7} />
                </mesh>
                {/* window sill */}
                <mesh position={[0, -0.8, 0.08]} castShadow>
                  <boxGeometry args={[1.25, 0.08, 0.16]} />
                  <meshStandardMaterial color={PAL.concrete} roughness={0.8} />
                </mesh>
              </group>
            );
          })
        );
      })}

      {/* AWNING for bars/food */}
      {(archetype === 'bar' || archetype === 'food') && (
        <group position={[0, -sy / 2 + 2.8, sz / 2 + 0.8]}>
          <mesh castShadow>
            <boxGeometry args={[sx * 0.7, 0.08, 1.4]} />
            <meshStandardMaterial
              color={archetype === 'bar' ? '#b45309' : '#dc2626'}
              roughness={0.7}
            />
          </mesh>
          {/* awning supports */}
          {[-0.35, 0.35].map(f => (
            <mesh key={f} position={[sx * f * 0.7 / 2, -0.04, 0.6]} rotation={[0.3, 0, 0]}>
              <cylinderGeometry args={[0.03, 0.03, 1.5, 4]} />
              <meshStandardMaterial color={PAL.metal} roughness={0.5} metalness={0.4} />
            </mesh>
          ))}
        </group>
      )}

      {/* PLANTERS for civic/community/residential */}
      {(archetype === 'civic' || archetype === 'community' || archetype === 'residential') && (
        <>
          {[-1, 1].map(dir => (
            <group key={`planter-${dir}`} position={[dir * (sx / 2 - 0.5), -sy / 2 + 0.3, sz / 2 + 0.6]}>
              <mesh castShadow>
                <boxGeometry args={[0.8, 0.6, 0.6]} />
                <meshStandardMaterial color="#8B5E3C" roughness={0.85} />
              </mesh>
              {/* plant */}
              <mesh position={[0, 0.5, 0]}>
                <sphereGeometry args={[0.35, 6, 5]} />
                <meshStandardMaterial color={PAL.grass} roughness={0.9} />
              </mesh>
            </group>
          ))}
        </>
      )}

      {/* MEDICAL cross */}
      {archetype === 'medical' && (
        <group position={[0, sy * 0.15, sz / 2 + 0.03]}>
          <mesh>
            <planeGeometry args={[2.0, 0.5]} />
            <meshStandardMaterial color="#dc2626" emissive="#dc2626" emissiveIntensity={0.5} />
          </mesh>
          <mesh>
            <planeGeometry args={[0.5, 2.0]} />
            <meshStandardMaterial color="#dc2626" emissive="#dc2626" emissiveIntensity={0.5} />
          </mesh>
        </group>
      )}

      {/* PORT crane */}
      {archetype === 'port' && (
        <group position={[sx * 0.3, sy / 2, sz * 0.3]}>
          <mesh castShadow>
            <boxGeometry args={[0.4, 5, 0.4]} />
            <meshStandardMaterial color={PAL.metal} roughness={0.5} metalness={0.4} />
          </mesh>
          <mesh position={[2, 2.5, 0]} rotation={[0, 0, -0.15]} castShadow>
            <boxGeometry args={[4, 0.25, 0.25]} />
            <meshStandardMaterial color="#d97706" roughness={0.5} metalness={0.3} />
          </mesh>
          {/* cable */}
          <mesh position={[3.5, 0.5, 0]}>
            <cylinderGeometry args={[0.02, 0.02, 4, 4]} />
            <meshStandardMaterial color="#1f2937" roughness={0.8} />
          </mesh>
        </group>
      )}
    </group>
  );
}

// ── park area ───────────────────────────────────────────────
function ParkArea({ size }: { size: [number, number, number] }) {
  const [sx, , sz] = size;
  return (
    <group>
      {/* grass surface */}
      <mesh receiveShadow position={[0, 0.05, 0]}>
        <boxGeometry args={[sx, 0.1, sz]} />
        <meshStandardMaterial color={PAL.grass} roughness={0.9} />
      </mesh>
      {/* grass edge ring */}
      <mesh position={[0, 0.02, 0]} receiveShadow>
        <boxGeometry args={[sx + 0.4, 0.06, sz + 0.4]} />
        <meshStandardMaterial color={PAL.grassDark} roughness={0.9} />
      </mesh>

      {/* fountain */}
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0.25, 0]} castShadow>
          <cylinderGeometry args={[1.2, 1.4, 0.4, 16]} />
          <meshStandardMaterial color={PAL.concrete} roughness={0.5} />
        </mesh>
        <mesh position={[0, 0.6, 0]} castShadow>
          <cylinderGeometry args={[0.2, 0.25, 0.4, 8]} />
          <meshStandardMaterial color={PAL.concrete} roughness={0.5} />
        </mesh>
        {/* water surface */}
        <mesh position={[0, 0.42, 0]}>
          <cylinderGeometry args={[1.1, 1.1, 0.02, 16]} />
          <meshStandardMaterial color={PAL.water} roughness={0.1} metalness={0.3} transparent opacity={0.8} />
        </mesh>
        <Sparkles count={15} size={1.5} scale={[1.5, 1, 1.5]} position={[0, 0.7, 0]} speed={0.3} color={PAL.water} opacity={0.4} />
      </group>

      {/* benches */}
      {[
        [-sx * 0.3, sz * 0.2, 0],
        [sx * 0.3, -sz * 0.2, Math.PI],
        [0, sz * 0.35, Math.PI / 2],
      ].map(([bx, bz, rot], i) => (
        <group key={`bench-${i}`} position={[bx as number, 0, bz as number]} rotation={[0, rot as number, 0]}>
          <mesh position={[0, 0.33, 0]} castShadow>
            <boxGeometry args={[1.6, 0.06, 0.42]} />
            <meshStandardMaterial color={PAL.wood} roughness={0.8} />
          </mesh>
          <mesh position={[0, 0.52, -0.18]} castShadow>
            <boxGeometry args={[1.6, 0.32, 0.04]} />
            <meshStandardMaterial color={PAL.wood} roughness={0.8} />
          </mesh>
          {[-0.65, 0.65].map(lx => (
            <mesh key={lx} position={[lx, 0.16, 0]} castShadow>
              <boxGeometry args={[0.06, 0.33, 0.42]} />
              <meshStandardMaterial color={PAL.metal} roughness={0.6} metalness={0.3} />
            </mesh>
          ))}
        </group>
      ))}

      {/* flower beds */}
      {[[-sx * 0.35, sz * 0.3], [sx * 0.35, -sz * 0.25], [-sx * 0.1, -sz * 0.35]].map(([fx, fz], i) => (
        <group key={`flowers-${i}`} position={[fx, 0.1, fz]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.7, 0.8, 0.2, 8]} />
            <meshStandardMaterial color="#5C3A1E" roughness={0.9} />
          </mesh>
          {/* flower clump */}
          {Array.from({ length: 5 }).map((_, j) => {
            const a = (j / 5) * Math.PI * 2;
            const r = 0.3;
            const cols = ['#f472b6', '#fbbf24', '#a78bfa', '#fb7185', '#34d399'];
            return (
              <mesh key={j} position={[Math.cos(a) * r, 0.22, Math.sin(a) * r]}>
                <sphereGeometry args={[0.12, 5, 4]} />
                <meshStandardMaterial color={cols[j % cols.length]} roughness={0.9} />
              </mesh>
            );
          })}
        </group>
      ))}

      {/* walking path */}
      <mesh position={[0, 0.07, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[1.5, sz * 0.85]} />
        <meshStandardMaterial color={PAL.sidewalk} roughness={0.9} />
      </mesh>
    </group>
  );
}

// ════════════════════════════════════════════════════════════
//  ENVIRONMENT ELEMENTS
// ════════════════════════════════════════════════════════════

function Ground({ bounds }: { bounds: ReturnType<typeof getTownBounds> }) {
  const size = Math.max(bounds.maxX - bounds.minX, bounds.maxZ - bounds.minZ) + 120;
  return (
    <group>
      {/* main ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[bounds.centerX, -0.05, bounds.centerZ]} receiveShadow>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color={PAL.ground} roughness={0.95} />
      </mesh>
      {/* grass patches around the edge */}
      {Array.from({ length: 20 }).map((_, i) => {
        const angle = (i / 20) * Math.PI * 2;
        const r = bounds.radius * 1.1 + 5;
        return (
          <mesh
            key={`grassp-${i}`}
            rotation={[-Math.PI / 2, 0, angle]}
            position={[
              bounds.centerX + Math.cos(angle) * r,
              -0.03,
              bounds.centerZ + Math.sin(angle) * r,
            ]}
            receiveShadow
          >
            <circleGeometry args={[3 + (i % 3) * 2, 8]} />
            <meshStandardMaterial color={PAL.grass} roughness={0.9} />
          </mesh>
        );
      })}
    </group>
  );
}

function Roads({ layouts }: { layouts: LocationLayout[] }) {
  const roads = useMemo(() => generateRoads(layouts), [layouts]);
  return (
    <>
      {roads.map((road, i) => {
        const dx = road.end[0] - road.start[0];
        const dz = road.end[1] - road.start[1];
        const len = Math.sqrt(dx * dx + dz * dz);
        const angle = Math.atan2(dx, dz);
        const cx = (road.start[0] + road.end[0]) / 2;
        const cz = (road.start[1] + road.end[1]) / 2;
        return (
          <group key={`road-${i}`}>
            {/* asphalt */}
            <mesh rotation={[-Math.PI / 2, angle, 0]} position={[cx, -0.01, cz]} receiveShadow>
              <planeGeometry args={[road.width, len]} />
              <meshStandardMaterial color={PAL.road} roughness={0.9} />
            </mesh>
            {/* sidewalks */}
            {[-1, 1].map(side => (
              <mesh
                key={side}
                rotation={[-Math.PI / 2, angle, 0]}
                position={[
                  cx + Math.cos(angle + Math.PI / 2) * side * (road.width / 2 + 0.5),
                  0.01,
                  cz - Math.sin(angle + Math.PI / 2) * side * (road.width / 2 + 0.5),
                ]}
                receiveShadow
              >
                <planeGeometry args={[1.0, len]} />
                <meshStandardMaterial color={PAL.sidewalk} roughness={0.85} />
              </mesh>
            ))}
          </group>
        );
      })}
    </>
  );
}

function AnimatedTree({ position, seed }: { position: [number, number, number]; seed: number }) {
  const canopyRef = useRef<THREE.Mesh>(null);
  const trunkH = 2.8 + (seed % 5) * 0.35;
  const canopyR = 1.5 + (seed % 7) * 0.2;
  const hue = 115 + (seed % 15) * 3;
  const sat = 45 + (seed % 8) * 4;
  const light = 32 + (seed % 6) * 3;

  useFrame(({ clock }) => {
    if (canopyRef.current) {
      const t = clock.elapsedTime;
      // wind sway
      canopyRef.current.rotation.z = Math.sin(t * 0.8 + seed) * 0.03;
      canopyRef.current.rotation.x = Math.sin(t * 0.6 + seed * 0.7) * 0.02;
      canopyRef.current.position.y = trunkH + canopyR * 0.55 + Math.sin(t * 1.2 + seed) * 0.03;
    }
  });

  return (
    <group position={position}>
      {/* trunk — tapered */}
      <mesh position={[0, trunkH / 2, 0]} castShadow>
        <cylinderGeometry args={[0.1, 0.22, trunkH, 6]} />
        <meshStandardMaterial color={PAL.wood} roughness={0.9} />
      </mesh>
      {/* roots */}
      {[0, 1.2, 2.4, 3.6].map((a, i) => (
        <mesh key={i} position={[Math.cos(a) * 0.15, 0.08, Math.sin(a) * 0.15]} rotation={[0.3 * Math.cos(a), a, 0.3 * Math.sin(a)]}>
          <cylinderGeometry args={[0.04, 0.08, 0.35, 4]} />
          <meshStandardMaterial color={PAL.woodDark} roughness={0.95} />
        </mesh>
      ))}
      {/* canopy — double sphere for lushness */}
      <mesh ref={canopyRef} position={[0, trunkH + canopyR * 0.55, 0]} castShadow>
        <sphereGeometry args={[canopyR, 10, 7]} />
        <meshStandardMaterial color={`hsl(${hue}, ${sat}%, ${light}%)`} roughness={0.85} />
      </mesh>
      <mesh position={[canopyR * 0.3, trunkH + canopyR * 0.3, canopyR * 0.2]} castShadow>
        <sphereGeometry args={[canopyR * 0.65, 8, 6]} />
        <meshStandardMaterial color={`hsl(${hue + 8}, ${sat + 5}%, ${light - 3}%)`} roughness={0.85} />
      </mesh>
    </group>
  );
}

function Trees({ layouts }: { layouts: LocationLayout[] }) {
  const positions = useMemo(() => generateTreePositions(layouts), [layouts]);
  return (
    <>
      {positions.map((pos, i) => (
        <AnimatedTree key={i} position={pos} seed={i * 17 + 3} />
      ))}
    </>
  );
}

function AnimatedWater({ bounds }: { bounds: ReturnType<typeof getTownBounds> }) {
  const ref = useRef<THREE.Mesh>(null);
  const waterX = bounds.maxX + 18;
  const waterZ = bounds.centerZ;
  const waterR = Math.max(10, bounds.radius * 0.25);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.elapsedTime;
    ref.current.position.y = -0.02 + Math.sin(t * 0.6) * 0.025;
    // subtle scale pulse
    const s = 1 + Math.sin(t * 0.3) * 0.005;
    ref.current.scale.set(s, s, 1);
  });

  return (
    <group position={[waterX, 0, waterZ]}>
      {/* water body */}
      <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[waterR, 40]} />
        <meshStandardMaterial
          color={PAL.water}
          roughness={0.08}
          metalness={0.35}
          transparent
          opacity={0.75}
        />
      </mesh>
      {/* deep center */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.04, 0]}>
        <circleGeometry args={[waterR * 0.6, 24]} />
        <meshStandardMaterial color={PAL.waterDeep} roughness={0.1} metalness={0.3} transparent opacity={0.6} />
      </mesh>
      {/* shoreline ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
        <ringGeometry args={[waterR - 0.3, waterR + 0.5, 32]} />
        <meshStandardMaterial color="#c9b99a" roughness={0.95} />
      </mesh>
      {/* sparkles on water */}
      <Sparkles count={30} size={2} scale={[waterR * 2, 0.5, waterR * 2]} position={[0, 0.1, 0]} speed={0.2} color="#ffffff" opacity={0.3} />
    </group>
  );
}

function StreetLamps({ layouts }: { layouts: LocationLayout[] }) {
  const nightGlow = useNightGlow();
  const lampIntensity = 0.15 + nightGlow * 0.8;
  const bulbEmissive = 0.4 + nightGlow * 1.5;
  const positions = useMemo(() => {
    const lamps: [number, number, number][] = [];
    for (const l of layouts) {
      if (l.size[1] < 1) continue;
      lamps.push([l.position[0] + l.size[0] * 0.55 + 2, 0, l.position[2] + l.size[2] * 0.4]);
      if (l.size[0] > 10) {
        lamps.push([l.position[0] - l.size[0] * 0.55 - 2, 0, l.position[2] + l.size[2] * 0.4]);
      }
    }
    return lamps;
  }, [layouts]);

  return (
    <>
      {positions.map((pos, i) => (
        <group key={`lamp-${i}`} position={pos}>
          {/* pole */}
          <mesh position={[0, 2, 0]} castShadow>
            <cylinderGeometry args={[0.05, 0.08, 4, 6]} />
            <meshStandardMaterial color="#374151" roughness={0.6} metalness={0.4} />
          </mesh>
          {/* curved arm */}
          <mesh position={[0.35, 3.7, 0]} rotation={[0, 0, -0.6]}>
            <cylinderGeometry args={[0.035, 0.035, 0.8, 5]} />
            <meshStandardMaterial color="#374151" roughness={0.6} metalness={0.4} />
          </mesh>
          {/* lamp housing */}
          <mesh position={[0.6, 3.8, 0]} castShadow>
            <cylinderGeometry args={[0.18, 0.1, 0.2, 8]} />
            <meshStandardMaterial color="#1f2937" roughness={0.5} metalness={0.4} />
          </mesh>
          {/* bulb */}
          <mesh position={[0.6, 3.65, 0]}>
            <sphereGeometry args={[0.08, 8, 6]} />
            <meshStandardMaterial color={PAL.lamp} emissive={PAL.lamp} emissiveIntensity={bulbEmissive} roughness={0.2} />
          </mesh>
          <pointLight position={[0.6, 3.65, 0]} color={PAL.lamp} intensity={lampIntensity} distance={8 + nightGlow * 4} />
          {/* base plate */}
          <mesh position={[0, 0.03, 0]}>
            <cylinderGeometry args={[0.15, 0.15, 0.06, 8]} />
            <meshStandardMaterial color="#374151" roughness={0.7} metalness={0.3} />
          </mesh>
        </group>
      ))}
    </>
  );
}

function Clouds({ bounds }: { bounds: ReturnType<typeof getTownBounds> }) {
  const cloudData = useMemo(() => {
    const clouds: { x: number; y: number; z: number; scale: number; speed: number }[] = [];
    for (let i = 0; i < 12; i++) {
      clouds.push({
        x: bounds.centerX + (Math.random() - 0.5) * bounds.radius * 3,
        y: 25 + Math.random() * 15,
        z: bounds.centerZ + (Math.random() - 0.5) * bounds.radius * 3,
        scale: 4 + Math.random() * 8,
        speed: 0.3 + Math.random() * 0.5,
      });
    }
    return clouds;
  }, [bounds]);

  return (
    <>
      {cloudData.map((c, i) => (
        <CloudBillboard key={i} data={c} index={i} bounds={bounds} />
      ))}
    </>
  );
}

function CloudBillboard({ data, index, bounds }: {
  data: { x: number; y: number; z: number; scale: number; speed: number };
  index: number;
  bounds: ReturnType<typeof getTownBounds>;
}) {
  const ref = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.elapsedTime;
    ref.current.position.x = data.x + t * data.speed;
    // wrap around
    if (ref.current.position.x > bounds.centerX + bounds.radius * 2) {
      ref.current.position.x = bounds.centerX - bounds.radius * 2;
    }
    ref.current.position.y = data.y + Math.sin(t * 0.2 + index) * 0.5;
  });

  return (
    <group ref={ref} position={[data.x, data.y, data.z]}>
      {/* cloud is 3 overlapping spheres */}
      <mesh>
        <sphereGeometry args={[data.scale * 0.5, 8, 6]} />
        <meshStandardMaterial color={PAL.cloud} roughness={1} transparent opacity={0.7} />
      </mesh>
      <mesh position={[data.scale * 0.3, -data.scale * 0.1, data.scale * 0.15]}>
        <sphereGeometry args={[data.scale * 0.4, 7, 5]} />
        <meshStandardMaterial color={PAL.cloud} roughness={1} transparent opacity={0.6} />
      </mesh>
      <mesh position={[-data.scale * 0.25, data.scale * 0.05, -data.scale * 0.1]}>
        <sphereGeometry args={[data.scale * 0.35, 7, 5]} />
        <meshStandardMaterial color={PAL.cloud} roughness={1} transparent opacity={0.65} />
      </mesh>
    </group>
  );
}

function AmbientParticles({ bounds }: { bounds: ReturnType<typeof getTownBounds> }) {
  return (
    <Sparkles
      count={80}
      size={1.2}
      scale={[bounds.radius * 2, 8, bounds.radius * 2]}
      position={[bounds.centerX, 3, bounds.centerZ]}
      speed={0.15}
      color="#e8dcc8"
      opacity={0.25}
    />
  );
}

// ════════════════════════════════════════════════════════════
//  MAIN COMPONENT
// ════════════════════════════════════════════════════════════

export function TownEnvironment() {
  const layouts = getLayouts();
  const bounds = useMemo(() => getTownBounds(layouts), [layouts]);

  return (
    <>
      {/* Lighting is handled by DayNightCycle in the parent Scene */}
      {/* Hemisphere fill for ground bounce — always active */}
      <hemisphereLight args={['#87ceeb', '#c2b280', 0.15]} />
      <fog attach="fog" args={['#e8e0d8', bounds.radius * 1.5, bounds.radius * 3.5]} />

      <Ground bounds={bounds} />
      <Roads layouts={layouts} />
      <AnimatedWater bounds={bounds} />
      <Trees layouts={layouts} />
      <StreetLamps layouts={layouts} />
      <Clouds bounds={bounds} />
      <AmbientParticles bounds={bounds} />

      {/* Buildings */}
      {layouts.map(loc => (
        <Building key={loc.id} {...loc} />
      ))}
    </>
  );
}

/**
 * Activity props — 3D objects that appear near agents based on what they're doing.
 *
 * A desk appears when someone's typing, a TV when watching, a beer when drinking,
 * a book when reading, etc. Props are small, stylised, and positioned relative
 * to the agent.
 */

import * as THREE from 'three';
import type { PropType, SubActivity } from '../activities';

export function ActivityProp({ prop, sub }: { prop: PropType | null; sub: SubActivity }) {
  if (!prop || prop === 'none') return null;

  switch (prop) {
    case 'computer':
      return (
        <group position={[0, 0.45, 0.4]}>
          {/* desk */}
          <mesh castShadow>
            <boxGeometry args={[0.7, 0.03, 0.4]} />
            <meshStandardMaterial color="#8B5E3C" roughness={0.8} />
          </mesh>
          {/* desk legs */}
          {[[-0.3, -0.3], [0.3, -0.3], [-0.3, 0.15], [0.3, 0.15]].map(([x, z], i) => (
            <mesh key={i} position={[x, -0.22, z]}>
              <boxGeometry args={[0.03, 0.44, 0.03]} />
              <meshStandardMaterial color="#5C3A1E" roughness={0.9} />
            </mesh>
          ))}
          {/* monitor */}
          <mesh position={[0, 0.18, 0.05]}>
            <boxGeometry args={[0.3, 0.22, 0.02]} />
            <meshStandardMaterial color="#1e293b" roughness={0.5} />
          </mesh>
          {/* screen */}
          <mesh position={[0, 0.18, 0.06]}>
            <planeGeometry args={[0.26, 0.18]} />
            <meshStandardMaterial color="#60a5fa" emissive="#60a5fa" emissiveIntensity={0.3} roughness={0.2} />
          </mesh>
          {/* monitor stand */}
          <mesh position={[0, 0.04, 0.05]}>
            <boxGeometry args={[0.06, 0.06, 0.04]} />
            <meshStandardMaterial color="#374151" roughness={0.6} />
          </mesh>
        </group>
      );

    case 'tv':
      return (
        <group position={[0, 0.5, 0.8]}>
          {/* TV */}
          <mesh>
            <boxGeometry args={[0.6, 0.35, 0.03]} />
            <meshStandardMaterial color="#1e293b" roughness={0.4} />
          </mesh>
          {/* screen glow */}
          <mesh position={[0, 0, 0.02]}>
            <planeGeometry args={[0.55, 0.3]} />
            <meshStandardMaterial
              color="#93c5fd" emissive="#93c5fd"
              emissiveIntensity={0.5} roughness={0.1}
            />
          </mesh>
          {/* stand */}
          <mesh position={[0, -0.2, 0]}>
            <boxGeometry args={[0.15, 0.06, 0.08]} />
            <meshStandardMaterial color="#374151" roughness={0.6} />
          </mesh>
        </group>
      );

    case 'phone':
      return (
        <group position={[0.15, 1.2, 0.15]}>
          <mesh rotation={[0.3, 0, 0]}>
            <boxGeometry args={[0.04, 0.08, 0.005]} />
            <meshStandardMaterial color="#1e293b" roughness={0.3} />
          </mesh>
          <mesh position={[0, 0, 0.003]} rotation={[0.3, 0, 0]}>
            <planeGeometry args={[0.035, 0.065]} />
            <meshStandardMaterial color="#60a5fa" emissive="#60a5fa" emissiveIntensity={0.2} />
          </mesh>
        </group>
      );

    case 'coffee_cup':
      return (
        <group position={[0.2, 0.75, 0.15]}>
          {/* cup */}
          <mesh>
            <cylinderGeometry args={[0.03, 0.025, 0.06, 8]} />
            <meshStandardMaterial color="#f5f5f0" roughness={0.5} />
          </mesh>
          {/* handle */}
          <mesh position={[0.035, 0, 0]}>
            <torusGeometry args={[0.015, 0.004, 6, 8, Math.PI]} />
            <meshStandardMaterial color="#f5f5f0" roughness={0.5} />
          </mesh>
          {/* steam (just a tiny transparent sphere) */}
          <mesh position={[0, 0.05, 0]}>
            <sphereGeometry args={[0.015, 4, 4]} />
            <meshStandardMaterial color="#ffffff" transparent opacity={0.2} />
          </mesh>
        </group>
      );

    case 'book':
      return (
        <group position={[0, 0.9, 0.2]} rotation={[0.5, 0, 0]}>
          {/* book body */}
          <mesh>
            <boxGeometry args={[0.1, 0.14, 0.015]} />
            <meshStandardMaterial color="#7c2d12" roughness={0.8} />
          </mesh>
          {/* pages */}
          <mesh position={[0, 0, 0.008]}>
            <boxGeometry args={[0.09, 0.13, 0.01]} />
            <meshStandardMaterial color="#fef3c7" roughness={0.9} />
          </mesh>
        </group>
      );

    case 'beer':
      return (
        <group position={[0.2, 0.75, 0.1]}>
          {/* glass */}
          <mesh>
            <cylinderGeometry args={[0.025, 0.02, 0.08, 8]} />
            <meshStandardMaterial color="#fbbf24" roughness={0.2} transparent opacity={0.7} />
          </mesh>
          {/* foam */}
          <mesh position={[0, 0.04, 0]}>
            <cylinderGeometry args={[0.026, 0.025, 0.015, 8]} />
            <meshStandardMaterial color="#fef9c3" roughness={0.8} />
          </mesh>
        </group>
      );

    case 'guitar':
      return (
        <group position={[-0.15, 0.7, 0.1]} rotation={[0.3, 0.5, -0.4]}>
          {/* body */}
          <mesh>
            <sphereGeometry args={[0.1, 8, 6]} />
            <meshStandardMaterial color="#8B5E3C" roughness={0.8} />
          </mesh>
          {/* neck */}
          <mesh position={[0, 0.2, 0]}>
            <boxGeometry args={[0.025, 0.3, 0.015]} />
            <meshStandardMaterial color="#5C3A1E" roughness={0.9} />
          </mesh>
        </group>
      );

    case 'papers':
      return (
        <group position={[0.15, 0.47, 0.3]}>
          {[0, 0.003, 0.006].map((y, i) => (
            <mesh key={i} position={[i * 0.02 - 0.02, y, i * 0.01]} rotation={[0, i * 0.1 - 0.1, 0]}>
              <boxGeometry args={[0.08, 0.002, 0.1]} />
              <meshStandardMaterial color="#f8fafc" roughness={0.9} />
            </mesh>
          ))}
        </group>
      );

    case 'broom':
      return (
        <group position={[0.3, 0.5, 0]} rotation={[0, 0, 0.2]}>
          {/* handle */}
          <mesh>
            <cylinderGeometry args={[0.01, 0.01, 0.8, 4]} />
            <meshStandardMaterial color="#8B5E3C" roughness={0.9} />
          </mesh>
          {/* bristles */}
          <mesh position={[0, -0.42, 0]}>
            <boxGeometry args={[0.08, 0.06, 0.04]} />
            <meshStandardMaterial color="#a3866a" roughness={0.95} />
          </mesh>
        </group>
      );

    case 'dumbbell':
      return (
        <group position={[0.25, 0.2, 0]}>
          {/* bar */}
          <mesh rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.008, 0.008, 0.15, 4]} />
            <meshStandardMaterial color="#6b7280" roughness={0.4} metalness={0.5} />
          </mesh>
          {/* weights */}
          {[-0.07, 0.07].map((x, i) => (
            <mesh key={i} position={[x, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.025, 0.025, 0.02, 8]} />
              <meshStandardMaterial color="#374151" roughness={0.5} metalness={0.4} />
            </mesh>
          ))}
        </group>
      );

    case 'pool_cue':
      return (
        <group position={[0.2, 0.7, 0.3]} rotation={[0.6, 0, -0.1]}>
          <mesh>
            <cylinderGeometry args={[0.006, 0.01, 0.9, 4]} />
            <meshStandardMaterial color="#8B5E3C" roughness={0.8} />
          </mesh>
          {/* tip */}
          <mesh position={[0, 0.45, 0]}>
            <sphereGeometry args={[0.008, 4, 4]} />
            <meshStandardMaterial color="#60a5fa" roughness={0.5} />
          </mesh>
        </group>
      );

    case 'bird_seed':
      return null; // too small to render meaningfully

    default:
      return null;
  }
}

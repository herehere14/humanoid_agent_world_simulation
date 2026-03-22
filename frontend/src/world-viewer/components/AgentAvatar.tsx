/**
 * Fully articulated 3D agent avatar.
 *
 * Each avatar has head, torso, two upper-arms, two forearms, two hands,
 * two upper-legs, two lower-legs.  Every limb is animated per-frame based
 * on the agent's simulation action, emotional state, and interaction context.
 *
 * Behavior catalogue (each with unique body language):
 *   confronting  — lean forward, arms thrust, jabbing gesture, body sway
 *   collapsed    — knees buckle, head drops, arms limp at sides
 *   fleeing      — fast sprint cycle, hunched, pumping arms
 *   withdrawing  — slow shuffle, arms crossed, head down
 *   ruminating   — still, hand on chin, subtle weight shift
 *   venting      — one arm gestures emphatically, torso twists
 *   socializing  — relaxed stance, open palms, gentle sway
 *   helping      — lean toward partner, one arm extended, head tilt
 *   celebrating  — jump, both arms up, head back
 *   working      — seated posture, subtle typing / writing motion
 *   resting      — lying / slumped, minimal motion
 *   idle         — weight shift, occasional hand-to-hip
 *   walking      — walk cycle between locations (interpolated)
 */

import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import type { AgentState } from '../types';
import {
  actionToBehavior,
  getEmotionColor,
  type BehaviorClass,
} from '../layout';
import { useWorldStore } from '../store';

// ─── constants ───────────────────────────────────────────────
const SKIN = '#e8b88a';
const SKIN_DARK = '#c69c6d';
const PANTS = '#374151';
const SHOE = '#1f2937';

// ─── helpers ─────────────────────────────────────────────────
function truncate(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}
function shouldShowThought(a: AgentState) {
  return (
    a.vulnerability > 0.35 ||
    a.divergence > 0.25 ||
    a.arousal > 0.55 ||
    a.action === 'RUMINATE' ||
    a.action === 'COLLAPSE'
  );
}

// ─── pose calculator ─────────────────────────────────────────
/** Returns joint angles for every limb given the current behaviour + time */
interface Pose {
  // torso
  torsoLean: number;   // forward lean (x rot)
  torsoTwist: number;  // side twist  (y rot)
  torsoSway: number;   // side sway   (z rot)
  torsoY: number;      // vertical offset (crouch / jump)
  // head
  headNod: number;
  headTilt: number;
  headTurn: number;
  // left arm
  lShoulderX: number; lShoulderZ: number;
  lElbowX: number;
  lWristX: number;
  // right arm
  rShoulderX: number; rShoulderZ: number;
  rElbowX: number;
  rWristX: number;
  // left leg
  lHipX: number; lKneeX: number;
  // right leg
  rHipX: number; rKneeX: number;
}

function computePose(behavior: BehaviorClass, t: number, phase: number): Pose {
  const s = Math.sin;
  const c = Math.cos;
  const p = t + phase; // per-agent phase offset

  const base: Pose = {
    torsoLean: 0, torsoTwist: 0, torsoSway: 0, torsoY: 0,
    headNod: 0, headTilt: 0, headTurn: 0,
    lShoulderX: 0.15, lShoulderZ: -0.1, lElbowX: 0.1, lWristX: 0,
    rShoulderX: 0.15, rShoulderZ: 0.1,  rElbowX: 0.1, rWristX: 0,
    lHipX: 0, lKneeX: 0, rHipX: 0, rKneeX: 0,
  };

  switch (behavior) {
    case 'idle': {
      // gentle weight shift, occasional hand to hip
      base.torsoSway = s(p * 0.7) * 0.02;
      base.lHipX = s(p * 0.5) * 0.03;
      base.rHipX = -s(p * 0.5) * 0.03;
      base.headTurn = s(p * 0.3) * 0.08;
      base.lShoulderZ = -0.12 + s(p * 0.4) * 0.03;
      break;
    }

    case 'walking': {
      // walk cycle
      const stride = s(p * 4);
      base.lHipX = stride * 0.4;
      base.rHipX = -stride * 0.4;
      base.lKneeX = Math.max(0, -stride) * 0.5;
      base.rKneeX = Math.max(0, stride) * 0.5;
      base.lShoulderX = -stride * 0.3 + 0.15;
      base.rShoulderX = stride * 0.3 + 0.15;
      base.lElbowX = 0.4;
      base.rElbowX = 0.4;
      base.torsoTwist = stride * 0.04;
      base.torsoSway = s(p * 8) * 0.015;
      base.torsoY = Math.abs(s(p * 4)) * 0.03;
      break;
    }

    case 'confronting': {
      // aggressive — leaning forward, jabbing with right hand, left fist cocked
      base.torsoLean = -0.18 + s(p * 3) * 0.04;
      base.torsoTwist = s(p * 2.5) * 0.08;
      base.torsoSway = s(p * 4) * 0.06;
      // right arm — jab forward repeatedly
      const jab = Math.max(0, s(p * 5));
      base.rShoulderX = -1.2 - jab * 0.5;
      base.rShoulderZ = 0.1;
      base.rElbowX = 0.3 + jab * 0.6;
      base.rWristX = -0.3;
      // left arm — fist guard
      base.lShoulderX = -0.8;
      base.lShoulderZ = -0.25;
      base.lElbowX = 1.4;
      base.lWristX = -0.2;
      // stomping feet
      base.lHipX = s(p * 3) * 0.15;
      base.rHipX = -s(p * 3) * 0.15;
      base.lKneeX = Math.max(0, s(p * 3)) * 0.2;
      base.rKneeX = Math.max(0, -s(p * 3)) * 0.2;
      // head — chin forward, glaring
      base.headNod = -0.15;
      base.headTurn = s(p * 2) * 0.1;
      base.torsoY = s(p * 6) * 0.02;
      break;
    }

    case 'collapsed': {
      // on knees, head down, arms limp
      base.torsoLean = 0.5;
      base.torsoY = -0.45;
      base.headNod = 0.5;
      base.headTilt = 0.15;
      base.lShoulderX = 0.3; base.lShoulderZ = -0.05;
      base.rShoulderX = 0.3; base.rShoulderZ = 0.05;
      base.lElbowX = 0.1; base.rElbowX = 0.1;
      base.lHipX = 0.9; base.rHipX = 0.9;
      base.lKneeX = 1.5; base.rKneeX = 1.5;
      // subtle breathing
      base.torsoLean += s(p * 1.2) * 0.03;
      break;
    }

    case 'fleeing': {
      // panicked sprint
      const run = s(p * 7);
      base.lHipX = run * 0.7;
      base.rHipX = -run * 0.7;
      base.lKneeX = Math.max(0, -run) * 0.9;
      base.rKneeX = Math.max(0, run) * 0.9;
      base.lShoulderX = -run * 0.5;
      base.rShoulderX = run * 0.5;
      base.lElbowX = 0.8; base.rElbowX = 0.8;
      base.torsoLean = -0.2;
      base.torsoTwist = run * 0.06;
      base.torsoY = Math.abs(s(p * 7)) * 0.06;
      base.headNod = -0.1;
      break;
    }

    case 'withdrawing': {
      // slow shuffle, arms hugging self
      base.torsoLean = 0.08;
      base.lShoulderX = -0.5; base.lShoulderZ = 0.3;
      base.rShoulderX = -0.5; base.rShoulderZ = -0.3;
      base.lElbowX = 1.5; base.rElbowX = 1.5;
      base.headNod = 0.2;
      const shuffle = s(p * 1.5);
      base.lHipX = shuffle * 0.08;
      base.rHipX = -shuffle * 0.08;
      base.torsoSway = s(p * 0.8) * 0.02;
      break;
    }

    case 'ruminating': {
      // hand on chin, weight shift
      base.rShoulderX = -0.7;
      base.rElbowX = 1.6;
      base.rWristX = 0.3;
      base.lShoulderX = -0.3; base.lShoulderZ = 0.15;
      base.lElbowX = 1.2;
      base.headNod = 0.1 + s(p * 0.4) * 0.04;
      base.headTilt = -0.08;
      base.torsoSway = s(p * 0.6) * 0.02;
      base.lHipX = s(p * 0.3) * 0.04;
      break;
    }

    case 'venting': {
      // one arm gesturing emphatically, body twisting
      const gesture = s(p * 3.5);
      base.rShoulderX = -1.0 - gesture * 0.4;
      base.rShoulderZ = gesture * 0.15;
      base.rElbowX = 0.5 + Math.abs(gesture) * 0.4;
      base.rWristX = gesture * 0.4;
      base.lShoulderX = -0.3;
      base.lElbowX = 0.6;
      base.torsoTwist = gesture * 0.1;
      base.torsoLean = -0.06;
      base.headTurn = gesture * 0.12;
      base.headNod = -0.08 + s(p * 5) * 0.04;
      base.lHipX = s(p * 1.5) * 0.05;
      base.rHipX = -s(p * 1.5) * 0.05;
      break;
    }

    case 'socializing': {
      // relaxed open palms, gentle sway
      base.lShoulderX = -0.3 + s(p * 1.2) * 0.12;
      base.rShoulderX = -0.3 + s(p * 1.2 + 1) * 0.12;
      base.lShoulderZ = -0.25; base.rShoulderZ = 0.25;
      base.lElbowX = 0.4; base.rElbowX = 0.4;
      base.lWristX = -0.2; base.rWristX = -0.2;
      base.torsoSway = s(p * 0.8) * 0.03;
      base.headTurn = s(p * 0.6) * 0.1;
      base.headNod = s(p * 1.5) * 0.05;
      base.lHipX = s(p * 0.5) * 0.02;
      break;
    }

    case 'helping': {
      // lean toward partner, arm extended
      base.torsoLean = -0.12;
      base.rShoulderX = -1.1;
      base.rShoulderZ = 0.1;
      base.rElbowX = 0.25;
      base.rWristX = -0.15;
      base.lShoulderX = -0.2;
      base.lElbowX = 0.5;
      base.headNod = -0.1;
      base.headTilt = 0.08;
      base.torsoSway = s(p * 0.8) * 0.02;
      break;
    }

    case 'celebrating': {
      // jump, arms up, head back
      const jump = Math.abs(s(p * 4));
      base.torsoY = jump * 0.25;
      base.lShoulderX = -2.8 + s(p * 5) * 0.2;
      base.rShoulderX = -2.8 + s(p * 5 + 0.5) * 0.2;
      base.lShoulderZ = -0.3; base.rShoulderZ = 0.3;
      base.lElbowX = 0.1; base.rElbowX = 0.1;
      base.lWristX = s(p * 8) * 0.3;
      base.rWristX = s(p * 8 + 1) * 0.3;
      base.headNod = -0.2;
      base.lHipX = s(p * 4) * 0.2;
      base.rHipX = -s(p * 4) * 0.2;
      base.lKneeX = Math.max(0, s(p * 4)) * 0.3;
      base.rKneeX = Math.max(0, -s(p * 4)) * 0.3;
      break;
    }

    case 'working': {
      // seated-ish posture, typing/writing
      base.torsoLean = 0.1;
      base.torsoY = -0.15;
      base.lShoulderX = -0.5;
      base.rShoulderX = -0.5;
      base.lElbowX = 1.3 + s(p * 4) * 0.08;
      base.rElbowX = 1.3 + s(p * 4 + 1.5) * 0.08;
      base.lWristX = s(p * 6) * 0.15;
      base.rWristX = s(p * 6 + 2) * 0.15;
      base.lHipX = 0.5; base.rHipX = 0.5;
      base.lKneeX = 0.5; base.rKneeX = 0.5;
      base.headNod = 0.1 + s(p * 1) * 0.03;
      break;
    }

    case 'resting': {
      // lying down / slumped
      base.torsoLean = 0.35;
      base.torsoY = -0.35;
      base.headNod = 0.3;
      base.lShoulderX = 0.2; base.rShoulderX = 0.2;
      base.lElbowX = 0.1; base.rElbowX = 0.1;
      base.lHipX = 0.6; base.rHipX = 0.6;
      base.lKneeX = 0.4; base.rKneeX = 0.4;
      // breathing
      base.torsoLean += s(p * 0.8) * 0.02;
      break;
    }
  }
  return base;
}

// ─── main component ──────────────────────────────────────────
export function AgentAvatar({
  agent,
  targetPosition,
  interactionPartner,
  interactionType,
}: {
  agent: AgentState;
  targetPosition: [number, number, number];
  interactionPartner: string | null;
  interactionType: string | null;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const currentPos = useRef(new THREE.Vector3(...targetPosition));

  // refs for animated joints
  const torsoRef   = useRef<THREE.Group>(null);
  const headRef    = useRef<THREE.Group>(null);
  const lUpperArm  = useRef<THREE.Group>(null);
  const lForearm   = useRef<THREE.Group>(null);
  const lHand      = useRef<THREE.Group>(null);
  const rUpperArm  = useRef<THREE.Group>(null);
  const rForearm   = useRef<THREE.Group>(null);
  const rHand      = useRef<THREE.Group>(null);
  const lUpperLeg  = useRef<THREE.Group>(null);
  const lLowerLeg  = useRef<THREE.Group>(null);
  const rUpperLeg  = useRef<THREE.Group>(null);
  const rLowerLeg  = useRef<THREE.Group>(null);

  const selectedAgentId = useWorldStore(s => s.selectedAgentId);
  const hoveredAgentId  = useWorldStore(s => s.hoveredAgentId);
  const selectAgent     = useWorldStore(s => s.selectAgent);
  const hoverAgent      = useWorldStore(s => s.hoverAgent);

  const isSelected = selectedAgentId === agent.id;
  const isHovered  = hoveredAgentId === agent.id;

  const behavior    = actionToBehavior(agent.action);
  const emotionColor = getEmotionColor(agent.valence, agent.arousal, agent.vulnerability);
  const showThought = shouldShowThought(agent);

  // per-agent phase offset so they don't all sync
  const phase = useRef(
    (agent.id.charCodeAt(0) * 7 + (agent.id.charCodeAt(1) || 0) * 13) * 0.1
  ).current;

  // ─── per-frame animation ────────────────────────────────
  useFrame((_, delta) => {
    if (!groupRef.current) return;
    const t = performance.now() * 0.001;
    const target = new THREE.Vector3(...targetPosition);
    const speed = behavior === 'fleeing' ? 5 : behavior === 'walking' ? 3.5 : 2.5;
    currentPos.current.lerp(target, Math.min(1, delta * speed));
    groupRef.current.position.copy(currentPos.current);

    // detect if we're still moving significantly → show walk cycle
    const moving = currentPos.current.distanceTo(target) > 0.3;
    const effectiveBehavior: BehaviorClass =
      moving && behavior !== 'fleeing' && behavior !== 'collapsed' && behavior !== 'resting'
        ? 'walking'
        : behavior;

    const pose = computePose(effectiveBehavior, t, phase);

    // apply pose to joints
    groupRef.current.position.y = currentPos.current.y + pose.torsoY;

    if (torsoRef.current) {
      torsoRef.current.rotation.x = pose.torsoLean;
      torsoRef.current.rotation.y = pose.torsoTwist;
      torsoRef.current.rotation.z = pose.torsoSway;
    }
    if (headRef.current) {
      headRef.current.rotation.x = pose.headNod;
      headRef.current.rotation.y = pose.headTurn;
      headRef.current.rotation.z = pose.headTilt;
    }
    if (lUpperArm.current) {
      lUpperArm.current.rotation.x = pose.lShoulderX;
      lUpperArm.current.rotation.z = pose.lShoulderZ;
    }
    if (lForearm.current) lForearm.current.rotation.x = pose.lElbowX;
    if (lHand.current)    lHand.current.rotation.x    = pose.lWristX;
    if (rUpperArm.current) {
      rUpperArm.current.rotation.x = pose.rShoulderX;
      rUpperArm.current.rotation.z = pose.rShoulderZ;
    }
    if (rForearm.current) rForearm.current.rotation.x = pose.rElbowX;
    if (rHand.current)    rHand.current.rotation.x    = pose.rWristX;
    if (lUpperLeg.current) lUpperLeg.current.rotation.x = pose.lHipX;
    if (lLowerLeg.current) lLowerLeg.current.rotation.x = pose.lKneeX;
    if (rUpperLeg.current) rUpperLeg.current.rotation.x = pose.rHipX;
    if (rLowerLeg.current) rLowerLeg.current.rotation.x = pose.rKneeX;

    // face partner during interactions
    if (interactionPartner && torsoRef.current) {
      // gentle turn toward partner would need partner pos;
      // for now slight extra twist toward interaction
      torsoRef.current.rotation.y += Math.sin(t * 2) * 0.04;
    }
  });

  const ringColor = isSelected ? '#0066ff' : isHovered ? '#60a5fa' : '';

  return (
    <group
      ref={groupRef}
      position={targetPosition}
      onClick={e => { e.stopPropagation(); selectAgent(isSelected ? null : agent.id); }}
      onPointerEnter={e => { e.stopPropagation(); hoverAgent(agent.id); document.body.style.cursor = 'pointer'; }}
      onPointerLeave={() => { hoverAgent(null); document.body.style.cursor = 'default'; }}
    >
      {/* selection ring */}
      {(isSelected || isHovered) && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
          <ringGeometry args={[0.65, 0.8, 32]} />
          <meshBasicMaterial color={ringColor} transparent opacity={0.7} />
        </mesh>
      )}

      {/* ─── articulated body ─── */}
      {/* hierarchy: root → torso → (head, arms, legs) */}
      <group ref={torsoRef}>
        {/* TORSO */}
        <mesh position={[0, 0.95, 0]} castShadow>
          <capsuleGeometry args={[0.2, 0.55, 4, 8]} />
          <meshStandardMaterial
            color={emotionColor} roughness={0.55} metalness={0.05}
            emissive={emotionColor}
            emissiveIntensity={isSelected ? 0.25 : agent.arousal > 0.6 ? 0.12 : 0.03}
          />
        </mesh>
        {/* NECK */}
        <mesh position={[0, 1.32, 0]}>
          <cylinderGeometry args={[0.06, 0.08, 0.12, 6]} />
          <meshStandardMaterial color={SKIN} roughness={0.7} />
        </mesh>

        {/* HEAD */}
        <group ref={headRef} position={[0, 1.48, 0]}>
          <mesh castShadow>
            <sphereGeometry args={[0.2, 14, 10]} />
            <meshStandardMaterial color={SKIN} roughness={0.65} />
          </mesh>
          {/* eyes */}
          <mesh position={[-0.07, 0.04, 0.17]}>
            <sphereGeometry args={[0.03, 6, 6]} />
            <meshStandardMaterial color="#1e293b" />
          </mesh>
          <mesh position={[0.07, 0.04, 0.17]}>
            <sphereGeometry args={[0.03, 6, 6]} />
            <meshStandardMaterial color="#1e293b" />
          </mesh>
          {/* hair */}
          <mesh position={[0, 0.1, -0.02]} castShadow>
            <sphereGeometry args={[0.19, 10, 8, 0, Math.PI * 2, 0, Math.PI * 0.55]} />
            <meshStandardMaterial color="#3d2b1f" roughness={0.9} />
          </mesh>
        </group>

        {/* ─── LEFT ARM ─── */}
        <group ref={lUpperArm} position={[-0.32, 1.18, 0]}>
          <mesh position={[0, -0.17, 0]} castShadow>
            <capsuleGeometry args={[0.065, 0.22, 4, 6]} />
            <meshStandardMaterial color={emotionColor} roughness={0.6} />
          </mesh>
          <group ref={lForearm} position={[0, -0.35, 0]}>
            <mesh position={[0, -0.14, 0]} castShadow>
              <capsuleGeometry args={[0.055, 0.2, 4, 6]} />
              <meshStandardMaterial color={SKIN_DARK} roughness={0.65} />
            </mesh>
            <group ref={lHand} position={[0, -0.3, 0]}>
              <mesh castShadow>
                <boxGeometry args={[0.08, 0.1, 0.05]} />
                <meshStandardMaterial color={SKIN} roughness={0.7} />
              </mesh>
              {/* fingers hint */}
              <mesh position={[0, -0.06, 0]}>
                <boxGeometry args={[0.07, 0.05, 0.04]} />
                <meshStandardMaterial color={SKIN} roughness={0.7} />
              </mesh>
            </group>
          </group>
        </group>

        {/* ─── RIGHT ARM ─── */}
        <group ref={rUpperArm} position={[0.32, 1.18, 0]}>
          <mesh position={[0, -0.17, 0]} castShadow>
            <capsuleGeometry args={[0.065, 0.22, 4, 6]} />
            <meshStandardMaterial color={emotionColor} roughness={0.6} />
          </mesh>
          <group ref={rForearm} position={[0, -0.35, 0]}>
            <mesh position={[0, -0.14, 0]} castShadow>
              <capsuleGeometry args={[0.055, 0.2, 4, 6]} />
              <meshStandardMaterial color={SKIN_DARK} roughness={0.65} />
            </mesh>
            <group ref={rHand} position={[0, -0.3, 0]}>
              <mesh castShadow>
                <boxGeometry args={[0.08, 0.1, 0.05]} />
                <meshStandardMaterial color={SKIN} roughness={0.7} />
              </mesh>
              <mesh position={[0, -0.06, 0]}>
                <boxGeometry args={[0.07, 0.05, 0.04]} />
                <meshStandardMaterial color={SKIN} roughness={0.7} />
              </mesh>
            </group>
          </group>
        </group>

        {/* HIPS */}
        <mesh position={[0, 0.6, 0]}>
          <boxGeometry args={[0.32, 0.12, 0.18]} />
          <meshStandardMaterial color={PANTS} roughness={0.8} />
        </mesh>

        {/* ─── LEFT LEG ─── */}
        <group ref={lUpperLeg} position={[-0.1, 0.55, 0]}>
          <mesh position={[0, -0.2, 0]} castShadow>
            <capsuleGeometry args={[0.085, 0.25, 4, 6]} />
            <meshStandardMaterial color={PANTS} roughness={0.8} />
          </mesh>
          <group ref={lLowerLeg} position={[0, -0.42, 0]}>
            <mesh position={[0, -0.17, 0]} castShadow>
              <capsuleGeometry args={[0.07, 0.22, 4, 6]} />
              <meshStandardMaterial color={PANTS} roughness={0.8} />
            </mesh>
            {/* shoe */}
            <mesh position={[0, -0.35, 0.04]} castShadow>
              <boxGeometry args={[0.1, 0.06, 0.16]} />
              <meshStandardMaterial color={SHOE} roughness={0.9} />
            </mesh>
          </group>
        </group>

        {/* ─── RIGHT LEG ─── */}
        <group ref={rUpperLeg} position={[0.1, 0.55, 0]}>
          <mesh position={[0, -0.2, 0]} castShadow>
            <capsuleGeometry args={[0.085, 0.25, 4, 6]} />
            <meshStandardMaterial color={PANTS} roughness={0.8} />
          </mesh>
          <group ref={rLowerLeg} position={[0, -0.42, 0]}>
            <mesh position={[0, -0.17, 0]} castShadow>
              <capsuleGeometry args={[0.07, 0.22, 4, 6]} />
              <meshStandardMaterial color={PANTS} roughness={0.8} />
            </mesh>
            <mesh position={[0, -0.35, 0.04]} castShadow>
              <boxGeometry args={[0.1, 0.06, 0.16]} />
              <meshStandardMaterial color={SHOE} roughness={0.9} />
            </mesh>
          </group>
        </group>
      </group>

      {/* vulnerability glow */}
      {agent.vulnerability > 0.45 && (
        <pointLight position={[0, 1, 0]} color={emotionColor} intensity={agent.vulnerability * 0.5} distance={3} />
      )}

      {/* ─── conflict VFX: impact sparks ─── */}
      {behavior === 'confronting' && (
        <ConflictVFX color={emotionColor} />
      )}

      {/* ─── HTML overlays ─── */}
      {isSelected && (
        <Html position={[0, 2.0, 0]} center distanceFactor={15} style={{ pointerEvents: 'none' }}>
          <div style={{ background: '#0066ff', color: '#fff', borderRadius: 6, padding: '2px 8px', fontSize: 10, fontWeight: 600, fontFamily: 'Inter, system-ui', whiteSpace: 'nowrap' }}>
            {agent.name}
          </div>
        </Html>
      )}
      {isHovered && !isSelected && (
        <Html position={[0, 2.2, 0]} center distanceFactor={15} style={{ pointerEvents: 'none' }}>
          <HoverTooltip agent={agent} />
        </Html>
      )}
      {interactionPartner && (
        <Html position={[0.4, 2.4, 0]} center distanceFactor={18} style={{ pointerEvents: 'none' }}>
          <SpeechBubble type={interactionType} agent={agent} />
        </Html>
      )}
      {showThought && !interactionPartner && (
        <Html position={[-0.3, 2.3, 0]} center distanceFactor={20} style={{ pointerEvents: 'none' }}>
          <ThoughtBubble agent={agent} />
        </Html>
      )}
    </group>
  );
}

// ─── conflict VFX ────────────────────────────────────────────
function ConflictVFX({ color }: { color: string }) {
  const ref = useRef<THREE.Group>(null);
  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y += 0.05;
      ref.current.children.forEach((c, i) => {
        const t = performance.now() * 0.001;
        c.position.y = 1.0 + Math.sin(t * 8 + i * 2) * 0.15;
        c.position.x = Math.cos(t * 6 + i * 1.5) * 0.3;
        c.position.z = Math.sin(t * 6 + i * 1.5) * 0.3;
        (c as THREE.Mesh).scale.setScalar(0.5 + Math.abs(Math.sin(t * 10 + i)) * 0.5);
      });
    }
  });

  return (
    <group ref={ref}>
      {[0, 1, 2, 3].map(i => (
        <mesh key={i} position={[0, 1, 0]}>
          <octahedronGeometry args={[0.04, 0]} />
          <meshBasicMaterial color={color} transparent opacity={0.8} />
        </mesh>
      ))}
    </group>
  );
}

// ─── UI bubbles ──────────────────────────────────────────────
function HoverTooltip({ agent }: { agent: AgentState }) {
  const ec = getEmotionColor(agent.valence, agent.arousal, agent.vulnerability);
  return (
    <div style={{ background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(8px)', borderRadius: 8, padding: '6px 10px', boxShadow: '0 2px 12px rgba(0,0,0,0.15)', fontSize: 11, fontFamily: 'Inter, system-ui', color: '#1a1a2e', whiteSpace: 'nowrap', lineHeight: 1.4, minWidth: 130 }}>
      <div style={{ fontWeight: 600, fontSize: 12 }}>{agent.name}</div>
      <div style={{ color: '#64748b', fontSize: 10 }}>{agent.action.replace(/_/g, ' ')} at {agent.location}</div>
      <div style={{ marginTop: 2, display: 'flex', gap: 6, fontSize: 10 }}>
        <span style={{ color: ec }}>{agent.surface}</span>
        {agent.divergence > 0.2 && <span style={{ color: '#94a3b8' }}>(actually {agent.internal})</span>}
      </div>
      <div style={{ marginTop: 3, fontSize: 9, color: '#94a3b8', fontStyle: 'italic' }}>
        {truncate(agent.primary_concern, 45)}
      </div>
    </div>
  );
}

function SpeechBubble({ type, agent }: { type: string | null; agent: AgentState }) {
  const bg = type === 'conflict' ? '#fef2f2' : type === 'support' ? '#f0fdf4' : type === 'positive' ? '#fffbeb' : '#f8fafc';
  const border = type === 'conflict' ? '#fca5a5' : type === 'support' ? '#86efac' : type === 'positive' ? '#fde68a' : '#e2e8f0';
  const icon = type === 'conflict' ? '⚡' : type === 'support' ? '🤝' : type === 'positive' ? '✨' : '💬';
  return (
    <div style={{ background: bg, border: `1.5px solid ${border}`, borderRadius: '10px 10px 10px 2px', padding: '4px 8px', fontSize: 10, fontFamily: 'Inter, system-ui', maxWidth: 170, lineHeight: 1.3, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
      <div style={{ fontWeight: 600, fontSize: 11 }}>{icon} {agent.name}</div>
      <div style={{ color: '#475569', marginTop: 2 }}>{truncate(agent.action_style, 50)}</div>
    </div>
  );
}

function ThoughtBubble({ agent }: { agent: AgentState }) {
  const thought = agent.divergence > 0.3
    ? `${agent.internal}… ${truncate(agent.primary_concern, 30)}`
    : truncate(agent.primary_concern, 40);
  return (
    <div style={{ position: 'relative', background: 'rgba(241,245,249,0.93)', border: '1px solid rgba(148,163,184,0.3)', borderRadius: '10px 10px 10px 2px', padding: '4px 8px', fontSize: 9, fontFamily: 'Inter, system-ui', color: '#64748b', fontStyle: 'italic', maxWidth: 140, lineHeight: 1.3 }}>
      <div style={{ position: 'absolute', bottom: -5, left: 6, width: 6, height: 6, borderRadius: '50%', background: 'rgba(148,163,184,0.25)' }} />
      <div style={{ position: 'absolute', bottom: -10, left: 2, width: 4, height: 4, borderRadius: '50%', background: 'rgba(148,163,184,0.18)' }} />
      💭 {thought}
    </div>
  );
}

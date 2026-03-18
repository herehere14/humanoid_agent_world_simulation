import React, { useRef, useMemo, useState, Component } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

// ─── Error Boundary ─────────────────────────────────────────────────────────
class Avatar3DErrorBoundary extends Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

// ─── Types ──────────────────────────────────────────────────────────────────
export interface AvatarState {
  confidence: number;
  stress: number;
  curiosity: number;
  fear: number;
  ambition: number;
  empathy: number;
  impulse: number;
  reflection: number;
  trust: number;
  motivation: number;
  fatigue: number;
  frustration: number;
}

type EmotionType =
  | 'neutral' | 'happy' | 'sad' | 'angry' | 'fearful'
  | 'curious' | 'reflective' | 'fatigued' | 'excited'
  | 'empathetic' | 'conflicted' | 'stressed';

interface EmotionPose {
  type: EmotionType;
  label: string;
  color: string;
}

// ─── Emotion Mapper ─────────────────────────────────────────────────────────
function computeEmotion(state: AvatarState): EmotionPose {
  const {
    confidence, stress, curiosity, fear, ambition,
    empathy, impulse, reflection, motivation,
    fatigue, frustration,
  } = state;

  const hasConflict =
    (curiosity > 0.6 && fear > 0.6) ||
    (impulse > 0.6 && reflection > 0.6);

  if (hasConflict) return { type: 'conflicted', label: 'Conflicted', color: '#9333ea' };

  const emotions: [EmotionType, number, string, string][] = [
    ['happy', confidence * 0.5 + motivation * 0.3 + ambition * 0.2, 'Happy', '#059669'],
    ['sad', frustration * 0.5 + stress * 0.3 + (1 - confidence) * 0.2, 'Sad', '#6366f1'],
    ['angry', frustration * 0.6 + impulse * 0.2 + stress * 0.2, 'Angry', '#dc2626'],
    ['fearful', fear * 0.6 + stress * 0.3 + (1 - confidence) * 0.1, 'Fearful', '#e11d48'],
    ['curious', curiosity * 0.6 + motivation * 0.2 + ambition * 0.2, 'Curious', '#0066ff'],
    ['reflective', reflection * 0.7 + (1 - impulse) * 0.3, 'Thinking', '#7c3aed'],
    ['fatigued', fatigue * 0.7 + (1 - motivation) * 0.3, 'Exhausted', '#94a3b8'],
    ['excited', impulse * 0.3 + motivation * 0.3 + ambition * 0.2 + curiosity * 0.2, 'Excited', '#ea580c'],
    ['empathetic', empathy * 0.7 + state.trust * 0.3, 'Caring', '#059669'],
    ['stressed', stress * 0.6 + frustration * 0.2 + fear * 0.2, 'Stressed', '#e11d48'],
  ];

  emotions.sort((a, b) => b[1] - a[1]);
  if (emotions[0][1] < 0.35) return { type: 'neutral', label: 'Neutral', color: '#64748b' };
  return { type: emotions[0][0], label: emotions[0][2], color: emotions[0][3] };
}

// ─── Tear Particle System ───────────────────────────────────────────────────
function TearParticles({ active, side }: { active: boolean; side: 'left' | 'right' }) {
  const particles = useRef<THREE.Group>(null!);
  const tearData = useRef(
    Array.from({ length: 6 }, () => ({
      pos: new THREE.Vector3(),
      vel: new THREE.Vector3(),
      life: 0,
      maxLife: 0,
      size: 0,
    }))
  );

  useFrame((_, delta) => {
    if (!particles.current) return;
    const xDir = side === 'left' ? -1 : 1;

    tearData.current.forEach((tear, i) => {
      const mesh = particles.current.children[i] as THREE.Mesh;
      if (!mesh) return;

      if (active && tear.life <= 0) {
        // Respawn: fly sideways and downward from eye area
        tear.pos.set(0, 0, 0);
        tear.vel.set(
          xDir * (0.8 + Math.random() * 1.2),
          0.5 + Math.random() * 1.0,
          (Math.random() - 0.5) * 0.5
        );
        tear.life = 0.6 + Math.random() * 0.5;
        tear.maxLife = tear.life;
        tear.size = 0.03 + Math.random() * 0.025;
      }

      if (tear.life > 0) {
        tear.life -= delta;
        tear.vel.y -= 4.5 * delta; // gravity
        tear.pos.addScaledVector(tear.vel, delta);

        const progress = 1 - tear.life / tear.maxLife;
        const scale = tear.size * (1 - progress * 0.5);

        mesh.position.copy(tear.pos);
        mesh.scale.setScalar(scale * 10);
        mesh.visible = true;
        (mesh.material as THREE.MeshStandardMaterial).opacity = 1 - progress;
      } else {
        mesh.visible = false;
      }
    });
  });

  return (
    <group ref={particles}>
      {Array.from({ length: 6 }).map((_, i) => (
        <mesh key={i} visible={false}>
          <sphereGeometry args={[0.01, 6, 6]} />
          <meshStandardMaterial color="#7dd3fc" transparent opacity={0.8} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Sweat Drops ────────────────────────────────────────────────────────────
function SweatDrops({ active }: { active: boolean }) {
  const ref = useRef<THREE.Group>(null!);
  const drops = useRef(
    Array.from({ length: 4 }, () => ({ y: 0, speed: 0, visible: false }))
  );

  useFrame((_, delta) => {
    if (!ref.current) return;
    drops.current.forEach((drop, i) => {
      const mesh = ref.current.children[i] as THREE.Mesh;
      if (!mesh) return;

      if (active && !drop.visible && Math.random() < 0.02) {
        drop.y = 0.15;
        drop.speed = 0.3 + Math.random() * 0.4;
        drop.visible = true;
      }

      if (drop.visible) {
        drop.y -= drop.speed * delta;
        if (drop.y < -0.3) drop.visible = false;
        mesh.position.y = drop.y;
        mesh.visible = drop.visible;
      } else {
        mesh.visible = false;
      }
    });
  });

  return (
    <group ref={ref}>
      {Array.from({ length: 4 }).map((_, i) => (
        <mesh key={i} position={[i < 2 ? -0.2 : 0.2, 0, 0.2]} visible={false}>
          <sphereGeometry args={[0.015, 6, 6]} />
          <meshStandardMaterial color="#93c5fd" transparent opacity={0.7} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Anger Steam Particles ──────────────────────────────────────────────────
function AngerSteam({ active }: { active: boolean }) {
  const ref = useRef<THREE.Group>(null!);
  const puffs = useRef(
    Array.from({ length: 8 }, () => ({
      pos: new THREE.Vector3(), vel: new THREE.Vector3(), life: 0, maxLife: 0
    }))
  );

  useFrame((_, delta) => {
    if (!ref.current) return;
    puffs.current.forEach((p, i) => {
      const mesh = ref.current.children[i] as THREE.Mesh;
      if (!mesh) return;

      if (active && p.life <= 0 && Math.random() < 0.08) {
        p.pos.set((Math.random() - 0.5) * 0.15, 0.25, (Math.random() - 0.5) * 0.1);
        p.vel.set((Math.random() - 0.5) * 0.3, 0.8 + Math.random() * 0.5, 0);
        p.life = 0.5 + Math.random() * 0.4;
        p.maxLife = p.life;
      }

      if (p.life > 0) {
        p.life -= delta;
        p.pos.addScaledVector(p.vel, delta);
        const progress = 1 - p.life / p.maxLife;
        mesh.position.copy(p.pos);
        mesh.scale.setScalar(0.03 + progress * 0.06);
        mesh.visible = true;
        (mesh.material as THREE.MeshStandardMaterial).opacity = (1 - progress) * 0.6;
      } else {
        mesh.visible = false;
      }
    });
  });

  return (
    <group ref={ref}>
      {Array.from({ length: 8 }).map((_, i) => (
        <mesh key={i} visible={false}>
          <sphereGeometry args={[1, 6, 6]} />
          <meshStandardMaterial color="#ef4444" transparent opacity={0.5} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Sparkle particles for happy/excited ────────────────────────────────────
function SparkleParticles({ active }: { active: boolean }) {
  const ref = useRef<THREE.Group>(null!);
  const sparkles = useRef(
    Array.from({ length: 10 }, () => ({
      pos: new THREE.Vector3(), vel: new THREE.Vector3(), life: 0, maxLife: 0
    }))
  );

  useFrame((_, delta) => {
    if (!ref.current) return;
    sparkles.current.forEach((s, i) => {
      const mesh = ref.current.children[i] as THREE.Mesh;
      if (!mesh) return;

      if (active && s.life <= 0 && Math.random() < 0.06) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 0.3 + Math.random() * 0.3;
        s.pos.set(Math.cos(angle) * radius, -0.2 + Math.random() * 0.8, Math.sin(angle) * radius);
        s.vel.set(0, 0.5 + Math.random() * 0.5, 0);
        s.life = 0.4 + Math.random() * 0.4;
        s.maxLife = s.life;
      }

      if (s.life > 0) {
        s.life -= delta;
        s.pos.addScaledVector(s.vel, delta);
        const progress = 1 - s.life / s.maxLife;
        mesh.position.copy(s.pos);
        const pulse = 0.02 + Math.sin(progress * Math.PI) * 0.02;
        mesh.scale.setScalar(pulse * 10);
        mesh.visible = true;
        mesh.rotation.z += delta * 5;
        (mesh.material as THREE.MeshStandardMaterial).opacity = Math.sin(progress * Math.PI);
      } else {
        mesh.visible = false;
      }
    });
  });

  return (
    <group ref={ref}>
      {Array.from({ length: 10 }).map((_, i) => (
        <mesh key={i} visible={false}>
          <boxGeometry args={[0.01, 0.01, 0.01]} />
          <meshStandardMaterial color="#fbbf24" transparent emissive="#fbbf24" emissiveIntensity={2} />
        </mesh>
      ))}
    </group>
  );
}

// ─── ZZZ floating letters for fatigue ───────────────────────────────────────
function ZzzBubbles({ active }: { active: boolean }) {
  const ref = useRef<THREE.Group>(null!);
  const bubbles = useRef(
    Array.from({ length: 3 }, (_, i) => ({ y: 0, x: 0, phase: i * 1.2, visible: false, life: 0 }))
  );

  useFrame(({ clock }, delta) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    bubbles.current.forEach((b, i) => {
      const mesh = ref.current.children[i] as THREE.Mesh;
      if (!mesh) return;

      if (active) {
        b.life += delta;
        const cycle = (b.life + b.phase) % 2.5;
        b.y = cycle * 0.25;
        b.x = Math.sin(t * 1.5 + b.phase) * 0.08;
        mesh.position.set(0.25 + b.x + i * 0.08, 0.3 + b.y, 0.15);
        mesh.scale.setScalar(0.04 + i * 0.015);
        mesh.visible = true;
        (mesh.material as THREE.MeshStandardMaterial).opacity = Math.max(0, 1 - cycle / 2.5);
      } else {
        mesh.visible = false;
      }
    });
  });

  return (
    <group ref={ref}>
      {Array.from({ length: 3 }).map((_, i) => (
        <mesh key={i} visible={false}>
          <boxGeometry args={[1, 1, 0.3]} />
          <meshStandardMaterial color="#94a3b8" transparent opacity={0.7} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Question marks for curious ─────────────────────────────────────────────
function QuestionMarks({ active }: { active: boolean }) {
  const ref = useRef<THREE.Group>(null!);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    ref.current.children.forEach((child, i) => {
      const mesh = child as THREE.Mesh;
      if (active) {
        const phase = i * 1.5;
        const bobY = Math.sin(t * 2 + phase) * 0.05;
        mesh.position.y = 0.45 + bobY + i * 0.12;
        mesh.position.x = 0.25 + Math.sin(t + phase) * 0.05;
        mesh.rotation.z = Math.sin(t * 1.5 + phase) * 0.2;
        mesh.visible = true;
        (mesh.material as THREE.MeshStandardMaterial).opacity = 0.5 + Math.sin(t * 2 + phase) * 0.3;
      } else {
        mesh.visible = false;
      }
    });
  });

  return (
    <group ref={ref}>
      {[0, 1].map(i => (
        <mesh key={i} visible={false}>
          <boxGeometry args={[0.04, 0.06, 0.02]} />
          <meshStandardMaterial color="#0066ff" transparent emissive="#0066ff" emissiveIntensity={0.5} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Minecraft-Style Blocky Character ───────────────────────────────────────
function MinecraftCharacter({ emotion }: { emotion: EmotionPose }) {
  // Body part refs
  const rootRef = useRef<THREE.Group>(null!);
  const bodyRef = useRef<THREE.Group>(null!);
  const headRef = useRef<THREE.Group>(null!);
  const leftArmRef = useRef<THREE.Group>(null!);
  const rightArmRef = useRef<THREE.Group>(null!);
  const leftLegRef = useRef<THREE.Group>(null!);
  const rightLegRef = useRef<THREE.Group>(null!);
  // Face refs
  const leftEyeRef = useRef<THREE.Mesh>(null!);
  const rightEyeRef = useRef<THREE.Mesh>(null!);
  const leftPupilRef = useRef<THREE.Mesh>(null!);
  const rightPupilRef = useRef<THREE.Mesh>(null!);
  const mouthRef = useRef<THREE.Group>(null!);
  const leftBrowRef = useRef<THREE.Mesh>(null!);
  const rightBrowRef = useRef<THREE.Mesh>(null!);
  const blushLRef = useRef<THREE.Mesh>(null!);
  const blushRRef = useRef<THREE.Mesh>(null!);

  // Smooth animation state
  const anim = useRef({
    headRot: [0, 0, 0],
    bodyY: 0,
    bodyRotX: 0,
    bodyRotZ: 0,
    leftArmRotX: 0,
    leftArmRotZ: 0.12,
    rightArmRotX: 0,
    rightArmRotZ: -0.12,
    leftLegRotX: 0,
    rightLegRotX: 0,
    eyeScaleY: 1,
    eyeScaleX: 1,
    pupilY: 0,
    browLeftRotZ: 0,
    browRightRotZ: 0,
    browLeftY: 0,
    browRightY: 0,
    mouthScaleX: 1,
    mouthScaleY: 1,
    mouthY: 0,
    blush: 0,
  });

  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const s = 0.08; // smoothing speed
    const a = anim.current;
    const e = emotion.type;

    // ─── Target values per emotion ───────────────────────────────────
    let targetHeadX = 0, targetHeadY = 0, targetHeadZ = 0;
    let targetBodyY = 0, targetBodyRotX = 0, targetBodyRotZ = 0;
    let targetLeftArmX = 0, targetLeftArmZ = 0.12;
    let targetRightArmX = 0, targetRightArmZ = -0.12;
    let targetLeftLegX = 0, targetRightLegX = 0;
    let targetEyeScaleY = 1, targetEyeScaleX = 1;
    let targetPupilY = 0;
    let targetBrowLZ = 0, targetBrowRZ = 0, targetBrowLY = 0, targetBrowRY = 0;
    let targetMouthSX = 1, targetMouthSY = 1, targetMouthY = 0;
    let targetBlush = 0;

    // Dynamic overlays
    let headShake = 0;
    let bodyBounce = 0;
    let bodyShake = 0;
    let armWaveL = 0, armWaveR = 0;
    let legWalk = 0;
    let jumpHeight = 0;

    switch (e) {
      case 'happy':
        // Waving left arm, bouncing, big smile, happy squint, head bobbing
        targetLeftArmX = -2.8;
        targetLeftArmZ = 0.3;
        armWaveL = Math.sin(t * 8) * 0.5;
        targetRightArmZ = -0.15;
        bodyBounce = Math.sin(t * 4) * 0.05;
        legWalk = Math.sin(t * 4) * 0.12;
        targetEyeScaleY = 0.35; // big happy squint — curved happy eyes
        targetEyeScaleX = 1.15;
        targetMouthSX = 2.0; // very wide smile
        targetMouthSY = 0.7;
        targetMouthY = -0.005;
        targetBlush = 0.55;
        targetHeadZ = Math.sin(t * 2) * 0.1;
        targetHeadY = Math.sin(t * 3) * 0.05;
        targetBrowLZ = -0.2;
        targetBrowRZ = 0.2;
        targetBrowLY = 0.015;
        targetBrowRY = 0.015;
        break;

      case 'sad':
        // Both hands up near face, head shaking slowly, tears streaming, sobbing body tremor
        targetLeftArmX = -1.6;
        targetLeftArmZ = 0.8;
        targetRightArmX = -1.6;
        targetRightArmZ = -0.8;
        headShake = Math.sin(t * 2.5) * 0.15;
        targetHeadX = -0.25; // head drooped down
        targetBodyRotX = 0.1;
        bodyShake = Math.sin(t * 6) * 0.004; // sobbing tremor
        targetEyeScaleY = 0.2; // eyes nearly shut, crying
        targetEyeScaleX = 0.9;
        targetMouthSX = 0.5; // small wobbly frown
        targetMouthSY = 0.9;
        targetMouthY = -0.04;
        targetBrowLZ = 0.4; // brows up in center = sad
        targetBrowRZ = -0.4;
        targetBrowLY = 0.03;
        targetBrowRY = 0.03;
        break;

      case 'angry':
        // Fist raised shaking, stomping feet, gritted teeth, veins popping
        targetRightArmX = -2.5;
        targetRightArmZ = -0.3;
        armWaveR = Math.sin(t * 10) * 0.3;
        targetLeftArmX = -0.6;
        targetLeftArmZ = 0.5;
        bodyShake = Math.sin(t * 15) * 0.01;
        targetHeadX = 0.12;
        targetBodyRotX = 0.06;
        targetLeftLegX = Math.sin(t * 6) > 0.7 ? -0.6 : 0;
        targetRightLegX = Math.sin(t * 6 + Math.PI) > 0.7 ? -0.6 : 0;
        targetEyeScaleY = 0.3; // angry squint
        targetEyeScaleX = 1.2; // wide angry
        targetMouthSX = 1.8;
        targetMouthSY = 0.25; // gritted teeth
        targetBrowLZ = -0.5; // brows angled down = angry V shape
        targetBrowRZ = 0.5;
        targetBrowLY = -0.025;
        targetBrowRY = -0.025;
        targetBlush = 0.3; // flushed with anger
        break;

      case 'fearful':
        // Cowering back, arms defensive, wide terrified eyes, trembling
        targetLeftArmX = -2.0;
        targetLeftArmZ = 0.7;
        targetRightArmX = -2.0;
        targetRightArmZ = -0.7;
        bodyShake = Math.sin(t * 20) * 0.015;
        targetHeadX = -0.15;
        targetBodyRotX = -0.12;
        targetBodyRotZ = Math.sin(t * 3) * 0.04;
        targetEyeScaleY = 1.6; // huge terrified eyes
        targetEyeScaleX = 1.4;
        targetPupilY = -0.008; // pupils looking down/away
        targetMouthSX = 0.5;
        targetMouthSY = 1.4; // mouth wide open in shock
        targetBrowLZ = 0.4; // brows raised high
        targetBrowRZ = -0.4;
        targetBrowLY = 0.04;
        targetBrowRY = 0.04;
        legWalk = Math.sin(t * 2.5) * 0.06;
        break;

      case 'curious':
        // Head tilted, hand on chin, one eyebrow raised high, big inquisitive eyes
        targetHeadY = 0.3 + Math.sin(t * 1.5) * 0.06;
        targetHeadZ = 0.18;
        targetRightArmX = -1.8;
        targetRightArmZ = -0.2;
        targetBodyRotX = 0.1;
        targetEyeScaleY = 1.35; // big curious eyes
        targetEyeScaleX = 1.15;
        targetPupilY = Math.sin(t * 2) * 0.005; // eyes scanning
        targetBrowLZ = -0.35; // one brow way up
        targetBrowRZ = 0.05;
        targetBrowLY = 0.03;
        targetBrowRY = 0;
        targetMouthSX = 0.6;
        targetMouthSY = 0.5;
        break;

      case 'reflective':
        // Hand on chin, eyes half-closed, slow peaceful breathing
        targetHeadX = -0.1;
        targetHeadY = -0.2;
        targetRightArmX = -1.5;
        targetRightArmZ = -0.15;
        targetEyeScaleY = 0.5; // half-closed contemplative
        targetEyeScaleX = 0.95;
        targetMouthSX = 0.8;
        targetMouthSY = 0.35;
        bodyBounce = Math.sin(t * 1) * 0.008;
        targetBrowLZ = 0.12;
        targetBrowRZ = -0.12;
        targetBrowLY = 0.005;
        targetBrowRY = 0.005;
        break;

      case 'fatigued':
        // Slouched, droopy everything, head drooping, barely-open eyes, yawning
        targetHeadX = -0.4;
        targetHeadZ = -0.12;
        targetBodyRotX = 0.18;
        targetLeftArmZ = 0.35;
        targetRightArmZ = -0.35;
        targetEyeScaleY = 0.15; // barely open slit
        targetEyeScaleX = 0.9;
        targetMouthSX = 1.2; // yawning
        targetMouthSY = Math.sin(t * 0.5) > 0.3 ? 1.8 : 0.4; // periodic yawn
        targetMouthY = -0.02;
        targetBrowLZ = 0.25;
        targetBrowRZ = -0.25;
        targetBrowLY = -0.01;
        targetBrowRY = -0.01;
        bodyBounce = Math.sin(t * 0.6) * 0.004;
        break;

      case 'excited':
        // Jumping, both arms up pumping, wide sparkly eyes, huge grin
        targetLeftArmX = -2.8;
        targetLeftArmZ = 0.5;
        targetRightArmX = -2.8;
        targetRightArmZ = -0.5;
        armWaveL = Math.sin(t * 6) * 0.25;
        armWaveR = Math.sin(t * 6 + 1) * 0.25;
        jumpHeight = Math.abs(Math.sin(t * 5)) * 0.18;
        targetEyeScaleY = 1.3; // big sparkly eyes
        targetEyeScaleX = 1.2;
        targetMouthSX = 1.6;
        targetMouthSY = 1.5; // big open excited mouth
        targetMouthY = -0.01;
        targetBlush = 0.4;
        targetHeadY = Math.sin(t * 3) * 0.12;
        targetBrowLZ = -0.3;
        targetBrowRZ = 0.3;
        targetBrowLY = 0.03;
        targetBrowRY = 0.03;
        legWalk = Math.sin(t * 10) * 0.25;
        break;

      case 'empathetic':
        // Arms open in hugging gesture, warm soft eyes, gentle head tilt
        targetLeftArmX = -0.8;
        targetLeftArmZ = 1.0;
        targetRightArmX = -0.8;
        targetRightArmZ = -1.0;
        targetBodyRotX = 0.07;
        targetHeadZ = Math.sin(t * 1.5) * 0.08;
        targetEyeScaleY = 0.7; // warm soft eyes
        targetEyeScaleX = 1.05;
        targetMouthSX = 1.5; // warm smile
        targetMouthSY = 0.5;
        targetBlush = 0.5;
        targetBrowLZ = 0.12;
        targetBrowRZ = -0.12;
        targetBrowLY = 0.015;
        targetBrowRY = 0.015;
        bodyBounce = Math.sin(t * 1.5) * 0.01;
        break;

      case 'conflicted':
        // Hands grabbing head, rocking, eyes darting wildly, asymmetric brows
        targetLeftArmX = -2.2;
        targetLeftArmZ = 0.5;
        targetRightArmX = -2.2;
        targetRightArmZ = -0.5;
        bodyShake = Math.sin(t * 4) * 0.035;
        targetBodyRotZ = Math.sin(t * 2) * 0.1;
        headShake = Math.sin(t * 5) * 0.18;
        targetEyeScaleY = 1.15;
        targetEyeScaleX = 1.1;
        targetPupilY = Math.sin(t * 7) * 0.012; // eyes darting wildly
        targetMouthSX = 0.6;
        targetMouthSY = 0.6;
        targetBrowLZ = Math.sin(t * 3) * 0.3; // brows moving independently
        targetBrowRZ = Math.sin(t * 3 + 1.5) * 0.3;
        targetBrowLY = Math.sin(t * 4) * 0.015;
        targetBrowRY = Math.sin(t * 4 + 2) * 0.015;
        legWalk = Math.sin(t * 3) * 0.06;
        break;

      case 'stressed':
        // Hands on head, leg bouncing fast, twitchy eyes, clenched jaw
        targetLeftArmX = -2.5;
        targetLeftArmZ = 0.3;
        targetRightArmX = -2.5;
        targetRightArmZ = -0.3;
        bodyShake = Math.sin(t * 12) * 0.006;
        targetHeadX = -0.12;
        targetBodyRotZ = Math.sin(t * 3) * 0.05;
        targetRightLegX = Math.sin(t * 14) * 0.1; // fast leg bounce
        targetEyeScaleY = 0.6; // tense squint
        targetEyeScaleX = 1.1;
        targetPupilY = Math.sin(t * 8) * 0.004; // twitchy
        targetMouthSX = 1.5;
        targetMouthSY = 0.2; // tightly clenched
        targetBrowLZ = -0.3; // furrowed V
        targetBrowRZ = 0.3;
        targetBrowLY = -0.015;
        targetBrowRY = -0.015;
        break;

      default: // neutral
        // Idle breathing, gentle sway, occasional blink
        bodyBounce = Math.sin(t * 1.5) * 0.008;
        targetHeadY = Math.sin(t * 0.8) * 0.03;
        // Periodic blinking
        const blinkPhase = t % 4;
        if (blinkPhase > 3.7 && blinkPhase < 3.9) {
          targetEyeScaleY = 0.1; // blink
        }
        legWalk = 0;
        break;
    }

    // ─── Apply smooth lerp ──────────────────────────────────────────
    a.headRot[0] = lerp(a.headRot[0], targetHeadX, s);
    a.headRot[1] = lerp(a.headRot[1], targetHeadY + headShake, s);
    a.headRot[2] = lerp(a.headRot[2], targetHeadZ, s);
    a.bodyY = lerp(a.bodyY, targetBodyY + bodyBounce + jumpHeight, s * 1.5);
    a.bodyRotX = lerp(a.bodyRotX, targetBodyRotX, s);
    a.bodyRotZ = lerp(a.bodyRotZ, targetBodyRotZ + bodyShake, s);
    a.leftArmRotX = lerp(a.leftArmRotX, targetLeftArmX + armWaveL, s * 1.2);
    a.leftArmRotZ = lerp(a.leftArmRotZ, targetLeftArmZ, s);
    a.rightArmRotX = lerp(a.rightArmRotX, targetRightArmX + armWaveR, s * 1.2);
    a.rightArmRotZ = lerp(a.rightArmRotZ, targetRightArmZ, s);
    a.leftLegRotX = lerp(a.leftLegRotX, targetLeftLegX + legWalk, s);
    a.rightLegRotX = lerp(a.rightLegRotX, targetRightLegX - legWalk, s);
    a.eyeScaleY = lerp(a.eyeScaleY, targetEyeScaleY, s);
    a.eyeScaleX = lerp(a.eyeScaleX, targetEyeScaleX, s);
    a.pupilY = lerp(a.pupilY, targetPupilY, s);
    a.browLeftRotZ = lerp(a.browLeftRotZ, targetBrowLZ, s);
    a.browRightRotZ = lerp(a.browRightRotZ, targetBrowRZ, s);
    a.browLeftY = lerp(a.browLeftY, targetBrowLY, s);
    a.browRightY = lerp(a.browRightY, targetBrowRY, s);
    a.mouthScaleX = lerp(a.mouthScaleX, targetMouthSX, s);
    a.mouthScaleY = lerp(a.mouthScaleY, targetMouthSY, s);
    a.mouthY = lerp(a.mouthY, targetMouthY, s);
    a.blush = lerp(a.blush, targetBlush, s);

    // ─── Apply to meshes ────────────────────────────────────────────
    if (bodyRef.current) {
      bodyRef.current.position.y = a.bodyY;
      bodyRef.current.rotation.set(a.bodyRotX, 0, a.bodyRotZ);
    }
    if (headRef.current) {
      headRef.current.rotation.set(a.headRot[0], a.headRot[1], a.headRot[2]);
    }
    if (leftArmRef.current) {
      leftArmRef.current.rotation.set(a.leftArmRotX, 0, a.leftArmRotZ);
    }
    if (rightArmRef.current) {
      rightArmRef.current.rotation.set(a.rightArmRotX, 0, a.rightArmRotZ);
    }
    if (leftLegRef.current) {
      leftLegRef.current.rotation.x = a.leftLegRotX;
    }
    if (rightLegRef.current) {
      rightLegRef.current.rotation.x = a.rightLegRotX;
    }

    // Eyes
    if (leftEyeRef.current) {
      leftEyeRef.current.scale.set(a.eyeScaleX, a.eyeScaleY, 1);
    }
    if (rightEyeRef.current) {
      rightEyeRef.current.scale.set(a.eyeScaleX, a.eyeScaleY, 1);
    }
    if (leftPupilRef.current) {
      leftPupilRef.current.position.y = a.pupilY;
    }
    if (rightPupilRef.current) {
      rightPupilRef.current.position.y = a.pupilY;
    }

    // Brows
    if (leftBrowRef.current) {
      leftBrowRef.current.rotation.z = Math.PI / 2 + a.browLeftRotZ;
      leftBrowRef.current.position.y = 0.115 + a.browLeftY;
    }
    if (rightBrowRef.current) {
      rightBrowRef.current.rotation.z = Math.PI / 2 + a.browRightRotZ;
      rightBrowRef.current.position.y = 0.115 + a.browRightY;
    }

    // Mouth
    if (mouthRef.current) {
      mouthRef.current.scale.set(a.mouthScaleX, a.mouthScaleY, 1);
      mouthRef.current.position.y = -0.12 + a.mouthY;
    }

    // Blush
    if (blushLRef.current) {
      (blushLRef.current.material as THREE.MeshStandardMaterial).opacity = a.blush;
    }
    if (blushRRef.current) {
      (blushRRef.current.material as THREE.MeshStandardMaterial).opacity = a.blush;
    }
  });

  // ─── Colors ─────────────────────────────────────────────────────────
  const skinColor = '#ffcc99';
  const hairColor = '#5c3317';
  const shirtColor = emotion.color;
  const pantsColor = '#3b5998';
  const shoeColor = '#333333';

  return (
    <group ref={rootRef} position={[0, -1.0, 0]}>
      <group ref={bodyRef}>

        {/* ═══ TORSO ═══ */}
        <mesh position={[0, 0.55, 0]}>
          <boxGeometry args={[0.5, 0.6, 0.25]} />
          <meshStandardMaterial color={shirtColor} roughness={0.5} />
        </mesh>
        {/* Collar */}
        <mesh position={[0, 0.82, 0.08]}>
          <boxGeometry args={[0.2, 0.05, 0.12]} />
          <meshStandardMaterial color={shirtColor} roughness={0.4} />
        </mesh>

        {/* ═══ HEAD (oversized chibi) ═══ */}
        <group ref={headRef} position={[0, 1.15, 0]}>
          {/* Head block */}
          <mesh>
            <boxGeometry args={[0.55, 0.55, 0.5]} />
            <meshStandardMaterial color={skinColor} roughness={0.5} />
          </mesh>

          {/* Hair top cap — above eyes */}
          <mesh position={[0, 0.22, -0.02]}>
            <boxGeometry args={[0.59, 0.14, 0.54]} />
            <meshStandardMaterial color={hairColor} roughness={0.7} />
          </mesh>
          {/* Short fringe — above forehead */}
          <mesh position={[0, 0.18, 0.24]}>
            <boxGeometry args={[0.5, 0.06, 0.06]} />
            <meshStandardMaterial color={hairColor} roughness={0.7} />
          </mesh>
          {/* Hair sides — back half only */}
          <mesh position={[-0.285, 0.05, -0.08]}>
            <boxGeometry args={[0.04, 0.25, 0.35]} />
            <meshStandardMaterial color={hairColor} roughness={0.7} />
          </mesh>
          <mesh position={[0.285, 0.05, -0.08]}>
            <boxGeometry args={[0.04, 0.25, 0.35]} />
            <meshStandardMaterial color={hairColor} roughness={0.7} />
          </mesh>
          {/* Hair back */}
          <mesh position={[0, 0.02, -0.26]}>
            <boxGeometry args={[0.59, 0.35, 0.04]} />
            <meshStandardMaterial color={hairColor} roughness={0.7} />
          </mesh>

          {/* ─ FACE ─ */}

          {/* Left eye white */}
          <mesh ref={leftEyeRef} position={[-0.12, 0.0, 0.251]}>
            <boxGeometry args={[0.14, 0.12, 0.005]} />
            <meshStandardMaterial color="white" />
          </mesh>
          {/* Left iris */}
          <mesh position={[-0.12, -0.01, 0.255]}>
            <boxGeometry args={[0.09, 0.09, 0.005]} />
            <meshStandardMaterial color="#4a7dbd" />
          </mesh>
          {/* Left pupil */}
          <mesh ref={leftPupilRef} position={[-0.12, -0.01, 0.258]}>
            <boxGeometry args={[0.055, 0.055, 0.005]} />
            <meshStandardMaterial color="#1a1a2e" />
          </mesh>
          {/* Left highlights */}
          <mesh position={[-0.1, 0.015, 0.261]}>
            <boxGeometry args={[0.025, 0.025, 0.003]} />
            <meshStandardMaterial color="white" />
          </mesh>
          <mesh position={[-0.14, -0.025, 0.261]}>
            <boxGeometry args={[0.015, 0.015, 0.003]} />
            <meshStandardMaterial color="white" emissive="white" emissiveIntensity={0.3} />
          </mesh>

          {/* Right eye white */}
          <mesh ref={rightEyeRef} position={[0.12, 0.0, 0.251]}>
            <boxGeometry args={[0.14, 0.12, 0.005]} />
            <meshStandardMaterial color="white" />
          </mesh>
          {/* Right iris */}
          <mesh position={[0.12, -0.01, 0.255]}>
            <boxGeometry args={[0.09, 0.09, 0.005]} />
            <meshStandardMaterial color="#4a7dbd" />
          </mesh>
          {/* Right pupil */}
          <mesh ref={rightPupilRef} position={[0.12, -0.01, 0.258]}>
            <boxGeometry args={[0.055, 0.055, 0.005]} />
            <meshStandardMaterial color="#1a1a2e" />
          </mesh>
          {/* Right highlights */}
          <mesh position={[0.14, 0.015, 0.261]}>
            <boxGeometry args={[0.025, 0.025, 0.003]} />
            <meshStandardMaterial color="white" />
          </mesh>
          <mesh position={[0.1, -0.025, 0.261]}>
            <boxGeometry args={[0.015, 0.015, 0.003]} />
            <meshStandardMaterial color="white" emissive="white" emissiveIntensity={0.3} />
          </mesh>

          {/* Eyelid lines */}
          <mesh position={[-0.12, 0.065, 0.253]}>
            <boxGeometry args={[0.15, 0.008, 0.003]} />
            <meshStandardMaterial color="#d4a574" />
          </mesh>
          <mesh position={[0.12, 0.065, 0.253]}>
            <boxGeometry args={[0.15, 0.008, 0.003]} />
            <meshStandardMaterial color="#d4a574" />
          </mesh>

          {/* Eyebrows */}
          <mesh ref={leftBrowRef} position={[-0.12, 0.115, 0.255]}>
            <boxGeometry args={[0.14, 0.025, 0.005]} />
            <meshStandardMaterial color={hairColor} />
          </mesh>
          <mesh ref={rightBrowRef} position={[0.12, 0.115, 0.255]}>
            <boxGeometry args={[0.14, 0.025, 0.005]} />
            <meshStandardMaterial color={hairColor} />
          </mesh>

          {/* Nose */}
          <mesh position={[0, -0.06, 0.26]}>
            <boxGeometry args={[0.04, 0.03, 0.03]} />
            <meshStandardMaterial color="#e8b78e" roughness={0.5} />
          </mesh>

          {/* Mouth */}
          <group ref={mouthRef} position={[0, -0.12, 0.255]}>
            <mesh>
              <boxGeometry args={[0.12, 0.04, 0.005]} />
              <meshStandardMaterial color="#cc6655" roughness={0.4} />
            </mesh>
            <mesh position={[0, 0.01, -0.002]}>
              <boxGeometry args={[0.08, 0.015, 0.003]} />
              <meshStandardMaterial color="#fff5ee" />
            </mesh>
          </group>

          {/* Blush */}
          <mesh ref={blushLRef} position={[-0.19, -0.05, 0.24]}>
            <boxGeometry args={[0.08, 0.04, 0.005]} />
            <meshStandardMaterial color="#ff9999" transparent opacity={0} />
          </mesh>
          <mesh ref={blushRRef} position={[0.19, -0.05, 0.24]}>
            <boxGeometry args={[0.08, 0.04, 0.005]} />
            <meshStandardMaterial color="#ff9999" transparent opacity={0} />
          </mesh>

          {/* Ears */}
          <mesh position={[-0.29, -0.02, 0.02]}>
            <boxGeometry args={[0.04, 0.08, 0.06]} />
            <meshStandardMaterial color={skinColor} roughness={0.5} />
          </mesh>
          <mesh position={[0.29, -0.02, 0.02]}>
            <boxGeometry args={[0.04, 0.08, 0.06]} />
            <meshStandardMaterial color={skinColor} roughness={0.5} />
          </mesh>

          {/* Particle effects */}
          <group position={[-0.12, -0.02, 0.26]}>
            <TearParticles active={emotion.type === 'sad'} side="left" />
          </group>
          <group position={[0.12, -0.02, 0.26]}>
            <TearParticles active={emotion.type === 'sad'} side="right" />
          </group>
          <SweatDrops active={emotion.type === 'stressed' || emotion.type === 'fearful'} />
          <AngerSteam active={emotion.type === 'angry'} />
          <ZzzBubbles active={emotion.type === 'fatigued'} />
          <QuestionMarks active={emotion.type === 'curious' || emotion.type === 'reflective'} />
        </group>

        {/* ═══ LEFT ARM ═══ */}
        <group ref={leftArmRef} position={[-0.35, 0.75, 0]}>
          <mesh position={[0, -0.15, 0]}>
            <boxGeometry args={[0.12, 0.3, 0.12]} />
            <meshStandardMaterial color={shirtColor} roughness={0.5} />
          </mesh>
          <mesh position={[0, -0.4, 0]}>
            <boxGeometry args={[0.1, 0.22, 0.1]} />
            <meshStandardMaterial color={skinColor} roughness={0.5} />
          </mesh>
          <mesh position={[0, -0.55, 0]}>
            <boxGeometry args={[0.09, 0.09, 0.09]} />
            <meshStandardMaterial color={skinColor} roughness={0.5} />
          </mesh>
        </group>

        {/* ═══ RIGHT ARM ═══ */}
        <group ref={rightArmRef} position={[0.35, 0.75, 0]}>
          <mesh position={[0, -0.15, 0]}>
            <boxGeometry args={[0.12, 0.3, 0.12]} />
            <meshStandardMaterial color={shirtColor} roughness={0.5} />
          </mesh>
          <mesh position={[0, -0.4, 0]}>
            <boxGeometry args={[0.1, 0.22, 0.1]} />
            <meshStandardMaterial color={skinColor} roughness={0.5} />
          </mesh>
          <mesh position={[0, -0.55, 0]}>
            <boxGeometry args={[0.09, 0.09, 0.09]} />
            <meshStandardMaterial color={skinColor} roughness={0.5} />
          </mesh>
        </group>

        {/* ═══ LEFT LEG ═══ */}
        <group ref={leftLegRef} position={[-0.12, 0.22, 0]}>
          <mesh position={[0, -0.15, 0]}>
            <boxGeometry args={[0.14, 0.3, 0.14]} />
            <meshStandardMaterial color={pantsColor} roughness={0.6} />
          </mesh>
          <mesh position={[0, -0.38, 0]}>
            <boxGeometry args={[0.12, 0.2, 0.12]} />
            <meshStandardMaterial color={pantsColor} roughness={0.6} />
          </mesh>
          <mesh position={[0, -0.52, 0.02]}>
            <boxGeometry args={[0.13, 0.08, 0.18]} />
            <meshStandardMaterial color={shoeColor} roughness={0.7} />
          </mesh>
        </group>

        {/* ═══ RIGHT LEG ═══ */}
        <group ref={rightLegRef} position={[0.12, 0.22, 0]}>
          <mesh position={[0, -0.15, 0]}>
            <boxGeometry args={[0.14, 0.3, 0.14]} />
            <meshStandardMaterial color={pantsColor} roughness={0.6} />
          </mesh>
          <mesh position={[0, -0.38, 0]}>
            <boxGeometry args={[0.12, 0.2, 0.12]} />
            <meshStandardMaterial color={pantsColor} roughness={0.6} />
          </mesh>
          <mesh position={[0, -0.52, 0.02]}>
            <boxGeometry args={[0.13, 0.08, 0.18]} />
            <meshStandardMaterial color={shoeColor} roughness={0.7} />
          </mesh>
        </group>
      </group>

      {/* Sparkle particles */}
      <SparkleParticles active={emotion.type === 'happy' || emotion.type === 'excited'} />
    </group>
  );
}

// ─── Scene ──────────────────────────────────────────────────────────────────
function Scene({ state }: { state: AvatarState }) {
  const emotion = useMemo(() => computeEmotion(state), [state]);
  const emotionColor = useMemo(() => new THREE.Color(emotion.color), [emotion.color]);

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[3, 5, 4]} intensity={0.9} castShadow />
      <directionalLight position={[-2, 3, -2]} intensity={0.3} color="#c4b5fd" />
      <pointLight position={[0, 3, 2]} intensity={0.4} color={emotion.color} />
      {/* Ground glow matching emotion */}
      <pointLight position={[0, -1, 0]} intensity={0.2} color={emotion.color} />

      <MinecraftCharacter emotion={emotion} />

      <ContactShadows
        position={[0, -1.35, 0]}
        opacity={0.3}
        scale={3}
        blur={3}
        far={3}
      />

      {/* Hemisphere light for ambient fill instead of Environment HDR */}
      <hemisphereLight args={['#ffffff', '#e0e7ff', 0.6]} />

      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={1.5}
        maxDistance={6}
        minPolarAngle={Math.PI * 0.15}
        maxPolarAngle={Math.PI * 0.8}
        autoRotate={false}
        target={[0, 0, 0]}
      />
    </>
  );
}

// ─── Exported Component ─────────────────────────────────────────────────────
export function AgentAvatar3D({
  state,
  size = 400,
  className = '',
}: {
  state: AvatarState;
  size?: number;
  className?: string;
}) {
  const emotion = useMemo(() => computeEmotion(state), [state]);

  const fallback = (
    <div
      className={`flex items-center justify-center bg-surface rounded-2xl border border-border ${className}`}
      style={{ width: size, height: size }}
    >
      <div className="text-center text-text-muted text-sm">
        <div className="text-2xl mb-2">🤖</div>
        <div className="font-medium" style={{ color: emotion.color }}>{emotion.label}</div>
        <div className="text-[10px] mt-1">3D preview unavailable</div>
      </div>
    </div>
  );

  return (
    <Avatar3DErrorBoundary fallback={fallback}>
      <div className={`relative ${className}`} style={{ width: size, height: size }}>
        {/* Emotion label */}
        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10">
          <span
            className="text-[11px] font-semibold px-3 py-1 rounded-full border backdrop-blur-sm"
            style={{
              color: emotion.color,
              backgroundColor: emotion.color + '15',
              borderColor: emotion.color + '30',
            }}
          >
            {emotion.label}
          </span>
        </div>

        <Canvas
          camera={{ position: [0, 0.5, 3], fov: 35 }}
          shadows
          dpr={[1, 2]}
          style={{ borderRadius: 16 }}
        >
          <Scene state={state} />
        </Canvas>

        {/* Hint */}
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[9px] text-text-muted opacity-60">
          Drag to rotate · Scroll to zoom
        </div>
      </div>
    </Avatar3DErrorBoundary>
  );
}

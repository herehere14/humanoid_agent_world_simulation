import { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

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

interface EmotionProfile {
  label: string;
  emoji: string;
  // Face
  eyeScale: number;
  pupilY: number;
  browAngle: number;
  browY: number;
  mouthPath: string;
  mouthColor: string;
  blush: number;
  // Body
  bodyTilt: number;
  shoulderY: number;
  // Arms
  leftArmAngle: number;
  rightArmAngle: number;
  leftHandY: number;
  rightHandY: number;
  // Posture
  headTilt: number;
  headY: number;
  // Aura
  auraColor: string;
  auraOpacity: number;
  auraPulse: boolean;
  // Extras
  sweatDrops: boolean;
  sparkles: boolean;
  exclamation: boolean;
  question: boolean;
  zzz: boolean;
  heartEyes: boolean;
  fireAura: boolean;
  thinkBubble: boolean;
  conflictSwirl: boolean;
}

// ─── Emotion mapping ────────────────────────────────────────────────────────
function computeEmotion(state: AvatarState): EmotionProfile {
  const {
    confidence, stress, curiosity, fear, ambition,
    empathy, impulse, reflection, trust, motivation,
    fatigue, frustration,
  } = state;

  // Determine dominant emotion
  const emotions: [string, number][] = [
    ['confident', confidence],
    ['stressed', stress],
    ['curious', curiosity],
    ['fearful', fear],
    ['ambitious', ambition],
    ['empathetic', empathy],
    ['impulsive', impulse],
    ['reflective', reflection],
    ['trusting', trust],
    ['motivated', motivation],
    ['fatigued', fatigue],
    ['frustrated', frustration],
  ];

  // Check for conflict state (two opposing drives both high)
  const hasConflict =
    (curiosity > 0.6 && fear > 0.6) ||
    (impulse > 0.6 && reflection > 0.6) ||
    (ambition > 0.6 && state.frustration > 0.6);

  const sorted = [...emotions].sort((a, b) => b[1] - a[1]);
  const dominant = sorted[0][0];
  const intensity = sorted[0][1];

  // Base profile
  const base: EmotionProfile = {
    label: 'Neutral',
    emoji: '',
    eyeScale: 1,
    pupilY: 0,
    browAngle: 0,
    browY: 0,
    mouthPath: 'M 85,175 Q 100,178 115,175', // neutral line
    mouthColor: '#e88b7a',
    blush: 0,
    bodyTilt: 0,
    shoulderY: 0,
    leftArmAngle: 15,
    rightArmAngle: -15,
    leftHandY: 0,
    rightHandY: 0,
    headTilt: 0,
    headY: 0,
    auraColor: '#0066ff',
    auraOpacity: 0.08,
    auraPulse: false,
    sweatDrops: false,
    sparkles: false,
    exclamation: false,
    question: false,
    zzz: false,
    heartEyes: false,
    fireAura: false,
    thinkBubble: false,
    conflictSwirl: false,
  };

  if (hasConflict) {
    return {
      ...base,
      label: 'Conflicted',
      eyeScale: 1.1,
      pupilY: -1,
      browAngle: -5,
      browY: -2,
      mouthPath: 'M 85,178 Q 95,172 100,178 Q 105,172 115,178',
      bodyTilt: -2,
      leftArmAngle: 35,
      rightArmAngle: -35,
      headTilt: -5,
      auraColor: '#9333ea',
      auraOpacity: 0.15,
      auraPulse: true,
      conflictSwirl: true,
    };
  }

  switch (dominant) {
    case 'confident':
      return {
        ...base,
        label: 'Confident',
        eyeScale: 1.05,
        browAngle: -3,
        browY: -1,
        mouthPath: 'M 85,173 Q 100,185 115,173',
        blush: 0.15 * intensity,
        leftArmAngle: 30 + intensity * 20,
        rightArmAngle: -30 - intensity * 20,
        headTilt: 3,
        headY: -2 * intensity,
        auraColor: '#0891b2',
        auraOpacity: 0.12 * intensity,
        sparkles: intensity > 0.7,
      };

    case 'stressed':
      return {
        ...base,
        label: 'Stressed',
        eyeScale: 1.15,
        pupilY: -1,
        browAngle: 8 * intensity,
        browY: -3 * intensity,
        mouthPath: 'M 85,178 Q 100,172 115,178',
        shoulderY: -3 * intensity,
        leftArmAngle: 5,
        rightArmAngle: -5,
        bodyTilt: -1,
        headY: 1,
        auraColor: '#e11d48',
        auraOpacity: 0.1 * intensity,
        auraPulse: intensity > 0.6,
        sweatDrops: intensity > 0.5,
      };

    case 'curious':
      return {
        ...base,
        label: 'Curious',
        eyeScale: 1.2 + intensity * 0.15,
        pupilY: -2,
        browAngle: -6,
        browY: -4 * intensity,
        mouthPath: 'M 90,175 Q 100,178 110,175',
        bodyTilt: 4 * intensity,
        leftArmAngle: 10,
        rightArmAngle: -50 * intensity,
        headTilt: 8 * intensity,
        headY: -3 * intensity,
        auraColor: '#0066ff',
        auraOpacity: 0.12 * intensity,
        question: intensity > 0.6,
        sparkles: intensity > 0.75,
      };

    case 'fearful':
      return {
        ...base,
        label: 'Fearful',
        eyeScale: 1.3 + intensity * 0.2,
        pupilY: 1,
        browAngle: 10 * intensity,
        browY: -5 * intensity,
        mouthPath: 'M 90,178 Q 100,182 110,178',
        mouthColor: '#d4a0a0',
        shoulderY: 3 * intensity,
        bodyTilt: -3 * intensity,
        leftArmAngle: -10,
        rightArmAngle: 10,
        leftHandY: -10 * intensity,
        rightHandY: -10 * intensity,
        headTilt: -4 * intensity,
        headY: 3 * intensity,
        auraColor: '#dc2626',
        auraOpacity: 0.08 * intensity,
        auraPulse: intensity > 0.7,
        sweatDrops: true,
        exclamation: intensity > 0.7,
      };

    case 'ambitious':
      return {
        ...base,
        label: 'Ambitious',
        eyeScale: 1.05,
        browAngle: -5 * intensity,
        browY: -2,
        mouthPath: 'M 85,174 Q 100,182 115,174',
        leftArmAngle: 60 * intensity,
        rightArmAngle: -20,
        rightHandY: -15 * intensity,
        headTilt: 2,
        headY: -4 * intensity,
        bodyTilt: 2,
        auraColor: '#d97706',
        auraOpacity: 0.15 * intensity,
        fireAura: intensity > 0.7,
        sparkles: intensity > 0.6,
      };

    case 'empathetic':
      return {
        ...base,
        label: 'Empathetic',
        eyeScale: 1.05,
        pupilY: 1,
        browAngle: 3,
        browY: 1,
        mouthPath: 'M 87,174 Q 100,182 113,174',
        blush: 0.25 * intensity,
        bodyTilt: 3 * intensity,
        leftArmAngle: 40 * intensity,
        rightArmAngle: -40 * intensity,
        headTilt: 5 * intensity,
        auraColor: '#059669',
        auraOpacity: 0.12 * intensity,
        heartEyes: intensity > 0.75,
      };

    case 'impulsive':
      return {
        ...base,
        label: 'Impulsive',
        eyeScale: 1.15,
        browAngle: -4,
        browY: -2,
        mouthPath: 'M 85,173 Q 100,186 115,173',
        bodyTilt: 6 * intensity,
        leftArmAngle: 70 * intensity,
        rightArmAngle: -60 * intensity,
        headTilt: 4,
        headY: -3,
        auraColor: '#ea580c',
        auraOpacity: 0.15 * intensity,
        auraPulse: true,
        exclamation: intensity > 0.5,
        fireAura: intensity > 0.6,
      };

    case 'reflective':
      return {
        ...base,
        label: 'Reflective',
        eyeScale: 0.9,
        pupilY: -2,
        browAngle: 2,
        mouthPath: 'M 88,176 Q 100,177 112,176',
        bodyTilt: -2,
        leftArmAngle: 10,
        rightArmAngle: -55 * intensity,
        rightHandY: -20 * intensity,
        headTilt: -6 * intensity,
        headY: 2,
        auraColor: '#7c3aed',
        auraOpacity: 0.1 * intensity,
        thinkBubble: intensity > 0.5,
      };

    case 'trusting':
      return {
        ...base,
        label: 'Trusting',
        eyeScale: 1.05,
        browAngle: 2,
        mouthPath: 'M 87,174 Q 100,183 113,174',
        blush: 0.2 * intensity,
        leftArmAngle: 45 * intensity,
        rightArmAngle: -45 * intensity,
        headTilt: 3,
        bodyTilt: 2,
        auraColor: '#0d9488',
        auraOpacity: 0.1 * intensity,
      };

    case 'motivated':
      return {
        ...base,
        label: 'Motivated',
        eyeScale: 1.1,
        browAngle: -5,
        browY: -2,
        mouthPath: 'M 84,173 Q 100,187 116,173',
        leftArmAngle: 55 * intensity,
        rightArmAngle: -55 * intensity,
        leftHandY: -10 * intensity,
        rightHandY: -10 * intensity,
        headY: -4 * intensity,
        bodyTilt: 1,
        auraColor: '#2563eb',
        auraOpacity: 0.15 * intensity,
        sparkles: true,
        auraPulse: true,
      };

    case 'fatigued':
      return {
        ...base,
        label: 'Fatigued',
        eyeScale: 0.7 - intensity * 0.15,
        pupilY: 2,
        browAngle: 5 * intensity,
        browY: 3 * intensity,
        mouthPath: 'M 88,178 Q 100,175 112,178',
        shoulderY: 5 * intensity,
        bodyTilt: -3 * intensity,
        leftArmAngle: 5,
        rightArmAngle: -5,
        headTilt: -8 * intensity,
        headY: 6 * intensity,
        auraColor: '#94a3b8',
        auraOpacity: 0.06,
        zzz: intensity > 0.5,
      };

    case 'frustrated':
      return {
        ...base,
        label: 'Frustrated',
        eyeScale: 0.85,
        browAngle: -10 * intensity,
        browY: -4 * intensity,
        mouthPath: 'M 85,180 Q 100,172 115,180',
        mouthColor: '#d45a5a',
        shoulderY: -2,
        bodyTilt: -2,
        leftArmAngle: -15,
        rightArmAngle: 15,
        headTilt: -3,
        headY: 2,
        auraColor: '#be123c',
        auraOpacity: 0.12 * intensity,
        auraPulse: true,
        exclamation: intensity > 0.6,
        sweatDrops: intensity > 0.5,
      };

    default:
      return base;
  }
}

// ─── Sub-components ─────────────────────────────────────────────────────────

function SweatDrop({ x, delay }: { x: number; delay: number }) {
  return (
    <motion.ellipse
      cx={x} cy={110}
      rx={2} ry={3}
      fill="#7dd3fc"
      initial={{ opacity: 0, y: -5 }}
      animate={{ opacity: [0, 0.8, 0], y: [0, 12] }}
      transition={{ duration: 1.2, repeat: Infinity, delay, ease: 'easeIn' }}
    />
  );
}

function Sparkle({ x, y, delay }: { x: number; y: number; delay: number }) {
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0 }}
      animate={{ opacity: [0, 1, 0], scale: [0, 1, 0], rotate: [0, 180] }}
      transition={{ duration: 1.5, repeat: Infinity, delay }}
    >
      <line x1={x - 4} y1={y} x2={x + 4} y2={y} stroke="#fbbf24" strokeWidth={1.5} strokeLinecap="round" />
      <line x1={x} y1={y - 4} x2={x} y2={y + 4} stroke="#fbbf24" strokeWidth={1.5} strokeLinecap="round" />
      <line x1={x - 3} y1={y - 3} x2={x + 3} y2={y + 3} stroke="#fbbf24" strokeWidth={1} strokeLinecap="round" />
      <line x1={x + 3} y1={y - 3} x2={x - 3} y2={y + 3} stroke="#fbbf24" strokeWidth={1} strokeLinecap="round" />
    </motion.g>
  );
}

function ZzzBubble() {
  return (
    <g>
      {[0, 1, 2].map(i => (
        <motion.text
          key={i}
          x={135 + i * 10}
          y={105 - i * 12}
          fontSize={10 + i * 3}
          fontWeight="bold"
          fill="#94a3b8"
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: [0, 0.7, 0], y: [-2, -8] }}
          transition={{ duration: 2, repeat: Infinity, delay: i * 0.5 }}
        >
          Z
        </motion.text>
      ))}
    </g>
  );
}

function ThinkBubble() {
  return (
    <g>
      <motion.circle cx={140} cy={100} r={3} fill="#c4b5fd" animate={{ opacity: [0.3, 0.7, 0.3] }} transition={{ duration: 2, repeat: Infinity }} />
      <motion.circle cx={148} cy={90} r={4} fill="#c4b5fd" animate={{ opacity: [0.4, 0.8, 0.4] }} transition={{ duration: 2, repeat: Infinity, delay: 0.3 }} />
      <motion.ellipse cx={160} cy={78} rx={12} ry={9} fill="#c4b5fd" fillOpacity={0.3} stroke="#c4b5fd" strokeWidth={1} animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 2, repeat: Infinity, delay: 0.5 }} />
      <motion.text x={154} y={82} fontSize={10} fill="#7c3aed" animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}>?</motion.text>
    </g>
  );
}

function ExclamationMark() {
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0, y: 5 }}
      animate={{ opacity: 1, scale: [1, 1.2, 1], y: 0 }}
      transition={{ duration: 0.6, repeat: Infinity, repeatDelay: 1 }}
    >
      <text x={130} y={95} fontSize={18} fontWeight="bold" fill="#ef4444">!</text>
    </motion.g>
  );
}

function QuestionMark() {
  return (
    <motion.g
      animate={{ y: [0, -3, 0], rotate: [0, 5, -5, 0] }}
      transition={{ duration: 2, repeat: Infinity }}
    >
      <text x={128} y={95} fontSize={18} fontWeight="bold" fill="#0066ff">?</text>
    </motion.g>
  );
}

function ConflictSwirl() {
  return (
    <g>
      {[0, 120, 240].map((angle, i) => (
        <motion.circle
          key={i}
          cx={100}
          cy={145}
          r={35}
          fill="none"
          stroke="#9333ea"
          strokeWidth={1}
          strokeDasharray="4 8"
          strokeOpacity={0.3}
          initial={{ rotate: angle }}
          animate={{ rotate: angle + 360 }}
          transition={{ duration: 4 + i, repeat: Infinity, ease: 'linear' }}
          style={{ transformOrigin: '100px 145px' }}
        />
      ))}
    </g>
  );
}

function FireAura() {
  return (
    <g>
      {[70, 85, 100, 115, 130].map((x, i) => (
        <motion.path
          key={i}
          d={`M ${x},290 Q ${x - 3},${270 - i * 5} ${x + 2},${260 - i * 3} Q ${x + 5},${270 - i * 5} ${x},290`}
          fill="#f97316"
          fillOpacity={0.15}
          animate={{
            d: [
              `M ${x},290 Q ${x - 3},${270 - i * 5} ${x + 2},${260 - i * 3} Q ${x + 5},${270 - i * 5} ${x},290`,
              `M ${x},290 Q ${x - 5},${265 - i * 5} ${x + 3},${255 - i * 3} Q ${x + 7},${265 - i * 5} ${x},290`,
              `M ${x},290 Q ${x - 3},${270 - i * 5} ${x + 2},${260 - i * 3} Q ${x + 5},${270 - i * 5} ${x},290`,
            ],
            opacity: [0.15, 0.25, 0.15],
          }}
          transition={{ duration: 0.8 + i * 0.2, repeat: Infinity }}
        />
      ))}
    </g>
  );
}

// ─── Main Avatar Component ──────────────────────────────────────────────────
export function AgentAvatar({ state, size = 300 }: { state: AvatarState; size?: number }) {
  const emotion = useMemo(() => computeEmotion(state), [state]);
  const spring = { type: 'spring' as const, stiffness: 120, damping: 15 };

  const skinColor = '#fddcb5';
  const skinShadow = '#f0c090';
  const hairColor = '#4a3728';
  const shirtColor = emotion.auraColor;
  const pantsColor = '#374151';
  const shoeColor = '#1f2937';

  return (
    <div className="relative" style={{ width: size, height: size }}>
      {/* Emotion label */}
      <motion.div
        className="absolute -top-1 left-1/2 -translate-x-1/2 z-10"
        key={emotion.label}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <span
          className="text-[11px] font-semibold px-3 py-1 rounded-full border"
          style={{
            color: emotion.auraColor,
            backgroundColor: emotion.auraColor + '10',
            borderColor: emotion.auraColor + '25',
          }}
        >
          {emotion.label}
        </span>
      </motion.div>

      <svg viewBox="0 0 200 310" fill="none" style={{ width: '100%', height: '100%' }}>
        <defs>
          {/* Body gradient */}
          <linearGradient id="av-shirt" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={shirtColor} stopOpacity={0.9} />
            <stop offset="100%" stopColor={shirtColor} stopOpacity={0.7} />
          </linearGradient>
          {/* Skin gradient */}
          <radialGradient id="av-skin" cx="0.4" cy="0.35">
            <stop offset="0%" stopColor={skinColor} />
            <stop offset="100%" stopColor={skinShadow} />
          </radialGradient>
          {/* Aura glow */}
          <radialGradient id="av-aura">
            <stop offset="0%" stopColor={emotion.auraColor} stopOpacity={emotion.auraOpacity} />
            <stop offset="100%" stopColor={emotion.auraColor} stopOpacity={0} />
          </radialGradient>
          {/* Shadow */}
          <radialGradient id="av-shadow" cx="0.5" cy="0.5">
            <stop offset="0%" stopColor="#000" stopOpacity={0.08} />
            <stop offset="100%" stopColor="#000" stopOpacity={0} />
          </radialGradient>
        </defs>

        {/* Ground shadow */}
        <motion.ellipse
          cx={100} cy={298}
          rx={35} ry={5}
          fill="url(#av-shadow)"
          animate={{ rx: emotion.bodyTilt !== 0 ? 30 : 35 }}
          transition={spring}
        />

        {/* Aura */}
        <motion.circle
          cx={100} cy={165}
          r={80}
          fill="url(#av-aura)"
          animate={{
            r: emotion.auraPulse ? [78, 85, 78] : 80,
            opacity: emotion.auraOpacity > 0 ? 1 : 0,
          }}
          transition={emotion.auraPulse ? { duration: 1.5, repeat: Infinity } : spring}
        />

        {/* Effects behind character */}
        {emotion.fireAura && <FireAura />}
        {emotion.conflictSwirl && <ConflictSwirl />}

        {/* ── BODY ── */}
        <motion.g
          animate={{ rotate: emotion.bodyTilt, y: 0 }}
          transition={spring}
          style={{ transformOrigin: '100px 250px' }}
        >
          {/* Legs */}
          <motion.g animate={{ y: 0 }} transition={spring}>
            {/* Left leg */}
            <rect x={82} y={248} width={14} height={35} rx={6} fill={pantsColor} />
            <rect x={80} y={278} width={18} height={10} rx={5} fill={shoeColor} />
            {/* Right leg */}
            <rect x={104} y={248} width={14} height={35} rx={6} fill={pantsColor} />
            <rect x={102} y={278} width={18} height={10} rx={5} fill={shoeColor} />
          </motion.g>

          {/* Torso */}
          <motion.g
            animate={{ y: emotion.shoulderY }}
            transition={spring}
          >
            {/* Body */}
            <rect x={72} y={195} width={56} height={58} rx={16} fill="url(#av-shirt)" />
            {/* Collar detail */}
            <path d="M 88,195 Q 100,205 112,195" stroke={shirtColor} strokeWidth={2} fill="none" strokeOpacity={0.5} />

            {/* Left arm */}
            <motion.g
              animate={{ rotate: emotion.leftArmAngle }}
              transition={spring}
              style={{ transformOrigin: '72px 205px' }}
            >
              <rect x={52} y={200} width={22} height={14} rx={7} fill="url(#av-shirt)" />
              <motion.circle
                cx={52} cy={207 + emotion.leftHandY}
                r={8}
                fill="url(#av-skin)"
                animate={{ cy: 207 + emotion.leftHandY }}
                transition={spring}
              />
            </motion.g>

            {/* Right arm */}
            <motion.g
              animate={{ rotate: emotion.rightArmAngle }}
              transition={spring}
              style={{ transformOrigin: '128px 205px' }}
            >
              <rect x={126} y={200} width={22} height={14} rx={7} fill="url(#av-shirt)" />
              <motion.circle
                cx={148} cy={207 + emotion.rightHandY}
                r={8}
                fill="url(#av-skin)"
                animate={{ cy: 207 + emotion.rightHandY }}
                transition={spring}
              />
            </motion.g>
          </motion.g>

          {/* ── HEAD ── */}
          <motion.g
            animate={{
              rotate: emotion.headTilt,
              y: emotion.headY,
            }}
            transition={spring}
            style={{ transformOrigin: '100px 145px' }}
          >
            {/* Neck */}
            <rect x={93} y={185} width={14} height={16} rx={5} fill="url(#av-skin)" />

            {/* Head shape */}
            <ellipse cx={100} cy={145} rx={38} ry={42} fill="url(#av-skin)" />

            {/* Hair */}
            <ellipse cx={100} cy={115} rx={40} ry={22} fill={hairColor} />
            <ellipse cx={100} cy={108} rx={35} ry={16} fill={hairColor} />
            {/* Side hair */}
            <ellipse cx={64} cy={135} rx={8} ry={18} fill={hairColor} />
            <ellipse cx={136} cy={135} rx={8} ry={18} fill={hairColor} />

            {/* Ears */}
            <ellipse cx={62} cy={148} rx={6} ry={8} fill={skinShadow} />
            <ellipse cx={138} cy={148} rx={6} ry={8} fill={skinShadow} />

            {/* Blush */}
            {emotion.blush > 0 && (
              <>
                <motion.ellipse cx={78} cy={160} rx={10} ry={5} fill="#f87171" fillOpacity={emotion.blush} animate={{ fillOpacity: [emotion.blush, emotion.blush * 1.3, emotion.blush] }} transition={{ duration: 2, repeat: Infinity }} />
                <motion.ellipse cx={122} cy={160} rx={10} ry={5} fill="#f87171" fillOpacity={emotion.blush} animate={{ fillOpacity: [emotion.blush, emotion.blush * 1.3, emotion.blush] }} transition={{ duration: 2, repeat: Infinity }} />
              </>
            )}

            {/* Eyes */}
            <motion.g animate={{ scaleY: emotion.eyeScale }} transition={spring} style={{ transformOrigin: '100px 148px' }}>
              {emotion.heartEyes ? (
                <>
                  <motion.text x={78} y={155} fontSize={14} animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 1, repeat: Infinity }}>
                    ❤️
                  </motion.text>
                  <motion.text x={108} y={155} fontSize={14} animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}>
                    ❤️
                  </motion.text>
                </>
              ) : (
                <>
                  {/* Left eye */}
                  <ellipse cx={86} cy={148} rx={8} ry={9} fill="white" />
                  <motion.circle cx={86} cy={148 + emotion.pupilY} r={4.5} fill="#1a1a2e" animate={{ cy: 148 + emotion.pupilY }} transition={spring} />
                  <circle cx={84} cy={146 + emotion.pupilY} r={1.5} fill="white" />

                  {/* Right eye */}
                  <ellipse cx={114} cy={148} rx={8} ry={9} fill="white" />
                  <motion.circle cx={114} cy={148 + emotion.pupilY} r={4.5} fill="#1a1a2e" animate={{ cy: 148 + emotion.pupilY }} transition={spring} />
                  <circle cx={112} cy={146 + emotion.pupilY} r={1.5} fill="white" />
                </>
              )}
            </motion.g>

            {/* Eyebrows */}
            <motion.line
              x1={78} y1={136}
              x2={94} y2={136}
              stroke={hairColor}
              strokeWidth={2.5}
              strokeLinecap="round"
              animate={{
                y1: 136 + emotion.browY,
                y2: 136 + emotion.browY - Math.sin(emotion.browAngle * Math.PI / 180) * 6,
              }}
              transition={spring}
            />
            <motion.line
              x1={106} y1={136}
              x2={122} y2={136}
              stroke={hairColor}
              strokeWidth={2.5}
              strokeLinecap="round"
              animate={{
                y1: 136 + emotion.browY - Math.sin(-emotion.browAngle * Math.PI / 180) * 6,
                y2: 136 + emotion.browY,
              }}
              transition={spring}
            />

            {/* Mouth */}
            <motion.path
              d={emotion.mouthPath}
              stroke={emotion.mouthColor}
              strokeWidth={2.5}
              strokeLinecap="round"
              fill="none"
              animate={{ d: emotion.mouthPath }}
              transition={spring}
            />

            {/* Nose */}
            <ellipse cx={100} cy={162} rx={3} ry={2} fill={skinShadow} fillOpacity={0.5} />
          </motion.g>
        </motion.g>

        {/* ── EFFECTS ── */}
        <AnimatePresence>
          {emotion.sweatDrops && (
            <g>
              <SweatDrop x={65} delay={0} />
              <SweatDrop x={135} delay={0.6} />
            </g>
          )}
        </AnimatePresence>

        {emotion.sparkles && (
          <g>
            <Sparkle x={55} y={120} delay={0} />
            <Sparkle x={145} y={130} delay={0.5} />
            <Sparkle x={70} y={190} delay={1} />
            <Sparkle x={130} y={100} delay={1.5} />
          </g>
        )}

        {emotion.zzz && <ZzzBubble />}
        {emotion.thinkBubble && <ThinkBubble />}
        {emotion.exclamation && <ExclamationMark />}
        {emotion.question && <QuestionMark />}
      </svg>
    </div>
  );
}

// ─── Compact Avatar for inline use ──────────────────────────────────────────
export function AvatarEmotionBadge({ state }: { state: AvatarState }) {
  const emotion = useMemo(() => computeEmotion(state), [state]);

  return (
    <div className="flex items-center gap-2">
      <div className="w-10 h-10">
        <AgentAvatar state={state} size={40} />
      </div>
      <span
        className="text-xs font-semibold px-2 py-0.5 rounded-full"
        style={{
          color: emotion.auraColor,
          backgroundColor: emotion.auraColor + '12',
        }}
      >
        {emotion.label}
      </span>
    </div>
  );
}

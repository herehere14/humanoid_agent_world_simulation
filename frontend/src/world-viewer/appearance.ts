/**
 * Procedural agent appearance — deterministic from agent ID.
 *
 * Generates unique skin tone, hair color, hair style, shirt color,
 * pants color, shoe color, and body proportions for each agent so
 * no two look the same but the look is stable across reloads.
 */

// Seeded hash from agent id
function hash(str: string): number {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h + str.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function seeded(id: string, salt: number): number {
  const h = hash(id + salt);
  return (h % 10000) / 10000; // 0-1
}

export interface AgentAppearance {
  skin: string;
  skinDark: string;  // slightly darker for forearms
  hair: string;
  hairStyle: 'short' | 'medium' | 'long' | 'bald' | 'mohawk';
  shirt: string;
  pants: string;
  shoes: string;
  height: number;    // multiplier 0.88–1.12
  bulk: number;      // torso width multiplier 0.85–1.15
}

// Palettes
const SKIN_TONES = [
  { base: '#FDDCB5', dark: '#D4A574' }, // light
  { base: '#F5C6A0', dark: '#C8946A' }, // peach
  { base: '#E8B88A', dark: '#C09060' }, // medium
  { base: '#D4956A', dark: '#A06830' }, // tan
  { base: '#C68642', dark: '#8B5E2B' }, // brown
  { base: '#8D5524', dark: '#5C3310' }, // dark brown
  { base: '#6B3A1F', dark: '#3D1F0A' }, // deep
  { base: '#4A2511', dark: '#2A1508' }, // very dark
];

const HAIR_COLORS = [
  '#1a1a1a', // black
  '#2c1a0e', // dark brown
  '#5c3a1e', // brown
  '#8B6914', // light brown
  '#c4842d', // auburn
  '#d4a04a', // blonde
  '#e8c86a', // light blonde
  '#b03010', // red
  '#8a8a8a', // grey
  '#f0f0f0', // white
];

const SHIRT_COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#22c55e', // green
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#1e293b', // navy
  '#6b7280', // grey
  '#854d0e', // brown
  '#166534', // forest
  '#7c2d12', // rust
  '#4338ca', // indigo
  '#be185d', // magenta
  '#0d9488', // teal
  '#ffffff', // white
  '#fbbf24', // yellow
];

const PANTS_COLORS = [
  '#1e293b', // dark navy
  '#374151', // charcoal
  '#4b5563', // grey
  '#78350f', // brown
  '#1e3a5f', // dark blue
  '#292524', // almost black
  '#365314', // olive
  '#44403c', // stone
];

const SHOE_COLORS = [
  '#1f2937', // black
  '#3d2b1f', // dark brown
  '#78350f', // brown
  '#1e293b', // navy
  '#ffffff', // white
  '#6b7280', // grey
];

/**
 * Movement personality — how this specific person moves differently from others.
 * Derived from action_style / temperament keywords in the simulation data.
 * Applied as multipliers to the base pose to make each agent physically unique.
 */
export interface MovementProfile {
  gestureScale: number;       // 0.5=reserved, 1.5=animated, how big are gestures
  tempo: number;              // 0.7=slow/deliberate, 1.3=quick/nervous, movement speed
  stiffness: number;          // 0=loose/fluid, 1=rigid/controlled, body rigidity
  swayAmount: number;         // how much idle sway (0.3=still, 1.5=fidgety)
  headMovement: number;       // how much the head moves independently (0.5=stiff-neck, 1.5=expressive)
  leanForward: number;        // resting forward lean (-0.05=upright/back, 0.1=hunched)
  shoulderDrop: number;       // 0=square shoulders, 0.1=slouched
  walkBounce: number;         // 0.5=gliding, 1.5=bouncy walk
  confrontStyle: 'jabbing' | 'looming' | 'pacing' | 'rigid';
  distressStyle: 'shaking' | 'freezing' | 'pacing' | 'curling' | 'rigid';
}

/** Infer movement profile from simulation text fields */
export function getMovementProfile(
  temperament: string,
  actionStyle: string,
): MovementProfile {
  const text = `${temperament} ${actionStyle}`.toLowerCase();

  // Defaults — average person
  const p: MovementProfile = {
    gestureScale: 1.0,
    tempo: 1.0,
    stiffness: 0.3,
    swayAmount: 1.0,
    headMovement: 1.0,
    leanForward: 0.0,
    shoulderDrop: 0.0,
    walkBounce: 1.0,
    confrontStyle: 'jabbing',
    distressStyle: 'shaking',
  };

  // Anxious / nervous types — fast, fidgety, small gestures
  if (/anxious|nervous|eager|worried|panick/i.test(text)) {
    p.tempo = 1.25;
    p.swayAmount = 1.4;
    p.gestureScale = 0.8;
    p.headMovement = 1.3;
    p.walkBounce = 1.2;
    p.distressStyle = 'shaking';
  }
  // Calm / stoic — slow, controlled, minimal gestures
  if (/calm|stoic|steady|methodical|analytical|guarded|patient/i.test(text)) {
    p.tempo = 0.75;
    p.swayAmount = 0.4;
    p.gestureScale = 0.6;
    p.stiffness = 0.6;
    p.headMovement = 0.6;
    p.walkBounce = 0.6;
    p.confrontStyle = 'rigid';
    p.distressStyle = 'freezing';
  }
  // Blunt / explosive — big gestures, fast, stompy
  if (/blunt|explosive|fierce|aggressive|sharp|quick to escalate/i.test(text)) {
    p.gestureScale = 1.5;
    p.tempo = 1.15;
    p.walkBounce = 1.4;
    p.headMovement = 1.3;
    p.confrontStyle = 'jabbing';
  }
  // Authoritative / commanding — upright, deliberate, controlled gestures
  if (/authorit|command|assertive|strategic|pragmatic|director|ceo/i.test(text)) {
    p.gestureScale = 0.9;
    p.tempo = 0.85;
    p.stiffness = 0.5;
    p.leanForward = -0.03; // leans back slightly = confidence
    p.shoulderDrop = 0.0; // square shoulders
    p.confrontStyle = 'looming';
    p.distressStyle = 'rigid';
  }
  // Empathetic / warm — fluid, open, leaning in
  if (/empath|compassion|warm|kind|gentle|caring|support/i.test(text)) {
    p.gestureScale = 1.1;
    p.stiffness = 0.15;
    p.leanForward = 0.05; // leans toward others
    p.headMovement = 1.3;
    p.shoulderDrop = 0.04;
    p.distressStyle = 'curling';
  }
  // Humorous / carefree — loose, bouncy, big head movements
  if (/humor|carefree|party|joke|easygoing|free spirit/i.test(text)) {
    p.gestureScale = 1.3;
    p.tempo = 1.1;
    p.stiffness = 0.1;
    p.swayAmount = 1.3;
    p.headMovement = 1.4;
    p.walkBounce = 1.3;
    p.shoulderDrop = 0.06;
    p.confrontStyle = 'pacing';
    p.distressStyle = 'pacing';
  }
  // Sensitive / introverted — small gestures, drawn inward
  if (/sensitive|introverted|withdraw|shy|quiet|artistic/i.test(text)) {
    p.gestureScale = 0.6;
    p.swayAmount = 0.7;
    p.leanForward = 0.04;
    p.shoulderDrop = 0.07;
    p.headMovement = 0.8;
    p.walkBounce = 0.7;
    p.distressStyle = 'curling';
  }
  // Proud / competitive — upright, controlled power
  if (/proud|competitive|ambitious|driven|confident/i.test(text)) {
    p.leanForward = -0.04;
    p.stiffness = 0.4;
    p.gestureScale = 1.1;
    p.walkBounce = 0.9;
    p.confrontStyle = 'looming';
  }

  return p;
}

export function getAgentAppearance(agentId: string): AgentAppearance {
  const skinIdx = Math.floor(seeded(agentId, 1) * SKIN_TONES.length);
  const hairIdx = Math.floor(seeded(agentId, 2) * HAIR_COLORS.length);
  const shirtIdx = Math.floor(seeded(agentId, 3) * SHIRT_COLORS.length);
  const pantsIdx = Math.floor(seeded(agentId, 4) * PANTS_COLORS.length);
  const shoeIdx = Math.floor(seeded(agentId, 5) * SHOE_COLORS.length);
  const hairStyleNum = seeded(agentId, 6);

  const hairStyle: AgentAppearance['hairStyle'] =
    hairStyleNum < 0.3 ? 'short' :
    hairStyleNum < 0.55 ? 'medium' :
    hairStyleNum < 0.75 ? 'long' :
    hairStyleNum < 0.9 ? 'bald' : 'mohawk';

  return {
    skin: SKIN_TONES[skinIdx].base,
    skinDark: SKIN_TONES[skinIdx].dark,
    hair: HAIR_COLORS[hairIdx],
    hairStyle,
    shirt: SHIRT_COLORS[shirtIdx],
    pants: PANTS_COLORS[pantsIdx],
    shoes: SHOE_COLORS[shoeIdx],
    height: 0.88 + seeded(agentId, 7) * 0.24,
    bulk: 0.85 + seeded(agentId, 8) * 0.30,
  };
}

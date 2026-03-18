import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SectionHeading, Card } from '@/components/ui';
import { AgentAvatar3D } from './AgentAvatar3D';
import type { AvatarState } from './AgentAvatar3D';
import {
  TrendingUp, AlertTriangle, Lightbulb, Shield, Target,
  Heart, Zap, Brain, Users, Sparkles, Timer, Flame,
  RotateCcw, Shuffle, Play, Pause,
} from 'lucide-react';

// ─── Drive metadata ─────────────────────────────────────────────────────────
interface DriveConfig {
  key: keyof AvatarState;
  label: string;
  icon: React.ReactNode;
  color: string;
  baseline: number;
  description: string;
}

const DRIVES: DriveConfig[] = [
  { key: 'confidence', label: 'Confidence', icon: <TrendingUp size={13} />, color: '#0891b2', baseline: 0.55, description: 'Self-assurance and optimism' },
  { key: 'stress', label: 'Stress', icon: <AlertTriangle size={13} />, color: '#e11d48', baseline: 0.20, description: 'Pressure and tension' },
  { key: 'curiosity', label: 'Curiosity', icon: <Lightbulb size={13} />, color: '#0066ff', baseline: 0.60, description: 'Drive to explore and learn' },
  { key: 'fear', label: 'Fear', icon: <Shield size={13} />, color: '#dc2626', baseline: 0.15, description: 'Threat detection and caution' },
  { key: 'ambition', label: 'Ambition', icon: <Target size={13} />, color: '#d97706', baseline: 0.55, description: 'Goal pursuit and determination' },
  { key: 'empathy', label: 'Empathy', icon: <Heart size={13} />, color: '#059669', baseline: 0.50, description: 'Social sensitivity and care' },
  { key: 'impulse', label: 'Impulse', icon: <Zap size={13} />, color: '#ea580c', baseline: 0.30, description: 'Gut reactions, fast decisions' },
  { key: 'reflection', label: 'Reflection', icon: <Brain size={13} />, color: '#7c3aed', baseline: 0.50, description: 'Deep thinking and analysis' },
  { key: 'trust', label: 'Trust', icon: <Users size={13} />, color: '#0d9488', baseline: 0.50, description: 'Openness to others' },
  { key: 'motivation', label: 'Motivation', icon: <Sparkles size={13} />, color: '#2563eb', baseline: 0.60, description: 'Energy and drive to act' },
  { key: 'fatigue', label: 'Fatigue', icon: <Timer size={13} />, color: '#94a3b8', baseline: 0.15, description: 'Cognitive exhaustion' },
  { key: 'frustration', label: 'Frustration', icon: <Flame size={13} />, color: '#be123c', baseline: 0.10, description: 'Accumulated failure stress' },
];

// ─── Presets ─────────────────────────────────────────────────────────────────
interface Preset {
  label: string;
  desc: string;
  emoji: string;
  values: Partial<AvatarState>;
}

const PRESETS: Preset[] = [
  {
    label: 'Calm Baseline',
    desc: 'Neutral resting state',
    emoji: '😐',
    values: {},
  },
  {
    label: 'Happy & Confident',
    desc: 'Everything is going great',
    emoji: '😄',
    values: { confidence: 0.95, motivation: 0.90, curiosity: 0.80, ambition: 0.75, trust: 0.70, stress: 0.05, fear: 0.05, fatigue: 0.05 },
  },
  {
    label: 'Anxious & Scared',
    desc: 'Overwhelmed by threats',
    emoji: '😰',
    values: { fear: 0.90, stress: 0.85, confidence: 0.10, trust: 0.15, curiosity: 0.15, impulse: 0.50, reflection: 0.20 },
  },
  {
    label: 'Angry & Frustrated',
    desc: 'Repeated failures mounting',
    emoji: '😡',
    values: { frustration: 0.95, stress: 0.80, impulse: 0.75, confidence: 0.25, empathy: 0.15, reflection: 0.15 },
  },
  {
    label: 'Deep Thinker',
    desc: 'Contemplating complex problems',
    emoji: '🤔',
    values: { reflection: 0.95, curiosity: 0.70, impulse: 0.05, motivation: 0.50, stress: 0.10, ambition: 0.40 },
  },
  {
    label: 'Exhausted',
    desc: 'Running on empty',
    emoji: '😴',
    values: { fatigue: 0.95, motivation: 0.10, curiosity: 0.10, reflection: 0.15, impulse: 0.55, stress: 0.60 },
  },
  {
    label: 'Excited Explorer',
    desc: 'Bursting with curiosity',
    emoji: '🤩',
    values: { curiosity: 0.95, motivation: 0.90, impulse: 0.65, ambition: 0.80, confidence: 0.75, fear: 0.05 },
  },
  {
    label: 'Torn Apart',
    desc: 'Curiosity vs fear deadlock',
    emoji: '😵',
    values: { curiosity: 0.85, fear: 0.80, ambition: 0.70, stress: 0.55, impulse: 0.50, reflection: 0.50 },
  },
  {
    label: 'Compassionate',
    desc: 'Deeply empathetic state',
    emoji: '🥰',
    values: { empathy: 0.95, trust: 0.85, confidence: 0.60, stress: 0.10, fear: 0.05, motivation: 0.65 },
  },
  {
    label: 'On Fire',
    desc: 'Peak ambition and drive',
    emoji: '🔥',
    values: { ambition: 0.95, motivation: 0.90, confidence: 0.85, impulse: 0.70, curiosity: 0.65, fear: 0.05, fatigue: 0.05 },
  },
];

function getDefaults(): AvatarState {
  const out: Record<string, number> = {};
  for (const d of DRIVES) out[d.key] = d.baseline;
  return out as unknown as AvatarState;
}

// ─── Emotion Slider ──────────────────────────────────────────────────────────
function EmotionSlider({
  drive,
  value,
  onChange,
}: {
  drive: DriveConfig;
  value: number;
  onChange: (key: keyof AvatarState, v: number) => void;
}) {
  return (
    <div className="group flex items-center gap-2.5 py-1.5">
      <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center" style={{ color: drive.color }}>
        {drive.icon}
      </span>
      <span className="text-[11px] font-medium text-text-base w-[72px] flex-shrink-0">{drive.label}</span>
      <div className="relative flex-1 h-7 flex items-center">
        {/* Track background */}
        <div className="absolute inset-x-0 h-2 rounded-full bg-surface border border-border" />
        {/* Filled track */}
        <div
          className="absolute left-0 h-2 rounded-full transition-all duration-100"
          style={{ width: `${value * 100}%`, backgroundColor: drive.color + '40' }}
        />
        {/* Input */}
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(value * 100)}
          onChange={(e) => onChange(drive.key, parseInt(e.target.value) / 100)}
          className="relative z-10 w-full h-7 appearance-none bg-transparent cursor-pointer"
          style={{
            // Thumb styling via inline for cross-browser
            WebkitAppearance: 'none',
          }}
        />
      </div>
      <span className="text-[11px] font-mono text-text-secondary w-8 text-right">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

// ─── Emotion Event Simulator ─────────────────────────────────────────────────
interface EmotionEvent {
  label: string;
  emoji: string;
  deltas: Partial<AvatarState>;
}

const EVENTS: EmotionEvent[] = [
  { label: 'Receive praise', emoji: '👏', deltas: { confidence: 0.15, motivation: 0.1, stress: -0.1, trust: 0.05 } },
  { label: 'Get rejected', emoji: '💔', deltas: { confidence: -0.2, trust: -0.15, stress: 0.15, fear: 0.1 } },
  { label: 'Discover something', emoji: '💡', deltas: { curiosity: 0.2, motivation: 0.15, fatigue: -0.05 } },
  { label: 'Make a mistake', emoji: '❌', deltas: { frustration: 0.2, stress: 0.15, confidence: -0.15 } },
  { label: 'Help someone', emoji: '🤝', deltas: { empathy: 0.15, trust: 0.1, motivation: 0.05, stress: -0.05 } },
  { label: 'Face a deadline', emoji: '⏰', deltas: { stress: 0.2, impulse: 0.15, reflection: -0.1, fatigue: 0.1 } },
  { label: 'Take a break', emoji: '☕', deltas: { fatigue: -0.2, stress: -0.15, reflection: 0.1, motivation: 0.05 } },
  { label: 'Win a challenge', emoji: '🏆', deltas: { confidence: 0.2, ambition: 0.15, motivation: 0.15, frustration: -0.15 } },
  { label: 'Witness injustice', emoji: '⚖️', deltas: { empathy: 0.15, frustration: 0.1, ambition: 0.1, impulse: 0.1 } },
  { label: 'Encounter danger', emoji: '⚡', deltas: { fear: 0.25, stress: 0.2, impulse: 0.15, curiosity: -0.1 } },
];

// ─── Main Component ──────────────────────────────────────────────────────────
export function AvatarPlayground() {
  const [state, setState] = useState<AvatarState>(getDefaults());
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simInterval, setSimInterval] = useState<ReturnType<typeof setInterval> | null>(null);

  const updateDrive = useCallback((key: keyof AvatarState, value: number) => {
    setState(prev => ({ ...prev, [key]: Math.max(0, Math.min(1, value)) }));
  }, []);

  const applyPreset = useCallback((preset: Preset) => {
    const base = getDefaults();
    setState({ ...base, ...preset.values } as AvatarState);
    setEventLog(prev => [`Applied: ${preset.emoji} ${preset.label}`, ...prev].slice(0, 15));
  }, []);

  const applyEvent = useCallback((event: EmotionEvent) => {
    setState(prev => {
      const next = { ...prev };
      for (const [key, delta] of Object.entries(event.deltas)) {
        const k = key as keyof AvatarState;
        next[k] = Math.max(0, Math.min(1, prev[k] + (delta as number)));
      }
      return next;
    });
    setEventLog(prev => [`${event.emoji} ${event.label}`, ...prev].slice(0, 15));
  }, []);

  const resetToBaseline = useCallback(() => {
    setState(getDefaults());
    setEventLog([]);
  }, []);

  const randomize = useCallback(() => {
    const next: Record<string, number> = {};
    for (const d of DRIVES) next[d.key] = Math.random();
    setState(next as unknown as AvatarState);
    setEventLog(prev => ['🎲 Randomized all drives', ...prev].slice(0, 15));
  }, []);

  // Auto-simulate: apply random events over time
  const toggleSimulation = useCallback(() => {
    if (isSimulating && simInterval) {
      clearInterval(simInterval);
      setSimInterval(null);
      setIsSimulating(false);
    } else {
      setIsSimulating(true);
      const interval = setInterval(() => {
        const event = EVENTS[Math.floor(Math.random() * EVENTS.length)];
        applyEvent(event);
      }, 2000);
      setSimInterval(interval);
    }
  }, [isSimulating, simInterval, applyEvent]);

  // Compute derived info
  const mood = (() => {
    const positive = state.confidence + state.motivation + state.curiosity + state.trust + state.ambition;
    const negative = state.stress + state.frustration + state.fear + state.fatigue;
    return (positive - negative) / (positive + negative + 0.001);
  })();

  const arousal = (state.stress + state.curiosity + state.fear + state.ambition + state.impulse) / 5;

  const dominantDrive = DRIVES.reduce((best, d) =>
    state[d.key] > state[best.key] ? d : best, DRIVES[0]);

  // Detect conflicts
  const conflictPairs: [string, string][] = [];
  const oppositions: [keyof AvatarState, keyof AvatarState][] = [
    ['curiosity', 'fear'],
    ['impulse', 'reflection'],
    ['ambition', 'frustration'],
    ['empathy', 'stress'],
  ];
  for (const [a, b] of oppositions) {
    if (state[a] > 0.5 && state[b] > 0.5) {
      conflictPairs.push([
        DRIVES.find(d => d.key === a)!.label,
        DRIVES.find(d => d.key === b)!.label,
      ]);
    }
  }

  return (
    <section id="avatar-playground" className="py-24">
      <div className="section-container">
        <SectionHeading
          badge="Interactive"
          title="Meet the Agent"
          subtitle="Drag the sliders to shape the agent's emotional state. Trigger life events. Watch how every internal shift changes posture, expression, and behavior in real-time — the same state that drives cognitive branch routing."
          center
        />

        <div className="mt-12 grid lg:grid-cols-[1fr,320px,1fr] gap-6 items-start">
          {/* LEFT: Sliders */}
          <Card className="p-5">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-text-base">Emotional Drives</h4>
              <div className="flex gap-1.5">
                <button
                  onClick={resetToBaseline}
                  className="p-1.5 rounded-lg text-text-muted hover:text-text-base hover:bg-surface transition-all"
                  title="Reset to baseline"
                >
                  <RotateCcw size={14} />
                </button>
                <button
                  onClick={randomize}
                  className="p-1.5 rounded-lg text-text-muted hover:text-text-base hover:bg-surface transition-all"
                  title="Randomize"
                >
                  <Shuffle size={14} />
                </button>
                <button
                  onClick={toggleSimulation}
                  className={`p-1.5 rounded-lg transition-all ${
                    isSimulating ? 'text-primary bg-primary-light' : 'text-text-muted hover:text-text-base hover:bg-surface'
                  }`}
                  title={isSimulating ? 'Stop simulation' : 'Auto-simulate events'}
                >
                  {isSimulating ? <Pause size={14} /> : <Play size={14} />}
                </button>
              </div>
            </div>

            <div className="space-y-0.5">
              {DRIVES.map(drive => (
                <EmotionSlider
                  key={drive.key}
                  drive={drive}
                  value={state[drive.key]}
                  onChange={updateDrive}
                />
              ))}
            </div>
          </Card>

          {/* CENTER: Avatar */}
          <div className="flex flex-col items-center gap-4 sticky top-20">
            <AgentAvatar3D state={state} size={340} />

            {/* State readout */}
            <div className="w-full space-y-2">
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] text-text-muted uppercase tracking-wider">Mood</span>
                <span className={`text-xs font-mono font-semibold ${
                  mood > 0.2 ? 'text-success-DEFAULT' : mood < -0.2 ? 'text-danger-DEFAULT' : 'text-text-muted'
                }`}>
                  {mood > 0 ? '+' : ''}{mood.toFixed(3)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-surface border border-border overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  animate={{
                    width: `${((mood + 1) / 2) * 100}%`,
                    backgroundColor: mood > 0.2 ? '#059669' : mood < -0.2 ? '#e11d48' : '#94a3b8',
                  }}
                  transition={{ duration: 0.3 }}
                />
              </div>

              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] text-text-muted uppercase tracking-wider">Arousal</span>
                <span className="text-xs font-mono text-text-secondary">{(arousal * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 rounded-full bg-surface border border-border overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  animate={{
                    width: `${arousal * 100}%`,
                    backgroundColor: arousal > 0.6 ? '#ea580c' : arousal > 0.3 ? '#d97706' : '#94a3b8',
                  }}
                  transition={{ duration: 0.3 }}
                />
              </div>

              <div className="flex items-center justify-between px-1 pt-1">
                <span className="text-[10px] text-text-muted uppercase tracking-wider">Dominant</span>
                <span className="text-xs font-semibold" style={{ color: dominantDrive.color }}>
                  {dominantDrive.label}
                </span>
              </div>

              {conflictPairs.length > 0 && (
                <div className="p-2 rounded-lg bg-[rgba(147,51,234,0.05)] border border-accent-purple/15 mt-1">
                  <div className="text-[9px] font-semibold text-accent-purple uppercase tracking-wider mb-1">Active Conflicts</div>
                  {conflictPairs.map(([a, b], i) => (
                    <div key={i} className="text-[11px] text-text-secondary">
                      {a} <span className="text-accent-purple">vs</span> {b}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: Events + Presets */}
          <div className="space-y-4">
            {/* Presets */}
            <Card className="p-5">
              <h4 className="text-sm font-semibold text-text-base mb-3">Emotional Presets</h4>
              <div className="grid grid-cols-2 gap-2">
                {PRESETS.map(preset => (
                  <button
                    key={preset.label}
                    onClick={() => applyPreset(preset)}
                    className="flex items-center gap-2 p-2.5 rounded-lg bg-surface border border-border hover:border-border-strong hover:shadow-card text-left transition-all group"
                  >
                    <span className="text-base">{preset.emoji}</span>
                    <div className="min-w-0">
                      <div className="text-[11px] font-medium text-text-base group-hover:text-primary transition-colors truncate">{preset.label}</div>
                      <div className="text-[9px] text-text-muted truncate">{preset.desc}</div>
                    </div>
                  </button>
                ))}
              </div>
            </Card>

            {/* Life Events */}
            <Card className="p-5">
              <h4 className="text-sm font-semibold text-text-base mb-3">Trigger Life Events</h4>
              <p className="text-[10px] text-text-muted mb-3">Click events to apply incremental emotional changes — watch the avatar react and adapt.</p>
              <div className="grid grid-cols-2 gap-1.5">
                {EVENTS.map(event => (
                  <button
                    key={event.label}
                    onClick={() => applyEvent(event)}
                    className="flex items-center gap-2 px-2.5 py-2 rounded-lg hover:bg-surface border border-transparent hover:border-border text-left transition-all active:scale-95"
                  >
                    <span className="text-sm">{event.emoji}</span>
                    <span className="text-[11px] text-text-secondary">{event.label}</span>
                  </button>
                ))}
              </div>
            </Card>

            {/* Event Log */}
            <Card className="p-5">
              <h4 className="text-sm font-semibold text-text-base mb-3">Event Log</h4>
              <div className="space-y-1 max-h-[140px] overflow-y-auto">
                <AnimatePresence initial={false}>
                  {eventLog.length === 0 ? (
                    <p className="text-[11px] text-text-muted italic">No events yet — try clicking a preset or life event</p>
                  ) : (
                    eventLog.map((entry, i) => (
                      <motion.div
                        key={`${entry}-${i}`}
                        initial={{ opacity: 0, x: -10, height: 0 }}
                        animate={{ opacity: 1, x: 0, height: 'auto' }}
                        className="text-[11px] text-text-secondary py-0.5 border-b border-border last:border-0"
                      >
                        {entry}
                      </motion.div>
                    ))
                  )}
                </AnimatePresence>
              </div>
            </Card>
          </div>
        </div>
      </div>

      {/* Custom slider styles */}
      <style>{`
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: white;
          border: 2px solid #d1d5db;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          cursor: pointer;
          transition: border-color 0.15s, box-shadow 0.15s;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
          border-color: #0066ff;
          box-shadow: 0 0 0 3px rgba(0,102,255,0.12);
        }
        input[type="range"]::-webkit-slider-thumb:active {
          border-color: #0066ff;
          box-shadow: 0 0 0 5px rgba(0,102,255,0.15);
        }
        input[type="range"]::-moz-range-thumb {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: white;
          border: 2px solid #d1d5db;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          cursor: pointer;
        }
        input[type="range"]::-moz-range-track {
          background: transparent;
          border: none;
        }
      `}</style>
    </section>
  );
}

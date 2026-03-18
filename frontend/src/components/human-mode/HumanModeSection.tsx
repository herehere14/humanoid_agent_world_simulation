import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SectionHeading, Card, Bar, GlowButton } from '@/components/ui';
import {
  Brain, Heart, Shield, Zap, Eye, Target, Scale, AlertTriangle,
  TrendingUp, TrendingDown, Activity, Sparkles, ArrowRight, ArrowDown,
  Lightbulb, Users, Timer, BarChart3, GitBranch, RefreshCw, Database,
  Cpu, Layers,
} from 'lucide-react';
import { AgentAvatar3D } from './AgentAvatar3D';
import type { AvatarState } from './AgentAvatar3D';

// ─── Drive definitions ─────────────────────────────────────────────────────
interface DriveState {
  name: string;
  value: number;
  baseline: number;
  color: string;
  icon: React.ReactNode;
  label: string;
}

const DRIVE_COLORS: Record<string, string> = {
  confidence: '#0891b2',
  stress: '#e11d48',
  curiosity: '#0066ff',
  fear: '#dc2626',
  ambition: '#d97706',
  empathy: '#059669',
  impulse: '#ea580c',
  reflection: '#7c3aed',
  trust: '#0d9488',
  motivation: '#2563eb',
  fatigue: '#94a3b8',
  frustration: '#be123c',
  self_protection: '#9333ea',
  caution: '#64748b',
};

const INITIAL_DRIVES: DriveState[] = [
  { name: 'confidence', value: 0.55, baseline: 0.55, color: '#0891b2', icon: <TrendingUp size={14} />, label: 'Confidence' },
  { name: 'stress', value: 0.20, baseline: 0.20, color: '#e11d48', icon: <AlertTriangle size={14} />, label: 'Stress' },
  { name: 'curiosity', value: 0.60, baseline: 0.60, color: '#0066ff', icon: <Lightbulb size={14} />, label: 'Curiosity' },
  { name: 'fear', value: 0.15, baseline: 0.15, color: '#dc2626', icon: <Shield size={14} />, label: 'Fear' },
  { name: 'ambition', value: 0.55, baseline: 0.55, color: '#d97706', icon: <Target size={14} />, label: 'Ambition' },
  { name: 'empathy', value: 0.50, baseline: 0.50, color: '#059669', icon: <Heart size={14} />, label: 'Empathy' },
  { name: 'impulse', value: 0.30, baseline: 0.30, color: '#ea580c', icon: <Zap size={14} />, label: 'Impulse' },
  { name: 'reflection', value: 0.50, baseline: 0.50, color: '#7c3aed', icon: <Brain size={14} />, label: 'Reflection' },
  { name: 'trust', value: 0.50, baseline: 0.50, color: '#0d9488', icon: <Users size={14} />, label: 'Trust' },
  { name: 'motivation', value: 0.60, baseline: 0.60, color: '#2563eb', icon: <Sparkles size={14} />, label: 'Motivation' },
  { name: 'fatigue', value: 0.15, baseline: 0.15, color: '#94a3b8', icon: <Timer size={14} />, label: 'Fatigue' },
];

type Scenario = 'baseline' | 'confident' | 'anxious' | 'conflicted' | 'fatigued';

const SCENARIOS: Record<Scenario, { label: string; desc: string; values: Record<string, number> }> = {
  baseline: {
    label: 'Baseline',
    desc: 'Homeostatic resting state — all drives at default',
    values: {},
  },
  confident: {
    label: 'Confident Explorer',
    desc: 'High confidence + curiosity, low fear — approach-oriented',
    values: { confidence: 0.90, curiosity: 0.85, ambition: 0.80, motivation: 0.85, fear: 0.10, stress: 0.10 },
  },
  anxious: {
    label: 'Anxious Defender',
    desc: 'High fear + stress, low confidence — avoidance-oriented',
    values: { confidence: 0.15, fear: 0.85, stress: 0.80, self_protection: 0.80, trust: 0.20, curiosity: 0.20 },
  },
  conflicted: {
    label: 'Internally Conflicted',
    desc: 'Curiosity vs fear both high — approach-avoidance tension',
    values: { curiosity: 0.80, fear: 0.75, ambition: 0.70, caution: 0.65, stress: 0.50 },
  },
  fatigued: {
    label: 'Exhausted Under Pressure',
    desc: 'High fatigue + deadline stress — cognitive degradation',
    values: { fatigue: 0.90, motivation: 0.20, stress: 0.70, impulse: 0.60, reflection: 0.20 },
  },
};

const CONFLICT_PAIRS = [
  ['curiosity', 'fear'],
  ['impulse', 'reflection'],
  ['ambition', 'caution'],
  ['empathy', 'self_protection'],
];

const COGNITIVE_BRANCHES = [
  { name: 'Reflective Reasoning', drive: 'reflection', desc: 'Slow, deliberate analysis', speed: 'slow', cost: 'high' },
  { name: 'Fear/Risk Assessment', drive: 'fear', desc: 'Threat detection, worst-case scanning', speed: 'fast', cost: 'low' },
  { name: 'Curiosity Exploration', drive: 'curiosity', desc: 'Novel angles, open-ended inquiry', speed: 'medium', cost: 'medium' },
  { name: 'Ambition/Reward', drive: 'ambition', desc: 'Goal pursuit, opportunity focus', speed: 'fast', cost: 'low' },
  { name: 'Impulse Response', drive: 'impulse', desc: 'Gut reaction, pattern matching', speed: 'instant', cost: 'very low' },
  { name: 'Empathy/Social', drive: 'empathy', desc: 'Theory of mind, perspective taking', speed: 'medium', cost: 'high' },
  { name: 'Conflict Resolver', drive: 'reflection', desc: 'Mediates competing drives', speed: 'slow', cost: 'high' },
];

// ─── Structural Isomorphism ─────────────────────────────────────────────────
function StructuralIsomorphism() {
  const mappings = [
    {
      forest: 'Competing Branches',
      forestIcon: <GitBranch size={16} />,
      cognition: 'Competing Cognitive Modules',
      cognitionIcon: <Brain size={16} />,
      desc: 'Multiple specialized branches execute in parallel — just as a person simultaneously processes impulse, reflection, fear, and ambition before acting. The forest doesn\'t simulate cognition. It IS cognition, architecturally.',
      color: '#0066ff',
    },
    {
      forest: 'Weighted Routing',
      forestIcon: <Layers size={16} />,
      cognition: 'Arousal & Drive Modulation',
      cognitionIcon: <Activity size={16} />,
      desc: 'Branch selection scores are modulated by internal state. High stress boosts fast/impulse branches and suppresses reflection — the same Yerkes-Dodson dynamic that governs human decision-making under pressure.',
      color: '#7c3aed',
    },
    {
      forest: 'RL Weight Updates',
      forestIcon: <RefreshCw size={16} />,
      cognition: 'Experiential Learning',
      cognitionIcon: <TrendingUp size={16} />,
      desc: 'After each task, reward signals adjust branch weights. Success with empathy-driven routing strengthens that pathway for similar future tasks. This is operant conditioning — the same mechanism that shapes human behavioral patterns.',
      color: '#059669',
    },
    {
      forest: 'Memory-Biased Routing',
      forestIcon: <Database size={16} />,
      cognition: 'Approach / Avoidance Bias',
      cognitionIcon: <Eye size={16} />,
      desc: 'Past outcomes create routing biases. Traumatic failures create avoidance of certain branch paths. Peak successes create approach bias. This is the "once bitten, twice shy" pattern — experiential memory shaping future behavior.',
      color: '#d97706',
    },
    {
      forest: 'Conflict Detection',
      forestIcon: <AlertTriangle size={16} />,
      cognition: 'Drive Opposition',
      cognitionIcon: <Scale size={16} />,
      desc: 'When opposing branches (curiosity vs fear, impulse vs long-term goals) both score highly, the system detects a conflict and activates resolution strategies — dominant, compromise, or noisy. Humans call this being "torn between options."',
      color: '#e11d48',
    },
    {
      forest: 'Homeostatic Decay',
      forestIcon: <ArrowDown size={16} />,
      cognition: 'Emotional Regulation',
      cognitionIcon: <Heart size={16} />,
      desc: 'All state variables decay toward baseline unless actively reinforced. Panic fades. Euphoria subsides. This models the biological regulatory mechanisms that prevent permanent emotional extremes — the same mean-reversion observed in psychological and market data.',
      color: '#0891b2',
    },
  ];

  return (
    <div className="space-y-3">
      {mappings.map((m, i) => (
        <motion.div
          key={m.forest}
          className="grid grid-cols-[1fr,auto,1fr] gap-4 items-center p-4 bg-white rounded-xl border border-border hover:shadow-card-md transition-all"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.06 }}
        >
          {/* Forest concept */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0" style={{ backgroundColor: m.color + '12' }}>
              <span style={{ color: m.color }}>{m.forestIcon}</span>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-text-muted font-medium">Forest Engine</div>
              <div className="text-xs font-semibold text-text-base">{m.forest}</div>
            </div>
          </div>

          {/* Equals sign */}
          <div className="flex flex-col items-center gap-1">
            <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold" style={{ backgroundColor: m.color + '15', color: m.color }}>
              =
            </div>
          </div>

          {/* Cognitive concept */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0" style={{ backgroundColor: m.color + '12' }}>
              <span style={{ color: m.color }}>{m.cognitionIcon}</span>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-text-muted font-medium">Human Cognition</div>
              <div className="text-xs font-semibold text-text-base">{m.cognition}</div>
            </div>
          </div>

          {/* Description spanning full width */}
          <div className="col-span-3 pt-2 border-t border-border">
            <p className="text-xs text-text-secondary leading-relaxed">{m.desc}</p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// ─── Interactive Drive Visualizer ──────────────────────────────────────────
function DriveVisualizer({ drives, conflicts }: { drives: DriveState[]; conflicts: string[][] }) {
  return (
    <div className="space-y-2">
      {drives.map((drive) => {
        const isInConflict = conflicts.some(([a, b]) =>
          (a === drive.name || b === drive.name) && drive.value > 0.4
        );
        return (
          <motion.div
            key={drive.name}
            className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all ${
              isInConflict ? 'bg-[rgba(147,51,234,0.06)] border border-accent-purple/15' : 'bg-surface'
            }`}
            layout
          >
            <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center" style={{ color: drive.color }}>
              {drive.icon}
            </span>
            <span className="text-xs font-medium text-text-base w-20 flex-shrink-0">{drive.label}</span>
            <div className="flex-1">
              <Bar value={drive.value} color={drive.color} height={6} />
            </div>
            <span className="text-xs font-mono text-text-secondary w-10 text-right">
              {drive.value.toFixed(2)}
            </span>
            {isInConflict && (
              <span className="text-[9px] font-medium text-accent-purple bg-[rgba(147,51,234,0.1)] px-1.5 py-0.5 rounded">
                CONFLICT
              </span>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}

// ─── Routing Comparison ───────────────────────────────────────────────────
function RoutingComparison({ scenario }: { scenario: Scenario }) {
  const values = SCENARIOS[scenario].values;

  const branchScores = COGNITIVE_BRANCHES.map((branch) => {
    const driveValue = values[branch.drive] ?? INITIAL_DRIVES.find(d => d.name === branch.drive)?.baseline ?? 0.5;
    const fatigue = values.fatigue ?? 0.15;

    let score = 0.5 + driveValue * 0.5;
    if (fatigue > 0.5 && (branch.cost === 'high')) {
      score -= fatigue * 0.2;
    }
    const arousal = ((values.stress ?? 0.2) + (values.curiosity ?? 0.6) + (values.fear ?? 0.15) + (values.ambition ?? 0.55) + (values.impulse ?? 0.3)) / 5;
    if (arousal > 0.6 && branch.speed === 'instant') score *= 1.15;
    if (arousal < 0.3 && branch.speed === 'slow') score *= 1.1;

    return { ...branch, score: Math.max(0.1, Math.min(1.0, score)), driveValue };
  }).sort((a, b) => b.score - a.score);

  const topBranches = branchScores.slice(0, 4);
  const maxScore = topBranches[0]?.score ?? 1;

  return (
    <div className="space-y-2">
      {topBranches.map((branch, i) => (
        <motion.div
          key={branch.name}
          className="flex items-center gap-3 p-3 bg-white rounded-lg border border-border"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.08 }}
        >
          <span className={`text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center ${
            i === 0 ? 'bg-primary text-white' : 'bg-surface text-text-muted'
          }`}>
            {i + 1}
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium text-text-base">{branch.name}</div>
            <div className="text-[10px] text-text-muted">{branch.desc}</div>
          </div>
          <div className="w-20">
            <Bar value={branch.score} max={maxScore} color={DRIVE_COLORS[branch.drive] ?? '#0066ff'} height={4} />
          </div>
          <span className="text-xs font-mono text-text-secondary w-10 text-right">{branch.score.toFixed(2)}</span>
        </motion.div>
      ))}
    </div>
  );
}

// ─── World Prediction Argument ────────────────────────────────────────────
function WorldPredictionSection() {
  const points = [
    {
      icon: <Brain size={20} />,
      title: 'Cognitive Fidelity via Structural Identity',
      desc: 'This isn\'t a model that approximates cognition — it uses the same architecture. Competing branches ARE competing cognitive modules. Routing IS attention allocation. Weight updates ARE reinforcement learning. Markets, elections, and social systems are driven by these exact mechanisms operating across millions of human minds.',
    },
    {
      icon: <Scale size={20} />,
      title: 'Competing Drive Resolution',
      desc: 'Real-world outcomes emerge from tension between opposing forces. Bull vs bear sentiment is curiosity vs fear. Innovation vs regulation is ambition vs caution. The forest\'s conflict detection identifies when opposing drives are both strong, activates both pathways, and models the resolution — exactly as human groups resolve competing priorities.',
    },
    {
      icon: <Activity size={20} />,
      title: 'Arousal-Performance Dynamics',
      desc: 'Under crisis, the forest shifts from slow reflective branches to fast impulsive ones — the Yerkes-Dodson effect. This mirrors how market panics produce irrational selling: cognitive resources are hijacked by threat detection, suppressing deliberative analysis. The routing change is automatic, state-driven, and reversible.',
    },
    {
      icon: <TrendingDown size={20} />,
      title: 'Failure Cascades & Momentum',
      desc: 'The frustration amplifier (1.0 + frustration) makes each subsequent failure compound exponentially. This models recession dynamics: job losses reduce confidence, confidence reduces spending, spending reductions cause more job losses. The same RL weight-update loop that optimizes the forest also produces these realistic spiral effects.',
    },
    {
      icon: <Eye size={20} />,
      title: 'Experiential Memory Bias',
      desc: 'The forest\'s memory system records emotional valence alongside routing outcomes. Traumatic losses create avoidance bias that persists long after the threat passes — the "once bitten, twice shy" effect that makes traders avoid asset classes for years after a crash. Positive memories create approach bias that drives overconfidence cycles.',
    },
    {
      icon: <RefreshCw size={20} />,
      title: 'Homeostatic Mean Reversion',
      desc: 'Every state variable decays toward its baseline at a configurable rate. Panic subsides. Euphoria fades. Markets mean-revert. This isn\'t added as a feature — it\'s an emergent property of the forest\'s weight decay mechanism. The same homeostatic pressure that prevents the engine from fixating on one branch also models psychological recovery.',
    },
  ];

  return (
    <div className="grid md:grid-cols-2 gap-6">
      {points.map((point, i) => (
        <motion.div
          key={point.title}
          className="p-6 bg-white rounded-xl border border-border hover:shadow-card-md transition-all"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.08 }}
        >
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-light flex items-center justify-center text-primary flex-shrink-0">
              {point.icon}
            </div>
            <div>
              <h4 className="text-sm font-semibold text-text-base mb-2">{point.title}</h4>
              <p className="text-xs text-text-secondary leading-relaxed">{point.desc}</p>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// ─── Shared Engine Diagram ──────────────────────────────────────────────────
function SharedEngineDiagram() {
  const sharedLayers = [
    { label: 'PromptForestEngine', desc: 'Core execution runtime', color: '#0066ff', icon: <Cpu size={14} /> },
    { label: 'Branch Execution', desc: 'Parallel module processing', color: '#059669', icon: <GitBranch size={14} /> },
    { label: 'Output Aggregation', desc: 'Best-of-N selection', color: '#d97706', icon: <Layers size={14} /> },
    { label: 'RL Weight Updates', desc: 'Reward-driven adaptation', color: '#7c3aed', icon: <RefreshCw size={14} /> },
  ];

  return (
    <Card className="p-6 border-primary/15">
      <h4 className="text-sm font-semibold text-text-base text-center mb-6">One Engine, Two Expressions</h4>
      <div className="grid md:grid-cols-[1fr,auto,1fr] gap-6 items-start">
        {/* Agent Mode */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 pb-2 border-b border-border">
            <BarChart3 size={14} className="text-primary" />
            <span className="text-xs font-semibold text-text-base">Agent Improvement</span>
          </div>
          <div className="space-y-2 text-xs text-text-secondary">
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-primary mt-1.5 flex-shrink-0" /><span>Performance-optimized branches (analytical, critique, planner, verification...)</span></div>
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-primary mt-1.5 flex-shrink-0" /><span>Routes by task type + historical performance</span></div>
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-primary mt-1.5 flex-shrink-0" /><span>Reward = output quality</span></div>
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-primary mt-1.5 flex-shrink-0" /><span>Stateless — same input, same routing</span></div>
          </div>
        </div>

        {/* Shared core */}
        <div className="space-y-2">
          <div className="text-[10px] text-text-muted text-center uppercase tracking-wider font-semibold mb-2">Shared Engine</div>
          {sharedLayers.map((layer) => (
            <div key={layer.label} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface border border-border">
              <span style={{ color: layer.color }}>{layer.icon}</span>
              <div>
                <div className="text-[10px] font-semibold text-text-base">{layer.label}</div>
                <div className="text-[9px] text-text-muted">{layer.desc}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Human Mode */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 pb-2 border-b border-accent-purple/30">
            <Brain size={14} className="text-accent-purple" />
            <span className="text-xs font-semibold text-text-base">Human Mode</span>
          </div>
          <div className="space-y-2 text-xs text-text-secondary">
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-accent-purple mt-1.5 flex-shrink-0" /><span>Cognitive-behavioral branches (reflection, impulse, fear, empathy, conflict...)</span></div>
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-accent-purple mt-1.5 flex-shrink-0" /><span>Routes by internal state + arousal + drive conflicts</span></div>
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-accent-purple mt-1.5 flex-shrink-0" /><span>Reward = behavioral coherence</span></div>
            <div className="flex items-start gap-2"><span className="w-1 h-1 rounded-full bg-accent-purple mt-1.5 flex-shrink-0" /><span>Stateful — same input, different routing per state</span></div>
          </div>
        </div>
      </div>
    </Card>
  );
}

// ─── Human Mode Pipeline ─────────────────────────────────────────────────
function HumanModeArchitecture() {
  const steps = [
    { label: 'Task Input', desc: 'Query enters system', color: '#64748b', icon: <Cpu size={14} /> },
    { label: 'State Snapshot', desc: 'Read 18 internal variables', color: '#0891b2', icon: <Activity size={14} /> },
    { label: 'Conflict Detection', desc: 'Check opposing drives', color: '#9333ea', icon: <AlertTriangle size={14} /> },
    { label: 'State-Conditioned Router', desc: 'Drives modulate branch scores', color: '#0066ff', icon: <GitBranch size={14} /> },
    { label: 'Branch Execution', desc: 'Cognitive modules run in parallel', color: '#059669', icon: <Layers size={14} /> },
    { label: 'Coherence Evaluation', desc: 'Score behavioral realism', color: '#d97706', icon: <Eye size={14} /> },
    { label: 'State Update', desc: 'Outcome → drive deltas + decay', color: '#e11d48', icon: <RefreshCw size={14} /> },
    { label: 'Memory Recording', desc: 'Emotional + experiential bias', color: '#7c3aed', icon: <Database size={14} /> },
  ];

  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-center gap-2">
          <div className="flex flex-col items-center">
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center text-white"
              style={{ backgroundColor: step.color }}
            >
              {step.icon}
            </div>
            <div className="mt-1.5 text-center max-w-[90px]">
              <div className="text-[10px] font-semibold text-text-base leading-tight">{step.label}</div>
              <div className="text-[9px] text-text-muted leading-tight">{step.desc}</div>
            </div>
          </div>
          {i < steps.length - 1 && (
            <ArrowRight size={14} className="text-text-placeholder mt-[-16px]" />
          )}
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────
export function HumanModeSection() {
  const [activeScenario, setActiveScenario] = useState<Scenario>('baseline');
  const [drives, setDrives] = useState<DriveState[]>(INITIAL_DRIVES);

  useEffect(() => {
    const scenario = SCENARIOS[activeScenario];
    setDrives(INITIAL_DRIVES.map(d => ({
      ...d,
      value: scenario.values[d.name] ?? d.baseline,
    })));
  }, [activeScenario]);

  const mood = (() => {
    const vals = Object.fromEntries(drives.map(d => [d.name, d.value]));
    const positive = (vals.confidence ?? 0.55) + (vals.motivation ?? 0.6) + (vals.curiosity ?? 0.6) +
                     (vals.trust ?? 0.5) + (vals.ambition ?? 0.55);
    const negative = (vals.stress ?? 0.2) + (vals.frustration ?? 0.1) + (vals.fear ?? 0.15) +
                     (vals.fatigue ?? 0.15) + (vals.self_protection ?? 0.3);
    return (positive - negative) / (positive + negative + 0.001);
  })();

  const activeConflicts = CONFLICT_PAIRS.filter(([a, b]) => {
    const valA = drives.find(d => d.name === a)?.value ?? 0;
    const valB = drives.find(d => d.name === b)?.value ?? 0;
    return valA > 0.4 && valB > 0.4 && Math.abs(valA - valB) < 0.25;
  });

  return (
    <section id="human-mode" className="py-24">
      <div className="section-container">
        {/* Header */}
        <SectionHeading
          badge="Core Insight"
          title="The Forest IS the Mind"
          subtitle="Most cognitive simulations bolt psychology onto a general-purpose engine. We realized the branch-and-forest architecture already mirrors human cognition — competing branches are competing cognitive modules, weighted routing is attention allocation, and RL weight updates are learning. Human Mode doesn't add cognition to the forest. It reveals the cognition that was always there."
          center
        />

        {/* Structural Isomorphism — THE key insight */}
        <div className="mt-16">
          <h3 className="text-lg font-semibold text-text-base text-center mb-3">
            Structural Isomorphism: Forest Architecture = Cognitive Architecture
          </h3>
          <p className="text-sm text-text-secondary text-center mb-8 max-w-3xl mx-auto">
            Every mechanism in the Adaptive Prompt Forest maps directly to a mechanism in human cognition.
            This is not metaphor — it is structural identity. The same math that makes the forest
            self-improving also makes it a faithful model of how humans think, decide, learn, and adapt.
          </p>
          <StructuralIsomorphism />
        </div>

        {/* One Engine, Two Expressions */}
        <div className="mt-20">
          <h3 className="text-lg font-semibold text-text-base text-center mb-3">
            Same Engine. Different Branch Taxonomy.
          </h3>
          <p className="text-sm text-text-secondary text-center mb-8 max-w-3xl mx-auto">
            Both modes run on the identical <code className="text-xs bg-surface px-1.5 py-0.5 rounded font-mono">PromptForestEngine</code>.
            Agent Improvement swaps in performance-optimized branches (analytical, critique, verification).
            Human Mode swaps in cognitive-behavioral branches (reflection, impulse, fear, empathy).
            The routing, execution, aggregation, and RL loops are shared.
          </p>
          <SharedEngineDiagram />
        </div>

        {/* Human Mode Pipeline */}
        <div className="mt-20">
          <h3 className="text-lg font-semibold text-text-base text-center mb-3">
            Human Mode Pipeline
          </h3>
          <p className="text-sm text-text-secondary text-center mb-8 max-w-2xl mx-auto">
            What Human Mode adds on top of the shared engine: persistent internal state, conflict detection,
            state-conditioned routing, coherence evaluation, and experiential memory with emotional amplification.
          </p>
          <Card className="p-8">
            <HumanModeArchitecture />
          </Card>
        </div>

        {/* Interactive Demo */}
        <div className="mt-20">
          <h3 className="text-lg font-semibold text-text-base text-center mb-3">
            Interactive: Same Task, Different Internal States
          </h3>
          <p className="text-sm text-text-secondary text-center mb-8 max-w-2xl mx-auto">
            Select a psychological profile below. Watch how the same question — "Should I invest in this risky opportunity?"
            — routes through completely different cognitive branches depending on the agent's internal state.
            This is the same forest engine. Different state = different routing = different behavior.
          </p>

          {/* Scenario selector */}
          <div className="flex flex-wrap justify-center gap-2 mb-8">
            {(Object.entries(SCENARIOS) as [Scenario, typeof SCENARIOS[Scenario]][]).map(([key, scenario]) => (
              <button
                key={key}
                onClick={() => setActiveScenario(key)}
                className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
                  activeScenario === key
                    ? 'bg-primary text-white shadow-glow-sm'
                    : 'bg-white border border-border text-text-secondary hover:border-border-strong hover:shadow-card'
                }`}
              >
                {scenario.label}
              </button>
            ))}
          </div>

          <p className="text-xs text-text-muted text-center mb-6">{SCENARIOS[activeScenario].desc}</p>

          {/* Avatar + Panels */}
          <div className="grid lg:grid-cols-[1fr,auto,1fr] gap-6 items-start">
            {/* Left: Drive panel */}
            {/* Drive panel */}
            <Card className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-semibold text-text-base">Internal State Variables</h4>
                <div className="flex items-center gap-3">
                  <span className={`text-xs font-mono px-2 py-0.5 rounded ${
                    mood > 0.2 ? 'bg-success-light text-success-DEFAULT' :
                    mood < -0.2 ? 'bg-danger-light text-danger-DEFAULT' :
                    'bg-surface text-text-muted'
                  }`}>
                    mood: {mood > 0 ? '+' : ''}{mood.toFixed(3)}
                  </span>
                </div>
              </div>
              <DriveVisualizer drives={drives} conflicts={activeConflicts} />
              {activeConflicts.length > 0 && (
                <div className="mt-4 p-3 rounded-lg bg-[rgba(147,51,234,0.04)] border border-accent-purple/10">
                  <div className="text-[10px] font-semibold text-accent-purple uppercase tracking-wider mb-2">Active Conflicts</div>
                  {activeConflicts.map(([a, b]) => (
                    <div key={`${a}-${b}`} className="text-xs text-text-secondary flex items-center gap-2">
                      <Scale size={12} className="text-accent-purple" />
                      <span className="font-medium" style={{ color: DRIVE_COLORS[a] }}>{a}</span>
                      <span className="text-text-muted">vs</span>
                      <span className="font-medium" style={{ color: DRIVE_COLORS[b] }}>{b}</span>
                      <span className="text-text-muted">— both branches activated, conflict resolver engaged</span>
                    </div>
                  ))}
                </div>
              )}
            </Card>

            {/* Center: Avatar */}
            <div className="flex flex-col items-center justify-start pt-4">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeScenario}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.4 }}
                >
                  <AgentAvatar3D
                    state={{
                      confidence: drives.find(d => d.name === 'confidence')?.value ?? 0.55,
                      stress: drives.find(d => d.name === 'stress')?.value ?? 0.2,
                      curiosity: drives.find(d => d.name === 'curiosity')?.value ?? 0.6,
                      fear: drives.find(d => d.name === 'fear')?.value ?? 0.15,
                      ambition: drives.find(d => d.name === 'ambition')?.value ?? 0.55,
                      empathy: drives.find(d => d.name === 'empathy')?.value ?? 0.5,
                      impulse: drives.find(d => d.name === 'impulse')?.value ?? 0.3,
                      reflection: drives.find(d => d.name === 'reflection')?.value ?? 0.5,
                      trust: drives.find(d => d.name === 'trust')?.value ?? 0.5,
                      motivation: drives.find(d => d.name === 'motivation')?.value ?? 0.6,
                      fatigue: drives.find(d => d.name === 'fatigue')?.value ?? 0.15,
                      frustration: drives.find(d => d.name === 'frustration')?.value ?? 0.1,
                    } as AvatarState}
                    size={280}
                  />
                </motion.div>
              </AnimatePresence>
              <p className="text-[10px] text-text-muted text-center mt-2 max-w-[200px]">
                Avatar reflects the agent's current emotional state in real-time
              </p>
            </div>

            {/* Right: Routing panel */}
            <Card className="p-5">
              <h4 className="text-sm font-semibold text-text-base mb-4">Cognitive Branch Routing</h4>
              <RoutingComparison scenario={activeScenario} />
              <div className="mt-4 p-3 rounded-lg bg-surface">
                <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1">Why this routing?</div>
                <p className="text-xs text-text-secondary leading-relaxed">
                  {activeScenario === 'baseline' && 'At baseline, reflective reasoning dominates with balanced curiosity. No strong drives or conflicts skew the routing — the forest explores evenly.'}
                  {activeScenario === 'confident' && 'High confidence + curiosity pushes exploration and ambition branches to the top. Fear/risk assessment is suppressed — the agent sees opportunities, not threats. The same RL weight modulation, just driven by internal state.'}
                  {activeScenario === 'anxious' && 'Fear dominates, activating threat detection. Self-protection suppresses curiosity. The agent routes defensively — the forest\'s weighted selection naturally produces avoidance behavior.'}
                  {activeScenario === 'conflicted' && 'Curiosity and fear are both high, triggering conflict detection. Both exploration and risk branches compete. The conflict resolver activates, producing nuanced output that acknowledges the tension — a structurally inevitable result of the forest architecture.'}
                  {activeScenario === 'fatigued' && 'High fatigue penalizes cognitively expensive branches (reflection). Impulse response gains priority. This is the Yerkes-Dodson effect emerging naturally from the forest\'s cost-weighted routing.'}
                </p>
              </div>
            </Card>
          </div>
        </div>

        {/* World Prediction Argument */}
        <div className="mt-24">
          <SectionHeading
            badge="Application"
            title="Why This Architecture Enables World Prediction"
            subtitle="Human behavior is the substrate of market movements, geopolitical shifts, and social dynamics. Because the forest architecture is structurally identical to human cognitive processing — not an approximation of it — the same engine that routes tasks through competing branches can predict outcomes in any domain where human decision-making is the driving force."
            center
          />
          <div className="mt-12">
            <WorldPredictionSection />
          </div>

          {/* Mathematical precision callout */}
          <Card className="mt-10 p-8 border-primary/15">
            <div className="max-w-3xl mx-auto">
              <h4 className="text-base font-semibold text-text-base mb-4 text-center">The Mathematical Core</h4>
              <div className="grid sm:grid-cols-3 gap-6 text-center">
                <div>
                  <div className="text-2xl font-bold text-primary mb-1">18</div>
                  <div className="text-xs text-text-secondary">Persistent state variables with bounded ranges, momentum, and homeostatic decay</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-accent-purple mb-1">5</div>
                  <div className="text-xs text-text-secondary">Opposition pairs that generate conflicts when both drives exceed threshold</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-accent-green mb-1">15</div>
                  <div className="text-xs text-text-secondary">Cognitive-behavioral branches mapped to the same forest engine as agent mode</div>
                </div>
              </div>

              <div className="mt-8 p-4 bg-surface rounded-lg font-mono text-xs text-text-secondary">
                <div className="text-[10px] text-text-muted mb-2 font-sans font-semibold uppercase tracking-wider">State Update Equation</div>
                <div>new_value = old * momentum + (old + delta) * (1 - momentum)</div>
                <div className="mt-1">decay = current + (baseline - current) * decay_rate</div>
                <div className="mt-1">branch_score *= (0.5 + drive_strength)  // state modulates forest routing</div>
                <div className="mt-1">conflict = drive_a &gt; 0.4 AND drive_b &gt; 0.4 AND |drive_a - drive_b| &lt; threshold</div>
                <div className="mt-1">frustration_amplifier = 1.0 + frustration  // compounding failure spiral</div>
              </div>
            </div>
          </Card>
        </div>

        {/* Cross-effects explanation */}
        <div className="mt-16">
          <h3 className="text-lg font-semibold text-text-base text-center mb-3">Cross-Variable Cascade Effects</h3>
          <p className="text-sm text-text-secondary text-center mb-8 max-w-2xl mx-auto">
            State variables don't change in isolation. High stress suppresses reflection and amplifies impulse —
            the same cascade effect that makes humans act impulsively under pressure. These cascades emerge from the
            forest's cross-variable update rules, producing realistic behavioral dynamics without explicit scripting.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { trigger: 'High Stress (>0.6)', effects: ['reflection ↓', 'impulse ↑'], desc: 'Prefrontal suppression — fast branches dominate', color: '#e11d48' },
              { trigger: 'High Fatigue (>0.7)', effects: ['motivation ↓', 'curiosity ↓'], desc: 'Cognitive resource depletion — expensive branches penalized', color: '#94a3b8' },
              { trigger: 'High Confidence (>0.7)', effects: ['ambition ↑'], desc: 'Winner effect — success breeds approach bias', color: '#0891b2' },
              { trigger: 'Repeated Failure', effects: ['frustration ↑↑', 'stress ↑ (amplified)'], desc: 'Compounding spiral — each failure hits harder via frustration amplifier', color: '#dc2626' },
            ].map(({ trigger, effects, desc, color }) => (
              <Card key={trigger} className="p-4">
                <div className="text-xs font-semibold mb-2" style={{ color }}>{trigger}</div>
                <div className="space-y-1 mb-3">
                  {effects.map(e => (
                    <div key={e} className="text-xs text-text-secondary flex items-center gap-1.5">
                      <ArrowRight size={10} className="text-text-muted" />
                      <span className="font-mono">{e}</span>
                    </div>
                  ))}
                </div>
                <div className="text-[10px] text-text-muted italic">{desc}</div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

import { useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';

// ═══════════════════════════════════════════════════════════
// Animated network canvas — shows cascading ripples
// ═══════════════════════════════════════════════════════════

interface Node { x: number; y: number; vx: number; vy: number; r: number; color: string; pulse: number; }

function CascadeCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const animRef = useRef(0);
  const rippleRef = useRef<{ x: number; y: number; r: number; alpha: number }[]>([]);

  const init = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#ef4444'];
    nodesRef.current = Array.from({ length: 50 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      r: Math.random() * 3 + 1.5,
      color: colors[Math.floor(Math.random() * colors.length)],
      pulse: Math.random() * Math.PI * 2,
    }));
  }, []);

  useEffect(() => {
    init();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    let time = 0;

    const animate = () => {
      time += 0.01;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const nodes = nodesRef.current;
      const ripples = rippleRef.current;

      // Randomly spawn ripple cascades
      if (Math.random() < 0.008) {
        const source = nodes[Math.floor(Math.random() * nodes.length)];
        ripples.push({ x: source.x, y: source.y, r: 0, alpha: 0.4 });
      }

      // Draw ripples
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rip = ripples[i];
        rip.r += 3;
        rip.alpha *= 0.97;
        if (rip.alpha < 0.01) { ripples.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(rip.x, rip.y, rip.r, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(59, 130, 246, ${rip.alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Draw connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 200) {
            const alpha = 0.06 * (1 - dist / 200);
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(100, 116, 139, ${alpha})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }

      // Draw and move nodes
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0) n.x = canvas.width;
        if (n.x > canvas.width) n.x = 0;
        if (n.y < 0) n.y = canvas.height;
        if (n.y > canvas.height) n.y = 0;

        n.pulse += 0.02;
        const pulseR = n.r + Math.sin(n.pulse) * 0.8;

        // Glow
        const glow = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, pulseR * 4);
        glow.addColorStop(0, n.color + '20');
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.fillRect(n.x - pulseR * 4, n.y - pulseR * 4, pulseR * 8, pulseR * 8);

        // Node
        ctx.beginPath();
        ctx.arc(n.x, n.y, pulseR, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      animRef.current = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(animRef.current);
  }, [init]);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />;
}


// ═══════════════════════════════════════════════════════════
// Feature Card
// ═══════════════════════════════════════════════════════════

function FeatureCard({ icon, title, description, delay }: {
  icon: string; title: string; description: string; delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.5 }}
      className="bg-white rounded-2xl border border-black/[0.04] p-6 shadow-[0_1px_3px_rgba(0,0,0,0.04)] hover:shadow-[0_4px_20px_rgba(0,0,0,0.06)] transition-shadow"
    >
      <div className="text-3xl mb-3">{icon}</div>
      <h3 className="text-[15px] font-bold text-slate-900 mb-2">{title}</h3>
      <p className="text-sm text-slate-500 leading-relaxed">{description}</p>
    </motion.div>
  );
}


// ═══════════════════════════════════════════════════════════
// Example Scenario Card
// ═══════════════════════════════════════════════════════════

function ScenarioCard({ title, description, result, color }: {
  title: string; description: string; result: string; color: string;
}) {
  return (
    <div className={`rounded-2xl p-5 border ${color}`}>
      <h4 className="text-sm font-bold text-slate-800 mb-1">{title}</h4>
      <p className="text-xs text-slate-500 mb-3">{description}</p>
      <div className="text-xs text-slate-600 bg-white/60 rounded-lg p-3 font-mono leading-relaxed">
        {result}
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════
// Main Landing Page
// ═══════════════════════════════════════════════════════════

export default function CascadeLanding() {
  return (
    <div className="min-h-screen bg-[#fafbfc]">

      {/* ═══ Hero ═══ */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0">
          <CascadeCanvas />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-white/40 via-white/60 to-white" />

        <div className="relative max-w-5xl mx-auto px-6 pt-20 pb-28 text-center">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-3 mb-8"
          >
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="5" r="3" />
                <circle cx="5" cy="19" r="3" />
                <circle cx="19" cy="19" r="3" />
                <path d="M12 8v3M8.5 17L10.5 12.5M15.5 17L13.5 12.5" />
              </svg>
            </div>
            <span className="text-3xl font-black tracking-tight text-slate-900">Cascade</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-5xl md:text-6xl font-black tracking-tight text-slate-900 mb-4 leading-[1.1]"
          >
            Simulate the ripple effects<br />
            <span className="gradient-text">of any world event</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="text-lg text-slate-500 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            Type any scenario. Cascade researches real people, injects their personalities,
            and runs hundreds of AI agents who make freeform decisions, talk to each other,
            and create butterfly effects that ripple through the simulated world.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-3"
          >
            <a
              href="#/simulate"
              className="px-8 py-3.5 rounded-xl text-white text-sm font-semibold transition-all hover:scale-[1.02]"
              style={{
                background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                boxShadow: '0 4px 15px rgba(59,130,246,0.3)',
              }}
            >
              Launch Simulator
            </a>
            <a
              href="#setup"
              className="px-8 py-3.5 rounded-xl text-slate-700 text-sm font-semibold border border-slate-200 bg-white hover:bg-slate-50 transition-all"
            >
              Run Locally
            </a>
          </motion.div>
        </div>
      </section>


      {/* ═══ How It Works ═══ */}
      <section className="py-20 border-t border-black/[0.04]">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">How it works</h2>
            <p className="text-slate-500 max-w-lg mx-auto">
              From your prediction to emergent butterfly effects in minutes.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-4 gap-6">
            <FeatureCard delay={0} icon="🔍" title="1. Research" description="GPT-5 researches all key real-world figures involved — politicians, CEOs, regulators, union leaders. Generates psychologically honest personality profiles for each person." />
            <FeatureCard delay={0.1} icon="🌍" title="2. Build World" description="Creates a simulated world with hundreds of agents, organizations, supply chains, coalitions, and rivalries. Each person has a heart engine tracking 6 emotional variables." />
            <FeatureCard delay={0.2} icon="🧠" title="3. Freeform Decisions" description="Key figures make LLM-powered decisions with no menu — they do whatever a real person would do. They send messages to each other, negotiate, threaten, and cooperate." />
            <FeatureCard delay={0.3} icon="🌊" title="4. Reactive Cascades" description="When a decision hits someone hard enough, they react immediately. Reactions trigger more reactions. A single CEO's decision can cascade through 6 levels of consequences." />
          </div>
        </div>
      </section>


      {/* ═══ What Makes It Different ═══ */}
      <section className="py-20 bg-white border-t border-black/[0.04]">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">Not a prediction engine. A discovery engine.</h2>
            <p className="text-slate-500 max-w-xl mx-auto">
              Normal simulations tell you what you already know. Cascade finds what you didn't expect.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard delay={0} icon="🎭" title="Emotional Masking" description="Leaders whose public composure hides internal panic. The TSMC CEO shows 'calm' while internally terrified his life's work is being destroyed. This masking costs energy and makes future decisions unpredictable." />
            <FeatureCard delay={0.1} icon="🔄" title="Sector Paradoxes" description="Healthcare sector booms during a crisis but healthcare workers are psychologically devastated. The sector gets funding; the humans get burned out. Sector-level data hides worker-level suffering." />
            <FeatureCard delay={0.2} icon="💸" title="Savings Depletion Waves" description="Finds which groups burn through savings first. When 142 office workers hit zero savings simultaneously, their spending collapse hits local businesses in a secondary wave nobody predicted." />
            <FeatureCard delay={0} icon="⚡" title="Cascade Sources" description="Identifies the single person whose decision triggered the most downstream consequences. Tim Cook committing Apple to relief corridors cascaded to 11 people — the biggest single action in the simulation." />
            <FeatureCard delay={0.1} icon="🗜️" title="Compound Squeeze" description="Finds groups caught as losers in 3+ policies simultaneously. Office professionals hit by housing, trade, labour, fiscal, healthcare, education, defence, and trade policies all at once." />
            <FeatureCard delay={0.2} icon="🧠" title="Psychological vs Financial" description="Regional bank tellers have zero financial exposure but pessimism surges +0.64. They're destroyed by anxiety about what MIGHT happen, not what has. The fear is worse than the reality." />
          </div>
        </div>
      </section>


      {/* ═══ Example Scenarios ═══ */}
      <section className="py-20 border-t border-black/[0.04]">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">What people have simulated</h2>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-5">
            <ScenarioCard
              title="Zombie bee virus in rural Australia"
              description="A novel virus crosses from insects to mammals via bee sting"
              result="435 decisions, 1680 ripples. Brad Banducci (Woolworths) negotiates with Sally McManus (ACTU) over worker safety. Every CEO independently calls Health Minister Mark Butler — he becomes the convergence point. 14 beekeepers burn through savings first."
              color="bg-emerald-50 border-emerald-200"
            />
            <ScenarioCard
              title="Nuclear weapon detonates in Shanghai"
              description="Geopolitical and economic fallout across global systems"
              result="435 decisions, 2081 ripples. Xi Jinping accepts Biden's forensic help while keeping 'full oversight.' Tim Cook's single decision cascades to 11 people. Oil/shipping workers in the Middle East are the hardest hit group — nobody predicted this."
              color="bg-red-50 border-red-200"
            />
            <ScenarioCard
              title="Elon Musk dies suddenly"
              description="Impact on Tesla, SpaceX, X/Twitter, xAI, Neuralink"
              result="19 key figures including Robyn Denholm, Gwynne Shotwell, Gary Gensler. DoD pauses SpaceX contracts. Neuralink trials frozen. X/Twitter ad sales team psychologically destroyed despite no financial hit. SpaceX engineers: financially stable but shattered by dread."
              color="bg-amber-50 border-amber-200"
            />
            <ScenarioCard
              title="Biggest US property developer goes bankrupt"
              description="Cascading effects through housing, banking, construction"
              result="1109 agents. Construction sector booms (+0.5) but construction workers are the MOST devastated (+0.77 pessimism) — sector paradox. Homebuyers mid-transaction psychologically destroyed despite debt actually decreasing. 511 people burn through savings."
              color="bg-blue-50 border-blue-200"
            />
          </div>
        </div>
      </section>


      {/* ═══ Architecture ═══ */}
      <section className="py-20 bg-white border-t border-black/[0.04]">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3">Under the hood</h2>
            <p className="text-slate-500 max-w-lg mx-auto">
              Hybrid deterministic + LLM architecture. The deterministic engine provides physics. LLMs provide intelligence.
            </p>
          </motion.div>

          <div className="bg-slate-900 rounded-2xl p-8 text-white font-mono text-sm leading-relaxed overflow-x-auto">
            <pre className="text-slate-300">{`
  User Prediction
       │
       ▼
  ┌─────────────────────────┐
  │  GPT-5 Research Layer   │  ← Researches real people, policies, economic impacts
  │  (2 parallel API calls) │
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │  World Builder          │  ← 500-1000 agents, org fabric, relationships
  │  + Policy Engine        │  ← Persistent conditions, sector impacts
  │  + Shock Injector       │  ← Narrative events at specific locations
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────┐
  │  Simulation Tick Loop (24 ticks/day)            │
  │                                                 │
  │  ┌──────────────┐  ┌──────────────────────────┐ │
  │  │ HeartState    │  │ Freeform LLM Agents      │ │
  │  │ Engine        │  │ (no menu, real decisions) │ │
  │  │ (numpy/SBERT) │  │                          │ │
  │  │ 6 emotional   │  │ ┌─ Scheduled decisions   │ │
  │  │ variables     │  │ ├─ Reactive cascades     │ │
  │  │ per agent     │  │ ├─ Agent conversations   │ │
  │  └──────┬───────┘  │ └─ Message delivery      │ │
  │         │          └──────────┬───────────────┘ │
  │         │                     │                 │
  │         ▼                     ▼                 │
  │  ┌──────────────────────────────────────┐       │
  │  │ Ripple Engine                        │       │
  │  │ Person → Person consequence chains   │       │
  │  │ Traceable cascades up to 6 levels    │       │
  │  └──────────────────────────────────────┘       │
  │                                                 │
  │  ┌──────────────────────────────────────┐       │
  │  │ Macro Aggregator                     │       │
  │  │ Consumer confidence, civil unrest,   │       │
  │  │ market pressure, institutional trust │       │
  │  └──────────────────────────────────────┘       │
  └─────────────────────┬───────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────┐
  │  Insight Engine                                 │
  │  Scans for non-obvious emergent patterns:       │
  │  - Savings depletion waves                      │
  │  - Sector paradoxes                             │
  │  - Emotional masking                            │
  │  - Cascade sources / victims                    │
  │  - Compound policy squeeze                      │
  │  - Counterintuitive resilience                  │
  └─────────────────────────────────────────────────┘
`}</pre>
          </div>
        </div>
      </section>


      {/* ═══ Setup Guide ═══ */}
      <section id="setup" className="py-20 border-t border-black/[0.04]">
        <div className="max-w-3xl mx-auto px-6">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
            <h2 className="text-3xl font-bold text-slate-900 mb-3 text-center">Run locally</h2>
            <p className="text-slate-500 text-center mb-10 max-w-lg mx-auto">
              Cascade runs entirely on your machine. You only need Python, Node.js, and an OpenAI API key.
            </p>

            <div className="space-y-6">
              {/* Step 1 */}
              <div className="bg-white rounded-2xl border border-black/[0.04] p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-7 h-7 rounded-lg bg-blue-50 text-blue-600 text-sm font-bold flex items-center justify-center">1</span>
                  <h3 className="text-[15px] font-bold text-slate-900">Clone and install</h3>
                </div>
                <div className="bg-slate-900 rounded-xl p-4 font-mono text-sm text-slate-300 overflow-x-auto">
                  <pre>{`git clone https://github.com/herehere14/Cascade-An-AI-Agent-World-Simulator.git
cd Cascade-An-AI-Agent-World-Simulator

# Backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..`}</pre>
                </div>
              </div>

              {/* Step 2 */}
              <div className="bg-white rounded-2xl border border-black/[0.04] p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-7 h-7 rounded-lg bg-blue-50 text-blue-600 text-sm font-bold flex items-center justify-center">2</span>
                  <h3 className="text-[15px] font-bold text-slate-900">Set your OpenAI API key</h3>
                </div>
                <div className="bg-slate-900 rounded-xl p-4 font-mono text-sm text-slate-300 overflow-x-auto">
                  <pre>{`export OPENAI_API_KEY="sk-your-key-here"`}</pre>
                </div>
                <p className="text-xs text-slate-400 mt-2">
                  Cascade uses GPT-5-mini for research and agent decisions. A typical 7-day simulation with LLM agents costs ~$1-3 in API calls.
                </p>
              </div>

              {/* Step 3 */}
              <div className="bg-white rounded-2xl border border-black/[0.04] p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-7 h-7 rounded-lg bg-blue-50 text-blue-600 text-sm font-bold flex items-center justify-center">3</span>
                  <h3 className="text-[15px] font-bold text-slate-900">Start the servers</h3>
                </div>
                <div className="bg-slate-900 rounded-xl p-4 font-mono text-sm text-slate-300 overflow-x-auto">
                  <pre>{`# Terminal 1: API server
python api_server.py

# Terminal 2: Frontend
cd frontend && npm run dev`}</pre>
                </div>
                <p className="text-xs text-slate-400 mt-2">
                  API runs on <code className="text-blue-400">localhost:8000</code>, frontend on <code className="text-blue-400">localhost:3000</code>.
                </p>
              </div>

              {/* Step 4 */}
              <div className="bg-white rounded-2xl border border-black/[0.04] p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-7 h-7 rounded-lg bg-blue-50 text-blue-600 text-sm font-bold flex items-center justify-center">4</span>
                  <h3 className="text-[15px] font-bold text-slate-900">Simulate</h3>
                </div>
                <p className="text-sm text-slate-600 mb-3">
                  Open <code className="text-blue-500 bg-blue-50 px-1.5 py-0.5 rounded text-xs">http://localhost:3000/#/simulate</code> and type any scenario.
                </p>
                <p className="text-sm text-slate-600 mb-3">
                  Or use Python directly:
                </p>
                <div className="bg-slate-900 rounded-xl p-4 font-mono text-sm text-slate-300 overflow-x-auto">
                  <pre>{`from world_sim.run_prediction import run_prediction

report = run_prediction(
    "What if a zombie virus breaks out in rural Australia?",
    llm_agents=True,       # Enable freeform LLM agent decisions
    llm_agents_per_tick=4,  # 4 decisions per simulated hour
    days=7,                 # 7 simulated days
)`}</pre>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>


      {/* ═══ Footer ═══ */}
      <footer className="py-12 border-t border-black/[0.04] bg-white">
        <div className="max-w-5xl mx-auto px-6 text-center">
          <div className="flex items-center justify-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <svg viewBox="0 0 24 24" className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="12" cy="5" r="3" />
                <circle cx="5" cy="19" r="3" />
                <circle cx="19" cy="19" r="3" />
                <path d="M12 8v3M8.5 17L10.5 12.5M15.5 17L13.5 12.5" />
              </svg>
            </div>
            <span className="text-sm font-bold text-slate-900">Cascade</span>
          </div>
          <p className="text-xs text-slate-400">
            Built with HeartState engine, SBERT embeddings, GPT-5 freeform agents, and reactive cascade architecture.
          </p>
        </div>
      </footer>
    </div>
  );
}

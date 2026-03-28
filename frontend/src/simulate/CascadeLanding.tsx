import { useEffect, useRef, useCallback, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

// ═══════════════════════════════════════════════════════════
// Ripple Canvas — dark, minimal, data-network aesthetic
// ═══════════════════════════════════════════════════════════

function RippleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef(0);

  interface Node { x: number; y: number; vx: number; vy: number; r: number; alpha: number; }

  const init = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
  }, []);

  useEffect(() => {
    init();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;

    const nodes: Node[] = Array.from({ length: 70 }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5, alpha: Math.random() * 0.4 + 0.1,
    }));

    const ripples: { x: number; y: number; r: number; a: number }[] = [];
    let frame = 0;

    const animate = () => {
      frame++;
      ctx.fillStyle = 'rgba(8,8,12,0.15)';
      ctx.fillRect(0, 0, W, H);

      // Spawn ripple
      if (frame % 120 === 0) {
        const n = nodes[Math.floor(Math.random() * nodes.length)];
        ripples.push({ x: n.x, y: n.y, r: 0, a: 0.25 });
      }

      // Draw ripples
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rip = ripples[i];
        rip.r += 1.5;
        rip.a *= 0.985;
        if (rip.a < 0.005) { ripples.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(rip.x, rip.y, rip.r, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(59,130,246,${rip.a})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 160) {
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(59,130,246,${0.04 * (1 - dist / 160)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      // Draw + move nodes
      for (const n of nodes) {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0) n.x = W; if (n.x > W) n.x = 0;
        if (n.y < 0) n.y = H; if (n.y > H) n.y = 0;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(59,130,246,${n.alpha})`;
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(animate);
    };

    ctx.fillStyle = '#08080c';
    ctx.fillRect(0, 0, W, H);
    animate();
    return () => cancelAnimationFrame(animRef.current);
  }, [init]);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />;
}

// ═══════════════════════════════════════════════════════════
// Scroll-reveal wrapper
// ═══════════════════════════════════════════════════════════

function Reveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.7, delay, ease: [0.4, 0, 0.2, 1] }}
    >
      {children}
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════
// Counter animation
// ═══════════════════════════════════════════════════════════

function AnimatedNumber({ value, suffix = '' }: { value: number; suffix?: string }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          let start = 0;
          const duration = 1500;
          const startTime = Date.now();
          const tick = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setDisplay(Math.round(eased * value));
            if (progress < 1) requestAnimationFrame(tick);
          };
          tick();
          observer.disconnect();
        }
      },
      { threshold: 0.5 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [value]);

  return <span ref={ref}>{display.toLocaleString()}{suffix}</span>;
}

// ═══════════════════════════════════════════════════════════
// Main Landing Page
// ═══════════════════════════════════════════════════════════

export default function CascadeLanding() {
  const { scrollYProgress } = useScroll();
  const headerOpacity = useTransform(scrollYProgress, [0, 0.05], [0, 1]);

  return (
    <div className="bg-[#08080c] text-white min-h-screen" style={{ fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif' }}>

      {/* ═══ Fixed Header ═══ */}
      <motion.header
        style={{ opacity: headerOpacity }}
        className="fixed top-0 left-0 right-0 z-50 border-b border-white/[0.06]"
      >
        <div className="backdrop-blur-xl bg-[#08080c]/80">
          <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <img src="./favicon.svg" alt="RippleSim" className="w-7 h-7 rounded-lg" />
              <span className="text-[15px] font-semibold tracking-tight">RippleSim</span>
            </div>
            <nav className="hidden md:flex items-center gap-8 text-[13px] text-white/50">
              <a href="#how" className="hover:text-white transition-colors">How it works</a>
              <a href="#capabilities" className="hover:text-white transition-colors">Capabilities</a>
              <a href="#scenarios" className="hover:text-white transition-colors">Scenarios</a>
              <a href="#setup" className="hover:text-white transition-colors">Run locally</a>
            </nav>
            <a href="#/simulate" className="text-[13px] font-medium text-white bg-white/[0.08] hover:bg-white/[0.14] px-4 py-2 rounded-lg transition-colors border border-white/[0.08]">
              Launch &rarr;
            </a>
          </div>
        </div>
      </motion.header>

      {/* ═══ Hero ═══ */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <RippleCanvas />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#08080c]" />

        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1.2 }}
          >
            <div className="inline-flex items-center gap-2 text-[13px] text-blue-400/70 mb-8 border border-blue-500/20 rounded-full px-4 py-1.5 bg-blue-500/[0.05]">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
              Agent-based world simulation engine
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-5xl md:text-7xl font-bold tracking-[-0.03em] leading-[1.05] mb-6"
          >
            One decision.<br />
            <span className="text-white/40">A thousand consequences.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="text-lg md:text-xl text-white/40 max-w-2xl mx-auto mb-12 leading-relaxed"
          >
            RippleSim builds a world of AI agents from any scenario you describe.
            They make freeform decisions, communicate with each other, and create
            butterfly effects that no one predicted.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <a
              href="#/simulate"
              className="px-8 py-3.5 rounded-lg text-[14px] font-medium bg-blue-600 hover:bg-blue-500 transition-colors"
            >
              Launch Simulator
            </a>
            <a
              href="#setup"
              className="px-8 py-3.5 rounded-lg text-[14px] font-medium text-white/60 border border-white/10 hover:border-white/20 hover:text-white/80 transition-colors"
            >
              Run Locally
            </a>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-5 h-8 rounded-full border border-white/20 flex justify-center pt-1.5"
          >
            <div className="w-1 h-2 bg-white/30 rounded-full" />
          </motion.div>
        </motion.div>
      </section>


      {/* ═══ Stats Strip ═══ */}
      <section className="border-y border-white/[0.06] py-16">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-12 text-center">
            {[
              { value: 1000, suffix: '+', label: 'Agents per simulation' },
              { value: 435, suffix: '', label: 'Freeform LLM decisions' },
              { value: 2081, suffix: '', label: 'Ripple cascade events' },
              { value: 6, suffix: '', label: 'Max cascade depth' },
            ].map((stat, i) => (
              <Reveal key={i} delay={i * 0.1}>
                <div className="text-4xl md:text-5xl font-bold tracking-tight text-white mb-2">
                  <AnimatedNumber value={stat.value} suffix={stat.suffix} />
                </div>
                <div className="text-[13px] text-white/30 uppercase tracking-wider">{stat.label}</div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>


      {/* ═══ How It Works ═══ */}
      <section id="how" className="py-28">
        <div className="max-w-6xl mx-auto px-6">
          <Reveal>
            <div className="text-[13px] text-blue-400/60 uppercase tracking-[0.15em] mb-4">Process</div>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-16">How it works</h2>
          </Reveal>

          <div className="grid md:grid-cols-4 gap-px bg-white/[0.04] rounded-xl overflow-hidden">
            {[
              { num: '01', title: 'Research', desc: 'GPT-5 researches real key figures — politicians, CEOs, regulators, union leaders. Generates psychologically honest personality profiles based on real public behavior.' },
              { num: '02', title: 'Build', desc: 'Creates a world with hundreds of agents, organizational fabrics, supply chains, coalitions, and rivalries. Each person has a heart engine tracking arousal, valence, tension, impulse, energy, vulnerability.' },
              { num: '03', title: 'Decide', desc: 'Key figures make freeform LLM decisions — no menu, no constraints. They negotiate, threaten, organize, leak information, comfort allies, or do anything a real person would do.' },
              { num: '04', title: 'Cascade', desc: 'When a decision hits someone hard enough, they react immediately. Their reaction triggers more reactions. A single CEO\'s choice cascades through 6 levels of consequences.' },
            ].map((step, i) => (
              <Reveal key={i} delay={i * 0.1}>
                <div className="bg-[#0c0c10] p-8 h-full">
                  <div className="text-blue-500/40 text-[13px] font-mono mb-4">{step.num}</div>
                  <h3 className="text-lg font-semibold mb-3">{step.title}</h3>
                  <p className="text-[14px] text-white/35 leading-relaxed">{step.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>


      {/* ═══ Capabilities ═══ */}
      <section id="capabilities" className="py-28 border-t border-white/[0.06]">
        <div className="max-w-6xl mx-auto px-6">
          <Reveal>
            <div className="text-[13px] text-blue-400/60 uppercase tracking-[0.15em] mb-4">Capabilities</div>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4">Not a prediction engine.</h2>
            <p className="text-xl text-white/30 mb-16 max-w-xl">A discovery engine. It finds what you didn't expect.</p>
          </Reveal>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              { title: 'Emotional Masking', desc: 'Detects leaders whose public composure hides internal panic. The TSMC CEO shows "calm" while internally terrified. This masking costs energy and makes future decisions unpredictable.', tag: 'Psychology' },
              { title: 'Sector Paradoxes', desc: 'Healthcare sector booms during a crisis but healthcare workers are psychologically devastated. The sector gets funding. The humans get burned out. Sector data hides worker suffering.', tag: 'Economics' },
              { title: 'Savings Depletion Waves', desc: 'Finds which groups burn through savings first. When 142 office workers hit zero simultaneously, their spending collapse hits local businesses in a secondary wave nobody saw coming.', tag: 'Emergent' },
              { title: 'Cascade Source Detection', desc: 'Identifies the single person whose decision triggered the most downstream consequences. Tim Cook committing Apple to relief corridors cascaded to 11 people — the biggest single action.', tag: 'Network' },
              { title: 'Compound Policy Squeeze', desc: 'Finds groups caught as losers in 3+ policies simultaneously. Office professionals hit by housing, trade, labour, fiscal, healthcare, education, and defence policies all at once.', tag: 'Policy' },
              { title: 'Reactive Conversations', desc: 'Agents at the same location have real-time multi-turn conversations. Brad Banducci messages Sally McManus about supply continuity. Sally responds with conditions on worker safety.', tag: 'Agent' },
            ].map((cap, i) => (
              <Reveal key={i} delay={i * 0.08}>
                <div className="border border-white/[0.06] rounded-xl p-6 hover:border-white/[0.12] transition-colors group h-full">
                  <div className="text-[11px] text-blue-400/50 uppercase tracking-wider mb-3">{cap.tag}</div>
                  <h3 className="text-[15px] font-semibold mb-3 group-hover:text-blue-400 transition-colors">{cap.title}</h3>
                  <p className="text-[13px] text-white/30 leading-relaxed">{cap.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>


      {/* ═══ Scenarios ═══ */}
      <section id="scenarios" className="py-28 border-t border-white/[0.06]">
        <div className="max-w-6xl mx-auto px-6">
          <Reveal>
            <div className="text-[13px] text-blue-400/60 uppercase tracking-[0.15em] mb-4">Validated scenarios</div>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-16">What people have simulated</h2>
          </Reveal>

          <div className="space-y-px">
            {[
              {
                title: 'Zombie bee virus — Rural Australia',
                agents: '925 agents', decisions: '435 decisions', ripples: '1,680 ripples',
                finding: 'Brad Banducci (Woolworths CEO) independently messages Sally McManus (ACTU Secretary) to negotiate worker safety conditions. Every CEO in the simulation independently calls Health Minister Mark Butler — he becomes the convergence point nobody designed.',
              },
              {
                title: 'Nuclear weapon detonates in Shanghai',
                agents: '918 agents', decisions: '435 decisions', ripples: '2,081 ripples',
                finding: 'Xi Jinping accepts Biden\'s forensic help while keeping "full oversight" — cooperating enough to control the narrative. Tim Cook\'s single decision to open Apple relief corridors cascades to 11 people. Middle East oil workers are the hardest hit group — nobody predicted this.',
              },
              {
                title: 'Elon Musk dies suddenly',
                agents: '718 agents', decisions: '222 decisions', ripples: '983 ripples',
                finding: 'X/Twitter ad sales team psychologically destroyed despite no financial impact — dread about the future does more damage than actual losses. DoD pauses SpaceX contracts and opens competitive rebidding. Neuralink human trials immediately frozen.',
              },
              {
                title: 'Biggest US property developer goes bankrupt',
                agents: '1,109 agents', decisions: '—', ripples: '—',
                finding: 'Construction sector booms (+0.5) but construction workers are the most devastated group (+0.77 pessimism). Homebuyers mid-transaction are psychologically destroyed despite their debt actually decreasing. 511 people burn through savings.',
              },
            ].map((scenario, i) => (
              <Reveal key={i} delay={i * 0.1}>
                <div className="bg-[#0c0c10] border border-white/[0.04] rounded-xl p-8 mb-4 hover:border-white/[0.08] transition-colors">
                  <div className="flex flex-col md:flex-row md:items-start gap-6">
                    <div className="md:w-1/3">
                      <h3 className="text-[16px] font-semibold mb-3">{scenario.title}</h3>
                      <div className="flex gap-3 text-[12px] text-white/25 font-mono">
                        <span>{scenario.agents}</span>
                        <span className="text-white/10">|</span>
                        <span>{scenario.decisions}</span>
                        <span className="text-white/10">|</span>
                        <span>{scenario.ripples}</span>
                      </div>
                    </div>
                    <div className="md:w-2/3">
                      <div className="text-[11px] text-blue-400/40 uppercase tracking-wider mb-2">Non-obvious finding</div>
                      <p className="text-[14px] text-white/40 leading-relaxed">{scenario.finding}</p>
                    </div>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>


      {/* ═══ Architecture ═══ */}
      <section className="py-28 border-t border-white/[0.06]">
        <div className="max-w-6xl mx-auto px-6">
          <Reveal>
            <div className="text-[13px] text-blue-400/60 uppercase tracking-[0.15em] mb-4">Architecture</div>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-16">Under the hood</h2>
          </Reveal>

          <Reveal delay={0.2}>
            <div className="border border-white/[0.06] rounded-xl p-8 font-mono text-[13px] text-white/30 leading-relaxed overflow-x-auto">
              <pre>{`User Prediction
     │
     ├─── GPT-5 Research ──── Key figures + personality profiles
     │                        Policy positions + economic impacts
     │
     ├─── World Builder ───── 500-1000 agents with HeartState engine
     │                        Organizational fabric + supply chains
     │                        Coalitions, rivalries, relationships
     │
     ├─── Simulation Loop ─── 24 ticks/day
     │    │
     │    ├── HeartState ──── arousal, valence, tension,
     │    │   (numpy/SBERT)   impulse_control, energy, vulnerability
     │    │
     │    ├── LLM Agents ─── Freeform decisions (no menu)
     │    │                   Agent-to-agent messaging
     │    │                   Reactive cascades (depth 6)
     │    │                   Multi-turn conversations
     │    │
     │    ├── Ripple Engine ─ Person → person consequence chains
     │    │                   Manager cuts hours → worker spends less
     │    │                   → vendor raises prices → retiree can't eat
     │    │
     │    └── Macro Agg ──── Consumer confidence, civil unrest,
     │                       market pressure, institutional trust
     │
     └─── Insight Engine ─── Savings depletion waves
                             Sector paradoxes
                             Emotional masking detection
                             Cascade source/victim identification
                             Compound policy squeeze
                             Counterintuitive resilience`}</pre>
            </div>
          </Reveal>
        </div>
      </section>


      {/* ═══ Setup ═══ */}
      <section id="setup" className="py-28 border-t border-white/[0.06]">
        <div className="max-w-3xl mx-auto px-6">
          <Reveal>
            <div className="text-[13px] text-blue-400/60 uppercase tracking-[0.15em] mb-4">Getting started</div>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-16">Run locally</h2>
          </Reveal>

          <div className="space-y-6">
            {[
              {
                num: '01', title: 'Clone and install',
                code: `git clone https://github.com/herehere14/Cascade-An-AI-Agent-World-Simulator.git
cd Cascade-An-AI-Agent-World-Simulator

# Backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..`,
              },
              {
                num: '02', title: 'Set your OpenAI API key',
                code: `export OPENAI_API_KEY="sk-your-key-here"`,
                note: 'Uses GPT-5-mini. A 7-day simulation with LLM agents costs ~$1-3.',
              },
              {
                num: '03', title: 'Start',
                code: `# Terminal 1 — API server
python api_server.py

# Terminal 2 — Frontend
cd frontend && npm run dev`,
                note: 'API on localhost:8000, frontend on localhost:3000.',
              },
              {
                num: '04', title: 'Simulate',
                code: `# Open http://localhost:3000/#/simulate
# Or use Python:
from world_sim.run_prediction import run_prediction

report = run_prediction(
    "What if a zombie virus breaks out in rural Australia?",
    llm_agents=True,
    llm_agents_per_tick=4,
    days=7,
)`,
              },
            ].map((step, i) => (
              <Reveal key={i} delay={i * 0.1}>
                <div className="border border-white/[0.06] rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-[13px] font-mono text-blue-500/50">{step.num}</span>
                    <h3 className="text-[15px] font-semibold">{step.title}</h3>
                  </div>
                  <div className="bg-[#0c0c10] rounded-lg p-4 font-mono text-[13px] text-white/40 overflow-x-auto">
                    <pre>{step.code}</pre>
                  </div>
                  {step.note && (
                    <p className="text-[12px] text-white/20 mt-3">{step.note}</p>
                  )}
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>


      {/* ═══ CTA ═══ */}
      <section className="py-28 border-t border-white/[0.06]">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <Reveal>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4">
              What happens if<span className="text-white/20">...</span>
            </h2>
            <p className="text-lg text-white/30 mb-10">
              Type any scenario. Watch the consequences unfold.
            </p>
            <a
              href="#/simulate"
              className="inline-block px-10 py-4 rounded-lg text-[15px] font-medium bg-blue-600 hover:bg-blue-500 transition-colors"
            >
              Launch Simulator
            </a>
          </Reveal>
        </div>
      </section>


      {/* ═══ Footer ═══ */}
      <footer className="border-t border-white/[0.06] py-12">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <img src="./favicon.svg" alt="RippleSim" className="w-5 h-5 rounded" />
              <span className="text-[13px] font-semibold text-white/60">RippleSim</span>
            </div>
            <div className="flex items-center gap-6 text-[12px] text-white/25">
              <a href="https://github.com/herehere14/Cascade-An-AI-Agent-World-Simulator" className="hover:text-white/50 transition-colors" target="_blank" rel="noopener">GitHub</a>
              <a href="#/simulate" className="hover:text-white/50 transition-colors">Simulator</a>
              <a href="#setup" className="hover:text-white/50 transition-colors">Documentation</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

import { useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { GlowButton } from '@/components/ui';
import { Play, GitCompare, Layers, Brain } from 'lucide-react';

interface Particle {
  x: number; y: number;
  vx: number; vy: number;
  size: number; alpha: number;
  color: string;
}

function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animRef = useRef<number>(0);

  const init = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;

    const colors = ['#0066ff', '#7c3aed', '#ec4899', '#059669', '#2563eb'];
    particlesRef.current = Array.from({ length: 60 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      size: Math.random() * 2 + 0.5,
      alpha: Math.random() * 0.25 + 0.05,
      color: colors[Math.floor(Math.random() * colors.length)],
    }));
  }, []);

  useEffect(() => {
    init();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const particles = particlesRef.current;

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.fill();
      }

      // Draw connections
      ctx.strokeStyle = '#0066ff';
      ctx.lineWidth = 0.5;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 180) {
            ctx.globalAlpha = 0.04 * (1 - dist / 180);
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
      animRef.current = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animRef.current);
  }, [init]);

  return <canvas ref={canvasRef} className="particle-canvas" />;
}

function GlowingSphere() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const size = 400;
    canvas.width = size * 2;
    canvas.height = size * 2;
    const ctx = canvas.getContext('2d')!;
    const cx = size;
    const cy = size;
    let time = 0;

    const animate = () => {
      time += 0.008;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Outer glow
      const outerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, size);
      outerGrad.addColorStop(0, 'rgba(0,102,255,0.04)');
      outerGrad.addColorStop(0.4, 'rgba(124,58,237,0.02)');
      outerGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = outerGrad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Main sphere
      const r = 120 + Math.sin(time) * 5;
      const grad = ctx.createRadialGradient(cx - 20, cy - 30, 0, cx, cy, r);
      grad.addColorStop(0, 'rgba(0,102,255,0.12)');
      grad.addColorStop(0.4, 'rgba(124,58,237,0.08)');
      grad.addColorStop(0.7, 'rgba(236,72,153,0.04)');
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Ring
      ctx.beginPath();
      ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(0,102,255,${0.12 + Math.sin(time * 2) * 0.04})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Orbiting nodes
      const branches = [
        { angle: time * 0.4, dist: r + 35, color: '#0066ff', size: 5 },
        { angle: time * 0.3 + 1, dist: r + 50, color: '#7c3aed', size: 4 },
        { angle: time * 0.5 + 2, dist: r + 28, color: '#059669', size: 4 },
        { angle: time * 0.35 + 3, dist: r + 45, color: '#ec4899', size: 3.5 },
        { angle: time * 0.45 + 4, dist: r + 38, color: '#d97706', size: 3.5 },
        { angle: time * 0.25 + 5, dist: r + 55, color: '#2563eb', size: 4.5 },
      ];

      for (const b of branches) {
        const bx = cx + Math.cos(b.angle) * b.dist;
        const by = cy + Math.sin(b.angle) * b.dist;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(bx, by);
        ctx.strokeStyle = b.color + '15';
        ctx.lineWidth = 1;
        ctx.stroke();

        const ng = ctx.createRadialGradient(bx, by, 0, bx, by, b.size * 4);
        ng.addColorStop(0, b.color + '25');
        ng.addColorStop(1, 'transparent');
        ctx.fillStyle = ng;
        ctx.fillRect(bx - b.size * 4, by - b.size * 4, b.size * 8, b.size * 8);

        ctx.beginPath();
        ctx.arc(bx, by, b.size, 0, Math.PI * 2);
        ctx.fillStyle = b.color;
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="w-[200px] h-[200px] sm:w-[300px] sm:h-[300px] lg:w-[400px] lg:h-[400px] animate-float"
      style={{ imageRendering: 'auto' }}
    />
  );
}

export function HeroSection({ onScrollTo }: { onScrollTo: (id: string) => void }) {
  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden hero-gradient">
      <ParticleCanvas />

      <div className="relative z-10 section-container flex flex-col lg:flex-row items-center gap-8 lg:gap-16 py-20">
        {/* Text */}
        <div className="flex-1 flex flex-col gap-6 items-center lg:items-start text-center lg:text-left">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-primary-light text-primary border border-primary/15">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              Open Source
            </span>
          </motion.div>

          <motion.h1
            className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-[1.1] tracking-tight"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <span className="text-text-base">OpenClaw</span>{' '}
            <span className="gradient-text">Adaptive</span>
            <br />
            <span className="gradient-text">Prompt Forest</span>
          </motion.h1>

          <motion.p
            className="text-lg sm:text-xl text-text-secondary max-w-lg"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            A self-improving forest of competing branches — the same architecture that models human cognition, enabling world prediction.
          </motion.p>

          <motion.div
            className="flex flex-col gap-3 text-sm text-text-muted"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            {[
              'Competing branches = competing cognitive modules',
              'RL weight updates = experiential learning',
              'Same engine powers both agent optimization and human cognition',
              'Predicts world outcomes by mirroring how humans decide',
            ].map((text) => (
              <div key={text} className="flex items-center gap-2">
                <span className="w-1 h-1 rounded-full bg-primary" />
                <span>{text}</span>
              </div>
            ))}
          </motion.div>

          <motion.div
            className="flex flex-wrap gap-3 mt-2"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <GlowButton onClick={() => onScrollTo('human-mode')} size="lg">
              <Brain size={16} /> Explore Human Mode
            </GlowButton>
            <GlowButton variant="secondary" onClick={() => onScrollTo('console')} size="lg">
              <Play size={16} /> Run Live Demo
            </GlowButton>
            <GlowButton variant="ghost" onClick={() => onScrollTo('architecture')} size="lg">
              <Layers size={16} /> Architecture
            </GlowButton>
          </motion.div>
        </div>

        {/* Sphere */}
        <motion.div
          className="flex-shrink-0"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
        >
          <GlowingSphere />
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
      >
        <span className="text-[10px] uppercase tracking-widest text-text-muted">Scroll to explore</span>
        <motion.div
          className="w-5 h-8 rounded-full border border-border flex items-start justify-center pt-1.5"
          animate={{ y: [0, 4, 0] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          <div className="w-1 h-2 rounded-full bg-primary" />
        </motion.div>
      </motion.div>
    </section>
  );
}

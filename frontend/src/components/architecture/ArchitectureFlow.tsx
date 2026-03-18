import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  MessageSquare, Brain, BarChart3, GitBranch, Cpu,
  CheckCircle, TrendingUp, Database, Compass, ArrowDown
} from 'lucide-react';
import { SectionHeading, Card } from '@/components/ui';
import { ARCHITECTURE_STEPS } from '@/lib/mockData';

const iconMap: Record<string, React.ElementType> = {
  MessageSquare, Brain, BarChart3, GitBranch, Cpu,
  CheckCircle, TrendingUp, Database, Compass,
};

const stepColors = [
  '#e4e4e7', '#7c3aed', '#3b82f6', '#0066ff',
  '#22d3ee', '#10b981', '#f59e0b', '#6366f1', '#ec4899',
];

export function ArchitectureFlow() {
  const [activeStep, setActiveStep] = useState(-1);
  const [autoPlay, setAutoPlay] = useState(false);

  useEffect(() => {
    if (!autoPlay) return;
    let step = 0;
    const interval = setInterval(() => {
      setActiveStep(step);
      step++;
      if (step >= ARCHITECTURE_STEPS.length) {
        step = 0;
        // Reset and loop
        setTimeout(() => setActiveStep(-1), 800);
      }
    }, 1200);
    return () => clearInterval(interval);
  }, [autoPlay]);

  return (
    <section id="architecture" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="System Design"
          title="Architecture Overview"
          subtitle="The complete adaptive prompt forest pipeline — from user query to future route biasing."
          center
        />

        <div className="mt-12 max-w-3xl mx-auto">
          {/* Auto-play toggle */}
          <div className="flex justify-center mb-8">
            <button
              onClick={() => { setAutoPlay(!autoPlay); if (!autoPlay) setActiveStep(0); }}
              className="px-4 py-2 rounded-lg glass text-xs font-medium text-text-secondary hover:text-text-base transition-colors"
            >
              {autoPlay ? 'Pause Animation' : 'Animate Flow'}
            </button>
          </div>

          {/* Flow steps */}
          <div className="space-y-2">
            {ARCHITECTURE_STEPS.map((step, idx) => {
              const Icon = iconMap[step.icon] ?? Brain;
              const isActive = activeStep === idx;
              const isPassed = activeStep > idx;
              const color = stepColors[idx];

              return (
                <div key={step.id}>
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    onMouseEnter={() => !autoPlay && setActiveStep(idx)}
                    onMouseLeave={() => !autoPlay && setActiveStep(-1)}
                  >
                    <Card
                      className={`p-4 transition-all duration-300 ${
                        isActive ? 'border-border-strong' : ''
                      }`}
                      glow={isActive}
                    >
                      <div className="flex items-center gap-4">
                        {/* Step number */}
                        <div
                          className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 transition-all duration-300"
                          style={{
                            background: isActive || isPassed ? color + '20' : 'rgba(0,0,0,0.03)',
                            boxShadow: isActive ? `0 0 16px ${color}25` : undefined,
                          }}
                        >
                          <Icon
                            size={18}
                            style={{ color: isActive || isPassed ? color : '#71717a' }}
                          />
                        </div>

                        {/* Content */}
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono text-text-muted">
                              {String(idx + 1).padStart(2, '0')}
                            </span>
                            <h3 className={`text-sm font-semibold transition-colors ${
                              isActive ? 'text-text-base' : 'text-text-secondary'
                            }`}>
                              {step.label}
                            </h3>
                          </div>
                          <motion.p
                            initial={false}
                            animate={{ height: isActive ? 'auto' : 0, opacity: isActive ? 1 : 0 }}
                            className="overflow-hidden text-xs text-text-muted mt-1"
                          >
                            {step.description}
                          </motion.p>
                        </div>

                        {/* Status dot */}
                        <div
                          className="w-2 h-2 rounded-full shrink-0 transition-all"
                          style={{
                            background: isActive ? color : isPassed ? color + '60' : 'rgba(0,0,0,0.04)',
                            boxShadow: isActive ? `0 0 8px ${color}40` : undefined,
                          }}
                        />
                      </div>
                    </Card>
                  </motion.div>

                  {/* Arrow between steps */}
                  {idx < ARCHITECTURE_STEPS.length - 1 && (
                    <div className="flex justify-center py-1">
                      <ArrowDown
                        size={14}
                        className="transition-colors"
                        style={{
                          color: isPassed || isActive ? color + '60' : 'rgba(0,0,0,0.06)',
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Loop indicator */}
          <div className="flex justify-center mt-4">
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg glass text-[10px] text-text-muted">
              <Compass size={12} className="text-accent-pink" />
              <span>Feedback loop: Memory biases future routing decisions</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

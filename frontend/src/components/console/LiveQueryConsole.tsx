import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Square, Zap, Monitor, FlaskConical, ToggleLeft, ToggleRight, ChevronDown } from 'lucide-react';
import { useChat } from '@/hooks/useChat';
import { useAppStore } from '@/store/useAppStore';
import { PRESET_QUERIES } from '@/lib/mockData';
import { SectionHeading, GlowButton, Chip } from '@/components/ui';

type PresetCategory = keyof typeof PRESET_QUERIES;
const categories: { key: PresetCategory; label: string; color: string }[] = [
  { key: 'coding', label: 'Coding', color: 'cyan' },
  { key: 'planning', label: 'Planning', color: 'purple' },
  { key: 'analysis', label: 'Analysis', color: 'blue' },
  { key: 'extraction', label: 'Extraction', color: 'green' },
  { key: 'strategy', label: 'Strategy', color: 'amber' },
  { key: 'shift', label: 'Domain Shift', color: 'pink' },
];

const backendModes = [
  { key: 'mock', label: 'Mock', icon: FlaskConical, desc: 'Deterministic simulation' },
  { key: 'real', label: 'Real API', icon: Monitor, desc: 'OpenAI-compatible' },
];

export function LiveQueryConsole() {
  const [prompt, setPrompt] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<PresetCategory>('coding');
  const [adaptiveMode, setAdaptiveMode] = useState(true);
  const [demoMode, setDemoMode] = useState(true);
  const [backendMode, setBackendMode] = useState('mock');
  const [showBackendDropdown, setShowBackendDropdown] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { sendMessage, abort } = useChat();
  const isStreaming = useAppStore((s) => s.isStreaming);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [prompt]);

  const handleSend = () => {
    if (!prompt.trim() || isStreaming) return;
    sendMessage(prompt.trim(), 'auto');
    setPrompt('');
  };

  const handlePreset = (query: string) => {
    setPrompt(query);
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <section id="console" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Interactive Demo"
          title="Live Query Console"
          subtitle="Run prompts through the adaptive forest in real-time. Compare against a normal single-model baseline."
          center
        />

        <div className="mt-12 max-w-3xl mx-auto space-y-6">
          {/* Mode toggles */}
          <div className="flex flex-wrap items-center justify-center gap-4">
            {/* Adaptive vs Normal toggle */}
            <button
              onClick={() => setAdaptiveMode(!adaptiveMode)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium transition-colors hover:bg-elevated"
            >
              {adaptiveMode ? (
                <ToggleRight size={18} className="text-primary" />
              ) : (
                <ToggleLeft size={18} className="text-text-muted" />
              )}
              <span className={adaptiveMode ? 'text-primary' : 'text-text-muted'}>
                {adaptiveMode ? 'Adaptive Forest' : 'Normal Model'}
              </span>
            </button>

            {/* Demo / Real toggle */}
            <button
              onClick={() => setDemoMode(!demoMode)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium transition-colors hover:bg-elevated"
            >
              {demoMode ? (
                <FlaskConical size={16} className="text-accent-purple" />
              ) : (
                <Monitor size={16} className="text-accent-green" />
              )}
              <span className="text-text-secondary">
                {demoMode ? 'Demo Mode' : 'Real Mode'}
              </span>
            </button>

            {/* Backend selector */}
            <div className="relative">
              <button
                onClick={() => setShowBackendDropdown(!showBackendDropdown)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium transition-colors hover:bg-elevated"
              >
                <Zap size={14} className="text-text-muted" />
                <span className="text-text-secondary">{backendModes.find(b => b.key === backendMode)?.label}</span>
                <ChevronDown size={14} className="text-text-muted" />
              </button>
              <AnimatePresence>
                {showBackendDropdown && (
                  <motion.div
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    className="absolute top-full mt-1 right-0 glass-strong rounded-lg py-1 min-w-[180px] z-30"
                  >
                    {backendModes.map((m) => (
                      <button
                        key={m.key}
                        onClick={() => { setBackendMode(m.key); setShowBackendDropdown(false); }}
                        className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-[rgba(0,0,0,0.03)] transition-colors"
                      >
                        <m.icon size={14} className={backendMode === m.key ? 'text-primary' : 'text-text-muted'} />
                        <div>
                          <div className="text-xs font-medium text-text-base">{m.label}</div>
                          <div className="text-[10px] text-text-muted">{m.desc}</div>
                        </div>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Preset categories */}
          <div className="flex flex-wrap justify-center gap-2">
            {categories.map((cat) => (
              <button
                key={cat.key}
                onClick={() => setSelectedCategory(cat.key)}
                className="transition-all"
              >
                <Chip
                  color={selectedCategory === cat.key ? cat.color : 'gray'}
                  dot={selectedCategory === cat.key}
                >
                  {cat.label}
                </Chip>
              </button>
            ))}
          </div>

          {/* Preset queries */}
          <div className="flex flex-col gap-2">
            {PRESET_QUERIES[selectedCategory].map((query) => (
              <button
                key={query}
                onClick={() => handlePreset(query)}
                className="text-left px-4 py-3 rounded-lg border border-border bg-surface hover:bg-elevated hover:border-border-strong transition-all text-sm text-text-secondary group"
              >
                <span className="group-hover:text-text-base transition-colors">{query}</span>
              </button>
            ))}
          </div>

          {/* Input area */}
          <div className="relative">
            <div className="glass-strong rounded-xl p-1 glow-border">
              <div className="flex items-end gap-2 p-3">
                <textarea
                  ref={textareaRef}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your prompt or select a preset above..."
                  rows={1}
                  className="flex-1 bg-transparent text-text-base placeholder:text-text-placeholder resize-none outline-none text-sm leading-relaxed"
                />
                {isStreaming ? (
                  <GlowButton onClick={abort} variant="secondary" size="sm">
                    <Square size={14} /> Stop
                  </GlowButton>
                ) : (
                  <GlowButton onClick={handleSend} disabled={!prompt.trim()} size="sm">
                    <Send size={14} /> Run
                  </GlowButton>
                )}
              </div>
            </div>
            <div className="flex items-center justify-between mt-2 px-1">
              <span className="text-[10px] text-text-muted">
                {adaptiveMode ? 'Adaptive routing · Multi-branch execution · Live optimization' : 'Single model · No routing · No optimization'}
              </span>
              <span className="text-[10px] text-text-muted">Shift+Enter for newline</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

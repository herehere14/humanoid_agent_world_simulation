import { useState, useEffect } from 'react';
import { useAppStore } from '@/store/useAppStore';
import { StatusDot } from '@/components/ui';
import { MOCK_ENGINE_STATE } from '@/lib/mockData';
import { Github, Star, Menu, X } from 'lucide-react';

const navLinks = [
  { href: '#human-mode', label: 'Human Mode' },
  { href: '#avatar-playground', label: 'Agent' },
  { href: '#sphere', label: 'Sphere' },
  { href: '#console', label: 'Demo' },
  { href: '#trace', label: 'Trace' },
  { href: '#comparison', label: 'Compare' },
  { href: '#architecture', label: 'Architecture' },
  { href: '#benchmarks', label: 'Benchmarks' },
];

export function Header() {
  const { isStreaming, engineState } = useAppStore();
  const state = engineState ?? MOCK_ENGINE_STATE;
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'glass-strong shadow-card' : 'bg-transparent'
      }`}
    >
      <div className="section-container h-16 flex items-center justify-between">
        {/* Logo */}
        <a href="#hero" className="flex items-center gap-2.5 group">
          <svg viewBox="0 0 28 28" fill="none" className="w-7 h-7">
            <circle cx="14" cy="14" r="13" stroke="rgba(0,102,255,0.2)" strokeWidth="1.5" />
            <circle cx="14" cy="14" r="4" fill="rgba(0,102,255,0.1)" />
            <circle cx="14" cy="14" r="2.5" fill="#0066ff" />
            {[
              [14, 2], [24, 8], [24, 20], [14, 26], [4, 20], [4, 8]
            ].map(([x, y], i) => (
              <g key={i}>
                <line x1="14" y1="14" x2={x} y2={y} stroke="#0066ff" strokeWidth="1" strokeOpacity="0.2" />
                <circle cx={x} cy={y} r="2" fill="#0066ff" fillOpacity="0.35" />
              </g>
            ))}
          </svg>
          <span className="text-base font-semibold text-text-base tracking-tight group-hover:text-primary transition-colors">
            OpenClaw
          </span>
        </a>

        {/* Nav links - desktop */}
        <nav className="hidden lg:flex items-center gap-1">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="px-3 py-1.5 rounded-lg text-xs font-medium text-text-muted hover:text-text-base hover:bg-surface transition-all"
            >
              {link.label}
            </a>
          ))}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-xs text-text-muted">
            <StatusDot active={!isStreaming} pulse={isStreaming} color={isStreaming ? '#0066ff' : '#059669'} />
            <span>{isStreaming ? 'Processing' : `${state.branches.filter(b => b.status === 'ACTIVE').length} branches`}</span>
          </div>

          <a
            href="https://github.com/openclaw/adaptive-prompt-forest"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border text-xs font-medium text-text-secondary hover:text-text-base hover:bg-surface hover:shadow-card transition-all"
          >
            <Github size={14} />
            <Star size={12} />
            <span className="hidden sm:inline">Star</span>
          </a>

          <button
            onClick={() => setMobileMenu(!mobileMenu)}
            className="lg:hidden p-2 rounded-lg hover:bg-surface"
          >
            {mobileMenu ? <X size={18} className="text-text-base" /> : <Menu size={18} className="text-text-muted" />}
          </button>
        </div>
      </div>

      {mobileMenu && (
        <div className="lg:hidden glass-strong border-t border-border">
          <nav className="section-container py-4 flex flex-col gap-1">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenu(false)}
                className="px-4 py-2 rounded-lg text-sm text-text-secondary hover:text-text-base hover:bg-surface transition-all"
              >
                {link.label}
              </a>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}

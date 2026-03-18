import { Github, Twitter, BookOpen, ExternalLink } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-border bg-surface/50 py-12">
      <div className="section-container">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <svg viewBox="0 0 28 28" fill="none" className="w-6 h-6">
              <circle cx="14" cy="14" r="13" stroke="rgba(0,102,255,0.2)" strokeWidth="1.5" />
              <circle cx="14" cy="14" r="2.5" fill="#0066ff" />
              {[[14,2],[24,8],[24,20],[14,26],[4,20],[4,8]].map(([x,y],i) => (
                <g key={i}>
                  <line x1="14" y1="14" x2={x} y2={y} stroke="#0066ff" strokeWidth="1" strokeOpacity="0.15" />
                  <circle cx={x} cy={y} r="1.5" fill="#0066ff" fillOpacity="0.3" />
                </g>
              ))}
            </svg>
            <div>
              <span className="text-sm font-semibold text-text-base">OpenClaw Adaptive Prompt Forest</span>
              <p className="text-[10px] text-text-muted">A self-improving control layer with human-like cognition</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <FooterLink href="https://github.com/openclaw/adaptive-prompt-forest" icon={<Github size={14} />} label="GitHub" />
            <FooterLink href="#" icon={<BookOpen size={14} />} label="Docs" />
            <FooterLink href="#" icon={<Twitter size={14} />} label="Twitter" />
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="text-[11px] text-text-muted">
            Built with reinforcement learning principles and cognitive-behavioral modeling.
          </p>
          <p className="text-[11px] text-text-muted">
            MIT License · Made for the open-source community
          </p>
        </div>
      </div>
    </footer>
  );
}

function FooterLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-base transition-colors"
    >
      {icon}
      {label}
      <ExternalLink size={10} className="opacity-50" />
    </a>
  );
}

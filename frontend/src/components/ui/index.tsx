import { clsx } from 'clsx';
import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

// ─── Chip ──────────────────────────────────────────────────────────────────
const chipColors: Record<string, string> = {
  cyan: 'bg-primary-light text-primary border-primary/15',
  green: 'bg-success-light text-success-DEFAULT border-success-DEFAULT/15',
  amber: 'bg-warn-light text-warn-DEFAULT border-warn-DEFAULT/15',
  red: 'bg-danger-light text-danger-DEFAULT border-danger-DEFAULT/15',
  purple: 'bg-[rgba(124,58,237,0.08)] text-accent-purple border-accent-purple/15',
  pink: 'bg-[rgba(236,72,153,0.08)] text-accent-pink border-accent-pink/15',
  blue: 'bg-[rgba(37,99,235,0.08)] text-accent-blue border-accent-blue/15',
  gray: 'bg-[rgba(0,0,0,0.03)] text-text-secondary border-border',
  white: 'bg-white text-text-base border-border shadow-card',
};

export function Chip({
  children, color = 'gray', dot, className,
}: { children: ReactNode; color?: string; dot?: boolean; className?: string }) {
  return (
    <span className={clsx(
      'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium border',
      chipColors[color] ?? chipColors.gray,
      className,
    )}>
      {dot && <span className={clsx('w-1.5 h-1.5 rounded-full', {
        'bg-primary': color === 'cyan',
        'bg-success-DEFAULT': color === 'green',
        'bg-warn-DEFAULT': color === 'amber',
        'bg-danger-DEFAULT': color === 'red',
        'bg-accent-purple': color === 'purple',
        'bg-accent-pink': color === 'pink',
        'bg-accent-blue': color === 'blue',
        'bg-text-muted': color === 'gray' || color === 'white',
      })} />}
      {children}
    </span>
  );
}

// ─── Bar ───────────────────────────────────────────────────────────────────
export function Bar({
  value, max = 1, color = '#0066ff', height = 4, className,
}: { value: number; max?: number; color?: string; height?: number; className?: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className={clsx('w-full rounded-full overflow-hidden', className)}
      style={{ height, background: 'rgba(0,0,0,0.06)' }}>
      <motion.div
        className="h-full rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      />
    </div>
  );
}

// ─── SectionTitle ──────────────────────────────────────────────────────────
export function SectionTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <h4 className={clsx('text-[10px] font-semibold uppercase tracking-[0.12em] text-text-muted', className)}>
      {children}
    </h4>
  );
}

// ─── Card ──────────────────────────────────────────────────────────────────
export function Card({
  children, className, onClick, glow, hover,
}: { children: ReactNode; className?: string; onClick?: () => void; glow?: boolean; hover?: boolean }) {
  return (
    <div
      className={clsx(
        'rounded-xl border border-border bg-white',
        glow && 'shadow-card-md',
        hover && 'hover:border-border-strong hover:shadow-card-md transition-all duration-200 cursor-pointer',
        onClick && 'cursor-pointer',
        className,
      )}
      onClick={onClick}
    >
      {children}
    </div>
  );
}

// ─── GlassCard ─────────────────────────────────────────────────────────────
export function GlassCard({
  children, className, glow,
}: { children: ReactNode; className?: string; glow?: boolean }) {
  return (
    <div className={clsx(
      'glass rounded-xl',
      glow && 'shadow-card-md',
      className,
    )}>
      {children}
    </div>
  );
}

// ─── Divider ───────────────────────────────────────────────────────────────
export function Divider({ className }: { className?: string }) {
  return <div className={clsx('h-px bg-border', className)} />;
}

// ─── Spinner ───────────────────────────────────────────────────────────────
export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return (
    <svg className={clsx('animate-spin', className)} width={size} height={size} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeDasharray="60" strokeDashoffset="20" opacity="0.2" />
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeDasharray="40" strokeDashoffset="60" />
    </svg>
  );
}

// ─── StatusDot ─────────────────────────────────────────────────────────────
export function StatusDot({
  active = false, color, pulse, size = 6,
}: { active?: boolean; color?: string; pulse?: boolean; size?: number }) {
  return (
    <span className="relative inline-flex">
      {pulse && active && (
        <span
          className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-40"
          style={{ width: size, height: size, background: color ?? '#059669' }}
        />
      )}
      <span
        className="rounded-full"
        style={{ width: size, height: size, background: color ?? (active ? '#059669' : '#94a3b8') }}
      />
    </span>
  );
}

// ─── EmptyState ────────────────────────────────────────────────────────────
export function EmptyState({
  icon, label, description,
}: { icon?: ReactNode; label: string; description?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center gap-2">
      {icon && <div className="text-text-muted">{icon}</div>}
      <p className="text-sm text-text-secondary font-medium">{label}</p>
      {description && <p className="text-xs text-text-muted max-w-[240px]">{description}</p>}
    </div>
  );
}

// ─── MetricBox ─────────────────────────────────────────────────────────────
export function MetricBox({
  label, value, suffix, color, small,
}: { label: string; value: string | number; suffix?: string; color?: string; small?: boolean }) {
  return (
    <div className={clsx('flex flex-col', small ? 'gap-0' : 'gap-0.5')}>
      <span className="text-[10px] uppercase tracking-wider text-text-muted">{label}</span>
      <span className={clsx('font-mono font-semibold', small ? 'text-sm' : 'text-lg')} style={{ color: color ?? '#1a1a2e' }}>
        {value}{suffix && <span className="text-text-muted text-xs ml-0.5">{suffix}</span>}
      </span>
    </div>
  );
}

// ─── GlowButton ────────────────────────────────────────────────────────────
export function GlowButton({
  children, onClick, variant = 'primary', size = 'md', className, disabled,
}: { children: ReactNode; onClick?: () => void; variant?: 'primary' | 'secondary' | 'ghost'; size?: 'sm' | 'md' | 'lg'; className?: string; disabled?: boolean }) {
  const variants = {
    primary: 'bg-primary text-white hover:bg-primary-dark shadow-glow-sm hover:shadow-glow-md',
    secondary: 'bg-white text-text-base border border-border hover:bg-surface hover:border-border-strong shadow-card',
    ghost: 'text-text-secondary hover:text-text-base hover:bg-surface',
  };
  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-5 py-2.5 text-sm',
    lg: 'px-7 py-3 text-base',
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'rounded-lg font-medium transition-all duration-200 inline-flex items-center gap-2',
        variants[variant],
        sizes[size],
        disabled && 'opacity-50 cursor-not-allowed',
        className,
      )}
    >
      {children}
    </button>
  );
}

// ─── SectionHeading ────────────────────────────────────────────────────────
export function SectionHeading({
  badge, title, subtitle, className, center,
}: { badge?: string; title: string; subtitle?: string; className?: string; center?: boolean }) {
  return (
    <div className={clsx('flex flex-col gap-3', center && 'items-center text-center', className)}>
      {badge && (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium bg-primary-light text-primary border border-primary/15 w-fit">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          {badge}
        </span>
      )}
      <h2 className="text-2xl sm:text-3xl font-bold text-text-base">{title}</h2>
      {subtitle && <p className="text-sm sm:text-base text-text-secondary max-w-2xl">{subtitle}</p>}
    </div>
  );
}

// ─── AnimatedCounter ───────────────────────────────────────────────────────
export function AnimatedCounter({ value, decimals = 0 }: { value: number; decimals?: number }) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {value.toFixed(decimals)}
    </motion.span>
  );
}

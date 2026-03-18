/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        base: '#ffffff',
        surface: '#f8f9fb',
        elevated: '#f1f3f6',
        panel: '#e8ebf0',
        border: {
          DEFAULT: 'rgba(0,0,0,0.08)',
          subtle: 'rgba(0,0,0,0.04)',
          strong: 'rgba(0,0,0,0.14)',
        },
        primary: {
          DEFAULT: '#0066ff',
          light: 'rgba(0,102,255,0.08)',
          mid: 'rgba(0,102,255,0.15)',
          dark: '#0052cc',
        },
        accent: {
          purple: '#7c3aed',
          pink: '#ec4899',
          green: '#059669',
          amber: '#d97706',
          red: '#dc2626',
          blue: '#2563eb',
          indigo: '#4f46e5',
          teal: '#0d9488',
        },
        text: {
          base: '#1a1a2e',
          secondary: '#64748b',
          muted: '#94a3b8',
          placeholder: '#cbd5e1',
        },
        success: {
          DEFAULT: '#059669',
          light: 'rgba(5,150,105,0.08)',
          mid: 'rgba(5,150,105,0.15)',
        },
        warn: {
          DEFAULT: '#d97706',
          light: 'rgba(217,119,6,0.08)',
        },
        danger: {
          DEFAULT: '#dc2626',
          light: 'rgba(220,38,38,0.08)',
        },
        branch: {
          analytical: '#0066ff',
          planner: '#7c3aed',
          retrieval: '#059669',
          critique: '#d97706',
          verification: '#2563eb',
          creative: '#ec4899',
          chain: '#4f46e5',
          code: '#0891b2',
          meta: '#ea580c',
        },
        // Human mode specific colors
        human: {
          curiosity: '#0066ff',
          fear: '#dc2626',
          ambition: '#d97706',
          empathy: '#059669',
          reflection: '#7c3aed',
          impulse: '#ea580c',
          stress: '#e11d48',
          confidence: '#0891b2',
          trust: '#059669',
          conflict: '#9333ea',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'card-md': '0 4px 12px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.04)',
        'card-lg': '0 8px 24px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.04)',
        'card-xl': '0 20px 50px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04)',
        'input': '0 0 0 3px rgba(0,102,255,0.12)',
        'glow-sm': '0 0 12px rgba(0,102,255,0.1)',
        'glow-md': '0 0 24px rgba(0,102,255,0.15)',
        'glow-lg': '0 0 48px rgba(0,102,255,0.12)',
        'glow-purple': '0 0 24px rgba(124,58,237,0.15)',
        'glow-green': '0 0 24px rgba(5,150,105,0.15)',
        'inner-soft': 'inset 0 2px 4px rgba(0,0,0,0.04)',
      },
      borderRadius: {
        '2xl': '16px',
        '3xl': '20px',
        '4xl': '24px',
      },
      animation: {
        'pulse-soft': 'pulseSoft 2.5s ease-in-out infinite',
        'pulse-glow': 'pulseGlow 3s ease-in-out infinite',
        'fade-in': 'fadeIn 0.4s ease-out',
        'fade-in-up': 'fadeInUp 0.6s cubic-bezier(0.16,1,0.3,1)',
        'slide-up': 'slideUp 0.3s cubic-bezier(0.16,1,0.3,1)',
        'slide-in-right': 'slideInRight 0.4s cubic-bezier(0.16,1,0.3,1)',
        'float': 'float 6s ease-in-out infinite',
        'spin-slow': 'spin 20s linear infinite',
        'gradient-x': 'gradientX 8s ease infinite',
        'trace-line': 'traceLine 1s ease-out forwards',
        'shimmer': 'shimmer 2s linear infinite',
        'orbit': 'orbit 12s linear infinite',
      },
      keyframes: {
        pulseSoft: {
          '0%,100%': { opacity: '0.7' },
          '50%': { opacity: '1' },
        },
        pulseGlow: {
          '0%,100%': { boxShadow: '0 0 20px rgba(0,102,255,0.05)' },
          '50%': { boxShadow: '0 0 40px rgba(0,102,255,0.12)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(8px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideInRight: {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        float: {
          '0%,100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-12px)' },
        },
        gradientX: {
          '0%,100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        traceLine: {
          '0%': { width: '0%', opacity: '0' },
          '100%': { width: '100%', opacity: '1' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg) translateX(60px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(60px) rotate(-360deg)' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'grid-pattern': 'linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)',
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
    },
  },
  plugins: [],
}

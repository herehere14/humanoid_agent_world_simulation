import React, { useEffect, useCallback, lazy, Suspense } from 'react';
import { Header } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';
import { HeroSection } from '@/components/hero/HeroSection';
import { MetricsStrip } from '@/components/metrics/MetricsStrip';
import { AdaptiveSphere } from '@/components/sphere/AdaptiveSphere';
import { LiveQueryConsole } from '@/components/console/LiveQueryConsole';
import { RoutingTracePanel } from '@/components/trace/RoutingTracePanel';
import { ComparisonPanel } from '@/components/comparison/ComparisonPanel';
import { MemoryPanel } from '@/components/memory/MemoryPanel';
import { RewardPanel } from '@/components/rewards/RewardPanel';
import { DomainShiftChallenge } from '@/components/shift/DomainShiftChallenge';
import { ArchitectureFlow } from '@/components/architecture/ArchitectureFlow';
import { BenchmarkSection } from '@/components/benchmarks/BenchmarkSection';
import { ReplayMode } from '@/components/replay/ReplayMode';
import { useAppStore } from '@/store/useAppStore';
import { fetchEngineState } from '@/lib/api';
import { MOCK_ENGINE_STATE } from '@/lib/mockData';

// Lazy-load heavy 3D components to prevent crash from blocking the whole app
const HumanModeSection = lazy(() => import('@/components/human-mode/HumanModeSection').then(m => ({ default: m.HumanModeSection })));
const AvatarPlayground = lazy(() => import('@/components/human-mode/AvatarPlayground').then(m => ({ default: m.AvatarPlayground })));

function SectionFallback({ label }: { label: string }) {
  return (
    <div className="py-24 text-center text-text-muted">
      <div className="text-sm">Loading {label}...</div>
    </div>
  );
}

// Top-level error boundary to prevent white screen of death
class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  state = { hasError: false, error: '' };
  static getDerivedStateFromError(err: Error) {
    return { hasError: true, error: err.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, fontFamily: 'monospace', color: '#1a1a2e' }}>
          <h1 style={{ color: '#dc2626' }}>Something went wrong</h1>
          <pre style={{ background: '#f8f9fb', padding: 16, borderRadius: 8, overflow: 'auto' }}>
            {this.state.error}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{ marginTop: 16, padding: '8px 16px', background: '#0066ff', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer' }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const { setEngineState } = useAppStore();

  useEffect(() => {
    fetchEngineState()
      .then(setEngineState)
      .catch(() => setEngineState(MOCK_ENGINE_STATE));
  }, [setEngineState]);

  const scrollTo = useCallback((id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  return (
    <AppErrorBoundary>
      <div className="min-h-screen bg-base text-text-base overflow-x-hidden">
        <Header />

        <main>
          {/* 1. Hero / Landing */}
          <HeroSection onScrollTo={scrollTo} />

          {/* 2. Metrics Strip */}
          <MetricsStrip />

          {/* NEW: Human Mode Section */}
          <Suspense fallback={<SectionFallback label="Human Mode" />}>
            <HumanModeSection />
          </Suspense>

          {/* Avatar Playground - Interactive Emotion Demo */}
          <div className="section-alt">
            <Suspense fallback={<SectionFallback label="3D Avatar" />}>
              <AvatarPlayground />
            </Suspense>
          </div>

          {/* 3. 3D Adaptive Sphere */}
          <div className="section-alt">
            <AdaptiveSphere />
          </div>

          {/* 4. Live Query Console */}
          <LiveQueryConsole />

          {/* 5. Real-time Routing Trace */}
          <div className="section-alt">
            <RoutingTracePanel />
          </div>

          {/* 6. Base vs Adaptive Comparison */}
          <ComparisonPanel />

          {/* 7. Memory Panel */}
          <div className="section-alt">
            <MemoryPanel />
          </div>

          {/* 8. Reward / Optimization Panel */}
          <RewardPanel />

          {/* 9. Domain Shift Challenge */}
          <div className="section-alt">
            <DomainShiftChallenge />
          </div>

          {/* 10. Architecture Flow */}
          <ArchitectureFlow />

          {/* 11. Benchmark Section */}
          <div className="section-alt">
            <BenchmarkSection />
          </div>

          {/* 12. Replay Mode */}
          <ReplayMode />
        </main>

        <Footer />
      </div>
    </AppErrorBoundary>
  );
}

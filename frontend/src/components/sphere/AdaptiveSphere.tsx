import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { MOCK_BRANCHES, getBranchColor, formatBranchName } from '@/lib/mockData';
import type { BranchState } from '@/types';
import { SectionHeading } from '@/components/ui';

interface Node3D {
  branch: BranchState;
  x: number; y: number; z: number;
  screenX: number; screenY: number;
  color: string;
  radius: number;
}

interface HoveredNode {
  branch: BranchState;
  x: number;
  y: number;
}

export function AdaptiveSphere() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const nodesRef = useRef<Node3D[]>([]);
  const rotRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef({ dragging: false, lastX: 0, lastY: 0 });
  const zoomRef = useRef(1);
  const [hoveredNode, setHoveredNode] = useState<HoveredNode | null>(null);
  const [pinnedNode, setPinnedNode] = useState<BranchState | null>(null);
  const [replayActive, setReplayActive] = useState(false);
  const [replayStep, setReplayStep] = useState(0);
  const engineState = useAppStore((s) => s.engineState);
  const activeTrace = useAppStore((s) => s.activeTrace);

  const branches = engineState?.branches ?? MOCK_BRANCHES;

  const initNodes = useCallback(() => {
    const n = branches.length;
    nodesRef.current = branches.map((branch, i) => {
      const phi = Math.acos(1 - 2 * (i + 0.5) / n);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const r = 140;
      return {
        branch,
        x: r * Math.sin(phi) * Math.cos(theta),
        y: r * Math.sin(phi) * Math.sin(theta),
        z: r * Math.cos(phi),
        screenX: 0,
        screenY: 0,
        color: getBranchColor(branch.name),
        radius: 4 + branch.weight * 3,
      };
    });
  }, [branches]);

  useEffect(() => { initNodes(); }, [initNodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const W = 600;
    const H = 600;
    canvas.width = W * 2;
    canvas.height = H * 2;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(2, 2);

    const cx = W / 2;
    const cy = H / 2;
    let time = 0;

    const activeBranches = activeTrace.activeBranches;
    const selectedBranch = activeTrace.evaluation?.selected_branch;

    const animate = () => {
      time += 0.003;
      ctx.clearRect(0, 0, W, H);

      if (!dragRef.current.dragging) {
        rotRef.current.y += 0.003;
      }

      const cosX = Math.cos(rotRef.current.x);
      const sinX = Math.sin(rotRef.current.x);
      const cosY = Math.cos(rotRef.current.y);
      const sinY = Math.sin(rotRef.current.y);
      const zoom = zoomRef.current;

      // Project nodes
      for (const node of nodesRef.current) {
        const x1 = node.x * cosY - node.z * sinY;
        const z1 = node.x * sinY + node.z * cosY;
        const y1 = node.y * cosX - z1 * sinX;
        const z2 = node.y * sinX + z1 * cosX;

        const scale = 300 / (300 + z2);
        node.screenX = cx + x1 * scale * zoom;
        node.screenY = cy + y1 * scale * zoom;
      }

      // Sort by depth for proper rendering
      const sorted = [...nodesRef.current].sort((a, b) => {
        const az = a.x * sinY + a.z * cosY;
        const bz = b.x * sinY + b.z * cosY;
        return az - bz;
      });

      // Draw sphere outline
      ctx.beginPath();
      ctx.arc(cx, cy, 140 * zoom, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,0,0,0.06)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Latitude rings
      for (let lat = -60; lat <= 60; lat += 60) {
        const latRad = (lat * Math.PI) / 180;
        const ringR = 140 * Math.cos(latRad) * zoom;
        const ringY = cy + 140 * Math.sin(latRad) * cosX * zoom;
        ctx.beginPath();
        ctx.ellipse(cx, ringY, ringR, ringR * 0.3, 0, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0,0,0,0.04)';
        ctx.stroke();
      }

      // Draw edges from center
      for (const node of sorted) {
        const isActive = activeBranches.includes(node.branch.name);
        const isSelected = selectedBranch === node.branch.name;
        const depth = (node.screenY - cy) / (140 * zoom);
        const depthAlpha = 0.15 + (1 - Math.abs(depth)) * 0.25;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(node.screenX, node.screenY);
        ctx.strokeStyle = isActive
          ? node.color + (isSelected ? '60' : '30')
          : `rgba(0,0,0,${depthAlpha * 0.08})`;
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.stroke();
      }

      // Draw center orb
      const centerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 15);
      const centerPulse = 0.3 + Math.sin(time * 5) * 0.1;
      centerGrad.addColorStop(0, `rgba(0,102,255,${centerPulse})`);
      centerGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = centerGrad;
      ctx.fillRect(cx - 15, cy - 15, 30, 30);
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = activeBranches.length > 0 ? '#0066ff' : 'rgba(0,0,0,0.15)';
      ctx.fill();

      // Draw nodes
      for (const node of sorted) {
        const isActive = activeBranches.includes(node.branch.name);
        const isSelected = selectedBranch === node.branch.name;
        const isRunning = activeTrace.branchOutputs[node.branch.name]?.status === 'running';

        // Replay highlight
        const isReplayHighlight = replayActive && replayStep >= 0 &&
          sorted.indexOf(node) === replayStep % sorted.length;

        // Confidence heat (reward-based color intensity)
        const lastReward = node.branch.historical_rewards[node.branch.historical_rewards.length - 1] ?? 0.5;
        const heatAlpha = 0.3 + lastReward * 0.5;

        // Node glow
        if (isActive || isSelected || isReplayHighlight) {
          const glow = ctx.createRadialGradient(
            node.screenX, node.screenY, 0,
            node.screenX, node.screenY, node.radius * (isSelected ? 6 : 4)
          );
          glow.addColorStop(0, node.color + (isSelected ? '50' : '30'));
          glow.addColorStop(1, 'transparent');
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(node.screenX, node.screenY, node.radius * (isSelected ? 6 : 4), 0, Math.PI * 2);
          ctx.fill();
        }

        // Pulsing ring for running
        if (isRunning) {
          const pulseR = node.radius + 3 + Math.sin(time * 10) * 3;
          ctx.beginPath();
          ctx.arc(node.screenX, node.screenY, pulseR, 0, Math.PI * 2);
          ctx.strokeStyle = node.color + '60';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Node body
        ctx.beginPath();
        ctx.arc(node.screenX, node.screenY, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = isActive || isSelected
          ? node.color
          : `rgba(${hexToRgb(node.color)},${heatAlpha})`;
        ctx.fill();

        // Label for active/selected
        if (isActive || isSelected) {
          ctx.font = '500 10px Inter';
          ctx.fillStyle = 'rgba(26,26,46,0.8)';
          ctx.textAlign = 'center';
          ctx.fillText(
            formatBranchName(node.branch.name),
            node.screenX,
            node.screenY - node.radius - 6
          );
        }
      }

      // Optimizer pulse effect
      if (activeTrace.stage === 'optimizing') {
        const pulseR = 140 * zoom + Math.sin(time * 8) * 10;
        ctx.beginPath();
        ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(124,58,237,${0.1 + Math.sin(time * 8) * 0.05})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Reward feedback arc
      if (activeTrace.evaluation) {
        const reward = activeTrace.evaluation.reward_score;
        const arcColor = reward >= 0.75 ? '#10b981' : reward >= 0.5 ? '#f59e0b' : '#ef4444';
        const arcAngle = reward * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 145 * zoom, -Math.PI / 2, -Math.PI / 2 + arcAngle);
        ctx.strokeStyle = arcColor + '40';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animRef.current);
  }, [branches, activeTrace, replayActive, replayStep]);

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { dragging: true, lastX: e.clientX, lastY: e.clientY };
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (dragRef.current.dragging) {
      const dx = e.clientX - dragRef.current.lastX;
      const dy = e.clientY - dragRef.current.lastY;
      rotRef.current.y += dx * 0.005;
      rotRef.current.x += dy * 0.005;
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastY = e.clientY;
      return;
    }

    // Hit test for hover
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / 2 / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / 2 / rect.height);

    let found: HoveredNode | null = null;
    for (const node of nodesRef.current) {
      const dx = mx - node.screenX;
      const dy = my - node.screenY;
      if (Math.sqrt(dx * dx + dy * dy) < node.radius + 8) {
        found = {
          branch: node.branch,
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        };
        break;
      }
    }
    setHoveredNode(found);
    canvas.style.cursor = found ? 'pointer' : 'grab';
  };
  const handleMouseUp = () => { dragRef.current.dragging = false; };
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    zoomRef.current = Math.max(0.5, Math.min(2, zoomRef.current - e.deltaY * 0.001));
  };
  const handleClick = () => {
    if (hoveredNode) {
      setPinnedNode(pinnedNode?.name === hoveredNode.branch.name ? null : hoveredNode.branch);
    }
  };

  const handleReplay = () => {
    setReplayActive(true);
    setReplayStep(0);
    const interval = setInterval(() => {
      setReplayStep((s) => {
        if (s >= branches.length - 1) {
          clearInterval(interval);
          setTimeout(() => setReplayActive(false), 1000);
          return s;
        }
        return s + 1;
      });
    }, 400);
  };

  const displayedBranch = pinnedNode ?? hoveredNode?.branch;

  return (
    <section id="sphere" className="py-20 relative">
      <div className="section-container">
        <SectionHeading
          badge="Interactive Visualization"
          title="Adaptive Branch Network"
          subtitle="Explore the living topology of specialized prompt branches. Rotate, zoom, hover to inspect, click to pin."
          center
        />

        <div className="mt-12 flex flex-col lg:flex-row items-center gap-8">
          {/* Canvas */}
          <div className="relative flex-1 flex justify-center">
            <div className="relative">
              <canvas
                ref={canvasRef}
                className="w-[300px] h-[300px] sm:w-[500px] sm:h-[500px] lg:w-[600px] lg:h-[600px] rounded-2xl"
                style={{ cursor: 'grab' }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
                onClick={handleClick}
              />

              {/* Hover tooltip */}
              <AnimatePresence>
                {hoveredNode && !pinnedNode && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="absolute glass-strong rounded-lg p-3 pointer-events-none z-20 min-w-[200px]"
                    style={{ left: hoveredNode.x + 15, top: hoveredNode.y - 10 }}
                  >
                    <NodeTooltip branch={hoveredNode.branch} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Replay button */}
            <button
              onClick={handleReplay}
              disabled={replayActive}
              className="absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg glass text-xs font-medium text-primary hover:bg-primary-light transition-colors disabled:opacity-50"
            >
              {replayActive ? `Replaying... (${replayStep + 1}/${branches.length})` : 'Replay Routing'}
            </button>
          </div>

          {/* Pinned details panel */}
          <AnimatePresence>
            {displayedBranch && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="w-full lg:w-[300px] glass rounded-xl p-5 space-y-4"
              >
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-text-base">
                    {formatBranchName(displayedBranch.name)}
                  </h3>
                  <span
                    className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                    style={{
                      background: getBranchColor(displayedBranch.name) + '20',
                      color: getBranchColor(displayedBranch.name),
                    }}
                  >
                    {displayedBranch.status}
                  </span>
                </div>
                <p className="text-xs text-text-secondary">{displayedBranch.purpose}</p>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <DetailItem label="Weight" value={displayedBranch.weight.toFixed(2)} />
                  <DetailItem label="Success Rate" value={`${(((displayedBranch.metadata as Record<string, number>)?.success_rate ?? 0) * 100).toFixed(0)}%`} />
                  <DetailItem label="Avg Reward" value={avgReward(displayedBranch.historical_rewards).toFixed(3)} />
                  <DetailItem label="Memory Bias" value={((displayedBranch.metadata as Record<string, number>)?.memory_bias ?? 0).toFixed(2)} />
                  <DetailItem label="Latency" value={`${(displayedBranch.metadata as Record<string, number>)?.latency_avg_ms ?? 0}ms`} />
                  <DetailItem label="Token Cost" value={`${(displayedBranch.metadata as Record<string, number>)?.token_avg ?? 0}`} />
                </div>
                <div className="pt-2">
                  <span className="text-[10px] text-text-muted uppercase tracking-wider">Reward History</span>
                  <div className="flex items-end gap-1 h-8 mt-1">
                    {displayedBranch.historical_rewards.map((r, i) => (
                      <div
                        key={i}
                        className="flex-1 rounded-sm transition-all"
                        style={{
                          height: `${r * 100}%`,
                          background: r >= 0.75 ? '#10b981' : r >= 0.5 ? '#f59e0b' : '#ef4444',
                          opacity: 0.4 + (i / displayedBranch.historical_rewards.length) * 0.6,
                        }}
                      />
                    ))}
                  </div>
                </div>
                {pinnedNode && (
                  <button
                    onClick={() => setPinnedNode(null)}
                    className="text-[10px] text-text-muted hover:text-text-secondary"
                  >
                    Click to unpin
                  </button>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-text-muted block">{label}</span>
      <span className="text-text-base font-mono font-medium">{value}</span>
    </div>
  );
}

function NodeTooltip({ branch }: { branch: BranchState }) {
  return (
    <div className="space-y-1.5">
      <div className="font-semibold text-xs text-text-base">{formatBranchName(branch.name)}</div>
      <div className="text-[10px] text-text-secondary">{branch.purpose}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
        <span className="text-text-muted">Weight</span>
        <span className="text-text-base font-mono">{branch.weight.toFixed(2)}</span>
        <span className="text-text-muted">Reward</span>
        <span className="text-text-base font-mono">{avgReward(branch.historical_rewards).toFixed(3)}</span>
        <span className="text-text-muted">Status</span>
        <span className="text-text-base">{branch.status}</span>
      </div>
    </div>
  );
}

function avgReward(rewards: number[]): number {
  if (!rewards.length) return 0;
  return rewards.reduce((a, b) => a + b, 0) / rewards.length;
}

function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

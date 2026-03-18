import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { MOCK_ENGINE_STATE, getBranchColor } from '@/lib/mockData';

// ─── Fibonacci sphere point distribution ─────────────────────────────────────
function fibonacciSphere(n: number): [number, number, number][] {
  const pts: [number, number, number][] = [];
  const phi = Math.PI * (3 - Math.sqrt(5)); // golden angle
  for (let i = 0; i < n; i++) {
    const y   = 1 - (i / (n - 1)) * 2;
    const r   = Math.sqrt(1 - y * y);
    const th  = phi * i;
    pts.push([Math.cos(th) * r, y, Math.sin(th) * r]);
  }
  return pts;
}

// Orthographic project + slow auto-rotation via elapsed time
function project(
  x: number, y: number, z: number,
  rotY: number, rotX: number,
  cx: number, cy: number, radius: number
): { sx: number; sy: number; depth: number } {
  // Rotate around Y axis
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const x1  = x * cosY + z * sinY;
  const z1  = -x * sinY + z * cosY;
  // Rotate around X axis
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const y1  = y * cosX - z1 * sinX;
  const depth = z1 * cosX + y * sinX;
  return {
    sx: cx + x1 * radius,
    sy: cy + y1 * radius,
    depth,
  };
}

const BRANCH_NAMES = [
  'analytical','planner','retrieval','critique',
  'verification','creative','chain_of_thought','code_solver',
  'deep_math','meta_verifier','logical_chain','step_decompose',
];

export function BranchSphere() {
  const { activeTrace, engineState } = useAppStore();
  const state    = engineState ?? MOCK_ENGINE_STATE;
  const { activeBranches, selectedPath, branchOutputs, routing } = activeTrace;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef  = useRef<number>(0);
  const startRef  = useRef<number>(Date.now());

  // Use engine branches or fallback list
  const branches = state.branches.length > 0
    ? state.branches.map(b => b.name)
    : BRANCH_NAMES;

  const weights  = state.branch_weights;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const RADIUS = Math.min(W, H) * 0.36;

    const pts = fibonacciSphere(branches.length);

    function draw() {
      const now   = Date.now();
      const t     = (now - startRef.current) / 1000;
      const rotY  = t * 0.22;
      const rotX  = Math.sin(t * 0.08) * 0.25;

      ctx.clearRect(0, 0, W, H);

      // Project all points
      const projected = pts.map(([x, y, z], i) => {
        const name   = branches[i];
        const isActive = activeBranches.includes(name);
        const isSelected = selectedPath.includes(name);
        const weight = weights[name] ?? 1.0;
        const reward = branchOutputs[name]?.reward ?? routing?.branch_scores?.[name];
        const p = project(x, y, z, rotY, rotX, cx, cy, RADIUS);
        return { ...p, name, isActive, isSelected, weight, reward };
      }).sort((a, b) => a.depth - b.depth); // back-to-front

      // Draw edges from center to each node (back nodes only, faint)
      for (const p of projected) {
        if (p.depth < -0.3 || !p.isActive) continue;
        const alpha = Math.max(0, (p.depth + 1) / 2) * 0.12;
        if (!p.isActive) continue;
        const color = getBranchColor(p.name);
        const [r, g, b] = hexToRgb(color);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(p.sx, p.sy);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha + (p.isSelected ? 0.25 : 0)})`;
        ctx.lineWidth = p.isSelected ? 1.5 : 0.75;
        ctx.stroke();
      }

      // Draw sphere outline circle (subtle)
      ctx.beginPath();
      ctx.arc(cx, cy, RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(229,231,235,0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Latitude rings (3)
      for (const lat of [-0.5, 0, 0.5]) {
        const r2 = Math.sqrt(1 - lat * lat) * RADIUS;
        const pts2: [number,number][] = [];
        for (let a = 0; a <= 64; a++) {
          const th2 = (a / 64) * Math.PI * 2 + rotY;
          const x3  = Math.cos(th2) * r2;
          const z3  = Math.sin(th2) * r2;
          const y3  = lat * RADIUS;
          const cosX2 = Math.cos(rotX), sinX2 = Math.sin(rotX);
          const yy = y3 * cosX2 - z3 * sinX2;
          const zz = z3 * cosX2 + y3 * sinX2;
          if (a === 0) pts2.push([cx + x3, cy + yy]);
          else if (zz > -0.1) pts2.push([cx + x3, cy + yy]);
        }
        if (pts2.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo(pts2[0][0], pts2[0][1]);
        for (let k = 1; k < pts2.length; k++) ctx.lineTo(pts2[k][0], pts2[k][1]);
        ctx.strokeStyle = 'rgba(229,231,235,0.3)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }

      // Draw nodes
      for (const p of projected) {
        const depthFactor = Math.max(0.3, (p.depth + 1) / 2);
        const baseR = 4 + (p.weight / 3) * 4;   // 4–8px
        const r3    = baseR * depthFactor;
        const color = getBranchColor(p.name);
        const [ri, gi, bi] = hexToRgb(color);

        if (p.isSelected) {
          // Outer glow ring
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, r3 + 6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${ri},${gi},${bi},0.08)`;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, r3 + 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${ri},${gi},${bi},0.14)`;
          ctx.fill();
        }

        if (p.isActive) {
          // Filled with color
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, r3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${ri},${gi},${bi},${0.55 * depthFactor + (p.isSelected ? 0.35 : 0)})`;
          ctx.fill();
          // White highlight
          ctx.beginPath();
          ctx.arc(p.sx - r3 * 0.25, p.sy - r3 * 0.25, r3 * 0.3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,255,255,${0.3 * depthFactor})`;
          ctx.fill();
          if (p.isSelected) {
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, r3 + 1.5, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(${ri},${gi},${bi},0.8)`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        } else {
          // Inactive: faint outline
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, r3 * 0.7, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(229,231,235,${0.5 * depthFactor})`;
          ctx.fill();
        }
      }

      // Center orb
      const grad = ctx.createRadialGradient(cx - 5, cy - 5, 0, cx, cy, 16);
      grad.addColorStop(0, 'rgba(37,99,235,0.25)');
      grad.addColorStop(1, 'rgba(37,99,235,0)');
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = activeBranches.length > 0 ? '#0066ff' : '#D1D5DB';
      ctx.fill();

      frameRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, [branches, weights, activeBranches, selectedPath, branchOutputs, routing]);

  return (
    <div className="relative flex items-center justify-center">
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        className="block"
        style={{ width: 280, height: 280 }}
      />
      {/* Active label overlay */}
      <AnimatePresence>
        {activeBranches.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-3 left-0 right-0 flex justify-center"
          >
            <span className="text-[10px] text-text-muted bg-surface/80 px-2 py-0.5 rounded-full border border-border">
              {activeBranches.length} branch{activeBranches.length !== 1 ? 'es' : ''} active
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  const num   = parseInt(clean, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

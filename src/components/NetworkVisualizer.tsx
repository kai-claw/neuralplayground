import { useRef, useEffect, useState, useCallback } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';

interface NetworkVisualizerProps {
  layers: LayerState[] | null;
  inputSize: number;
  width?: number;
  height?: number;
  /** Increment to trigger a signal flow animation */
  signalFlowTrigger?: number;
}

function getColor(value: number, alpha = 1): string {
  if (value > 0) {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(99, 222, 255, ${intensity * alpha})`;
  } else {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(255, 99, 132, ${intensity * alpha})`;
  }
}

function getWeightColor(value: number): string {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped > 0) {
    return `rgba(99, 222, 255, ${Math.abs(clamped) * 0.6})`;
  } else {
    return `rgba(255, 99, 132, ${Math.abs(clamped) * 0.6})`;
  }
}

/* ── Signal flow particle types ── */
interface Particle {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number; // progress per second
  r: number;
  g: number;
  b: number;
  alpha: number;
  size: number;
  layerIdx: number;
  delay: number; // seconds before this particle starts
  alive: boolean;
}

interface NodePos {
  x: number;
  y: number;
}

/** Compute node positions (shared between static render + particle gen) */
function computeNodePositions(
  layerSizes: number[],
  width: number,
  height: number,
  padding: number,
): NodePos[][] {
  const numLayers = layerSizes.length;
  const layerSpacing = (width - padding * 2) / (numLayers - 1);
  const positions: NodePos[][] = [];

  for (let l = 0; l < numLayers; l++) {
    const nodes: NodePos[] = [];
    const numNodes = Math.min(layerSizes[l], 16);
    const maxH = height - padding * 2;
    const nodeSpacing = Math.min(maxH / (numNodes + 1), 25);
    const startY = height / 2 - (numNodes - 1) * nodeSpacing / 2;

    for (let n = 0; n < numNodes; n++) {
      nodes.push({
        x: padding + l * layerSpacing,
        y: startY + n * nodeSpacing,
      });
    }
    // "+N" overflow node
    if (layerSizes[l] > 16) {
      nodes.push({
        x: padding + l * layerSpacing,
        y: startY + numNodes * nodeSpacing,
      });
    }
    positions.push(nodes);
  }
  return positions;
}

/** Generate signal flow particles for all connections */
function generateParticles(
  nodePositions: NodePos[][],
  layerSizes: number[],
  layers: LayerState[],
): Particle[] {
  const particles: Particle[] = [];
  const numLayers = nodePositions.length;
  const LAYER_DELAY = 0.35; // seconds between each layer's particles starting

  for (let l = 1; l < numLayers; l++) {
    const prevNodes = nodePositions[l - 1];
    const currNodes = nodePositions[l];
    const layer = layers[l - 1];
    const maxPrev = Math.min(prevNodes.length, layerSizes[l - 1] > 16 ? 15 : prevNodes.length);
    const maxCurr = Math.min(currNodes.length, layerSizes[l] > 16 ? 15 : currNodes.length);

    // Only sample a subset of connections for visual clarity
    const sampleRate = maxPrev * maxCurr > 100 ? 0.3 : maxPrev * maxCurr > 40 ? 0.5 : 1.0;

    for (let j = 0; j < maxCurr; j++) {
      for (let i = 0; i < maxPrev; i++) {
        if (Math.random() > sampleRate) continue;
        if (j >= layer.weights.length || i >= (layer.weights[j]?.length || 0)) continue;

        const weight = layer.weights[j][i];
        const absW = Math.abs(weight);
        if (absW < 0.05) continue; // skip near-zero connections

        const activation = layer.activations[j] || 0;
        const isPositive = activation >= 0;

        particles.push({
          fromX: prevNodes[i].x,
          fromY: prevNodes[i].y,
          toX: currNodes[j].x,
          toY: currNodes[j].y,
          progress: 0,
          speed: 1.8 + Math.random() * 0.8, // 1.8-2.6x per second
          r: isPositive ? 99 : 255,
          g: isPositive ? 222 : 99,
          b: isPositive ? 255 : 132,
          alpha: 0.4 + absW * 0.6,
          size: 1.5 + absW * 2.5,
          layerIdx: l - 1,
          delay: (l - 1) * LAYER_DELAY + Math.random() * 0.1,
          alive: true,
        });
      }
    }
  }
  return particles;
}

// Scratch arrays to avoid per-frame allocation in glow rendering
const GLOW_STOPS = [0.3, 0.0] as const;

export function NetworkVisualizer({
  layers,
  inputSize,
  width: propWidth,
  height: propHeight,
  signalFlowTrigger = 0,
}: NetworkVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: propWidth || 620, height: propHeight || 420 });

  // Signal flow state (refs to avoid re-renders during animation)
  const particlesRef = useRef<Particle[]>([]);
  const animStartRef = useRef(0);
  const rafRef = useRef(0);
  const lastTriggerRef = useRef(0);
  // Store node glow state for arrival effects
  const nodeGlowRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    if (propWidth && propHeight) {
      setDims({ width: propWidth, height: propHeight });
      return;
    }
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      if (w > 0) setDims({ width: w, height: Math.round(w * 0.68) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [propWidth, propHeight]);

  const { width, height } = dims;

  // Draw the static network (connections + nodes)
  const drawStatic = useCallback((ctx: CanvasRenderingContext2D, nodePositions: NodePos[][], layerSizes: number[]) => {
    if (!layers || layers.length === 0) return;
    const numLayers = layerSizes.length;

    // Draw connections
    for (let l = 1; l < numLayers; l++) {
      const prevNodes = nodePositions[l - 1];
      const currNodes = nodePositions[l];
      const layer = layers[l - 1];
      const maxPrev = Math.min(prevNodes.length, layerSizes[l - 1] > 16 ? 15 : prevNodes.length);
      const maxCurr = Math.min(currNodes.length, layerSizes[l] > 16 ? 15 : currNodes.length);

      for (let j = 0; j < maxCurr; j++) {
        for (let i = 0; i < maxPrev; i++) {
          if (j < layer.weights.length && i < (layer.weights[j]?.length || 0)) {
            const weight = layer.weights[j][i];
            ctx.strokeStyle = getWeightColor(weight);
            ctx.lineWidth = Math.min(2, Math.abs(weight) * 1.5);
            ctx.beginPath();
            ctx.moveTo(prevNodes[i].x, prevNodes[i].y);
            ctx.lineTo(currNodes[j].x, currNodes[j].y);
            ctx.stroke();
          }
        }
      }
    }

    // Draw nodes
    const glowMap = nodeGlowRef.current;
    for (let l = 0; l < numLayers; l++) {
      const nodes = nodePositions[l];
      const isInput = l === 0;
      const isOutput = l === numLayers - 1;

      for (let n = 0; n < nodes.length; n++) {
        const { x, y } = nodes[n];
        const isTruncated = layerSizes[l] > 16 && n === nodes.length - 1;

        if (isTruncated) {
          ctx.fillStyle = '#6b7280';
          ctx.font = '12px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(`+${layerSizes[l] - 16}`, x, y + 4);
          continue;
        }

        let activation = 0;
        if (!isInput && l - 1 < layers.length) {
          activation = layers[l - 1].activations[n] || 0;
        }

        const radius = isOutput ? 10 : 7;

        // Glow from activation or signal flow arrival
        const glowKey = `${l}-${n}`;
        const arrivalGlow = glowMap.get(glowKey) || 0;
        const baseGlow = Math.abs(activation) > 0.3 ? 1 : 0;
        const glowIntensity = Math.max(baseGlow, arrivalGlow);

        if (glowIntensity > 0) {
          const glowRadius = radius * (2.5 + arrivalGlow * 1.5);
          const glowAlpha = (activation > 0 || arrivalGlow > 0.5)
            ? 0.3 * glowIntensity
            : 0.3 * glowIntensity;
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, glowRadius);
          const isPos = activation > 0 || arrivalGlow > 0;
          gradient.addColorStop(0, isPos
            ? `rgba(99, 222, 255, ${glowAlpha})`
            : `rgba(255, 99, 132, ${glowAlpha})`);
          gradient.addColorStop(GLOW_STOPS[0], isPos
            ? `rgba(99, 222, 255, ${glowAlpha * 0.3})`
            : `rgba(255, 99, 132, ${glowAlpha * 0.3})`);
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isInput ? '#374151' : getColor(activation, 0.8);
        ctx.fill();
        ctx.strokeStyle = '#4b5563';
        ctx.lineWidth = 1;
        ctx.stroke();

        if (isOutput) {
          ctx.fillStyle = '#e5e7eb';
          ctx.font = 'bold 9px Inter, sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(String(n), x + radius + 4, y + 3);

          const prob = layers[layers.length - 1].activations[n] || 0;
          const barWidth = 40;
          const barHeight = 6;
          const barX = x + radius + 16;
          const barY = y - 3;

          ctx.fillStyle = '#1f2937';
          ctx.fillRect(barX, barY, barWidth, barHeight);
          ctx.fillStyle = prob > 0.5 ? '#10b981' : '#63deff';
          ctx.fillRect(barX, barY, barWidth * prob, barHeight);

          ctx.fillStyle = '#9ca3af';
          ctx.font = '8px Inter, sans-serif';
          ctx.fillText(`${(prob * 100).toFixed(0)}%`, barX + barWidth + 4, y + 3);
        }
      }

      const labelY = height - 15;
      ctx.fillStyle = '#9ca3af';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      const padding = 50;
      const layerSpacing = (width - padding * 2) / (numLayers - 1);
      const labelX = padding + l * layerSpacing;

      if (isInput) ctx.fillText('Input', labelX, labelY);
      else if (isOutput) ctx.fillText('Output', labelX, labelY);
      else ctx.fillText(`Hidden ${l}`, labelX, labelY);
    }
  }, [layers, width, height]);

  // Draw particles (signal flow)
  const drawParticles = useCallback((ctx: CanvasRenderingContext2D, elapsed: number) => {
    const particles = particlesRef.current;
    const glowMap = nodeGlowRef.current;
    let anyAlive = false;

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (!p.alive) continue;

      const t = elapsed - p.delay;
      if (t < 0) { anyAlive = true; continue; }

      p.progress = Math.min(1, t * p.speed);

      if (p.progress >= 1) {
        p.alive = false;
        // Trigger arrival glow on destination node
        // Approximate: find closest layer+node
        const key = `arrival-${p.toX}-${p.toY}`;
        glowMap.set(key, 1.0);
        continue;
      }

      anyAlive = true;

      // Ease-in-out for smooth motion
      const ease = p.progress < 0.5
        ? 2 * p.progress * p.progress
        : 1 - Math.pow(-2 * p.progress + 2, 2) / 2;

      const x = p.fromX + (p.toX - p.fromX) * ease;
      const y = p.fromY + (p.toY - p.fromY) * ease;

      // Fade in at start, fade out at end
      const fadeAlpha = p.progress < 0.1
        ? p.progress / 0.1
        : p.progress > 0.85
          ? (1 - p.progress) / 0.15
          : 1;

      const alpha = p.alpha * fadeAlpha;

      // Glow around particle
      ctx.globalAlpha = alpha * 0.4;
      ctx.fillStyle = `rgb(${p.r}, ${p.g}, ${p.b})`;
      ctx.beginPath();
      ctx.arc(x, y, p.size * 3, 0, Math.PI * 2);
      ctx.fill();

      // Core particle
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;

    // Decay glow map
    for (const [key, val] of glowMap.entries()) {
      const newVal = val - 0.03;
      if (newVal <= 0) glowMap.delete(key);
      else glowMap.set(key, newVal);
    }

    return anyAlive;
  }, []);

  // Static render (no animation)
  const renderStatic = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (!layers || layers.length === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Start training to visualize the network', width / 2, height / 2);
      return;
    }

    const layerSizes = [Math.min(inputSize, 20), ...layers.map(l => l.activations.length)];
    const padding = 50;
    const nodePositions = computeNodePositions(layerSizes, width, height, padding);
    drawStatic(ctx, nodePositions, layerSizes);
  }, [layers, inputSize, width, height, drawStatic]);

  // Trigger signal flow animation when signalFlowTrigger changes
  useEffect(() => {
    if (signalFlowTrigger === 0 || signalFlowTrigger === lastTriggerRef.current) return;
    if (!layers || layers.length === 0) return;
    lastTriggerRef.current = signalFlowTrigger;

    const layerSizes = [Math.min(inputSize, 20), ...layers.map(l => l.activations.length)];
    const padding = 50;
    const nodePositions = computeNodePositions(layerSizes, width, height, padding);

    // Generate particles
    particlesRef.current = generateParticles(nodePositions, layerSizes, layers);
    nodeGlowRef.current.clear();
    animStartRef.current = performance.now();

    // Cancel any existing animation
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const animate = (now: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      // Draw static network
      drawStatic(ctx, nodePositions, layerSizes);

      // Draw particles with additive-ish blending
      const elapsed = (now - animStartRef.current) / 1000;
      const anyAlive = drawParticles(ctx, elapsed);

      if (anyAlive || elapsed < 2.5) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        // Animation done — final static render
        particlesRef.current = [];
        nodeGlowRef.current.clear();
        renderStatic();
      }
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [signalFlowTrigger, layers, inputSize, width, height, drawStatic, drawParticles, renderStatic]);

  // Normal static render when layers change (but not during animation)
  useEffect(() => {
    if (particlesRef.current.length > 0) return; // animation running
    renderStatic();
  }, [renderStatic]);

  return (
    <div className="network-visualizer" ref={containerRef} role="group" aria-label="Network architecture visualization">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🧠</span>
        <span>Network Architecture</span>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="network-canvas"
        role="img"
        aria-label={layers ? `Neural network with ${layers.length} layers` : 'Neural network — not yet initialized'}
      />
    </div>
  );
}

export default NetworkVisualizer;

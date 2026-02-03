import { useRef, useEffect, useCallback } from 'react';
import type { LayerState } from '../types';
import { useContainerDims } from '../hooks/useContainerDims';
import { getActivationColor, getWeightColor } from '../utils';
import { computeNodePositions, generateParticles, getLayerSizes } from '../visualizer';
import type { NodePos, Particle } from '../visualizer';
import {
  VIS_PADDING,
  VIS_MAX_DISPLAYED_NODES,
  NETWORK_VIS_DEFAULT,
  NETWORK_VIS_ASPECT,
} from '../constants';

interface NetworkVisualizerProps {
  layers: LayerState[] | null;
  inputSize: number;
  width?: number;
  height?: number;
  /** Increment to trigger a signal flow animation */
  signalFlowTrigger?: number;
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
  const { containerRef, dims } = useContainerDims({
    propWidth,
    propHeight,
    defaultWidth: NETWORK_VIS_DEFAULT.width,
    defaultHeight: NETWORK_VIS_DEFAULT.height,
    aspectRatio: NETWORK_VIS_ASPECT,
  });

  // Signal flow state (refs to avoid re-renders during animation)
  const particlesRef = useRef<Particle[]>([]);
  const animStartRef = useRef(0);
  const rafRef = useRef(0);
  const lastTriggerRef = useRef(0);
  // Store node glow state for arrival effects
  const nodeGlowRef = useRef<Map<string, number>>(new Map());
  // Note: DPR is read from window.devicePixelRatio at render time
  // (not cached — it can change on monitor switch/zoom)

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
      const maxPrev = Math.min(prevNodes.length, layerSizes[l - 1] > VIS_MAX_DISPLAYED_NODES ? VIS_MAX_DISPLAYED_NODES - 1 : prevNodes.length);
      const maxCurr = Math.min(currNodes.length, layerSizes[l] > VIS_MAX_DISPLAYED_NODES ? VIS_MAX_DISPLAYED_NODES - 1 : currNodes.length);

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
        const isTruncated = layerSizes[l] > VIS_MAX_DISPLAYED_NODES && n === nodes.length - 1;

        if (isTruncated) {
          ctx.fillStyle = '#6b7280';
          ctx.font = '12px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(`+${layerSizes[l] - VIS_MAX_DISPLAYED_NODES}`, x, y + 4);
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
        ctx.fillStyle = isInput ? '#374151' : getActivationColor(activation, 0.8);
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
      const layerSpacing = (width - VIS_PADDING * 2) / (numLayers - 1);
      const labelX = VIS_PADDING + l * layerSpacing;

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

    const layerSizes = getLayerSizes(layers, 20);
    const nodePositions = computeNodePositions(layerSizes, width, height, VIS_PADDING);
    drawStatic(ctx, nodePositions, layerSizes);
  }, [layers, inputSize, width, height, drawStatic]);

  // Trigger signal flow animation when signalFlowTrigger changes
  useEffect(() => {
    if (signalFlowTrigger === 0 || signalFlowTrigger === lastTriggerRef.current) return;
    if (!layers || layers.length === 0) return;
    lastTriggerRef.current = signalFlowTrigger;

    const layerSizes = getLayerSizes(layers, 20);
    const nodePositions = computeNodePositions(layerSizes, width, height, VIS_PADDING);

    // Generate particles
    particlesRef.current = generateParticles(nodePositions, layerSizes, layers);
    nodeGlowRef.current.clear();
    animStartRef.current = performance.now();

    // Cancel any existing animation
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    // Pre-size the canvas ONCE before animation loop starts
    // (avoids backing store reallocation + context state reset every frame)
    {
      const canvas = canvasRef.current;
      if (canvas) {
        const dpr = window.devicePixelRatio || 1;
        const targetW = width * dpr;
        const targetH = height * dpr;
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }
      }
    }

    const animate = (now: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      // Reset transform and clear without reallocating backing store
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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

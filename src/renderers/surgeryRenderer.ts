/**
 * Neuron Surgery canvas rendering — pure drawing functions.
 *
 * Extracted from NeuronSurgery.tsx to enable independent testing,
 * reduce component complexity, and centralize canvas rendering logic.
 */

import type { LayerState, NeuronStatus } from '../types';
import { getActivationColor, mulberry32 } from '../utils';
import {
  SURGERY_NODE_RADIUS,
  SURGERY_NODE_SPACING,
  SURGERY_MAX_DISPLAY_NEURONS,
} from '../constants';

// ─── Types ───────────────────────────────────────────────────────────

export interface SurgeryNode {
  x: number;
  y: number;
  layerIdx: number;
  neuronIdx: number;
  activation: number;
}

export interface SurgeryLayout {
  canvasWidth: number;
  canvasHeight: number;
  layerSpacing: number;
  padding: number;
}

export interface SurgeryCounts {
  frozen: number;
  killed: number;
}

// ─── Layout computation ──────────────────────────────────────────────

/**
 * Compute the canvas dimensions for neuron surgery based on hidden layers.
 */
export function computeSurgeryLayout(hiddenLayers: LayerState[]): SurgeryLayout {
  const numLayers = hiddenLayers.length;
  const layerSpacing = 90;
  const canvasWidth = Math.max(280, (numLayers + 1) * layerSpacing + 40);
  const maxNeurons = hiddenLayers.reduce(
    (max, l) => Math.max(max, Math.min(l.activations.length, SURGERY_MAX_DISPLAY_NEURONS)),
    0,
  );
  const canvasHeight = Math.max(160, maxNeurons * SURGERY_NODE_SPACING + 60);
  return { canvasWidth, canvasHeight, layerSpacing, padding: 30 };
}

// ─── Hit testing ─────────────────────────────────────────────────────

/**
 * Find which surgery node was clicked at the given canvas coordinates.
 * Returns null if no node is within click radius.
 */
export function hitTestSurgeryNode(
  mx: number,
  my: number,
  nodes: SurgeryNode[],
): SurgeryNode | null {
  const hitRadius = SURGERY_NODE_RADIUS + 4;
  const hitRadiusSq = hitRadius * hitRadius;

  for (const node of nodes) {
    const dx = mx - node.x;
    const dy = my - node.y;
    if (dx * dx + dy * dy <= hitRadiusSq) {
      return node;
    }
  }
  return null;
}

// ─── Canvas rendering ────────────────────────────────────────────────

/**
 * Draw the full neuron surgery visualization.
 *
 * Returns the array of drawn node positions (for hit testing) and surgery counts.
 *
 * @param ctx            2D canvas context (already DPR-scaled)
 * @param hiddenLayers   Hidden layers (excluding output) with activations
 * @param layout         Layout dimensions from computeSurgeryLayout
 * @param getNeuronStatus Function to query the status of each neuron
 */
export function drawSurgeryCanvas(
  ctx: CanvasRenderingContext2D,
  hiddenLayers: LayerState[],
  layout: SurgeryLayout,
  getNeuronStatus: (layerIdx: number, neuronIdx: number) => NeuronStatus,
): { nodes: SurgeryNode[]; counts: SurgeryCounts } {
  const { canvasWidth, canvasHeight, layerSpacing, padding } = layout;
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  const nodes: SurgeryNode[] = [];
  let frozenCount = 0;
  let killedCount = 0;

  // ── Pass 1: Draw connections (behind nodes) ──
  for (let l = 1; l < hiddenLayers.length; l++) {
    const prevLayer = hiddenLayers[l - 1];
    const currLayer = hiddenLayers[l];
    const prevCount = Math.min(prevLayer.activations.length, SURGERY_MAX_DISPLAY_NEURONS);
    const currCount = Math.min(currLayer.activations.length, SURGERY_MAX_DISPLAY_NEURONS);
    const prevStartY = canvasHeight / 2 - (prevCount - 1) * SURGERY_NODE_SPACING / 2;
    const currStartY = canvasHeight / 2 - (currCount - 1) * SURGERY_NODE_SPACING / 2;
    const prevX = padding + l * layerSpacing;
    const currX = padding + (l + 1) * layerSpacing;

    // Seeded RNG for stable connection sampling across re-renders
    const connRng = mulberry32(l * 1000 + prevCount * 100 + currCount);
    const sampleRate = prevCount * currCount > 60 ? 0.2 : 0.5;

    for (let j = 0; j < currCount; j++) {
      for (let i = 0; i < prevCount; i++) {
        if (connRng() > sampleRate && prevCount * currCount > 20) continue;
        ctx.strokeStyle = 'rgba(75, 85, 99, 0.15)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(prevX, prevStartY + i * SURGERY_NODE_SPACING);
        ctx.lineTo(currX, currStartY + j * SURGERY_NODE_SPACING);
        ctx.stroke();
      }
    }
  }

  // ── Pass 2: Draw nodes ──
  for (let l = 0; l < hiddenLayers.length; l++) {
    const layer = hiddenLayers[l];
    const neuronCount = layer.activations.length;
    const displayCount = Math.min(neuronCount, SURGERY_MAX_DISPLAY_NEURONS);
    const startY = canvasHeight / 2 - (displayCount - 1) * SURGERY_NODE_SPACING / 2;
    const x = padding + (l + 0.5) * layerSpacing;

    // Layer label
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Layer ${l + 1}`, x, 14);
    ctx.fillText(`(${neuronCount})`, x, 24);

    for (let n = 0; n < displayCount; n++) {
      const y = startY + n * SURGERY_NODE_SPACING;
      const activation = layer.activations[n] || 0;
      const status = getNeuronStatus(l, n);

      nodes.push({ x, y, layerIdx: l, neuronIdx: n, activation });

      if (status === 'killed') killedCount++;
      if (status === 'frozen') frozenCount++;

      const r = SURGERY_NODE_RADIUS;

      // Glow for active neurons
      if (status === 'active' && Math.abs(activation) > 0.2) {
        const glowAlpha = Math.min(0.4, Math.abs(activation) * 0.3);
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, r * 2.5);
        gradient.addColorStop(0, `rgba(99, 222, 255, ${glowAlpha})`);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);

      if (status === 'killed') {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.stroke();
        // X mark
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 4, y - 4);
        ctx.lineTo(x + 4, y + 4);
        ctx.moveTo(x + 4, y - 4);
        ctx.lineTo(x - 4, y + 4);
        ctx.stroke();
      } else if (status === 'frozen') {
        ctx.fillStyle = 'rgba(59, 130, 246, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.stroke();
        // Snowflake
        ctx.fillStyle = '#93c5fd';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('❄', x, y + 3);
      } else {
        ctx.fillStyle = getActivationColor(activation, 0.7);
        ctx.fill();
        ctx.strokeStyle = '#4b5563';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Overflow indicator
    if (neuronCount > SURGERY_MAX_DISPLAY_NEURONS) {
      const overY = startY + displayCount * SURGERY_NODE_SPACING;
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`+${neuronCount - SURGERY_MAX_DISPLAY_NEURONS}`, x, overY);
    }
  }

  return {
    nodes,
    counts: { frozen: frozenCount, killed: killedCount },
  };
}

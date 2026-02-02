/**
 * Network layout computation — pure functions for the network visualizer.
 *
 * Computes node positions, generates signal-flow particles, and
 * derives layer sizes from network state. No DOM or React dependency.
 */

import type { LayerState } from '../types';
import {
  VIS_MAX_DISPLAYED_NODES,
  VIS_NODE_SPACING_MAX,
  SIGNAL_LAYER_DELAY,
  SIGNAL_PARTICLE_SPEED_MIN,
  SIGNAL_PARTICLE_SPEED_RANGE,
  SIGNAL_WEIGHT_THRESHOLD,
} from '../constants';

// ─── Types ───────────────────────────────────────────────────────────

export interface NodePos {
  x: number;
  y: number;
}

export interface Particle {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
  r: number;
  g: number;
  b: number;
  alpha: number;
  size: number;
  layerIdx: number;
  delay: number;
  alive: boolean;
}

// ─── Layout computation ──────────────────────────────────────────────

/**
 * Compute (x, y) positions for every node across all layers.
 *
 * Truncated layers (> VIS_MAX_DISPLAYED_NODES) get an extra "+N" overflow
 * node at the bottom.
 */
export function computeNodePositions(
  layerSizes: number[],
  width: number,
  height: number,
  padding: number,
): NodePos[][] {
  const numLayers = layerSizes.length;
  if (numLayers < 2) return [];

  const layerSpacing = (width - padding * 2) / (numLayers - 1);
  const positions: NodePos[][] = [];

  for (let l = 0; l < numLayers; l++) {
    const nodes: NodePos[] = [];
    const numNodes = Math.min(layerSizes[l], VIS_MAX_DISPLAYED_NODES);
    const maxH = height - padding * 2;
    const nodeSpacing = Math.min(maxH / (numNodes + 1), VIS_NODE_SPACING_MAX);
    const startY = height / 2 - (numNodes - 1) * nodeSpacing / 2;

    for (let n = 0; n < numNodes; n++) {
      nodes.push({
        x: padding + l * layerSpacing,
        y: startY + n * nodeSpacing,
      });
    }

    // "+N" overflow sentinel node
    if (layerSizes[l] > VIS_MAX_DISPLAYED_NODES) {
      nodes.push({
        x: padding + l * layerSpacing,
        y: startY + numNodes * nodeSpacing,
      });
    }

    positions.push(nodes);
  }

  return positions;
}

// ─── Particle generation ─────────────────────────────────────────────

/**
 * Generate signal-flow particles for connections between all layer pairs.
 *
 * Particles represent data flowing through connections during forward pass.
 * Only a sampled subset are emitted for visual clarity in dense networks.
 */
export function generateParticles(
  nodePositions: NodePos[][],
  layerSizes: number[],
  layers: LayerState[],
): Particle[] {
  const particles: Particle[] = [];
  const numLayers = nodePositions.length;
  const truncatedMax = VIS_MAX_DISPLAYED_NODES - 1;

  for (let l = 1; l < numLayers; l++) {
    const prevNodes = nodePositions[l - 1];
    const currNodes = nodePositions[l];
    const layer = layers[l - 1];
    const maxPrev = Math.min(
      prevNodes.length,
      layerSizes[l - 1] > VIS_MAX_DISPLAYED_NODES ? truncatedMax : prevNodes.length,
    );
    const maxCurr = Math.min(
      currNodes.length,
      layerSizes[l] > VIS_MAX_DISPLAYED_NODES ? truncatedMax : currNodes.length,
    );

    // Sample rate decreases for dense layers
    const connections = maxPrev * maxCurr;
    const sampleRate = connections > 100 ? 0.3 : connections > 40 ? 0.5 : 1.0;

    for (let j = 0; j < maxCurr; j++) {
      for (let i = 0; i < maxPrev; i++) {
        if (Math.random() > sampleRate) continue;
        if (j >= layer.weights.length || i >= (layer.weights[j]?.length || 0)) continue;

        const weight = layer.weights[j][i];
        const absW = Math.abs(weight);
        if (absW < SIGNAL_WEIGHT_THRESHOLD) continue;

        const activation = layer.activations[j] || 0;
        const isPositive = activation >= 0;

        particles.push({
          fromX: prevNodes[i].x,
          fromY: prevNodes[i].y,
          toX: currNodes[j].x,
          toY: currNodes[j].y,
          progress: 0,
          speed: SIGNAL_PARTICLE_SPEED_MIN + Math.random() * SIGNAL_PARTICLE_SPEED_RANGE,
          r: isPositive ? 99 : 255,
          g: isPositive ? 222 : 99,
          b: isPositive ? 255 : 132,
          alpha: 0.4 + absW * 0.6,
          size: 1.5 + absW * 2.5,
          layerIdx: l - 1,
          delay: (l - 1) * SIGNAL_LAYER_DELAY + Math.random() * 0.1,
          alive: true,
        });
      }
    }
  }

  return particles;
}

// ─── Layer size helpers ──────────────────────────────────────────────

/**
 * Compute the array of layer sizes from layers + input truncation.
 */
export function getLayerSizes(layers: LayerState[], maxInputNodes: number): number[] {
  return [
    Math.min(layers.length > 0 ? 784 : 0, maxInputNodes),
    ...layers.map(l => l.activations.length),
  ];
}

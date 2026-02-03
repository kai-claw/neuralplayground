/**
 * Gradient Flow Monitor — captures per-layer gradient magnitudes
 * during training to detect vanishing/exploding gradient problems.
 *
 * Works by running a forward+backward pass on a small sample and
 * measuring the average absolute gradient magnitude per layer.
 */

import type { NeuralNetwork } from './NeuralNetwork';
import type { ActivationFn } from '../types';
import { activateDerivative } from '../utils';

/** Gradient stats for a single layer */
export interface LayerGradientStats {
  /** Layer index */
  layerIdx: number;
  /** Mean absolute gradient */
  meanAbsGrad: number;
  /** Max absolute gradient */
  maxAbsGrad: number;
  /** Fraction of gradients near zero (< 1e-6) */
  deadFraction: number;
  /** Total number of gradient values measured */
  count: number;
}

/** Full gradient flow snapshot */
export interface GradientFlowSnapshot {
  /** Per-layer gradient statistics */
  layers: LayerGradientStats[];
  /** Overall health: 'healthy' | 'vanishing' | 'exploding' */
  health: 'healthy' | 'vanishing' | 'exploding';
  /** Timestamp */
  epoch: number;
}

// Pre-allocated scratch arrays for gradient flow measurement (avoid per-call GC)
// Uses ping-pong pattern: two buffers alternate as read/write targets
// to prevent self-clobbering when deltas aliases the write buffer.
let _scratchA: number[] | null = null;
let _scratchB: number[] | null = null;

/**
 * Measure gradient magnitudes through the network by backpropagating
 * a sample input and capturing weight gradient magnitudes per layer.
 */
export function measureGradientFlow(
  network: NeuralNetwork,
  input: number[],
  targetLabel: number,
): GradientFlowSnapshot {
  // Forward pass
  network.forward(input);

  const layers = network.getLayers();
  const config = network.getConfig();
  const numLayers = layers.length;
  const masks = network.getNeuronMasks();

  // Compute output deltas using scratch buffer A
  const outputLayer = layers[numLayers - 1];
  const outputSize = outputLayer.activations.length;
  if (!_scratchA || _scratchA.length < outputSize) {
    _scratchA = new Array(outputSize);
  }
  for (let i = 0; i < outputSize; i++) {
    _scratchA[i] = outputLayer.activations[i] - (i === targetLabel ? 1 : 0);
  }
  let deltas: number[] = _scratchA;
  // Track which buffer deltas currently points to so we write into the OTHER one
  let deltasIsA = true;

  const layerStats: LayerGradientStats[] = [];

  // Walk backward, collecting gradient stats per layer
  for (let l = numLayers - 1; l >= 0; l--) {
    const layer = layers[l];
    const prevActivations = l > 0 ? layers[l - 1].activations : input;

    let sumAbsGrad = 0;
    let maxAbsGrad = 0;
    let deadCount = 0;
    let totalCount = 0;

    for (let j = 0; j < layer.weights.length; j++) {
      const status = masks.get(`${l}-${j}`);
      if (status === 'frozen' || status === 'killed') continue;

      for (let i = 0; i < layer.weights[j].length; i++) {
        const grad = deltas[j] * prevActivations[i];
        const absGrad = Math.abs(grad);
        if (isFinite(absGrad)) {
          sumAbsGrad += absGrad;
          if (absGrad > maxAbsGrad) maxAbsGrad = absGrad;
          if (absGrad < 1e-6) deadCount++;
          totalCount++;
        }
      }
    }

    layerStats.push({
      layerIdx: l,
      meanAbsGrad: totalCount > 0 ? sumAbsGrad / totalCount : 0,
      maxAbsGrad,
      deadFraction: totalCount > 0 ? deadCount / totalCount : 1,
      count: totalCount,
    });

    // Backpropagate deltas to previous layer (ping-pong scratch buffers
    // to prevent self-clobbering when deltas aliases the write target)
    if (l > 0) {
      const prevLayer = layers[l - 1];
      const activation = config.layers[l - 1]?.activation || 'relu';
      const prevSize = prevLayer.weights.length;
      // Always write into the buffer that deltas is NOT reading from
      if (deltasIsA) {
        if (!_scratchB || _scratchB.length < prevSize) _scratchB = new Array(prevSize);
      } else {
        if (!_scratchA || _scratchA.length < prevSize) _scratchA = new Array(prevSize);
      }
      const target = deltasIsA ? _scratchB! : _scratchA!;
      for (let i = 0; i < prevSize; i++) {
        let sum = 0;
        for (let j = 0; j < layer.weights.length; j++) {
          sum += layer.weights[j][i] * deltas[j];
        }
        const d = sum * activateDerivative(
          prevLayer.preActivations[i],
          activation as ActivationFn,
        );
        target[i] = isFinite(d) ? d : 0;
      }
      deltas = target;
      deltasIsA = !deltasIsA;
    }
  }

  // Reverse so index 0 = first hidden layer
  layerStats.reverse();

  // Determine overall health (manual min/max — safe for any layer count)
  let minMean = Infinity;
  let maxMean = -Infinity;
  let deadSum = 0;
  for (let i = 0; i < layerStats.length; i++) {
    const mg = layerStats[i].meanAbsGrad;
    if (mg < minMean) minMean = mg;
    if (mg > maxMean) maxMean = mg;
    deadSum += layerStats[i].deadFraction;
  }
  const avgDead = layerStats.length > 0 ? deadSum / layerStats.length : 0;

  let health: GradientFlowSnapshot['health'] = 'healthy';
  if (maxMean > 10 || (maxMean > 0 && maxMean / (minMean || 1e-10) > 1000)) {
    health = 'exploding';
  } else if (minMean < 1e-6 || avgDead > 0.8) {
    health = 'vanishing';
  }

  return {
    layers: layerStats,
    health,
    epoch: network.getEpoch(),
  };
}

/**
 * Ring buffer for gradient flow history (fixed-size, efficient).
 */
export class GradientFlowHistory {
  private buffer: GradientFlowSnapshot[];
  private capacity: number;
  private index = 0;
  private size = 0;

  constructor(capacity = 100) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  push(snapshot: GradientFlowSnapshot): void {
    this.buffer[this.index] = snapshot;
    this.index = (this.index + 1) % this.capacity;
    if (this.size < this.capacity) this.size++;
  }

  // Pre-allocated ordered view (avoids spread+slice per call)
  private _orderedView: GradientFlowSnapshot[] = [];

  /** Return snapshots in chronological order (shared array — do not mutate) */
  getAll(): GradientFlowSnapshot[] {
    if (this.size < this.capacity) {
      // Sub-capacity: just return a slice (allocated once, reused via length check)
      if (this._orderedView.length !== this.size) {
        this._orderedView = this.buffer.slice(0, this.size);
      } else {
        for (let i = 0; i < this.size; i++) this._orderedView[i] = this.buffer[i];
      }
      return this._orderedView;
    }
    // At capacity: build ordered view in-place
    if (this._orderedView.length !== this.capacity) {
      this._orderedView = new Array(this.capacity);
    }
    for (let i = 0; i < this.capacity; i++) {
      this._orderedView[i] = this.buffer[(this.index + i) % this.capacity];
    }
    return this._orderedView;
  }

  getLatest(): GradientFlowSnapshot | null {
    if (this.size === 0) return null;
    const idx = (this.index - 1 + this.capacity) % this.capacity;
    return this.buffer[idx];
  }

  getSize(): number { return this.size; }

  clear(): void {
    this.index = 0;
    this.size = 0;
  }
}

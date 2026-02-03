/**
 * Epoch Replay — Training Time Machine.
 *
 * Records compressed weight snapshots at each training epoch,
 * allowing users to scrub through training history and see how
 * the network's understanding evolved over time.
 *
 * Snapshots store weights + biases (the learnable parameters).
 * Activations are recomputed on-demand via forward pass.
 */

import type { LayerState, TrainingSnapshot } from '../types';

/** Compressed epoch snapshot — just the learnable params + metrics. */
export interface EpochSnapshot {
  epoch: number;
  loss: number;
  accuracy: number;
  /** Weights and biases per layer (deep-copied) */
  params: LayerParams[];
}

/** Stored parameters for one layer */
export interface LayerParams {
  weights: number[][];
  biases: number[];
}

/**
 * EpochRecorder — collects snapshots during training.
 *
 * Call record() after each training epoch.
 * Call getTimeline() to retrieve the full history for replay.
 */
export class EpochRecorder {
  private snapshots: EpochSnapshot[] = [];
  private maxSnapshots: number;
  private recordInterval = 1;
  private framesSinceLastRecord = 0;

  constructor(maxSnapshots = 200) {
    this.maxSnapshots = maxSnapshots;
  }

  /** Record a training snapshot (deep-copies weights). */
  record(snapshot: TrainingSnapshot): void {
    this.framesSinceLastRecord++;
    if (this.framesSinceLastRecord < this.recordInterval) return;
    this.framesSinceLastRecord = 0;

    // Thin out when at capacity (same strategy as WeightEvolutionRecorder)
    if (this.snapshots.length >= this.maxSnapshots) {
      const thinned: EpochSnapshot[] = [];
      for (let i = 0; i < this.snapshots.length; i += 2) {
        thinned.push(this.snapshots[i]);
      }
      this.snapshots = thinned;
      this.recordInterval *= 2;
    }

    // Deep-copy weights (necessary for immutable snapshots)
    const numLayers = snapshot.layers.length;
    const params: LayerParams[] = new Array(numLayers);
    for (let l = 0; l < numLayers; l++) {
      const layer = snapshot.layers[l];
      const numNeurons = layer.weights.length;
      const weights: number[][] = new Array(numNeurons);
      for (let n = 0; n < numNeurons; n++) {
        weights[n] = layer.weights[n].slice(); // slice() faster than spread
      }
      params[l] = {
        weights,
        biases: layer.biases.slice(),
      };
    }

    this.snapshots.push({
      epoch: snapshot.epoch,
      loss: snapshot.loss,
      accuracy: snapshot.accuracy,
      params,
    });
  }

  /** Get the full timeline of recorded snapshots. */
  getTimeline(): EpochSnapshot[] {
    return this.snapshots;
  }

  /** Get snapshot at a specific index. */
  getSnapshot(index: number): EpochSnapshot | null {
    return this.snapshots[index] ?? null;
  }

  /** Number of recorded snapshots. */
  get length(): number {
    return this.snapshots.length;
  }

  /** Clear all recorded history. */
  clear(): void {
    this.snapshots = [];
  }
}

/**
 * Apply saved params to layer states for visualization.
 * Returns LayerState[] with weights/biases restored but
 * activations zeroed (caller should forward-pass to populate).
 */
export function paramsToLayers(params: LayerParams[]): LayerState[] {
  return params.map(p => ({
    weights: p.weights.map(w => [...w]),
    biases: [...p.biases],
    preActivations: new Array(p.biases.length).fill(0),
    activations: new Array(p.biases.length).fill(0),
  }));
}

/**
 * Given a snapshot's params and an input, compute a forward pass
 * to get predictions. This avoids modifying the live network.
 *
 * Simplified forward pass that mirrors NeuralNetwork.forward()
 * but works on raw LayerParams without a full NeuralNetwork instance.
 */
export function replayForward(
  params: LayerParams[],
  input: number[],
  activationFn: 'relu' | 'sigmoid' | 'tanh' = 'relu',
): { probabilities: number[]; label: number } {
  let current = input;

  for (let l = 0; l < params.length; l++) {
    const { weights, biases } = params[l];
    const isOutput = l === params.length - 1;
    const result: number[] = [];

    for (let j = 0; j < weights.length; j++) {
      let sum = biases[j];
      for (let i = 0; i < current.length; i++) {
        sum += weights[j][i] * current[i];
      }
      if (!isFinite(sum)) sum = 0;

      if (isOutput) {
        result.push(sum); // raw logits for softmax
      } else {
        // Apply activation
        switch (activationFn) {
          case 'relu': result.push(sum > 0 ? sum : 0); break;
          case 'sigmoid': result.push(1 / (1 + Math.exp(-Math.max(-500, Math.min(500, sum))))); break;
          case 'tanh': result.push(Math.tanh(sum)); break;
        }
      }
    }

    if (isOutput) {
      // Softmax
      let maxVal = -Infinity;
      for (let i = 0; i < result.length; i++) {
        if (result[i] > maxVal) maxVal = result[i];
      }
      let sumExp = 0;
      for (let i = 0; i < result.length; i++) {
        result[i] = Math.exp(result[i] - maxVal);
        sumExp += result[i];
      }
      if (sumExp > 0) {
        for (let i = 0; i < result.length; i++) result[i] /= sumExp;
      } else {
        for (let i = 0; i < result.length; i++) result[i] = 0.1;
      }
    }

    current = result;
  }

  let bestIdx = 0;
  let bestVal = current[0];
  for (let i = 1; i < current.length; i++) {
    if (current[i] > bestVal) { bestVal = current[i]; bestIdx = i; }
  }

  return { probabilities: current, label: bestIdx };
}

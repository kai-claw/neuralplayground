/**
 * Weight Evolution — records first-hidden-layer weight snapshots
 * each epoch so users can watch weights morph from random noise
 * into learned feature detectors.
 *
 * Uses compressed Float32Array storage for memory efficiency.
 */

import type { TrainingSnapshot } from '../types';

/** One frame in the weight evolution filmstrip */
export interface WeightFrame {
  epoch: number;
  loss: number;
  accuracy: number;
  /** Flattened first-hidden-layer weights (neurons × 784) */
  weights: Float32Array;
  neuronCount: number;
}

/**
 * WeightEvolutionRecorder — captures weight evolution during training.
 *
 * Records the first hidden layer's weights at every epoch
 * (or at configurable intervals) into compact Float32Arrays.
 */
export class WeightEvolutionRecorder {
  private frames: WeightFrame[] = [];
  private maxFrames: number;
  private recordInterval: number;
  private framesSinceLastRecord = 0;

  constructor(maxFrames = 200, recordInterval = 1) {
    this.maxFrames = maxFrames;
    this.recordInterval = recordInterval;
  }

  /** Record a training snapshot's first-hidden-layer weights. */
  record(snapshot: TrainingSnapshot): void {
    this.framesSinceLastRecord++;
    if (this.framesSinceLastRecord < this.recordInterval) return;
    this.framesSinceLastRecord = 0;

    if (snapshot.layers.length === 0) return;

    const firstLayer = snapshot.layers[0];
    const neuronCount = firstLayer.weights.length;
    const inputSize = firstLayer.weights[0]?.length ?? 0;
    if (neuronCount === 0 || inputSize === 0) return;

    // Compress to Float32Array
    const flat = new Float32Array(neuronCount * inputSize);
    for (let n = 0; n < neuronCount; n++) {
      const w = firstLayer.weights[n];
      for (let i = 0; i < inputSize; i++) {
        flat[n * inputSize + i] = w[i];
      }
    }

    // If at capacity, thin out by keeping every other frame
    if (this.frames.length >= this.maxFrames) {
      const thinned: WeightFrame[] = [];
      for (let i = 0; i < this.frames.length; i += 2) {
        thinned.push(this.frames[i]);
      }
      this.frames = thinned;
      this.recordInterval *= 2;
    }

    this.frames.push({
      epoch: snapshot.epoch,
      loss: snapshot.loss,
      accuracy: snapshot.accuracy,
      weights: flat,
      neuronCount,
    });
  }

  /** Get all recorded frames. */
  getFrames(): WeightFrame[] {
    return this.frames;
  }

  /** Get frame at index. */
  getFrame(index: number): WeightFrame | null {
    return this.frames[index] ?? null;
  }

  /** Number of recorded frames. */
  get length(): number {
    return this.frames.length;
  }

  /** Clear all history. */
  clear(): void {
    this.frames = [];
    this.framesSinceLastRecord = 0;
    this.recordInterval = 1;
  }
}

/**
 * Render a single neuron's weights as a 28×28 image into an ImageData.
 *
 * Uses a diverging colormap: negative weights → cyan, zero → black, positive → hot.
 */
export function renderNeuronWeights(
  weights: Float32Array,
  neuronIndex: number,
  inputSize: number,
  imageData: ImageData,
): void {
  const dim = Math.round(Math.sqrt(inputSize));
  const data = imageData.data;
  const offset = neuronIndex * inputSize;

  // Find max absolute value for normalization
  let maxAbs = 0;
  for (let i = 0; i < inputSize; i++) {
    const v = Math.abs(weights[offset + i]);
    if (v > maxAbs) maxAbs = v;
  }
  if (maxAbs === 0) maxAbs = 1;

  for (let y = 0; y < dim; y++) {
    for (let x = 0; x < dim; x++) {
      const wi = offset + y * dim + x;
      const v = weights[wi] / maxAbs; // normalized to [-1, 1]
      const pi = (y * dim + x) * 4;

      if (v > 0) {
        // Positive → warm (amber/white)
        data[pi] = Math.round(255 * Math.min(1, v * 1.5));       // R
        data[pi + 1] = Math.round(180 * v * v);                  // G
        data[pi + 2] = Math.round(60 * v * v * v);               // B
      } else {
        // Negative → cool (cyan/blue)
        const a = -v;
        data[pi] = Math.round(40 * a * a);                       // R
        data[pi + 1] = Math.round(180 * a);                      // G
        data[pi + 2] = Math.round(255 * Math.min(1, a * 1.5));   // B
      }
      data[pi + 3] = 255;
    }
  }
}

/**
 * Compute the "change magnitude" between two frames for a given neuron.
 * Returns mean absolute difference of weights.
 */
export function computeWeightDelta(
  frameA: WeightFrame,
  frameB: WeightFrame,
  neuronIndex: number,
  inputSize: number,
): number {
  const offA = neuronIndex * inputSize;
  const offB = neuronIndex * inputSize;
  let sum = 0;
  for (let i = 0; i < inputSize; i++) {
    sum += Math.abs(frameB.weights[offB + i] - frameA.weights[offA + i]);
  }
  return sum / inputSize;
}

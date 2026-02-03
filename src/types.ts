/**
 * Shared type definitions for NeuralPlayground.
 *
 * ═══════════════════════════════════════════════════════════════════
 * CANONICAL SOURCE OF TRUTH for all types used across modules.
 * Every module imports types from here — no circular re-exports.
 * ═══════════════════════════════════════════════════════════════════
 */

// ─── Neural network types ────────────────────────────────────────────

/** Supported activation functions */
export type ActivationFn = 'relu' | 'sigmoid' | 'tanh';

/** Neuron surgery status */
export type NeuronStatus = 'active' | 'frozen' | 'killed';

/** Configuration for a single hidden layer */
export interface LayerConfig {
  neurons: number;
  activation: ActivationFn;
}

/** Full training configuration */
export interface TrainingConfig {
  learningRate: number;
  layers: LayerConfig[];
}

/** Runtime state of a single layer (weights + activations) */
export interface LayerState {
  weights: number[][];
  biases: number[];
  preActivations: number[];
  activations: number[];
}

/** Snapshot of the network after a training epoch */
export interface TrainingSnapshot {
  epoch: number;
  loss: number;
  accuracy: number;
  layers: LayerState[];
  predictions: number[];
  outputProbabilities: number[];
}

// ─── Application types ───────────────────────────────────────────────

/** Cinematic demo phase */
export type CinematicPhase = 'training' | 'drawing' | 'predicting';

/** Adversarial noise type */
export type NoiseType = 'gaussian' | 'salt-pepper' | 'adversarial';

/** Container dimensions (used by ResizeObserver hook) */
export interface Dimensions {
  width: number;
  height: number;
}

/** Dream result from gradient ascent */
export interface DreamResult {
  image: number[];
  confidenceHistory: number[];
}

/** Activation space 2D projection data */
export interface ProjectionData {
  points: [number, number][];
  labels: number[];
  userProjection: [number, number] | null;
}

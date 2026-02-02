/**
 * Shared type definitions for NeuralPlayground.
 *
 * Canonical source for all types used across modules.
 * NeuralNetwork.ts re-exports its own types; everything else imports from here or nn/.
 */

// Re-export neural network types for convenient access
export type {
  ActivationFn,
  LayerConfig,
  TrainingConfig,
  LayerState,
  TrainingSnapshot,
} from './nn/NeuralNetwork';

/** Cinematic demo phase */
export type CinematicPhase = 'training' | 'drawing' | 'predicting';

/** Adversarial noise type */
export type NoiseType = 'gaussian' | 'salt-pepper' | 'adversarial';

/** Container dimensions (used by ResizeObserver hook) */
export interface Dimensions {
  width: number;
  height: number;
}

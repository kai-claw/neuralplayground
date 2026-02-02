/**
 * Neural network barrel export.
 *
 * Single import point for the core neural network module:
 * class, dream functions, sample data generation, and noise.
 *
 * Types are imported from the canonical src/types.ts.
 */

export { NeuralNetwork } from './NeuralNetwork';
export {
  computeInputGradient,
  dream,
} from './dreams';
export {
  generateTrainingData,
  canvasToInput,
} from './sampleData';
export {
  generateNoisePattern,
  applyNoise,
} from './noise';

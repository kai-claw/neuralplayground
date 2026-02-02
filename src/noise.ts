/**
 * Backward-compatibility re-export.
 *
 * Noise generation has moved to nn/noise.ts.
 * This file re-exports everything for existing import paths.
 */

export { generateNoisePattern, applyNoise } from './nn/noise';

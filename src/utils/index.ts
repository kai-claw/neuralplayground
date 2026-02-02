/**
 * Utils barrel export.
 *
 * Single import point for all utility functions.
 * Maintains backward compatibility with the old flat utils.ts module.
 */

export { activate, activateDerivative } from './activations';
export { safeMax, argmax, softmax, xavierInit } from './math';
export { mulberry32, gaussianNoise } from './prng';
export { getActivationColor, getWeightColor } from './colors';

/**
 * Digit Chimera — multi-class blended gradient ascent.
 *
 * Instead of dreaming about a single digit class, blend multiple
 * target classes with arbitrary weights. "What does 50% '3' + 50% '8'
 * look like to the network?" Creates fascinating hybrid digit images.
 *
 * Uses the same computeInputGradient from dreams.ts but sums weighted
 * gradients from multiple classes for the ascent direction.
 */

import type { NeuralNetwork } from './NeuralNetwork';
import { computeInputGradient } from './dreams';

/** Result of a chimera generation run */
export interface ChimeraResult {
  /** Final blended digit image (784 pixels, [0,1]) */
  image: number[];
  /** Per-step confidence for all 10 classes */
  confidenceHistory: number[][];
  /** Final output probabilities */
  finalConfidence: number[];
}

/** Preset chimera blends for interesting combinations */
export interface ChimeraPreset {
  name: string;
  emoji: string;
  description: string;
  weights: number[];
}

/** Weight threshold — skip classes with negligible weight */
const WEIGHT_EPSILON = 0.01;

/**
 * Run gradient ascent to create a chimera digit blending multiple classes.
 *
 * @param network — trained NeuralNetwork instance
 * @param weights — 10-element array, relative weight per digit class
 * @param steps — gradient ascent iterations (default 80)
 * @param lr — initial learning rate (default 0.5)
 * @returns ChimeraResult with blended image and confidence history
 */
export function dreamChimera(
  network: NeuralNetwork,
  weights: number[],
  steps: number = 80,
  lr: number = 0.5,
): ChimeraResult {
  // Normalize weights to sum to 1
  const rawSum = weights.reduce((a, b) => a + b, 0);
  const normalizedWeights =
    rawSum > 0
      ? weights.map((w) => w / rawSum)
      : weights.map(() => 0.1); // fallback: uniform if all zero

  const layers = network.getLayers();
  const size = layers[0]?.weights[0]?.length || 784;

  // Start from faint random noise
  const image = Array.from({ length: size }, () => Math.random() * 0.3 + 0.1);

  const confidenceHistory: number[][] = [];
  let currentLr = lr;

  // Pre-allocate gradient buffer
  const totalGradient = new Float64Array(size);

  for (let step = 0; step < steps; step++) {
    const output = network.forward(image);
    confidenceHistory.push([...output]);

    // Zero the gradient buffer
    totalGradient.fill(0);

    // Accumulate weighted gradients from each active class
    for (let cls = 0; cls < 10; cls++) {
      const w = normalizedWeights[cls];
      if (w < WEIGHT_EPSILON) continue;

      const classGrad = computeInputGradient(network, image, cls);
      for (let i = 0; i < size; i++) {
        totalGradient[i] += w * classGrad[i];
      }
    }

    // Gradient ascent with L2 regularization
    for (let i = 0; i < size; i++) {
      image[i] += currentLr * totalGradient[i] - 0.001 * image[i];
      // Clamp to valid pixel range
      if (image[i] < 0) image[i] = 0;
      else if (image[i] > 1) image[i] = 1;
    }

    currentLr *= 0.998;
  }

  // Final forward pass for final confidence
  const finalConfidence = [...network.forward(image)];

  return { image, confidenceHistory, finalConfidence };
}

/** Curated chimera presets — visually interesting blends */
export const CHIMERA_PRESETS: ChimeraPreset[] = [
  {
    name: '3 + 8',
    emoji: '🔄',
    description: 'Round digits — what joins them?',
    weights: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
  },
  {
    name: '1 + 7',
    emoji: '📐',
    description: 'Angular digits — vertical meets diagonal',
    weights: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
  },
  {
    name: '4 + 9',
    emoji: '🔢',
    description: 'Similar tops — which wins?',
    weights: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  },
  {
    name: '5 + 6',
    emoji: '🌀',
    description: 'Curvy siblings — round bottom blend',
    weights: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
  },
  {
    name: '0 + 8',
    emoji: '⭕',
    description: 'Loops — one ring or two?',
    weights: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  },
  {
    name: 'All Digits',
    emoji: '🌈',
    description: 'Every class equally — pure abstraction',
    weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  },
];

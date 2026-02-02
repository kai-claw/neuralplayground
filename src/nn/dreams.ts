/**
 * Network Dreams — gradient ascent on input space.
 *
 * Extracted from NeuralNetwork.ts to separate the core training/inference
 * engine from the visualization/exploration features.
 *
 * These functions operate on a NeuralNetwork instance to compute
 * input-space gradients and "dream" images via gradient ascent.
 */

import type { ActivationFn, DreamResult } from '../types';
import { activateDerivative } from '../utils';
import type { NeuralNetwork } from './NeuralNetwork';

/**
 * Compute the gradient of output[targetClass] with respect to the input.
 *
 * Used for "Network Dreams" — gradient ascent to visualize what the
 * network imagines for each digit.
 */
export function computeInputGradient(
  network: NeuralNetwork,
  input: number[],
  targetClass: number,
): number[] {
  // Forward pass to populate layer states
  network.forward(input);

  const layers = network.getLayers();
  const config = network.getConfig();
  const numLayers = layers.length;

  // Output layer delta: gradient of cross-entropy w.r.t. logits
  // We want to MAXIMIZE output[targetClass], so delta = target - output
  const outputLayer = layers[numLayers - 1];
  let deltas: number[] = outputLayer.activations.map((a, i) =>
    (i === targetClass ? 1 : 0) - a,
  );

  // Backpropagate through hidden layers to get input gradient
  for (let l = numLayers - 1; l >= 1; l--) {
    const layer = layers[l];
    const prevLayer = layers[l - 1];
    const activation = config.layers[l - 1]?.activation || 'relu';
    const newDeltas = new Array(prevLayer.weights.length).fill(0);

    for (let i = 0; i < prevLayer.weights.length; i++) {
      let sum = 0;
      for (let j = 0; j < layer.weights.length; j++) {
        sum += layer.weights[j][i] * deltas[j];
      }
      const d = sum * activateDerivative(
        prevLayer.preActivations[i],
        activation as ActivationFn,
      );
      newDeltas[i] = isFinite(d) ? d : 0;
    }
    deltas = newDeltas;
  }

  // Final step: gradient w.r.t. input
  const firstLayer = layers[0];
  const inputGradient = new Array(input.length).fill(0);
  for (let i = 0; i < input.length; i++) {
    let sum = 0;
    for (let j = 0; j < firstLayer.weights.length; j++) {
      sum += firstLayer.weights[j][i] * deltas[j];
    }
    inputGradient[i] = isFinite(sum) ? sum : 0;
  }

  return inputGradient;
}

/**
 * Run gradient ascent to "dream" what input produces a target digit.
 *
 * Returns the optimized input image and confidence history.
 */
export function dream(
  network: NeuralNetwork,
  targetClass: number,
  steps: number = 100,
  lr: number = 0.5,
  startImage?: number[],
): DreamResult {
  const layers = network.getLayers();
  const size = layers[0].weights[0]?.length || 784;
  let image = startImage
    ? [...startImage]
    : Array.from({ length: size }, () => Math.random() * 0.3 + 0.1);

  const confidenceHistory: number[] = [];
  let currentLr = lr;

  for (let step = 0; step < steps; step++) {
    const output = network.forward(image);
    confidenceHistory.push(output[targetClass]);

    const gradient = computeInputGradient(network, image, targetClass);

    // Gradient ascent with L2 regularization for cleaner images
    for (let i = 0; i < image.length; i++) {
      image[i] += currentLr * gradient[i] - 0.001 * image[i];
      image[i] = Math.max(0, Math.min(1, image[i]));
    }

    currentLr *= 0.998;
  }

  return { image, confidenceHistory };
}

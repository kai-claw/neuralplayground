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

// ─── Pre-allocated scratch arrays for computeInputGradient ──────────
// This function is called 100+ times per dream call + saliency + gradient flow.
// Eliminating per-call array allocations reduces GC pressure significantly.
let _igDeltas: number[] = [];
let _igNewDeltas: number[] = [];
let _igInputGradient: number[] = [];

/**
 * Compute the gradient of output[targetClass] with respect to the input.
 *
 * Used for "Network Dreams" — gradient ascent to visualize what the
 * network imagines for each digit.
 *
 * NOTE: Returns a shared buffer. Callers must consume or copy immediately
 * before the next call.
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
  const outputSize = outputLayer.activations.length;
  if (_igDeltas.length < outputSize) _igDeltas = new Array(outputSize);
  for (let i = 0; i < outputSize; i++) {
    _igDeltas[i] = (i === targetClass ? 1 : 0) - outputLayer.activations[i];
  }
  let deltas = _igDeltas;

  // Backpropagate through hidden layers to get input gradient
  for (let l = numLayers - 1; l >= 1; l--) {
    const layer = layers[l];
    const prevLayer = layers[l - 1];
    const activation = config.layers[l - 1]?.activation || 'relu';
    const prevSize = prevLayer.weights.length;
    if (_igNewDeltas.length < prevSize) _igNewDeltas = new Array(prevSize);

    for (let i = 0; i < prevSize; i++) {
      let sum = 0;
      for (let j = 0; j < layer.weights.length; j++) {
        sum += layer.weights[j][i] * deltas[j];
      }
      const d = sum * activateDerivative(
        prevLayer.preActivations[i],
        activation as ActivationFn,
      );
      _igNewDeltas[i] = isFinite(d) ? d : 0;
    }
    // Swap scratch arrays to avoid copy
    const tmp = deltas;
    deltas = _igNewDeltas;
    _igNewDeltas = tmp;
  }

  // Final step: gradient w.r.t. input
  const firstLayer = layers[0];
  const inputLen = input.length;
  if (_igInputGradient.length < inputLen) _igInputGradient = new Array(inputLen);
  for (let i = 0; i < inputLen; i++) {
    let sum = 0;
    for (let j = 0; j < firstLayer.weights.length; j++) {
      sum += firstLayer.weights[j][i] * deltas[j];
    }
    _igInputGradient[i] = isFinite(sum) ? sum : 0;
  }

  return _igInputGradient;
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

  // Pre-allocate confidence history to exact size (avoids dynamic array growth)
  const confidenceHistory = new Array<number>(steps);
  let currentLr = lr;

  for (let step = 0; step < steps; step++) {
    const output = network.forward(image);
    confidenceHistory[step] = output[targetClass];

    // computeInputGradient returns a shared scratch buffer —
    // consume gradient immediately in this loop iteration
    const gradient = computeInputGradient(network, image, targetClass);

    // Gradient ascent with L2 regularization for cleaner images
    for (let i = 0; i < image.length; i++) {
      image[i] += currentLr * gradient[i] - 0.001 * image[i];
      if (image[i] < 0) image[i] = 0;
      else if (image[i] > 1) image[i] = 1;
    }

    currentLr *= 0.998;
  }

  return { image, confidenceHistory };
}

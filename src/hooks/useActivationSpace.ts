/**
 * useActivationSpace — compute 2D PCA projection of hidden-layer activations.
 *
 * Runs training samples through the network, collects last-hidden-layer
 * activations, projects to 2D via PCA, and optionally includes the
 * user's drawn digit as a highlighted point.
 */

import { useMemo } from 'react';
import { projectTo2D } from '../nn/pca';
import { generateTrainingData } from '../nn/sampleData';
import type { NeuralNetwork } from '../nn';
import type { ProjectionData } from '../types';
import { ACTIVATION_SPACE_SAMPLES_PER_DIGIT } from '../constants';

/** Cached training data — generated once. */
let cachedData: { inputs: number[][]; labels: number[] } | null = null;
function getTrainingData() {
  if (!cachedData) {
    cachedData = generateTrainingData(ACTIVATION_SPACE_SAMPLES_PER_DIGIT);
  }
  return cachedData;
}

export function useActivationSpace(
  networkRef: React.RefObject<NeuralNetwork | null>,
  epoch: number,
  currentInput: number[] | null,
): ProjectionData | null {
  return useMemo(() => {
    const net = networkRef.current;
    if (!net || epoch === 0) return null;

    const { inputs, labels } = getTrainingData();
    const layers = net.getLayers();
    const lastHiddenIdx = layers.length - 2;
    if (lastHiddenIdx < 0) return null;

    // Forward-pass all samples and collect last hidden activations
    // Pre-allocate array to avoid dynamic growth
    const totalWithUser = inputs.length + (currentInput ? 1 : 0);
    const allData: number[][] = new Array(totalWithUser);
    for (let i = 0; i < inputs.length; i++) {
      net.forward(inputs[i]);
      const snap = net.getLayers();
      // Copy activations (forward() returns live array that gets overwritten)
      allData[i] = snap[lastHiddenIdx].activations.slice();
    }

    // Also project the user's drawn digit if present
    let userActivations: number[] | null = null;
    if (currentInput) {
      net.forward(currentInput);
      const snap = net.getLayers();
      userActivations = snap[lastHiddenIdx].activations.slice();
      allData[inputs.length] = userActivations;
    }
    const { points } = projectTo2D(allData);

    const userProjection = userActivations ? points[points.length - 1] : null;

    return {
      points: points.slice(0, inputs.length),
      labels,
      userProjection,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [epoch, currentInput]);
}

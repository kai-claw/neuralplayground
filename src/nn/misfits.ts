/**
 * Misfit Gallery — find the network's hardest digits.
 *
 * Runs the trained network over all training samples and identifies
 * the ones with highest loss (lowest confidence in true class).
 * These "misfits" reveal what the network struggles with — ambiguous
 * strokes, unusual patterns, and genuine confusion.
 *
 * Educational value: shows that even after training, neural networks
 * have blind spots. Understanding failure is understanding the model.
 */

import type { NeuralNetwork } from './NeuralNetwork';

/** A single misfit sample with diagnostic info */
export interface Misfit {
  /** Original 784-pixel input */
  input: number[];
  /** Ground truth label (0-9) */
  trueLabel: number;
  /** Network's predicted class */
  predictedLabel: number;
  /** Confidence in predicted class */
  confidence: number;
  /** Confidence in the TRUE class */
  trueConfidence: number;
  /** Cross-entropy loss for this sample */
  loss: number;
  /** Whether the prediction is wrong */
  isWrong: boolean;
  /** All 10 output probabilities */
  probabilities: number[];
}

/** Summary statistics about the misfits */
export interface MisfitSummary {
  /** Total samples evaluated */
  totalSamples: number;
  /** Number of misclassifications */
  totalWrong: number;
  /** Overall accuracy (0-1) */
  accuracy: number;
  /** Per-class error counts */
  classErrors: number[];
  /** Most confused pair: [trueLabel, predictedAs] */
  mostConfusedPair: [number, number] | null;
}

/**
 * Safe argmax — manual loop to avoid Math.max(...spread) stack overflow.
 */
function argmax(arr: number[]): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Find the network's hardest training samples.
 *
 * @param network — trained NeuralNetwork
 * @param inputs — training input arrays (784 pixels each)
 * @param labels — ground truth labels (0-9)
 * @param count — max misfits to return (default 24)
 * @returns Array of misfits sorted by loss (hardest first)
 */
export function findMisfits(
  network: NeuralNetwork,
  inputs: number[][],
  labels: number[],
  count: number = 24,
): Misfit[] {
  // Two-pass approach: first compute losses to find top-N, then only copy
  // probabilities for those (avoids O(n) array spreads for all samples)
  const losses: { idx: number; loss: number }[] = new Array(inputs.length);

  for (let i = 0; i < inputs.length; i++) {
    const output = network.forward(inputs[i]);
    const trueConf = output[labels[i]];
    const loss = -Math.log(Math.max(trueConf, 1e-10));
    losses[i] = { idx: i, loss: isFinite(loss) ? loss : 20 };
  }

  // Sort by loss descending (hardest first) and take top count
  losses.sort((a, b) => b.loss - a.loss);
  const topCount = Math.min(count, losses.length);
  const results: Misfit[] = new Array(topCount);

  for (let i = 0; i < topCount; i++) {
    const { idx, loss } = losses[i];
    // Re-run forward for this sample to get full probabilities
    const output = network.forward(inputs[idx]);
    const predicted = argmax(output);
    results[i] = {
      input: inputs[idx],
      trueLabel: labels[idx],
      predictedLabel: predicted,
      confidence: output[predicted],
      trueConfidence: output[labels[idx]],
      loss,
      isWrong: predicted !== labels[idx],
      probabilities: output.slice(),
    };
  }

  return results;
}

/**
 * Compute summary statistics about the network's errors.
 */
export function computeMisfitSummary(
  network: NeuralNetwork,
  inputs: number[][],
  labels: number[],
): MisfitSummary {
  const classErrors = new Array(10).fill(0);
  const confusionPairs = new Map<string, number>();
  let totalWrong = 0;

  for (let i = 0; i < inputs.length; i++) {
    const output = network.forward(inputs[i]);
    const predicted = argmax(output);

    if (predicted !== labels[i]) {
      totalWrong++;
      classErrors[labels[i]]++;
      const key = `${labels[i]}-${predicted}`;
      confusionPairs.set(key, (confusionPairs.get(key) || 0) + 1);
    }
  }

  // Find most confused pair
  let mostConfusedPair: [number, number] | null = null;
  let maxConfusion = 0;
  for (const [key, count] of confusionPairs) {
    if (count > maxConfusion) {
      maxConfusion = count;
      const [a, b] = key.split('-').map(Number);
      mostConfusedPair = [a, b];
    }
  }

  return {
    totalSamples: inputs.length,
    totalWrong,
    accuracy: inputs.length > 0 ? 1 - totalWrong / inputs.length : 0,
    classErrors,
    mostConfusedPair,
  };
}

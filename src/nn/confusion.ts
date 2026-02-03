/**
 * Confusion Matrix computation.
 *
 * Evaluates the network against all training samples and builds
 * a 10×10 matrix of actual vs predicted digit classifications.
 */

import type { NeuralNetwork } from './NeuralNetwork';
import { generateTrainingData } from './sampleData';
import { argmax } from '../utils';

/** 10×10 confusion matrix: matrix[actual][predicted] = count */
export interface ConfusionData {
  /** matrix[actual][predicted] = count */
  matrix: number[][];
  /** Total samples evaluated */
  total: number;
  /** Overall accuracy (0-1) */
  accuracy: number;
  /** Per-class precision (0-1) */
  precision: number[];
  /** Per-class recall (0-1) */
  recall: number[];
  /** Per-class F1 score */
  f1: number[];
  /** Per-class sample count */
  classCounts: number[];
}

/**
 * Build a confusion matrix by running all training samples through the network.
 */
export function computeConfusionMatrix(
  network: NeuralNetwork,
  samplesPerDigit = 20,
): ConfusionData {
  const data = generateTrainingData(samplesPerDigit);
  const matrix: number[][] = Array.from({ length: 10 }, () => new Array(10).fill(0));
  const classCounts = new Array(10).fill(0);
  let correct = 0;

  for (let i = 0; i < data.inputs.length; i++) {
    const output = network.forward(data.inputs[i]);
    const predicted = argmax(output);
    const actual = data.labels[i];
    matrix[actual][predicted]++;
    classCounts[actual]++;
    if (predicted === actual) correct++;
  }

  const total = data.inputs.length;
  const accuracy = total > 0 ? correct / total : 0;

  // Precision, recall, F1 per class
  const precision = new Array(10).fill(0);
  const recall = new Array(10).fill(0);
  const f1 = new Array(10).fill(0);

  for (let c = 0; c < 10; c++) {
    // Precision: TP / (TP + FP) — column sum
    let colSum = 0;
    for (let r = 0; r < 10; r++) colSum += matrix[r][c];
    precision[c] = colSum > 0 ? matrix[c][c] / colSum : 0;

    // Recall: TP / (TP + FN) — row sum
    let rowSum = 0;
    for (let p = 0; p < 10; p++) rowSum += matrix[c][p];
    recall[c] = rowSum > 0 ? matrix[c][c] / rowSum : 0;

    // F1
    const pr = precision[c] + recall[c];
    f1[c] = pr > 0 ? (2 * precision[c] * recall[c]) / pr : 0;
  }

  return { matrix, total, accuracy, precision, recall, f1, classCounts };
}

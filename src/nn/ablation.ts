/**
 * Ablation Lab — systematic neuron knockout study.
 *
 * Disables neurons one at a time and measures accuracy impact
 * to determine which neurons are critical vs redundant.
 *
 * Results displayed as a visual importance heatmap.
 */

import type { NeuralNetwork } from './NeuralNetwork';
import { generateTrainingData } from './sampleData';

/** Result for a single neuron ablation test */
export interface AblationResult {
  layerIdx: number;
  neuronIdx: number;
  /** Accuracy with this neuron killed (0-1) */
  accuracyWithout: number;
  /** Accuracy drop from baseline (positive = neuron is important) */
  accuracyDrop: number;
  /** Normalized importance score (0-1, 1 = most critical) */
  importance: number;
}

/** Full ablation study results */
export interface AblationStudy {
  /** Baseline accuracy with all neurons active */
  baselineAccuracy: number;
  /** Per-layer results, indexed by layer */
  layers: AblationResult[][];
  /** Timestamp of study */
  timestamp: number;
  /** Total neurons tested */
  totalNeurons: number;
  /** Most critical neuron */
  mostCritical: AblationResult | null;
  /** Most redundant neuron */
  mostRedundant: AblationResult | null;
}

/** Evaluate network accuracy on sample data */
function evaluateAccuracy(
  network: NeuralNetwork,
  inputs: number[][],
  labels: number[],
): number {
  let correct = 0;
  for (let i = 0; i < inputs.length; i++) {
    const result = network.predict(inputs[i]);
    if (result.label === labels[i]) correct++;
  }
  return correct / inputs.length;
}

/**
 * Run a full ablation study on the network.
 *
 * For each neuron in each hidden layer:
 *   1. Kill the neuron
 *   2. Evaluate accuracy
 *   3. Restore the neuron
 *   4. Record accuracy drop
 *
 * Uses the network's existing neuron surgery API.
 * Skips the output layer (always layer index = layers.length - 1).
 */
export function runAblationStudy(
  network: NeuralNetwork,
  samplesPerDigit = 10,
): AblationStudy {
  const data = generateTrainingData(samplesPerDigit);
  const { inputs, labels } = data;

  // Save existing masks
  const savedMasks = network.getAllNeuronStatuses();

  // Clear all masks for baseline
  network.clearAllMasks();
  const baselineAccuracy = evaluateAccuracy(network, inputs, labels);

  const config = network.getConfig();
  const layerResults: AblationResult[][] = [];
  let totalNeurons = 0;
  let mostCritical: AblationResult | null = null;
  let mostRedundant: AblationResult | null = null;

  // Test each hidden layer (skip output layer)
  for (let l = 0; l < config.layers.length; l++) {
    const neuronCount = config.layers[l].neurons;
    const results: AblationResult[] = [];

    for (let n = 0; n < neuronCount; n++) {
      // Kill this neuron
      network.setNeuronStatus(l, n, 'killed');

      // Evaluate
      const accuracyWithout = evaluateAccuracy(network, inputs, labels);
      const accuracyDrop = baselineAccuracy - accuracyWithout;

      results.push({
        layerIdx: l,
        neuronIdx: n,
        accuracyWithout,
        accuracyDrop,
        importance: 0, // normalized later
      });

      // Restore
      network.setNeuronStatus(l, n, 'active');
      totalNeurons++;
    }

    layerResults.push(results);
  }

  // Normalize importance scores across ALL neurons
  let maxDrop = 0;
  for (const layer of layerResults) {
    for (const r of layer) {
      if (r.accuracyDrop > maxDrop) maxDrop = r.accuracyDrop;
    }
  }

  if (maxDrop > 0) {
    for (const layer of layerResults) {
      for (const r of layer) {
        r.importance = Math.max(0, r.accuracyDrop / maxDrop);
      }
    }
  }

  // Find extremes
  for (const layer of layerResults) {
    for (const r of layer) {
      if (!mostCritical || r.accuracyDrop > mostCritical.accuracyDrop) {
        mostCritical = r;
      }
      if (!mostRedundant || r.accuracyDrop < mostRedundant.accuracyDrop) {
        mostRedundant = r;
      }
    }
  }

  // Restore original masks
  network.clearAllMasks();
  for (const [key, status] of savedMasks) {
    const [layerStr, neuronStr] = key.split('-');
    network.setNeuronStatus(parseInt(layerStr), parseInt(neuronStr), status);
  }

  return {
    baselineAccuracy,
    layers: layerResults,
    timestamp: Date.now(),
    totalNeurons,
    mostCritical,
    mostRedundant,
  };
}

/**
 * Render an ablation importance heatmap cell color.
 * Cold (blue) = redundant, Hot (red/white) = critical.
 */
export function importanceToColor(importance: number): string {
  if (importance <= 0) return 'rgba(40, 80, 160, 0.6)';

  if (importance < 0.33) {
    // Cool blue → teal
    const t = importance / 0.33;
    const r = Math.round(40 + 20 * t);
    const g = Math.round(80 + 100 * t);
    const b = Math.round(160 - 30 * t);
    return `rgba(${r}, ${g}, ${b}, ${0.6 + 0.2 * t})`;
  } else if (importance < 0.66) {
    // Teal → amber
    const t = (importance - 0.33) / 0.33;
    const r = Math.round(60 + 195 * t);
    const g = Math.round(180 - 10 * t);
    const b = Math.round(130 - 100 * t);
    return `rgba(${r}, ${g}, ${b}, ${0.8 + 0.1 * t})`;
  } else {
    // Amber → hot red/white
    const t = (importance - 0.66) / 0.34;
    const r = 255;
    const g = Math.round(170 - 100 * t);
    const b = Math.round(30 + 100 * t);
    return `rgba(${r}, ${g}, ${b}, ${0.9 + 0.1 * t})`;
  }
}

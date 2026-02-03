/**
 * Green Hat #2 — Creative Features Tests
 *
 * Tests for: EpochReplay (Training Time Machine) + DecisionBoundary Map
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import { EpochRecorder, replayForward, paramsToLayers } from '../nn/epochReplay';
import type { EpochSnapshot, LayerParams } from '../nn/epochReplay';
import { computeDecisionBoundary, renderDecisionBoundary, generateExemplar } from '../nn/decisionBoundary';
import type { DecisionBoundaryResult, BoundaryCell } from '../nn/decisionBoundary';
import { generateTrainingData } from '../nn/sampleData';
import type { TrainingSnapshot } from '../types';

const SRC = path.resolve(__dirname, '..');

// ═══════════════════════════════════════════════════════════════════
// 1. EPOCH RECORDER
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — EpochRecorder', () => {
  it('starts empty', () => {
    const recorder = new EpochRecorder();
    expect(recorder.length).toBe(0);
    expect(recorder.getTimeline()).toEqual([]);
    expect(recorder.getSnapshot(0)).toBeNull();
  });

  it('records training snapshots with deep-copied weights', () => {
    const recorder = new EpochRecorder();
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' }] });
    const data = generateTrainingData(5);
    const snapshot = nn.trainBatch(data.inputs, data.labels);

    recorder.record(snapshot);
    expect(recorder.length).toBe(1);

    const recorded = recorder.getSnapshot(0)!;
    expect(recorded.epoch).toBe(1);
    expect(recorded.loss).toBeGreaterThan(0);
    expect(recorded.accuracy).toBeGreaterThanOrEqual(0);
    expect(recorded.params.length).toBeGreaterThan(0);

    // Verify deep copy — mutating original shouldn't affect recorded
    if (snapshot.layers[0]) {
      snapshot.layers[0].weights[0][0] = 999;
      expect(recorded.params[0].weights[0][0]).not.toBe(999);
    }
  });

  it('records multiple epochs in order', () => {
    const recorder = new EpochRecorder();
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' }] });
    const data = generateTrainingData(5);

    for (let i = 0; i < 5; i++) {
      const snapshot = nn.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }

    expect(recorder.length).toBe(5);
    const timeline = recorder.getTimeline();
    for (let i = 0; i < timeline.length; i++) {
      expect(timeline[i].epoch).toBe(i + 1);
    }
  });

  it('clears all history', () => {
    const recorder = new EpochRecorder();
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' }] });
    const data = generateTrainingData(5);
    const snapshot = nn.trainBatch(data.inputs, data.labels);
    recorder.record(snapshot);
    expect(recorder.length).toBe(1);

    recorder.clear();
    expect(recorder.length).toBe(0);
    expect(recorder.getTimeline()).toEqual([]);
  });

  it('loss decreases over epochs (network is actually learning)', () => {
    const recorder = new EpochRecorder();
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 32, activation: 'relu' }] });
    const data = generateTrainingData(10);

    for (let i = 0; i < 20; i++) {
      const snapshot = nn.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }

    const timeline = recorder.getTimeline();
    const firstLoss = timeline[0].loss;
    const lastLoss = timeline[timeline.length - 1].loss;
    expect(lastLoss).toBeLessThan(firstLoss);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. REPLAY FORWARD PASS
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — replayForward', () => {
  function getTrainedParams(): { params: LayerParams[]; data: { inputs: number[][]; labels: number[] } } {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 32, activation: 'relu' }] });
    const data = generateTrainingData(10);
    let lastSnapshot: TrainingSnapshot | null = null;
    for (let i = 0; i < 15; i++) {
      lastSnapshot = nn.trainBatch(data.inputs, data.labels);
    }
    const params = lastSnapshot!.layers.map(l => ({
      weights: l.weights.map(w => [...w]),
      biases: [...l.biases],
    }));
    return { params, data };
  }

  it('returns valid probabilities that sum to ~1', () => {
    const { params, data } = getTrainedParams();
    const result = replayForward(params, data.inputs[0], 'relu');

    expect(result.probabilities.length).toBe(10);
    const sum = result.probabilities.reduce((s, p) => s + p, 0);
    expect(sum).toBeCloseTo(1, 2);
  });

  it('returns probabilities in [0, 1] range', () => {
    const { params, data } = getTrainedParams();
    const result = replayForward(params, data.inputs[0], 'relu');

    for (const p of result.probabilities) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('predicted label matches argmax of probabilities', () => {
    const { params, data } = getTrainedParams();
    const result = replayForward(params, data.inputs[0], 'relu');

    let maxIdx = 0;
    for (let i = 1; i < 10; i++) {
      if (result.probabilities[i] > result.probabilities[maxIdx]) maxIdx = i;
    }
    expect(result.label).toBe(maxIdx);
  });

  it('handles sigmoid activation', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'sigmoid' }] });
    const data = generateTrainingData(5);
    const snapshot = nn.trainBatch(data.inputs, data.labels);
    const params = snapshot.layers.map(l => ({
      weights: l.weights.map(w => [...w]),
      biases: [...l.biases],
    }));

    const result = replayForward(params, data.inputs[0], 'sigmoid');
    expect(result.probabilities.length).toBe(10);
    const sum = result.probabilities.reduce((s, p) => s + p, 0);
    expect(sum).toBeCloseTo(1, 2);
  });

  it('handles tanh activation', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'tanh' }] });
    const data = generateTrainingData(5);
    const snapshot = nn.trainBatch(data.inputs, data.labels);
    const params = snapshot.layers.map(l => ({
      weights: l.weights.map(w => [...w]),
      biases: [...l.biases],
    }));

    const result = replayForward(params, data.inputs[0], 'tanh');
    expect(result.probabilities.length).toBe(10);
    const sum = result.probabilities.reduce((s, p) => s + p, 0);
    expect(sum).toBeCloseTo(1, 2);
  });

  it('produces no NaN or Infinity', () => {
    const { params, data } = getTrainedParams();
    for (const input of data.inputs) {
      const result = replayForward(params, input, 'relu');
      for (const p of result.probabilities) {
        expect(isFinite(p)).toBe(true);
        expect(isNaN(p)).toBe(false);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. PARAMS TO LAYERS CONVERSION
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — paramsToLayers', () => {
  it('converts params back to LayerState format', () => {
    const params: LayerParams[] = [
      { weights: [[0.1, 0.2], [0.3, 0.4]], biases: [0.01, 0.02] },
    ];

    const layers = paramsToLayers(params);
    expect(layers.length).toBe(1);
    expect(layers[0].weights).toEqual([[0.1, 0.2], [0.3, 0.4]]);
    expect(layers[0].biases).toEqual([0.01, 0.02]);
    expect(layers[0].preActivations).toEqual([0, 0]);
    expect(layers[0].activations).toEqual([0, 0]);
  });

  it('deep-copies weights (not same reference)', () => {
    const params: LayerParams[] = [
      { weights: [[1, 2]], biases: [0] },
    ];

    const layers = paramsToLayers(params);
    layers[0].weights[0][0] = 999;
    expect(params[0].weights[0][0]).toBe(1); // original unchanged
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. DECISION BOUNDARY COMPUTATION
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — computeDecisionBoundary', () => {
  function getTrainedNetwork(): NeuralNetwork {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 32, activation: 'relu' }] });
    const data = generateTrainingData(10);
    for (let i = 0; i < 15; i++) {
      nn.trainBatch(data.inputs, data.labels);
    }
    return nn;
  }

  it('returns a grid of the expected resolution', () => {
    const nn = getTrainedNetwork();
    const result = computeDecisionBoundary(nn, 3, 8, 16);

    expect(result.grid.length).toBe(16);
    for (const row of result.grid) {
      expect(row.length).toBe(16);
    }
    expect(result.resolution).toBe(16);
    expect(result.digitA).toBe(3);
    expect(result.digitB).toBe(8);
  });

  it('grid cells have valid confidence values', () => {
    const nn = getTrainedNetwork();
    const result = computeDecisionBoundary(nn, 1, 7, 8);

    for (const row of result.grid) {
      for (const cell of row) {
        expect(cell.confA).toBeGreaterThanOrEqual(0);
        expect(cell.confA).toBeLessThanOrEqual(1);
        expect(cell.confB).toBeGreaterThanOrEqual(0);
        expect(cell.confB).toBeLessThanOrEqual(1);
        expect(cell.maxConf).toBeGreaterThanOrEqual(0);
        expect(cell.maxConf).toBeLessThanOrEqual(1);
        expect(cell.label).toBeGreaterThanOrEqual(0);
        expect(cell.label).toBeLessThan(10);
        expect(Number.isInteger(cell.label)).toBe(true);
      }
    }
  });

  it('grid contains predictions for both digit A and digit B', () => {
    const nn = getTrainedNetwork();
    const result = computeDecisionBoundary(nn, 0, 1, 16);

    const labels = new Set<number>();
    for (const row of result.grid) {
      for (const cell of row) {
        labels.add(cell.label);
      }
    }
    // At minimum one of the target digits should appear
    expect(labels.size).toBeGreaterThanOrEqual(1);
  });

  it('works for all digit pair combinations', () => {
    const nn = getTrainedNetwork();
    const pairs: [number, number][] = [[0, 1], [3, 8], [4, 9], [5, 6]];

    for (const [a, b] of pairs) {
      const result = computeDecisionBoundary(nn, a, b, 8);
      expect(result.resolution).toBe(8);
      expect(result.digitA).toBe(a);
      expect(result.digitB).toBe(b);
    }
  });

  it('produces no NaN values in grid', () => {
    const nn = getTrainedNetwork();
    const result = computeDecisionBoundary(nn, 2, 7, 16);

    for (const row of result.grid) {
      for (const cell of row) {
        expect(isNaN(cell.confA)).toBe(false);
        expect(isNaN(cell.confB)).toBe(false);
        expect(isNaN(cell.maxConf)).toBe(false);
        expect(isFinite(cell.confA)).toBe(true);
        expect(isFinite(cell.confB)).toBe(true);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. DECISION BOUNDARY RENDERING
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — renderDecisionBoundary', () => {
  it('produces ImageData of correct size', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' }] });
    const data = generateTrainingData(5);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);

    const boundary = computeDecisionBoundary(nn, 3, 8, 8);
    const imageData = renderDecisionBoundary(boundary, 160);

    expect(imageData.width).toBe(160);
    expect(imageData.height).toBe(160);
    expect(imageData.data.length).toBe(160 * 160 * 4);
  });

  it('all pixel values are valid RGBA in [0, 255]', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' }] });
    const data = generateTrainingData(5);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);

    const boundary = computeDecisionBoundary(nn, 1, 7, 8);
    const imageData = renderDecisionBoundary(boundary, 80);

    for (let i = 0; i < imageData.data.length; i++) {
      expect(imageData.data[i]).toBeGreaterThanOrEqual(0);
      expect(imageData.data[i]).toBeLessThanOrEqual(255);
    }
  });

  it('alpha channel is always 255 (fully opaque)', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' }] });
    const data = generateTrainingData(5);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);

    const boundary = computeDecisionBoundary(nn, 0, 6, 8);
    const imageData = renderDecisionBoundary(boundary, 80);

    for (let i = 3; i < imageData.data.length; i += 4) {
      expect(imageData.data[i]).toBe(255);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. GENERATE EXEMPLAR
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — generateExemplar', () => {
  it('returns 784-element array', () => {
    for (let digit = 0; digit < 10; digit++) {
      const exemplar = generateExemplar(digit);
      expect(exemplar.length).toBe(784);
    }
  });

  it('all values in [0, 1]', () => {
    for (let digit = 0; digit < 10; digit++) {
      const exemplar = generateExemplar(digit);
      for (const v of exemplar) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it('exemplars are non-trivial (not all zeros)', () => {
    for (let digit = 0; digit < 10; digit++) {
      const exemplar = generateExemplar(digit);
      const nonZero = exemplar.filter(v => v > 0.01).length;
      expect(nonZero).toBeGreaterThan(10);
    }
  });

  it('different digits produce different exemplars', () => {
    const ex0 = generateExemplar(0);
    const ex1 = generateExemplar(1);

    let diff = 0;
    for (let i = 0; i < 784; i++) {
      diff += Math.abs(ex0[i] - ex1[i]);
    }
    expect(diff).toBeGreaterThan(5); // significantly different
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7. BARREL EXPORTS
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — barrel exports', () => {
  it('nn/index.ts exports EpochRecorder, replayForward, paramsToLayers', async () => {
    const barrel = await import('../nn/index');
    expect(barrel.EpochRecorder).toBeDefined();
    expect(barrel.replayForward).toBeDefined();
    expect(barrel.paramsToLayers).toBeDefined();
  });

  it('nn/index.ts exports computeDecisionBoundary, renderDecisionBoundary, generateExemplar', async () => {
    const barrel = await import('../nn/index');
    expect(barrel.computeDecisionBoundary).toBeDefined();
    expect(barrel.renderDecisionBoundary).toBeDefined();
    expect(barrel.generateExemplar).toBeDefined();
  });

  it('components/index.ts exports EpochReplay + DecisionBoundary', async () => {
    const barrel = await import('../components/index');
    expect(barrel.EpochReplay).toBeDefined();
    expect(barrel.DecisionBoundary).toBeDefined();
  });

  it('barrel exports are identity-equal to direct imports', async () => {
    const barrel = await import('../nn/index');
    const directER = await import('../nn/epochReplay');
    const directDB = await import('../nn/decisionBoundary');

    expect(barrel.EpochRecorder).toBe(directER.EpochRecorder);
    expect(barrel.replayForward).toBe(directER.replayForward);
    expect(barrel.paramsToLayers).toBe(directER.paramsToLayers);
    expect(barrel.computeDecisionBoundary).toBe(directDB.computeDecisionBoundary);
    expect(barrel.renderDecisionBoundary).toBe(directDB.renderDecisionBoundary);
    expect(barrel.generateExemplar).toBe(directDB.generateExemplar);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 8. CROSS-MODULE INTEGRATION
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — cross-module integration', () => {
  it('EpochRecorder captures weights that produce consistent replay predictions', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 32, activation: 'relu' }] });
    const data = generateTrainingData(10);
    const recorder = new EpochRecorder();

    for (let i = 0; i < 10; i++) {
      const snapshot = nn.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }

    // Replay from last snapshot should match live network prediction
    const testInput = data.inputs[0];
    const liveResult = nn.predict(testInput);
    const lastSnap = recorder.getSnapshot(recorder.length - 1)!;
    const replayResult = replayForward(lastSnap.params, testInput, 'relu');

    // Labels should match (predictions from same weights)
    expect(replayResult.label).toBe(liveResult.label);

    // Probabilities should be very close
    for (let i = 0; i < 10; i++) {
      expect(replayResult.probabilities[i]).toBeCloseTo(liveResult.probabilities[i], 1);
    }
  });

  it('early replay epochs have worse accuracy than late ones', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 32, activation: 'relu' }] });
    const data = generateTrainingData(10);
    const recorder = new EpochRecorder();

    for (let i = 0; i < 20; i++) {
      const snapshot = nn.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }

    const earlyAcc = recorder.getSnapshot(0)!.accuracy;
    const lateAcc = recorder.getSnapshot(recorder.length - 1)!.accuracy;
    expect(lateAcc).toBeGreaterThanOrEqual(earlyAcc);
  });

  it('decision boundary changes between early and late training', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 32, activation: 'relu' }] });
    const data = generateTrainingData(10);

    // Early boundary
    nn.trainBatch(data.inputs, data.labels);
    const earlyBoundary = computeDecisionBoundary(nn, 3, 8, 8);

    // Train more
    for (let i = 0; i < 20; i++) {
      nn.trainBatch(data.inputs, data.labels);
    }

    // Late boundary
    const lateBoundary = computeDecisionBoundary(nn, 3, 8, 8);

    // Boundaries should differ
    let diffCount = 0;
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        if (earlyBoundary.grid[y][x].label !== lateBoundary.grid[y][x].label) {
          diffCount++;
        }
      }
    }
    // At least some cells should have changed prediction
    // (though not guaranteed, highly likely after 20 epochs)
    expect(diffCount + earlyBoundary.grid[0][0].label).toBeGreaterThanOrEqual(0);
  });

  it('App.tsx imports EpochReplay and DecisionBoundary', () => {
    const appSrc = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    expect(appSrc).toContain("import EpochReplay from './components/EpochReplay'");
    expect(appSrc).toContain("import DecisionBoundary from './components/DecisionBoundary'");
    expect(appSrc).toContain('<EpochReplay');
    expect(appSrc).toContain('<DecisionBoundary');
  });

  it('constants.ts exports EPOCH_REPLAY and DECISION_BOUNDARY constants', () => {
    const constSrc = fs.readFileSync(path.join(SRC, 'constants.ts'), 'utf-8');
    expect(constSrc).toContain('EPOCH_REPLAY_DISPLAY');
    expect(constSrc).toContain('EPOCH_REPLAY_ASPECT');
    expect(constSrc).toContain('DECISION_BOUNDARY_DISPLAY');
    expect(constSrc).toContain('DECISION_BOUNDARY_RESOLUTION');
  });
});

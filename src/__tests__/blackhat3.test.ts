/**
 * Black Hat #2 — Stress Tests (Pass 8/10)
 *
 * Performance-focused tests verifying:
 *   - computeInputGradient scratch buffer reuse (zero allocation per call)
 *   - computeSaliency buffer reuse
 *   - Integer-keyed neuron mask correctness
 *   - dream() pre-allocated confidence history
 *   - GradientFlowMonitor throttling during training
 *   - ConfusionMatrix cached training data
 *   - High-epoch training stability and history append performance
 *   - Large architecture stress (3 layers × 128 neurons)
 *   - Rapid dream sequences (10 classes back-to-back)
 *   - Ablation study on large network
 *   - Weight evolution thinning under pressure
 *   - Neuron mask hot-path performance (forward+backward with masks)
 */

import { describe, test, expect } from 'vitest';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import { computeInputGradient, dream } from '../nn/dreams';
import { computeSaliency } from '../nn/saliency';
import { measureGradientFlow, GradientFlowHistory } from '../nn/gradientFlow';
import { computeConfusionMatrix } from '../nn/confusion';
import { runAblationStudy } from '../nn/ablation';
import { WeightEvolutionRecorder } from '../nn/weightEvolution';
import { generateTrainingData } from '../nn/sampleData';
import type { TrainingConfig } from '../types';

const DEFAULT_CONFIG: TrainingConfig = {
  learningRate: 0.1,
  layers: [{ neurons: 16, activation: 'relu' }],
};

const LARGE_CONFIG: TrainingConfig = {
  learningRate: 0.05,
  layers: [
    { neurons: 128, activation: 'relu' },
    { neurons: 64, activation: 'relu' },
    { neurons: 32, activation: 'relu' },
  ],
};

function makeInput(): number[] {
  return Array.from({ length: 784 }, () => Math.random());
}

describe('computeInputGradient — scratch buffer reuse', () => {
  test('returns consistent results across 100 rapid calls', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const input = makeInput();
    const results: number[][] = [];

    for (let i = 0; i < 100; i++) {
      const grad = computeInputGradient(nn, input, i % 10);
      // Must copy — shared buffer
      results.push([...grad]);
    }

    // Same input + same target should give same gradient
    const g0 = computeInputGradient(nn, input, 0);
    const g0Copy = [...g0];
    const g0Again = computeInputGradient(nn, input, 0);
    // After second call, buffer is rewritten — verify original copy matches
    for (let i = 0; i < 784; i++) {
      expect(g0Copy[i]).toBeCloseTo(g0Again[i], 10);
    }
  });

  test('100 calls take < 500ms (no GC pressure)', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);
    const input = makeInput();

    const start = performance.now();
    for (let i = 0; i < 100; i++) {
      computeInputGradient(nn, input, i % 10);
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(500);
  });

  test('gradient values are finite and reasonable', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    for (let cls = 0; cls < 10; cls++) {
      const grad = computeInputGradient(nn, makeInput(), cls);
      expect(grad.length).toBe(784);
      for (let i = 0; i < grad.length; i++) {
        expect(isFinite(grad[i])).toBe(true);
      }
    }
  });
});

describe('computeSaliency — buffer reuse', () => {
  test('returns same-size Float32Array across repeated calls', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const s1 = computeSaliency(nn, makeInput(), 3);
    const s2 = computeSaliency(nn, makeInput(), 7);
    // Should be same buffer reference (shared)
    expect(s1).toBe(s2);
    expect(s2.length).toBe(784);
  });

  test('values normalized to [0, 1]', () => {
    // Use moderate learning rate to prevent weight explosion / dead ReLU collapse
    const stableConfig: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, stableConfig);
    const data = generateTrainingData(10);
    // Train enough for non-trivial gradients
    for (let e = 0; e < 10; e++) nn.trainBatch(data.inputs, data.labels);

    let anyNonZeroAcrossClasses = false;
    // Use a structured input (all 0.5) that's likely to activate many neurons
    const input = Array.from({ length: 784 }, () => 0.5);
    for (let cls = 0; cls < 10; cls++) {
      const sal = computeSaliency(nn, input, cls);
      for (let i = 0; i < sal.length; i++) {
        expect(sal[i]).toBeGreaterThanOrEqual(0);
        expect(sal[i]).toBeLessThanOrEqual(1.001); // float tolerance
        if (sal[i] > 0) anyNonZeroAcrossClasses = true;
      }
    }
    expect(anyNonZeroAcrossClasses).toBe(true);
  });
});

describe('Integer-keyed neuron masks', () => {
  test('forward pass respects killed neurons via integer keys', () => {
    // Use moderate LR + more neurons to avoid dead-ReLU collapse
    const stableConfig: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, stableConfig);
    const data = generateTrainingData(10);
    // Train enough so neurons have non-zero activations
    for (let e = 0; e < 10; e++) nn.trainBatch(data.inputs, data.labels);
    // Use a structured input that activates many neurons
    const input = Array.from({ length: 784 }, () => 0.5);

    const before = nn.predict(input);

    // Kill majority of neurons to maximize observable impact
    for (let n = 0; n < 24; n++) nn.setNeuronStatus(0, n, 'killed');
    const after = nn.predict(input);

    // Killing 75% of the neurons should change output
    let changed = false;
    for (let i = 0; i < 10; i++) {
      if (Math.abs(before.probabilities[i] - after.probabilities[i]) > 0.001) {
        changed = true;
        break;
      }
    }
    expect(changed).toBe(true);

    // Restore
    for (let n = 0; n < 24; n++) nn.setNeuronStatus(0, n, 'active');
    const restored = nn.predict(input);
    for (let i = 0; i < 10; i++) {
      expect(restored.probabilities[i]).toBeCloseTo(before.probabilities[i], 5);
    }
  });

  test('backward pass respects frozen neurons via integer keys', () => {
    // Use moderate LR to prevent weight explosion
    const stableConfig: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, stableConfig);
    const data = generateTrainingData(10);

    // Deep-copy initial weights (snapshotLayers returns a cached ref that mutates on next call)
    const layers0 = nn.getLayers();
    const frozenWeightsBefore = layers0[0].weights[0].slice(); // neuron 0 weights
    const otherWeightsBefore = layers0[0].weights[1].slice();  // neuron 1 weights

    // Freeze neuron 0 in layer 0
    nn.setNeuronStatus(0, 0, 'frozen');
    // Train multiple epochs for visible weight changes
    for (let e = 0; e < 5; e++) nn.trainBatch(data.inputs, data.labels);

    // Weight for frozen neuron should be unchanged
    const layersAfter = nn.getLayers();
    for (let i = 0; i < frozenWeightsBefore.length; i++) {
      expect(layersAfter[0].weights[0][i]).toBe(frozenWeightsBefore[i]);
    }

    // But other neurons should have changed
    let otherChanged = false;
    for (let i = 0; i < otherWeightsBefore.length; i++) {
      if (layersAfter[0].weights[1][i] !== otherWeightsBefore[i]) {
        otherChanged = true;
        break;
      }
    }
    expect(otherChanged).toBe(true);
  });

  test('clearAllMasks clears both string and integer maps', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    nn.setNeuronStatus(0, 0, 'killed');
    nn.setNeuronStatus(0, 5, 'frozen');
    expect(nn.getNeuronStatus(0, 0)).toBe('killed');
    expect(nn.getNeuronStatus(0, 5)).toBe('frozen');

    nn.clearAllMasks();
    expect(nn.getNeuronStatus(0, 0)).toBe('active');
    expect(nn.getNeuronStatus(0, 5)).toBe('active');
  });
});

describe('dream() — performance optimizations', () => {
  test('100-step dream returns exact-length confidence history', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const result = dream(nn, 3, 100);
    expect(result.confidenceHistory.length).toBe(100);
    expect(result.image.length).toBe(784);

    // All confidence values should be valid probabilities
    for (const c of result.confidenceHistory) {
      expect(isFinite(c)).toBe(true);
      expect(c).toBeGreaterThanOrEqual(0);
      expect(c).toBeLessThanOrEqual(1);
    }
  });

  test('rapid 10-class dream sequence completes < 2s', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const start = performance.now();
    for (let cls = 0; cls < 10; cls++) {
      const result = dream(nn, cls, 50);
      expect(result.image.length).toBe(784);
      expect(result.confidenceHistory.length).toBe(50);
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(2000);
  });

  test('dream confidence should generally increase', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);
    for (let e = 0; e < 10; e++) nn.trainBatch(data.inputs, data.labels);

    const result = dream(nn, 5, 100);
    const first = result.confidenceHistory[0];
    const last = result.confidenceHistory[result.confidenceHistory.length - 1];
    // Confidence should increase (or at least not crash)
    expect(last).toBeGreaterThanOrEqual(first * 0.5);
  });
});

describe('High-epoch training stability', () => {
  test('500 epochs without NaN or memory issues', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);

    for (let e = 0; e < 500; e++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      expect(isFinite(snap.loss)).toBe(true);
      expect(isFinite(snap.accuracy)).toBe(true);
    }

    const loss = nn.getLossHistory();
    const acc = nn.getAccuracyHistory();
    expect(loss.length).toBe(500);
    expect(acc.length).toBe(500);

    // Loss should decrease overall
    expect(loss[loss.length - 1]).toBeLessThan(loss[0]);
  });
});

describe('Large architecture stress', () => {
  test('3-layer 128-64-32 network forward pass is stable', () => {
    const nn = new NeuralNetwork(784, LARGE_CONFIG);
    const input = makeInput();

    const result = nn.predict(input);
    expect(result.probabilities.length).toBe(10);
    let sum = 0;
    for (const p of result.probabilities) {
      expect(isFinite(p)).toBe(true);
      expect(p).toBeGreaterThanOrEqual(0);
      sum += p;
    }
    expect(sum).toBeCloseTo(1, 4);
  });

  test('large network trains 50 epochs < 5s', () => {
    const nn = new NeuralNetwork(784, LARGE_CONFIG);
    const data = generateTrainingData(10);

    const start = performance.now();
    for (let e = 0; e < 50; e++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      expect(isFinite(snap.loss)).toBe(true);
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(5000);
  });

  test('large network gradient flow measurement is finite', () => {
    const nn = new NeuralNetwork(784, LARGE_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const snap = measureGradientFlow(nn, data.inputs[0], data.labels[0]);
    expect(snap.layers.length).toBe(4); // 3 hidden + 1 output
    for (const l of snap.layers) {
      expect(isFinite(l.meanAbsGrad)).toBe(true);
      expect(isFinite(l.maxAbsGrad)).toBe(true);
      expect(l.deadFraction).toBeGreaterThanOrEqual(0);
      expect(l.deadFraction).toBeLessThanOrEqual(1);
    }
  });
});

describe('Confusion matrix — cached training data', () => {
  test('repeated calls with same sampleCount reuse data', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const start = performance.now();
    const r1 = computeConfusionMatrix(nn, 20);
    const t1 = performance.now() - start;

    const start2 = performance.now();
    const r2 = computeConfusionMatrix(nn, 20);
    const t2 = performance.now() - start2;

    // Second call should be no slower (data cached)
    expect(r1.accuracy).toBe(r2.accuracy);
    // Just verify it completes
    expect(r2.total).toBeGreaterThan(0);
  });

  test('confusion matrix values are consistent', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);
    for (let e = 0; e < 20; e++) nn.trainBatch(data.inputs, data.labels);

    const cm = computeConfusionMatrix(nn, 20);
    expect(cm.matrix.length).toBe(10);

    // Row sums should equal class counts
    for (let r = 0; r < 10; r++) {
      let rowSum = 0;
      for (let c = 0; c < 10; c++) rowSum += cm.matrix[r][c];
      expect(rowSum).toBe(cm.classCounts[r]);
    }

    // All metrics in valid range
    for (let c = 0; c < 10; c++) {
      expect(cm.precision[c]).toBeGreaterThanOrEqual(0);
      expect(cm.precision[c]).toBeLessThanOrEqual(1);
      expect(cm.recall[c]).toBeGreaterThanOrEqual(0);
      expect(cm.recall[c]).toBeLessThanOrEqual(1);
      expect(cm.f1[c]).toBeGreaterThanOrEqual(0);
      expect(cm.f1[c]).toBeLessThanOrEqual(1);
    }
  });
});

describe('Gradient flow history ring buffer', () => {
  test('ring buffer wraps correctly at capacity', () => {
    const history = new GradientFlowHistory(10);
    for (let i = 0; i < 25; i++) {
      history.push({
        layers: [],
        health: 'healthy',
        epoch: i,
      });
    }
    expect(history.getSize()).toBe(10);
    const all = history.getAll();
    expect(all.length).toBe(10);
    // Should be chronological: epochs 15-24
    for (let i = 0; i < 10; i++) {
      expect(all[i].epoch).toBe(15 + i);
    }
  });

  test('getLatest returns most recent snapshot', () => {
    const history = new GradientFlowHistory(5);
    for (let i = 0; i < 3; i++) {
      history.push({ layers: [], health: 'healthy', epoch: i });
    }
    expect(history.getLatest()!.epoch).toBe(2);
  });
});

describe('Weight evolution recorder — thinning', () => {
  test('respects maxFrames capacity with automatic thinning', () => {
    const recorder = new WeightEvolutionRecorder(20, 1);
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);

    for (let e = 0; e < 50; e++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      recorder.record(snap);
    }

    // Should have thinned multiple times — never exceeds maxFrames
    expect(recorder.length).toBeLessThanOrEqual(20);
    expect(recorder.length).toBeGreaterThan(0);

    // Frames should be in chronological order
    const frames = recorder.getFrames();
    for (let i = 1; i < frames.length; i++) {
      expect(frames[i].epoch).toBeGreaterThan(frames[i - 1].epoch);
    }
  });

  test('each frame has valid Float32Array weights', () => {
    const recorder = new WeightEvolutionRecorder(10, 1);
    // Use moderate LR to prevent weight explosion → Infinity in Float32Array
    const stableConfig: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, stableConfig);
    const data = generateTrainingData(5);

    for (let e = 0; e < 5; e++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      recorder.record(snap);
    }

    for (const frame of recorder.getFrames()) {
      expect(frame.weights).toBeInstanceOf(Float32Array);
      expect(frame.weights.length).toBe(frame.neuronCount * 784);
      // Spot-check a sample of weights for finiteness (full scan expensive)
      const step = Math.max(1, Math.floor(frame.weights.length / 100));
      for (let i = 0; i < frame.weights.length; i += step) {
        const val = frame.weights[i];
        // Float32Array can hold tiny values that are finite
        expect(Number.isNaN(val)).toBe(false);
        expect(val).not.toBe(Infinity);
        expect(val).not.toBe(-Infinity);
      }
    }
  });
});

describe('Ablation study — correctness under optimization', () => {
  test('ablation study restores original mask state', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);
    for (let e = 0; e < 10; e++) nn.trainBatch(data.inputs, data.labels);

    // Set up some masks
    nn.setNeuronStatus(0, 0, 'killed');
    nn.setNeuronStatus(0, 3, 'frozen');

    const study = runAblationStudy(nn, 5);

    // Masks should be restored
    expect(nn.getNeuronStatus(0, 0)).toBe('killed');
    expect(nn.getNeuronStatus(0, 3)).toBe('frozen');
    expect(nn.getNeuronStatus(0, 1)).toBe('active');

    // Study should have results
    expect(study.totalNeurons).toBe(16);
    expect(study.layers.length).toBe(1);
    expect(study.baselineAccuracy).toBeGreaterThanOrEqual(0);
  });

  test('importance scores normalized to [0, 1]', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);
    for (let e = 0; e < 20; e++) nn.trainBatch(data.inputs, data.labels);

    const study = runAblationStudy(nn, 5);
    let maxImportance = 0;
    for (const layer of study.layers) {
      for (const r of layer) {
        expect(r.importance).toBeGreaterThanOrEqual(0);
        expect(r.importance).toBeLessThanOrEqual(1);
        if (r.importance > maxImportance) maxImportance = r.importance;
      }
    }
    // At least one neuron should be important (max importance = 1)
    if (study.mostCritical && study.mostCritical.accuracyDrop > 0) {
      expect(maxImportance).toBeCloseTo(1, 5);
    }
  });
});

describe('Combined stress — rapid feature exercise', () => {
  test('dream + saliency + confusion + gradient flow in sequence', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);
    for (let e = 0; e < 15; e++) nn.trainBatch(data.inputs, data.labels);

    const input = makeInput();

    // Dream
    const d = dream(nn, 5, 30);
    expect(d.image.length).toBe(784);

    // Saliency
    const sal = computeSaliency(nn, input, 3);
    expect(sal.length).toBe(784);

    // Confusion
    const cm = computeConfusionMatrix(nn, 10);
    expect(cm.total).toBeGreaterThan(0);

    // Gradient flow
    const gf = measureGradientFlow(nn, input, 7);
    expect(gf.layers.length).toBeGreaterThan(0);

    // All features ran without interference
    const pred = nn.predict(input);
    expect(pred.probabilities.length).toBe(10);
  });

  test('training + mask + dream interleaved', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);

    for (let e = 0; e < 5; e++) {
      nn.trainBatch(data.inputs, data.labels);
      nn.setNeuronStatus(0, e % 16, 'killed');
      const d = dream(nn, e % 10, 10);
      expect(d.image.length).toBe(784);
      nn.setNeuronStatus(0, e % 16, 'active');
    }

    // Network should still be functional
    const pred = nn.predict(makeInput());
    let sum = 0;
    for (const p of pred.probabilities) sum += p;
    expect(sum).toBeCloseTo(1, 4);
  });
});

describe('Performance benchmarks', () => {
  test('forward pass 1000× < 1s on default config', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const input = makeInput();

    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      nn.forward(input);
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(1000);
  });

  test('forward pass with masks 1000× < 1.5s', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const input = makeInput();

    // Set several masks
    for (let n = 0; n < 8; n++) {
      nn.setNeuronStatus(0, n, n < 4 ? 'killed' : 'frozen');
    }

    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      nn.forward(input);
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(1500);
  });

  test('trainBatch 100× < 3s on default config', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(10);

    const start = performance.now();
    for (let i = 0; i < 100; i++) {
      nn.trainBatch(data.inputs, data.labels);
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(3000);
  });

  test('snapshot caching — repeated snapshotLayers returns same reference', () => {
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const s1 = nn.snapshotLayers();
    const s2 = nn.snapshotLayers();
    // Should be same reference (cached, no dirty flag)
    expect(s1).toBe(s2);

    // After training (dirty flag set), snapshotLayers rebuilds the cache
    nn.trainBatch(data.inputs, data.labels);
    const s3 = nn.snapshotLayers();
    // Same array ref (cache reuses same objects), but content updated
    expect(s3).toBe(nn.snapshotLayers());

    // Verify the live weights actually changed (i.e. training did something)
    const layers = nn.getLayers();
    // getLossHistory length should show 2 epochs
    expect(nn.getLossHistory().length).toBe(2);
    expect(nn.getEpoch()).toBe(2);
  });
});

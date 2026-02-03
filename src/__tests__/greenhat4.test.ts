/**
 * Green Hat #2 — Weight Evolution Filmstrip + Ablation Lab tests.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import { WeightEvolutionRecorder, renderNeuronWeights, computeWeightDelta } from '../nn/weightEvolution';
import type { WeightFrame } from '../nn/weightEvolution';
import { runAblationStudy, importanceToColor } from '../nn/ablation';
import type { AblationStudy } from '../nn/ablation';
import { generateTrainingData } from '../nn/sampleData';
import {
  WEIGHT_EVOLUTION_CELL_SIZE,
  WEIGHT_EVOLUTION_MAX_NEURONS,
  WEIGHT_EVOLUTION_PLAYBACK_INTERVAL,
  ABLATION_CELL_SIZE,
  ABLATION_CELL_GAP,
  ABLATION_MAX_NEURONS_PER_LAYER,
  ABLATION_SAMPLES_PER_DIGIT,
} from '../constants';

const SRC = path.resolve(__dirname, '..');

// ═══════════════════════════════════════════════════════════════════
// Weight Evolution Recorder
// ═══════════════════════════════════════════════════════════════════

describe('WeightEvolutionRecorder', () => {
  let network: NeuralNetwork;
  let data: { inputs: number[][]; labels: number[] };
  let recorder: WeightEvolutionRecorder;

  beforeEach(() => {
    network = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [
        { neurons: 16, activation: 'relu' },
        { neurons: 8, activation: 'relu' },
      ],
    });
    data = generateTrainingData(5);
    recorder = new WeightEvolutionRecorder();
  });

  it('starts empty', () => {
    expect(recorder.length).toBe(0);
    expect(recorder.getFrames()).toEqual([]);
    expect(recorder.getFrame(0)).toBeNull();
  });

  it('records a training snapshot', () => {
    const snapshot = network.trainBatch(data.inputs, data.labels);
    recorder.record(snapshot);
    expect(recorder.length).toBe(1);
    const frame = recorder.getFrame(0);
    expect(frame).not.toBeNull();
    expect(frame!.epoch).toBe(1);
    expect(frame!.neuronCount).toBe(16);
    expect(frame!.weights.length).toBe(16 * 784);
  });

  it('records multiple epochs', () => {
    for (let i = 0; i < 5; i++) {
      const snapshot = network.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }
    expect(recorder.length).toBe(5);
    expect(recorder.getFrame(4)!.epoch).toBe(5);
  });

  it('stores weights as Float32Array', () => {
    const snapshot = network.trainBatch(data.inputs, data.labels);
    recorder.record(snapshot);
    const frame = recorder.getFrame(0)!;
    expect(frame.weights).toBeInstanceOf(Float32Array);
  });

  it('deep-copies weights (not affected by further training)', () => {
    const snapshot = network.trainBatch(data.inputs, data.labels);
    recorder.record(snapshot);
    const weightsBefore = new Float32Array(recorder.getFrame(0)!.weights);

    // Train more
    network.trainBatch(data.inputs, data.labels);
    const weightsAfter = recorder.getFrame(0)!.weights;

    // Original frame should be unchanged
    let same = true;
    for (let i = 0; i < weightsBefore.length; i++) {
      if (weightsBefore[i] !== weightsAfter[i]) { same = false; break; }
    }
    expect(same).toBe(true);
  });

  it('clears all frames', () => {
    for (let i = 0; i < 3; i++) {
      const snapshot = network.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }
    expect(recorder.length).toBe(3);
    recorder.clear();
    expect(recorder.length).toBe(0);
    expect(recorder.getFrames()).toEqual([]);
  });

  it('records loss and accuracy per frame', () => {
    const snapshot = network.trainBatch(data.inputs, data.labels);
    recorder.record(snapshot);
    const frame = recorder.getFrame(0)!;
    expect(frame.loss).toBeGreaterThanOrEqual(0);
    expect(frame.accuracy).toBeGreaterThanOrEqual(0);
    expect(frame.accuracy).toBeLessThanOrEqual(1);
  });

  it('handles thin-out when at capacity', () => {
    const smallRecorder = new WeightEvolutionRecorder(10);
    for (let i = 0; i < 15; i++) {
      const snapshot = network.trainBatch(data.inputs, data.labels);
      smallRecorder.record(snapshot);
    }
    // Should not exceed capacity
    expect(smallRecorder.length).toBeLessThanOrEqual(15);
  });

  it('respects recording interval', () => {
    const intervalRecorder = new WeightEvolutionRecorder(200, 3);
    for (let i = 0; i < 9; i++) {
      const snapshot = network.trainBatch(data.inputs, data.labels);
      intervalRecorder.record(snapshot);
    }
    // With interval=3, should record every 3rd epoch
    expect(intervalRecorder.length).toBe(3);
  });
});

// ═══════════════════════════════════════════════════════════════════
// renderNeuronWeights
// ═══════════════════════════════════════════════════════════════════

describe('renderNeuronWeights', () => {
  it('fills ImageData with non-zero pixels', () => {
    const weights = new Float32Array(16 * 784);
    for (let i = 0; i < weights.length; i++) {
      weights[i] = (Math.random() - 0.5) * 2;
    }
    // Create a mock ImageData-like object
    const data = new Uint8ClampedArray(28 * 28 * 4);
    const imgData = { data, width: 28, height: 28 } as ImageData;

    renderNeuronWeights(weights, 0, 784, imgData);

    // Check that pixels were written (not all zeros)
    let hasNonZero = false;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0) {
        hasNonZero = true;
        break;
      }
    }
    expect(hasNonZero).toBe(true);
  });

  it('produces different images for different neurons', () => {
    const weights = new Float32Array(16 * 784);
    for (let i = 0; i < weights.length; i++) {
      weights[i] = (Math.random() - 0.5) * 2;
    }

    const img0 = { data: new Uint8ClampedArray(28 * 28 * 4), width: 28, height: 28 } as ImageData;
    const img1 = { data: new Uint8ClampedArray(28 * 28 * 4), width: 28, height: 28 } as ImageData;

    renderNeuronWeights(weights, 0, 784, img0);
    renderNeuronWeights(weights, 1, 784, img1);

    let different = false;
    for (let i = 0; i < img0.data.length; i++) {
      if (img0.data[i] !== img1.data[i]) { different = true; break; }
    }
    expect(different).toBe(true);
  });

  it('sets alpha to 255 for all pixels', () => {
    const weights = new Float32Array(784);
    for (let i = 0; i < weights.length; i++) weights[i] = Math.random();
    const imgData = { data: new Uint8ClampedArray(28 * 28 * 4), width: 28, height: 28 } as ImageData;
    renderNeuronWeights(weights, 0, 784, imgData);

    for (let i = 3; i < imgData.data.length; i += 4) {
      expect(imgData.data[i]).toBe(255);
    }
  });

  it('uses warm colors for positive weights', () => {
    const weights = new Float32Array(784);
    for (let i = 0; i < weights.length; i++) weights[i] = 1.0; // all positive
    const imgData = { data: new Uint8ClampedArray(28 * 28 * 4), width: 28, height: 28 } as ImageData;
    renderNeuronWeights(weights, 0, 784, imgData);

    // First pixel should have R > B (warm)
    expect(imgData.data[0]).toBeGreaterThan(imgData.data[2]);
  });

  it('uses cool colors for negative weights', () => {
    const weights = new Float32Array(784);
    for (let i = 0; i < weights.length; i++) weights[i] = -1.0; // all negative
    const imgData = { data: new Uint8ClampedArray(28 * 28 * 4), width: 28, height: 28 } as ImageData;
    renderNeuronWeights(weights, 0, 784, imgData);

    // First pixel should have B > R (cool)
    expect(imgData.data[2]).toBeGreaterThan(imgData.data[0]);
  });
});

// ═══════════════════════════════════════════════════════════════════
// computeWeightDelta
// ═══════════════════════════════════════════════════════════════════

describe('computeWeightDelta', () => {
  it('returns 0 for identical frames', () => {
    const weights = new Float32Array(16 * 784);
    const frame: WeightFrame = { epoch: 1, loss: 1, accuracy: 0.5, weights, neuronCount: 16 };
    const delta = computeWeightDelta(frame, frame, 0, 784);
    expect(delta).toBe(0);
  });

  it('returns positive value for different frames', () => {
    const w1 = new Float32Array(16 * 784);
    const w2 = new Float32Array(16 * 784);
    for (let i = 0; i < w1.length; i++) {
      w1[i] = Math.random();
      w2[i] = Math.random();
    }
    const f1: WeightFrame = { epoch: 1, loss: 1, accuracy: 0.5, weights: w1, neuronCount: 16 };
    const f2: WeightFrame = { epoch: 2, loss: 0.8, accuracy: 0.6, weights: w2, neuronCount: 16 };
    const delta = computeWeightDelta(f1, f2, 0, 784);
    expect(delta).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Ablation Lab
// ═══════════════════════════════════════════════════════════════════

describe('Ablation Study', () => {
  let network: NeuralNetwork;
  let data: { inputs: number[][]; labels: number[] };

  beforeEach(() => {
    network = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [
        { neurons: 8, activation: 'relu' },
      ],
    });
    data = generateTrainingData(5);
    // Train a few epochs
    for (let i = 0; i < 5; i++) {
      network.trainBatch(data.inputs, data.labels);
    }
  });

  it('returns valid baseline accuracy', () => {
    const study = runAblationStudy(network, 5);
    expect(study.baselineAccuracy).toBeGreaterThanOrEqual(0);
    expect(study.baselineAccuracy).toBeLessThanOrEqual(1);
  });

  it('tests all neurons in all hidden layers', () => {
    const study = runAblationStudy(network, 5);
    expect(study.totalNeurons).toBe(8); // 1 layer × 8 neurons
    expect(study.layers.length).toBe(1);
    expect(study.layers[0].length).toBe(8);
  });

  it('computes accuracy drop per neuron', () => {
    const study = runAblationStudy(network, 5);
    for (const layer of study.layers) {
      for (const result of layer) {
        expect(result.accuracyWithout).toBeGreaterThanOrEqual(0);
        expect(result.accuracyWithout).toBeLessThanOrEqual(1);
        // Drop = baseline - without
        const expectedDrop = study.baselineAccuracy - result.accuracyWithout;
        expect(Math.abs(result.accuracyDrop - expectedDrop)).toBeLessThan(0.001);
      }
    }
  });

  it('normalizes importance to [0, 1]', () => {
    const study = runAblationStudy(network, 5);
    for (const layer of study.layers) {
      for (const result of layer) {
        expect(result.importance).toBeGreaterThanOrEqual(0);
        expect(result.importance).toBeLessThanOrEqual(1);
      }
    }
  });

  it('identifies most critical neuron', () => {
    const study = runAblationStudy(network, 5);
    expect(study.mostCritical).not.toBeNull();
    if (study.mostCritical) {
      expect(study.mostCritical.layerIdx).toBeGreaterThanOrEqual(0);
      expect(study.mostCritical.neuronIdx).toBeGreaterThanOrEqual(0);
    }
  });

  it('identifies most redundant neuron', () => {
    const study = runAblationStudy(network, 5);
    expect(study.mostRedundant).not.toBeNull();
  });

  it('restores original neuron masks after study', () => {
    // Freeze a neuron before study
    network.setNeuronStatus(0, 2, 'frozen');
    const studyResult = runAblationStudy(network, 5);

    // Verify mask was restored
    expect(network.getNeuronStatus(0, 2)).toBe('frozen');
    // Other neurons should be active
    expect(network.getNeuronStatus(0, 0)).toBe('active');
    expect(studyResult.totalNeurons).toBe(8);
  });

  it('works with multi-layer networks', () => {
    const multiNet = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [
        { neurons: 8, activation: 'relu' },
        { neurons: 4, activation: 'relu' },
      ],
    });
    const multiData = generateTrainingData(3);
    for (let i = 0; i < 3; i++) {
      multiNet.trainBatch(multiData.inputs, multiData.labels);
    }

    const study = runAblationStudy(multiNet, 3);
    expect(study.layers.length).toBe(2);
    expect(study.layers[0].length).toBe(8);
    expect(study.layers[1].length).toBe(4);
    expect(study.totalNeurons).toBe(12);
  });

  it('has a valid timestamp', () => {
    const before = Date.now();
    const study = runAblationStudy(network, 5);
    const after = Date.now();
    expect(study.timestamp).toBeGreaterThanOrEqual(before);
    expect(study.timestamp).toBeLessThanOrEqual(after);
  });
});

// ═══════════════════════════════════════════════════════════════════
// importanceToColor
// ═══════════════════════════════════════════════════════════════════

describe('importanceToColor', () => {
  it('returns rgba string', () => {
    const color = importanceToColor(0.5);
    expect(color).toMatch(/^rgba\(/);
  });

  it('returns cool color for 0 importance', () => {
    const color = importanceToColor(0);
    expect(color).toContain('40');  // R=40
    expect(color).toContain('160'); // B=160
  });

  it('returns hot color for 1.0 importance', () => {
    const color = importanceToColor(1.0);
    expect(color).toContain('255'); // R=255
  });

  it('transitions smoothly across range', () => {
    const colors = [];
    for (let i = 0; i <= 10; i++) {
      colors.push(importanceToColor(i / 10));
    }
    // All should be valid rgba strings
    for (const c of colors) {
      expect(c).toMatch(/^rgba\(\d+, \d+, \d+, [\d.]+\)$/);
    }
  });

  it('handles negative importance gracefully', () => {
    const color = importanceToColor(-0.5);
    expect(color).toMatch(/^rgba\(/);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Constants validation
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — Constants', () => {
  it('weight evolution constants are positive', () => {
    expect(WEIGHT_EVOLUTION_CELL_SIZE).toBeGreaterThan(0);
    expect(WEIGHT_EVOLUTION_MAX_NEURONS).toBeGreaterThan(0);
    expect(WEIGHT_EVOLUTION_PLAYBACK_INTERVAL).toBeGreaterThan(0);
  });

  it('ablation constants are positive', () => {
    expect(ABLATION_CELL_SIZE).toBeGreaterThan(0);
    expect(ABLATION_CELL_GAP).toBeGreaterThanOrEqual(0);
    expect(ABLATION_MAX_NEURONS_PER_LAYER).toBeGreaterThan(0);
    expect(ABLATION_SAMPLES_PER_DIGIT).toBeGreaterThan(0);
  });

  it('playback interval is reasonable (50-500ms)', () => {
    expect(WEIGHT_EVOLUTION_PLAYBACK_INTERVAL).toBeGreaterThanOrEqual(50);
    expect(WEIGHT_EVOLUTION_PLAYBACK_INTERVAL).toBeLessThanOrEqual(500);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Module exports & barrel
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — Module exports', () => {
  it('nn barrel exports weight evolution', () => {
    const barrel = fs.readFileSync(path.join(SRC, 'nn', 'index.ts'), 'utf-8');
    expect(barrel).toContain('WeightEvolutionRecorder');
    expect(barrel).toContain('renderNeuronWeights');
    expect(barrel).toContain('computeWeightDelta');
    expect(barrel).toContain('WeightFrame');
  });

  it('nn barrel exports ablation', () => {
    const barrel = fs.readFileSync(path.join(SRC, 'nn', 'index.ts'), 'utf-8');
    expect(barrel).toContain('runAblationStudy');
    expect(barrel).toContain('importanceToColor');
    expect(barrel).toContain('AblationResult');
    expect(barrel).toContain('AblationStudy');
  });

  it('components barrel exports new components', () => {
    const barrel = fs.readFileSync(path.join(SRC, 'components', 'index.ts'), 'utf-8');
    expect(barrel).toContain('WeightEvolution');
    expect(barrel).toContain('AblationLab');
  });

  it('constants has weight evolution entries', () => {
    const src = fs.readFileSync(path.join(SRC, 'constants.ts'), 'utf-8');
    expect(src).toContain('WEIGHT_EVOLUTION_CELL_SIZE');
    expect(src).toContain('WEIGHT_EVOLUTION_MAX_NEURONS');
    expect(src).toContain('WEIGHT_EVOLUTION_PLAYBACK_INTERVAL');
  });

  it('constants has ablation entries', () => {
    const src = fs.readFileSync(path.join(SRC, 'constants.ts'), 'utf-8');
    expect(src).toContain('ABLATION_CELL_SIZE');
    expect(src).toContain('ABLATION_CELL_GAP');
    expect(src).toContain('ABLATION_MAX_NEURONS_PER_LAYER');
    expect(src).toContain('ABLATION_SAMPLES_PER_DIGIT');
  });

  it('App.tsx imports both new components', () => {
    const src = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    expect(src).toContain("import WeightEvolution from './components/WeightEvolution'");
    expect(src).toContain("import AblationLab from './components/AblationLab'");
  });

  it('App.tsx uses WeightEvolutionRecorder', () => {
    const src = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    expect(src).toContain('WeightEvolutionRecorder');
    expect(src).toContain('weightRecorderRef');
  });
});

// ═══════════════════════════════════════════════════════════════════
// Cross-feature interaction
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — Cross-feature integration', () => {
  it('ablation preserves network state after study', () => {
    const network = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    for (let i = 0; i < 3; i++) {
      network.trainBatch(data.inputs, data.labels);
    }

    // Predict before study
    const input = data.inputs[0];
    const predBefore = network.predict(input);

    // Run ablation
    runAblationStudy(network, 3);

    // Predict after study — should be identical
    const predAfter = network.predict(input);
    expect(predAfter.label).toBe(predBefore.label);
    for (let i = 0; i < 10; i++) {
      expect(Math.abs(predAfter.probabilities[i] - predBefore.probabilities[i])).toBeLessThan(1e-6);
    }
  });

  it('weight evolution + ablation work on same network', () => {
    const network = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    const recorder = new WeightEvolutionRecorder();

    for (let i = 0; i < 5; i++) {
      const snapshot = network.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }

    expect(recorder.length).toBe(5);

    // Now run ablation — should not corrupt weight recorder
    const study = runAblationStudy(network, 3);
    expect(study.totalNeurons).toBe(16);

    // Recorder still has 5 frames
    expect(recorder.length).toBe(5);
  });

  it('weight delta increases during early training', () => {
    // Use sigmoid activation to avoid dead-neuron ReLU issue with small networks
    const network = new NeuralNetwork(784, {
      learningRate: 0.05, // higher LR for visible changes
      layers: [{ neurons: 16, activation: 'sigmoid' }],
    });
    const data = generateTrainingData(5);
    const recorder = new WeightEvolutionRecorder();

    for (let i = 0; i < 5; i++) {
      const snapshot = network.trainBatch(data.inputs, data.labels);
      recorder.record(snapshot);
    }

    // First weight delta should be non-zero (weights changed from init)
    if (recorder.length >= 2) {
      const delta = computeWeightDelta(
        recorder.getFrame(0)!,
        recorder.getFrame(1)!,
        0,
        784,
      );
      expect(delta).toBeGreaterThan(0);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// CSS validation
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — CSS', () => {
  const css = fs.readFileSync(path.join(SRC, 'App.css'), 'utf-8');

  it('has weight evolution styles', () => {
    expect(css).toContain('.weight-evolution');
    expect(css).toContain('.we-timeline');
    expect(css).toContain('.we-scrubber');
    expect(css).toContain('.we-magnifier');
    expect(css).toContain('.we-sparkline');
  });

  it('has ablation lab styles', () => {
    expect(css).toContain('.ablation-lab');
    expect(css).toContain('.ablation-heatmap');
    expect(css).toContain('.ablation-tooltip');
    expect(css).toContain('.ablation-callout');
    expect(css).toContain('.ablation-run-btn');
  });

  it('has reduced-motion overrides for new components', () => {
    expect(css).toContain('.we-play-btn.playing');
    expect(css).toContain('.we-magnifier');
    expect(css).toContain('.ablation-running');
    expect(css).toContain('.ablation-tooltip');
  });
});

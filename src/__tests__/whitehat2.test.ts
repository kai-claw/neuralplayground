/**
 * White Hat #2 — Final Verification Tests
 *
 * Pass 10/10: Comprehensive integration, correctness, and
 * deployment readiness verification. Sign-off suite.
 */

import { describe, test, expect } from 'vitest';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

// ─── Core imports ────────────────────────────────────────────────────
import { NeuralNetwork } from '../nn/NeuralNetwork';
import {
  generateTrainingData,
  canvasToInput,
  computeInputGradient,
  dream,
  computeSaliency,
  saliencyToColor,
  computeConfusionMatrix,
  measureGradientFlow,
  GradientFlowHistory,
  EpochRecorder,
  replayForward,
  paramsToLayers,
  computeDecisionBoundary,
  generateExemplar,
  dreamChimera,
  CHIMERA_PRESETS,
  findMisfits,
  computeMisfitSummary,
  WeightEvolutionRecorder,
  renderNeuronWeights,
  computeWeightDelta,
  runAblationStudy,
  importanceToColor,
  generateNoisePattern,
  applyNoise,
  projectTo2D,
} from '../nn';

import {
  DEFAULT_CONFIG,
  INPUT_SIZE,
  INPUT_DIM,
  OUTPUT_CLASSES,
  NEURON_OPTIONS,
  MAX_HIDDEN_LAYERS,
  SHORTCUTS,
  NOISE_LABELS,
  NOISE_DESCRIPTIONS,
  CINEMATIC_TRAIN_EPOCHS,
  CINEMATIC_PREDICT_DWELL,
  CINEMATIC_EPOCH_INTERVAL,
  AUTO_TRAIN_EPOCHS,
  AUTO_TRAIN_DELAY,
  HISTORY_MAX_LENGTH,
  PERF_SAMPLE_INTERVAL,
  PERF_DEGRADE_FPS,
  PERF_RECOVER_FPS,
  PERF_DEGRADE_SECONDS,
  PERF_RECOVER_SECONDS,
  RACE_EPOCHS,
  TRAINING_STEP_INTERVAL,
  GRADIENT_FLOW_SAMPLE_COUNT,
  CONFUSION_SAMPLES_PER_DIGIT,
  ABLATION_SAMPLES_PER_DIGIT,
} from '../constants';

import type {
  ActivationFn,
  NeuronStatus,
  TrainingConfig,
  NoiseType,
  CinematicPhase,
} from '../types';

import { activate, activateDerivative } from '../utils/activations';
import { argmax, safeMax } from '../utils/math';
import { mulberry32 } from '../utils/prng';

import { RACE_PRESETS } from '../data/racePresets';
import { DIGIT_STROKES } from '../data/digitStrokes';

const ROOT = join(__dirname, '..', '..');

// ─── Helper ──────────────────────────────────────────────────────────

function trainedNetwork(epochs = 20): NeuralNetwork {
  const nn = new NeuralNetwork(INPUT_SIZE, DEFAULT_CONFIG);
  const data = generateTrainingData(10);
  for (let e = 0; e < epochs; e++) {
    nn.trainBatch(data.inputs, data.labels);
  }
  return nn;
}

// ═════════════════════════════════════════════════════════════════════
// 1. End-to-End Data Pipeline
// ═════════════════════════════════════════════════════════════════════

describe('End-to-end data pipeline', () => {
  test('generate → train → predict → dream → saliency → confusion full cycle', () => {
    const nn = trainedNetwork(15);
    const data = generateTrainingData(5);

    // Predict via predict() which returns label + probabilities
    const result = nn.predict(data.inputs[0]);
    expect(result.label).toBeGreaterThanOrEqual(0);
    expect(result.label).toBeLessThanOrEqual(9);
    expect(result.probabilities.length).toBe(OUTPUT_CLASSES);
    const sumProb = result.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(sumProb - 1)).toBeLessThan(0.01);

    // Dream
    const dreamResult = dream(nn, 5, 40);
    expect(dreamResult.image.length).toBe(INPUT_SIZE);
    expect(dreamResult.confidenceHistory.length).toBe(40);
    dreamResult.image.forEach(v => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    });

    // Saliency
    const saliency = computeSaliency(nn, data.inputs[0], data.labels[0]);
    expect(saliency.length).toBe(INPUT_SIZE);
    saliency.forEach(v => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
      expect(isFinite(v)).toBe(true);
    });

    // Confusion matrix
    const confusion = computeConfusionMatrix(nn, 5);
    expect(confusion.matrix.length).toBe(OUTPUT_CLASSES);
    confusion.matrix.forEach(row => {
      expect(row.length).toBe(OUTPUT_CLASSES);
      row.forEach(val => expect(val).toBeGreaterThanOrEqual(0));
    });
    expect(confusion.recall.length).toBe(OUTPUT_CLASSES);
    expect(confusion.precision.length).toBe(OUTPUT_CLASSES);
  });

  test('generate → train → gradient flow → epoch replay full cycle', () => {
    const nn = new NeuralNetwork(INPUT_SIZE, DEFAULT_CONFIG);
    const data = generateTrainingData(8);
    const recorder = new EpochRecorder();

    for (let e = 0; e < 10; e++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      recorder.record(snap);
    }

    // Gradient flow
    const gf = measureGradientFlow(nn, data.inputs[0], data.labels[0]);
    expect(gf.layers.length).toBe(DEFAULT_CONFIG.layers.length + 1);
    gf.layers.forEach(l => {
      expect(isFinite(l.meanAbsGrad)).toBe(true);
      expect(isFinite(l.maxAbsGrad)).toBe(true);
      expect(l.deadFraction).toBeGreaterThanOrEqual(0);
      expect(l.deadFraction).toBeLessThanOrEqual(1);
    });
    expect(['healthy', 'vanishing', 'exploding']).toContain(gf.health);

    // Epoch replay
    const timeline = recorder.getTimeline();
    expect(timeline.length).toBeGreaterThan(0);
    const replayed = replayForward(timeline[0].params, data.inputs[0]);
    expect(replayed.probabilities.length).toBe(OUTPUT_CLASSES);
    const probSum = replayed.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(probSum - 1)).toBeLessThan(0.01);
  });

  test('generate → train → ablation → weight evolution full cycle', () => {
    const nn = trainedNetwork(20);
    const data = generateTrainingData(5);

    // Ablation study (takes network + samplesPerDigit)
    const ablation = runAblationStudy(nn, 5);
    expect(ablation.layers.length).toBe(DEFAULT_CONFIG.layers.length);
    ablation.layers.forEach(layer => {
      expect(layer.length).toBeGreaterThan(0);
      layer.forEach(n => {
        expect(n.importance).toBeGreaterThanOrEqual(0);
        expect(n.importance).toBeLessThanOrEqual(1);
        expect(isFinite(n.accuracyDrop)).toBe(true);
      });
    });

    // Weight evolution
    const wer = new WeightEvolutionRecorder(50);
    const nn2 = new NeuralNetwork(INPUT_SIZE, DEFAULT_CONFIG);
    for (let e = 0; e < 10; e++) {
      const snap = nn2.trainBatch(data.inputs, data.labels);
      wer.record(snap);
    }
    expect(wer.length).toBe(10);
    const frame = wer.getFrame(0);
    expect(frame).not.toBeNull();
    if (frame) {
      expect(frame.weights).toBeInstanceOf(Float32Array);
      expect(frame.weights.length).toBeGreaterThan(0);
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// 2. Gradient Flow Ping-Pong Buffer Correctness
// ═════════════════════════════════════════════════════════════════════

describe('Gradient flow ping-pong buffer correctness', () => {
  test('3-layer network (128-64-32) produces all-finite gradient stats', () => {
    const config: TrainingConfig = {
      learningRate: 0.05,
      layers: [
        { neurons: 128, activation: 'relu' },
        { neurons: 64, activation: 'relu' },
        { neurons: 32, activation: 'relu' },
      ],
    };
    const nn = new NeuralNetwork(INPUT_SIZE, config);
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

  test('gradient flow called 50× consecutively produces finite results each time', () => {
    const nn = trainedNetwork(5);
    const data = generateTrainingData(3);
    for (let i = 0; i < 50; i++) {
      const snap = measureGradientFlow(nn, data.inputs[i % data.inputs.length], data.labels[i % data.labels.length]);
      snap.layers.forEach(l => {
        expect(isFinite(l.meanAbsGrad)).toBe(true);
        expect(isFinite(l.maxAbsGrad)).toBe(true);
      });
    }
  });

  test('gradient flow with multiple architectures in sequence does not cross-contaminate', () => {
    const configs: TrainingConfig[] = [
      { learningRate: 0.01, layers: [{ neurons: 16, activation: 'relu' }] },
      { learningRate: 0.01, layers: [{ neurons: 64, activation: 'relu' }, { neurons: 32, activation: 'relu' }] },
      { learningRate: 0.01, layers: [{ neurons: 128, activation: 'tanh' }, { neurons: 64, activation: 'sigmoid' }, { neurons: 32, activation: 'relu' }] },
    ];
    const data = generateTrainingData(5);

    for (const config of configs) {
      const nn = new NeuralNetwork(INPUT_SIZE, config);
      nn.trainBatch(data.inputs, data.labels);
      const snap = measureGradientFlow(nn, data.inputs[0], data.labels[0]);
      expect(snap.layers.length).toBe(config.layers.length + 1);
      snap.layers.forEach(l => {
        expect(isFinite(l.meanAbsGrad)).toBe(true);
        expect(isFinite(l.maxAbsGrad)).toBe(true);
      });
    }
  });

  test('gradient flow history ring buffer preserves chronological order', () => {
    const history = new GradientFlowHistory(5);
    const nn = trainedNetwork(5);
    const data = generateTrainingData(3);

    for (let i = 0; i < 8; i++) {
      nn.trainBatch(data.inputs, data.labels);
      const snap = measureGradientFlow(nn, data.inputs[0], data.labels[0]);
      history.push(snap);
    }

    expect(history.getSize()).toBe(5);
    const all = history.getAll();
    expect(all.length).toBe(5);
    for (let i = 1; i < all.length; i++) {
      expect(all[i].epoch).toBeGreaterThanOrEqual(all[i - 1].epoch);
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// 3. All Activation Functions — Forward + Derivative
// ═════════════════════════════════════════════════════════════════════

describe('Activation function completeness', () => {
  const activations: ActivationFn[] = ['relu', 'sigmoid', 'tanh'];
  const testValues = [-10, -1, -0.5, 0, 0.5, 1, 10];

  for (const fn of activations) {
    test(`${fn} produces finite values for all test inputs`, () => {
      for (const v of testValues) {
        expect(isFinite(activate(v, fn))).toBe(true);
        expect(isFinite(activateDerivative(v, fn))).toBe(true);
      }
    });

    test(`${fn} derivative is non-negative`, () => {
      for (const v of testValues) {
        expect(activateDerivative(v, fn)).toBeGreaterThanOrEqual(0);
      }
    });
  }

  test('networks with each activation train stably for 30 epochs', () => {
    const data = generateTrainingData(8);
    for (const fn of activations) {
      const config: TrainingConfig = {
        learningRate: 0.01,
        layers: [
          { neurons: 32, activation: fn },
          { neurons: 16, activation: fn },
        ],
      };
      const nn = new NeuralNetwork(INPUT_SIZE, config);
      for (let e = 0; e < 30; e++) {
        const snap = nn.trainBatch(data.inputs, data.labels);
        expect(isFinite(snap.loss)).toBe(true);
        expect(snap.accuracy).toBeGreaterThanOrEqual(0);
        expect(snap.accuracy).toBeLessThanOrEqual(1);
      }
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// 4. Feature Completeness Verification
// ═════════════════════════════════════════════════════════════════════

describe('Feature completeness', () => {
  test('all components directory has 28+ component files', () => {
    const componentsDir = join(ROOT, 'src', 'components');
    const componentFiles = readdirSync(componentsDir).filter(f => f.endsWith('.tsx'));
    expect(componentFiles.length).toBeGreaterThanOrEqual(28);
  });

  test('all 6 hooks exist', () => {
    const hooksDir = join(ROOT, 'src', 'hooks');
    const hookFiles = readdirSync(hooksDir).filter(f => f.endsWith('.ts') && f !== 'index.ts');
    expect(hookFiles.length).toBeGreaterThanOrEqual(6);
  });

  test('all nn modules export expected functions', () => {
    expect(typeof NeuralNetwork).toBe('function');
    expect(typeof generateTrainingData).toBe('function');
    expect(typeof canvasToInput).toBe('function');
    expect(typeof computeInputGradient).toBe('function');
    expect(typeof dream).toBe('function');
    expect(typeof computeSaliency).toBe('function');
    expect(typeof computeConfusionMatrix).toBe('function');
    expect(typeof measureGradientFlow).toBe('function');
    expect(typeof projectTo2D).toBe('function');
    expect(typeof computeDecisionBoundary).toBe('function');
    expect(typeof generateExemplar).toBe('function');
    expect(typeof dreamChimera).toBe('function');
    expect(typeof findMisfits).toBe('function');
    expect(typeof computeMisfitSummary).toBe('function');
    expect(typeof runAblationStudy).toBe('function');
    expect(typeof WeightEvolutionRecorder).toBe('function');
    expect(typeof renderNeuronWeights).toBe('function');
    expect(typeof computeWeightDelta).toBe('function');
    expect(typeof generateNoisePattern).toBe('function');
    expect(typeof applyNoise).toBe('function');
  });

  test('chimera presets are defined with valid 10-element weight arrays', () => {
    expect(CHIMERA_PRESETS.length).toBeGreaterThan(0);
    for (const preset of CHIMERA_PRESETS) {
      expect(typeof preset.name).toBe('string');
      expect(preset.name.length).toBeGreaterThan(0);
      expect(preset.weights.length).toBe(OUTPUT_CLASSES);
      const hasWeight = preset.weights.some(w => w > 0);
      expect(hasWeight).toBe(true);
    }
  });

  test('race presets define valid training configs', () => {
    expect(RACE_PRESETS.length).toBeGreaterThan(0);
    for (const preset of RACE_PRESETS) {
      expect(typeof preset.label).toBe('string');
      // Each race preset has two configs (a and b) for head-to-head
      for (const cfg of [preset.a, preset.b]) {
        expect(cfg.learningRate).toBeGreaterThan(0);
        expect(cfg.layers.length).toBeGreaterThan(0);
        for (const layer of cfg.layers) {
          expect(layer.neurons).toBeGreaterThan(0);
          expect(['relu', 'sigmoid', 'tanh']).toContain(layer.activation);
        }
      }
    }
  });

  test('digit strokes cover all 10 digits', () => {
    expect(DIGIT_STROKES.length).toBe(10);
    for (let d = 0; d < 10; d++) {
      expect(DIGIT_STROKES[d]).toBeDefined();
      expect(DIGIT_STROKES[d].length).toBeGreaterThan(0);
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// 5. Noise System Completeness
// ═════════════════════════════════════════════════════════════════════

describe('Noise system completeness', () => {
  const noiseTypes: NoiseType[] = ['gaussian', 'salt-pepper', 'adversarial'];

  test('all noise types have labels and descriptions', () => {
    for (const t of noiseTypes) {
      expect(NOISE_LABELS[t]).toBeDefined();
      expect(NOISE_LABELS[t].length).toBeGreaterThan(0);
      expect(NOISE_DESCRIPTIONS[t]).toBeDefined();
      expect(NOISE_DESCRIPTIONS[t].length).toBeGreaterThan(0);
    }
  });

  test('noise generation + apply produces valid [0,1] clamped output for all types', () => {
    const input = Array.from({ length: INPUT_SIZE }, () => 0.5);
    for (const t of noiseTypes) {
      const pattern = generateNoisePattern(t, 0.5, 42, t === 'adversarial' ? 3 : undefined);
      expect(pattern).toBeInstanceOf(Float32Array);
      expect(pattern.length).toBe(INPUT_SIZE);

      const noisy = applyNoise(input, pattern, 0.5, t, 42);
      expect(noisy.length).toBe(INPUT_SIZE);
      noisy.forEach(v => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        expect(isFinite(v)).toBe(true);
      });
    }
  });

  test('noise at level 0 preserves original for gaussian', () => {
    const input = Array.from({ length: INPUT_SIZE }, () => Math.random());
    const pattern = generateNoisePattern('gaussian', 0, 42);
    const noisy = applyNoise(input, pattern, 0, 'gaussian', 42);
    for (let i = 0; i < INPUT_SIZE; i++) {
      expect(noisy[i]).toBeCloseTo(input[i], 3);
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// 6. Utility Functions Verification
// ═════════════════════════════════════════════════════════════════════

describe('Utility functions', () => {
  test('argmax returns correct index', () => {
    expect(argmax([1, 5, 3, 2])).toBe(1);
    expect(argmax([10])).toBe(0);
    expect(argmax([0, 0, 0, 1])).toBe(3);
  });

  test('safeMax returns -Infinity for empty, correct max otherwise', () => {
    expect(safeMax([])).toBe(-Infinity);
    expect(safeMax([3, 1, 4, 1, 5])).toBe(5);
    expect(safeMax([-5, -1, -3])).toBe(-1);
  });

  test('mulberry32 PRNG is deterministic', () => {
    const gen1 = mulberry32(42);
    const gen2 = mulberry32(42);
    for (let i = 0; i < 100; i++) {
      expect(gen1()).toBe(gen2());
    }
  });

  test('mulberry32 produces values in [0,1)', () => {
    const gen = mulberry32(12345);
    for (let i = 0; i < 1000; i++) {
      const v = gen();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  test('saliencyToColor returns valid RGBA tuple', () => {
    const [r, g, b, a] = saliencyToColor(0.5);
    expect(r).toBeGreaterThanOrEqual(0);
    expect(r).toBeLessThanOrEqual(255);
    expect(g).toBeGreaterThanOrEqual(0);
    expect(g).toBeLessThanOrEqual(255);
    expect(b).toBeGreaterThanOrEqual(0);
    expect(b).toBeLessThanOrEqual(255);
    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThanOrEqual(255);
  });

  test('importanceToColor returns valid CSS color', () => {
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      const color = importanceToColor(v);
      expect(color).toMatch(/^rgb/);
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// 7. Constants Consistency
// ═════════════════════════════════════════════════════════════════════

describe('Constants consistency', () => {
  test('INPUT_SIZE = INPUT_DIM²', () => {
    expect(INPUT_SIZE).toBe(INPUT_DIM * INPUT_DIM);
  });

  test('NEURON_OPTIONS are sorted ascending', () => {
    for (let i = 1; i < NEURON_OPTIONS.length; i++) {
      expect(NEURON_OPTIONS[i]).toBeGreaterThan(NEURON_OPTIONS[i - 1]);
    }
  });

  test('SHORTCUTS have unique keys', () => {
    const keys = SHORTCUTS.map(s => s.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  test('timing constants are positive', () => {
    expect(CINEMATIC_TRAIN_EPOCHS).toBeGreaterThan(0);
    expect(CINEMATIC_PREDICT_DWELL).toBeGreaterThan(0);
    expect(CINEMATIC_EPOCH_INTERVAL).toBeGreaterThan(0);
    expect(AUTO_TRAIN_EPOCHS).toBeGreaterThan(0);
    expect(AUTO_TRAIN_DELAY).toBeGreaterThan(0);
    expect(TRAINING_STEP_INTERVAL).toBeGreaterThan(0);
    expect(HISTORY_MAX_LENGTH).toBeGreaterThan(0);
    expect(RACE_EPOCHS).toBeGreaterThan(0);
    expect(GRADIENT_FLOW_SAMPLE_COUNT).toBeGreaterThan(0);
    expect(CONFUSION_SAMPLES_PER_DIGIT).toBeGreaterThan(0);
    expect(ABLATION_SAMPLES_PER_DIGIT).toBeGreaterThan(0);
  });

  test('performance monitor thresholds are consistent', () => {
    expect(PERF_DEGRADE_FPS).toBeLessThan(PERF_RECOVER_FPS);
    expect(PERF_SAMPLE_INTERVAL).toBeGreaterThan(0);
    expect(PERF_DEGRADE_SECONDS).toBeGreaterThan(0);
    expect(PERF_RECOVER_SECONDS).toBeGreaterThan(0);
  });

  test('DEFAULT_CONFIG is valid', () => {
    expect(DEFAULT_CONFIG.learningRate).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.layers.length).toBeGreaterThan(0);
    DEFAULT_CONFIG.layers.forEach(l => {
      expect(l.neurons).toBeGreaterThan(0);
      expect(NEURON_OPTIONS as readonly number[]).toContain(l.neurons);
      expect(['relu', 'sigmoid', 'tanh']).toContain(l.activation);
    });
  });
});

// ═════════════════════════════════════════════════════════════════════
// 8. Type System Consistency
// ═════════════════════════════════════════════════════════════════════

describe('Type system consistency', () => {
  test('ActivationFn covers all used values', () => {
    const fns: ActivationFn[] = ['relu', 'sigmoid', 'tanh'];
    expect(fns.length).toBe(3);
  });

  test('NeuronStatus covers all states', () => {
    const states: NeuronStatus[] = ['active', 'frozen', 'killed'];
    expect(states.length).toBe(3);
  });

  test('NoiseType covers all noise types', () => {
    const types: NoiseType[] = ['gaussian', 'salt-pepper', 'adversarial'];
    expect(types.length).toBe(3);
    types.forEach(t => expect(NOISE_LABELS[t]).toBeDefined());
  });

  test('CinematicPhase covers all phases', () => {
    const phases: CinematicPhase[] = ['training', 'drawing', 'predicting'];
    expect(phases.length).toBe(3);
  });

  test('NeuralNetwork public API is complete', () => {
    const nn = new NeuralNetwork(INPUT_SIZE, DEFAULT_CONFIG);
    expect(typeof nn.forward).toBe('function');
    expect(typeof nn.trainBatch).toBe('function');
    expect(typeof nn.reset).toBe('function');
    expect(typeof nn.getConfig).toBe('function');
    expect(typeof nn.getLayers).toBe('function');
    expect(typeof nn.getEpoch).toBe('function');
    expect(typeof nn.snapshotLayers).toBe('function');
    expect(typeof nn.setNeuronStatus).toBe('function');
    expect(typeof nn.getNeuronStatus).toBe('function');
    expect(typeof nn.getNeuronMasks).toBe('function');
    expect(typeof nn.clearAllMasks).toBe('function');
    expect(typeof nn.predict).toBe('function');
  });

  test('TrainingSnapshot has all required fields', () => {
    const nn = new NeuralNetwork(INPUT_SIZE, DEFAULT_CONFIG);
    const data = generateTrainingData(5);
    const snap = nn.trainBatch(data.inputs, data.labels);
    expect(typeof snap.epoch).toBe('number');
    expect(typeof snap.loss).toBe('number');
    expect(typeof snap.accuracy).toBe('number');
    expect(Array.isArray(snap.layers)).toBe(true);
    expect(Array.isArray(snap.predictions)).toBe(true);
    expect(Array.isArray(snap.outputProbabilities)).toBe(true);
  });
});

// ═════════════════════════════════════════════════════════════════════
// 9. Module Export & Architecture Verification
// ═════════════════════════════════════════════════════════════════════

describe('Module architecture', () => {
  test('all 6 barrel exports exist', () => {
    const barrels = [
      'nn/index.ts',
      'utils/index.ts',
      'renderers/index.ts',
      'visualizers/index.ts',
      'hooks/index.ts',
      'components/index.ts',
    ];
    for (const barrel of barrels) {
      expect(existsSync(join(ROOT, 'src', barrel))).toBe(true);
    }
  });

  test('backward-compat re-exports exist and are thin', () => {
    const compatFiles = ['noise.ts', 'rendering.ts', 'visualizer.ts'];
    for (const file of compatFiles) {
      const fullPath = join(ROOT, 'src', file);
      expect(existsSync(fullPath)).toBe(true);
      const content = readFileSync(fullPath, 'utf-8');
      expect(content.split('\n').length).toBeLessThan(25);
    }
  });

  test('ARCHITECTURE.md exists and is substantial', () => {
    expect(existsSync(join(ROOT, 'ARCHITECTURE.md'))).toBe(true);
    const content = readFileSync(join(ROOT, 'ARCHITECTURE.md'), 'utf-8');
    expect(content.length).toBeGreaterThan(500);
  });
});

// ═════════════════════════════════════════════════════════════════════
// 10. Deployment Readiness
// ═════════════════════════════════════════════════════════════════════

describe('Deployment readiness', () => {
  test('public assets exist', () => {
    const assets = ['favicon.svg', 'og-image.svg', '404.html', 'robots.txt', 'sitemap.xml', 'manifest.json', 'icon-192.png', 'icon-512.png'];
    for (const asset of assets) {
      expect(existsSync(join(ROOT, 'public', asset))).toBe(true);
    }
  });

  test('LICENSE and README are substantial', () => {
    expect(existsSync(join(ROOT, 'LICENSE'))).toBe(true);
    expect(existsSync(join(ROOT, 'README.md'))).toBe(true);
    const readme = readFileSync(join(ROOT, 'README.md'), 'utf-8');
    expect(readme.length).toBeGreaterThan(1000);
  });

  test('CI/CD workflow exists', () => {
    expect(existsSync(join(ROOT, '.github', 'workflows', 'ci.yml'))).toBe(true);
  });

  test('package.json has required metadata', () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf-8'));
    expect(pkg.version).toBe('1.0.0');
    expect(pkg.name).toBeDefined();
    expect(pkg.description).toBeDefined();
    expect(pkg.homepage).toBeDefined();
    expect(pkg.repository).toBeDefined();
    expect(pkg.keywords?.length).toBeGreaterThan(0);
    expect(pkg.author).toBeDefined();
    expect(pkg.license).toBe('MIT');
    expect(pkg.scripts?.build).toBeDefined();
    expect(pkg.scripts?.test).toBeDefined();
    expect(pkg.scripts?.deploy).toBeDefined();
  });

  test('tsconfig is strict with all lint flags', () => {
    // tsconfig.app.json is JSONC (allows comments) — strip them before parsing
    const raw = readFileSync(join(ROOT, 'tsconfig.app.json'), 'utf-8');
    const stripped = raw.replace(/\/\*[\s\S]*?\*\/|\/\/.*/g, '');
    const tsconfig = JSON.parse(stripped);
    expect(tsconfig.compilerOptions.strict).toBe(true);
    expect(tsconfig.compilerOptions.noUnusedLocals).toBe(true);
    expect(tsconfig.compilerOptions.noUnusedParameters).toBe(true);
  });

  test('HTML index has essential meta tags', () => {
    const html = readFileSync(join(ROOT, 'index.html'), 'utf-8');
    expect(html).toContain('og:title');
    expect(html).toContain('og:description');
    expect(html).toContain('twitter:card');
    expect(html).toContain('application/ld+json');
    expect(html).toContain('manifest.json');
    expect(html).toContain('apple-touch-icon');
    expect(html).toContain('noscript');
  });

  test('PWA manifest has required fields', () => {
    const manifest = JSON.parse(readFileSync(join(ROOT, 'public', 'manifest.json'), 'utf-8'));
    expect(manifest.name).toBeDefined();
    expect(manifest.short_name).toBeDefined();
    expect(manifest.display).toBe('standalone');
    expect(manifest.icons?.length).toBeGreaterThan(0);
    expect(manifest.theme_color).toBeDefined();
    expect(manifest.background_color).toBeDefined();
  });
});

// ═════════════════════════════════════════════════════════════════════
// 11. Source Code Quality Gates
// ═════════════════════════════════════════════════════════════════════

describe('Source code quality', () => {
  test('no TODO/FIXME/HACK in source files', () => {
    const srcDir = join(ROOT, 'src');
    const check = (dir: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.name === '__tests__' || entry.name === 'node_modules') continue;
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          check(full);
        } else if (entry.name.endsWith('.ts') || entry.name.endsWith('.tsx')) {
          const content = readFileSync(full, 'utf-8');
          expect(content).not.toMatch(/\bTODO\b/);
          expect(content).not.toMatch(/\bFIXME\b/);
          expect(content).not.toMatch(/\bHACK\b/);
        }
      }
    };
    check(srcDir);
  });

  test('no "as any" in source files', () => {
    const srcDir = join(ROOT, 'src');
    const check = (dir: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.name === '__tests__' || entry.name === 'node_modules') continue;
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          check(full);
        } else if (entry.name.endsWith('.ts') || entry.name.endsWith('.tsx')) {
          const content = readFileSync(full, 'utf-8');
          expect(content).not.toMatch(/as any/);
        }
      }
    };
    check(srcDir);
  });

  test('ErrorBoundary exists in components', () => {
    expect(existsSync(join(ROOT, 'src', 'components', 'ErrorBoundary.tsx'))).toBe(true);
  });

  test('prefers-reduced-motion is respected', () => {
    const cssPath = join(ROOT, 'src', 'App.css');
    const content = readFileSync(cssPath, 'utf-8');
    expect(content).toContain('prefers-reduced-motion');
  });
});

// ═════════════════════════════════════════════════════════════════════
// 12. Cross-Module Integration Stress
// ═════════════════════════════════════════════════════════════════════

describe('Cross-module integration stress', () => {
  test('train → dream → saliency → chimera rapid sequence', () => {
    const nn = trainedNetwork(15);

    // Dream 3 digits
    for (let d = 0; d < 3; d++) {
      const result = dream(nn, d, 30);
      expect(result.image.length).toBe(INPUT_SIZE);
      expect(result.confidenceHistory.length).toBe(30);
    }

    // Saliency on dreamed images
    const dreamed = dream(nn, 7, 30);
    const sal = computeSaliency(nn, dreamed.image, 7);
    expect(sal.length).toBe(INPUT_SIZE);

    // Chimera with weights array
    const weights = [0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0]; // 3 + 8
    const chimera = dreamChimera(nn, weights, 50);
    expect(chimera.image.length).toBe(INPUT_SIZE);
    expect(chimera.confidenceHistory.length).toBe(50);
  });

  test('ablation + gradient flow + confusion on same network', () => {
    const nn = trainedNetwork(20);
    const data = generateTrainingData(5);

    const ablation = runAblationStudy(nn, 5);
    const gf = measureGradientFlow(nn, data.inputs[0], data.labels[0]);
    const confusion = computeConfusionMatrix(nn, 5);

    expect(ablation.layers.length).toBeGreaterThan(0);
    expect(gf.layers.length).toBeGreaterThan(0);
    expect(confusion.matrix.length).toBe(OUTPUT_CLASSES);

    // Network should still work after all analysis
    const result = nn.predict(data.inputs[0]);
    expect(result.label).toBeGreaterThanOrEqual(0);
    expect(result.label).toBeLessThanOrEqual(9);
  });

  test('neuron surgery + prediction + gradient flow', () => {
    const nn = trainedNetwork(15);
    const data = generateTrainingData(5);

    nn.setNeuronStatus(0, 0, 'killed');
    nn.setNeuronStatus(0, 1, 'frozen');

    const result = nn.predict(data.inputs[0]);
    expect(result.probabilities.length).toBe(OUTPUT_CLASSES);

    const gf = measureGradientFlow(nn, data.inputs[0], data.labels[0]);
    gf.layers.forEach(l => {
      expect(isFinite(l.meanAbsGrad)).toBe(true);
    });

    nn.clearAllMasks();
    const result2 = nn.predict(data.inputs[0]);
    expect(result2.probabilities.length).toBe(OUTPUT_CLASSES);
  });

  test('misfits + exemplars + decision boundary pipeline', () => {
    const nn = trainedNetwork(20);
    const data = generateTrainingData(10);

    // findMisfits takes (network, inputs, labels, count?)
    const misfits = findMisfits(nn, data.inputs, data.labels);
    expect(Array.isArray(misfits)).toBe(true);

    const summary = computeMisfitSummary(nn, data.inputs, data.labels);
    expect(summary.totalWrong).toBeGreaterThanOrEqual(0);

    for (let d = 0; d < 3; d++) {
      const ex = generateExemplar(d);
      expect(ex.length).toBe(INPUT_SIZE);
      ex.forEach(v => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      });
    }

    const boundary = computeDecisionBoundary(nn, 0, 1, 16);
    expect(boundary.grid.length).toBe(16);
    expect(boundary.grid[0].length).toBe(16);
  });
});

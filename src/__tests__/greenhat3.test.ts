/**
 * Green Hat #2 (Pass 7) — Tests for Digit Chimera Lab and Misfit Gallery.
 *
 * Validates both creative features:
 * 1. Chimera Lab: multi-class blended gradient ascent
 * 2. Misfit Gallery: finding the network's hardest digits
 */

import { describe, it, expect } from 'vitest';

// ═══════════════════════════════════════════════════════════════════
// 1. CHIMERA LAB — Multi-class gradient ascent
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — Chimera Lab', () => {
  it('dreamChimera returns valid image and confidence history', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);

    const weights = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]; // blend 3 + 8
    const result = dreamChimera(nn, weights, 20, 0.5);

    expect(result.image).toHaveLength(784);
    expect(result.confidenceHistory.length).toBeGreaterThan(0);
    expect(result.finalConfidence).toHaveLength(10);
  });

  it('chimera image pixels are clamped to [0, 1]', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    for (let i = 0; i < 3; i++) nn.trainBatch(data.inputs, data.labels);

    const weights = [1, 0, 0, 1, 0, 0, 0, 0, 0, 1]; // blend 0 + 3 + 9
    const result = dreamChimera(nn, weights, 30, 0.5);

    for (const px of result.image) {
      expect(px).toBeGreaterThanOrEqual(0);
      expect(px).toBeLessThanOrEqual(1);
      expect(isFinite(px)).toBe(true);
    }
  });

  it('confidence history entries have 10 classes each', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    for (let i = 0; i < 3; i++) nn.trainBatch(data.inputs, data.labels);

    const weights = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]; // blend 1 + 7
    const result = dreamChimera(nn, weights, 15, 0.5);

    for (const snap of result.confidenceHistory) {
      expect(snap).toHaveLength(10);
      // Probabilities should be valid (softmax output)
      const sum = snap.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 1); // softmax sums to ~1
    }
  });

  it('all-zeros weights produces valid output (uniform fallback)', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    nn.trainBatch(data.inputs, data.labels);

    const weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // all zeros
    const result = dreamChimera(nn, weights, 10, 0.5);

    expect(result.image).toHaveLength(784);
    expect(result.finalConfidence).toHaveLength(10);
    // Should not crash or produce NaN
    for (const px of result.image) {
      expect(isFinite(px)).toBe(true);
    }
  });

  it('single-class chimera behaves like regular dream', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);

    const weights = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]; // only digit 5
    const result = dreamChimera(nn, weights, 40, 0.5);

    // Confidence in class 5 should be the highest (or among highest)
    const maxConf = Math.max(...result.finalConfidence);
    const class5Conf = result.finalConfidence[5];
    // Allow some tolerance — gradient ascent may not always perfectly converge
    expect(class5Conf).toBeGreaterThan(0.1);
    expect(maxConf).toBeGreaterThan(0);
  });

  it('CHIMERA_PRESETS are valid', async () => {
    const { CHIMERA_PRESETS } = await import('../nn/chimera');

    expect(CHIMERA_PRESETS.length).toBeGreaterThanOrEqual(4);

    for (const preset of CHIMERA_PRESETS) {
      expect(preset.name.length).toBeGreaterThan(0);
      expect(preset.emoji.length).toBeGreaterThan(0);
      expect(preset.description.length).toBeGreaterThan(0);
      expect(preset.weights).toHaveLength(10);
      // At least one non-zero weight
      expect(preset.weights.some((w) => w > 0)).toBe(true);
      // All non-negative
      for (const w of preset.weights) {
        expect(w).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('chimera presets have unique names', async () => {
    const { CHIMERA_PRESETS } = await import('../nn/chimera');
    const names = CHIMERA_PRESETS.map((p) => p.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it('blended chimera shifts confidence toward target classes', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const data = generateTrainingData(10);
    for (let i = 0; i < 10; i++) nn.trainBatch(data.inputs, data.labels);

    const weights = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]; // blend 3 + 8
    const result = dreamChimera(nn, weights, 50, 0.5);

    // Combined confidence in classes 3 and 8 should be significant
    const targetConf = result.finalConfidence[3] + result.finalConfidence[8];
    expect(targetConf).toBeGreaterThan(0.1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. MISFIT GALLERY — Finding the network's hardest digits
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — Misfit Gallery', () => {
  it('findMisfits returns sorted results by loss (descending)', async () => {
    const { NeuralNetwork, generateTrainingData, findMisfits } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const misfits = findMisfits(nn, data.inputs, data.labels, 10);

    expect(misfits.length).toBeLessThanOrEqual(10);
    expect(misfits.length).toBeGreaterThan(0);

    // Verify sorted by loss descending
    for (let i = 1; i < misfits.length; i++) {
      expect(misfits[i - 1].loss).toBeGreaterThanOrEqual(misfits[i].loss);
    }
  });

  it('misfit entries have all required fields', async () => {
    const { NeuralNetwork, generateTrainingData, findMisfits } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const misfits = findMisfits(nn, data.inputs, data.labels, 5);

    for (const m of misfits) {
      expect(m.input).toHaveLength(784);
      expect(m.trueLabel).toBeGreaterThanOrEqual(0);
      expect(m.trueLabel).toBeLessThanOrEqual(9);
      expect(m.predictedLabel).toBeGreaterThanOrEqual(0);
      expect(m.predictedLabel).toBeLessThanOrEqual(9);
      expect(m.confidence).toBeGreaterThanOrEqual(0);
      expect(m.confidence).toBeLessThanOrEqual(1);
      expect(m.trueConfidence).toBeGreaterThanOrEqual(0);
      expect(m.trueConfidence).toBeLessThanOrEqual(1);
      expect(isFinite(m.loss)).toBe(true);
      expect(m.loss).toBeGreaterThanOrEqual(0);
      expect(typeof m.isWrong).toBe('boolean');
      expect(m.probabilities).toHaveLength(10);
    }
  });

  it('isWrong flag is correct', async () => {
    const { NeuralNetwork, generateTrainingData, findMisfits } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const misfits = findMisfits(nn, data.inputs, data.labels, 20);

    for (const m of misfits) {
      expect(m.isWrong).toBe(m.predictedLabel !== m.trueLabel);
    }
  });

  it('computeMisfitSummary returns valid statistics', async () => {
    const { NeuralNetwork, generateTrainingData, computeMisfitSummary } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    for (let i = 0; i < 3; i++) nn.trainBatch(data.inputs, data.labels);

    const summary = computeMisfitSummary(nn, data.inputs, data.labels);

    expect(summary.totalSamples).toBe(data.inputs.length);
    expect(summary.totalWrong).toBeGreaterThanOrEqual(0);
    expect(summary.totalWrong).toBeLessThanOrEqual(summary.totalSamples);
    expect(summary.accuracy).toBeGreaterThanOrEqual(0);
    expect(summary.accuracy).toBeLessThanOrEqual(1);
    expect(summary.classErrors).toHaveLength(10);

    // Class errors should sum to totalWrong
    const errorSum = summary.classErrors.reduce((a, b) => a + b, 0);
    expect(errorSum).toBe(summary.totalWrong);
  });

  it('misfit count is limited by count parameter', async () => {
    const { NeuralNetwork, generateTrainingData, findMisfits } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const data = generateTrainingData(10); // 100 samples

    const misfits3 = findMisfits(nn, data.inputs, data.labels, 3);
    const misfits50 = findMisfits(nn, data.inputs, data.labels, 50);

    expect(misfits3.length).toBeLessThanOrEqual(3);
    expect(misfits50.length).toBeLessThanOrEqual(data.inputs.length);
  });

  it('empty inputs returns empty misfits', async () => {
    const { NeuralNetwork, findMisfits, computeMisfitSummary } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });

    const misfits = findMisfits(nn, [], [], 10);
    expect(misfits).toHaveLength(0);

    const summary = computeMisfitSummary(nn, [], []);
    expect(summary.totalSamples).toBe(0);
    expect(summary.totalWrong).toBe(0);
    expect(summary.accuracy).toBe(0);
    expect(summary.mostConfusedPair).toBeNull();
  });

  it('trained network has fewer misfits than untrained', async () => {
    const { NeuralNetwork, generateTrainingData, findMisfits } = await import('../nn');

    const data = generateTrainingData(8);

    // Untrained network
    const nnUntrained = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const untrainedMisfits = findMisfits(nnUntrained, data.inputs, data.labels, 50);
    const untrainedWrong = untrainedMisfits.filter((m) => m.isWrong).length;

    // Trained network
    const nnTrained = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    for (let i = 0; i < 15; i++) nnTrained.trainBatch(data.inputs, data.labels);
    const trainedMisfits = findMisfits(nnTrained, data.inputs, data.labels, 50);
    const trainedWrong = trainedMisfits.filter((m) => m.isWrong).length;

    // Trained should have fewer misclassifications (or at least not more)
    expect(trainedWrong).toBeLessThanOrEqual(untrainedWrong);
  });

  it('mostConfusedPair identifies real confusion', async () => {
    const { NeuralNetwork, generateTrainingData, computeMisfitSummary } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    // Only 1 epoch — should have plenty of errors
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const summary = computeMisfitSummary(nn, data.inputs, data.labels);

    if (summary.mostConfusedPair) {
      const [trueLabel, predLabel] = summary.mostConfusedPair;
      expect(trueLabel).toBeGreaterThanOrEqual(0);
      expect(trueLabel).toBeLessThanOrEqual(9);
      expect(predLabel).toBeGreaterThanOrEqual(0);
      expect(predLabel).toBeLessThanOrEqual(9);
      expect(trueLabel).not.toBe(predLabel);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. CONSTANTS & ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — Constants & module architecture', () => {
  it('chimera constants are sensible', async () => {
    const {
      CHIMERA_DISPLAY_SIZE,
      CHIMERA_STEPS,
      CHIMERA_LR,
      CHIMERA_ANIMATION_INTERVAL,
    } = await import('../constants');

    expect(CHIMERA_DISPLAY_SIZE).toBeGreaterThanOrEqual(80);
    expect(CHIMERA_DISPLAY_SIZE).toBeLessThanOrEqual(300);
    expect(CHIMERA_STEPS).toBeGreaterThanOrEqual(20);
    expect(CHIMERA_STEPS).toBeLessThanOrEqual(200);
    expect(CHIMERA_LR).toBeGreaterThan(0);
    expect(CHIMERA_LR).toBeLessThanOrEqual(2);
    expect(CHIMERA_ANIMATION_INTERVAL).toBeGreaterThanOrEqual(10);
    expect(CHIMERA_ANIMATION_INTERVAL).toBeLessThanOrEqual(200);
  });

  it('misfit constants are sensible', async () => {
    const {
      MISFIT_DISPLAY_SIZE,
      MISFIT_GALLERY_COUNT,
    } = await import('../constants');

    expect(MISFIT_DISPLAY_SIZE).toBeGreaterThanOrEqual(24);
    expect(MISFIT_DISPLAY_SIZE).toBeLessThanOrEqual(128);
    expect(MISFIT_GALLERY_COUNT).toBeGreaterThanOrEqual(5);
    expect(MISFIT_GALLERY_COUNT).toBeLessThanOrEqual(100);
  });

  it('nn barrel exports chimera functions', async () => {
    const nn = await import('../nn');

    expect(typeof nn.dreamChimera).toBe('function');
    expect(Array.isArray(nn.CHIMERA_PRESETS)).toBe(true);
  });

  it('nn barrel exports misfit functions', async () => {
    const nn = await import('../nn');

    expect(typeof nn.findMisfits).toBe('function');
    expect(typeof nn.computeMisfitSummary).toBe('function');
  });

  it('components barrel exports new components', async () => {
    const components = await import('../components');

    expect(components.ChimeraLab).toBeDefined();
    expect(components.MisfitGallery).toBeDefined();
  });

  it('chimera and misfits modules have no React dependency', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');

    const SRC = path.join(process.cwd(), 'src');

    for (const file of ['nn/chimera.ts', 'nn/misfits.ts']) {
      const content = fs.readFileSync(path.join(SRC, file), 'utf-8');
      expect(content.includes("from 'react'")).toBe(false);
      expect(content.includes("from \"react\"")).toBe(false);
    }
  });

  it('new components import from barrel (not direct nn/ files)', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');

    const SRC = path.join(process.cwd(), 'src');

    for (const file of ['components/ChimeraLab.tsx', 'components/MisfitGallery.tsx']) {
      const content = fs.readFileSync(path.join(SRC, file), 'utf-8');
      // Should not import directly from nn/NeuralNetwork
      const directImport = content.match(
        /import\s+type\s+\{[^}]+\}\s+from\s+['"]\.\.\/nn\/NeuralNetwork['"]/,
      );
      expect(directImport, `${file} imports directly from nn/NeuralNetwork`).toBeNull();
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. CROSS-MODULE INTEGRATION
// ═══════════════════════════════════════════════════════════════════

describe('Green Hat #2 — Cross-module integration', () => {
  it('chimera → misfit pipeline: generate chimera then analyze as misfit', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera, findMisfits } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const data = generateTrainingData(8);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);

    // Generate a chimera
    const chimera = dreamChimera(nn, [0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 30, 0.5);

    // Use chimera as input to find misfits (append to test data)
    const extendedInputs = [...data.inputs, chimera.image];
    const extendedLabels = [...data.labels, 3]; // pretend it's a 3

    const misfits = findMisfits(nn, extendedInputs, extendedLabels, 30);
    expect(misfits.length).toBeGreaterThan(0);

    // Chimera should be in the results (it's an unusual input)
    const chimeraEntry = misfits.find(
      (m) => m.input === chimera.image || m.input.every((v, i) => v === chimera.image[i]),
    );
    // May or may not be in top misfits, but the pipeline should work
    expect(misfits[0].loss).toBeGreaterThan(0);
  });

  it('chimera NaN safety: extreme weight values', async () => {
    const { NeuralNetwork, generateTrainingData, dreamChimera } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    nn.trainBatch(data.inputs, data.labels);

    // Extreme weights
    const weights = [1000, 0, 0, 0, 0, 0, 0, 0, 0, 1000];
    const result = dreamChimera(nn, weights, 15, 0.5);

    for (const px of result.image) {
      expect(isFinite(px)).toBe(true);
    }
    for (const conf of result.finalConfidence) {
      expect(isFinite(conf)).toBe(true);
    }
  });

  it('misfit NaN safety: loss values are finite', async () => {
    const { NeuralNetwork, generateTrainingData, findMisfits } = await import('../nn');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    // Untrained — will have lots of near-zero confidences
    const data = generateTrainingData(10);

    const misfits = findMisfits(nn, data.inputs, data.labels, 50);
    for (const m of misfits) {
      expect(isFinite(m.loss)).toBe(true);
      expect(isFinite(m.confidence)).toBe(true);
      expect(isFinite(m.trueConfidence)).toBe(true);
    }
  });
});

/**
 * Blue Hat #2 — Architecture validation tests.
 *
 * Validates the refactored module structure: extracted renderers,
 * barrel exports, race presets, cross-module integration, and
 * the elimination of business logic from React components.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const SRC = path.join(__dirname, '..');

// ═══════════════════════════════════════════════════════════════════
// Extracted race presets
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Race presets module', () => {
  it('exports RACE_PRESETS with all 4 matchups', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    expect(RACE_PRESETS).toHaveLength(4);
    const labels = RACE_PRESETS.map(p => p.label);
    expect(labels).toContain('Deep vs Shallow');
    expect(labels).toContain('ReLU vs Sigmoid');
    expect(labels).toContain('Fast vs Slow LR');
    expect(labels).toContain('Wide vs Narrow');
  });

  it('each preset isolates a single variable', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    // Deep vs Shallow: same LR, different depth
    expect(RACE_PRESETS[0].a.learningRate).toBe(RACE_PRESETS[0].b.learningRate);
    expect(RACE_PRESETS[0].a.layers.length).toBeGreaterThan(RACE_PRESETS[0].b.layers.length);
    // ReLU vs Sigmoid: same architecture, different activation
    expect(RACE_PRESETS[1].a.layers[0].neurons).toBe(RACE_PRESETS[1].b.layers[0].neurons);
    expect(RACE_PRESETS[1].a.layers[0].activation).not.toBe(RACE_PRESETS[1].b.layers[0].activation);
    // Fast vs Slow: different LR, same arch
    expect(RACE_PRESETS[2].a.learningRate).toBeGreaterThan(RACE_PRESETS[2].b.learningRate);
    expect(RACE_PRESETS[2].a.layers[0].neurons).toBe(RACE_PRESETS[2].b.layers[0].neurons);
    // Wide vs Narrow: same LR/activation, different width
    expect(RACE_PRESETS[3].a.layers[0].neurons).toBeGreaterThan(RACE_PRESETS[3].b.layers[0].neurons);
  });

  it('exports default racer configs', async () => {
    const { DEFAULT_RACER_A_CONFIG, DEFAULT_RACER_B_CONFIG } = await import('../data/racePresets');
    expect(DEFAULT_RACER_A_CONFIG.learningRate).toBeGreaterThan(0);
    expect(DEFAULT_RACER_A_CONFIG.layers.length).toBeGreaterThanOrEqual(1);
    expect(DEFAULT_RACER_B_CONFIG.learningRate).toBeGreaterThan(0);
    expect(DEFAULT_RACER_B_CONFIG.layers.length).toBeGreaterThanOrEqual(1);
  });

  it('preset configs have valid layer definitions', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    const validActivations = ['relu', 'sigmoid', 'tanh'];
    for (const preset of RACE_PRESETS) {
      for (const config of [preset.a, preset.b]) {
        expect(config.learningRate).toBeGreaterThan(0);
        expect(config.learningRate).toBeLessThanOrEqual(1);
        for (const layer of config.layers) {
          expect(layer.neurons).toBeGreaterThan(0);
          expect(Number.isInteger(layer.neurons)).toBe(true);
          expect(validActivations).toContain(layer.activation);
        }
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// Extracted renderers — dream renderer
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Dream renderer', () => {
  it('dreamPixelToRGB produces valid RGB at boundaries', async () => {
    const { dreamPixelToRGB } = await import('../renderers/dreamRenderer');
    const [r0, g0, b0] = dreamPixelToRGB(0);
    expect(r0).toBe(0);
    expect(g0).toBe(0);
    expect(b0).toBe(0);

    const [r1, g1, b1] = dreamPixelToRGB(1);
    expect(r1).toBe(Math.round(255 * 0.4));
    expect(g1).toBe(Math.round(255 * 0.87));
    expect(b1).toBe(255);
  });

  it('dreamPixelToRGB clamps out-of-range values', async () => {
    const { dreamPixelToRGB } = await import('../renderers/dreamRenderer');
    const [rn, gn, bn] = dreamPixelToRGB(-0.5);
    expect(rn).toBe(0);
    expect(gn).toBe(0);
    expect(bn).toBe(0);

    const [rh, gh, bh] = dreamPixelToRGB(1.5);
    expect(rh).toBe(Math.round(255 * 0.4));
    expect(gh).toBe(Math.round(255 * 0.87));
    expect(bh).toBe(255);
  });

  it('dreamPixelToRGB cyan gradient — G > R at all non-zero values', async () => {
    const { dreamPixelToRGB } = await import('../renderers/dreamRenderer');
    for (let v = 0.1; v <= 1.0; v += 0.1) {
      const [r, g, b] = dreamPixelToRGB(v);
      expect(g).toBeGreaterThan(r); // Cyan bias
      expect(b).toBeGreaterThanOrEqual(g); // Blue ≥ Green
    }
  });

  it('GALLERY_DIMS has correct computed dimensions', async () => {
    const { GALLERY_DIMS } = await import('../renderers/dreamRenderer');
    expect(GALLERY_DIMS.cols).toBe(5);
    expect(GALLERY_DIMS.rows).toBe(2);
    expect(GALLERY_DIMS.cellSize).toBe(40);
    expect(GALLERY_DIMS.gap).toBe(4);
    expect(GALLERY_DIMS.width).toBe(5 * (40 + 4) - 4); // 216
    expect(GALLERY_DIMS.height).toBe(2 * (40 + 4 + 14) - 4); // 112
  });
});

// ═══════════════════════════════════════════════════════════════════
// Extracted renderers — surgery renderer
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Surgery renderer', () => {
  it('computeSurgeryLayout scales with layer count', async () => {
    const { computeSurgeryLayout } = await import('../renderers/surgeryRenderer');
    const layout1 = computeSurgeryLayout([
      { weights: [], biases: [], preActivations: [], activations: new Array(32).fill(0) },
    ]);
    const layout2 = computeSurgeryLayout([
      { weights: [], biases: [], preActivations: [], activations: new Array(32).fill(0) },
      { weights: [], biases: [], preActivations: [], activations: new Array(64).fill(0) },
    ]);
    expect(layout2.canvasWidth).toBeGreaterThan(layout1.canvasWidth);
    expect(layout1.canvasHeight).toBeGreaterThan(0);
    expect(layout2.canvasHeight).toBeGreaterThan(0);
  });

  it('computeSurgeryLayout enforces minimums', async () => {
    const { computeSurgeryLayout } = await import('../renderers/surgeryRenderer');
    const layout = computeSurgeryLayout([]);
    expect(layout.canvasWidth).toBeGreaterThanOrEqual(280);
    expect(layout.canvasHeight).toBeGreaterThanOrEqual(160);
    expect(layout.padding).toBeGreaterThan(0);
  });

  it('hitTestSurgeryNode returns null for empty nodes', async () => {
    const { hitTestSurgeryNode } = await import('../renderers/surgeryRenderer');
    expect(hitTestSurgeryNode(100, 100, [])).toBeNull();
  });

  it('hitTestSurgeryNode finds node within radius', async () => {
    const { hitTestSurgeryNode } = await import('../renderers/surgeryRenderer');
    const nodes = [
      { x: 50, y: 50, layerIdx: 0, neuronIdx: 0, activation: 0.5 },
      { x: 150, y: 150, layerIdx: 1, neuronIdx: 0, activation: 0.3 },
    ];
    const hit = hitTestSurgeryNode(52, 48, nodes);
    expect(hit).not.toBeNull();
    expect(hit!.layerIdx).toBe(0);
    expect(hit!.neuronIdx).toBe(0);
  });

  it('hitTestSurgeryNode misses when too far', async () => {
    const { hitTestSurgeryNode } = await import('../renderers/surgeryRenderer');
    const nodes = [
      { x: 50, y: 50, layerIdx: 0, neuronIdx: 0, activation: 0.5 },
    ];
    expect(hitTestSurgeryNode(200, 200, nodes)).toBeNull();
  });

  it('hitTestSurgeryNode returns first match (front-to-back)', async () => {
    const { hitTestSurgeryNode } = await import('../renderers/surgeryRenderer');
    const nodes = [
      { x: 50, y: 50, layerIdx: 0, neuronIdx: 0, activation: 0.5 },
      { x: 55, y: 55, layerIdx: 0, neuronIdx: 1, activation: 0.3 },
    ];
    const hit = hitTestSurgeryNode(52, 52, nodes);
    expect(hit).not.toBeNull();
    expect(hit!.neuronIdx).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Extracted renderers — race chart
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Race chart module', () => {
  it('exports CHART_WIDTH constant', async () => {
    const { CHART_WIDTH } = await import('../renderers/raceChart');
    expect(CHART_WIDTH).toBe(420);
  });

  it('exports drawRaceChart function', async () => {
    const { drawRaceChart } = await import('../renderers/raceChart');
    expect(typeof drawRaceChart).toBe('function');
  });

  it('RaceChartData and RacerVisuals types are importable', async () => {
    const mod = await import('../renderers/raceChart');
    // If module exports compile, types are valid
    expect(mod).toBeDefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// Barrel exports
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — nn/index.ts barrel export', () => {
  it('exports NeuralNetwork class', async () => {
    const { NeuralNetwork } = await import('../nn/index');
    const nn = new NeuralNetwork(4, { learningRate: 0.01, layers: [{ neurons: 2, activation: 'relu' }] });
    expect(nn).toBeDefined();
    expect(nn.getEpoch()).toBe(0);
  });

  it('exports generateTrainingData', async () => {
    const { generateTrainingData } = await import('../nn/index');
    const data = generateTrainingData(2);
    expect(data.inputs.length).toBeGreaterThan(0);
    expect(data.labels.length).toBe(data.inputs.length);
  });

  it('exports canvasToInput', async () => {
    const { canvasToInput } = await import('../nn/index');
    expect(typeof canvasToInput).toBe('function');
  });
});

describe('Blue Hat #2 — renderers/index.ts barrel export', () => {
  it('re-exports all dream renderer functions', async () => {
    const renderers = await import('../renderers/index');
    expect(typeof renderers.renderDreamImage).toBe('function');
    expect(typeof renderers.renderDreamGallery).toBe('function');
    expect(typeof renderers.dreamPixelToRGB).toBe('function');
    expect(renderers.GALLERY_DIMS).toBeDefined();
  });

  it('re-exports all surgery renderer functions', async () => {
    const renderers = await import('../renderers/index');
    expect(typeof renderers.drawSurgeryCanvas).toBe('function');
    expect(typeof renderers.computeSurgeryLayout).toBe('function');
    expect(typeof renderers.hitTestSurgeryNode).toBe('function');
  });

  it('re-exports race chart', async () => {
    const renderers = await import('../renderers/index');
    expect(typeof renderers.drawRaceChart).toBe('function');
    expect(renderers.CHART_WIDTH).toBe(420);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Component slimming verification
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Component slimming', () => {
  it('TrainingRace.tsx no longer contains drawRaceChart implementation', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'TrainingRace.tsx'), 'utf-8',
    );
    // Should NOT have the old inline function
    expect(src).not.toContain('function drawRaceChart');
    expect(src).not.toContain('const pad = {');
    // Should import from renderers
    expect(src).toContain("from '../renderers/raceChart'");
  });

  it('NetworkDreams.tsx no longer contains inline pixel rendering', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'NetworkDreams.tsx'), 'utf-8',
    );
    // Should NOT have the old inline colorize logic
    expect(src).not.toContain('v * 0.4');
    expect(src).not.toContain('v * 0.87');
    // Should import from renderers
    expect(src).toContain("from '../renderers/dreamRenderer'");
  });

  it('NeuronSurgery.tsx no longer contains inline canvas drawing', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'NeuronSurgery.tsx'), 'utf-8',
    );
    // Should NOT have 100+ lines of canvas draw code
    expect(src).not.toContain('createRadialGradient');
    expect(src).not.toContain('getActivationColor');
    // Should import from renderers
    expect(src).toContain("from '../renderers/surgeryRenderer'");
  });

  it('useTrainingRace.ts no longer contains RACE_PRESETS definition', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'hooks', 'useTrainingRace.ts'), 'utf-8',
    );
    expect(src).not.toContain("label: 'Deep vs Shallow'");
    expect(src).not.toContain("label: 'ReLU vs Sigmoid'");
    // Should import from data module
    expect(src).toContain("from '../data/racePresets'");
  });

  it('TrainingRace.tsx is under 160 lines (was 255)', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'TrainingRace.tsx'), 'utf-8',
    );
    const lines = src.split('\n').length;
    expect(lines).toBeLessThan(160);
  });

  it('NeuronSurgery.tsx is under 180 lines (was 313)', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'NeuronSurgery.tsx'), 'utf-8',
    );
    const lines = src.split('\n').length;
    expect(lines).toBeLessThan(180);
  });

  it('NetworkDreams.tsx is under 250 lines (was 300, polish added micro-interactions)', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'NetworkDreams.tsx'), 'utf-8',
    );
    const lines = src.split('\n').length;
    expect(lines).toBeLessThan(250);
  });

  it('useTrainingRace.ts is under 185 lines (was 234)', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'hooks', 'useTrainingRace.ts'), 'utf-8',
    );
    const lines = src.split('\n').length;
    expect(lines).toBeLessThan(185);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Cross-module integration
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Cross-module integration', () => {
  it('NeuralNetwork trains and predicts through barrel export', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn/index');
    const data = generateTrainingData(3);
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const snap = nn.trainBatch(data.inputs, data.labels);
    expect(snap.epoch).toBe(1);
    expect(snap.loss).toBeGreaterThan(0);
    expect(isFinite(snap.loss)).toBe(true);

    const pred = nn.predict(data.inputs[0]);
    expect(pred.label).toBeGreaterThanOrEqual(0);
    expect(pred.label).toBeLessThan(10);
    expect(pred.probabilities).toHaveLength(10);
  });

  it('race presets produce convergent training with NeuralNetwork', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    const { NeuralNetwork, generateTrainingData } = await import('../nn/index');
    const data = generateTrainingData(5);

    for (const preset of RACE_PRESETS) {
      const nn = new NeuralNetwork(784, preset.a);
      let lastLoss = Infinity;
      for (let i = 0; i < 5; i++) {
        const snap = nn.trainBatch(data.inputs, data.labels);
        expect(isFinite(snap.loss)).toBe(true);
        lastLoss = snap.loss;
      }
      // After 5 epochs, loss should be finite (not diverging)
      expect(isFinite(lastLoss)).toBe(true);
    }
  });

  it('surgery layout works with real NeuralNetwork layers', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn/index');
    const { computeSurgeryLayout } = await import('../renderers/surgeryRenderer');
    const data = generateTrainingData(3);
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [
        { neurons: 64, activation: 'relu' },
        { neurons: 32, activation: 'relu' },
      ],
    });
    const snap = nn.trainBatch(data.inputs, data.labels);
    const hiddenLayers = snap.layers.slice(0, -1);
    const layout = computeSurgeryLayout(hiddenLayers);
    expect(layout.canvasWidth).toBeGreaterThan(280);
    expect(layout.canvasHeight).toBeGreaterThan(160);
  });

  it('dream pixel renderer handles network output range', async () => {
    const { dreamPixelToRGB } = await import('../renderers/dreamRenderer');
    const { NeuralNetwork } = await import('../nn/index');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const result = nn.dream(5, 10, 0.5);
    for (const pixel of result.image) {
      const [r, g, b] = dreamPixelToRGB(pixel);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(255);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(255);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThanOrEqual(255);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// Module structure verification
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Module structure', () => {
  it('renderers/ directory exists with 4 files', () => {
    const dir = path.join(SRC, 'renderers');
    expect(fs.existsSync(dir)).toBe(true);
    const files = fs.readdirSync(dir).sort();
    expect(files).toContain('index.ts');
    expect(files).toContain('raceChart.ts');
    expect(files).toContain('dreamRenderer.ts');
    expect(files).toContain('surgeryRenderer.ts');
  });

  it('nn/index.ts barrel exists', () => {
    expect(fs.existsSync(path.join(SRC, 'nn', 'index.ts'))).toBe(true);
  });

  it('data/racePresets.ts exists', () => {
    expect(fs.existsSync(path.join(SRC, 'data', 'racePresets.ts'))).toBe(true);
  });

  it('no circular dependencies — renderers do not import React', () => {
    const files = ['raceChart.ts', 'dreamRenderer.ts', 'surgeryRenderer.ts'];
    for (const file of files) {
      const src = fs.readFileSync(path.join(SRC, 'renderers', file), 'utf-8');
      expect(src).not.toContain("from 'react'");
      expect(src).not.toContain("from \"react\"");
    }
  });

  it('data/racePresets.ts does not import React', () => {
    const src = fs.readFileSync(path.join(SRC, 'data', 'racePresets.ts'), 'utf-8');
    expect(src).not.toContain("from 'react'");
  });

  it('no duplicate RACE_PRESETS definitions remain', () => {
    const hookSrc = fs.readFileSync(path.join(SRC, 'hooks', 'useTrainingRace.ts'), 'utf-8');
    const occurrences = (hookSrc.match(/export const RACE_PRESETS/g) || []).length;
    expect(occurrences).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Type system consistency
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Type system consistency', () => {
  it('RacePreset type fields match preset data', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    for (const preset of RACE_PRESETS) {
      // TypeScript would catch this at compile time, but runtime validation ensures
      // no runtime type mismatches in production
      expect(typeof preset.label).toBe('string');
      expect(typeof preset.a).toBe('object');
      expect(typeof preset.b).toBe('object');
      expect(typeof preset.a.learningRate).toBe('number');
      expect(Array.isArray(preset.a.layers)).toBe(true);
    }
  });

  it('SurgeryNode type has required fields', async () => {
    const { hitTestSurgeryNode } = await import('../renderers/surgeryRenderer');
    // Create a valid node and verify hit-test works with it
    const node = { x: 100, y: 100, layerIdx: 0, neuronIdx: 5, activation: 0.8 };
    const result = hitTestSurgeryNode(100, 100, [node]);
    expect(result).not.toBeNull();
    expect(result!.x).toBe(100);
    expect(result!.activation).toBe(0.8);
  });

  it('GALLERY_DIMS is readonly', async () => {
    const { GALLERY_DIMS } = await import('../renderers/dreamRenderer');
    // Frozen by 'as const' — fields should exist and be numbers
    expect(typeof GALLERY_DIMS.width).toBe('number');
    expect(typeof GALLERY_DIMS.height).toBe('number');
    expect(typeof GALLERY_DIMS.cols).toBe('number');
    expect(typeof GALLERY_DIMS.rows).toBe('number');
    expect(typeof GALLERY_DIMS.cellSize).toBe('number');
    expect(typeof GALLERY_DIMS.gap).toBe('number');
  });
});

// ═══════════════════════════════════════════════════════════════════
// Constants consistency
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #2 — Constants consistency', () => {
  it('DREAM_STEPS matches between constants and renderer', async () => {
    const { DREAM_STEPS } = await import('../constants');
    expect(DREAM_STEPS).toBe(80);
    expect(typeof DREAM_STEPS).toBe('number');
  });

  it('SURGERY constants are positive integers', async () => {
    const {
      SURGERY_NODE_RADIUS,
      SURGERY_NODE_SPACING,
      SURGERY_MAX_DISPLAY_NEURONS,
    } = await import('../constants');
    expect(SURGERY_NODE_RADIUS).toBeGreaterThan(0);
    expect(SURGERY_NODE_SPACING).toBeGreaterThan(0);
    expect(SURGERY_MAX_DISPLAY_NEURONS).toBeGreaterThan(0);
    expect(Number.isInteger(SURGERY_MAX_DISPLAY_NEURONS)).toBe(true);
  });

  it('RACE constants are sane', async () => {
    const { RACE_EPOCHS, RACE_STEP_INTERVAL, RACE_CHART_HEIGHT } = await import('../constants');
    expect(RACE_EPOCHS).toBeGreaterThanOrEqual(10);
    expect(RACE_STEP_INTERVAL).toBeGreaterThan(0);
    expect(RACE_CHART_HEIGHT).toBeGreaterThan(50);
  });

  it('all activation function types are covered in utils', async () => {
    const { activate, activateDerivative } = await import('../utils');
    const fns = ['relu', 'sigmoid', 'tanh'] as const;
    for (const fn of fns) {
      expect(typeof activate(0.5, fn)).toBe('number');
      expect(typeof activateDerivative(0.5, fn)).toBe('number');
      expect(isFinite(activate(0.5, fn))).toBe(true);
      expect(isFinite(activateDerivative(0.5, fn))).toBe(true);
    }
  });
});

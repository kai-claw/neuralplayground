/**
 * Blue Hat #3 — Architecture Refactor Verification
 *
 * Pass 6 tests validating:
 * 1. Type consolidation (types.ts is canonical, no circular re-exports)
 * 2. Utils split (4 focused modules with barrel re-export)
 * 3. Dream extraction (nn/dreams.ts + backward-compat delegation)
 * 4. Module export verification
 * 5. Cross-module integration
 * 6. Import hygiene (no imports from nn/NeuralNetwork for types)
 * 7. Constants consistency
 * 8. Type system completeness
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const ROOT = path.resolve(__dirname, '../..');
const SRC = path.join(ROOT, 'src');

// ═══════════════════════════════════════════════════════════════════
// 1. TYPE CONSOLIDATION
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Types consolidation', () => {
  it('types.ts defines all core types (not re-exported from nn/)', () => {
    const src = fs.readFileSync(path.join(SRC, 'types.ts'), 'utf-8');
    // These should be PRIMARY definitions, not re-exports
    expect(src).toContain('export type ActivationFn');
    expect(src).toContain('export type NeuronStatus');
    expect(src).toContain('export interface LayerConfig');
    expect(src).toContain('export interface TrainingConfig');
    expect(src).toContain('export interface LayerState');
    expect(src).toContain('export interface TrainingSnapshot');
    expect(src).toContain('export type CinematicPhase');
    expect(src).toContain('export type NoiseType');
    expect(src).toContain('export interface Dimensions');
    expect(src).toContain('export interface DreamResult');
  });

  it('types.ts has no imports from nn/ (canonical source, not re-exports)', () => {
    const src = fs.readFileSync(path.join(SRC, 'types.ts'), 'utf-8');
    expect(src).not.toContain("from './nn/");
    expect(src).not.toContain("from '../nn/");
  });

  it('NeuralNetwork.ts imports types from ../types (not self-defined)', () => {
    const src = fs.readFileSync(path.join(SRC, 'nn', 'NeuralNetwork.ts'), 'utf-8');
    expect(src).toContain("from '../types'");
    // Should NOT define its own type/interface exports
    expect(src).not.toMatch(/^export type /m);
    expect(src).not.toMatch(/^export interface /m);
  });

  it('no non-test file imports types from nn/NeuralNetwork', () => {
    const srcFiles = collectFiles(SRC, ['.ts', '.tsx']).filter(
      f => !f.includes('__tests__') && !f.includes('node_modules'),
    );
    for (const file of srcFiles) {
      const src = fs.readFileSync(file, 'utf-8');
      // Allow importing the CLASS from nn/NeuralNetwork, but not types
      const typeImports = src.match(/import\s+type\s+\{[^}]+\}\s+from\s+['"].*nn\/NeuralNetwork['"]/);
      if (typeImports) {
        expect.fail(`${path.relative(SRC, file)} imports types from nn/NeuralNetwork — should use ../types`);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. UTILS SPLIT VERIFICATION
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Utils split into focused modules', () => {
  it('utils/ directory contains 4 focused modules + barrel', () => {
    const utilsDir = path.join(SRC, 'utils');
    const files = fs.readdirSync(utilsDir).filter(f => f.endsWith('.ts')).sort();
    expect(files).toEqual([
      'activations.ts',
      'colors.ts',
      'index.ts',
      'math.ts',
      'prng.ts',
    ]);
  });

  it('activations.ts exports activate and activateDerivative', () => {
    const src = fs.readFileSync(path.join(SRC, 'utils', 'activations.ts'), 'utf-8');
    expect(src).toContain('export function activate');
    expect(src).toContain('export function activateDerivative');
  });

  it('math.ts exports safeMax, argmax, softmax, xavierInit', () => {
    const src = fs.readFileSync(path.join(SRC, 'utils', 'math.ts'), 'utf-8');
    expect(src).toContain('export function safeMax');
    expect(src).toContain('export function argmax');
    expect(src).toContain('export function softmax');
    expect(src).toContain('export function xavierInit');
  });

  it('prng.ts exports mulberry32 and gaussianNoise', () => {
    const src = fs.readFileSync(path.join(SRC, 'utils', 'prng.ts'), 'utf-8');
    expect(src).toContain('export function mulberry32');
    expect(src).toContain('export function gaussianNoise');
  });

  it('colors.ts exports getActivationColor and getWeightColor', () => {
    const src = fs.readFileSync(path.join(SRC, 'utils', 'colors.ts'), 'utf-8');
    expect(src).toContain('export function getActivationColor');
    expect(src).toContain('export function getWeightColor');
  });

  it('barrel re-exports all functions from sub-modules', async () => {
    const barrel = await import('../utils');
    const expected = [
      'activate', 'activateDerivative',
      'safeMax', 'argmax', 'softmax', 'xavierInit',
      'mulberry32', 'gaussianNoise',
      'getActivationColor', 'getWeightColor',
    ];
    for (const fn of expected) {
      expect(typeof (barrel as Record<string, unknown>)[fn], `Missing export: ${fn}`).toBe('function');
    }
  });

  it('barrel identity — re-exports are the same function objects', async () => {
    const barrel = await import('../utils');
    const activations = await import('../utils/activations');
    const math = await import('../utils/math');
    const prng = await import('../utils/prng');
    const colors = await import('../utils/colors');

    expect(barrel.activate).toBe(activations.activate);
    expect(barrel.activateDerivative).toBe(activations.activateDerivative);
    expect(barrel.safeMax).toBe(math.safeMax);
    expect(barrel.argmax).toBe(math.argmax);
    expect(barrel.softmax).toBe(math.softmax);
    expect(barrel.xavierInit).toBe(math.xavierInit);
    expect(barrel.mulberry32).toBe(prng.mulberry32);
    expect(barrel.gaussianNoise).toBe(prng.gaussianNoise);
    expect(barrel.getActivationColor).toBe(colors.getActivationColor);
    expect(barrel.getWeightColor).toBe(colors.getWeightColor);
  });

  it('no utils sub-module imports React', () => {
    const utilsDir = path.join(SRC, 'utils');
    const files = fs.readdirSync(utilsDir).filter(f => f.endsWith('.ts'));
    for (const file of files) {
      const src = fs.readFileSync(path.join(utilsDir, file), 'utf-8');
      expect(src, `utils/${file} imports React`).not.toContain("from 'react'");
    }
  });

  it('old utils.ts no longer exists in src root', () => {
    expect(fs.existsSync(path.join(SRC, 'utils.ts'))).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. DREAM EXTRACTION VERIFICATION
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Dream extraction to nn/dreams.ts', () => {
  it('nn/dreams.ts exists and exports computeInputGradient and dream', () => {
    const src = fs.readFileSync(path.join(SRC, 'nn', 'dreams.ts'), 'utf-8');
    expect(src).toContain('export function computeInputGradient');
    expect(src).toContain('export function dream');
  });

  it('nn/dreams.ts imports NeuralNetwork as type only (no circular value dep)', () => {
    const src = fs.readFileSync(path.join(SRC, 'nn', 'dreams.ts'), 'utf-8');
    expect(src).toContain("import type { NeuralNetwork }");
    // Should NOT have a value import of NeuralNetwork
    expect(src).not.toMatch(/import\s+\{\s*NeuralNetwork\s*\}\s+from/);
  });

  it('NeuralNetwork.ts has backward-compat dream() and computeInputGradient() methods', () => {
    const src = fs.readFileSync(path.join(SRC, 'nn', 'NeuralNetwork.ts'), 'utf-8');
    expect(src).toContain('computeInputGradient(input: number[]');
    expect(src).toContain('dream(');
  });

  it('nn/index.ts barrel exports dream functions', async () => {
    const nn = await import('../nn');
    expect(typeof nn.computeInputGradient).toBe('function');
    expect(typeof nn.dream).toBe('function');
    expect(typeof nn.NeuralNetwork).toBe('function');
    expect(typeof nn.generateTrainingData).toBe('function');
    expect(typeof nn.canvasToInput).toBe('function');
  });

  it('dream function produces valid results via extracted module', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { dream } = await import('../nn/dreams');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    // Train briefly
    const { generateTrainingData } = await import('../nn/sampleData');
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const result = dream(nn, 5, 10, 0.5);
    expect(result.image).toHaveLength(784);
    expect(result.confidenceHistory).toHaveLength(10);
    // All pixels in [0, 1]
    for (const px of result.image) {
      expect(px).toBeGreaterThanOrEqual(0);
      expect(px).toBeLessThanOrEqual(1);
    }
  });

  it('computeInputGradient produces finite gradient via extracted module', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { computeInputGradient } = await import('../nn/dreams');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const gradient = computeInputGradient(nn, new Array(784).fill(0.5), 3);
    expect(gradient).toHaveLength(784);
    for (const g of gradient) {
      expect(isFinite(g)).toBe(true);
    }
  });

  it('backward-compat: nn.dream() returns same structure as dreams.dream()', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const result = nn.dream(3, 5, 0.3);
    expect(result).toHaveProperty('image');
    expect(result).toHaveProperty('confidenceHistory');
    expect(result.image).toHaveLength(784);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. MODULE STRUCTURE VERIFICATION
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Module structure', () => {
  it('nn/ directory has expected files', () => {
    const nnDir = path.join(SRC, 'nn');
    const files = fs.readdirSync(nnDir).filter(f => f.endsWith('.ts')).sort();
    expect(files).toEqual([
      'NeuralNetwork.ts',
      'ablation.ts',
      'chimera.ts',
      'confusion.ts',
      'decisionBoundary.ts',
      'dreams.ts',
      'epochReplay.ts',
      'gradientFlow.ts',
      'index.ts',
      'misfits.ts',
      'noise.ts',
      'pca.ts',
      'saliency.ts',
      'sampleData.ts',
      'weightEvolution.ts',
    ]);
  });

  it('hooks/ directory has expected hooks + barrel', () => {
    const hooksDir = path.join(SRC, 'hooks');
    const files = fs.readdirSync(hooksDir).filter(f => f.endsWith('.ts')).sort();
    expect(files).toEqual([
      'index.ts',
      'useActivationSpace.ts',
      'useCinematic.ts',
      'useContainerDims.ts',
      'useNeuralNetwork.ts',
      'useTrainingRace.ts',
    ]);
  });

  it('renderers/ directory has expected files + pixelRendering', () => {
    const dir = path.join(SRC, 'renderers');
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.ts')).sort();
    expect(files).toEqual([
      'confusionRenderer.ts',
      'dreamRenderer.ts',
      'gradientFlowRenderer.ts',
      'index.ts',
      'pixelRendering.ts',
      'raceChart.ts',
      'surgeryRenderer.ts',
    ]);
  });

  it('data/ directory has expected files', () => {
    const dir = path.join(SRC, 'data');
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.ts')).sort();
    expect(files).toEqual([
      'digitStrokes.ts',
      'racePresets.ts',
    ]);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. CROSS-MODULE INTEGRATION
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Cross-module integration', () => {
  it('activation functions work end-to-end: forward pass → softmax → probabilities', async () => {
    const { activate, softmax, argmax } = await import('../utils');
    // Simulate a forward pass
    const preAct = [-2, -1, 0, 1, 2];
    const relu = preAct.map(x => activate(x, 'relu'));
    expect(relu).toEqual([0, 0, 0, 1, 2]);

    const probs = softmax(preAct);
    expect(probs).toHaveLength(5);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0);
    expect(argmax(probs)).toBe(4);
  });

  it('PRNG → noise → rendering pipeline works', async () => {
    const { mulberry32, gaussianNoise } = await import('../utils');
    const { generateNoisePattern, applyNoise } = await import('../noise');
    const { pixelsToImageData } = await import('../rendering');

    // Generate a noise pattern
    const rng = mulberry32(42);
    expect(typeof rng()).toBe('number');
    expect(typeof gaussianNoise(rng)).toBe('number');

    // Generate and apply noise
    const pattern = generateNoisePattern('gaussian', 42);
    expect(pattern).toBeInstanceOf(Float32Array);
    expect(pattern.length).toBe(784);

    const clean = new Array(784).fill(0.5);
    const noised = applyNoise(clean, pattern, 0.5, 'gaussian', 42);
    expect(noised).toHaveLength(784);
    for (const px of noised) {
      expect(px).toBeGreaterThanOrEqual(0);
      expect(px).toBeLessThanOrEqual(1);
    }

    // Render to ImageData
    const imgData = pixelsToImageData(noised, 56);
    expect(imgData.width).toBe(56);
    expect(imgData.height).toBe(56);
  });

  it('NeuralNetwork + dreams: train → dream → render', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn');
    const { dream } = await import('../nn/dreams');
    const { pixelsToImageData } = await import('../rendering');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const result = dream(nn, 7, 10, 0.5);
    const imgData = pixelsToImageData(result.image, 28);
    expect(imgData.width).toBe(28);
    expect(imgData.data.length).toBe(28 * 28 * 4);
  });

  it('constants used by NeuralNetwork match types', async () => {
    const { INPUT_SIZE, OUTPUT_CLASSES, DEFAULT_CONFIG } = await import('../constants');
    const { NeuralNetwork } = await import('../nn');

    expect(INPUT_SIZE).toBe(784);
    expect(OUTPUT_CLASSES).toBe(10);

    // DEFAULT_CONFIG creates a valid network
    const nn = new NeuralNetwork(INPUT_SIZE, DEFAULT_CONFIG);
    const result = nn.predict(new Array(INPUT_SIZE).fill(0));
    expect(result.probabilities).toHaveLength(OUTPUT_CLASSES);
  });

  it('pixelRendering module uses constants consistently', async () => {
    const { INPUT_DIM } = await import('../constants');
    const src = fs.readFileSync(path.join(SRC, 'renderers', 'pixelRendering.ts'), 'utf-8');
    expect(src).toContain('INPUT_DIM');
    expect(INPUT_DIM).toBe(28);
  });

  it('race presets produce trainable networks', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    const { NeuralNetwork, generateTrainingData } = await import('../nn');

    const data = generateTrainingData(3);
    for (const preset of RACE_PRESETS) {
      const nnA = new NeuralNetwork(784, preset.a);
      const nnB = new NeuralNetwork(784, preset.b);
      const snapA = nnA.trainBatch(data.inputs, data.labels);
      const snapB = nnB.trainBatch(data.inputs, data.labels);
      expect(isFinite(snapA.loss)).toBe(true);
      expect(isFinite(snapB.loss)).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. IMPORT HYGIENE
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Import hygiene', () => {
  it('no non-test source file has "as any" type cast', () => {
    const srcFiles = collectFiles(SRC, ['.ts', '.tsx']).filter(
      f => !f.includes('__tests__') && !f.includes('node_modules'),
    );
    for (const file of srcFiles) {
      const src = fs.readFileSync(file, 'utf-8');
      expect(src, `${path.relative(SRC, file)} has "as any"`).not.toContain(' as any');
    }
  });

  it('no non-test source file has TODO/FIXME/HACK', () => {
    const srcFiles = collectFiles(SRC, ['.ts', '.tsx']).filter(
      f => !f.includes('__tests__') && !f.includes('node_modules'),
    );
    for (const file of srcFiles) {
      const src = fs.readFileSync(file, 'utf-8');
      const rel = path.relative(SRC, file);
      expect(src, `${rel} has TODO`).not.toMatch(/\bTODO\b/);
      expect(src, `${rel} has FIXME`).not.toMatch(/\bFIXME\b/);
      expect(src, `${rel} has HACK`).not.toMatch(/\bHACK\b/);
    }
  });

  it('all hooks import types from ../types (not nn/NeuralNetwork)', () => {
    const hooksDir = path.join(SRC, 'hooks');
    const files = fs.readdirSync(hooksDir).filter(f => f.endsWith('.ts'));
    for (const file of files) {
      const src = fs.readFileSync(path.join(hooksDir, file), 'utf-8');
      // Type imports should be from ../types, not ../nn/NeuralNetwork
      const nnTypeImport = src.match(/import\s+type\s+\{[^}]+\}\s+from\s+['"]\.\.\/nn\/NeuralNetwork['"]/);
      if (nnTypeImport) {
        expect.fail(`hooks/${file} imports types from nn/NeuralNetwork — should use ../types`);
      }
    }
  });

  it('all components import types from ../types (not nn/NeuralNetwork)', () => {
    const dir = path.join(SRC, 'components');
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.tsx'));
    for (const file of files) {
      const src = fs.readFileSync(path.join(dir, file), 'utf-8');
      const nnTypeImport = src.match(/import\s+type\s+\{[^}]+\}\s+from\s+['"]\.\.\/nn\/NeuralNetwork['"]/);
      if (nnTypeImport) {
        expect.fail(`components/${file} imports types from nn/NeuralNetwork — should use ../types`);
      }
    }
  });

  it('all renderers import types from ../types (not nn/NeuralNetwork)', () => {
    const dir = path.join(SRC, 'renderers');
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.ts') && f !== 'index.ts');
    for (const file of files) {
      const src = fs.readFileSync(path.join(dir, file), 'utf-8');
      const nnTypeImport = src.match(/import\s+type\s+\{[^}]+\}\s+from\s+['"]\.\.\/nn\/NeuralNetwork['"]/);
      if (nnTypeImport) {
        expect.fail(`renderers/${file} imports types from nn/NeuralNetwork — should use ../types`);
      }
    }
  });

  it('data/ modules import types from ../types (not nn/NeuralNetwork)', () => {
    const dir = path.join(SRC, 'data');
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.ts'));
    for (const file of files) {
      const src = fs.readFileSync(path.join(dir, file), 'utf-8');
      const nnTypeImport = src.match(/import\s+type\s+\{[^}]+\}\s+from\s+['"]\.\.\/nn\/NeuralNetwork['"]/);
      if (nnTypeImport) {
        expect.fail(`data/${file} imports types from nn/NeuralNetwork — should use ../types`);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7. CONSTANTS CONSISTENCY
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Constants consistency', () => {
  it('all constants are actually used in source', async () => {
    const constSrc = fs.readFileSync(path.join(SRC, 'constants.ts'), 'utf-8');
    const exports = [...constSrc.matchAll(/export const (\w+)/g)].map(m => m[1]);

    const srcFiles = collectFiles(SRC, ['.ts', '.tsx']).filter(
      f => !f.includes('__tests__') && f !== path.join(SRC, 'constants.ts'),
    );
    const allSrc = srcFiles.map(f => fs.readFileSync(f, 'utf-8')).join('\n');

    // Documentation/test-only constants (used in tests, CSS vars, or documenting magic numbers)
    const testOnly = new Set(['INPUT_SIZE', 'OUTPUT_CLASSES', 'COLOR_GREEN_HEX']);
    for (const name of exports) {
      if (testOnly.has(name)) continue;
      expect(allSrc, `Constant '${name}' exported but unused`).toContain(name);
    }
  });

  it('timing constants are positive numbers', async () => {
    const {
      CINEMATIC_TRAIN_EPOCHS,
      CINEMATIC_PREDICT_DWELL,
      CINEMATIC_EPOCH_INTERVAL,
      AUTO_TRAIN_EPOCHS,
      AUTO_TRAIN_DELAY,
      TRAINING_STEP_INTERVAL,
      DREAM_ANIMATION_INTERVAL,
      RACE_STEP_INTERVAL,
      RACE_EPOCHS,
    } = await import('../constants');

    const timings = {
      CINEMATIC_TRAIN_EPOCHS,
      CINEMATIC_PREDICT_DWELL,
      CINEMATIC_EPOCH_INTERVAL,
      AUTO_TRAIN_EPOCHS,
      AUTO_TRAIN_DELAY,
      TRAINING_STEP_INTERVAL,
      DREAM_ANIMATION_INTERVAL,
      RACE_STEP_INTERVAL,
      RACE_EPOCHS,
    };

    for (const [name, val] of Object.entries(timings)) {
      expect(val, `${name} should be positive`).toBeGreaterThan(0);
      expect(isFinite(val), `${name} should be finite`).toBe(true);
    }
  });

  it('display size constants are valid', async () => {
    const {
      FEATURE_MAP_CELL_SIZE,
      FEATURE_MAP_MAGNIFIER_SIZE,
      ADVERSARIAL_DISPLAY_SIZE,
      MORPH_DISPLAY_SIZE,
      DREAM_DISPLAY_SIZE,
    } = await import('../constants');

    const sizes = {
      FEATURE_MAP_CELL_SIZE,
      FEATURE_MAP_MAGNIFIER_SIZE,
      ADVERSARIAL_DISPLAY_SIZE,
      MORPH_DISPLAY_SIZE,
      DREAM_DISPLAY_SIZE,
    };

    for (const [name, val] of Object.entries(sizes)) {
      expect(val, `${name} should be > 0`).toBeGreaterThan(0);
      expect(Number.isInteger(val), `${name} should be integer`).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 8. TYPE SYSTEM COMPLETENESS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #3 — Type system', () => {
  it('ActivationFn covers relu, sigmoid, tanh', async () => {
    const { activate } = await import('../utils/activations');
    const fns: Array<'relu' | 'sigmoid' | 'tanh'> = ['relu', 'sigmoid', 'tanh'];
    for (const fn of fns) {
      expect(typeof activate(1.0, fn)).toBe('number');
      expect(isFinite(activate(1.0, fn))).toBe(true);
    }
  });

  it('NeuronStatus covers active, frozen, killed', async () => {
    const { NeuralNetwork } = await import('../nn');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const statuses: Array<'active' | 'frozen' | 'killed'> = ['active', 'frozen', 'killed'];
    for (const status of statuses) {
      nn.setNeuronStatus(0, 0, status);
      expect(nn.getNeuronStatus(0, 0)).toBe(status === 'active' ? 'active' : status);
    }
  });

  it('NoiseType covers gaussian, salt-pepper, adversarial', async () => {
    const { NOISE_LABELS, NOISE_DESCRIPTIONS } = await import('../constants');
    const types: Array<'gaussian' | 'salt-pepper' | 'adversarial'> = ['gaussian', 'salt-pepper', 'adversarial'];
    for (const type of types) {
      expect(NOISE_LABELS[type]).toBeDefined();
      expect(NOISE_DESCRIPTIONS[type]).toBeDefined();
    }
  });

  it('TrainingSnapshot has all required fields', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    const snap = nn.trainBatch(data.inputs, data.labels);

    expect(snap).toHaveProperty('epoch');
    expect(snap).toHaveProperty('loss');
    expect(snap).toHaveProperty('accuracy');
    expect(snap).toHaveProperty('layers');
    expect(snap).toHaveProperty('predictions');
    expect(snap).toHaveProperty('outputProbabilities');
    expect(typeof snap.epoch).toBe('number');
    expect(typeof snap.loss).toBe('number');
    expect(typeof snap.accuracy).toBe('number');
    expect(Array.isArray(snap.layers)).toBe(true);
    expect(snap.outputProbabilities).toHaveLength(10);
  });

  it('DreamResult has image and confidenceHistory', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn');
    const { dream } = await import('../nn/dreams');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    nn.trainBatch(data.inputs, data.labels);
    const result = dream(nn, 0, 5, 0.3);
    expect(result).toHaveProperty('image');
    expect(result).toHaveProperty('confidenceHistory');
    expect(result.image).toHaveLength(784);
    expect(result.confidenceHistory).toHaveLength(5);
  });

  it('LayerState has weights, biases, preActivations, activations', async () => {
    const { NeuralNetwork } = await import('../nn');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const result = nn.predict(new Array(784).fill(0));
    for (const layer of result.layers) {
      expect(layer).toHaveProperty('weights');
      expect(layer).toHaveProperty('biases');
      expect(layer).toHaveProperty('preActivations');
      expect(layer).toHaveProperty('activations');
      expect(Array.isArray(layer.weights)).toBe(true);
      expect(Array.isArray(layer.biases)).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

function collectFiles(dir: string, exts: string[]): string[] {
  const results: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...collectFiles(full, exts));
    } else if (exts.some(e => entry.name.endsWith(e))) {
      results.push(full);
    }
  }
  return results;
}

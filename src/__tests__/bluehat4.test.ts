/**
 * Blue Hat #4 — Architecture & integration tests.
 *
 * Pass 6 comprehensive tests covering:
 * - Barrel export identity verification
 * - Module boundary enforcement
 * - Extracted component contracts
 * - Cross-module integration pipelines
 * - Backward compatibility re-exports
 * - Constants completeness
 * - Type system consistency
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect } from 'vitest';

const SRC = path.resolve(__dirname, '..');

// ═══════════════════════════════════════════════════════════════════
// 1. BARREL EXPORT IDENTITY — verify re-exports reference same objects
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Barrel export identity', () => {
  it('nn/index re-exports NeuralNetwork identical to direct import', async () => {
    const barrel = await import('../nn/index');
    const direct = await import('../nn/NeuralNetwork');
    expect(barrel.NeuralNetwork).toBe(direct.NeuralNetwork);
  });

  it('nn/index re-exports dream functions identical to direct', async () => {
    const barrel = await import('../nn/index');
    const direct = await import('../nn/dreams');
    expect(barrel.computeInputGradient).toBe(direct.computeInputGradient);
    expect(barrel.dream).toBe(direct.dream);
  });

  it('nn/index re-exports noise functions identical to direct', async () => {
    const barrel = await import('../nn/index');
    const direct = await import('../nn/noise');
    expect(barrel.generateNoisePattern).toBe(direct.generateNoisePattern);
    expect(barrel.applyNoise).toBe(direct.applyNoise);
  });

  it('nn/index re-exports sampleData functions identical to direct', async () => {
    const barrel = await import('../nn/index');
    const direct = await import('../nn/sampleData');
    expect(barrel.generateTrainingData).toBe(direct.generateTrainingData);
    expect(barrel.canvasToInput).toBe(direct.canvasToInput);
  });

  it('utils/index re-exports all utility functions', async () => {
    const barrel = await import('../utils/index');
    expect(typeof barrel.activate).toBe('function');
    expect(typeof barrel.activateDerivative).toBe('function');
    expect(typeof barrel.safeMax).toBe('function');
    expect(typeof barrel.argmax).toBe('function');
    expect(typeof barrel.softmax).toBe('function');
    expect(typeof barrel.xavierInit).toBe('function');
    expect(typeof barrel.mulberry32).toBe('function');
    expect(typeof barrel.gaussianNoise).toBe('function');
    expect(typeof barrel.getActivationColor).toBe('function');
    expect(typeof barrel.getWeightColor).toBe('function');
  });

  it('renderers/index re-exports pixelRendering functions', async () => {
    const barrel = await import('../renderers/index');
    const direct = await import('../renderers/pixelRendering');
    expect(barrel.weightsToImageData).toBe(direct.weightsToImageData);
    expect(barrel.pixelsToImageData).toBe(direct.pixelsToImageData);
    expect(barrel.lerpPixels).toBe(direct.lerpPixels);
  });

  it('renderers/index re-exports all renderer modules', async () => {
    const barrel = await import('../renderers/index');
    expect(typeof barrel.drawRaceChart).toBe('function');
    expect(typeof barrel.renderDreamImage).toBe('function');
    expect(typeof barrel.renderDreamGallery).toBe('function');
    expect(typeof barrel.drawSurgeryCanvas).toBe('function');
    expect(typeof barrel.computeSurgeryLayout).toBe('function');
    expect(typeof barrel.hitTestSurgeryNode).toBe('function');
    expect(typeof barrel.weightsToImageData).toBe('function');
    expect(typeof barrel.pixelsToImageData).toBe('function');
    expect(typeof barrel.lerpPixels).toBe('function');
  });

  it('visualizers/index re-exports networkLayout functions', async () => {
    const barrel = await import('../visualizers/index');
    const direct = await import('../visualizers/networkLayout');
    expect(barrel.computeNodePositions).toBe(direct.computeNodePositions);
    expect(barrel.generateParticles).toBe(direct.generateParticles);
    expect(barrel.getLayerSizes).toBe(direct.getLayerSizes);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. BACKWARD-COMPATIBILITY RE-EXPORTS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Backward-compatible re-exports', () => {
  it('src/noise.ts re-exports from nn/noise.ts', async () => {
    const compat = await import('../noise');
    const direct = await import('../nn/noise');
    expect(compat.generateNoisePattern).toBe(direct.generateNoisePattern);
    expect(compat.applyNoise).toBe(direct.applyNoise);
  });

  it('src/visualizer.ts re-exports from visualizers/networkLayout.ts', async () => {
    const compat = await import('../visualizer');
    const direct = await import('../visualizers/networkLayout');
    expect(compat.computeNodePositions).toBe(direct.computeNodePositions);
    expect(compat.generateParticles).toBe(direct.generateParticles);
    expect(compat.getLayerSizes).toBe(direct.getLayerSizes);
  });

  it('src/rendering.ts re-exports from renderers/pixelRendering.ts', async () => {
    const compat = await import('../rendering');
    const direct = await import('../renderers/pixelRendering');
    expect(compat.weightsToImageData).toBe(direct.weightsToImageData);
    expect(compat.pixelsToImageData).toBe(direct.pixelsToImageData);
    expect(compat.lerpPixels).toBe(direct.lerpPixels);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. MODULE BOUNDARY ENFORCEMENT
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Module boundaries', () => {
  it('nn/noise.ts imports from utils/ and types (not sibling nn/ modules)', () => {
    const src = fs.readFileSync(path.join(SRC, 'nn', 'noise.ts'), 'utf-8');
    // Should import from parent directories
    expect(src).toContain("from '../utils'");
    expect(src).toContain("from '../types'");
    expect(src).toContain("from '../constants'");
    // Should NOT import from sibling nn/ modules
    expect(src).not.toContain("from './NeuralNetwork'");
    expect(src).not.toContain("from './dreams'");
  });

  it('visualizers/networkLayout.ts imports from types and constants only', () => {
    const src = fs.readFileSync(path.join(SRC, 'visualizers', 'networkLayout.ts'), 'utf-8');
    expect(src).toContain("from '../types'");
    expect(src).toContain("from '../constants'");
    // Should NOT import from nn/, utils/, components/, etc.
    expect(src).not.toContain("from '../nn");
    expect(src).not.toContain("from '../components");
  });

  it('renderers/pixelRendering.ts imports from constants only', () => {
    const src = fs.readFileSync(path.join(SRC, 'renderers', 'pixelRendering.ts'), 'utf-8');
    expect(src).toContain("from '../constants'");
    // Pure rendering — no nn/ or React dependency
    expect(src).not.toContain("from '../nn");
    expect(src).not.toContain("from 'react'");
  });

  it('utils/ modules have no internal cross-dependencies', () => {
    const utilsDir = path.join(SRC, 'utils');
    const files = fs.readdirSync(utilsDir).filter(f => f.endsWith('.ts') && f !== 'index.ts');
    for (const file of files) {
      const src = fs.readFileSync(path.join(utilsDir, file), 'utf-8');
      // Utils should not import from other utils (except via types)
      const imports = [...src.matchAll(/from ['"]\.\/(\w+)['"]/g)].map(m => m[1]);
      const otherUtils = files.map(f => f.replace('.ts', '')).filter(f => f !== file.replace('.ts', ''));
      for (const imp of imports) {
        expect(
          otherUtils.includes(imp),
          `${file} imports sibling util ${imp} — utils should be independent`
        ).toBe(false);
      }
    }
  });

  it('nn/dreams.ts depends on NeuralNetwork type only (not class internals)', () => {
    const src = fs.readFileSync(path.join(SRC, 'nn', 'dreams.ts'), 'utf-8');
    expect(src).toContain("import type { NeuralNetwork }");
    // Uses public API methods only
    expect(src).toContain('.forward(');
    expect(src).toContain('.getLayers()');
    expect(src).toContain('.getConfig()');
  });

  it('data/ modules have no React imports', () => {
    const dataDir = path.join(SRC, 'data');
    const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.ts'));
    for (const file of files) {
      const src = fs.readFileSync(path.join(dataDir, file), 'utf-8');
      expect(src).not.toContain("from 'react'");
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. DIRECTORY STRUCTURE VERIFICATION
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Directory structure', () => {
  it('visualizers/ directory has expected files', () => {
    const dir = path.join(SRC, 'visualizers');
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.ts')).sort();
    expect(files).toEqual([
      'index.ts',
      'networkLayout.ts',
    ]);
  });

  it('components/ has barrel index.ts', () => {
    const idx = path.join(SRC, 'components', 'index.ts');
    expect(fs.existsSync(idx)).toBe(true);
    const src = fs.readFileSync(idx, 'utf-8');
    expect(src).toContain('StatsPanel');
    expect(src).toContain('HelpOverlay');
    expect(src).toContain('ExperiencePanel');
  });

  it('hooks/ has barrel index.ts', () => {
    const idx = path.join(SRC, 'hooks', 'index.ts');
    expect(fs.existsSync(idx)).toBe(true);
    const src = fs.readFileSync(idx, 'utf-8');
    expect(src).toContain('useNeuralNetwork');
    expect(src).toContain('useCinematic');
    expect(src).toContain('useContainerDims');
    expect(src).toContain('useTrainingRace');
  });

  it('all barrel index.ts files exist across all directories', () => {
    const dirs = ['nn', 'utils', 'renderers', 'visualizers', 'hooks', 'components'];
    for (const dir of dirs) {
      const idx = path.join(SRC, dir, 'index.ts');
      expect(fs.existsSync(idx), `${dir}/index.ts missing`).toBe(true);
    }
  });

  it('extracted components exist as standalone files', () => {
    const expected = ['StatsPanel.tsx', 'HelpOverlay.tsx', 'ExperiencePanel.tsx'];
    const compDir = path.join(SRC, 'components');
    for (const file of expected) {
      expect(fs.existsSync(path.join(compDir, file)), `${file} missing`).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. CROSS-MODULE INTEGRATION PIPELINES
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Cross-module integration', () => {
  it('full pipeline: NeuralNetwork → predict → pixelRendering', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn');
    const { weightsToImageData } = await import('../renderers/pixelRendering');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    nn.trainBatch(data.inputs, data.labels);

    const layers = nn.predict(new Array(784).fill(0.5)).layers;
    // First hidden layer weights can be rendered as feature maps
    const imgData = weightsToImageData(layers[0].weights[0], 28);
    expect(imgData.width).toBe(28);
    expect(imgData.height).toBe(28);
    expect(imgData.data.length).toBe(28 * 28 * 4);
  });

  it('full pipeline: NeuralNetwork → noise → predict', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn');
    const { generateNoisePattern, applyNoise } = await import('../nn/noise');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    nn.trainBatch(data.inputs, data.labels);

    const clean = new Array(784).fill(0.5);
    const pattern = generateNoisePattern('gaussian', 42);
    const noised = applyNoise(clean, pattern, 0.5, 'gaussian', 42);

    const cleanResult = nn.predict(clean);
    const noisedResult = nn.predict(noised);

    expect(cleanResult.probabilities).toHaveLength(10);
    expect(noisedResult.probabilities).toHaveLength(10);
  });

  it('full pipeline: NeuralNetwork → visualizer layout → particles', async () => {
    const { NeuralNetwork } = await import('../nn');
    const { computeNodePositions, generateParticles, getLayerSizes } = await import('../visualizers');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }, { neurons: 16, activation: 'relu' }],
    });
    const result = nn.predict(new Array(784).fill(0));
    const layers = result.layers;
    const sizes = getLayerSizes(layers, 16);

    expect(sizes).toHaveLength(4); // input + 2 hidden + output
    const positions = computeNodePositions(sizes, 600, 400, 50);
    expect(positions).toHaveLength(4);

    const particles = generateParticles(positions, sizes, layers);
    expect(particles.length).toBeGreaterThan(0);
    for (const p of particles) {
      expect(p.alive).toBe(true);
      expect(p.progress).toBe(0);
      expect(p.speed).toBeGreaterThan(0);
    }
  });

  it('full pipeline: NeuralNetwork → dream → pixelRendering', async () => {
    const { NeuralNetwork, generateTrainingData, dream } = await import('../nn');
    const { pixelsToImageData } = await import('../renderers/pixelRendering');

    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    nn.trainBatch(data.inputs, data.labels);

    const result = dream(nn, 5, 10, 0.5);
    expect(result.image).toHaveLength(784);
    expect(result.confidenceHistory.length).toBeGreaterThan(0);

    const imgData = pixelsToImageData(result.image, 56);
    expect(imgData.width).toBe(56);
    expect(imgData.data.length).toBe(56 * 56 * 4);
  });

  it('race presets all produce valid configs for NeuralNetwork', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    const { NeuralNetwork } = await import('../nn');
    const { INPUT_SIZE, OUTPUT_CLASSES } = await import('../constants');

    for (const preset of RACE_PRESETS) {
      const nnA = new NeuralNetwork(INPUT_SIZE, preset.a);
      const nnB = new NeuralNetwork(INPUT_SIZE, preset.b);

      const resA = nnA.predict(new Array(INPUT_SIZE).fill(0));
      const resB = nnB.predict(new Array(INPUT_SIZE).fill(0));

      expect(resA.probabilities).toHaveLength(OUTPUT_CLASSES);
      expect(resB.probabilities).toHaveLength(OUTPUT_CLASSES);
    }
  });

  it('lerpPixels correctly interpolates between two images', async () => {
    const { lerpPixels } = await import('../renderers/pixelRendering');

    const a = new Array(784).fill(0);
    const b = new Array(784).fill(1);

    const mid = lerpPixels(a, b, 0.5);
    expect(mid).toHaveLength(784);
    for (const v of mid) {
      expect(v).toBeCloseTo(0.5, 5);
    }

    const atA = lerpPixels(a, b, 0);
    for (const v of atA) expect(v).toBeCloseTo(0, 5);

    const atB = lerpPixels(a, b, 1);
    for (const v of atB) expect(v).toBeCloseTo(1, 5);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. CONSTANTS COMPLETENESS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Constants completeness', () => {
  it('all timing constants are positive numbers', async () => {
    const c = await import('../constants');
    const timingKeys = [
      'CINEMATIC_TRAIN_EPOCHS',
      'CINEMATIC_PREDICT_DWELL',
      'CINEMATIC_EPOCH_INTERVAL',
      'AUTO_TRAIN_EPOCHS',
      'AUTO_TRAIN_DELAY',
      'TRAINING_STEP_INTERVAL',
      'DREAM_STEPS',
      'DREAM_ANIMATION_INTERVAL',
      'RACE_EPOCHS',
      'RACE_STEP_INTERVAL',
    ] as const;

    for (const key of timingKeys) {
      const val = (c as Record<string, unknown>)[key];
      expect(typeof val).toBe('number');
      expect(val as number).toBeGreaterThan(0);
    }
  });

  it('display constants produce valid aspect ratios', async () => {
    const c = await import('../constants');
    expect(c.NETWORK_VIS_ASPECT).toBeGreaterThan(0);
    expect(c.NETWORK_VIS_ASPECT).toBeLessThan(2);
    expect(c.LOSS_CHART_ASPECT).toBeGreaterThan(0);
    expect(c.LOSS_CHART_ASPECT).toBeLessThan(1);
    expect(c.ACTIVATION_VIS_ASPECT).toBeGreaterThan(0);
    expect(c.ACTIVATION_VIS_ASPECT).toBeLessThan(2);
  });

  it('INPUT_SIZE equals INPUT_DIM squared', async () => {
    const { INPUT_SIZE, INPUT_DIM } = await import('../constants');
    expect(INPUT_SIZE).toBe(INPUT_DIM * INPUT_DIM);
  });

  it('NEURON_OPTIONS are sorted ascending and all positive', async () => {
    const { NEURON_OPTIONS } = await import('../constants');
    for (let i = 0; i < NEURON_OPTIONS.length; i++) {
      expect(NEURON_OPTIONS[i]).toBeGreaterThan(0);
      if (i > 0) expect(NEURON_OPTIONS[i]).toBeGreaterThan(NEURON_OPTIONS[i - 1]);
    }
  });

  it('SHORTCUTS have unique keys and non-empty descriptions', async () => {
    const { SHORTCUTS } = await import('../constants');
    expect(SHORTCUTS.length).toBeGreaterThanOrEqual(4);
    const keys = SHORTCUTS.map(s => s.key);
    expect(new Set(keys).size).toBe(keys.length);
    for (const s of SHORTCUTS) {
      expect(s.key.length).toBeGreaterThan(0);
      expect(s.description.length).toBeGreaterThan(0);
    }
  });

  it('NOISE_LABELS and NOISE_DESCRIPTIONS cover all NoiseType values', async () => {
    const { NOISE_LABELS, NOISE_DESCRIPTIONS } = await import('../constants');
    const expected: string[] = ['gaussian', 'salt-pepper', 'adversarial'];
    for (const key of expected) {
      expect(NOISE_LABELS[key as keyof typeof NOISE_LABELS]).toBeDefined();
      expect(NOISE_DESCRIPTIONS[key as keyof typeof NOISE_DESCRIPTIONS]).toBeDefined();
    }
  });

  it('DEFAULT_CONFIG produces a valid NeuralNetwork', async () => {
    const { DEFAULT_CONFIG, INPUT_SIZE, OUTPUT_CLASSES } = await import('../constants');
    const { NeuralNetwork } = await import('../nn');

    expect(DEFAULT_CONFIG.learningRate).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.layers.length).toBeGreaterThan(0);

    const nn = new NeuralNetwork(INPUT_SIZE, DEFAULT_CONFIG);
    const result = nn.predict(new Array(INPUT_SIZE).fill(0));
    expect(result.probabilities).toHaveLength(OUTPUT_CLASSES);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7. TYPE SYSTEM CONSISTENCY
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Type system consistency', () => {
  it('all ActivationFn values are handled by activate()', async () => {
    const { activate, activateDerivative } = await import('../utils/activations');
    const fns: Array<'relu' | 'sigmoid' | 'tanh'> = ['relu', 'sigmoid', 'tanh'];
    for (const fn of fns) {
      const val = activate(0.5, fn);
      expect(isFinite(val)).toBe(true);
      const deriv = activateDerivative(0.5, fn);
      expect(isFinite(deriv)).toBe(true);
    }
  });

  it('NeuralNetwork public API completeness', async () => {
    const { NeuralNetwork } = await import('../nn');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });

    // Training API
    expect(typeof nn.trainBatch).toBe('function');
    expect(typeof nn.predict).toBe('function');
    expect(typeof nn.forward).toBe('function');
    expect(typeof nn.reset).toBe('function');

    // Surgery API
    expect(typeof nn.setNeuronStatus).toBe('function');
    expect(typeof nn.getNeuronStatus).toBe('function');
    expect(typeof nn.getAllNeuronStatuses).toBe('function');
    expect(typeof nn.clearAllMasks).toBe('function');

    // Accessor API
    expect(typeof nn.getLossHistory).toBe('function');
    expect(typeof nn.getAccuracyHistory).toBe('function');
    expect(typeof nn.getEpoch).toBe('function');
    expect(typeof nn.getLayers).toBe('function');
    expect(typeof nn.getNeuronMasks).toBe('function');
    expect(typeof nn.getConfig).toBe('function');
    expect(typeof nn.snapshotLayers).toBe('function');

    // Dream delegation API
    expect(typeof nn.computeInputGradient).toBe('function');
    expect(typeof nn.dream).toBe('function');
  });

  it('TrainingSnapshot has all required fields', async () => {
    const { NeuralNetwork, generateTrainingData } = await import('../nn');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    const snapshot = nn.trainBatch(data.inputs, data.labels);

    expect(typeof snapshot.epoch).toBe('number');
    expect(typeof snapshot.loss).toBe('number');
    expect(typeof snapshot.accuracy).toBe('number');
    expect(Array.isArray(snapshot.layers)).toBe(true);
    expect(Array.isArray(snapshot.predictions)).toBe(true);
    expect(Array.isArray(snapshot.outputProbabilities)).toBe(true);
    expect(snapshot.outputProbabilities).toHaveLength(10);
  });

  it('DreamResult has required fields', async () => {
    const { NeuralNetwork, generateTrainingData, dream } = await import('../nn');
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const data = generateTrainingData(3);
    nn.trainBatch(data.inputs, data.labels);

    const result = dream(nn, 3, 5, 0.5);
    expect(Array.isArray(result.image)).toBe(true);
    expect(result.image).toHaveLength(784);
    expect(Array.isArray(result.confidenceHistory)).toBe(true);
    expect(result.confidenceHistory).toHaveLength(5);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 8. EXTRACTED COMPONENT CONTRACTS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Extracted component contracts', () => {
  it('StatsPanel has expected prop interface', () => {
    const src = fs.readFileSync(path.join(SRC, 'components', 'StatsPanel.tsx'), 'utf-8');
    expect(src).toContain('epoch: number');
    expect(src).toContain('isTraining: boolean');
    expect(src).toContain('loss: number | null');
    expect(src).toContain('accuracy: number | null');
    expect(src).toContain('predictedLabel: number | null');
    expect(src).toContain('export default function StatsPanel');
  });

  it('HelpOverlay has expected prop interface', () => {
    const src = fs.readFileSync(path.join(SRC, 'components', 'HelpOverlay.tsx'), 'utf-8');
    expect(src).toContain('onClose: () => void');
    expect(src).toContain("from '../constants'");
    expect(src).toContain('SHORTCUTS');
    expect(src).toContain('export default function HelpOverlay');
  });

  it('ExperiencePanel has expected prop interface', () => {
    const src = fs.readFileSync(path.join(SRC, 'components', 'ExperiencePanel.tsx'), 'utf-8');
    expect(src).toContain('cinematicActive: boolean');
    expect(src).toContain('onStartCinematic: () => void');
    expect(src).toContain('export default function ExperiencePanel');
  });

  it('App.tsx imports the 3 extracted components', () => {
    const src = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    expect(src).toContain("from './components/StatsPanel'");
    expect(src).toContain("from './components/HelpOverlay'");
    expect(src).toContain("from './components/ExperiencePanel'");
  });

  it('App.tsx is slimmer after extraction (no inline stats/help JSX)', () => {
    const src = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    // The inline stats-grid and help-list should be gone
    expect(src).not.toContain('stats-grid');
    expect(src).not.toContain('help-list');
    expect(src).not.toContain('help-row');
    // Should use the extracted components
    expect(src).toContain('<StatsPanel');
    expect(src).toContain('<HelpOverlay');
    expect(src).toContain('<ExperiencePanel');
  });
});

// ═══════════════════════════════════════════════════════════════════
// 9. CODE QUALITY CHECKS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat #4 — Code quality', () => {
  it('no as-any casts in source files (excluding tests)', () => {
    const srcFiles = getAllSourceFiles(SRC);
    for (const file of srcFiles) {
      if (file.includes('__tests__')) continue;
      const src = fs.readFileSync(file, 'utf-8');
      expect(src).not.toContain('as any');
    }
  });

  it('no TODO/FIXME/HACK in source files', () => {
    const srcFiles = getAllSourceFiles(SRC);
    for (const file of srcFiles) {
      if (file.includes('__tests__')) continue;
      const src = fs.readFileSync(file, 'utf-8');
      const lower = src.toLowerCase();
      // Allow "TODO" in comments about what's NOT there
      const lines = src.split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].toLowerCase();
        if (line.includes('todo') || line.includes('fixme') || line.includes('hack')) {
          // Ignore lines that are part of test descriptions or string literals
          if (lines[i].includes("'") || lines[i].includes('"')) continue;
          throw new Error(`${file}:${i + 1} contains TODO/FIXME/HACK`);
        }
      }
    }
  });

  it('ErrorBoundary component exists', () => {
    const file = path.join(SRC, 'components', 'ErrorBoundary.tsx');
    expect(fs.existsSync(file)).toBe(true);
    const src = fs.readFileSync(file, 'utf-8');
    expect(src).toContain('componentDidCatch');
  });

  it('all backward-compat files are thin re-exports (< 15 lines)', () => {
    const compatFiles = ['noise.ts', 'visualizer.ts', 'rendering.ts'];
    for (const file of compatFiles) {
      const src = fs.readFileSync(path.join(SRC, file), 'utf-8');
      const lines = src.split('\n').filter(l => l.trim().length > 0);
      expect(lines.length, `${file} should be a thin re-export`).toBeLessThanOrEqual(15);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

function getAllSourceFiles(dir: string): string[] {
  const files: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
      files.push(...getAllSourceFiles(fullPath));
    } else if (entry.isFile() && (entry.name.endsWith('.ts') || entry.name.endsWith('.tsx'))) {
      files.push(fullPath);
    }
  }
  return files;
}

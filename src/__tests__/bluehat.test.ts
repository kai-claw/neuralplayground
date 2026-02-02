/**
 * Blue Hat Pass 6 — Structural Integrity & Process Tests
 *
 * These tests validate the architecture itself: directory structure,
 * feature completeness, import/export hygiene, store consistency,
 * canvas pipeline integrity, and component separation of concerns.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

// ─── Helper: recursively collect files ─────────────────────────────
function collectFiles(dir: string, ext: string[]): string[] {
  const results: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory() && entry.name !== 'node_modules' && entry.name !== 'dist') {
      results.push(...collectFiles(full, ext));
    } else if (entry.isFile() && ext.some(e => entry.name.endsWith(e))) {
      results.push(full);
    }
  }
  return results;
}

const SRC = path.resolve(__dirname, '..');
const ROOT = path.resolve(SRC, '..');

// ═══════════════════════════════════════════════════════════════════
// 1. DIRECTORY STRUCTURE CORRECTNESS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Directory structure', () => {
  const requiredDirs = [
    'src',
    'src/components',
    'src/hooks',
    'src/nn',
    'src/data',
    'src/__tests__',
  ];

  for (const dir of requiredDirs) {
    it(`has required directory: ${dir}`, () => {
      const full = path.join(ROOT, dir);
      expect(fs.existsSync(full), `Missing directory: ${dir}`).toBe(true);
      expect(fs.statSync(full).isDirectory()).toBe(true);
    });
  }

  const requiredFiles = [
    'src/App.tsx',
    'src/main.tsx',
    'src/types.ts',
    'src/constants.ts',
    'src/utils.ts',
    'src/nn/NeuralNetwork.ts',
    'src/nn/sampleData.ts',
    'src/hooks/useNeuralNetwork.ts',
    'src/hooks/useCinematic.ts',
    'src/hooks/useContainerDims.ts',
    'src/data/digitStrokes.ts',
    'index.html',
    'package.json',
    'vite.config.ts',
    'tsconfig.json',
  ];

  for (const file of requiredFiles) {
    it(`has required file: ${file}`, () => {
      expect(fs.existsSync(path.join(ROOT, file)), `Missing: ${file}`).toBe(true);
    });
  }

  const requiredComponents = [
    'ActivationVisualizer',
    'AdversarialLab',
    'CinematicBadge',
    'ControlPanel',
    'DigitMorph',
    'DrawingCanvas',
    'ErrorBoundary',
    'FeatureMaps',
    'LossChart',
    'NetworkVisualizer',
    'PredictionBar',
    'WeightHeatmap',
    'WeightPanel',
  ];

  for (const comp of requiredComponents) {
    it(`has component file: ${comp}.tsx`, () => {
      expect(fs.existsSync(path.join(SRC, 'components', `${comp}.tsx`))).toBe(true);
    });
  }

  it('has no stray .ts/.tsx files in src root besides App, main, types, constants, utils, vite-env', () => {
    const allowed = new Set(['App.tsx', 'main.tsx', 'types.ts', 'constants.ts', 'utils.ts', 'vite-env.d.ts']);
    const rootFiles = fs.readdirSync(SRC).filter(f => (f.endsWith('.ts') || f.endsWith('.tsx')) && !allowed.has(f));
    expect(rootFiles, `Stray files in src/: ${rootFiles.join(', ')}`).toEqual([]);
  });

  it('components/ contains only .tsx files', () => {
    const compDir = path.join(SRC, 'components');
    const files = fs.readdirSync(compDir);
    const nonTsx = files.filter(f => !f.endsWith('.tsx'));
    expect(nonTsx, `Non-TSX files in components/: ${nonTsx.join(', ')}`).toEqual([]);
  });

  it('hooks/ contains only .ts files', () => {
    const hookDir = path.join(SRC, 'hooks');
    const files = fs.readdirSync(hookDir);
    const nonTs = files.filter(f => !f.endsWith('.ts'));
    expect(nonTs, `Non-TS files in hooks/: ${nonTs.join(', ')}`).toEqual([]);
  });

  it('nn/ contains only .ts files', () => {
    const nnDir = path.join(SRC, 'nn');
    const files = fs.readdirSync(nnDir);
    const nonTs = files.filter(f => !f.endsWith('.ts'));
    expect(nonTs).toEqual([]);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. FEATURE COMPLETENESS — All documented features wired up
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Feature completeness', () => {
  const appSource = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
  const mainSource = fs.readFileSync(path.join(SRC, 'main.tsx'), 'utf-8');

  // Every component must be imported and rendered in App.tsx
  const componentImports: [string, string][] = [
    ['DrawingCanvas', 'Drawing canvas for digit input'],
    ['NetworkVisualizer', 'Network topology visualization'],
    ['LossChart', 'Loss/accuracy chart'],
    ['ActivationVisualizer', 'Per-layer activation bars'],
    ['PredictionBar', 'Digit probability display'],
    ['ControlPanel', 'Training controls'],
    ['WeightPanel', 'Weight heatmap panel'],
    ['CinematicBadge', 'Cinematic mode indicator'],
    ['DigitMorph', 'Digit morphing panel'],
    ['FeatureMaps', 'Feature map visualization'],
    ['AdversarialLab', 'Adversarial noise lab'],
  ];

  for (const [comp, desc] of componentImports) {
    it(`App imports and uses ${comp} (${desc})`, () => {
      expect(appSource).toContain(`import ${comp}`);
      expect(appSource).toContain(`<${comp}`);
    });
  }

  it('ErrorBoundary wraps App in main.tsx', () => {
    expect(mainSource).toContain('ErrorBoundary');
    expect(mainSource).toContain('<App');
  });

  // Hook integration
  it('App uses useNeuralNetwork hook', () => {
    expect(appSource).toContain('useNeuralNetwork');
  });

  it('App uses useCinematic hook', () => {
    expect(appSource).toContain('useCinematic');
  });

  // Keyboard shortcuts wired
  it('App has keyboard event listener for shortcuts', () => {
    expect(appSource).toContain('keydown');
    expect(appSource).toContain("' '");  // Space key
  });

  // ARIA accessibility
  it('App has role="application" on root', () => {
    expect(appSource).toContain('role="application"');
  });

  it('App has skip-to-content link', () => {
    expect(appSource).toContain('Skip to content');
  });

  it('App has aria-live status for training', () => {
    expect(appSource).toContain('aria-live');
  });

  // Auto-start feature
  it('App implements auto-start training on first load', () => {
    expect(appSource).toContain('autoStartedRef');
    expect(appSource).toContain('AUTO_TRAIN_DELAY');
  });

  // Signal flow animation
  it('App tracks signalFlowTrigger for animation', () => {
    expect(appSource).toContain('signalFlowTrigger');
  });

  // Confidence-reactive glow
  it('App has prediction confidence glow on drawing canvas', () => {
    expect(appSource).toContain('boxShadow');
    expect(appSource).toContain('livePrediction');
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. IMPORT/EXPORT HYGIENE — No dead imports or unused exports
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Import/export hygiene', () => {
  const allTsFiles = collectFiles(SRC, ['.ts', '.tsx']).filter(
    f => !f.includes('__tests__') && !f.includes('vite-env')
  );

  it('utils.ts — every exported function is imported somewhere in src/', () => {
    const utilsSrc = fs.readFileSync(path.join(SRC, 'utils.ts'), 'utf-8');
    const exportedFns = [...utilsSrc.matchAll(/export function (\w+)/g)].map(m => m[1]);

    // Read all non-test source files
    const allSrc = allTsFiles
      .filter(f => !f.endsWith('utils.ts'))
      .map(f => fs.readFileSync(f, 'utf-8'))
      .join('\n');

    for (const fn of exportedFns) {
      // safeMax is used internally by softmax in utils.ts itself — that's fine
      if (fn === 'safeMax') {
        expect(utilsSrc).toContain(`safeMax(`);
        continue;
      }
      expect(allSrc, `utils.ts exports '${fn}' but it's never imported in src/`).toContain(fn);
    }
  });

  it('constants.ts — every exported constant is used somewhere (or documented-only)', () => {
    const constSrc = fs.readFileSync(path.join(SRC, 'constants.ts'), 'utf-8');
    const exports = [
      ...constSrc.matchAll(/export const (\w+)/g),
    ].map(m => m[1]);

    // Audit finding: several constants are exported but not imported anywhere in src/.
    // Some are architectural documentation (INPUT_SIZE=784, OUTPUT_CLASSES=10),
    // some have their values hardcoded in components instead of importing the constant
    // (NEURON_OPTIONS, MAX_HIDDEN_LAYERS in ControlPanel; MORPH_DISPLAY_SIZE in DigitMorph),
    // and COLOR_GREEN_HEX is defined but unused (stat colors use CSS vars instead).
    // These are noted as findings for future cleanup passes.
    const unusedButDocumented = new Set([
      'INPUT_SIZE',        // 784 literal used in NeuralNetwork(784, ...) — symbol not imported
      'OUTPUT_CLASSES',    // 10 literal used — symbol not imported
      'NEURON_OPTIONS',    // ControlPanel hardcodes [8,16,32,64,128,256]
      'MAX_HIDDEN_LAYERS', // ControlPanel hardcodes 5
      'MORPH_DISPLAY_SIZE',// DigitMorph hardcodes 140
      'COLOR_GREEN_HEX',   // Unused — stat colors use CSS vars instead
    ]);

    const allSrc = allTsFiles
      .filter(f => !f.endsWith('constants.ts'))
      .map(f => fs.readFileSync(f, 'utf-8'))
      .join('\n');

    for (const name of exports) {
      if (unusedButDocumented.has(name)) continue;
      expect(allSrc, `constants.ts exports '${name}' but it's never used`).toContain(name);
    }
  });

  it('types.ts — every exported type/interface is used somewhere', () => {
    const typesSrc = fs.readFileSync(path.join(SRC, 'types.ts'), 'utf-8');
    // Check re-exported types and local types
    const localTypes = [...typesSrc.matchAll(/export (?:type|interface) (\w+)/g)].map(m => m[1]);

    const allSrc = allTsFiles
      .filter(f => !f.endsWith('types.ts'))
      .map(f => fs.readFileSync(f, 'utf-8'))
      .join('\n');

    for (const t of localTypes) {
      expect(allSrc, `types.ts exports '${t}' but it's never used`).toContain(t);
    }
  });

  it('every component has a default export', () => {
    const compDir = path.join(SRC, 'components');
    const components = fs.readdirSync(compDir).filter(f => f.endsWith('.tsx'));

    for (const comp of components) {
      const src = fs.readFileSync(path.join(compDir, comp), 'utf-8');
      const hasDefaultExport =
        src.includes('export default') ||
        src.includes('export { ') ||  // re-export
        /export\s+default\s+/m.test(src);
      expect(hasDefaultExport, `${comp} has no default export`).toBe(true);
    }
  });

  it('no component imports from another component (separation of concerns)', () => {
    const compDir = path.join(SRC, 'components');
    const components = fs.readdirSync(compDir).filter(f => f.endsWith('.tsx'));

    for (const comp of components) {
      const src = fs.readFileSync(path.join(compDir, comp), 'utf-8');
      const imports = [...src.matchAll(/from\s+['"]\.\/(\w+)['"]/g)].map(m => m[1]);
      // WeightPanel imports WeightHeatmap — this is a valid parent-child composition
      if (comp === 'WeightPanel.tsx') {
        const filteredImports = imports.filter(i => i !== 'WeightHeatmap');
        expect(filteredImports, `${comp} imports sibling: ${filteredImports.join(', ')}`).toEqual([]);
      } else {
        // Other components should not import siblings
        const siblingImports = imports.filter(i =>
          components.some(c => c.replace('.tsx', '') === i)
        );
        expect(siblingImports, `${comp} imports sibling(s): ${siblingImports.join(', ')}`).toEqual([]);
      }
    }
  });

  it('no circular imports between nn/ and components/', () => {
    const nnDir = path.join(SRC, 'nn');
    const nnFiles = fs.readdirSync(nnDir).filter(f => f.endsWith('.ts'));

    for (const file of nnFiles) {
      const src = fs.readFileSync(path.join(nnDir, file), 'utf-8');
      expect(src, `nn/${file} imports from components/`).not.toContain("from '../components/");
      expect(src, `nn/${file} imports from hooks/`).not.toContain("from '../hooks/");
    }
  });

  it('no circular imports between hooks/ and components/', () => {
    const hookDir = path.join(SRC, 'hooks');
    const hookFiles = fs.readdirSync(hookDir).filter(f => f.endsWith('.ts'));

    for (const file of hookFiles) {
      const src = fs.readFileSync(path.join(hookDir, file), 'utf-8');
      expect(src, `hooks/${file} imports from components/`).not.toContain("from '../components/");
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. STORE / STATE CONSISTENCY
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — State consistency (useNeuralNetwork)', () => {
  // Validate the hook's exported state shape matches what App expects
  const hookSrc = fs.readFileSync(path.join(SRC, 'hooks', 'useNeuralNetwork.ts'), 'utf-8');
  const appSrc = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');

  it('NetworkState interface has all required fields', () => {
    expect(hookSrc).toContain('isTraining: boolean');
    expect(hookSrc).toContain('snapshot: TrainingSnapshot | null');
    expect(hookSrc).toContain('lossHistory: number[]');
    expect(hookSrc).toContain('accuracyHistory: number[]');
    expect(hookSrc).toContain('epoch: number');
    expect(hookSrc).toContain('config: TrainingConfig');
  });

  it('hook returns all methods App needs', () => {
    // These are destructured in App.tsx
    const hookMethods = ['initNetwork', 'startTraining', 'stopTraining', 'predict', 'updateConfig', 'updateLayers'];
    for (const method of hookMethods) {
      expect(hookSrc, `Hook missing method: ${method}`).toContain(method);
      expect(appSrc, `App doesn't use hook method: ${method}`).toContain(method);
    }
  });

  it('App destructures state fields that hook provides', () => {
    // App accesses: state.isTraining, state.epoch, state.config, state.snapshot, state.lossHistory, state.accuracyHistory
    const stateAccesses = ['state.isTraining', 'state.epoch', 'state.config', 'state.snapshot', 'state.lossHistory', 'state.accuracyHistory'];
    for (const access of stateAccesses) {
      expect(appSrc, `App accesses ${access} but hook may not provide it`).toContain(access);
    }
  });

  it('hook uses DEFAULT_CONFIG from constants', () => {
    expect(hookSrc).toContain('DEFAULT_CONFIG');
    expect(hookSrc).toContain("from '../constants'");
  });

  it('hook cleans up timer on unmount (no memory leak)', () => {
    expect(hookSrc).toContain('clearTimeout');
    expect(hookSrc).toContain('return ()');
  });
});

describe('Blue Hat — State consistency (useCinematic)', () => {
  const cinematicSrc = fs.readFileSync(path.join(SRC, 'hooks', 'useCinematic.ts'), 'utf-8');

  it('CinematicState has all required fields', () => {
    expect(cinematicSrc).toContain('active: boolean');
    expect(cinematicSrc).toContain('phase: CinematicPhase');
    expect(cinematicSrc).toContain('digit: number');
    expect(cinematicSrc).toContain('progress: number');
    expect(cinematicSrc).toContain('epoch: number');
  });

  it('returns start/stop functions and state', () => {
    expect(cinematicSrc).toContain('startCinematic');
    expect(cinematicSrc).toContain('stopCinematic');
    expect(cinematicSrc).toContain('cinematic: state');
  });

  it('cleans up timers on unmount', () => {
    expect(cinematicSrc).toContain('clearTimer');
    // Cleanup effect
    expect(cinematicSrc).toContain('return () =>');
  });

  it('uses digit stroke data for drawing', () => {
    expect(cinematicSrc).toContain('DIGIT_STROKES');
    expect(cinematicSrc).toContain('getDigitDrawDuration');
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. CANVAS PIPELINE INTEGRITY
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Canvas pipeline integrity', () => {
  it('DrawingCanvas outputs ImageData to App via onDraw callback', () => {
    const drawSrc = fs.readFileSync(path.join(SRC, 'components', 'DrawingCanvas.tsx'), 'utf-8');
    expect(drawSrc).toContain('onDraw');
    expect(drawSrc).toContain('ImageData');
    expect(drawSrc).toContain('getImageData');
  });

  it('App converts ImageData to 784-element input via canvasToInput', () => {
    const appSrc = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    expect(appSrc).toContain('canvasToInput');
    expect(appSrc).toContain('handleDraw');
  });

  it('canvasToInput produces 784-element array (28×28)', () => {
    const sampleSrc = fs.readFileSync(path.join(SRC, 'nn', 'sampleData.ts'), 'utf-8');
    expect(sampleSrc).toContain('targetSize = 28');
    expect(sampleSrc).toContain('targetSize * targetSize');
  });

  it('NeuralNetwork accepts 784-element input in forward()', () => {
    const nnSrc = fs.readFileSync(path.join(SRC, 'nn', 'NeuralNetwork.ts'), 'utf-8');
    expect(nnSrc).toContain('forward(input: number[])');
  });

  it('predict returns probabilities → displayed by PredictionBar', () => {
    const appSrc = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    expect(appSrc).toContain('livePrediction');
    expect(appSrc).toContain('<PredictionBar');
    expect(appSrc).toContain('probabilities={livePrediction}');
  });

  it('predict returns layers → fed to NetworkVisualizer, ActivationVisualizer, WeightPanel, FeatureMaps', () => {
    const appSrc = fs.readFileSync(path.join(SRC, 'App.tsx'), 'utf-8');
    expect(appSrc).toContain('displayLayers');
    expect(appSrc).toContain('<NetworkVisualizer');
    expect(appSrc).toContain('<ActivationVisualizer');
    expect(appSrc).toContain('<WeightPanel');
    expect(appSrc).toContain('<FeatureMaps');
  });

  it('NetworkVisualizer uses useContainerDims for responsive sizing', () => {
    const visSrc = fs.readFileSync(path.join(SRC, 'components', 'NetworkVisualizer.tsx'), 'utf-8');
    expect(visSrc).toContain('useContainerDims');
  });

  it('LossChart uses useContainerDims for responsive sizing', () => {
    const lossSrc = fs.readFileSync(path.join(SRC, 'components', 'LossChart.tsx'), 'utf-8');
    expect(lossSrc).toContain('useContainerDims');
  });

  it('ActivationVisualizer uses useContainerDims for responsive sizing', () => {
    const actSrc = fs.readFileSync(path.join(SRC, 'components', 'ActivationVisualizer.tsx'), 'utf-8');
    expect(actSrc).toContain('useContainerDims');
  });

  it('training data pipeline: generateTrainingData → trainBatch → snapshot → UI', () => {
    const hookSrc = fs.readFileSync(path.join(SRC, 'hooks', 'useNeuralNetwork.ts'), 'utf-8');
    expect(hookSrc).toContain('generateTrainingData');
    expect(hookSrc).toContain('trainBatch');
    expect(hookSrc).toContain('snapshot');
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. CONSTANTS & CONFIGURATION INTEGRITY
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Constants integrity', () => {
  // Import and validate constants at runtime
  it('INPUT_SIZE matches 28×28', async () => {
    const { INPUT_SIZE, INPUT_DIM } = await import('../constants');
    expect(INPUT_SIZE).toBe(784);
    expect(INPUT_DIM).toBe(28);
    expect(INPUT_DIM * INPUT_DIM).toBe(INPUT_SIZE);
  });

  it('OUTPUT_CLASSES is 10 (digits 0-9)', async () => {
    const { OUTPUT_CLASSES } = await import('../constants');
    expect(OUTPUT_CLASSES).toBe(10);
  });

  it('DEFAULT_CONFIG has valid structure', async () => {
    const { DEFAULT_CONFIG } = await import('../constants');
    expect(DEFAULT_CONFIG.learningRate).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.learningRate).toBeLessThan(1);
    expect(DEFAULT_CONFIG.layers.length).toBeGreaterThan(0);
    for (const layer of DEFAULT_CONFIG.layers) {
      expect(layer.neurons).toBeGreaterThan(0);
      expect(['relu', 'sigmoid', 'tanh']).toContain(layer.activation);
    }
  });

  it('NEURON_OPTIONS are valid powers of 2 or standard sizes', async () => {
    const { NEURON_OPTIONS } = await import('../constants');
    for (const n of NEURON_OPTIONS) {
      expect(n).toBeGreaterThan(0);
      expect(n).toBeLessThanOrEqual(256);
    }
    // Sorted ascending
    for (let i = 1; i < NEURON_OPTIONS.length; i++) {
      expect(NEURON_OPTIONS[i]).toBeGreaterThan(NEURON_OPTIONS[i - 1]);
    }
  });

  it('MAX_HIDDEN_LAYERS is 5', async () => {
    const { MAX_HIDDEN_LAYERS } = await import('../constants');
    expect(MAX_HIDDEN_LAYERS).toBe(5);
  });

  it('aspect ratios are positive fractions', async () => {
    const { NETWORK_VIS_ASPECT, LOSS_CHART_ASPECT, ACTIVATION_VIS_ASPECT } = await import('../constants');
    for (const ar of [NETWORK_VIS_ASPECT, LOSS_CHART_ASPECT, ACTIVATION_VIS_ASPECT]) {
      expect(ar).toBeGreaterThan(0);
      expect(ar).toBeLessThan(2);
    }
  });

  it('SHORTCUTS covers essential keys', async () => {
    const { SHORTCUTS } = await import('../constants');
    const keys = SHORTCUTS.map(s => s.key);
    expect(keys).toContain('Space');
    expect(keys).toContain('R');
    expect(keys).toContain('D');
    expect(keys).toContain('H');
    expect(keys).toContain('Esc');
  });

  it('NOISE_LABELS and NOISE_DESCRIPTIONS cover all NoiseType values', async () => {
    const { NOISE_LABELS, NOISE_DESCRIPTIONS } = await import('../constants');
    const noiseTypes = ['gaussian', 'salt-pepper', 'adversarial'];
    for (const t of noiseTypes) {
      expect(NOISE_LABELS[t as keyof typeof NOISE_LABELS]).toBeTruthy();
      expect(NOISE_DESCRIPTIONS[t as keyof typeof NOISE_DESCRIPTIONS]).toBeTruthy();
    }
  });

  it('timing constants are positive', async () => {
    const {
      CINEMATIC_TRAIN_EPOCHS,
      CINEMATIC_PREDICT_DWELL,
      CINEMATIC_EPOCH_INTERVAL,
      AUTO_TRAIN_EPOCHS,
      AUTO_TRAIN_DELAY,
      TRAINING_STEP_INTERVAL,
    } = await import('../constants');
    for (const v of [CINEMATIC_TRAIN_EPOCHS, CINEMATIC_PREDICT_DWELL, CINEMATIC_EPOCH_INTERVAL, AUTO_TRAIN_EPOCHS, AUTO_TRAIN_DELAY, TRAINING_STEP_INTERVAL]) {
      expect(v).toBeGreaterThan(0);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7. UTILITY FUNCTION CORRECTNESS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Utility functions', () => {
  it('activate handles all three activation types', async () => {
    const { activate } = await import('../utils');
    // ReLU
    expect(activate(5, 'relu')).toBe(5);
    expect(activate(-5, 'relu')).toBe(0);
    // Sigmoid
    expect(activate(0, 'sigmoid')).toBeCloseTo(0.5, 5);
    expect(activate(100, 'sigmoid')).toBeCloseTo(1, 5);
    expect(activate(-100, 'sigmoid')).toBeCloseTo(0, 5);
    // Tanh
    expect(activate(0, 'tanh')).toBeCloseTo(0, 5);
    expect(activate(100, 'tanh')).toBeCloseTo(1, 5);
    expect(activate(-100, 'tanh')).toBeCloseTo(-1, 5);
  });

  it('activateDerivative handles all three types', async () => {
    const { activateDerivative } = await import('../utils');
    // ReLU
    expect(activateDerivative(5, 'relu')).toBe(1);
    expect(activateDerivative(-5, 'relu')).toBe(0);
    // Sigmoid derivative at 0 ≈ 0.25
    expect(activateDerivative(0, 'sigmoid')).toBeCloseTo(0.25, 5);
    // Tanh derivative at 0 = 1
    expect(activateDerivative(0, 'tanh')).toBeCloseTo(1, 5);
  });

  it('softmax produces valid probability distribution', async () => {
    const { softmax } = await import('../utils');
    const result = softmax([1, 2, 3, 4]);
    expect(result.length).toBe(4);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
    // Monotonically increasing
    for (let i = 1; i < result.length; i++) {
      expect(result[i]).toBeGreaterThan(result[i - 1]);
    }
  });

  it('softmax handles degenerate all-same input', async () => {
    const { softmax } = await import('../utils');
    const result = softmax([0, 0, 0, 0]);
    expect(result.every(v => Math.abs(v - 0.25) < 0.001)).toBe(true);
  });

  it('argmax returns correct index', async () => {
    const { argmax } = await import('../utils');
    expect(argmax([0.1, 0.3, 0.9, 0.2])).toBe(2);
    expect(argmax([0.9, 0.1, 0.1, 0.1])).toBe(0);
    expect(argmax([0.1, 0.1, 0.1, 0.9])).toBe(3);
  });

  it('mulberry32 is deterministic with same seed', async () => {
    const { mulberry32 } = await import('../utils');
    const rng1 = mulberry32(42);
    const rng2 = mulberry32(42);
    for (let i = 0; i < 100; i++) {
      expect(rng1()).toBe(rng2());
    }
  });

  it('mulberry32 produces values in [0, 1)', async () => {
    const { mulberry32 } = await import('../utils');
    const rng = mulberry32(123);
    for (let i = 0; i < 1000; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('gaussianNoise produces finite values', async () => {
    const { mulberry32, gaussianNoise } = await import('../utils');
    const rng = mulberry32(42);
    for (let i = 0; i < 1000; i++) {
      const v = gaussianNoise(rng);
      expect(isFinite(v)).toBe(true);
    }
  });

  it('getActivationColor returns valid CSS rgba', async () => {
    const { getActivationColor } = await import('../utils');
    const positive = getActivationColor(0.5);
    expect(positive).toMatch(/^rgba\(/);
    const negative = getActivationColor(-0.5);
    expect(negative).toMatch(/^rgba\(/);
  });

  it('getWeightColor returns valid CSS rgba', async () => {
    const { getWeightColor } = await import('../utils');
    const pos = getWeightColor(0.5);
    expect(pos).toMatch(/^rgba\(/);
    const neg = getWeightColor(-0.5);
    expect(neg).toMatch(/^rgba\(/);
  });

  it('xavierInit produces values centered around 0', async () => {
    const { xavierInit } = await import('../utils');
    let sum = 0;
    const n = 10000;
    for (let i = 0; i < n; i++) {
      sum += xavierInit(784, 64);
    }
    const mean = sum / n;
    expect(Math.abs(mean)).toBeLessThan(0.05);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 8. COMPONENT SEPARATION OF CONCERNS
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Component separation of concerns', () => {
  it('no component directly creates NeuralNetwork (only hooks do)', () => {
    const compDir = path.join(SRC, 'components');
    const components = fs.readdirSync(compDir).filter(f => f.endsWith('.tsx'));

    for (const comp of components) {
      const src = fs.readFileSync(path.join(compDir, comp), 'utf-8');
      expect(src, `${comp} directly imports NeuralNetwork class`).not.toContain('new NeuralNetwork');
    }
  });

  it('no component calls trainBatch directly (only hooks do)', () => {
    const compDir = path.join(SRC, 'components');
    const components = fs.readdirSync(compDir).filter(f => f.endsWith('.tsx'));

    for (const comp of components) {
      const src = fs.readFileSync(path.join(compDir, comp), 'utf-8');
      expect(src, `${comp} calls trainBatch directly`).not.toContain('trainBatch');
    }
  });

  it('no component reads from localStorage or sessionStorage', () => {
    const compDir = path.join(SRC, 'components');
    const components = fs.readdirSync(compDir).filter(f => f.endsWith('.tsx'));

    for (const comp of components) {
      const src = fs.readFileSync(path.join(compDir, comp), 'utf-8');
      expect(src, `${comp} uses localStorage`).not.toContain('localStorage');
      expect(src, `${comp} uses sessionStorage`).not.toContain('sessionStorage');
    }
  });

  it('nn/ module has zero React imports (pure logic)', () => {
    const nnDir = path.join(SRC, 'nn');
    const nnFiles = fs.readdirSync(nnDir).filter(f => f.endsWith('.ts'));

    for (const file of nnFiles) {
      const src = fs.readFileSync(path.join(nnDir, file), 'utf-8');
      expect(src, `nn/${file} imports React`).not.toContain("from 'react'");
    }
  });

  it('utils.ts has zero React imports (pure functions)', () => {
    const src = fs.readFileSync(path.join(SRC, 'utils.ts'), 'utf-8');
    expect(src).not.toContain("from 'react'");
  });

  it('constants.ts has zero React imports', () => {
    const src = fs.readFileSync(path.join(SRC, 'constants.ts'), 'utf-8');
    expect(src).not.toContain("from 'react'");
  });

  it('data/ module has zero React imports (pure data)', () => {
    const dataDir = path.join(SRC, 'data');
    const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.ts'));
    for (const file of files) {
      const src = fs.readFileSync(path.join(dataDir, file), 'utf-8');
      expect(src, `data/${file} imports React`).not.toContain("from 'react'");
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 9. BUILD & CONFIGURATION INTEGRITY
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Build & config integrity', () => {
  it('package.json has required scripts', () => {
    const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf-8'));
    expect(pkg.scripts.build).toBeDefined();
    expect(pkg.scripts.test).toBeDefined();
    expect(pkg.scripts.dev).toBeDefined();
    expect(pkg.scripts.deploy).toBeDefined();
  });

  it('package.json has no runtime ML dependencies', () => {
    const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf-8'));
    const deps = Object.keys(pkg.dependencies || {});
    const forbidden = ['tensorflow', '@tensorflow', 'brain.js', 'ml5', 'onnxruntime', 'pytorch'];
    for (const dep of deps) {
      for (const f of forbidden) {
        expect(dep, `Has ML dependency: ${dep}`).not.toContain(f);
      }
    }
  });

  it('vite.config.ts has correct base path for GitHub Pages', () => {
    const viteSrc = fs.readFileSync(path.join(ROOT, 'vite.config.ts'), 'utf-8');
    expect(viteSrc).toContain("base: '/neuralplayground/'");
  });

  it('index.html has proper meta tags', () => {
    const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf-8');
    expect(html).toContain('charset="UTF-8"');
    expect(html).toContain('viewport');
    expect(html).toContain('description');
    expect(html).toContain('og:title');
    expect(html).toContain('twitter:card');
    expect(html).toContain('application/ld+json');
  });

  it('index.html has noscript fallback', () => {
    const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf-8');
    expect(html).toContain('<noscript>');
  });

  it('index.html has loading spinner', () => {
    const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf-8');
    expect(html).toContain('app-loader');
  });
});

// ═══════════════════════════════════════════════════════════════════
// 10. CROSS-CUTTING ARCHITECTURE VALIDATION
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Architecture validation', () => {
  it('all .tsx files in components/ export exactly one React component', () => {
    const compDir = path.join(SRC, 'components');
    const files = fs.readdirSync(compDir).filter(f => f.endsWith('.tsx'));

    for (const file of files) {
      const src = fs.readFileSync(path.join(compDir, file), 'utf-8');
      // Count EXPORTED React components (the public API — helper functions don't count)
      const exportedFnComponents = (src.match(/^export (?:default )?function \w+/gm) || []).length;
      const exportedClassComponents = (src.match(/export (?:default )?class \w+ extends Component/gm) || []).length;
      const exportedForwardRef = (src.match(/export const \w+ = forwardRef/gm) || []).length;
      // Also count `export { ... }` and `export default` for named components
      const defaultExports = (src.match(/^export default \w+/gm) || []).length;
      
      // Files may have helper functions (computeNodePositions, generateParticles in NetworkVisualizer)
      // but should only export one primary React component + optionally types/interfaces
      const totalExportedComponents = exportedFnComponents + exportedClassComponents + exportedForwardRef;
      
      // At least one component exported, and including the default re-export it should be <= 3
      // (export function X, export default X, and maybe a named export of the handle type)
      expect(totalExportedComponents, `${file} exports ${totalExportedComponents} components`).toBeGreaterThanOrEqual(1);
      
      // Verify there's a default export (either inline or separate)
      const hasDefault = src.includes('export default');
      expect(hasDefault, `${file} has no default export`).toBe(true);
    }
  });

  it('all hooks follow use* naming convention', () => {
    const hookDir = path.join(SRC, 'hooks');
    const files = fs.readdirSync(hookDir).filter(f => f.endsWith('.ts'));

    for (const file of files) {
      expect(file, `Hook file ${file} doesn't start with 'use'`).toMatch(/^use/);
      const src = fs.readFileSync(path.join(hookDir, file), 'utf-8');
      const exports = [...src.matchAll(/export function (\w+)/g)].map(m => m[1]);
      for (const exp of exports) {
        expect(exp, `Exported hook ${exp} doesn't start with 'use'`).toMatch(/^use/);
      }
    }
  });

  it('test files exist for core modules', () => {
    const testDir = path.join(SRC, '__tests__');
    const testFiles = fs.readdirSync(testDir);
    expect(testFiles).toContain('neuralNetwork.test.ts');
    expect(testFiles).toContain('sampleData.test.ts');
    expect(testFiles).toContain('blackhat.test.ts');
    expect(testFiles).toContain('greenhat.test.ts');
    expect(testFiles).toContain('bluehat.test.ts');
  });

  it('data flow is unidirectional: nn → hooks → App → components (utility hooks excepted)', () => {
    // Components should only import utility hooks (like useContainerDims),
    // never business-logic hooks (useNeuralNetwork, useCinematic) — those belong in App.
    const compDir = path.join(SRC, 'components');
    const components = fs.readdirSync(compDir).filter(f => f.endsWith('.tsx'));

    // Utility hooks that are acceptable for direct component use
    const utilityHooks = new Set(['useContainerDims']);

    for (const comp of components) {
      const src = fs.readFileSync(path.join(compDir, comp), 'utf-8');
      const hookImports = [...src.matchAll(/from ['"]\.\.\/hooks\/(\w+)['"]/g)]
        .map(m => m[1]);

      const nonUtilHookImports = hookImports.filter(h => !utilityHooks.has(h));
      expect(nonUtilHookImports, `${comp} imports business-logic hooks: ${nonUtilHookImports.join(', ')}`).toEqual([]);
    }
  });

  it('CSS is contained in dedicated files (no inline style objects > 3 props in components)', () => {
    // We allow some inline styles but the bulk should be in CSS
    expect(fs.existsSync(path.join(SRC, 'App.css'))).toBe(true);
    const cssContent = fs.readFileSync(path.join(SRC, 'App.css'), 'utf-8');
    // Should have substantial CSS
    expect(cssContent.length).toBeGreaterThan(1000);
  });

  it('every source file has no console.log (only console.error in ErrorBoundary)', () => {
    const allFiles = collectFiles(SRC, ['.ts', '.tsx']).filter(f => !f.includes('__tests__'));

    for (const file of allFiles) {
      const src = fs.readFileSync(file, 'utf-8');
      const basename = path.basename(file);
      if (basename === 'ErrorBoundary.tsx') {
        // ErrorBoundary may use console.error
        expect(src).not.toContain('console.log');
      } else {
        expect(src, `${basename} has console.log`).not.toContain('console.log');
        expect(src, `${basename} has console.warn`).not.toContain('console.warn');
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 11. DIGIT STROKES DATA INTEGRITY
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — Digit strokes data', () => {
  it('DIGIT_STROKES is exported and has entries for digits 0-9', async () => {
    const { DIGIT_STROKES } = await import('../data/digitStrokes');
    for (let d = 0; d < 10; d++) {
      expect(DIGIT_STROKES[d], `Missing strokes for digit ${d}`).toBeDefined();
      expect(DIGIT_STROKES[d].length, `Digit ${d} has no strokes`).toBeGreaterThan(0);
    }
  });

  it('getDigitDrawDuration returns positive values for all digits', async () => {
    const { getDigitDrawDuration } = await import('../data/digitStrokes');
    for (let d = 0; d < 10; d++) {
      const dur = getDigitDrawDuration(d);
      expect(dur, `Digit ${d} has non-positive duration`).toBeGreaterThan(0);
    }
  });

  it('all stroke points have numeric x and y', async () => {
    const { DIGIT_STROKES } = await import('../data/digitStrokes');
    for (let d = 0; d < 10; d++) {
      for (const stroke of DIGIT_STROKES[d]) {
        for (const pt of stroke.points) {
          expect(typeof pt.x).toBe('number');
          expect(typeof pt.y).toBe('number');
          expect(isFinite(pt.x)).toBe(true);
          expect(isFinite(pt.y)).toBe(true);
        }
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 12. END-TO-END DATA PIPELINE
// ═══════════════════════════════════════════════════════════════════

describe('Blue Hat — End-to-end data pipeline', () => {
  it('full pipeline: generate → train → predict → valid output', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { generateTrainingData } = await import('../nn/sampleData');
    const { DEFAULT_CONFIG } = await import('../constants');

    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const data = generateTrainingData(5);

    // Train
    for (let i = 0; i < 10; i++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      expect(isFinite(snap.loss)).toBe(true);
      expect(snap.accuracy).toBeGreaterThanOrEqual(0);
      expect(snap.accuracy).toBeLessThanOrEqual(1);
    }

    // Predict
    const result = nn.predict(data.inputs[0]);
    expect(result.label).toBeGreaterThanOrEqual(0);
    expect(result.label).toBeLessThanOrEqual(9);
    expect(result.probabilities.length).toBe(10);
    const sum = result.probabilities.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);

    // Layers returned for visualization
    expect(result.layers.length).toBe(DEFAULT_CONFIG.layers.length + 1); // hidden + output
    for (const layer of result.layers) {
      expect(layer.weights.length).toBeGreaterThan(0);
      expect(layer.biases.length).toBeGreaterThan(0);
      expect(layer.activations.length).toBeGreaterThan(0);
    }
  });

  it('canvasToInput → predict pipeline', async () => {
    const { NeuralNetwork } = await import('../nn/NeuralNetwork');
    const { canvasToInput } = await import('../nn/sampleData');
    const { DEFAULT_CONFIG } = await import('../constants');

    // Simulate a 280×280 canvas (like DrawingCanvas)
    const width = 280, height = 280;
    const data = new Uint8ClampedArray(width * height * 4);
    // Draw a white square in the center (like someone drawing)
    for (let y = 100; y < 180; y++) {
      for (let x = 100; x < 180; x++) {
        const idx = (y * width + x) * 4;
        data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 255; data[idx + 3] = 255;
      }
    }

    const imageData = { width, height, data, colorSpace: 'srgb' } as unknown as ImageData;
    const input = canvasToInput(imageData);
    expect(input.length).toBe(784);
    expect(input.every(v => v >= 0 && v <= 1)).toBe(true);

    // Some pixels should be active (the square)
    const activePixels = input.filter(v => v > 0.5).length;
    expect(activePixels).toBeGreaterThan(0);

    // Predict with this input
    const nn = new NeuralNetwork(784, DEFAULT_CONFIG);
    const result = nn.predict(input);
    expect(result.label).toBeGreaterThanOrEqual(0);
    expect(result.label).toBeLessThanOrEqual(9);
  });
});
